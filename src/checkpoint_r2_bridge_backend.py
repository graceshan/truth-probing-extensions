"""Offline pinned HF backend, imported ONLY by explicitly acknowledged bridge runs.

This module does not execute at import. There is no download, fallback, fit,
research metric, chat prompt, generation or final/test route.
"""
import gc
import inspect
import importlib.metadata
import subprocess
import json
import os
import platform
from pathlib import Path
import numpy as np
from src.checkpoint_r2_bridge_runner import digest, require, text_hash


def read_hidden_at_last(states, mask, layers, width):
    """Shared actual/synthetic tensor readout; never invokes a model."""
    import torch
    mask=np.asarray(mask)
    require(mask.ndim==2 and set(np.unique(mask))<={0,1} and np.all(mask.sum(1)>0),'invalid readout mask')
    require(len(states)==layers+1,'incomplete HF layer coverage')
    readout=np.where(mask,np.arange(mask.shape[1]),-1).max(1)
    selected=[]
    for layer in states[1:]:
        require(tuple(layer.shape)==(len(mask),mask.shape[1],width) and layer.dtype==torch.bfloat16,'HF layer shape/dtype')
        selected.append(layer[torch.arange(len(mask),device=layer.device),torch.tensor(readout,device=layer.device)].float().cpu().numpy())
    return np.stack(selected,axis=1),readout


class HFBackend:
    def __init__(self,model_name,runner_config):
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        from transformers.utils.hub import cached_file
        self.torch=torch;self.name=model_name;self.spec=runner_config['models'][model_name];self.settings=runner_config['fresh_execution']
        cfg=self.settings
        versions={'python':platform.python_version(),'torch':torch.__version__.split('+')[0],'transformers':transformers.__version__,'numpy':np.__version__}
        for k,v in versions.items():require(v==cfg[k],f'pinned runtime required: {k}={cfg[k]}, found {v}')
        require(torch.cuda.is_available() and torch.cuda.is_bf16_supported(),'CUDA BF16 required')
        require(torch.version.cuda==cfg['cuda'] and torch.backends.cudnn.version()==cfg['cudnn'],'CUDA/cuDNN runtime mismatch')
        require(torch.cuda.get_device_name(0)==cfg['gpu_name'],'different GPU configuration; no automatic cross-configuration fallback')
        torch.cuda.set_device(0);torch.set_num_threads(1)
        if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
        torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
        torch.set_float32_matmul_precision(cfg['float32_matmul_precision'])
        torch.backends.cuda.matmul.allow_tf32=cfg['allow_tf32'];torch.backends.cudnn.allow_tf32=cfg['allow_tf32']
        torch.use_deterministic_algorithms(cfg['deterministic_algorithms'])
        torch.backends.cuda.enable_flash_sdp(cfg['sdpa_flash_enabled']);torch.backends.cuda.enable_mem_efficient_sdp(cfg['sdpa_mem_efficient_enabled']);torch.backends.cuda.enable_math_sdp(cfg['sdpa_math_enabled'])
        model_id,revision=self.spec['model_id'],self.spec['revision'];files={}
        def cached(name):
            p=Path(cached_file(model_id,name,revision=revision,local_files_only=True))
            require(p.parent.name==revision,'resolved snapshot revision mismatch')
            files[name]={'sha256':digest(p),'bytes':p.stat().st_size};return p
        for name,expected in self.spec['files_sha256'].items():
            cached(name);require(files[name]['sha256']==expected,'pinned tokenizer/config file mismatch: '+name)
        config=AutoConfig.from_pretrained(model_id,revision=revision,local_files_only=True,trust_remote_code=False)
        require(config._commit_hash==revision,'resolved config revision')
        require(config.num_hidden_layers==self.spec['layers'] and config.hidden_size==self.spec['width'],'model dimensions')
        # Require the reviewed snapshot's safetensors index; never try a different format/revision.
        index=cached('model.safetensors.index.json');weight_map=json.loads(index.read_text())['weight_map']
        for name in sorted(set(weight_map.values())):cached(name)
        self.tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,local_files_only=True,trust_remote_code=False,use_fast=True)
        require(self.tokenizer.is_fast,'reviewed fast tokenizer required')
        require(type(self.tokenizer).__name__==self.spec['tokenizer_class'],'tokenizer class differs from recorded producer')
        require(text_hash(self.tokenizer.backend_tokenizer.to_str())==self.spec['tokenizer_backend_sha256'],'tokenizer backend hash differs; no fallback')
        self.model=AutoModelForCausalLM.from_pretrained(model_id,revision=revision,local_files_only=True,trust_remote_code=False,use_safetensors=True,config=config,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda:0').eval()
        require(self.model.config._commit_hash==revision and self.model.dtype==torch.bfloat16 and self.model.config._attn_implementation=='sdpa','loaded representation mismatch')
        # The fallback pad ID is explicit and only applies to declared calibration padding.
        self.pad_id=self.tokenizer.pad_token_id
        if self.pad_id is None:self.pad_id=self.tokenizer.eos_token_id
        require(isinstance(self.pad_id,int),'missing explicit padding token')
        driver=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],text=True,capture_output=True,timeout=5)
        require(driver.returncode==0,'driver observation unavailable')
        dependencies={name:importlib.metadata.version(name) for name in ('tokenizers','safetensors','huggingface-hub')}
        self.runtime={'driver_version':driver.stdout.strip(),'runtime_dependencies':dependencies,'hostname':platform.node(),'platform':platform.platform(),'versions':versions,'torch_build':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu_name':torch.cuda.get_device_name(0),'gpu_total_bytes':torch.cuda.get_device_properties(0).total_memory,'model_id':model_id,'model_revision':revision,'tokenizer_revision':revision,'files':files,'model_implementation_sha256':digest(Path(inspect.getfile(type(self.model)))),'tokenizer_class':type(self.tokenizer).__name__,'tokenizer_backend_sha256':text_hash(self.tokenizer.backend_tokenizer.to_str()),'actual_settings':{'device':str(next(self.model.parameters()).device),'compute_dtype':str(self.model.dtype),'saved_dtype':'float16','attention':self.model.config._attn_implementation,'eval_mode':not self.model.training,'float32_matmul_precision':torch.get_float32_matmul_precision(),'allow_tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'allow_tf32_cudnn':torch.backends.cudnn.allow_tf32,'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),'sdpa_flash_enabled':torch.backends.cuda.flash_sdp_enabled(),'sdpa_mem_efficient_enabled':torch.backends.cuda.mem_efficient_sdp_enabled(),'sdpa_math_enabled':torch.backends.cuda.math_sdp_enabled(),'seed':cfg['seed'],'torch_threads':torch.get_num_threads(),'torch_interop_threads':torch.get_num_interop_threads()},'layer_convention':{'HF_indices':list(range(1,self.spec['layers']+1)),'saved_indices':list(range(self.spec['layers'])),'embedding_excluded':True,'final_saved_layer':'post-final-RMSNorm'},'special_tokens':{'bos_token_id':self.tokenizer.bos_token_id,'eos_token_id':self.tokenizer.eos_token_id,'native_pad_token_id':self.tokenizer.pad_token_id,'calibration_pad_token_id':self.pad_id},'historical_execution_verified':False}

    def forward(self,rows,padding):
        require(padding in (None,'left','right') and len(rows)==(1 if padding is None else 2),'unapproved batch/padding configuration')
        require(all(r['split']=='train' for r in rows) if padding else all(r['split'] in ('train','validation') for r in rows),'E or non-training padding input')
        torch=self.torch
        tokens=[self.tokenizer.encode(r['statement'],add_special_tokens=True,truncation=False) for r in rows]
        require(all(0<len(t)<=512 for t in tokens),'bridge token safety bound; reject, never truncate')
        length=max(map(len,tokens));ids=[];mask=[]
        for t in tokens:
            extra=length-len(t);pad=[self.pad_id]*extra
            ids.append(pad+t if padding=='left' else t+pad)
            mask.append([0]*extra+[1]*len(t) if padding=='left' else [1]*len(t)+[0]*extra)
        mask_np=np.array(mask,dtype=np.int64);pos=np.maximum(mask_np.cumsum(1)-1,0);pos[mask_np==0]=0
        readout=np.where(mask_np,np.arange(length),-1).max(1)
        encoded={'input_ids':torch.tensor(ids,dtype=torch.long,device='cuda:0'),'attention_mask':torch.tensor(mask,dtype=torch.long,device='cuda:0'),'position_ids':torch.tensor(pos,dtype=torch.long,device='cuda:0')}
        torch.cuda.synchronize()
        with torch.inference_mode():
            result=self.model(**encoded,output_hidden_states=True,use_cache=False,return_dict=True,logits_to_keep=1)
            compute,readout=read_hidden_at_last(result.hidden_states,mask_np,self.spec['layers'],self.spec['width'])
        torch.cuda.synchronize()
        physical=[{'token_ids':ids[i],'attention_mask':mask[i],'position_ids':pos[i].tolist(),'readout_index':int(readout[i]),'semantic_readout_position':len(tokens[i])-1,'pad_token_id':self.pad_id} for i in range(len(rows))]
        return {'ids':[r['id'] for r in rows],'statement_sha256':[text_hash(r['statement']) for r in rows],'compute':compute,'physical':physical,'execution':{'batch_size':len(rows),'padding':padding,'add_special_tokens':True,'truncation':False,'chat_template':False,'use_cache':False,'output_hidden_states':True,'logits_to_keep':1,'compute_dtype':'bfloat16','compute_serialization':'float32 exact widening of BF16','saved_dtype':'float16','layer_indices':list(range(self.spec['layers']))}}

    def close(self):
        del self.model;gc.collect();self.torch.cuda.empty_cache()
