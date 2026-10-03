"""Frozen chat rendering, full continuation likelihood, and strict generation parsing."""
import math
from src.checkpoint_next_inventory import digest, require


def render_packet(tokenizer, user_content, model, max_tokens=1024):
    messages=[{'role':'user','content':user_content}]
    native=tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    rendered=native
    removed=''
    # Qwen's pinned native template inserts a default system even with user-only messages.
    # Remove ONLY that complete native prefix to honor the explicit no-system contract.
    # User and assistant-generation segments remain byte-for-byte native template output.
    if model=='qwen' and native.startswith('<|im_start|>system\n'):
        end=native.find('<|im_end|>\n')
        require(end>=0, 'unterminated implicit native system')
        removed=native[:end+len('<|im_end|>\n')]
        rendered=native[len(removed):]
    require(not rendered.startswith('<|im_start|>system') and
            '<|start_header_id|>system<|end_header_id|>' not in rendered, 'extra system message')
    if model=='qwen':
        require(rendered.startswith('<|im_start|>user\n') and rendered.endswith('<|im_start|>assistant\n'), 'native Qwen boundaries')
    elif model=='llama':
        require(rendered.startswith('<|begin_of_text|><|start_header_id|>user<|end_header_id|>') and
                rendered.endswith('<|start_header_id|>assistant<|end_header_id|>\n\n'), 'native Llama boundaries')
    else:
        raise ValueError('unknown model')
    ids=tokenizer.encode(rendered,add_special_tokens=False,truncation=False)
    native_ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    require(native_ids==tokenizer.encode(native,add_special_tokens=False,truncation=False), 'native chat tokenization mismatch')
    if removed:
        prefix=tokenizer.encode(removed,add_special_tokens=False,truncation=False)
        require(native_ids==prefix+ids,'implicit system removal changed token boundary')
    else:
        require(native_ids==ids, 'native packet mismatch')
    require(0<len(ids)<=max_tokens, 'prompt length/truncation bound')
    candidates={}
    for answer in ('True','False'):
        whole=tokenizer.encode(rendered+answer,add_special_tokens=False,truncation=False)
        require(whole[:len(ids)]==ids and len(whole)>len(ids), 'continuation retokenizes prompt boundary')
        continuation=whole[len(ids):]
        require(tokenizer.decode(continuation,skip_special_tokens=False,clean_up_tokenization_spaces=False)==answer,
                'continuation is not exact unspaced answer')
        require(not set(continuation)&set(tokenizer.all_special_ids), 'continuation contains termination/special token')
        require(len(whole)<=max_tokens, 'continuation length bound')
        candidates[answer]=continuation
    return dict(schema='checkpoint-next-token-packet-v1',rendered=rendered,native_template_sha256=digest(tokenizer.chat_template),
        removed_implicit_system_prefix=removed,system_messages=0,prompt_ids=ids,attention_mask=[1]*len(ids),
        position_ids=list(range(len(ids))),readout_index=len(ids)-1,candidates=candidates,
        special_token_ids=list(tokenizer.all_special_ids),
        boundary_verified=True,add_special_tokens=False,padding=False,truncation=False)


def continuation_logprob(logits, prompt_ids, continuation):
    """Pure-Python reference: logits[t] predicts token[t+1]; sum all candidate tokens."""
    require(prompt_ids and continuation and len(logits)==len(prompt_ids)+len(continuation), 'sequence dimensions')
    terms=[]
    for offset, token in enumerate(continuation):
        row=logits[len(prompt_ids)-1+offset]
        require(all(math.isfinite(float(v)) for v in row) and 0<=token<len(row), 'nonfinite logits/token bounds')
        maximum=max(row)
        terms.append(float(row[token])-maximum-math.log(math.fsum(math.exp(v-maximum) for v in row)))
    return dict(logprob=math.fsum(terms),token_logprobs=terms)


def likelihood_result(true, false):
    margin=true['logprob']-false['logprob']
    require(math.isfinite(margin), 'nonfinite likelihood margin')
    return dict(true=true,false=false,margin=margin,decision=int(margin>=0),rule='zero_margin_ties_true')


def parse_generation(tokenizer, ids, termination_ids):
    require(len(ids)<=8,'generation token cap')
    cleaned=list(ids)
    while cleaned and cleaned[-1] in termination_ids:
        cleaned.pop()
    # An interior special/termination token is never silently discarded.
    raw=tokenizer.decode(ids,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    content=tokenizer.decode(cleaned,skip_special_tokens=False,clean_up_tokenization_spaces=False)
    canonical=content.strip()
    valid=canonical in ('True','False') and not set(cleaned)&set(tokenizer.all_special_ids)
    return dict(raw_output=raw,content=content,canonical=canonical,token_ids=list(ids),
        content_token_ids=cleaned,termination_ids=list(termination_ids),format_failure=not valid,
        decision=int(canonical=='True') if valid else None)


def native_terminations(tokenizer, config_eos, model):
    # The raw snapshot excluded generation_config.json. Recover native termination
    # semantics from the pinned special-token vocabulary, including end-of-text
    # and Llama end-of-message, rather than fetching a moving generation config.
    names={'qwen':('<|im_end|>','<|endoftext|>'),
           'llama':('<|end_of_text|>','<|eom_id|>','<|eot_id|>')}[model]
    ids=[]
    for name in names:
        token=tokenizer.convert_tokens_to_ids(name)
        require(isinstance(token,int) and token in tokenizer.all_special_ids and
                tokenizer.decode([token],skip_special_tokens=False,clean_up_tokenization_spaces=False)==name,
                'native termination vocabulary mismatch')
        ids.append(token)
    eos=[config_eos] if isinstance(config_eos,int) else list(config_eos or [])
    require(set(eos+[tokenizer.eos_token_id])<=set(ids),'unexpected native config/tokenizer EOS')
    return sorted(set(ids))


class ChatBackend:
    """Uses the accepted reviewed offline loader and numerical runtime, new paths below."""
    def __init__(self, spec, snapshot, settings, model, layers):
        from src.checkpoint_r2_fresh_backend import FreshBackend
        self.raw=FreshBackend(spec,snapshot,settings)
        self.model_name,self.layers=model,layers
        self.torch,self.tokenizer,self.model=self.raw.torch,self.raw.tokenizer,self.raw.model
        # Raw extraction deliberately excluded generation_config.json. Use native EOS
        # from the exact cached model config and tokenizer, never fetch new defaults.
        self.termination_ids=native_terminations(self.tokenizer,self.model.config.eos_token_id,model)

    def packet(self, job):
        return render_packet(self.tokenizer,job['user_content'],self.model_name,self.raw.settings['max_tokens'])

    def arguments(self, ids):
        torch=self.torch
        return dict(input_ids=torch.tensor([ids],device='cuda:0'),
            attention_mask=torch.ones((1,len(ids)),dtype=torch.long,device='cuda:0'),
            position_ids=torch.arange(len(ids),device='cuda:0').unsqueeze(0))

    def score(self, packet):
        results={}
        with self.torch.inference_mode():
            for answer, continuation in packet['candidates'].items():
                ids=packet['prompt_ids']+continuation
                out=self.model(**self.arguments(ids),use_cache=False,output_hidden_states=False,return_dict=True,logits_to_keep=0)
                logits=out.logits[0,len(packet['prompt_ids'])-1:len(ids)-1].float()
                require(self.torch.isfinite(logits).all().item(),'nonfinite scored logits')
                terms=self.torch.log_softmax(logits,dim=-1).gather(1,self.torch.tensor(continuation,device=logits.device).unsqueeze(1))[:,0].double().cpu().tolist()
                results[answer]=dict(logprob=math.fsum(terms),token_logprobs=terms)
                del out,logits
        self.torch.cuda.synchronize()
        return likelihood_result(results['True'],results['False'])

    def generate(self, packet):
        from transformers import GenerationConfig
        # Minimal explicit config avoids inherited sampling/temperature or max-length defaults.
        cfg=GenerationConfig(do_sample=False,num_beams=1,max_new_tokens=8,use_cache=False,
            eos_token_id=self.termination_ids,pad_token_id=self.raw.pad_id,
            bos_token_id=self.tokenizer.bos_token_id)
        with self.torch.inference_mode():
            arguments=self.arguments(packet['prompt_ids'])
            # GenerationMixin grows the mask, but retains explicit positions unchanged.
            # Pinned Qwen/Llama derive positions from the growing mask when omitted.
            arguments.pop('position_ids')
            values=self.model.generate(**arguments,generation_config=cfg)
        self.torch.cuda.synchronize()
        require(values[0,:len(packet['prompt_ids'])].tolist()==packet['prompt_ids'],'generation prompt changed')
        return parse_generation(self.tokenizer,values[0,len(packet['prompt_ids']):].tolist(),self.termination_ids)

    def chat(self, packet):
        from src.checkpoint_r2_fresh_backend import verify_layer_states
        torch=self.torch
        captured,handles={},[]
        if not self.raw.semantic_checked:
            def hook(key):
                def capture(module,args,value):
                    captured[key]=(value[0] if isinstance(value,tuple) else value).detach()
                return capture
            for i,b in enumerate(self.model.model.layers):
                handles.append(b.register_forward_hook(hook(i)))
            handles.append(self.model.model.norm.register_forward_hook(hook('norm')))
        try:
            with torch.inference_mode():
                out=self.model(**self.arguments(packet['prompt_ids']),use_cache=False,output_hidden_states=True,return_dict=True,logits_to_keep=1)
                if not self.raw.semantic_checked:
                    verify_layer_states(out.hidden_states,captured,self.raw.spec['layers'])
                    self.raw.semantic_checked=True
                values=torch.stack([out.hidden_states[l+1][0,packet['readout_index']].float() for l in self.layers]).cpu().numpy()
            self.torch.cuda.synchronize()
            require(values.shape==(2,self.raw.spec['width']),'saved chat layer dimensions')
            import numpy as np
            with np.errstate(over='ignore'):
                saved=values.astype(np.float16)
            require(np.isfinite(saved).all(),'nonfinite/overflow chat state')
            return saved
        finally:
            for h in handles:
                h.remove()

    def preflight_checks(self, packet, likelihood, generation, atol=0.05):
        """Independent prefix indexing and manual greedy oracle; training inputs only.

        Absolute log-probability tolerance is prospective for BF16 full vs prefix
        kernels. Token choices/positions must match exactly; no tolerance relaxation.
        """
        torch=self.torch
        prefix_terms={}
        with torch.inference_mode():
            for answer, continuation in packet['candidates'].items():
                prefix=list(packet['prompt_ids']);terms=[]
                for token in continuation:
                    out=self.model(**self.arguments(prefix),use_cache=False,return_dict=True,logits_to_keep=1)
                    terms.append(float(torch.log_softmax(out.logits[0,-1].float(),dim=-1)[token].item()))
                    prefix.append(token)
                expected=likelihood[answer.lower()]['token_logprobs']
                require(len(expected)==len(terms) and all(abs(a-b)<=atol for a,b in zip(expected,terms)),
                        'causal prefix/full likelihood disagreement')
                prefix_terms[answer]=terms
            prefix=list(packet['prompt_ids']);manual=[]
            for _ in range(8):
                out=self.model(**self.arguments(prefix),use_cache=False,return_dict=True,logits_to_keep=1)
                token=int(out.logits[0,-1].argmax().item());manual.append(token);prefix.append(token)
                if token in self.termination_ids:break
        trace=[]
        original=self.model.prepare_inputs_for_generation
        def checked(*args,**kwargs):
            prepared=original(*args,**kwargs)
            ids=prepared['input_ids'];mask=prepared['attention_mask'];positions=prepared['position_ids']
            n=ids.shape[-1]
            require(mask.shape[-1]==positions.shape[-1]==n and
                    positions[0].tolist()==list(range(n)), 'growing generation positions/mask')
            trace.append(dict(input_length=n,mask_length=mask.shape[-1],positions=positions[0].tolist()))
            return prepared
        self.model.prepare_inputs_for_generation=checked
        try:checked_generation=self.generate(packet)
        finally:self.model.prepare_inputs_for_generation=original
        require(manual==generation['token_ids']==checked_generation['token_ids'], 'manual greedy sequence disagreement')
        require(len(trace)==len(manual) and [r['input_length'] for r in trace]==
                list(range(len(packet['prompt_ids']),len(packet['prompt_ids'])+len(manual))), 'decoding trace coverage')
        return dict(causal_prefix_token_logprobs=prefix_terms,absolute_tolerance=atol,
                    manual_greedy_ids=manual,prepared_steps=trace,independent_checks_passed=True)

    def close(self):
        # Release the alias before the reviewed backend frees the model/cache.
        del self.model
        self.raw.close()
