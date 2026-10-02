"""Fresh pinned HF snapshots on a supported GPU; never imports historical heads."""
import importlib.metadata
import inspect
import json
import platform
import subprocess
import time
from pathlib import Path
import numpy as np
from src.checkpoint_r2_fresh_inputs import file_hash, require, text_hash
from src.checkpoint_r2_fresh_store import write_json


def verify_layer_states(states, captured, layers):
    import torch
    require(len(states) == layers + 1 and len(captured) == layers + 1, 'missing block/norm capture')
    require(all(torch.equal(states[i + 1], captured[i]) for i in range(layers - 1)),
            'intermediate block layer semantics')
    require(torch.equal(states[-1], captured['norm']), 'final layer must be post-RMSNorm')


def readout_states(states, mask, spec):
    import torch
    require(len(states) == spec['layers'] + 1, 'missing HF layer')
    require(mask.ndim == 2 and set(np.unique(mask)) <= {0, 1} and np.all(mask.sum(1) > 0), 'invalid readout mask')
    indices = np.where(mask, np.arange(mask.shape[1]), -1).max(1)
    selected = []
    for state in states[1:]:
        require(tuple(state.shape) == (len(mask), mask.shape[1], spec['width']) and state.dtype == torch.bfloat16,
                'hidden layer shape/dtype')
        selected.append(state[torch.arange(len(mask), device=state.device),
                              torch.tensor(indices, device=state.device)].float().cpu().numpy())
    values = np.stack(selected, axis=1)
    require(np.isfinite(values).all(), 'nonfinite computation')
    return values


def runtime_receipt(lock):
    import torch
    import transformers
    versions = {name: importlib.metadata.version(name) for name in lock['packages']}
    require(versions == lock['packages'], 'installed packages differ from fresh runtime lock')
    require(platform.python_version().startswith(lock['python_series'] + '.'), 'Python series mismatch')
    require(torch.cuda.is_available() and torch.cuda.device_count() >= 1 and torch.cuda.is_bf16_supported(),
            'visible CUDA BF16 GPU required')
    driver = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,memory.total',
                             '--format=csv,noheader'], capture_output=True, text=True, timeout=10, check=True)
    freeze = subprocess.run([__import__('sys').executable, '-m', 'pip', 'freeze', '--all'],
                            capture_output=True, text=True, timeout=30, check=True)
    return dict(python=platform.python_version(), executable=__import__('sys').executable,
                packages=versions, pip_freeze=freeze.stdout.splitlines(), platform=platform.platform(),
                hostname=platform.node(), torch_build=torch.__version__, transformers=transformers.__version__,
                cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                device='cuda:0', gpu=torch.cuda.get_device_name(0),
                gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
                compute_capability=list(torch.cuda.get_device_capability(0)), driver=driver.stdout.strip(),
                attention='sdpa', sdpa_kernel='math (flash and memory-efficient disabled)',
                compute_dtype='bfloat16', stored_dtype='float16')


def prepare_snapshots(config, cache, output):
    """Download only the two exact snapshots into a new cache; hash every resolved file."""
    from huggingface_hub import HfApi, get_token, hf_hub_download
    require(not cache.exists(), 'fresh cache must not exist; preserve old snapshots')
    cache.mkdir(parents=True)
    receipt = {}
    # Resolve credentials in memory only. The wrapper preserves the original token
    # file path across HF_HOME isolation; an existing HF_TOKEN also remains available.
    token = get_token()
    require(bool(token), 'Hugging Face authentication required; use the secure terminal procedure in the handoff')
    api = HfApi(token=token)
    infos = {m: api.model_info(s['model_id'], revision=s['revision'])
             for m, s in config['models'].items()}
    for model, spec in config['models'].items():
        require(infos[model].sha == spec['revision'], 'remote resolved revision mismatch')
    # Public model_info is not proof of gated download access. Force an actual
    # authenticated protected-file request before either model's large weights.
    llama = config['models']['llama']
    access_started = time.monotonic()
    try:
        protected = Path(hf_hub_download(llama['model_id'], 'config.json', revision=llama['revision'],
                                        cache_dir=str(cache), token=token, force_download=True))
    except Exception as exc:
        # Never persist response headers, URLs or arbitrary authentication errors.
        response = getattr(exc, 'response', None)
        status = getattr(response, 'status_code', None)
        write_json(output / 'llama-protected-file-access.json', dict(
            model_id=llama['model_id'], revision=llama['revision'], filename='config.json',
            authenticated_token_resolved=True, protected_file_access_verified=False,
            error_type=type(exc).__name__, http_status=status if isinstance(status, int) else None,
            access_seconds=time.monotonic() - access_started))
        raise PermissionError('Pinned Llama protected-file access failed; complete the secure authentication '
                              'procedure in the handoff before reviewed execution') from None
    require(protected.parent.name == llama['revision'] and protected.stat().st_size > 0,
            'protected file resolved snapshot/size')
    require(file_hash(protected) == llama['files_sha256']['config.json'], 'protected config pin mismatch')
    write_json(output / 'llama-protected-file-access.json', dict(
        model_id=llama['model_id'], revision=llama['revision'], filename='config.json',
        authenticated_token_resolved=True, protected_file_access_verified=True,
        method='authenticated hf_hub_download(force_download=True)',
        resolved_snapshot_revision=protected.parent.name, sha256=file_hash(protected),
        bytes=protected.stat().st_size, access_seconds=time.monotonic() - access_started))
    for model in ('qwen', 'llama'):
        spec = config['models'][model]
        require(infos[model].sha == spec['revision'], 'remote resolved revision mismatch')
        started = time.monotonic()
        def fetch(name):
            return Path(hf_hub_download(spec['model_id'], name, revision=spec['revision'], cache_dir=str(cache), token=token))
        index = fetch('model.safetensors.index.json')
        files = set(json.loads(index.read_text())['weight_map'].values()) | set(spec['files_sha256'])
        files.add('model.safetensors.index.json')
        # Generation config is optional and irrelevant to raw extraction; exclude it explicitly.
        resolved = {name: fetch(name) for name in sorted(files)}
        for name, expected in spec['files_sha256'].items():
            require(file_hash(resolved[name]) == expected, 'model/tokenizer/config pin: ' + name)
        pins = {name: dict(path=str(p), resolved_path=str(p.resolve()), sha256=file_hash(p),
                           bytes=p.stat().st_size) for name, p in resolved.items()}
        require(all(v['bytes'] > 0 for v in pins.values()), 'empty snapshot file')
        receipt[model] = dict(model_id=spec['model_id'], revision=spec['revision'],
                             tokenizer_revision=spec['revision'], snapshot=str(index.parent), files=pins,
                             download_and_hash_seconds=time.monotonic() - started,
                             downloaded_bytes=sum(v['bytes'] for v in pins.values()))
        write_json(output / (model + '-snapshot.json'), receipt[model])
    return receipt


class FreshBackend:
    def __init__(self, spec, snapshot, settings):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        self.torch, self.spec, self.settings = torch, spec, settings
        torch.cuda.set_device(0)
        # Start the single per-model interval BEFORE loading, including transient
        # allocations during from_pretrained/device transfer; never reset afterward.
        torch.cuda.reset_peak_memory_stats(0)
        torch.manual_seed(settings['seed'])
        torch.cuda.manual_seed_all(settings['seed'])
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        path = snapshot['snapshot']
        require(Path(path).name == spec['revision'], 'local snapshot revision')
        # Recheck the exact resolved files before loading.
        for identity in snapshot['files'].values():
            require(file_hash(identity['path']) == identity['sha256'], 'snapshot changed after receipt')
        cfg = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
        require((cfg.num_hidden_layers, cfg.hidden_size) == (spec['layers'], spec['width']), 'model dimensions')
        self.tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
        require(self.tokenizer.is_fast, 'fast pinned tokenizer required')
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.tokenizer.eos_token_id
        require(isinstance(self.pad_id, int), 'no padding ID')
        self.model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                         use_safetensors=True, torch_dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda:0').eval()
        require(self.model.dtype == torch.bfloat16 and self.model.config._attn_implementation == 'sdpa', 'precision/backend')
        self.semantic_checked = False
        self.details = dict(tokenizer_class=type(self.tokenizer).__name__,
                            tokenizer_backend_sha256=text_hash(self.tokenizer.backend_tokenizer.to_str()),
                            model_class=type(self.model).__name__,
                            model_implementation_sha256=file_hash(inspect.getfile(type(self.model))),
                            tokenizer_implementation_sha256=file_hash(inspect.getfile(type(self.tokenizer))),
                            native_bos_id=self.tokenizer.bos_token_id, native_eos_id=self.tokenizer.eos_token_id,
                            native_pad_id=self.tokenizer.pad_token_id, padding_id=self.pad_id,
                            eval=not self.model.training, device=str(next(self.model.parameters()).device),
                            actual_flags=dict(tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
                                              tf32_cudnn=torch.backends.cudnn.allow_tf32,
                                              matmul_precision=torch.get_float32_matmul_precision(),
                                              deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                                              flash_sdpa=torch.backends.cuda.flash_sdp_enabled(),
                                              memory_efficient_sdpa=torch.backends.cuda.mem_efficient_sdp_enabled(),
                                              math_sdpa=torch.backends.cuda.math_sdp_enabled()),
                            settings=settings, max_position_embeddings=cfg.max_position_embeddings,
                            peak_memory_tracking='one reset before model loading; load and execution share the same peak interval')

    def forward(self, rows, batch_size):
        torch = self.torch
        require(0 < len(rows) <= batch_size and batch_size in (1, 2, 4, 8), 'undeclared batch')
        # Single and batched tokenization APIs must agree exactly, without truncation.
        tokens = [self.tokenizer.encode(r['statement'], add_special_tokens=True, truncation=False) for r in rows]
        batch_encoding = self.tokenizer([r['statement'] for r in rows], add_special_tokens=True,
                                        truncation=False, padding=False, return_special_tokens_mask=True)
        batched = batch_encoding['input_ids']
        special_masks = batch_encoding['special_tokens_mask']
        plain = [self.tokenizer.encode(r['statement'], add_special_tokens=False, truncation=False) for r in rows]
        require(all([t for t, special in zip(tokens[i], special_masks[i]) if not special] == plain[i]
                    for i in range(len(rows))), 'native special-token construction')
        require(tokens == batched and all(0 < len(t) <= min(self.settings['max_tokens'],
                self.model.config.max_position_embeddings) for t in tokens), 'tokenization/truncation bound')
        length = max(map(len, tokens))
        ids = [t + [self.pad_id] * (length - len(t)) for t in tokens]
        mask = np.array([[1] * len(t) + [0] * (length - len(t)) for t in tokens], dtype=np.int64)
        pos = np.maximum(mask.cumsum(1) - 1, 0)
        pos[mask == 0] = 0
        readout = mask.sum(1) - 1  # right padding only
        arguments = dict(input_ids=torch.tensor(ids, device='cuda:0'),
                         attention_mask=torch.tensor(mask, device='cuda:0'),
                         position_ids=torch.tensor(pos, device='cuda:0'))
        # Once/model, verify HF layer identities against actual decoder block outputs and final RMSNorm.
        captured, handles = {}, []
        if not self.semantic_checked:
            def hook(key):
                def capture(module, args, value):
                    value = value[0] if isinstance(value, tuple) else value
                    captured[key] = value.detach()
                return capture
            for i, block in enumerate(self.model.model.layers):
                handles.append(block.register_forward_hook(hook(i)))
            handles.append(self.model.model.norm.register_forward_hook(hook('norm')))
        try:
            with torch.inference_mode():
                output = self.model(**arguments, output_hidden_states=True, use_cache=False,
                                    return_dict=True, logits_to_keep=1)
                states = output.hidden_states
                require(len(states) == self.spec['layers'] + 1, 'missing layer')
                if not self.semantic_checked:
                    verify_layer_states(states, captured, self.spec['layers'])
                    self.semantic_checked = True
                compute = readout_states(states, mask, self.spec)
            torch.cuda.synchronize()
        finally:
            for h in handles:
                h.remove()
        require(np.isfinite(compute).all(), 'nonfinite computation')
        physical = [dict(id=r['id'], statement_sha256=r['statement_sha256'],
                         unpadded_token_ids=tokens[i], raw_token_ids=plain[i],
                         unpadded_special_tokens_mask=special_masks[i],
                         token_ids=ids[i], attention_mask=mask[i].tolist(),
                         position_ids=pos[i].tolist(), readout_index=int(readout[i]),
                         semantic_readout_position=len(tokens[i]) - 1, pad_token_id=self.pad_id,
                         add_special_tokens=True, truncation=False, chat_template=False)
                    for i, r in enumerate(rows)]
        return compute, physical

    def peak(self):
        return dict(allocated=self.torch.cuda.max_memory_allocated(), reserved=self.torch.cuda.max_memory_reserved())

    def close(self):
        import gc
        del self.model
        gc.collect()
        self.torch.cuda.empty_cache()
