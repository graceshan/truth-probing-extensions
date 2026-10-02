"""No network/GPU: gated-file access ordering, credential isolation and load peaks."""
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
from types import SimpleNamespace

import pytest
from src.checkpoint_r2_fresh_backend import FreshBackend, prepare_snapshots
from src.checkpoint_r2_fresh_inputs import ROOT, file_hash, text_hash


@pytest.fixture
def hub(monkeypatch):
    import huggingface_hub
    calls = []
    token = 'synthetic-test-credential-never-a-real-token'
    cfg = {'models': {m: {'model_id': 'test/' + m, 'revision': c * 40,
                          'files_sha256': {'config.json': text_hash('{}')}}
                      for m, c in [('qwen','a'), ('llama','b')]}}
    def model_info(repo, revision):
        calls.append(('info', repo, revision))
        return SimpleNamespace(sha=revision)
    def api(**kwargs):
        assert kwargs == {'token':token}
        return SimpleNamespace(model_info=model_info)
    def download(repo, name, *, revision, cache_dir, token=None, force_download=False):
        assert token == 'synthetic-test-credential-never-a-real-token'
        calls.append(('download', repo, name, force_download))
        p = Path(cache_dir) / repo.split('/')[-1] / 'snapshots' / revision / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name == 'model.safetensors.index.json':
            p.write_text(json.dumps({'weight_map': {'weight':'model-00001.safetensors'}}))
        elif name.endswith('.safetensors'):
            p.write_bytes(b'synthetic weights')
        else:
            p.write_text('{}')
        return str(p)
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: token)
    monkeypatch.setattr(huggingface_hub, 'HfApi', api)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', download)
    return cfg, calls, download, token


def test_authenticated_protected_file_precedes_all_weights(tmp_path, hub):
    cfg, calls, _, token = hub
    out = tmp_path / 'output'
    out.mkdir()
    snapshots = prepare_snapshots(cfg, tmp_path / 'cache', out)
    downloads = [call for call in calls if call[0] == 'download']
    assert downloads[0] == ('download','test/llama','config.json',True)
    assert all(call[3] is False for call in downloads[1:])
    receipt = json.loads((out / 'llama-protected-file-access.json').read_text())
    assert receipt['protected_file_access_verified'] is True
    assert receipt['revision'] == receipt['resolved_snapshot_revision'] == 'b' * 40
    assert receipt['bytes'] == 2 and receipt['sha256'] == text_hash('{}')
    assert set(snapshots) == {'qwen','llama'}
    assert token not in ''.join(p.read_text() for p in out.iterdir())


def test_public_model_info_does_not_bypass_denied_download(tmp_path, hub, monkeypatch):
    import huggingface_hub
    cfg, calls, _, token = hub
    out = tmp_path / 'output'
    out.mkdir()
    def denied(repo, name, **kwargs):
        calls.append(('denied', repo, name))
        exc = RuntimeError('server failure including sensitive content: ' + token)
        exc.response = SimpleNamespace(status_code=403)
        raise exc
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', denied)
    with pytest.raises(PermissionError) as error:
        prepare_snapshots(cfg, tmp_path / 'cache', out)
    # Both public metadata calls succeed; denied protected file prevents ALL weights.
    assert len([c for c in calls if c[0] == 'info']) == 2
    assert [c for c in calls if c[0] == 'denied'] == [('denied','test/llama','config.json')]
    assert not any(c[0] == 'download' for c in calls)
    receipt = json.loads((out / 'llama-protected-file-access.json').read_text())
    assert receipt['protected_file_access_verified'] is False and receipt['http_status'] == 403
    assert token not in (out / 'llama-protected-file-access.json').read_text()
    # The worker persists formatted tracebacks; the suppressed authentication cause
    # must not leak arbitrary server text into that receipt or the terminal traceback.
    formatted = ''.join(traceback.format_exception(type(error.value), error.value, error.value.__traceback__))
    assert token not in formatted


def test_missing_authentication_stops_before_network(tmp_path, hub, monkeypatch):
    import huggingface_hub
    cfg, calls, _, _ = hub
    monkeypatch.setattr(huggingface_hub, 'get_token', lambda: None)
    with pytest.raises(ValueError, match='authentication required'):
        prepare_snapshots(cfg, tmp_path / 'cache', tmp_path)
    assert not calls


def test_mismatched_protected_file_stops_before_weights(tmp_path, hub, monkeypatch):
    import huggingface_hub
    cfg, calls, download, _ = hub
    def changed(*args, **kwargs):
        p = Path(download(*args, **kwargs))
        p.write_text('{"changed":true}')
        return str(p)
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', changed)
    with pytest.raises(ValueError, match='protected config pin'):
        prepare_snapshots(cfg, tmp_path / 'cache', tmp_path)
    assert [c for c in calls if c[0] == 'download'] == [('download','test/llama','config.json',True)]


@pytest.mark.parametrize('authentication', ['original_home_file', 'custom_token_path', 'environment', 'legacy_environment'])
def test_wrapper_authentication_survives_fresh_hf_home(tmp_path, authentication):
    """Execute the actual shell wrapper with provisioning/inference stand-ins."""
    bindir = tmp_path / 'bin'
    bindir.mkdir()
    old_home = tmp_path / 'original-hf-home'
    old_home.mkdir()
    token_file = old_home / 'token'
    token_file.write_text('synthetic-test-credential-never-a-real-token')
    token_file.chmod(0o600)
    if authentication == 'custom_token_path':
        token_file = tmp_path / 'external-credentials' / 'token'
        token_file.parent.mkdir()
        token_file.write_text('synthetic-test-credential-never-a-real-token')
        token_file.chmod(0o600)
    # This interpreter simulates pip only. The wrapper's credential-path command
    # and child authentication resolution execute the real Hub get_token API.
    shim = bindir / 'python-shim'
    shim.write_text(f'''#!{sys.executable}
import json, os, subprocess, sys
if sys.argv[1:3] == ['-m', 'pip']:
    raise SystemExit(0)
if sys.argv[1:2] == ['-c']:
    raise SystemExit(subprocess.call([{sys.executable!r}, '-B', *sys.argv[1:]]))
if sys.argv[1:4] == ['-B', '-m', 'src.checkpoint_r2_fresh_pilot']:
    from huggingface_hub import get_token
    assert get_token() == 'synthetic-test-credential-never-a-real-token'
    assert os.environ['HF_HOME'] == os.environ['TEST_ROOT'] + '/hf-home'
    assert os.environ['HF_TOKEN_PATH'] == os.environ['TEST_ORIGINAL_TOKEN_PATH']
    assert not os.path.exists(os.environ['HF_HOME'] + '/token')
    with open(os.environ['TEST_OBSERVATION'], 'w') as f:
        json.dump({{'authentication_available': True, 'token_in_arguments': any('synthetic-test-credential' in a for a in sys.argv)}}, f)
    raise SystemExit(0)
raise AssertionError('unexpected shim invocation')
''')
    shim.chmod(0o755)
    for name, script in {
        'uname': '#!/bin/bash\necho Linux\n',
        'git': '#!/bin/bash\nif [[ $1 == rev-parse ]]; then echo ' + 'c' * 40 + '; fi\n',
        'python3.12': '#!/bin/bash\nmkdir -p "$3/bin"\ncp "$TEST_SHIM" "$3/bin/python"\n',
    }.items():
        p = bindir / name
        p.write_text(script)
        p.chmod(0o755)
    root = tmp_path / 'new-runtime'
    observation = tmp_path / 'observation.json'
    import huggingface_hub
    env = dict(PATH=str(bindir) + ':/usr/bin:/bin', HOME=str(tmp_path), HF_HOME=str(old_home),
               PYTHONPATH=str(Path(huggingface_hub.__file__).resolve().parent.parent),
               TEST_ROOT=str(root), TEST_SHIM=str(shim), TEST_ORIGINAL_TOKEN_PATH=str(token_file),
               TEST_OBSERVATION=str(observation))
    if authentication == 'custom_token_path':
        env['HF_TOKEN_PATH'] = str(token_file)
    elif authentication == 'environment':
        env['HF_TOKEN'] = 'synthetic-test-credential-never-a-real-token'
        token_file.unlink()
    elif authentication == 'legacy_environment':
        env['HUGGING_FACE_HUB_TOKEN'] = 'synthetic-test-credential-never-a-real-token'
        token_file.unlink()
    result = subprocess.run(['/bin/bash', '-x', str(ROOT / 'scripts/checkpoint_r2_fresh_gpu_setup.sh'),
                             str(root), 'c' * 40], env=env, cwd=tmp_path, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert 'synthetic-test-credential-never-a-real-token' not in result.stdout + result.stderr
    assert json.loads(observation.read_text()) == {'authentication_available':True,'token_in_arguments':False}
    assert not list(root.rglob('token'))


def test_peak_tracking_includes_transient_loading_allocations(tmp_path, monkeypatch):
    import transformers
    events, memory = [], {'live':0,'peak':0,'reserved':0}
    def reset(device):
        events.append('reset')
        assert device == 0
        memory.update(peak=memory['live'], reserved=memory['live'])
    flags = SimpleNamespace(allow_tf32=False, version=lambda:1)
    cuda_backends = SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False),
        enable_flash_sdp=lambda x:None, enable_mem_efficient_sdp=lambda x:None, enable_math_sdp=lambda x:None,
        flash_sdp_enabled=lambda:False,mem_efficient_sdp_enabled=lambda:False,math_sdp_enabled=lambda:True)
    fake_torch = SimpleNamespace(bfloat16='BF16', manual_seed=lambda x:None,
        set_float32_matmul_precision=lambda x:None, get_float32_matmul_precision=lambda:'highest',
        use_deterministic_algorithms=lambda x:None,are_deterministic_algorithms_enabled=lambda:False,
        backends=SimpleNamespace(cuda=cuda_backends,cudnn=flags),
        cuda=SimpleNamespace(set_device=lambda x:events.append('device'),reset_peak_memory_stats=reset,
            manual_seed_all=lambda x:None, max_memory_allocated=lambda:memory['peak'],
            max_memory_reserved=lambda:memory['reserved']))
    cfg = SimpleNamespace(num_hidden_layers=2,hidden_size=3,max_position_embeddings=32,_attn_implementation='sdpa')
    class Tokenizer:
        is_fast=True
        bos_token_id=1
        eos_token_id=2
        pad_token_id=None
        backend_tokenizer=SimpleNamespace(to_str=lambda:'{}')
    class Model:
        dtype='BF16'
        training=False
        config=cfg
        def to(self, device):
            events.append('transfer')
            # Loading/device transfer has a transient peak greater than steady state.
            memory.update(live=500,peak=max(memory['peak'],1200),reserved=max(memory['reserved'],1300))
            return self
        def eval(self):
            return self
        def parameters(self):
            return iter([SimpleNamespace(device='cuda:0')])
    def load_model(*args, **kwargs):
        events.append('load')
        memory.update(live=900,peak=max(memory['peak'],900),reserved=max(memory['reserved'],1000))
        return Model()
    monkeypatch.setattr(transformers, 'AutoConfig', SimpleNamespace(from_pretrained=lambda *a,**k:cfg))
    monkeypatch.setattr(transformers, 'AutoTokenizer', SimpleNamespace(from_pretrained=lambda *a,**k:Tokenizer()))
    monkeypatch.setattr(transformers, 'AutoModelForCausalLM', SimpleNamespace(from_pretrained=load_model))
    monkeypatch.setitem(sys.modules, 'torch', fake_torch)
    snapshot = tmp_path / ('a' * 40)
    snapshot.mkdir()
    backend = FreshBackend({'revision':'a'*40,'layers':2,'width':3},
                           {'snapshot':str(snapshot),'files':{}}, {'seed':0})
    assert events == ['device','reset','load','transfer']
    assert backend.peak() == {'allocated':1200,'reserved':1300}
    assert memory['live'] == 500
    # Later execution shares that interval and cannot erase the loading peak.
    memory.update(live=600,peak=max(memory['peak'],600),reserved=max(memory['reserved'],700))
    assert backend.peak() == {'allocated':1200,'reserved':1300}
