"""Focused fresh-contract tests; fixtures are synthetic, never research observations."""
import copy
import json
import os
from pathlib import Path
import signal
import sys
import time
import numpy as np
import pytest
from src.checkpoint_r2_fresh_inputs import ROOT, build_manifest, file_hash, hash_value, text_hash
from src.checkpoint_r2_fresh_store import load_features, publish, storage_probe, write_json, write_shard
from src.checkpoint_r2_fresh_pilot import (Budget, accepted, calibrate, differences, exceeded, numerical,
                                         plan, run_model, semantic_view, supervise)


@pytest.fixture(scope='module')
def frozen():
    return plan()


def test_frozen_selection(frozen):
    cfg, manifest = frozen
    assert manifest == build_manifest()
    assert sum(cfg['raw_inventory']['fp16_bytes'].values()) == 16589860864
    assert cfg['raw_inventory']['unique_inputs_per_model'] == 35843
    cal, ver = manifest['stages']['calibration'], manifest['stages']['verification']
    assert len(cal) == 80 and len(ver) == 240
    assert all(r['split'] == 'train' for r in cal)
    assert not {r['id'] for r in cal} & {r['id'] for r in ver}
    for rows in (cal, ver):
        assert len({r['topic'] for r in rows}) == 5
        assert all(r['id'] == 'text_' + text_hash(r['statement']) for r in rows)
        assert all([b['inventory_offset'] for b in r['bindings']] == sorted(b['inventory_offset'] for b in r['bindings']) for r in rows)
    groups = {b['group'] for r in ver for b in r['bindings']}
    assert {'P15', 'atomic_D', 'D_bare', 'and_both_following_v1', 'or_explicit_or_both_v1', 'or_at_least_one_v1'} <= groups
    for topic in {r['topic'] for r in ver}:
        assert {'AND/AB', 'AND/BA', 'OR/AB', 'OR/BA'} <= {
            b['operator'] + '/' + b['ordering'] for r in ver if r['topic'] == topic for b in r['bindings']}
    # Identical text retains multiple logical uses; never just a canonical ID.
    assert any(len(r['bindings']) > 1 for r in cal + ver)


def row(identity, text='one two', split='train'):
    return dict(id=identity, statement=text, statement_sha256=text_hash(text), split=split,
                bindings=[dict(group='test', logical_id=identity, inventory_offset=0)])


def packet(rows):
    tokens = [[1] + list(range(2, len(r['statement'].split()) + 2)) for r in rows]
    size = max(map(len, tokens))
    return [dict(id=r['id'], statement_sha256=r['statement_sha256'], unpadded_token_ids=t,
                 raw_token_ids=t[1:], unpadded_special_tokens_mask=[1] + [0] * (len(t) - 1),
                 token_ids=t + [0] * (size - len(t)), attention_mask=[1] * len(t) + [0] * (size - len(t)),
                 position_ids=list(range(len(t))) + [0] * (size - len(t)), readout_index=len(t) - 1,
                 semantic_readout_position=len(t) - 1, pad_token_id=0, add_special_tokens=True,
                 truncation=False, chat_template=False) for r, t in zip(rows, tokens)]


@pytest.mark.parametrize('field,value', [('id','wrong'), ('readout_index',0), ('semantic_readout_position',0),
        ('attention_mask',[1,0,1]), ('position_ids',[0,2,1]), ('unpadded_token_ids',[999]),
        ('raw_token_ids',[999]), ('unpadded_special_tokens_mask',[0,0,0]),
        ('add_special_tokens',False), ('truncation',True), ('chat_template',True)])
def test_semantic_failures(field, value):
    rows = [row('x')]
    p = packet(rows)
    p[0][field] = value
    with pytest.raises(ValueError):
        semantic_view(p, rows)


def test_semantic_unpadding():
    rows = [row('x'), row('y', 'one two three four')]
    assert semantic_view(packet(rows), rows)[0] == semantic_view(packet(rows[:1]), rows[:1])[0]
    with pytest.raises(ValueError):
        semantic_view(packet(rows)[::-1], rows)
    with pytest.raises(ValueError):
        semantic_view(packet(rows)[:1], rows)


def test_layer_numerical_metrics():
    a = np.ones((4, 3, 2), dtype=np.float32)
    b = a.copy()
    b[:, 1] += .25
    result = numerical(a, b)
    assert result['max_absolute'] == [0., .25, 0.]
    assert result['rms_absolute'] == [0., .25, 0.]
    assert result['relative_rms'] == [0., .25, 0.]
    assert all(np.isfinite(numerical(np.zeros_like(a), a)['relative_rms']))
    with pytest.raises(ValueError):
        numerical(a, b[:, :2])
    b[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        numerical(a, b)


def test_computation_separate_from_storage():
    a = np.full((2, 2, 3), 1.0001, dtype=np.float32)
    b = a + np.float32(.0001)
    report = differences(a, b)
    assert max(report['computation']['max_absolute']) > 0
    assert max(report['stored_fp16']['max_absolute']) == 0
    assert max(report['reference_storage_effect']['max_absolute']) > 0


def test_calibration_lock_and_independent_failure(frozen):
    cfg, _ = frozen
    acceptance = cfg['numerical_acceptance']
    a = np.ones((4, 2, 3), dtype=np.float32)
    limits = calibrate(differences(a, a + .0001), acceptance)
    locked = copy.deepcopy(limits)
    assert accepted(differences(a, a + .0002), limits, acceptance)
    assert not accepted(differences(a, a + .0005), limits, acceptance)
    assert limits == locked
    with pytest.raises(ValueError):
        calibrate(differences(a, a + .02), acceptance)
    # Zero calibration gets a prospective floor, not universal bitwise equality.
    zero_limits = calibrate(differences(a, a), acceptance)
    assert zero_limits['computation']['max_absolute'] == [1e-5, 1e-5]
    assert not accepted(differences(a, a + .5), zero_limits, acceptance)


SPEC = dict(model_id='synthetic', revision='0'*40, layers=2, width=3)


def shard(tmp_path):
    rows = [row('x'), row('y', 'one two three')]
    values = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    provenance = dict(contract='test')
    write_shard(tmp_path / 'shard', rows, values, SPEC, provenance, packet(rows))
    return rows, values, provenance


def test_real_shard_reopen(tmp_path):
    rows, values, provenance = shard(tmp_path)
    loaded, restored, _ = load_features(tmp_path / 'shard', expected_rows=rows, expected_spec=SPEC,
                                        expected_provenance=provenance)
    assert np.array_equal(loaded, values.astype(np.float16))
    assert restored == rows
    with pytest.raises(ValueError):
        load_features(tmp_path / 'shard', expected_rows=rows[::-1])
    with pytest.raises(ValueError):
        load_features(tmp_path / 'shard', expected_provenance={'contract':'other'})
    with pytest.raises(FileExistsError):
        write_shard(tmp_path / 'shard', rows, values, SPEC, provenance, packet(rows))


@pytest.mark.parametrize('file', ['activations.npy','rows.json','tokens.json'])
def test_shard_corruption(tmp_path, file):
    shard(tmp_path)
    p = tmp_path / 'shard' / file
    p.write_bytes(p.read_bytes() + b'corruption')
    with pytest.raises(ValueError, match='shard hash'):
        load_features(tmp_path / 'shard')


@pytest.mark.parametrize('field,value', [('axes',['layer','input','feature']), ('final_layer','pre-norm'),
                                       ('saved_layers',[1,0]), ('ordered_ids',['y','x']), ('dtype','float32')])
def test_shard_receipt_semantics(tmp_path, field, value):
    shard(tmp_path)
    p = tmp_path / 'shard' / 'receipt.json'
    receipt = json.loads(p.read_text())
    receipt[field] = value
    p.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        load_features(tmp_path / 'shard')


def test_nonfinite_and_storage_overflow(tmp_path):
    for bad in (float('nan'), float('inf'), 70000.):
        values = np.full((1, 2, 3), bad, dtype=np.float32)
        with pytest.raises(ValueError):
            write_shard(tmp_path / str(bad), [row('x')], values, SPEC, {}, packet([row('x')]))


def test_storage_publication_and_orphan_recovery(tmp_path):
    result = storage_probe(tmp_path, 3*1024*1024)
    assert result['actual_write_and_readback_bytes'] == 3*1024*1024
    assert not list(tmp_path.iterdir())
    publish(tmp_path / 'x', lambda f: f.write(b'first'))
    with pytest.raises(FileExistsError):
        publish(tmp_path / 'x', lambda f: f.write(b'second'))
    assert (tmp_path / 'x').read_bytes() == b'first'
    assert len(list(tmp_path.glob('*.partial'))) == 1


def test_budget_accumulates_across_models(tmp_path):
    now = [10.]
    b = Budget(tmp_path, clock=lambda: now[0])
    now[0] += 20
    b.begin_model('qwen')
    now[0] += 30
    b.transition('active')
    now[0] += 40
    b.end_model()
    b.transition('setup')
    b.begin_model('llama')
    now[0] += 50
    b.end_model()
    assert b.snapshot()['gpu_seconds'] == {'qwen':70., 'llama':50.}
    assert b.snapshot()['active_seconds'] == 40.
    assert b.snapshot()['setup_seconds'] == 100.
    assert b.snapshot()['total_billable_wall_seconds'] == 140.
    state = b.snapshot()
    state.update(gpu_seconds={'qwen':3600.,'llama':0.})
    assert exceeded(state, now[0]) == 'one_gpu_hour_qwen'
    state.update(gpu_seconds={'qwen':0.,'llama':0.},active_seconds=7200.)
    assert exceeded(state, now[0]) == 'two_active_hours'


def test_supervisor_kills_hung_worker(tmp_path):
    now = time.monotonic()
    write_json(tmp_path / 'budget.json', dict(as_of_monotonic=now, phase='active',phase_seconds=0.,
                active_seconds=7199.8, gpu_seconds={'qwen':0.,'llama':0.},gpu_inflight=None))
    start = time.monotonic()
    assert supervise([sys.executable, '-c', 'import time; time.sleep(20)'], tmp_path) == 124
    assert time.monotonic() - start < 3
    assert json.loads((tmp_path / 'supervisor-exit.json').read_text())['cap_reason'] == 'two_active_hours'


class SyntheticBackend:
    def __init__(self, spec, snapshot, settings):
        self.spec = spec
        self.details = {'fixture':'synthetic'}
        self.calls = []

    def forward(self, rows, batch):
        self.calls.append((tuple(r['id'] for r in rows), batch))
        value = np.array([1 + int(text_hash(r['id'])[:6],16) / 16777216 for r in rows], dtype=np.float32)
        return np.broadcast_to(value[:,None,None], (len(rows), self.spec['layers'], self.spec['width'])).copy(), packet(rows)

    def peak(self):
        return {'allocated':0,'reserved':0}

    def close(self):
        pass


def test_full_pipeline_locks_before_verification_and_reopens(tmp_path, frozen):
    cfg, manifest = frozen
    cfg = copy.deepcopy(cfg)
    cfg['models']['qwen'] = SPEC
    write_json(tmp_path / 'launch.json', {'commit':'fixture'})
    class Witness(SyntheticBackend):
        def forward(self, rows, batch):
            if any(r['id'] in {v['id'] for v in manifest['stages']['verification']} for r in rows):
                assert (tmp_path / 'qwen' / 'locked-calibration.json').exists()
            return super().forward(rows, batch)
    result = run_model(tmp_path,'qwen',cfg,manifest,{'download_and_hash_seconds':0.,'downloaded_bytes':0},
                       {'gpu':'synthetic','packages':{}},Budget(tmp_path),Witness,None)
    assert result['input_count'] == 320 and result['accepted_policy']['batch_size'] == 8
    assert len(result['steady_state_inputs_per_second']) == 3
    assert result['projection']['dollar_raw_range'] is None
    assert result['projection']['behavior_chat_included'] is False
    assert load_features(tmp_path / 'qwen' / 'transfer-ready')[0].shape == (8,2,3)


def test_verification_failure_retains_batch1_without_widening(tmp_path, frozen):
    cfg, manifest = frozen
    cfg = copy.deepcopy(cfg)
    cfg['models']['qwen'] = SPEC
    write_json(tmp_path / 'launch.json', {'commit':'fixture'})
    ver_ids = {r['id'] for r in manifest['stages']['verification']}
    class FailingBatch(SyntheticBackend):
        def forward(self, rows, batch):
            values, tokens = super().forward(rows,batch)
            if batch > 1 and any(r['id'] in ver_ids for r in rows):
                values += np.float32(.01)
            return values, tokens
    result = run_model(tmp_path,'qwen',cfg,manifest,{'download_and_hash_seconds':0.,'downloaded_bytes':0},
                       {'gpu':'synthetic','packages':{}},Budget(tmp_path),FailingBatch,.5)
    assert result['accepted_policy']['batch_size'] == 1
    assert result['accepted_policy']['fallback']
    lock = json.loads((tmp_path / 'qwen' / 'locked-calibration.json').read_text())
    assert result['accepted_policy']['locked_calibration_sha256'] == file_hash(tmp_path/'qwen'/'locked-calibration.json')
    assert lock['tolerance_widening_allowed'] is False
    assert result['projection']['dollar_raw_range'][0] >= 0


@pytest.mark.parametrize('architecture', ['qwen2', 'llama'])
def test_actual_tiny_bf16_architecture_readout_and_final_norm(architecture):
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM, LlamaConfig, LlamaForCausalLM
    from src.checkpoint_r2_fresh_backend import readout_states, verify_layer_states
    config_class, model_class = ((Qwen2Config,Qwen2ForCausalLM) if architecture == 'qwen2'
                                else (LlamaConfig,LlamaForCausalLM))
    cfg = config_class(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                       num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    model = model_class(cfg).to(dtype=torch.bfloat16).eval()
    captured, handles = {}, []
    def hook(key):
        def capture(module, args, value):
            captured[key] = value[0] if isinstance(value,tuple) else value
        return capture
    for i, block in enumerate(model.model.layers):
        handles.append(block.register_forward_hook(hook(i)))
    handles.append(model.model.norm.register_forward_hook(hook('norm')))
    mask = np.array([[1,1,0,0],[1,1,1,1]], dtype=np.int64)
    with torch.inference_mode():
        output = model(input_ids=torch.tensor([[1,2,0,0],[1,2,3,4]]), attention_mask=torch.tensor(mask),
                       position_ids=torch.tensor([[0,1,0,0],[0,1,2,3]]), output_hidden_states=True,
                       use_cache=False, logits_to_keep=1)
    for handle in handles:
        handle.remove()
    verify_layer_states(output.hidden_states, captured, 2)
    features = readout_states(output.hidden_states, mask, {'layers':2,'width':16})
    assert features.shape == (2,2,16) and features.dtype == np.float32
    assert np.array_equal(features[0,-1], captured['norm'][0,1].float().numpy())
    assert np.array_equal(features[1,0], captured[0][1,3].float().numpy())
    bad = list(output.hidden_states)
    bad[-1] = captured[1]  # pre-final-norm residual is forbidden
    with pytest.raises(ValueError, match='post-RMSNorm'):
        verify_layer_states(bad,captured,2)
    with pytest.raises(ValueError, match='missing HF'):
        readout_states(output.hidden_states[:-1],mask,{'layers':2,'width':16})
    with pytest.raises(ValueError, match='dtype'):
        readout_states([s.float() for s in output.hidden_states],mask,{'layers':2,'width':16})
