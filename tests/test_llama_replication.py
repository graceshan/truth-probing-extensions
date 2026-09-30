"""Synthetic only: no model downloads, scientific results, TEST or production arrays."""
import copy
from pathlib import Path
from types import SimpleNamespace
import subprocess

import numpy as np
import pandas as pd
import pytest

from src import clean_transfer_contracts as c
from src import llama_replication_contracts as l
from src import llama_replication_extraction as ex
from src import llama_replication_probes as pr
from src import llama_replication_transfer as tr
from src import llama_replication_evaluation as ev
from src import priority2_input_controls as p
from src.clean_transfer_statistics import EntityBootstrap
from src.clean_transfer_evaluation import validate_metadata
from src.clean_extraction import ExtractionWriter
from clean_transfer_fixtures import graph
from test_priority2_input_controls import raw_facts


def synthetic_pin():
    descriptor = l.representation('a'*40)
    return dict(schema_version=1, representation=descriptor, fingerprint=c.digest(c.canonical(descriptor)),
        config=dict(model_type='llama', architectures=['LlamaForCausalLM'], num_hidden_layers=l.LAYERS, hidden_size=l.WIDTH),
        files={name: dict(sha256='b'*64, bytes=10) for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json']},
        tokenizer_backend_sha256='c'*64, tokenizer_class='SyntheticTokenizer',
        special_tokens=dict(bos_token_id=1, eos_token_id=2, add_bos_token=True, add_eos_token=False))


def write_cache(root, stage, rows, binding, values):
    gate = dict(passed=True, policy='llama-unpadded-repeat-native-position-exact-v1', samples=len(ex.smoke_statements(stage, rows)),
                statements_sha256=c.ordered_hash(ex.smoke_statements(stage, rows)), batch_size=1, padding=False)
    manifest = dict(pin=l.read_pin(root), stage=stage, input_binding=binding,
        runtime=dict(torch='2.11.0', transformers='5.12.1', gpu='synthetic'),
        data=dict(**rows.hashes, metadata_columns=list(rows.frame), expected_activation_shape=list(values.shape)),
        code=dict(implementation_version='llama31-replication-v1', source_sha256={}), smoke_test=gate,
        atomic_replay=dict(passed=True, float16_exact=True, samples=4) if stage == 'transfer' else None)
    with ExtractionWriter(ex.cache_path(root, stage), rows, manifest) as writer:
        writer.append(0, rows.frame, values)
        writer.finalize()


@pytest.fixture
def suite(tmp_path, monkeypatch):
    root = tmp_path
    monkeypatch.setattr(l, 'LAYERS', 2)
    monkeypatch.setattr(l, 'WIDTH', 3)
    c.publish_json(root/l.PIN, synthetic_pin())
    monkeypatch.setattr(l, 'sources', lambda root: {'synthetic': 'd'*64})
    groups = {}
    rng = np.random.default_rng(4)
    for split in ['train', 'validation']:
        frame = pd.DataFrame([dict(example_id=f'{split}_{i}', dataset=topic, row_index=i,
            entity_id=f'{split}_{i}', statement=f'Synthetic {split} {i}.', topic=topic,
            form='negated' if form else 'affirmative', split=split, label=label)
            for i, (topic, form, label) in enumerate((t, f, y) for t in c.TOPICS for f in [0, 1] for y in [0, 1])])
        groups[split] = ex.Rows(frame)
    atomic_identity = {'synthetic_only': True}
    monkeypatch.setattr(ex, 'atomic_rows', lambda root: (groups, atomic_identity))
    for split, rows in groups.items():
        values = rng.normal(size=(len(rows.frame), l.LAYERS, l.WIDTH))
        values[:, :, 0] += rows.frame.label.to_numpy()[:, None]*3
        write_cache(root, split, rows, atomic_identity, values.astype(np.float16))
    return root, groups


def test_pin_architecture_and_fixed_policy():
    pin = synthetic_pin()
    assert l.validate_pin(pin) == pin
    assert l.discover(pin['config']) == (32, 4096)
    for revision in ['main', '', 'a'*39, '../test']:
        with pytest.raises(ValueError): l.representation(revision)
    for key, value in [('hidden_size', 3584), ('num_hidden_layers', 28), ('model_type', 'qwen2')]:
        bad = copy.deepcopy(pin)
        bad['config'][key] = value
        with pytest.raises(ValueError): l.validate_pin(bad)
    assert l.COUNTS == {p.RAW_ID: 8384, p.BOTH: 4192, p.JUX: 4192, p.ISO: 524}
    assert l.policy(c.ROOT)['bootstrap']['replicates'] == 2000
    assert l.policy(c.ROOT)['bootstrap']['minimum_valid'] == 1800


def test_loader_selector_readonly_verifier(suite, monkeypatch):
    root, groups = suite
    data = pr.AtomicData(root)
    with pytest.raises(PermissionError): data.partition('test')
    with pytest.raises(ValueError): ex.cache_path(root, '../test')
    assert pr.select(root, check_only=True)['activation_rows_read'] == dict(train=0, validation=0, test=0)
    monkeypatch.setattr(ex, 'transfer_rows', lambda root: pytest.fail('selection opened compound data'))
    selected = pr.select(root)
    assert selected['test_evaluated'] is False
    assert selected['selection_rule'] == pr.selector.SELECTION_RULE
    assert selected['activation_rows_read'] == dict(train=40, validation=40, test=0)
    with pytest.raises(FileExistsError): pr.select(root)
    monkeypatch.setattr(pr.selector, 'fit_converged_probe', lambda *a, **k: pytest.fail('verification refit'))
    selection, coef, intercept, _ = pr.verify_probe(root)
    _, sklearn_probe = pr.selector.load_frozen_probe(pr.AtomicData(root), root/l.PROBE)
    values = groups['validation'].frame.label.to_numpy()[:, None]*np.ones((20, 3))
    np.testing.assert_allclose(tr.affine(values, coef, intercept), sklearn_probe.decision_function(values), rtol=0, atol=1e-14)
    with (root/l.PROBE/'selected_probe.npz').open('ab') as handle: handle.write(b'bad')
    with pytest.raises(ValueError, match='hash mismatch'): pr.verify_probe(root)


def test_tiny_llama_native_positions_last_token_and_hidden_indexing():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.tokenization_utils_base import BatchEncoding
    from src.extract import last_real_token_indices, saved_hidden_states
    torch.manual_seed(4)
    config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                        num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=64)
    config._attn_implementation = 'sdpa'
    model = LlamaForCausalLM(config).eval()
    class Tokenizer:
        padded = False
        def __call__(self, text, **kwargs):
            assert len(text) == 1 and kwargs['padding'] is False and kwargs['truncation'] is False
            assert kwargs['add_special_tokens'] is True
            return BatchEncoding(dict(input_ids=torch.tensor([[1, 4, 5, 6]]),
                attention_mask=torch.tensor([[1, 1, 1, 0 if self.padded else 1]])))
    tokenizer = Tokenizer()
    assert ex.smoke(model, tokenizer, ['Synthetic one.', 'Synthetic two.'])['passed']
    values = ex.readout(model, tokenizer, 'Synthetic one.')
    with torch.inference_mode():
        direct = model(input_ids=torch.tensor([[1, 4, 5, 6]]), output_hidden_states=True, use_cache=False)
    expected = torch.stack([state[0, -1] for state in direct.hidden_states[1:]]).numpy()
    np.testing.assert_array_equal(values, expected)
    assert values.shape == (2, 16)
    assert last_real_token_indices(torch.tensor([[1, 1, 0], [0, 1, 1]])).tolist() == [1, 2]
    with pytest.raises(ValueError): saved_hidden_states([1, 2], 2)
    tokenizer.padded = True
    with pytest.raises(ValueError, match='no-padding'): ex.readout(model, tokenizer, 'Synthetic.')


def make_transfer(root, monkeypatch):
    policy = l.policy(c.ROOT)
    raw, benchmark = graph(dict(zip(c.TOPICS, [5]*5)))
    raw = raw_facts(raw)
    monkeypatch.setattr(c, 'ROWS', len(raw))
    monkeypatch.setattr(c, 'BENCHMARK', benchmark)
    tables, facts, mapping = p.variants(raw)
    frames = [table[p.STATEMENT_FIELDS] for key, table in tables.items()]
    frames.append(facts.rename(columns={'fact_key': 'example_id'}).assign(condition_id=p.ISO)[p.STATEMENT_FIELDS])
    statements = pd.concat(frames, ignore_index=True)
    contents = {p.RAW: raw, p.DATA+'/statements.csv': statements, p.DATA+'/constituent_map.csv': mapping,
        p.DATA+'/isolated_facts.csv': facts, p.DATA+'/scoring_index.csv': pd.concat(
            [table[['example_id', 'condition_id', 'base_example_id']] for table in tables.values()], ignore_index=True),
        **{p.DATA+'/'+key+'.csv': table for key, table in tables.items() if key in [p.BOTH, p.JUX]}}
    for path, frame in contents.items():
        (root/path).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(root/path, index=False)
    files = {name: c.record(root/name) for name in contents}
    def verify_data(root):
        for name, identity in files.items(): assert c.record(root/name) == identity
        return files
    monkeypatch.setattr(l, 'data_files', verify_data)
    monkeypatch.setattr(l, 'COUNTS', {p.RAW_ID: len(raw), p.BOTH: len(raw)//2, p.JUX: len(raw)//2, p.ISO: len(facts)})
    policy['benchmark'] = benchmark
    policy['conditions'] = l.COUNTS
    policy['bootstrap'] = {**policy['bootstrap'], 'replicates': 20, 'minimum_valid': 18}
    policy['schedule_sha256'] = EntityBootstrap(raw, policy['bootstrap']).sha256
    monkeypatch.setattr(l, 'policy', lambda root: policy)
    selection, _, _, structural = pr.verify_probe(root)
    rows, data = ex.transfer_rows(root)
    binding = dict(data=data, atomic=structural, probe={name: c.record(root/l.PROBE/name)
        for name in ['selection.json', 'selected_probe.npz', 'validation_metrics.csv']})
    values = np.random.default_rng(10).normal(size=(len(rows.frame), l.LAYERS, l.WIDTH)).astype(np.float16)
    write_cache(root, 'transfer', rows, binding, values)
    return raw, tables, mapping, policy


def test_end_to_end_truth_barrier_same_schedule_paired_contrasts(suite, monkeypatch):
    root, _ = suite
    pr.select(root)
    raw, tables, mapping, policy = make_transfer(root, monkeypatch)
    original_read = pd.read_csv
    forbidden = {'compound_label', 'canonical_truth_a', 'canonical_truth_b', 'surface_first_truth', 'surface_second_truth'}
    reads = []
    def truth_guard(path, *args, **kwargs):
        if str(path).startswith(str(root/'data')):
            projection = kwargs.get('usecols')
            assert projection and not forbidden.intersection(projection)
            reads.append(projection)
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', truth_guard)
    assert tr.preflight(root)['scores_computed'] == 0
    assert tr.freeze(root)['scores_computed'] == 0
    assert not (root/l.OUTPUT/'scores').exists()
    monkeypatch.setattr(l, 'committed', lambda *args: None)
    monkeypatch.setattr(pr.selector, 'fit_converged_probe', lambda *a, **k: pytest.fail('scoring refit'))
    events = []
    publish = c.publish_json
    def track(path, value):
        if path.name == 'scoring_manifest.json':
            assert (path.parent/'scores.csv').exists()
            events.append('manifest_last')
        return publish(path, value)
    monkeypatch.setattr(c, 'publish_json', track)
    assert tr.score(root)['scores_computed'] == sum(l.COUNTS.values())
    assert reads and events == ['manifest_last']
    with pytest.raises(FileExistsError): tr.score(root)
    monkeypatch.setattr(pd, 'read_csv', original_read)
    original_open = Path.open
    def no_arrays(path, *args, **kwargs):
        assert '/acts/' not in str(path) and '/atomic_probes_' not in str(path), 'evaluator opened model representation'
        return original_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', no_arrays)
    spec, frame, tables, mapping, scores, _ = ev.load(root)
    metrics, draws, schedule = ev.compute(frame, tables, mapping, scores, policy)
    assert schedule.sha256 == policy['schedule_sha256']
    assert metrics[metrics.condition_id == p.JUX].category.eq('geometry').all()
    assert metrics[metrics.condition_id == p.BOOLEAN].category.eq('threshold').all()
    indices = {value: i for i, value in enumerate(metrics.metric_id)}
    for contrast in policy['paired_contrasts']:
        for metric in contrast['metrics']:
            tail = '/pooled/all/'+metric
            left, right = [indices[contrast[k]+tail] for k in ['left', 'right']]
            diff = indices[contrast['left']+'_minus_'+contrast['right']+tail]
            np.testing.assert_allclose(draws[:, diff], draws[:, left]-draws[:, right], equal_nan=True, atol=0, rtol=0)
    shuffled = frame.sample(frac=1, random_state=3)
    metrics2, draws2, schedule2 = ev.compute(shuffled, tables, mapping, scores, policy)
    np.testing.assert_array_equal(draws, draws2)
    assert schedule.sha256 == schedule2.sha256
    assert ev.evaluate(root)['complete']
    with pytest.raises(ValueError, match='no overwrite'): ev.evaluate(root)


def test_composition_formulas_and_canonical_code_unchanged():
    continuous, boolean = ev.composition([-2, 0, 1, -1], [1, -1, 2, -2], ['AND', 'OR', 'AND', 'OR'])
    np.testing.assert_array_equal(continuous, [-2, 0, 1, -1])
    np.testing.assert_array_equal(boolean, [False, True, True, False])
    files = ['src/clean_atomic_probes.py', 'src/clean_atomic_extraction.py', 'src/extract.py',
        'src/clean_extraction.py', 'src/pinned_compound_scoring.py', 'src/priority2_input_controls.py',
        'src/priority2_evaluation.py', 'config/clean_protocol/pinned_qwen25_lr_transfer_v1.json',
        'config/clean_protocol/priority2_input_controls_v1.json']
    for name in files:
        before = subprocess.run(['git', 'show', '205b0344161f8d8b2505b6b181f163eb16af65e9:'+name],
                                cwd=c.ROOT, check=True, capture_output=True).stdout
        assert (c.ROOT/name).read_bytes() == before


@pytest.mark.parametrize('mutation', ['incomplete', 'wrong_layer', 'test_access', 'grid_missing', 'wrong_pin', 'wrong_activation'])
def test_production_binding_failures(suite, mutation):
    root, _ = suite
    pr.select(root)
    receipt_path = root/l.PROBE/'completion.json'
    receipt = c.read_json(receipt_path)
    selection_path = root/l.PROBE/'selection.json'
    selection = c.read_json(selection_path)
    if mutation == 'incomplete':
        receipt['complete'] = False
    elif mutation == 'wrong_layer':
        selection['selected_layer'] = 1-selection['selected_layer']
    elif mutation == 'test_access':
        selection['test_evaluated'] = True
    elif mutation == 'grid_missing':
        path = root/l.PROBE/'validation_metrics.csv'
        frame = pd.read_csv(path).iloc[:-1]
        frame.to_csv(path, index=False)
        selection['validation_metrics_sha256'] = c.file_hash(path)
        receipt['files']['validation_metrics.csv'] = c.record(path)
    elif mutation == 'wrong_pin':
        pin_path = root/l.PIN
        pin = c.read_json(pin_path)
        pin['representation']['model'] = 'Qwen/Qwen2.5-7B-Instruct'
        pin_path.write_bytes(c.canonical(pin))
    elif mutation == 'wrong_activation':
        with (root/l.ATOMIC/'train/activations.npy').open('ab') as handle: handle.write(b'bad')
    selection_path.write_bytes(c.canonical(selection))
    receipt['files']['selection.json'] = c.record(selection_path)
    receipt_path.write_bytes(c.canonical(receipt))
    with pytest.raises(ValueError): pr.verify_probe(root)


def test_committed_spec_required(tmp_path):
    path = tmp_path/l.SPEC
    path.parent.mkdir(parents=True)
    path.write_text('{}')
    with pytest.raises(ValueError, match='commit'): l.committed(tmp_path, l.SPEC)


def test_selection_tie_policy_and_no_qwen_layer_assumption():
    candidates = [dict(layer=layer, C=C, validation_overall_auroc=.9)
                  for layer in [1, 17, 31] for C in [1., .001]]
    assert min(candidates, key=pr.selector.selection_key)['layer'] == 1
    assert min(candidates, key=pr.selector.selection_key)['C'] == .001
    candidates[-1]['validation_overall_auroc'] = .95
    assert min(candidates, key=pr.selector.selection_key)['layer'] == 31


def test_production_atomic_counts_are_shared():
    assert ex.atomic.COUNTS == {'train': 3144, 'validation': 1040}
    assert ex.atomic.INPUT_DIGEST == '1faf186cb28b48311e85fd41921316fad9d06b3d28c9756aa5e8b381a950f1d6'
    assert set(ex.atomic.COUNTS) == {'train', 'validation'}
