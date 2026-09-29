"""Synthetic scoring isolation and immutable artifact-chain tests."""
import builtins
import copy
import io
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src import clean_transfer_contracts as c
from src import pinned_compound_scoring as s
from src import clean_atomic_probes as old
from src import pinned_atomic_probes as adapter
from src.repaired_atomic_cache import RepairedAtomicCache
from clean_transfer_fixtures import artifacts, rewrite


def guard(root, monkeypatch, evaluation=False):
    allowed = {root / c.SPEC, root / c.PREFLIGHT}
    if evaluation:
        allowed.add(root / c.COMPOUND / 'metadata.csv')
    else:
        for directory, names in [(c.ATOMIC, c.ATOMIC_FILES), (c.PROBE, c.PROBE_FILES), (c.COMPOUND, c.COMPOUND_FILES)]:
            allowed.update(root / directory / n for n in names)
    opened = []
    for module in [builtins, io, os]:
        original = module.open
        def checked(path, *args, _original=original, **kwargs):
            if not isinstance(path, int):
                path = Path(os.fsdecode(path)).absolute()
                if path.is_relative_to(root):
                    assert path in allowed or path.is_relative_to(root / c.OUTPUT) or path.is_dir() or path.name.startswith('.'), path
                    opened.append(path)
                elif path.is_relative_to(c.ROOT):
                    assert path.suffix in ['.py', '.pyc'], path
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(module, 'open', checked)
    for module, name in [(old, 'AtomicData'), (old, 'load_frozen_probe'), (old, 'fit_and_save'),
                          (adapter, 'run')]:
        monkeypatch.setattr(module, name, Mock(side_effect=AssertionError('forbidden API ' + name)))
    return opened


def freeze(root):
    s.preflight(root)
    s.freeze_spec(root)


def test_full_truth_blind_scoring_and_finalize_last(artifacts, monkeypatch):
    root, frame, values = artifacts
    guard(root, monkeypatch)
    real_read = pd.read_csv
    reads = []
    def projected(path, *args, **kwargs):
        if str(path).endswith('/metadata.csv') and str(c.COMPOUND) in str(path):
            assert callable(kwargs.get('usecols'))
            for truth in ['compound_label', 'canonical_truth_a', 'canonical_truth_b', 'surface_first_truth', 'surface_second_truth', 'cell']:
                assert not kwargs['usecols'](truth)
            reads.append(path)
        return real_read(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', projected)
    real_load = np.load
    def no_activation_values_in_preflight(path, *args, **kwargs):
        assert str(path).endswith('selected_probe.npz')
        return real_load(path, *args, **kwargs)
    monkeypatch.setattr(np, 'load', no_activation_values_in_preflight)
    freeze(root)
    class SelectedLayerOnly:
        def __init__(self, array): self.array = array
        def __getitem__(self, key):
            assert len(key) == 3 and isinstance(key[0], slice) and key[1] == 17 and key[2] == slice(None)
            return self.array[key]
    def selected_layer_only(path, *args, **kwargs):
        loaded = real_load(path, *args, **kwargs)
        return SelectedLayerOnly(loaded) if str(path).endswith('activations.npy') else loaded
    monkeypatch.setattr(np, 'load', selected_layer_only)
    events = []
    real_publish = c.publish_json
    def publish(path, value):
        events.append(Path(path).name)
        if Path(path).name == 'scoring_manifest.json':
            assert (root / c.OUTPUT / 'row_scores.csv').exists()
            assert value['row_scores'] == c.record(root / c.OUTPUT / 'row_scores.csv')
        return real_publish(path, value)
    monkeypatch.setattr(c, 'publish_json', publish)
    s.score(root)
    assert events == ['representation_binding.json', 'scoring_manifest.json']
    table = real_read(root / c.OUTPUT / 'row_scores.csv', float_precision='round_trip')
    assert table.columns.tolist() == ['example_id', 'frozen_probe_score']
    np.testing.assert_array_equal(table.frozen_probe_score, s.affine(values[:, 17, :], np.array([.3, -.2, .7]), .1))
    assert table.example_id.tolist() == frame.example_id.tolist() and reads
    binding = c.read_json(root / c.OUTPUT / 'representation_binding.json')
    assert binding['producer_source_audit']['files']['src/pinned_atomic_probes.py']['recorded_sha256'] == 'a'*64
    with pytest.raises(ValueError, match='existing'):
        s.score(root)


def test_affine_equals_sklearn_and_no_fit_api():
    rng = np.random.default_rng(3)
    X, coef = rng.normal(size=(31, 5)).astype(np.float16), rng.normal(size=5)
    probe = LogisticRegression()
    probe.coef_, probe.intercept_, probe.classes_, probe.n_features_in_ = coef[None], np.array([.4]), np.array([0, 1]), 5
    np.testing.assert_array_equal(s.affine(X, coef, .4), probe.decision_function(X.astype(np.float64)))
    assert not hasattr(s, 'fit')


def test_refactored_adapter_hash_is_provenance_not_cache_identity(artifacts):
    root, _, _ = artifacts
    path = root / 'src/pinned_atomic_probes.py'
    path.parent.mkdir()
    path.write_text('# later adapter refactor\n')
    before = {name: c.record(root / c.PROBE / name) for name in c.PROBE_FILES}
    _, binding, _, _ = s.inspect_inputs(root)
    audit = binding['producer_source_audit']['files']['src/pinned_atomic_probes.py']
    assert audit['current_sha256'] == c.file_hash(path)
    assert audit['recorded_sha256'] == 'a'*64 != audit['current_sha256']
    assert before == {name: c.record(root / c.PROBE / name) for name in c.PROBE_FILES}


@pytest.mark.parametrize('failure', ['probe_hash', 'grid_hash', 'cache_identity', 'fingerprint', 'activation_hash',
                                     'metadata_hash', 'layer', 'C', 'replay', 'smoke', 'coverage', 'incomplete'])
def test_corruption_fails_before_scoring(artifacts, monkeypatch, failure):
    root, _, _ = artifacts
    probe_dir, compound = root / c.PROBE, root / c.COMPOUND
    selection, manifest, progress = [c.read_json(p) for p in [probe_dir / 'selection.json', compound / 'extraction_manifest.json', compound / 'progress.json']]
    if failure == 'probe_hash': selection['selected_probe_sha256'] = '0'*64
    elif failure == 'grid_hash': selection['validation_metrics_sha256'] = '0'*64
    elif failure == 'cache_identity': selection['structural']['repaired_cache_files']['completion.json']['sha256'] = '0'*64
    elif failure in ['layer', 'C']: selection['selected_layer' if failure == 'layer' else 'selected_C'] = 2
    elif failure == 'fingerprint': manifest['representation_fingerprint'] = '0'*64
    elif failure in ['activation_hash', 'metadata_hash']:
        path = compound / ('activations.npy' if failure == 'activation_hash' else 'metadata.csv')
        with path.open('ab') as handle: handle.write(b'corrupted')
    elif failure == 'replay': manifest['smoke_test']['atomic_replay']['saved_float16_byte_equal'] = False
    elif failure == 'smoke': manifest['smoke_test']['compound_smoke']['passed'] = False
    elif failure == 'coverage': progress['chunks'][0]['start'] = 1
    elif failure == 'incomplete': progress['complete'] = False
    rewrite(probe_dir / 'selection.json', selection)
    rewrite(compound / 'extraction_manifest.json', manifest)
    progress['identity_sha256'] = s.extraction_identity(manifest)
    rewrite(compound / 'progress.json', progress)
    monkeypatch.setattr(s, 'affine', Mock(side_effect=AssertionError('scoring in preflight')))
    with pytest.raises(ValueError):
        s.preflight(root)
    assert not (root / c.OUTPUT).exists()


@pytest.mark.parametrize('field,value', [('layer', np.array(22)), ('C', np.array(.01)), ('classes', np.array([1, 0]))])
def test_npz_layer_and_classes_mismatch(artifacts, field, value):
    root, _, _ = artifacts
    path = root / c.PROBE / 'selected_probe.npz'
    with np.load(path) as z: fields = {k: z[k] for k in z.files}
    fields[field] = value
    np.savez_compressed(path, **fields)
    selpath = root / c.PROBE / 'selection.json'
    selection = c.read_json(selpath)
    selection['selected_probe_sha256'] = c.file_hash(path)
    rewrite(selpath, selection)
    with pytest.raises(ValueError):
        s.inspect_inputs(root)


def test_production_descriptor_matches_unchanged_producer():
    from src.clean_atomic_extraction import contract
    from src.pinned_compound_extraction import representation_fields, fingerprint
    expected = representation_fields(contract())
    assert c.representation(contract(), expected['layer_convention']) == expected
    assert fingerprint(expected) == c.FINGERPRINT


def test_strict_spec_and_preflight_freeze(artifacts):
    root, _, _ = artifacts
    freeze(root)
    spec = c.read_json(root / c.SPEC)
    for key in ['unknown', 'scoring', 'bootstrap', 'inputs']:
        changed = copy.deepcopy(spec)
        if key == 'unknown': changed[key] = True
        elif key == 'inputs': changed[key]['compound']['files']['extra'] = {'sha256': 'a'*64, 'bytes': 1}
        else: changed[key]['unknown'] = True
        with pytest.raises(ValueError): c.validate_spec(changed)
    with pytest.raises(ValueError, match='overwrite'): s.freeze_spec(root)


def test_test_scope_and_symlink_blocked(artifacts):
    root, frame, _ = artifacts
    frame['split'] = 'test'
    path = root / c.COMPOUND / 'metadata.csv'
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError, match='test'): c.projected_metadata(path)
    with pytest.raises(ValueError): c.safe_path(root, '../test.csv')
    path.unlink()
    path.symlink_to(root / 'test.csv')
    with pytest.raises(ValueError, match='symlink'): s.inspect_inputs(root)


def test_failed_scoring_never_publishes_complete(artifacts, monkeypatch):
    root, _, _ = artifacts
    freeze(root)
    monkeypatch.setattr(s, 'affine', Mock(side_effect=ValueError('injected scoring failure')))
    with pytest.raises(ValueError): s.score(root)
    assert not (root / c.OUTPUT / 'scoring_manifest.json').exists()
