"""Synthetic-only multi-method transfer and tamper tests; never reads real artifacts."""
import builtins
from contextlib import contextmanager
import copy
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import clean_transfer_contracts as c
from src import method_transfer_contracts as m
from src import pinned_atomic_method_suite as atomic_suite
from src import pinned_compound_scoring as lr_scoring
from src import clean_transfer_evaluation as lr_eval
from src import pinned_method_binding as binding
from src import pinned_method_scoring as scoring
from src import pinned_method_evaluation as evaluation
from src.clean_transfer_statistics import EntityBootstrap
from src.repaired_atomic_cache import RepairedAtomicCache
from test_pinned_atomic_method_suite import atomic
from clean_transfer_fixtures import graph


@pytest.fixture
def artifacts(atomic, monkeypatch):
    root, _, selection = atomic
    # Add real-schema tokenizer/config provenance to this synthetic atomic producer.
    cache = root / c.ATOMIC
    manifest = c.read_json(cache / 'extraction_manifest.json')
    manifest['resolved_model'].update(config_file_sha256='c'*64, tokenizer_config_file_sha256='d'*64, tokenizer_backend_sha256='e'*64)
    (cache / 'extraction_manifest.json').write_bytes(c.canonical(manifest))
    receipt = c.read_json(cache / 'completion.json')
    receipt['manifest_sha256'] = c.file_hash(cache / 'extraction_manifest.json')
    (cache / 'completion.json').write_bytes(c.canonical(receipt))
    data = RepairedAtomicCache(root)
    selection['structural'].update(data.structural)
    (root / c.PROBE / 'selection.json').write_bytes(c.canonical(selection))
    atomic_suite.run(root, suite='both')  # small-width synthetic fits ONLY
    frame, benchmark = graph(dict(zip(c.TOPICS, [5, 6, 7, 8, 9])))
    monkeypatch.setattr(c, 'ROWS', len(frame))
    monkeypatch.setattr(c, 'BENCHMARK', benchmark)
    original_template = c.spec_template
    def synthetic_template(inputs, descriptor):
        spec = original_template(inputs, descriptor)
        spec['bootstrap'].update(replicates=40, minimum_valid=36)
        return spec
    monkeypatch.setattr(c, 'spec_template', synthetic_template)
    directory = root / c.COMPOUND
    directory.mkdir(parents=True)
    frame.to_csv(directory / 'metadata.csv', index=False)
    X = np.random.default_rng(811).normal(size=(len(frame), 28, c.WIDTH)).astype(np.float16)
    np.save(directory / 'activations.npy', X)
    monkeypatch.setattr(c, 'ACTIVATION_SHA', c.file_hash(directory / 'activations.npy'))
    monkeypatch.setattr(c, 'METADATA_SHA', c.file_hash(directory / 'metadata.csv'))
    descriptor = atomic_suite.representation(data.manifest['contract'])
    compound = dict(representation=descriptor, representation_fingerprint=c.FINGERPRINT,
        execution_contract=data.manifest['contract'], resolved_model=data.manifest['resolved_model'],
        repaired_atomic_binding=dict(passed=True, repaired_cache_files=data.files,
            repaired_cache_identity_sha256=c.digest(c.canonical(data.files)), representation=descriptor, representation_fingerprint=c.FINGERPRINT),
        atomic_replay_sample_sha256='b'*64,
        smoke_test=dict(live_binding_passed=True,
            atomic_replay=dict(passed=True, saved_float16_byte_equal=True, sample_sha256='b'*64, batch_size=1, padding=False, policy='exact-repaired-atomic-float16-replay-v1'),
            compound_smoke=dict(passed=True, batch_size=1, padding=False, policy='unpadded-single-repeat-direct-exact-v1')),
        code=dict(implementation_version='synthetic', source_sha256={}),
        data=dict(expected_activation_shape=list(X.shape), number_of_examples=len(frame), sidecar_sha256=c.METADATA_SHA,
                  benchmark_sha256=c.METADATA_SHA, ordered_example_id_sha256=c.ordered_hash(frame.example_id)))
    (directory / 'extraction_manifest.json').write_bytes(c.canonical(compound))
    (directory / 'progress.json').write_bytes(c.canonical(dict(complete=True, next_row=len(frame),
        activation_file_sha256=c.ACTIVATION_SHA, identity_sha256=lr_scoring.extraction_identity(compound),
        chunks=[dict(start=0, stop=len(frame), ordered_example_id_sha256=c.ordered_hash(frame.example_id))])))
    lr_scoring.preflight(root)
    lr_scoring.freeze_spec(root)
    lr_scoring.score(root)
    lr_eval.evaluate(root)  # synthetic LR reference, not any real LR output
    monkeypatch.setattr(m, 'LR_SPEC_SHA', c.file_hash(root / c.SPEC))
    monkeypatch.setattr(m, 'LR_EVALUATION_SHA', c.file_hash(root / c.OUTPUT / 'evaluation/evaluation_manifest.json'))
    monkeypatch.setattr(m, 'LR_PRIMARY_SHA', c.file_hash(root / c.OUTPUT / 'evaluation/primary_metrics.csv'))
    return root, frame, X


def freeze(root):
    scoring.preflight(root)
    scoring.freeze_spec(root)


@contextmanager
def guard(monkeypatch, root, *, stage):
    """No prohibited paths; Stage 1 pandas metadata projection; Stage 2 no arrays/NPZ."""
    with monkeypatch.context() as patch:
        def checked(path):
            if isinstance(path, int) or not isinstance(path, (str, bytes, os.PathLike)): return
            p = Path(os.fsdecode(path)).absolute()
            if p.is_relative_to(root):
                name = str(p.relative_to(root))
                assert not any(x in p.parts for x in ['test', 'atomic_test', 'compound_test']), name
                assert not ('atomic_method_suite_v1' in name or 'atomic_method_matched_v1' in name), name
                if stage == 'evaluation':
                    assert not name.endswith(('.npy', '.npz')) or name.startswith(m.OUTPUT + '/evaluation/'), name
                    assert not any(name.startswith(x + '/') for x in [c.ATOMIC, c.PROBE, *m.SUITES.values()]), name
        for module in [builtins, io, os]:
            original = module.open
            def opened(path, *args, _original=original, **kwargs):
                checked(path)
                return _original(path, *args, **kwargs)
            patch.setattr(module, 'open', opened)
        read = pd.read_csv
        def read_csv(path, *args, **kwargs):
            if stage == 'scoring' and isinstance(path, (str, Path)) and Path(path) == root / c.COMPOUND / 'metadata.csv':
                assert kwargs.get('usecols') is not None, 'full compound truth dataframe materialized during scoring'
                projection = kwargs['usecols']
                if callable(projection):
                    assert all(not projection(k) for k in ['compound_label', 'canonical_truth_a', 'canonical_truth_b', 'surface_first_truth', 'surface_second_truth', 'cell'])
                else:
                    assert set(projection) <= set(c.PROJECTION)
            return read(path, *args, **kwargs)
        patch.setattr(pd, 'read_csv', read_csv)
        yield


def update_receipt(root, mode, name):
    path = root / m.SUITES[mode] / 'verification.json'
    value = c.read_json(path)
    value['outputs'][name] = c.record(path.parent / name)
    path.write_bytes(c.canonical(value))


@pytest.mark.parametrize('failure', ['archive', 'incomplete', 'burger', 'secondary', 'test', 'compound', 'matched_C', 'metrics'])
def test_strict_suite_failures(artifacts, failure):
    root, _, _ = artifacts
    mode = 'matched' if failure in ['burger', 'matched_C'] else 'faithful'
    directory = root / m.SUITES[mode]
    if failure == 'archive':
        with (directory / 'layers/layer_17.npz').open('ab') as handle: handle.write(b'bad')
    elif failure == 'burger':
        path = directory / 'burger_training_rows.csv'
        table = pd.read_csv(path)
        table.iloc[::-1].to_csv(path, index=False)
        update_receipt(root, mode, path.name)
    elif failure in ['secondary', 'matched_C']:
        path = directory / 'summary.json'
        summary = c.read_json(path)
        if failure == 'secondary': summary['comparisons']['secondary_method_selected_layer']['difference_of_means']['layer'] = 22
        else: summary['canonical_lr_C'] = .01
        path.write_bytes(c.canonical(summary))
        update_receipt(root, mode, path.name)
    elif failure == 'metrics':
        path = directory / 'validation_metrics.csv'
        frame = pd.read_csv(path)
        frame.loc[0, 'validation_overall_auroc'] = .123
        frame.to_csv(path, index=False, float_format='%.17g')
        update_receipt(root, mode, path.name)
    else:
        path = directory / 'verification.json'
        receipt = c.read_json(path)
        receipt[{'incomplete': 'complete', 'test': 'atomic_test_accessed', 'compound': 'compound_data_accessed'}[failure]] = failure != 'incomplete'
        path.write_bytes(c.canonical(receipt))
    with pytest.raises((ValueError, AssertionError)):
        scoring.preflight(root)
    assert not (root / m.SPEC).exists()
    assert not (root / m.OUTPUT).exists()


def test_full_workflow_truth_blind_reproduction_and_immutability(artifacts, monkeypatch):
    root, _, X = artifacts
    sources = {p: p.read_bytes() for base in [root / c.PROBE, root / c.OUTPUT, *(root / v for v in m.SUITES.values())]
               for p in base.rglob('*') if p.is_file()}
    sources[root / c.SPEC] = (root / c.SPEC).read_bytes()
    monkeypatch.setattr(atomic_suite, 'fit_layer', lambda *a, **k: pytest.fail('transfer performed fitting'))
    monkeypatch.setattr(atomic_suite, 'fit_matched_methods', lambda *a, **k: pytest.fail('transfer performed fitting'))
    with guard(monkeypatch, root, stage='scoring'):
        with monkeypatch.context() as patch:
            load = np.load
            def preflight_load(path, *args, **kwargs):
                assert not isinstance(path, Path) or path != root / c.COMPOUND / 'activations.npy'
                return load(path, *args, **kwargs)
            patch.setattr(np, 'load', preflight_load)
            freeze(root)
        with pytest.raises(ValueError, match='committed'):
            scoring.score(root)
        monkeypatch.setattr(m, 'committed_spec', lambda root: None)  # temp synthetic fixture has no Git repository
        result = scoring.score(root)
    assert result['rows'] == 15*c.ROWS
    spec = m.load(root)
    table = pd.read_csv(root / m.OUTPUT / 'method_scores.csv', float_precision='round_trip')
    assert list(table) == m.COLUMNS
    assert set(table.analysis_group) == set(m.GROUPS)
    assert set(table.method) == set(m.METHODS)
    assert len(table.groupby(['analysis_group', 'method'])) == 15
    for condition in spec['conditions']:
        archive = root / m.SUITES[condition['suite']] / condition['archive']
        with np.load(archive, allow_pickle=False) as params:
            prefix = condition['method'] + '__'
            expected = X[:, condition['atomic_layer'], :].astype(np.float64) @ params[prefix+'coef'] + params[prefix+'intercept']
        actual = table[(table.analysis_group == condition['analysis_group']) & (table.method == condition['method'])]
        np.testing.assert_allclose(actual.frozen_probe_score, expected, atol=1e-12, rtol=1e-12)
    # Output-writing NPZ is permitted; evaluation guard blocks *input* archive reads.
    with monkeypatch.context() as patch:
        original = np.load
        patch.setattr(np, 'load', lambda *a, **k: pytest.fail('evaluation opened an archive'))
        with guard(patch, root, stage='evaluation'):
            evaluation.evaluate(root)
    output = root / m.OUTPUT / 'evaluation'
    metrics = pd.read_csv(output / 'bootstrap_summary.csv')
    assert set(metrics.loc[metrics.category == 'threshold', 'method']) == {'l2_logistic', 'ttpd'}
    assert set(metrics.loc[metrics.metric.str.contains('geometry'), 'category']) == {'geometry'}
    assert len(metrics[metrics.category == 'paired_contrast']) == 9*5*7
    assert len(metrics[metrics.category == 'primary']) == 15*5*7
    assert not any('rank' in k or 'winner' in k for k in metrics.columns)
    with np.load(output / 'bootstrap_draws.npz', allow_pickle=False) as draws:
        lookup = {v: i for i, v in enumerate(draws['metric_ids'])}
        values = draws['values']
        for contrast in spec['paired_contrasts']:
            for metric in contrast['metrics']:
                suffix = 'pooled/all/' + metric
                np.testing.assert_equal(values[:, lookup[contrast['id']+'/'+suffix]],
                    values[:, lookup['/'.join(contrast['left'])+'/'+suffix]] - values[:, lookup['/'.join(contrast['right'])+'/'+suffix]])
        for group in m.GROUPS:
            for method in m.METHODS:
                prefix = group+'/'+method+'/'
                expected = np.mean([values[:, lookup[prefix+'topic/'+t+'/and_auroc']] for t in c.TOPICS], axis=0)
                np.testing.assert_allclose(values[:, lookup[prefix+'topic_macro/all/and_auroc']], expected, atol=1e-15, equal_nan=True)
    assert c.read_json(output / 'evaluation_manifest.json')['schedule_sha256'] == spec['lr_reference']['schedule_sha256']
    # Gate failures cannot be bypassed even when a file retains valid metric schema.
    primary = metrics.copy()
    target = (primary.analysis_group == m.GROUPS[0]) & (primary.method == 'l2_logistic') & (primary.metric == 'and_auroc')
    primary.loc[target, 'estimate'] += .1
    with pytest.raises(ValueError, match='primary reproduction'):
        evaluation.primary_gate(root, primary, spec)
    scores, frame, _ = evaluation.load_scores(root, spec)
    wrong_schedule = copy.deepcopy(spec)
    wrong_schedule['lr_reference']['schedule_sha256'] = '0'*64
    with pytest.raises(ValueError, match='schedule identity'):
        evaluation.compute(scores, frame, wrong_schedule)
    assert all(p.read_bytes() == payload for p, payload in sources.items())
    for fn in [scoring.preflight, scoring.freeze_spec, scoring.score, evaluation.evaluate]:
        with pytest.raises(ValueError, match='refusing'):
            fn(root)


def test_spec_strictness_lr_gates_and_row_integrity(artifacts, monkeypatch):
    root, _, _ = artifacts
    freeze(root)
    original = m.load(root)
    for mutate in [lambda s: s.update(extra=True), lambda s: s['bootstrap'].update(seed=7),
                   lambda s: s['threshold_applicability']['burger_t_g'].update(applicable=True),
                   lambda s: s['conditions'][0].update(atomic_layer=22),
                   lambda s: s['suites']['faithful']['files']['summary.json'].update(sha256='placeholder')]:
        changed = copy.deepcopy(original)
        mutate(changed)
        with pytest.raises(ValueError): m.validate(changed)
    with pytest.raises(ValueError, match='LR per-row'):
        scoring.lr_score_gate(np.array([0., 2.]), np.array([0., 2.1]), original['reproduction'])
    monkeypatch.setattr(m, 'committed_spec', lambda root: None)
    scoring.score(root)
    path = root / m.OUTPUT / 'method_scores.csv'
    frame = pd.read_csv(path, float_precision='round_trip')
    frame.loc[1, 'example_id'] = frame.loc[0, 'example_id']
    frame.to_csv(path, index=False, float_format='%.17g')
    mp = path.parent / 'scoring_manifest.json'
    manifest = c.read_json(mp)
    manifest['method_scores'] = c.record(path)
    mp.write_bytes(c.canonical(manifest))
    with pytest.raises(ValueError, match='duplicates'):
        evaluation.load_scores(root, original)


def test_parameter_schema_rejects_nonfinite_and_wrong_fused_readout(tmp_path, monkeypatch):
    monkeypatch.setattr(c, 'WIDTH', 2)
    p = dict(coef=np.array([1., 2.]), intercept=np.array(0.), t_g=np.array([1., 0.]), t_p=np.array([0., 1.]),
             dataset_names=np.asarray([f'd{i}' for i in range(10)]), dataset_means=np.zeros((10, 2)), ols_gram=np.eye(2),
             polarity_coef=np.array([0., 1.]), polarity_intercept=np.array([0.]), polarity_classes=np.array([0, 1]),
             head_coef=np.array([1., 2.]), head_intercept=np.array(0.), classes=np.array([0, 1]))
    path = tmp_path / 'synthetic.npz'
    def save(): np.savez(path, **{'ttpd__'+k: v for k, v in p.items()})
    save()
    resolved = binding.parameters(path, ['ttpd'])['ttpd']
    X = np.array([[2., 3.], [1., -1.]])
    np.testing.assert_array_equal(X @ resolved['coef'], (np.column_stack([X @ resolved['t_g'], X @ resolved['polarity_coef']]) @ resolved['head_coef']))
    p['coef'][0] = 3.
    save()
    with pytest.raises(ValueError, match='fused'): binding.parameters(path, ['ttpd'])
    p['coef'][0] = np.nan
    save()
    with pytest.raises(ValueError, match='nonfinite'): binding.parameters(path, ['ttpd'])


def test_schedule_policy_and_thresholds_are_predeclared():
    base = c.spec_template({}, {})
    assert base['bootstrap']['replicates'] == 2000 and base['bootstrap']['minimum_valid'] == 1800
    assert base['bootstrap']['seed'] == 1729
    assert all(m.THRESHOLDS[x]['applicable'] is (x in ['l2_logistic', 'ttpd']) for x in m.METHODS)
    assert len(m.contrasts()) == 9
    assert len([x for x in m.contrasts() if x['role'] == 'primary_method_contrast']) == 4
