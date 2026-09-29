"""Synthetic transfer metrics and graph bootstrap only; no scientific scores."""
import copy

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import clean_transfer_contracts as c
from src import clean_transfer_evaluation as e
from src import pinned_compound_scoring as scoring
from src.clean_transfer_statistics import EntityBootstrap, MetricPlan, describe, interval, weighted_auc
from clean_transfer_fixtures import artifacts, graph, rewrite
from test_pinned_compound_scoring import freeze, guard


def analysis(frame=None, expected=None):
    if frame is None: frame, expected = graph()
    f = e.validate_metadata(frame, expected)
    # Labels are allowed in synthetic evaluation fixtures.
    f['frozen_probe_score'] = np.where(f.compound_label, 1., -1.)
    spec = c.spec_template({}, {})
    spec['benchmark'] = expected
    return f, spec


def plan_values(frame, spec, weights=None):
    plan = MetricPlan(frame, e.match_and_or(frame), spec)
    values = plan.evaluate(np.ones((1, len(frame))) if weights is None else weights)
    return plan, {r['metric_id']: values[:, i] for i, r in enumerate(plan.records)}


def test_perfect_primary_boundary_and_geometry_taxonomy():
    f, spec = analysis()
    plan, values = plan_values(f, spec)
    for key in ['and_auroc', 'or_auroc', 'and_tt_vs_mixed_auroc', 'or_mixed_vs_ff_auroc']:
        assert values['pooled/all/' + key][0] == 1
    assert values['pooled/all/and_minus_or_auroc'][0] == 0
    for row in plan.records:
        if '_geometry_' in row['metric']:
            assert row['category'] == 'geometry' and not row['primary']
    assert values['pooled/all/or_tt_vs_mixed_geometry_auroc'][0] == .5


def test_weak_or_boundary_threshold_and_balanced_accuracy():
    f, spec = analysis()
    f['frozen_probe_score'] = np.where(f.cell == 'TT', 2., 0.)
    _, values = plan_values(f, spec)
    assert values['pooled/all/or_mixed_vs_ff_auroc'][0] == .5
    assert values['pooled/all/and_accuracy'][0] == .25
    assert values['pooled/all/or_accuracy'][0] == .75
    assert values['pooled/all/and_balanced_accuracy'][0] == .5
    assert values['pooled/all/or_balanced_accuracy'][0] == .5
    assert values['pooled/all/and_ff_true_response_fraction'][0] == 1  # zero predicts true


def test_population_sd_linear_quantiles_and_topic_macro():
    summary = describe([0., 1., 2., 10.])
    assert summary['std'] == np.std([0., 1., 2., 10.], ddof=0)
    assert summary['q25'] == .75
    f, spec = analysis()
    f.loc[f.topic == 'animal_class', 'frozen_probe_score'] *= -1
    _, values = plan_values(f, spec)
    assert values['topic_macro/all/and_auroc'][0] == .8
    assert values['pooled/all/and_auroc'][0] != .8


@pytest.mark.parametrize('failure', ['duplicate', 'self_pair', 'degree', 'topic', 'label', 'surface_truth', 'fact'])
def test_structure_rejects_invalid_graph(failure):
    f, expected = graph()
    if failure == 'duplicate': f.loc[0, 'example_id'] = f.loc[1, 'example_id']
    elif failure == 'self_pair': f.loc[0, 'entity_b_id'] = f.loc[0, 'entity_a_id']
    elif failure == 'degree': expected['degree'] = 3
    elif failure == 'topic': f.loc[0, 'topic'] = 'unknown'
    elif failure == 'label': f.loc[0, 'compound_label'] = not f.loc[0, 'compound_label']
    elif failure == 'surface_truth': f.loc[0, 'surface_first_truth'] = not f.loc[0, 'surface_first_truth']
    elif failure == 'fact': f.loc[0, 'fact_a_id'] = 'wrong'
    with pytest.raises(ValueError): e.validate_metadata(f, expected)


def test_full_8384_graph_exact_matches_and_contrast_direction():
    f, expected = graph(dict(cities=149, sp_en_trans=34, inventors=45, element_symb=18, animal_class=16))
    f, _ = analysis(f, expected)
    f['frozen_probe_score'] = np.where(f.operator == 'OR', 3., 1.)
    matched = e.match_and_or(f)
    assert len(f) == 8384 and len(matched) == 4192
    assert matched.groupby('cell').size().to_dict() == {cell: 1048 for cell in c.CELLS}
    assert matched.delta_or_minus_and.eq(2.).all()
    with pytest.raises(ValueError): e.match_and_or(f.iloc[1:])


@pytest.mark.parametrize('weighted', [False, True])
def test_weighted_auc_sklearn_with_ties(weighted):
    rng = np.random.default_rng(17)
    scores, labels = rng.integers(0, 4, 100), rng.integers(0, 2, 100)
    weights = rng.integers(0, 5, 100) if weighted else np.ones(100)
    assert weighted_auc(labels, scores, weights) == pytest.approx(roc_auc_score(labels, scores, sample_weight=weights), abs=1e-15)
    assert np.isnan(weighted_auc(labels, scores, np.zeros(100)))


def test_endpoint_weights_permutation_and_paired_replicates():
    f, spec = analysis()
    f['frozen_probe_score'] += np.random.default_rng(4).normal(size=len(f))
    schedule = EntityBootstrap(f, spec['bootstrap'])
    allweights = schedule.row_weights(0, 2000)
    for pid, idx in f.groupby('pair_id').indices.items():
        assert np.all(allweights[:, idx] == allweights[:, idx[:1]])
    plan, values = plan_values(f, spec, allweights[:13])
    np.testing.assert_equal(values['pooled/all/and_minus_or_auroc'],
                            values['pooled/all/and_auroc'] - values['pooled/all/or_auroc'])
    shuffled = f.sample(frac=1, random_state=27).reset_index(drop=True)
    other = EntityBootstrap(shuffled, spec['bootstrap'])
    assert other.sha256 == schedule.sha256
    np.testing.assert_array_equal(other.weights, schedule.weights)
    _, shuffled_values = plan_values(shuffled, spec, other.row_weights(0, 13))
    for key in values: np.testing.assert_allclose(values[key], shuffled_values[key], atol=1e-14, rtol=1e-14, equal_nan=True)
    # Check multiplicity products directly, not just within-pair equality.
    pairs = f[['pair_id', 'topic', 'entity_a_id', 'entity_b_id']].drop_duplicates().set_index('pair_id')
    for col, pid in enumerate(schedule.pair_ids):
        row = pairs.loc[pid]
        entities = schedule.entities[row.topic]
        m = schedule.multiplicities[row.topic]
        np.testing.assert_array_equal(schedule.weights[:, col], m[:, entities.index(row.entity_a_id)] * m[:, entities.index(row.entity_b_id)])


def test_invalid_replicates_and_macro_all_topics_required():
    options = c.spec_template({}, {})['bootstrap']
    samples = np.r_[np.ones(1799), np.full(201, np.nan)]
    result = interval(samples, options)
    assert result['ci_status'] == 'insufficient_valid_replicates' and result['ci_low'] is None
    assert result['invalid_replicates'] == 201 and result['valid_fraction'] == 1799/2000
    samples[1799] = 1
    assert interval(samples, options)['ci_status'] == 'ok'
    f, spec = analysis()
    weights = np.ones((1, len(f)))
    weights[:, f.topic.eq('cities')] = 0
    _, values = plan_values(f, spec, weights)
    assert np.isnan(values['topic_macro/all/and_auroc'][0])
    assert np.isnan(values['topic/cities/and_auroc'][0])
    assert np.isfinite(values['pooled/all/and_auroc'][0])


def test_end_to_end_cpu_evaluation_only_allowed_files(artifacts, monkeypatch):
    root, _, _ = artifacts
    freeze(root)
    scoring.score(root)
    guard(root, monkeypatch, evaluation=True)
    result = e.evaluate(root)
    assert result['complete'] is True
    output = root / c.OUTPUT / 'evaluation'
    manifest = c.read_json(output / 'evaluation_manifest.json')
    assert manifest['activation_arrays_opened'] is False and manifest['probe_archives_opened'] is False
    assert sorted(p.name for p in output.iterdir()) == sorted(c.spec_template({}, {})['outputs']['evaluation_files'])
    boundary = pd.read_csv(output / 'boundary_metrics.csv')
    assert not boundary.metric.str.contains('geometry').any()
    with np.load(output / 'bootstrap_draws.npz') as z:
        assert z['values'].shape[0] == 2000
        np.testing.assert_array_equal(z['valid'], np.isfinite(z['values']))
    with pytest.raises(ValueError, match='existing'): e.evaluate(root)


@pytest.mark.parametrize('failure', ['duplicate', 'missing', 'wrong_id', 'metadata_hash', 'score_hash', 'incomplete', 'spec', 'truth_projection'])
def test_join_hash_failures_before_truth_read(artifacts, monkeypatch, failure):
    root, _, _ = artifacts
    freeze(root)
    scoring.score(root)
    out = root / c.OUTPUT
    manifest = c.read_json(out / 'scoring_manifest.json')
    table = pd.read_csv(out / 'row_scores.csv')
    if failure == 'duplicate': table.loc[0, 'example_id'] = table.loc[1, 'example_id']
    elif failure == 'missing': table = table.iloc[1:]
    elif failure == 'wrong_id': table.loc[0, 'example_id'] = 'not_present'
    elif failure == 'metadata_hash': manifest['inputs']['compound']['files']['metadata.csv']['sha256'] = '0'*64
    elif failure == 'score_hash': manifest['row_scores']['sha256'] = '0'*64
    elif failure == 'incomplete': manifest['complete'] = False
    elif failure == 'spec': manifest['analysis_spec_sha256'] = '0'*64
    elif failure == 'truth_projection': manifest['metadata_columns_materialized'].append('compound_label')
    if failure in ['duplicate', 'missing', 'wrong_id']:
        table.to_csv(out / 'row_scores.csv', index=False)
        manifest['row_scores'] = c.record(out / 'row_scores.csv')
    rewrite(out / 'scoring_manifest.json', manifest)
    original = pd.read_csv
    def no_labels(path, *args, **kwargs):
        if str(path).endswith('metadata.csv'):
            assert 'usecols' in kwargs, 'labels loaded before verification'
        return original(path, *args, **kwargs)
    monkeypatch.setattr(pd, 'read_csv', no_labels)
    with pytest.raises(ValueError): e.evaluate(root)
