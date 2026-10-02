"""Synthetic checks of paired precision, graph support, and binding failures."""
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src import selection_repair_precision_proxy_v1 as p

OPTIONS = dict(replicates=2000, seed=1729, minimum_valid=1800)


def fixtures():
    atomic = pd.DataFrame(dict(source_row_id=list('abcdefgh'), topic=['t'] * 8,
                               person_key=np.repeat(list('ABCD'), 2), label=[0, 1] * 4))
    rows = []
    for i, (a, b) in enumerate([('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'A')]):
        for operator in ['AND', 'OR']:
            for cell in ['TT', 'TF', 'FT', 'FF']:
                for ordering in ['ab', 'ba']:
                    rows.append(dict(pair_id=str(i), topic='t', entity_a_id=a, entity_b_id=b,
                                     example_id=f'{i}-{operator}-{cell}-{ordering}', operator=operator,
                                     cell=cell, ordering=ordering,
                                     compound_label=int(cell == 'TT' if operator == 'AND' else cell != 'FF')))
    return atomic, pd.DataFrame(rows)


def test_weighted_auc_against_independent_pair_formula_and_sklearn():
    scores = np.array([2., -1., 2., 8., -10.])
    labels = np.array([0, 0, 1, 1, 1])
    weights = np.array([[1., 2., 3., 4., 0.], [3., 1., 1., 1., 5.]])
    fn = p.AUC(np.arange(5), scores, labels)
    for got, w in zip(fn(weights), weights):
        num = sum(w[i] * w[j] * ((scores[i] > scores[j]) + .5 * (scores[i] == scores[j]))
                  for i in np.flatnonzero(labels) for j in np.flatnonzero(1 - labels))
        expected = num / (w[labels == 1].sum() * w[labels == 0].sum())
        assert got == pytest.approx(expected, abs=1e-15)
        assert got == pytest.approx(roc_auc_score(labels, scores, sample_weight=w), abs=1e-15)


def test_pair_endpoint_product_and_variant_dependence_reproducible():
    atomic, bare = fixtures()
    w, receipt = p.schedules(atomic, bare, OPTIONS)
    w2, receipt2 = p.schedules(atomic, bare, OPTIONS)
    assert receipt == receipt2
    assert all(np.array_equal(w[k], w2[k]) for k in w)
    np.testing.assert_array_equal(w['atomic'][:, 0], w['atomic'][:, 1])
    for pair, group in bare.groupby('pair_id'):
        a, b = group.iloc[0][['entity_a_id', 'entity_b_id']]
        ia, ib = list('ABCD').index(a) * 2, list('ABCD').index(b) * 2
        expected = w['atomic'][:, ia] * w['atomic'][:, ib]
        for i in group.index:
            np.testing.assert_array_equal(w['bare'][:, i], expected)


def test_duplicate_all_surface_rows_does_not_change_auc_or_draws():
    atomic, bare = fixtures()
    weights, _ = p.schedules(atomic, bare, OPTIONS)
    scores = np.arange(len(bare)) % 7
    labels = bare.compound_label.to_numpy()
    a = p.evaluate(scores, labels, np.arange(len(scores)), weights['bare'])
    b = p.evaluate(np.tile(scores, 2), np.tile(labels, 2), np.arange(2 * len(scores)),
                   np.tile(weights['bare'], (1, 2)))
    assert a[0] == b[0]
    np.testing.assert_allclose(a[1], b[1], atol=0, rtol=0, equal_nan=True)


def test_identical_methods_have_zero_paired_uncertainty():
    atomic, bare = fixtures()
    weights, _ = p.schedules(atomic, bare, OPTIONS)
    scores = np.arange(len(atomic), dtype=float)
    point, draws = p.evaluate(scores, atomic.label.to_numpy(), np.arange(len(atomic)), weights['atomic'])
    ci = p.precision_interval(draws - draws, point - point, OPTIONS)
    assert ci['ci_low'] == ci['ci_high'] == ci['half_width'] == 0


def test_missing_class_and_empty_graph_replicates_preserved_without_redraw():
    fn = p.AUC(np.arange(2), np.array([0., 1.]), np.array([0, 1]))
    weights = np.array([[1, 1], [0, 1], [1, 0], [0, 0]])
    values = fn(weights)
    assert values[0] == 1 and np.isnan(values[1:]).all()
    ci = p.precision_interval(values, 1, dict(minimum_valid=3))
    assert ci['total_replicates'] == 4 and ci['valid_replicates'] == 1 and ci['invalid_replicates'] == 3
    assert ci['ci_low'] is None and ci['ci_status'] == 'insufficient_valid_replicates'


def test_macro_does_not_drop_undefined_topic_and_percentiles_are_linear():
    values = np.array([[.1, np.nan], [.3, .5]])
    np.testing.assert_allclose(p.macro(values), [.2, np.nan], equal_nan=True)
    ci = p.precision_interval(np.arange(100.) / 100, .5, dict(minimum_valid=90))
    assert ci['ci_low'] == pytest.approx(.02475)
    assert ci['ci_high'] == pytest.approx(.96525)
    assert ci['half_width'] == pytest.approx(.47025)


def test_rows_fail_on_identity_duplicates_and_incomplete_variants():
    atomic, bare = fixtures()
    p.validate_rows(atomic, bare, ['t'])
    with pytest.raises(ValueError, match='incomplete'):
        p.validate_rows(atomic, bare.iloc[:-1], ['t'])
    bad = atomic.copy(); bad.loc[1, 'source_row_id'] = 'a'
    with pytest.raises(ValueError, match='duplicate'):
        p.validate_rows(bad, bare, ['t'])
    bad = bare.copy(); bad.loc[0, 'compound_label'] = 0
    with pytest.raises(ValueError, match='label mismatch'):
        p.validate_rows(atomic, bad, ['t'])


def test_stale_hash_rejected_even_equal_bytes():
    expected = p.binding(b'abcd')
    assert p.check_bytes(b'abcd', expected, 'fixture') == b'abcd'
    with pytest.raises(ValueError, match='stale or mixed'):
        p.check_bytes(b'abdc', expected, 'fixture')


def test_graph_support_requires_actual_edges_and_atomic_counts_separate():
    _, bare = fixtures()
    graph = p.graph_summary(bare[['pair_id', 'topic', 'entity_a_id', 'entity_b_id']].drop_duplicates())
    assert graph['t']['degree_min'] == 2 and graph['t']['component_sizes'] == [4]
    diagnostics = dict(D_graph=graph, E_graph={'t':dict(all_degree_four=True)},
                       D_atomic={'t':dict(persons=7)}, E_atomic={'t':dict(persons=7)})
    assert not p.projection_supported('bare', 'topic', 't', diagnostics)[0]
    assert not p.projection_supported('bare', 'topic_macro', 'all', diagnostics)[0]
    assert p.projection_supported('atomic', 'pooled', 'all', diagnostics)[0]
    diagnostics['E_atomic']['t']['persons'] = 4
    assert not p.projection_supported('atomic', 'pooled', 'all', diagnostics)[0]


def test_independent_roc_ties_and_undefined_values():
    from src.selection_repair_precision_validation_v1 import roc_integral, linear_quantile
    scores = np.array([2., 2., -3., 10.])
    labels = np.array([1, 0, 0, 1])
    weights = np.array([[1., 3., 2., 4.], [0., 3., 2., 0.], [1., 0., 0., 4.]])
    got = roc_integral(labels, scores, weights)
    assert got[0] == pytest.approx(roc_auc_score(labels, scores, sample_weight=weights[0]))
    assert np.isnan(got[1:]).all()
    assert linear_quantile(np.array([3., 1., 2., 0.]), .25) == .75


def test_compute_uses_same_paired_draws_and_applies_count_scaling():
    atomic, bare = fixtures()
    weights, _ = p.schedules(atomic, bare, OPTIONS)
    config = dict(cohorts=['P10'], models={'toy': 0}, contrasts=['r0'], reference='l2_logistic',
                  topics=['t'], bootstrap=dict(OPTIONS, minimum_valid=1),
                  endpoints=[dict(id='or_mixed_vs_ff_auroc', kind='bare', operator='OR',
                                  positive=['TF', 'FT'], negative=['FF'], role='main')])
    diagnostics = dict(D_graph={'t':dict(all_degree_four=True, entities=16)},
                       E_graph={'t':dict(all_degree_four=True, entities=4)})
    x = np.arange(len(bare), dtype=float)
    scores = {('P10', 'toy', 'r0'):dict(bare=np.sin(x)),
              ('P10', 'toy', 'l2_logistic'):dict(bare=np.cos(x))}
    records, draws = p.compute(config, dict(atomic=atomic, bare=bare), scores, weights, diagnostics)
    native = next(r for r in records if r['scope'] == 'topic' and r['precision_mode'] == 'native_D')
    projected = next(r for r in records if r['scope'] == 'topic' and r['precision_mode'] != 'native_D')
    center = native['delta_D_proxy']
    assert projected['delta_D_proxy'] == center
    assert projected['transport_scale'] == 2
    np.testing.assert_allclose(draws[:, projected['draw_column']],
                               center + 2 * (draws[:, native['draw_column']] - center), atol=0, rtol=0)
    assert projected['half_width'] == pytest.approx(2 * native['half_width'])
    assert sum(r['status'] == 'unsupported' for r in records) == 2
