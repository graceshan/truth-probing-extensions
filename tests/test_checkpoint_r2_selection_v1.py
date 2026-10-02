"""Synthetic Revision 2 policy checks; no research scores or arrays."""
import dataclasses
import itertools
import numpy as np
import pandas as pd
import pytest
from src import checkpoint_r2_selection_v1 as s


def candidate(layer, pooled=.95, macro=.9, model='qwen', valid=True, method='r0'):
    return s.AtomicCandidate(s.candidate_id(model, layer, method), model, layer, method, None, valid,
                             pooled, {t: macro for t in s.TOPICS})


def select(candidates, **kwargs):
    model = kwargs.pop('model', 'qwen')
    eligible = [r['candidate_id'] for r in s.atomic_eligibility(candidates, model, .95) if r['eligible']]
    return s.select_atom_all(candidates, model=model, reduced_lr_atomic_auc=.95,
                             expected_s_all_eligible_ids=eligible, **kwargs)


def source_fixture():
    rows = []
    for topic in s.TOPICS:
        for a, b, order in itertools.product([0, 1], [0, 1], ['AB', 'BA']):
            rows.append(dict(topic=topic, canonical_truth_a=a, canonical_truth_b=b, ordering=order,
                             operator='AND', example_id=f'{topic}-{a}-{b}-{order}'))
    frame = pd.DataFrame(rows)
    first, second = s.surface_labels(frame)
    return frame, first, second


def layer_select(frame, scores, **kwargs):
    return s.select_constituent_layer(source_operator='AND', role='T_C_source_validation',
                                      validation_binding='synthetic-only', rows=frame, scores_by_layer=scores, **kwargs)


def test_atomic_order_invariance_and_reproducible_independent_models():
    bank = [candidate(i) for i in range(6)]
    expected = select(bank)
    for seed in range(10):
        order = np.random.default_rng(seed).permutation(6)
        assert select([bank[i] for i in order]) == expected
    assert expected['randomized_tie_order'] == np.random.Generator(np.random.PCG64(20261005)).permutation(sorted(c.candidate_id for c in bank)).tolist()
    llama = [candidate(i, model='llama') for i in range(6)]
    assert select(llama, model='llama') == select(llama[::-1], model='llama')
    assert select(bank) == expected  # selecting another model cannot consume its RNG


def test_atomic_pooled_then_macro_tolerance_not_chained():
    bank = [candidate(0, .96, .99), candidate(1, .96 + 5e-13, .995), candidate(2, .96 + 1.4e-12, .98)]
    got = select(bank)
    assert bank[0].candidate_id not in got['pooled_tie_ids']
    assert got['selected_id'] == bank[1].candidate_id
    assert not got['compound_inputs_used']


def test_atomic_shared_eligibility_invalid_candidates_and_empty_bank():
    bank = [candidate(0), candidate(1, .944), candidate(2, valid=False), candidate(3, pooled=None)]
    got = select(bank)
    assert got['eligible_ids'] == [bank[0].candidate_id]
    assert all(r['reasons'] for r in got['eligibility'][1:])
    with pytest.raises(ValueError, match='S_all eligibility'):
        s.select_atom_all(bank, model='qwen', reduced_lr_atomic_auc=.95, expected_s_all_eligible_ids=[])
    assert select(bank[1:])['status'] == 'no_valid_candidate'
    bad = dataclasses.replace(bank[0], topic_atomic_auc={s.TOPICS[0]: .9})
    assert select([bad])['status'] == 'no_valid_candidate'
    with pytest.raises(ValueError, match='duplicate'):
        select([bank[0], bank[0]])
    with pytest.raises(TypeError):
        select([bank[0]], compound_scores=[999])


def test_original_r0_remains_separate_exact_tie_rule():
    bank = [candidate(7), candidate(2)]
    assert s.select_original_atomic_r0(bank, model='qwen')['selected_id'] == bank[1].candidate_id
    bank[0] = dataclasses.replace(bank[0], pooled_atomic_auc=.95 + 1e-14)
    assert s.select_original_atomic_r0(bank, model='qwen')['selected_id'] == bank[0].candidate_id


def test_conditional_trap_and_worst_of_six_selection():
    frame, first, second = source_fixture()
    trapped = np.column_stack([first + 10 * second, second + 10 * first])
    endpoints = s.constituent_endpoints(frame, trapped)
    assert endpoints['macro']['first_marginal'] == endpoints['macro']['second_marginal'] == .75
    assert all(endpoints['macro'][e] == 1 for e in s.ENDPOINTS[2:])
    ideal = np.column_stack([first, second])
    got = layer_select(frame, {1: trapped, 9: ideal})
    assert got['selected_layer'] == 9 and got['curves'][0]['minimum_macro'] == .75
    assert len(got['curves'][0]['macro']) == 6


def test_source_layer_order_invariance_ties_and_undefined_endpoints():
    frame, first, second = source_fixture();scores = np.column_stack([first, second])
    got = layer_select(frame, {5: scores, 2: scores})
    assert got == layer_select(frame, {2: scores, 5: scores})
    assert got['selected_layer'] == 2 and got['mean_tie_layers'] == [2, 5]
    missing = frame.topic != s.TOPICS[0]
    assert layer_select(frame.loc[missing], {2: scores[missing]})['status'] == 'no_valid_layer'
    bad = scores.astype(float);bad[0, 0] = np.nan
    invalid = layer_select(frame, {2: bad, 3: scores})
    assert invalid['selected_layer'] == 3 and invalid['curves'][0]['status'] == 'invalid_scores'
    only_true = first == 1
    assert not s.constituent_endpoints(frame.loc[only_true], scores[only_true])['valid']


def test_source_only_selection_cannot_use_target_or_D():
    frame, first, second = source_fixture();scores = np.column_stack([first, second])
    before = layer_select(frame, {1: scores})
    target = frame.copy();target.operator = 'OR'
    s.constituent_endpoints(target, -scores)  # poor target results do not change selection
    assert before == layer_select(frame, {1: scores})
    with pytest.raises(ValueError, match='target operator'):
        layer_select(target, {1: scores})
    with pytest.raises(ValueError, match='source-only'):
        s.select_constituent_layer(source_operator='AND', role='D', validation_binding='x', rows=frame, scores_by_layer={1: scores})
    with pytest.raises(TypeError):
        layer_select(frame, {1: scores}, target_scores=-scores)


def test_reversed_surface_truth_and_metadata_consistency():
    frame, first, second = source_fixture()
    reverse = frame.ordering == 'BA'
    np.testing.assert_array_equal(first[reverse], frame.loc[reverse].canonical_truth_b)
    np.testing.assert_array_equal(second[reverse], frame.loc[reverse].canonical_truth_a)
    bad = frame.copy();bad['surface_first_truth'] = 1 - first
    with pytest.raises(ValueError, match='surface truth'):
        s.surface_labels(bad)


def test_sample_aggregation_preserves_head_specific_metrics_and_undefined_draws():
    # Two heads can have perfect within-head ranking but incompatible offsets.
    from src.clean_transfer_statistics import weighted_auc
    y = np.array([0, 1]);a = np.array([0., 1.]);b = np.array([100., 101.])
    samples = {11: weighted_auc(y, a), 23: weighted_auc(y, b), 37: weighted_auc(y, a)}
    assert s.average_sample_metrics(samples) == 1
    assert weighted_auc(np.tile(y, 3), np.r_[a, b, a]) < 1
    draws = {11: [1., np.nan], 23: [.5, .5], 37: [.75, 1.]}
    np.testing.assert_allclose(s.average_sample_metrics(draws), [.75, np.nan], equal_nan=True)
    with pytest.raises(ValueError, match='sample'):
        s.average_sample_metrics({11: 1.})


def test_constituent_secondary_mean_before_lower_layer():
    frame, first, second = source_fixture()
    both_trapped=np.column_stack([first+10*second, second+10*first])
    first_trapped=np.column_stack([first+10*second, second])
    got=layer_select(frame,{1:both_trapped,9:first_trapped})
    assert got['minimum_tie_layers']==[1,9] and got['mean_tie_layers']==[9]
    assert got['selected_layer']==9


def test_bank_grid_schema_and_no_fabricated_results():
    from pathlib import Path
    from src import checkpoint_r2_cpu_preflight_v1 as prep
    from src import checkpoint_r2_result_schema_v1 as schemas
    cfg=prep.config(Path(__file__).resolve().parents[1]);grid=prep.grid(cfg)
    assert len(grid)==600 and sum(r['model']=='qwen' for r in grid)==280
    assert sum(r['method']=='l2_logistic' for r in grid)==300
    assert all(r['planned_rows']==700 for r in grid if r['method'] in ('burger_t_g','ttpd'))
    records=schemas.templates();assert records and all(schemas.validate(r) for r in records)
    bad=schemas.row(panel='b25_contrasts',status='unrun',metric_value=.99)
    with pytest.raises(ValueError,match='invented'):schemas.validate(bad)
    bad=schemas.row(panel='b25_contrasts',sample_seed='mean_11_23_37',aggregation='pooled_scores')
    with pytest.raises(ValueError,match='pooled heads'):schemas.validate(bad)
    bad=schemas.row(panel='source_validation_curves',endpoint='first_marginal',fit_stage='full_TC_refitted_20_per_topic')
    with pytest.raises(ValueError,match='leaked'):schemas.validate(bad)
