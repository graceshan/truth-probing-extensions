"""Actual frozen metadata, synthetic scores only; no cache or local-file dependency."""
import copy
import csv
from pathlib import Path

import numpy as np
import pytest

from src.checkpoint_r2_common_v1 import (
    ROOT, PACKAGE, FrozenBindings, binary, contract, local_preservation,
    membership_receipt, preservation_receipt, select_atomic_synthetic,
    sha, shared_atomic_bank,
)
from src.checkpoint_r2_selection_v1 import AtomicCandidate, TOPICS, candidate_id


@pytest.fixture(scope='module')
def frozen():
    return FrozenBindings()


def packet(block, scores=None):
    if scores is None:
        scores = [[r['surface_first_truth'], r['surface_second_truth']] for r in block['rows']]
    return dict(metadata_sha256=block['metadata_sha256'], ordered_keys=copy.deepcopy(block['ordered_keys']),
                score_kind='synthetic_validation', scores=np.asarray(scores, dtype=float))


@pytest.mark.parametrize('bad', ['False', 'True', 'false', '', '2', ' 1', '1.0', None, 0.0, 1.0, 2, float('nan')])
def test_strict_truth_rejects(bad):
    with pytest.raises(ValueError):
        binary(bad)


@pytest.mark.parametrize('value,expected', [('0', 0), ('1', 1), (False, 0), (True, 1), (0, 0), (1, 1)])
def test_strict_truth_accepts(value, expected):
    assert binary(value) == expected


def test_exact_memberships_and_preservation():
    receipt = membership_receipt()
    assert receipt['counts'] == dict(outer_train=3040, P15=2778, balanced=700, A15=75,
                                     atomic_D=1012, compound_D_pairs=483, TC=100)
    assert receipt['historical_fit_provenance'] == 'v4'
    assert receipt['conditional_P15_reuse_matches'] == 36
    assert preservation_receipt()['preserved_union_files'] == 1149


def test_portable_explicit_local_preservation(tmp_path):
    (tmp_path / 'fixture').write_bytes(b'original')
    expected = {'fixture': sha(b'original')}
    assert local_preservation(tmp_path, expected)['all_preserved']
    (tmp_path / 'fixture').write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash mismatch'):
        local_preservation(tmp_path, expected)


def test_frozen_input_mutation_rejected(tmp_path):
    # Shadow just one frozen input; all other paths remain read-only symlinks.
    import json
    cfg = contract()
    names = set(cfg['bindings']) | {'config/checkpoint_r2/common_integration_v1.json'}
    for name in names:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.symlink_to(ROOT / name)
    target = tmp_path / PACKAGE / 'memberships.json'
    target.unlink()
    data = json.loads((ROOT / PACKAGE / 'memberships.json').read_bytes())
    data['balanced'][0], data['balanced'][1] = data['balanced'][1], data['balanced'][0]
    target.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='hash mismatch.*memberships'):
        membership_receipt(tmp_path)


def test_adapter_counts_and_occurrences(frozen):
    receipt = frozen.receipt()
    assert receipt['rows'] == 25408
    assert receipt['repeated_example_occurrences'] == 5184
    a = frozen.block('constituent_source_validation', 'AND')
    assert len(a['rows']) == 240
    rows = [r for g in ('constituent_source_fit', 'TC_control_and_final_refit')
            for r in frozen.block(g, 'AND')['rows']]
    by_example = {}
    for r in rows:
        by_example.setdefault(r['example_id'], []).append(r)
    shared = next(v for v in by_example.values() if len(v) == 2)
    assert shared[0]['binding_id'] != shared[1]['binding_id']
    assert shared[0]['role'] != shared[1]['role']


def test_surface_swap_and_conditional_trap(frozen):
    b = frozen.block('constituent_source_validation', 'AND')
    for r in b['rows']:
        first, second = ('a', 'b') if r['ordering'] == 'AB' else ('b', 'a')
        assert r['surface_first_truth'] == r['canonical_truth_' + first]
        assert r['surface_second_truth'] == r['canonical_truth_' + second]
        assert r['surface_first_person_key'] == r['person_' + first]
        assert r['surface_first_fact_id'] == r['fact_' + first + '_id']
    ideal = packet(b)
    trap = packet(b, ideal['scores'] + 10 * ideal['scores'][:, ::-1])
    result = frozen.select_synthetic(b, {17: trap, 18: ideal}, source_operator='AND')
    assert result['selected_layer'] == 18
    curve = result['curves'][0]['macro']
    assert curve['first_marginal'] == curve['second_marginal'] == .75
    assert all(v == 1 for k, v in curve.items() if 'given' in k)
    assert result['target_or_D_used'] is False


@pytest.mark.parametrize('group', ['D_bare', 'constituent_source_fit', 'TC_control_and_final_refit', 'B25_11'])
def test_nonvalidation_roles_rejected(frozen, group):
    b = frozen.block(group, 'AND')
    with pytest.raises(ValueError, match='source-validation-only'):
        frozen.select_synthetic(b, {17: packet(b)}, source_operator='AND')


def test_target_and_score_alignment_guards(frozen):
    b = frozen.block('constituent_source_validation', 'AND')
    with pytest.raises(ValueError, match='target operator'):
        frozen.select_synthetic(b, {17: packet(b)}, source_operator='OR')
    for field, value, message in [
        ('metadata_sha256', 'old-version', 'metadata binding'),
        ('ordered_keys', b['ordered_keys'][::-1], 'ordered score alignment'),
        ('score_kind', 'research_scores', 'only synthetic'),
        ('scores', np.zeros((239, 2)), 'invalid score'),
        ('scores', np.full((240, 2), np.nan), 'invalid score'),
    ]:
        p = packet(b); p[field] = value
        with pytest.raises(ValueError, match=message):
            frozen.align(b, p)
    changed = copy.deepcopy(b); changed['rows'].reverse()
    with pytest.raises(ValueError, match='altered metadata'):
        frozen.align(changed, packet(b))
    # Same example IDs can occur in another group, but its binding is distinct.
    full = frozen.block('TC_control_and_final_refit', 'AND')
    source = frozen.block('constituent_source_fit', 'AND')
    with pytest.raises(ValueError, match='metadata binding'):
        frozen.align(full, packet(source))


@pytest.mark.parametrize('field,value', [
    ('truth_a', 'False'), ('truth_a', '1'), ('surface_first_truth', '1'),
    ('person_a', 'wrong-person'), ('fact_a_id', 'wrong-fact'),
    ('example_id', 'wrong-example'), ('pair_id', 'wrong-pair'),
    ('group', 'D_bare'), ('ordering', 'BA'), ('statement', 'substituted text'),
])
def test_corrupt_row_rejected(frozen, field, value):
    with (ROOT / PACKAGE / 'compound_bindings.csv').open() as f:
        row = next(csv.DictReader(f))  # first row FF/AND/AB
    row[field] = value
    with pytest.raises(ValueError):
        frozen.adapt_row(row)


def candidates(model):
    return [AtomicCandidate(candidate_id(model, layer, 'r0'), model, layer, 'r0', None,
                            True, .8, {t: .8 for t in TOPICS}) for layer in range(6)]


@pytest.mark.parametrize('model', ['qwen', 'llama'])
def test_shared_bank_and_prospective_permutation_tie(model):
    cs = candidates(model)
    bank = shared_atomic_bank(cs, model=model, reduced_lr_atomic_auc=.8)
    first = select_atomic_synthetic(cs, bank=bank, s_all_eligible_ids=bank['eligible_ids'])
    second = select_atomic_synthetic(cs[::-1], bank=bank, s_all_eligible_ids=bank['eligible_ids'][::-1])
    assert first == second
    expected = np.random.Generator(np.random.PCG64(20261005)).permutation(sorted(c.candidate_id for c in cs)).tolist()
    assert first['randomized_tie_order'] == expected
    assert first['selected_id'] == expected[0]
    assert not first['compound_inputs_used']
    with pytest.raises(ValueError, match='eligibility mismatch'):
        select_atomic_synthetic(cs, bank=bank, s_all_eligible_ids=bank['eligible_ids'][:-1])
    mutated = copy.deepcopy(bank); mutated['eligibility'][0]['eligible'] = False
    with pytest.raises(ValueError, match='stale shared'):
        select_atomic_synthetic(cs, bank=mutated, s_all_eligible_ids=bank['eligible_ids'])
