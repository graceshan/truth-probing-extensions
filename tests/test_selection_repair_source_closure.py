"""Portable source-only closure tests; historical tests remain untouched."""
import copy
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from src import selection_repair_source_closure as c

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    p = ROOT / c.CLOSURE
    return (c.prior.rows(ROOT / c.prior.PACKAGE / 'row_manifest.csv'),
            c.prior.load(p / 'queue.json'), c.prior.load(p / 'decisions.json'), c.prior.load(p / 'evidence.json'))


def test_decisions_and_identity_preservation(inputs):
    manifest, queue, decisions, evidence = inputs
    c.validate_decisions(*inputs)
    result = c.apply_successor(manifest, decisions)
    for old, new in zip(manifest, result):
        assert all(new[k] == v for k, v in old.items())
    assert c.admitted_identity_bytes(manifest, 'candidate_status') == c.admitted_identity_bytes(result, 'successor_status')
    assert Counter(r['successor_status'] for r in result) == {'admitted': 5062, 'excluded': 32, 'quarantined': 118}
    assert Counter(r['split'] for r in result if r['successor_status'] == 'admitted') == {'train': 3048, 'validation': 1012, 'test': 1002}


def test_exact_four_paired_status_changes(inputs):
    result = c.apply_successor(inputs[0], inputs[2])
    changed = {r['source_row_id'] for r in result if r['candidate_status'] != r['successor_status']}
    assert changed == {'inventors:2', 'neg_inventors:2', 'inventors:234', 'neg_inventors:234'}
    by_id = {r['source_row_id']: r for r in result}
    for r in result:
        assert r['successor_status'] == by_id[r['paired_source_row_id']]['successor_status']
    assert all(by_id[d['source_row_id']]['successor_status'] != 'admitted' for d in inputs[2])


def test_aliases_and_unchanged_original_labels(inputs):
    result = c.apply_successor(inputs[0], inputs[2])
    simjian = [r for r in result if 'Simjian' in r['entity']]
    assert len(simjian) == 8 and {r['successor_status'] for r in simjian} == {'quarantined'}
    assert {r['split'] for r in simjian} == {'train', 'validation'}
    farnsworth = [r for r in result if 'Farnsworth' in r['entity']]
    assert len({r['person_key'] for r in farnsworth}) == 1
    assert len({r['entity_id'] for r in farnsworth}) == 2
    assert {r['split'] for r in farnsworth} == {'train'}
    assert all(r['label'] == '0' for r in result if r['source_row_id'] in c.EXPECTED_IDS)


@pytest.mark.parametrize('field,value', [('person_key', 'wrong'), ('original_label', '1'),
    ('paired_source_row_id', 'inventors:1'), ('split', 'validation'), ('statement_sha256', 'wrong'),
    ('previous_v1_judgment', 'unresolved'), ('successor_status', 'admitted'),
    ('sample_denominator_contribution', 1), ('evidence_ids', ['unknown'])])
def test_mutation_rejected(inputs, field, value):
    inputs[2][0][field] = value
    with pytest.raises(ValueError):
        c.validate_decisions(*inputs)


def test_incomplete_or_reordered_review_rejected(inputs):
    inputs[2].reverse()
    with pytest.raises(ValueError, match='coverage/order'):
        c.validate_decisions(*inputs)
    inputs[2].pop()
    with pytest.raises(ValueError, match='coverage/order'):
        c.validate_decisions(*inputs)


def test_unresolved_not_promoted_without_corroboration(inputs):
    d = inputs[2][0]
    d.update(judgment='supported_true', successor_status='excluded', classification='corroborated_label_conflict')
    with pytest.raises(ValueError, match='No residential corroboration'):
        c.validate_decisions(*inputs)


def test_duplicate_pair_rejected(inputs):
    with pytest.raises(ValueError, match='Duplicate paired'):
        c.apply_successor(inputs[0], inputs[2] + [copy.deepcopy(inputs[2][0])])


def test_missing_locator_rejected(inputs):
    inputs[3]['kapany_ucsc']['observations'][0]['locator'] = ''
    with pytest.raises(ValueError, match='source text/locator'):
        c.validate_decisions(*inputs)


def test_historical_hash_mutation_rejected():
    load = c.prior.load
    def changed(path):
        data = load(path)
        if path.name == 'historical_lock.json':
            data[next(iter(data))] = '0' * 64
        return data
    with patch.object(c.prior, 'load', changed), pytest.raises(ValueError, match='Historical file changed'):
        c.build(ROOT)


def test_disjoint_role_misinterpretation_rejected():
    load = c.prior.load
    def changed(path):
        data = load(path)
        if path.name == 'recommendation.json':
            data['role_policy']['T_C_may_overlap'] = []
        return data
    with patch.object(c.prior, 'load', changed), pytest.raises(ValueError, match='Role interpretation'):
        c.build(ROOT)


def test_deterministic_no_overwrite_and_capacity(tmp_path):
    outputs, receipt = c.build(ROOT)
    repeated, _ = c.build(ROOT)
    assert repeated == outputs
    assert receipt['v1_admitted_identity_sha256'] == receipt['v2_admitted_identity_sha256']
    assert receipt['classification_counts'] == {'corroborated_label_conflict': 5, 'unresolved_not_confirmed_error': 2}
    assert receipt['changed_rows'] == 4 and not receipt['production_default']
    with patch.object(c, 'build', return_value=(outputs, receipt)):
        c.run(tmp_path)
        c.run(tmp_path, check_only=True)
        old = c.prior.rows(ROOT / c.prior.PACKAGE / 'capacity.csv')
        new = c.prior.rows(tmp_path / c.SUCCESSOR / 'capacity.csv')
        for before, after in zip(old, new):
            for k in before:
                if not k.startswith(('excluded_', 'quarantined_')):
                    assert before[k] == after[k]
        spanish = next(r for r in new if r['topic'] == 'sp_en_trans' and r['split'] == 'validation')
        assert spanish['historical_external_negative_facts'] == '34'
        path = tmp_path / c.SUCCESSOR / 'row_manifest.csv'
        path.write_text('changed')
        with pytest.raises(ValueError, match='Existing closure output differs'):
            c.run(tmp_path)
        assert path.read_text() == 'changed'


def test_role_policy_and_sample_hashes():
    policy = c.prior.load(ROOT / c.CLOSURE / 'policy.json')
    assert 'May overlap P and A; excludes D/E' in policy['roles']['T_C']
    assert '15/topic pilot; 10 fallback; 20 only deliberate B=50' in policy['roles']['reservation_plan']
    lock = c.prior.load(ROOT / c.CLOSURE / 'historical_lock.json')
    for name in (c.prior.AUDIT / 'review_queue_30.csv', c.prior.REVIEW / 'reviews.json',
                 c.prior.PACKAGE / 'targeted_queue.json', c.prior.PACKAGE / 'targeted_reviews.json'):
        assert c.prior.sha((ROOT / name).read_bytes()) == lock[str(name)]
