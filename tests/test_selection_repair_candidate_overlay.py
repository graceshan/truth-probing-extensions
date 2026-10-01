"""Source-only T2C checks; no activation, model, prediction or compound reads."""
import ast
import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from src import selection_repair_candidate_overlay as c

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def source():
    original, reviews, old = c.validate_inputs(ROOT)
    recs = c.rows(ROOT / c.REVIEW / 'recommended_exclusions.csv')
    manifest = c.apply_overlay(original, recs, reviews)
    return original, reviews, old, recs, manifest


def test_original_columns_hashes_and_order(source):
    original, _, _, _, manifest = source
    assert len(original) == len(manifest)
    for before, after in zip(original, manifest):
        assert all(after[k] == v for k, v in before.items())
        assert c.sha(after['statement'].encode()) == after['statement_sha256']


def test_paired_handling_and_reconciliation(source):
    original, _, _, _, manifest = source
    by_id = {r['source_row_id']: r for r in manifest}
    counts = Counter(r['candidate_status'] for r in manifest)
    assert counts == {'admitted': 5062, 'excluded': 36, 'quarantined': 114}
    assert sum(counts.values()) == len(original)
    for r in manifest:
        p = by_id[r['paired_source_row_id']]
        assert r['candidate_status'] == p['candidate_status']
        assert int(r['label']) + int(p['label']) == 1


def test_duplicate_restrictions_deduplicated(source):
    original, reviews, _, recs, manifest = source
    assert c.apply_overlay(original, recs + recs, reviews + reviews) == manifest


def test_old_exclusions_never_readmitted(source):
    for r in source[4]:
        if r['status'] == 'excluded':
            assert r['candidate_status'] != 'admitted'
            assert 't2a:' + r['exclusion_reasons'] in json.loads(r['candidate_reasons'])


def test_simjian_quarantine_and_farnsworth_grouping(source):
    manifest = source[4]
    simjian = [r for r in manifest if 'Simjian' in r['entity']]
    assert len(simjian) == 8
    assert {r['split'] for r in simjian} == {'train', 'validation'}
    assert {r['candidate_status'] for r in simjian} == {'quarantined'}
    assert len({r['person_key'] for r in simjian}) == 1
    farnsworth = [r for r in manifest if 'Farnsworth' in r['entity']]
    assert len({r['entity_id'] for r in farnsworth}) == 2
    assert len({r['person_key'] for r in farnsworth}) == 1
    assert {r['split'] for r in farnsworth} == {'train'}


def test_incomplete_identity_not_merged(source):
    kwolek = [r for r in source[4] if r['entity'] == 'Kwolek']
    assert kwolek and all(r['candidate_status'] == 'quarantined' for r in kwolek)
    assert all(r['person_key'] == r['entity_id'] for r in kwolek)


def test_t2b_error_vs_unresolved_carries(source):
    recs, manifest = source[3:]
    by_id = {r['source_row_id']: r for r in manifest}
    assert len(recs) == 52
    assert Counter(by_id[r['source_row_id']]['candidate_status'] for r in recs) == {'excluded': 4, 'quarantined': 48}


def test_fixed_sample_order_and_outcomes(source):
    old = source[2]
    assert [r['draw_index'] for r in old] == list(map(str, range(30)))
    assert Counter(r['judgment'] for r in old) == {'supported_false': 4, 'supported_true': 2, 'unresolved': 24}
    assert all(r['sample_denominator_contribution'] == 0 for r in source[1])


def test_no_screen_trigger_is_not_false_evidence(source):
    screen = c.load(ROOT / c.PACKAGE / 'screening_inventory.json')
    assert len(screen) == 140
    assert all(r['screen_status'] != 'supported_false' for r in screen)
    assert any(r['substantive_residence_screen_incomplete'] for r in screen)
    by_id = {r['source_row_id']: r for r in source[4]}
    assert all(by_id[r['source_row_id']]['candidate_status'] == 'admitted'
               for r in screen if r['screen_status'] == 'screened_no_trigger_in_bounded_material')


@pytest.mark.parametrize('field,value', [('judgment', 'false'), ('person_key', 'changed'),
    ('original_label', '1'), ('sample_denominator_contribution', 1), ('evidence_ids', ['missing'])])
def test_review_corruption_rejected(field, value):
    original_load = c.load
    def mutated(path):
        data = original_load(path)
        if path.name == 'targeted_reviews.json':
            data[0][field] = value
        return data
    with patch.object(c, 'load', mutated), pytest.raises(ValueError):
        c.validate_inputs(ROOT)


def test_missing_target_rejected():
    original_load = c.load
    def mutated(path):
        data = original_load(path)
        if path.name == 'targeted_reviews.json':
            data.pop()
        return data
    with patch.object(c, 'load', mutated), pytest.raises(ValueError, match='coverage incomplete'):
        c.validate_inputs(ROOT)


def test_capacity_tiers_and_spanish(source):
    facts, _ = c.inventory(source[4], source[1], source[2], [(p, c.rows(ROOT / p)) for p in c.REGISTRIES])
    people, capacity = c.capacity(source[4], facts)
    assert sum(p['audited_source_pair'] for p in people) == 3
    assert all(p['audited_positive_facts'] == 0 for p in people if p['topic'] != 'inventors')
    spanish = {r['split']: r for r in capacity if r['topic'] == 'sp_en_trans'}
    assert [spanish[s]['provisional_source_pair'] for s in ('train', 'validation', 'test')] == [4, 1, 0]
    assert [spanish[s]['positive_entity_upper_bound'] for s in ('train', 'validation', 'test')] == [106, 34, 34]
    assert spanish['validation']['positive_plus_historical_reviewed_negative'] == 34
    franklin = [f for f in facts if f['fact_ref'] == c.FRANKLIN]
    assert len(franklin) == 1 and not franklin[0]['eligible']
    assert franklin[0]['status'] == 'rejected_true_or_ambiguous'
    assert all(r['entities_10_capacity'] != 'audited_bound_only_pending_review' for r in capacity)


def test_counts_reconcile_by_label_form(source):
    counts = c.count_rows(source[4])
    lookup = {(r['topic'], r['split'], r['stage'], r['label'], r['form']): r['rows'] for r in counts}
    for topic, split in {(r['topic'], r['split']) for r in counts}:
        for form in ('all', 'affirmative', 'negated'):
            for label in ('all', '0', '1'):
                assert lookup[topic, split, 'original', label, form] == sum(
                    lookup[topic, split, stage, label, form] for stage in ('admitted', 'excluded', 'quarantined'))


def test_deterministic_build_and_no_overwrite(tmp_path):
    outputs, receipt = c.build(ROOT)
    second, second_receipt = c.build(ROOT)
    assert outputs == second and receipt == second_receipt
    directory = tmp_path / c.PACKAGE
    directory.mkdir(parents=True)
    with patch.object(c, 'build', return_value=(outputs, receipt)):
        c.run(tmp_path)
        c.run(tmp_path, check_only=True)
        path = directory / 'capacity.csv'
        path.write_text('corrupt')
        with pytest.raises(ValueError, match='Existing candidate differs'):
            c.run(tmp_path)
        assert path.read_text() == 'corrupt'


def test_unrelated_13_files_preserved(tmp_path):
    # Portable behavior test: mac 1's historical files are deliberately untracked.
    # The original baseline remains evidence, not a checkout dependency.
    baseline = {}
    for index in range(13):
        relative = f'unrelated/fixture_{index}.txt'
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(f'synthetic unrelated content {index}\n')
        baseline[relative] = c.sha(path.read_bytes())
    (tmp_path / c.PACKAGE).mkdir(parents=True)
    outputs = {'row_manifest.csv': b'synthetic_fixture_only\n'}
    with patch.object(c, 'build', return_value=(outputs, {'synthetic': True})):
        c.run(tmp_path)
        c.run(tmp_path, check_only=True)
    assert all(c.sha((tmp_path / p).read_bytes()) == digest for p, digest in baseline.items())


def test_historical_preservation_evidence_is_retained():
    baseline = c.load(ROOT / c.PACKAGE / 'preservation_baseline.json')
    assert baseline['baseline_head'] == '68f861cac10d0e631fc215e1f09210f6d5e9ca90'
    assert len(baseline['unrelated_untracked_sha256']) == 13
    # c.build / validate_inputs still hash every tracked historical input, including
    # preservation_baseline.json. Actual mac 1 files require an explicit local check.


def test_explicit_local_preservation_diagnostics(tmp_path):
    from scripts.check_selection_repair_local_preservation import check_preservation
    (tmp_path / 'good.txt').write_bytes(b'good')
    (tmp_path / 'changed.txt').write_bytes(b'changed')
    baseline = {'good.txt': c.sha(b'good'), 'changed.txt': c.sha(b'original'),
                'missing.txt': c.sha(b'missing')}
    result = check_preservation(tmp_path, baseline)
    assert result == {'matched': ['good.txt'], 'missing': ['missing.txt'], 'mismatched': ['changed.txt']}


def test_source_only_imports():
    tree = ast.parse((ROOT / 'src/selection_repair_candidate_overlay.py').read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imports <= {'csv', 'hashlib', 'io', 'json', 'collections', 'pathlib'}
