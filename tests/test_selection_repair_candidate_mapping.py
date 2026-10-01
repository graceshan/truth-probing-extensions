"""Synthetic alignment failures and pinned receipt checks; no activation or SSH I/O."""
import ast
import copy
import json
from pathlib import Path

import pytest
from src import selection_repair_candidate_mapping as m

ROOT = Path(__file__).resolve().parents[1]


def fixture():
    manifest, originals, sidecar = [], [], []
    for index in range(3):
        raw = {'statement': 'Same exact statement.' if index < 2 else 'Other statement.', 'label': '1'}
        original = dict(source_row_id=f'cities:{index}', dataset='cities', row_index=str(index),
                        source_row=str(index + 1), source_file='data/cities.csv', source_sha256='f'*64,
                        original_row_json=m.canonical(raw).decode(), source_row_sha256=m.sha(m.canonical(raw)),
                        statement=raw['statement'], statement_sha256=m.sha(raw['statement'].encode()),
                        label='1', topic='cities', form='affirmative', entity_id='entity_a',
                        person_key='person_a', split='train', partition_role='outer_train', status='admitted')
        originals.append(original)
        manifest.append(dict(original, candidate_status='admitted' if index < 2 else 'quarantined',
                             candidate_manifest_status='candidate_pending_review_not_production'))
        sidecar.append({key: original[key] for key in
                        ('dataset', 'row_index', 'statement', 'entity_id', 'topic', 'form', 'split', 'label')})
    return manifest, originals, list(reversed(sidecar))


def test_authoritative_sidecar_order_candidate_status_and_duplicate_text():
    manifest, originals, sidecar = fixture()
    m.validate_manifest(manifest, originals, {'data/cities.csv': 'f'*64})
    mapped, diagnostic = m.align_rows(manifest, sidecar, 'train')
    assert diagnostic['exact_coverage'] and diagnostic['expected_rows_from_candidate_manifest'] == 2
    assert [(r['source_row_id'], r['tensor_row_index']) for r in mapped] == [('cities:1', 1), ('cities:0', 2)]
    assert diagnostic['duplicate_statement_rows'] == 2
    assert len(diagnostic['duplicate_statement_groups']) == 1
    assert all(r['person_key'] == 'person_a' and r['allocation'] == 'outer_training_unallocated' for r in mapped)
    assert all(r['source_row_sha256'] == manifest[int(r['row_index'])]['source_row_sha256'] for r in mapped)
    # Historical status is not an admission switch; candidate_status alone controls this join.
    manifest[0]['status'] = 'excluded'
    assert m.align_rows(manifest, sidecar, 'train')[0] == mapped
    manifest[1]['candidate_status'] = 'excluded'
    assert [r['source_row_id'] for r in m.align_rows(manifest, sidecar, 'train')[0]] == ['cities:0']


@pytest.mark.parametrize('field,value', [('statement', 'changed'), ('label', '0'), ('split', 'validation'),
                                        ('entity_id', 'other'), ('topic', 'inventors'), ('form', 'negated')])
def test_mismatched_sidecar_binding_is_explicit(field, value):
    manifest, _, sidecar = fixture()
    sidecar[1][field] = value
    mapped, diagnostic = m.align_rows(manifest, sidecar, 'train')
    assert not diagnostic['exact_coverage'] and len(mapped) == 1
    assert diagnostic['issue_counts']['mismatched_binding'] == 1
    assert field in diagnostic['issues'][0]['fields']


def test_missing_ambiguous_and_duplicate_identities_not_text_fallback():
    manifest, _, sidecar = fixture()
    _, missing = m.align_rows(manifest, sidecar[:-1], 'train')
    assert missing['issue_counts'] == {'missing_binding': 1}
    assert missing['issues'][0]['source_row_id'] == 'cities:0'
    _, ambiguous = m.align_rows(manifest, sidecar + [sidecar[-1]], 'train')
    assert ambiguous['issue_counts'] == {'duplicate_sidecar_identity': 1, 'ambiguous_binding': 1}
    _, duplicated = m.align_rows(manifest + [manifest[0]], sidecar, 'train')
    assert not duplicated['exact_coverage'] and duplicated['issue_counts']['duplicate_candidate_identity'] == 1
    sidecar[0]['row_index'] = '99'
    _, unexpected = m.align_rows(manifest, sidecar, 'train')
    assert unexpected['issue_counts'] == {'unexpected_sidecar_identity': 1, 'missing_binding': 1}


def test_llama_example_id_and_final_test_forbidden():
    manifest, _, sidecar = fixture()
    sidecar[0]['example_id'] = 'cities:999'
    assert m.align_rows(manifest, sidecar, 'train')[1]['issue_counts']['mismatched_example_id'] == 1
    with pytest.raises(ValueError, match='only train/validation'):
        m.align_rows(manifest, sidecar, 'test')


@pytest.mark.parametrize('field,value,message', [
    ('source_row_sha256', '0'*64, 'source-row hash'),
    ('statement_sha256', '0'*64, 'statement hash'),
    ('source_sha256', '0'*64, 'source-file hash'),
    ('source_row', '99', 'one-based'),
    ('source_row_id', 'cities:99', 'ID/index'),
    ('partition_role', 'P', 'partition role'),
    ('person_key', '', 'person key'),
])
def test_original_row_hash_source_identity_and_partition_validation(field, value, message):
    manifest, originals, _ = fixture()
    manifest[0][field] = originals[0][field] = value
    with pytest.raises(ValueError, match=message):
        m.validate_manifest(manifest, originals, {'data/cities.csv': 'f'*64})


def test_overlay_cannot_change_historical_person_or_drop_original_rows():
    manifest, originals, _ = fixture()
    manifest[0]['person_key'] = 'invented'
    with pytest.raises(ValueError, match='historical fields'):
        m.validate_manifest(manifest, originals, {'data/cities.csv': 'f'*64})
    with pytest.raises(ValueError, match='coverage'):
        m.validate_manifest(manifest[1:], originals, {'data/cities.csv': 'f'*64})
    with pytest.raises(ValueError, match='duplicate candidate'):
        m.validate_manifest(manifest + [manifest[0]], originals, {'data/cities.csv': 'f'*64})


@pytest.mark.parametrize('kind', ['overlay_receipt', 'physical_receipt', 'session_receipt'])
def test_pinned_version_bytes_reject_stale_or_mixed_inputs(tmp_path, kind):
    lock = json.loads((ROOT / m.LOCK).read_text())
    pin = lock[kind]
    raw = (ROOT / pin['path']).read_bytes()
    path = tmp_path / 'metadata.json'
    path.write_bytes(raw)
    reader = m.MetadataReader()
    assert reader.read(tmp_path, path.name, pin) == raw
    path.write_bytes(raw + b' ')
    with pytest.raises(ValueError, match='stale/mixed'):
        m.MetadataReader().read(tmp_path, path.name, pin)
    with pytest.raises(ValueError, match='changed during build'):
        reader.unchanged()


def physical_fixture():
    lock = json.loads((ROOT / m.LOCK).read_text())
    return tuple(json.loads((ROOT / lock[k]['path']).read_text()) for k in ('physical_receipt', 'session_receipt'))


@pytest.mark.parametrize('mutation', ['pending', 'hash', 'shape', 'dtype', 'order', 'stability', 'companion', 'test_path', 'extra_cache'])
def test_mixed_physical_measurement_rejected(mutation):
    physical, session = physical_fixture()
    m.verify_physical_receipt(physical, session)
    actual = physical['remote']['cache_results']['qwen/train']
    if mutation == 'pending':
        physical['remote']['status'] = 'pending'
    elif mutation == 'hash':
        actual['tensor']['sha256'] = '0'*64
    elif mutation == 'shape':
        actual['tensor']['header']['shape'][0] -= 1
    elif mutation == 'dtype':
        actual['tensor']['header']['dtype'] = '<f4'
    elif mutation == 'order':
        actual['tensor']['header']['fortran_order'] = True
    elif mutation == 'stability':
        actual['tensor']['stable_during_read'] = False
    elif mutation == 'companion':
        next(iter(actual['companions'].values()))['sha256'] = '0'*64
    elif mutation == 'test_path':
        actual['tensor']['path'] = actual['tensor']['path'].replace('/train/', '/test/')
    else:
        physical['remote_requests']['qwen/test'] = {}
    with pytest.raises(ValueError):
        m.verify_physical_receipt(physical, session)


def test_metadata_reader_forbids_tensor_paths_and_links(tmp_path):
    with pytest.raises(ValueError, match='non-metadata'):
        m.MetadataReader().read(tmp_path, 'activations.npy')
    (tmp_path / 'metadata.csv').write_text('safe')
    (tmp_path / 'link.csv').symlink_to(tmp_path / 'metadata.csv')
    with pytest.raises(ValueError, match='linked'):
        m.MetadataReader().read(tmp_path, 'link.csv')
    with pytest.raises(ValueError, match='unsafe'):
        m.MetadataReader().read(tmp_path, '../metadata.csv')


def test_output_checks_fail_closed_no_overwrite_or_production_adoption(tmp_path, monkeypatch):
    outputs = {'mapping_report.json': b'{"production_usable": false}\n'}
    report = {'production_usable': False}
    monkeypatch.setattr(m, 'build', lambda *_: (outputs, report))
    output = tmp_path / 'output'
    payload = tmp_path / 'backup' / 'payload'
    assert m.run(ROOT, payload, output) == report
    assert m.run(ROOT, payload, output, check_only=True) == report
    with pytest.raises(ValueError, match='exists'):
        m.run(ROOT, payload, output)
    (output / 'mapping_report.json').write_text('stale')
    with pytest.raises(ValueError, match='stale/mixed mapping'):
        m.run(ROOT, payload, output, check_only=True)
    assert (output / 'mapping_report.json').read_text() == 'stale'
    with pytest.raises(ValueError, match='adoption'):
        m.require_production_ready(report)


def test_mapper_imports_are_metadata_only():
    tree = ast.parse((ROOT / 'src/selection_repair_candidate_mapping.py').read_text())
    imports = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    imports |= {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imports <= {'csv', 'hashlib', 'io', 'json', 'collections', 'pathlib'}
