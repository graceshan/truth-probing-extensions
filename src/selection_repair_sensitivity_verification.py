"""Read-only receipt/export audit; stream hashes and headers, never score or fit."""
import hashlib
import json
import os
from pathlib import Path

from src import selection_repair_sensitivity_inputs as s
from src.selection_repair_cache_export_remote import header, token, DIRECTORIES


def audit_cache(path, expected, record):
    """Validate a selected layer against its verified source and exact row list."""
    s.require(record['status'] == 'verified', 'source verification unavailable')
    s.require(record['saved_layer_index'] == expected['layer']
              and record['original_row_indices'] == expected['row_indices']
              and record['row_identity_sha256'] == expected['row_identity_sha256'], 'row/layer export binding mismatch')
    source = record['source']
    s.require(all(source[k] == expected[k] for k in ('path','bytes','sha256')), 'source tensor identity mismatch')
    meta = source['header']
    s.require(meta['shape'] == expected['shape'] and meta['dtype'] == '<f2' and meta['fortran_order'] is False
              and meta['computed_file_bytes'] == source['bytes'], 'source header mismatch')
    s.require(source['stable_during_read'] is True and record['source_stable_through_export'] is True, 'unstable source')
    rows = expected['row_indices']
    s.require(rows and rows == sorted(set(rows)) and all(type(i) is int and 0 <= i < expected['shape'][0] for i in rows),
              'invalid original row indices')
    s.require(0 <= expected['layer'] < expected['shape'][1], 'invalid saved layer')
    s.require(set(record['companions']) == set(expected['companions']), 'companion set mismatch')
    for name, pin in expected['companions'].items():
        actual = record['companions'][name]
        s.require(actual['path'] == name and actual['stable_during_read'] is True
                  and all(actual[k] == pin[k] for k in ('bytes','sha256')), 'companion identity mismatch')
    path = Path(path)
    before = token(path.stat())
    with path.open('rb') as stream:
        local_header = header(stream)
        stream.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
        s.require(before == token(os.fstat(stream.fileno())) == token(path.stat()), 'local export changed during audit')
    actual = dict(bytes=path.stat().st_size, sha256=digest.hexdigest())
    exported = record['export']
    s.require(actual == {k:exported[k] for k in ('bytes','sha256')}, 'local export identity mismatch')
    s.require(local_header['shape'] == exported['shape'] == [len(rows),expected['shape'][2]]
              and local_header['computed_file_bytes'] == actual['bytes']
              and exported['dtype'] == '<f2' and exported['fortran_order'] is False, 'export header mismatch')
    return dict(status='verified',source=source,companions=record['companions'],export=exported,
                original_row_indices_sha256=s.sha(s.canonical(rows)),row_identity_sha256=record['row_identity_sha256'],
                saved_layer_index=record['saved_layer_index'],source_stable_through_export=True,local_export_stable=True)


def verify_exports(root, prepared, artifacts):
    root, prepared, artifacts = map(Path, (root, prepared, artifacts))
    _, inventory = s.verify_preparation(root, prepared)
    request_bytes = (prepared/'export_request.json').read_bytes()
    request = json.loads(request_bytes)
    staging_bytes = (artifacts/'staging_receipt.json').read_bytes()
    staging = json.loads(staging_bytes)
    remote_bytes = (artifacts/'export_receipt.json').read_bytes()
    remote = json.loads(remote_bytes)
    s.require(staging['remote_receipt'] == s.identity(remote_bytes), 'stale remote receipt')
    s.require(staging['preparation_receipt_sha256'] == s.sha((prepared/'preparation_receipt.json').read_bytes())
              and staging['export_request_sha256'] == remote['request_sha256'] == s.sha(request_bytes), 'mixed staging/preparation')
    s.require(staging['remote_program_sha256'] == s.sha((root/'src/selection_repair_cache_export_remote.py').read_bytes()),
              'exporter source mismatch')
    s.require(staging['local_platform'] == 'Darwin'
              and staging['remote_hostname'] == remote['remote_hostname'] == 'ef7f7534c328', 'host mismatch')
    s.require(all(remote[k] is False for k in ('model_extraction','final_test_access','remote_files_written'))
              and all(staging[k] is False for k in ('model_extraction','final_test_access','interactive_shell')), 'export scope mismatch')
    s.require(set(request['caches']) == set(remote['caches']) == set(staging['cache_status']) == set(DIRECTORIES),
              'cache set mismatch')
    results = {}
    for key, expected in request['caches'].items():
        record = remote['caches'][key]
        s.require(staging['cache_status'][key] == record['status'], 'staging/cache status mismatch')
        if record['status'] != 'verified':
            s.require(record['status'] == 'not_computed', 'unknown cache status')
            results[key] = record
            continue
        results[key] = audit_cache(artifacts/(key+'.npy'), expected, record)
    complete = all(r['status']=='verified' for r in results.values())
    s.require(staging['status'] == ('complete' if complete else 'partial'), 'staging completion mismatch')
    s.verify_preparation(root, prepared)
    s.require((artifacts/'export_receipt.json').read_bytes() == remote_bytes
              and (artifacts/'staging_receipt.json').read_bytes() == staging_bytes, 'receipts changed during audit')
    return dict(status='verified' if complete else 'partial',physical_identity_only=True,current_corrected_fit_authorized=False,
                local_platform=staging['local_platform'],remote_hostname=remote['remote_hostname'],
                adoption_sha256=inventory['adoption_sha256'],preparation_receipt=s.identity((prepared/'preparation_receipt.json').read_bytes()),
                export_request=s.identity(request_bytes),export_receipt=s.identity(remote_bytes),staging_receipt=s.identity(staging_bytes),
                caches=results,activation_values_inspected=False,real_fits=0)
