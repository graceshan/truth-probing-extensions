"""Pinned metadata-only candidate correspondence; never an admitted production map."""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

LOCK = Path('config/clean_protocol/selection_repair_mapping_v1.json')
OVERLAY = Path('data/clean_protocol/selection_repair_v1/candidate_overlay_v1')
ORIGINAL = Path('data/clean_protocol/selection_repair_v1/source_audit_v1/row_manifest.csv')
REMOTE = '/workspace/truth-probing-artifacts/'
CACHES = {f'{model}/{split}': REMOTE + prefix + '/' + split
          for model, prefix in [('qwen', 'atomic_repair/qwen25_a09a354_bs1_bf16_v1'),
                                ('llama', 'llama31_replication_v1/atomic')]
          for split in ('train', 'validation')}
STATUS = 'provisional_not_for_production'
FIELDS = ('tensor_row_index', 'source_row_id', 'dataset', 'row_index', 'source_sha256',
          'source_row_sha256', 'statement_sha256', 'entity_id', 'person_key', 'split',
          'partition_role', 'label', 'candidate_status', 'allocation', 'mapping_status')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()


def identity(data):
    return {'bytes': len(data), 'sha256': sha(data)}


def parse_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode('utf-8'), newline='')))


def csv_bytes(rows):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


class MetadataReader:
    """Read bounded metadata files, refuse links/path escapes and detect changed inputs."""

    def __init__(self):
        self.receipts = {}

    def read(self, root, relative, expected=None):
        root = Path(root).resolve()
        relative = Path(relative)
        path = root / relative
        require(not relative.is_absolute() and '..' not in relative.parts, 'unsafe metadata path')
        require(path.suffix in ('.json', '.csv', '.py', '.md'), 'non-metadata file forbidden: ' + str(relative))
        require(path.resolve() == path and path.is_file(), 'missing or linked metadata: ' + str(relative))
        require(path.stat().st_size < 128 * 1024 * 1024, 'metadata size limit exceeded')
        data = path.read_bytes()
        actual = identity(data)
        if expected:
            require(all(actual[k] == expected[k] for k in actual if k in expected),
                    'stale/mixed metadata bytes: ' + str(relative))
        if path in self.receipts:
            require(actual == self.receipts[path], 'metadata changed during build: ' + str(relative))
        self.receipts[path] = actual
        return data

    def unchanged(self):
        for path, expected in self.receipts.items():
            require(identity(path.read_bytes()) == expected, 'metadata changed during build: ' + str(path))


def validate_manifest(manifest, originals, source_hashes):
    """Bind candidate rows to locked original-row JSON, hashes and unchanged partitions."""
    original_by_id = {r['source_row_id']: r for r in originals}
    require(len(original_by_id) == len(originals), 'duplicate original source-row identity')
    require(len({r['source_row_id'] for r in manifest}) == len(manifest), 'duplicate candidate source-row identity')
    require({r['source_row_id'] for r in manifest} == set(original_by_id), 'candidate/original identity coverage differs')
    for row in manifest:
        original = original_by_id[row['source_row_id']]
        require(all(row.get(k) == v for k, v in original.items()),
                'candidate changed historical fields: ' + row['source_row_id'])
        require(row['candidate_status'] in ('admitted', 'excluded', 'quarantined'), 'invalid candidate_status')
        require(row['candidate_manifest_status'] == 'candidate_pending_review_not_production', 'candidate approval/version differs')
        require(row['source_row_id'] == row['dataset'] + ':' + str(int(row['row_index'])), 'source-row ID/index mismatch')
        require(int(row['source_row']) == int(row['row_index']) + 1, 'one-based source-row mismatch')
        raw = json.loads(row['original_row_json'])
        require(sha(canonical(raw)) == row['source_row_sha256'], 'original source-row hash mismatch')
        require(raw['statement'] == row['statement'] and str(raw['label']) == row['label'], 'original row text/label mismatch')
        require(sha(row['statement'].encode()) == row['statement_sha256'], 'exact statement hash mismatch')
        require(row['source_sha256'] == source_hashes.get(row['source_file']), 'original source-file hash mismatch')
        require(row['partition_role'] == {'train': 'outer_train', 'validation': 'D', 'test': 'E'}[row['split']], 'partition role mismatch')
        require(bool(row['person_key']), 'missing person key')


def align_rows(manifest, sidecar, split):
    """Enumerate authoritative sidecar positions, never alternate exports or text joins."""
    require(split in ('train', 'validation'), 'only train/validation mapping allowed')
    selected = [r for r in manifest if r['split'] == split and r['candidate_status'] == 'admitted']
    originals = [r for r in manifest if r['split'] == split]
    lookup = defaultdict(list)
    issues = []
    for position, row in enumerate(sidecar):
        key = row['dataset'] + ':' + str(int(row['row_index']))
        lookup[key].append((position, row))
        if 'example_id' in row and row['example_id'] != key:
            issues.append({'kind': 'mismatched_example_id', 'source_row_id': key, 'tensor_row_index': position})
    for key, matches in lookup.items():
        if len(matches) != 1:
            issues.append({'kind': 'duplicate_sidecar_identity', 'source_row_id': key,
                           'tensor_row_indices': [p for p, _ in matches]})
    if len({r['source_row_id'] for r in manifest}) != len(manifest):
        issues.append({'kind': 'duplicate_candidate_identity'})
    original_ids = {r['source_row_id'] for r in originals}
    for key in sorted(set(lookup) - original_ids):
        issues.append({'kind': 'unexpected_sidecar_identity', 'source_row_id': key})
    mapped = []
    selected_ids = {r['source_row_id'] for r in selected}
    for row in originals:
        key = row['source_row_id']
        matches = lookup.get(key, [])
        if len(matches) != 1:
            issues.append({'kind': 'missing_binding' if not matches else 'ambiguous_binding',
                           'source_row_id': key, 'candidate_status': row['candidate_status']})
            continue
        position, actual = matches[0]
        mismatches = [name for name in ('dataset', 'row_index', 'statement', 'entity_id', 'topic', 'form', 'split', 'label')
                      if str(actual[name]) != str(row[name])]
        if sha(actual['statement'].encode()) != row['statement_sha256']:
            mismatches.append('statement_sha256')
        if mismatches:
            issues.append({'kind': 'mismatched_binding', 'source_row_id': key,
                           'tensor_row_index': position, 'fields': mismatches})
            continue
        if key in selected_ids:
            mapped.append({**{field: row[field] for field in FIELDS if field in row},
                           'tensor_row_index': position,
                           'allocation': 'outer_training_unallocated' if split == 'train' else 'D_unallocated',
                           'mapping_status': STATUS})
    mapped.sort(key=lambda r: r['tensor_row_index'])
    duplicate_text = defaultdict(list)
    for row in mapped:
        duplicate_text[row['statement_sha256']].append({'source_row_id': row['source_row_id'],
                                                       'tensor_row_index': row['tensor_row_index']})
    duplicates = [{'statement_sha256': key, 'distinct_source_rows': records}
                  for key, records in sorted(duplicate_text.items()) if len(records) > 1]
    failures = Counter(r['kind'] for r in issues)
    diagnostics = {'expected_rows_from_candidate_manifest': len(selected), 'mapped_rows': len(mapped),
                   'binding_failure_counts': {
                       'missing': failures['missing_binding'], 'ambiguous': failures['ambiguous_binding'],
                       'duplicate_identities': failures['duplicate_sidecar_identity'] + failures['duplicate_candidate_identity'],
                       'mismatched': failures['mismatched_binding'] + failures['mismatched_example_id'],
                       'unexpected': failures['unexpected_sidecar_identity']},
                   'partition_candidate_status_counts': dict(Counter(r['candidate_status'] for r in originals)),
                   'mapped_person_keys': len({r['person_key'] for r in mapped}),
                   'original_partition_rows': len(originals), 'sidecar_rows': len(sidecar),
                   'issue_counts': dict(Counter(r['kind'] for r in issues)), 'issues': issues,
                   'duplicate_statement_groups': duplicates,
                   'duplicate_statement_rows': sum(len(g['distinct_source_rows']) for g in duplicates),
                   'exact_coverage': len(mapped) == len(selected) and not issues}
    return mapped, diagnostics


def verify_physical_receipt(physical, session):
    require(physical['remote']['status'] == 'verified' and session['physical_identity_verification_passed'] is True,
            'physical verification receipt is pending or failed')
    require(set(physical['caches']) == set(physical['remote']['cache_results']) == set(physical['remote_requests']) == set(CACHES),
            'physical receipt must bind exactly four allowlisted caches')
    require(session['ssh']['remote_hostname'] == 'ef7f7534c328', 'unexpected saved remote host')
    for cache, directory in CACHES.items():
        recorded = physical['caches'][cache]
        request = physical['remote_requests'][cache]
        measured = physical['remote']['cache_results'][cache]
        tensor = measured['tensor']
        require(recorded['physical_status'] == recorded['local_binding_status'] == measured['status'] == 'verified', 'unverified cache')
        require(tensor['path'] == request['path'] == directory + '/activations.npy', 'tensor path outside allowlist')
        for key in ('bytes', 'sha256'):
            require(tensor[key] == request[key] == recorded['expected_tensor'][key], 'mixed tensor identity')
        header = tensor['header']
        require(header['shape'] == request['shape'] == recorded['expected_tensor']['shape'], 'mixed tensor shape')
        require(header['shape'][0] == recorded['rows'] and header['dtype'] == '<f2' and header['fortran_order'] is False
                and tensor['stable_during_read'] is True and header['computed_file_bytes'] == tensor['bytes'], 'invalid physical header/stability')
        companion_paths = {directory + '/metadata.csv'}
        if cache.startswith('qwen/'):
            parent = directory.rsplit('/', 1)[0]
            companion_paths |= {parent + '/completion.json', parent + '/extraction_manifest.json'}
        else:
            companion_paths |= {directory + '/progress.json', directory + '/extraction_manifest.json'}
        require(set(request['companions']) == set(measured['companions']) == companion_paths, 'mixed companion allowlist')
        for path, expected in request['companions'].items():
            actual = measured['companions'][path]
            require(actual['path'] == path and actual['stable_during_read'] is True
                    and all(actual[k] == expected[k] for k in ('bytes', 'sha256')), 'mixed companion receipt')


def build(root, payload_root):
    root, payload_root = Path(root).resolve(), Path(payload_root).resolve()
    reader = MetadataReader()
    lock_bytes = reader.read(root, LOCK)
    lock = json.loads(lock_bytes)
    require(lock['schema_version'] == 1 and lock['status'] == STATUS, 'unrecognized mapping lock')
    pinned = {key: reader.read(root, lock[key]['path'], lock[key])
              for key in ('overlay_receipt', 'physical_receipt', 'session_receipt')}
    candidate, physical, session = [json.loads(pinned[k]) for k in ('overlay_receipt', 'physical_receipt', 'session_receipt')]
    require(session['checker_report_files']['compatibility.json'] == identity(pinned['physical_receipt']), 'mixed session/physical receipts')
    verify_physical_receipt(physical, session)
    require(candidate['status'] == 'candidate_pending_review_not_production' and candidate['production_default'] is False, 'overlay version/adoption changed')
    candidate_inputs = {path: reader.read(root, path, {'sha256': digest}) for path, digest in candidate['inputs'].items()}
    candidate_outputs = {name: reader.read(root, OVERLAY / name, expected) for name, expected in candidate['outputs'].items()}
    manifest = parse_rows(candidate_outputs['row_manifest.csv'])
    originals = parse_rows(candidate_inputs[str(ORIGINAL)])
    source_hashes = {r['location']: r['sha256'] for r in physical['input_identities'] if r.get('origin') == 'checkout'}
    validate_manifest(manifest, originals, source_hashes)
    require(dict(Counter(r['candidate_status'] for r in manifest)) == candidate['row_counts']
            and len(manifest) == candidate['original_rows'], 'candidate receipt counts differ')
    require(parse_rows(candidate_outputs['admitted_rows.csv']) == [r for r in manifest if r['candidate_status'] == 'admitted'], 'candidate admitted subset differs')
    inventory_record = next(r for r in physical['input_identities'] if r['location'] == 'inventory.json')
    inventory = json.loads(reader.read(payload_root, 'inventory.json', inventory_record))
    by_remote = defaultdict(list)
    for record in inventory:
        by_remote[record['original_path']].append(record)
    restored = {}
    for cache, request in physical['remote_requests'].items():
        for path, expected in request['companions'].items():
            matches = by_remote[path]
            require(len(matches) == 1, 'missing/ambiguous restored companion: ' + path)
            record = matches[0]
            require(all(record[k] == expected[k] for k in ('bytes', 'sha256')), 'restored/live companion identities differ')
            restored[path] = reader.read(payload_root, record['archive_path'], expected)
    maps, coverage = {}, {}
    for cache, directory in CACHES.items():
        sidecar = parse_rows(restored[directory + '/metadata.csv'])
        maps[cache], coverage[cache] = align_rows(manifest, sidecar, cache.split('/')[1])
        expected = physical['caches'][cache]
        require(len(sidecar) == expected['rows'], 'sidecar/physical receipt row count mismatch')
        ids = [r['dataset'] + ':' + r['row_index'] for r in sidecar]
        # Match the saved verifier's ordered_hash: canonical JSON list, exact UTF-8.
        require(sha(canonical(ids)) == expected['ordered_source_row_ids_sha256'], 'authoritative source order mismatch')
        require(sha(canonical([r['statement'] for r in sidecar])) == expected['ordered_statement_sha256'], 'authoritative statement order mismatch')
        coverage[cache]['tensor_identity'] = physical['remote']['cache_results'][cache]['tensor']
        coverage[cache]['sidecar_identity'] = identity(restored[directory + '/metadata.csv'])
    for split in ('train', 'validation'):
        require(maps['qwen/' + split] == maps['llama/' + split], 'cross-model source/position/person correspondence differs')
    bindings = {'mapping_lock': identity(lock_bytes),
                **{key: {**identity(value), 'path': lock[key]['path']} for key, value in pinned.items()},
                'candidate_manifest': identity(candidate_outputs['row_manifest.csv']),
                'original_manifest': identity(candidate_inputs[str(ORIGINAL)]),
                'restored_inventory': {k: inventory_record[k] for k in ('bytes', 'sha256')},
                'reviewed_objective_commit': lock['reviewed_objective_commit'],
                'reviewed_overlay_commit': lock['reviewed_overlay_commit']}
    proposal = {'status': 'proposal_pending_adoption', 'representation_adopted': False,
                'correction_overlay_adopted': False, 'production_usable': False, 'bindings': bindings,
                'verified_facts': ['Saved physical receipts match the four allowlisted tensor identities and NPY headers.',
                                   'Restored companion bytes match the live companion hashes saved by physical verification.',
                                   'Candidate manifest and original-row metadata match the pinned receipt chain.'],
                'physical_measurement_time_utc': physical['completed_utc'],
                'new_remote_measurement_performed': False, 'models': {},
                'missing_evidence': physical['missing_evidence'][1:],
                'historical_producer_conflict': physical['historical_producer_conflict'],
                'extraction_source_hash_differences': physical['extraction_source_hash_differences'],
                'remaining_decisions': ['Record explicit adoption of the exact correction-overlay receipt.',
                                        'Record representation adoption against these exact model/tokenizer/rendering/readout/layer/dtype identities and provenance limitations.',
                                        'Approve future membership, held-out-pair removal and P/A allocation independently; train rows here are unallocated.']}
    for model in ('qwen', 'llama'):
        cache = model + '/train'; recorded = physical['caches'][cache]
        directory = CACHES[cache]
        manifest_path = (directory.rsplit('/', 1)[0] if model == 'qwen' else directory) + '/extraction_manifest.json'
        extraction = json.loads(restored[manifest_path])
        config_evidence = extraction['resolved_model'] if model == 'qwen' else extraction['pin']
        proposal['models'][model] = {'recorded_contract_not_replayed': recorded['representation'],
            'representation_fingerprint': recorded['representation_fingerprint'],
            'tokenizer_recorded_identity_not_replayed': recorded['tokenizer_recorded_identity'],
            'embedded_config_and_recorded_file_identities': config_evidence,
            'standalone_tokenizer_config_bytes_available': False,
            'execution_provenance_established': False,
            'saved_axis_index_convention': 'zero-based tensor layer axis j is HF hidden_states[j+1]; final saved layer post-final-RMSNorm',
            'physical_cache_identities': {key: {'tensor': physical['remote']['cache_results'][key]['tensor'],
                                               'companions': physical['remote']['cache_results'][key]['companions']}
                                          for key in CACHES if key.startswith(model + '/')}}
    outputs = {'representation_proposal.json': json_bytes(proposal)}
    exact = all(r['exact_coverage'] for r in coverage.values())
    # Failure receipts expose every alignment issue but emit no partial map files.
    if exact:
        outputs.update({cache.replace('/', '_') + '.provisional.csv': csv_bytes(rows) for cache, rows in maps.items()})
    report = {'schema_version': 1, 'status': STATUS if exact else 'alignment_failed',
              'metadata_alignment_passed': exact, 'production_usable': False,
              'correction_overlay_adopted': False, 'representation_adopted': False,
              'final_row_map': False, 'P_A_reserved': False,
              'admission_column': 'candidate_status', 'historical_status_used_for_admission': False,
              'bindings': bindings, 'cache_coverage': coverage,
              'excluded_scope': 'E has no cache mapping; no tensor/SSH/model/fit/prediction access',
              'alternate_method_export_offsets_used': False,
              'saved_physical_receipt_time_utc': physical['completed_utc'],
              'outputs': {name: identity(data) for name, data in outputs.items()},
              'code_sha256': {name: sha(reader.read(root, name)) for name in
                              ('src/selection_repair_candidate_mapping.py', 'scripts/48_map_selection_repair_candidates.py')}}
    outputs['mapping_report.json'] = json_bytes(report)
    reader.unchanged()
    return outputs, report


def run(root, payload_root, output_root, check_only=False):
    """Create a fresh report directory or verify every existing report byte; never overwrite."""
    root, payload_root, output_root = map(lambda p: Path(p).resolve(), (root, payload_root, output_root))
    require(not output_root.is_relative_to(payload_root.parent), 'output must be outside restored backup')
    outputs, report = build(root, payload_root)
    if check_only:
        require(output_root.is_dir() and {p.name for p in output_root.iterdir()} == set(outputs), 'output set differs')
        for name, data in outputs.items():
            require(not (output_root/name).is_symlink() and (output_root/name).read_bytes() == data, 'stale/mixed mapping output: ' + name)
    else:
        require(not output_root.exists(), 'output directory exists; no overwrite')
        output_root.mkdir(parents=True)
        for name, data in outputs.items():
            (output_root/name).write_bytes(data)
    return report


def require_production_ready(report):
    """This proposal-only module cannot grant adoption or create a production loader."""
    raise ValueError('Provisional candidate maps require separately recorded correction-overlay and representation adoption; production use is unavailable here.')
