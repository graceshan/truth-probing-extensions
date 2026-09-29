"""Pinned extraction orchestration; imports GPU dependencies only when invoked."""
from pathlib import Path
import csv

import pandas as pd
import numpy as np

from src import clean_transfer_contracts as c
from src import priority2_input_controls as p
from src.repaired_atomic_cache import RepairedAtomicCache, array_header

EXTRACTION_FILES = ['extraction_manifest.json', 'progress.json', 'metadata.csv', 'activations.npy']


class Statements:
    """Truth-free metadata adapter for the existing durable ExtractionWriter."""
    def __init__(self, path):
        self.path = Path(path)
        self.payload = self.path.read_bytes()
        with self.path.open(encoding='utf-8', newline='') as handle:
            c.require(next(csv.reader(handle)) == p.STATEMENT_FIELDS, 'truth-free statement header required')
        self.frame = pd.read_csv(path, usecols=p.STATEMENT_FIELDS, dtype=str, keep_default_na=False)
        c.require(list(self.frame.columns) == p.STATEMENT_FIELDS and self.frame.example_id.is_unique and
                  len(self.frame) > 0 and self.frame.statement.str.strip().ne('').all(), 'statement schema/identity')
        c.require(set(self.frame.condition_id) <= set(p.TEXT_CONDITIONS+[p.ISO]) and
                  set(self.frame.split) <= set(c.BENCHMARK['allowed_splits']) and
                  set(self.frame.evaluation_phase) <= set(c.BENCHMARK['allowed_phases']) and
                  set(self.frame.protocol) == {'entity_disjoint'}, 'forbidden extraction scope')
        self.hashes = dict(benchmark_sha256=c.digest(self.payload), sidecar_sha256=c.digest(self.payload),
            ordered_example_id_sha256=c.ordered_hash(self.frame.example_id), ordered_statement_sha256=c.ordered_hash(self.frame.statement))

    def verify_source(self):
        c.require(c.file_hash(self.path) == self.hashes['benchmark_sha256'], 'statement benchmark changed')

    def verify_rows(self, start, rows):
        c.require(rows.reset_index(drop=True).equals(self.frame.iloc[start:start+len(rows)].reset_index(drop=True)), 'row order changed')

    def verify_sidecar(self, path):
        other = Statements(path)
        c.require(other.hashes == self.hashes and other.frame.equals(self.frame), 'sidecar changed')


def require_gates(manifest):
    from src import pinned_compound_extraction as shared
    shared.require_canonical_gates(manifest)
    smoke = manifest['smoke_test']['compound_smoke']
    expected = set(manifest['condition_counts'])
    c.require(set(smoke['conditions']) == expected and all(r['passed'] is True and r['batch_size'] == 1 and
              r['padding'] is False for r in smoke['conditions'].values()), 'every extracted condition needs unpadded smoke')


def extract_loop(model, tokenizer, benchmark, manifest, output):
    from src.clean_extraction import ExtractionWriter
    from src.pinned_compound_extraction import pinned_readout
    require_gates(manifest)
    with ExtractionWriter(output, benchmark, manifest, resume=False) as writer:
        for start in range(len(benchmark.frame)):
            rows = benchmark.frame.iloc[start:start+1].copy()
            values = pinned_readout(model, tokenizer, rows.statement.iloc[0]).numpy().astype(np.float16)[None]
            writer.append(start, rows, values)
        writer.finalize()


def run(mode='plan', root=c.ROOT):
    from src import pinned_compound_extraction as shared
    from src import clean_atomic_extraction as atomic
    from types import SimpleNamespace
    c.require(mode in ['plan', 'smoke', 'extract'], 'unknown extraction mode')
    output = c.safe_path(root, p.ACTS)
    c.require(not output.exists(), 'refusing existing Priority-2 extraction output; no resume')
    cache, binding = shared.bind_repaired_cache(root)
    generation, files = p.verify_generation(root, cache)
    benchmark = Statements(c.safe_path(root, p.DATA+'/statements.csv'))
    counts = benchmark.frame.groupby('condition_id').size().to_dict()
    expected = {k: c.ROWS//2 for k in p.TEXT_CONDITIONS}
    if generation['missing_facts']: expected[p.ISO] = generation['missing_facts']
    c.require(counts == expected and binding['representation_fingerprint'] == c.FINGERPRINT, 'extraction shape/representation')
    plan = dict(condition_counts=counts, shape=[len(benchmark.frame), c.LAYERS, c.WIDTH],
                representation_fingerprint=c.FINGERPRINT, output=p.ACTS, model_loaded=False)
    if mode == 'plan': return plan
    model, tokenizer, info = atomic.load_pinned_model()
    shared.validate_live_binding(cache, info)
    samples = shared.atomic_sample(cache)
    replay = shared.replay_atomic(model, tokenizer, cache, samples)
    c.require(replay['passed'] is True, 'repaired replay failed')
    smokes = {condition: shared.compound_smoke(model, tokenizer, SimpleNamespace(frame=benchmark.frame[benchmark.frame.condition_id == condition]))
              for condition in sorted(counts)}
    manifest = dict(schema_version=1, representation=binding['representation'], representation_fingerprint=c.FINGERPRINT,
        repaired_atomic_binding=binding, atomic_replay_sample_sha256=shared.fingerprint(samples),
        execution_contract=atomic.contract(), resolved_model=info, generation_files=files, condition_counts=counts,
        superseded_historical_representation=dict(status='failed_compatibility', used_as_gate=False, historical_caches_opened=False),
        numerics=dict(batch_size=1, runtime=shared.runtime_provenance()),
        data=dict(**benchmark.hashes, metadata_columns=list(benchmark.frame.columns), number_of_examples=len(benchmark.frame),
                  expected_activation_shape=plan['shape']),
        code=dict(implementation_version=p.VERSION, source_sha256={name: c.file_hash(c.ROOT/name) for name in
                 ['src/priority2_extraction.py', 'src/priority2_input_controls.py', 'src/pinned_compound_extraction.py',
                  'src/clean_extraction.py', 'src/clean_atomic_extraction.py', 'src/extract.py']}),
        smoke_test=dict(live_binding_passed=True, atomic_replay=replay,
                        compound_smoke=dict(passed=all(r['passed'] for r in smokes.values()), conditions=smokes)))
    require_gates(manifest)
    if mode == 'smoke':
        c.publish_json(c.safe_path(root, p.OUTPUT+'/extraction_smoke.json'), manifest)
    else:
        c.require(not output.exists(), 'extraction output appeared during smoke')
        extract_loop(model, tokenizer, benchmark, manifest, output)
    return {**plan, 'model_loaded': True, 'extracted': mode == 'extract'}


def verify(root, cache, generation_files):
    """CPU header/hash verification; no numerical activation values or model imports."""
    from src.clean_atomic_extraction import contract
    from src.pinned_atomic_method_suite import representation
    paths = {n: c.safe_path(root, p.ACTS+'/'+n) for n in EXTRACTION_FILES}
    c.require(not c.safe_path(root, p.ACTS+'/activations.partial.npy').exists(), 'partial array forbidden')
    files = {n: c.record(path) for n, path in paths.items()}
    manifest, progress = [c.read_json(paths[n]) for n in ['extraction_manifest.json', 'progress.json']]
    benchmark = Statements(c.safe_path(root, p.DATA+'/statements.csv'))
    benchmark.verify_sidecar(paths['metadata.csv'])
    descriptor = representation(cache.manifest['contract'])
    c.require(manifest['representation'] == descriptor and c.digest(c.canonical(descriptor)) == manifest['representation_fingerprint'] == c.FINGERPRINT and
              manifest['execution_contract'] == contract() and manifest['generation_files'] == generation_files,
              'Priority-2 extraction representation/input mismatch')
    runtime = manifest['numerics']['runtime']
    c.require(manifest['numerics']['batch_size'] == 1 and runtime['torch'].split('+')[0] == '2.11.0' and
              runtime['transformers'] == '5.12.1', 'pinned extraction runtime mismatch')
    binding = manifest['repaired_atomic_binding']
    c.require(binding['passed'] is True and binding['repaired_cache_files'] == cache.files and
              binding['repaired_cache_identity_sha256'] == c.digest(c.canonical(cache.files)), 'repaired binding mismatch')
    for key in ['model_resolved_revision', 'tokenizer_resolved_revision', 'config_file_sha256', 'tokenizer_config_file_sha256', 'tokenizer_backend_sha256']:
        c.require(manifest['resolved_model'].get(key) == cache.manifest['resolved_model'].get(key) and
                  isinstance(manifest['resolved_model'].get(key), str), 'model/tokenizer identity mismatch')
    gates = manifest['smoke_test']
    replay = gates['atomic_replay']
    c.require(gates['live_binding_passed'] is True and replay['passed'] is True and replay['saved_float16_byte_equal'] is True and
              replay['sample_sha256'] == manifest['atomic_replay_sample_sha256'] and replay['batch_size'] == 1 and replay['padding'] is False and
              replay['policy'] == 'exact-repaired-atomic-float16-replay-v1',
              'replay gate failed')
    counts = benchmark.frame.groupby('condition_id').size().to_dict()
    smoke = gates['compound_smoke']
    c.require(manifest['condition_counts'] == counts and smoke['passed'] is True and set(smoke['conditions']) == set(counts) and
              all(r['passed'] is True and r['batch_size'] == 1 and r['padding'] is False and r['policy'] == 'unpadded-single-repeat-direct-exact-v1'
                  for r in smoke['conditions'].values()), 'condition smoke failed')
    shape = [len(benchmark.frame), c.LAYERS, c.WIDTH]
    c.require(manifest['data']['expected_activation_shape'] == shape and all(manifest['data'][k] == v for k,v in benchmark.hashes.items()), 'extraction data mismatch')
    array_header(paths['activations.npy'], tuple(shape))
    # Same identity rule as the shared durable writer, without importing GPU modules.
    identity = dict(manifest)
    identity['code'] = {k: manifest['code'][k] for k in ['implementation_version', 'source_sha256']}
    identity.pop('smoke_test', None)
    c.require(progress['complete'] is True and progress['next_row'] == shape[0] and
              progress['activation_file_sha256'] == files['activations.npy']['sha256'] and
              progress['identity_sha256'] == c.digest(c.canonical(identity)), 'incomplete extraction identity')
    cursor = 0
    for chunk in progress['chunks']:
        stop = chunk['stop']
        c.require(chunk['start'] == cursor and cursor < stop <= shape[0] and
                  chunk['ordered_example_id_sha256'] == c.ordered_hash(benchmark.frame.example_id.iloc[cursor:stop]) and
                  chunk['ordered_statement_sha256'] == c.ordered_hash(benchmark.frame.statement.iloc[cursor:stop]), 'extraction coverage/order mismatch')
        cursor = stop
    c.require(cursor == shape[0], 'incomplete extraction coverage')
    return files, benchmark.frame
