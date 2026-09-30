"""Truth-blind Llama preflight, immutable policy freeze, and affine scoring only."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import clean_transfer_contracts as c
from src import llama_replication_contracts as l
from src import llama_replication_extraction as extraction
from src import llama_replication_probes as probes


def candidate(root):
    selection, coef, intercept, structural = probes.verify_probe(root)
    rows, data = extraction.transfer_rows(root)
    probe_files = {name: c.record(c.safe_path(root, l.PROBE+'/'+name)) for name in
                   ['selection.json', 'selected_probe.npz', 'validation_metrics.csv']}
    binding = dict(data=data, atomic=structural, probe=probe_files)
    files = extraction.verify_cache(root, 'transfer', rows, binding)
    spec = dict(schema_version=1, analysis_id='llama31-cross-family-replication-v1',
        pin=l.read_pin(root), inputs=dict(data=data, probe=probe_files, atomic=structural, transfer=files),
        selected=dict(layer=selection['selected_layer'], C=selection['selected_C'],
                      validation_auroc=selection['selected_validation_metrics']['validation_overall_auroc']),
        policy=l.policy(root), source_sha256=l.sources(root),
        ordered_example_id_sha256=c.ordered_hash(rows.frame.example_id))
    return validate_spec(spec, root), rows, coef, intercept


def validate_spec(spec, root):
    c.require(set(spec) == {'schema_version', 'analysis_id', 'pin', 'inputs', 'selected', 'policy',
        'source_sha256', 'ordered_example_id_sha256'} and spec['schema_version'] == 1 and
        spec['analysis_id'] == 'llama31-cross-family-replication-v1', 'analysis spec schema')
    l.validate_pin(spec['pin'])
    c.require(spec['policy'] == l.policy(root) and spec['source_sha256'] == l.sources(root), 'frozen policy/code changed')
    selected = spec['selected']
    c.require(set(selected) == {'layer', 'C', 'validation_auroc'} and type(selected['layer']) is int and
              0 <= selected['layer'] < l.LAYERS and selected['C'] in probes.selector.C_VALUES and
              0 <= selected['validation_auroc'] <= 1, 'selected assertions')
    c.require(set(spec['inputs']) == {'data', 'probe', 'atomic', 'transfer'}, 'input schema')
    c.require(set(spec['inputs']['transfer']) == set(l.CACHE_FILES) and
              set(spec['inputs']['probe']) == {'selection.json', 'selected_probe.npz', 'validation_metrics.csv'}, 'file schema')
    for group in ['data', 'probe', 'transfer']:
        for record in spec['inputs'][group].values(): l.validate_record(record)
    # Atomic structural schema originates in the strictly verified, frozen selector.
    c.require(spec['inputs']['atomic']['pin'] == spec['pin'], 'atomic representation mismatch')
    for split in ['train', 'validation']:
        c.require(set(spec['inputs']['atomic']['cache_files'][split]) == set(l.CACHE_FILES), 'atomic file schema')
        for record in spec['inputs']['atomic']['cache_files'][split].values(): l.validate_record(record)
    c.require(set(spec['inputs']['atomic']['cache_files']) == {'train', 'validation'}, 'atomic partition schema')
    c.require(set(spec['inputs']['atomic']) == {'pin', 'cache_files', 'approved_atomic_input', 'counts',
        'training_policy', 'test_accessed', 'compound_accessed'} and
        spec['inputs']['atomic']['test_accessed'] is False and spec['inputs']['atomic']['compound_accessed'] is False,
        'atomic binding schema')
    return spec


def preflight(root=c.ROOT):
    spec, _, _, _ = candidate(root)
    receipt = dict(complete=True, scores_computed=0, activation_matrices_materialized=0,
                   compound_truth_columns_materialized=False, candidate_spec=spec)
    c.publish_json(c.safe_path(root, l.OUTPUT+'/preflight.json'), receipt)
    return dict(output=l.OUTPUT+'/preflight.json', scores_computed=0)


def freeze(root=c.ROOT):
    receipt = c.read_json(c.safe_path(root, l.OUTPUT+'/preflight.json'))
    spec, _, _, _ = candidate(root)
    c.require(receipt['complete'] is True and receipt['scores_computed'] == 0 and receipt['candidate_spec'] == spec,
              'preflight changed; freeze refused')
    c.require(not c.safe_path(root, l.OUTPUT+'/scores').exists(), 'scores already exist')
    c.publish_json(c.safe_path(root, l.SPEC), spec)
    return dict(spec=l.SPEC, sha256=c.file_hash(c.safe_path(root, l.SPEC)), scores_computed=0)


def affine(values, coef, intercept):
    values = np.asarray(values, dtype=np.float64)
    c.require(values.ndim == 2 and values.shape[1] == coef.shape[1] and np.isfinite(values).all(), 'invalid scoring batch')
    result = (values @ coef.T + intercept).reshape(-1)
    c.require(np.isfinite(result).all(), 'nonfinite scores')
    return result


def score(root=c.ROOT):
    l.committed(root, l.SPEC)
    spec = validate_spec(c.read_json(c.safe_path(root, l.SPEC)), root)
    current, rows, coef, intercept = candidate(root)
    c.require(current == spec, 'frozen artifact identities changed')
    output = c.safe_path(root, l.OUTPUT+'/scores')
    output.mkdir(parents=True, exist_ok=False)
    layer = spec['selected']['layer']
    array = np.load(extraction.cache_path(root, 'transfer')/'activations.npy', mmap_mode='r', allow_pickle=False)
    values = np.empty(len(rows.frame), dtype=np.float64)
    with threadpool_limits(limits=1):
        for start in range(0, len(values), 256):
            stop = min(start+256, len(values))
            values[start:stop] = affine(array[start:stop, layer, :], coef, intercept)
    frame = rows.frame[['example_id', 'condition_id']].assign(frozen_probe_score=values)
    with (output/'scores.csv').open('x') as handle:
        frame.to_csv(handle, index=False, float_format='%.17g')
        handle.flush()
        import os
        os.fsync(handle.fileno())
    # Recheck opaque data/activation identities before publishing the completion marker.
    c.require(candidate(root)[0] == spec, 'inputs changed during scoring')
    c.publish_json(output/'scoring_manifest.json', dict(complete=True, analysis_spec_sha256=c.file_hash(c.safe_path(root, l.SPEC)),
        score_file=c.record(output/'scores.csv'), rows=len(values), columns=l.SCORE_COLUMNS,
        ordered_example_id_sha256=c.ordered_hash(frame.example_id), compound_truth_columns_materialized=False,
        fit_operations=0, test_accessed=False, selected=spec['selected'], representation_fingerprint=spec['pin']['fingerprint']))
    return dict(output=str(output), scores_computed=len(values))
