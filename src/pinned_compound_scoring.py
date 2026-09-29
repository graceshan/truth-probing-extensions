"""Truth-column-blind, frozen affine scoring. No fitting or evaluation API."""
from datetime import datetime, timezone
from pathlib import Path
import os
import re
import subprocess
import json

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits, threadpool_info

from src import clean_transfer_contracts as c
from src.repaired_atomic_cache import RepairedAtomicCache, array_header

VERSION = 'pinned-qwen25-lr-scoring-v1'
PROBE_CONFIG = dict(penalty='l2', solver='lbfgs', fit_intercept=True, max_iter=2000,
                    class_weight=None, tol=.0001)


def source_audit(root, selection):
    """Audit code only; recorded producer hashes are never replaced by current hashes."""
    revision = selection['provenance'].get('git_revision')
    recorded = {**selection['provenance'].get('code_sha256', {}),
                **selection['structural'].get('adapter_code_sha256', {})}
    audit = {}
    for name, expected in recorded.items():
        c.require(re.fullmatch(r'(src|scripts)/[A-Za-z0-9_/]+\.py', name) is not None and '..' not in name,
                  'invalid producer source path')
        c.require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'invalid producer source hash')
        current = c.safe_path(root, name)
        entry = dict(recorded_sha256=expected, current_sha256=c.file_hash(current) if current.is_file() else None,
                     git_verification='unavailable')
        if isinstance(revision, str) and re.fullmatch('[0-9a-f]{40}', revision):
            result = subprocess.run(['git', 'show', f'{revision}:{name}'], cwd=root, capture_output=True)
            if result.returncode == 0:
                entry.update(git_sha256=c.digest(result.stdout),
                             git_verification='matches' if c.digest(result.stdout) == expected else 'differs_from_recorded_commit')
        audit[name] = entry
    return dict(selection_git_revision=revision, files=audit,
                policy='source hashes are producer provenance; cache and artifact bytes are the binding')


def verify_probe(root, cache):
    directory = c.safe_path(root, c.PROBE)
    paths = {name: c.safe_path(root, f'{c.PROBE}/{name}') for name in c.PROBE_FILES}
    selection = c.read_json(paths['selection.json'])
    c.require(c.file_hash(paths['selected_probe.npz']) == selection['selected_probe_sha256'], 'probe hash mismatch')
    c.require(c.file_hash(paths['validation_metrics.csv']) == selection['validation_metrics_sha256'], 'validation metrics hash mismatch')
    structure = selection['structural']
    # Deliberately compare scientific data identity, NOT adapter/source-code identity.
    for key in ('model', 'cache_dir', 'counts', 'repaired_cache_files', 'approved_input_digest', 'extraction_contract'):
        c.require(structure.get(key) == cache.structural[key], f'selection repaired-cache identity mismatch: {key}')
    c.require(structure.get('test_metrics_evaluated') is False, 'atomic test provenance invalid')
    c.require(selection['model'] == c.MODEL and selection['preprocessing'] == 'none', 'model/preprocessing mismatch')
    c.require(selection['selection_rule'] == c.SELECTION_RULE and selection['C_values'] == c.C_VALUES,
              'frozen selection rule/grid mismatch')
    c.require(selection['probe_configuration'] == PROBE_CONFIG, 'probe configuration mismatch')
    c.require(selection['label_convention'] == '0=false, 1=true; score >= 0 predicts true' and
              selection['layer_convention'] == 'zero-based transformer-block output; embedding excluded',
              'label/layer convention mismatch')
    c.require(selection['test_evaluated'] is False and selection['test_decision_scores_computed'] == 0 and
              selection['activation_rows_read']['test'] == 0 and selection['all_final_fits_converged'] is True,
              'test access or unconverged selection')
    config = selection['provenance']['config']
    producer_config = {'schema_version': 1, 'loader': 'pinned-repaired-cache-v1', 'cache_dir': c.ATOMIC,
        'C_values': c.C_VALUES, 'models': {'qwen25_7b': {'model': c.MODEL, 'n_layers': c.LAYERS,
        'hidden_size': c.WIDTH, 'dtype': 'float16', 'results_dir': c.PROBE}}}
    c.require(config == producer_config, 'selection configuration mismatch')
    # The atomic producer uses insertion-order JSON here, unlike fingerprint JSON.
    c.require(selection['provenance']['config_sha256'] == c.digest(json.dumps(
        producer_config, ensure_ascii=False, separators=(',', ':')).encode()), 'selection config hash mismatch')
    c.require(config['loader'] == 'pinned-repaired-cache-v1' and config['cache_dir'] == c.ATOMIC and
              config['C_values'] == c.C_VALUES and selection['provenance']['model_key'] == 'qwen25_7b',
              'selection training source mismatch')
    model_config = config['models']['qwen25_7b']
    c.require(model_config['model'] == c.MODEL and model_config['n_layers'] == c.LAYERS and
              model_config['hidden_size'] == c.WIDTH and model_config['dtype'] == 'float16', 'selection dimensions mismatch')
    grid = pd.read_csv(paths['validation_metrics.csv'], keep_default_na=False, float_precision='round_trip')
    c.require(len(grid) == c.LAYERS * len(c.C_VALUES) and not grid.duplicated(['layer', 'C']).any() and
              set(zip(grid.layer, grid.C)) == {(l, C) for l in range(c.LAYERS) for C in c.C_VALUES},
              'expected exactly 140 validation configurations')
    c.require(grid.final_converged.astype(str).str.lower().eq('true').all() and
              grid.final_convergence_status.eq('converged').all() and
              grid.convergence_warning.astype(str).str.lower().eq('false').all() and
              np.isfinite(grid.validation_overall_auroc).all() and grid.validation_overall_auroc.between(0, 1).all(),
              'invalid convergence/validation grid')
    best = grid.sort_values(['validation_overall_auroc', 'C', 'layer'], ascending=[False, True, True]).iloc[0]
    layer, C = selection['selected_layer'], selection['selected_C']
    c.require(type(layer) is int and 0 <= layer < c.LAYERS and C in c.C_VALUES and
              layer == best.layer and C == best.C, 'selected layer/C inconsistent with frozen grid')
    chosen = selection['selected_validation_metrics']
    c.require(chosen['layer'] == layer and chosen['C'] == C and
              chosen['validation_overall_auroc'] == best.validation_overall_auroc and chosen['final_converged'] is True,
              'selected validation record mismatch')
    c.require(layer == c.EXPECTED_SELECTION['layer'] and C == c.EXPECTED_SELECTION['C'] and
              best.validation_overall_auroc == c.EXPECTED_SELECTION['validation_auroc'] and
              len(grid) == c.EXPECTED_SELECTION['converged_configurations'], 'unexpected frozen selection')
    with np.load(paths['selected_probe.npz'], allow_pickle=False) as archive:
        c.require(set(archive.files) == {'coef', 'intercept', 'classes', 'layer', 'C', 'n_iter', 'max_iter'},
                  'unexpected frozen archive schema')
        p = {k: archive[k].copy() for k in archive.files}
    c.require(all(np.issubdtype(v.dtype, np.number) and np.isfinite(v).all() for v in p.values()), 'nonfinite/invalid archive')
    c.require(p['layer'].shape == () and p['C'].shape == () and p['layer'].item() == layer and p['C'].item() == C,
              'NPZ selected layer/C mismatch')
    c.require(p['coef'].shape == (1, c.WIDTH) and p['intercept'].shape == (1,) and
              np.array_equal(p['classes'], [0, 1]), 'coefficient/intercept/classes mismatch')
    c.require(p['n_iter'].shape == (1,) and p['max_iter'].shape == () and
              p['max_iter'].item() == selection['selected_probe_max_iter'] == chosen['final_max_iter'] and
              p['n_iter'].item() == chosen['final_n_iter'] and p['n_iter'].item() > 0,
              'archive optimization mismatch')
    # Immutable arrays, no estimator or fit API in production scoring.
    coef = np.frombuffer(p['coef'].astype(np.float64).tobytes(), dtype=np.float64)
    return dict(coef=coef, intercept=float(p['intercept'][0]), layer=layer, C=C,
                selection=selection, files={name: c.record(path) for name, path in paths.items()},
                source_audit=source_audit(root, selection))


def extraction_identity(manifest):
    value = dict(manifest)
    value['code'] = {k: manifest['code'][k] for k in ('implementation_version', 'source_sha256')}
    value.pop('smoke_test', None)
    value.pop('historical_atomic_compatibility', None)
    return c.digest(c.canonical(value))


def verify_compound(root, cache):
    paths = {name: c.safe_path(root, f'{c.COMPOUND}/{name}') for name in c.COMPOUND_FILES}
    c.require(not c.safe_path(root, f'{c.COMPOUND}/activations.partial.npy').exists(), 'partial compound array forbidden')
    files = {name: c.record(path) for name, path in paths.items()}
    c.require(files['activations.npy']['sha256'] == c.ACTIVATION_SHA, 'activation hash mismatch')
    c.require(files['metadata.csv']['sha256'] == c.METADATA_SHA, 'metadata hash mismatch')
    manifest, progress = c.read_json(paths['extraction_manifest.json']), c.read_json(paths['progress.json'])
    descriptor = manifest['representation']
    c.require(c.digest(c.canonical(descriptor)) == manifest['representation_fingerprint'] == c.FINGERPRINT,
              'representation fingerprint mismatch')
    expected = c.representation(cache.manifest['contract'], descriptor['layer_convention'])
    c.require(descriptor == expected and manifest['execution_contract'] == cache.manifest['contract'],
              'atomic/compound representation convention mismatch')
    binding = manifest['repaired_atomic_binding']
    c.require(binding['passed'] is True and binding['repaired_cache_files'] == cache.files and
              binding['repaired_cache_identity_sha256'] == c.digest(c.canonical(cache.files)) and
              binding['representation'] == descriptor and binding['representation_fingerprint'] == c.FINGERPRINT,
              'compound repaired-cache identity mismatch')
    for key in ('model_resolved_revision', 'tokenizer_resolved_revision', 'config_file_sha256',
                'tokenizer_config_file_sha256', 'tokenizer_backend_sha256'):
        expected_identity = cache.manifest['resolved_model'].get(key)
        c.require(isinstance(expected_identity, str) and
                  re.fullmatch('[0-9a-f]{40}' if key.endswith('revision') else '[0-9a-f]{64}', expected_identity) and
                  manifest['resolved_model'].get(key) == expected_identity,
                  'compound model/tokenizer identity mismatch')
    gates = manifest['smoke_test']
    replay, smoke = gates['atomic_replay'], gates['compound_smoke']
    c.require(gates['live_binding_passed'] is True and replay['passed'] is True and
              replay['saved_float16_byte_equal'] is True and replay['sample_sha256'] == manifest['atomic_replay_sample_sha256'] and
              replay['batch_size'] == 1 and replay['padding'] is False and
              replay['policy'] == 'exact-repaired-atomic-float16-replay-v1' and
              smoke['passed'] is True and smoke['batch_size'] == 1 and smoke['padding'] is False and
              smoke['policy'] == 'unpadded-single-repeat-direct-exact-v1', 'canonical replay/smoke evidence failed')
    data = manifest['data']
    c.require(data['expected_activation_shape'] == [c.ROWS, c.LAYERS, c.WIDTH] and data['number_of_examples'] == c.ROWS,
              'manifest shape mismatch')
    c.require(data['sidecar_sha256'] == data['benchmark_sha256'] == c.METADATA_SHA, 'manifest metadata hash mismatch')
    c.require(progress['complete'] is True and progress['next_row'] == c.ROWS and
              progress['activation_file_sha256'] == c.ACTIVATION_SHA and
              progress['identity_sha256'] == extraction_identity(manifest), 'incomplete/mismatched extraction identity')
    array_header(paths['activations.npy'], (c.ROWS, c.LAYERS, c.WIDTH))
    identities = c.projected_metadata(paths['metadata.csv'])
    c.require(c.ordered_hash(identities.example_id) == data['ordered_example_id_sha256'], 'ordered identity mismatch')
    cursor = 0
    for chunk in progress['chunks']:
        stop = chunk['stop']
        c.require(chunk['start'] == cursor and type(stop) is int and cursor < stop <= c.ROWS, 'chunk coverage gap/overlap')
        c.require(chunk['ordered_example_id_sha256'] == c.ordered_hash(identities.example_id.iloc[cursor:stop]),
                  'chunk identity mismatch')
        cursor = stop
    c.require(cursor == c.ROWS, 'incomplete chunk coverage')
    return dict(files=files, descriptor=descriptor, identities=identities, manifest=manifest,
                extraction_identity=progress['identity_sha256'])


def inspect_inputs(root):
    cache = RepairedAtomicCache(root)
    probe = verify_probe(root, cache)
    compound = verify_compound(root, cache)
    inputs = {key: dict(directory=directory, files=files) for key, directory, files in [
        ('atomic', c.ATOMIC, cache.files), ('probe', c.PROBE, probe['files']),
        ('compound', c.COMPOUND, compound['files'])]}
    spec = c.validate_spec(c.spec_template(inputs, compound['descriptor']))
    binding = dict(schema_version=1, representation_fingerprint=c.FINGERPRINT,
                   representation=compound['descriptor'], repaired_cache_identity_sha256=c.digest(c.canonical(cache.files)),
                   repaired_cache_files=cache.files, selection_files=probe['files'],
                   producer_source_audit=probe['source_audit'], refit_performed=False, probe_modified=False)
    return spec, binding, probe, compound


def preflight(root=c.ROOT):
    spec, binding, _, compound = inspect_inputs(root)
    report = dict(schema_version=1, purpose='metadata_hash_only_preflight', candidate_spec=spec, binding=binding,
                  ordered_example_id_sha256=c.ordered_hash(compound['identities'].example_id),
                  activation_values_materialized=False, compound_truth_columns_materialized=False, scores_computed=0)
    c.publish_json(c.safe_path(root, c.PREFLIGHT), report)
    return dict(preflight=c.PREFLIGHT, scores_computed=0)


def freeze_spec(root=c.ROOT):
    report = c.read_json(c.safe_path(root, c.PREFLIGHT))
    c.require(report['purpose'] == 'metadata_hash_only_preflight' and report['scores_computed'] == 0, 'invalid preflight')
    spec = c.validate_spec(report['candidate_spec'])
    current, _, _, _ = inspect_inputs(root)
    c.require(current == spec, 'artifacts changed since preflight')
    c.publish_json(c.safe_path(root, c.SPEC), spec)
    return dict(spec=c.SPEC, sha256=c.file_hash(c.safe_path(root, c.SPEC)), scores_computed=0)


def affine(X, coef, intercept):
    X = np.asarray(X, dtype=np.float64)
    c.require(X.ndim == 2 and X.shape[1] == len(coef) and np.isfinite(X).all(), 'invalid selected-layer values')
    result = X @ coef + intercept
    c.require(np.isfinite(result).all(), 'nonfinite scores')
    return result


def score(root=c.ROOT):
    spec_path, destination = c.safe_path(root, c.SPEC), c.safe_path(root, c.OUTPUT)
    c.require(not destination.exists(), 'refusing existing scoring output')
    spec = c.validate_spec(c.read_json(spec_path))
    spec_sha = c.file_hash(spec_path)
    current, binding, probe, compound = inspect_inputs(root)
    c.require(current == spec, 'artifacts differ from frozen spec')
    started = datetime.now(timezone.utc).isoformat()
    destination.mkdir(parents=True, exist_ok=False)
    c.publish_json(destination / 'representation_binding.json', binding)
    path = c.safe_path(root, f'{c.COMPOUND}/activations.npy')
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    scores = np.empty(c.ROWS, dtype=np.float64)
    batch = spec['scoring']['batch_size']
    with threadpool_limits(limits=1):
        runtime_blas = threadpool_info()
        for start in range(0, c.ROWS, batch):
            scores[start:start+batch] = affine(array[start:start+batch, probe['layer'], :], probe['coef'], probe['intercept'])
    del array
    table = pd.DataFrame(dict(example_id=compound['identities'].example_id, frozen_probe_score=scores))
    with (destination / 'row_scores.csv').open('x', encoding='utf-8', newline='') as handle:
        table.to_csv(handle, index=False, float_format='%.17g')
        handle.flush()
        os.fsync(handle.fileno())
    # Recheck byte identities before publication to catch changes during scoring.
    for key in ['compound', 'probe', 'atomic']:
        for name, expected in spec['inputs'][key]['files'].items():
            c.require(c.record(c.safe_path(root, spec['inputs'][key]['directory'] + '/' + name)) == expected,
                      'input changed during scoring')
    c.require(c.file_hash(spec_path) == spec_sha, 'spec changed during scoring')
    manifest = dict(schema_version=1, implementation_version=VERSION, complete=True,
        model=c.MODEL, model_revision=c.REVISION, tokenizer_revision=c.REVISION,
        representation_fingerprint=c.FINGERPRINT, inputs=spec['inputs'], analysis_spec_sha256=spec_sha,
        repaired_cache_identity_sha256=binding['repaired_cache_identity_sha256'],
        representation_binding=c.record(destination / 'representation_binding.json'),
        selected_probe_sha256=probe['files']['selected_probe.npz']['sha256'], selected_layer=probe['layer'], C=probe['C'],
        classes=[0, 1], coefficient_shape=[1, c.WIDTH], intercept_shape=[1], rows=c.ROWS,
        fit_split='train', selection_split='validation', refit_on_validation=False,
        ordered_example_id_sha256=c.ordered_hash(table.example_id), score_columns=list(table.columns),
        row_scores=c.record(destination / 'row_scores.csv'), scoring=spec['scoring'],
        metadata_columns_materialized=list(compound['identities'].columns),
        compound_truth_columns_materialized=False, compound_labels_used=False, fit_operations=0,
        test_artifacts_accessed=False, provenance=c.code_provenance(), blas=runtime_blas,
        started_utc=started, completed_utc=datetime.now(timezone.utc).isoformat())
    c.publish_json(destination / 'scoring_manifest.json', manifest)  # completion marker LAST
    return dict(output=c.OUTPUT, rows=c.ROWS, complete=True)
