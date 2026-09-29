"""CPU-only frozen v1 specification and strict I/O for pinned transfer.

No production spec is checked in: hashes must be collected from real artifacts.
"""
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import uuid
import subprocess
from importlib.metadata import version

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ATOMIC = 'acts/clean_protocol/atomic/qwen25_a09a354_bs1_bf16_v1'
PROBE = 'results/clean_protocol/atomic_probes_pinned_v1/qwen25_7b'
COMPOUND = 'acts/clean_protocol/entity_disjoint/development_validation_v1/qwen2_5_7b_a09a354_bs1_bf16_v1'
OUTPUT = 'results/clean_protocol/entity_disjoint_transfer_qwen25_pinned_v1'
SPEC = 'config/clean_protocol/pinned_qwen25_lr_transfer_v1.json'
PREFLIGHT = 'results/clean_protocol/pinned_qwen25_lr_transfer_preflight_v1.json'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
MODEL = 'Qwen/Qwen2.5-7B-Instruct'
FINGERPRINT = '59df237b800f49892889e72ee6f852314b8430eb14f9c64dc0d7fff83e832821'
ACTIVATION_SHA = 'ed9f6a642f35ec185b6f6b63c4bc4a7dafb754012e967274560bc8aef43d07ac'
METADATA_SHA = '96e96515e780086065694155e3061bdfe6f4bb80af062a63c2deecd6bfad6f94'
ROWS, LAYERS, WIDTH = 8384, 28, 3584
TOPICS = ['cities', 'sp_en_trans', 'inventors', 'element_symb', 'animal_class']
CELLS = ['TT', 'TF', 'FT', 'FF']
PROJECTION = ['example_id', 'split', 'protocol', 'evaluation_phase', 'benchmark_label']
ATOMIC_FILES = ['completion.json', 'extraction_manifest.json', 'train/activations.npy',
                'train/metadata.csv', 'validation/activations.npy', 'validation/metadata.csv']
PROBE_FILES = ['selection.json', 'selected_probe.npz', 'validation_metrics.csv']
COMPOUND_FILES = ['extraction_manifest.json', 'progress.json', 'activations.npy', 'metadata.csv']
SOURCES = ['src/clean_transfer_contracts.py', 'src/pinned_compound_scoring.py',
           'src/clean_transfer_statistics.py', 'src/clean_transfer_evaluation.py',
           'scripts/36_score_pinned_compounds.py', 'scripts/37_evaluate_clean_transfer.py',
           'src/repaired_atomic_cache.py', 'src/clean_atomic_extraction.py', 'src/clean_compounds.py']
SELECTION_RULE = 'maximize pooled validation AUROC; exact ties: smaller C, then lower zero-based layer index'
C_VALUES = [.001, .01, .1, 1., 10.]
EXPECTED_SELECTION = dict(layer=17, C=1., validation_auroc=.9996523668639054,
                          converged_configurations=140, role='assertions_only')
BENCHMARK = dict(rows=ROWS, entities=262, pairs=524, degree=4, variants_per_pair=16,
                 topics=TOPICS, operators=['AND', 'OR'], cells=CELLS, orderings=['AB', 'BA'],
                 pairs_per_topic=dict(cities=298, sp_en_trans=68, inventors=90, element_symb=36, animal_class=32),
                 allowed_splits=['development', 'validation'], protocol='entity_disjoint',
                 allowed_phases=['development', 'development_validation'])


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def ordered_hash(values):
    return digest(canonical(list(values)))


def safe_path(root, relative):
    root = Path(root).resolve()
    relative = Path(relative)
    require(not relative.is_absolute() and '..' not in relative.parts, 'path outside fixed allowlist')
    path = root / relative
    require(path.resolve() == path.absolute(), 'symlink/path escape forbidden')
    return path


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def record(path):
    return dict(sha256=file_hash(path), bytes=Path(path).stat().st_size)


def no_duplicates(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path):
    return json.loads(Path(path).read_bytes(), object_pairs_hook=no_duplicates,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f'nonfinite JSON: {x}')))


def fsync_dir(directory):
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_json(path, value):
    """No-clobber atomic publication, after durable temporary contents."""
    path = Path(path)
    require(not path.exists(), f'refusing overwrite: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / ('.' + path.name + '.' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as handle:
            handle.write(canonical(value) + b'\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)  # fails atomically if another publisher won
        fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def projected_metadata(path):
    # Only the header is decoded with csv.reader. No full row is ever returned.
    with Path(path).open(encoding='utf-8', newline='') as handle:
        header = next(csv.reader(handle))
    require(len(header) == len(set(header)), 'duplicate metadata columns')
    require(set(PROJECTION[:4]) <= set(header), 'missing identity/scope columns')
    frame = pd.read_csv(path, usecols=lambda c: c in PROJECTION, dtype=str, keep_default_na=False)
    require(set(frame.columns) <= set(PROJECTION), 'truth column materialized')
    require(len(frame) == ROWS and frame.example_id.is_unique and frame.example_id.str.strip().ne('').all(),
            'invalid projected identities/count')
    require(set(frame.split) <= set(BENCHMARK['allowed_splits']), 'test/unknown split forbidden')
    require(set(frame.protocol) == {'entity_disjoint'}, 'wrong protocol')
    require(set(frame.evaluation_phase) <= set(BENCHMARK['allowed_phases']), 'test/unknown phase forbidden')
    if 'benchmark_label' in frame:
        require(set(frame.benchmark_label) <= {'DEVELOPMENT / VALIDATION - NOT FINAL TEST'}, 'unknown benchmark label')
    return frame


def representation(settings, layer_convention):
    keys = ('model', 'model_revision', 'tokenizer_revision', 'compute_dtype', 'attention',
            'batch_size', 'padding', 'input', 'add_special_tokens', 'chat_template', 'truncation',
            'explicit_semantic_position_ids', 'use_cache', 'output_hidden_states', 'saved_dtype', 'readout')
    return dict(schema_version=1, **{k: settings[k] for k in keys}, num_hidden_layers=LAYERS,
                hidden_size=WIDTH, layer_convention=layer_convention)


def metric_definitions():
    records = []
    def auc(name, op, pos, neg, category):
        records.append(dict(id=name, kind='auc', operator=op, positive=pos, negative=neg,
                            category=category, ci=True))
    auc('and_auroc', 'AND', ['TT'], ['TF', 'FT', 'FF'], 'primary')
    auc('or_auroc', 'OR', ['TT', 'TF', 'FT'], ['FF'], 'primary')
    records.append(dict(id='and_minus_or_auroc', kind='difference', operands=['and_auroc', 'or_auroc'],
                        category='primary', ci=True))
    auc('and_tt_vs_mixed_auroc', 'AND', ['TT'], ['TF', 'FT'], 'primary')
    auc('or_mixed_vs_ff_auroc', 'OR', ['TF', 'FT'], ['FF'], 'primary')
    for op, comparisons in [('AND', [('TT', 'TF'), ('TT', 'FT'), ('TT', 'FF')]),
                            ('OR', [('TF', 'FF'), ('FT', 'FF'), ('TT', 'FF')])]:
        for pos, neg in comparisons:
            auc(f'{op.lower()}_{pos.lower()}_vs_{neg.lower()}_auroc', op, [pos], [neg], 'boundary')
    auc('or_tt_vs_mixed_geometry_auroc', 'OR', ['TT'], ['TF', 'FT'], 'geometry')
    auc('and_mixed_vs_ff_geometry_auroc', 'AND', ['TF', 'FT'], ['FF'], 'geometry')
    records.append(dict(id='pooled_and_or_auroc', kind='compound_auc', category='geometry', ci=True))
    for op in ['AND', 'OR']:
        for kind in ['accuracy', 'balanced_accuracy']:
            records.append(dict(id=f'{op.lower()}_{kind}', kind=kind, operator=op, category='threshold', ci=True))
        for cell in CELLS:
            records.append(dict(id=f'{op.lower()}_{cell.lower()}_true_response_fraction', kind='true_response',
                                operator=op, cell=cell, category='threshold', ci=True))
    for cell in ['all', *CELLS]:
        records.append(dict(id=f'or_minus_and_{cell.lower()}_mean', kind='matched_mean', cell=cell,
                            category='geometry', ci=True))
    return records


def spec_template(inputs, descriptor):
    return dict(schema_version=1, analysis_id='pinned_qwen25_lr_transfer_v1', inputs=inputs,
        expected_selection=EXPECTED_SELECTION, representation=dict(fingerprint=FINGERPRINT, descriptor=descriptor),
        benchmark=BENCHMARK,
        scoring=dict(operation='direct_affine_decision_function', dtype='float64', preprocessing='none',
                     batch_size=256, blas_threads=1, metadata_projection=PROJECTION),
        metrics=metric_definitions(),
        reporting=dict(scopes=['pooled', 'topic', 'topic_macro'], macro='equal_mean_of_all_five_topics',
                       matched_keys=['pair_id', 'canonical_truth_a', 'canonical_truth_b', 'ordering'],
                       majority_accuracy_baseline=.75),
        statistics=dict(threshold=0., threshold_operator='>=', auc_ties='half_credit', sd_ddof=0,
                        quantiles=[.1, .25, .5, .75, .9], quantile_method='linear',
                        ci='pointwise_percentile', confidence=.95, ci_quantile_method='linear',
                        multiplicity_adjusted=False),
        bootstrap=dict(method='topic_stratified_endpoint', pair_weight='m_i*m_j', replicates=2000,
                       seed=1729, rng='numpy.random.Generator(PCG64)', order='sorted_topics_entities_pairs',
                       minimum_valid=1800, undefined='retain_invalid_no_redraw_no_imputation',
                       macro_requires_all_topics=True, normalize_topic_weights=False),
        prohibitions={k: True for k in ['compound_fitting', 'calibration', 'sign_selection', 'layer_selection',
                       'C_selection', 'threshold_tuning', 'atomic_test_access', 'compound_test_access',
                       'canonical_artifact_modification']},
        outputs=dict(root=OUTPUT, score_columns=['example_id', 'frozen_probe_score'],
                     score_files=['row_scores.csv', 'representation_binding.json', 'scoring_manifest.json'],
                     evaluation_directory='evaluation', evaluation_files=[
                         'primary_metrics.csv', 'primary_metrics.json', 'boundary_metrics.csv', 'threshold_metrics.csv',
                         'descriptive_geometry.csv', 'cell_statistics.csv', 'topic_metrics.csv', 'matched_and_or.csv',
                         'bootstrap_summary.csv', 'bootstrap_summary.json', 'bootstrap_draws.npz',
                         'bootstrap_pair_weights.npz', 'evaluation_manifest.json']))


def validate_spec(spec):
    inputs = spec.get('inputs', {})
    require(set(inputs) == {'atomic', 'probe', 'compound'}, 'invalid input sections')
    for key, directory, names in [('atomic', ATOMIC, ATOMIC_FILES), ('probe', PROBE, PROBE_FILES),
                                  ('compound', COMPOUND, COMPOUND_FILES)]:
        entry = inputs[key]
        require(set(entry) == {'directory', 'files'} and entry['directory'] == directory, 'input path/keys forbidden')
        require(set(entry['files']) == set(names), 'file allowlist mismatch')
        for item in entry['files'].values():
            require(set(item) == {'sha256', 'bytes'} and isinstance(item['sha256'], str) and
                    re.fullmatch('[0-9a-f]{64}', item['sha256']) and type(item['bytes']) is int and item['bytes'] > 0,
                    'invalid file hash/size')
    descriptor = spec.get('representation', {}).get('descriptor', {})
    require(digest(canonical(descriptor)) == FINGERPRINT, 'representation fingerprint mismatch')
    require(canonical(spec) == canonical(spec_template(inputs, descriptor)), 'unknown keys or altered frozen specification')
    require(inputs['compound']['files']['activations.npy']['sha256'] == ACTIVATION_SHA, 'canonical activation hash mismatch')
    require(inputs['compound']['files']['metadata.csv']['sha256'] == METADATA_SHA, 'canonical metadata hash mismatch')
    return spec


def code_provenance():
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True)
    return dict(source_sha256={p: file_hash(ROOT / p) for p in SOURCES},
                git_commit=head.stdout.strip() if head.returncode == 0 else None,
                python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                sklearn=version('scikit-learn'), threadpoolctl=version('threadpoolctl'))
