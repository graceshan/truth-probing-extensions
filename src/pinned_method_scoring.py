"""Truth-column-blind multi-method affine scoring; fixed paths, zero fits."""
import os

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import clean_transfer_contracts as c
from src import method_transfer_contracts as m
from src.pinned_compound_scoring import affine
from src.pinned_method_binding import inspect


def recheck(root, spec):
    for entry in [*spec['lr_spec']['inputs'].values(), *spec['suites'].values(), spec['lr_reference']]:
        for name, expected in entry['files'].items():
            c.require(c.record(c.safe_path(root, entry['directory'] + '/' + name)) == expected, 'input bytes changed')
    c.require(c.file_hash(c.safe_path(root, c.SPEC)) == spec['lr_spec_sha256'], 'LR spec changed')


def preflight(root=c.ROOT):
    spec, binding, _, _, _ = inspect(root)
    recheck(root, spec)
    c.publish_json(c.safe_path(root, m.PREFLIGHT), dict(candidate_spec=spec, binding=binding,
        compound_scores_computed=0, fit_operations=0, compound_truth_columns_materialized=False,
        atomic_validation_readouts_verified=True))
    return dict(preflight=m.PREFLIGHT, compound_scores_computed=0, fit_operations=0)


def freeze_spec(root=c.ROOT):
    report = c.read_json(c.safe_path(root, m.PREFLIGHT))
    c.require(report['compound_scores_computed'] == report['fit_operations'] == 0 and
              report['compound_truth_columns_materialized'] is False, 'invalid preflight receipt')
    prior = m.validate(report['candidate_spec'])
    current, _, _, _, _ = inspect(root)
    c.require(current == prior, 'artifacts changed since preflight')
    recheck(root, current)
    c.publish_json(c.safe_path(root, m.SPEC), current)
    return dict(spec=m.SPEC, sha256=c.file_hash(c.safe_path(root, m.SPEC)), compound_scores_computed=0)


def lr_score_gate(actual, expected, policy):
    c.require(np.shape(actual) == np.shape(expected) and np.allclose(actual, expected,
        rtol=policy['score_rtol'], atol=policy['score_atol']), 'canonical LR per-row reproduction failed')


def score(root=c.ROOT):
    output = c.safe_path(root, m.OUTPUT)
    c.require(not output.exists(), 'refusing existing scoring output')
    spec = m.load(root)
    m.committed_spec(root)
    spec_sha = c.file_hash(c.safe_path(root, m.SPEC))
    current, binding, parameters, identities, lr_scores = inspect(root)
    c.require(current == spec, 'runtime artifacts differ from frozen specification')
    c.require(identities.example_id.tolist() == lr_scores.example_id.tolist(), 'LR score row order mismatch')
    array = np.load(c.safe_path(root, c.COMPOUND + '/activations.npy'), mmap_mode='r', allow_pickle=False)
    tables = []
    # Explicit duplicate conditions are intentional. There is no method/layer selection here.
    with threadpool_limits(limits=1):
        for condition in spec['conditions']:
            group, method, layer = [condition[k] for k in ['analysis_group', 'method', 'atomic_layer']]
            coef, intercept = parameters[group, method]
            scores = np.empty(c.ROWS, dtype=np.float64)
            for start in range(0, c.ROWS, spec['scoring']['batch_size']):
                stop = min(start + spec['scoring']['batch_size'], c.ROWS)
                scores[start:stop] = affine(array[start:stop, layer, :], coef, intercept)
            if group == m.GROUPS[0] and method == 'l2_logistic':
                lr_score_gate(scores, lr_scores.frozen_probe_score.to_numpy(), spec['reproduction'])
            tables.append(pd.DataFrame(dict(example_id=identities.example_id, analysis_group=group,
                                           method=method, atomic_layer=layer, frozen_probe_score=scores)))
    del array
    recheck(root, spec)
    c.require(c.file_hash(c.safe_path(root, m.SPEC)) == spec_sha, 'spec changed while scoring')
    table = pd.concat(tables, ignore_index=True)
    c.require(list(table.columns) == m.COLUMNS and len(table) == 15*c.ROWS, 'score schema/cardinality')
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'method_scores.csv').open('x', encoding='utf-8', newline='') as handle:
        table.to_csv(handle, index=False, float_format='%.17g')
        handle.flush()
        os.fsync(handle.fileno())
    c.publish_json(output / 'method_binding.json', binding)
    manifest = dict(complete=True, schema_version=1, implementation_version=m.VERSION,
        analysis_spec_sha256=spec_sha, representation_fingerprint=c.FINGERPRINT,
        compound_inputs=spec['lr_spec']['inputs']['compound'], suites=spec['suites'], conditions=spec['conditions'],
        ordered_example_id_sha256=c.ordered_hash(identities.example_id), rows=len(table), examples=c.ROWS,
        score_columns=m.COLUMNS, method_scores=c.record(output / 'method_scores.csv'),
        method_binding=c.record(output / 'method_binding.json'), lr_scores_reproduced=True,
        metadata_columns_materialized=list(identities.columns), compound_truth_columns_materialized=False,
        compound_labels_used=False, test_artifacts_accessed=False, fit_operations=0,
        scoring=spec['scoring'], provenance=m.code())
    c.publish_json(output / 'scoring_manifest.json', manifest)  # LAST
    return dict(output=m.OUTPUT, complete=True, rows=len(table))
