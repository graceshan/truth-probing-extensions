"""Evaluate immutable method scores. No activation or parameter archive access."""
import csv

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import clean_transfer_contracts as c
from src import method_transfer_contracts as m
from src.clean_transfer_evaluation import validate_metadata, match_and_or, finite_records
from src.clean_transfer_statistics import EntityBootstrap, MetricPlan, interval


def load_scores(root, spec):
    output = c.safe_path(root, m.OUTPUT)
    manifest = c.read_json(c.safe_path(root, m.OUTPUT + '/scoring_manifest.json'))
    c.require(manifest['complete'] is True and manifest['implementation_version'] == m.VERSION and
              manifest['analysis_spec_sha256'] == c.file_hash(c.safe_path(root, m.SPEC)) and
              manifest['representation_fingerprint'] == c.FINGERPRINT and manifest['score_columns'] == m.COLUMNS and
              manifest['rows'] == 15*c.ROWS and manifest['examples'] == c.ROWS and
              manifest['scoring'] == spec['scoring'] and manifest['compound_inputs'] == spec['lr_spec']['inputs']['compound'] and
              manifest['suites'] == spec['suites'] and manifest['conditions'] == spec['conditions'], 'scoring manifest binding mismatch')
    c.require(manifest['fit_operations'] == 0 and manifest['test_artifacts_accessed'] is False and
              manifest['compound_truth_columns_materialized'] is False and manifest['compound_labels_used'] is False and
              manifest['lr_scores_reproduced'] is True and set(manifest['metadata_columns_materialized']) <= set(c.PROJECTION),
              'invalid truth-blind scoring provenance')
    for name, key in [('method_scores.csv', 'method_scores'), ('method_binding.json', 'method_binding')]:
        c.require(c.record(c.safe_path(root, m.OUTPUT + '/' + name)) == manifest[key], 'scoring output hash mismatch')
    binding = c.read_json(output / 'method_binding.json')
    c.require(binding['suites'] == spec['suites'] and binding['conditions'] == spec['conditions'] and
              binding['representation_fingerprint'] == c.FINGERPRINT and binding['fit_operations'] == 0 and
              binding['atomic_test_accessed'] is False and binding['compound_labels_used'] is False and
              binding['ttpd_explicit_fused_verified'] is True and binding['atomic_validation_archives_reproduced'] is True,
              'method binding differs')
    path = c.safe_path(root, c.COMPOUND + '/metadata.csv')
    c.require(c.record(path) == spec['lr_spec']['inputs']['compound']['files']['metadata.csv'], 'metadata hash mismatch')
    projected = c.projected_metadata(path)
    with (output / 'method_scores.csv').open(newline='', encoding='utf-8') as handle:
        c.require(next(csv.reader(handle)) == m.COLUMNS, 'score columns mismatch')
    scores = pd.read_csv(output / 'method_scores.csv', keep_default_na=False, float_precision='round_trip', dtype={'example_id': str})
    c.require(len(scores) == 15*c.ROWS and np.isfinite(scores.frozen_probe_score).all() and
              not scores.duplicated(['analysis_group', 'method', 'example_id']).any(), 'score cardinality/duplicates')
    expected = [(r['analysis_group'], r['method']) for r in spec['conditions']]
    c.require(set(zip(scores.analysis_group, scores.method)) == set(expected), 'score conditions differ')
    for condition in spec['conditions']:
        part = scores[(scores.analysis_group == condition['analysis_group']) & (scores.method == condition['method'])]
        c.require(part.example_id.tolist() == projected.example_id.tolist() and
                  part.atomic_layer.eq(condition['atomic_layer']).all(), 'exact one-to-one score coverage/layer mismatch')
    c.require(c.ordered_hash(projected.example_id) == manifest['ordered_example_id_sha256'], 'ordered identity mismatch')
    # Recheck old immutable LR references without opening coefficients/activation arrays.
    for name, expected in spec['lr_reference']['files'].items():
        c.require(c.record(c.safe_path(root, c.OUTPUT + '/' + name)) == expected, 'LR reference bytes changed')
    old = pd.read_csv(c.safe_path(root, c.OUTPUT + '/row_scores.csv'), keep_default_na=False, float_precision='round_trip')
    common = scores[(scores.analysis_group == m.GROUPS[0]) & (scores.method == 'l2_logistic')]
    c.require(old.example_id.tolist() == common.example_id.tolist() and np.allclose(old.frozen_probe_score,
        common.frozen_probe_score, rtol=0, atol=spec['reproduction']['score_atol']), 'LR score reproduction failed')
    # All identity/cardinality gates above precede the FIRST truth-bearing read.
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    c.require(c.record(path) == spec['lr_spec']['inputs']['compound']['files']['metadata.csv'] and
              raw.example_id.tolist() == projected.example_id.tolist(), 'metadata changed during read')
    frame = validate_metadata(raw, spec['benchmark']).sort_values('example_id').reset_index(drop=True)
    return scores, frame, manifest


def metrics_for_method(spec, method):
    return [d for d in spec['metrics'] if d['category'] != 'threshold' or spec['threshold_applicability'][method]['applicable']]


def compute(scores, frame, spec):
    schedule = EntityBootstrap(frame, spec['bootstrap'])  # ONE schedule for every condition and contrast.
    c.require(schedule.sha256 == spec['lr_reference']['schedule_sha256'], 'LR bootstrap schedule identity mismatch')
    records, points, draws, lookup = [], [], [], {}
    with threadpool_limits(limits=1):
        for condition in spec['conditions']:
            group, method = condition['analysis_group'], condition['method']
            part = scores[(scores.analysis_group == group) & (scores.method == method)][['example_id', 'frozen_probe_score']]
            f = frame.merge(part, on='example_id', validate='one_to_one').sort_values('example_id').reset_index(drop=True)
            plan = MetricPlan(f, match_and_or(f), {**spec, 'metrics': metrics_for_method(spec, method)})
            point = plan.evaluate(np.ones((1, len(f))))[0]
            values = np.empty((spec['bootstrap']['replicates'], len(plan.records)))
            for start in range(0, len(values), 64):
                stop = min(start + 64, len(values))
                values[start:stop] = plan.evaluate(schedule.row_weights(start, stop))
            for i, record in enumerate(plan.records):
                lookup[group, method, record['metric_id']] = len(records)
                records.append({**record, 'metric_id': group + '/' + method + '/' + record['metric_id'],
                                'analysis_group': group, 'method': method, 'atomic_layer': condition['atomic_layer']})
                points.append(point[i])
                draws.append(values[:, i])
    for contrast in spec['paired_contrasts']:
        for scope, topic in [('pooled', 'all')] + [('topic', t) for t in spec['benchmark']['topics']] + [('topic_macro', 'all')]:
            for metric in contrast['metrics']:
                suffix = f'{scope}/{topic}/{metric}'
                left, right = [lookup[*side, suffix] for side in [contrast['left'], contrast['right']]]
                records.append(dict(metric_id=contrast['id'] + '/' + suffix, metric=metric, scope=scope, topic=topic,
                    category='paired_contrast', primary=False, contrast=contrast['id'], contrast_role=contrast['role'],
                    left_condition='/'.join(contrast['left']), right_condition='/'.join(contrast['right'])))
                points.append(points[left] - points[right])
                draws.append(draws[left] - draws[right])  # NaN if either component is undefined. Never redraw.
    values = np.column_stack(draws).astype(np.float64)
    table = pd.DataFrame([{**r, 'estimate': points[i], **interval(values[:, i], spec['bootstrap'])} for i, r in enumerate(records)])
    table['majority_accuracy_baseline'] = np.where(table.metric.isin(['and_accuracy', 'or_accuracy']), .75, np.nan)
    return table, values, schedule


def primary_gate(root, metrics, spec):
    expected = pd.read_csv(c.safe_path(root, c.OUTPUT + '/evaluation/primary_metrics.csv'), keep_default_na=False, float_precision='round_trip')
    actual = metrics[(metrics.analysis_group == m.GROUPS[0]) & (metrics.method == 'l2_logistic') & (metrics.category == 'primary')]
    keys = ['scope', 'topic', 'metric']
    c.require(len(expected) == len(actual) == 35 and not expected.duplicated(keys).any(), 'LR primary metric coverage')
    joined = actual.merge(expected, on=keys, validate='one_to_one', suffixes=('_new', '_old'))
    c.require(len(joined) == 35, 'LR primary identities differ')
    for col in ['estimate', 'ci_low', 'ci_high', 'valid_fraction']:
        old = pd.to_numeric(joined[col + '_old'].replace('', np.nan)).to_numpy(float)
        c.require(np.allclose(joined[col + '_new'], old, rtol=0, atol=spec['reproduction']['primary_atol'], equal_nan=True),
                  'canonical LR primary reproduction failed: ' + col)
    for col in ['total_replicates', 'valid_replicates', 'invalid_replicates', 'ci_status']:
        c.require(joined[col + '_new'].eq(joined[col + '_old']).all(), 'LR CI validity differs')


def evaluate(root=c.ROOT):
    spec = m.load(root)
    spec_sha = c.file_hash(c.safe_path(root, m.SPEC))
    output = c.safe_path(root, m.OUTPUT + '/evaluation')
    c.require(not output.exists(), 'refusing existing evaluation output')
    scores, frame, manifest = load_scores(root, spec)
    metrics, draws, schedule = compute(scores, frame, spec)
    primary_gate(root, metrics, spec)
    output.mkdir(parents=True, exist_ok=False)
    tables = {name: metrics[metrics.category == category] for name, category in [
        ('primary_metrics', 'primary'), ('boundary_metrics', 'boundary'), ('geometry_metrics', 'geometry'),
        ('threshold_metrics', 'threshold'), ('paired_method_contrasts', 'paired_contrast')]}
    tables.update(topic_metrics=metrics[metrics.scope != 'pooled'], bootstrap_summary=metrics)
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False, float_format='%.17g')
    for name in ['primary_metrics', 'bootstrap_summary']:
        c.publish_json(output / (name + '.json'), finite_records(tables[name]))
    np.savez_compressed(output / 'bootstrap_draws.npz', metric_ids=metrics.metric_id.to_numpy(str), values=draws, valid=np.isfinite(draws))
    np.savez_compressed(output / 'bootstrap_pair_weights.npz', pair_ids=np.asarray(schedule.pair_ids), weights=schedule.weights)
    c.require(c.file_hash(c.safe_path(root, m.SPEC)) == spec_sha and
              c.record(c.safe_path(root, m.OUTPUT + '/method_scores.csv')) == manifest['method_scores'] and
              c.record(c.safe_path(root, c.COMPOUND + '/metadata.csv')) == spec['lr_spec']['inputs']['compound']['files']['metadata.csv'],
              'evaluation inputs changed')
    result = dict(complete=True, schema_version=1, implementation_version=m.VERSION, analysis_spec_sha256=spec_sha,
        scoring_manifest=c.record(c.safe_path(root, m.OUTPUT + '/scoring_manifest.json')), method_scores=manifest['method_scores'],
        metadata=spec['lr_spec']['inputs']['compound']['files']['metadata.csv'], representation_fingerprint=c.FINGERPRINT,
        examples=len(frame), conditions=spec['conditions'], schedule_sha256=schedule.sha256,
        bootstrap=spec['bootstrap'], statistics=spec['statistics'], threshold_applicability=spec['threshold_applicability'],
        lr_primary_reproduced=True, lr_schedule_reproduced=True, atomic_or_compound_test_accessed=False,
        activation_arrays_opened=False, probe_archives_opened=False, fit_operations=0, provenance=m.code(),
        outputs={p.name: c.record(p) for p in sorted(output.iterdir())})
    c.publish_json(output / 'evaluation_manifest.json', result)
    return dict(output=str(output.relative_to(root)), complete=True, examples=len(frame))
