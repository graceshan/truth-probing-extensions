"""Label-aware evaluation of finalized scores; never opens activations or probes."""
from datetime import datetime, timezone
import csv

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import clean_transfer_contracts as c
from src.clean_transfer_statistics import EntityBootstrap, MetricPlan, describe, interval
from src.clean_compounds import boolean_truth

VERSION = 'clean-pinned-transfer-evaluation-v1'


def parse_bool(values):
    values = values.astype(str)
    c.require(values.isin(['True', 'False', '1', '0']).all(), 'invalid Boolean metadata')
    return values.isin(['True', '1'])


def validate_metadata(frame, expected):
    required = {'example_id', 'pair_id', 'topic', 'split', 'operator', 'ordering', 'entity_a_id', 'entity_b_id',
                'fact_a_id', 'fact_b_id', 'canonical_truth_a', 'canonical_truth_b', 'compound_label',
                'protocol', 'evaluation_phase'}
    c.require(required <= set(frame), 'missing evaluation metadata')
    f = frame.copy().reset_index(drop=True)
    c.require(len(f) == expected['rows'] and f.example_id.is_unique and
              all(f[k].notna().all() and f[k].astype(str).str.strip().ne('').all() for k in required), 'row count/IDs/missing metadata')
    c.require(set(f.split) <= set(expected['allowed_splits']) and set(f.protocol) == {expected['protocol']} and
              set(f.evaluation_phase) <= set(expected['allowed_phases']), 'test/unknown scope forbidden')
    c.require(set(f.topic) == set(expected['topics']) and set(f.operator) == {'AND', 'OR'} and
              set(f.ordering) == {'AB', 'BA'}, 'topic/operator/order mismatch')
    for name in ['canonical_truth_a', 'canonical_truth_b', 'compound_label']:
        f[name] = parse_bool(f[name])
    a, b = f.canonical_truth_a.to_numpy(), f.canonical_truth_b.to_numpy()
    cells = np.char.add(np.where(a, 'T', 'F'), np.where(b, 'T', 'F'))
    if 'cell' in f:
        c.require(np.array_equal(f.cell, cells), 'cell mismatch')
    f['cell'] = cells
    truth = [boolean_truth(op, [bool(x), bool(y)]) for op, x, y in zip(f.operator, a, b)]
    c.require(np.array_equal(f.compound_label, truth), 'compound Boolean semantics mismatch')
    for name, values in [('surface_first_truth', np.where(f.ordering == 'AB', a, b)),
                         ('surface_second_truth', np.where(f.ordering == 'AB', b, a))]:
        if name in f:
            c.require(np.array_equal(parse_bool(f[name]), values), 'surface truth mismatch')
    for name, values in [('surface_first_entity_id', np.where(f.ordering == 'AB', f.entity_a_id, f.entity_b_id)),
                         ('surface_second_entity_id', np.where(f.ordering == 'AB', f.entity_b_id, f.entity_a_id))]:
        if name in f:
            c.require(np.array_equal(f[name], values), 'surface endpoint mismatch')
    c.require((f.entity_a_id != f.entity_b_id).all(), 'self pair')
    pairs = f[['pair_id', 'topic', 'entity_a_id', 'entity_b_id']].drop_duplicates()
    c.require(pairs.pair_id.is_unique and len(pairs) == expected['pairs'], 'pair identity/count mismatch')
    endpoints = [tuple(sorted(x)) for x in zip(pairs.entity_a_id, pairs.entity_b_id)]
    c.require(len(set(endpoints)) == len(pairs), 'duplicate unordered pair')
    topics, degree = {}, {}
    for row in pairs.itertuples():
        for entity in (row.entity_a_id, row.entity_b_id):
            c.require(topics.get(entity, row.topic) == row.topic, 'entity crosses topics')
            topics[entity] = row.topic
            degree[entity] = degree.get(entity, 0) + 1
    c.require(len(degree) == expected['entities'] and set(degree.values()) == {expected['degree']}, 'entity count/degree mismatch')
    c.require(pairs.groupby('topic').size().to_dict() == expected['pairs_per_topic'], 'topic pair counts mismatch')
    c.require(set(f.groupby('pair_id').size()) == {16} and
              not f.duplicated(['pair_id', 'operator', 'cell', 'ordering']).any(), 'incomplete/duplicate 16 variants')
    facts = pd.concat([f[[f'entity_{s}_id', f'canonical_truth_{s}', f'fact_{s}_id']].set_axis(
        ['entity', 'truth', 'fact'], axis=1) for s in ['a', 'b']], ignore_index=True)
    c.require(facts.groupby(['entity', 'truth']).fact.nunique().eq(1).all(), 'fact identity varies for entity/truth')
    return f


def match_and_or(frame):
    keys = ['pair_id', 'canonical_truth_a', 'canonical_truth_b', 'ordering']
    cols = keys + ['example_id', 'topic', 'entity_a_id', 'entity_b_id', 'fact_a_id', 'fact_b_id', 'cell', 'frozen_probe_score']
    left, right = [frame.loc[frame.operator == op, cols] for op in ['AND', 'OR']]
    result = left.merge(right, on=keys, how='outer', suffixes=('_and', '_or'), validate='one_to_one', indicator=True)
    c.require(result['_merge'].eq('both').all() and len(result) * 2 == len(frame), 'missing AND/OR match')
    for col in ['topic', 'entity_a_id', 'entity_b_id', 'fact_a_id', 'fact_b_id', 'cell']:
        c.require(result[col + '_and'].eq(result[col + '_or']).all(), 'matched invariant mismatch')
        result[col] = result[col + '_and']
    result['delta_or_minus_and'] = result.frozen_probe_score_or - result.frozen_probe_score_and
    c.require(result.groupby('cell').size().to_dict() == {cell: len(frame)//8 for cell in c.CELLS}, 'matched cell counts mismatch')
    result['interpretation'] = np.where(result.cell.isin(['TT', 'FF']), 'same_truth_connective_shift', 'connective_and_truth_change')
    return result.drop(columns='_merge').sort_values(keys).reset_index(drop=True)


def verify_score_inputs(root, spec):
    output = c.safe_path(root, c.OUTPUT)
    manifest_path = c.safe_path(root, f'{c.OUTPUT}/scoring_manifest.json')
    scores_path = c.safe_path(root, f'{c.OUTPUT}/row_scores.csv')
    manifest = c.read_json(manifest_path)
    c.require(manifest['complete'] is True and manifest['implementation_version'] == 'pinned-qwen25-lr-scoring-v1', 'incomplete scoring')
    c.require(manifest['analysis_spec_sha256'] == c.file_hash(c.safe_path(root, c.SPEC)) and manifest['inputs'] == spec['inputs'], 'score spec/input mismatch')
    c.require(manifest['model'] == c.MODEL and manifest['model_revision'] == manifest['tokenizer_revision'] == c.REVISION and
              manifest['scoring'] == spec['scoring'] and
              set(manifest['metadata_columns_materialized']) <= set(c.PROJECTION) and
              manifest['selected_probe_sha256'] == spec['inputs']['probe']['files']['selected_probe.npz']['sha256'] and
              manifest['repaired_cache_identity_sha256'] == c.digest(c.canonical(spec['inputs']['atomic']['files'])) and
              manifest['classes'] == [0, 1] and manifest['coefficient_shape'] == [1, c.WIDTH] and
              manifest['intercept_shape'] == [1] and manifest['fit_split'] == 'train' and
              manifest['selection_split'] == 'validation' and manifest['refit_on_validation'] is False,
              'scoring provenance/binding mismatch')
    c.require(manifest['representation_fingerprint'] == c.FINGERPRINT and manifest['rows'] == c.ROWS and
              manifest['selected_layer'] == spec['expected_selection']['layer'] and manifest['C'] == spec['expected_selection']['C'] and
              manifest['score_columns'] == ['example_id', 'frozen_probe_score'] and
              manifest['compound_truth_columns_materialized'] is False and manifest['compound_labels_used'] is False and
              manifest['test_artifacts_accessed'] is False and manifest['fit_operations'] == 0, 'invalid scoring contract')
    c.require(c.record(scores_path) == manifest['row_scores'], 'score file hash/size mismatch')
    metadata_path = c.safe_path(root, f'{c.COMPOUND}/metadata.csv')
    c.require(c.record(metadata_path) == spec['inputs']['compound']['files']['metadata.csv'], 'canonical metadata hash mismatch')
    with scores_path.open(encoding='utf-8', newline='') as handle:
        c.require(next(csv.reader(handle)) == ['example_id', 'frozen_probe_score'], 'score schema mismatch')
    scores = pd.read_csv(scores_path, dtype={'example_id': str}, keep_default_na=False, float_precision='round_trip')
    c.require(len(scores) == c.ROWS and scores.example_id.is_unique and scores.example_id.str.strip().ne('').all(), 'duplicate/missing score IDs')
    scores['frozen_probe_score'] = scores.frozen_probe_score.astype(np.float64)
    c.require(np.isfinite(scores.frozen_probe_score).all(), 'nonfinite scores')
    projected = c.projected_metadata(metadata_path)
    c.require(set(scores.example_id) == set(projected.example_id), 'score/metadata ID coverage mismatch')
    c.require(c.ordered_hash(scores.example_id) == c.ordered_hash(projected.example_id) == manifest['ordered_example_id_sha256'],
              'ordered ID hash mismatch')
    return scores, projected, metadata_path, manifest


def load_evaluation_rows(root, spec):
    scores, projected, path, manifest = verify_score_inputs(root, spec)
    # First and only full truth-bearing read: all score/hash/ID gates above passed.
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    c.require(c.record(path) == spec['inputs']['compound']['files']['metadata.csv'] and
              raw.example_id.tolist() == projected.example_id.tolist(), 'metadata changed during label read')
    frame = validate_metadata(raw, spec['benchmark'])
    frame = frame.merge(scores, on='example_id', how='left', validate='one_to_one')
    c.require(np.isfinite(frame.frozen_probe_score).all(), 'missing joined scores')
    return frame.sort_values('example_id').reset_index(drop=True), manifest


def finite_records(frame):
    # JSON null for undefined metrics; never emit NaN tokens.
    return frame.astype(object).where(pd.notna(frame), None).to_dict('records')


def evaluate(root=c.ROOT):
    spec_path = c.safe_path(root, c.SPEC)
    spec = c.validate_spec(c.read_json(spec_path))
    spec_sha = c.file_hash(spec_path)
    output = c.safe_path(root, f'{c.OUTPUT}/evaluation')
    c.require(not output.exists(), 'refusing existing evaluation output')
    frame, scoring_manifest = load_evaluation_rows(root, spec)
    matched = match_and_or(frame)
    plan, schedule = MetricPlan(frame, matched, spec), EntityBootstrap(frame, spec['bootstrap'])
    with threadpool_limits(limits=1):
        point = plan.evaluate(np.ones((1, len(frame))))[0]
        draws = np.empty((spec['bootstrap']['replicates'], len(plan.records)), dtype=np.float64)
        for start in range(0, len(draws), 64):
            stop = min(start + 64, len(draws))
            draws[start:stop] = plan.evaluate(schedule.row_weights(start, stop))
    metrics = pd.DataFrame([{**r, 'estimate': point[i], **interval(draws[:, i], spec['bootstrap'])}
                            for i, r in enumerate(plan.records)])
    metrics['majority_accuracy_baseline'] = np.where(metrics.metric.isin(['and_accuracy', 'or_accuracy']), .75, np.nan)
    cells, geometry = [], []
    for scope, topic in [('pooled', 'all')] + [('topic', t) for t in spec['benchmark']['topics']]:
        subset = frame if scope == 'pooled' else frame[frame.topic == topic]
        for (op, cell), rows in subset.groupby(['operator', 'cell']):
            cells.append(dict(scope=scope, topic=topic, operator=op, cell=cell, **describe(rows.frozen_probe_score)))
        subset = matched if scope == 'pooled' else matched[matched.topic == topic]
        for cell in ['all', *c.CELLS]:
            values = subset if cell == 'all' else subset[subset.cell == cell]
            geometry.append(dict(metric='or_minus_and_summary', category='geometry', scope=scope, topic=topic,
                                 cell=cell, **describe(values.delta_or_minus_and)))
    output.mkdir(parents=True, exist_ok=False)
    tables = {'primary_metrics': metrics[metrics.category == 'primary'],
              'boundary_metrics': metrics[metrics.category == 'boundary'],
              'threshold_metrics': metrics[metrics.category == 'threshold'],
              'descriptive_geometry': pd.concat([metrics[metrics.category == 'geometry'], pd.DataFrame(geometry)], ignore_index=True),
              'cell_statistics': pd.DataFrame(cells), 'topic_metrics': metrics[metrics.scope != 'pooled'],
              'matched_and_or': matched, 'bootstrap_summary': metrics}
    for name, table in tables.items():
        table.to_csv(output / (name + '.csv'), index=False, float_format='%.17g')
    for name in ['primary_metrics', 'bootstrap_summary']:
        c.publish_json(output / (name + '.json'), finite_records(tables[name]))
    np.savez_compressed(output / 'bootstrap_draws.npz', metric_ids=np.asarray(metrics.metric_id),
                        values=draws, valid=np.isfinite(draws))
    np.savez_compressed(output / 'bootstrap_pair_weights.npz', pair_ids=np.asarray(schedule.pair_ids), weights=schedule.weights)
    c.require(c.file_hash(spec_path) == spec_sha, 'spec changed during evaluation')
    manifest = dict(schema_version=1, implementation_version=VERSION, complete=True,
                    analysis_spec_sha256=spec_sha, scoring_manifest_sha256=c.file_hash(c.safe_path(root, f'{c.OUTPUT}/scoring_manifest.json')),
                    row_scores=scoring_manifest['row_scores'], metadata=spec['inputs']['compound']['files']['metadata.csv'],
                    representation_fingerprint=c.FINGERPRINT, rows=len(frame), matched_rows=len(matched),
                    statistics=spec['statistics'], bootstrap=spec['bootstrap'], schedule_sha256=schedule.sha256,
                    bootstrap_entities=schedule.entities, provenance=c.code_provenance(),
                    atomic_or_compound_test_accessed=False, activation_arrays_opened=False, probe_archives_opened=False,
                    completed_utc=datetime.now(timezone.utc).isoformat(),
                    outputs={p.name: c.record(p) for p in sorted(output.iterdir())})
    c.publish_json(output / 'evaluation_manifest.json', manifest)
    return dict(output=str(output.relative_to(root)), complete=True, rows=len(frame))
