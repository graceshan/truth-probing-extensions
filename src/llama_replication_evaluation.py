"""Label-aware Llama evaluation from frozen score CSVs; no model/cache/archive reads."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import clean_transfer_contracts as c
from src import llama_replication_contracts as l
from src import priority2_input_controls as p
from src.clean_transfer_evaluation import validate_metadata, match_and_or, parse_bool, finite_records
from src.clean_transfer_statistics import EntityBootstrap, MetricPlan, AUC, interval
from src.priority2_evaluation import composition


def load(root):
    from src.llama_replication_transfer import validate_spec
    from src.llama_replication_extraction import transfer_rows
    spec = validate_spec(c.read_json(c.safe_path(root, l.SPEC)), root)
    manifest = c.read_json(c.safe_path(root, l.OUTPUT+'/scores/scoring_manifest.json'))
    score_path = c.safe_path(root, l.OUTPUT+'/scores/scores.csv')
    rows, files = transfer_rows(root)  # hashes and projection only, no truth columns
    c.require(files == spec['inputs']['data'] and manifest == dict(complete=True,
        analysis_spec_sha256=c.file_hash(c.safe_path(root, l.SPEC)), score_file=c.record(score_path),
        rows=sum(l.COUNTS.values()), columns=l.SCORE_COLUMNS,
        ordered_example_id_sha256=spec['ordered_example_id_sha256'], compound_truth_columns_materialized=False,
        fit_operations=0, test_accessed=False, selected=spec['selected'], representation_fingerprint=spec['pin']['fingerprint']),
        'scoring completion/binding mismatch')
    scores = pd.read_csv(score_path, keep_default_na=False, float_precision='round_trip')
    c.require(list(scores) == l.SCORE_COLUMNS and scores.example_id.is_unique and
              scores[['example_id', 'condition_id']].equals(rows.frame[['example_id', 'condition_id']]) and
              np.isfinite(scores.frozen_probe_score).all() and
              c.ordered_hash(scores.example_id) == spec['ordered_example_id_sha256'], 'score one-to-one identity')
    # FIRST truth-bearing read, after score/spec/metadata identities and coverage pass.
    raw = pd.read_csv(c.safe_path(root, p.RAW), dtype=str, keep_default_na=False)
    frame = validate_metadata(raw, spec['policy']['benchmark']).merge(
        scores.loc[scores.condition_id == p.RAW_ID, ['example_id', 'frozen_probe_score']],
        on='example_id', validate='one_to_one').sort_values('example_id').reset_index(drop=True)
    tables = {key: pd.read_csv(c.safe_path(root, p.DATA+'/'+key+'.csv'), dtype=str, keep_default_na=False)
              for key in [p.BOTH, p.JUX]}
    mapping = pd.read_csv(c.safe_path(root, p.DATA+'/constituent_map.csv'), dtype=str, keep_default_na=False)
    facts = pd.read_csv(c.safe_path(root, p.DATA+'/isolated_facts.csv'), dtype=str, keep_default_na=False)
    c.require(set(mapping.base_example_id) == set(frame.example_id) and mapping.base_example_id.is_unique and
              facts.fact_key.is_unique and set(facts.fact_key) == set(scores.loc[scores.condition_id == p.ISO, 'example_id']),
              'same-fact coverage mismatch')
    fact_lookup = facts.set_index('fact_key')
    bound = frame.merge(mapping, left_on='example_id', right_on='base_example_id', validate='one_to_one')
    for side in ['a', 'b']:
        keys = bound['fact_'+side+'_key']
        for target, source in [('fact_id', 'fact_'+side+'_id'), ('entity_id', 'entity_'+side+'_id'),
                               ('statement', 'fact_'+side+'_statement'), ('topic', 'topic')]:
            c.require(keys.map(fact_lookup[target]).eq(bound[source]).all(), 'constituent provenance mismatch')
    return spec, frame, tables, mapping, scores, manifest


def compute(frame, tables, mapping, scores, policy):
    """Pure numeric computation; shared schedule across every condition/contrast."""
    frame = frame.sort_values('example_id').reset_index(drop=True)
    schedule = EntityBootstrap(frame, policy['bootstrap'])
    c.require(schedule.sha256 == policy['schedule_sha256'], 'Qwen endpoint schedule identity mismatch')
    lookup = scores.set_index('example_id').frozen_probe_score
    variants = tables[p.BOTH].set_index('base_example_id').example_id.map(lookup)
    mask = frame.operator == 'OR'
    c.require(variants.index.is_unique and set(variants.index) == set(frame.loc[mask, 'example_id']) and
              np.isfinite(variants).all(), 'OR-both coverage')
    variant = frame.copy()
    variant.loc[mask, 'frozen_probe_score'] = frame.loc[mask, 'example_id'].map(variants)
    mapped = frame[['example_id', 'operator']].merge(mapping, left_on='example_id', right_on='base_example_id', validate='one_to_one')
    continuous, boolean = composition(mapped.fact_a_key.map(lookup), mapped.fact_b_key.map(lookup), mapped.operator)
    conditions = {p.RAW_ID: frame, p.BOTH: variant,
                  p.EXTERNAL: frame.assign(frozen_probe_score=continuous),
                  p.BOOLEAN: frame.assign(frozen_probe_score=np.where(boolean, 1., -1.))}
    records, points, draws, indices = [], [], [], {}

    def append(condition, record, point, samples):
        suffix = record['metric_id']
        indices[condition, suffix] = len(records)
        records.append({**record, 'condition_id': condition, 'metric_id': condition+'/'+suffix})
        points.append(float(point))
        draws.append(samples)

    with threadpool_limits(limits=1):
        for condition, f in conditions.items():
            definitions = policy['boolean_metrics'] if condition == p.BOOLEAN else policy['formal_metrics']
            plan = MetricPlan(f, match_and_or(f), {**policy, 'metrics': definitions})
            point = plan.evaluate(np.ones((1, len(f))))[0]
            values = np.empty((policy['bootstrap']['replicates'], len(plan.records)))
            for start in range(0, len(values), 64):
                stop = min(start+64, len(values))
                values[start:stop] = plan.evaluate(schedule.row_weights(start, stop))
            for i, record in enumerate(plan.records): append(condition, record, point[i], values[:, i])
        jux = tables[p.JUX].copy()
        c.require('operator' not in jux and 'compound_label' not in jux, 'juxtaposition cannot have formal labels')
        jux['frozen_probe_score'] = jux.example_id.map(lookup)
        c.require(np.isfinite(jux.frozen_probe_score).all(), 'juxtaposition score coverage')
        jux['cell'] = np.char.add(np.where(parse_bool(jux.canonical_truth_a), 'T', 'F'),
                                  np.where(parse_bool(jux.canonical_truth_b), 'T', 'F'))
        pair_lookup = {pid: i for i, pid in enumerate(schedule.pair_ids)}
        weights = schedule.weights[:, [pair_lookup[x] for x in jux.pair_id]]
        for scope, topic in [('pooled', 'all')]+[('topic', x) for x in policy['benchmark']['topics']]:
            mask = np.ones(len(jux), bool) if scope == 'pooled' else (jux.topic == topic).to_numpy()
            for definition in policy['juxtaposition_metrics']:
                positive = jux.cell.isin(definition['positive']).to_numpy()
                selected = np.flatnonzero(mask & (positive | jux.cell.isin(definition['negative']).to_numpy()))
                auc = AUC(selected, jux.frozen_probe_score.to_numpy()[selected], positive[selected])
                append(p.JUX, dict(metric_id=f'{scope}/{topic}/'+definition['id'], metric=definition['id'],
                    scope=scope, topic=topic, category='geometry'), auc(np.ones((1, len(jux))))[0], auc(weights))
        for definition in policy['juxtaposition_metrics']:
            ix = [indices[p.JUX, 'topic/'+topic+'/'+definition['id']] for topic in policy['benchmark']['topics']]
            append(p.JUX, dict(metric_id='topic_macro/all/'+definition['id'], metric=definition['id'],
                scope='topic_macro', topic='all', category='geometry'), np.mean([points[i] for i in ix]),
                np.mean([draws[i] for i in ix], axis=0))
    for contrast in policy['paired_contrasts']:
        for scope, topic in [('pooled', 'all')]+[('topic', x) for x in policy['benchmark']['topics']]+[('topic_macro', 'all')]:
            for metric in contrast['metrics']:
                suffix = f'{scope}/{topic}/{metric}'
                left, right = [indices[key, suffix] for key in [contrast['left'], contrast['right']]]
                append(contrast['left']+'_minus_'+contrast['right'], dict(metric_id=suffix, metric=metric,
                    scope=scope, topic=topic, category='paired_contrast'), points[left]-points[right], draws[left]-draws[right])
    values = np.column_stack(draws)
    metrics = pd.DataFrame([{**record, 'estimate': points[i], **interval(values[:, i], policy['bootstrap'])}
                           for i, record in enumerate(records)])
    return metrics, values, schedule


def evaluate(root=c.ROOT):
    output = c.safe_path(root, l.OUTPUT+'/evaluation')
    c.require(not output.exists(), 'no overwrite/resume')
    spec, frame, tables, mapping, scores, manifest = load(root)
    metrics, draws, schedule = compute(frame, tables, mapping, scores, spec['policy'])
    output.mkdir(parents=True, exist_ok=False)
    categories = dict(primary_metrics='primary', boundary_metrics='boundary', boolean_metrics='threshold',
                      juxtaposition_geometry='geometry', paired_contrasts='paired_contrast')
    for name, category in categories.items():
        table = metrics[metrics.category == category]
        table.to_csv(output/(name+'.csv'), index=False, float_format='%.17g')
        c.publish_json(output/(name+'.json'), finite_records(table))
    metrics.to_csv(output/'bootstrap_summary.csv', index=False, float_format='%.17g')
    np.savez_compressed(output/'bootstrap_draws.npz', metric_ids=metrics.metric_id.to_numpy(str),
                        values=draws, valid=np.isfinite(draws))
    np.savez_compressed(output/'bootstrap_pair_weights.npz', pair_ids=np.asarray(schedule.pair_ids), weights=schedule.weights)
    c.require(l.data_files(root) == spec['inputs']['data'] and
              c.record(c.safe_path(root, l.OUTPUT+'/scores/scores.csv')) == manifest['score_file'], 'evaluation inputs changed')
    c.publish_json(output/'evaluation_manifest.json', dict(complete=True, analysis_spec_sha256=manifest['analysis_spec_sha256'],
        score_file=manifest['score_file'], schedule_sha256=schedule.sha256, bootstrap=spec['policy']['bootstrap'],
        rows=len(frame), activation_arrays_opened=False, probe_archives_opened=False, test_accessed=False,
        outputs={path.name: c.record(path) for path in output.iterdir()}))
    return dict(output=str(output), complete=True)
