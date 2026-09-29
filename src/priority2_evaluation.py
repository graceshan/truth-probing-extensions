"""Label-aware Priority-2 evaluation; no activation arrays or probe archives."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import clean_transfer_contracts as c
from src import priority2_input_controls as p
from src import priority2_transfer as t
from src.clean_transfer_evaluation import validate_metadata, match_and_or, finite_records
from src.clean_transfer_statistics import EntityBootstrap, MetricPlan, AUC, describe, interval


def composition(a,b,operator):
    a,b,operator = np.asarray(a,float),np.asarray(b,float),np.asarray(operator)
    c.require(a.shape == b.shape == operator.shape and np.isfinite(a).all() and np.isfinite(b).all() and
              np.isin(operator,['AND','OR']).all(), 'composition inputs')
    continuous = np.where(operator == 'AND',np.minimum(a,b),np.maximum(a,b))
    boolean = np.where(operator == 'AND',(a>=0)&(b>=0),(a>=0)|(b>=0))
    return continuous,boolean


def load(root,spec):
    output = c.safe_path(root,p.OUTPUT+'/scores')
    manifest = c.read_json(c.safe_path(root,p.OUTPUT+'/scores/scoring_manifest.json'))
    c.require(manifest['complete'] is True and manifest['version'] == p.VERSION and
              manifest['analysis_spec_sha256'] == c.file_hash(c.safe_path(root,p.SPEC)) and
              manifest['inputs'] == spec['inputs'] and manifest['representation_fingerprint'] == c.FINGERPRINT and
              manifest['selected_layer'] == spec['expected_selection']['layer'] and manifest['C'] == spec['expected_selection']['C'] and
              manifest['selected_probe_sha256'] == spec['lr_spec']['inputs']['probe']['files']['selected_probe.npz']['sha256'] and
              manifest['score_columns'] == t.SCORE_COLUMNS and manifest['isolated_columns'] == t.ISOLATED_COLUMNS and
              manifest['fit_operations'] == 0 and manifest['compound_truth_columns_materialized'] is False and
              manifest['compound_labels_used'] is False and manifest['test_accessed'] is False, 'scoring binding mismatch')
    for name in ['condition_scores.csv','isolated_scores.csv']:
        c.require(c.record(c.safe_path(root,p.OUTPUT+'/scores/'+name)) == manifest['outputs'][name], 'score hash mismatch')
    generation,files = p.verify_generation(root)
    c.require(files == spec['inputs']['generation']['files'], 'frozen generation mismatch')
    path = c.safe_path(root,p.RAW)
    c.require(c.record(path) == generation['raw_benchmark'], 'raw benchmark identity mismatch')
    projected = c.projected_metadata(path)
    scores = pd.read_csv(output/'condition_scores.csv',keep_default_na=False,float_precision='round_trip')
    isolated = pd.read_csv(output/'isolated_scores.csv',keep_default_na=False,float_precision='round_trip')
    index = pd.read_csv(c.safe_path(root,p.DATA+'/scoring_index.csv'),dtype=str,keep_default_na=False)
    sources = pd.read_csv(c.safe_path(root,p.DATA+'/isolated_sources.csv'),dtype=str,keep_default_na=False)
    c.require(list(scores) == t.SCORE_COLUMNS and scores.iloc[:,:3].equals(index) and scores.example_id.is_unique and
              len(scores) == manifest['condition_rows'] == 3*c.ROWS//2 and np.isfinite(scores.frozen_probe_score).all() and
              c.ordered_hash(scores.example_id) == manifest['ordered_example_id_sha256'], 'condition identity/cardinality mismatch')
    c.require(list(isolated) == t.ISOLATED_COLUMNS and isolated.fact_key.tolist() == sources.fact_key.tolist() and
              isolated.fact_key.is_unique and len(isolated) == manifest['isolated_rows'] == spec['condition_counts'][p.ISO] and
              c.ordered_hash(isolated.fact_key) == manifest['ordered_fact_key_sha256'] and np.isfinite(isolated.frozen_probe_score).all(), 'isolated score identities')
    for name,expected in spec['inputs']['lr_reference']['files'].items():
        c.require(c.record(c.safe_path(root,c.OUTPUT+'/'+name)) == expected, 'raw LR reference changed')
    raw_scores = pd.read_csv(c.safe_path(root,c.OUTPUT+'/row_scores.csv'),keep_default_na=False,float_precision='round_trip')
    c.require(list(raw_scores) == ['example_id','frozen_probe_score'] and raw_scores.example_id.tolist() == projected.example_id.tolist() and
              raw_scores.example_id.is_unique and np.isfinite(raw_scores.frozen_probe_score).all(), 'raw LR exact coverage')
    # FIRST full compound truth read: all hashes, score IDs and coverage checked above.
    raw = pd.read_csv(path,dtype=str,keep_default_na=False)
    c.require(c.record(path) == generation['raw_benchmark'] and raw.example_id.tolist() == projected.example_id.tolist(), 'raw changed during read')
    tables,facts,mapping = p.variants(raw)
    for name,frame in {**{k+'.csv':v for k,v in tables.items()},'isolated_facts.csv':facts,'constituent_map.csv':mapping}.items():
        payload = p.csv_bytes(frame)
        c.require(dict(sha256=c.digest(payload),bytes=len(payload)) == files[name], 'same-fact generated table mismatch')
    frame = validate_metadata(raw,spec['benchmark']).merge(raw_scores,on='example_id',validate='one_to_one').sort_values('example_id').reset_index(drop=True)
    return frame,tables,mapping,scores,isolated,manifest


def compute(frame,tables,mapping,scores,isolated,spec):
    frame = frame.sort_values('example_id').reset_index(drop=True)
    schedule = EntityBootstrap(frame,spec['bootstrap'])
    c.require(schedule.sha256 == spec['inputs']['lr_reference']['schedule_sha256'], 'raw endpoint schedule identity mismatch')
    conditions = {p.RAW_ID:frame.copy()}
    for condition in [p.BOTH,p.LEAST]:
        variant = scores[scores.condition_id == condition].set_index('base_example_id').frozen_probe_score
        f = frame.copy()
        mask = f.operator == 'OR'
        c.require(set(variant.index) == set(f.loc[mask,'example_id']), 'OR one-to-one base mapping')
        f.loc[mask,'frozen_probe_score'] = f.loc[mask,'example_id'].map(variant)
        conditions[condition] = f
    lookup = isolated.set_index('fact_key').frozen_probe_score
    mapped = frame[['example_id','operator']].merge(mapping,left_on='example_id',right_on='base_example_id',validate='one_to_one')
    continuous,boolean = composition(mapped.fact_a_key.map(lookup),mapped.fact_b_key.map(lookup),mapped.operator)
    conditions[p.EXTERNAL] = frame.assign(frozen_probe_score=continuous)
    # MetricPlan's frozen >=0 threshold: encode booleans as -1/+1, NEVER use 0 for false.
    conditions[p.BOOLEAN] = frame.assign(frozen_probe_score=np.where(boolean,1.,-1.))
    records,points,draws,indices = [],[],[],{}
    def append(condition,record,point,values):
        suffix = record['metric_id']
        indices[condition,suffix] = len(records)
        records.append({**record,'condition_id':condition,'metric_id':condition+'/'+suffix})
        points.append(float(point))
        draws.append(values)
    with threadpool_limits(limits=1):
        for condition,f in conditions.items():
            definitions = spec['boolean_metrics'] if condition == p.BOOLEAN else spec['formal_metrics']
            plan = MetricPlan(f,match_and_or(f),{**spec,'metrics':definitions})
            point = plan.evaluate(np.ones((1,len(f))))[0]
            values = np.empty((spec['bootstrap']['replicates'],len(plan.records)))
            for start in range(0,len(values),64):
                stop = min(start+64,len(values))
                values[start:stop] = plan.evaluate(schedule.row_weights(start,stop))
            for i,r in enumerate(plan.records): append(condition,r,point[i],values[:,i])
        # Juxtaposition has no operator or compound label. Use AUC directly, never MetricPlan.
        jux = tables[p.JUX].merge(scores[scores.condition_id == p.JUX][['example_id','frozen_probe_score']],on='example_id',validate='one_to_one')
        jux['cell'] = np.char.add(np.where(jux.canonical_truth_a,'T','F'),np.where(jux.canonical_truth_b,'T','F'))
        pair_index = {pid:i for i,pid in enumerate(schedule.pair_ids)}
        weights = schedule.weights[:,[pair_index[x] for x in jux.pair_id]]
        scopes = [('pooled','all')]+[('topic',topic) for topic in spec['benchmark']['topics']]
        summaries = []
        for scope,topic in scopes:
            mask = np.ones(len(jux),bool) if scope == 'pooled' else (jux.topic == topic).to_numpy()
            for cell in c.CELLS:
                summaries.append(dict(condition_id=p.JUX,category='geometry',scope=scope,topic=topic,cell=cell,
                                      **describe(jux.loc[mask & (jux.cell == cell),'frozen_probe_score'])))
            for definition in spec['juxtaposition_metrics']:
                positive = jux.cell.isin(definition['positive']).to_numpy()
                selected = np.flatnonzero(mask & (positive | jux.cell.isin(definition['negative']).to_numpy()))
                auc = AUC(selected,jux.frozen_probe_score.to_numpy()[selected],positive[selected])
                append(p.JUX,dict(metric_id=f'{scope}/{topic}/'+definition['id'],metric=definition['id'],scope=scope,topic=topic,category='geometry'),
                       auc(np.ones((1,len(jux))))[0],auc(weights))
        for definition in spec['juxtaposition_metrics']:
            ix = [indices[p.JUX,'topic/'+topic+'/'+definition['id']] for topic in spec['benchmark']['topics']]
            append(p.JUX,dict(metric_id='topic_macro/all/'+definition['id'],metric=definition['id'],scope='topic_macro',topic='all',category='geometry'),
                   np.mean([points[i] for i in ix]),np.mean([draws[i] for i in ix],axis=0))
    for contrast in spec['paired_contrasts']:
        for scope,topic in [('pooled','all')]+[('topic',x) for x in spec['benchmark']['topics']]+[('topic_macro','all')]:
            for metric in contrast['metrics']:
                suffix = f'{scope}/{topic}/{metric}'
                left,right = [indices[k,suffix] for k in [contrast['left'],contrast['right']]]
                append(contrast['left']+'_minus_'+contrast['right'],dict(metric_id=suffix,metric=metric,scope=scope,topic=topic,category='paired_contrast'),
                       points[left]-points[right],draws[left]-draws[right])
    values = np.column_stack(draws)
    metrics = pd.DataFrame([{**r,'estimate':points[i],**interval(values[:,i],spec['bootstrap'])} for i,r in enumerate(records)])
    metrics['majority_accuracy_baseline'] = np.where(metrics.metric.isin(['and_accuracy','or_accuracy']),.75,np.nan)
    return metrics,values,schedule,pd.DataFrame(summaries)


def evaluate(root=c.ROOT):
    spec = t.validate(c.read_json(c.safe_path(root,p.SPEC)))
    spec_sha = c.file_hash(c.safe_path(root,p.SPEC))
    output = c.safe_path(root,p.OUTPUT+'/evaluation')
    c.require(not output.exists(), 'refusing existing evaluation output')
    frame,tables,mapping,scores,isolated,manifest = load(root,spec)
    metrics,draws,schedule,summaries = compute(frame,tables,mapping,scores,isolated,spec)
    # Reproduce the known LR primary table before publishing new condition results.
    from src.pinned_method_evaluation import primary_gate
    from src import method_transfer_contracts as m
    raw = metrics[metrics.condition_id == p.RAW_ID].assign(analysis_group=m.GROUPS[0],method='l2_logistic')
    primary_gate(root,raw,{'reproduction':{'primary_atol':1e-14}})
    output.mkdir(parents=True)
    outputs = dict(formal_metrics=metrics[metrics.category.isin(['primary','boundary'])],
        paired_contrasts=metrics[metrics.category == 'paired_contrast'],boolean_metrics=metrics[metrics.category == 'threshold'],
        juxtaposition_geometry=metrics[metrics.category == 'geometry'],juxtaposition_cell_statistics=summaries,bootstrap_summary=metrics)
    for name,table in outputs.items(): table.to_csv(output/(name+'.csv'),index=False,float_format='%.17g')
    c.publish_json(output/'formal_metrics.json',finite_records(outputs['formal_metrics']))
    np.savez_compressed(output/'bootstrap_draws.npz',metric_ids=metrics.metric_id.to_numpy(str),values=draws,valid=np.isfinite(draws))
    np.savez_compressed(output/'bootstrap_pair_weights.npz',pair_ids=np.asarray(schedule.pair_ids),weights=schedule.weights)
    c.require(c.file_hash(c.safe_path(root,p.SPEC)) == spec_sha, 'spec changed')
    for name,record in manifest['outputs'].items():
        c.require(c.record(c.safe_path(root,p.OUTPUT+'/scores/'+name)) == record, 'scores changed during evaluation')
    c.publish_json(output/'evaluation_manifest.json',dict(complete=True,version=p.VERSION,analysis_spec_sha256=spec_sha,
        scoring_manifest=c.record(c.safe_path(root,p.OUTPUT+'/scores/scoring_manifest.json')),schedule_sha256=schedule.sha256,
        bootstrap=spec['bootstrap'],composition=spec['composition'],raw_primary_reproduced=True,
        activation_arrays_opened=False,probe_archives_opened=False,test_accessed=False,fit_operations=0,
        provenance=t.code(),outputs={path.name:c.record(path) for path in output.iterdir()}))
    return dict(output=str(output),complete=True)
