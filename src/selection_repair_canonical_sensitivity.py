"""Two fixed canonical raw-feature LR references and paired development sensitivity."""
import io
import json
import platform
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.special import expit
from threadpoolctl import threadpool_limits

from src import clean_atomic_probes as canonical_lr
from src.clean_transfer_statistics import AUC, EntityBootstrap, interval
from src.selection_repair_cache_export_remote import header, DIRECTORIES
from src import selection_repair_sensitivity_inputs as s


def gradient_diagnostics(X, y, coef, intercept, C):
    """sklearn binary LR: mean BCE + ||w||²/(2*C*n), free intercept."""
    logits = X @ coef + intercept
    residual = expit(logits) - y
    gradient = np.r_[X.T @ residual / len(y) + coef / (C*len(y)), residual.mean()]
    loss = np.logaddexp(0, np.where(y == 1, -logits, logits)).mean() + np.dot(coef,coef)/(2*C*len(y))
    return dict(objective=float(loss), gradient_infinity_norm=float(np.max(np.abs(gradient))),
                finite=bool(np.isfinite(gradient).all() and np.isfinite(loss) and np.isfinite(coef).all() and np.isfinite(intercept)),
                objective_definition='mean BCE + sum(w**2)/(2*C*n); intercept unpenalized; raw features')


def fit_reference(X, y, C, layer):
    s.require(X.dtype == np.float64 and np.isfinite(X).all(), 'finite raw float64 matrix required')
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            probe, diagnostics = canonical_lr.fit_converged_probe(X, y, C, layer=layer)
    diagnostics.update(gradient_diagnostics(X, y, probe.coef_[0], float(probe.intercept_[0]), C))
    diagnostics['library_warnings'] = [str(w.message) for w in caught]
    diagnostics['raw_gradient_target_met'] = diagnostics['gradient_infinity_norm'] <= canonical_lr.PROBE_CONFIG['tol']
    diagnostics['valid_for_scoring'] = diagnostics['finite'] and diagnostics['final_converged'] and diagnostics['raw_gradient_target_met']
    return probe, diagnostics


def atomic_weights(rows, eligible, options):
    weights = np.zeros((options['replicates'], len(rows)), dtype=np.int16)
    rng = np.random.Generator(np.random.PCG64(options['seed']))
    for topic in sorted({r['topic'] for r in rows}):
        indices = [i for i,r in enumerate(rows) if eligible[i] and r['topic']==topic]
        people = sorted({rows[i]['person_key'] for i in indices})
        if not people: continue
        counts = rng.multinomial(len(people), np.full(len(people),1/len(people)), size=options['replicates'])
        lookup = {key:i for i,key in enumerate(people)}
        weights[:,indices] = counts[:,[lookup[rows[i]['person_key']] for i in indices]]
    return weights


def compound_weights(frame, eligible, options):
    retained = frame.loc[eligible].reset_index(drop=True)
    schedule = EntityBootstrap(retained, options)
    lookup = {key:i for i,key in enumerate(schedule.pair_ids)}
    weights = np.zeros((options['replicates'],len(frame)),dtype=np.int16)
    indices = np.flatnonzero(eligible)
    weights[:,indices] = schedule.weights[:,[lookup[frame.iloc[i].pair_id] for i in indices]]
    return weights, schedule.sha256


def auc_draws(scores, labels, indices, weights):
    fn = AUC(indices, scores[indices], labels[indices])
    point = float(fn(np.ones((1,len(scores))))[0])
    draws = np.concatenate([fn(weights[start:start+64]) for start in range(0,len(weights),64)])
    return point, draws


def compare_endpoint(old, new, labels, base_mask, eligible, topics, weights, definition, model, options, all_topics):
    """Same retained rows and bootstrap weights for both probes; full old coverage separate."""
    records, draws = [], []
    groups = [('pooled','all',np.ones(len(old),bool))] + [('topic',topic,topics==topic) for topic in all_topics]
    topic_values = []
    for scope, topic, group in groups:
        idx = np.flatnonzero(base_mask & eligible & group)
        full = np.flatnonzero(base_mask & group)
        old_point, old_draws = auc_draws(old,labels,idx,weights)
        new_point, new_draws = auc_draws(new,labels,idx,weights)
        historical_full = float(AUC(full,old[full],labels[full])(np.ones((1,len(old))))[0])
        delta = new_draws-old_draws
        record = dict(model=model,condition=definition['condition'],metric=definition['metric'],scope=scope,topic=topic,
                      status='computed',historical_full_auroc=historical_full,historical_retained_auroc=old_point,
                      corrected_retained_auroc=new_point,training_delta=new_point-old_point,
                      coverage_delta=old_point-historical_full,total_delta=new_point-historical_full,
                      original_rows=len(full),retained_rows=len(idx),excluded_rows=len(full)-len(idx),
                      historical_ci=interval(old_draws,options),corrected_ci=interval(new_draws,options),
                      paired_delta_ci=interval(delta,options))
        records.append(record); draws.append((old_draws,new_draws))
        if scope=='topic':topic_values.append((record,old_draws,new_draws))
    historical = np.mean([r['historical_retained_auroc'] for r,_,_ in topic_values])
    corrected = np.mean([r['corrected_retained_auroc'] for r,_,_ in topic_values])
    old_draws = np.mean([a for _,a,_ in topic_values],axis=0); new_draws=np.mean([b for _,_,b in topic_values],axis=0)
    historical_full = np.mean([r['historical_full_auroc'] for r,_,_ in topic_values])
    records.append(dict(model=model,condition=definition['condition'],metric=definition['metric'],scope='topic_macro',topic='all',
                        status='computed',historical_full_auroc=float(historical_full),historical_retained_auroc=float(historical),
                        corrected_retained_auroc=float(corrected),training_delta=float(corrected-historical),
                        coverage_delta=float(historical-historical_full),total_delta=float(corrected-historical_full),
                        original_rows=sum(r['original_rows'] for r,_,_ in topic_values),retained_rows=sum(r['retained_rows'] for r,_,_ in topic_values),
                        excluded_rows=sum(r['excluded_rows'] for r,_,_ in topic_values),
                        historical_ci=interval(old_draws,options),corrected_ci=interval(new_draws,options),
                        paired_delta_ci=interval(new_draws-old_draws,options)))
    draws.append((old_draws,new_draws))
    # Preserve all declared topics; undefined endpoints must never look computed.
    for record in records:
        if not all(np.isfinite(record[k]) for k in
                   ('historical_full_auroc','historical_retained_auroc','corrected_retained_auroc')):
            record['status'] = 'not_computed'
            record['dependency'] = 'Both truth classes required in every declared endpoint/topic; no topic dropping'
        for key, value in list(record.items()):
            if isinstance(value, float) and not np.isfinite(value):
                record[key] = None
    return records,draws


def refresh_triggers(records, draws, config):
    summary=[(i,r) for i,r in enumerate(records) if r['status']=='computed' and r['scope'] in config['refresh']['summary_scopes']]
    triggers=[]; threshold=config['refresh']['absolute_auroc_movement_threshold']
    for i,r in summary:
        name='/'.join(str(r[k]) for k in ('model','condition','metric','scope'))
        for field,kind in [('training_delta','training_cleanup_auroc'),('coverage_delta','evaluation_coverage_auroc')]:
            if abs(r[field])>threshold:triggers.append(dict(kind=kind,endpoint=name,movement=r[field],threshold=threshold))
        if np.sign(r['historical_retained_auroc']-.5)!=np.sign(r['corrected_retained_auroc']-.5):
            triggers.append(dict(kind='chance_side_changed',endpoint=name))
    comparisons=[]
    for pos,(i,left) in enumerate(summary):
        for j,right in summary[pos+1:]:
            if left['scope']!=right['scope']:continue
            compatible=(left['model']!=right['model'] and left['condition']==right['condition'] and left['metric']==right['metric'])
            compatible|=(left['model']==right['model'] and left['condition']!=right['condition'] and left['metric']==right['metric'])
            compatible|=(left['model']==right['model'] and left['condition']==right['condition'] and {left['metric'],right['metric']}=={'and_auroc','or_auroc'})
            if not compatible:continue
            old=left['historical_retained_auroc']-right['historical_retained_auroc'];new=left['corrected_retained_auroc']-right['corrected_retained_auroc']
            old_ci=interval(draws[i][0]-draws[j][0],config['bootstrap']);new_ci=interval(draws[i][1]-draws[j][1],config['bootstrap'])
            def category(ci):
                if ci['ci_status']!='ok':return 'insufficient'
                return 'positive' if ci['ci_low']>0 else 'negative' if ci['ci_high']<0 else 'includes_zero'
            item=dict(left='/'.join(str(left[k]) for k in ('model','condition','metric','scope')),
                      right='/'.join(str(right[k]) for k in ('model','condition','metric','scope')),
                      historical_difference=old,corrected_difference=new,historical_ci=old_ci,corrected_ci=new_ci,
                      ranking_changed=bool(np.sign(old)!=np.sign(new)),interpretation_changed=category(old_ci)!=category(new_ci))
            comparisons.append(item)
            if item['ranking_changed'] or item['interpretation_changed']:
                triggers.append(dict(kind='ranking_or_contrast_interpretation_changed',**item))
    return triggers, comparisons


def run(root, payload, prepared, artifacts, output):
    root,payload,prepared,artifacts,output=map(Path,(root,payload,prepared,artifacts,output))
    from src.selection_repair_sensitivity_hold import require_current_correction
    require_current_correction(root)
    s.require(platform.system()=='Darwin','fit/evaluation must remain on macOS')
    config,inventory=s.verify_preparation(root,prepared)
    from src.selection_repair_sensitivity_verification import verify_exports
    verify_exports(root,prepared,artifacts)
    s.require(not output.exists(),'sensitivity output exists; no overwrite')
    staging=json.loads((artifacts/'staging_receipt.json').read_text());remote=json.loads((artifacts/'export_receipt.json').read_text())
    s.require(staging['preparation_receipt_sha256']==s.sha((prepared/'preparation_receipt.json').read_bytes())
              and staging['remote_receipt']==s.identity((artifacts/'export_receipt.json').read_bytes()),'staging/preparation binding')
    request=json.loads((prepared/'export_request.json').read_text())
    s.require(remote['request_sha256']==s.sha((prepared/'export_request.json').read_bytes()),'stale export request')
    physical=json.loads((root/json.loads((root/config['adoption_path']).read_text())['bindings']['physical_receipt']['path']).read_text())
    inventory_pin=next(r for r in physical['input_identities'] if r['location']=='inventory.json')
    package=s.Inputs(root,payload,inventory_pin)
    used_metadata={}; arrays={}
    def read(relative):
        key=s.ROOT_REMOTE+relative; rec=inventory['metadata_inputs'][key]
        data=(payload/rec['archive_path']).read_bytes()
        s.require(s.identity(data)=={k:rec[k] for k in ('bytes','sha256')},'restored metadata/probe changed: '+key)
        used_metadata[key]={k:rec[k] for k in ('bytes','sha256')}
        return data
    def array(key):
        if key in arrays:return arrays[key]
        rec=remote['caches'][key]
        if rec['status']!='verified':return None
        req=request['caches'][key]
        s.require(rec['saved_layer_index']==req['layer'] and rec['original_row_indices']==req['row_indices']
                  and rec['row_identity_sha256']==req['row_identity_sha256'] and rec['source_stable_through_export'], 'export binding changed')
        for field in ('path','bytes','sha256'):s.require(rec['source'][field]==req[field],'source tensor changed')
        path=artifacts/(key+'.npy')
        s.require(s.identity(path.read_bytes())=={k:rec['export'][k] for k in ('bytes','sha256')},'export file changed')
        with path.open('rb') as stream:meta=header(stream)
        s.require(meta['shape']==[len(req['row_indices']),req['shape'][2]] and meta['computed_file_bytes']==path.stat().st_size,'export header changed')
        values=np.load(path,mmap_mode='r',allow_pickle=False)
        s.require(np.isfinite(values).all(),'nonfinite selected states');arrays[key]=values
        return values
    raw=pd.read_csv(io.BytesIO(read(s.RAW+'metadata.csv')))
    raw_records=s.parse_rows(read(s.RAW+'metadata.csv'))
    raw['cell']=np.where(raw.canonical_truth_a,'T','F')+np.where(raw.canonical_truth_b,'T','F')
    both=s.parse_rows(read(s.GEN+'or_explicit_or_both_v1.csv'))
    mapping=s.parse_rows(read(s.GEN+'constituent_map.csv'))
    facts=s.parse_rows(read(s.GEN+'isolated_facts.csv'))
    atomic=s.parse_rows((prepared/'atomic_D_eligibility.csv').read_bytes())
    pairs=s.parse_rows((prepared/'development_pair_eligibility.csv').read_bytes())
    retained_pairs={r['pair_id'] for r in pairs if r['eligible']=='True'}
    atomic_keep=np.array([r['eligible']=='True' for r in atomic]);compound_keep=raw.pair_id.isin(retained_pairs).to_numpy()
    atom_w=atomic_weights(atomic,atomic_keep,config['bootstrap'])
    comp_w,comp_schedule=compound_weights(raw,compound_keep,config['bootstrap'])
    # Both models use the exact same eligibility and resampling schedules.
    output.mkdir(parents=True);(artifacts/'sensitivity_scores').mkdir(exist_ok=False)
    records=[];all_draws=[];fits={};missing=[];baseline_checks={}
    for model,settings in config['models'].items():
        train=array(model+'_train')
        if train is None:
            missing.append(dict(model=model,endpoint='all',dependency=remote['caches'][model+'_train']))
            continue
        train_rows=s.parse_rows((prepared/(model+'_train.mapping_v2.csv')).read_bytes())
        s.require([int(r['tensor_row_index']) for r in train_rows]==request['caches'][model+'_train']['row_indices'],'fit membership/export order')
        y=np.array([int(r['label']) for r in train_rows]);X=np.asarray(train,dtype=np.float64)
        prefix='atomic_probe_selection/qwen25_7b/' if model=='qwen' else 'llama31_replication_v1/probe/'
        selection=json.loads(read(prefix+'selection.json'));probe_bytes=read(prefix+'selected_probe.npz')
        s.require(selection['selected_layer']==settings['layer'] and selection['selected_C']==settings['C']
                  and selection['preprocessing']=='none' and selection['probe_configuration']==canonical_lr.PROBE_CONFIG
                  and selection['optimization_policy']==canonical_lr.OPTIMIZATION_POLICY
                  and selection['selected_probe_sha256']==s.sha(probe_bytes) and selection['all_final_fits_converged'] is True,'historical probe contract')
        with np.load(io.BytesIO(probe_bytes),allow_pickle=False) as saved:
            old_w=saved['coef'][0].copy();old_b=float(saved['intercept'][0])
            s.require(saved['coef'].shape==(1,settings['width']) and saved['intercept'].shape==(1,)
                      and np.array_equal(saved['classes'],[0,1]) and int(saved['layer'])==settings['layer']
                      and float(saved['C'])==settings['C'] and np.isfinite(old_w).all() and np.isfinite(old_b),'historical probe archive')
        print('Fitting '+model+' corrected canonical reference on '+str(len(y))+' rows',flush=True)
        probe,diagnostics=fit_reference(X,y,settings['C'],settings['layer'])
        del X
        fits[model]=dict(**diagnostics,training_rows=len(y),class_counts={str(k):int((y==k).sum()) for k in (0,1)},
                         training_membership=s.identity((prepared/(model+'_train.mapping_v2.csv')).read_bytes()),
                         source_tensor=request['caches'][model+'_train'],selected_export=remote['caches'][model+'_train']['export'],
                         C=settings['C'],saved_layer_index=settings['layer'],preprocessing='none',probe_configuration=canonical_lr.PROBE_CONFIG,
                         optimization_policy=canonical_lr.OPTIMIZATION_POLICY,historical_probe_sha256=s.sha(probe_bytes))
        print(model+' iterations='+str(diagnostics['final_n_iter'])+' gradient_inf='+str(diagnostics['gradient_infinity_norm']),flush=True)
        if not diagnostics['valid_for_scoring']:
            missing.append(dict(model=model,endpoint='all',dependency='corrected canonical fit flagged; see convergence diagnostics'));continue
        np.savez_compressed(output/(model+'_corrected_probe.npz'),coef=probe.coef_,intercept=probe.intercept_,classes=probe.classes_,
                            layer=settings['layer'],C=settings['C'],n_iter=probe.n_iter_,max_iter=probe.max_iter)
        old_scores={};new_scores={}
        def score_cache(key,ids):
            states=array(key)
            if states is None:return {},{}
            with threadpool_limits(limits=1):
                values=np.asarray(states,dtype=np.float64)
                a=values@old_w+old_b;b=probe.decision_function(values)
            s.require(len(ids)==len(a) and len(set(ids))==len(ids),'scoring identity coverage')
            return dict(zip(ids,a)),dict(zip(ids,b))
        validation=array(model+'_validation')
        if validation is not None:
            with threadpool_limits(limits=1):
                values=np.asarray(validation,dtype=np.float64)
                old_scores['atomic_D']=values@old_w+old_b;new_scores['atomic_D']=probe.decision_function(values)
            labels=np.array([int(r['label']) for r in atomic]);full=float(AUC(np.arange(len(labels)),old_scores['atomic_D'],labels)(np.ones((1,len(labels))))[0])
            recorded=selection['selected_validation_metrics']['validation_overall_auroc']
            s.require(abs(full-recorded)<1e-12,'historical atomic AUROC replay mismatch')
            baseline_checks[model]=dict(historical_atomic_full_auroc=full,recorded_atomic_full_auroc=recorded,matched=True)
        model_caches=['qwen_raw','qwen_controls'] if model=='qwen' else ['llama_transfer']
        a_lookup={};b_lookup={}
        for key in model_caches:
            rows=s.parse_rows(read(DIRECTORIES[key]+'/metadata.csv'))
            ids=[rows[i]['example_id'] for i in request['caches'][key]['row_indices']]
            a,b=score_cache(key,ids);a_lookup.update(a);b_lookup.update(b)
        raw_ids=raw.example_id.tolist()
        if all(e in a_lookup for e in raw_ids):
            old_scores['raw_reference']=np.array([a_lookup[e] for e in raw_ids]);new_scores['raw_reference']=np.array([b_lookup[e] for e in raw_ids])
        both_by_base={r['base_example_id']:r['example_id'] for r in both}
        if all(e in a_lookup for e in both_by_base.values()):
            old_scores['or_explicit_or_both_v1']=np.array([a_lookup[both_by_base[e]] if e in both_by_base else 0.0 for e in raw_ids])
            new_scores['or_explicit_or_both_v1']=np.array([b_lookup[both_by_base[e]] if e in both_by_base else 0.0 for e in raw_ids])
        if all(f['fact_key'] in a_lookup for f in facts):
            by_id={r['base_example_id']:r for r in mapping}
            for scores,lookup in [(old_scores,a_lookup),(new_scores,b_lookup)]:
                first=np.array([lookup[by_id[e]['fact_a_key']] for e in raw_ids]);second=np.array([lookup[by_id[e]['fact_b_key']] for e in raw_ids])
                scores['isolated_external_minmax_v1']=np.where(raw.operator.to_numpy()=='AND',np.minimum(first,second),np.maximum(first,second))
        # Row-level replay check against the bound saved scores, not aggregates alone.
        checks=[]
        score_groups=([('pinned_transfer_qwen25_v1/scoring/','qwen_spec','row_scores.csv','row_scores'),
                       ('priority2_input_controls_v1/scoring/','qwen_controls_spec','condition_scores.csv','outputs'),
                       ('priority2_input_controls_v1/scoring/','qwen_controls_spec','isolated_scores.csv','outputs')]
                      if model=='qwen' else [('llama31_replication_v1/analysis/scores/','llama_spec','scores.csv','score_file')])
        decision=json.loads((root/config['adoption_path']).read_text())
        for score_prefix,spec_key,filename,record_key in score_groups:
            manifest=json.loads(package.remote(score_prefix+'scoring_manifest.json'))
            s.require(manifest['complete'] is True and manifest['analysis_spec_sha256']==decision['bindings'][spec_key]['sha256'],
                      'historical score/spec binding')
            expected=manifest['outputs'][filename] if record_key=='outputs' else manifest[record_key]
            score_data=package.remote(score_prefix+filename,expected)
            saved_scores=pd.read_csv(io.BytesIO(score_data),float_precision='round_trip')
            id_column='fact_key' if filename=='isolated_scores.csv' else 'example_id'
            s.require(saved_scores[id_column].is_unique,'duplicate historical score identity')
            matched=[(eid,value) for eid,value in zip(saved_scores[id_column],saved_scores.frozen_probe_score) if eid in a_lookup]
            if matched:
                error=max(abs(a_lookup[eid]-float(value)) for eid,value in matched)
                s.require(error<1e-10,'historical row-score replay mismatch')
                checks.append(dict(file=score_prefix+filename,matched_rows=len(matched),maximum_absolute_error=error,verified=True))
        baseline_checks.setdefault(model,{})['row_level_saved_score_checks']=checks
        np.savez_compressed(artifacts/'sensitivity_scores'/(model+'_scores.npz'),**{'historical_'+k:v for k,v in old_scores.items()},
                            **{'corrected_'+k:v for k,v in new_scores.items()})
        for definition in config['endpoints']:
            condition=definition['condition']
            if condition not in old_scores:
                missing.append(dict(model=model,endpoint=condition+'/'+definition['metric'],dependency={k:v for k,v in remote['caches'].items()
                               if k.startswith(model) and v['status']!='verified'}));continue
            if condition=='atomic_D':
                labels=np.array([int(r['label']) for r in atomic]);base_mask=np.ones(len(atomic),bool)
                eligible=atomic_keep;topics=np.array([r['topic'] for r in atomic]);weights=atom_w
            else:
                labels=raw.cell.isin(definition['positive']).to_numpy(int)
                base_mask=((raw.operator==definition['operator']) & raw.cell.isin(definition['positive']+definition['negative'])).to_numpy()
                eligible=compound_keep;topics=raw.topic.to_numpy();weights=comp_w
            found,draws=compare_endpoint(old_scores[condition],new_scores[condition],labels,base_mask,eligible,topics,weights,
                                        definition,model,config['bootstrap'],config['topics'])
            for record in found:
                if record['status'] != 'computed' or record['paired_delta_ci']['ci_status'] != 'ok':
                    missing.append(dict(model=model, endpoint=condition+'/'+definition['metric'],
                                        scope=record['scope'], topic=record['topic'],
                                        dependency=record.get('dependency', 'Insufficient valid paired bootstrap replicates')))
            records.extend(found);all_draws.extend(draws)
    triggers,comparisons=refresh_triggers(records,all_draws,config)
    # Large score and bootstrap artifacts remain outside Git; receipts bind their exact bytes.
    if all_draws:np.savez_compressed(artifacts/'sensitivity_scores/bootstrap_draws.npz',historical=np.array([d[0] for d in all_draws]).T,
                                   corrected=np.array([d[1] for d in all_draws]).T)
    compact=[]
    for r in records:
        item={k:v for k,v in r.items() if k not in ('historical_ci','corrected_ci','paired_delta_ci')}
        item.update({key:value for key,value in r['paired_delta_ci'].items()})
        compact.append(item)
    if compact:(output/'metrics.csv').write_bytes(s.csv_bytes(compact))
    (output/'metrics.json').write_bytes(s.json_bytes(records))
    (output/'fits.json').write_bytes(s.json_bytes(fits))
    (output/'rankings_and_contrasts.json').write_bytes(s.json_bytes(comparisons))
    refresh=dict(triggered=bool(triggers),triggers=triggers,automatic_refresh_launched=False,
                 affected_downstream=['canonical raw LR compound transfer tables/figures and critical boundaries',
                    'Priority-2 raw versus or-both and isolated min/max comparisons',
                    'Qwen/Llama canonical-reference comparisons and conclusions',
                    'method-comparison downstream analyses that reuse the canonical LR baseline or the old development eligibility masks'] if triggers else [])
    (output/'refresh_decisions.json').write_bytes(s.json_bytes(refresh))
    s.verify_preparation(root,prepared)
    package.unchanged()
    receipt=dict(status='complete_sensitivity' if not missing else 'incomplete_sensitivity',completed_utc=datetime.now(timezone.utc).isoformat(),
                 local_platform=platform.system(),adoption_sha256=config['adoption_sha256'],preparation_receipt=s.identity((prepared/'preparation_receipt.json').read_bytes()),
                 staging_receipt=s.identity((artifacts/'staging_receipt.json').read_bytes()),export_receipt=s.identity((artifacts/'export_receipt.json').read_bytes()),
                 historical_score_inputs=package.used,
                 runtime=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,pandas=pd.__version__,blas_threads=1),
                 endpoint_results=len(records),missing_endpoints=missing,baseline_checks=baseline_checks,
                 bootstrap=dict(**config['bootstrap'],compound_schedule_sha256=comp_schedule,atomic_weight_sha256=s.sha(atom_w.tobytes())),
                 fitting_operations=len(fits),production_bank_fits=0,reservations=0,new_model_extractions=0,final_test_predictions=0,
                 artifacts={p.name:s.identity(p.read_bytes()) for p in sorted(output.iterdir())},
                 external_artifacts={str(p):s.identity(p.read_bytes()) for p in sorted((artifacts/'sensitivity_scores').iterdir())},
                 source_sha256={name:s.sha((root/name).read_bytes()) for name in ['src/selection_repair_canonical_sensitivity.py','src/clean_atomic_probes.py','src/clean_transfer_statistics.py']})
    (output/'sensitivity_receipt.json').write_bytes(s.json_bytes(receipt))
    return receipt
