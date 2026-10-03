"""Read-only independent saved-prediction/metric rebuild; never refits a head."""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score,accuracy_score,balanced_accuracy_score
from threadpoolctl import threadpool_limits

from src.checkpoint_r2_constituent_adapter import ConstituentAdapter,FIT,VALID,FULL,BARE,check_production_packet
from src.checkpoint_r2_fresh_constituents import configuration,fit_folder
from src.checkpoint_r2_fresh_recoverability import verify_current_inputs,save
from src.checkpoint_r2_fresh_inputs import ROOT,file_hash,hash_value,require
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_full_inputs import WORDING
from src.checkpoint_r2_selection_v1 import TOPICS,ENDPOINTS


def read(p):return json.loads(Path(p).read_text())
def npz(p):
    with np.load(p,allow_pickle=False) as d:return {k:d[k].copy() for k in d.files}


def labels(rows):
    a=np.array([int(r['truth_a']) for r in rows]);b=np.array([int(r['truth_b']) for r in rows]);ab=np.array([r['ordering']=='AB' for r in rows])
    return np.where(ab,a,b),np.where(ab,b,a)


def manual_definitions(rows,score,operator):
    first,second=labels(rows);all_rows=np.ones(len(rows),bool);decision=score>=0
    defs={}
    for e,y,h,mask in zip(ENDPOINTS,[first,second,first,first,second,second],[0,1,0,0,1,1],
            [all_rows,all_rows,second==0,second==1,first==0,first==1]):defs[e]=('auc',y,score[:,h],mask)
    for i,(name,y) in enumerate([('first',first),('second',second)]):
        defs[name+'_accuracy']=('accuracy',y,decision[:,i],all_rows)
        defs[name+'_balanced_accuracy']=('ba',y,decision[:,i],all_rows)
    y=np.array([int(r['label']) for r in rows]);boolean=decision.all(axis=1) if operator=='AND' else decision.any(axis=1)
    scalar=np.minimum(score[:,0],score[:,1]) if operator=='AND' else np.maximum(score[:,0],score[:,1])
    defs['joint_correctness']=('mean',y,(decision[:,0]==first)&(decision[:,1]==second),all_rows)
    defs['external_composition_auroc']=('auc',y,scalar,all_rows)
    defs['external_boundary_auroc']=('auc',y,scalar,(first+second)>0 if operator=='AND' else (first+second)<2)
    defs['boolean_composition_accuracy']=('accuracy',y,boolean,all_rows)
    defs['boolean_composition_balanced_accuracy']=('ba',y,boolean,all_rows)
    return defs


def point_values(rows,scores,op,orders=('all','AB','BA')):
    result={};topic=np.array([r['topic'] for r in rows]);order=np.array([r['ordering'] for r in rows])
    for o in orders:
        om=np.ones(len(rows),bool) if o=='all' else order==o
        for metric,(kind,y,value,mask) in manual_definitions(rows,scores,op).items():
            points=[]
            for scope,t in [('pooled','all')]+[('topic',t) for t in TOPICS]:
                idx=mask&om&(np.ones(len(rows),bool) if scope=='pooled' else topic==t)
                if not idx.any() or (kind in ('auc','ba') and len(np.unique(y[idx]))<2):v=np.nan
                elif kind=='auc':v=roc_auc_score(y[idx],value[idx])
                elif kind=='accuracy':v=accuracy_score(y[idx],value[idx])
                elif kind=='ba':v=balanced_accuracy_score(y[idx],value[idx])
                else:v=value[idx].mean()
                result[metric,scope,t,o]=float(v)
                if scope=='topic':points.append(v)
            result[metric,'topic_macro','all',o]=float(np.mean(points))
    return result


def independent_auc_draws(y,score,weights):
    """Independent stable rank-group calculation; no production AUC/auc_draws call."""
    sorting=np.argsort(score,kind='stable');sorted_score=score[sorting];truth=y[sorting]
    boundaries=np.r_[0,np.flatnonzero(sorted_score[1:]!=sorted_score[:-1])+1] if len(score) else np.array([],int)
    if not len(score):return np.full(len(weights),np.nan)
    out=[]
    for start in range(0,len(weights),64):
        w=weights[start:start+64,sorting].astype(float)
        p=np.add.reduceat(w*truth,boundaries,axis=1);n=np.add.reduceat(w*(1-truth),boundaries,axis=1)
        numerator=(p*(np.cumsum(n,axis=1)-n+n/2)).sum(axis=1);den=p.sum(axis=1)*n.sum(axis=1)
        out.append(np.divide(numerator,den,out=np.full(len(w),np.nan),where=den!=0))
    return np.concatenate(out)


def independent_endpoint_draws(rows,score,op,weights):
    result={};topics=np.array([r['topic'] for r in rows])
    for metric,(kind,y,v,mask) in manual_definitions(rows,score,op).items():
        if metric not in ENDPOINTS:continue
        ds=[]
        for scope,t in [('pooled','all')]+[('topic',t) for t in TOPICS]:
            keep=mask&(np.ones(len(rows),bool) if scope=='pooled' else topics==t)
            draw=independent_auc_draws(y[keep],v[keep],weights[:,keep]);result[metric,scope,t,'all']=draw
            if scope=='topic':ds.append(draw)
        result[metric,'topic_macro','all','all']=np.mean(ds,axis=0)
    return result


def audit(run,expected):
    started=time.monotonic();cfg=configuration();completion=read(run/'completion.json')
    require(completion['producer_sha']==expected and completion['config_sha256']==file_hash(ROOT/'config/checkpoint_r2/fresh_constituents_v1.json'),'fitting producer/config')
    for rel,pinned in completion['artifacts'].items():require(pin(run/rel)==pinned,'derived corruption: '+rel)
    acceptance=read(ROOT/'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json');verify_current_inputs(acceptance,cfg)
    adapter=ConstituentAdapter(acceptance);fit_records=[read(p) for p in sorted((run/'fits').glob('*/fit.json'))]
    require(len(fit_records)==completion['valid_fits'] and all(r['valid'] for r in fit_records),'complete valid head inventory')
    scores=[read(p) for p in sorted((run/'scores').glob('*/binding.json'))]
    by_layer={}
    for rec in fit_records:by_layer.setdefault((rec['model'],rec['saved_layer']),dict(fits=[],scores=[]))['fits'].append(rec)
    for rec in scores:by_layer[rec['packet']['model'],rec['packet']['saved_layer']]['scores'].append(rec)
    reconstructed=0;objective_checks=0;point_checks=0;endpoint_draw_checks=0;max_score_delta=0.;rebuilt_scores={};selection_points={}
    for (model,layer),items in sorted(by_layer.items()):
        groups={'P15'}|{r['group'] for r in items['fits']}|{r['packet']['group'] for r in items['scores']}
        features=adapter.layer_groups(model,layer,sorted(groups));pre=npz(run/f'preprocessing/{model}/L{layer:02d}.npz')
        active=~pre['constant_mask'];standardized={}
        for g,X in features.items():
            z=np.zeros_like(X);z[:,active]=(X[:,active]-pre['mean'][active])/pre['denominator'][active];standardized[g]=z
        params={}
        for rec in items['fits']:
            params[rec['id']]=npz(fit_folder(run,rec['id'])/'head.npz')
            idx=adapter.indices_for(rec['group'],rec['operator']);rows=adapter.block(rec['group'],rec['operator'])['rows']
            y=labels(rows)[0 if rec['position']=='FIRST' else 1];X=features[rec['group']][idx]
            require(rec['training_bindings_sha256']==hash_value(adapter.block(rec['group'],rec['operator'])['bindings']),'training observations')
            require(hashlib.sha256(X.tobytes()).hexdigest()==rec['feature_float64_sha256'],'training feature exposure')
            require(hashlib.sha256(y.tobytes()).hexdigest()==rec['labels_sha256'],'training surface labels')
            Z=standardized[rec['group']][idx];p=params[rec['id']];w=p['weights'];b=float(p['intercept']);logit=Z@w+b
            objective=np.logaddexp(0,np.where(y==1,-logit,logit)).mean()+.001*np.dot(w,w)
            residual=np.where(y==1,-expit(-logit),expit(logit));grad=np.r_[Z.T@residual/len(y)+.002*w,residual.mean()]
            require(np.max(np.abs(grad))<=1e-4 and np.isfinite(objective),'independent objective/gradient gate')
            np.testing.assert_allclose(objective,rec['attempts'][-1]['objective'],atol=1e-12,rtol=0)
            objective_checks+=1
        for rec in items['scores']:
            packet=rec['packet'];group=packet['group'];op=packet['operator'];idx=adapter.indices_for(group,op);rows=adapter.block(group,op)['rows'];Z=standardized[group][idx]
            pred=np.column_stack([Z@params[name]['weights']+float(params[name]['intercept']) for name in rec['head_ids']])
            folder=run/'scores'/hash_value(rec['id'])[:24];saved=np.load(folder/'scores.npy',allow_pickle=False);check_production_packet(adapter,packet,saved)
            delta=float(np.max(np.abs(pred-saved)));max_score_delta=max(max_score_delta,delta)
            require(np.array_equal(pred,saved),'independent prediction mismatch')
            reconstructed+=2;points=point_values(rows,pred,op,orders=('all','AB','BA') if (folder/'metrics.json').exists() else ('all',))
            rebuilt_scores[str(folder.relative_to(run))]=dict(pred=pred,rows=rows,packet=packet,points=points)
            if group==VALID:
                source=fit_records[[r['id'] for r in fit_records].index(rec['head_ids'][0])]['operator']
                macros={e:points[e,'topic_macro','all','all'] for e in ENDPOINTS}
                selection_points.setdefault((model,source),{})[layer]=dict(minimum=min(macros.values()),mean=float(np.mean(list(macros.values()))),macro=macros)
            if (folder/'metrics.json').exists():
                metric_meta=read(folder/'metrics.json');draws=np.load(folder/'bootstrap-metrics.npy',allow_pickle=False)
                for r in metric_meta['records']:
                    value=points[r['metric'],r['scope'],r['topic'],r['surface_order']]
                    require((r['point'] is None and not np.isfinite(value)) or abs(value-r['point'])<=1e-12,'independent principal metric mismatch');point_checks+=1
        del features,standardized
    for r in pd.read_csv(run/'constituent-layer-curves.csv').to_dict('records'):
        value=rebuilt_scores[r['score_artifact']]['points'][r['metric'],r['scope'],r['topic'],r['surface_order']]
        require((pd.isna(r['point']) and not np.isfinite(value)) or abs(value-r['point'])<=1e-12,'all-layer descriptive metric mismatch');point_checks+=1
    locked=read(run/'locked-selection.json')
    for (model,op),curves in selection_points.items():
        best=max(c['minimum'] for c in curves.values());tied=[l for l,c in curves.items() if best-c['minimum']<=1e-12]
        mean=max(curves[l]['mean'] for l in tied);winner=min(l for l in tied if mean-curves[l]['mean']<=1e-12)
        selection=locked['selections'][model][op];require(winner==selection['selected_layer'],'independent source-only layer selection')
        for rec in selection['curves']:
            for e in ENDPOINTS:np.testing.assert_allclose(rec['macro'][e],curves[rec['layer']]['macro'][e],atol=1e-12,rtol=0)
    # Independently rebuild the reviewed PCG64 multinomial endpoint-product schedule.
    bare=adapter.rows[BARE];rng=np.random.Generator(np.random.PCG64(1729));weights=np.zeros((2000,len(bare)),dtype=np.int16)
    for topic in TOPICS:
        people=sorted({r[k] for r in bare if r['topic']==topic for k in ('person_a','person_b')});lookup={p:i for i,p in enumerate(people)}
        counts=rng.multinomial(len(people),np.full(len(people),1/len(people)),size=2000)
        idx=[i for i,r in enumerate(bare) if r['topic']==topic]
        weights[:,idx]=counts[:,[lookup[bare[i]['person_a']] for i in idx]]*counts[:,[lookup[bare[i]['person_b']] for i in idx]]
    require(np.array_equal(weights,np.load(run/'bootstrap-weights.npy',allow_pickle=False)),'independent shared schedule mismatch')
    bare_indices={r['example_id']:i for i,r in enumerate(bare)};rebuilt_draws={}
    for folder,v in rebuilt_scores.items():
        path=run/folder
        if not (path/'metrics.json').exists():continue
        wb=weights[:,[bare_indices[r['example_id']] for r in v['rows']]]
        draws=independent_endpoint_draws(v['rows'],v['pred'],v['packet']['operator'],wb);rebuilt_draws[folder]=draws
        saved=np.load(path/'bootstrap-metrics.npy',allow_pickle=False);records=[r for r in read(path/'metrics.json')['records'] if r['surface_order']=='all']
        for i,r in enumerate(records):
            key=(r['metric'],r['scope'],r['topic'],'all')
            if r['metric'] in ENDPOINTS:
                np.testing.assert_allclose(draws[key],saved[i],atol=1e-12,rtol=0,equal_nan=True);endpoint_draw_checks+=2000
    contrasts=read(run/'paired-contrasts.json')['records'];contrast_checks=0
    for r in contrasts:
        if r['metric'] not in ENDPOINTS:continue
        key=(r['metric'],r['scope'],r['topic'],'all');lhs=r['lhs_score_folder'];rhs=r['rhs_score_folder']
        require(rebuilt_scores[lhs]['packet']==rebuilt_scores[rhs]['packet'],'paired target row/representation mismatch')
        np.testing.assert_allclose(r['point'],rebuilt_scores[lhs]['points'][key]-rebuilt_scores[rhs]['points'][key],atol=1e-12,rtol=0)
        d=rebuilt_draws[lhs][key]-rebuilt_draws[rhs][key];finite=d[np.isfinite(d)];require(len(finite)==r['valid_replicates'],'paired valid-draw mismatch')
        if len(finite)>=1800:
            lo,hi=np.quantile(finite,[.025,.975],method='linear');np.testing.assert_allclose([lo,hi],[r['ci_low'],r['ci_high']],atol=1e-12,rtol=0)
        else:require(r['ci_low'] is None and r['ci_high'] is None,'undefined interval rules')
        contrast_checks+=1
    adapter.unchanged()
    return dict(status='independent_constituent_saved_prediction_and_metric_audit_passed',producer_sha=expected,
        predictions_reconstructed=reconstructed,maximum_score_absolute_difference=max_score_delta,objective_gradient_checks=objective_checks,
        principal_point_checks=point_checks,endpoint_bootstrap_values_checked=endpoint_draw_checks,paired_endpoint_contrasts_checked=contrast_checks,
        selectors_independently_reconstructed=4,shared_bootstrap_schedule_reconstructed=True,refitting_performed=False,
        behavior_chat_E_accessed=False,seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--expected-producer',required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    with threadpool_limits(limits=1):result=audit(a.run,a.expected_producer)
    save(a.receipt,result);print(json.dumps(result,sort_keys=True))


if __name__=='__main__':main()
