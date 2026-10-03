"""Six separate constituent endpoints, fixed decisions and external composition."""
import numpy as np
import pandas as pd
from src.checkpoint_r2_selection_v1 import TOPICS, ENDPOINTS, surface_labels
from src.clean_transfer_statistics import AUC, Mean, weighted_auc, interval
from src.checkpoint_r2_fresh_inputs import require
from src.checkpoint_r2_fresh_recoverability import finite


def definitions(rows,scores,operator):
    require(set(r['operator'] for r in rows)=={operator},'single evaluation operator')
    first,second=surface_labels(pd.DataFrame(rows));scores=np.asarray(scores,float)
    require(scores.shape==(len(rows),2) and np.isfinite(scores).all(),'invalid predictions')
    all_rows=np.ones(len(rows),bool);decision=scores>=0  # inherited .5 probability, exact tie True
    defs=[]
    for name,labels,head,mask in zip(ENDPOINTS,[first,second,first,first,second,second],[0,1,0,0,1,1],
            [all_rows,all_rows,second==0,second==1,first==0,first==1]):
        defs.append((name,'auc',labels,scores[:,head],mask))
    for i,(name,label) in enumerate((('first',first),('second',second))):
        correct=(decision[:,i]==label).astype(float)
        defs.extend([(name+'_accuracy','mean',label,correct,all_rows),(name+'_balanced_accuracy','ba',label,correct,all_rows)])
    label=np.array([int(r['label']) for r in rows])
    composition=scores.min(axis=1) if operator=='AND' else scores.max(axis=1)
    boolean=decision.all(axis=1) if operator=='AND' else decision.any(axis=1)
    cell=first+second
    defs.extend([('joint_correctness','mean',label,(decision[:,0]==first)&(decision[:,1]==second),all_rows),
        ('external_composition_auroc','auc',label,composition,all_rows),
        ('external_boundary_auroc','auc',label,composition,cell>0 if operator=='AND' else cell<2),
        ('boolean_composition_accuracy','mean',label,boolean==label,all_rows),
        ('boolean_composition_balanced_accuracy','ba',label,boolean==label,all_rows)])
    return defs


def summaries(rows,scores,operator,weights=None,options=None,orders=True):
    topics=np.array([r['topic'] for r in rows]);ordering=np.array([r['ordering'] for r in rows])
    records=[];samples=[]
    for order in (('all','AB','BA') if orders else ('all',)):
        om=np.ones(len(rows),bool) if order=='all' else ordering==order
        for metric,kind,labels,values,mask in definitions(rows,scores,operator):
            pts=[];ds=[]
            for scope,topic in [('pooled','all'),*[('topic',t) for t in TOPICS]]:
                keep=mask&om&(np.ones(len(rows),bool) if scope=='pooled' else topics==topic)
                idx=np.flatnonzero(keep)
                if kind=='auc':
                    fn=AUC(idx,values[idx],labels[idx])
                elif kind=='mean':fn=Mean(idx,values[idx])
                else:
                    a=Mean(idx[labels[idx]==0],values[idx[labels[idx]==0]])
                    b=Mean(idx[labels[idx]==1],values[idx[labels[idx]==1]])
                    fn=lambda w,a=a,b=b:.5*(a(w)+b(w))
                point=float(fn(np.ones((1,len(rows))))[0])
                rec=dict(metric=metric,scope=scope,topic=topic,surface_order=order,point=finite(point),
                         observations=len(idx),status='defined' if np.isfinite(point) else 'undefined_endpoint')
                if weights is not None and order=='all':
                    d=np.concatenate([fn(weights[j:j+64]) for j in range(0,len(weights),64)])
                    rec.update(interval(d,options));samples.append(d)
                    if scope=='topic':ds.append(d)
                records.append(rec)
                if scope=='topic':pts.append(point)
            rec=dict(metric=metric,scope='topic_macro',topic='all',surface_order=order,point=finite(np.mean(pts)),
                     status='defined' if np.isfinite(pts).all() else 'undefined_endpoint')
            if weights is not None and order=='all':
                d=np.mean(ds,axis=0);rec.update(interval(d,options));samples.append(d)
            records.append(rec)
    return records,None if weights is None else np.asarray(samples)


def absolute_status(record):
    if record['point'] is None:return 'undefined'
    if record['point']<.90:return 'point_below_90'
    if record.get('ci_low') is not None and record['ci_low']>=.90:return 'lower_bound_at_least_90'
    return 'point_at_least_90_bound_uncertain'


def transfer_status(record):
    if record.get('ci_high') is None:return 'unresolved_insufficient_draws'
    if record['ci_high']<.05:return 'upper_bound_below_05'
    if record['ci_low']>.05:return 'material_deficit_lower_bound_above_05'
    return 'unresolved_crosses_05'


def paired_contrast(left,right,left_draws,right_draws,options):
    key=lambda r:(r['metric'],r['scope'],r['topic'],r['surface_order'])
    lhs=[r for r in left if r['surface_order']=='all'];rhs=[r for r in right if r['surface_order']=='all']
    require([key(r) for r in lhs]==[key(r) for r in rhs] and left_draws.shape==right_draws.shape==(len(lhs),options['replicates']),'paired endpoint/draw alignment')
    result=[]
    for i,(l,r) in enumerate(zip(lhs,rhs)):
        point=None if l['point'] is None or r['point'] is None else l['point']-r['point']
        rec=dict(metric=l['metric'],scope=l['scope'],topic=l['topic'],surface_order='all',point=point,
            **interval(left_draws[i]-right_draws[i],options))
        if l['metric'] in ENDPOINTS:rec.update(planning_margin=.05,classification=transfer_status(rec))
        result.append(rec)
    return result
