"""Frozen metric/report adapter, usable on synthetic/D data without E authorization.

No fitting, ranking, sign orientation or thresholds occur here. E caller must pass
final_e_access before opening representations. Scores are never averaged across heads.
"""
import numpy as np
import pandas as pd
from src.final_e_inventory import CONDITIONS
from src.checkpoint_r2_fresh_inputs import hash_value, require
from src.checkpoint_r2_b25_statistics import summaries as scalar_summaries
from src.checkpoint_r2_constituent_metrics import summaries as constituent_summaries, paired_contrast, absolute_status
from src.checkpoint_r2_selection_v1 import average_sample_metrics, ENDPOINTS
from src.selection_repair_canonical_sensitivity import atomic_weights, compound_weights
from src.clean_transfer_statistics import interval


def finite(x):
    return float(x) if np.isfinite(x) else None


def schedules(metadata, options):
    """Separate inherited membership frames; shared within frame, never D indices."""
    atom=metadata.rows['atomic_E'];bare=metadata.rows['bare']
    frame=pd.DataFrame([dict(pair_id=r['pair_id'],topic=r['topic'],entity_a_id=r['person_a'],entity_b_id=r['person_b']) for r in bare])
    cw, schedule=compound_weights(frame,np.ones(len(bare),bool),options)
    aw=atomic_weights(atom,np.ones(len(atom),bool),options)
    weights=dict(atomic_E=aw,bare=cw)
    for c in CONDITIONS[1:]:
        require([r['base_example_id'] for r in metadata.rows[c]]==[r['example_id'] for r in bare], 'shared E support')
        weights[c]=cw  # exact object/draws, never independent wordings
    receipt=dict(schema='final-e-bootstrap-v1',options=options,compound_schedule_sha256=schedule,
        atomic_people=metadata.receipt['atomic_people'],compound_people=metadata.receipt['compound_people'],
        shared_people=metadata.receipt['shared_people'],membership_sha256=metadata.receipt['ordered_groups'],
        arrays_sha256={g:__import__('hashlib').sha256(w.astype('<i2').tobytes()).hexdigest() for g,w in weights.items()},
        statistical_scope='separate inherited atomic and compound frames; shared across models/heads/samples/wordings within frame',
        joint_atomic_compound_inference='pending; no cross-frame joint confidence statement',
        undefined='no redraw/imputation/topic dropping; all-five mean; minimum 1800/2000 finite draws',
        E_scores_used=False)
    return weights,receipt


def scalar_result(rows, values, weights=None, options=None):
    records=[];draws=[]
    require(set(values)==set(rows) and (weights is None or set(weights)==set(rows)), 'exact report coverage')
    for group, rs in rows.items():
        scores=np.asarray(values[group],float)
        require(scores.shape==(len(rs),) and np.isfinite(scores).all(), 'score coverage/nonfinite')
        rec, d=scalar_summaries(rs,scores,atomic=group.startswith('atomic_'),weights=None if weights is None else weights[group],options=options)
        records.extend(dict(condition=group,**r) for r in rec)
        if d is not None:draws.extend(d)
    return dict(records=records,points=np.array([np.nan if r['point'] is None else r['point'] for r in records]),
                draws=None if weights is None else np.asarray(draws))


def endpoint_keys(result):
    return [(r['condition'],r['metric'],r['scope'],r['topic']) for r in result['records']]


def scalar_tables(registry, results, contrast_spec, options):
    """One result per saved head; alias copies keep each of the three samples."""
    procedures={};metrics=[];contrasts=[]
    for a in registry['scalar_aliases']:
        result=results[a['head']];key=(a['panel'],a['model'],a['arm'],a['sample'])
        require(key not in procedures, 'duplicate procedure alias')
        procedures[key]=result
        metrics.extend(dict(panel=a['panel'],model=a['model'],arm=a['arm'],sample=a['sample'],category=a['category'],head=a['head'],**r) for r in result['records'])
    # Full-bank comparisons reference original B25 arms, keeping their original heads.
    for model in ('qwen','llama'):
        for arm in ('R_all','S_all','reduced_LR','bank_oracle'):
            for s in (11,23,37):
                result=procedures['b25',model,arm,s]
                procedures['full_bank',model,arm,s]=result
                metrics.extend(dict(panel='full_bank',model=model,arm=arm,sample=s,category='original_B25_reference',**r) for r in result['records'])
    for panel, model, arm in sorted({k[:3] for k in procedures}):
        rs={s:procedures[panel,model,arm,s] for s in (11,23,37)}
        template=rs[11]
        require(all(endpoint_keys(r)==endpoint_keys(template) for r in rs.values()), 'metric order/alignment')
        result=dict(records=template['records'],points=average_sample_metrics({s:r['points'] for s,r in rs.items()}),
                    draws=average_sample_metrics({s:r['draws'] for s,r in rs.items()}))
        procedures[panel,model,arm,'mean_11_23_37']=result
        metrics.extend(dict(panel=panel,model=model,arm=arm,sample='mean_11_23_37',condition=r['condition'],metric=r['metric'],
            scope=r['scope'],topic=r['topic'],point=finite(result['points'][i]),**interval(result['draws'][i],options)) for i,r in enumerate(template['records']))
    for panel,pairs in contrast_spec.items():
        for model in ('qwen','llama'):
            for left,right in pairs:
                for sample in (11,23,37,'mean_11_23_37'):
                    lhs,rhs=procedures[panel,model,left,sample],procedures[panel,model,right,sample]
                    require(endpoint_keys(lhs)==endpoint_keys(rhs),'paired endpoint identity')
                    points,draws=lhs['points']-rhs['points'],lhs['draws']-rhs['draws']
                    for i,r in enumerate(lhs['records']):
                        rec=dict(panel=panel,model=model,sample=sample,left=left,right=right,condition=r['condition'],
                            metric=r['metric'],scope=r['scope'],topic=r['topic'],point=finite(points[i]),**interval(draws[i],options))
                        if right=='reduced_LR' and r['metric'] in ('atomic_auroc','AND_auroc'):
                            margin=.005 if r['metric']=='atomic_auroc' else .02
                            rec.update(retention_margin=margin,point_retained=None if rec['point'] is None else rec['point']>=-margin,
                                interval_supported=rec['ci_low'] is not None and rec['ci_low']>=-margin)
                        contrasts.append(rec)
    for sample in (11,23,37,'mean_11_23_37'):
        q,qs,l,ls=[procedures['b25',m,a,sample] for m,a in [('qwen','R_all'),('qwen','S_all'),('llama','R_all'),('llama','S_all')]]
        require(all(endpoint_keys(x)==endpoint_keys(q) for x in (qs,l,ls)), 'cross-model paired identities')
        points=(q['points']-qs['points'])-(l['points']-ls['points'])
        draws=(q['draws']-qs['draws'])-(l['draws']-ls['draws'])
        contrasts.extend(dict(panel='b25',model='qwen_minus_llama',sample=sample,left='R_all_minus_S_all_qwen',right='R_all_minus_S_all_llama',
            condition=r['condition'],metric=r['metric'],scope=r['scope'],topic=r['topic'],point=finite(points[i]),**interval(draws[i],options)) for i,r in enumerate(q['records']))
    return metrics,contrasts


def constituent_result(rows, scores, operator, weights, options):
    from src.final_e_inventory import truth
    normalized=[]
    for row in rows:
        a,b=int(row['truth_a']),int(row['truth_b'])
        require(a in (0,1) and b in (0,1), 'constituent binary targets')
        if 'canonical_truth_a' in row:
            require(truth(row['canonical_truth_a'])==a and truth(row['canonical_truth_b'])==b, 'canonical targets changed')
        first,second=(a,b) if row['ordering']=='AB' else (b,a)
        for k,v in (('surface_first_truth',first),('surface_second_truth',second)):
            if k in row:require(truth(row[k])==v, 'surface constituent target changed')
        normalized.append(dict(row,canonical_truth_a=a,canonical_truth_b=b,surface_first_truth=first,surface_second_truth=second))
    records,draws=constituent_summaries(normalized,scores,operator,weights,options)
    for r in records:
        if r['metric'] in ENDPOINTS:r.update(absolute_planning_target=.90,absolute_classification=absolute_status(r))
    return dict(records=records,draws=draws)


def constituent_tables(cases, results, options):
    metrics=[];contrasts=[]
    for case in cases:
        for condition,result in results[hash_value(case)].items():
            metrics.extend(dict(**{k:v for k,v in case.items() if k!='heads'},condition=condition,**r) for r in result['records'])
    for case in [c for c in cases if c['arm']=='source_on_target']:
        other=next(c for c in cases if c['model']==case['model'] and c['source_operator']==case['source_operator'] and c['arm']=='target_on_target')
        for condition,rhs in results[hash_value(case)].items():
            lhs=results[hash_value(other)][condition]
            contrasts.extend(dict(model=case['model'],source_operator=case['source_operator'],target_operator=case['target_operator'],
                saved_layer=case['saved_layer'],condition=condition,contrast='target_trained_minus_source_trained',paired=True,**r)
                for r in paired_contrast(lhs['records'],rhs['records'],lhs['draws'],rhs['draws'],options))
    return metrics,contrasts
