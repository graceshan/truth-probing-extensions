"""Targeted independent checks for the locked E report, also usable on saved D.

Reuse the reviewed full-bank sorted-pair AUROC reference; do not call the E
report evaluator, production summaries, interval or constituent definitions.
"""
from pathlib import Path
import numpy as np
from src.checkpoint_next_full_bank_audit import endpoint, independent_auc
from src.checkpoint_r2_selection_v1 import TOPICS
from src.checkpoint_r2_fresh_inputs import require, hash_value
from src.checkpoint_r2_fresh_store import pin
from src.final_e_inventory import read


class Checks:
    def __init__(self):self.count=0;self.maximum=0.
    def compare(self,a,b):
        a=np.asarray(a,float);b=np.asarray(b,float)
        require(a.shape==b.shape and np.array_equal(np.isnan(a),np.isnan(b)), 'independent undefined/shape mismatch')
        require(np.array_equal(np.isfinite(a),np.isfinite(b)), 'independent nonfinite mismatch')
        d=float(np.max(np.abs(a[np.isfinite(a)]-b[np.isfinite(b)]))) if np.isfinite(a).any() else 0.
        self.maximum=max(self.maximum,d);self.count+=a.size
        require(d<=2e-12,'independent numerical mismatch')
    def interval(self,row,draws,options):
        finite=np.asarray(draws)[np.isfinite(draws)]
        require(row['valid_replicates']==len(finite) and row['invalid_replicates']==options['replicates']-len(finite),'independent undefined counts')
        if len(finite)>=options['minimum_valid']:
            self.compare(np.quantile(finite,[.025,.975],method='linear'),[row['ci_low'],row['ci_high']])
        else:require(row['ci_low'] is None and row['ci_high'] is None,'independent insufficient draws')
    def retention(self,row):
        if 'retention_margin' not in row:return
        margin=.005 if row['metric']=='atomic_auroc' else .02
        require(row['retention_margin']==margin and row['point_retained']==(None if row['point'] is None else row['point']>=-margin)
            and row['interval_supported']==(row['ci_low'] is not None and row['ci_low']>=-margin),'independent retention decision')
        self.count+=3
    def receipt(self):return dict(status='independent_principal_calculation_passed',checks=self.count,maximum_absolute_difference=self.maximum,
        scope='principal scalar points/draws, sample means, paired contrasts/retention and constituent decisions; reproducibility separately covers the complete report')


def constituent_endpoint(rows,scores,metric,scope,topic,order='all',weights=None):
    """Explicit truth/order, confusion/Boolean and min/max formulas."""
    if scope=='topic_macro':
        return np.mean([constituent_endpoint(rows,scores,metric,'topic',t,order,weights) for t in TOPICS],axis=0)
    a=np.array([int(r['truth_a']) for r in rows]);b=np.array([int(r['truth_b']) for r in rows])
    ab=np.array([r['ordering']=='AB' for r in rows]);first=np.where(ab,a,b);second=np.where(ab,b,a)
    scores=np.asarray(scores,float);mask=np.ones(len(rows),bool)
    if scope=='topic':mask &= np.array([r['topic']==topic for r in rows])
    if order!='all':mask &= np.array([r['ordering']==order for r in rows])
    w=np.ones((1,len(rows))) if weights is None else np.asarray(weights,float)
    labels=np.array([int(r['label']) for r in rows]);dec=scores>=0
    operator=rows[0]['operator'];joint=(dec[:,0]==first)&(dec[:,1]==second)
    composed=dec.all(1) if operator=='AND' else dec.any(1)
    composition=scores.min(1) if operator=='AND' else scores.max(1)
    conditionals={'first_given_second_0':(first,scores[:,0],second==0),'first_given_second_1':(first,scores[:,0],second==1),
        'second_given_first_0':(second,scores[:,1],first==0),'second_given_first_1':(second,scores[:,1],first==1)}
    if metric in ('first_marginal','second_marginal'):
        y,s=(first,scores[:,0]) if metric=='first_marginal' else (second,scores[:,1])
    elif metric in conditionals:
        y,s,c=conditionals[metric];mask &= c
    elif metric in ('external_composition_auroc','external_boundary_auroc'):
        y,s=labels,composition
        if metric=='external_boundary_auroc':mask &= (first+second>0 if operator=='AND' else first+second<2)
    else:
        if metric.startswith('first_'):y=first;correct=dec[:,0]==y
        elif metric.startswith('second_'):y=second;correct=dec[:,1]==y
        elif metric=='joint_correctness':y=labels;correct=joint
        else:
            require(metric in ('boolean_composition_accuracy','boolean_composition_balanced_accuracy'),'declared constituent reference metric')
            y=labels;correct=composed==y
        def mean(m):
            total=w[:,m].sum(1);num=(w[:,m]*correct[m]).sum(1)
            return np.divide(num,total,out=np.full(len(total),np.nan),where=total>0)
        return .5*(mean(mask&(y==0))+mean(mask&(y==1))) if metric.endswith('balanced_accuracy') else mean(mask)
    return independent_auc(y[mask],s[mask],w[:,mask])


def principal(row):
    if row['scope'] not in ('pooled','topic_macro'):return False
    return (row['condition'].startswith('atomic') and row['metric']=='atomic_auroc') or (
        row['condition'] in ('bare','D_bare') and row['metric'] in ('AND_auroc','OR_mixed_vs_FF_auroc')) or (
        row['condition'] in ('or_at_least_one_v1','or_explicit_or_both_v1') and row['metric']=='OR_mixed_vs_FF_auroc')


def verify_report(original, metadata, registry, weights, options):
    """Called after full reproducibility; checks original saved scores separately."""
    root=Path(original);checks=Checks();cache={}
    metrics=read(root/'scalar-metrics.json')['records'];contrasts=read(root/'scalar-contrasts.json')['records']
    arms={'R_all','S_all','S_atom_all','reduced_LR','S_full','S_atom_full'}
    aliases={(a['panel'],a['model'],a['arm'],str(a['sample'])):a['head'] for a in registry['scalar_aliases']}
    for m in ('qwen','llama'):
        for a in ('R_all','S_all','reduced_LR','bank_oracle'):
            for seed in (11,23,37):aliases['full_bank',m,a,str(seed)]=aliases['b25',m,a,str(seed)]
    def values(panel,model,arm,sample,row):
        seeds=(11,23,37) if sample=='mean_11_23_37' else (int(sample),)
        pts=[];ds=[]
        for seed in seeds:
            head=aliases[panel,model,arm,str(seed)];key=(head,row['condition'],row['metric'],row['scope'],row['topic'])
            if key not in cache:
                g=row['condition'];folder=root/'scores'/hash_value(head)[:24]
                binding=read(folder/'binding.json')
                require(binding['head']==head and binding['head_sha256']==hash_value(registry['heads'][head]) and binding['artifact']==pin(folder/'scores.npz'),'independent score binding')
                with np.load(folder/'scores.npz',allow_pickle=False) as z:s=z[g]
                cache[key]=(endpoint(metadata.rows[g],s,row['metric'],row['scope'],row['topic'])[0],
                    endpoint(metadata.rows[g],s,row['metric'],row['scope'],row['topic'],weights[g]))
                mb=read(folder/'metric-binding.json')
                j=next(i for i,r in enumerate(mb['records']) if (r['condition'],r['metric'],r['scope'],r['topic'])==key[1:])
                with (folder/'metric-draws.npy').open('rb') as f:draws=np.load(f,allow_pickle=False)
                checks.compare(cache[key][1],draws[j])
            p,d=cache[key];pts.append(p);ds.append(d)
        return np.mean(pts),np.mean(ds,axis=0)
    for r in metrics:
        if r['arm'] not in arms or not principal(r):continue
        p,d=values(r['panel'],r['model'],r['arm'],str(r['sample']),r)
        checks.compare([p],[np.nan if r['point'] is None else r['point']]);checks.interval(r,d,options)
    for r in contrasts:
        if not principal(r):continue
        if r['model']=='qwen_minus_llama':
            qp,qd=values('b25','qwen','R_all',str(r['sample']),r);sp,sd=values('b25','qwen','S_all',str(r['sample']),r)
            lp,ld=values('b25','llama','R_all',str(r['sample']),r);rp,rd=values('b25','llama','S_all',str(r['sample']),r)
            lp,ld,rp,rd=qp-sp,qd-sd,lp-rp,ld-rd
        else:
            if r['left'] not in arms or r['right'] not in arms:continue
            lp,ld=values(r['panel'],r['model'],r['left'],str(r['sample']),r);rp,rd=values(r['panel'],r['model'],r['right'],str(r['sample']),r)
        checks.compare([lp-rp],[np.nan if r['point'] is None else r['point']]);checks.interval(r,ld-rd,options);checks.retention(r)
    for r in read(root/'wording-paired.json')['records']:
        if r['scope'] not in ('pooled','topic_macro') or r['metric']!='OR_mixed_vs_FF_auroc':continue
        head=r['head'];folder=root/'scores'/hash_value(head)[:24]
        with np.load(folder/'scores.npz',allow_pickle=False) as z:
            word=z[r['condition']];bare=z['bare']
        p=endpoint(metadata.rows[r['condition']],word,r['metric'],r['scope'],r['topic'])[0]-endpoint(metadata.rows['bare'],bare,r['metric'],r['scope'],r['topic'])[0]
        d=endpoint(metadata.rows[r['condition']],word,r['metric'],r['scope'],r['topic'],weights[r['condition']])-endpoint(metadata.rows['bare'],bare,r['metric'],r['scope'],r['topic'],weights['bare'])
        checks.compare([p],[np.nan if r['point'] is None else r['point']]);checks.interval(r,d,options)
    # Decisions and constituent ranking on identical support; no production formulas.
    cmetrics=read(root/'constituent-metrics.json')['records'];ccache={}
    for case in registry['constituent_cases']:
        selected=[r for r in cmetrics if all(r[k]==case[k] for k in ('model','source_operator','arm','evaluation_operator','saved_layer'))]
        for r in selected:
            if r['scope'] not in ('pooled','topic_macro'):continue
            group=r['condition'];indices=[i for i,x in enumerate(metadata.rows[group]) if x['operator']==case['evaluation_operator']]
            rows=[metadata.rows[group][i] for i in indices];key=(tuple(case['heads']),case['evaluation_operator'],group,r['metric'],r['scope'],r['surface_order'])
            if key not in ccache:
                cols=[]
                for h in case['heads']:
                    with np.load(root/'scores'/hash_value(h)[:24]/'scores.npz',allow_pickle=False) as z:cols.append(z[group][indices])
                scores=np.column_stack(cols)
                p=constituent_endpoint(rows,scores,r['metric'],r['scope'],r['topic'],r['surface_order'])[0]
                d=constituent_endpoint(rows,scores,r['metric'],r['scope'],r['topic'],weights=weights[group][:,indices]) if r['surface_order']=='all' else None
                ccache[key]=(p,d)
            p,d=ccache[key];checks.compare([p],[np.nan if r['point'] is None else r['point']])
            if d is not None:checks.interval(r,d,options)
            if 'absolute_classification' in r:
                status='undefined' if r['point'] is None else 'point_below_90' if r['point']<.90 else 'lower_bound_at_least_90' if r.get('ci_low') is not None and r['ci_low']>=.90 else 'point_at_least_90_bound_uncertain'
                require(r['absolute_planning_target']==.90 and r['absolute_classification']==status,'independent constituent absolute decision');checks.count+=1
    for r in read(root/'constituent-contrasts.json')['records']:
        if r['scope'] not in ('pooled','topic_macro'):continue
        source=next(c for c in registry['constituent_cases'] if c['model']==r['model'] and c['source_operator']==r['source_operator'] and c['arm']=='source_on_target')
        target=next(c for c in registry['constituent_cases'] if c['model']==r['model'] and c['source_operator']==r['source_operator'] and c['arm']=='target_on_target')
        key=lambda c:(tuple(c['heads']),c['evaluation_operator'],r['condition'],r['metric'],r['scope'],'all')
        lp,ld=ccache[key(target)];rp,rd=ccache[key(source)]
        checks.compare([lp-rp],[np.nan if r['point'] is None else r['point']]);checks.interval(r,ld-rd,options)
        if 'planning_margin' in r:
            require(r['planning_margin']==.05,'constituent transfer margin')
            status='unresolved_insufficient_draws' if r['ci_high'] is None else 'upper_bound_below_05' if r['ci_high']<.05 else 'material_deficit_lower_bound_above_05' if r['ci_low']>.05 else 'unresolved_crosses_05'
            require(r['classification']==status,'independent transfer decision');checks.count+=1
    return checks.receipt()
