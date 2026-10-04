"""Future independent sorted-pair calculation. Preparation tests use synthetic values only."""
from pathlib import Path
import numpy as np
from src.checkpoint_next_inventory import require, sha
from src.checkpoint_next_full_bank_inputs import read, FullTrainingAdapter
from src.checkpoint_next_full_bank_provenance import FrozenB25
from src.checkpoint_next_full_bank_selection import bank_specs
from src.checkpoint_next_full_bank_worker import load_fit, folder
from src.checkpoint_next_store import publish_json
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_fresh_adapter import check_packet


def independent_auc(labels,scores,weights=None):
    """Pair-mass formula, grouped exact ties; undefined total classes remain NaN."""
    labels=np.asarray(labels,int);scores=np.asarray(scores,float)
    require(labels.shape==scores.shape and np.isfinite(scores).all(),'audit score axes/finite')
    w=np.ones((1,len(labels))) if weights is None else np.atleast_2d(weights).astype(float)
    require(w.shape[1]==len(labels) and np.isfinite(w).all() and (w>=0).all(),'audit weight axes')
    order=np.argsort(scores,kind='stable');s=scores[order];y=labels[order];w=w[:,order]
    starts=np.flatnonzero(np.r_[True,s[1:]!=s[:-1]]) if len(s) else np.array([],int)
    if not len(starts):return np.full(w.shape[0],np.nan)
    pos=np.add.reduceat(w*(y==1),starts,axis=1);neg=np.add.reduceat(w*(y==0),starts,axis=1)
    lower=np.cumsum(neg,axis=1)-neg
    numerator=np.sum(pos*(lower+.5*neg),axis=1);denominator=pos.sum(1)*neg.sum(1)
    return np.divide(numerator,denominator,out=np.full(len(denominator),np.nan),where=denominator>0)


def endpoint(rows,scores,metric,scope,topic,weights=None):
    labels=np.array([int(r['label']) for r in rows]);topics=np.array([r['topic'] for r in rows])
    if metric=='atomic_auroc':mask=np.ones(len(rows),bool)
    else:
        op=np.array([r['operator'] for r in rows]);cell=np.array([int(r['truth_a'])+int(r['truth_b']) for r in rows])
        mask={'AND_auroc':op=='AND','OR_auroc':op=='OR','OR_mixed_vs_FF_auroc':(op=='OR')&(cell<2),
              'AND_TT_vs_mixed_auroc':(op=='AND')&(cell>0)}[metric]
    if scope=='topic_macro':
        # Fixed five topics, propagate each missing endpoint.
        return np.mean([endpoint(rows,scores,metric,'topic',t,weights) for t in
            ('animal_class','cities','element_symb','inventors','sp_en_trans')],axis=0)
    if scope=='topic':mask &= topics==topic
    return independent_auc(labels[mask],np.asarray(scores)[mask],None if weights is None else weights[:,mask])


def audit(args,identity):
    root=Path(args.output);completion=read(root/'completion.json');require(completion['status']=='completed' and completion['identity']==identity,'completed exact producer required')
    adapter=FullTrainingAdapter(args.handoffs);frozen=FrozenB25(adapter)
    for path,p in completion['evaluation']['files'].items():require(pin(root/path)==p,'report pin')
    records=[load_fit(root,s,identity) for s in bank_specs()]
    require(len(records)==600 and all(r and r['valid_for_scoring'] for r in records),'all600 valid candidates')
    maximum=0.;checks=0
    def compare(a,b):
        nonlocal maximum,checks
        a=np.asarray(a,float);b=np.asarray(b,float)
        require(np.array_equal(np.isnan(a),np.isnan(b)),'audit undefined counts')
        err=float(np.max(np.abs(a[np.isfinite(a)]-b[np.isfinite(b)]))) if np.isfinite(a).any() else 0.
        maximum=max(maximum,err);checks+=a.size
        require(err<=1e-12,'independent metric/contrast mismatch')
    for record in records:
        with np.load(folder(root,record['spec']['fit_id'])/'scores.npz',allow_pickle=False) as z:
            for g,packet in record['score_bindings'].items():check_packet(adapter.bindings[g],dict(packet,scores=z[g]))
            compare(independent_auc([int(r['label']) for r in adapter.rows['atomic_D']],z['atomic_D']),[record['atomic']['pooled']])
            for seed in (11,23,37):
                compare(endpoint(adapter.rows[f'B25_{seed}'],z[f'B25_{seed}'],'OR_mixed_vs_FF_auroc','pooled','all'),[record['allocated'][str(seed)]['primary']])
    frozen.verify(frozen.root/'bootstrap_weights.npz')
    with np.load(frozen.root/'bootstrap_weights.npz',allow_pickle=False) as z:weights={g:z[g].copy() for g in z.files}
    for path in sorted((root/'evaluations').glob('*/metrics.json')):
        receipt=read(path);binding=read(path.parent/'score-binding.json');require(binding['lock_sha256']==sha(root/'locked.json'),'audit evaluation lock')
        require(pin(path.parent/'scores.npz')==binding['artifact'] and pin(path.parent/'bootstrap-metrics.npy')==receipt['draws'],'audit evaluation pins')
        draws=np.load(path.parent/'bootstrap-metrics.npy',allow_pickle=False)
        with np.load(path.parent/'scores.npz',allow_pickle=False) as values:
            for i,r in enumerate(receipt['records']):
                g=r['condition'];check_packet(adapter.bindings[g],dict(binding['groups'][g],scores=values[g]))
                point=np.nan if r['point'] is None else r['point']
                compare(endpoint(adapter.rows[g],values[g],r['metric'],r['scope'],r['topic']),[point])
                compare(endpoint(adapter.rows[g],values[g],r['metric'],r['scope'],r['topic'],weights[g]),draws[i])
    metrics=read(root/'metrics.json')['records'];contrasts=read(root/'contrasts.json')['records']
    # Reconstruct per-seed and paired mean uncertainty using original frozen
    # bootstrap metrics plus independently checked new full-bank draw arrays.
    lock=read(root/'locked.json');sample_draws={};full_draws={}
    for model in ('qwen','llama'):
        for seed in (11,23,37):
            for arm in ('R_all','S_all','reduced_LR','bank_oracle'):
                _,_,destination,_=frozen.head(model,seed,arm)
                frozen.verify(destination/'metrics.json');receipt=read(destination/'metrics.json')
                frozen.verify(destination/'bootstrap_metrics.npy')
                require(pin(destination/'bootstrap_metrics.npy')==receipt['draws'],'original metric draw pin')
                sample_draws[model,str(seed),arm]=np.load(destination/'bootstrap_metrics.npy',allow_pickle=False)
            needed=dict(lock['models'][model]['procedures'][str(seed)],
                full_bank_eligible=lock['models'][model]['oracle']['eligible']['fit_id'],
                full_bank_unrestricted=lock['models'][model]['oracle']['unrestricted']['fit_id'])
            from src.checkpoint_r2_selection_v1 import candidate_id
            from src.checkpoint_next_full_bank_inputs import configuration
            fixed=configuration()['models'][model]
            needed['full_canonical_LR']=candidate_id(model,fixed['fixed_layer'],'l2_logistic',fixed['fixed_C']).replace('/P15/','/full_train/')
            for arm,name in needed.items():
                if name=='frozen:reduced_LR':samples=sample_draws[model,str(seed),'reduced_LR']
                else:
                    if name not in full_draws:
                        destination=root/'evaluations'/folder(root,name).name
                        full_draws[name]=np.load(destination/'bootstrap-metrics.npy',allow_pickle=False)
                    samples=full_draws[name]
                sample_draws[model,str(seed),arm]=samples
        arms={r['arm'] for r in metrics if r['model']==model}
        for arm in arms:sample_draws[model,'mean_11_23_37',arm]=np.mean([sample_draws[model,str(s),arm] for s in (11,23,37)],axis=0)
    template=next((read(p)['records'] for p in (root/'evaluations').glob('*/metrics.json')))
    endpoint_index={(r['condition'],r['metric'],r['scope'],r['topic']):i for i,r in enumerate(template)}
    def check_interval(row,draws):
        valid=draws[np.isfinite(draws)]
        require(row['total_replicates']==2000 and row['valid_replicates']==len(valid) and row['invalid_replicates']==2000-len(valid),'independent undefined draw accounting')
        if len(valid)>=1800:
            compare(np.quantile(valid,[.025,.975],method='linear'),[row['ci_low'],row['ci_high']])
        else:require(row['ci_low'] is None and row['ci_high'] is None,'undefined interval must stay undefined')
    for r in metrics:
        i=endpoint_index[r['condition'],r['metric'],r['scope'],r['topic']]
        check_interval(r,sample_draws[r['model'],str(r['seed']),r['arm']][i])
    for r in contrasts:
        i=endpoint_index[r['condition'],r['metric'],r['scope'],r['topic']]
        samples=sample_draws[r['model'],str(r['seed']),r['left']]-sample_draws[r['model'],str(r['seed']),r['right']]
        check_interval(r,samples[i])
    key=lambda r:(r['model'],str(r['seed']),r['arm'],r['condition'],r['metric'],r['scope'],r['topic'])
    lookup={key(r):r for r in metrics}
    for r in contrasts:
        values=[]
        for arm in (r['left'],r['right']):
            v=lookup[(r['model'],str(r['seed']),arm,r['condition'],r['metric'],r['scope'],r['topic'])]['point'];values.append(np.nan if v is None else v)
        compare([values[0]-values[1]],[np.nan if r['point'] is None else r['point']])
    for r in metrics:
        if r['seed']!='mean_11_23_37':continue
        values=[lookup[(r['model'],str(s),r['arm'],r['condition'],r['metric'],r['scope'],r['topic'])]['point'] for s in (11,23,37)]
        compare([np.mean([np.nan if v is None else v for v in values])],[np.nan if r['point'] is None else r['point']])
    receipt=dict(status='independent_calculation_passed',identity=identity,checks=checks,maximum_absolute_difference=maximum,
        all600_candidate_points=True,locked_full_score_points_and_2000_draws=True,strict_undefined_propagation=True,
        limitations='conditional frozen-winner audit; no second-machine replication; no refitting',completion_sha256=sha(root/'completion.json'))
    publish_json(Path(args.audit_output),receipt)
    return receipt
