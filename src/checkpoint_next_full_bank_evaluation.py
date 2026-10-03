"""Future locked sensitivity report, shared original draws and metric averaging only."""
from pathlib import Path
import numpy as np
import pandas as pd
from src.checkpoint_next_inventory import require, sha
from src.checkpoint_next_full_bank_inputs import read
from src.checkpoint_next_full_bank_selection import CONTRASTS, average_contrast
from src.checkpoint_next_store import same_or_publish
from src.checkpoint_r2_fresh_store import pin, publish
from src.checkpoint_r2_b25_statistics import summaries
from src.checkpoint_r2_b25_store import npz
from src.checkpoint_r2_selection_v1 import average_sample_metrics
from src.checkpoint_r2_fresh_recoverability import finite
from src.clean_transfer_statistics import interval
from src.selection_repair_capacity_methods_v1 import score


def summarize(adapter,values,weights,options):
    template=[];draws=[]
    for g in ['atomic_D','D_bare']+sorted(adapter.receipt['heldout_wording_rows']):
        records,samples=summaries(adapter.rows[g],values[g],atomic=g=='atomic_D',weights=weights[g],options=options)
        template.extend(dict(condition=g,**r) for r in records);draws.extend(samples)
    return dict(records=template,points=np.array([np.nan if r['point'] is None else r['point'] for r in template]),draws=np.asarray(draws))


def write_csv(path,rows):
    text=pd.DataFrame(rows).to_csv(index=False)
    if path.exists():require(path.read_text()==text,'resumed compact table changed')
    else:
        with path.open('x') as f:f.write(text)


def evaluate(root,adapter,frozen,bank,lock,identity,cfg):
    from src.checkpoint_next_full_bank_worker import folder
    root=Path(root);require(read(root/'locked.json')==lock and lock['wording_used_before_lock'] is False,'selectors locked before wording')
    options=cfg['bootstrap'];groups=['atomic_D','D_bare']+sorted(adapter.receipt['heldout_wording_rows'])
    binding=frozen.json('bootstrap_binding.json');frozen.verify(frozen.root/'bootstrap_weights.npz')
    require(pin(frozen.root/'bootstrap_weights.npz')==binding['weights'] and binding['replicates']==2000,'original paired schedule pin/count')
    with np.load(frozen.root/'bootstrap_weights.npz',allow_pickle=False) as z:weights={g:z[g].copy() for g in groups}
    require(all(w.shape==(2000,len(adapter.rows[g])) for g,w in weights.items()),'draw/support axes')
    bare={r['example_id']:i for i,r in enumerate(adapter.rows['D_bare'])}
    for g in groups[2:]:require(np.array_equal(weights[g],weights['D_bare'][:,[bare[r['example_id']] for r in adapter.rows[g]]]),'wording shares existing draw identities')
    same_or_publish(root/'bootstrap-provenance.json',dict(original_binding=binding,original_weights_pin=pin(frozen.root/'bootstrap_weights.npz'),
        conditional='frozen fits/selectors/support; 95% pointwise percentile',metric_mean_within_draw=True,score_pooling=False,redrawing=False))
    cache={};procedures={};by_id={r['spec']['fit_id']:r for r in bank}
    for model in ('qwen','llama'):
        for seed in (11,23,37):
            arms=lock['models'][model]['procedures'][str(seed)]
            for arm in ('R_all','S_all','reduced_LR','bank_oracle'):
                key=(model,seed,arm);cache[key]=summarize(adapter,frozen.scores(model,seed,arm),weights,options)
                procedures[model,seed,arm]=cache[key]
            needed=dict(arms,full_bank_eligible=lock['models'][model]['oracle']['eligible']['fit_id'],
                full_bank_unrestricted=lock['models'][model]['oracle']['unrestricted']['fit_id'])
            fixed=cfg['models'][model]
            from src.checkpoint_r2_selection_v1 import candidate_id
            needed['full_canonical_LR']=candidate_id(model,fixed['fixed_layer'],'l2_logistic',fixed['fixed_C']).replace('/P15/','/full_train/')
            for arm,name in needed.items():
                if name=='frozen:reduced_LR':result=procedures[model,seed,'reduced_LR']
                else:
                    if name not in cache:
                        record=by_id[name];spec=record['spec'];fitfolder=folder(root,name)
                        require(pin(fitfolder/'parameters.npz')==record['parameters'] and pin(fitfolder/'scores.npz')==record['scores'],'locked fit pin')
                        features=adapter.layer(model,spec['saved_layer'],wording=True);params=npz(fitfolder/'parameters.npz')
                        values={g:score(spec['method'],params,features[g]) for g in groups}
                        previous=npz(fitfolder/'scores.npz')
                        require(all(np.array_equal(previous[g],values[g]) for g in ('atomic_D','D_bare')),'locked bare score reconstruction')
                        destination=root/'evaluations'/fitfolder.name;destination.mkdir(parents=True,exist_ok=True)
                        if (destination/'scores.npz').exists():
                            with np.load(destination/'scores.npz',allow_pickle=False) as saved:require(all(np.array_equal(saved[g],v) for g,v in values.items()),'resume locked scores changed')
                        else:publish(destination/'scores.npz',lambda f:np.savez(f,**values))
                        from src.checkpoint_next_full_bank_worker import score_packet
                        same_or_publish(destination/'score-binding.json',dict(identity=identity,fit_id=name,lock_sha256=sha(root/'locked.json'),
                            artifact=pin(destination/'scores.npz'),groups={g:score_packet(adapter,g,v) for g,v in values.items()}))
                        result=summarize(adapter,values,weights,options);cache[name]=result
                        if (destination/'bootstrap-metrics.npy').exists():
                            saved=np.load(destination/'bootstrap-metrics.npy',allow_pickle=False)
                            require(np.array_equal(saved,result['draws'],equal_nan=True),'resume metric draws changed')
                        else:publish(destination/'bootstrap-metrics.npy',lambda f:np.save(f,result['draws'],allow_pickle=False))
                        same_or_publish(destination/'metrics.json',dict(records=result['records'],draws=pin(destination/'bootstrap-metrics.npy')))
                        del features
                    result=cache[name]
                procedures[model,seed,arm]=result
    template=next(iter(procedures.values()))['records'];keys=lambda rs:[(r['condition'],r['metric'],r['scope'],r['topic']) for r in rs]
    require(all(keys(r['records'])==keys(template) for r in procedures.values()),'metric identities aligned')
    rows=[];contrasts=[]
    for model in ('qwen','llama'):
        arms=sorted({arm for m,s,arm in procedures if m==model})
        for arm in arms:
            ps={s:procedures[model,s,arm] for s in (11,23,37)}
            for s in (11,23,37):
                rows.extend(dict(model=model,seed=s,arm=arm,**r) for r in ps[s]['records'])
            mean=dict(points=average_sample_metrics({s:ps[s]['points'] for s in ps}),draws=average_sample_metrics({s:ps[s]['draws'] for s in ps}))
            procedures[model,'mean_11_23_37',arm]=mean
            rows.extend(dict(model=model,seed='mean_11_23_37',arm=arm,condition=r['condition'],metric=r['metric'],scope=r['scope'],topic=r['topic'],
                point=finite(mean['points'][i]),**interval(mean['draws'][i],options)) for i,r in enumerate(template))
        pairs=list(CONTRASTS)+[(arm,'reduced_LR') for arm in ('S_full','S_atom_full','R_all','full_canonical_LR','full_bank_eligible','full_bank_unrestricted')]
        for left,right in pairs:
            for seed in (11,23,37,'mean_11_23_37'):
                lhs,rhs=procedures[model,seed,left],procedures[model,seed,right]
                delta=lhs['draws']-rhs['draws'];points=lhs['points']-rhs['points']
                if seed=='mean_11_23_37':
                    direct=average_contrast({s:procedures[model,s,left]['draws'] for s in (11,23,37)},
                        {s:procedures[model,s,right]['draws'] for s in (11,23,37)})
                    require(np.allclose(direct,delta,rtol=0,atol=1e-15,equal_nan=True),'within-draw paired metric averaging')
                for i,r in enumerate(template):
                    c=dict(model=model,seed=seed,left=left,right=right,condition=r['condition'],metric=r['metric'],scope=r['scope'],topic=r['topic'],
                           point=finite(points[i]),**interval(delta[i],options))
                    if right=='reduced_LR' and r['metric'] in ('atomic_auroc','AND_auroc'):
                        margin=.005 if r['metric']=='atomic_auroc' else .02
                        c.update(retention_margin=margin,point_retained=bool(points[i]>=-margin),interval_supported=c['ci_low'] is not None and c['ci_low']>=-margin)
                    contrasts.append(c)
    # All original and new candidate points stay visible; endpoint-specific maxima
    # are descriptive. Primary fixed winners above have conditional intervals.
    original=frozen.json('bank.json');original=original['records'] if isinstance(original,dict) else original
    maxima=[]
    for model in ('qwen','llama'):
        ref=lock['models'][model]['reference_atomic'];allowed={r['candidate_id'] for r in lock['models'][model]['eligibility'] if r['eligible']}
        for cohort,records in [('full_train',[r for r in bank if r['spec']['model']==model]),('P15',[r for r in original if r['model']==model])]:
            for gate in ('eligible','unrestricted'):
                accepted=[]
                for r in records:
                    if not r['valid_for_scoring']:continue
                    if cohort=='full_train':
                        okay=r['spec']['canonical_id'] in allowed;values=r['bare_metrics'];name=r['spec']['fit_id']
                    else:
                        atom=next(t['point'] for t in r['metrics'] if t['metric']=='atomic_auroc' and t['scope']=='pooled')
                        okay=atom>=ref-.005;values=r['all_bare_points'];name=r['id']
                    if gate=='eligible' and not okay:continue
                    accepted.append((name,values))
                for metric,scope,topic in {(r['metric'],r['scope'],r['topic']) for _,rs in accepted for r in rs}:
                    candidates=[(name,r['point']) for name,rs in accepted for r in rs if (r['metric'],r['scope'],r['topic'])==(metric,scope,topic) and r['point'] is not None]
                    best=max(v for _,v in candidates) if candidates else None
                    maxima.append(dict(model=model,cohort=cohort,gate=gate,metric=metric,scope=scope,topic=topic,point=best,
                        exact_maximum_ids=[n for n,v in candidates if v==best],interval_scope='endpoint maximum descriptive; primary fixed-winner intervals in metrics.csv',reference_atomic=ref))
    write_csv(root/'metrics.csv',rows);write_csv(root/'contrasts.csv',contrasts);write_csv(root/'bank-maxima.csv',maxima)
    same_or_publish(root/'metrics.json',dict(records=rows));same_or_publish(root/'contrasts.json',dict(records=contrasts))
    same_or_publish(root/'bank-maxima.json',dict(records=maxima))
    return dict(metric_rows=len(rows),contrast_rows=len(contrasts),maxima_rows=len(maxima),
        interpretation='A identities occur in atomic training: advantaged full-training comparator versus existing reduced-P repair; no full-training repair',
        files={p:pin(root/p) for p in ('metrics.csv','contrasts.csv','bank-maxima.csv','metrics.json','contrasts.json','bank-maxima.json')})
