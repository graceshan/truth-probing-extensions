"""Independent capacity-pilot validation from scores/parameters; no refit or activation load."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src import selection_repair_capacity_pilot_v1 as p
from src.clean_transfer_statistics import interval
s=p.s


def check(actual,expected):
    s.require(np.allclose(actual,expected,rtol=0,atol=1e-12),'independent pilot value mismatch')


def validate(root,prepared,output):
    root,prepared,output=map(Path,(root,prepared,output))
    config,_=s.verify_preparation(root,prepared)
    spec=json.loads((root/p.CONFIG).read_text());binding=json.loads((output/'binding.json').read_text())
    s.require(binding['config']==s.identity((root/p.CONFIG).read_bytes()) and binding['preparation']==s.identity((prepared/'preparation_receipt.json').read_bytes()),'stale pilot validation input')
    for path,digest in binding['source_sha256'].items():s.require(s.sha((root/path).read_bytes())==digest,'pilot source changed')
    receipt=json.loads((output/'completion.json').read_text())
    s.require(receipt['correction_commit']==s.CORRECTION_COMMIT and receipt['unique_configurations']==72,'wrong completed grid/version')
    for path,pin in receipt['artifacts'].items():s.require(s.identity((output/path).read_bytes())==pin,'pilot artifact changed: '+path)
    fits=json.loads((output/'fits.json').read_text());expected=p.configurations(spec)
    s.require([r['configuration'] for r in fits]==expected,'ordered grid mismatch')
    memberships,samples=p.cohorts(root)
    for r in fits:
        c=r['configuration'];rows=(samples if c['method'] in ('burger_t_g','ttpd') else memberships)[c['cohort']]
        s.require(r['fit_rows']==len(rows) and r['exposure']==p.counts(rows) and r['fitting_identity']['membership_sha256']==s.sha(s.canonical(rows)),'fit exposure mismatch')
        s.require(r['A_exposed_diagnostic']==(c['cohort']=='full') and not r['primary_bank_eligible'],'bank contamination flag')
        if r['valid_for_scoring']:
            s.require(r['status']=='converged' and r['finite_parameters'],'false convergence')
            with np.load(output/'fits'/r['id']/'parameters.npz',allow_pickle=False) as saved:
                s.require(all(np.isfinite(saved[k]).all() for k in saved.files if saved[k].dtype.kind not in 'US'),'nonfinite saved parameters')
            if c['method']=='l2_logistic':s.require(r['gradient_infinity_norm']<=1e-4 and r['final_converged'],'LR convergence check')
            if c['method']=='r0':s.require(r['attempts'][-1]['gradient_infinity_norm']<=1e-4 and r['attempts'][-1]['status']=='converged','R0 convergence check')
            if c['method']=='ttpd':
                for key in ('polarity_optimizer','head_optimizer'):s.require(r[key]['valid_for_scoring'] and r[key]['attempts'][-1]['gradient_infinity_norm']<=1e-4,'TTPD convergence check')
    atom=pd.read_csv(output/'atomic_D_rows.csv');raw=pd.read_csv(output/'bare_D_rows.csv')
    s.require(len(atom)==1012 and len(raw)==7728 and set(atom.tc_current_eligible)=={True},'pilot eval coverage')
    metrics=json.loads((output/'metrics.json').read_text());max_error=0.;seen=set();points={}
    with np.load(output/'scores'/'paired_draws.npz',allow_pickle=False) as archive:full=archive['full'];P15=archive['P15']
    s.require(full.shape==P15.shape==(spec['bootstrap']['replicates'],len(metrics)),'paired draw dimensions')
    definitions={r['metric']:r for r in spec['endpoints']}
    valid_fit_ids={r['id'] for r in fits if r['valid_for_scoring']}
    expected_comparisons={(m,l,method) for m,v in spec['models'].items() for l in v['layers'] for method in spec['methods']
        if all(f'{m}_L{l}_{method}_{c}' in valid_fit_ids for c in ('full','P15'))}
    for i,r in enumerate(metrics):
        key=(r['model'],r['layer'],r['method'],r['metric'],r['scope'],r['topic'])
        s.require(key not in seen,'duplicate metric');seen.add(key)
        d=definitions[r['metric']];frame=atom if d['condition']=='atomic_D' else raw
        if d['condition']=='atomic_D':mask=np.ones(len(frame),bool);labels=frame.label.to_numpy();score_key='atomic'
        else:
            mask=((raw.operator==d['operator'])&raw.cell.isin(d['positive']+d['negative'])).to_numpy();labels=raw.cell.isin(d['positive']).to_numpy(int);score_key='bare'
        if r['scope']=='topic':mask = mask & (frame.topic.to_numpy()==r['topic'])
        observed=[]
        for cohort in ('full','P15'):
            with np.load(output/'scores'/f"{r['model']}_L{r['layer']}_{r['method']}_{cohort}.npz",allow_pickle=False) as archive:scores=archive[score_key]
            if r['scope']=='topic_macro':
                value=np.mean([roc_auc_score(labels[mask&(frame.topic.to_numpy()==t)],scores[mask&(frame.topic.to_numpy()==t)]) for t in spec['topics']])
            else:value=roc_auc_score(labels[mask],scores[mask])
            field='full_auroc' if cohort=='full' else 'P15_auroc'
            check(value,r[field]);max_error=max(max_error,abs(value-r[field]));observed.append(value)
        check(observed[1]-observed[0],r['withholding_delta'])
        for arr,name in [(full,'full_ci'),(P15,'P15_ci'),(P15-full,'paired_delta_ci')]:
            s.require(interval(arr[:,i],spec['bootstrap'])==r[name],'paired interval mismatch')
        threshold=spec['review_drop_thresholds'].get(r['metric'])
        s.require(r['triggered']==(threshold is not None and r['withholding_delta'] < -threshold),'review threshold mismatch')
    expected_metrics={(m,l,method,d['metric'],scope,topic) for m,l,method in expected_comparisons for d in spec['endpoints']
                      for scope,topic in [('pooled','all'),('topic_macro','all')]+[('topic',t) for t in spec['topics']]}
    s.require(seen==expected_metrics,'missing or extra computed endpoints')
    s.require(json.loads((output/'threshold_triggers.json').read_text())==[r for r in metrics if r['triggered']],'threshold receipt mismatch')
    s.require(receipt['failures']==[r['id'] for r in fits if not r['valid_for_scoring']],'failure reporting mismatch')
    return dict(status='passed',unique_configurations=len(fits),summaries_checked=len(metrics),failures=receipt['failures'],
                maximum_independent_auroc_error=max_error,intervals_recomputed=True,artifact_hashes_verified=True,
                exact_memberships_and_samplers_verified=True,real_fits=0,activation_loads=0,
                completion=s.identity((output/'completion.json').read_bytes()))
