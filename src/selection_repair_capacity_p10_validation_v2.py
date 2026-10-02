"""Independent P10 score/receipt validation; no activation loading or new fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src import selection_repair_capacity_p10_v2 as p
from src.clean_transfer_statistics import interval
s=p.s


def close(actual,expected):
    s.require(np.allclose(actual,expected,rtol=0,atol=1e-12,equal_nan=False),'independent P10 metric mismatch')


def verify_convergence(record):
    method=record['configuration']['method']
    if not record['valid_for_scoring']:
        s.require(record['status'] in ('failed','flagged'),'unreported invalid fit');return
    s.require(record['status']=='converged' and record['finite_parameters'],'invalid finite/convergence status')
    if method=='l2_logistic':s.require(record['final_converged'] and record['gradient_infinity_norm']<=1e-4,'LR gradient/convergence failure')
    if method=='r0':s.require(record['attempts'][-1]['status']=='converged' and record['attempts'][-1]['gradient_infinity_norm']<=1e-4,'R0 gradient failure')
    if method=='ttpd':
        for key in ('polarity_optimizer','head_optimizer'):
            a=record[key]['attempts'][-1]
            s.require(record[key]['valid_for_scoring'] and a['library_converged'] and a['finite'] and a['gradient_infinity_norm']<=1e-4,'TTPD gradient failure')


def validate(root,output):
    root,output=Path(root),Path(output);config=p.verify_config(root)
    binding=json.loads((output/'binding.json').read_text());receipt=json.loads((output/'completion.json').read_text())
    old=Path(binding['prior_root']);prepared=Path(binding['prepared_root'])
    s.require(binding['config']==p.file_identity(root/p.CONFIG) and binding['prior_completion']==p.file_identity(old/'completion.json')
              and binding['prior_completion']==p.file_identity(root/p.PACKAGE/'completion.json'),'stale completed predecessor')
    s.require(binding['preparation']==p.file_identity(prepared/'preparation_receipt.json'),'stale P10 preparation')
    for name,digest in binding['source_sha256'].items():s.require(s.sha((root/name).read_bytes())==digest,'P10 source changed')
    for name,pin in receipt['artifacts'].items():s.require(p.file_identity(output/name)==pin,'P10 artifact changed: '+name)
    old_validation=p.prior_validation.validate(root,prepared,old)
    s.require(receipt['unique_new_configurations']==36 and receipt['cumulative_unique_configurations']==108
              and receipt['correction_commit']==s.CORRECTION_COMMIT and not receipt['P_A_frozen'],'P10 completed scope mismatch')
    rows,sample,membership=p.membership(root)
    s.require(json.loads((output/'membership.json').read_text())==membership and binding['membership']==s.sha(s.canonical(membership)),'P10 membership receipt differs')
    s.require((output/'P10_membership.csv').read_bytes()==s.csv_bytes(rows) and (output/'P10_balanced.csv').read_bytes()==s.csv_bytes(sample),'P10 exact ordered membership differs')
    fits=json.loads((output/'fits.json').read_text());s.require([r['configuration'] for r in fits]==p.configurations(config),'P10 configuration grid mismatch')
    for r in fits:
        c=r['configuration'];actual=sample if c['method'] in ('burger_t_g','ttpd') else rows
        s.require(r['fit_rows']==len(actual) and r['exposure']==p.prior.counts(actual)
                  and r['fitting_identity']['membership_sha256']==s.sha(s.canonical(actual)),'P10 actual fitting exposure mismatch')
        s.require(not r['A_exposed_diagnostic'] and not r['primary_bank_eligible'] and r['preprocessing_fitting_cohort']=='P10','P10 bank/scope contamination')
        verify_convergence(r)
        if 'parameters' in r:
            path=output/'fits'/r['id']/'parameters.npz';s.require(p.file_identity(path)==r['parameters'],'fit parameter hash mismatch')
            with np.load(path,allow_pickle=False) as a:
                s.require(all(np.isfinite(a[k]).all() for k in a.files if a[k].dtype.kind not in 'US'),'nonfinite P10 parameters')
                if c['method']=='r0':s.require(r['preprocessing_fit_rows']==2864,'R0 preprocessing cohort wrong')
    replay=json.loads((output/'predecessor_replay.json').read_text())
    prior_fits=json.loads((old/'fits.json').read_text())
    s.require({r['configuration'] for r in replay}=={r['id'] for r in prior_fits} and len(replay)==72
              and all(r['fit_identity_verified'] and r['score_replay_bitwise_equal'] for r in replay),'incomplete predecessor replay')
    for name in ('atomic_D_rows.csv','bare_D_rows.csv'):s.require((output/name).read_bytes()==(old/name).read_bytes(),'D identities/labels/order differ')
    atom=pd.read_csv(output/'atomic_D_rows.csv');raw=pd.read_csv(output/'bare_D_rows.csv')
    s.require(len(atom)==1012 and len(raw)==7728,'P10 D coverage')
    atom_records=s.parse_rows((output/'atomic_D_rows.csv').read_bytes())
    atom_w=p.prior.atomic_weights(atom_records,np.ones(len(atom),bool),config['bootstrap'])
    comp_w,schedule=p.prior.compound_weights(raw,np.ones(len(raw),bool),config['bootstrap'])
    expected_binding=dict(compound_schedule_sha256=schedule,atomic_weight_sha256=s.sha(atom_w.tobytes()))
    s.require(expected_binding==json.loads((old/'bootstrap_binding.json').read_text())==json.loads((output/'bootstrap_binding.json').read_text()),'entity bootstrap schedule mismatch')
    metrics=json.loads((output/'metrics.json').read_text());prior_metrics=json.loads((old/'metrics.json').read_text())
    index={p.metric_key(r):i for i,r in enumerate(prior_metrics)}
    with np.load(output/'scores'/'paired_draws.npz',allow_pickle=False) as a:base_draws=a['baseline'];new_draws=a['P10']
    with np.load(old/'scores'/'paired_draws.npz',allow_pickle=False) as a:old_draws={k:a[k] for k in a.files}
    s.require(base_draws.shape==new_draws.shape==(2000,len(metrics)),'P10 draw shape')
    scores={};definitions={r['metric']:r for r in config['endpoints']};seen=set();checked_new={};max_point=0.;max_weighted=0.;weighted_checks=0
    def load_score(model,layer,method,cohort):
        name=f'{model}_L{layer}_{method}_{cohort}'
        if name not in scores:
            path=(output if cohort=='P10' else old)/'scores'/(name+'.npz')
            with np.load(path,allow_pickle=False) as a:scores[name]={k:a[k] for k in a.files}
        return scores[name]
    for i,r in enumerate(metrics):
        key=p.metric_key(r);unique=(r['comparison'],key)
        s.require(unique not in seen,'duplicate P10 comparison');seen.add(unique)
        j=index[key];baseline=r['baseline']
        s.require(r['comparison']=='P10_minus_'+baseline and baseline in ('full','P15') and r['prior_metric_index']==j,'comparison/baseline identity differs')
        p.require_draw_alignment(base_draws[:,i],old_draws[baseline][:,j])
        if key in checked_new:p.require_draw_alignment(new_draws[:,i],new_draws[:,checked_new[key]])
        else:checked_new[key]=i
        d=definitions[r['metric']];frame=atom if d['condition']=='atomic_D' else raw
        if d['condition']=='atomic_D':
            mask=np.ones(len(frame),bool);labels=atom.label.to_numpy();score_key='atomic';weights=atom_w
        else:
            mask=((raw.operator==d['operator'])&raw.cell.isin(d['positive']+d['negative'])).to_numpy();labels=raw.cell.isin(d['positive']).to_numpy(int);score_key='bare';weights=comp_w
        if r['scope']=='topic':mask=mask & (frame.topic.to_numpy()==r['topic'])
        s.require(r['rows']==int(mask.sum()),'retained endpoint count mismatch')
        def auc(values,draw=None):
            groups=[mask&(frame.topic.to_numpy()==topic) for topic in config['topics']] if r['scope']=='topic_macro' else [mask]
            return float(np.mean([roc_auc_score(labels[g],values[g],sample_weight=None if draw is None else weights[draw,g]) for g in groups]))
        old_scores=load_score(r['model'],r['layer'],r['method'],baseline)[score_key]
        new_scores=load_score(r['model'],r['layer'],r['method'],'P10')[score_key]
        a,b=auc(old_scores),auc(new_scores);close([a,b,b-a],[r['baseline_auroc'],r['P10_auroc'],r['delta']])
        max_point=max(max_point,abs(a-r['baseline_auroc']),abs(b-r['P10_auroc']))
        s.require(r['baseline_auroc']==prior_metrics[j][baseline+'_auroc'],'baseline point differs from old result')
        for values,name in [(base_draws[:,i],'baseline_ci'),(new_draws[:,i],'P10_ci'),(new_draws[:,i]-base_draws[:,i],'paired_delta_ci')]:
            s.require(interval(values,config['bootstrap'])==r[name],'independent paired interval mismatch')
        # Independent sklearn weighted AUROC checks of new draws across all summaries.
        if r['comparison']=='P10_minus_full':
            for draw in (0,73,1999):
                value=auc(new_scores,draw);close(value,new_draws[draw,i]);max_weighted=max(max_weighted,abs(value-new_draws[draw,i]));weighted_checks+=1
        threshold=config['review_drop_thresholds'].get(r['metric'])
        s.require(r['review_drop_threshold']==threshold and r['triggered']==(threshold is not None and r['delta'] < -threshold),'P10 review threshold mismatch')
    valid={r['id'] for r in fits if r['valid_for_scoring']}
    expected={(comparison,(model,layer,method,d['metric'],scope,topic)) for model,v in config['models'].items() for layer in v['layers'] for method in config['methods']
              if f'{model}_L{layer}_{method}_P10' in valid for comparison in ('P10_minus_full','P10_minus_P15') for d in config['endpoints']
              for scope,topic in [('pooled','all'),('topic_macro','all')]+[('topic',t) for t in config['topics']]}
    s.require(seen==expected and len(metrics)==receipt['endpoint_summaries'],'missing/extra P10 comparison coverage')
    followup=json.loads((output/'trigger_followup.json').read_text())
    s.require(followup==p.trigger_followup(prior_metrics,metrics),'original/new trigger tracking mismatch')
    s.require(receipt['failures']==[r['id'] for r in fits if not r['valid_for_scoring']],'fit failures not preserved')
    return dict(status='passed',new_configurations=len(fits),cumulative_configurations=108,comparisons_checked=len(metrics),failures=receipt['failures'],
        maximum_independent_auroc_error=max_point,independent_weighted_AUROC_draw_checks=weighted_checks,maximum_weighted_draw_error=max_weighted,
        all_intervals_recomputed=True,baseline_draws_bitwise_match=True,P10_draws_shared_across_comparisons=True,
        all_artifact_hashes_verified=True,exact_membership_and_sampler_verified=True,original_triggers_tracked=14,
        predecessor_validation=old_validation,real_fits=0,activation_arrays_loaded=0,completion=p.file_identity(output/'completion.json'),
        validator_sha256=s.sha((root/'src/selection_repair_capacity_p10_validation_v2.py').read_bytes()))
