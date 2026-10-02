"""Authorized bounded fresh CPU stage: 2 reduced LR + 60 R0 + 60 C_clean fits."""
import argparse
from dataclasses import asdict
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from src import selection_repair_objectives as objective
from src.checkpoint_r2_fresh_adapter import FreshAdapter, clean_block, check_packet
from src.checkpoint_r2_fresh_inputs import ROOT, file_hash, hash_value, require
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_selection_v1 import AtomicCandidate, candidate_id, select_original_atomic_r0
from src.clean_transfer_statistics import interval, weighted_auc
from src.selection_repair_canonical_sensitivity import fit_reference, atomic_weights, compound_weights, auc_draws

CONFIG = 'config/checkpoint_r2/fresh_recoverability_v1.json'
TOPICS = ('animal_class','cities','element_symb','inventors','sp_en_trans')


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.exists(), 'refuse artifact overwrite: '+str(path))
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def finite(value):
    return float(value) if np.isfinite(value) else None


def head_id(model, family, layer):
    return f'{model}_{family}_L{layer:02d}'


def configuration():
    cfg = json.loads((ROOT/CONFIG).read_text())
    require(cfg['planned_fits']==122 and cfg['processes']==cfg['blas_threads']==1, 'bounded execution configuration')
    for path,expected in cfg['frozen_hashes'].items():
        require(file_hash(ROOT/path)==expected, 'frozen implementation/config/input changed: '+path)
    require({p:importlib.metadata.version(p) for p in cfg['local_packages']}==cfg['local_packages'], 'local numerical runtime changed')
    return cfg


def verify_current_inputs(acceptance,cfg):
    from src.checkpoint_r2_full_raw import plan,verify_output
    root=Path(acceptance['handoff'])
    inventory_path=root/'control/complete-copy-checksums.json'
    require(file_hash(inventory_path)==cfg['inventory_sha256']==acceptance['inventory_sha256'],'pinned full inventory changed')
    inventory=json.loads(inventory_path.read_text())
    for name,identity in inventory['files'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts,'unsafe inventory path')
        path=root/name
        require(path.is_file() and not path.is_symlink() and pin(path)==identity,'accepted input changed: '+name)
    full_cfg,manifest=plan()
    verify_output(acceptance['output'],full_cfg,manifest)


def options(cfg):
    return cfg['bootstrap']


def metric_masks(adapter):
    atom = adapter.rows['atomic_D']; comp = adapter.rows['D_bare']
    a = np.array([int(r['label']) for r in atom]); c = np.array([int(r['label']) for r in comp])
    op = np.array([r['operator'] for r in comp]); cells = np.array([int(r['truth_a'])+int(r['truth_b']) for r in comp])
    return [('atomic_auroc',a,np.ones(len(a),bool),'atomic_D'),
            ('AND_auroc',c,op=='AND','D_bare'),('OR_auroc',c,op=='OR','D_bare'),
            ('OR_mixed_vs_FF_auroc',c,(op=='OR')&(cells<2),'D_bare')]


def point_metrics(adapter, scores):
    records = []
    for metric,labels,mask,group in metric_masks(adapter):
        topics = np.array([r['topic'] for r in adapter.rows[group]])
        topic_points = []
        for scope,topic,selection in [('pooled','all',mask)]+[('topic',t,mask&(topics==t)) for t in TOPICS]:
            value = weighted_auc(labels[selection],scores[group][selection])
            records.append(dict(metric=metric,scope=scope,topic=topic,point=finite(value)))
            if scope=='topic': topic_points.append(value)
        records.append(dict(metric=metric,scope='topic_macro',topic='all',point=finite(np.mean(topic_points))))
    return records


def separation(p_scores, compound_scores, rows):
    scale = float(np.std(p_scores,ddof=0))
    if not np.isfinite(scale) or scale<=0: return None
    cells = np.array([int(r['truth_a'])+int(r['truth_b']) for r in rows])
    op = np.array([r['operator'] for r in rows])
    mixed = compound_scores[(op=='OR')&(cells==1)]/scale
    ff = compound_scores[(op=='OR')&(cells==0)]/scale
    return finite((mixed.mean()-ff.mean())/np.sqrt((mixed.var(ddof=0)+ff.var(ddof=0))/2+1e-8))


def choose_clean(records, cfg):
    """Inherited D OR-boundary, atomic, continuous-separation ranking."""
    require(records and all(r['valid_for_scoring'] for r in records), 'failed C_clean layer blocks all-layer selector')
    candidates = records[:]; stages = []
    for key in ('OR_mixed_vs_FF_auroc','atomic_auroc','separation'):
        values = [r[key] for r in candidates]
        require(all(v is not None and np.isfinite(v) for v in values), 'undefined C_clean selection endpoint')
        best = max(values); candidates = [r for r in candidates if best-r[key]<=cfg['tie_tolerance']]
        stages.append(dict(metric=key,best=best,tied_ids=sorted(r['id'] for r in candidates)))
    tied = sorted(r['id'] for r in candidates)
    # Protocol's shared final tie policy: fresh PCG64 seed over sorted IDs.
    selected = str(np.random.Generator(np.random.PCG64(cfg['tie_seed'])).choice(tied))
    return dict(selected_id=selected,stages=stages,final_tie_ids=tied,tie_seed=cfg['tie_seed'],
                label='optimistic development selection on D; intervals conditional on selected head')


def bootstrap_head(adapter, scores, weights, cfg):
    records = []; draws = []
    for metric,labels,mask,group in metric_masks(adapter):
        topics = np.array([r['topic'] for r in adapter.rows[group]])
        topic_draws = []; topic_points = []
        for scope,topic,selection in [('pooled','all',mask)]+[('topic',t,mask&(topics==t)) for t in TOPICS]:
            point, sample = auc_draws(scores[group],labels,np.flatnonzero(selection),weights[group])
            records.append(dict(metric=metric,scope=scope,topic=topic,point=finite(point),**interval(sample,options(cfg))))
            draws.append(sample)
            if scope=='topic': topic_draws.append(sample);topic_points.append(point)
        # Ordinary all-five mean deliberately propagates undefined topics.
        sample = np.mean(topic_draws,axis=0)
        records.append(dict(metric=metric,scope='topic_macro',topic='all',point=finite(np.mean(topic_points)),**interval(sample,options(cfg))))
        draws.append(sample)
    return records,np.array(draws)


def fit_one(adapter, features, model, layer, family, pre, cfg, destination, identity):
    name = head_id(model,family,layer); folder = destination/'fits'/name;folder.mkdir(parents=True)
    t = time.monotonic()
    group = 'TC_control_and_final_refit' if family=='C_clean' else 'P15'
    source = features[group]
    meta = dict(identity,id=name,model=model,family=family,saved_layer=layer,training_group=group,
                fit_rows=len(source),training_bindings_sha256=hash_value(adapter.bindings[group]),
                feature_float64_sha256=__import__('hashlib').sha256(source.tobytes()).hexdigest(),
                preprocessing_group=None if family=='reduced_LR' else 'P15',historical_parameters_reused=False)
    try:
        if family=='reduced_LR':
            y = np.array([int(r['label']) for r in adapter.rows['P15']])
            head,detail = fit_reference(source,y,cfg['models'][model]['C'],layer)
            parameters = dict(coef=head.coef_[0],intercept=head.intercept_[0])
            valid = detail['valid_for_scoring']
            scorer = lambda x: x@parameters['coef']+parameters['intercept']
            meta.update(C=cfg['models'][model]['C'],preprocessing='none',optimizer=detail)
        else:
            block = clean_block(source,adapter.rows[group],pre) if family=='C_clean' else objective.prepare_block(
                source,[int(r['label']) for r in adapter.rows[group]],pre)
            head = objective.fit_readout('r0',block,pre)
            valid = head.converged
            parameters = dict(standardized_coef=head.weights,standardized_intercept=head.intercept)
            attempts = []
            for attempt in head.attempts:
                entry = asdict(attempt); entry.pop('initial_parameters');entry.pop('final_parameters'); attempts.append(entry)
            meta.update(optimizer=dict(status=head.status,attempts=attempts,settings=asdict(head.settings)),
                        objective='mean TC BCE + .001 ||w||^2' if family=='C_clean' else 'mean P15 BCE + .001 ||w||^2',
                        intercept='unpenalized',preprocessing=pre.policy)
            scorer = head.decision_function
        np.savez(folder/'head.npz',**parameters)
        meta.update(valid_for_scoring=bool(valid),fit_seconds=time.monotonic()-t,parameters=pin(folder/'head.npz'),
                    process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if valid:
            scores = {g:scorer(features[g]) for g in ('P15','atomic_D','D_bare')}
            packets = {}
            for g in scores:
                packet=dict(row_binding_sha256=hash_value(adapter.bindings[g]),ordered_keys=[[r['group'],r['logical_id']] for r in adapter.bindings[g]],scores=scores[g])
                check_packet(adapter.bindings[g],packet)
                packets[g]={k:v for k,v in packet.items() if k!='scores'}
            np.savez(folder/'scores.npz',**scores)
            meta['score_bindings']=packets; meta['scores']=pin(folder/'scores.npz')
            meta['metrics']=point_metrics(adapter,scores)
            meta['separation']=separation(scores['P15'],scores['D_bare'],adapter.rows['D_bare'])
        else:
            meta['failure']='reviewed convergence/gradient/finite policy failed; prediction prohibited'
    except Exception as exc:
        meta.update(valid_for_scoring=False,failure=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc(),
                    fit_seconds=time.monotonic()-t)
    save(folder/'fit.json',meta)
    optimizer=meta.get('optimizer',{})
    print(json.dumps(dict(id=name,valid=meta['valid_for_scoring'],seconds=meta['fit_seconds'],
        iterations=[a['iterations'] for a in optimizer.get('attempts',[])],
        gradient=[a['gradient_infinity_norm'] for a in optimizer.get('attempts',[])],
        failure=meta.get('failure'))),flush=True)
    return meta


def evaluate(adapter, cfg, destination, fit_records, identity):
    t = time.monotonic(); atom = adapter.rows['atomic_D']; comp = adapter.rows['D_bare']
    weights = dict(atomic_D=atomic_weights(atom,np.ones(len(atom),bool),options(cfg)))
    frame = pd.DataFrame(comp).rename(columns={'person_a':'entity_a_id','person_b':'entity_b_id'})
    weights['D_bare'],schedule = compound_weights(frame,np.ones(len(frame),bool),options(cfg))
    np.savez(destination/'bootstrap_weights.npz',**weights)
    save(destination/'bootstrap_binding.json',dict(identity,seed=1729,replicates=2000,minimum_valid=1800,
        compound_schedule_sha256=schedule,atomic_weights_sha256=__import__('hashlib').sha256(weights['atomic_D'].tobytes()).hexdigest(),
        artifact=pin(destination/'bootstrap_weights.npz'),shared_across_heads_and_models=True,
        undefined='no redraw, imputation or surviving-topic macro; pointwise conditional intervals'))
    metrics = []; selected = {}; boot = {}; point = {}; contrast_records = []
    for r in fit_records:
        if not r['valid_for_scoring']:
            for metric,_,_,_ in metric_masks(adapter):
                for scope,topic in [('pooled','all')]+[('topic',t) for t in TOPICS]+[('topic_macro','all')]:
                    metrics.append(dict(identity,model=r['model'],family=r['family'],saved_layer=r['saved_layer'],id=r['id'],
                        metric=metric,scope=scope,topic=topic,point=None,status='fit_failed',failure=r['failure'],
                        ci_low=None,ci_high=None,valid_replicates=0,invalid_replicates=2000,ci_status='fit_failed'))
            continue
        folder = destination/'fits'/r['id']
        with np.load(folder/'scores.npz',allow_pickle=False) as a:
            scores = {g:check_packet(adapter.bindings[g],dict(r['score_bindings'][g],scores=a[g])) for g in ('P15','atomic_D','D_bare')}
        records,draws = bootstrap_head(adapter,scores,weights,cfg)
        np.save(folder/'bootstrap_metrics.npy',draws,allow_pickle=False)
        save(folder/'metrics.json',dict(identity,id=r['id'],records=records,draws=pin(folder/'bootstrap_metrics.npy')))
        boot[r['id']]=draws; point[r['id']]=np.array([np.nan if x['point'] is None else x['point'] for x in records])
        metrics.extend(dict(identity,model=r['model'],family=r['family'],saved_layer=r['saved_layer'],id=r['id'],**x) for x in records)
        print('evaluated',r['id'],flush=True)
    for model,spec in cfg['models'].items():
        reduced = head_id(model,'reduced_LR',spec['canonical_layer'])
        r0 = [r for r in fit_records if r['model']==model and r['family']=='R0']
        cc = [r for r in fit_records if r['model']==model and r['family']=='C_clean']
        def pooled(r,metric):
            return next(x['point'] for x in r.get('metrics',[]) if x['scope']=='pooled' and x['metric']==metric)
        candidates = [AtomicCandidate(candidate_id(model,r['saved_layer'],'r0'),model,r['saved_layer'],'r0',None,
                      r['valid_for_scoring'],pooled(r,'atomic_auroc') if r['valid_for_scoring'] else None,
                      {t:next(x['point'] for x in r['metrics'] if x['scope']=='topic' and x['metric']=='atomic_auroc' and x['topic']==t) for t in TOPICS} if r['valid_for_scoring'] else {}) for r in r0]
        atomic = select_original_atomic_r0(candidates,model=model)
        atomic_id = None if atomic['selected_id'] is None else head_id(model,'R0',next(c.layer for c in candidates if c.candidate_id==atomic['selected_id']))
        try:
            clean = choose_clean([dict(r,id=r['id'],atomic_auroc=pooled(r,'atomic_auroc') if r['valid_for_scoring'] else None,
                                      OR_mixed_vs_FF_auroc=pooled(r,'OR_mixed_vs_FF_auroc') if r['valid_for_scoring'] else None) for r in cc],cfg['C_clean_selector'])
        except ValueError as exc:
            clean=dict(selected_id=None,blocker=str(exc))
        selected[model]=dict(reduced_LR=reduced,canonical_layer=spec['canonical_layer'],R0_atomic_selector=atomic,
                            R0_atomic_selected_id=atomic_id,C_clean_D_selector=clean,
                            all_layer_valid=all(r['valid_for_scoring'] for r in r0+cc))
        comparisons = {(head_id(model,'C_clean',l),head_id(model,'R0',l),'same_layer') for l in spec['layers']}
        comparisons |= {(head_id(model,'C_clean',l),reduced,'against_fresh_reduced_LR') for l in spec['layers']}
        controls=[(head_id(model,'C_clean',spec['canonical_layer']),'fixed_canonical'),
                  (clean['selected_id'],'optimistic_D_selected'),
                  (None if atomic_id is None else head_id(model,'C_clean',next(r['saved_layer'] for r in r0 if r['id']==atomic_id)),'at_atomic_R0_layer')]
        for left,label in controls:
            if left:
                comparisons.add((left,head_id(model,'R0',next(r['saved_layer'] for r in cc if r['id']==left)),label+'_same_layer'))
                if atomic_id: comparisons.add((left,atomic_id,label+'_vs_atomic_selected_R0'))
                comparisons.add((left,reduced,label+'_vs_reduced_LR'))
        for r in r0:
            comparisons.add((r['id'],reduced,'R0_retention_against_fresh_reduced_LR'))
        for left,right,label in sorted(comparisons):
            if left not in boot or right not in boot:
                contrast_records.append(dict(identity,model=model,left=left,right=right,comparison=label,status='fit_failed'))
                continue
            deltas = boot[left]-boot[right]; delta_points=point[left]-point[right]
            templates = [r for r in metrics if r['id']==left]
            for i,template in enumerate(templates):
                entry=dict(identity,model=model,left=left,right=right,comparison=label,metric=template['metric'],scope=template['scope'],
                           topic=template['topic'],point=finite(delta_points[i]),**interval(deltas[i],options(cfg)),
                           optimistic_development=label.startswith('optimistic') or left==clean['selected_id'] or right==atomic_id,
                           interval_scope='conditional on fitted and selected heads; selection not repeated in bootstrap')
                if right==reduced and entry['metric'] in ('atomic_auroc','AND_auroc'):
                    margin=.005 if entry['metric']=='atomic_auroc' else .02
                    entry.update(retention_margin=margin,point_retained=entry['point'] is not None and entry['point']>=-margin,
                                 interval_supported=entry['ci_low'] is not None and entry['ci_low']>=-margin)
                contrast_records.append(entry)
    save(destination/'selected.json',dict(identity,models=selected))
    save(destination/'metrics.json',dict(identity,records=metrics))
    save(destination/'contrasts.json',dict(identity,records=contrast_records))
    pd.DataFrame(metrics).drop(columns=['code_hashes']).to_csv(destination/'all_layer_metrics.csv',index=False)
    pd.DataFrame(contrast_records).drop(columns=['code_hashes']).to_csv(destination/'paired_contrasts.csv',index=False)
    plot_curves(metrics,cfg,destination)
    return dict(seconds=time.monotonic()-t,metric_records=len(metrics),contrast_records=len(contrast_records),selections=selected)


def plot_curves(metrics,cfg,destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,4,figsize=(16,7),sharey=True)
    for row,model in enumerate(('qwen','llama')):
        for col,metric in enumerate(('atomic_auroc','AND_auroc','OR_auroc','OR_mixed_vs_FF_auroc')):
            ax=axes[row,col]
            for family in ('R0','C_clean'):
                for scope,style in [('pooled','-'),('topic_macro','--')]:
                    data=sorted([r for r in metrics if r['model']==model and r['family']==family and r['metric']==metric and r['scope']==scope],key=lambda r:r['saved_layer'])
                    ax.plot([r['saved_layer'] for r in data],[np.nan if r['point'] is None else r['point'] for r in data],style,label=family+(' pooled' if scope=='pooled' else ' macro'))
            reduced=next((r['point'] for r in metrics if r['model']==model and r['family']=='reduced_LR' and r['metric']==metric and r['scope']=='pooled'),None)
            if reduced is not None:ax.axhline(reduced,color='gray',linewidth=1,label='fresh reduced LR')
            ax.set(title=model+' '+metric,xlabel='Saved layer',ylim=(.45,1.01));ax.grid(alpha=.25)
    axes[0,0].legend(fontsize=8);fig.suptitle('Fresh bare-D recoverability — pooled and equal-topic macro');fig.tight_layout()
    fig.savefig(destination/'all_layer_curves.png',dpi=150);fig.savefig(destination/'all_layer_curves.svg');plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['validate','run'])
    p.add_argument('--acceptance',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expected-commit');args=p.parse_args();cfg=configuration()
    require(platform.system()=='Darwin','this approved local run is macOS; peak RSS units are bytes')
    require(file_hash(args.acceptance)==cfg['acceptance_sha256'],'input acceptance hash')
    acceptance=json.loads(args.acceptance.read_text())
    verify_current_inputs(acceptance,cfg)
    adapter=FreshAdapter(acceptance['output'],acceptance)
    if args.mode=='validate':
        save(args.output,dict(status='fresh_adapter_validated_no_fits',configuration_sha256=file_hash(ROOT/CONFIG),**adapter.receipt));return
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True,cwd=ROOT).strip()
    require(head==args.expected_commit and len(head)==40,'exact committed producer required')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],text=True,cwd=ROOT).strip(),'clean committed code required before first fit')
    destination=args.output.resolve();require(not destination.exists(),'fresh result directory required')
    require(not destination.is_relative_to(ROOT) and not destination.is_relative_to(Path(acceptance['handoff'])),'derived results must be outside checkout and backup')
    destination.mkdir(parents=True)
    identity=dict(producer_sha=head,configuration_sha256=file_hash(ROOT/CONFIG),input_manifest_sha256=acceptance['manifest_sha256'],
                  inventory_sha256=acceptance['inventory_sha256'],acceptance_sha256=cfg['acceptance_sha256'],
                  code_hashes={name:file_hash(ROOT/name) for name in cfg['record_code_paths']})
    save(destination/'execution.json',dict(identity,config=cfg,acceptance=acceptance,adapter=adapter.receipt,host=platform.node(),
         platform=platform.platform(),interpreter=sys.executable,python=sys.version))
    started=time.monotonic();records=[]
    with threadpool_limits(limits=1):
        require(all(x['num_threads']==1 for x in threadpool_info()),'one BLAS/OpenMP thread required')
        save(destination/'threadpools.json',dict(identity,pid=os.getpid(),processes=1,threads=threadpool_info()))
        for model in ('qwen','llama'):
            spec=cfg['models'][model]
            for layer in spec['layers']:
                layer_started=time.monotonic();features=adapter.layer(model,layer)
                pre=objective.fit_p_preprocessing(features['P15'])
                prepath=destination/'preprocessing'/model/f'L{layer:02d}.npz';prepath.parent.mkdir(parents=True,exist_ok=True)
                np.savez(prepath,mean=pre.mean,population_std=pre.population_std,denominator=pre.denominator,constant_mask=pre.constant_mask)
                save(prepath.with_suffix('.json'),dict(identity,model=model,saved_layer=layer,fit_group='P15',fit_rows=2778,
                     binding_sha256=hash_value(adapter.bindings['P15']),policy=pre.policy,artifact=pin(prepath)))
                families=['R0','C_clean']+(['reduced_LR'] if layer==spec['canonical_layer'] else [])
                for family in families:records.append(fit_one(adapter,features,model,layer,family,pre,cfg,destination,identity))
                print('layer completed',model,layer,'seconds',time.monotonic()-layer_started,flush=True)
                del features,pre;gc.collect()
        fit_seconds=time.monotonic()-started
        require(len(records)==122 and len({r['id'] for r in records})==122,'planned fit inventory')
        save(destination/'fit_inventory.json',dict(identity,records=records,planned=122,actual=len(records),fit_phase_seconds=fit_seconds))
        evaluation=evaluate(adapter,cfg,destination,records,identity)
    adapter.unchanged();failures=[r['id'] for r in records if not r['valid_for_scoring']]
    artifacts={str(path.relative_to(destination)):pin(path) for path in sorted(destination.rglob('*')) if path.is_file()}
    save(destination/'completion.json',dict(identity,status='completed' if not failures else 'completed_with_fit_failures',planned_fits=122,
         valid_fits=len(records)-len(failures),failures=failures,retries=[r['id'] for r in records if len(r.get('optimizer',{}).get('attempts',[]))>1 or r.get('optimizer',{}).get('retry_needed',False)],
         fit_phase_seconds=fit_seconds,evaluation=evaluation,total_seconds=time.monotonic()-started,
         peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,backup_unchanged=True,
         historical_compatibility='unverified',wording_evaluated=False,behavior_chat_E_evaluated=False,artifacts=artifacts))
    print('COMPLETED',destination,flush=True)


if __name__=='__main__':main()
