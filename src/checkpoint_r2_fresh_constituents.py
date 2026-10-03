"""Frozen raw constituent transfer panel; local CPU only, no E/behavior/chat."""
import argparse
from dataclasses import asdict
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info,threadpool_limits

from src import selection_repair_objectives as objective
from src.checkpoint_r2_constituent_adapter import ConstituentAdapter,FIT,VALID,FULL,BARE,score_packet,check_production_packet
from src.checkpoint_r2_constituent_metrics import summaries,paired_contrast,absolute_status
from src.checkpoint_r2_fresh_inputs import ROOT,file_hash,hash_value,require
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_fresh_recoverability import save,verify_current_inputs
from src.checkpoint_r2_b25_store import read,npz,array_hash
from src.checkpoint_r2_selection_v1 import TOPICS,ENDPOINTS,surface_labels,constituent_endpoints
from src.checkpoint_r2_full_inputs import WORDING
from src.selection_repair_canonical_sensitivity import compound_weights

CONFIG='config/checkpoint_r2/fresh_constituents_v1.json'


def configuration():
    cfg=read(ROOT/CONFIG)
    require(cfg['processes']==cfg['blas_threads']==1 and cfg['initial_heads']==240 and cfg['maximum_final_heads']==16,'frozen execution inventory')
    require(cfg['objective_settings']==asdict(objective.OptimizerSettings()),'reviewed solver settings')
    for name,digest in cfg['frozen_hashes'].items():require(file_hash(ROOT/name)==digest,'frozen file changed: '+name)
    require({p:importlib.metadata.version(p) for p in cfg['local_packages']}==cfg['local_packages'],'numerical runtime drift')
    return cfg


class FreshPreprocessing:
    """Reuse only accepted fresh P15 statistics and D bootstrap; no old heads."""
    def __init__(self,adapter,cfg):
        delivery=read(ROOT/cfg['recoverability_delivery'])
        self.root=Path(delivery['external_results']);self.completion=read(self.root/'completion.json')
        require(file_hash(self.root/'completion.json')==delivery['external_completion_sha256'],'fresh preprocessing delivery')
        require(self.completion['producer_sha']==cfg['preprocessing_producer_sha'] and self.completion['inventory_sha256']==cfg['inventory_sha256'],'fresh preprocessing producer/representation')
        self.stats={}
        for model,spec in adapter.cfg['models'].items():
            for layer in range(spec['layers']):
                for suffix in ('npz','json'):
                    rel=f'preprocessing/{model}/L{layer:02d}.{suffix}';p=self.root/rel
                    require(pin(p)==self.completion['artifacts'][rel],'fresh P15 statistics hash')
                    self.stats[p]=(p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)
                meta=read(self.root/f'preprocessing/{model}/L{layer:02d}.json')
                require(meta['fit_group']=='P15' and meta['fit_rows']==2778 and meta['binding_sha256']==hash_value(adapter.bindings['P15']),'P15 preprocessing identity')
        self.bootstrap_path=self.root/'bootstrap_weights.npz'
        require(pin(self.bootstrap_path)==self.completion['artifacts']['bootstrap_weights.npz'],'shared bootstrap source hash')
        self.stats[self.bootstrap_path]=(self.bootstrap_path.stat().st_size,self.bootstrap_path.stat().st_mtime_ns,self.bootstrap_path.stat().st_ctime_ns)

    def pre(self,model,layer,P,destination):
        path=self.root/f'preprocessing/{model}/L{layer:02d}.npz';data=npz(path)
        actual=objective.fit_p_preprocessing(P)
        require(set(data)=={'mean','population_std','denominator','constant_mask'},'P15 statistics fields')
        for name,value in data.items():require(np.array_equal(value,getattr(actual,name)),'fresh P15 statistics not reproducible')
        target=destination/f'preprocessing/{model}/L{layer:02d}.npz'
        if not target.exists():
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
            save(target.with_suffix('.json'),dict(model=model,saved_layer=layer,source=str(path),source_artifact=pin(path),
                preprocessing_producer_sha=self.completion['producer_sha'],fit_group='P15',fit_rows=2778,
                policy=actual.policy,exact_recomputed_from_P15=True,artifact=pin(target)))
        require(pin(target)==pin(path),'derived preprocessing changed')
        return actual

    def unchanged(self):
        require(all((p.stat().st_size,p.stat().st_mtime_ns,p.stat().st_ctime_ns)==s for p,s in self.stats.items()),'fresh preprocessing source modified')


def fitting_spec(adapter,model,layer,operator,position,group):
    require(group in (FIT,FULL) and position in ('FIRST','SECOND'),'constituent-only fit role')
    block=adapter.block(group,operator)
    return dict(id=f'{model}/{group}/{operator}/L{layer:02d}/{position}',model=model,saved_layer=layer,
                operator=operator,position=position,group=group,rows=len(block['rows']),
                training_bindings_sha256=hash_value(block['bindings']),training_metadata_sha256=hash_value(block['rows']),
                objective='mean surface-constituent BCE + .001*||w||^2; free intercept',
                settings=asdict(objective.OptimizerSettings()),preprocessing_group='P15')


def initial_inventory(adapter):
    return [fitting_spec(adapter,m,l,o,p,FIT) for m in ('qwen','llama') for l in range(adapter.cfg['models'][m]['layers'])
            for o in ('AND','OR') for p in ('FIRST','SECOND')]


def fit_folder(root,name):return root/'fits'/hash_value(name)[:24]


def fit_head(adapter,features,pre,spec,root,identity):
    folder=fit_folder(root,spec['id']);folder.mkdir(parents=True)
    idx=adapter.indices_for(spec['group'],spec['operator']);rows=adapter.block(spec['group'],spec['operator'])['rows']
    labels=surface_labels(pd.DataFrame(rows))[0 if spec['position']=='FIRST' else 1]
    X=features[spec['group']][idx];block=objective.prepare_block(X,labels,pre)
    start=time.monotonic();record=dict(identity,**spec,feature_float64_sha256=array_hash(X),labels_sha256=array_hash(labels),
        preprocessing_artifact=pin(root/f'preprocessing/{spec["model"]}/L{spec["saved_layer"]:02d}.npz'),historical_reuse=False)
    try:
        head=objective.fit_readout('r0',block,pre)  # mean-single-block primitive, not P BCE or half-P repair
        attempts=[]
        for a in head.attempts:
            attempts.append({k:v for k,v in asdict(a).items() if k not in ('initial_parameters','final_parameters')})
        np.savez(folder/'head.npz',weights=head.weights,intercept=head.intercept)
        record.update(valid=head.converged,optimizer_status=head.status,attempts=attempts,parameters=pin(folder/'head.npz'))
        if not head.converged:record['failure']='reviewed mean-constituent finite/success/gradient gate failed; scoring forbidden'
    except Exception as exc:
        record.update(valid=False,failure=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc())
    record['fit_seconds']=time.monotonic()-start;save(folder/'fit.json',record)
    print(json.dumps(dict(id=spec['id'],valid=record['valid'],seconds=record['fit_seconds'],
        iterations=[a['iterations'] for a in record.get('attempts',[])],gradient=[a['gradient_infinity_norm'] for a in record.get('attempts',[])],failure=record.get('failure'))),flush=True)
    return record


def predict_pair(root,heads,pre,X):
    require(len(heads)==2 and all(h['valid'] for h in heads),'invalid fitted heads prohibit predictions')
    require([h['position'] for h in heads]==['FIRST','SECOND'],'head surface ordering')
    Z=pre.transform(X);values=[]
    for h in heads:
        path=fit_folder(root,h['id'])/'head.npz';require(pin(path)==h['parameters'],'head corruption')
        p=npz(path);values.append(Z@p['weights']+float(p['intercept']))
    return np.column_stack(values)


def save_scores(adapter,root,name,model,layer,group,operator,values,heads,lock=None):
    folder=root/'scores'/hash_value(name)[:24];folder.mkdir(parents=True)
    np.save(folder/'scores.npy',values,allow_pickle=False)
    meta=dict(id=name,head_ids=[h['id'] for h in heads],packet=score_packet(adapter,model,layer,group,operator,values),
              artifact=pin(folder/'scores.npy'),lock_sha256=lock)
    save(folder/'binding.json',meta)
    return dict(folder=str(folder.relative_to(root)),**meta)


def run(adapter,cfg,pre_source,destination,identity):
    started=time.monotonic();fits=[];curve_records=[];validation={};initial=initial_inventory(adapter)
    require(len(initial)==240,'initial 240 head inventory')
    save(destination/'frozen-fitting-inventory.json',dict(identity,initial=initial,final_recipe=dict(group=FULL,
        all_frozen_100_of_190_pairs_per_topic=True,maximum_heads=16,dedup_key='model,layer,training operator,surface position,exact ordered exposure,objective,preprocessing'),
        exclusions=adapter.exclusions))
    for model in ('qwen','llama'):
        validation[model]={o:{} for o in ('AND','OR')}
        for layer in range(adapter.cfg['models'][model]['layers']):
            features=adapter.layer_groups(model,layer,('P15',FIT,VALID,BARE))
            pre=pre_source.pre(model,layer,features['P15'],destination)
            for op in ('AND','OR'):
                heads=[]
                for position in ('FIRST','SECOND'):
                    rec=fit_head(adapter,features,pre,fitting_spec(adapter,model,layer,op,position,FIT),destination,identity)
                    fits.append(rec);heads.append(rec)
                    require(rec['valid'],'invalid initial constituent fit: '+rec['id'])
                for group in (VALID,BARE):
                    eval_ops=(op,) if group==VALID else ('AND','OR')
                    for eval_op in eval_ops:
                        idx=adapter.indices_for(group,eval_op);values=predict_pair(destination,heads,pre,features[group][idx])
                        score=save_scores(adapter,destination,f'initial/{model}/L{layer:02d}/{op}/{group}/{eval_op}',model,layer,group,eval_op,values,heads)
                        if group==VALID:validation[model][op][layer]=values
                        metrics,_=summaries(adapter.block(group,eval_op)['rows'],values,eval_op,orders=False)
                        curve_records.extend(dict(model=model,saved_layer=layer,source_operator=op,evaluation_operator=eval_op,condition=group,
                            fit_stage='source_fit_all_layer_descriptive',score_artifact=score['folder'],**r) for r in metrics if r['metric'] in ENDPOINTS and r['surface_order']=='all')
            print('source layer complete',model,layer,flush=True);del features,pre;gc.collect()
    selections={m:{o:adapter.select_production(m,o,validation[m][o]) for o in ('AND','OR')} for m in ('qwen','llama')}
    save(destination/'source-validation-curves.json',dict(identity,selections=selections))
    require(all(s['status']=='selected' for ops in selections.values() for s in ops.values()),'no valid source-validation layer')
    save(destination/'locked-selection.json',dict(identity,selections=selections,source_only=True,D_or_wording_used=False,
        target_validation_used=False,absolute_planning_target=.90,transfer_loss_margin=.05))
    pd.DataFrame(curve_records).to_csv(destination/'constituent-layer-curves.csv',index=False)
    lock=file_hash(destination/'locked-selection.json');specs={};aliases={}
    for m,ops in selections.items():
        for source,s in ops.items():
            layer=s['selected_layer'];aliases[m,source]={}
            for trained in ('AND','OR'):
                names=[]
                for pos in ('FIRST','SECOND'):
                    spec=fitting_spec(adapter,m,layer,trained,pos,FULL);specs.setdefault(spec['id'],spec);names.append(spec['id'])
                aliases[m,source][trained]=names
    require(len(specs)<=16,'at most 16 distinct final head fits')
    save(destination/'final-fitting-inventory.json',dict(identity,specs=list(specs.values()),alias_records=[dict(model=m,source_operator=o,heads=values) for (m,o),values in aliases.items()],lock_sha256=lock))
    final={}
    for m,layer in sorted({(s['model'],s['saved_layer']) for s in specs.values()}):
        features=adapter.layer_groups(m,layer,('P15',FULL));pre=pre_source.pre(m,layer,features['P15'],destination)
        for spec in [s for s in specs.values() if s['model']==m and s['saved_layer']==layer]:
            rec=fit_head(adapter,features,pre,spec,destination,identity);fits.append(rec);final[spec['id']]=rec
            require(rec['valid'],'invalid final constituent fit: '+rec['id'])
        del features,pre;gc.collect()
    save(destination/'fit-inventory.json',dict(identity,expected_initial=240,expected_final=len(specs),records=fits))
    opts=cfg['bootstrap'];bare=adapter.rows[BARE]
    frame=pd.DataFrame(bare).rename(columns={'person_a':'entity_a_id','person_b':'entity_b_id'})
    weights,schedule=compound_weights(frame,np.ones(len(frame),bool),opts)
    with np.load(pre_source.bootstrap_path,allow_pickle=False) as old:require(np.array_equal(weights,old['D_bare']),'shared reviewed entity bootstrap drift')
    np.save(destination/'bootstrap-weights.npy',weights,allow_pickle=False)
    save(destination/'bootstrap-binding.json',dict(identity,options=opts,compound_schedule_sha256=schedule,
        artifact=pin(destination/'bootstrap-weights.npy'),reuses_reviewed_D_schedule_exactly=True,
        shared_across_models_heads_operators_wordings=True,endpoint_product=True))
    bare_lookup={r['example_id']:i for i,r in enumerate(bare)}
    evaluated={};metric_records=[];cases=[]
    # Reuse score/metric artifacts only for identical head IDs and identical logical observations.
    for m,source in aliases:
        layer=selections[m][source]['selected_layer'];target='OR' if source=='AND' else 'AND'
        cases.extend(dict(model=m,source_operator=source,target_operator=target,saved_layer=layer,arm=arm,
                          trained_operator=trained,evaluation_operator=op,heads=aliases[m,source][trained])
            for arm,trained,op in [('source_on_source',source,source),('source_on_target',source,target),('target_on_target',target,target)])
    for m,layer in sorted({(c['model'],c['saved_layer']) for c in cases}):
        features=adapter.layer_groups(m,layer,('P15',BARE,*sorted(WORDING)));pre=pre_source.pre(m,layer,features['P15'],destination)
        for case in [c for c in cases if c['model']==m and c['saved_layer']==layer]:
            op=case['evaluation_operator'];heads=[final[n] for n in case['heads']]
            groups=[BARE]+[g for g in sorted(WORDING) if g.startswith('and_')==(op=='AND')]
            for group in groups:
                name=hash_value(dict(heads=case['heads'],group=group,operator=op));key=(m,case['source_operator'],case['arm'],group)
                if name not in evaluated:
                    idx=adapter.indices_for(group,op);rows=adapter.block(group,op)['rows']
                    values=predict_pair(destination,heads,pre,features[group][idx])
                    score=save_scores(adapter,destination,name,m,layer,group,op,values,heads,lock)
                    wb=weights[:,[bare_lookup[r['example_id']] for r in rows]]
                    for i,r in enumerate(rows):require(r['pair_id']==bare[bare_lookup[r['example_id']]]['pair_id'],'wording pairing identity')
                    records,draws=summaries(rows,values,op,wb,opts)
                    folder=destination/score['folder'];np.save(folder/'bootstrap-metrics.npy',draws,allow_pickle=False)
                    for r in records:
                        if r['metric'] in ENDPOINTS:r.update(absolute_planning_target=.90,absolute_classification=absolute_status(r))
                    save(folder/'metrics.json',dict(records=records,draws=pin(folder/'bootstrap-metrics.npy')))
                    evaluated[name]=dict(records=records,draws=draws,score=score)
                result=evaluated[name];metric_records.extend(dict(**{k:v for k,v in case.items() if k!='heads'},condition=group,score_folder=result['score']['folder'],**r) for r in result['records'])
                evaluated[key]=result
        del features,pre;gc.collect()
    contrasts=[]
    for m,source in aliases:
        target='OR' if source=='AND' else 'AND'
        for group in [BARE]+[g for g in sorted(WORDING) if g.startswith('and_')==(target=='AND')]:
            lhs=evaluated[m,source,'target_on_target',group];rhs=evaluated[m,source,'source_on_target',group]
            require(lhs['score']['packet']==rhs['score']['packet'],'identical paired target observations/representation')
            records=paired_contrast(lhs['records'],rhs['records'],lhs['draws'],rhs['draws'],opts)
            contrasts.extend(dict(model=m,source_operator=source,target_operator=target,condition=group,
                saved_layer=selections[m][source]['selected_layer'],contrast='target_trained_minus_source_trained',
                lhs_score_folder=lhs['score']['folder'],rhs_score_folder=rhs['score']['folder'],paired=True,**r) for r in records)
    save(destination/'metrics.json',dict(identity,records=metric_records));save(destination/'paired-contrasts.json',dict(identity,records=contrasts))
    pd.DataFrame(metric_records).to_csv(destination/'constituent-transfer.csv',index=False)
    pd.DataFrame(contrasts).to_csv(destination/'paired-transfer-losses.csv',index=False)
    adapter.unchanged();pre_source.unchanged()
    artifacts={str(p.relative_to(destination)):pin(p) for p in sorted(destination.rglob('*')) if p.is_file()}
    save(destination/'completion.json',dict(identity,status='completed',expected_initial=240,completed_initial=240,
        expected_final=len(specs),completed_final=len(specs),valid_fits=len(fits),failed_fits=0,
        optimizer_retries=sum(len(r.get('attempts',[]))-1 for r in fits),seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,artifacts=artifacts,
        historical_compatibility='unverified',behavior_chat_E_accessed=False,scientific_scope='raw development constituent diagnostics only'))
    print('COMPLETED',destination,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('validate','run'))
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-commit');a=p.parse_args()
    cfg=configuration();require(platform.system()=='Darwin','approved macOS CPU host required')
    acceptance_path=ROOT/'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json'
    require(file_hash(acceptance_path)==cfg['acceptance_sha256'],'accepted fresh receipt changed')
    acceptance=read(acceptance_path);verify_current_inputs(acceptance,cfg)
    adapter=ConstituentAdapter(acceptance);pre_source=FreshPreprocessing(adapter,cfg)
    inventory=initial_inventory(adapter)
    if a.mode=='validate':
        save(a.output,dict(status='production_constituent_bindings_validated_no_fits',config_sha256=file_hash(ROOT/CONFIG),
            initial_heads=len(inventory),initial_inventory_sha256=hash_value(inventory),adapter=adapter.receipt));return
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(head==a.expected_commit and len(head)==40 and not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=ROOT,text=True).strip(),'exact clean committed fitting producer required')
    dest=a.output.resolve();require(not dest.exists() and not dest.is_relative_to(ROOT) and not dest.is_relative_to(Path(acceptance['handoff'])),'fresh external result directory required')
    dest.mkdir(parents=True);identity=dict(producer_sha=head,config_sha256=file_hash(ROOT/CONFIG),
        input_manifest_sha256=acceptance['manifest_sha256'],inventory_sha256=cfg['inventory_sha256'],acceptance_sha256=cfg['acceptance_sha256'])
    save(dest/'execution.json',dict(identity,host=platform.node(),platform=platform.platform(),python=sys.version,interpreter=sys.executable,
        cfg=cfg,adapter=adapter.receipt,preprocessing_source=str(pre_source.root),pid=os.getpid(),processes=1))
    try:
        with threadpool_limits(limits=1):
            require(all(x['num_threads']==1 for x in threadpool_info()),'single BLAS/OpenMP thread required')
            save(dest/'threads.json',dict(pid=os.getpid(),processes=1,pools=threadpool_info(),environment={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS')}))
            run(adapter,cfg,pre_source,dest,identity)
    except Exception as exc:
        completed=[read(p) for p in sorted((dest/'fits').glob('*/fit.json'))]
        save(dest/'validity-halt.json',dict(identity,reason=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc(),
            expected_initial=240,completed_fits=len(completed),valid=sum(r['valid'] for r in completed),failed=sum(not r['valid'] for r in completed),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,records=completed,downstream_execution_stopped=True))
        raise


if __name__=='__main__':main()
