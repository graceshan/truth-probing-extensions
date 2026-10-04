"""Future serialized full-bank CPU worker. Never called by local preparation/tests."""
from pathlib import Path
from dataclasses import asdict
import gc
import json
import time
import traceback
import resource
import platform
import sys
import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info
from src.checkpoint_next_inventory import ROOT, require, sha
from src.checkpoint_next_full_bank_inputs import FullTrainingAdapter, configuration, read, PRODUCERS
from src.checkpoint_next_full_bank_selection import bank_specs, selectors
from src.checkpoint_next_full_bank_provenance import FrozenB25
from src.checkpoint_next_chat import diagnostic_json
from src.checkpoint_next_store import same_or_publish, publish_json
from src.checkpoint_r2_fresh_store import pin, publish
from src.checkpoint_r2_fresh_inputs import hash_value
from src.checkpoint_r2_b25_store import array_hash, normalized_parameters
from src.checkpoint_r2_b25_statistics import summaries, allocated_statistics
from src import selection_repair_objectives as objective
from src import checkpoint_r2_logistic_continuation as methods
from src.selection_repair_capacity_methods_v1 import score


def folder(root,name):return Path(root)/'fits'/hash_value(name)[:24]


def load_fit(root,spec,identity):
    path=folder(root,spec['fit_id'])/'fit.json'
    if not path.exists():return None
    record=read(path)
    require(record['identity']==identity and record['spec']==spec,'resume producer/input/config/candidate changed')
    for name,key in [('parameters.npz','parameters'),('scores.npz','scores')]:
        if key in record:require(pin(path.parent/name)==record[key],'fit checkpoint corruption')
    return record


def score_packet(adapter,group,values):
    from src.checkpoint_r2_fresh_adapter import check_packet
    b=adapter.bindings[group];p=dict(row_binding_sha256=hash_value(b),ordered_keys=[[r['group'],r['logical_id']] for r in b])
    check_packet(b,dict(p,scores=values));return p


def candidate(root,spec,adapter,features,identity,pre):
    old=load_fit(root,spec,identity)
    if old is not None:
        require(old['valid_for_scoring'],'persistent fit failure preserved; no automatic refit')
        return old
    target=folder(root,spec['fit_id'])
    if target.exists():
        quarantine=Path(root)/'quarantine';quarantine.mkdir(exist_ok=True)
        target.rename(quarantine/(target.name+'-'+str(time.time_ns())))
    target.mkdir(parents=True)
    start=time.monotonic();X=features['outer_train'];rows=adapter.train
    if spec['training_group']=='balanced_full':
        X=X[adapter.full_balanced];rows=[dict(adapter.train[i],dataset=adapter.train[i]['source_row_id'].rsplit(':',1)[0]) for i in adapter.full_balanced]
    record=dict(identity=identity,spec=spec,valid_for_scoring=False,params_producer=identity['cpu_producer'],historical_reuse=False,
        extraction_producers=dict(existing='7656a0eb68a5fd7ff7dac1c73efeab982d540cc2',supplement=PRODUCERS[spec['model']]),
        training_rows=len(rows),training_source_sha256=hash_value(rows),training_features_sha256=array_hash(X))
    params=None
    try:
        require(len(rows)==spec['fit_rows'],'candidate exposure rows')
        if spec['method']=='r0':
            fit=objective.fit_readout('r0',objective.prepare_block(X,[int(r['label']) for r in rows],pre),pre)
            params=normalized_parameters(fit.weights,fit.intercept,pre)
            detail=dict(valid_for_scoring=fit.converged,status=fit.status,attempts=[asdict(a) for a in fit.attempts],
                settings=asdict(fit.settings),preprocessing=pre.policy,preprocessing_fit_rows=pre.p_rows,
                objective='mean full train BCE + .001 standardized squared weight norm; free intercept')
        else:params,detail=methods.fit(spec['method'],X,[int(r['label']) for r in rows],rows,spec['C'],spec['saved_layer'])
        # Preserve parameters and all diagnostics BEFORE rejecting failed optimizers.
        publish(target/'parameters.npz',lambda f:np.savez(f,**params));record['parameters']=pin(target/'parameters.npz')
        record['optimizer']=diagnostic_json(detail)
        valid=bool(detail['valid_for_scoring'] and all(np.isfinite(v).all() for v in params.values() if np.asarray(v).dtype.kind not in 'US'))
        require(valid,'existing convergence/finite rule failed; predictions blocked')
        groups=['P15','atomic_D','D_bare']+[f'B25_{s}' for s in (11,23,37)]
        values={g:score(spec['method'],params,features[g]) for g in groups}
        packets={g:score_packet(adapter,g,v) for g,v in values.items()}
        publish(target/'scores.npz',lambda f:np.savez(f,**values));record.update(scores=pin(target/'scores.npz'),score_bindings=packets)
        atomic,_=summaries(adapter.rows['atomic_D'],values['atomic_D'],atomic=True)
        point=lambda scope,topic:next(r['point'] for r in atomic if r['scope']==scope and r['topic']==topic)
        record['atomic']=dict(pooled=point('pooled','all'),topics={t:point('topic',t) for t in ('animal_class','cities','element_symb','inventors','sp_en_trans')})
        record['allocated']={str(s):allocated_statistics(adapter.rows[f'B25_{s}'],values[f'B25_{s}'],values['P15'],record['atomic']['pooled']) for s in (11,23,37)}
        # Full/P15 sensitivity changes the fitting procedure. Keep the inherited
        # B25 selection separation convention using original P15 score dispersion.
        record['oracle']=allocated_statistics(adapter.rows['D_bare'],values['D_bare'],values['P15'],record['atomic']['pooled'])
        record['bare_metrics'],_=summaries(adapter.rows['D_bare'],values['D_bare'])
        record['valid_for_scoring']=True
    except Exception as exc:
        record.update(valid_for_scoring=False,failure=str(exc),error_type=type(exc).__name__,traceback=traceback.format_exc())
    record['seconds']=time.monotonic()-start
    publish_json(target/'fit.json',diagnostic_json(record))
    require(record['valid_for_scoring'],'affected candidate stopped; failed fit and diagnostics preserved')
    return record


def accounting(root,identity):
    records=[]
    for spec in bank_specs():
        path=folder(root,spec['fit_id'])
        r=load_fit(root,spec,identity)
        records.append(dict(fit_id=spec['fit_id'],canonical_id=spec['canonical_id'],
            status=('valid' if r['valid_for_scoring'] else 'failed') if r else ('incomplete' if path.exists() else 'unrun'),
            fit_pin=pin(path/'fit.json') if r else None,failure=r.get('failure') if r else None))
    return dict(expected=600,counts={s:sum(r['status']==s for r in records) for s in ('valid','failed','incomplete','unrun')},records=records)


def worker(args,identity):
    start=time.monotonic();cfg=configuration();adapter=FullTrainingAdapter(args.handoffs)
    require(adapter.supplement_status=='transferred_independently_reopened','actual supplement acceptance required')
    # Full existing input-byte verification is mandatory at future launch/resume,
    # not replaced by today's metadata-only preparation.
    from src.checkpoint_r2_fresh_recoverability import verify_current_inputs
    verify_current_inputs(read(ROOT/'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json'),
        read(ROOT/'config/checkpoint_r2/fresh_recoverability_v1.json'))
    frozen=FrozenB25(adapter);records=[]
    same_or_publish(Path(args.output)/'input-acceptance.json',dict(existing=adapter.verify_existing_metadata(),supplement=adapter.supplement))
    with threadpool_limits(limits=1):
        require(all(p['num_threads']==1 for p in threadpool_info()),'one BLAS/OpenMP thread')
        for model in ('qwen','llama'):
            for layer in range(cfg['models'][model]['layers']):
                features=adapter.layer(model,layer)
                pre=objective.fit_p_preprocessing(features['outer_train'])
                require(pre.p_rows==3040,'full admitted preprocessing')
                prepath=Path(args.output)/'preprocessing'/model/f'L{layer:02d}.npz';prepath.parent.mkdir(parents=True,exist_ok=True)
                fields={k:getattr(pre,k) for k in ('mean','population_std','denominator','constant_mask')}
                if prepath.exists():
                    with np.load(prepath,allow_pickle=False) as saved:require(all(np.array_equal(saved[k],v) for k,v in fields.items()),'resume full preprocessing changed')
                else:publish(prepath,lambda f:np.savez(f,**fields))
                for spec in [s for s in bank_specs() if s['model']==model and s['saved_layer']==layer]:
                    records.append(candidate(args.output,spec,adapter,features,identity,pre))
                del features,pre;gc.collect()
    require(len(records)==600 and all(r['valid_for_scoring'] for r in records),'complete 600-candidate bank required')
    locks={}
    for model in ('qwen','llama'):
        original=frozen.existing_metrics(model,11,'reduced_LR')
        ref=next(r['point'] for r in original if r['condition']=='atomic_D' and r['scope']=='pooled')
        locks[model]=selectors([r for r in records if r['spec']['model']==model],model,ref,'frozen:reduced_LR')
    lock=dict(identity=identity,models=locks,wording_used_before_lock=False,repair_fits=0)
    same_or_publish(Path(args.output)/'locked.json',lock)
    from src.checkpoint_next_full_bank_evaluation import evaluate
    result=evaluate(args.output,adapter,frozen,records,lock,identity,cfg)
    adapter.unchanged()
    same_or_publish(Path(args.output)/'completion.json',dict(identity=identity,status='completed',bank=accounting(args.output,identity),
        lock_sha256=sha(Path(args.output)/'locked.json'),evaluation=result,repair_fits=0,full_training_repair_fitted=False,
        worker_seconds=time.monotonic()-start,process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        rss_units='bytes on macOS, KiB on Linux',hostname=platform.node(),interpreter=sys.executable))
