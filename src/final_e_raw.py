"""Versioned E raw interface; historical development-only guards remain untouched.

All public E I/O enters through the final_e_runner authorization gate. This module
reuses the accepted backend/writer/loader and watchdog, never chat or generation.
"""
import ctypes
import fcntl
import inspect
import json
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
import numpy as np
from src.final_e_inventory import ROOT, Metadata, read
from src.checkpoint_r2_fresh_inputs import file_hash, hash_value, require, text_hash
from src.checkpoint_r2_fresh_store import pin, write_json, load_features, storage_probe
from src.checkpoint_r2_full_support import evidence, verify_snapshot, single_packet, filesystem_prerequisites, NON_SECRET_ARTIFACT_CONTRACT
from src.checkpoint_r2_full_raw import runtime_matches, extract_model, verify_completed, supervise, unchanged_or_publish, recover_budget, cap_reason
from src.checkpoint_r2_fresh_pilot import atomic_pointer


def raw_contract():
    cfg=read(ROOT/'config/checkpoint_r2/full_raw_v1.json')
    for path,sha in cfg['evidence_sha256'].items():require(file_hash(ROOT/path)==sha, 'accepted raw evidence')
    return cfg


def input_rows(metadata):
    bytext={r['id']:[] for r in metadata.texts}
    for b in metadata.bindings:bytext[b['text_id']].append(b)
    return [dict(r,bindings=bytext[r['id']]) for r in metadata.texts]


def campaign_identity(access, metadata):
    return dict(schema='final-e-raw-campaign-v1',producer_sha=access['producer_sha'],lock_sha256=access['lock_sha256'],
        campaign_id=access['campaign_id'],metadata_sha256=hash_value(metadata.receipt),
        raw_contract_sha256=file_hash(ROOT/'config/checkpoint_r2/full_raw_v1.json'),batch_size=1,padding='none',
        scientific_evaluation=False)


def provenance(identity, model, origin):
    return dict(identity,model=model,runtime_receipt_sha256=file_hash(origin),purpose='final-E-raw-v1')


def offline_tokenizer(cfg, model, cache):
    from transformers import AutoTokenizer
    spec=cfg['models'][model];snapshot=verify_snapshot(evidence(cfg,model+'-snapshot.json'),spec,cache)
    tok=AutoTokenizer.from_pretrained(snapshot['snapshot'],local_files_only=True,trust_remote_code=False,use_fast=True)
    expected=evidence(cfg,model+'/runtime-before-inference.json')['backend']
    require(text_hash(tok.backend_tokenizer.to_str())==expected['tokenizer_backend_sha256'] and
            file_hash(inspect.getfile(type(tok)))==expected['tokenizer_implementation_sha256'], 'pinned tokenizer backend/implementation')
    return snapshot,tok


def shard_plan(rows, packets, spec, size, storage):
    """Use the inherited exact payload/JSON and receipt/commit allowances."""
    from src.checkpoint_r2_fresh_inputs import canonical
    return [dict(start=i, rows=len(rows[i:i+size]),
        tokens_sha256=hash_value(packets[i:i+size]),
        reserved_bytes=len(rows[i:i+size])*spec['layers']*spec['width']*2+128+
            len(canonical(rows[i:i+size]))+len(canonical(packets[i:i+size]))+
            storage['per_shard_receipt_reserve_bytes']+storage['per_shard_commit_reserve_bytes'])
        for i in range(0,len(rows),size)]


def worker(args, access):
    require(platform.system()=='Linux', 'GPU execution requires approved Linux runtime')
    require(os.getppid()==args.parent_pid and ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)==0 and os.getppid()==args.parent_pid, 'parent death watchdog')
    output,attempt=Path(args.output),Path(args.attempt)
    metadata=Metadata();rows=input_rows(metadata);cfg=raw_contract();model=args.model
    identity=campaign_identity(access,metadata);unchanged_or_publish(output/'campaign.json',identity)
    modelroot=output/model;modelroot.mkdir(exist_ok=True)
    origin=modelroot/'runtime-origin.json'
    progress=dict(model=None,intervals=[],phase_started=time.monotonic());atomic_pointer(attempt/'progress.json',progress)
    costs={};backend=None;started=time.monotonic();status='failed';error=None
    def timed(name, fn):
        t=time.monotonic()
        try:return fn()
        finally:costs[name]=costs.get(name,0.)+time.monotonic()-t
    try:
        from src.checkpoint_r2_fresh_backend import FreshBackend,runtime_receipt
        runtime=timed('runtime_verification',lambda:runtime_receipt(cfg['runtime_lock']))
        runtime_matches(runtime,evidence(cfg,'host-runtime.json'));write_json(attempt/'runtime.json',runtime)
        volume=Path(args.volume).resolve();cache=Path(args.cache).resolve()
        require(os.path.ismount(volume) and output.resolve().is_relative_to(volume) and cache.is_relative_to(volume), 'persistent output/cache mount')
        require(not output.resolve().is_relative_to(cache) and not output.resolve().is_relative_to(ROOT), 'preserve code/cache')
        fs=timed('filesystem',lambda:filesystem_prerequisites(volume,artifact_contract=NON_SECRET_ARTIFACT_CONTRACT))
        write_json(attempt/'filesystem.json',fs)
        snapshot,tok=timed('snapshot_and_tokenizer',lambda:offline_tokenizer(cfg,model,cache))
        packets=timed('tokenization',lambda:[single_packet(tok,r,cfg['execution']['max_tokens']) for r in rows]);del tok
        unchanged_or_publish(modelroot/'token-plan.json',packets)
        size=read(ROOT/'config/final_e/preparation_v1.json')['shard_rows'];spec=cfg['models'][model]
        # Two complete model copies, one temporary shard, exact serialized metadata,
        # and the inherited safety reserve; allocation/fsync/readback, never statvfs alone.
        from src.checkpoint_r2_fresh_inputs import canonical
        plan=shard_plan(rows,packets,spec,size,cfg['storage'])
        payload_all=sum(len(rows)*s['layers']*s['width']*2 for s in cfg['models'].values())
        # Other-model token storage is a prospective upper allowance, not a measured length.
        other_token_allowance=len(rows)*cfg['execution']['max_tokens']*128
        shard_allowance=cfg['storage']['per_shard_receipt_reserve_bytes']+cfg['storage']['per_shard_commit_reserve_bytes']+128
        reserve=2*(payload_all+2*len(canonical(rows))+len(canonical(packets))+other_token_allowance+
            2*((len(rows)+size-1)//size)*shard_allowance+cfg['storage']['global_receipt_reserve_bytes'])+max(p['reserved_bytes'] for p in plan)+cfg['storage']['safety_margin_bytes']
        storage=timed('storage_allocation_verification',lambda:storage_probe(volume,reserve));write_json(attempt/'storage.json',dict(storage,reserve_bytes=reserve))
        progress.update(model=model,phase_started=time.monotonic());atomic_pointer(attempt/'progress.json',progress)
        backend=timed('model_loading',lambda:FreshBackend(spec,snapshot,cfg['execution']))
        expected=evidence(cfg,model+'/runtime-before-inference.json')['backend']
        # Peak tracking metadata was added in the reviewed successor; numerical flags stay exact.
        details=dict(backend.details);details.pop('peak_memory_tracking',None)
        old=dict(expected);old.pop('peak_memory_tracking',None)
        require(details==old,'accepted raw backend/flags')
        current=dict(runtime=runtime,backend=backend.details,snapshot_sha256=hash_value(snapshot))
        if origin.exists():
            previous=read(origin);runtime_matches(previous['runtime'],runtime)
            require(previous['backend']==current['backend'] and previous['snapshot_sha256']==current['snapshot_sha256'],'resume representation')
        else:write_json(origin,current)
        # The reviewed extractor publishes each shard, verifies through the independent
        # loader, and quarantines only interrupted unpublished shards. No overwrite.
        indices=timed('forward_writing_verification',lambda:extract_model(modelroot/'shards',rows,spec,
            provenance(identity,model,origin),backend,size,plan,output/'quarantine'/model))
        unchanged_or_publish(modelroot/'index.json',dict(identity=identity,model=model,shards=indices,unique_texts=len(rows),logical_bindings=len(metadata.bindings)))
        status='completed'
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc))
        raise
    finally:
        # Preserve measured peaks before teardown, including failed attempts.
        if backend is not None:
            try:write_json(attempt/'peak-memory.json',backend.peak())
            finally:
                try:timed('teardown',backend.close)
                except BaseException as exc:
                    write_json(attempt/'teardown-failure.json',dict(type=type(exc).__name__,message=str(exc)))
                    status='failed'
        if progress['model']:
            progress['intervals'].append(dict(model=model,seconds=time.monotonic()-progress['phase_started']))
        progress.update(model=None,phase_started=time.monotonic());atomic_pointer(attempt/'progress.json',progress)
        write_json(attempt/'resource.json',dict(status=status,error=error,costs_seconds=costs,wall_seconds=time.monotonic()-started,
            producer_sha=access['producer_sha'],access=access,measured=True,attempts_never_reset=True))
        if status!='completed' and error is None:raise RuntimeError('teardown failed; acceptance blocked')


def launch(args, access):
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=True)
    require(not (output/args.model/'index.json').exists(), 'raw model already complete; run audit/accept/verify, never re-extract')
    identity=campaign_identity(access,Metadata());unchanged_or_publish(output/'campaign.json',identity)
    with (output/'campaign.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        attempts=output/'attempts';attempts.mkdir(exist_ok=True)
        carry=read(ROOT/'config/final_e/carryforward_v1.json')
        for name,sha in carry['source_files'].items():require(file_hash(ROOT/name)==sha,'carried attempt evidence')
        seed=dict(total_seconds=sum(carry['charged_seconds'].values()),model_seconds=carry['charged_seconds'],
                  running=False,model=None,as_of_unix=time.time())
        adoption=output/'budget-adoption.json'
        if not adoption.exists():
            require(not any(attempts.iterdir()) and not (output/'full-budget.json').exists(),'no ledger/adoption substitution')
            write_json(adoption,dict(campaign=identity,carryforward=pin(ROOT/'config/final_e/carryforward_v1.json'),initial=seed))
        else:
            previous=read(adoption)
            require(previous['campaign']==identity and previous['carryforward']==pin(ROOT/'config/final_e/carryforward_v1.json'), 'budget adoption changed')
            seed=previous['initial']
        if not (output/'full-budget.json').exists():
            require(not any(attempts.iterdir()),'lost cumulative budget; cannot reset')
            atomic_pointer(output/'full-budget.json',seed)
        current=read(output/'full-budget.json')
        require(all(current['model_seconds'][m]>=carry['charged_seconds'][m] for m in ('qwen','llama')), 'carried charges decreased')
        require((output/'full-budget.json').exists() or not any(attempts.iterdir()), 'missing cumulative ledger; cannot reset')
        if any(attempts.iterdir()):settled_accounting(output,None,access)
        attempt=attempts/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex);attempt.mkdir()
        write_json(attempt/'access.json',access)
        # Exact argv, exact checkout; no shell commands or moving branch.
        cmd=[sys.executable,'-m','src.final_e_runner','extract','--worker','--parent-pid',str(os.getpid()),'--attempt',str(attempt)]
        for key in ('lock','lock_sha256','authorization','authorization_sha256','output','model','cache','volume'):
            cmd.extend(['--'+key.replace('_','-'),str(getattr(args,key))])
        rc=supervise(cmd,output,attempt,read(args.lock)['limits'])
        require(rc==0,'E extraction failed/stopped; preserve attempt and do not retry automatically')


def verify_origin(root, cfg, model):
    origin=read(Path(root)/'runtime-origin.json')
    runtime_matches(origin['runtime'],evidence(cfg,'host-runtime.json'))
    actual=dict(origin['backend']);actual.pop('peak_memory_tracking',None)
    expected=dict(evidence(cfg,model+'/runtime-before-inference.json')['backend']);expected.pop('peak_memory_tracking',None)
    require(actual==expected,'loader runtime/backend contract')
    require(isinstance(origin['snapshot_sha256'],str) and len(origin['snapshot_sha256'])==64,'snapshot identity receipt')
    return origin


def settled_accounting(output, model, access):
    """Check the durable supervisor chain, not merely the mutable running flag."""
    root=Path(output);carry=read(ROOT/'config/final_e/carryforward_v1.json')
    for name,sha in carry['source_files'].items():require(file_hash(ROOT/name)==sha,'carried attempt evidence')
    adoption=read(root/'budget-adoption.json');identity=campaign_identity(access,Metadata())
    require(adoption['campaign']==identity and adoption['carryforward']==pin(ROOT/'config/final_e/carryforward_v1.json'),'accounting adoption identity')
    baseline=adoption['initial']
    require(baseline['model_seconds']==carry['charged_seconds'] and baseline['total_seconds']==sum(carry['charged_seconds'].values()) and baseline['running'] is False and baseline['model'] is None,'carried accounting baseline')
    limits=access['limits'];selected=None
    for attempt in sorted((root/'attempts').iterdir()):
        a=read(attempt/'access.json');exit=read(attempt/'supervisor-exit.json')
        require(all(a[k]==access[k] for k in ('producer_sha','lock_sha256','campaign_id','limits')) and a['phase']=='extract' and a['model'] in ('qwen','llama'),'attempt producer/lock/accounting scope')
        b=exit['budget']
        require(b['running'] is False and b['model'] is None,'unsettled supervisor accounting')
        recover_budget(b,time.time())  # inherited finite/schema/sum checks; no mutation
        require(exit['limits']==limits and exit['budget_reset_allowed'] is False,'attempt resource limits')
        require(b['total_seconds']>=baseline['total_seconds'] and all(b['model_seconds'][m]>=baseline['model_seconds'][m] for m in ('qwen','llama')),'reduced cumulative accounting')
        if (attempt/'resource.json').exists():
            resource=read(attempt/'resource.json')
            require(resource['access']==a and resource['producer_sha']==access['producer_sha'],'worker accounting identity')
            require(np.isfinite(resource['wall_seconds']) and resource['wall_seconds']>=0 and b['total_seconds']-baseline['total_seconds']+1e-6>=resource['wall_seconds'],'worker wall accounting')
        if (attempt/'progress.json').exists():
            progress=read(attempt/'progress.json')
            require(progress['model'] is None or exit['returncode']!=0,'unfinished successful worker progress')
            for m in ('qwen','llama'):
                durations=[i['seconds'] for i in progress['intervals'] if i['model']==m]
                require(all(np.isfinite(v) and v>=0 for v in durations),'nonfinite attempt charge')
                require(b['model_seconds'][m]-baseline['model_seconds'][m]+1e-6>=sum(durations),'reduced measured model charge')
        if a['model']==model:selected=attempt
        baseline=b
    require(selected is not None or model is None,'missing model attempt accounting')
    current=read(root/'full-budget.json')
    require(current==baseline and current['running'] is False,'ledger differs from settled supervisor receipt')
    require(cap_reason(current,limits) is None,'campaign cap exhausted; acceptance blocked')
    if model is None:return dict(adoption=pin(root/'budget-adoption.json'),ledger=pin(root/'full-budget.json'))
    resource=read(selected/'resource.json');exit=read(selected/'supervisor-exit.json')
    require(resource['status']=='completed' and exit['returncode']==0 and not exit['interrupted'] and exit['cap_reason'] is None,'completed successful model attempt required')
    return dict(adoption=pin(root/'budget-adoption.json'),attempt=str(selected.relative_to(root)),
        resource=pin(selected/'resource.json'),supervisor=pin(selected/'supervisor-exit.json'))


def verify(output, model, access, require_acceptance=False):
    metadata=Metadata();identity=campaign_identity(access,metadata);root=Path(output);mroot=root/model
    require(read(root/'campaign.json')==identity,'E campaign identity')
    cfg=raw_contract();verify_origin(mroot,cfg,model);rows=input_rows(metadata);size=read(ROOT/'config/final_e/preparation_v1.json')['shard_rows']
    plan=read(mroot/'token-plan.json');require(len(plan)==len(rows),'token plan coverage')
    names={f'{i:05d}' for i in range((len(rows)+size-1)//size)}
    require({p.name for p in (mroot/'shards').iterdir()}==names,'exact shard coverage')
    receipts=[]
    for index,start in enumerate(range(0,len(rows),size)):
        p=mroot/'shards'/f'{index:05d}';part=rows[start:start+size]
        r=verify_completed(p,part,cfg['models'][model],provenance(identity,model,mroot/'runtime-origin.json'))
        require(read(p/'tokens.json')==plan[start:start+size],'token binding mismatch')
        r['completed_marker']=pin(p/'completed.json');receipts.append(r)
    expected=dict(identity=identity,model=model,shards=receipts,unique_texts=len(rows),logical_bindings=len(metadata.bindings))
    require(read(mroot/'index.json')==expected,'complete ordered index')
    result=dict(schema='final-e-loader-verification-v1',campaign=identity,model=model,status='loader_verified',
        unique_texts=len(rows),logical_bindings=len(metadata.bindings),index=pin(mroot/'index.json'),
        token_plan=pin(mroot/'token-plan.json'),shards=len(receipts))
    if require_acceptance:
        accounting=settled_accounting(root,model,access)
        audit=read(mroot/'token-audit.json');acceptance=read(mroot/'acceptance.json')
        require(audit==dict(campaign=identity,model=model,status='independently_reconstructed',token_plan=result['token_plan'],unique_texts=len(rows),model_loaded=False),'independent token audit')
        require(acceptance==dict(schema='final-e-acceptance-v1',verification=result,audit=pin(mroot/'token-audit.json'),accounting=accounting,status='independently_accepted_E_raw'),'E acceptance')
    return result


def audit(args, access):
    metadata=Metadata();cfg=raw_contract();root=Path(args.output)/args.model
    # Independent tokenizer-only process, no research model load.
    _,tok=offline_tokenizer(cfg,args.model,args.cache)
    packets=[single_packet(tok,r,cfg['execution']['max_tokens']) for r in input_rows(metadata)]
    require(packets==read(root/'token-plan.json'),'independent E token audit failed')
    result=dict(campaign=campaign_identity(access,metadata),model=args.model,status='independently_reconstructed',
        token_plan=pin(root/'token-plan.json'),unique_texts=len(packets),model_loaded=False)
    unchanged_or_publish(root/'token-audit.json',result)
    return result


def accept(args, access):
    root=Path(args.output)/args.model;verification=verify(args.output,args.model,access)
    a=read(root/'token-audit.json')
    require(a['campaign']==verification['campaign'] and a['model']==args.model and a['token_plan']==verification['token_plan'] and
            a['status']=='independently_reconstructed' and a['unique_texts']==len(input_rows(Metadata())) and a['model_loaded'] is False,'independent audit acceptance')
    accounting=settled_accounting(args.output,args.model,access)
    result=dict(schema='final-e-acceptance-v1',status='independently_accepted_E_raw',verification=verification,audit=pin(root/'token-audit.json'),accounting=accounting)
    unchanged_or_publish(root/'acceptance.json',result)
    return result


def layer(output, model, saved_layer, metadata, access):
    # Caller must verify accepted full coverage once immediately before scoring.
    cfg=raw_contract();spec=cfg['models'][model]
    require(type(saved_layer) is int and 0<=saved_layer<spec['layers'],'saved layer range')
    rows=input_rows(metadata);size=read(ROOT/'config/final_e/preparation_v1.json')['shard_rows'];values=[]
    identity=campaign_identity(access,metadata);root=Path(output)/model
    for i,start in enumerate(range(0,len(rows),size)):
        a,_,_=load_features(root/'shards'/f'{i:05d}',expected_rows=rows[start:start+size],expected_spec=spec,
            expected_provenance=provenance(identity,model,root/'runtime-origin.json'))
        values.append(np.array(a[:,saved_layer,:],dtype=np.float64))
    X=np.concatenate(values)
    return {g:X[metadata.indices(g)] for g in metadata.rows}
