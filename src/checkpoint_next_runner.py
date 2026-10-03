"""Preparation CLI and separately authorized bounded offline future execution.

Nothing contacts a Pod. prepare/verify/missing-raw never load models. Preflight and
launch require an exact-commit authorization; this preparation has no such receipt.
"""
import argparse
import fcntl
import gzip
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path
from src.checkpoint_next_inventory import CONFIG, ROOT, config, digest, inventory, require, sha, missing_raw_interface
from src.checkpoint_next_store import publish_json, same_or_publish, recover, save_shard, verify_shard

PACKAGE=ROOT/'data/checkpoint_next_behavior_prep_20261003'


def producer():
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def frozen():
    x=inventory()
    require(json.loads(gzip.decompress((PACKAGE/'inventory.json.gz').read_bytes()))==x,'frozen prepared inventory changed')
    return x


def identity(model, x):
    return dict(producer_commit=producer(),config_sha256=sha(CONFIG),inventory_sha256=digest(x),model=model,
                paths='full-sequence-likelihood/greedy/last-input-chat-v1')


def prepare(out):
    x=inventory();out=Path(out);out.mkdir(parents=True,exist_ok=True)
    same_or_publish(out/'accounting.json',x['accounting'])
    same_or_publish(out/'heads.json',x['heads'])
    same_or_publish(out/'preflight-inputs.json',x['preflight'])
    same_or_publish(out/'inventory-hashes.json',{k:digest(v) for k,v in x.items()})
    return x


def authorized(args, phase, x):
    c=config();a=json.loads(Path(args.authorization).read_text())
    require(a['phase']==phase and a['producer_commit']==producer() and a['config_sha256']==sha(CONFIG)
            and a['inventory_sha256']==digest(x) and a['model']==args.model,'reviewed authorization mismatch')
    require(len(a['producer_commit'])==40 and a['reviewed'] is True and a['E'] is False,'reviewed exact scope required')
    require(subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True)=='','dirty producer')
    prior=a['missing_raw_active_seconds']
    require(isinstance(prior,(int,float)) and 0<=prior<c['caps']['active_gpu_seconds_per_model']
            and a['missing_raw_accounting_sha256'],'missing-raw budget/coverage receipt required')
    ledger_path=Path(a['missing_raw_accounting_path'])
    require(sha(ledger_path)==a['missing_raw_accounting_sha256'],'missing-raw ledger changed')
    ledger=json.loads(ledger_path.read_text())
    require(ledger['model']==args.model and ledger['active_gpu_seconds']==prior and
            ledger['status'] in ('completed','no_missing_inputs_verified'), 'missing-raw coverage/budget unresolved')
    return a


def selected_jobs(x, phase):
    return x['preflight'] if phase=='preflight' else x['jobs']


def attempts_elapsed(output, model):
    total=0
    for p in sorted((Path(output)/'attempts'/model).glob('*/start.json')):
        start=json.loads(p.read_text());end=p.with_name('end.json')
        if end.exists():
            charge=json.loads(end.read_text())['active_seconds']
            require(isinstance(charge,(int,float)) and charge>=0,'invalid saved budget')
            total+=charge
        else:
            active=p.with_name('active.json')
            # Conservative recovery: interrupted active worker is charged wall time
            # since its start until this recovery. Downtime never replenishes caps.
            if active.exists():
                charge=time.time()-json.loads(active.read_text())['utc']
                require(charge>=0,'clock reversal during recovery; accounting review required')
                same_or_publish(end,dict(active_seconds=charge,status='interruption_conservative_charge'))
                total+=charge
            else:
                same_or_publish(end,dict(active_seconds=0,status='interruption_during_setup'))
    return total


def worker(args, attempt, phase, approval):
    """Child only: exact offline runtime/model validation then new bounded paths."""
    from src.checkpoint_r2_fresh_backend import runtime_receipt
    from src.checkpoint_r2_full_raw import runtime_matches
    from src.checkpoint_r2_full_support import evidence, verify_snapshot
    from src.checkpoint_next_measurement import ChatBackend
    from src.checkpoint_r2_fresh_inputs import text_hash
    c=config();x=frozen();spec=c['models'][args.model]
    require(platform.system()=='Linux' and os.getppid()==args.parent_pid,'supervised Linux worker required')
    import ctypes
    require(ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)==0 and os.getppid()==args.parent_pid,
            'parent-death watchdog required')
    runtime=runtime_receipt(c['runtime_lock'])
    # Accepted runtime evidence, not the obsolete original protocol candidate pins.
    expected=evidence(c,args.model+'/runtime-before-inference.json')
    runtime_matches(runtime,evidence(c,'host-runtime.json'))
    snapshot=verify_snapshot(evidence(c,args.model+'-snapshot.json'),spec,args.cache)
    backend=ChatBackend(spec,snapshot,c['execution'],args.model,c['layers'][args.model])
    require(backend.raw.details['tokenizer_backend_sha256']==expected['backend']['tokenizer_backend_sha256']
            and backend.raw.details['model_implementation_sha256']==expected['backend']['model_implementation_sha256']
            and backend.raw.details['tokenizer_implementation_sha256']==expected['backend']['tokenizer_implementation_sha256']
            and backend.raw.details['actual_flags']==expected['backend']['actual_flags'], 'accepted implementations/flags differ')
    publish_json(attempt/'runtime.json',dict(runtime=runtime,backend=backend.raw.details,snapshot=snapshot))
    publish_json(attempt/'active.json',dict(utc=time.time(),monotonic=time.monotonic()))
    rows=selected_jobs(x,phase);idn=identity(args.model,x)
    root=Path(args.output)/phase/args.model;root.mkdir(parents=True,exist_ok=True)
    same_or_publish(root/'identity.json',idn)
    timings=[]
    try:
        # Tokenize entire bounded planned workload before research inference. This
        # establishes exact token IDs/boundaries and metadata/storage size.
        tokens=[];t=time.monotonic()
        for j in x['jobs']:
            tokens.append(dict(id=j['id'],packet=backend.packet(j)))
        token_seconds=time.monotonic()-t
        same_or_publish(root/'token-plan.json',tokens)
        token_by_id={r['id']:r['packet'] for r in tokens}
        names={f'{i//c["shard_rows"]:05d}' for i in range(0,len(rows),c['shard_rows'])}
        shards=root/'shards';shards.mkdir(exist_ok=True)
        require(all(p.name in names for p in shards.iterdir()),'unexpected extra shard')
        for start in range(0,len(rows),c['shard_rows']):
            jobs=rows[start:start+c['shard_rows']];path=shards/f'{start//c["shard_rows"]:05d}'
            if recover(path,jobs,idn,c['layers'][args.model],spec['width'],root/'quarantine'):
                old=json.loads((path/'records.json').read_text());timings.extend(r['seconds'] for r in old)
                continue
            records,chat=[],[]
            for j in jobs:
                t=time.monotonic();packet=token_by_id[j['id']] if phase=='launch' else backend.packet(j)
                times=dict(tokenization=time.monotonic()-t)
                r=dict(id=j['id'],job_sha256=digest(j),packet=packet)
                if j['behavior']:
                    t=time.monotonic();r['likelihood']=backend.score(packet);times['likelihood']=time.monotonic()-t
                    t=time.monotonic();r['generation']=backend.generate(packet);times['generation']=time.monotonic()-t
                    if phase=='preflight':
                        t=time.monotonic()
                        require(r['likelihood']==backend.score(packet) and r['generation']==backend.generate(packet), 'behavior training repeat mismatch')
                        times['behavior_repeat_checks']=time.monotonic()-t
                if j['chat']:
                    t=time.monotonic();state=backend.chat(packet);times['chat']=time.monotonic()-t
                    if phase=='preflight':
                        # New-path repeat check is prospective; old raw success is insufficient.
                        import numpy as np
                        require(np.array_equal(state,backend.chat(packet)), 'chat training repeat mismatch')
                        times['repeat_checks']=time.monotonic()-t-times['chat']
                    r.update(chat_index=len(chat),saved_layers=c['layers'][args.model]);chat.append(state)
                r['seconds']=times;records.append(r);timings.append(times)
            save_shard(path,jobs,records,chat,idn,c['layers'][args.model],spec['width'])
        summary=dict(identity=idn,phase=phase,status='completed',jobs=len(rows),
                     new_paths_measured=phase=='preflight',tokenization_full_workload_seconds=token_seconds,
                     token_plan_sha256=sha(root/'token-plan.json'))
        if phase=='preflight':
            # Conservative maxima extrapolation, explicitly a forecast rather than a guarantee.
            path_counts={k:sum(j['behavior'] if k in ('likelihood','generation') else j['chat'] for j in x['jobs']) for k in ('likelihood','generation','chat')}
            maxima={k:max(t[k] for t in timings if k in t) for k in path_counts}
            forecast=token_seconds+sum(path_counts[k]*maxima[k] for k in path_counts)
            training_max=max(len(backend.packet(j)['prompt_ids']) for j in rows)
            workload_max=max(len(r['packet']['prompt_ids']) for r in tokens)
            # Exact tensor bytes plus measured token-plan metadata and explicit reserves;
            # two copies, temporary largest shard, safety, retained cache/raw from quota review.
            payload=x['accounting']['fp16_chat_payload_bytes'][args.model]
            record_reserve=sum(len(json.dumps(r).encode()) for r in tokens)*4
            copy=payload+record_reserve+c['storage']['metadata_planning_reserve_per_copy_bytes']
            needed=2*copy+c['shard_rows']*2*spec['width']*2+c['storage']['safety_margin_bytes']
            free=os.statvfs(args.output).f_bavail*os.statvfs(args.output).f_frsize
            summary.update(path_counts=path_counts,max_training_seconds=maxima,forecast_gpu_seconds=forecast,
                forecast_method='max fixed training sample cost per path times exact input count; forecast, not guarantee',
                storage_reserved_bytes=needed,statvfs_free_bytes=free,quota_guarantee=False,
                training_max_prompt_tokens=training_max,workload_max_prompt_tokens=workload_max,
                training_length_coverage=training_max>=workload_max,
                fits_proposed_caps=training_max>=workload_max and forecast+approval['missing_raw_active_seconds']<c['caps']['active_gpu_seconds_per_model'],
                observed_statvfs_fits=needed<=free,requires_review_before_launch=True)
        same_or_publish(root/'completion.json',summary)
    finally:
        backend.close()


def supervise(args, phase):
    c=config();x=frozen();a=authorized(args,phase,x)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/(args.model+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        same_or_publish(out/(args.model+'-campaign.json'),dict(identity(args.model,x),
            missing_raw_accounting_sha256=a['missing_raw_accounting_sha256'],missing_raw_active_seconds=a['missing_raw_active_seconds']))
        if (out/phase/args.model/'completion.json').exists():
            print(json.dumps(verify(out,phase,args.model),indent=2))
            return
        charged=attempts_elapsed(out,args.model)+a['missing_raw_active_seconds']
        remaining=c['caps']['active_gpu_seconds_per_model']-charged
        require(remaining>0,'cumulative model cap exhausted; no automatic cap reset')
        if phase=='launch':
            p=out/'preflight'/args.model/'completion.json';pre=json.loads(p.read_text())
            require(a['preflight_sha256']==sha(p) and pre['identity']==identity(args.model,x) and
                    pre['fits_proposed_caps'] and pre['forecast_gpu_seconds']<remaining,'preflight forecast/cap review required')
            require(a['storage_reviewed'] is True and a['storage_receipt_sha256'] and a['forecast_reviewed'] is True,'quota/cap review missing')
            # Reviewed writer allocation and second-copy reservation must be performed
            # at first use; statvfs from benchmark alone is never a quota proof.
            receipt=json.loads(Path(a['storage_receipt_path']).read_text())
            require(sha(a['storage_receipt_path'])==a['storage_receipt_sha256'] and receipt['identity']==identity(args.model,x)
                    and receipt['reserved_bytes']>=pre['storage_reserved_bytes'] and receipt['allocation_verified'] is True,'verified storage reservation missing')
            require(receipt['device']==out.stat().st_dev and 0<=time.time()-receipt['utc']<=86400,
                    'storage mount/restart or stale reservation; rerun storage check')
        from uuid import uuid4
        attempt=out/'attempts'/args.model/uuid4().hex;attempt.mkdir(parents=True)
        publish_json(attempt/'start.json',dict(utc=time.time(),remaining_gpu_seconds=remaining,phase=phase))
        command=[sys.executable,'-m','src.checkpoint_next_runner','_worker','--phase',phase,'--model',args.model,
                 '--output',str(out),'--cache',str(Path(args.cache).resolve()),'--authorization',str(Path(args.authorization).resolve()),'--attempt',str(attempt),'--parent-pid',str(os.getpid())]
        start=time.monotonic();active_start=None;reason=None
        with (attempt/'stdout.log').open('xb') as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            def interrupted(signum, frame):
                raise KeyboardInterrupt('supervisor stop requested')
            old_term=signal.signal(signal.SIGTERM,interrupted)
            try:
                while proc.poll() is None:
                    if active_start is None and (attempt/'active.json').exists():
                        active_start=json.loads((attempt/'active.json').read_text())['monotonic']
                    if active_start is None and time.monotonic()-start>=c['caps']['setup_seconds_per_phase']-1:
                        reason='setup cap'
                    if active_start is not None and time.monotonic()-active_start>=remaining-1:
                        reason='active cumulative GPU cap'
                    if reason:
                        os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        break
                    time.sleep(.1)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                if active_start is None and (attempt/'active.json').exists():
                    active_start=json.loads((attempt/'active.json').read_text())['monotonic']
                duration=0 if active_start is None else time.monotonic()-active_start
                publish_json(attempt/'end.json',dict(active_seconds=duration,status=reason or 'finished',exit_code=proc.returncode))
                signal.signal(signal.SIGTERM,old_term)
        require(not reason and proc.returncode==0,'affected task stopped; inspect attempt logs; no cap/tolerance relaxation')


def verify(output, phase, model):
    c=config();x=frozen();root=Path(output)/phase/model;idn=json.loads((root/'identity.json').read_text())
    require(idn==identity(model,x),'transferred producer identity mismatch')
    rows=selected_jobs(x,phase);names=set()
    for start in range(0,len(rows),c['shard_rows']):
        name=f'{start//c["shard_rows"]:05d}';names.add(name)
        verify_shard(root/'shards'/name,rows[start:start+c['shard_rows']],idn,c['layers'][model],c['models'][model]['width'])
    require({p.name for p in (root/'shards').iterdir()}==names,'missing/extra coverage')
    completion=json.loads((root/'completion.json').read_text())
    require(completion['identity']==idn and completion['status']=='completed' and completion['jobs']==len(rows),'completion coverage')
    require(sha(root/'token-plan.json')==completion['token_plan_sha256'],'token plan corrupt')
    tokens=json.loads((root/'token-plan.json').read_text())
    require([r['id'] for r in tokens]==[j['id'] for j in x['jobs']],'full token plan coverage/order')
    if phase=='launch':
        lookup={r['id']:r['packet'] for r in tokens}
        for path in sorted((root/'shards').glob('*/records.json')):
            require(all(r['packet']==lookup[r['id']] for r in json.loads(path.read_text())), 'scored/extracted packet differs from token preflight')
    return dict(status='verified',phase=phase,model=model,shards=len(names),jobs=len(rows),identity=idn)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','verify','missing-raw','preflight','launch','_worker'])
    p.add_argument('--output',required=True);p.add_argument('--model',choices=['qwen','llama']);p.add_argument('--phase',choices=['preflight','launch'],default='launch')
    p.add_argument('--cache');p.add_argument('--authorization');p.add_argument('--attempt');p.add_argument('--parent-pid',type=int);p.add_argument('--required');p.add_argument('--available');p.add_argument('--representation-sha256')
    args=p.parse_args()
    if args.action=='prepare':
        print(json.dumps(prepare(args.output)['accounting'],indent=2))
    elif args.action=='verify':
        print(json.dumps(verify(args.output,args.phase,args.model),indent=2))
    elif args.action=='missing-raw':
        publish_json(args.output,missing_raw_interface(json.loads(Path(args.required).read_text()),json.loads(Path(args.available).read_text()),args.representation_sha256))
    else:
        require(args.authorization and args.cache and args.model,'exact reviewed authorization/cache/model required')
        if args.action=='_worker':
            a=authorized(args,args.phase,frozen());worker(args,Path(args.attempt),args.phase,a)
        else:supervise(args,args.action)


if __name__=='__main__':
    main()
