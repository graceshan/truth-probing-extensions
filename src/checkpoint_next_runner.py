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
                integration_config_sha256=sha(ROOT/'config/checkpoint_next/integration_fix_v1.json'),
                paths='full-sequence-likelihood/greedy/last-input-chat-v2')


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
    from src.checkpoint_next_missing import settings, SETTINGS
    settings()
    require(a['integration_config_sha256']==sha(SETTINGS),'successor contract mismatch')
    if phase!='missing_raw':
        acceptance=Path(args.output)/'missing_raw'/args.model/'acceptance.json'
        require(sha(acceptance)==a['missing_raw_acceptance_sha256'],'missing supplement acceptance binding')
        require(json.loads(acceptance.read_text())==missing_acceptance(args.output,args.model,x),
                'missing supplement/ledger acceptance changed')
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


def missing_acceptance(output, model, x):
    from src.checkpoint_next_missing import verify_supplement
    receipt=verify_supplement(output,model,identity(model,x))
    root=Path(output)/'missing_raw'/model
    completion=json.loads((root/'completion.json').read_text())
    require(completion['status']=='completed' and completion['verification']==receipt,'supplement incomplete')
    verify_execution_evidence(output,model,completion)
    audit=json.loads((root/'token-audit.json').read_text())
    from src.checkpoint_r2_full_support import evidence
    expected=evidence(config(),model+'/runtime-before-inference.json')['backend']
    require(audit['identity']==identity(model,x) and audit['token_plan_sha256']==receipt['token_plan_sha256']
        and audit['queue_sha256']==receipt['queue_sha256'] and audit['status']=='independently_reconstructed'
        and audit['rows']==131 and audit['model_loaded'] is False
        and all(audit[k]==expected[k] for k in ('tokenizer_backend_sha256','tokenizer_implementation_sha256')),
        'independent token audit missing/mismatch')
    receipt['token_audit_sha256']=sha(root/'token-audit.json')
    ledger=[]
    for p in sorted((Path(output)/'attempts'/model).glob('*/start.json')):
        if json.loads(p.read_text())['phase']!='missing_raw':continue
        end=p.with_name('end.json');require(end.exists(),'missing raw attempt unsettled')
        value=json.loads(end.read_text());require(value['active_seconds']>=0,'invalid raw charge')
        ledger.append(dict(attempt=p.parent.name,start_sha256=sha(p),end_sha256=sha(end),**value))
    require(ledger,'missing extraction ledger')
    return dict(verification=receipt,completion_sha256=sha(root/'completion.json'),ledger=ledger,
                active_seconds=sum(r['active_seconds'] for r in ledger),status='accepted')


def worker(args, attempt, phase, approval):
    """Child only: accepted offline runtime, shared cap, durable failure evidence."""
    from src.checkpoint_r2_fresh_backend import runtime_receipt
    from src.checkpoint_r2_full_raw import runtime_matches
    from src.checkpoint_r2_full_support import evidence, verify_snapshot
    from src.checkpoint_next_measurement import ChatBackend
    from src.checkpoint_next_missing import settings, extract
    c=config();x=frozen();spec=c['models'][args.model];policy=settings()
    require(platform.system()=='Linux' and os.getppid()==args.parent_pid,'supervised Linux worker required')
    import ctypes
    require(ctypes.CDLL(None).prctl(1,signal.SIGKILL,0,0,0)==0 and os.getppid()==args.parent_pid,
            'parent-death watchdog required')
    began=time.monotonic();backend=None;summary=None
    costs={k:0. for k in ('loading','tokenization','likelihood','generation','chat_extraction',
                         'raw_extraction','preflight_checks','writing_verification','teardown')}
    try:
        runtime=runtime_receipt(c['runtime_lock'])
        expected=evidence(c,args.model+'/runtime-before-inference.json')
        runtime_matches(runtime,evidence(c,'host-runtime.json'))
        snapshot=verify_snapshot(evidence(c,args.model+'-snapshot.json'),spec,args.cache)
        # Loading and every later stage consume the same active model allowance.
        publish_json(attempt/'active.json',dict(utc=time.time(),monotonic=time.monotonic()))
        t=time.monotonic()
        try:backend=ChatBackend(spec,snapshot,c['execution'],args.model,c['layers'][args.model])
        finally:costs['loading']=time.monotonic()-t
        require(all(backend.raw.details[k]==expected['backend'][k] for k in
            ('tokenizer_backend_sha256','model_implementation_sha256','tokenizer_implementation_sha256','actual_flags')),
            'accepted implementations/flags differ')
        publish_json(attempt/'runtime.json',dict(runtime=runtime,backend=backend.raw.details,snapshot=snapshot))
        idn=identity(args.model,x)
        if phase=='missing_raw':
            summary=extract(args,backend,idn,costs)
        else:
            rows=selected_jobs(x,phase);root=Path(args.output)/phase/args.model;root.mkdir(parents=True,exist_ok=True)
            same_or_publish(root/'identity.json',idn)
            tokens=[];t=time.monotonic()
            for j in x['jobs']:tokens.append(dict(id=j['id'],packet=backend.packet(j)))
            costs['tokenization']+=time.monotonic()-t
            t=time.monotonic();same_or_publish(root/'token-plan.json',tokens);costs['writing_verification']+=time.monotonic()-t
            token_by_id={r['id']:r['packet'] for r in tokens}
            names={f'{i//c["shard_rows"]:05d}' for i in range(0,len(rows),c['shard_rows'])}
            shards=root/'shards';shards.mkdir(exist_ok=True)
            require(all(p.name in names for p in shards.iterdir()),'unexpected extra shard')
            timings=[]
            for start in range(0,len(rows),c['shard_rows']):
                jobs=rows[start:start+c['shard_rows']];path=shards/f'{start//c["shard_rows"]:05d}';t=time.monotonic()
                reused=recover(path,jobs,idn,c['layers'][args.model],spec['width'],root/'quarantine')
                costs['writing_verification']+=time.monotonic()-t
                if reused:
                    timings.extend(r['seconds'] for r in json.loads((path/'records.json').read_text()));continue
                records,chat=[],[]
                for j in jobs:
                    t=time.monotonic();packet=token_by_id[j['id']] if phase=='launch' else backend.packet(j)
                    times=dict(tokenization=time.monotonic()-t);costs['tokenization']+=times['tokenization']
                    r=dict(id=j['id'],job_sha256=digest(j),packet=packet)
                    if j['behavior']:
                        t=time.monotonic();r['likelihood']=backend.score(packet);times['likelihood']=time.monotonic()-t
                        costs['likelihood']+=times['likelihood']
                        t=time.monotonic();r['generation']=backend.generate(packet);times['generation']=time.monotonic()-t
                        costs['generation']+=times['generation'];times['generated_tokens']=len(r['generation']['token_ids'])
                        if phase=='preflight':
                            t=time.monotonic()
                            r['independent_preflight']=backend.preflight_checks(packet,r['likelihood'],r['generation'],
                                policy['causal_prefix_logprob_absolute_tolerance'])
                            require(r['likelihood']==backend.score(packet) and r['generation']==backend.generate(packet),'behavior repeat mismatch')
                            costs['preflight_checks']+=time.monotonic()-t
                    if j['chat']:
                        t=time.monotonic();state=backend.chat(packet);times['chat']=time.monotonic()-t
                        costs['chat_extraction']+=times['chat']
                        if phase=='preflight':
                            import numpy as np
                            t=time.monotonic();require(np.array_equal(state,backend.chat(packet)),'chat repeat mismatch')
                            costs['preflight_checks']+=time.monotonic()-t
                        r.update(chat_index=len(chat),saved_layers=c['layers'][args.model]);chat.append(state)
                    r['seconds']=times;records.append(r);timings.append(times)
                t=time.monotonic();save_shard(path,jobs,records,chat,idn,c['layers'][args.model],spec['width'])
                costs['writing_verification']+=time.monotonic()-t
            summary=dict(identity=idn,phase=phase,status='completed',jobs=len(rows),token_plan_sha256=sha(root/'token-plan.json'))
            if phase=='preflight':
                counts={k:sum(j['behavior'] if k in ('likelihood','generation') else j['chat'] for j in x['jobs']) for k in ('likelihood','generation','chat')}
                maxima={k:max(t[k] for t in timings if k in t) for k in counts}
                # Charge eight times a full observed generation, even if it ended at token one.
                # Later positions can still cost more; this remains a reviewed forecast.
                maxima['generation']*=8
                forecast=costs['loading']+costs['tokenization']+sum(counts[k]*maxima[k] for k in counts)
                training_max=max(len(backend.packet(j)['prompt_ids']) for j in rows)
                workload_max=max(len(r['packet']['prompt_ids']) for r in tokens)
                payload=x['accounting']['fp16_chat_payload_bytes'][args.model]
                copy=payload+sum(len(json.dumps(r).encode()) for r in tokens)*4+c['storage']['metadata_planning_reserve_per_copy_bytes']
                needed=2*copy+c['shard_rows']*2*spec['width']*2+c['storage']['safety_margin_bytes']
                free=os.statvfs(args.output).f_bavail*os.statvfs(args.output).f_frsize
                # Parent settles current charge after exit; launch rechecks actual remaining budget.
                summary.update(new_paths_measured=True,path_counts=counts,forecast_gpu_seconds=forecast,max_training_seconds=maxima,
                    forecast_method='max path costs; generation observed full-call maximum times eight-token cap; forecast only',
                    unmeasured_allowances=['future writing/verification','teardown','longer sequence cost'],
                    storage_reserved_bytes=needed,statvfs_free_bytes=free,quota_guarantee=False,
                    training_max_prompt_tokens=training_max,workload_max_prompt_tokens=workload_max,
                    training_length_coverage=training_max>=workload_max,
                    fits_proposed_caps=training_max>=workload_max and forecast<7200,
                    independent_checks_required=True,observed_statvfs_fits=needed<=free,requires_review_before_launch=True)
    finally:
        # Capture before teardown; reviewed loader reset is before from_pretrained.
        peak=None
        try:
            if backend is not None:
                peak=backend.raw.peak()
                publish_json(attempt/'peak-before-teardown.json',dict(peak_cuda_bytes=peak,scope='loading and execution'))
                t=time.monotonic()
                try:backend.close()
                finally:costs['teardown']=time.monotonic()-t
            elif (attempt/'active.json').exists():
                import torch
                peak=dict(allocated=torch.cuda.max_memory_allocated(),reserved=torch.cuda.max_memory_reserved())
        finally:
            wall=time.monotonic()-began
            publish_json(attempt/'execution-evidence.json',dict(status='completed' if summary and sys.exc_info()[0] is None else 'failed',
                measured_seconds=costs,worker_wall_seconds=wall,unattributed_seconds=wall-sum(costs.values()),
                peak_cuda_bytes=peak,peak_scope='model loading and execution before teardown',
                accounting='disjoint path timers; wall includes setup; reused shards excluded from current execution costs'))
    summary['execution_evidence_sha256']=sha(attempt/'execution-evidence.json')
    summary['attempt']=attempt.name
    same_or_publish(Path(args.output)/phase/args.model/'completion.json',summary)


def supervise(args, phase):
    c=config();x=frozen();a=authorized(args,phase,x)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    with (out/(args.model+'.lock')).open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        same_or_publish(out/(args.model+'-campaign.json'),identity(args.model,x))
        if (out/phase/args.model/'completion.json').exists():
            print(json.dumps(verify(out,phase,args.model),indent=2))
            return
        charged=attempts_elapsed(out,args.model)
        remaining=c['caps']['active_gpu_seconds_per_model']-charged
        require(remaining>0,'cumulative model cap exhausted; no automatic cap reset')
        if phase=='launch':
            p=out/'preflight'/args.model/'completion.json';pre=json.loads(p.read_text())
            require(a['preflight_sha256']==sha(p) and pre['identity']==identity(args.model,x) and
                    pre['fits_proposed_caps'] and pre['forecast_gpu_seconds']+a['unmeasured_allowance_seconds']<remaining,'preflight forecast/cap review required')
            require(isinstance(a['unmeasured_allowance_seconds'],(int,float)) and a['unmeasured_allowance_seconds']>=0,
                    'explicit reviewed unmeasured allowance required')
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


def verify_execution_evidence(output,model,completion):
    require(Path(completion['attempt']).name==completion['attempt'],'attempt path')
    path=Path(output)/'attempts'/model/completion['attempt']/'execution-evidence.json'
    require(sha(path)==completion['execution_evidence_sha256'],'execution evidence hash')
    evidence=json.loads(path.read_text())
    require(evidence['status']=='completed' and all(v>=0 for v in evidence['measured_seconds'].values()),'execution timings')
    peak=json.loads(path.with_name('peak-before-teardown.json').read_text())
    require(peak['peak_cuda_bytes']==evidence['peak_cuda_bytes'] and
            all(v>=0 for v in evidence['peak_cuda_bytes'].values()),'peak before teardown')
    return evidence


def verify(output, phase, model):
    if phase=='missing_raw':return missing_acceptance(output,model,frozen())
    c=config();x=frozen();root=Path(output)/phase/model;idn=json.loads((root/'identity.json').read_text())
    require(idn==identity(model,x),'transferred producer identity mismatch')
    rows=selected_jobs(x,phase);names=set()
    for start in range(0,len(rows),c['shard_rows']):
        name=f'{start//c["shard_rows"]:05d}';names.add(name)
        verify_shard(root/'shards'/name,rows[start:start+c['shard_rows']],idn,c['layers'][model],c['models'][model]['width'])
    require({p.name for p in (root/'shards').iterdir()}==names,'missing/extra coverage')
    completion=json.loads((root/'completion.json').read_text())
    verify_execution_evidence(output,model,completion)
    require(completion['identity']==idn and completion['status']=='completed' and completion['jobs']==len(rows),'completion coverage')
    require(sha(root/'token-plan.json')==completion['token_plan_sha256'],'token plan corrupt')
    tokens=json.loads((root/'token-plan.json').read_text())
    require([r['id'] for r in tokens]==[j['id'] for j in x['jobs']],'full token plan coverage/order')
    if phase=='preflight':
        for path in sorted((root/'shards').glob('*/records.json')):
            for r in json.loads(path.read_text()):
                if 'likelihood' in r:
                    require(r['independent_preflight']['independent_checks_passed'] is True and
                        r['independent_preflight']['manual_greedy_ids']==r['generation']['token_ids'],
                        'independent preflight evidence missing')
    if phase=='launch':
        from src.checkpoint_next_missing import verify_supplement
        require(json.loads((Path(output)/'missing_raw'/model/'acceptance.json').read_text())==
                missing_acceptance(output,model,x),'launch supplement acceptance/ledger changed')
        lookup={r['id']:r['packet'] for r in tokens}
        for path in sorted((root/'shards').glob('*/records.json')):
            require(all(r['packet']==lookup[r['id']] for r in json.loads(path.read_text())), 'scored/extracted packet differs from token preflight')
    return dict(status='verified',phase=phase,model=model,shards=len(names),jobs=len(rows),identity=idn)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','verify','missing-raw','extract-missing','audit-missing-tokens','accept-missing','preflight','launch','_worker'])
    p.add_argument('--output',required=True);p.add_argument('--model',choices=['qwen','llama']);p.add_argument('--phase',choices=['preflight','launch','missing_raw'],default='launch')
    p.add_argument('--cache');p.add_argument('--authorization');p.add_argument('--attempt');p.add_argument('--parent-pid',type=int);p.add_argument('--required');p.add_argument('--available');p.add_argument('--representation-sha256')
    args=p.parse_args()
    if args.action=='prepare':
        print(json.dumps(prepare(args.output)['accounting'],indent=2))
    elif args.action=='verify':
        print(json.dumps(verify(args.output,args.phase,args.model),indent=2))
    elif args.action=='missing-raw':
        from src.checkpoint_next_missing import metadata
        publish_json(args.output,metadata())
    elif args.action=='audit-missing-tokens':
        from src.checkpoint_next_missing import audit_tokens
        require(args.cache and args.model,'offline verified cache/model required')
        with (Path(args.output)/(args.model+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            receipt=audit_tokens(args.output,args.model,identity(args.model,frozen()),args.cache)
            same_or_publish(Path(args.output)/'missing_raw'/args.model/'token-audit.json',receipt)
    elif args.action=='accept-missing':
        with (Path(args.output)/(args.model+'.lock')).open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            attempts_elapsed(args.output,args.model)
            receipt=missing_acceptance(args.output,args.model,frozen())
            same_or_publish(Path(args.output)/'missing_raw'/args.model/'acceptance.json',receipt)
    else:
        require(args.authorization and args.cache and args.model,'exact reviewed authorization/cache/model required')
        if args.action=='_worker':
            a=authorized(args,args.phase,frozen());worker(args,Path(args.attempt),args.phase,a)
        else:supervise(args,'missing_raw' if args.action=='extract-missing' else args.action)


if __name__=='__main__':
    main()
