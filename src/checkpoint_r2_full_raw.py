"""Complete fresh raw feature preparation/execution; no fitting, E, behavior or chat route."""
import argparse
import ctypes
import fcntl
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
from src.checkpoint_r2_fresh_inputs import ROOT, canonical, file_hash, hash_value, require
from src.checkpoint_r2_fresh_store import fsync_dir, load_features, pin, write_json, write_shard
from src.checkpoint_r2_fresh_pilot import atomic_pointer, semantic_view
from src.checkpoint_r2_full_inputs import full_manifest
from src.checkpoint_r2_full_support import (additional_capacity, check_review, check_storage_receipt, evidence,
    storage_check, verify_cached_models, verify_handoff)

CONFIG = 'config/checkpoint_r2/full_raw_v1.json'
CONFIG_SHA256 = 'ed15c662289d587cc9328bc43d980b42afe3d39e4dc352e334e0f504ebcd4ec1'


def plan():
    require(file_hash(ROOT/CONFIG)==CONFIG_SHA256, 'full contract hash mismatch')
    cfg=json.loads((ROOT/CONFIG).read_text())
    require(cfg['batch_size']==1 and cfg['padding']=='none' and 0<cfg['shard_rows']<=320, 'validated batch/shard policy')
    for path,expected in {**cfg['evidence_sha256'],**cfg['preserved_sha256'],**cfg['review_sha256']}.items():
        require(file_hash(ROOT/path)==expected, 'pilot evidence changed: '+path)
    check_review(cfg)
    manifest=full_manifest(cfg)
    require(hash_value(manifest)==cfg['manifest_sha256'], 'complete inventory manifest mismatch')
    return cfg,manifest


def unchanged_or_publish(path,value):
    path=Path(path)
    if path.exists():require(json.loads(path.read_text())==value, 'immutable metadata mismatch: '+str(path))
    else:write_json(path,value)


def shard_provenance(campaign, model, runtime_pin):
    return dict(commit=campaign['commit'],contract_sha256=campaign['contract_sha256'],manifest_sha256=campaign['manifest_sha256'],
        model=model,batch_size=1,padding='none',purpose='full-fresh-raw',runtime_receipt_sha256=runtime_pin)


def verify_completed(path, rows, spec, provenance):
    values,restored,receipt=load_features(path,expected_rows=rows,expected_spec=spec,expected_provenance=provenance)
    packet=json.loads((path/'tokens.json').read_text())
    semantic_view(packet,rows)
    require(all(all(p['attention_mask']) and p['token_ids']==p['unpadded_token_ids'] for p in packet), 'full features must be unpadded batch1')
    marker=path/'completed.json'
    if marker.exists():
        commit=json.loads(marker.read_text())
        require(commit['schema']=='r2-full-raw-shard-commit-v1' and commit['receipt']==pin(path/'receipt.json')
                and commit['ordered_rows_sha256']==hash_value(rows), 'completed shard corruption')
    return dict(path=path.name,receipt=pin(path/'receipt.json'),rows=len(restored),
        bindings=sum(len(r['bindings']) for r in restored),ordered_ids=[r['id'] for r in restored],tokens_sha256=hash_value(packet))


def recover_shard(path, rows, spec, provenance, quarantine):
    if not path.exists():return None
    require(path.is_dir() and not path.is_symlink(), 'unexpected shard path')
    # A published receipt is authoritative even if interrupted before completion marker.
    if (path/'receipt.json').exists():
        verified=verify_completed(path,rows,spec,provenance)  # corruption always stops, never overwrites
        if not (path/'completed.json').exists():
            write_json(path/'completed.json',dict(schema='r2-full-raw-shard-commit-v1',receipt=verified['receipt'],
                ordered_rows_sha256=hash_value(rows),adopted_after_interruption=True,timing_seconds=None))
        return verified
    require(not (path/'completed.json').exists(), 'completed marker without published receipt')
    quarantine.mkdir(parents=True,exist_ok=True)
    target=quarantine/(path.name+'-'+uuid.uuid4().hex)
    require(not target.exists(), 'quarantine collision')
    os.rename(path,target);fsync_dir(path.parent);fsync_dir(quarantine)
    return None


def extract_model(destination, rows, spec, provenance, backend, shard_rows, token_plan, quarantine):
    completed=[]
    destination.mkdir(parents=True,exist_ok=True)
    expected_names={f'{i:05d}' for i in range((len(rows)+shard_rows-1)//shard_rows)}
    require(all(p.name in expected_names for p in destination.iterdir()), 'unexpected extra shard/rows')
    for index,start in enumerate(range(0,len(rows),shard_rows)):
        selected=rows[start:start+shard_rows];path=destination/f'{index:05d}'
        prior=recover_shard(path,selected,spec,provenance,quarantine)
        if prior is None:
            require(backend is not None, 'pending rows require backend')
            begin=time.monotonic();arrays=[];packet=[]
            for row in selected:
                values,tokens=backend.forward([row],1)
                semantic_view(tokens,[row])
                require(values.dtype==np.float32 and values.shape==(1,spec['layers'],spec['width']), 'full feature shape/dtype')
                arrays.append(values);packet.extend(tokens)
            require(all(all(p['attention_mask']) for p in packet), 'batch1 padding forbidden')
            require(hash_value(packet)==token_plan[index]['tokens_sha256'], 'tokenization differs from full storage preflight')
            write_shard(path,selected,np.concatenate(arrays),spec,provenance,packet)
            prior=verify_completed(path,selected,spec,provenance)
            require(sum(p.stat().st_size for p in path.iterdir())+4096 <= token_plan[index]['reserved_bytes'], 'shard exceeds storage reservation')
            write_json(path/'completed.json',dict(schema='r2-full-raw-shard-commit-v1',receipt=prior['receipt'],
                ordered_rows_sha256=hash_value(selected),adopted_after_interruption=False,timing_seconds=time.monotonic()-begin))
        require(prior['tokens_sha256']==token_plan[index]['tokens_sha256'], 'recovered token sequence changed')
        prior['completed_marker']=pin(path/'completed.json');completed.append(prior)
    require([identity for s in completed for identity in s['ordered_ids']]==[r['id'] for r in rows], 'missing/reordered/duplicate rows')
    require(sum(s['bindings'] for s in completed)==sum(len(r['bindings']) for r in rows), 'missing logical bindings')
    return completed


def pending_shards(destination, rows, spec, provenance, size, quarantine):
    pending=False
    destination.mkdir(parents=True,exist_ok=True)
    expected={f'{i:05d}' for i in range((len(rows)+size-1)//size)}
    require(all(p.name in expected for p in destination.iterdir()), 'unexpected extra shard/rows')
    for index,start in enumerate(range(0,len(rows),size)):
        if recover_shard(destination/f'{index:05d}',rows[start:start+size],spec,provenance,quarantine) is None:pending=True
    return pending


def recover_budget(previous, now_utc):
    require(set(previous['model_seconds'])=={'qwen','llama'} and isinstance(previous['running'],bool)
            and previous['model'] in (None,'qwen','llama'), 'budget schema corrupt')
    require(all(isinstance(v,(int,float)) and np.isfinite(v) and v>=0 for v in [previous['total_seconds'],*previous['model_seconds'].values()]), 'budget values corrupt')
    require(sum(previous['model_seconds'].values())<=previous['total_seconds']+1., 'budget accounting corrupt')
    state=dict(previous,model_seconds=dict(previous['model_seconds']))
    if state['running']:
        gap=now_utc-state['as_of_unix']
        require(gap>=0, 'clock moved backwards after unclean interruption; accounting review required')
        state['total_seconds']+=gap
        if state['model'] is not None:state['model_seconds'][state['model']]+=gap
        state['unclean_gap_charged_seconds']=gap
        state['running']=False
    return state


def cap_reason(state,limits,setup_elapsed=0):
    if state['total_seconds']>=limits['total_seconds']:return 'full_total_cap'
    for model,seconds in state['model_seconds'].items():
        if seconds>=limits['model_seconds']:return 'full_model_cap_'+model
    if setup_elapsed>=limits['setup_phase_seconds']:return 'full_setup_phase_cap'
    return None


def supervise(command,output,attempt,limits):
    now=time.time()
    path=output/'full-budget.json'
    initial=dict(total_seconds=0.,model_seconds={'qwen':0.,'llama':0.},running=False,model=None,as_of_unix=now)
    baseline=recover_budget(json.loads(path.read_text()) if path.exists() else initial,now)
    require(cap_reason(baseline,limits) is None, 'full-run execution budget exhausted; do not reset')
    start=time.monotonic();process=None;reason=None;interrupted=False
    state=dict(baseline,running=True,as_of_unix=time.time())
    atomic_pointer(path,state)
    def snapshot():
        mono=time.monotonic();progress=attempt/'progress.json'
        p=json.loads(progress.read_text()) if progress.exists() else dict(model=None,intervals=[],phase_started=start)
        models=dict(baseline['model_seconds'])
        for interval in p['intervals']:models[interval['model']]+=interval['seconds']
        if p['model']:models[p['model']]+=max(0.,mono-p['phase_started'])
        return dict(total_seconds=baseline['total_seconds']+mono-start,model_seconds=models,
            running=True,model=p['model'],as_of_unix=time.time(),unclean_gap_charged_seconds=baseline.get('unclean_gap_charged_seconds',0.)),p
    def interrupt(*_):raise KeyboardInterrupt()
    old_term=signal.signal(signal.SIGTERM,interrupt)
    try:
        process=subprocess.Popen(command,start_new_session=True)
        while process.poll() is None:
            state,p=snapshot();atomic_pointer(path,state)
            reason=cap_reason(state,limits,time.monotonic()-p['phase_started'] if p['model'] is None else 0.)
            if reason:
                os.killpg(process.pid,signal.SIGKILL);process.wait();break
            time.sleep(.25)
    except KeyboardInterrupt:
        interrupted=True
    finally:
        signal.signal(signal.SIGTERM,old_term)
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
        state,_=snapshot();state['running']=False;state['model']=None;atomic_pointer(path,state)
        write_json(attempt/'supervisor-exit.json',dict(returncode=process.returncode if process else None,
            cap_reason=reason,interrupted=interrupted,budget=state,limits=limits,budget_reset_allowed=False))
    return 130 if interrupted else 124 if reason else process.returncode


def runtime_matches(actual, expected):
    for key in ('python','packages','cuda','cudnn','gpu','gpu_total_bytes','compute_capability'):
        require(actual[key]==expected[key], 'validated runtime changed: '+key)
    require(sorted(actual['pip_freeze'])==sorted(expected['pip_freeze']), 'installed dependency freeze changed')
    require(actual['driver'].splitlines()[0].split(',')[2].strip()==expected['driver'].splitlines()[0].split(',')[2].strip(), 'GPU driver changed; review runtime')


def worker(args,cfg,manifest,campaign,attempt):
    # Linux parent death kills this worker even if the supervisor is forcibly killed.
    parent=os.getppid();require(parent==args.parent_pid, 'supervised worker parent')
    libc=ctypes.CDLL(None);require(libc.prctl(1,signal.SIGKILL,0,0,0)==0 and os.getppid()==parent, 'parent-death watchdog prerequisite')
    progress=dict(model=None,intervals=[],phase_started=time.monotonic());atomic_pointer(attempt/'progress.json',progress)
    backend=None
    try:
        from src.checkpoint_r2_fresh_backend import FreshBackend,runtime_receipt
        runtime=runtime_receipt(cfg['runtime_lock']);runtime_matches(runtime,evidence(cfg,'host-runtime.json'))
        write_json(attempt/'host-runtime.json',runtime)
        snapshots=verify_cached_models(cfg,args.pilot_cache)
        storage=json.loads(args.storage_receipt.read_text())
        check_storage_receipt(storage,cfg,manifest,args.pilot_cache,args.output,snapshots)
        write_json(attempt/'verified-snapshots.json',snapshots)
        for model in ('qwen','llama'):
            model_root=args.output/model;model_root.mkdir(exist_ok=True)
            origin=model_root/'runtime-origin.json'
            prov=shard_provenance(campaign,model,file_hash(origin) if origin.exists() else 'not_loaded')
            quarantine=args.output/'quarantine'/model
            pending=pending_shards(model_root/'shards',manifest['rows'],cfg['models'][model],prov,cfg['shard_rows'],quarantine)
            if pending:
                progress.update(model=model,phase_started=time.monotonic());atomic_pointer(attempt/'progress.json',progress)
                try:
                    backend=FreshBackend(cfg['models'][model],snapshots[model],cfg['execution'])
                    expected=evidence(cfg,model+'/runtime-before-inference.json')['backend']
                    require(backend.details==expected, 'validated backend implementation/flags changed')
                    current=dict(runtime=runtime,backend=backend.details,snapshot_sha256=hash_value(snapshots[model]))
                    write_json(attempt/(model+'-runtime-before-inference.json'),current)
                    if not origin.exists():write_json(origin,current)
                    else:
                        original=json.loads(origin.read_text());runtime_matches(original['runtime'],runtime)
                        require(original['backend']==backend.details and original['snapshot_sha256']==current['snapshot_sha256'], 'resume runtime/snapshot changed')
                    prov=shard_provenance(campaign,model,file_hash(origin))
                    completed=extract_model(model_root/'shards',manifest['rows'],cfg['models'][model],prov,backend,
                        cfg['shard_rows'],storage['plan']['models'][model]['shards'],quarantine)
                    write_json(attempt/(model+'-peak-memory.json'),backend.peak())
                finally:
                    if backend is not None:backend.close();backend=None
                    progress['intervals'].append(dict(model=model,seconds=time.monotonic()-progress['phase_started']))
                    progress.update(model=None,phase_started=time.monotonic());atomic_pointer(attempt/'progress.json',progress)
            else:
                completed=extract_model(model_root/'shards',manifest['rows'],cfg['models'][model],prov,None,
                    cfg['shard_rows'],storage['plan']['models'][model]['shards'],quarantine)
            index=dict(model=model,unique_texts=sum(s['rows'] for s in completed),logical_bindings=sum(s['bindings'] for s in completed),
                manifest_sha256=campaign['manifest_sha256'],shards=completed)
            unchanged_or_publish(model_root/'index.json',index)
        write_json(args.output/'completion.json',dict(schema='r2-full-raw-completion-v1',manifest_sha256=campaign['manifest_sha256'],
            indices={m:pin(args.output/m/'index.json') for m in cfg['models']},raw_only=True,scientific_evaluation=False))
    except BaseException as exc:
        write_json(attempt/'failure.json',dict(error_type=type(exc).__name__,error=str(exc),partial_outputs_preserved=True))
        raise


def verify_output(output,cfg,manifest):
    """Portable final audit of the published manifest, complete coverage and shard bytes."""
    output=Path(output)
    campaign=json.loads((output/'campaign.json').read_text())
    require(campaign['schema']=='r2-full-raw-campaign-v1' and campaign['contract_sha256']==CONFIG_SHA256
            and campaign['manifest_sha256']==hash_value(manifest), 'output campaign identity')
    require(file_hash(output/'input-manifest.json')==hash_value(manifest), 'published manifest corrupt')
    completion=json.loads((output/'completion.json').read_text())
    require(completion['schema']=='r2-full-raw-completion-v1' and completion['manifest_sha256']==hash_value(manifest)
            and completion['raw_only'] is True and completion['scientific_evaluation'] is False
            and set(completion['indices'])==set(cfg['models']), 'completion identity/scope')
    expected_names={f'{i:05d}' for i in range(math_ceil_shards(manifest,cfg))}
    for m,spec in cfg['models'].items():
        require(completion['indices'][m]==pin(output/m/'index.json'), 'model index hash')
        index=json.loads((output/m/'index.json').read_text());expected=[]
        require(index['model']==m and index['manifest_sha256']==hash_value(manifest), 'model index identity')
        destination=output/m/'shards'
        require(set(p.name for p in destination.iterdir())==expected_names, 'missing/extra shard directories')
        prov=shard_provenance(campaign,m,file_hash(output/m/'runtime-origin.json'))
        for i,start in enumerate(range(0,len(manifest['rows']),cfg['shard_rows'])):
            path=destination/f'{i:05d}'
            require(path.is_dir() and not path.is_symlink(), 'unexpected shard path')
            r=verify_completed(path,manifest['rows'][start:start+cfg['shard_rows']],spec,prov)
            r['completed_marker']=pin(path/'completed.json');expected.append(r)
        require(index['shards']==expected and index['unique_texts']==manifest['unique_texts']
                and index['logical_bindings']==manifest['logical_bindings'], 'full index/coverage')
    return dict(status='full_output_hashes_and_coverage_verified',scientific_evaluation=False)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['dry-run','verify-handoff','preflight','storage-check','run','verify-output'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--handoff',type=Path)
    p.add_argument('--pilot-cache',type=Path)
    p.add_argument('--volume',type=Path)
    p.add_argument('--destination',type=Path)
    p.add_argument('--storage-receipt',type=Path)
    p.add_argument('--expected-commit')
    p.add_argument('--acknowledge-reviewed-full-raw',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--internal-worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--parent-pid',type=int,help=argparse.SUPPRESS)
    p.add_argument('--attempt',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args();cfg,manifest=plan()
    if args.mode=='dry-run':
        payload=sum(manifest['fp16_payload_bytes'].values());reserve=cfg['storage']['metadata_planning_reserve_per_copy_bytes']
        write_json(args.output,dict(status='metadata_verified_no_inference',contract_sha256=CONFIG_SHA256,
            manifest_sha256=hash_value(manifest),unique_texts=manifest['unique_texts'],logical_bindings=manifest['logical_bindings'],
            ordered_ids_sha256=manifest['ordered_ids_sha256'],ordered_bindings_sha256=manifest['ordered_bindings_sha256'],
            group_counts=manifest['group_counts'],excluded_workloads=manifest['excluded_workloads'],shards_per_model=math_ceil_shards(manifest,cfg),
            fp16_payload_bytes=manifest['fp16_payload_bytes'],cached_weights_bytes=sum(evidence(cfg,m+'-snapshot.json')['downloaded_bytes'] for m in cfg['models']),
            additional_storage_planning_bytes=2*(payload+reserve)+cfg['storage']['safety_margin_bytes']+cfg['shard_rows']*32*4096*2,
            storage_estimate_status='planning allowance only; CPU full-token plan and actual simultaneous allocation required',
            limits=cfg['execution_limits'],full_run_launched=False))
        return
    if args.mode=='verify-handoff':
        require(args.handoff is not None,'handoff directory required');print(json.dumps(verify_handoff(cfg,args.handoff,args.output)));return
    if args.mode=='preflight':
        require(platform.system()=='Linux' and args.pilot_cache is not None,'remote Linux and cache required for runtime preflight')
        from src.checkpoint_r2_fresh_backend import runtime_receipt
        actual=runtime_receipt(cfg['runtime_lock']);runtime_matches(actual,evidence(cfg,'host-runtime.json'))
        snapshots=verify_cached_models(cfg,args.pilot_cache)
        write_json(args.output,dict(status='validated_gpu_runtime_and_offline_snapshots_no_inference',runtime=actual,
            snapshots=snapshots,contract_sha256=CONFIG_SHA256,manifest_sha256=hash_value(manifest),new_download_bytes=0))
        print(json.dumps(dict(gpu=actual['gpu'],python=actual['python'],new_download_bytes=0,inference=False)));return
    if args.mode=='storage-check':
        require(args.pilot_cache is not None and args.volume is not None and args.destination is not None,'cache, actual volume and separate destination required')
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_XET='1')
        print(json.dumps(storage_check(cfg,manifest,args.pilot_cache,args.volume,args.destination,args.output)));return
    if args.mode=='verify-output':
        print(json.dumps(verify_output(args.output,cfg,manifest)));return
    require(platform.system()=='Linux' and args.acknowledge_reviewed_full_raw,'full run requires later explicit review/authorization on remote Linux')
    require(args.pilot_cache is not None and args.storage_receipt is not None,'cache and storage proof required')
    args.output=args.output.resolve();args.pilot_cache=args.pilot_cache.resolve()
    require(args.output.is_absolute() and args.output.parent.is_dir() and not str(args.output).startswith('/tmp/'),'durable output parent required')
    require(not args.output.is_relative_to(ROOT) and not args.output.is_relative_to(args.pilot_cache.parent)
            and not args.pilot_cache.is_relative_to(args.output),'separate fresh output; preserve pilot/cache')
    head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    require(head==args.expected_commit and len(head)==40 and not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],text=True).strip(),'exact clean reviewed commit required')
    campaign=dict(schema='r2-full-raw-campaign-v1',commit=head,contract_sha256=CONFIG_SHA256,manifest_sha256=hash_value(manifest),limits=cfg['execution_limits'])
    if args.internal_worker:
        require(args.attempt is not None and args.parent_pid is not None,'worker supervision required')
        worker(args,cfg,manifest,campaign,args.attempt);return
    require(args.resume==args.output.exists(),'fresh output or explicit same-campaign --resume required')
    if not args.output.exists():args.output.mkdir();fsync_dir(args.output.parent)
    with (args.output/'full-run.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise ValueError('full campaign already active; monitor it') from None
        require(not (args.output/'completion.json').exists(),'campaign already complete; use verify-output')
        unchanged_or_publish(args.output/'campaign.json',campaign)
        unchanged_or_publish(args.output/'input-manifest.json',manifest)
        attempts=args.output/'attempts';attempts.mkdir(exist_ok=True)
        attempt=attempts/f'{len(list(attempts.iterdir())):05d}';attempt.mkdir()
        write_json(attempt/'launch.json',dict(argv=sys.argv,commit=head,parent_pid=os.getpid()))
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_XET='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false')
        command=[sys.executable,'-B','-m','src.checkpoint_r2_full_raw',*sys.argv[1:],'--internal-worker','--parent-pid',str(os.getpid()),'--attempt',str(attempt)]
        raise SystemExit(supervise(command,args.output,attempt,cfg['execution_limits']))


def valid_primary_bytes(destination,cfg,manifest):
    destination=Path(destination)
    if not destination.exists():return 0
    campaign=json.loads((destination/'campaign.json').read_text())
    require(campaign['contract_sha256']==file_hash(ROOT/CONFIG) and campaign['manifest_sha256']==hash_value(manifest), 'resume storage campaign identity')
    total=0
    if (destination/'input-manifest.json').exists():
        require(file_hash(destination/'input-manifest.json')==hash_value(manifest), 'published manifest corrupt')
        total+=(destination/'input-manifest.json').stat().st_size
    for m,spec in cfg['models'].items():
        origin=destination/m/'runtime-origin.json'
        for i,start in enumerate(range(0,len(manifest['rows']),cfg['shard_rows'])):
            path=destination/m/'shards'/f'{i:05d}'
            if (path/'receipt.json').exists():
                require(origin.exists(),'runtime origin missing')
                verify_completed(path,manifest['rows'][start:start+cfg['shard_rows']],spec,shard_provenance(campaign,m,file_hash(origin)))
                total+=sum((path/name).stat().st_size for name in ('activations.npy','rows.json','tokens.json','receipt.json'))
                if (path/'completed.json').exists():total+=(path/'completed.json').stat().st_size
    return total


def math_ceil_shards(manifest,cfg):
    return (manifest['unique_texts']+cfg['shard_rows']-1)//cfg['shard_rows']


if __name__=='__main__':main()
