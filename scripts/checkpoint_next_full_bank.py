"""Preparation CLI and separately authorized future CPU run/resume/audit. No Pod code."""
import argparse
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import uuid
import signal
import fcntl
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.checkpoint_next_inventory import ROOT, sha, require
from src.checkpoint_next_store import publish_json, same_or_publish

THREAD_ENV=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def gate(producer):
    require(subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==producer,'exact CPU producer checkout required')
    require(not subprocess.check_output(['git','status','--porcelain','--untracked-files=no'],cwd=ROOT,text=True).strip(),'tracked producer must be clean')
    require(all(os.environ.get(k)=='1' for k in THREAD_ENV),'one CPU/BLAS/OpenMP thread env required')
    from src.checkpoint_next_full_bank_inputs import configuration
    cfg=configuration()
    import importlib.metadata
    require({p:importlib.metadata.version(p) for p in cfg['local_packages']}==cfg['local_packages'],'pinned CPU numerical packages required')
    return cfg


def prepare(args,cfg):
    import pandas as pd
    from threadpoolctl import threadpool_limits,threadpool_info
    from src.checkpoint_next_full_bank_inputs import FullTrainingAdapter,read,CONFIG,EXTRACTION_PRODUCER
    from src.checkpoint_next_full_bank_provenance import FrozenB25,raw_exports
    from src.checkpoint_next_full_bank_selection import bank_specs,CONTRASTS
    from src.checkpoint_r2_fresh_inputs import hash_value
    from src.checkpoint_r2_fresh_store import pin
    out=Path(args.output).resolve();exports=Path(args.export_output).resolve()
    require(not out.exists() and not exports.exists(),'new preparation and export outputs required')
    out.mkdir(parents=True);exports.mkdir(parents=True)
    start=time.monotonic()
    with threadpool_limits(limits=1):
        adapter=FullTrainingAdapter();verification=adapter.verify_existing_metadata();frozen=FrozenB25(adapter)
        raw=raw_exports(adapter)
        publish_json(exports/'raw-chat-comparators.json',raw)
        compact=[{k:v for k,v in r.items() if k not in ('scores','logical_ids','support_800_scores','support_800_logical_ids')} for r in raw]
        refs=[frozen.describe(m,s,arm) for m in ('qwen','llama') for s in (11,23,37) for arm in ('R_all','S_all','reduced_LR','bank_oracle')]
        reference={m:dict(frozen_head=frozen.describe(m,11,'reduced_LR'),atomic_D_reference=next(r['point'] for r in frozen.existing_metrics(m,11,'reduced_LR') if r['condition']=='atomic_D' and r['scope']=='pooled'),margin=.005)
                   for m in ('qwen','llama')}
        specs=bank_specs();pd.DataFrame(specs).to_csv(out/'candidate-inventory.csv',index=False)
        publish_json(out/'adapter-bindings.json',dict(outer_train=adapter.bindings['outer_train'],P15=adapter.bindings['P15'],sampler_positions=adapter.full_balanced,
            sampler_source_ids=[adapter.train[i]['source_row_id'] for i in adapter.full_balanced],B25={str(s):adapter.allocations[s] for s in (11,23,37)}))
        publish_json(out/'raw-export-provenance.json',dict(heads=compact,external_export=str(exports/'raw-chat-comparators.json'),export_pin=pin(exports/'raw-chat-comparators.json'),
            source_parameter_producer='3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f',fit_counts={'R0':2778,'C_clean':8000},full_D_rows=7728,matching_chat_rows=800,no_new_fits=True))
        bootstrap=frozen.json('bootstrap_binding.json')
        publish_json(out/'frozen-references.json',dict(heads=refs,eligibility=reference,bootstrap=bootstrap,
            schedule_verification='source receipt/hash binding only during preparation; future worker verifies schedule bytes and all row axes',
            repair_fits_prepared=0,repair_fold_fits_prepared=0))
        publish_json(out/'reporting-contract.json',dict(required_contrasts=[list(p) for p in CONTRASTS],
            reference='original reduced-P15 LR atomic D minus .005, unchanged',full_canonical_LR='separate arm, never eligibility reference',
            retention_margins={'atomic':.005,'AND':.02},practical_repair_margin=.02,
            criteria='pointwise pooled and macro per model; reaches repair if upper bound of R_all-comparator <.02 with supported retention; robust gap if lower>.02; crossing unresolved',
            reports=['each seed and within-draw metric mean','atomic/AND retention','OR/all and TF/FT-vs-FF boundary','all three existing wordings','five topics and fixed-five macro','eligible/unrestricted full and P15 maxima'],
            selectors='existing B25 rank/PCG64(20261008) choice; atomic-only once/model PCG64(20261005) permutation',
            tie_identity='canonical inherited recipe IDs retained for ties; full-fit identities separately mapped',
            wording_lock=True,bootstrap=cfg['bootstrap'],undefined='no redraw/imputation/topic dropping; ordinary mean propagates undefined',
            oracle_uncertainty='primary fixed-winner conditional; endpoint maxima descriptive',A_exposure=True,
            interpretation='A identities now occur in atomic training: advantaged full-training comparator against existing reduced-P repair; does not test full-training repair'))
        metadata=dict(status='preparation_only',cpu_producer=args.producer,supplement_extraction_producer=EXTRACTION_PRODUCER,
            config_sha256=sha(CONFIG),input_requirements=adapter.receipt,input_verification=verification,eligibility=reference,
            all600_candidates_unrun=True,repair_fits=0,chat_fits=0,research_fits=0,E_access=False,
            raw_export_pin=pin(exports/'raw-chat-comparators.json'),runtime_seconds=time.monotonic()-start,
            process_peak_rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            rss_units='bytes on macOS, KiB on Linux',hostname=platform.node(),platform=platform.platform(),interpreter=sys.executable,
            threadpools=threadpool_info(),cap_seconds=10800,cap_is_runtime_estimate=False)
        publish_json(out/'preparation.json',metadata)
        adapter.unchanged()
    print(json.dumps(metadata,indent=2))


def identity(args,cfg):
    from src.checkpoint_next_full_bank_inputs import CONFIG
    return dict(cpu_producer=args.producer,config_sha256=sha(CONFIG),supplement_transfer_sha256=sha(args.transfer_manifest),
        prepared_manifest_sha256=sha(Path(args.prepared)/'preparation.json'),existing_acceptance_sha256=sha(ROOT/'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json'))


def authorized(args,cfg):
    from src.checkpoint_next_full_bank_inputs import read
    auth=read(args.authorization);expected=identity(args,cfg)
    require(auth['identity']==expected and auth['reviewed'] is True and auth['research_fit_authorized'] is True and
        auth['CPU_slot_exclusive'] is True and auth['candidates']==600 and auth['repair_fits']==0 and
        auth['worker_cap_seconds']==10800 and auth['E'] is False,'separate exact-producer full-bank authorization required')
    require(read(Path(args.prepared)/'preparation.json')['cpu_producer']==args.producer,'wrong preparation producer')
    return expected


def charges(output):
    from src.checkpoint_next_full_bank_inputs import read
    total=0.
    for p in sorted((output/'attempts').glob('*/start.json')):
        end=p.with_name('end.json')
        if not end.exists():
            # Interrupted worker time/downtime is conservatively charged, never replenished.
            same_or_publish(end,dict(seconds=max(0,time.time()-read(p)['utc']),status='interrupted_conservative_charge'))
        v=read(end)['seconds'];require(v>=0,'CPU attempt charge');total+=v
    return total


def supervise(args,cfg,idn):
    from src.checkpoint_next_full_bank_worker import accounting
    output=Path(args.output).resolve()
    require(not str(output).startswith(str(ROOT)+os.sep),'research outputs must remain outside Git worktree')
    if args.command=='run':require(not output.exists(),'fresh full-bank directory required')
    else:require(output.is_dir(),'resume needs existing task output')
    output.mkdir(parents=True,exist_ok=True)
    same_or_publish(output/'identity.json',idn)
    lock=Path('/private/tmp/truth-probing-cpu-exclusive.lock') if platform.system()=='Darwin' else Path('/tmp/truth-probing-cpu-exclusive.lock')
    with lock.open('a+') as exclusive:
        fcntl.flock(exclusive,fcntl.LOCK_EX|fcntl.LOCK_NB)
        remaining=10800-charges(output);require(remaining>0,'cumulative full-bank CPU cap exhausted')
        attempt=output/'attempts'/uuid.uuid4().hex;attempt.mkdir(parents=True)
        publish_json(attempt/'start.json',dict(utc=time.time(),remaining_seconds=remaining,identity=idn))
        command=[sys.executable,__file__,'_worker','--producer',args.producer,'--output',str(output),'--supplement',args.supplement,
                 '--transfer-manifest',args.transfer_manifest,'--authorization',args.authorization,'--prepared',args.prepared]
        env=dict(os.environ,**{k:'1' for k in THREAD_ENV},FULL_BANK_SUPERVISOR_PID=str(os.getpid()),FULL_BANK_REMAINING_SECONDS=str(remaining));start=time.monotonic();proc=None;reason='finished'
        try:
            with (attempt/'stdout.log').open('xb') as log:
                proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                def stop(*_):raise KeyboardInterrupt('supervisor stop')
                old=signal.signal(signal.SIGTERM,stop)
                try:proc.wait(timeout=remaining)
                except subprocess.TimeoutExpired:reason='three-hour cumulative worker cap hit';os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                finally:signal.signal(signal.SIGTERM,old)
        finally:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                reason='interrupted'
            publish_json(attempt/'end.json',dict(seconds=time.monotonic()-start,status=reason,exit_code=proc.returncode if proc else None))
            publish_json(attempt/'candidate-accounting.json',accounting(output,idn))
        require(reason=='finished' and proc.returncode==0,'affected bank stopped; all600 statuses/attempts preserved')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['prepare','run','resume','audit','_worker'])
    for k in ('producer','output'):p.add_argument('--'+k,required=True)
    for k in ('export-output','supplement','transfer-manifest','authorization','prepared','audit-output'):p.add_argument('--'+k)
    args=p.parse_args();cfg=gate(args.producer)
    if args.command=='prepare':require(args.export_output,'export output required');prepare(args,cfg);return
    require(all(getattr(args,k) for k in ('supplement','transfer_manifest','authorization','prepared')),'actual inputs and separate authorization required')
    idn=authorized(args,cfg)
    if args.command in ('run','resume'):supervise(args,cfg,idn)
    elif args.command=='_worker':
        require(os.environ.get('FULL_BANK_SUPERVISOR_PID')==str(os.getppid()),'supervised CPU child required')
        def timeout(*_):raise TimeoutError('cumulative CPU worker cap reached')
        signal.signal(signal.SIGALRM,timeout)
        signal.setitimer(signal.ITIMER_REAL,float(os.environ['FULL_BANK_REMAINING_SECONDS']))
        from src.checkpoint_next_full_bank_worker import worker
        worker(args,idn)
    else:
        require(args.audit_output,'fresh independent audit receipt path required')
        from src.checkpoint_next_full_bank_audit import audit
        print(json.dumps(audit(args,idn),indent=2))


if __name__=='__main__':main()
