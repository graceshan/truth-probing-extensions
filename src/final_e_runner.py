"""Local preparation and future separately authorized E execution entry point."""
import argparse
import fcntl
import json
import os
import signal
import time
from pathlib import Path
import numpy as np
from src.final_e_inventory import ROOT, Metadata, read
from src.checkpoint_r2_fresh_inputs import require, file_hash, hash_value
from src.checkpoint_r2_fresh_store import write_json, pin, publish
from src.final_e_access import authorize


def evaluate(args, access):
    from src import final_e_raw as raw
    from src.final_e_registry import resolve, apply_head
    from src.final_e_reporting import schedules, scalar_result, scalar_tables, constituent_result, constituent_tables
    metadata=Metadata();registry=read(ROOT/'config/final_e/head_registry_v1.json');roots=read(args.roots)
    for ref in registry['selection_evidence'].values():resolve(ref,roots)
    opts=read(ROOT/'config/final_e/preparation_v1.json')['bootstrap']
    weights,schedule=schedules(metadata,opts)
    require(schedule==read(ROOT/'results/final_e_preparation_20261003/bootstrap.json'),'E schedule changed')
    # Do not touch E outputs until authorization and metadata/registry validation.
    for model in ('qwen','llama'):raw.verify(args.output,model,access,require_acceptance=True)
    destination=Path(args.report);require(not destination.resolve().is_relative_to(ROOT),'report arrays outside Git checkout');require(not destination.exists(),'new report destination; preserve prior failed attempts')
    destination.mkdir(parents=True)
    write_json(destination/'access.json',access);write_json(destination/'bootstrap.json',schedule)
    scalar={};constituents={};cache={};started=time.monotonic()
    try:
        for model,layer in sorted({(h['model'],h['saved_layer']) for h in registry['heads'].values()}):
            features=raw.layer(args.output,model,layer,metadata,access)
            for name,h in registry['heads'].items():
                if (h['model'],h['saved_layer'])!=(model,layer):continue
                vals={g:apply_head(h,X,roots) for g,X in features.items()}
                folder=destination/'scores'/hash_value(name)[:24];folder.mkdir(parents=True)
                publish(folder/'scores.npz',lambda f:np.savez(f,**vals))
                write_json(folder/'binding.json',dict(head=name,head_sha256=hash_value(h),artifact=pin(folder/'scores.npz'),
                    group_bindings=metadata.receipt['ordered_groups'],access=access))
                if h['family']=='scalar':scalar[name]=scalar_result(metadata.rows,vals,weights,opts)
                else:cache[name]=vals
            del features
        sm,sc=scalar_tables(registry,scalar,read(ROOT/'config/final_e/contrasts_v1.json'),opts)
        for case in registry['constituent_cases']:
            groups=['bare']+[c for c in metadata.rows if c not in ('atomic_E','bare') and c.startswith('and_')==(case['evaluation_operator']=='AND')]
            results={}
            for group in groups:
                indices=[i for i,r in enumerate(metadata.rows[group]) if r['operator']==case['evaluation_operator']]
                rows=[metadata.rows[group][i] for i in indices]
                scores=np.column_stack([cache[name][group][indices] for name in case['heads']])
                results[group]=constituent_result(rows,scores,case['evaluation_operator'],weights[group][:,indices],opts)
            constituents[hash_value(case)]=results
            for group,result in results.items():
                folder=destination/'constituent-draws'/hash_value(dict(case=case,condition=group))[:24]
                folder.mkdir(parents=True)
                publish(folder/'draws.npy',lambda f:np.save(f,result['draws'],allow_pickle=False))
                write_json(folder/'binding.json',dict(case=case,condition=group,records=result['records'],draws=pin(folder/'draws.npy'),
                    row_binding_sha256=metadata.receipt['ordered_groups'][group],access=access))
        cm,cc=constituent_tables(registry['constituent_cases'],constituents,opts)
        for name,records in [('scalar-metrics',sm),('scalar-contrasts',sc),('constituent-metrics',cm),('constituent-contrasts',cc)]:
            write_json(destination/(name+'.json'),dict(access=access,records=records))
        # Wording changes use the same head and the same endpoint draws. No re-selection.
        wording=[]
        from src.clean_transfer_statistics import interval
        for name,res in scalar.items():
            records=res['records'];lookup={(r['condition'],r['metric'],r['scope'],r['topic']):i for i,r in enumerate(records)}
            for i,r in enumerate(records):
                if r['condition'] in ('bare','atomic_E'):continue
                j=lookup['bare',r['metric'],r['scope'],r['topic']]
                delta=res['points'][i]-res['points'][j]
                wording.append(dict(head=name,condition=r['condition'],metric=r['metric'],scope=r['scope'],topic=r['topic'],
                    point=None if not np.isfinite(delta) else float(delta),**interval(res['draws'][i]-res['draws'][j],opts)))
        write_json(destination/'wording-paired.json',dict(records=wording))
        # Retain draw arrays outside Git, bound to their exact metric identities.
        for name,result in scalar.items():
            folder=destination/'scores'/hash_value(name)[:24]
            publish(folder/'metric-draws.npy',lambda f:np.save(f,result['draws'],allow_pickle=False))
            write_json(folder/'metric-binding.json',dict(records=result['records'],draws=pin(folder/'metric-draws.npy')))
        write_json(destination/'completion.json',dict(status='completed',access=access,seconds=time.monotonic()-started,
            files={str(p.relative_to(destination)):pin(p) for p in sorted(destination.rglob('*')) if p.is_file()}))
    except BaseException as exc:
        write_json(destination/'failure.json',dict(type=type(exc).__name__,message=str(exc),seconds=time.monotonic()-started,acceptance=False))
        raise


def audit_report(args,access):
    """Independent invocation: verify pins, reapply heads, reconstruct every report."""
    require(args.source_report and args.report, 'original report and new audit reconstruction paths required')
    original=Path(args.source_report).resolve();completion=read(original/'completion.json')
    require(completion['status']=='completed' and completion['access']['producer_sha']==access['producer_sha']
            and completion['access']['lock_sha256']==access['lock_sha256'] and completion['access']['campaign_id']==access['campaign_id'], 'report identity')
    actual={str(p.relative_to(original)) for p in original.rglob('*') if p.is_file()}
    require(actual==set(completion['files'])|{'completion.json'},'report file coverage')
    for name,expected in completion['files'].items():
        p=original/name
        require(p.resolve().is_relative_to(original) and not p.is_symlink() and pin(p)==expected,'report artifact hash')
    evaluate(args,access)
    rebuilt=Path(args.report)
    checked=[]
    for name in ('scalar-metrics','scalar-contrasts','constituent-metrics','constituent-contrasts','wording-paired'):
        old,new=read(original/(name+'.json')),read(rebuilt/(name+'.json'))
        require(old['records']==new['records'],'complete metric reconstruction '+name)
        checked.append(dict(report=name,rows=len(new['records'])))
    registry=read(ROOT/'config/final_e/head_registry_v1.json')
    for name in registry['heads']:
        rel=Path('scores')/hash_value(name)[:24]
        ob,nb=read(original/rel/'binding.json'),read(rebuilt/rel/'binding.json')
        require(all(ob[k]==nb[k] for k in ('head','head_sha256','group_bindings')),'score identity reconstruction')
        with np.load(original/rel/'scores.npz',allow_pickle=False) as a,np.load(rebuilt/rel/'scores.npz',allow_pickle=False) as b:
            require(a.files==b.files and all(np.array_equal(a[g],b[g]) for g in a.files),'reapplied frozen predictions')
    write_json(Path(args.operation_receipt).with_suffix('.audit.json'),dict(status='complete_report_reconstruction_passed',access=access,
        original_completion=pin(original/'completion.json'),reconstructed_completion=pin(rebuilt/'completion.json'),reports=checked,
        head_predictions=len(registry['heads']),new_fits=0,new_selection=False))


def transfer(args, access):
    root=Path(args.output).resolve();path=Path(args.manifest).resolve()
    require(not path.is_relative_to(root),'external checksum manifest required')
    require(read(root/'full-budget.json')['running'] is False,'settled workers required before transfer')
    if args.command=='transfer':
        files={}
        for p in sorted(root.rglob('*')):
            require(not p.is_symlink(),'handoff symlink forbidden')
            if p.is_file():files[str(p.relative_to(root))]=pin(p)
        write_json(path,dict(schema='final-e-external-transfer-v1',access=access,files=files,bytes=sum(v['bytes'] for v in files.values())))
    else:
        m=read(path);actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
        require(actual==set(m['files']),'transfer exact file coverage')
        for name,expected in m['files'].items():
            p=root/name
            require(p.resolve().is_relative_to(root) and not p.is_symlink() and pin(p)==expected,'transfer checksum '+name)
        print(json.dumps(dict(status='checksum_verified_only',files=len(actual),manifest_sha256=file_hash(path),full_loader_acceptance=False)))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['metadata','draft','extract','token_audit','accept','verify','evaluate','audit','transfer','verify-transfer'])
    for name in ('output','lock','lock-sha256','authorization','authorization-sha256','cache','volume','attempt','roots','report','manifest','operation-receipt','source-report'):
        p.add_argument('--'+name)
    p.add_argument('--model',choices=['qwen','llama','both'],default='both')
    p.add_argument('--worker',action='store_true');p.add_argument('--parent-pid',type=int)
    return p


def main():
    args=parser().parse_args()
    if args.command=='metadata':
        print(json.dumps(Metadata().receipt,indent=2));return
    if args.command=='draft':
        from src.final_e_access import draft
        require(args.output,'new draft file required');write_json(Path(args.output),draft());return
    access=authorize(args,phase='transfer' if args.command=='verify-transfer' else args.command)
    if args.command in ('extract','token_audit','accept'):require(args.model in ('qwen','llama'),'exactly one model per raw phase')
    if args.command in ('evaluate','audit'):require(args.model=='both','paired two-model evaluation required')
    return run_authorized(args,access)


def dispatch(args,access):
    from src import final_e_raw as raw
    if args.command=='extract':return raw.worker(args,access) if args.worker else raw.launch(args,access)
    if args.command=='token_audit':result=raw.audit(args,access)
    elif args.command=='accept':result=raw.accept(args,access)
    elif args.command=='verify':
        result={m:raw.verify(args.output,m,access,require_acceptance=True) for m in (('qwen','llama') if args.model=='both' else (args.model,))}
    elif args.command in ('evaluate','audit'):
        require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')), 'one CPU thread required')
        with open('/tmp/truth-probing-cpu-exclusive.lock','a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            if args.command=='audit':audit_report(args,access)
            else:evaluate(args,access)
        return
    else:return transfer(args,access)
    print(json.dumps(result,indent=2))


def run_authorized(args,access):
    if args.command=='extract':return dispatch(args,access)  # reviewed process watchdog + durable attempt ledger
    require(args.operation_receipt, 'external exclusive operation receipt required')
    receipt=Path(args.operation_receipt).resolve()
    require(not receipt.exists() and not receipt.is_relative_to(Path(args.output).resolve()), 'preserve campaign during independent checks/transfer')
    seconds=10800 if args.command in ('evaluate','audit') else 1800
    def stop(*_):raise TimeoutError('frozen '+args.command+' phase cap reached')
    previous=signal.signal(signal.SIGALRM,stop);signal.setitimer(signal.ITIMER_REAL,seconds)
    started=time.monotonic();status='failed';error=None
    try:
        # All operations share the campaign lock; transfer cannot race audit or extraction.
        with (Path(args.output)/'campaign.lock').open('r') as campaign_lock:
            fcntl.flock(campaign_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            result=dispatch(args,access);status='completed';return result
    except BaseException as exc:
        error=dict(type=type(exc).__name__,message=str(exc));raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
        write_json(receipt,dict(schema='final-e-operation-receipt-v1',access=access,status=status,error=error,
            seconds=time.monotonic()-started,cap_seconds=seconds,measured=True,model_loaded=False,
            scope='CPU/setup phase; GPU active time remains in cumulative campaign ledger'))


if __name__=='__main__':main()
