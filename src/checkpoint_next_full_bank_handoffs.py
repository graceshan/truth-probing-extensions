"""Read-only, per-model handoffs verified in exact extraction-producer checkouts.

CPU HEAD is never an extraction identity. No worker, model, tokenizer or recovery
command is invoked; unsettled ledgers are rejected before verifier dispatch.
"""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from src.checkpoint_next_inventory import ROOT, require, sha
from src.checkpoint_next_store import publish_json
from src.checkpoint_next_adoption import policy

PRODUCERS = {'qwen': '9b0ac1d84b35d7faf2148d43664569aa867cba52',
             'llama': '6865bfa43c52d9c5d7d3c96df90351ad8f895aea'}


def read(path):
    return json.loads(Path(path).read_text())


def transfer(root, manifest):
    root=Path(root).resolve();manifest=Path(manifest).resolve()
    require(root.is_dir() and not manifest.is_relative_to(root),'distinct external transfer manifest required')
    paths=list(root.rglob('*'))
    require(not any(p.is_symlink() for p in paths),'transfer symlinks forbidden')
    actual={str(p.relative_to(root)):sha(p) for p in sorted(paths) if p.is_file()}
    require(actual==read(manifest),'complete transfer file/hash mismatch')
    return dict(campaign=str(root),transfer_manifest=str(manifest),transfer_manifest_sha256=sha(manifest),files=actual)


def settled(root, model, successor=False):
    root=Path(root);folder=root/'attempts'/model
    starts=sorted(folder.glob('*/start.json'))
    require(starts,'real extraction attempt ledger required')
    require({p.parent for p in folder.glob('*/end.json')}=={p.parent for p in starts},'incomplete or orphan attempt ledger')
    rows=[]
    for p in starts:
        start=read(p);end=p.with_name('end.json');v=read(end);charge=v['active_seconds']
        require(type(charge) in (int,float) and math.isfinite(charge) and charge>=0,'nonfinite/negative attempt charge')
        require(start['phase'] in (('missing_raw',) if successor else ('missing_raw','preflight')),'unexpected attempt phase')
        require(type(start['utc']) in (int,float) and math.isfinite(start['utc']), 'invalid attempt timestamp')
        rows.append(dict(attempt=p.parent.name,start_sha256=sha(p),end_sha256=sha(end),start=start,end=v))
    return rows


def expected_identity(model):
    p=policy();idn=dict(p['original_identities'][model],producer_commit=PRODUCERS[model])
    if model=='llama':idn['raw_only_config_sha256']=sha(ROOT/'config/checkpoint_next/raw_only_fix_v1.json')
    return idn


# This script runs only in the supplied exact producer checkout. Its verification
# path loads saved shards and receipts; it never calls accept, audit or recovery.
VERIFY_CODE = '''
import json,sys
from pathlib import Path
from src.checkpoint_next_runner import frozen,verify,identity,verify_execution_evidence
from src.checkpoint_next_inventory import config
from src.checkpoint_r2_full_support import evidence
root,model,original,manifest=sys.argv[1:]
if model=='llama':
    from src.checkpoint_next_adoption import verify as adoption
    adoption(root,original,manifest,frozen(),execution=False)
accepted=verify(root,'missing_raw',model)
completion=json.loads((Path(root)/'missing_raw'/model/'completion.json').read_text())
print(json.dumps(dict(accepted=accepted,identity=identity(model,frozen()),
    measured_execution=verify_execution_evidence(root,model,completion),
    model_spec=config()['models'][model],backend=evidence(config(),model+'/runtime-before-inference.json')['backend']),allow_nan=False))
'''


def verify_in_checkout(model, entry, original, original_manifest):
    checkout=Path(entry['verifier_checkout']).resolve()
    # Inherited Git/Python overrides must not redirect imports or HEAD resolution.
    env={k:v for k,v in os.environ.items() if not k.startswith('GIT_') and k not in ('PYTHONPATH','PYTHONHOME')}
    env.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    def git(*args):return subprocess.check_output(['git',*args],cwd=checkout,env=env,text=True).strip()
    require(git('rev-parse','--show-toplevel')==str(checkout),'verifier must be checkout root')
    require(git('rev-parse','HEAD')==PRODUCERS[model],'wrong/swapped extraction verifier producer')
    require(not git('status','--porcelain','--untracked-files=all'),'extraction verifier must be clean')
    result=subprocess.run([sys.executable,'-c',VERIFY_CODE,str(Path(entry['campaign']).resolve()),model,
                           str(original),str(original_manifest)],cwd=checkout,env=env,text=True,capture_output=True)
    require(result.returncode==0,'exact-producer read-only verification failed: '+result.stderr[-4000:])
    require(git('rev-parse','HEAD')==PRODUCERS[model] and not git('status','--porcelain','--untracked-files=all'),
            'verifier changed during validation')
    return read_json_output(result.stdout)


def read_json_output(value):
    return json.loads(value)


def model_receipt(model, entry, original, original_manifest):
    t=transfer(entry['campaign'],entry['transfer_manifest']);root=Path(t['campaign'])/'missing_raw'/model
    idn=read(root/'identity.json')
    require(idn==expected_identity(model),'supplement extraction identity: wrong/swapped producer or altered bindings')
    completion=read(root/'completion.json')
    require(completion['identity']==idn and completion['phase']=='missing_raw' and completion['jobs']==131,
            'completion identity/phase/coverage mismatch')
    ledger=settled(t['campaign'],model,successor=model=='llama')
    if model=='llama':
        require(not (Path(t['campaign'])/'attempts/qwen').exists(),'Qwen successor attempts forbidden')
    result=verify_in_checkout(model,entry,original,original_manifest)
    require(result['identity']==idn and read(root/'acceptance.json')==result['accepted'],'actual acceptance changed')
    require(transfer(entry['campaign'],entry['transfer_manifest'])==t,'handoff changed during verification')
    names=('identity.json','token-plan.json','token-audit.json','completion.json','acceptance.json')
    shards={str(p.relative_to(root)):sha(p) for p in sorted((root/'shards').rglob('*')) if p.is_file()}
    return dict(identity=idn,extraction_producer=PRODUCERS[model],campaign=t['campaign'],
        transfer=t,verification=result['accepted']['verification'],acceptance_sha256=sha(root/'acceptance.json'),
        artifacts={n:sha(root/n) for n in names},shard_files=shards,attempts=ledger,
        ordered_token_bindings=read(root/'token-plan.json'),measured_execution=result['measured_execution'],
        model_spec=result['model_spec'],backend=result['backend'],exact_producer_verified=True)


def acceptance(bindings, require_complete=True):
    b=read(bindings) if isinstance(bindings,(str,Path)) else bindings
    require(set(b)=={'schema','qwen','llama'} and b['schema']=='checkpoint-next-full-bank-handoffs-v2','handoff descriptor schema')
    require(b['qwen'] is not None,'original Qwen handoff required')
    for model in PRODUCERS:
        if b[model] is not None:require(set(b[model])=={'campaign','transfer_manifest','verifier_checkout'},'explicit per-model handoff fields')
    p=policy();original=Path(b['qwen']['campaign']).resolve();manifest=Path(b['qwen']['transfer_manifest']).resolve()
    require(sha(manifest)==p['original_manifest_sha256'],'unapproved original transfer manifest')
    qwen=model_receipt('qwen',b['qwen'],original,manifest)
    require(qwen['acceptance_sha256']==p['original_acceptance_sha256'],'changed pinned Qwen acceptance')
    original_ledgers={m:settled(original,m) for m in PRODUCERS}
    for m,rows in original_ledgers.items():
        pins=[dict(attempt=r['attempt'],start_sha256=r['start_sha256'],end_sha256=r['end_sha256'],
                   phase=r['start']['phase'],active_seconds=r['end']['active_seconds']) for r in rows]
        require(pins==sorted(p['original_attempts'][m],key=lambda r:r['attempt']),'original accounting mismatch')
    models={'qwen':qwen};adoption=None;status={'qwen':'verified_original_transferred_acceptance','llama':'pending_actual_transferred_acceptance'}
    if b['llama'] is not None:
        successor=Path(b['llama']['campaign']).resolve()
        require(not successor.is_relative_to(original) and not original.is_relative_to(successor),'distinct artifact roots required')
        require(Path(b['llama']['transfer_manifest']).resolve()!=manifest,'distinct transfer manifests required')
        # Validate transfer before any subprocess that could inspect accounting.
        transfer(successor,b['llama']['transfer_manifest'])
        a=read(successor/'adoption.json')
        require(a['successor_producer']==PRODUCERS['llama'] and a['policy_sha256']==sha(ROOT/'config/checkpoint_next/raw_only_fix_v1.json')
                and a['full_preflight_acceptance'] is False,'adoption producer/policy changed')
        carry={m:sum(r['end']['active_seconds'] for r in rows) for m,rows in original_ledgers.items()}
        require(a['original']['original_producer']==PRODUCERS['qwen'] and a['original']['original_manifest_sha256']==sha(manifest)
                and a['original']['original_identities']==p['original_identities'] and a['original']['original_attempts']==p['original_attempts']
                and a['original']['carry_seconds']==carry and a['original']['qwen_acceptance_sha256']==qwen['acceptance_sha256'],
                'adoption original provenance/accounting mismatch')
        rows=settled(successor,'llama',successor=True);total=carry['llama']
        for r in sorted(rows,key=lambda r:r['start']['utc']):
            before=dict(original_seconds=carry['llama'],successor_seconds=total-carry['llama'],charged_seconds=total,remaining_seconds=7200-total)
            require(r['start']['adoption_sha256']==sha(successor/'adoption.json') and
                    all(math.isclose(r['start']['campaign_budget_before_attempt'][k],v,rel_tol=0,abs_tol=1e-9) for k,v in before.items()) and
                    math.isclose(r['start']['remaining_gpu_seconds'],before['remaining_seconds'],rel_tol=0,abs_tol=1e-9),
                    'inconsistent cumulative attempt accounting')
            total+=r['end']['active_seconds']
        require(total<=7200,'cumulative model cap exceeded')
        models['llama']=model_receipt('llama',b['llama'],original,manifest)
        status['llama']='verified_successor_transferred_acceptance'
        adoption=dict(campaign=str(successor),sha256=sha(successor/'adoption.json'),receipt=a,
                      original_ledgers=original_ledgers,total_llama_seconds=total)
    require(not require_complete or len(models)==2,'Llama pending actual completed transferred acceptance; full-bank fitting blocked')
    return dict(schema=b['schema'],status='transferred_independently_reopened' if len(models)==2 else 'pending_actual_transferred_acceptance',
                model_status=status,extraction_producers=PRODUCERS,models=models,adoption=adoption,
                original_attempt_ledgers=original_ledgers)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--handoffs',required=True)
    p.add_argument('--output',required=True);p.add_argument('--require-complete',action='store_true');a=p.parse_args()
    require(not Path(a.output).exists(),'new read-only acceptance output required')
    result=acceptance(a.handoffs,a.require_complete);publish_json(Path(a.output),result)
    print(json.dumps(dict(status=result['status'],model_status=result['model_status'],extraction_producers=PRODUCERS),indent=2))


if __name__=='__main__':main()
