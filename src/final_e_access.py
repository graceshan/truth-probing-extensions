"""Closed-by-default, hash-bound final-lock AND separate phase authorization gate."""
import subprocess
from pathlib import Path
from src.final_e_inventory import ROOT, read
from src.checkpoint_r2_fresh_inputs import require, file_hash, hash_value

PHASES = ('extract', 'token_audit', 'accept', 'verify', 'evaluate', 'audit', 'transfer')


def contract():
    paths=['config/final_e/preparation_v1.json','config/final_e/head_registry_v1.json','config/final_e/contrasts_v1.json','config/final_e/carryforward_v1.json',
           'results/final_e_preparation_20261003/inventory.json','results/final_e_preparation_20261003/bootstrap.json']
    return {p:file_hash(ROOT/p) for p in paths}


def draft():
    cfg=read(ROOT/'config/final_e/preparation_v1.json')
    return dict(schema='final-e-scientific-lock-v1',status='DRAFT',scientific_lock_approved=False,
        immutable_contract=contract(),pending={k:None for k in cfg['pending']},resolutions={},approval=None,
        producer_sha=None,limits=None,claims_finalized=False,carried_budget='config/final_e/carryforward_v1.json',
        note='Preparation is neither scientific lock nor permission for E extraction/predictions/metrics. No finalizer is provided.')


def validate_gate(lock, authorization, *, lock_sha256, authorization_sha256, phase, producer, output, model):
    require(phase in PHASES, 'undeclared E phase')
    require(lock['schema']=='final-e-scientific-lock-v1' and lock['status']=='FINAL' and
            lock['scientific_lock_approved'] is True and lock['claims_finalized'] is True, 'E CLOSED: finalized scientific lock required')
    require(lock['immutable_contract']==contract(), 'final lock altered frozen inputs/heads/statistics')
    required=set(read(ROOT/'config/final_e/preparation_v1.json')['pending'])
    require(not lock['pending'] and set(lock['resolutions'])==required and all(isinstance(v,str) and v.strip() for v in lock['resolutions'].values()), 'unresolved lock decisions')
    require(isinstance(lock['approval'],dict) and all(lock['approval'].get(k) for k in ('reviewer','instruction','reviewed_delivery_sha')), 'lock approval receipt required')
    limits=lock['limits']
    require(set(limits)=={'model_seconds','total_seconds','setup_phase_seconds'} and all(type(v) in (int,float) and 0<v<1e7 for v in limits.values()) and limits['setup_phase_seconds']==1800 and limits['model_seconds']==7200, 'explicit reviewed resource caps')
    require(lock['producer_sha']==producer and len(producer)==40, 'exact runnable producer')
    require(authorization['schema']=='final-e-phase-authorization-v1' and authorization['approved'] is True and
            authorization['phase']==phase and authorization['model']==model and authorization['producer_sha']==producer and
            authorization['lock_sha256']==lock_sha256 and authorization['output']==str(Path(output).resolve()), 'separate scoped execution authorization')
    require(authorization.get('instruction') and authorization.get('reviewer') and authorization.get('campaign_id'), 'actual instruction/identity required')
    require(len(lock_sha256)==len(authorization_sha256)==64, 'external hash pins required')
    return dict(schema='final-e-access-v1',producer_sha=producer,lock_sha256=lock_sha256,
        authorization_sha256=authorization_sha256,phase=phase,model=model,campaign_id=authorization['campaign_id'],
        immutable_contract_sha256=hash_value(lock['immutable_contract']),limits=dict(limits))


def authorize(args, phase=None):
    # This check happens before any E activations/predictions/metrics path is opened.
    require(args.lock and args.lock_sha256 and args.authorization and args.authorization_sha256, 'E CLOSED: lock and independent phase authorization pins required')
    require(file_hash(args.lock)==args.lock_sha256 and file_hash(args.authorization)==args.authorization_sha256, 'lock/authorization file hashes')
    require(not Path(args.output).resolve().is_relative_to(ROOT), 'E outputs must remain outside the frozen checkout')
    producer=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    require(not subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip(), 'clean detached producer required')
    require(subprocess.run(['git','symbolic-ref','-q','HEAD'],cwd=ROOT,stdout=subprocess.DEVNULL).returncode!=0, 'detached producer checkout required')
    return validate_gate(read(args.lock),read(args.authorization),lock_sha256=args.lock_sha256,
        authorization_sha256=args.authorization_sha256,phase=phase or args.command,producer=producer,output=args.output,model=args.model)
