"""Mac-only audit launcher. The unchanged producer acquires its own CPU lock.

Preserves source report, existing controls, raw campaign and prerequisite blocker.
No retries, repairs, generated controls or E scoring outside the pinned runner.
"""
from pathlib import Path
import datetime
import hashlib
import json
import os
import resource
import socket
import subprocess
import sys
import time

PACKAGE = Path('/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1')
SCIENCE = PACKAGE/'scientific-evaluation-20261005-v1'
CODE = PACKAGE/'execution-code'
EVIDENCE = PACKAGE/'independent-audit-evidence-20261005-v1'
REPO = Path('/Users/apple/projects/truth-probing-extensions')
DELIVERY = '75030617daa1987d22d7d550714bf78bc2b0c48e'
PRODUCER = 'b2c6dde71e82f5f0af308761bf19ed3341670587'
COMPLETION = '632ed93f295027701e1d798a94c758e332c747e071aee410ebb6972114d408bc'
LOCK = '88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e'
AUTH = '41af4c83bafc4d216b5df836cb6c1c018ecdc3c6659d41d324227aad2bfa6b62'


def read(p): return json.loads(Path(p).read_bytes())


def pin(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())


def write(p,d):
    with Path(p).open('x') as f: f.write(json.dumps(d,indent=2,sort_keys=True)+'\n')


def git(root,*args):
    return subprocess.check_output(['git','-C',str(root),*args],text=True).strip()


def inventory(root,expected):
    actual={str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    assert actual==set(expected),(str(root),'file coverage')
    for rel,x in expected.items():
        p=root/rel
        assert not p.is_symlink() and p.resolve().is_relative_to(root.resolve())
        assert pin(p)==x,(str(p),'hash mismatch')
    return dict(files=len(actual),bytes=sum(x['bytes'] for x in expected.values()),full_sha256=True,exact_coverage=True)


def preflight():
    started=time.monotonic();receipt=dict(status='checking',evaluation_delivery=DELIVERY,
        report_completion_pin=COMPLETION,host=socket.gethostname(),RunPod_contacted=False,
        user_target_identity_confirmation=True,scientific_approval_renewed=False)
    try:
        assert git(CODE,'rev-parse','HEAD')==PRODUCER and git(CODE,'status','--porcelain')==''
        assert subprocess.run(['git','-C',str(CODE),'symbolic-ref','-q','HEAD'],stdout=subprocess.DEVNULL).returncode!=0
        handoff=json.loads(git(REPO,'show',DELIVERY+':results/final_e_scientific_evaluation_20261005/evaluation-handoff.json'))
        assert handoff['status']=='completed_pending_Mac_2_independent_audit'
        assert handoff['report_completion']['sha256']==COMPLETION and pin(SCIENCE/'evaluation/completion.json')['sha256']==COMPLETION
        assert handoff['producer_sha']==PRODUCER and handoff['lock']['sha256']==LOCK and handoff['audit_authorization']['sha256']==AUTH
        assert handoff['raw_campaign']==str(PACKAGE/'handoff/campaign') and handoff['report']==str(SCIENCE/'evaluation')
        for key in ['lock','operation_receipt','process_receipt','roots','raw_manifest','authorization_index','report_completion']:
            x=handoff[key];assert pin(x['path'])=={k:x[k] for k in ('bytes','sha256')}
        process=read(handoff['process_receipt']['path']);operation=read(handoff['operation_receipt']['path'])
        assert process['returncode']==0 and operation['status']=='completed' and operation['error'] is None
        lock=read(PACKAGE/'transport/final_lock_20261004.json');auth=read(SCIENCE/'control/local-both-audit.json')
        assert pin(PACKAGE/'transport/final_lock_20261004.json')['sha256']==LOCK
        assert pin(SCIENCE/'control/local-both-audit.json')['sha256']==AUTH
        assert lock['status']=='FINAL' and lock['claims_finalized'] is True and lock['scientific_lock_approved'] is True
        assert lock['approval']['approved_candidate_sha256']=='be5b15fe38c6c32a344e321d767d04a6f4c3b6f529ecfb338c2cc6e6edd5d1d3'
        assert lock['approval']['independent_review_sha']=='523e756346b373cc6e8be1d788901e5b3291c1d3'
        assert lock['approval']['approval_reply']=='Yes I approve.'
        for rel,h in lock['immutable_contract'].items():assert pin(CODE/rel)['sha256']==h
        provenance=json.loads(git(REPO,'show',DELIVERY+':results/final_e_scientific_evaluation_20261005/approval-provenance.json'))
        assert provenance['date']=='2026-10-05' and provenance['reply']=='Yes I approve'
        assert auth['instruction']==provenance['instruction'] and auth['reviewer']==provenance['reviewer']
        assert read(SCIENCE/'control/artifact-roots.json')==read(PACKAGE/'transport/artifact_roots.json')
        assert (SCIENCE/'control/artifact-roots.json').read_bytes()==subprocess.check_output(['git','-C',str(REPO),'show',DELIVERY+':results/final_e_scientific_evaluation_20261005/controls/artifact-roots.json'])
        original_preflight=json.loads(git(REPO,'show',DELIVERY+':results/final_e_scientific_evaluation_20261005/preflight.json'))
        for x in original_preflight['artifacts']:assert pin(x['path'])=={k:x[k] for k in ('bytes','sha256')}
        receipt['saved_references_verified']=len(original_preflight['artifacts'])
        manifest=read(PACKAGE/'handoff/campaign-checksums.json')
        receipt['raw_inventory']=inventory(PACKAGE/'handoff/campaign',manifest['files'])
        source=read(SCIENCE/'evaluation/completion.json')
        assert source['status']=='completed' and source['access']['producer_sha']==PRODUCER and source['access']['lock_sha256']==LOCK
        receipt['source_inventory']=inventory(SCIENCE/'evaluation',{**source['files'],'completion.json':pin(SCIENCE/'evaluation/completion.json')})
        receipt['immutable_contract']=lock['immutable_contract']
        receipt['source_completion']=pin(SCIENCE/'evaluation/completion.json')
        receipt['raw_manifest']=pin(PACKAGE/'handoff/campaign-checksums.json')
        receipt['controls']={str(p):pin(p) for p in SCIENCE.joinpath('control').iterdir() if p.is_file()}
        receipt['existing_receipts']={str(p):pin(p) for p in SCIENCE.joinpath('operation-receipts').iterdir() if p.is_file()}
        blocker=PACKAGE/'audit-prerequisite-inspection-20261005-v1/blocked.json'
        receipt['historical_blocker']=dict(path=str(blocker),**pin(blocker),status=read(blocker)['status'])
        receipt['blocker_resolution']='Original missing-handoff receipt preserved; evaluation now completed and exact target identities supplied by user. No failed audit was retried.'
        receipt['unrelated_pins']=read(Path('/Users/apple/truth-probing-backups/20261001T164224Z/checkpoint-final-e-lock-review-20261004/results/final_e_lock_review_20261004/preservation.json'))['standalone_untracked_pins']
        for rel,x in receipt['unrelated_pins'].items(): assert pin(REPO/rel)==x
        receipt['worktrees']=[]
        for block in git(REPO,'worktree','list','--porcelain').split('\n\n'):
            p=Path(block.splitlines()[0][9:])
            receipt['worktrees'].append(dict(path=str(p),head=git(p,'rev-parse','HEAD'),status=git(p,'status','--porcelain')))
        for p in [SCIENCE/'audit-reconstruction',SCIENCE/'operation-receipts/audit.json',SCIENCE/'operation-receipts/audit.audit.json']:
            assert not p.exists(),('fresh path required',str(p))
        sys.path.insert(0,str(CODE))
        from src.final_e_runner import parser
        from src.final_e_access import authorize
        receipt['access']=authorize(parser().parse_args(command()[3:]))
        receipt['status']='passed'
    except BaseException as e:
        receipt.update(status='failed_preserved',error=dict(type=type(e).__name__,message=str(e)))
        raise
    finally:
        receipt['seconds']=time.monotonic()-started
        write(EVIDENCE/'preflight.json',receipt)
    return receipt


def command():
    return ['/opt/homebrew/opt/python@3.11/bin/python3.11','-m','src.final_e_runner','audit',
        '--model','both','--output',str(PACKAGE/'handoff/campaign'),
        '--roots',str(SCIENCE/'control/artifact-roots.json'),
        '--source-report',str(SCIENCE/'evaluation'),'--report',str(SCIENCE/'audit-reconstruction'),
        '--operation-receipt',str(SCIENCE/'operation-receipts/audit.json'),
        '--authorization',str(SCIENCE/'control/local-both-audit.json'),'--authorization-sha256',AUTH,
        '--lock',str(PACKAGE/'transport/final_lock_20261004.json'),'--lock-sha256',LOCK]


def main():
    EVIDENCE.mkdir(exist_ok=False)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    names=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')
    for k in names:os.environ[k]='1'
    preflight()
    argv=command();started=time.monotonic()
    with (EVIDENCE/'audit.stdout.log').open('x') as out,(EVIDENCE/'audit.stderr.log').open('x') as err:
        p=subprocess.Popen(argv,cwd=CODE,env=os.environ,stdout=out,stderr=err)
        write(EVIDENCE/'audit-launch.json',dict(argv=argv,cwd=str(CODE),pid=p.pid,supervising_pid=os.getpid(),
            environment={k:os.environ[k] for k in (*names,'PYTHONDONTWRITEBYTECODE')},
            started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cap_seconds=10800,
            cpu_lock='acquired by frozen runner itself; launcher holds no CPU lock',attempts=1))
        print(json.dumps(dict(status='audit_started',pid=p.pid)),flush=True)
        rc=p.wait()
    result=dict(returncode=rc,seconds=time.monotonic()-started,
        peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
        attempts=1,retry=False,RunPod_contacted=False)
    write(EVIDENCE/'audit-process.json',result)
    print(json.dumps(result),flush=True)
    raise SystemExit(rc)


if __name__=='__main__':main()
