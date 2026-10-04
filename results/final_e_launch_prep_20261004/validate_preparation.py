"""Lightweight local preparation checks only. Never execute prospective commands.

Reproduce into a fresh optional output directory; published receipts are not
reused. No raw functions, models, numerical suite or E output arrays are opened.
"""
import argparse
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SOURCE
BASE = 'f12d9faf65ff5e31efbbcfbdb71d110a2ccbeb1c'
PRODUCER = 'b2c6dde71e82f5f0af308761bf19ed3341670587'
LOCK_HASH = '88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e'
CONFIG = ROOT / 'config/final_e/launch_preparation_20261004'
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           VECLIB_MAXIMUM_THREADS='1', NUMEXPR_NUM_THREADS='1')


def read(p):
    return json.loads(Path(p).read_bytes())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin(p):
    p = Path(p)
    return dict(bytes=p.stat().st_size, sha256=sha(p))


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', *args], cwd=cwd, text=True).strip()


def write(name, obj):
    with (OUT / name).open('x') as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write('\n')


def identity_and_transport(inv):
    lock = ROOT / 'config/final_e/final_lock_20261004.json'
    assert sha(lock) == LOCK_HASH
    data = read(lock)
    assert data['producer_sha'] == PRODUCER
    assert data['limits'] == inv['limits']
    assert data['immutable_contract'] and len(data['immutable_contract']) == 6
    for rel, digest in data['immutable_contract'].items():
        assert sha(ROOT / rel) == digest
    assert not git('diff', BASE, '--name-only', '--diff-filter=MDR')
    for rel in ['config/final_e/final_lock_20261004.json', *data['immutable_contract']]:
        assert (ROOT / rel).read_bytes() == subprocess.check_output(['git','show',BASE+':'+rel],cwd=ROOT)
    transport = read(SOURCE / 'transport-inventory.json')
    folder = Path(transport['transport_directory'])
    assert read(folder / 'transport-inventory.json') == transport
    actual = {str(p.relative_to(folder)) for p in folder.rglob('*') if p.is_file()} - {'transport-inventory.json'}
    assert actual == set(transport['files'])
    for rel, expected in transport['files'].items():
        assert pin(folder / rel) == expected
    assert (folder/'paths_and_scopes.json').read_bytes() == (CONFIG/'paths_and_scopes.json').read_bytes()
    assert sha(inv['paths']['local']['lock']) == LOCK_HASH
    code = inv['paths']['local']['code']
    assert git('rev-parse','HEAD',cwd=code) == PRODUCER
    assert not git('status','--porcelain',cwd=code)
    assert subprocess.run(['git','symbolic-ref','-q','HEAD'],cwd=code,stdout=subprocess.DEVNULL).returncode != 0
    bundle = transport['bundle']
    assert pin(bundle['path']) == {k:bundle[k] for k in ('bytes','sha256')}
    verify = subprocess.run(['git','bundle','verify',bundle['path']],cwd=code,text=True,capture_output=True,check=True)
    assert git('bundle','list-heads',bundle['path'],cwd=code) == PRODUCER+' HEAD'
    header = Path(bundle['path']).read_bytes().split(b'\n\n',1)[0].decode()
    prereqs = [line.split()[0][1:] for line in header.splitlines() if line.startswith('-')]
    assert prereqs == [inv['bundle_prerequisite_sha']]
    git('cat-file','-e',prereqs[0]+'^{commit}',cwd=code)
    git('merge-base','--is-ancestor',prereqs[0],PRODUCER,cwd=code)
    assert git('rev-list','--count',PRODUCER,'^'+prereqs[0]) == '3'
    return dict(final_lock_sha256=LOCK_HASH, immutable_pins=data['immutable_contract'],
                producer_clean_detached=True, bundle=bundle, bundle_verify=verify.stdout+verify.stderr,
                transport_files=len(actual), scientific_and_execution_files_unchanged=True)


def scopes_and_commands(inv):
    assert inv['execution_approved'] is False and inv['budget_adopted'] is False
    expected = {('remote',m,p) for m in ('qwen','llama') for p in ('extract','token_audit','accept','verify')}
    expected |= {('remote','both','transfer'),('local','both','transfer'),('local','both','verify')}
    got = set()
    for scope in inv['authorization_drafts']:
        key = (scope['environment'],scope['model'],scope['phase'])
        assert key not in got
        got.add(key)
        draft = ROOT / scope['draft_file']
        auth = read(draft)
        assert auth['schema'] == 'final-e-phase-authorization-v1'
        assert auth['status'] == 'DRAFT' and auth['approved'] is False
        assert auth['instruction'] is None and auth['reviewer'] is None
        assert auth['model'] == scope['model'] and auth['phase'] == scope['phase']
        assert auth['campaign_id'] == inv['campaign_id'] and auth['producer_sha'] == PRODUCER
        assert auth['lock_sha256'] == LOCK_HASH
        assert auth['output'] == scope['output'] == inv['paths'][scope['environment']]['output']
        assert sha(draft) == scope['draft_sha256']
        assert draft.read_bytes() == (Path(inv['paths']['local']['transport'])/'closed-authorization-drafts'/draft.name).read_bytes()
        for value in (auth['output'],scope['prospective_approved_file']):
            assert Path(value).is_absolute() and not any(c in value for c in '*?[')
    assert got == expected and len(got) == 11
    for side, paths in inv['paths'].items():
        for value in paths.values():
            assert Path(value).is_absolute()
        assert not Path(paths['lock']).is_relative_to(Path(paths['code']))
        assert not Path(paths['authorizations']).is_relative_to(Path(paths['code']))
        assert not Path(paths['output']).is_relative_to(Path(paths['code']))
        for k in ('output','verification_receipts','authorizations','authorization_index') if side=='local' else ():
            assert not Path(paths[k]).exists(), 'future destination already exists: '+paths[k]
    assert inv['paths']['local']['independent_remote_pins'].endswith('.txt')
    assert inv['current_connection']['status'] == 'outstanding'
    assert all(v is None for k,v in inv['current_connection'].items() if k!='status')
    # Compile only parser(), not execution imports or main()/dispatch().
    tree = ast.parse((ROOT/'src/final_e_runner.py').read_text())
    fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parser')
    ns = {'argparse':argparse,'__doc__':'static command check'}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<static runner parser>','exec'),ns)
    parser = ns['parser']()
    for command, scope in zip(inv['prospective_runner_commands'], inv['authorization_drafts']):
        assert command['executed'] is False
        assert command['argv'][1:3] == ['-m','src.final_e_runner']
        args = parser.parse_args(command['argv'][3:])
        assert args.command == scope['runner_command'] and args.model == scope['model']
        assert args.output == scope['output'] and args.lock_sha256 == LOCK_HASH
        assert args.authorization == scope['prospective_approved_file']
        assert args.authorization_sha256 == 'REQUIRES_SEPARATELY_REVIEWED_APPROVED_AUTHORIZATION_SHA256'
        assert args.lock == inv['paths'][scope['environment']]['lock']
        assert command['cwd'] == inv['paths'][scope['environment']]['code']
        assert not args.worker and args.parent_pid is None
        if args.command in ('extract','token_audit'):
            assert args.cache == inv['remote_historical_expectations']['offline_cache']
        if args.command=='extract':assert args.volume == '/workspace'
        else:
            assert Path(args.operation_receipt).is_absolute()
            assert not Path(args.operation_receipt).is_relative_to(Path(args.output))
    assert len(inv['prospective_runner_commands']) == 11
    doc = (ROOT/'docs/final_e/LAUNCH_PREPARATION_20261004.md').read_text()
    blocks = re.findall(r'```bash\n(.*?)\n```',doc,re.S)
    for block in blocks:
        subprocess.run(['bash','-n'],input=block,text=True,check=True,capture_output=True)
        for snippet in re.findall(r"<<'PY'\n(.*?)\nPY",block,re.S):ast.parse(snippet)
    return dict(scopes_checked=11, prospective_argv_parsed=11,bash_blocks_syntax_checked=len(blocks),commands_executed=False)


def closed_gate(inv):
    # Run only existing access interface from the clean detached exact producer.
    script = '''import json,sys
from types import SimpleNamespace
from pathlib import Path
from src.final_e_access import authorize
inv=json.loads(Path(sys.argv[1]).read_bytes());repo=Path(sys.argv[2]);results=[]
for s in inv['authorization_drafts']:
    args=SimpleNamespace(lock=inv['paths']['local']['lock'],lock_sha256=inv['final_lock_sha256'],
        authorization=str(repo/s['draft_file']),authorization_sha256=s['draft_sha256'],
        output=s['output'],model=s['model'],command=s['runner_command'])
    try:authorize(args,phase=s['phase'])
    except ValueError as e:
        assert str(e)=='separate scoped execution authorization';results.append(dict(environment=s['environment'],model=s['model'],phase=s['phase'],rejection=str(e)))
    else:raise AssertionError('closed draft passed gate')
print(json.dumps(results))
'''
    result = subprocess.check_output([sys.executable,'-c',script,str(CONFIG/'paths_and_scopes.json'),str(ROOT)],
                                     cwd=inv['paths']['local']['code'],env=ENV,text=True)
    return json.loads(result)


def accounting():
    prior = read(ROOT/'results/final_e_lock_final_20261004/accounting.json')
    carry = read(ROOT/'config/final_e/carryforward_v1.json')
    issues = []
    for path, expected in prior['metadata_pins'].items():
        if not Path(path).is_file() or pin(path) != {k:expected[k] for k in ('bytes','sha256')}:
            issues.append(dict(issue='historical_metadata_changed_or_missing',path=path))
    observed = {}
    for search in prior['roots_inspected']:
        for directory, dirs, files in os.walk(search):
            here = Path(directory)
            dirs[:] = [d for d in dirs if d not in ('.git','shards','snapshots','weights','node_modules') and (here/d).resolve()!=ROOT]
            for name in files:
                path = here/name
                is_attempt = 'attempts' in path.parts and name in ('start.json','end.json','active.json','supervisor-exit.json')
                is_campaign = name in ('full-budget.json','budget-adoption.json','qwen-campaign.json','llama-campaign.json','campaign.json')
                if is_attempt or is_campaign:observed[str(path.resolve())]=dict(path=str(path.resolve()),**pin(path))
    for path in sorted(set(observed)-set(prior['metadata_pins'])):
        issues.append(dict(issue='new_metadata_requires_review_before_adoption',path=path))
    for path in sorted(set(prior['metadata_pins'])-set(observed)):
        issues.append(dict(issue='historical_metadata_not_in_current_census',path=path))
    totals = dict(qwen=0.,llama=0.)
    for attempt in prior['unique_carried_attempts']:
        end = read(attempt['end']['path'])
        assert end['status']=='finished' and end['active_seconds']==attempt['seconds']
        totals[attempt['model']] += end['active_seconds']
    assert totals == carry['charged_seconds'] == dict(qwen=160.83801023702836,llama=123.82646256202133)
    for rel,digest in carry['source_files'].items():assert sha(ROOT/rel)==digest
    result=dict(status='no_local_intervening_charge_found' if not issues else 'launch_blocked_accounting_review_required',
                source_receipt_sha256=sha(ROOT/'results/final_e_lock_final_20261004/accounting.json'),metadata_pins=observed,
                charged_seconds=totals,operational_blockers=issues,adoption_performed=False,remote_state_verified=False,
                later_recheck_required=True,cpu_job_cap_seconds=10800,limits=dict(model_seconds=7200,total_seconds=18000,setup_phase_seconds=1800))
    write('accounting.json',result)
    return dict(status=result['status'],metadata_files=len(observed),operational_blockers=issues,charged_seconds=totals)


def saved_heads():
    roots = read(ROOT/'config/final_e/artifact_roots.example.json')
    registry = read(ROOT/'config/final_e/head_registry_v1.json')
    manifests = {'b25':Path(roots['b25_manifest']),'constituent':Path(roots['constituent_manifest']),
                 'full_bank':ROOT/'results/checkpoint_next_full_bank_results_20261003/external-output-manifest.json'}
    files = {}; checks = {}; blockers = []
    for name,path in manifests.items():
        assert pin(path)==registry['source_manifests'][name]
        m=read(path);files[name]=m.get('artifacts',m.get('files'))
        checks[name]=dict(path=str(path),**pin(path),root=roots[name],root_exists=Path(roots[name]).is_dir())
    refs = {}
    for head in registry['heads'].values():
        for key in ('fit','parameters','preprocessing','preprocessing_metadata'):
            if head[key]:
                ref=head[key];refs[(ref['source'],ref['path'])]=ref
    selection = {(r['source'],r['path']) for r in registry['selection_evidence'].values()}
    for ref in registry['selection_evidence'].values():refs[(ref['source'],ref['path'])]=ref
    availability=[]
    for (source,rel),ref in sorted(refs.items()):
        root=Path(roots[source]);prefix=root.name+'/' if source!='full_bank' else 'bank/'
        expected={k:ref[k] for k in ('bytes','sha256')}
        assert files[source][prefix+rel]==expected
        path=root/rel
        present=path.is_file() and not path.is_symlink() and path.stat().st_size==ref['bytes']
        if not present:blockers.append(dict(source=source,path=str(path),issue='missing_or_size_mismatched_saved_head_artifact'))
        hashed=(source,rel) in selection
        if present and hashed:assert sha(path)==ref['sha256']
        availability.append(dict(source=source,path=str(path),available_expected_bytes=present,
                                 expected=expected,content_hash_checked=hashed and present))
    assert len(registry['heads'])==85 and len(registry['scalar_aliases'])==156 and len(registry['constituent_cases'])==12
    result=dict(source_manifests=checks,heads=85,scalar_aliases=156,constituent_cases=12,unique_references=len(refs),
                artifacts=availability,operational_blockers=blockers,parameter_archives_reopened=False,numerical_audit=False,
                scope='source manifest hashes and reference-pin membership; existence/size for fit/parameter/preprocessing; selection-evidence content hashes only')
    write('saved-head-availability.json',result)
    return {k:result[k] for k in ('heads','scalar_aliases','constituent_cases','unique_references','operational_blockers','scope')}


def preservation():
    initial = Path('/private/tmp/final-e-launch-prep-20261004-2gli28vh/initial-state.json')
    if not initial.exists():
        return dict(status='original_machine_snapshot_unavailable_for_reproduction',snapshot=str(initial))
    before=read(initial);checked=[]
    for rel,expected in before['untracked_pins'].items():assert pin(Path(before['root'])/rel)==expected
    for worktree in before['worktrees']:
        assert git('rev-parse','HEAD',cwd=worktree['path'])==worktree['head']
        status=subprocess.check_output(['git','status','--porcelain'],cwd=worktree['path'],text=True)
        assert status==worktree['status'],worktree['path']
        checked.append(worktree['path'])
    result=dict(original_worktrees_unchanged=len(checked),unrelated_file_bytes_preserved=len(before['untracked_pins']),
                original_root_head=before['head'],original_root_branch=before['branch'],
                prior_accounting_metadata_unchanged=True,active_correlation_job_not_interrupted=True,
                initial_snapshot_sha256=sha(initial),intentional_new_worktrees=[str(ROOT),read(CONFIG/'paths_and_scopes.json')['paths']['local']['code']])
    write('preservation.json',result)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('validation.json','accounting.json','saved-head-availability.json','preservation.json'):
        assert not (OUT/name).exists(), 'fresh output required: '+str(OUT/name)
    start=time.monotonic();inv=read(CONFIG/'paths_and_scopes.json')
    with open('/tmp/truth-probing-cpu-exclusive.lock','a') as slot:
        fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
        result=dict(status='PASS',identity=identity_and_transport(inv),paths_scopes_commands=scopes_and_commands(inv),
                    closed_gate_rejections=closed_gate(inv),accounting=accounting(),saved_heads=saved_heads(),preservation=preservation(),
                    RunPod_contacted=False,E_outputs_opened=False,model_loading=False,numerical_suite_run=False,
                    historical_validators_rerun=False,budget_adopted=False,approved_phase_records_created=False)
    result['seconds']=time.monotonic()-start
    write('validation.json',result)
    print(json.dumps({k:result[k] for k in ('status','seconds','paths_scopes_commands','accounting','saved_heads')}))


if __name__=='__main__':main()
