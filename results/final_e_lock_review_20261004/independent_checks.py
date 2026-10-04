"""Independent lock review: metadata and tiny synthetic boundary checks only.

Usage: python -B independent_checks.py NEW_EXTERNAL_OUTPUT_DIRECTORY
Never executes the supplied candidate validator or writes its published receipts.
No FINAL lock, approval, phase authorization, model, E array or fit is created.
"""
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = 'e8f087e53b96b2172cfcde03e3a451e6a65bb0fe'
CANDIDATE = 'c2efef5fef91b989d66c3b6228d2ec5e26c2c0f4'
PRODUCER = 'b2c6dde71e82f5f0af308761bf19ed3341670587'
EXPECTED = 'be5b15fe38c6c32a344e321d767d04a6f4c3b6f529ecfb338c2cc6e6edd5d1d3'
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
sys.path.insert(0, str(ROOT))


def read(p):
    return json.loads(Path(p).read_bytes())


def pin(p):
    p = Path(p)
    b = p.read_bytes()
    return dict(path=str(p), bytes=len(b), sha256=hashlib.sha256(b).hexdigest())


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def identity():
    path = ROOT / 'config/final_e/lock_candidate_20261004.json'
    c = read(path)
    assert pin(path)['sha256'] == EXPECTED
    original = json.loads(git('show', BASE + ':config/final_e/draft_lock_v1.json'))
    assert c['immutable_contract'] == original['immutable_contract']
    hashes = {p: pin(ROOT / p) for p in c['immutable_contract']}
    assert all(hashes[p]['sha256'] == sha for p, sha in c['immutable_contract'].items())
    for p in ['config/final_e/draft_lock_v1.json', *hashes]:
        before = subprocess.check_output(['git', '-C', str(ROOT), 'show', BASE + ':' + p])
        assert (ROOT / p).read_bytes() == before
    changes = git('diff', '--name-status', BASE, CANDIDATE).splitlines()
    assert len(changes) == 7 and all(s.startswith('A\t') for s in changes)
    edges = [(PRODUCER, BASE), (BASE, CANDIDATE)]
    for a, b in edges:
        assert subprocess.run(['git', '-C', str(ROOT), 'merge-base', '--is-ancestor', a, b]).returncode == 0
    assert c['producer_sha'] == PRODUCER and PRODUCER != CANDIDATE
    assert c['status'] == 'DRAFT' and c['approval'] is None
    assert c['claims_finalized'] is False and c['scientific_lock_approved'] is False
    assert c['pending'] == {'independent_review_and_execution_authorization': None}
    assert set(c['resolutions']) | set(c['pending']) == set(original['pending'])
    assert c['limits'] == dict(model_seconds=7200, total_seconds=18000, setup_phase_seconds=1800)
    carry = read(ROOT / c['carried_budget'])
    assert carry['cpu_cap_seconds'] == 10800
    assert c['interpretation_contract']['resource_accounting']['carried_model_seconds'] == carry['charged_seconds']
    assert c['interpretation_contract']['resource_accounting']['carried_total_seconds'] == sum(carry['charged_seconds'].values())
    vroot = ROOT / 'results/final_e_lock_candidate_20261004'
    v = read(vroot / 'validation.json')
    assert v['accounting']['receipt_sha256'] == pin(vroot / 'accounting.json')['sha256']
    assert v['focused_tests']['log_sha256'] == pin(vroot / 'focused-tests.log')['sha256']
    assert v['candidate_checks']['candidate_sha256'] == EXPECTED
    assert 'Ran 34 tests' in (vroot / 'focused-tests.log').read_text()
    plan = ROOT / 'docs/final_e/Truth_Probing_Next_Decision_Checkpoint_Revision_3.docx'
    assert pin(plan)['sha256'] == c['candidate_preparation']['governing_plan_sha256']
    return dict(candidate=pin(path), immutable_files=hashes, original_draft=pin(ROOT / 'config/final_e/draft_lock_v1.json'),
                additions=changes, ancestry_edges=edges, receipt_links_checked=True,
                published_validation_head=v['checkout_head_at_validation'],
                validation_head_note='Pre-commit validation ran while HEAD was the base; this review checks the committed candidate.',
                limits=c['limits'], cpu_job_cap_seconds=10800)


def accounting():
    published = read(ROOT / 'results/final_e_lock_candidate_20261004/accounting.json')
    carry = read(ROOT / 'config/final_e/carryforward_v1.json')
    def confirm(spec):
        actual = pin(spec['path'])
        assert actual['bytes'] == spec['bytes'] and actual['sha256'] == spec['sha256']
        return actual
    checked = [confirm(x) for x in published['metadata_pins']]
    for x in published['immutable_source_pins'].values():
        # Historical receipt paths cite mac 1's checkout; read them without writing.
        checked.append(confirm(x))
    adoption = confirm(published['local_adoption'])
    report = confirm(published['local_execution_report'])
    old = read(adoption['path'])
    final = read(report['path'])
    totals = {'qwen': 0.0, 'llama': 0.0}
    ids = {}
    for a in published['unique_carried_attempts']:
        assert a['attempt'] not in ids
        ids[a['attempt']] = a
        confirm(a['start']); confirm(a['end'])
        end = read(a['end']['path'])
        assert end['status'] == 'finished' and end['active_seconds'] == a['seconds']
        totals[a['model']] += end['active_seconds']
    for model, attempts in carry['old_attempts_preserved'].items():
        assert old['original']['original_attempts'][model] == attempts
        assert old['original']['carry_seconds'][model] == sum(a['active_seconds'] for a in attempts)
    assert totals == carry['charged_seconds'] == published['charged_seconds']
    assert {m: b['charged_seconds'] for m, b in final['final_budgets'].items()} == totals
    remaining = {m: 7200 - s for m, s in totals.items()}
    assert remaining == carry['remaining_seconds']
    known_paths = {str(Path(x['path'])) for x in published['metadata_pins']}
    observed, additional = [], []
    roots = [Path(p) for p in published['roots_inspected']]
    names = {'full-budget.json', 'budget-adoption.json', 'qwen-campaign.json', 'llama-campaign.json', 'campaign.json'}
    attempt_names = {'start.json', 'end.json', 'active.json', 'supervisor-exit.json'}
    for search in roots:
        for folder, dirs, files in os.walk(search):
            dirs[:] = [d for d in dirs if d not in ('.git', 'shards', 'snapshots', 'weights', 'node_modules', '__pycache__')]
            here = Path(folder)
            for name in files:
                p = here / name
                if name not in names and not ('attempts' in p.parts and name in attempt_names):
                    continue
                spec = pin(p)
                observed.append(spec)
                content = read(p)
                if str(p) not in known_paths:
                    additional.append(dict(**spec, schema=content.get('schema'), kind=name))
                if 'attempts' in p.parts and p.parent.name in ids:
                    a = ids[p.parent.name]
                    if name in ('start.json', 'end.json'):
                        assert spec['sha256'] == a[name.removesuffix('.json')]['sha256']
                if content.get('schema') == 'final-e-raw-campaign-v1' or name == 'budget-adoption.json':
                    raise AssertionError('New E accounting requires review: ' + str(p))
    assert not additional, additional
    duplicates = {}
    for a in ids:
        copies = [p['path'] for p in observed if Path(p['path']).parent.name == a and Path(p['path']).name == 'end.json']
        duplicates[a] = copies
    earlier = published['separate_historical_full_raw']
    confirm(earlier['campaign']); confirm(earlier['ledger'])
    assert read(earlier['ledger']['path']) == earlier['ledger_values']
    assert earlier['ledger_values']['running'] is False
    cpu = published['separate_historical_full_bank_cpu']
    confirm(cpu['receipt']); assert read(cpu['receipt']['path']) == cpu['values']
    assert cpu['values']['status'] == 'finished'
    return dict(status='local_receipts_reconcile', cited_metadata_pins=checked,
                unique_attempts=list(ids.values()), duplicate_end_copies=duplicates,
                totals=totals, remaining=remaining, seed_total=sum(totals.values()),
                remaining_total_from_18000=18000 - sum(totals.values()),
                roots_inspected=[str(p) for p in roots], discovered_metadata=observed,
                additional_metadata=additional, budget_adoption_performed=False,
                remote_state_verified=False, later_recheck_required=True,
                separate_historical_full_raw=earlier, separate_historical_full_bank_cpu=cpu)


def focused():
    """Small synthetic arrays (<2 MB), not a numerical suite/reproduction."""
    import numpy as np
    from src.final_e_access import PHASES, validate_gate
    from src.final_e_reporting import scalar_tables
    from src.checkpoint_r2_b25_statistics import masks
    from src.checkpoint_r2_selection_v1 import average_sample_metrics
    from src.clean_transfer_statistics import interval
    from src.checkpoint_r2_full_raw import cap_reason
    c = read(ROOT / 'config/final_e/lock_candidate_20261004.json')
    closed = []
    for phase in PHASES:
        for model in ('qwen', 'llama', 'both'):
            try:
                validate_gate(c, {}, lock_sha256=EXPECTED, authorization_sha256='0'*64,
                              phase=phase, producer=PRODUCER, output='/nonexistent/E-closed', model=model)
            except ValueError as e:
                assert str(e) == 'E CLOSED: finalized scientific lock required'
                closed.append(dict(phase=phase, model=model, reason=str(e)))
            else:
                raise AssertionError('Candidate gate accepted')
    opts = read(ROOT / 'config/final_e/preparation_v1.json')['bootstrap']
    rows = [dict(operator=op, truth_a=a, truth_b=b) for op in ('AND','OR') for a,b in ((0,0),(1,0),(0,1),(1,1))]
    mask = dict(masks(rows))['OR_mixed_vs_FF_auroc']
    assert np.flatnonzero(mask).tolist() == [4,5,6]
    avg = average_sample_metrics({11: np.array([1.,np.nan]), 23: np.array([4.,2.]), 37: np.array([7.,3.])})
    assert avg[0] == 4. and np.isnan(avg[1])
    finite_cases = []
    for n in (1799,1800,2000):
        ci = interval(np.r_[np.zeros(n),np.full(2000-n,np.nan)],opts)
        assert (ci['ci_low'] is not None) == (n>=1800)
        assert ci['invalid_replicates'] == 2000-n
        finite_cases.append(ci)
    aliases = [dict(panel='b25', model=m, arm=a, sample=s, category='synthetic', head=m+'/'+a+'/'+str(s))
               for m in ('qwen','llama') for a in ('R_all','S_all','reduced_LR','bank_oracle') for s in (11,23,37)]
    cases = []
    for metric, margin in [('atomic_auroc',.005),('AND_auroc',.02)]:
        boundary = -margin
        for label, lower, upper, expected in [('lower_equal',boundary,0.,'supported'),
                ('upper_equal',boundary-.001,boundary,'unresolved'),
                ('crossing',boundary-.001,boundary+.001,'unresolved'),
                ('entire_below',boundary-.002,boundary-.001,'loss'),
                ('point_equal_interval_below',boundary-.002,boundary-.001,'loss'),
                ('undefined',None,None,'unresolved')]:
            records = [dict(condition='atomic_E' if metric=='atomic_auroc' else 'bare',metric=metric,scope=scope,topic='all')
                       for scope in ('pooled','topic_macro')]
            vals = np.full(2000,np.nan) if lower is None else np.r_[np.full(1000,lower),np.full(1000,upper)]
            point = boundary if label=='point_equal_interval_below' else (np.nan if lower is None else (lower+upper)/2)
            results = {}
            for a in aliases:
                v = vals if a['arm']=='R_all' else np.zeros(2000)
                results[a['head']] = dict(records=records,points=np.full(2,point if a['arm']=='R_all' else 0.),draws=np.vstack([v,v]))
            _, contrasts = scalar_tables(dict(scalar_aliases=aliases),results,dict(b25=[['R_all','reduced_LR']]),opts)
            kept = [x for x in contrasts if x['model']=='qwen' and x['sample']=='mean_11_23_37' and x['right']=='reduced_LR']
            assert len(kept)==2
            for x in kept:
                # Interpret emitted intervals; never infer loss from a False support flag.
                status = 'unresolved' if x['ci_low'] is None else ('supported' if x['ci_low']>=boundary else ('loss' if x['ci_high']<boundary else 'unresolved'))
                assert status == expected
                assert x['interval_supported'] == (status=='supported')
            cases.append(dict(metric=metric, case=label, interpreted_status=expected, rows=kept))
    gain_cases=[]
    for low,high in [(0.,.03),(.001,.019),(.001,.02),(.001,.03),(.02,.03),(.020001,.03),(None,None)]:
        gain_cases.append(dict(low=low,high=high,positive=low is not None and low>0,
            beyond_practical=low is not None and low>.02,
            practical_status='unresolved' if low is None else ('beyond' if low>.02 else ('below' if high<.02 else 'unresolved'))))
    caps=[]
    for state,setup,reason in [(dict(total_seconds=18000,model_seconds={'qwen':160.,'llama':124.}),0,'full_total_cap'),
            (dict(total_seconds=8000,model_seconds={'qwen':7200.,'llama':124.}),0,'full_model_cap_qwen'),
            (dict(total_seconds=8000,model_seconds={'qwen':160.,'llama':7200.}),0,'full_model_cap_llama'),
            (dict(total_seconds=284.7,model_seconds={'qwen':160.,'llama':124.}),1800,'full_setup_phase_cap')]:
        assert cap_reason(state,c['limits'],setup) == reason
        caps.append(dict(state=state,setup_elapsed=setup,reason=reason))
    return dict(real_candidate_gate_rejections=closed,primary_mask_selected_indices=[4,5,6],
                sample_metric_mean=4.,undefined_sample_propagated=True,finite_draw_boundaries=finite_cases,
                retention_boundary_cases=cases,gain_interpretation_cases=gain_cases,cap_boundary_cases=caps,
                full_suites_rerun=False,models_loaded=False,E_outputs_opened=False)


def main():
    output=Path(sys.argv[1]).resolve()
    assert not output.exists() and not output.is_relative_to(ROOT)
    output.mkdir(parents=True)
    started=time.monotonic()
    receipt=dict(reviewed_commit=CANDIDATE,base_commit=BASE,execution_producer=PRODUCER,
                 utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=socket.gethostname(),
                 python=sys.executable,version=platform.python_version(),platform=platform.platform(),
                 command=sys.argv,numerical_threads=1,RunPod_contacted=False,approval_created=False)
    try:
        receipt['identity']=identity()
        receipt['accounting']=accounting()
        with open('/tmp/truth-probing-cpu-exclusive.lock','r') as slot:
            fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
            receipt['focused']=focused()
        receipt['status']='passed'
    except BaseException as e:
        receipt.update(status='failed_preserved',error=dict(type=type(e).__name__,message=str(e)))
        raise
    finally:
        receipt['seconds']=time.monotonic()-started
        (output/'independent-checks.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status=receipt['status'],seconds=receipt['seconds'],candidate_hash=EXPECTED,
                         gate_rejections=len(receipt['focused']['real_candidate_gate_rejections']),
                         cited_metadata=len(receipt['accounting']['discovered_metadata']))))


if __name__=='__main__':
    main()
