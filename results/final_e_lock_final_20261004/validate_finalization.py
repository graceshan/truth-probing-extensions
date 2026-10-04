"""Small finalization checks; no historical validator, remote or E output I/O.

An optional output-directory argument permits reproduction into fresh evidence.
The default is this new finalization-evidence directory; existing receipts are
never overwritten. Only three existing focused access tests run, not a suite
of substantial numerical work while the separate correlation job is active.
"""
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parent
ROOT = SOURCE.parents[1]
OUT = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else SOURCE
REVIEW = '523e756346b373cc6e8be1d788901e5b3291c1d3'
CANDIDATE_DELIVERY = 'c2efef5fef91b989d66c3b6228d2ec5e26c2c0f4'
CANDIDATE_PATH = 'config/final_e/lock_candidate_20261004.json'
CANDIDATE_HASH = 'be5b15fe38c6c32a344e321d767d04a6f4c3b6f529ecfb338c2cc6e6edd5d1d3'
FINAL_PATH = 'config/final_e/final_lock_20261004.json'
PRODUCER = 'b2c6dde71e82f5f0af308761bf19ed3341670587'
QUESTION = 'Do you approve this exact candidate, including its interpretation and caps, and authorize lock finalization?'
REPLY = 'Yes I approve.'
ADMIN = {'status', 'scientific_lock_approved', 'claims_finalized', 'approval',
         'pending', 'resolutions', 'note'}
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pin(path):
    p = Path(path)
    return dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=sha(p))


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def blob(ref, path):
    return subprocess.check_output(['git', 'show', ref + ':' + path], cwd=ROOT)


def check_identity():
    assert sha(ROOT / CANDIDATE_PATH) == CANDIDATE_HASH
    assert (ROOT / CANDIDATE_PATH).read_bytes() == blob(CANDIDATE_DELIVERY, CANDIDATE_PATH)
    c, f = read(ROOT / CANDIDATE_PATH), read(ROOT / FINAL_PATH)
    assert set(c) == set(f)
    changed = {k: dict(candidate=c[k], final=f[k]) for k in c if c[k] != f[k]}
    assert set(changed) == ADMIN
    assert read(SOURCE / 'administrative-diff.json')['changed_fields'] == changed
    for k in set(c) - ADMIN:
        assert f[k] == c[k], k
    assert f['interpretation_contract'] == c['interpretation_contract']
    assert f['candidate_preparation'] == c['candidate_preparation']
    assert f['candidate_preparation']['exact_final_lock_approved'] is False
    assert f['candidate_preparation']['E_execution_authorized'] is False
    assert f['status'] == 'FINAL' and f['scientific_lock_approved'] is True
    assert f['claims_finalized'] is True and f['pending'] == {}
    assert f['producer_sha'] == c['producer_sha'] == PRODUCER
    assert f['limits'] == c['limits'] == dict(model_seconds=7200, total_seconds=18000, setup_phase_seconds=1800)
    assert f['carried_budget'] == c['carried_budget']
    required = set(read(ROOT / 'config/final_e/preparation_v1.json')['pending'])
    assert set(f['resolutions']) == required
    assert all(isinstance(v, str) and v.strip() for v in f['resolutions'].values())
    for k, v in c['resolutions'].items():
        prefix = 'PROPOSED FOR REVIEW, NOT FINAL APPROVAL: ' if k == 'comparator_contrast_approval' else 'PROPOSED FOR REVIEW: '
        assert v.startswith(prefix)
        assert f['resolutions'][k] == 'APPROVED BY USER/PROJECT OWNER ON 2026-10-04: ' + v[len(prefix):]
    review_resolution = f['resolutions']['independent_review_and_execution_authorization']
    for item in [REVIEW, CANDIDATE_DELIVERY, CANDIDATE_HASH, 'remain mandatory', 'have not been issued', 'budget adoption remain unauthorized']:
        assert item in review_resolution
    a = f['approval']
    assert a['reviewer'] == 'user/project owner'
    assert a['reviewed_delivery_sha'] == CANDIDATE_DELIVERY
    assert a['independent_review_sha'] == REVIEW
    assert a['approved_candidate_path'] == CANDIDATE_PATH
    assert a['approved_candidate_sha256'] == CANDIDATE_HASH
    assert (a['approval_question'], a['approval_reply'], a['approval_date']) == (QUESTION, REPLY, '2026-10-04')
    assert 'does not authorize E execution, budget adoption or phase authorization' in a['instruction']
    assert 'current finalization instruction' in a['record_source']
    assert not ({'signature', 'message_id', 'personal_name'} & set(a))
    assert 'does not assert that E supports' in f['note']
    assert sha(ROOT / FINAL_PATH) not in (ROOT / FINAL_PATH).read_text()
    immutable = {p: sha(ROOT / p) for p in c['immutable_contract']}
    assert len(immutable) == 6 and immutable == f['immutable_contract'] == c['immutable_contract']
    for p in ['config/final_e/draft_lock_v1.json', CANDIDATE_PATH, *immutable]:
        assert (ROOT / p).read_bytes() == blob(REVIEW, p)
    inventory_path = 'results/final_e_lock_review_20261004/evidence-inventory.json'
    inv = read(ROOT / inventory_path)
    checked = {}
    for name, expected in inv['files'].items():
        p = ROOT / 'results/final_e_lock_review_20261004' / name
        assert p.stat().st_size == expected['bytes'] and sha(p) == expected['sha256']
        assert p.read_bytes() == blob(REVIEW, str(p.relative_to(ROOT)))
        checked[name] = expected
    review_doc = ROOT / 'docs/final_e/SCIENTIFIC_LOCK_REVIEW_20261004.md'
    assert sha(review_doc) == inv['review_document_sha256']
    assert '**PASS' in review_doc.read_text()
    assert subprocess.check_output(['git', 'diff', REVIEW, '--name-only', '--diff-filter=MDR'], cwd=ROOT, text=True).strip() == ''
    for ancestor, descendant in [(PRODUCER, CANDIDATE_DELIVERY), (CANDIDATE_DELIVERY, REVIEW)]:
        subprocess.run(['git', 'merge-base', '--is-ancestor', ancestor, descendant], cwd=ROOT, check=True)
    return dict(candidate_sha256=CANDIDATE_HASH, final_lock_sha256=sha(ROOT / FINAL_PATH),
                administrative_fields_changed=sorted(ADMIN), other_fields_equal=True,
                immutable_pins=immutable, review_inventory=checked,
                review_document_sha256=sha(review_doc), approval_provenance=a,
                original_tracked_files_unchanged=True)


def check_accounting():
    historical = read(ROOT / 'results/final_e_lock_candidate_20261004/accounting.json')
    carry = read(ROOT / 'config/final_e/carryforward_v1.json')
    blockers = []
    for p in historical['metadata_pins']:
        actual = Path(p['path'])
        if not actual.is_file() or actual.stat().st_size != p['bytes'] or sha(actual) != p['sha256']:
            blockers.append(dict(issue='reviewed_local_metadata_changed_or_missing', path=p['path']))
    totals = {'qwen': 0., 'llama': 0.}
    for attempt in historical['unique_carried_attempts']:
        for field in ('start', 'end'):
            p = attempt[field]
            assert Path(p['path']).stat().st_size == p['bytes'] and sha(p['path']) == p['sha256']
        e = read(attempt['end']['path'])
        assert e['status'] == 'finished' and e['active_seconds'] == attempt['seconds']
        totals[attempt['model']] += e['active_seconds']
    assert totals == carry['charged_seconds']
    for p, h in carry['source_files'].items():
        assert sha(ROOT / p) == h
    for key in ('local_adoption', 'local_execution_report'):
        p = historical[key]
        assert Path(p['path']).stat().st_size == p['bytes'] and sha(p['path']) == p['sha256']
    report = read(historical['local_execution_report']['path'])
    assert {m: v['charged_seconds'] for m, v in report['final_budgets'].items()} == totals
    observed = {}
    for search in historical['roots_inspected']:
        for directory, dirs, files in os.walk(search):
            here = Path(directory)
            dirs[:] = [d for d in dirs if d not in ('.git', 'shards', 'snapshots', 'weights', 'node_modules') and (here / d).resolve() != ROOT]
            for name in files:
                p = here / name
                attempt = 'attempts' in p.parts and name in ('start.json', 'end.json', 'active.json', 'supervisor-exit.json')
                campaign = name in ('full-budget.json', 'budget-adoption.json', 'qwen-campaign.json', 'llama-campaign.json', 'campaign.json')
                if attempt or campaign:
                    observed[str(p.resolve())] = pin(p)
    expected = {p['path']: p for p in historical['metadata_pins']}
    for p in sorted(set(observed) - set(expected)):
        blockers.append(dict(issue='new_accounting_metadata_requires_review_before_adoption', path=p))
    for p in sorted(set(expected) - set(observed)):
        blockers.append(dict(issue='reviewed_accounting_metadata_absent_from_census', path=p))
    result = dict(status='local_receipts_reconciled_no_intervening_charge_found' if not blockers else 'adoption_blocked_pending_accounting_review',
                  roots_inspected=historical['roots_inspected'], metadata_pins=observed,
                  unique_carried_attempts=historical['unique_carried_attempts'], charged_seconds=totals,
                  remaining_seconds=carry['remaining_seconds'], carried_total_seed=sum(totals.values()),
                  carryforward_sha256=sha(ROOT / 'config/final_e/carryforward_v1.json'),
                  cpu_job_cap_seconds=carry['cpu_cap_seconds'], operational_blockers=blockers,
                  duplicate_copies_not_double_charged=True, other_historical_campaigns_preserved=True,
                  adoption_performed=False, remote_state_verified=False, later_recheck_required=True)
    write(OUT / 'accounting.json', result)
    return dict(status=result['status'], metadata_files=len(observed), operational_blockers=blockers)


def check_access():
    from src.final_e_access import PHASES, authorize, validate_gate
    c, f = read(ROOT / CANDIDATE_PATH), read(ROOT / FINAL_PATH)
    results = []
    for phase in PHASES:
        for model in ('qwen', 'llama', 'both'):
            args = SimpleNamespace(lock=str(ROOT / FINAL_PATH), lock_sha256=sha(ROOT / FINAL_PATH),
                authorization=None, authorization_sha256=None, command=phase,
                model=model, output='/nonexistent/E-must-not-open')
            # Operational interface must stop before opening any files/outputs.
            with patch('src.final_e_access.file_hash', side_effect=AssertionError('unexpected I/O')):
                try:
                    authorize(args, phase=phase)
                except ValueError as exc:
                    assert str(exc) == 'E CLOSED: lock and independent phase authorization pins required'
                    operational_error = str(exc)
                else:
                    raise AssertionError('real final lock ran without phase authorization')
            kw = dict(lock_sha256=sha(ROOT / FINAL_PATH), authorization_sha256='0' * 64,
                      phase=phase, producer=PRODUCER, output=args.output, model=model)
            try:
                # Unapproved in-memory negative fixture; never written or approved.
                validate_gate(f, {'schema': 'final-e-phase-authorization-v1', 'approved': False}, **kw)
            except ValueError as exc:
                assert str(exc) == 'separate scoped execution authorization'
                final_gate_error = str(exc)
            else:
                raise AssertionError('real final lock accepted unapproved authorization')
            try:
                validate_gate(c, {}, **dict(kw, lock_sha256=CANDIDATE_HASH))
            except ValueError as exc:
                assert str(exc) == 'E CLOSED: finalized scientific lock required'
                candidate_error = str(exc)
            else:
                raise AssertionError('real candidate passed gate')
            results.append(dict(phase=phase, model=model, final_operational_rejection=operational_error,
                                final_unapproved_fixture_rejection=final_gate_error,
                                candidate_rejection=candidate_error))
    return results


def boundary_checks():
    margin = read(ROOT / FINAL_PATH)['interpretation_contract']['retention']['margins']
    def retention(lo, hi, m):
        if lo is None or hi is None:
            return 'unresolved'
        return 'supported' if lo >= -m else ('failure' if hi < -m else 'unresolved')
    results = []
    for endpoint, m in margin.items():
        for lo, hi, expected in [(-m, .01, 'supported'), (-m, -m, 'supported'),
                                  (-m-.01, -m, 'unresolved'), (-m-.01, -m+.01, 'unresolved'),
                                  (-m-.02, -m-.01, 'failure'), (None, None, 'unresolved')]:
            assert retention(lo, hi, m) == expected
            results.append(dict(endpoint=endpoint, lower=lo, upper=hi, result=expected))
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ('validation.json', 'focused-access-tests.log', 'accounting.json'):
        assert not (OUT / name).exists(), 'fresh output required: ' + str(OUT / name)
    receipt = dict(review_base_sha=REVIEW, candidate_delivery_sha=CANDIDATE_DELIVERY,
                   execution_producer_sha=PRODUCER, worktree=str(ROOT),
                   hostname=socket.gethostname(), platform=platform.platform(),
                   interpreter=sys.executable, python_version=platform.python_version(),
                   numerical_threads=1, active_correlation_job_preserved=True,
                   substantial_checks_deferred=True, historical_validators_rerun=False,
                   RunPod_contacted=False, E_outputs_opened=False, model_loading=False,
                   new_fits=0, full_saved_D_reproduction=False, budget_adoption=False,
                   approved_real_phase_authorization_created=False)
    start = time.monotonic()
    try:
        with open('/tmp/truth-probing-cpu-exclusive.lock', 'r') as slot:
            fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt['identity'] = check_identity()
            receipt['accounting'] = check_accounting()
            receipt['access_closure'] = check_access()
            receipt['retention_boundaries'] = boundary_checks()
            names = ['tests.test_final_e_locked_path.ClosedGateTests.' + name for name in (
                'test_no_authorization_blocks_before_output_IO', 'test_draft_cannot_execute',
                'test_hash_scope_producer_and_unresolved_decisions')]
            suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(n) for n in names)
            log = io.StringIO()
            t = time.monotonic()
            result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
            (OUT / 'focused-access-tests.log').write_text(log.getvalue())
            receipt['focused_access_tests'] = dict(names=names, count=result.testsRun,
                errors=len(result.errors), failures=len(result.failures), seconds=time.monotonic()-t,
                log_sha256=sha(OUT / 'focused-access-tests.log'), positive_fixtures='synthetic only')
            assert result.wasSuccessful() and result.testsRun == 3
            receipt['status'] = 'passed_final_scientific_lock_execution_still_closed'
    except BaseException as exc:
        receipt.update(status='failed_preserved', error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        receipt['seconds'] = time.monotonic() - start
        write(OUT / 'validation.json', receipt)
    print(json.dumps(dict(status=receipt['status'], final_lock_sha256=receipt['identity']['final_lock_sha256'],
                         focused_access_tests=3, operational_final_rejections=len(receipt['access_closure']),
                         real_candidate_rejections=len(receipt['access_closure']),
                         immutable_pins=6, accounting=receipt['accounting'], seconds=receipt['seconds'])))


if __name__ == '__main__':
    main()
