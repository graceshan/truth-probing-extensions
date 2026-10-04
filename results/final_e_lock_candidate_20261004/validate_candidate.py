"""Review evidence only: static candidate checks and existing synthetic suites.

No finalization, phase authorization, remote contact or E output I/O.
Run from the candidate checkout with the reviewed Python 3.11 interpreter.
"""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import resource
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
BASE = 'e8f087e53b96b2172cfcde03e3a451e6a65bb0fe'
PRODUCER = 'b2c6dde71e82f5f0af308761bf19ed3341670587'
CANDIDATE = 'config/final_e/lock_candidate_20261004.json'
EXPECTED_CANDIDATE_SHA256 = 'be5b15fe38c6c32a344e321d767d04a6f4c3b6f529ecfb338c2cc6e6edd5d1d3'
sys.path.insert(0, str(ROOT))
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT, text=True).strip()


def base_bytes(path):
    return subprocess.check_output(['git', 'show', BASE + ':' + path], cwd=ROOT)


def pin(path):
    p = Path(path)
    return dict(path=str(p.resolve()), bytes=p.stat().st_size, sha256=digest(p))


def check_contract():
    c = read(ROOT / CANDIDATE)
    original = json.loads(base_bytes('config/final_e/draft_lock_v1.json'))
    assert c['schema'] == original['schema'] == 'final-e-scientific-lock-v1'
    assert c['status'] == 'DRAFT'
    assert c['scientific_lock_approved'] is False
    assert c['claims_finalized'] is False and c['approval'] is None
    assert c['producer_sha'] == PRODUCER
    assert c['immutable_contract'] == original['immutable_contract']
    assert digest(ROOT / CANDIDATE) == EXPECTED_CANDIDATE_SHA256
    assert c['pending'] == {'independent_review_and_execution_authorization': None}
    assert set(c['resolutions']) == set(original['pending']) - set(c['pending'])
    assert all(isinstance(v, str) and v.strip() for v in c['resolutions'].values())
    assert c['limits'] == dict(model_seconds=7200, total_seconds=18000,
                             setup_phase_seconds=1800)
    assert c['carried_budget'] == original['carried_budget']
    immutable = {p: digest(ROOT / p) for p in c['immutable_contract']}
    assert immutable == c['immutable_contract']
    for p in ['config/final_e/draft_lock_v1.json', *immutable]:
        assert (ROOT / p).read_bytes() == base_bytes(p)
    for ancestor in (PRODUCER, '754ff1d7d362cf76ddb2bca681b32b2aeb7118f5'):
        subprocess.run(['git', 'merge-base', '--is-ancestor', ancestor, BASE],
                       cwd=ROOT, check=True)
    # Existing tracked files may not change in this additive delivery.
    assert git('diff', BASE, '--name-only', '--diff-filter=MDR') == ''
    t = c['interpretation_contract']
    p = t['original_primary']
    assert p['contrast'] == ['R_all', 'S_all']
    assert (p['condition'], p['operator'], p['metric'], p['scope']) == (
        'bare', 'OR', 'AUROC', 'pooled')
    assert p['positive_cells'] == ['TF', 'FT']
    assert p['negative_cells'] == ['FF'] and p['excluded_cells'] == ['TT']
    assert p['models_separately'] == ['qwen', 'llama']
    assert p['samples'] == [11, 23, 37]
    assert 'within every bootstrap draw' in p['aggregation']
    assert 'never average or concatenate scores' in p['aggregation']
    f = t['full_training_sensitivity']
    assert f['contrast'] == ['R_all', 'S_full']
    assert f['may_replace_failed_original_primary'] is False
    contrasts = read(ROOT / 'config/final_e/contrasts_v1.json')
    for pair in [f['contrast'], *f['additional_preserved_contrasts']]:
        assert pair in contrasts['full_bank']
    assert p['contrast'] in contrasts['b25']
    registry = read(ROOT / 'config/final_e/head_registry_v1.json')
    assert len(registry['heads']) == 85
    assert len(registry['constituent_cases']) == 12
    g = t['gain']
    assert g['positive'] == 'paired interval lower bound > 0'
    assert g['practical_margin'] == .02
    assert g['beyond_practical_margin'] == 'paired interval lower bound > .02'
    assert 'crosses or touches .02' in g['practical_unresolved']
    assert 'upper bound < .02' in g['below_practical_margin']
    w = t['topic_weighting']
    assert w['primary'] == 'pooled' and 'lower bound > 0' in w['additional_equal_topic_claim']
    for key in ('macro_is_coprimary', 'macro_practical_margin_required',
                'every_topic_or_sample_significant_required'):
        assert w[key] is False
    r = t['retention']
    assert r['contrast'] == ['R_all', 'reduced_LR']
    assert r['margins'] == {'atomic': .005, 'AND': .02}
    assert r['required_scopes'] == ['pooled', 'topic_macro']
    assert 'lower bound >= negative loss margin' in r['supported']
    assert 'upper bound < negative loss margin' in r['failure']
    i = t['inference']
    frozen = read(ROOT / 'config/final_e/preparation_v1.json')['bootstrap']
    assert (i['replicates'], i['seed'], i['rng'], i['minimum_finite_draws']) == (
        frozen['replicates'], frozen['seed'], frozen['rng'], frozen['minimum_valid'])
    assert 'separate atomic (294 people) and compound (225 people)' in i['frames']
    assert i['compound_row_weights'].startswith('product of endpoint entity multiplicities')
    assert 'all five fixed topics' in i['macro']
    assert 'no redraw, imputation or topic dropping' in i['undefined_handling']
    assert i['joint_atomic_compound_confidence_claim'] is False
    assert i['simultaneous_confidence_claim'] is False
    assert 'pointwise' in i['interval'] and 'conditional' in i['interval']
    scope = t['scope']
    assert scope['deferred'] == ['behavior', 'matched chat', 'calibration',
                                 'causal work', 'loss-deletion fits']
    assert 'R_fixed' in scope['deletion_ablation_note'] and 'R_all' in scope['deletion_ablation_note']
    assert 'atomic-loss term causes retention' in scope['omitted_claims']
    a = t['resource_accounting']
    carry = read(ROOT / c['carried_budget'])
    assert a['cpu_job_cap_seconds'] == carry['cpu_cap_seconds'] == 10800
    assert a['carried_model_seconds'] == carry['charged_seconds']
    assert a['remaining_model_seconds'] == carry['remaining_seconds']
    assert a['carried_total_seconds'] == sum(carry['charged_seconds'].values())
    # Directly submit the unaltered real candidate to every declared gate phase.
    # Empty authorization is deliberate; no real authorization file is created.
    from src.final_e_access import PHASES, validate_gate
    closed = []
    for phase in PHASES:
        for model in ('qwen', 'llama'):
            try:
                validate_gate(c, {}, lock_sha256=digest(ROOT / CANDIDATE),
                              authorization_sha256='0' * 64, phase=phase,
                              producer=PRODUCER, output='/nonexistent/E-must-not-open',
                              model=model)
            except ValueError as exc:
                assert str(exc) == 'E CLOSED: finalized scientific lock required'
                closed.append(dict(phase=phase, model=model, rejection=str(exc)))
            else:
                raise AssertionError('real DRAFT candidate passed E gate')
    return dict(candidate_sha256=digest(ROOT / CANDIDATE), immutable_hashes=immutable,
                original_draft_sha256=digest(ROOT / 'config/final_e/draft_lock_v1.json'),
                existing_tracked_files_unchanged=True, registry_heads=85,
                constituent_cases=12, real_candidate_gate_rejections=closed)


def accounting():
    project = Path('/Users/apple/projects/truth-probing-extensions')
    backup = Path('/Users/apple/truth-probing-backups')
    search_roots = [project / '.worktrees', backup]
    carry = read(ROOT / 'config/final_e/carryforward_v1.json')
    source_pins = {}
    for path, sha in carry['source_files'].items():
        assert digest(ROOT / path) == sha
        source_pins[path] = pin(ROOT / path)
    original = project / '.worktrees/checkpoint-next-gpu-execution-9b0ac1d/handoffs/preflight'
    successor = project / '.worktrees/checkpoint-next-raw-only-execution-6865bfa/handoff'
    known = {x['attempt']: (model, x['active_seconds'], x['start_sha256'], x['end_sha256'])
             for model, records in carry['old_attempts_preserved'].items() for x in records}
    successor_id = '4ca9f0fcabc04ed5bb6b8ec955168fa7'
    end = successor / 'attempts/llama' / successor_id / 'end.json'
    known[successor_id] = ('llama', carry['successor_attempt']['active_seconds'],
                           digest(end.with_name('start.json')), digest(end))
    attempts = []
    totals = {'qwen': 0., 'llama': 0.}
    for attempt, (model, seconds, start_sha, end_sha) in known.items():
        folder = (successor if attempt == successor_id else original) / 'attempts' / model / attempt
        assert digest(folder / 'start.json') == start_sha
        assert digest(folder / 'end.json') == end_sha
        actual = read(folder / 'end.json')
        assert actual['status'] == 'finished' and actual['active_seconds'] == seconds
        totals[model] += seconds
        attempts.append(dict(attempt=attempt, model=model, seconds=seconds,
                             start=pin(folder / 'start.json'), end=pin(folder / 'end.json')))
    assert totals == carry['charged_seconds']
    adoption = successor / 'adoption.json'
    report = project / '.worktrees/checkpoint-next-raw-only-execution-6865bfa/execution-report.json'
    assert digest(adoption) == carry['source_files']['results/final_e_preparation_20261003/accounting-evidence/adoption.json']
    assert digest(report) == carry['source_files']['results/final_e_preparation_20261003/accounting-evidence/execution-report.json']
    assert {m: v['charged_seconds'] for m, v in read(report)['final_budgets'].items()} == totals
    inspected, unexpected = [], []
    # Scan only accounting metadata names; do not open activation or prediction arrays.
    for search in search_roots:
        for directory, dirs, files in os.walk(search):
            here = Path(directory)
            dirs[:] = [d for d in dirs if d not in ('.git', 'shards', 'snapshots', 'weights', 'node_modules')
                       and (here / d).resolve() != ROOT]
            for name in files:
                p = here / name
                is_attempt = 'attempts' in p.parts and name in ('start.json', 'end.json', 'active.json', 'supervisor-exit.json')
                is_campaign = name in ('full-budget.json', 'budget-adoption.json', 'qwen-campaign.json', 'llama-campaign.json', 'campaign.json')
                if not (is_attempt or is_campaign):
                    continue
                inspected.append(pin(p))
                if is_attempt and p.parent.parent.name in ('qwen', 'llama'):
                    if p.parent.name not in known:
                        unexpected.append(str(p))
                    elif name == 'end.json':
                        assert digest(p) == known[p.parent.name][3], 'altered duplicate attempt receipt'
                if name in ('budget-adoption.json',) or (is_campaign and read(p).get('schema') == 'final-e-raw-campaign-v1'):
                    unexpected.append(str(p))
    # Distinct earlier full-raw and CPU campaigns are recorded, not reset/reclassified.
    older_root = backup / '20261002T190142Z-r2-full-raw-7656a0e-resized/output'
    older = read(older_root / 'campaign.json')
    assert older['commit'] == '7656a0eb68a5fd7ff7dac1c73efeab982d540cc2'
    older_budget = read(older_root / 'full-budget.json')
    assert older_budget['running'] is False
    cpu = backup / '20261001T164224Z/full-bank-execution-20261004T002800Z/bank/attempts/955a1cb283464a72bb99b06e66dd69a6/end.json'
    assert read(cpu)['status'] == 'finished'
    result = dict(status='local_receipts_reconciled_no_intervening_campaign_charge_found' if not unexpected else 'adoption_unresolved',
                  roots_inspected=[str(p) for p in search_roots], metadata_pins=inspected,
                  unique_carried_attempts=attempts, duplicate_copies_not_double_charged=True,
                  carryforward_sha256=digest(ROOT / 'config/final_e/carryforward_v1.json'),
                  immutable_source_pins=source_pins, local_adoption=pin(adoption), local_execution_report=pin(report),
                  charged_seconds=totals, remaining_seconds=carry['remaining_seconds'],
                  total_seed_seconds=sum(totals.values()), unexpected_accounting_metadata=unexpected,
                  separate_historical_full_raw=dict(campaign=pin(older_root / 'campaign.json'),
                      ledger=pin(older_root / 'full-budget.json'), ledger_values=older_budget,
                      interpretation='separate earlier raw campaign; excluded by frozen E carryforward contract, preserved without reset'),
                  separate_historical_full_bank_cpu=dict(receipt=pin(cpu), values=read(cpu)),
                  budget_adoption_performed=False, remote_state_verified=False,
                  later_recheck_required=True)
    write(OUT / 'accounting.json', result)
    assert not unexpected, 'new/unreconciled accounting: adoption remains unresolved'
    return dict(status=result['status'], receipt_sha256=digest(OUT / 'accounting.json'))


def main():
    started = time.monotonic()
    command = [sys.executable, '-m', 'unittest', '-v',
               'tests.test_final_e_locked_path', 'tests.test_final_e_review']
    receipt = dict(base_sha=BASE, execution_producer_sha=PRODUCER,
                   checkout_head_at_validation=git('rev-parse', 'HEAD'),
                   branch=git('branch', '--show-current'), worktree=str(ROOT),
                   hostname=socket.gethostname(), platform=platform.platform(),
                   interpreter=sys.executable, python_version=platform.python_version(),
                   numerical_threads=1, E_outputs_opened=False, RunPod_contacted=False,
                   model_loading=False, new_fits=0, full_saved_D_reproduction=False,
                   real_phase_authorization_created=False, real_candidate_finalized=False)
    try:
        with open('/tmp/truth-probing-cpu-exclusive.lock', 'r') as slot:
            fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt['candidate_checks'] = check_contract()
            receipt['accounting'] = accounting()
            t = time.monotonic()
            with (OUT / 'focused-tests.log').open('w') as log:
                result = subprocess.run(command, cwd=ROOT, env=os.environ,
                                        stdout=log, stderr=subprocess.STDOUT)
            receipt['focused_tests'] = dict(command=command, returncode=result.returncode,
                seconds=time.monotonic() - t, log_sha256=digest(OUT / 'focused-tests.log'),
                peak_child_rss_bytes=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
            assert result.returncode == 0, 'focused suite failed'
            assert 'Ran 34 tests' in (OUT / 'focused-tests.log').read_text()
            receipt['status'] = 'passed_candidate_closed'
    except BaseException as exc:
        receipt.update(status='failed_preserved', error=dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        receipt['total_seconds'] = time.monotonic() - started
        write(OUT / 'validation.json', receipt)
    print(json.dumps(dict(status=receipt['status'], focused_tests=34,
                         real_candidate_rejections=len(receipt['candidate_checks']['real_candidate_gate_rejections']),
                         immutable_pins=6, accounting=receipt['accounting']['status'],
                         seconds=receipt['total_seconds'])))


if __name__ == '__main__':
    main()
