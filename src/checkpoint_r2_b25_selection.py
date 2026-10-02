"""Frozen B25 ranking/retention. No fitting, wording inputs or score pooling."""
import numpy as np
from src.checkpoint_r2_selection_v1 import require

TOLERANCE = 1e-12
TIE_SEED = 20261008


def mean_fold_statistics(folds):
    require(len(folds) == 5 and sorted(f['fold'] for f in folds) == list(range(5)), 'five unique held-out folds required')
    require(all(f['valid_for_scoring'] for f in folds), 'invalid fold cannot disappear')
    keys = ('primary', 'atomic', 'separation', 'balanced')
    # Undefined ranking statistics remain explicit exclusions; never drop a fold.
    return {k: float(np.mean([f[k] for f in folds])) if all(f[k] is not None and np.isfinite(f[k]) for f in folds) else None for k in keys}


def rank(records, *, reference_atomic, fallback_id, arm, eligible_ids=None, balanced=False):
    require(len({r['id'] for r in records}) == len(records), 'duplicate selection identity')
    require(np.isfinite(reference_atomic), 'undefined reduced LR reference')
    declared = set(eligible_ids) if eligible_ids is not None else None
    eligibility = []
    primary_key = 'balanced' if balanced else 'primary'
    for r in sorted(records, key=lambda r: r['id']):
        reasons = []
        if not r['valid_for_scoring']:
            reasons.append('invalid_fit')
        if declared is not None and r['id'] not in declared:
            reasons.append('shared_atomic_bank_ineligible')
        if r['atomic'] is None or not np.isfinite(r['atomic']):
            reasons.append('undefined_atomic')
        elif r['atomic'] < reference_atomic - .005:
            reasons.append('atomic_retention_failed')
        if r[primary_key] is None or not np.isfinite(r[primary_key]):
            reasons.append('undefined_primary')
        if r['separation'] is None or not np.isfinite(r['separation']):
            reasons.append('degenerate_standardized_separation')
        eligibility.append(dict(id=r['id'], eligible=not reasons, reasons=reasons))
    allowed = {r['id'] for r in eligibility if r['eligible']}
    candidates = [r for r in records if r['id'] in allowed]
    trace = dict(arm=arm, status='fallback_no_eligible_candidate', selected_id=fallback_id,
                 chosen_id=None, fallback_id=fallback_id, eligibility=eligibility,
                 declared_candidates=len(records), eligible_count=len(candidates),
                 shared_atomic_eligible_ids=None if declared is None else sorted(declared),
                 tie_seed=TIE_SEED, tie_tolerance=TOLERANCE, stages=[], AND_gate=False,
                 primary='equal-weight boundary sensitivity' if balanced else 'OR mixed-versus-FF')
    if not candidates:
        return trace
    for key in (primary_key, 'atomic', 'separation'):
        best = max(r[key] for r in candidates)
        candidates = [r for r in candidates if best - r[key] <= TOLERANCE]
        trace['stages'].append(dict(statistic=key, best=best, tied_ids=sorted(r['id'] for r in candidates)))
    tied = sorted(r['id'] for r in candidates)
    chosen = str(np.random.Generator(np.random.PCG64(TIE_SEED)).choice(tied))
    trace.update(status='selected', selected_id=chosen, chosen_id=chosen, final_tie_ids=tied)
    return trace


def final_refit(trace, *, fit_id, valid, atomic, reference_atomic):
    result = dict(trace)
    passed = bool(valid and atomic is not None and np.isfinite(atomic) and atomic >= reference_atomic - .005)
    result.update(refit_id=fit_id, refit_valid=bool(valid), refit_atomic=atomic, final_refit_retention_passed=passed)
    if passed:
        result.update(selected_fit_id=fit_id)
    else:
        result.update(status='fallback_final_refit_invalid_or_retention', selected_fit_id=trace['fallback_id'])
    return result
