"""Post-hoc reduced-bank E analysis primitives. No fitting or procedure selection."""
import numpy as np
from src.clean_transfer_statistics import weighted_auc

TOLERANCE = 1e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_bank(records, expected_specs):
    """Exact declared identities/settings, not just the number of available files."""
    expected = {s['candidate_id']: s for s in expected_specs}
    ids = [r['id'] for r in records]
    require(len(ids) == len(set(ids)) == len(expected), 'missing or duplicate bank candidate')
    require(set(ids) == set(expected), 'candidate identity coverage differs')
    for r in records:
        require('/P15/' in r['id'] and r['spec'] == expected[r['id']], 'not original P15 candidate/settings')
        require(r['valid_for_scoring'] is True, 'invalid candidate blocks complete-bank maximum')


def validate_order(actual, expected):
    require(actual == expected, 'E row identity/order/label binding differs')
    require(len({r['example_id'] for r in actual}) == len(actual), 'duplicate E identity')


def boundary_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    require(labels.ndim == scores.ndim == 1 and labels.shape == scores.shape, 'score/label alignment')
    require(np.isfinite(scores).all() and set(labels.tolist()) == {0, 1}, 'nonfinite scores or invalid labels')
    return weighted_auc(labels, scores)


def maxima(records, eligible_only):
    """Return descriptive E maxima; preserve exact and historical-tolerance ties."""
    require(all(type(r['eligible']) is bool for r in records), 'eligibility must be original Boolean decision')
    eligible = [r for r in records if r['eligible'] or not eligible_only]
    require(bool(eligible), 'no candidates for maximum')
    require(all(r['status'] == 'scored' and np.isfinite(r['E_auroc']) for r in records), 'unscored candidate')
    best = max(r['E_auroc'] for r in eligible)
    return dict(maximum=best,
                exact_winner_ids=sorted(r['candidate_id'] for r in eligible if r['E_auroc'] == best),
                tolerance_tie_ids=sorted(r['candidate_id'] for r in eligible if best-r['E_auroc'] <= TOLERANCE))


def independent_auc(labels, scores):
    """Separate Mann--Whitney average-rank implementation for audit and fixtures."""
    from scipy.stats import rankdata
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    require(labels.shape == scores.shape and labels.ndim == 1 and np.isfinite(scores).all(), 'audit alignment/finite')
    require(set(labels.tolist()) == {0, 1}, 'audit needs both classes')
    positives = int(labels.sum())
    negatives = len(labels)-positives
    return float((rankdata(scores, method='average')[labels == 1].sum()
                  - positives*(positives+1)/2)/(positives*negatives))
