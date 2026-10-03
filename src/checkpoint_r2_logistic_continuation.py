"""Successor-only logistic numerics; unchanged first attempt and scientific loss."""
from contextlib import contextmanager
import warnings

import numpy as np
from scipy import optimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from src.checkpoint_r2_fresh_inputs import require
from src.selection_repair_canonical_sensitivity import gradient_diagnostics
from src import selection_repair_capacity_methods_v1 as original

SETTINGS = dict(gtol=1e-4, ftol=0.0, maxls=50, maxcor=10, maxfun=50000)
TOTAL_ITERATIONS = 10000


def loss_gradient(theta, X, y, C):
    """Mean BCE + ||w||²/(2*C*n); C=inf gives mean unregularized BCE."""
    theta = np.asarray(theta, dtype=np.float64)
    n = len(y); w, intercept = theta[:-1], theta[-1]
    logits = X @ w + intercept
    residual = expit(logits) - y
    loss = np.logaddexp(0, np.where(y == 1, -logits, logits)).mean()
    loss += np.dot(w, w) / (2 * C * n)
    gradient = np.r_[X.T @ residual / n + w / (C * n), residual.mean()]
    return float(loss), gradient


def diagnostics(theta, X, y, C):
    # Acceptance is independent of the analytic objective supplied to SciPy.
    return gradient_diagnostics(X, y, theta[:-1], float(theta[-1]), C)


def accepted(attempt):
    return bool(attempt['library_success'] and attempt['finite'] and
                attempt['gradient_infinity_norm'] <= SETTINGS['gtol'])


@contextmanager
def capture_minimize(records):
    """Observe sklearn's unchanged optimizer call; one process/thread only."""
    original_minimize = optimize.minimize
    def observed(*args, **kwargs):
        result = original_minimize(*args, **kwargs)
        records.append(dict(library_success=bool(result.success), termination_code=int(result.status),
                            termination_message=str(result.message), n_iter=int(result.nit),
                            function_evaluations=int(result.nfev)))
        return result
    optimize.minimize = observed
    try:
        yield
    finally:
        optimize.minimize = original_minimize


def first_attempt(X, y, C):
    """Original zero-initialized sklearn call, without the old cold retry."""
    initial_budget = 100 if np.isinf(C) else 2000
    model = LogisticRegression(C=1.0 if np.isinf(C) else C,
                               penalty=None if np.isinf(C) else 'l2',
                               fit_intercept=True, solver='lbfgs', tol=1e-4,
                               max_iter=initial_budget, class_weight=None)
    observed = []
    with capture_minimize(observed), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always'); model.fit(X, y)
    require(len(observed) == 1, 'one original first attempt required')
    theta = np.r_[model.coef_[0], float(model.intercept_[0])]
    attempt = dict(observed[0], phase='original_first_attempt', maximum_iterations=initial_budget,
                   initialization='original sklearn zero initialization',
                   warnings=[str(w.message) for w in caught], **diagnostics(theta, X, y, C))
    return theta, attempt


class EvaluationLimit(Exception):
    pass


def complete(X, y, C, theta, attempts):
    """At most one warm continuation, counting all original spent iterations."""
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
    theta = np.asarray(theta, dtype=np.float64).copy()
    require(X.ndim == 2 and y.shape == (len(X),) and theta.shape == (X.shape[1]+1,), 'logistic shapes')
    require(np.isfinite(X).all() and set(np.unique(y)) == {0, 1} and C > 0, 'logistic inputs')
    attempts = [dict(a) for a in attempts]
    require(bool(attempts) and all(a['n_iter'] >= 0 for a in attempts), 'spent iterations required')
    spent = sum(a['n_iter'] for a in attempts)
    require(spent <= TOTAL_ITERATIONS, 'prior attempts exceed cumulative budget')
    # Recompute the acceptance diagnostic even for already-valid imported heads.
    attempts[-1].update(diagnostics(theta, X, y, C))
    if not accepted(attempts[-1]) and not any(a['phase'] == 'warm_continuation' for a in attempts):
        remaining = TOTAL_ITERATIONS - spent
        initial = diagnostics(theta, X, y, C); initial_theta = theta.copy()
        if remaining and initial['finite']:
            calls = 0; last_theta = theta.copy()
            def bounded_objective(value):
                nonlocal calls, last_theta
                if calls >= SETTINGS['maxfun']:
                    raise EvaluationLimit('hard function evaluation limit')
                calls += 1; last_theta = value.copy()
                return loss_gradient(value, X, y, C)
            try:
                result = optimize.minimize(bounded_objective, initial_theta, method='L-BFGS-B', jac=True,
                                           options=dict(SETTINGS, maxiter=remaining))
                theta = np.asarray(result.x, dtype=np.float64)
                status = dict(library_success=bool(result.success), termination_code=int(result.status),
                              termination_message=str(result.message), n_iter=int(result.nit))
            except EvaluationLimit:
                theta = last_theta
                # Iteration count is unknown after hard interruption: charge all
                # remaining budget so this failed subfit can never be restarted.
                status = dict(library_success=False, termination_code=None,
                              termination_message='hard function evaluation limit', n_iter=remaining,
                              iteration_count_policy='unknown at interruption; remaining budget charged')
            attempts.append(dict(status, phase='warm_continuation', maximum_iterations=remaining,
                settings=dict(SETTINGS), function_evaluations=calls,
                warm_start_sha256=original_array_hash(initial_theta), before=initial,
                **diagnostics(theta, X, y, C)))
        else:
            attempts.append(dict(phase='warm_continuation', library_success=False, n_iter=0,
                maximum_iterations=remaining, settings=dict(SETTINGS), function_evaluations=0,
                termination_code=None, termination_message='budget exhausted or nonfinite warm start',
                before=initial, **initial))
    require(sum(a['n_iter'] for a in attempts) <= TOTAL_ITERATIONS, 'continuation exceeded cumulative iteration budget')
    details = dict(attempts=attempts, valid_for_scoring=accepted(attempts[-1]),
                   total_iterations=sum(a['n_iter'] for a in attempts), total_iteration_limit=TOTAL_ITERATIONS,
                   objective='mean BCE + ||w||^2/(2*C*n); free intercept; C=inf unregularized',
                   numerical_policy='r2-b25-one-warm-continuation-v1')
    return dict(coef=theta[:-1], intercept=np.array(theta[-1])), details


def original_array_hash(array):
    import hashlib
    return hashlib.sha256(np.asarray(array, dtype=np.float64).tobytes()).hexdigest()


def logistic(X, y, C):
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
    theta, first = first_attempt(X, y, C)
    return complete(X, y, C, theta, [first])


def fit(method, X, y, rows, C, layer):
    if method == 'l2_logistic':
        return logistic(X, y, C)
    if method != 'ttpd':
        return original.fit(method, X, y, rows, C, layer)
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y, dtype=np.int64)
    polarity_labels = np.array([r['form'] == 'affirmative' for r in rows], dtype=np.int64)
    head = original.old.learn_truth_directions(X, y, 2*polarity_labels-1,
                                              np.array([r['dataset'] for r in rows]))
    polarity, polarity_record = logistic(X, polarity_labels, np.inf)
    # A failed subfit cannot feed a downstream logistic objective or predictions.
    if not polarity_record['valid_for_scoring']:
        return dict(head.parameters, polarity_coef=polarity['coef'], polarity_intercept=polarity['intercept']), dict(
            valid_for_scoring=False, polarity_optimizer=polarity_record, head_optimizer=None,
            truth_fit=head.details, failure='invalid polarity subfit; no composition or predictions')
    features = np.column_stack((X @ head.parameters['t_g'], X @ polarity['coef']))
    truth, truth_record = logistic(features, y, np.inf)
    params = dict(head.parameters, coef=truth['coef'][0]*head.parameters['t_g']+truth['coef'][1]*polarity['coef'],
                  intercept=truth['intercept'], polarity_coef=polarity['coef'],
                  polarity_intercept=np.array([polarity['intercept']]), polarity_classes=np.array([0, 1]),
                  head_coef=truth['coef'], head_intercept=truth['intercept'], classes=np.array([0, 1]))
    return params, dict(valid_for_scoring=truth_record['valid_for_scoring'], polarity_optimizer=polarity_record,
                        head_optimizer=truth_record, truth_fit=head.details,
                        inference='raw X@t_G and X@polarity coefficient; no form inputs, intercept projection or eval centering')
