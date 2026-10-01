"""Synthetic-validated shared readout numerics; no data loading or membership policy."""
from dataclasses import dataclass
from typing import Literal, Sequence
import warnings

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

Objective = Literal['r0', 'compound', 'isolated']
PENALTY = 0.001
STD_FLOOR = 1e-6


def _readonly(values, dtype=np.float64):
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _matrix(values):
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or min(result.shape) == 0 or not np.isfinite(result).all():
        raise ValueError('features must be a nonempty finite rows-by-coordinates matrix')
    return result


@dataclass(frozen=True)
class PPreprocessing:
    """Per-layer P-only statistics. Construct with fit_p_preprocessing."""

    mean: np.ndarray
    population_std: np.ndarray
    denominator: np.ndarray
    constant_mask: np.ndarray
    p_rows: int
    fit_source: str = 'P only'
    policy: str = 'float64; ddof=0; denominator=max(std,1e-6); zero-P-variance outputs zero'

    def transform(self, features):
        x = _matrix(features)
        if x.shape[1] != len(self.mean):
            raise ValueError('feature dimension differs from fitted P preprocessing')
        # Do not subtract or divide constant coordinates, even for distant A/eval values.
        transformed = np.zeros_like(x, dtype=np.float64)
        active = ~self.constant_mask
        transformed[:, active] = (x[:, active] - self.mean[active]) / self.denominator[active]
        if not np.isfinite(transformed).all():
            raise ValueError('nonfinite standardized features')
        return transformed


def fit_p_preprocessing(p_features):
    """Call once per layer with P alone; reuse this object for A and evaluation."""
    p = _matrix(p_features)
    mean = p.mean(axis=0, dtype=np.float64)
    std = p.std(axis=0, ddof=0, dtype=np.float64)
    # Exact constants can acquire rounding noise in mean/std reduction (e.g. 0.1).
    constant = np.all(p == p[0], axis=0) | (std == 0)
    std[constant] = 0.0
    if not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError('nonfinite P statistics')
    return PPreprocessing(_readonly(mean), _readonly(std),
                          _readonly(np.maximum(std, STD_FLOOR)),
                          _readonly(constant, bool), len(p))


@dataclass(frozen=True)
class PreparedBlock:
    """Validated transformed features, binary targets and normalized row weights."""

    features: np.ndarray
    targets: np.ndarray
    weights: np.ndarray
    preprocessing: PPreprocessing

    def __post_init__(self):
        x = _matrix(self.features)
        y = np.asarray(self.targets, dtype=np.float64)
        weights = np.asarray(self.weights, dtype=np.float64)
        if y.shape != (len(x),) or not np.isin(y, [0.0, 1.0]).all():
            raise ValueError('targets must be one binary 0/1 value per row')
        if (weights.shape != y.shape or not np.isfinite(weights).all()
                or np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, rtol=0, atol=1e-12)):
            raise ValueError('within-block weights must be nonnegative and normalized to one')
        if x.shape[1] != len(self.preprocessing.mean):
            raise ValueError('block and P preprocessing dimensions differ')
        for name, values in [('features', x), ('targets', y), ('weights', weights)]:
            object.__setattr__(self, name, _readonly(values))


def prepare_block(features, targets, preprocessing, weights=None):
    """Transform with fixed P statistics; explicit weights must already sum to one."""
    transformed = preprocessing.transform(features)
    if weights is None:
        weights = np.full(len(transformed), 1.0 / len(transformed), dtype=np.float64)
    return PreparedBlock(transformed, targets, weights, preprocessing)


def _groups(columns: Sequence[Sequence]):
    columns = [list(column) for column in columns]
    if not columns or not columns[0] or any(len(c) != len(columns[0]) for c in columns):
        raise ValueError('group columns must have equal nonzero lengths')
    groups = {}
    try:
        for index, key in enumerate(zip(*columns)):
            groups.setdefault(key, []).append(index)
    except TypeError as exc:
        raise ValueError('group identifiers must be hashable') from exc
    return columns, groups


def compound_pair_weights(pair_ids, operators, truth_cells, surface_orders):
    """Equal pairs/AND-OR/four cells/two orders, then equal rows in each leaf.

    Cells are (0,0), (0,1), (1,0), (1,1); orders are 0 and 1. Require all
    sixteen leaves per pair. Repeated observations share their leaf's mass.
    Structural checks do not establish audited membership or factual correctness.
    """
    columns, groups = _groups([pair_ids, operators, truth_cells, surface_orders])
    pairs = set(columns[0])
    leaves = {(op, cell, order) for op in ['AND', 'OR']
              for cell in [(0, 0), (0, 1), (1, 0), (1, 1)] for order in [0, 1]}
    for pair in pairs:
        if {key[1:] for key in groups if key[0] == pair} != leaves:
            raise ValueError('each compound pair requires both operators, four cells and two orders')
    weights = np.empty(len(columns[0]), dtype=np.float64)
    for indices in groups.values():
        weights[indices] = 1.0 / (len(pairs) * 16 * len(indices))
    return weights


def isolated_pair_weights(pair_ids, affirmative_fact_ids):
    """Equal pairs/four distinct fact IDs, then equal rows per fact.

    The caller must establish that IDs denote the exact allocated affirmative facts.
    """
    columns, groups = _groups([pair_ids, affirmative_fact_ids])
    pairs = set(columns[0])
    for pair in pairs:
        if len({key[1] for key in groups if key[0] == pair}) != 4:
            raise ValueError('each isolated pair requires four distinct affirmative fact IDs')
    weights = np.empty(len(columns[0]), dtype=np.float64)
    for indices in groups.values():
        weights[indices] = 1.0 / (len(pairs) * 4 * len(indices))
    return weights


def _validate_blocks(mode, p, adaptation, preprocessing):
    if mode not in ('r0', 'compound', 'isolated'):
        raise ValueError('mode must be r0, compound or isolated')
    if p.preprocessing is not preprocessing:
        raise ValueError('P must use the supplied fixed preprocessing object')
    if not np.allclose(p.weights, 1.0 / len(p.targets), rtol=0, atol=1e-15):
        raise ValueError('P must use the ordinary mean over all supplied P rows')
    if mode == 'r0':
        if adaptation is not None:
            raise ValueError('R0 accepts only P')
    elif adaptation is None or adaptation.preprocessing is not preprocessing:
        raise ValueError('adaptation must use the same fixed P preprocessing object')


def _evaluate(parameters, mode, p, adaptation):
    w, intercept = parameters[:-1], parameters[-1]
    loss = PENALTY * np.dot(w, w)
    gradient = np.r_[2.0 * PENALTY * w, 0.0]
    blocks = [(1.0, p)] if mode == 'r0' else [(0.5, p), (0.5, adaptation)]
    for factor, block in blocks:
        logits = block.features @ w + intercept
        # logaddexp on signed logits avoids subtraction/cancellation at large |z|.
        signed = np.where(block.targets == 1, -logits, logits)
        loss += factor * np.dot(block.weights, np.logaddexp(0.0, signed))
        residual = np.where(block.targets == 1, -expit(-logits), expit(logits))
        residual = factor * block.weights * residual
        gradient[:-1] += block.features.T @ residual
        gradient[-1] += residual.sum()
    return float(loss), gradient


def objective_and_gradient(parameters, mode, p, adaptation=None):
    """Exact objective and analytic gradient; parameters are [w..., intercept]."""
    _validate_blocks(mode, p, adaptation, p.preprocessing)
    parameters = np.asarray(parameters, dtype=np.float64)
    if parameters.shape != (p.features.shape[1] + 1,) or not np.isfinite(parameters).all():
        raise ValueError('parameters must be finite [w..., intercept]')
    return _evaluate(parameters, mode, p, adaptation)


@dataclass(frozen=True)
class OptimizerSettings:
    initial_max_iterations: int = 2000
    total_max_iterations: int = 10000
    gradient_tolerance: float = 1e-4

    def __post_init__(self):
        if (type(self.initial_max_iterations) is not int or type(self.total_max_iterations) is not int
                or not 0 < self.initial_max_iterations < self.total_max_iterations <= 10000
                or self.initial_max_iterations > 2000):
            raise ValueError('require 0 < initial <= 2000, initial < total <= 10000')
        if not np.isfinite(self.gradient_tolerance) or not 0 < self.gradient_tolerance <= 1e-4:
            raise ValueError('gradient tolerance must be finite and in (0, 1e-4]')


@dataclass(frozen=True)
class AttemptRecord:
    maximum_iterations: int
    initial_parameters: np.ndarray
    final_parameters: np.ndarray
    initial_objective: float
    objective: float
    iterations: int
    function_evaluations: int
    termination_code: int
    termination_message: str
    library_success: bool
    relative_objective_change: float
    gradient_infinity_norm: float
    finite_parameters: bool
    finite_objective: bool
    finite_gradient: bool
    status: str


@dataclass(frozen=True)
class ReadoutFit:
    """One linear head; flagged results cannot be used through prediction methods."""

    mode: Objective
    weights: np.ndarray
    intercept: float
    preprocessing: PPreprocessing
    settings: OptimizerSettings
    attempts: tuple[AttemptRecord, ...]
    status: str

    @property
    def converged(self):
        return self.status == 'converged'

    def require_converged(self):
        if not self.converged:
            raise RuntimeError('flagged readout fit: inspect both attempt records')
        return self

    def decision_function(self, features):
        self.require_converged()
        return self.preprocessing.transform(features) @ self.weights + self.intercept

    def predict_probability(self, features):
        return expit(self.decision_function(features))


def _attempt(initial, maximum_iterations, settings, mode, p, adaptation):
    evaluate = lambda parameters: _evaluate(parameters, mode, p, adaptation)
    initial_loss = evaluate(initial)[0]
    objectives = [initial_loss]

    def callback(parameters):
        objectives.append(evaluate(parameters)[0])

    # Unconstrained L-BFGS-B: float64, deterministic, full-batch, no C conversion.
    result = minimize(evaluate, initial, jac=True, method='L-BFGS-B', callback=callback,
                      options={'maxiter': maximum_iterations, 'gtol': settings.gradient_tolerance,
                               'ftol': 0.0, 'maxls': 50, 'maxfun': 1000000, 'maxcor': 10})
    parameters = np.asarray(result.x, dtype=np.float64)
    objective, gradient = evaluate(parameters)
    previous = objectives[-2] if len(objectives) > 1 else initial_loss
    finite_parameters = bool(np.isfinite(parameters).all())
    finite_objective = bool(np.isfinite(objective))
    finite_gradient = bool(np.isfinite(gradient).all())
    norm = float(np.max(np.abs(gradient)))
    relative_change = float(abs(objective - previous) / max(1.0, abs(objective), abs(previous)))
    converged = (bool(result.success) and finite_parameters and finite_objective
                 and finite_gradient and norm <= settings.gradient_tolerance)
    record = AttemptRecord(maximum_iterations, _readonly(initial), _readonly(parameters),
                           initial_loss, objective, int(result.nit), int(result.nfev), int(result.status),
                           str(result.message), bool(result.success), relative_change, norm,
                           finite_parameters, finite_objective, finite_gradient,
                           'converged' if converged else 'flagged')
    return parameters, record


def fit_readout(mode, p, preprocessing, adaptation=None, settings=None):
    """Fit prepared blocks with fixed P statistics; zero init then one warm retry.

    A flagged first attempt receives the remaining budget (10000 minus actual
    first-attempt iterations by default), with exactly the same objective.
    A final flagged result emits a warning, retains diagnostics, and blocks prediction.
    No audited membership, pair allocation, or held-out exclusion is inferred here.
    """
    _validate_blocks(mode, p, adaptation, preprocessing)
    settings = OptimizerSettings() if settings is None else settings
    initial = np.zeros(p.features.shape[1] + 1, dtype=np.float64)
    parameters, first = _attempt(initial, settings.initial_max_iterations, settings, mode, p, adaptation)
    attempts = [first]
    if first.status == 'flagged':
        remaining = settings.total_max_iterations - first.iterations
        parameters, retry = _attempt(parameters, remaining, settings, mode, p, adaptation)
        attempts.append(retry)
    status = attempts[-1].status
    artifact = ReadoutFit(mode, _readonly(parameters[:-1]), float(parameters[-1]), preprocessing,
                          settings, tuple(attempts), status)
    if status == 'flagged':
        warnings.warn('Readout fit remains flagged after warm-start retry; inspect attempts.',
                      RuntimeWarning, stacklevel=2)
    return artifact
