"""Only small in-memory synthetic arrays; no research files or historical imports."""
from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from src import selection_repair_objectives as objectives
from src.selection_repair_objectives import (
    OptimizerSettings, compound_pair_weights, fit_p_preprocessing, fit_readout,
    isolated_pair_weights, objective_and_gradient, prepare_block,
)


def tiny_blocks():
    raw_p = np.array([[-1., 1.], [1., -1.]])
    raw_a = np.array([[0.2, -0.8], [1.2, 0.7], [-0.5, 0.4]])
    stats = fit_p_preprocessing(raw_p)
    return stats, prepare_block(raw_p, [0, 1], stats), prepare_block(raw_a, [1, 0, 1], stats)


def synthetic_blocks():
    rng = np.random.default_rng(20261001)
    raw_p = rng.normal(size=(37, 4))
    raw_a = rng.normal(loc=0.6, size=(13, 4))
    beta = np.array([0.6, -0.8, 0.2, 0.4])
    p_targets = (raw_p @ beta + rng.normal(size=37) > 0).astype(float)
    a_targets = (raw_a @ beta + rng.normal(size=13) > 0).astype(float)
    stats = fit_p_preprocessing(raw_p)
    return stats, prepare_block(raw_p, p_targets, stats), prepare_block(raw_a, a_targets, stats)


@pytest.mark.parametrize('mode', ['r0', 'compound', 'isolated'])
def test_central_difference_including_intercept(mode):
    stats, p, a = tiny_blocks()
    a = prepare_block([[0.2, -0.8], [1.2, 0.7], [-0.5, 0.4]], [1, 0, 1], stats,
                      weights=[0.2, 0.3, 0.5])
    theta = np.array([0.27, -0.42, 0.31])
    adaptation = None if mode == 'r0' else a
    value, analytic = objective_and_gradient(theta, mode, p, adaptation)
    step = 1e-6
    numerical = np.empty_like(theta)
    for index in range(len(theta)):
        delta = np.zeros_like(theta)
        delta[index] = step
        numerical[index] = (objective_and_gradient(theta + delta, mode, p, adaptation)[0]
                            - objective_and_gradient(theta - delta, mode, p, adaptation)[0]) / (2 * step)
    error = np.max(np.abs(analytic - numerical))
    print(f'gradient {mode}: max_absolute_error={error:.3e}; intercept_error={abs(analytic[-1]-numerical[-1]):.3e}')
    assert np.isfinite(value)
    assert_allclose(analytic, numerical, rtol=1e-7, atol=1e-9)


def test_three_objectives_independent_tiny_example_and_full_r0_weight():
    stats, p, compound = tiny_blocks()
    isolated = prepare_block([[0.1, 0.9], [-0.3, -0.6], [1.7, 0.2], [-0.2, 0.4]], [0, 1, 1, 0], stats)
    w, intercept = np.array([0.2, -0.3]), 0.4
    theta = np.r_[w, intercept]

    # Independent probability-space formula is safe for this deliberately tiny example.
    def direct(block):
        probabilities = 1 / (1 + np.exp(-(block.features @ w + intercept)))
        return float(np.mean(-block.targets * np.log(probabilities)
                             - (1 - block.targets) * np.log(1 - probabilities)))

    penalty = 0.001 * (0.2**2 + (-0.3)**2)
    expected = {'r0': direct(p) + penalty,
                'compound': 0.5 * direct(p) + 0.5 * direct(compound) + penalty,
                'isolated': 0.5 * direct(p) + 0.5 * direct(isolated) + penalty}
    for mode, a in [('r0', None), ('compound', compound), ('isolated', isolated)]:
        value, gradient = objective_and_gradient(theta, mode, p, a)
        assert_allclose(value, expected[mode], rtol=0, atol=2e-16)
        print(f'tiny objective {mode}: {value:.12f}')
        if mode == 'r0':
            probabilities = 1 / (1 + np.exp(-(p.features @ w + intercept)))
            residual = probabilities - p.targets
            assert_allclose(gradient[:-1], p.features.T @ residual / len(p.targets) + 0.002*w)
            assert_allclose(gradient[-1], residual.mean())
            assert abs(value - (0.5 * direct(p) + penalty)) > 0.1


@pytest.mark.parametrize('mode,duplicated', [('r0', 'p'), ('compound', 'p'), ('compound', 'a'),
                                           ('isolated', 'p'), ('isolated', 'a')])
def test_block_duplication_invariance(mode, duplicated):
    stats, p, a = tiny_blocks()
    a = prepare_block([[0.2, -0.8], [1.2, 0.7], [-0.5, 0.4]], [1, 0, 1], stats, [0.2, 0.3, 0.5])
    theta = np.array([0.2, -0.3, 0.4])

    def duplicate(block):
        return objectives.PreparedBlock(np.repeat(block.features, 2, axis=0),
                                        np.repeat(block.targets, 2), np.repeat(block.weights / 2, 2), stats)

    before = objective_and_gradient(theta, mode, p, None if mode == 'r0' else a)
    after = objective_and_gradient(theta, mode, duplicate(p) if duplicated == 'p' else p,
                                   None if mode == 'r0' else duplicate(a) if duplicated == 'a' else a)
    loss_error = abs(before[0] - after[0]); gradient_error = np.max(np.abs(before[1] - after[1]))
    print(f'duplication {mode}/{duplicated}: loss_error={loss_error:.3e}; gradient_error={gradient_error:.3e}')
    assert_allclose(before[0], after[0], rtol=0, atol=1e-15)
    assert_allclose(before[1], after[1], rtol=0, atol=1e-15)


@pytest.mark.parametrize('mode', ['compound', 'isolated'])
def test_unequal_block_sizes_keep_half_weights(mode):
    stats = fit_p_preprocessing([[2.], [2.]])
    p = prepare_block([[2.], [2.]], [0, 0], stats)
    a = prepare_block([[9.]] * 9, [1] * 9, stats)
    theta = np.array([0.0, 0.7])
    actual, gradient = objective_and_gradient(theta, mode, p, a)
    expected = 0.5*np.log1p(np.exp(0.7)) + 0.5*np.log1p(np.exp(-0.7))
    concatenated = (2*np.log1p(np.exp(0.7)) + 9*np.log1p(np.exp(-0.7))) / 11
    assert_allclose(actual, expected)
    assert abs(actual - concatenated) > 0.2
    assert_allclose(gradient[-1], 1/(1+np.exp(-0.7)) - 0.5)


def test_intercept_unpenalized_and_exact_weight_penalty_gradient():
    stats = fit_p_preprocessing([[4., 2.], [4., 2.], [4., 2.]])
    p = prepare_block([[4., 2.]] * 3, [0, 0, 1], stats)
    w = np.array([3., -4.]); intercept = 12.0
    value, gradient = objective_and_gradient(np.r_[w, intercept], 'r0', p)
    expected = (2*np.log1p(np.exp(intercept)) + np.log1p(np.exp(-intercept))) / 3 + 0.001*25
    assert_allclose(value, expected)
    assert_allclose(gradient[:-1], 0.002*w)
    assert_allclose(gradient[-1], 1/(1+np.exp(-intercept)) - 1/3)


def test_preprocessing_p_only_population_std_floor_constants_and_artifact():
    raw = np.array([[1., 0., 7.], [3., 2e-7, 7.]], dtype=np.float64)
    stats = fit_p_preprocessing(raw)
    assert_allclose(stats.mean, [2., 1e-7, 7.], rtol=0, atol=0)
    assert_allclose(stats.population_std, [1., 1e-7, 0.], rtol=0, atol=0)
    assert_allclose(stats.denominator, [1., 1e-6, 1e-6], rtol=0, atol=0)
    assert_array_equal(stats.constant_mask, [False, False, True])
    assert_allclose(stats.transform(raw), [[-1., -0.1, 0.], [1., 0.1, 0.]])
    frozen_mean = stats.mean.copy()
    assert_allclose(stats.transform([[102., 2.1e-6, -1e308]]), [[100., 2., 0.]])
    assert_array_equal(stats.mean, frozen_mean)
    p = prepare_block(raw, [0, 1], stats)
    a = prepare_block([[2., 1e-7, 98.], [3., 2e-7, -32.]], [0, 1], stats)
    fit = fit_readout('compound', p, stats, a)
    assert fit.preprocessing is stats and fit.converged
    assert fit.weights.dtype == stats.mean.dtype == p.features.dtype == np.float64
    assert stats.fit_source == 'P only' and 'ddof=0' in stats.policy and stats.p_rows == 2
    assert_allclose(fit.decision_function(raw), stats.transform(raw) @ fit.weights + fit.intercept)
    with pytest.raises(ValueError):
        stats.mean[0] = 9
    # Exact repeated decimal coordinates must not become spurious varying features.
    decimal = fit_p_preprocessing(np.full((17, 2), 0.1))
    assert_array_equal(decimal.population_std, [0., 0.])
    assert_array_equal(decimal.transform([[1e300, -1e300]]), [[0., 0.]])


def test_complete_pair_weights_equal_row_mean_and_hierarchical_leaf_mass():
    rows = list(product(['pair1', 'pair2'], ['AND', 'OR'], [(0, 0), (0, 1), (1, 0), (1, 1)], [0, 1]))
    weights = compound_pair_weights(*zip(*rows))
    assert_array_equal(weights, np.full(32, 1/32))
    values = np.linspace(-2, 3, 32)**2
    assert_allclose(weights @ values, values.mean(), rtol=0, atol=1e-15)
    stats, p, _ = tiny_blocks()
    features = np.column_stack([values, np.linspace(-1, 1, len(rows))])
    targets = np.arange(len(rows)) % 2
    weighted_block = prepare_block(features, targets, stats, weights)
    ordinary_block = prepare_block(features, targets, stats)
    weighted_value, weighted_gradient = objective_and_gradient([0.2, -0.3, 0.4], 'compound', p, weighted_block)
    ordinary_value, ordinary_gradient = objective_and_gradient([0.2, -0.3, 0.4], 'compound', p, ordinary_block)
    assert weighted_value == ordinary_value
    assert_array_equal(weighted_gradient, ordinary_gradient)
    repeated = rows + [rows[0]] * 3
    weighted = compound_pair_weights(*zip(*repeated))
    assert_allclose(weighted[[0, 32, 33, 34]].sum(), 1/32)
    assert_allclose(weighted[:16].sum() + weighted[32:].sum(), 0.5)
    assert_allclose(weighted.sum(), 1)
    with pytest.raises(ValueError, match='requires both operators'):
        compound_pair_weights(*zip(*rows[:-1]))


def test_isolated_pair_weights_equal_four_facts_even_with_repetitions():
    rows = list(product(['pair1', 'pair2'], ['fact1', 'fact2', 'fact3', 'fact4']))
    assert_array_equal(isolated_pair_weights(*zip(*rows)), np.full(8, 1/8))
    repeated = rows + [rows[0]] * 3
    weights = isolated_pair_weights(*zip(*repeated))
    assert_allclose(weights[[0, 8, 9, 10]].sum(), 1/8)
    assert_allclose(weights[:4].sum() + weights[8:].sum(), 0.5)
    with pytest.raises(ValueError, match='four distinct'):
        isolated_pair_weights(*zip(*rows[:-1]))


@pytest.mark.parametrize('mode', ['r0', 'compound', 'isolated'])
def test_extreme_logits_have_finite_stable_loss_and_gradient(mode):
    stats = fit_p_preprocessing([[-1.], [1.]])
    p = prepare_block([[-1.], [1.]], [1, 0], stats)
    a = prepare_block([[-1.], [1.]], [0, 1], stats)
    value, gradient = objective_and_gradient([10000., 0.], mode, p, None if mode == 'r0' else a)
    expected = (10000. if mode == 'r0' else 5000.) + 100000.
    assert value == expected
    assert np.isfinite(gradient).all()
    assert_allclose(gradient, [21. if mode == 'r0' else 20.5, 0.])


@pytest.mark.parametrize('mode', ['r0', 'compound', 'isolated'])
def test_repeated_fits_are_deterministic_with_complete_diagnostics(mode):
    stats, p, a = synthetic_blocks()
    a = None if mode == 'r0' else a
    first = fit_readout(mode, p, stats, a)
    second = fit_readout(mode, p, stats, a)
    assert first.converged and second.converged
    assert_array_equal(first.weights, second.weights)
    assert first.intercept == second.intercept
    assert_array_equal(first.attempts[0].initial_parameters, np.zeros(5))
    assert len(first.attempts) == len(second.attempts) == 1
    record = first.attempts[-1]
    assert record.maximum_iterations == 2000 and first.settings.total_max_iterations == 10000
    assert record.gradient_infinity_norm <= 1e-4
    assert record.finite_parameters and record.finite_objective and record.finite_gradient
    assert record.library_success and record.status == 'converged'
    assert record.iterations > 0 and record.termination_message
    assert np.isfinite(record.relative_objective_change) and record.relative_objective_change >= 0
    assert record.objective <= record.initial_objective
    print(f'fit {mode}: objective={record.objective:.12f}; iterations={record.iterations}; '
          f'gradient_inf={record.gradient_infinity_norm:.3e}; relative_change={record.relative_objective_change:.3e}; deterministic=True')


def test_flagged_first_fit_warm_retries_same_objective_and_converges():
    stats, p, a = synthetic_blocks()
    fit = fit_readout('compound', p, stats, a, OptimizerSettings(initial_max_iterations=1))
    assert fit.converged and len(fit.attempts) == 2
    first, retry = fit.attempts
    assert first.status == 'flagged' and retry.status == 'converged'
    assert_array_equal(retry.initial_parameters, first.final_parameters)
    assert retry.initial_objective == first.objective
    assert retry.maximum_iterations == 10000 - first.iterations
    assert first.iterations + retry.iterations <= 10000
    for attempt in fit.attempts:
        assert attempt.objective == objective_and_gradient(attempt.final_parameters, 'compound', p, a)[0]
    print(f'warm retry: iterations={first.iterations}+{retry.iterations}; gradient_inf={retry.gradient_infinity_norm:.3e}; status={fit.status}')


def test_intentionally_failed_fit_is_visibly_flagged_and_cannot_predict():
    stats, p, a = synthetic_blocks()
    settings = OptimizerSettings(initial_max_iterations=1, total_max_iterations=2, gradient_tolerance=1e-12)
    with pytest.warns(RuntimeWarning, match='remains flagged'):
        fit = fit_readout('compound', p, stats, a, settings)
    assert not fit.converged and fit.status == 'flagged' and len(fit.attempts) == 2
    assert sum(attempt.iterations for attempt in fit.attempts) <= 2
    assert all(attempt.status == 'flagged' for attempt in fit.attempts)
    assert fit.attempts[-1].gradient_infinity_norm > settings.gradient_tolerance
    with pytest.raises(RuntimeError, match='flagged'):
        fit.predict_probability([[0., 0., 0., 0.]])
    print(f'intentional failure: iterations={[a.iterations for a in fit.attempts]}; '
          f'gradient_inf={fit.attempts[-1].gradient_infinity_norm:.3e}; status={fit.status}; prediction=blocked')


@pytest.mark.parametrize('nonfinite', [False, True])
def test_library_success_cannot_override_gradient_or_finite_checks(monkeypatch, nonfinite):
    stats, p, a = synthetic_blocks()

    def false_success(fun, initial, **kwargs):
        x = initial.copy()
        if nonfinite:
            x[0] = np.nan
        return SimpleNamespace(x=x, success=True, nit=0, nfev=1, status=0, message='claimed success')

    monkeypatch.setattr(objectives, 'minimize', false_success)
    # The injected NaN is deliberate; the API's explicit flagged warning is required.
    with np.errstate(invalid='ignore'), pytest.warns(RuntimeWarning, match='remains flagged'):
        fit = fit_readout('compound', p, stats, a)
    assert fit.status == 'flagged' and len(fit.attempts) == 2
    assert all(attempt.library_success and attempt.status == 'flagged' for attempt in fit.attempts)
    if nonfinite:
        assert not fit.attempts[-1].finite_parameters
        assert not fit.attempts[-1].finite_objective
        assert not fit.attempts[-1].finite_gradient
    else:
        assert fit.attempts[-1].gradient_infinity_norm > 1e-4


def test_api_rejects_invalid_weights_wrong_stats_and_wrong_objective_blocks():
    stats, p, a = tiny_blocks()
    other = fit_p_preprocessing([[-1., 1.], [1., -1.]])
    wrong = prepare_block([[0., 0.]], [0], other)
    with pytest.raises(ValueError, match='same fixed P'):
        fit_readout('compound', p, stats, wrong)
    with pytest.raises(ValueError, match='supplied fixed'):
        fit_readout('r0', p, other)
    with pytest.raises(ValueError, match='only P'):
        fit_readout('r0', p, stats, a)
    nonuniform = prepare_block([[-1., 1.], [1., -1.]], [0, 1], stats, [0.2, 0.8])
    with pytest.raises(ValueError, match='ordinary mean'):
        fit_readout('r0', nonuniform, stats)
    for weights in [[1., 1.], [-0.1, 1.1], [np.nan, 0.5]]:
        with pytest.raises(ValueError, match='normalized'):
            prepare_block([[-1., 1.], [1., -1.]], [0, 1], stats, weights)
    with pytest.raises(ValueError, match='binary'):
        prepare_block([[-1., 1.], [1., -1.]], [0, 0.5], stats)
    for kwargs in [{'initial_max_iterations': 2001}, {'total_max_iterations': 10001},
                   {'gradient_tolerance': 1e-3}, {'initial_max_iterations': 0}]:
        with pytest.raises(ValueError):
            OptimizerSettings(**kwargs)
