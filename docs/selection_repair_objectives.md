# Shared readout objectives: synthetic numerical validation

This isolated module implements the October objective specification for one layer
at a time. It loads no data and is independent of the ongoing source audit. It
adds no production CLI, candidate bank, selector, cross-validation orchestration,
or dependency. Historical evaluators, probes, cache checkers, and reports remain
unchanged. The unrelated compound-transfer implementation is not used.

## Equations and shared head

For a standardized row z, logit s = z·w + b and binary target y, use stable BCE
`logaddexp(0, -s)` when y = 1 and `logaddexp(0, s)` when y = 0. Its logit derivative
is `-expit(-s)` for y = 1 and `expit(s)` for y = 0. There is no class weighting.

Let L(B) = sum_i q_i BCE_i, with nonnegative within-block weights q summing to one.

- R0: `L(P) + 0.001 * sum(w**2)`.
- Joint compound repair: `0.5 * L(P) + 0.5 * L(A_compound) + 0.001 * sum(w**2)`.
- Matched atomic-only adaptation: `0.5 * L(P) + 0.5 * L(A_isolated) + 0.001 * sum(w**2)`.

The intercept b is free and unpenalized. The weight penalty gradient is exactly
`0.002*w`; its intercept component is zero. No library C parameter is used. R0
uses the full P mean. Adaptation retains separate half-weight block means even
when P and A have different sizes. Each fit returns one linear head; compound
rows for AND and OR use that same head, without operator indicators, constituent
scores, or parse features.

P weights must be uniform over all supplied P rows. Explicit adaptation weights
must already be normalized; the API rejects invalid weights rather than silently
rescaling them. The compound helper assigns equal mass to each pair, then its two
operators, four truth cells and two surface orders. Each pair must include all
16 leaves. Repeated observations share a leaf's mass equally. Complete 16-row
pairs give the ordinary row mean across pairs. The isolated helper assigns equal
mass to pairs and four distinct affirmative fact IDs per pair; repetitions share
a fact's mass equally. These helpers check group structure, not the truth or
membership of supplied identifiers.

## P-only preprocessing

`fit_p_preprocessing(P_features)` computes coordinate means and population standard
deviations (`ddof=0`) in float64 on P alone. The denominator is `max(std, 1e-6)`.
Exactly constant P coordinates are masked to zero for every block and evaluation
input, even when those inputs differ from P. Exact repeated coordinates are also
detected directly so reduction rounding on constants such as 0.1 cannot create
spurious variance. Small positive standard deviations are floored, not masked.

The returned `PPreprocessing` stores mean, population standard deviation,
denominator, constant mask, P row count, fit-source label, and transformation
policy. Statistics and prepared arrays are copied and marked read-only. Fit once
per layer, then reuse the same object for P, each adaptation block and evaluation.
Never fit preprocessing on A, D, E, or concatenated blocks. A caller can supply
misidentified arrays; this numerical module does not establish that P was audited.

## API

Public functions live in `src/selection_repair_objectives.py`. For example, this
entire example uses synthetic arrays:

```python
import numpy as np
from src.selection_repair_objectives import (
    fit_p_preprocessing, prepare_block, fit_readout,
)

P_x = np.array([[-1., 0.], [0., 1.], [1., -1.], [2., 2.]])
P_y = np.array([0., 1., 0., 1.])
A_x = np.array([[-0.5, 1.], [1.5, 0.]])
A_y = np.array([1., 0.])
stats = fit_p_preprocessing(P_x)
p = prepare_block(P_x, P_y, stats)
a = prepare_block(A_x, A_y, stats)
r0 = fit_readout('r0', p, stats)
repair = fit_readout('compound', p, stats, adaptation=a)
repair.require_converged()
probabilities = repair.predict_probability(np.array([[0.2, 0.3]]))
```

`prepare_block(raw_features, targets, preprocessing, weights=None)` applies the
fixed statistics; omitted weights give the ordinary mean. Do not pass already
standardized arrays to this function. Prepared blocks retain the preprocessing
object; fitting rejects adaptation prepared with a different object, even if its
statistics happen to match. `objective_and_gradient([w..., b], mode, p, adaptation)`
returns the scalar objective and analytic gradient without optimization. Use
`mode='isolated'` with the prepared isolated block for matched adaptation.

`compound_pair_weights(pair_ids, operators, truth_cells, surface_orders)` expects
operators `AND`/`OR`, tuple cells `(0,0)`, `(0,1)`, `(1,0)`, `(1,1)`, and surface
orders 0/1. `isolated_pair_weights(pair_ids, affirmative_fact_ids)` expects four
distinct fact IDs in each pair. Pass returned weights explicitly to `prepare_block`.
These identifiers are used only to construct weights, never as model features.

## Optimization and validity policy

SciPy's unconstrained `L-BFGS-B` implements deterministic full-batch limited-memory
BFGS with analytic gradients and float64 arrays. Defaults are zero initialization,
2,000 initial maximum iterations, gradient infinity-norm tolerance 1e-4, `ftol=0`,
`maxcor=10`, `maxls=50`, and `maxfun=1000000`. Setting ftol to zero avoids a positive
relative-objective tolerance terminating a fit before the gradient target is met.

An attempt is converged only if the library reports success, every parameter,
objective and gradient component is finite, and the actual gradient infinity norm
is at most the target. Library success alone never makes a flagged fit valid.
After any flagged first attempt, retry the identical objective from its final
parameters, allowing `10000 - first_attempt_iterations` further iterations. Thus
10,000 is the total actual iteration cap, not a second independent 10,000 budget.
No regularization, preprocessing, weights or objective changes on retry.

`OptimizerSettings` permits smaller budgets and stricter gradient tolerances for
synthetic diagnostics, but not larger caps or looser tolerances. Each
`AttemptRecord` stores start/end parameters, start/final objective, maximum and
actual iterations, evaluation count, termination code/message, library success,
relative objective change, gradient infinity norm, separate finite-value checks,
and `converged`/`flagged` status. Relative change is the absolute final objective
difference from the preceding accepted iterate, divided by the maximum of one
and their absolute values; a zero-step attempt compares to its starting objective.

`ReadoutFit` retains the head, preprocessing, settings and all attempts. A final
flagged fit emits `RuntimeWarning`, exposes `status='flagged'` and
`converged=False`, and raises on `require_converged`, `decision_function`, or
`predict_probability`. It is never silently accepted for evaluation. A converged
fit after retry retains its flagged first attempt for inspection.

## Focused synthetic verification

Run from the task worktree:

```sh
/tmp/clean-extraction-venv/bin/python -B -m pytest -q -s -p no:cacheprovider \
  tests/test_selection_repair_objectives.py
git diff --check
```

Tests cover central finite differences including intercept; independent tiny
probability-space calculations of all three objectives; full R0 weighting and
exact regularization; unpenalized intercept; duplication of either block;
unequal block sizes; P-only statistics, constant masking and the floor; hierarchical
weights including repeated leaves/facts; extreme logits; deterministic repeated
fits; successful warm retry; visible intentional failure; library success with a
bad gradient or nonfinite output; and API rejection of invalid block weights or
preprocessing objects. Every array is small and constructed in memory.

### Recorded synthetic run — 2026-10-01

Validated on macOS 12.6, Python 3.11.6, NumPy 2.4.6, SciPy 1.17.1 and pytest
9.1.1 in the existing `/tmp/clean-extraction-venv` environment. The new task branch
is `t2-objective-validation-20261001`, based on reviewed ancestor
`88c27c625e7307bd8871e2c4506121c8cd71c983`. The baseline had no tracked or untracked
changes; origin fetch and fast-forward-only pull found it synchronized.

All **26 focused tests passed**. Maximum absolute central-difference errors were
`2.591e-11` for R0 and `3.000e-11` for each adaptation objective, including the
intercept (largest intercept error `1.667e-11`). Across all five duplication
checks, maximum loss and gradient changes were each `5.551e-17`. Complete-pair
weights gave exactly the same objective and gradient as ordinary row weights.
The independent tiny-example objectives were R0 `0.492905267403`, compound
`0.571208007432`, and isolated `0.554983489144`.

| Synthetic fit | Objective | Iterations | Gradient infinity norm | Relative objective change | Status |
|---|---:|---:|---:|---:|---|
| R0 | 0.405352285885 | 7 | 2.968e-5 | 2.097e-7 | converged |
| Compound | 0.300924611108 | 7 | 8.832e-5 | 1.802e-6 | converged |
| Isolated | 0.300924611108 | 7 | 8.832e-5 | 1.802e-6 | converged |

Repeated fits returned identical parameters. The compound and isolated fit
fixtures deliberately use the same synthetic adaptation block, demonstrating
their identical numerical form; the independent tiny example uses distinct
blocks. A forced first-attempt iteration limit retried from the exact last
parameters and converged in `1 + 6` iterations (gradient norm `5.305e-5`). A
stricter two-iteration total limit produced two flagged attempts with final
gradient norm `2.496e-2` against a `1e-12` target, emitted the required warning,
and refused prediction. Separate tests confirmed that fabricated library success
cannot override either a bad gradient or nonfinite parameters/loss/gradient.

## Remaining production integration requirements

Future data assembly must enforce audited membership, exact allocated facts,
removal of held-out pairs, permitted row coverage, and the approved split/leakage
policy before preparing any block. It must verify complete compound structure,
affirmative isolated facts, original cache/row binding, and compatible adopted
representation for each layer. Any intended within-block policy beyond the
ordinary mean must supply the specified normalized weights. Prepared arrays alone
cannot prove those upstream guarantees.

This task does not approve the source audit or representation contract, create a
final admitted-row map, reserve P/A, or authorize real-data fitting. No research
activations, score packages, source datasets or final-test material are loaded.
Synthetic convergence validates the numerical implementation, not a scientific
result or real-data optimizer behavior. No production pipeline is connected.
