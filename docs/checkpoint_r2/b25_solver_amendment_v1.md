# Authorized fresh B25 logistic solver amendment

This successor begins exactly at
`60e651370e5c08635f151822073a8b9c01e092ca`. The user explicitly authorized
a bounded numerical-policy amendment after the original B25 validity halt.
The original implementation, configuration, run and halt-review package remain
unchanged. Scientific authority, allocations, membership, preprocessing,
objectives, eligibility, ties, bootstrap schedules, margins and interpretation
remain those frozen in `fresh_b25_v1.json` and its approved sources.

## Training-only diagnosis

All four failed raw-LR saved parameters reproduce the original mean-loss
objectives and independent gradient infinity norms exactly using only P15.
Ordered source rows, logical bindings, labels, features and parameter hashes
are verified. The installed pinned sklearn 1.3.2 logistic path calls SciPy
L-BFGS-B with gtol and maxiter, leaving ftol at SciPy 1.11.4's default
2.2204460492503131e-9. SciPy supports successful relative-loss termination as
well as gradient termination. A library-success flag therefore does not prove
the independent mean-loss gradient criterion. Original solver task messages
were not retained; the four original stopping reasons are unknown. We do not
infer those reasons from gradients or performance. Installed source files are
hashed in the training diagnosis receipt.

## Declared bounded successor policy

Only new raw-LR and TTPD logistic subfits use this amendment, uniformly across
both models, all saved layers and all declared Cs. Each first sklearn attempt
retains its original objective, inputs, raw feature transformation, C, free
intercept and zero initialization: raw LR uses 2,000 iterations and L2;
each unregularized TTPD component uses 100 iterations and penalty=None.
The first attempt is observed without changing its optimizer arguments.

Raw LR's exact objective is mean BCE + ||w||²/(2*C*n), with no intercept
penalty. TTPD's logistic subfits use mean unregularized BCE, equivalent to
C=infinity. The polarity subfit uses the same 700 raw rows; the truth subfit
uses the same two columns X@t_G and X@polarity_coef. Composition and inference
are unchanged. A failed polarity subfit cannot feed the truth fit or scoring.
R0, repair, C_clean and closed-form methods retain the inherited implementation.

If the independent acceptance criteria fail, even with library success and no
warning, allow **one** deterministic warm continuation from the last parameters.
There is no successor cold retry or solver-setting search. The settings are
declared before any real continuation:

| Setting | Frozen value |
|---|---:|
| Arithmetic and analytic gradients | float64 |
| Method | unbounded L-BFGS-B |
| gtol | 0.0001 |
| ftol | 0 |
| maxls | 50 |
| maxcor | 10 |
| maxfun, with hard objective-call guard | 50,000 |
| Total optimizer iterations per logistic subfit | 10,000 |

Continuation maxiter is 10,000 minus all recorded prior spent iterations,
including saved original attempts. No spent iterations reset. Diagnostics
outside optimization do not count as optimizer function calls. If the hard
function-call bound interrupts SciPy, termination is unsuccessful; the exact
iteration count is unavailable, and all remaining iteration budget is charged
to prevent further continuation. A subfit already carrying a continuation
attempt cannot receive another, even on resume.

Acceptance requires finite parameters, loss and gradient; successful optimizer
termination; and an independently recomputed gradient infinity norm <=1e-4.
The unchanged diagnostic implementation is separate from the analytic loss
supplied to SciPy. Persistent invalidity publishes no predictions and blocks
the complete primary comparison. No candidate/layer removal substitutes for
completion. No D, allocated-A validation, wording or performance result guides
the amendment's settings.

## Commit, gate and compatibility

The amendment, numerical implementation, successor configuration and focused
tests must be committed before continuing any real failed head. `gate` reads
only P15 arrays and labels, preserves each original receipt/parameter hash,
and saves before/after loss, independent gradient, termination, iterations and
settings. All four must pass before the production successor is permitted.

The halt-review manifest is located through committed original delivery.json
and all 226 hashes are checked before and after each successor stage. The
original scientific sources/configuration must still match their pins. Valid
original checkpoints are compatible because their exact objectives, inputs,
settings and independent acceptance already passed. All 68 are copied by hash,
without fitting or changing their parameter/score bytes. Successor checkpoint
envelopes carry current execution identity plus original source-fit hash,
configuration and parameter-producer SHA. Existing recoverability-origin
parameter SHAs remain intact. Resume checks current identity and all file
hashes; it never disables the inherited producer/config checks.

Successful continuations bind both the original failed head and the new
parameter producer. Original invalid artifacts remain unchanged and excluded
from prediction. The interrupted fold without a checkpoint may be rerun;
any partial files remain preserved. Inventories distinguish imported valid
fits, four continuations, interrupted reruns, original recoverability reuses
and genuinely new fits. Cumulative attempts and timings remain visible.

The production runner invokes the inherited scientific stage, selectors,
refits, fallbacks and locked evaluation without changing their definitions.
Wordings remain unread until procedure lock. A new persistent bank/fold failure
halts dependent work promptly and remains explicit; the full planned 600/900
inventory is delivered even when work is unrun.

## Reproduction

Use one CPU fitting process and one BLAS/OpenMP thread. Freeze the producer
SHA before either command, with a clean worktree and fresh external directories:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
/private/tmp/clean-extraction-venv/bin/python -B -m unittest \
  tests.test_checkpoint_r2_logistic_continuation tests.test_checkpoint_r2_fresh_b25 -v
/private/tmp/clean-extraction-venv/bin/python -B -m src.checkpoint_r2_b25_convergence_fix gate \
  --expected-commit EXACT_SUCCESSOR_SHA --output NEW_EXTERNAL_GATE
/private/tmp/clean-extraction-venv/bin/python -B -m src.checkpoint_r2_b25_convergence_fix run \
  --expected-commit EXACT_SUCCESSOR_SHA --gate NEW_EXTERNAL_GATE --output NEW_EXTERNAL_RUN
```

Use `--resume` only with that same producer, configuration and output path;
completed checkpoint hashes are checked before reuse. No GPU, constituent
fitting, behavior/chat, interventions, allocation search, new budgets,
scientific-objective tuning or E evaluation is authorized by this amendment.
