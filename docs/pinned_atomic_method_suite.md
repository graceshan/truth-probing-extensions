# Canonical pinned Qwen2.5 atomic-method suite v1

This CPU-only implementation fits atomic TRAIN directions and evaluates/selects
on atomic VALIDATION. It has no atomic TEST or compound-data interface. The local
implementation tests use fabricated atomic caches; they do not produce scientific
results or substitute hashes for the absent RunPod artifacts.

## Inputs and binding

Only these runtime input trees are opened:

- `acts/clean_protocol/atomic/qwen25_a09a354_bs1_bf16_v1/`
- `results/clean_protocol/atomic_probes_pinned_v1/qwen25_7b/`

`RepairedAtomicCache` validates its six fixed files, completion receipt,
extraction contract, approved input digest, exact metadata, and NPY headers and
hashes. There are exactly 3,144 TRAIN and 1,040 VALIDATION rows. The adapter uses
the repaired metadata in stored order, never the old mixed-source loader or its
source exports. It preserves `(dataset, row_index)` identities, reads arrays by
read-only mmap, materializes only a requested layer as float64, and counts every
activation row materialized (including archive-validation rereads).

The representation is Qwen/Qwen2.5-7B-Instruct at revision
`a09a35458c702b33eeacc393d103063234e8bc28`, with 28 saved layers, width 3584,
float16 storage, and fingerprint
`59df237b800f49892889e72ee6f852314b8430eb14f9c64dc0d7fff83e832821`.
The descriptor is derived from the validated atomic contract and the producer's
layer convention. No compound artifact or frozen transfer specification is read.

The atomic-only `pinned_compound_scoring.verify_probe` function is reused; none of
that module's compound orchestration is called. It verifies the immutable
selection, NPZ and validation-grid hashes; 140 converged configurations; frozen
selection/solver/label semantics; classes `[0,1]`; layer/C; and exact repaired
cache files, sizes, approved digest and contract. Expected layer 17, C=1.0 and
validation AUROC 0.9996523668639054 are assertions after artifact verification.
Selection-time source hashes remain recorded producer provenance and are audited
against recorded Git bytes where available. They need not match refactored code.
Frozen classifier and cache bytes must match. No existing artifact is rewritten.

## Faithful comparison

| Method | TRAIN rows | Policy |
|---|---:|---|
| `l2_logistic` | 3,144 | Reuse frozen layer-17, C=1.0 coefficients and intercept; zero LR fits |
| `difference_of_means` | 3,144 | Existing unit true-minus-false mean direction |
| `mass_mean_covariance` | 3,144 | Existing within-class covariance pseudoinverse, absolute cutoff 1e-3 |
| `burger_t_g` | 1,000 | Existing paired balanced sampling and dataset-centered OLS |
| `ttpd` | 1,000 | Same exact subset; existing polarity LR and two-coordinate truth LR |

All methods evaluate the same ordered 1,040 VALIDATION rows. Primary comparisons
use saved layer 17. Each non-LR method is also fit at every saved layer 0–27;
secondary selection maximizes pooled validation AUROC, with an exact tie going
to the lower layer. LR's secondary entry is its existing canonical winner.
The frozen LR validation diagnostics must reproduce the selection record within
absolute tolerance 1e-12; otherwise the run cannot complete.

Sampling calls `balanced_burger_indices(..., seed=0)` unchanged: minimum
affirmative-dataset size, a shared `RandomState(0)`, topic order from
`BURGER_TOPICS`, and the same chosen positions for affirmative/negated rows.
The result must have exactly 100 rows per topic/form dataset and 1,000 unique
ordered TRAIN rows. There is no additional class balancing or reordering.

`atomic_method_suite.fit_layer` and `choose_layers` are reused, while its
historical `run_model`, `MethodData` and baseline loader are not called. All
method mathematics in `atomic_probe_methods.py` remain unchanged. In particular,
no direction is flipped using validation performance. TRAIN orientation gaps
are recorded; t_G's partial-regressor sign diagnostic is also retained.

## Supplementary matched control

The matched mode uses the identical deterministic 1,000 ordered Bürger rows for
all five methods and the same 1,040 VALIDATION rows, at saved layer 17 only.
It reuses `atomic_methods_matched.fit_matched_methods`. Its LR is refit with
C=1.0 read from the verified frozen artifact, using the canonical convergence
policy (2,000 iterations, retry at 10,000 only on a convergence warning, fail
if still unconverged). There is no C tuning or layer selection. The control
does not modify or depend on generated faithful-suite outputs. The synthetic
integration test verifies identical t_G/TTPD parameters in the two modes.

## Commands on the canonical artifact host

Run from the repository root, using the host's CPU Python environment:

```bash
python scripts/38_pinned_atomic_method_suite.py --suite both --check-only
```

Check-only validates/cache-hashes and inspects the frozen coefficient archive,
resolves ordered row identities and the exact sample, and prints JSON provenance.
It performs zero fits, zero scores and zero numerical activation-matrix reads.
Hashing the full array files as bytes is required. It writes no result tree.
Missing canonical files are a hard failure; there is no synthetic fallback.

After successful check-only, the future production command is:

```bash
python scripts/38_pinned_atomic_method_suite.py --suite both
```

`--suite faithful` or `--suite matched` runs either independently. Every fit and
verification score runs inside `threadpool_limits(limits=1)`. No model weights,
GPU or forward pass are involved. Do not run production fits in the local Mac
session without the actual canonical inputs. This implementation task does not
authorize or produce compound scores.

## Output schema and completion

Fresh output roots:

- `results/clean_protocol/atomic_method_suite_pinned_v1/qwen25_7b/`
- `results/clean_protocol/atomic_method_matched_pinned_v1/qwen25_7b/`

Any existing destination, including an incomplete run, is refused. There is no
automatic overwrite or resume. Historical outputs are never opened or modified.

| File | Contents |
|---|---|
| `provenance.json` | Schema/version, policy, upstream references, current code hashes/runtime, representation, six cache identities, three LR identities, producer-source audit, row/sample hashes and counts |
| `allowed_atomic_rows.csv` | Exact eight repaired metadata fields; TRAIN followed by VALIDATION, preserving each partition's order |
| `burger_training_rows.csv` | Same eight fields for the exact ordered 1,000-row TRAIN subset |
| `layers/layer_XX.npz` | Safe numeric/Unicode arrays, keys `method__parameter` |
| `layers/layer_XX.json` | Per-method diagnostics, fit details/convergence, archive and activation-slice hashes, representation fingerprint, readback verification |
| `validation_metrics.csv` | 113 faithful rows (4×28 + frozen LR) or five matched rows |
| `summary.json` | Full primary records and faithful secondary winners; no secondary selection for matched mode |
| `report.md` | Primary and, for faithful, secondary comparison tables |
| `verification.json` | Atomic completion receipt, published last; output hashes/sizes, input rehash verification, activation counters, BLAS information, fit counts and zero prohibited-access flags |

Every method/layer metric row includes `method`, `layer`, `comparison`,
`fit_rows`, `validation_rows`, TRAIN true-minus-false score gap, overall,
affirmative, negated, each topic and topic-macro validation AUROCs (plus the
shared diagnostic's accuracy at zero), parameter path/hash, ordered fit and
validation identity hashes, and actual fit/validation activation-slice hashes.
Slice hashes encode C-order float16 bytes: full TRAIN for LR/DoM/MM in faithful
mode, the ordered subset for t_G/TTPD and every matched fit. Layer JSON also
records the full TRAIN slice hash. Validation slices always use canonical order.

All parameters needed for later label-blind scoring are saved. Ordinary methods
use raw `X @ coef + intercept`. TTPD includes `t_g`, `t_p`, `dataset_names`,
`dataset_means`, `ols_gram`, `polarity_coef`, `polarity_intercept`,
`polarity_classes`, `head_coef`, `head_intercept`, and `classes` (the truth-head
classes). Its `coef`/`intercept` keys are the equivalent fused affine readout.
The original explicit readout projects onto t_G and the polarity *coefficient*,
without adding the polarity intercept or centering inference activations.
Every archive is loaded with `allow_pickle=False`, checked for safe finite arrays,
and used to reproduce all saved validation diagnostics. TTPD explicit versus
fused scores must agree with rtol=1e-9, atol=1e-8. Completion rehashes all nine
canonical input files. A failed check leaves no completed verification receipt.

## Tests and expected cost

```bash
python -m pytest -q tests/test_pinned_atomic_method_suite.py
```

Fixtures retain 3,144/1,040 rows and 28 layers, reducing only hidden width to six
and substituting explicitly synthetic identities. Tests guard all data I/O,
exercise strict verifier failures, full fitting/output/readback, sample identity,
no faithful LR refit, matched C=1.0, no validation sign flip, source/parameter
preservation and historical output sentinels. They never read real activations.

Production cost is unmeasured. The likely bottleneck is 28 dense 3584×3584
covariance eigendecompositions (plus one for matched), followed by full-width
unregularized TTPD polarity LR, which may retry up to 10,000 iterations. Budget
minutes to hours on a CPU depending on hardware/convergence, and roughly a few
GB of available RAM for matrices, covariance/eigenvectors and workspace. A
float64 TRAIN layer alone is about 90 MB and a covariance matrix about 103 MB.
Repeated full-cache hashing also incurs disk I/O. All reported scientific
results still require the future RunPod run.
