# Clean atomic probe-method suite

This suite fits only primary atomic TRAIN activations and selects/evaluates only
atomic VALIDATION activations. There is no test-evaluation or compound-transfer
mode. Historical activation files and the frozen clean LR artifacts are read-only.
The shared entity manifest defines both models' row assignments; no new split is
drawn. The primary fixed layers are saved index 17 for Qwen2.5 and 28 for Qwen3.
Layer indices address the existing cached axis directly: embedding state dropped,
saved index s = HF hidden_states[s+1], including the post-final-RMSNorm final
entry. Historical HF revisions remain unrecorded; fitting and validation here
stay entirely within those cached representations. This run does not establish
compatibility with any newly extracted model snapshot.

Before implementation, the repository audit found a reusable unit-normalized
true-minus-false direction in `src/diff_means.py`, clean L2 fitting/selection and
diagnostics in `src/clean_atomic_probes.py`, and no covariance-MM, Bürger general
truth or TTPD implementation. The old exploratory `src/probes.py` splits and
default C=0.1 are not the clean baseline and are not used. The clean baseline
loader structurally parses test labels, so it is also not called by this suite.

Method definitions are grounded in the original paper and pinned author code,
with URLs, commits and content hashes in
`config/clean_protocol/atomic_probe_methods.json`:

- Bürger et al., [Truth is Universal, Sections 3 and 5](https://arxiv.org/html/2407.12831v2),
  and [`probes.py`](https://github.com/sciai-lab/Truth_is_Universal/blob/605ef00514415deb4806969a172f6e13e0798df7/probes.py),
  [`utils.py`](https://github.com/sciai-lab/Truth_is_Universal/blob/605ef00514415deb4806969a172f6e13e0798df7/utils.py),
  and the `truth_directions.ipynb` raw-projection readout at that same commit.
- Marks–Tegmark, [`MMProbe`](https://github.com/saprmarks/geometry-of-truth/blob/5d1c630c44f7e50bda7ad86d601ccadf9abc5ddb/probes.py),
  specifically its covariance-adjusted `iid=True` path.

The five method identifiers have deliberately different meanings:

| Identifier | Fit and readout |
| --- | --- |
| `l2_logistic` | The existing clean TRAIN-fit L2 logistic baseline, including its intercept, C and class orientation. Exact saved coefficients are reused. |
| `difference_of_means` | Unit-normalized `mean_true - mean_false`; score is the raw activation projection. |
| `mass_mean_covariance` | `Sigma^+ @ (mean_true - mean_false)`, where Sigma is pooled **within-class** residual covariance divided by N. Score is the raw linear readout, the logit of the original MM `iid=True` sigmoid. |
| `burger_t_g` | Dataset-centered two-regressor OLS, returning the general-truth coefficient; score is raw `activation @ t_G`. |
| `ttpd` | The full original method: learn t_G, fit an unregularized polarity LR, project raw activations onto t_G and the polarity LR coefficient, then fit an unregularized truth LR on those two coordinates. |

Terminology matters: Bürger's paper calls ordinary mean difference plus a learned
one-dimensional logistic bias “MM.” That is **not** covariance-adjusted MM. The
third method here is explicitly the Marks–Tegmark covariance-adjusted variant
requested for this suite. It is not mislabeled as Bürger's MM baseline, and no
shrinkage/ridge LDA or covariance approximation is substituted.

For the covariance method,

```text
mu_c = mean(X_train[y_train == c])
R_i = X_i - mu_(y_i)
Sigma = R.T @ R / N
w = pinv(Sigma, hermitian=True, atol=1e-3, rtol=0) @ (mu_1 - mu_0)
```

The absolute pseudoinverse cutoff is the upstream default, not validation-tuned.
We use a symmetric eigendecomposition and apply the retained inverse spectrum
directly to the mean difference. This is algebraically the same full covariance
pseudoinverse, without materializing its dense inverse. It is not a diagonal,
low-rank approximation, or learned regularization. Eigenvalues, retained rank,
cutoff, both class means and the resulting classifier coefficient are saved.

For Bürger's truth directions, `tau = 2*y-1` and polarity is +1 for affirmative,
-1 for negated. Each topic/form dataset is centered separately using **only the
sampled training rows**. With `Z = [tau, tau*polarity]`, solve

```text
[t_G; t_P] = solve(Z.T @ Z, Z.T @ (X - dataset_training_mean))
```

Both directions and the training means are saved. We do not fit an activation-to-
truth regression and call it t_G. We do not whiten or orthogonalize t_G/t_P for
the readout. Validation activations are projected raw, exactly as in the authors'
AUROC code; no validation/topic mean is estimated or subtracted.

The authors sample an equal number of rows from each topic/form dataset, using
the same selected row positions for affirmative and negated versions. This is
preserved with a fixed MT19937 seed of 0, and is restricted to eligible TRAIN rows.
The five primary topics give 100 rows per dataset, 1,000 rows total, for both
models. Pair indices, entity identities and complementary training labels are
checked, and the exact sampled training rows are saved. No additional label
stratification or validation-based seed choice is introduced. The original paper
uses a sixth `facts` topic in some experiments; the user's primary atomic scope
here remains the established five topics.

This method-specific sampling budget is reported explicitly. L2, mean difference
and covariance-MM use all 3,144 clean TRAIN rows; t_G and TTPD use the original
balanced 1,000-row subset. All methods are evaluated on the same 1,040 clean
VALIDATION rows. This is a comparison of the specified original methods under
the shared split, not a claim that every method sees an identical row count.

Full TTPD uses a learned polarity direction **p**, not the polarity-sensitive truth
direction **t_P**, and not a supplied polarity indicator at inference. Its polarity
LR maps negated to class 0 and affirmative to class 1, with an intercept during
fitting. As in the authors' `_project_acts`, the polarity projection uses only the
coefficient vector, excluding that intercept. The two-dimensional truth head has
its own intercept. Both LRs are unregularized (`penalty=None`, `lbfgs`, tolerance
1e-4). The original 100-iteration budget is attempted first; a convergence warning
permits one fresh fit with 10,000 iterations and the same objective. A further
warning raises rather than changing the method. Optimization attempts are saved.

All numerical fitting uses float64 conversions of the existing float16 caches,
with one BLAS thread. This numerical precision is explicit; it does not replace
the original formulas. The clean L2 baseline remains unchanged: L2/lbfgs,
intercept, original C grid and train/validation selection, no preprocessing. Both
models' frozen selected C is 0.01. Its archive and validation grid hashes, config,
entity assignments and activation metadata are verified, and recomputed
validation diagnostics must reproduce the saved values to 1e-12.

The primary comparison reports every method at the existing model-level layer.
The secondary comparison evaluates each new method at **every** saved layer and
chooses maximum pooled validation AUROC, breaking exact ties by lower layer.
LR's secondary winner is the existing clean all-layer/C validation winner; there
is no reason to refit that unchanged baseline. Topic-specific or form-specific
metrics do not break ties. No model is refit on train+validation.

Sign checks use training data/formulas only. Mean difference is true-minus-false;
covariance-MM must preserve its nonnegative alignment through the PSD inverse.
t_G is the coefficient of true-positive tau, checked through the partial truth
regressor after accounting for tau*polarity. The polarity LR is
affirmative-positive; the TTPD head and baseline LR have classes `[0,1]`.
Training true-minus-false score means are recorded. No direction is flipped using
validation AUROC or any compound label. A low validation AUROC remains a low
AUROC. In particular, dataset offsets can affect t_G's pooled raw-score separation
without changing the sign of its OLS truth coefficient.

`MethodData` reads only statement columns on its first CSV pass. It joins those
identities to the split manifest, then skips every test row before requesting
labels on a second pass. Both source and activation sidecar must agree on the
allowed statements and labels. The primary atomic files use one statement per
physical CSV line; multiline statements are rejected for this strict skip-row
loader. Test labels are never checked for binary validity or included in hashes.
Activation files are memory-mapped read-only and indexed by permitted partition
rows before values are materialized. The loader has no test partition interface.
It does not open compound activation paths.

Run from the repository root:

```bash
python3 scripts/29_clean_atomic_method_suite.py --check-only
python3 -m unittest discover -s tests -p test_atomic_probe_methods.py -v
python3 scripts/29_clean_atomic_method_suite.py
```

Use `--model qwen25_7b` or `--model qwen3_8b` for one model. `--resume` verifies
provenance, per-layer parameter hashes and train/validation activation hashes
before reusing completed layers. Results use the new directory
`results/clean_protocol/atomic_method_suite_v1/{model}/`:

- `provenance.json`: method source URLs/hashes, implementation hashes, numerical
  versions, config, split identity, frozen LR identity, cache metadata, row counts
  and zero test/compound access declarations.
- `allowed_atomic_rows.csv`: exact permitted train/validation identities and
  labels; `burger_training_rows.csv`: the exact paired balanced training subset.
- `layers/layer_XX.npz`: fitted parameters for each new method at that layer,
  plus the unchanged L2 parameters at the fixed model layer. TTPD saves t_G,
  t_P, dataset means, polarity LR coefficient/intercept/classes, the two-dimensional
  head and the equivalent combined linear coefficient/intercept. The explicit
  two-projection score is checked against that combined coefficient.
- `layers/layer_XX.json`: metrics, training-only sign checks, optimizer details,
  parameter hash, and hashes of the selected train and validation activation
  slices. No full activation-file hash reads test values.
- `validation_metrics.csv`: all evaluated method/layer candidates, including
  pooled, affirmative, negated, all five topics and topic-macro AUROC.
- `summary.json`: primary and secondary comparisons and their parameter paths.

The reused diagnostic helper also emits zero-threshold accuracy. For an
uncalibrated direction such as t_G, that threshold has no fitted classification
interpretation; the requested comparisons and all selection use AUROC only.

No outputs overwrite historical activations, frozen clean LR parameters, or
prior exploratory analyses. Implementation, execution results and the source
sampling adaptations are kept distinct in the provenance.

## Completed validation run

The [comparison report](../results/clean_protocol/atomic_method_suite_v1/report.md)
contains both comparisons, including affirmative, negated, every topic and
topic-macro AUROC. All 28 Qwen2.5 and 36 Qwen3 layers completed. The
[verification record](../results/clean_protocol/atomic_method_suite_v1/verification.json)
confirms pickle-disabled loading of every parameter archive, finite parameters,
reproduction of all selected validation diagnostics, unchanged historical-cache
size/timestamps, and unchanged frozen-LR provenance. The 16 method-suite unit
tests pass, including formulas, sign orientation, paired sampling, forbidden-test
access guards and safe parameter serialization.

The first completed run encoded dataset-name metadata as NumPy object strings.
The serializer now writes Unicode strings. A one-time finalization reconstructed
only these names from the saved training identities, verified every numerical
array and dtype unchanged, and updated archive hashes. Original fitting
provenance remains intact; the exact fitting source is retained in
`fit_implementation_snapshot/atomic_probe_methods.py`, and the verification
record preserves old/new archive hashes. This repair does not alter any fitted
classifier or validation metric.

Because strict resume checks include implementation hashes, the current
serialization-fixed source correctly refuses to resume that original completed
run. Use the finalizer to verify existing completed outputs without refitting:

```bash
python3 scripts/30_finalize_atomic_method_report.py
```

Fresh runs use the corrected serialization and need a new configured output
directory; existing results are protected from overwrite.
