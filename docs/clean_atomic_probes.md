# Clean atomic union probes

The shared implementation is `src/clean_atomic_probes.py`; the two cache layouts
are configured in `config/clean_protocol/atomic_probes.json`. The exploratory
scripts and `src/probes.py` are not changed or called.

```bash
python3 -B scripts/20_clean_atomic_probe.py --check-only
python3 -B -m unittest discover -s tests -p 'test_clean_atomic_probes.py'
```

Structural checks read atomic source/sidecar metadata and activation headers,
without inspecting activation values. Exact parsed `(topic, entity)` joins to
the manifest determine all splits. Affirmative and negated entity coverage must
match each other and the manifest. Every entity is retained regardless of its
compound usability. Source hashes and row-aligned sidecars are verified; no old
exploratory split or `GroupShuffleSplit` is used.

The default command runs train/validation selection. Use `--model qwen25_7b` or
`--model qwen3_8b` for one model; the default is both. Both models must produce
the same assignment hash. Configuration paths are relative to the repository.

For every transformer layer and exactly `C=[0.001,0.01,0.1,1.0,10.0]`, fit L2
logistic regression with `lbfgs`, an intercept, `max_iter=2000`, no class weights,
and `tol=0.0001`. Train rows alone enter `.fit()`. Validation rows alone enter
`.decision_function()` during selection. There is no fitted preprocessing or
normalization; activation values are cast to float64 for the solver. BLAS runs
with one thread. Each fit starts with `max_iter=2000`. A sklearn
`ConvergenceWarning` triggers exactly one refit from scratch with identical data
and parameters except `max_iter=10000`. No warning means no retry. Only the
converged fit is scored on validation and entered into the selection table. If
the retry also emits a convergence warning, the run raises an error before
scoring that fit or saving a selection. Other warnings do not trigger retries.

Every grid row records `initial_max_iter`, `initial_n_iter`,
`initial_convergence_warning`, `retry_needed`, `final_max_iter`, `final_n_iter`,
`final_converged`, `final_convergence_status`, and initial warning messages.
Legacy `n_iter` and `convergence_warning` columns now describe the accepted final
fit. This is an optimization policy, not a change to the grid or selection rule.

Selection maximizes pooled validation AUROC, then smaller C, then the lower
zero-based transformer layer (embedding output excluded). Diagnostics are
affirmative, negated, per-topic and topic-macro validation AUROC, plus accuracy
at score >= 0. Undefined diagnostic AUROCs are null; undefined pooled AUROC is
an error. Diagnostics do not break ties. Labels remain 0=false, 1=true. The
selected coefficients are frozen as fitted on train; no train+validation refit.

## Test isolation

Structural loading may check source test labels for binary validity and sidecar
alignment, but default data objects retain labels only for train and validation.
Default access to a test partition raises an error. Activation matrices load
only the requested partition's row indices; there is no all-row decision pass.
Selection receives distinct train and validation partition interfaces and
rejects reversed or overlapping partitions.

Final evaluation is a separate explicit `--evaluate-test` mode. Do not run that
mode during development. It requires an existing `selection.json`, loads the
frozen coefficient archive, verifies archive/validation hashes, model, manifest,
source metadata and configuration, and evaluates only the saved layer and C.
It never fits or selects. Its metrics are written under `final_test/`, separately
from immutable selection artifacts. Only this branch authorizes retaining test
labels and reading test activation values.

## Outputs

Canonical per-model outputs live under
`results/clean_protocol/atomic_probes_converged/`. The initial 2,000-iteration
run under `atomic_probes/` and the separate convergence audit remain unchanged.

- `structural_checks.json`: optional check-only output; entity/row counts, source
  and sidecar hashes, activation shapes/dtypes/sizes/mtimes, manifest hashes.
- `validation_metrics.csv`: every layer/C combination and validation diagnostics.
- `selected_probe.npz`: coefficients, intercept, classes, layer, C, iteration count,
  and the final optimizer iteration budget.
- `selection.json`: exact rule, chosen validation metrics, hyperparameters,
  structural provenance, code/config hashes, Git revision, runtime versions,
  optimization policy and retry count, `test_evaluated=false`, and
  `test_decision_scores_computed=0`.
- `split_counts.csv`: train/validation/test counts, including each atomic form.
- `final_test/test_metrics.json`: created only by explicitly authorized final
  evaluation; records `test_evaluated=true` and frozen probe identity. Selection
  metadata continues to describe the original validation-only run.

Source activation provenance records file metadata and extraction-manifest hash
where available; it does not compute full activation-array hashes. Saved output
files cannot be overwritten with differing contents. A changed run needs a new
clean results location. No row-level labels or scores are saved by default.
