# Matched-training-data control

This addendum uses the exact ordered 1,000 TRAIN rows saved in each faithful
run's `burger_training_rows.csv`, and the same 1,040 VALIDATION rows. No new
sampling is performed. All five methods are fitted on that same training matrix:
L2 logistic regression, difference of means, full covariance-adjusted mass mean,
Bürger t_G and full TTPD. The existing formulas, sign checks, float64 numerics and
one-thread BLAS regime are reused without changing the faithful implementation.

The primary layers remain saved index 17 for Qwen2.5 and 28 for Qwen3. LR is
**refitted**, retaining C=0.01 and the existing clean solver/convergence policy.
C is not retuned, so the LR comparison changes the training subset while keeping
the established hyperparameters fixed. No optional secondary layer sweep is
performed. There are no validation-based sign flips or train+validation refits.

The strict atomic loader continues to skip TEST label rows before loading labels
and materializes only TRAIN/VALIDATION activation slices. No compound data is
opened. Original saved row identities, labels and order must match the current
TRAIN partition and the faithful full-record hash. The validation identities
and activation slice must match the faithful run too. Both models must use
identical full-record hashes for their training and validation rows.

Outputs are separate under `results/clean_protocol/atomic_method_matched_v1/`:

- `report.md`: overall, affirmative, negated, per-topic and topic-macro AUROC,
  plus the fixed-layer overall AUROC difference from the faithful result.
- Each model directory: fitted `parameters.npz`, `validation_metrics.csv`,
  `summary.json`, and `provenance.json` with source hashes, numerical versions,
  method settings, optimizer diagnostics, row hashes and activation-slice hashes.
- `training_rows.csv` is a byte-identical copy of the saved Bürger rows.
  `validation_rows.csv` preserves the common validation order. `row_ids.json`
  records each ID as `dataset:original_zero_based_row_index`; provenance includes
  hashes of both ordered IDs and complete records including statement/label.
- `faithful_files_before.json` and `verification.json` record a before/after
  content-hash audit of every file in `atomic_method_suite_v1`.

Every parameter archive is loaded back with pickle disabled, and every validation
diagnostic is reproduced. Because t_G and TTPD already used this training subset,
their newly fitted parameters and diagnostics must exactly reproduce the faithful
fixed-layer artifacts. A mismatch fails rather than being accepted silently.

```bash
python3 -m unittest discover -s tests -p 'test_atomic*method*.py' -v
python3 scripts/31_matched_atomic_methods.py
```

The runner refuses an existing output directory and rejects paths overlapping
the faithful output tree. Existing faithful and matched results are never
overwritten. Run once; a future fresh run requires a new configured output path.
Historical activation files and frozen LR artifacts are read-only.

The covariance method keeps its original absolute pseudoinverse cutoff of 1e-3;
there is no shrinkage or diagonal approximation. t_G and TTPD retain the original
dataset centering, learned polarity projection and unregularized two-feature
truth head described in [the faithful-method documentation](clean_atomic_method_suite.md).
