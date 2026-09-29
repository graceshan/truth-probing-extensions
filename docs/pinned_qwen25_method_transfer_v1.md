# Pinned Qwen2.5 multi-method compound transfer v1

Implementation only. No production spec or missing scientific outputs are
fabricated in this checkout. The canonical repaired atomic arrays, completed
method suites, compound cache and LR reference outputs must exist on the artifact
host before preflight can succeed. Existing scripts 33–38, the LR v1 spec,
canonical caches and fitted probes are unchanged.

## Frozen comparisons

All five methods are included: `l2_logistic`, `difference_of_means`,
`mass_mean_covariance`, `burger_t_g`, `ttpd`.

| Group | Role | Atomic artifact/layer |
|---|---|---|
| `faithful_common_layer` | Primary method comparison | Faithful suite, all methods at saved layer 17 |
| `faithful_method_selected_layer` | Secondary robustness | Faithful suite's atomic-VALIDATION AUROC winners, ties to lower layer |
| `matched_1000_common_layer` | Supplementary training-data control | Matched suite, all methods at saved layer 17 |

Faithful LR remains the canonical unchanged TRAIN-fitted layer-17/C=1 classifier.
DoM/MM use 3,144 TRAIN rows; t_G/TTPD use the exact paired balanced 1,000-row
Bürger sample. Matched methods all use those same ordered 1,000 rows, including
a separately fitted C=1 LR. Both suites use the same 1,040 atomic validation rows.
No fitting is exposed by the new CLIs. Compound outcomes never determine method
inclusion, layer, C, sign, threshold, scale, normalization or training subset.
The previously observed LR results are reproduction targets, never selection
criteria. There is no method winner or ranking score.

## Artifact validation and preflight

Fixed runtime inputs are the existing repaired atomic cache, canonical frozen LR,
canonical compound cache, both pinned atomic-method output trees, and the existing
LR v1 score/evaluation tree. Paths are the repository-relative canonical paths
used by scripts 35–38; arbitrary path overrides and symlinks are rejected. If
RunPod persistent copies are elsewhere, restore verified byte-identical copies
to these fixed locations before using this pipeline.

The read-only verifier reuses `RepairedAtomicCache`, the atomic-only frozen LR
verifier, the repaired method adapter, `choose_layers`, and the archive validation
routine. It checks completion; every expected output hash and byte size; suite
role and policy; exact repaired representation and LR byte identity; sample and
partition identities; no test/compound access or sign flip; pickle-free finite
parameter schemas; 113 faithful or five matched metric rows; all 28 non-LR layers;
LR only at 17; and exact agreement between layer records, metric CSV and summary.
Secondary winners are reconstructed from atomic validation records, never from
compound scores. Faithful and matched Bürger CSVs must be byte-identical.

Preflight additionally reapplies archived readouts to **atomic VALIDATION only**
to reproduce their archived diagnostics and verify TTPD explicit/fused scores
(rtol=1e-9, atol=1e-8). It performs zero fits and zero compound scores. This is
stronger than hash-only checking: it materializes atomic validation layer slices,
recorded in `method_binding.json` counters. It never materializes compound
activation values or compound truth columns during preflight/freeze. The
compound completion, hashes, representation, replay/smoke evidence and projected
row identities use the existing LR scoring verifier.

LR reference integrity is pinned to the already-frozen LR specification and
recorded canonical primary/evaluation identities. `row_scores.csv` is verified
through its complete scoring manifest, frozen probe/cache binding, exact row
order and one-to-one coverage. The existing primary file is hashed as opaque
bytes during scoring preflight; its metric values are read only by evaluation.
Missing references block production use rather than disabling reproduction gates.

## Strict specification and commands

The new schema/validator is implemented in `src/method_transfer_contracts.py`.
It rejects unknown keys, changed scientific policies, missing/placeholder hashes,
unexpected paths, methods or group/layer combinations. Every suite output and
LR reference identity is measured on the artifact host. The new spec embeds the
validated LR scientific definitions without modifying the LR spec itself.

Run later from the repository root on RunPod CPU:

```bash
python scripts/39_score_pinned_method_transfer.py --mode preflight
python scripts/39_score_pinned_method_transfer.py --mode freeze-spec
```

Preflight writes
`results/clean_protocol/pinned_qwen25_method_transfer_preflight_v1.json`.
Freeze revalidates all inputs and publishes
`config/clean_protocol/pinned_qwen25_method_transfer_v1.json` exclusively.
Review and **commit that exact specification before scoring**. Scoring checks
its bytes against `git show HEAD:config/clean_protocol/pinned_qwen25_method_transfer_v1.json`.
These commands neither fit nor compute compound scores.

Only after the frozen spec is reviewed and committed:

```bash
python scripts/39_score_pinned_method_transfer.py --mode score
python scripts/40_evaluate_pinned_method_transfer.py
```

There are no CLI overrides for paths, methods, layers, thresholds, bootstrap,
sign or normalization. Every preflight/spec/scoring/evaluation destination
refuses overwrite, including incomplete result trees. A changed scientific
policy requires a new version. The above production commands were not executed
on real artifacts during this implementation.

## Stage 1: finalized, truth-column-blind scores

Full metadata is hashed as opaque bytes. Parsing uses the strict existing
projection of example ID and scope fields. It never loads canonical/surface
truths, compound labels, operators or derived cells into scoring logic.
Selected activation-layer batches become float64 and use one BLAS thread:

`score = X @ coef + intercept`

TTPD uses its saved fused affine coefficients. All methods retain their original
raw scale and intercept; no normalization, centering, z-scoring, calibration or
sign change is introduced. The five-column long artifact is exactly:

```text
example_id,analysis_group,method,atomic_layer,frozen_probe_score
```

There are 15 conditions × 8,384 examples = 125,760 rows. Each condition retains
canonical activation-row order, unique IDs and finite float64 scores serialized
with 17 significant digits. Identical conditions are stored explicitly; no alias
resolution is required. The common-layer faithful LR scores must reproduce the
existing LR per-row scores at rtol=0, atol=1e-12 before publication.

New output root:
`results/clean_protocol/entity_disjoint_method_transfer_qwen25_pinned_v1/`

- `method_scores.csv`
- `method_binding.json`: exact suites/archives/method/layer conditions, atomic
  validation reproduction evidence and counters, no fits/test/compound labels.
- `scoring_manifest.json`: spec SHA, all input bindings, ordered example-ID hash,
  exact schema/count, output hashes, source hashes, truth-blind declarations and
  LR reproduction evidence. Published complete=true **last**, atomically.

## Stage 2: labels after identity gates

Evaluation validates the complete scoring manifest, binding, spec hash, score
and canonical metadata hashes, exact condition/layer set, unique IDs and exact
one-to-one coverage for every condition **before** loading truth-bearing
metadata. It then validates the same graph/Boolean semantics as LR v1 and joins
by example ID. It never opens atomic or compound activation arrays or fitted
parameter archives, and never fits anything.

The unchanged LR v1 metric planner implements identical primary and direct
boundary AUROCs, descriptive same-label geometry, pooled AND+OR geometry and
matched OR−AND mean shifts. Each primary/direct metric is reported pooled,
for each of the five topics, and as the equal five-topic macro. Geometry remains
separate from direct truth boundaries. The headline method comparison is ranking
transfer, not threshold accuracy.

Threshold applicability is frozen per method:

- LR and TTPD: score >= 0 is the frozen binary logistic decision threshold;
  report raw/balanced accuracy and cell true-response fractions, with the 75%
  majority-class raw-accuracy baseline.
- DoM, covariance-MM and raw t_G: no threshold metrics are emitted. Their
  origin-based raw projection zero is not asserted to be a calibrated classifier.
  This exclusion and reason are recorded explicitly in the spec/manifests.

## Shared endpoint bootstrap and paired contrasts

One `EntityBootstrap` instance is shared by all conditions: 2,000 replicates,
seed 1729, PCG64, sorted topic/entity/pair order, topic-stratified entity
multiplicities, pair weight m_i*m_j, all 16 variants sharing the pair weight.
No topic-total renormalization, redraw or imputation. Topic macro requires all
five topic metrics in the same replicate. Intervals are 95% pointwise percentile,
linear quantiles, minimum 1,800 valid draws. Every CI reports valid/invalid/total
counts, valid fraction and status; inadequate coverage gives null bounds. The
same-label geometry and raw-score scales remain method-specific descriptive
quantities; no cross-method normalization is used.

Predeclared contrasts, for each of the five primary metrics and all seven scopes:

1. Each non-LR minus LR within `faithful_common_layer` (four comparisons).
2. Matched minus faithful common-layer for each method (five supplementary
   comparisons isolating changes in training exposure).

Both sides are subtracted within the identical replicate. A contrast is invalid
when either component is invalid; macro validity requires all five topics.
There is no winner score, selective metric reporting, or multiplicity-adjusted
inference claim. These are paired effect estimates conditional on the frozen
atomic fits, not uncertainty over retraining.

The regenerated schedule SHA must equal the canonical LR evaluation schedule.
Common-layer faithful LR primary estimates and CIs must reproduce the frozen
LR primary table (atol=1e-14, rtol=0), including identical validity counts and CI
status. A reproduction failure prevents a completed evaluation manifest.

## Evaluation outputs and reproducibility

Under the new root's `evaluation/`:

- `primary_metrics.csv/json`, `boundary_metrics.csv`, `geometry_metrics.csv`
- `topic_metrics.csv`, `threshold_metrics.csv`, `paired_method_contrasts.csv`
- `bootstrap_summary.csv/json`
- `bootstrap_draws.npz`: stable condition/contrast metric IDs, float64 replicate
  values and validity masks; safe to read with allow_pickle=False.
- `bootstrap_pair_weights.npz`: ordered pair IDs and common integer schedule.
- `evaluation_manifest.json`: published last with spec/score/metadata/source and
  output identities, schedule SHA, threshold applicability and reproduction gates.

Scientific calculations are reproducible on CPU from scores, canonical metadata
and the frozen spec without model weights. Production gate verification also
requires the immutable LR reference score/primary/manifest files. The evaluator
requires neither activation caches nor atomic parameter archives.

Tests use temporary synthetic atomic fits and a synthetic degree-four compound
benchmark, never production data. They cover suite corruption, sample/selection
binding, affine resolution, TTPD equality, truth-blind scoring, LR reproduction,
shared bootstrap/paired differences, topic macro, threshold applicability,
forbidden evaluation I/O and preservation of every existing synthetic source
artifact. Existing generic statistics tests cover weighted AUROC ties and
undefined-replicate behavior.
