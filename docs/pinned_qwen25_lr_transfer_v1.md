# Frozen pinned Qwen2.5 compound transfer v1

This is a CPU pipeline with two separate processes: truth-column-blind scoring,
then label-aware evaluation. Implementation tests use temporary synthetic data
only. No production analysis specification is shipped: the canonical runtime
artifacts were absent from the implementation checkout. No real compound scores
or transfer metrics were computed during implementation.

The existing repaired atomic cache, frozen probe, compound cache, their producer
modules, and scripts 34/35 are unchanged. Historical Qwen2.5 activations and
compatibility reports are never inputs. Historical compatibility failure remains
informational extraction provenance, never a passing gate.

## Fixed inputs and future commands

Run these later from the repository root on the machine containing the artifacts:

```bash
# Hash/metadata validation only; no activation values or scores computed.
python scripts/36_score_pinned_compounds.py --mode preflight

# Revalidate current artifacts against preflight and exclusively publish real spec.
python scripts/36_score_pinned_compounds.py --mode freeze-spec

# Only after freezing/reviewing the spec: future scoring command.
python scripts/36_score_pinned_compounds.py --mode score

# Separate process; only after scoring_manifest.json declares complete.
python scripts/37_evaluate_clean_transfer.py
```

The first two commands inspect frozen coefficient parameters for integrity but
never evaluate them on activations. Whole activation files are streamed as opaque
bytes for SHA256 and their NPY headers are read. No activation matrix is
materialized. All commands are fixed-path; there is no test, alternate probe,
layer/C, threshold, or arbitrary input-path override.

Inputs:

- `acts/clean_protocol/atomic/qwen25_a09a354_bs1_bf16_v1/`
- `results/clean_protocol/atomic_probes_pinned_v1/qwen25_7b/`
- `acts/clean_protocol/entity_disjoint/development_validation_v1/qwen2_5_7b_a09a354_bs1_bf16_v1/`

Preflight publishes
`results/clean_protocol/pinned_qwen25_lr_transfer_preflight_v1.json`.
Freeze publishes
`config/clean_protocol/pinned_qwen25_lr_transfer_v1.json`.
Both refuse overwrite. The spec is a strict versioned schema implemented in
`src/clean_transfer_contracts.py::validate_spec`: unknown keys, altered frozen
scientific definitions, placeholders, noncanonical input paths, missing or extra
file identities, malformed hashes and sizes are rejected. Every production hash
must be measured. The preflight report contains a candidate only; scoring requires
the separately finalized spec. Review and commit the finalized spec before
running score. Any scientific change requires a new analysis version and output
root; v1 intentionally has no override flags.

CPU runtime libraries: NumPy, pandas, scikit-learn (for recorded version/test
comparison), threadpoolctl; the evaluator's centralized Boolean helper also
imports the repository's normal Python dependencies. No model weights, CUDA,
Torch or Transformers are required by the new production modules. Pytest and the
existing extraction test dependencies are needed for the synthetic test suite.

## Binding and selection-time code provenance

`RepairedAtomicCache` validates only its six fixed files: manifest, completion,
and train/validation activation/metadata files. All hashes, byte sizes, schemas,
counts, disjoint entity identities, approved input digest, shapes/dtypes, model
and tokenizer revisions and the canonical extraction contract are checked.

The scoring-specific verifier follows the archive format and semantics of
`clean_atomic_probes.fit_and_save`, without calling its fitting APIs or changing
its stricter general-purpose frozen loader. It checks archive and validation-grid
hashes, [0,1] classes, coefficient/intercept dimensions and finiteness, selected
layer/C, all 140 converged grid entries, the frozen rule and validation winner,
no preprocessing, true-positive orientation, no test scoring, and the repaired
selection configuration. The producer's insertion-order configuration hash is
reconstructed in its original key order; representation JSON uses sorted keys.
The expected layer 17/C=1.0/AUROC is an assertion after validated artifacts supply
the parameters. The saved TRAIN-fitted coefficient archive is never modified or
refitted, and validation data are never added to its fit.

Selection-time repaired file identities, model, counts, approved digest and
extraction contract must equal the current strict cache. Adapter source hashes
need not equal today's files. `representation_binding.json` preserves recorded
source hashes, current hashes where available, and checks recorded source bytes
against the recorded Git revision when that object is locally available. A
missing object or differing committed source is explicitly reported; it is never
rewritten as verified. The frozen spec pins selection.json itself, and byte-exact
cache/probe identities are mandatory regardless of source-audit availability.
This is provenance reporting, not a bypass of any cache or coefficient check.

The compound descriptor must hash to
`59df237b800f49892889e72ee6f852314b8430eb14f9c64dc0d7fff83e832821`.
Its descriptor, pinned model/tokenizer identities and repaired-cache identity
must agree with the repaired atomic producer. Recorded exact atomic replay and
unpadded compound smoke must have passed. Model forwards are never rerun.
The completion journal is `progress.json`; its identity, contiguous coverage and
final hash must agree with the full-file digest. Header shape is
`[8384,28,3584]`, C-order float16. A partial file blocks scoring.

Pinned hashes:

- Activation: `ed9f6a642f35ec185b6f6b63c4bc4a7dafb754012e967274560bc8aef43d07ac`
- Metadata: `96e96515e780086065694155e3061bdfe6f4bb80af062a63c2deecd6bfad6f94`

Paths are explicitly allowlisted and symlinks/escapes rejected. No original
atomic CSV, entity partition manifest, historical cache or test artifact is
opened. Repaired atomic TRAIN/VALIDATION metadata are permitted for cache
validation; compound truth columns are not permitted in scoring.

## Stage 1: finalized truth-blind scores

Metadata is hashed in full as opaque bytes. CSV projection returns only
`example_id`, `split`, `protocol`, `evaluation_phase`, and optional
`benchmark_label`. It never returns `compound_label`, canonical/surface truth
columns, or a derived cell. CSV tokenization may scan their bytes; no full
truth-bearing dataframe is constructed and subsequently filtered.

Only batches from the validated selected activation layer are materialized as
float64. Scores are `X @ coef + intercept`, without normalization, centering,
calibration, sign change or thresholding, using one BLAS thread and batch size
256. Synthetic tests compare this operation to reconstructed sklearn
`LogisticRegression.decision_function`.

Output root:
`results/clean_protocol/entity_disjoint_transfer_qwen25_pinned_v1/`

- `row_scores.csv`: exactly `example_id,frozen_probe_score`, 8,384 rows in canonical
  activation-row order, finite float64 scores written with 17 significant digits.
- `representation_binding.json`: additive receipt in the NEW output directory;
  no changes to selection.json, selected_probe.npz or extraction artifacts.
- `scoring_manifest.json`: complete marker published LAST using durable temporary
  bytes and no-clobber atomic linking. Includes input hashes/sizes, model and
  fingerprint, repaired cache identity, selected probe/layer/C, spec hash,
  ordered ID hash, output hash/schema, source/runtime hashes and truth-blindness
  declarations. Input bytes are rechecked before publication.

An existing output directory, including an incomplete run, is refused. No resume
or overwrite exists. Evaluation rejects any missing/incomplete scoring manifest.

## Stage 2: identity gate, then labels

The evaluator checks completion, score/spec/metadata hashes, exact schema/count,
finite scores, unique IDs, exact one-to-one coverage and ordered ID hash using
an identity-only projection. Only then does it load full canonical metadata.
It joins by example_id with one-to-one validation, rechecks metadata identity,
and validates Boolean labels with `clean_compounds.boolean_truth`.

Graph checks: 262 entities, 524 unique unordered non-self pairs, degree four,
five topics, topic-consistent entities, invariant constituent facts, 16 variants
per pair covering AND/OR × canonical TT/TF/FT/FF × AB/BA, and 8,384 rows.
Pair counts by topic are 298/68/90/36/32 for cities/sp_en_trans/inventors/
element_symb/animal_class. No test scope is allowed. BA never relabels canonical
TF as FT.

The methodological statement supported by this design is:
**The frozen compound scores were finalized before evaluation code loaded
compound truth labels.**

## Frozen metrics and statistics

Primary direct transfer:

1. AND AUROC: TT positive; TF/FT/FF negative.
2. OR AUROC: TT/TF/FT positive; FF negative.
3. AND AUROC minus OR AUROC.
4. AND TT-vs-mixed: TT positive; TF/FT negative.
5. OR mixed-vs-FF: TF/FT positive; FF negative.

Additional direct boundaries: AND TT/TF, TT/FT, TT/FF and OR TF/FF, FT/FF, TT/FF
(first group positive). AUROCs use individual examples, not cell-averaged scores.
Ties receive half credit; no score orientation is chosen using compounds.
All primary, boundary and threshold metrics report pooled, each of five topics,
and an equal five-topic macro.

Descriptive same-label geometry: OR TT-vs-mixed and AND mixed-vs-FF. These have
explicit `geometry` names/categories and never enter `boundary_metrics.csv`.
Pooled AND+OR AUROC is also descriptive, not primary.

Frozen threshold is score >= 0 (including exact zero). Report AND/OR raw and
balanced accuracy and all eight cell true-response fractions. Balanced accuracy
is (TPR+TNR)/2. The AND 25%/75% and OR 75%/25% imbalance gives a 75% majority-class
raw-accuracy baseline, reported with operator-level accuracy metrics.

Cell distributions and matched shift summaries use n, mean, population SD
(ddof=0), median, q10/q25/q75/q90. Quantiles use NumPy method='linear'. These
summaries are descriptive point estimates, not selection criteria.

AND/OR matching uses pair_id, canonical truth pattern and ordering with invariant
topic, endpoints and fact IDs. Require 4,192 matches, 1,048/cell. Save OR minus
AND per match; summarize overall, by canonical cell and by topic. TT/FF preserve
compound truth (connective-associated shifts); TF/FT change truth (connective
and logical truth effects combined). These are descriptive geometry. Their mean
shifts receive endpoint-bootstrap intervals.

## Endpoint uncertainty

Use 2,000 replicates, seed 1729, NumPy Generator(PCG64). Sort topics, entity IDs
and pair IDs. Within each topic draw n-entity multinomial multiplicities, with
uniform 1/n probabilities; pair weight is m_i*m_j. All 16 variants share it.
No row or ordinary pair bootstrap is used. Pooled weights are not renormalized
by topic. Every macro replicate averages all five topic metrics from that same
replicate. Contrasts subtract within the same replicate; matched differences
receive their pair weight once.

Compute weighted tie-aware AUROC, weighted accuracy/TPR/TNR/means. Synthetic tests
verify sklearn AUROC agreement for unit and nonuniform weights including ties.
Metric calculations canonically sort IDs so row permutations preserve results.

Intervals: 95% pointwise percentile, NumPy method='linear'; no multiplicity
adjustment claimed. Undefined comparisons remain invalid: zero class weight,
either undefined contrast component, or any missing macro topic. No redraw,
imputation or topic dropping. Require at least 1,800 valid replicates. Otherwise
bounds are null with ci_status='insufficient_valid_replicates'. Every interval
reports total/valid/invalid counts and valid fraction. Point estimates always
use original unit weights. Inference is conditional on the frozen probe and
benchmark construction, not atomic-training uncertainty.

Evaluation writes the specified primary/boundary/threshold/geometry/cell/topic/
matched tables, bootstrap summary CSV/JSON, float64 draws with validity masks,
and [2000,524] integer pair weights with ordered pair IDs. The final evaluation
manifest records all hashes, the spec hash, schedule hash and numerical versions.
It opens no activation array or coefficient archive. All paper metrics are
rebuildable on CPU from the two-column score table, frozen metadata and spec.

## Synthetic verification

```bash
python -m pytest -q tests/test_pinned_compound_scoring.py tests/test_clean_transfer_evaluation.py
```

Fixtures fabricate their arrays, labels, graph and frozen archive in temporary
directories. Tests include the full 8,384-row graph without real data, a full
2,000-replicate synthetic evaluation, forbidden-read guards, truth projection,
corruption/ID failures, frozen sklearn equivalence, finalization order and
bootstrap invariance/validity. They do not compute production scores.
