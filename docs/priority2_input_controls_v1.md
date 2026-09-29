# Priority-2 same-fact input controls v1

Implementation only. No production generation, coverage measurement, extraction,
new probe scores, bootstrap or scientific-result inspection was performed locally.
This adds one shared pipeline for the first Priority-2 bridge experiments; it does
not modify the canonical raw generator, cache, LR spec/results, scripts 33–40,
entity split or fitted probe. No chat, behavioral prompting, additional models,
new split seeds, atomic TEST or compound test paths are implemented.

## Conditions, identities and exact templates

The sole source benchmark is the byte-verified canonical
`data/clean_protocol/compounds/entity_disjoint/development_validation_v1/development_validation_compounds.csv`.
It has 8,384 examples, 524 pairs, 262 entities and five topics, with complete
AND/OR × TT/TF/FT/FF × AB/BA coverage. The raw reference is read, never regenerated
or written. Existing benchmark hash, scopes, graph invariants, Boolean semantics,
fact identities and AND/OR constituent agreement must pass before generation.

`first` and `second` below are the complete unchanged atomic fact sentences in
surface order: A/B for AB, B/A for BA. Every sentence must retain its original
text and terminal period. The versioned template ID equals its condition ID.

| Condition ID | Rows | Exact rendering |
|---|---:|---|
| `raw_reference` | 8,384 existing | Existing raw benchmark and frozen LR scores |
| `or_explicit_or_both_v1` | 4,192 new | `render_binary(first, second, 'OR').removesuffix('.') + ', or both.'` |
| `or_at_least_one_v1` | 4,192 new | `'At least one of the following is true: ' + first + ' ' + second` |
| `juxtaposition_v1` | 4,192 new | `first + ' ' + second` |
| `isolated_constituents_v1` | 524 required unique facts | Exact isolated fact sentence; only missing facts are newly extracted |

The existing `render_binary` primitive is reused unchanged for OR-both: it strips
one final period from each clause and lowercases only a leading `The ` in the
second clause, preserving the established raw grammar. The resulting text is
exactly the old raw OR sentence without its last period, followed by `, or both.`
The other two templates preserve sentence capitalization and punctuation, joined
by exactly one space. There is no arbitrary filler connective.

Every wording row retains pair/topic/split/protocol/evaluation phase, canonical
truths, order, entity IDs, fact IDs and exact fact sentences. OR controls map
one-to-one to the original raw OR `base_example_id`. Stable `base_tuple_id` is a
SHA256 identity of the full same-fact tuple without operator. Juxtaposition uses
that tuple ID as `base_example_id` and additionally records the exact corresponding
`raw_and_example_id` and `raw_or_example_id`. Its metadata contains **neither
operator nor compound_label**. No facts, negatives, pairs or labels are resampled.
New example IDs bind condition, base tuple and rendered text through SHA256.

## Exact isolated coverage audit

The benchmark's 524 unique canonical fact identities are audited against only
the validated repaired atomic TRAIN/VALIDATION metadata. The exact match key is
`(statement, entity_id, topic, truth)`. No stripping, lowercasing, fuzzy matching,
text-only matching, historical-cache fallback or test partition is permitted.

The repaired cache does not contain compound fact IDs. The explicit provenance
bridge is therefore:

`canonical fact_id -> exact fact tuple -> repaired dataset/row_index + cache split/index`.

A reused row binds to the six-file validated repaired-cache identity. Multiple
exact candidates are rejected as ambiguous. Missing identities are listed with
fact ID, entity ID, statement and their deterministic extraction ID. They become
new DEVELOPMENT/VALIDATION extraction rows. Full coverage produces **zero** new
isolated rows. The number of missing facts M is measured, never guessed;
0 ≤ M ≤ 524. Extraction contains exactly 12,576 + M statements.

Generation writes exclusively under
`data/clean_protocol/priority2_input_controls_v1/`:

- One label-aware metadata CSV per new wording/juxtaposition condition.
- `isolated_facts.csv`: exact fact identity and coverage-audit metadata.
- `isolated_sources.csv`: truth-free pointers to repaired rows or missing-fact
  extraction IDs.
- `constituent_map.csv`: each of 8,384 base IDs mapped to its A/B isolated fact keys.
- `scoring_index.csv`: only example ID, condition ID and base ID.
- `statements.csv`: only example ID, condition ID, statement, split, protocol and
  evaluation phase; no truth labels or formal operators.
- `coverage.json`: exact counts, missing/existing identities and extraction requirement.
- `generation_manifest.json`: complete marker published last, input/output hashes,
  repaired-cache identities, frozen templates and implementation source hashes.

Generation is byte-deterministic, refuses an existing destination and provides
an audit-only mode that creates no output files.

## Pinned extraction

`src/priority2_extraction.py` adapts the truth-free statement table to the existing
`ExtractionWriter`. It reuses `load_pinned_model`, the canonical readout, live
model/tokenizer binding, deterministic ten-row repaired-atomic replay and
independent-forward/repeat smoke. There is no new forward implementation.

The exact canonical contract remains: pinned model/tokenizer revision
`a09a35458c702b33eeacc393d103063234e8bc28`, Torch 2.11.0, Transformers 5.12.1,
BF16, SDPA, batch size one, no padding, exact raw input, special tokens enabled,
no chat or truncation, explicit semantic positions, cache disabled, last real
token, saved HF hidden_states[1:], embedding excluded and float16 storage.
The representation fingerprint must equal
`59df237b800f49892889e72ee6f852314b8430eb14f9c64dc0d7fff83e832821`.

Every extracted condition receives its own deterministic unpadded smoke sample
(up to four rows), checked against repeat and independent direct forward. The
exact repaired atomic float16 replay must pass. All gates run before opening the
writer. A single future `extract` invocation loads the model once, runs these
gates, then processes all condition rows at batch size one. Extraction has no
probe dependency and performs no scoring.

New cache:
`acts/clean_protocol/priority2_input_controls_v1/qwen25_a09a354_bs1_bf16_v1/`

- `metadata.csv`: byte-identical truth-free `statements.csv` sidecar.
- `activations.npy`: C-order float16, shape `[12576 + M, 28, 3584]`.
- `extraction_manifest.json`: representation, tokenizer/model identities,
  generation hashes, per-condition counts, runtime, code and replay/smoke evidence.
- `progress.json`: durable contiguous row journal and completed full-file hash.

The shared writer uses a partial array until finalization. Existing output trees
are refused; this v1 entry point has no resume or overwrite option. Historical
compatibility failure remains informational, never an extraction gate.

## Scoring, freeze and truth separation

Only the canonical repaired frozen LR is used: saved layer 17, C=1.0, read from
validated artifacts. Its NPZ/grid hashes, classes, convergence/selection semantics,
repaired cache bytes and frozen LR spec binding are checked using the existing
scoring-specific verifier. No LR or other method is fitted or recalibrated.

CPU preflight checks all generated/extracted file hashes, NPY shape/order/dtype,
complete contiguous progress, canonical representation and passing gates. It
reads only identity/scope projections for raw compound metadata. Truth-bearing
generated fact metadata is projected to fact key/ID, statement, entity and topic
to recheck exact repaired-cache pointers; no constituent truth column is parsed.
Statement/pointer/index headers are checked before dataframe parsing. Activation
files are hashed as opaque bytes; no numerical activation matrix or probe scores
are materialized during preflight/freeze.

The new strict production spec is
`config/clean_protocol/priority2_input_controls_v1.json`.
It pins actual generation/extraction/LR-reference hashes and sizes, counts,
fixed templates, the LR selection, metrics, bootstrap, contrasts, composition,
prohibitions and output schemas. Unknown keys and changed policies are rejected.
No production spec is shipped with this implementation. Preflight must succeed on
the real artifact host, then freeze must revalidate it. Review and commit the
exact spec before scoring: the scorer checks its bytes against Git HEAD.

Scoring uses one BLAS thread, selected-layer float64 batches and unchanged
`X @ coef + intercept`. All isolated scores come from actual isolated activation
rows, either the exact repaired row or a freshly extracted missing-fact row.
Compound activations are never used to infer an isolated score.

New result root: `results/clean_protocol/priority2_input_controls_v1/`.
Under `scores/`:

- `condition_scores.csv`: exactly
  `example_id,condition_id,base_example_id,frozen_probe_score` (12,576 rows).
- `isolated_scores.csv`: exactly `fact_key,frozen_probe_score` (524 unique facts).
- `scoring_manifest.json`: spec/source/input identities, frozen probe hash, layer/C,
  exact schemas/counts, ordered ID hashes, zero fits/test/compound-label use;
  published complete=true last.

Scores use 17 significant digits. No predictions, operators or cells are copied
into these tables. The evaluator verifies score/spec/metadata hashes and exact
coverage before loading truth-bearing raw metadata, joins by identities, and
reconstructs expected generated metadata to check unchanged same-fact content.
It never opens activation arrays or probe archives.

## Evaluation and scientific interpretation

Formal tables report the unchanged LR primary and direct-boundary metrics for:
raw reference, each OR wording condition (with **exactly unchanged raw AND
scores**), and continuous external composition. All report pooled, each of five
topics, and equal-weight topic macro. This includes AND/OR AUROC, their gap,
AND TT-vs-mixed, OR mixed-vs-FF, and all six direct component boundaries.

External scores and threshold decisions are frozen:

- AND continuous = min(s_A, s_B); OR continuous = max(s_A, s_B).
- b_A = s_A >= 0; b_B = s_B >= 0.
- AND Boolean = b_A and b_B; OR Boolean = b_A or b_B.

These are **two isolated model evaluations plus a known parse/external
composition rule**, a positive-control-style comparison. They do not show that
the model internally implements min/max. Context evaluations can be reused for
repeated identical facts; every base row still maps to its exact A/B scores.
Boolean metrics are separate accuracy, balanced accuracy and cell true-response
rates; no Boolean AUROC is reported. Internally false/true map to -1/+1 for the
shared >=0 metric planner, so false is never mistakenly treated as threshold zero.
The 75% majority-class raw-accuracy baseline is included.

Juxtaposition is **descriptive geometry only**. Its separate output reports:
TT-vs-mixed, mixed-vs-FF, TT-vs-FF and TF-vs-FT geometry AUROCs, plus cell n, mean,
population SD, median and q10/q25/q75/q90, pooled and by topic. Geometry AUROCs also
have topic macro. It receives no formal operator, truth-decoding accuracy or
connective AUROC and never enters the formal transfer tables.

One unchanged LR-v1 endpoint schedule is shared across all conditions:
2,000 draws, seed 1729, PCG64, sorted topic/entities/pairs, topic-stratified entity
multiplicities, m_i*m_j pair weights, no topic-total normalization. Every variant
of a pair shares that weight, including juxtaposition. Its schedule SHA must
match the raw reference. Raw LR primary estimates/CIs must reproduce before a
completed new evaluation can be published.

Predeclared paired differences, in every scope:

- Each OR wording condition minus raw: OR AUROC, AND-minus-OR gap,
  OR mixed-vs-FF, OR TF-vs-FF, OR FT-vs-FF, OR TT-vs-FF.
- External min/max minus raw: AND AUROC, OR AUROC, their gap,
  AND TT-vs-mixed, OR mixed-vs-FF.

Subtract both sides inside the same replicate. Undefined components propagate;
no redraw, imputation or topic dropping. Require at least 1,800 valid replicates
for 95% pointwise percentile CIs, linear interpolation. Every interval reports
valid/invalid/total counts, fraction and status; insufficient draws give null
bounds. AUROC ties receive half credit. No multiplicity adjustment, template
ranking, winner selection, sign flipping or compound-driven tuning is performed.

Under `evaluation/`: formal metric CSV/JSON, paired contrasts, Boolean metrics,
separate juxtaposition geometry and cell statistics, bootstrap summary, float64
draws/validity masks, common integer pair weights, and a completion manifest
published last. Calculations need only frozen scores, generated/base metadata,
spec and immutable raw LR references—not model weights or activation arrays.

## Future artifact-host commands

First CPU generation/audit (no model or probe scores):

```bash
python scripts/41_generate_priority2_input_controls.py --mode audit
python scripts/41_generate_priority2_input_controls.py --mode generate
```

Then future GPU session:

```bash
python scripts/44_extract_priority2_input_controls.py --mode plan
python scripts/44_extract_priority2_input_controls.py --mode smoke
python scripts/44_extract_priority2_input_controls.py --mode extract
```

The standalone smoke is optional; `extract` always reruns all live gates in its
own model session. `plan` loads no model. Canonical input files must already be at
the fixed repository-relative locations; no symlinks or arbitrary path overrides.

Finally CPU preflight/freeze, followed by review and commit before scoring:

```bash
python scripts/42_score_priority2_input_controls.py --mode preflight
python scripts/42_score_priority2_input_controls.py --mode freeze-spec
# Review and commit config/clean_protocol/priority2_input_controls_v1.json.
python scripts/42_score_priority2_input_controls.py --mode score
python scripts/43_evaluate_priority2_input_controls.py
```

Every output refuses overwrite. No step generates missing canonical inputs or
fabricates production hashes. Scientific choices live in versioned code/spec,
not command-line overrides.

## Synthetic tests

`tests/test_priority2_input_controls.py` uses synthetic facts, tiny activation
widths and deterministic fake readouts. It exercises full 8,384-row generation,
byte determinism, coverage/missing/ambiguous identities, all extraction gates and
shared writer, frozen affine scores, truth-blind parsing, label-aware evaluation,
Boolean zero handling, paired contrasts, no archive reads in evaluation and
byte preservation of all synthetic raw references. No real model loads or new
scientific results are inspected by these tests.
