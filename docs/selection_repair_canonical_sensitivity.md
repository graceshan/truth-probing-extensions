# Canonical sensitivity implementation and correction hold (2026-10-01)

**Current status: no real fits or scores. A reviewed successor correction is required.**
The user's subsequent notification identified `inventors:23`, `inventors:157`,
and their paired negations as requiring quarantine. The v2 adoption, mappings,
and score-independent masks were prepared before that notification and remain
unchanged historical preparation. They are not current corrected membership.
The separately pinned `canonical_sensitivity_correction_hold_20261001.json`
supersedes the earlier fitting authorization. `run()` rejects execution before
reading prepared inputs, loading arrays, creating output directories, or fitting.
Deleting or editing the hold does not enable this v2 runner; there is no bypass flag.

## Scope and provenance

Branch `t2-canonical-sensitivity-20261001` normally merged exactly
`827f01460802afde5dd82da424159f284812623c` into reviewed
`455016040208d0304b4b7547bb96dc626c979ac5`, yielding merge
`ed117ab11e68009e4af848c9194d14421d34848d`. No subsequent source-review commit
was incorporated. The user's reference to `9bbb2d2` is a notification, not a
reviewed correction base. Execution stayed on Darwin / `MacBook-Pro-7.lan`.

The new adoption record accepts conservative v2 corrections and the recorded
raw-statement representation only for two within-cache development references.
It pins the closure, manifest, ordered identities, physical receipts, proposal,
and analysis specifications. It does not assert human factual verification,
independent replay of historical extraction, or production-bank adoption.
The historical producer-code conflicts and missing standalone tokenizer/model
configuration bytes remain unresolved. Fresh extraction needs separate
compatibility verification. Historical pending recommendations remain unchanged.

## Frozen preparation and cache verification

`selection_repair_sensitivity_inputs.py` verifies the exact v2 adoption and
closure, equality of ordered v1/v2 admitted identities, and authoritative
sidecar order. New mapping files use `successor_status`; identities include the
original source row/hash, exact statement hash, person key, partition, and actual
tensor index. Method-export offsets are never tensor indices. Each frozen map
has 3,048 train or 1,012 D rows per model; no missing or ambiguous bindings were
accepted. These counts are v2 historical preparation, not forthcoming counts or
P allocation. The reviewed successor must derive its own coverage from its manifest.

Before any scoring, preparation declared 11 endpoints per model: atomic D;
bare AND/OR and both critical boundaries; or-both OR and its critical boundary;
isolated min/max AND/OR and both critical boundaries. It froze exact constituent
eligibility from admitted source facts or bound historical registry evidence,
with restricted source claims taking precedence. Every retained compound pair
must retain all 16 variants and all four constituent facts. Frozen v2 coverage:
1,012/1,040 atomic D rows; 483/524 pairs (7,728/8,384 bare rows);
510/524 isolated facts eligible. These masks must be refreshed, not reused by default.

The one-shot SSH exporter checked host `ef7f7534c328` before artifacts, with
strict host-key verification. All seven allowlisted source tensors matched their
expected streamed SHA-256, byte length, float16 C-order NPY header, stable read,
and live metadata/manifest/completion or progress companions. It exported only
zero-based saved layer 17 (Qwen) or 15 (Llama), binding exact original row indices
and ordered identities to verified full source tensors. No model ran remotely.
A separate local verifier rechecks receipts, source/companion bindings, export
hashes and headers without reading activation values numerically or scoring.

Full frozen mappings, masks, request, and receipts are preserved under:
`/Users/apple/projects/t2-canonical-sensitivity-artifacts-20261001/prepared_v2_frozen`.
The parent directory holds seven selected-layer NPY files and the original tar,
remote receipt, and staging receipt. Large artifacts remain outside Git. The
compact repository receipts bind every frozen preparation file and export; the
versioned preparation is reproducible with the pinned restored metadata and code.

Local independent verification (allowed while fitting is held):

```sh
/tmp/clean-extraction-venv/bin/python -B scripts/49_canonical_sensitivity.py verify-exports \
  --prepared-root /Users/apple/projects/t2-canonical-sensitivity-artifacts-20261001/prepared_v2_frozen \
  --artifact-root /Users/apple/projects/t2-canonical-sensitivity-artifacts-20261001
```

## Implemented sensitivity procedure (not run on research states)

The evaluator calls unchanged `clean_atomic_probes.fit_converged_probe` on raw
float64 features: Qwen saved index 17 / C=1; Llama index 15 / C=10. No standardizing,
new feature, class weight, layer/C selection, A/P allocation, or repair objective
is introduced. This is sklearn's canonical `mean BCE + ||w||²/(2*C*n)` with a free
intercept, not the shared R0 objective. L-BFGS starts at 2,000 iterations, tol=1e-4;
the historical convergence-warning retry is from scratch at 10,000 iterations.
One BLAS thread is used. Settings, library versions, membership, warning/iteration
records, finite checks and the final gradient infinity norm are recorded.
Nonfinite, unconverged, or gradient-flagged fits cannot be scored as valid.

The implementation binds saved historical probe bytes and settings, compares
historical and corrected probes on identical retained examples, and separately
reports historical full-versus-retained coverage changes. Historical atomic
aggregate replay and compound/control row-score replay checks are implemented;
**these replays have not been executed yet**. End-to-end numerical validation on
research states remains pending the successor correction.

AUROC uses the existing tie-aware statistics. Paired uncertainty uses 2,000
seed-1729 PCG64 topic-stratified draws, atomic person multiplicities or compound
entity-endpoint products, shared across probes/models. Pooled, each topic, and
equal-topic macro summaries are declared. At least 1,800 finite replicates are
required for a 95% linear-percentile interval. There is no redraw, imputation,
or dropping unavailable topics. Undefined endpoints and insufficient intervals
make the sensitivity incomplete.

Strict absolute summary movement >0.002, changed ranking/contrast interpretation,
or changed side of chance identifies affected canonical-reference downstream
analyses. Coverage movement is reported separately. No downstream refresh runs
automatically. No trigger has been assessed on real data in this task.

## Validation and resumption

Synthetic tests exercise export row/layer correctness, identity/header/companion
and stability failures, exact constituent evidence and complete-pair masks,
stale/mixed preparation and receipts, the correction hold, canonical raw-gradient
finite differences and sklearn agreement, flagged fits, identical-score paired
zero differences, separate coverage movement, undefined-topic handling, and
refresh thresholds/rank changes. The integrated source-audit, review, overlay,
closure, mapping, shared-objective and cache-checker tests/validators also run.
Exact results are in the companion validation receipt.

Resumption requires an explicitly reviewed correction commit; new versioned
adoption/mappings and masks; quarantine of the two negatives and paired
negations; renewed ordered admission and export-index bindings; and a successor
entry point with focused stale-version checks. Preserve every v2 artifact. Do
not merge latest mac 1 work opportunistically. Physical cache identity passing
does not resolve source membership, producer provenance, or representation
questions outside the bounded acceptance.
