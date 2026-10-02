# Revision 2 common integration v1

Integration and validation only, performed locally on Darwin, MacBook-Pro-7.lan,
in `/Users/apple/projects/checkpoint-r2-common-20261002`. No new extraction,
fitting, research selection, behavior evaluation, or E scoring was performed.
No SSH, installations, or tensor/score-file reads were needed.

## Exact history and successor scope

The persistent `checkpoint-r2-common-20261002` worktree starts at mac 1
`7dd63105c75b894b602fb93754e53eede4483a63`. Normal merge
`70f5ef8b159c077fb1339390d3bf9c2d0d1281ad` has that commit as first parent
and mac 2 `a8082bd7bb6a321690e615b078c9651a6c6a2d49` as second parent.
Both exact commits are ancestors. Origin was fetched; no pull, moving-tip
integration, rebase, or changes to another worktree were made.

`config/checkpoint_r2/common_integration_v1.json` is the successor contract.
Its SHA-256 is
`895ced573ad47af73e3bceaf5dd80a0b39693ca82367d0f5264b85de71755a57`.
Commit `2ae3ecbc5d504c62a14a8db50ea0864e6fdb1df6` froze it before the
synthetic selector validation. It pins source trees, original configurations,
implementations, frozen metadata, bridge receipts and preparation inventories.
The final branch HEAD, after committing this report and adapter, is the common
handoff SHA; the receipt's `validation_parent_head` records the earlier contract
commit, not a claim that the final receipt can contain its own commit hash.

Mac 1's adopted checkpoint memberships supersede the earlier preparation's
unfrozen/pending-common-base status. Original statuses remain byte-preserved
snapshots. This integration does not enable production or validate historical
representation execution. The 36 historical P15 matches retain v4 provenance
and conditional status; no historical fit has been newly admitted.

## Membership and preparation bindings

| Frozen membership | Verified count |
| --- | ---: |
| Corrected outer train | 3,040 |
| P15 | 2,778 |
| Balanced P15 exposure | 700 |
| A15 entities | 75 |
| Atomic D | 1,012 |
| Compound D pairs | 483 |
| T_C entities | 100 |

The validator rebuilds mac 1's metadata package from its pinned inputs and
compares every output byte with the frozen package. It verifies exact ordered
v4/v5 admission and sampler identities, A15 and T_C identities, all source
variants excluded from P15, fact evidence, source-validation split, pair groups
and D bindings. Counts alone are insufficient. The successor receipt carries
all six ordered equivalence hashes and every adapted block's metadata hash.

Preserved planning inventory contains 600 unique candidate IDs (Qwen 280,
Llama 320), 900 repair fold IDs before refits/controls, and 36 unique conditional
P15 reuse entries. Only their metadata was checked here. External fit parameters
and tensors were not reread or readmitted. Execution prerequisites still apply.

## Prospective selector clarification

S_atom_all uses the same atomic retention eligibility receipt as S_all:
valid fits, five defined topic atomic endpoints, and pooled atomic D at least
the reduced LR reference minus .005. `shared_atomic_bank` emits the canonical
sorted eligibility records and binding hash. `select_atomic_synthetic` rejects
a stale bank or an S_all eligible-ID set differing by even one candidate.
Future S_all execution must consume this exact shared bank. Its compound
ranking is not implemented or executed by this integration.

Within the eligible bank, S_atom_all maximizes pooled atomic AUROC, then
within 1e-12 of that maximum maximizes equal-topic atomic AUROC, then within
1e-12 of that maximum permutes lexically sorted canonical IDs using a **fresh
PCG64(20261005) per model**. The first permuted ID wins. This prospectively
resolves mac 1's `choice` wording in favor of mac 2's implemented permutation
convention: the two RNG operations need not produce the same winner. No
research scores informed this decision. Original configuration bytes remain
unchanged. S_all/R_all retain their separately specified seed/choice rules;
original atomic R0 remains separate with its exact-tie lower-layer rule.

## Frozen compound adapter

`src/checkpoint_r2_common_v1.py:FrozenBindings` loads only hash-bound metadata.
CSV truth values accept literal `0` or `1`; native Boolean/integer binary values
are also accepted by the parser. Strings such as `False`/`True`, floats, blanks
and missing values are rejected. No string-truthiness conversion is used.

The adapter checks fact eligibility, statement hashes, person/topic/split,
canonical truth, operator truth, pair/group membership, example identity and
exact rendered text. It maps `truth_a`/`truth_b` to
`canonical_truth_a`/`canonical_truth_b`. AB/BA surface labels, fact IDs and
person IDs are explicitly carried in surface order. D retains the original
authoritative IDs; it is never reconstructed from text or training-ID recipes.

All 25,408 occurrences remain distinct by `(group, example_id)`. There are
5,184 repeated example occurrences across groups; none are deduplicated away.
Every group/pair has all 16 operator/cell/order rows. Each source-validation
operator block has 240 rows (30 pairs), with its original CSV ordering.

`block(group, operator)` returns the ordered metadata, keys, role and SHA-256.
A score packet must supply that exact metadata hash, exact ordered composite
keys, `score_kind='synthetic_validation'`, and a finite N-by-2 array. `align`
rejects reordering, stale hashes, altered blocks and cross-group bindings;
it never reorders scores to make them fit. Such a packet is an explicit
caller alignment declaration, not proof of future score-generation provenance.
A future production runner must bind score artifacts and representation too.

Only `constituent_source_validation` with role `T_C_source_validation` and
the specified source operator can enter `select_synthetic`. Source-fit,
full-T_C-refit, target-operator, B25 and D uses cannot enter this route.
Source-fit and full-T_C blocks have separate role and metadata hashes even
where examples overlap. Source-validation persons are verified against the
frozen split, and the upstream rebuild checks fitting/validation separation.
The existing six-endpoint selector and objective code remain unchanged.

## Validation and preservation

The three focused suites passed: **69 tests in 6.40 seconds**, including 40 new
integration cases. Tests use actual frozen metadata and synthetic scores only.
They cover strict bits, AB/BA identities, the conditional-AUROC trap, deterministic
model-independent permutation, shared eligibility, missing/invalid endpoints,
wrong roles, ordered score/hash failures, same-count sampler reordering,
immutable metadata, exact memberships and bridge failure/unverifiable rules.
No historical experiment was rerun. `git diff --check` passed.

Preservation compared working-tree bytes against both exact source Git trees:
1,128 mac 1 paths and 1,002 mac 2 paths, **1,149 union paths**, all unchanged.
The explicit local preservation check also matched the supplied DOCX and all
13 unrelated files against mac 1's saved hashes. Normal tests do not require
those local files; their preservation behavior uses temporary synthetic files.
Only new successor paths were added.

Results live in `results/checkpoint_r2_common_v1_20261002/`:
`integration_receipt.json`, `test_results.json`, and `delivery_manifest.json`.
The integration receipt binds both sources, contract/code hashes, membership
identities, adapter blocks, inventory and optional local preservation check.
The delivery manifest binds the new files without self-reference.

## Remaining bridge and execution prerequisites

Fresh/historical representation compatibility is **unverified**. Production
execution and E scoring remain **disabled**. No tolerance was relaxed. Equal
tensor hashes do not establish historical producer execution or standalone
tokenizer/config provenance, and unknown history cannot become a bridge pass.

A reviewed, bounded extraction runner restricted to the frozen IDs is still
needed; historical production entry points must not substitute for it. Resolve
pinned model/tokenizer availability (including Llama inventory), GPU/runtime
compatibility, storage quota and durable outputs, then obtain actual bounded
GPU timing. The inherited remote observation reported no GPU, Torch 2.8 versus
recorded 2.11, and missing Transformers/tokenizers; this task made no fresh
remote observation and does not assert that state is current.

Bridge review still needs exact tokens, readout positions, masks, all-layer
shapes/states and frozen-head scores, plus repeatability and padding/batch
invariance and the specified calibration checks. Preserve zero hidden-state
absolute/relative and frozen-score tolerances and the two-hour active / one
GPU-hour-per-model caps. Stop for early review where evidence is unavailable;
do not infer provenance from agreement or automatically regenerate caches.
Separate authorization is required before production execution or E scoring.

## Reproduce

From the common worktree, using the existing numerical environment:

```sh
/tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_checkpoint_r2_common_v1.py tests/test_checkpoint_r2_inputs.py tests/test_checkpoint_r2_selection_v1.py
/tmp/clean-extraction-venv/bin/python -B scripts/68_validate_checkpoint_r2_common.py
git diff --check
```

The validator is read-only by default. For an explicit local preservation check,
add `--original-worktree /Users/apple/projects/truth-probing-extensions`.
To save a new receipt, add `--receipt <new-path>`; it refuses to overwrite an
existing file. Do not rerun the historical preparation timing/output writers.
