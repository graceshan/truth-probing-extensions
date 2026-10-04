# Independent locked-E readiness review, October 4

**Verdict: use the targeted successor for final scientific-lock review; E launch remains blocked by the draft lock and unmeasured runtime prerequisites.** The reviewed producer has two deterministic extraction/publication blockers and incomplete accounting enforcement. The successor fixes those operational defects and adds separate principal numerical checks while retaining full report reproducibility. It changes no scientific input, trained parameter, preprocessing rule, selector, margin, metric, bootstrap schedule or interpretation rule.

Repository: https://github.com/graceshan/truth-probing-extensions. Review branch: `checkpoint-final-e-review-20261004`.

- Review base: `ce3600d3e858c56476f9e098d5c582c6c01376af`.
- Original reviewed runnable producer: `880345c2bea9982d91786516219da873ddb4405d` (preserved).
- Full-bank ancestor: `754ff1d7d362cf76ddb2bca681b32b2aeb7118f5`.
- Successor runnable producer: `b2c6dde71e82f5f0af308761bf19ed3341670587`.

All three ancestry edges passed `git merge-base --is-ancestor`. Origin was fetched; no moving branch was pulled/merged, no original worktree was changed, and no history was rewritten. The original E preparation, full-bank results and CPU execution worktrees remain clean at their original SHAs. The original draft lock and all existing configurations/receipts retain their bytes. No approved real-E lock or execution authorization was created. In-memory synthetic authorizations and fake campaigns are confined to temporary synthetic fixtures. No Pod contact, model inference, fitting, E activations, predictions or metrics occurred.

Reviewed `PREPARATION_20261003.md`, `RUNBOOK_20261003.md`, `MAC2_HANDOFF_20261003.md`, all new E modules/tests, registry, draft lock and statistical contracts, and approved Revision 3 Sections 5–7. The approved document's SHA-256 is `ae135d188dddac61d921c0efc8b79499b6344396aae1046f5c9cce1283003035`. This review is technical acceptance of the successor's synthetic/development behavior, not claim approval or E execution permission.

## Findings and targeted fixes

| Severity | Finding in original producer | Evidence and successor resolution |
|---|---|---|
| High — execution blocker | E `worker` passes shard plans containing only token hashes. The inherited `extract_model` requires `reserved_bytes` after writing/reopening a shard. | Fake backend reaches `KeyError: 'reserved_bytes'`; `red-regression.txt` preserves the traceback. New `shard_plan` reuses the inherited exact FP16/JSON/header/receipt/commit allowances. The real pre-forward storage probe includes the complete temporary shard and inherited global/commit reserves. No backend or representation flags change. |
| High — acceptance blocker | Extractor includes `completed_marker` pins in index entries; E loader reconstruction omits them. Every complete extractor-produced index therefore differs. | After fixing the first defect, the same fixture fails `complete ordered index`; `red-index-regression.txt`. Verification now requires and pins each completion marker exactly as publication does. The original isolated-shard test was aligned with the actual publication contract. |
| High — operational integrity | Acceptance checks only `full-budget.running == false`; later accepted verification does not recheck accounting. Reduced/mismatched ledgers, exhausted caps and altered attempt evidence can escape those checks. Transfer publication also checks only the running flag. | Focused mutations cover running, missing, reduced and altered cumulative charges and cap exhaustion. `settled_accounting` checks carry-forward/adoption pins, producer/lock/campaign/limits, settled supervisor snapshots, monotonic charges, recorded worker wall/interval lower bounds, the final ledger equality and successful selected attempt. Acceptance pins that attempt; later loader verification rechecks the chain. Resume checks existing settled evidence; transfer requires both models' complete accepted loader verification. Access receipts now carry the reviewed limits explicitly. |
| Medium — numerical assurance | `audit_report` calls the same evaluator. Complete replay can repeat the same formula error. | Replay is preserved. Separate principal checks reuse the previously independently audited full-bank sorted-pair AUROC implementation and explicit constituent confusion/Boolean/min-max formulas. A deliberately consistent synthetic evaluator error reproduces identically but fails the new reference check. Saved D supplies an additional independent numerical check. This is separate code on the same physical Mac, not second-machine replication. |

No historical failed fit/attempt is removed, and no tolerance, numerical objective, threshold, sign, inclusion rule or cap is relaxed. Prior preparation validation remains historical evidence; it did not exercise the complete extractor-produced handoff and therefore missed the two blockers above.

## Frozen procedures and input identities

Deterministic registry reconstruction resolves exactly **85 valid heads** from committed delivery manifests and their verified external artifacts: 64 original B25/control heads, 9 full-bank heads and 12 final constituent heads. All **156 scalar aliases**, seeds **11/23/37**, and **12 constituent cases** reconstruct exactly. Full per-head provenance is in `head-provenance.json`; manifest pins and ancestry are in `provenance-review.json`.

| Actual parameter producer | Heads |
|---|---:|
| `5e1bf47a457558cbc0735bd8f9f13ca029cbf3a4` | 43 |
| `3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f` | 21 |
| `2d28c0b2733a4433aa6ceb4de020bfb815c2f7db` | 12 |
| `b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8` | 9 |

The CPU successor does not replace those parameter or extraction identities. Saved scalar archives retain their method-specific preprocessing; R0 and repair retain saved standardizers; constituents retain their saved P15 standardizers; full-bank arms retain their full-training exposures. Constituent FIRST/SECOND identities, source-selected Qwen L18/L19 and Llama L12, and source-on-source/source-on-target/target-on-target comparisons are preserved.

Metadata-only validation matches current v5 eligibility: **974 atomic rows**, **225 compound entities**, **450 pairs**, **7,200 bare observations**, **450 isolated facts**, and three existing **3,600-row wording additions**. These yield **19,424 logical bindings / 19,190 unique texts per model**, retaining all 234 deduplication bindings. Each wording reporting condition retains 7,200 rows and uses the other operator's unchanged bare support; no combined AND-plus-OR condition is invented. Every fact/text/logical binding, truth cell and AB/BA order remains pinned. Inventor compound support remains only **11 entities / 22 pairs**.

The E interface scores only registry heads. The D-selected bank/C_clean winners remain descriptive fixed references; it has no E candidate-bank search, fit, layer/sign/threshold selection, calibration or best-seed choice. Original B25 controls remain visible. Full-training arms are sensitivities with advantaged atomic exposure including A identities; repair remains reduced-P and frozen.

## Statistical review

Atomic and compound membership frames remain **294 and 225 people**, respectively; 225 overlap and 69 are atomic-only. Atomic resampling samples people within the atomic frame. Compound rows receive endpoint-multiplicity products within the compound frame; repeated cells/orders share pair weights. Models, methods, samples, wordings and constituent comparisons share draws within their frame. E schedules are reconstructed from actual E metadata using PCG64 seed 1729 and 2,000 draws; D row-index schedules are not substituted.

Each head is evaluated separately. The sample mean is the ordinary mean of the three metrics or paired differences **within each draw**, never concatenated or averaged fitted-head scores. Wording comparisons use the same fixed head/support/draws. The cross-model interaction subtracts paired method differences, not significance labels.

Pooled AUROC uses positive-negative comparison mass and half credit for ties. Equal-topic macro is the ordinary mean of all five topic endpoints. A missing topic endpoint propagates undefined to macro; a draw lacking either class stays undefined. No redraw, imputation or surviving-topic average is allowed. Intervals are pointwise percentile intervals conditional on frozen fits, selectors and support, requiring 1,800 finite draws of 2,000; valid/invalid counts are retained. **There is no simultaneous retention guarantee or joint atomic–compound confidence claim.** The different frames cannot silently be replaced with a union population.

Original margins remain atomic .005, AND .02, practical comparator .02 and constituent absolute/transfer .90/.05. Zero-logit constituent decisions retain `>= 0`, ties True, and original denominators; balanced accuracy is undefined without both classes. FIRST/SECOND targets swap with AB/BA order, while canonical truth identities do not. AND/OR external ranking remains min/max. Scalar B25/full-bank arms remain AUROC-only, without invented thresholds. Positive scaling preserves individual ranking and signs; relative head scales can still affect min/max ranking.

## Validation and resources

Final stage receipts, test logs and numerical evidence are under `results/final_e_review_20261004/`; exact reproduction commands are in `REVIEW_RUNBOOK_20261004.md`. All substantial numerical work ran serially on `MacBook-Pro-7.local`, Darwin/macOS, Python 3.11.6 at `/opt/homebrew/opt/python@3.11/bin/python3.11`, with one numerical thread and the shared exclusive CPU lock.

The original suite passed 24/24 before changes. The frozen successor passes **34/34 focused tests**. Tests exercise the actual E worker/writer/loader orchestration with a fake backend, independent token audit, acceptance, transfer/copy/local verification, scoring/reporting, replay and separate reference checks. They also cover corrupt/missing shards and markers, wrong producer/lock, altered head/preprocessing, changed audits/acceptance, incomplete transfer, reduced/unsettled accounting, caps, failure receipts, fixed-five macro/undefined handling, ties and truth/order rules. A synthetic shortened phase timer proves cap stop/receipt preservation; it changes no production cap. An exhausted synthetic worker budget proves no child process is spawned. Temporary synthetic files are removed after tests.

The serialized full saved-D reproduction passed **53,928 report rows with zero numerical difference**, recomputing all **2,000 draws** for 73 scalar heads and 25 unique constituent score panels and reconstructing the original D schedules. Published coverage is B25 12,936 metric / 18,788 contrast rows, full-bank 5,544 / 6,160, and constituent 9,450 / 1,050.

Feature reapplication is **bounded**, not full-D feature reapplication: the original validator selects the first 16 atomic D and first 16 bare D logical rows per model, using operator-specific subsets for constituents. All 85 heads were applied, with 158 head/group checks totaling 2,432 score applications; maximum difference was 0.0. In contrast, saved-score/report reconstruction uses the complete frozen saved-score supports and recomputes every draw. No head was fitted.

The separate saved-D reference passed **2,979,284 checks with maximum difference 0.0**: 730 principal scalar endpoints across 73 heads, 600 sample means, 3,240 paired contrasts, 608 retention decisions, and 2,250 constituent pooled/macro/order records across all 25 final score panels. It directly reconstructs principal weighted AUROC draws using separate code and independently checks quantiles/counts, ordinary fixed-five means, paired arithmetic and retention. Constituent checks independently derive AB/BA targets, confusion/accuracy, balanced accuracy, joint correctness, Boolean composition and min/max AUROC. Full reproducibility covers every published row; this independent calculation is deliberately scoped to principal endpoints/decisions rather than claiming every report field has a second implementation. Synthetic audit also checks paired wording/model/constituent contrasts and constituent .90/.05 classifications.

| Stage | Measured wall seconds | Peak child RSS bytes |
|---|---:|---:|
| tests | 16.251307 | 452526080 |
| prepare | 10.958660 | 835878912 |
| D-reproduction | 202.446801 | 1396588544 |
| D-independent | 195.466884 | 1799995392 |

macOS RSS units are bytes; these are actual local stage measurements. The independent reference's own internal time was 194.788600 seconds. No E runtime is measured or certified. All final numerical stages completed once from the frozen successor; no fits or scientific retries occurred. Preserved earlier synthetic diagnostics include the two original-producer blockers and a temporary reference-fixture cache-key issue corrected before freezing the successor. The original 24-test suite was run before changes, then focused additions during development, then the final 34-test suite against the committed successor.


## Remaining lock decisions and later runtime gates

Revision 3 requires claim review and a preserved scientific lock before E. This review does not finalize the submission claim, primary endpoints/comparator designation, wording/retention interpretation, optional-work omissions or any joint-frame inference choice. Record separate endpoint inference explicitly unless a separately reviewed statistical successor is chosen; do not imply joint or simultaneous coverage. Preserve pointwise contradictions and undefined results. A changed scientific inventory/statistical procedure requires an explicit decision and reviewed successor.

Later execution must adopt the exact successor producer in a clean detached checkout and reviewed FINAL lock, leaving the immutable registry/config/statistical contract unchanged, plus separately supplied phase-specific authorizations and hash pins. No real finalized lock/authorization is supplied here. Retain GPU cumulative model caps **7,200 seconds/model**, setup caps **1,800 seconds**, and CPU job caps **10,800 seconds**; the total-wall cap still needs explicit review.

Carried charges reconstruct unchanged adoption/attempt evidence: Qwen **160.83801023702836** and Llama **123.82646256202133** seconds, leaving **7039.161989762972 / 7076.173537437979** model seconds. Confirm no intervening charges before lock. A new directory, producer or attempt does not reset accounting. Normal supervised interruption retains the charged ledger and supervisor exit. Explicit resume is allowed only after reviewing the failure and with new scoped instructions, same campaign/producer/lock and intact settled evidence. **Hard supervisor loss, missing settlement evidence or ledger/receipt inconsistency blocks resume pending explicit accounting review**; the successor will not fabricate a settlement or silently reset/retry.

Original extraction forecasts remain estimates: about **865/978 active seconds**, with fourfold planning scenarios **3459/3913 seconds**. Actual E token lengths, CUDA peaks, persistent quota, dependency/model/tokenizer readiness, storage metadata and transfer throughput remain unmeasured. The exact FP16 payload is **8,882,053,120 bytes** across models; two payload copies alone are **17,764,106,240 bytes**, excluding metadata, temporary shards, receipts, reserves, caches, scoring/audit arrays and transfer copies. Free-space inspection is not the required real allocation/fsync/readback test. The successor uses inherited per-shard/global/commit reserves and a complete temporary-shard allowance in that later pre-forward check.

Later verify the accepted offline runtime/snapshots, complete token plans, independent tokenizer-only audit, all 150 shards/model, completion/index/acceptance/accounting receipts and transfer manifests. Complete local checksums and full loader acceptance before scoring. Retain the original runbook's independently captured remote campaign/receipt manifest pins, durable Mac handoff, stopped-worker checks and explicit safe-stop announcement; start no new remote task after that checkpoint. One local CPU job holds the exclusive slot; extraction may overlap only that later authorized local job. No automated retry, replacement runtime/download, tolerance relaxation or procedure redesign follows a technical failure.

Publication is a technical review delivery, not scientific-lock or E execution approval. Keep RunPod stopped.
