# Final scientific lock, October 4, 2026

**The scientific interpretation and resource caps are approved and finalized. E execution remains unauthorized.** This delivery records the actual user approval of the exact reviewed candidate and stops. It neither adopts accounting nor issues a phase authorization, starts RunPod, loads models, or accesses E activations, predictions or metrics.

## Approved identity and provenance

- Approved candidate delivery: `c2efef5fef91b989d66c3b6228d2ec5e26c2c0f4`.
- Approved candidate: `config/final_e/lock_candidate_20261004.json`.
- Approved candidate SHA-256: `be5b15fe38c6c32a344e321d767d04a6f4c3b6f529ecfb338c2cc6e6edd5d1d3`.
- Independent PASS review and finalization base: `523e756346b373cc6e8be1d788901e5b3291c1d3`.
- Final lock: `config/final_e/final_lock_20261004.json`.
- Final lock SHA-256: `88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e`.
- Reviewed runnable E producer, unchanged: `b2c6dde71e82f5f0af308761bf19ed3341670587`.
- Finalization branch: `checkpoint-final-e-lock-final-20261004`.

The current finalization instruction supplies the actual approval from the project conversation on 2026-10-04. The recorded question was:

> Do you approve this exact candidate, including its interpretation and caps, and authorize lock finalization?

The user replied:

> Yes I approve.

The approver is recorded as **user/project owner**. No personal name, message identifier or signature was supplied or invented. Approval covers recording the exact approved interpretation/caps in a final lock and publishing this delivery. It does not cover E execution, budget adoption or phase authorization. The candidate delivery is the approval record's `reviewed_delivery_sha`; the independent-review SHA and approved candidate hash are separate provenance fields.

The original draft, approved candidate and all published preparation/review evidence remain byte-for-byte unchanged. Candidate, review and finalization commits are delivery identities, never replacements for the runnable producer or original head/extraction producers.

## Administrative finalization only

The final lock is derived from the exact approved candidate. Its `interpretation_contract`, `immutable_contract`, all six pinned files, `limits`, `carried_budget`, `producer_sha` and historical `candidate_preparation` are unchanged. The historical preparation flags remain false: they describe what preparation authorized at that time, not the present scientific approval.

Only these seven top-level administrative fields differ:

| Field | Final record |
|---|---|
| `status` | `FINAL` |
| `scientific_lock_approved` | `true` |
| `claims_finalized` | `true`; freezes interpretation rules, without asserting that E supports any claim |
| `approval` | Actual question/reply, approver, scope, date and exact candidate/review identities |
| `pending` | Empty; scientific decisions and their review are settled |
| `resolutions` | The four proposed decisions now recorded as approved, with their substantive wording unchanged; the combined review/execution item records PASS/user approval and explicitly requires later scoped phase authorization |
| `note` | Distinguishes scientific finalization, historical preparation and still-unauthorized execution/adoption |

`results/final_e_lock_final_20261004/administrative-diff.json` records the full before/after values. Final-lock hashes are in separate evidence; the lock contains no self-referential hash. No executable, gate or frozen protocol is changed.

## Unchanged interpretation and boundary rules

Keep `R_all − S_all` as the original pooled bare OR TF/FT-versus-FF AUROC primary, excluding TT, separately for Qwen and Llama. Use samples 11/23/37 and ordinary metric/difference means inside every draw, never different fitted heads' scores.

Positive gain, gain beyond the .02 practical margin, positive equal-topic macro support and `R_all − S_full` full-training robustness remain separate claims. Macro is not co-primary; the sensitivity cannot rescue a failed primary. Retention uses frozen reduced-training canonical LR, atomic .005 and AND .02 margins, at pooled and macro scopes. Secondary analyses cannot rescue primary failure. All other approved scope, aggregation, bootstrap, unfavorable-outcome and no-tuning rules remain exactly as in the candidate.

The PASS review clarified consumption of the existing contract without changing it:

- Lower bound equal to the negative retention margin supports retention.
- Lower bound below the margin with upper bound exactly on it is unresolved.
- A crossing or undefined interval is unresolved; unsupported retention is not automatically demonstrated loss.
- An interval entirely strictly below the loss boundary supports retention failure.
- An interval touching/crossing .02 leaves practical gain unresolved, even if positivity is supported.

Intervals remain pointwise and conditional, with separate atomic/compound frames and the frozen 2,000 PCG64(1729) draws, fixed five-topic macro and 1,800-finite-draw requirement. No joint or simultaneous confidence claim is introduced.

## Caps, accounting and remaining operational prerequisites

Accepted `limits` remain exactly:

```json
{"model_seconds": 7200, "total_seconds": 18000, "setup_phase_seconds": 1800}
```

The separate CPU-job cap remains **10,800 seconds**, without an unsupported key in `limits`. The 18,000-second cumulative total uses the existing carry-forward definition, including frozen carried model-active charges plus new extraction-worker/supervisor wall. It is a stopping cap, not total project elapsed time or a runtime forecast. Historical charges and setup receipts are not reset or reclassified.

| Model | Frozen carried seconds | Remaining model seconds |
|---|---:|---:|
| Qwen | 160.83801023702836 | 7039.161989762972 |
| Llama | 123.82646256202133 | 7076.173537437979 |

The carried seed is 284.6644727990497 seconds. New evidence rechecks the reviewed local metadata, four unique attempt charges, original/successor adoption and execution-report links, and the metadata census for intervening records. This is bounded local reconciliation; remote state is not certified. No budget adoption is performed. Any new charge or missing/inconsistent settlement blocks later adoption pending review, preserving all existing records and immutable pins.

Separate scoped, hash-pinned phase authorizations remain mandatory. Later work also requires a clean detached exact producer, accepted offline runtime/snapshots, actual token/CUDA checks, real persistent storage allocation and two-copy reservations, settled accounting, complete shard/acceptance checks and durable verified transfer. Actual E runtime readiness remains unmeasured. No launch instruction is supplied here.

## Validation, preservation and publication

New evidence lives only in `results/final_e_lock_final_20261004/`. Neither the candidate validator nor the independent-review validator is rerun over historical receipts. The new finalization check independently verifies the candidate hash, review evidence inventory, six immutable hashes, exact scientific-contract equality, producer/caps/carry-forward equality, the seven-field administrative diff and genuine supplied approval provenance.

Command from the isolated finalization worktree:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.11/bin/python3.11 \
  results/final_e_lock_final_20261004/validate_finalization.py
```

Only small static checks and three existing `ClosedGateTests` are run under the shared CPU lock with one numerical thread. The separate active `evaluate_primary_weight_correlations.py` job was observed using a CPU core without holding that lock; it is preserved and not interrupted. Substantial suites are deferred while it is active. The selected tests exercise no-authorization closure, DRAFT rejection, and hash/scope/producer/unresolved-decision validation using synthetic positive fixtures only.

The unchanged operational `final_e_access.authorize` interface rejects the real final lock when separate authorization/pins are missing, before file/output I/O. Direct gate checks reach the authorization requirement for the final lock using an unapproved in-memory negative fixture, while the original candidate remains rejected as DRAFT. No approved authorization is constructed for the real final lock. There is no gate weakening or execution-module change.

The preservation receipt compares original worktree HEAD/status and the 15 standalone unrelated untracked files, plus original candidate/review evidence and accounting pins. External arrays, ledgers and failed attempts are preserved. No fit, saved-D reproduction, wording diagnosis, model loading or E-output access occurs.

Publication uses a normal non-force push on the isolated finalization branch, followed by independent remote-SHA verification; no merge is performed. Stop at this delivery. Scientific approval is now recorded; accounting adoption, actual runtime checks and separate phase authorizations remain outstanding.
