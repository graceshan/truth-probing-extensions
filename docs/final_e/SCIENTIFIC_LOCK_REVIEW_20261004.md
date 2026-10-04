# Independent scientific-lock review, October 4, 2026

**PASS — recommend the exact candidate for user scientific approval. No blocking defect or substantive mismatch was found. This review grants neither scientific approval nor E execution permission. E remains closed.**

Reviewer: mac 2, independent implementation/result review on the shared local Mac. Review category: scientific-lock review. Environment: `MacBook-Pro-7.local`, macOS 12.6 arm64, `/opt/homebrew/opt/python@3.11/bin/python3.11` (3.11.6). No RunPod contact.

## Exact reviewed identity and scope

- Candidate delivery: `c2efef5fef91b989d66c3b6228d2ec5e26c2c0f4`.
- Candidate file: `config/final_e/lock_candidate_20261004.json`.
- Independently calculated candidate SHA-256: `be5b15fe38c6c32a344e321d767d04a6f4c3b6f529ecfb338c2cc6e6edd5d1d3`.
- Original reviewed delivery: `e8f087e53b96b2172cfcde03e3a451e6a65bb0fe`.
- Proposed execution producer: `b2c6dde71e82f5f0af308761bf19ed3341670587`. The candidate/review delivery SHAs are not runnable producers and do not replace frozen head or extraction provenance.
- Review branch: `checkpoint-final-e-lock-review-20261004`, in a separate worktree under `/Users/apple/truth-probing-backups/20261001T164224Z/`.

Fetched the exact candidate commit, created a new review branch at it, and independently verified producer → original delivery → candidate ancestry. The complete committed diff contains exactly seven additions: candidate JSON, interpretation document, accounting, focused-test log, preservation, validator and validation receipt. There are no modifications/deletions/renames of existing execution modules, scientific inputs or historical receipts. All six immutable files and the original draft match their original bytes and pins. The governing Revision 3 DOCX hash also matches `ae135d188dddac61d921c0efc8b79499b6344396aae1046f5c9cce1283003035`.

Read the candidate and every accompanying file, the original mac 2 handoff, readiness review and successor review runbook, and the relevant frozen reporting/access/accounting functions. No applicable `AGENTS.md` or separate session-continuation instruction file was found in the inspected repository/ancestors; the supplied handoff and repository review documents governed restoration. The historical handoff's original producer is superseded by the explicit successor in the later readiness review.

## Findings

| Severity | Finding and disposition | Supporting locations |
|---|---|---|
| Blocking | None. The exact candidate is recommended for approval, with the later decisions below still required. | Candidate JSON; independent identity/accounting/boundary receipt. |
| Informational | The evaluator computes endpoints, paired intervals and `interval_supported`; it does not implement a complete scientific claim classifier. A false support boolean must not be reported as supported loss. The candidate correctly supplies the distinction between loss and unresolved retention, plus separate gain/robustness/macro claims. | `src/final_e_reporting.py:59–105`, particularly 93–96; candidate `interpretation_contract.retention`, `.gain`, `.interpretation`. |
| Informational | Retention prose uses “crosses” for uncertainty. With the explicit inclusive support and strict loss rules, an upper bound exactly on the loss boundary with a lower bound below it is unresolved. This reading is recorded explicitly here; it requires no criterion or candidate change. Undefined intervals are also unresolved. | Candidate JSON 130–148; interpretation document's retention rows; boundary cases in `independent-checks.json`. |
| Informational | The gate requires approval fields and scoped authorization, but field presence alone cannot establish authentic user approval provenance. The actual candidate honestly carries none. Later records must cite actual user approval/instructions; this review or a recommendation to proceed cannot manufacture them. | `src/final_e_access.py:24–54`; candidate JSON 2, 14, 197–210. |

No candidate or execution-module correction is proposed. These informational findings describe how to consume the unchanged reports and gates, rather than introducing implementation changes or expanding the audit.

## Scientific contract versus frozen computation

| Agreed rule | Review result / frozen computation |
|---|---|
| Original pooled primary | Candidate fixes bare `R_all − S_all`, separately Qwen/Llama, OR TF/FT versus FF, TT excluded. `checkpoint_r2_b25_statistics.masks` uses OR and truth-sum `<2`; inventory labels are truth-functional OR. A synthetic eight-row mask selected exactly FF/TF/FT OR rows and excluded TT/AND. |
| Samples | Seeds 11/23/37 remain in the registry. `final_e_reporting.scalar_tables` uses `average_sample_metrics` on already computed points/draws, then paired differences. Ordinary means preserve undefined samples; no fitted-head scores are averaged or concatenated. |
| Gain and topic weighting | Positive requires lower bound `>0`; practical gain requires `>.02`. A positive interval can leave practical gain unresolved or lie entirely below .02. Pooled remains primary; positive fixed-five macro supports only the additional equal-topic claim. Macro is not co-primary and need not exceed .02; every topic/sample need not be significant. The evaluator provides intervals, while the candidate governs these claims. |
| Full-training sensitivity | `R_all − S_full` remains a separate robustness contrast. It cannot rescue an unsuccessful original primary. All frozen contrasts, including `S_full − S_atom_full` and `R_all − S_atom_full`, remain byte-identical in `contrasts_v1.json`. Advantaged exposure, preprocessing and sampling change together; `S_full` uses compound labels; full-training repair was not tested. |
| Reference and retention | Registry aliases preserve the frozen reduced-training canonical LR, not full canonical LR or D maxima. `R_all − reduced_LR` is evaluated on E with atomic margin .005 and AND margin .02. Support requires inclusive lower bounds at both pooled and macro scopes, giving four required retention checks for the basic claim. Point estimates alone do not establish retention. |
| Bootstrap | Frozen 2,000 PCG64(1729) draws, separate 294-person atomic / 225-person compound frames, shared within each frame across models/methods/samples/wordings/comparisons. Atomic weights use entity counts; compound weights use endpoint-count products, with repeated cells/orders sharing pair weights. `final_e_reporting.schedules`, inherited `atomic_weights` / `compound_weights`, and `EntityBootstrap` agree. |
| Undefined and macro | `summaries` takes the ordinary mean across all five fixed topics; it does not drop undefined topics. `interval` uses linear percentile quantiles only with at least 1,800 finite draws, retaining invalid counts. No redraw, imputation or union-frame substitution. |
| Inference and model claims | Intervals are pointwise, conditional on frozen fits/selectors/support. Positive original-primary plus four retention checks gives the basic model claim. Practical magnitude, equal-topic gain and full-training robustness each need separate checks. Both models permit replication; one permits model-specific support. Different significance labels do not establish interaction. Passing a conjunction does not create simultaneous/joint 95% coverage. |
| Unfavorable outcomes and scope | Unsuccessful, undefined, contradictory and inconclusive outcomes remain reportable; secondary endpoints cannot rescue primary failure. AUROC retention does not establish calibration/decisions/behavior. Held-out entities are unseen by readout fitting/selection, not necessarily by pretraining. Behavior, chat, calibration, causal work and loss-deletion fits remain deferred; the atomic-loss-causes-retention claim is omitted. No E tuning, comparator search or optional expansion is authorized. |

Key implementation locations: `src/checkpoint_r2_b25_statistics.py:19–48`; `src/final_e_inventory.py:23–39`; `src/final_e_reporting.py:21–105`; `src/checkpoint_r2_selection_v1.py:192–197`; `src/selection_repair_canonical_sensitivity.py:46–73`; `src/clean_transfer_statistics.py:52–83`. `final_e_runner.evaluate` produces metrics/contrasts, not automatic final claim approval.

Boundary interpretation checked explicitly:

| Interval situation | Result |
|---|---|
| Retention lower bound equals `−margin` | Supported, including a degenerate interval on the boundary. |
| Lower bound below `−margin`, upper bound equals it | Unresolved; strict supported-loss criterion is not met. |
| Interval crosses `−margin` | Unresolved. |
| Entire interval strictly below `−margin` | Supported loss. A retained point alone cannot override the interval. |
| Undefined / fewer than 1,800 finite draws | Unresolved; `interval_supported=false` does not establish loss. |
| Gain lower bound equals 0 | Positive gain not supported. |
| Gain lower bound equals .02, or interval touches/crosses .02 | Gain beyond .02 unresolved; positivity remains a separate question. |
| Gain upper bound strictly below .02 | Does not reach the practical margin; may still support positive gain. |

## Limits and local accounting

Accepted limit keys/values are exactly `model_seconds=7200`, `total_seconds=18000`, `setup_phase_seconds=1800`; separate CPU-job cap is 10,800 seconds. `final_e_access.validate_gate` accepts these keys, and `final_e_runner.run_authorized` retains the CPU/setup operation caps. Equality at each stopping cap triggers the inherited `cap_reason`; the focused checks call that pure function without spawning a worker.

`final_e_raw.launch` seeds the cumulative total with the sum of frozen carried model-active charges. The inherited supervisor adds new extraction-worker/supervisor wall time, including setup within that worker envelope; model counters add model-active intervals. The total is not project elapsed time, transfer/CPU elapsed time, or an extraction forecast. The 18,000-second limit is a stopping cap; no budget was adopted.

Independently rehashed all **32 cited local accounting metadata files**, the immutable source-receipt links, local historical adoption and execution report, and the four unique settled attempt start/end pairs. Deduplicated by attempt identity and verified duplicate end/start bytes; copied preflight/missing-only/Llama handoffs are not new executions. Failed preflight charge is retained, not discarded. Re-scanning the cited project `.worktrees` and backup roots found the same 32 campaign/attempt metadata files and no additional relevant accounting metadata or E adoption/campaign.

| Model | Independently summed unique attempt charges | Carried total | Remaining from 7,200 |
|---|---|---:|---:|
| Qwen | 100.81950526899891 + 60.01850496802945 | 160.83801023702836 | 7039.161989762972 |
| Llama | 41.1570940479869 + 82.66936851403443 | 123.82646256202133 | 7076.173537437979 |

Combined carried seed: **284.6644727990497 seconds**; remaining cumulative total allowance before any new E work: **17715.33552720095 seconds**. Historical adoption/execution final budgets agree exactly. The older full-raw campaign ledger and full-bank CPU attempt remain distinct historical campaigns and match their recorded bytes; neither is reset or silently reclassified. `carryforward_v1.json` is unchanged.

This is a bounded local metadata reconciliation. It does not inspect or certify current remote state, network-mounted receipts, new receipts after the scan, current GPU/runtime/storage readiness, or completion of future E work. Actual current charges must be rechecked before later adoption. Any new charge or missing/inconsistent settlement needs review while preserving all prior records and immutable pins.

## Access closure, checks and preservation

Submitted the unchanged real candidate to `validate_gate` for all seven declared phases and Qwen/Llama/both model values: **21 rejections**, each `E CLOSED: finalized scientific lock required`. No FINAL variant, real approval or phase-authorization record was created. Later gating still requires FINAL status, both flags true, exact immutable contract, all decisions resolved, actual approval reviewer/instruction/reviewed-delivery evidence, clean detached exact producer, lock/authorization hashes, and separate scoped phase/model/output/campaign authorization. Gate field validation does not replace checking actual user provenance.

New evidence is in `results/final_e_lock_review_20261004/`:

- `independent_checks.py`: independent identity, local accounting and tiny synthetic boundary checks; fresh external-output argument; no reuse of the candidate validator.
- `independent-checks.json`: command/runtime, seven-addition identity, six hashes, receipt links, scan census, gain/retention/draw/cap boundaries and 21 real-candidate gate rejections. Measured check time: 1.47 seconds, one numerical thread, exclusive CPU lock held for tiny synthetic checks.
- `initial-state.json` and `preservation.json`: existing worktree HEAD/status and standalone-file pins before/after, plus committed-candidate/published-receipt preservation.

Command executed from the isolated review worktree:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -B \
  results/final_e_lock_review_20261004/independent_checks.py \
  /Users/apple/truth-probing-backups/20261001T164224Z/final-e-lock-review-evidence-20261004/checks
```

The supplied validator was read, never run. Its accounting/log/validation outputs are untouched. Its published 34-test pass is hash-linked evidence, not a suite rerun by this reviewer. The validation receipt records the base as HEAD during pre-commit preparation; this independent review verifies the final candidate commit and exact bytes. Tiny focused checks resolved the boundary/status questions without completed fits, full saved-D reproduction or wording diagnosis.

A separate local `evaluate_primary_weight_correlations.py` job was active near one CPU core during inspection. It had no observed handle on the shared lock. It was left untouched; no numerical suite or substantial CPU job was launched by mac 2. No model loading, RunPod contact, E activations/predictions/metrics access, array mutation, candidate editing, lock finalization, accounting adoption, approval fabrication, reset/stash/clean/pull/merge or force push occurred. Existing worktree HEADs/statuses, 13 unrelated files plus two DOCX files, accounting metadata and all seven committed candidate additions are preserved. External arrays were neither opened nor written by these checks.

## Remaining decisions

The user must decide whether to approve this exact candidate and its interpretation/caps. A later finalized lock must preserve this reviewed content and immutable contract while recording genuine exact-lock approval, finalized flags, completed review resolution and hash-bound identity. Any further substantive change needs review. Current accounting adoption/runtime prerequisites and separate phase authorizations remain outstanding. This technical recommendation is not any of those records.

Publish only this review material on the isolated review branch, verify its remote SHA, and stop. No merge or E action follows delivery.
