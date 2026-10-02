# Fresh B25 comparison: frozen validity blocker

**Inputs, pre-fit validation and independent checkpoint audit PASS. The full
B25 scientific comparison is BLOCKED.** Four fresh Qwen raw-LR candidates
failed the inherited independent gradient check. Execution was halted; no
bank restriction, tolerance change, additional retry, selection or refitting
was substituted. There is no estimate of the primary R_all-minus-S_all effect.

The isolated branch starts exactly at
`098a79fcb03114c0809f91dc192c446b3ba392d0`. Implementation and configuration
were committed before the first real fit at producer
`b25581ca9c22d93c8c988c93c0c5491219d6cfec`.
Configuration SHA-256:
`c3c7cc923185847002ff7f741fc8a0cafe7728d5d76451b91f14728c281314cb`.
Origin was fetched without pulling or merging a moving tip; existing worktrees,
historical artifacts, the accepted recoverability run and feature backup were
preserved.

## Accepted inputs and frozen implementation

The fresh full-raw inventory matches
`7d54672272be2cbb30cc20559868c2d39581b219761db09fea76bb40da07f2cb`.
All inventoried files and complete producer verify-output checks passed.
Recoverability producer `3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f` supplied
742 verified artifacts. All 62 potential exact atomic-bank reuses passed
source/exposure, representation, preprocessing and settings checks; their
186 saved P15/atomic-D/bare-D score vectors reconstructed exactly while
streaming all 60 layers. Only three R0 reuses were consumed before the halt.
C_clean is excluded from the atomic bank; historical parameters, statistics
and the 36 previously conditional matches remain ineligible.

The adapter preserves ordered logical bindings and the exact 700-row sampler,
three allocations (seeds 11/23/37), their facts and five entity-disjoint folds.
Each allocation has 25 pairs, 50 persons, 100 affirmative statements
(50 true/50 false) and 400 compound observations. Each fold has 320 adaptation
and 80 held-out compound observations, with held-out persons and facts excluded.
P15 retains all 75 A15-person/source exclusions. Wording bindings were checked
as metadata; wording feature arrays remained unread and unscored.

The committed runner implements the 600-candidate bank and 900 repair folds,
unchanged method preprocessing, reviewed objectives/solvers, shared atomic
eligibility, ties and fallbacks, metric averaging across fold heads, controls,
locked-head wording evaluation, paired bootstrap and precision reporting.
The frozen rules and source hashes are in
[the implementation documentation](../../docs/checkpoint_r2/fresh_b25_v1.md)
and configuration. The 23 focused tests passed before execution in 7.375 s;
they cover leakage, held-out facts, block weights, duplicated observations,
alignment, eligibility, ties/fallbacks, invalid candidates and metric averaging,
including synthetic locked evaluation. Real downstream procedures remain
unvalidated by this blocked run.

## Specific failure and complete accounting

All four heads had finite parameters and library convergence, but exceeded
the frozen gradient infinity-norm maximum of **0.0001**:

| Canonical bank ID | Recomputed gradient infinity norm |
|---|---:|
| qwen/P15/L01/l2_logistic/C=0.1 | 0.000335508513846723 |
| qwen/P15/L01/l2_logistic/C=1 | 0.00015809270829964369 |
| qwen/P15/L02/l2_logistic/C=0.1 | 0.0002259840184617968 |
| qwen/P15/L02/l2_logistic/C=1 | 0.00020199397623376134 |

No convergence warnings occurred. The inherited raw-LR cold retry is triggered
only by an initial convergence warning, so it did not authorize an additional
attempt for these failures. No failed-head predictions were published or used.
There were no retries among the published fresh fits.

| Inventory | Valid fresh | Valid reused | Failed | Reuse ready, unconsumed | Interrupted | Unrun after halt | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Atomic bank | 23 | 3 | 4 | 59 | 0 | 511 | 600 |
| Repair folds | 42 | 0 | 0 | 0 | 1 | 857 | 900 |

The 72 published checkpoints contain 68 valid heads (65 newly fitted and three
reused) and four invalid heads. Qwen L02/seed37/fold2 was interrupted during
optimization and has no published checkpoint. Full canonical identities,
statuses, failure/exclusion reasons and source bindings appear in
[candidate-inventory.csv](candidate-inventory.csv),
[fold-inventory.csv](fold-inventory.csv) and
[published-fit-inventory.json](published-fit-inventory.json).
Eligibility/ranking exclusions and fallbacks were not evaluated; failed fits
are explicitly invalid, and no unrun candidate is treated as excluded.

## Scientific panels and interpretation

S_atom_all, S_all, R_all, full refits and every requested fixed/same-layer,
atomic-exposure, LR-only, R0, C_clean and bank-oracle control are **unrun**.
Therefore per-seed and three-seed mean contrasts, pooled/macro/per-topic
evaluation, paired intervals, atomic/AND retention, actual-B25 precision and
three-wording transfer are **unavailable**, as recorded in
[selection-evaluation-status.json](selection-evaluation-status.json).
No procedure lock or bootstrap output exists. No partial bank or fold metrics
are presented as a completed procedure comparison.

The accepted recoverability results still support complete fresh raw batch-1
extraction. They show C_clean's failed atomic retention, including fixed and
D-selected heads; no saved C_clean layer passes joint atomic/AND retention at
pooled or equal-topic macro scope. That finding remains explicit. Strong
compound AUROC does not establish joint sufficiency. D-selected C_clean and
the bank oracle remain optimistic development references. The .02 sufficiency
and .005/.02 retention margins were not changed; this run adds no sufficiency
or wording-transfer conclusion. Historical compatibility remains unverified.

## Independent audit, runtime and reproducibility

The independent read-only audit verified all 72 checkpoint head/score hashes
and source feature hashes, reconstructed all **408** valid score vectors
exactly, checked **1,904** point AUROCs with sklearn and **42** held-out-fold
OR-boundary metrics, and recomputed the four failed training gradients from
their saved parameters. Failed scores were absent. Audit time was 25.340 s.
There are no principal contrasts to audit because procedures never ran.
See [independent-audit.json](independent-audit.json) and the byte-identical
[audit script](../../scripts/checkpoint_r2_audit_b25_halt.py).

Actual host: **MacBook-Pro-7.lan**, macOS 12.6 arm64. Actual interpreter:
`/opt/homebrew/opt/python@3.11/bin/python3.11` (Python 3.11.6), invoked through
`/private/tmp/clean-extraction-venv/bin/python`. This is independent CPU
review/re-execution on the actual Mac, **not second physical-machine
replication**. One fitting process and one BLAS/OpenMP thread were recorded.
The numerical package versions, code/input hashes and threadpool libraries
are in [execution-summary.json](execution-summary.json) and
[threads.json](threads.json).

Elapsed process time at SIGINT halt: **6 min 47 s**, including startup/input
verification. Published-fit time totals **295.681 s**, excluding interrupted
optimization and other work. Resident memory sampled at halt was
**2,943,500,288 bytes (2.74 GiB)**. Peak RSS is **unavailable** because the halt
preceded the final peak-memory receipt; the sample is not a peak estimate.
The process exited 130. Input/reuse verification and audit ran separately.

Reusable heads, tensors, full row-bound scores and full per-fit metadata stay
outside Git at the paths in [delivery.json](delivery.json). A separate halt
review manifest binds 226 preserved external files and supporting logs:
[halt-review-manifest.json](halt-review-manifest.json). It records a blocked
review, not successful scientific completion. The original run checkpoints
are preserved. Reproducing the audit requires the exact implementation and
external artifacts; no fit restart is needed for audit.

No GPU/SSH/RunPod work, constituent fitting, behavior/chat, interventions,
allocation searches, new budgets, objective tuning or E evaluation ran.
