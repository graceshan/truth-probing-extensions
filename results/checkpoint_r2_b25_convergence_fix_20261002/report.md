# Fresh B25 successor: complete comparison

**Numerical correction, complete execution and independent audit PASS. No remaining blocker.** Repair improves pooled bare-D OR mixed-versus-FF over selection by **0.1223 for Qwen** and **0.1532 for Llama**, averaged over seeds 11/23/37. The paired intervals exclude zero and the .02 margin. These are frozen development-D results; no E evaluation or joint C_clean sufficiency claim follows.

## Lineage, inputs and numerical correction

The isolated successor starts exactly at `60e651370e5c08635f151822073a8b9c01e092ca`. The amendment, implementation, configuration, diagnosis and 38 focused tests were committed before real continuation at **`5e1bf47a457558cbc0735bd8f9f13ca029cbf3a4`**. Configuration SHA-256: `88db93cd63d3d2f24229d76506d0a0128931f22a3bf5d08960247a24a088c248`. Origin was fetched; no pull or merge of moving tips occurred.

Original fitting producer: `b25581ca9c22d93c8c988c93c0c5491219d6cfec`; its [blocked delivery](../checkpoint_r2_fresh_b25_20261002/delivery.json) and [report](../checkpoint_r2_fresh_b25_20261002/report.md) remain unchanged. All 226 halt-review artifact hashes were rechecked before and after each successor stage. The read-only full feature inventory remains `7d54672272be2cbb30cc20559868c2d39581b219761db09fea76bb40da07f2cb`. Complete producer verify-output and all inventory checks passed again. Recoverability producer `3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f` supplied verified fresh heads/statistics; historical fits/statistics and the 36 conditional matches remain ineligible.

All four original failed raw-LR gradients and objectives reproduced exactly using P15 only. Installed pinned sklearn/SciPy sources show separate library relative-loss and gradient stopping tests; original termination messages were not retained, so their specific original stopping reasons remain unknown. The successor preserves each first attempt and raw objective **mean BCE + ||w||²/(2*C*n)** with a free intercept. TTPD subfits preserve their unregularized objectives, original features and composition. One float64 warm continuation uses analytic gradients, gtol=1e-4, ftol=0, maxls=50, maxcor=10, a hard 50,000 function-call cap, and <=10,000 cumulative iterations per subfit. R0/repair/C_clean/closed-form numerics and every scientific rule remain unchanged. No D/A-validation/wording performance guided the settings. See [the authorized amendment](../../docs/checkpoint_r2/b25_solver_amendment_v1.md).

| Failed original candidate | Original gradient | Continued gradient | Iterations: original + warm | Objective change |
|---|---:|---:|---:|---:|
| qwen/P15/L01/l2_logistic/C=0.1 | 0.000335508514 | 3.11428065e-05 | 118 + 1 | -3.01e-09 |
| qwen/P15/L01/l2_logistic/C=1 | 0.000158092708 | 2.07203601e-05 | 537 + 1 | -7.38e-10 |
| qwen/P15/L02/l2_logistic/C=0.1 | 0.000225984018 | 2.09859575e-05 | 430 + 1 | -1.1e-09 |
| qwen/P15/L02/l2_logistic/C=1 | 0.000201993976 | 1.78079392e-05 | 1329 + 1 | -8.89e-10 |

All four warm continuations explicitly terminated with `CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL`, passed finite/loss/gradient checks and the unchanged 1e-4 limit, and were independently reconstructed before production. Original invalid heads remain preserved as failed attempts. Full before/after losses, iterations, messages and parameter hashes are in [continuation-gate.json](continuation-gate.json) and [gate-independent-audit.json](gate-independent-audit.json).

## Complete fit and selection accounting

| Family | Published valid records |
|---|---:|
| C_clean | 10 |
| atomic_only | 12 |
| bank | 600 |
| repair_fold | 900 |
| repair_refit | 19 |

All **600 bank identities** and **900 fold identities** completed; C_clean contributed zero bank candidates. **68 original valid checkpoints were imported without refitting**, retaining original parameter-producer SHAs and byte-identical parameters/scores. Four original failures became new continued artifacts; the interrupted original fold was rerun. The bank contains 59 further exact recoverability reuses and 511 genuinely new fits. The fold inventory contains 42 imports, one interrupted rerun and 857 genuinely new fits. There are 19 deduplicated full repair refits, 12 atomic-only controls and ten exact C_clean reuses. No fit failure remains. There were 28 raw-LR warm continuations (the four saved failures plus 24 new first-attempt failures), 0 TTPD polarity continuations and 3 TTPD truth continuations. Full inventories, optimizer attempts, source/exposure hashes and explicit aliases are linked below.

P15 excludes all 75 A15 persons/source variants. Allocations remain 25 pairs/50 persons/100 isolated facts/400 compound rows; folds are five entity-disjoint 320-training/80-held-out partitions with held-out facts excluded. Logical observations and exposure weights are preserved. Repair ranks mean five fold AUROCs; cross-head scores are never pooled. Shared atomic eligibility has no AND gate. Raw-LR/TTPD continuations apply uniformly across models/layers/Cs.

| Model / seed | S_all canonical ID | R_all refit layer | Final-refit retention |
|---|---|---:|---|
| qwen / 11 | qwen/P15/L15/l2_logistic/C=1 | 20 | PASS |
| qwen / 23 | qwen/P15/L18/l2_logistic/C=10 | 22 | PASS |
| qwen / 37 | qwen/P15/L18/l2_logistic/C=10 | 18 | PASS |
| llama / 11 | llama/P15/L09/ttpd | 14 | PASS |
| llama / 23 | llama/P15/L09/ttpd | 13 | PASS |
| llama / 37 | llama/P15/L09/ttpd | 13 | PASS |

The shared atomic-eligible bank contains 144/280 Qwen and 182/320 Llama candidates; the other 274 valid candidates fail atomic retention and remain in the complete inventory. No primary S_all/R_all fallback was invoked. All selector eligibility exclusions, ranking/tie stages, fallback checks, original atomic-selected R0, S_atom_all, fixed/LR-only/equal-boundary controls and optimistic bank-oracle/C_clean references are preserved in [locked.json](locked.json). D-selected C_clean and the D bank oracle are optimistic development references.

## Primary contrast and actual precision

Pooled bare-D OR mixed-versus-FF AUROC difference, R_all minus S_all; 95% paired percentile intervals. Within every shared person/entity draw, the three seed **metrics** are averaged, rather than head scores. Bootstrap seed 1729, 2,000 draws, frozen undefined-endpoint rules and >=1,800 valid-draw requirement are unchanged. Every primary interval has 2,000 valid draws. Intervals are pointwise and conditional on the fixed locked heads; bootstrap draws do not rerun selection.

| Model | Seed | Paired difference [95% interval] |
|---|---|---|
| qwen | 11 | 0.1325 [0.1093, 0.1536] |
| qwen | 23 | 0.1107 [0.0881, 0.1319] |
| qwen | 37 | 0.1238 [0.1016, 0.1436] |
| qwen | mean_11_23_37 | 0.1223 [0.1038, 0.1388] |
| llama | 11 | 0.1493 [0.1234, 0.1730] |
| llama | 23 | 0.1536 [0.1269, 0.1788] |
| llama | 37 | 0.1568 [0.1292, 0.1824] |
| llama | mean_11_23_37 | 0.1532 [0.1266, 0.1780] |

qwen: actual primary interval half-width **0.0175**; maximum distance from the point **0.0186**. The .02 margin remains unchanged.

llama: actual primary interval half-width **0.0257**; maximum distance from the point **0.0267**. The .02 margin remains unchanged.

Qwen achieves a half-width below .02; Llama does not. Both repair-versus-selection intervals are far above .02, so the exploratory selection-sufficiency repair-gap criterion fails for both models. The control gap also fails; there is no selection-sufficiency conclusion. Full actual-D and supported per-topic count-proxy precision is in [precision.csv](precision.csv); pooled/macro/inventors compound transport remains unsupported, with no E predictions or exact power claim.

## Atomic and compound performance and retention

Three-seed mean AUROC; pooled and equal-topic macro. Complete per-seed, per-topic, fixed-layer and selected-control intervals are in [metrics.csv](metrics.csv).

| Model | Scope | Arm | Atomic | AND | OR | OR mixed/FF |
|---|---|---|---:|---:|---:|---:|
| qwen | pooled | reduced_LR | 0.9998 | 0.9419 | 0.8478 | 0.7783 |
| qwen | pooled | S_all | 0.9997 | 0.9709 | 0.8991 | 0.8503 |
| qwen | pooled | R_all | 0.9996 | 0.9934 | 0.9813 | 0.9726 |
| qwen | topic_macro | reduced_LR | 1.0000 | 0.9144 | 0.8457 | 0.7788 |
| qwen | topic_macro | S_all | 1.0000 | 0.9665 | 0.8840 | 0.8290 |
| qwen | topic_macro | R_all | 0.9999 | 0.9893 | 0.9702 | 0.9565 |
| llama | pooled | reduced_LR | 1.0000 | 0.9756 | 0.7606 | 0.6546 |
| llama | pooled | S_all | 0.9990 | 0.9676 | 0.8897 | 0.8373 |
| llama | pooled | R_all | 1.0000 | 0.9973 | 0.9937 | 0.9905 |
| llama | topic_macro | reduced_LR | 1.0000 | 0.9893 | 0.8114 | 0.7229 |
| llama | topic_macro | S_all | 0.9997 | 0.9680 | 0.8755 | 0.8146 |
| llama | topic_macro | R_all | 1.0000 | 0.9951 | 0.9911 | 0.9866 |

Retention against fresh reduced LR keeps .005 atomic and .02 AND margins. The table gives point/interval-supported results.

| Model | Arm | Scope | Atomic retention | AND retention |
|---|---|---|---|---|
| qwen | S_all | pooled | PASS/PASS | PASS/PASS |
| qwen | S_all | topic_macro | PASS/PASS | PASS/PASS |
| qwen | R_all | pooled | PASS/PASS | PASS/PASS |
| qwen | R_all | topic_macro | PASS/PASS | PASS/PASS |
| qwen | C_clean_D_selected | pooled | FAIL/FAIL | PASS/PASS |
| qwen | C_clean_D_selected | topic_macro | FAIL/FAIL | PASS/PASS |
| llama | S_all | pooled | PASS/PASS | PASS/PASS |
| llama | S_all | topic_macro | PASS/PASS | FAIL/FAIL |
| llama | R_all | pooled | PASS/PASS | PASS/PASS |
| llama | R_all | topic_macro | PASS/PASS | PASS/PASS |
| llama | C_clean_D_selected | pooled | FAIL/FAIL | PASS/PASS |
| llama | C_clean_D_selected | topic_macro | FAIL/FAIL | PASS/PASS |

**R_all retains atomic and AND performance at pooled and macro scope for both models. Llama S_all fails macro AND retention. C_clean still fails atomic retention**; its strong compound AUROC does not establish joint sufficiency. Fixed and procedure-layer C_clean retention, every topic and every seed remain explicit in [contrasts.csv](contrasts.csv).

## Same-layer and atomic-exposure controls

Three-seed mean pooled bare-D OR mixed/FF paired differences:

| Model | Contrast | Difference [95% interval] |
|---|---|---|
| qwen | R_all − R0_at_R_all | 0.2724 [0.2503, 0.2906] |
| qwen | R_all − atomic_only_at_R_all | 0.2821 [0.2577, 0.3027] |
| qwen | R_fixed − S_fixed | 0.1655 [0.1476, 0.1822] |
| qwen | R_fixed − atomic_only_fixed | 0.1740 [0.1539, 0.1927] |
| llama | R_all − R0_at_R_all | 0.3521 [0.3240, 0.3825] |
| llama | R_all − atomic_only_at_R_all | 0.3520 [0.3237, 0.3834] |
| llama | R_fixed − S_fixed | 0.3407 [0.3127, 0.3739] |
| llama | R_fixed − atomic_only_fixed | 0.3659 [0.3385, 0.3990] |

Repair gains persist against same-layer R0 and matched isolated-fact adaptation. Required fixed, LR-only, repair-at-S_all, S_atom_all, atomic-selected R0, oracle and C_clean controls are all included, with deduplication only for identical fit exposure/settings.

## Held-out wording transfer

Heads were locked before wording lookup/scoring. No wording influenced selection, fitting or calibration. Three-seed mean pooled R_all-minus-S_all differences on the same held-out D pair/fact/order identities:

| Model | Wording | Endpoint | Difference [95% interval] |
|---|---|---|---|
| qwen | and_both_following_v1 | AND_auroc | 0.0414 [0.0364, 0.0464] |
| qwen | or_explicit_or_both_v1 | OR_mixed_vs_FF_auroc | 0.1689 [0.1476, 0.1903] |
| qwen | or_at_least_one_v1 | OR_mixed_vs_FF_auroc | 0.2402 [0.2240, 0.2582] |
| llama | and_both_following_v1 | AND_auroc | 0.1039 [0.0872, 0.1185] |
| llama | or_explicit_or_both_v1 | OR_mixed_vs_FF_auroc | 0.1539 [0.1198, 0.1901] |
| llama | or_at_least_one_v1 | OR_mixed_vs_FF_auroc | 0.2659 [0.2452, 0.2896] |

Repair gains transfer to each wording at these endpoints. Full OR/AND endpoints, macro and per-topic results and paired intervals remain in the complete tables. These wordings reuse D entities; they are not E generalization results.

## Independent audit, runtime and delivery

The read-only independent audit verified **1541** fit receipts and all run artifact hashes, reconstructed **9246** fit score vectors exactly, checked **43148** point AUROCs with sklearn, and checked all **900** held-out fold boundaries. It independently reconstructed locked wording/atomic/bare scores and the principal weighted bootstrap draws, seed-specific contrasts and metric-averaged three-seed intervals. Audit duration: **624.1 s**. No refitting was used. See [independent-audit.json](independent-audit.json) and [audit code](../../scripts/checkpoint_r2_audit_b25_successor.py).

Production stage runtime (after startup/input validation/import): **4004.7 s (66.74 min)**; peak RSS **4,638,343,168 bytes (4.32 GiB)**. Gate fitting time was 0.429 s. Original published fits totaled 295.681 s (including the four failed first attempts); successor published new-fit time totaled 2505.454 s. Summing those recorded fit times plus the gate gives 2801.564 s. This excludes the original interrupted optimization, startup/validation, scoring, bootstrap and audit overhead, and is not total wall time. Each logistic subfit's cumulative iterations/attempts are retained in the optimizer inventory.

Actual host: **MacBook-Pro-7.lan**, macOS 12.6 arm64. Actual interpreter: `/opt/homebrew/opt/python@3.11/bin/python3.11`, Python 3.11.6, invoked through `/private/tmp/clean-extraction-venv/bin/python`. One fitting process and one BLAS/OpenMP thread were recorded. This is independent CPU review/re-execution on the actual Mac, not second physical-machine replication. Numerical environment and code/input hashes are in [execution.json](execution.json) and [threads.json](threads.json).

The original blocked run, accepted recoverability run and read-only feature backup remain unchanged. The accepted evidence continues to support complete fresh raw batch-1 extraction. No GPU/SSH/RunPod, constituent fitting, behavior/chat, interventions, new supervision budgets, allocation search, scientific-objective tuning or E evaluation ran. Historical compatibility remains unverified.

Reproducible external artifact locations and a hash-bound delivery manifest are in [delivery.json](delivery.json). Heads, tensors, row-bound scores and bootstrap arrays stay outside Git. The committed tables are [candidate inventory](candidate-inventory.csv), [fold inventory](fold-inventory.csv), [complete fit inventory](fit-inventory.csv), [optimizer attempts](optimizer-attempts.csv), [all-layer bank metrics](all-layer-bank-metrics.csv), [complete metrics](metrics.csv), [paired contrasts/retention](contrasts.csv), [precision](precision.csv) and [selection traces/aliases](locked.json).
