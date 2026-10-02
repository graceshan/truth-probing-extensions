# Fresh R2 bounded CPU clean-recoverability results

Input acceptance, execution and independent result re-execution **PASS**. The completed artifacts support acceptance of complete fresh raw batch-1 extraction for both models. All 122 planned fits converged; no retries or fit failures. C_clean demonstrates strong bare-compound recoverability, but **no saved C_clean layer passes both atomic/AND retention margins** at pooled or equal-topic macro scope. D-selected C_clean and its intervals are **optimistic development results**.

## Acceptance and scope

Origin was fetched and the isolated branch `checkpoint-r2-fresh-recoverability-20261002` started exactly at `7656a0eb68a5fd7ff7dac1c73efeab982d540cc2`; no newer tips were pulled or merged. The isolated checkout was clean at launch. The implementation/configuration was committed before the first real fit as `3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f`. Historical artifacts/configurations and the source backup were preserved.

Actual host: **MacBook-Pro-7.lan**, macOS-12.6-arm64-arm-64bit. Interpreter: `/opt/homebrew/opt/python@3.11/bin/python3.11`, Python 3.11.6. This is independent review/re-execution on the actual Mac, **not second physical-machine replication**.

The delivery receipt located the payload in `/Users/apple/truth-probing-backups/20261002T190142Z-r2-full-raw-7656a0e-resized/output`. The pinned inventory SHA-256 `7d54672272be2cbb30cc20559868c2d39581b219761db09fea76bb40da07f2cb`, all 1,174 listed files (16,919,772,945 bytes), and exact producer `verify-output` passed. Direct array inspection covered **all 113 transferred shards per model**, not just pilot samples: 35,843 unique texts / 41,490 ordered logical bindings each; finite FP16 Qwen 28×3584 and Llama 32×4096. Exact model/tokenizer/runtime provenance, token packets, source/group identities and saved-layer conventions passed. Saved layer s is HF hidden_states[s+1], embeddings excluded, final layer post-RMSNorm; fresh batch-1, unpadded, BF16-compute/FP16-storage representation. These checks do not repeat GPU inference or independently reproduce producer numerical comparisons.

P15=2,778; A15=75 persons; atomic D=1,012 observations; compound D=483 pairs / 7,728 bare rows; T_C=100 entities / 500 pairs / 8,000 rows. Every A-person source variant is excluded from P15; training/D persons are disjoint. Identity/hash mapping preserves frozen order, source roles and duplicate-observation exposure: atomic D has 1,006 distinct texts but all 1,012 logical observations. The documented NumPy 2.4.6 environment reproduced every frozen metadata byte. Historical compatibility remains **unverified**; no historical probe, preprocessing statistic or 36 conditional bank matches were admitted.

Model snapshots: Qwen/Qwen2.5-7B-Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28`; meta-llama/Llama-3.1-8B-Instruct revision `0e9e39f249a16976918f6564b8830bc894c89659`. Exact tokenizer/configuration hashes, backend implementation hashes, snapshot receipt hashes and complete extraction runtime identity are in [representation-identity.json](representation-identity.json). Recorded producer runtime is Python 3.12.3, torch 2.8.0+cu128, transformers 4.56.2, CUDA 12.8/cuDNN 91002, A40 driver 595.91.07; this CPU review reads those receipts and does not invoke that GPU runtime.

## Fits and numerical rules

Two inherited raw LR fits: Qwen L17/C=1 and Llama L15/C=10. R0 and C_clean each fit all 28 Qwen and 32 Llama saved layers. R0 uses mean P15 BCE; C_clean uses mean BCE over all 8,000 frozen T_C rows and one shared AND/OR truth head. Both use P15-only population standardization, floor 1e-6, zero constant coordinates, .001 ||w||² and an unpenalized intercept. The reviewed solver, 2,000-initial/10,000-total warm-retry policy and finite/success/gradient-infinity≤1e-4 checks are unchanged. Raw LR retains its inherited convergence policy. No observed outcome changed configuration.

One fitting process; all BLAS/OpenMP libraries recorded one thread. Fit/stream/score phase: 628.67 s; bootstrap/evaluation/plotting: 86.13 s; recorded total: 717.43 s (11.96 min). Peak process RSS: 4,782,358,528 bytes (4.454 GiB). The total starts after input revalidation and includes layer streaming; input acceptance took 57.09 s separately. Independent result audit took 190.04 s. See fit-summary for every attempt and gradient. No failures/retries; no validity blocker.

## Fixed and selected results

AUROC; columns are atomic / AND / OR / OR mixed-versus-FF. Macro gives each of the five frozen topics equal weight. R0 uses original atomic-D selection with exact ties choosing the lower layer; compound performance never enters R0 selection. C_clean uses inherited D OR-boundary selection, then atomic AUROC and standardized separation (1e-12 ties; frozen final tie seed). Selected layers and tie receipts are in selected.json; full candidate metrics are in all-layer-metrics.csv.

### Pooled

| Model | Head | Layer | Atomic | AND | OR | OR mixed/FF |
|---|---|---:|---:|---:|---:|---:|
| qwen | reduced LR | 17 | 0.9998 | 0.9419 | 0.8478 | 0.7783 |
| qwen | R0 fixed | 17 | 0.9998 | 0.9315 | 0.8375 | 0.7651 |
| qwen | R0 atomic-D selected | 16 | 0.9998 | 0.8819 | 0.8353 | 0.7736 |
| qwen | C_clean fixed | 17 | 0.8018 | 0.9971 | 0.9876 | 0.9814 |
| qwen | C_clean at R0-selected layer | 16 | 0.7201 | 0.9959 | 0.9852 | 0.9778 |
| qwen | C_clean D-selected* | 18 | 0.9297 | 0.9984 | 0.9938 | 0.9907 |
| llama | reduced LR | 15 | 1.0000 | 0.9756 | 0.7606 | 0.6546 |
| llama | R0 fixed | 15 | 1.0000 | 0.9604 | 0.7264 | 0.6131 |
| llama | R0 atomic-D selected | 10 | 1.0000 | 0.9473 | 0.7517 | 0.6508 |
| llama | C_clean fixed | 15 | 0.7499 | 0.9994 | 0.9980 | 0.9970 |
| llama | C_clean at R0-selected layer | 10 | 0.6863 | 0.9988 | 0.9928 | 0.9892 |
| llama | C_clean D-selected* | 13 | 0.7305 | 0.9997 | 0.9986 | 0.9979 |

### Equal-topic macro

| Model | Head | Layer | Atomic | AND | OR | OR mixed/FF |
|---|---|---:|---:|---:|---:|---:|
| qwen | reduced LR | 17 | 1.0000 | 0.9144 | 0.8457 | 0.7788 |
| qwen | R0 fixed | 17 | 1.0000 | 0.9018 | 0.8373 | 0.7739 |
| qwen | R0 atomic-D selected | 16 | 1.0000 | 0.8754 | 0.8491 | 0.7902 |
| qwen | C_clean fixed | 17 | 0.7723 | 0.9963 | 0.9855 | 0.9783 |
| qwen | C_clean at R0-selected layer | 16 | 0.7077 | 0.9958 | 0.9840 | 0.9760 |
| qwen | C_clean D-selected* | 18 | 0.8673 | 0.9983 | 0.9908 | 0.9862 |
| llama | reduced LR | 15 | 1.0000 | 0.9893 | 0.8114 | 0.7229 |
| llama | R0 fixed | 15 | 1.0000 | 0.9895 | 0.8030 | 0.7163 |
| llama | R0 atomic-D selected | 10 | 1.0000 | 0.9807 | 0.7831 | 0.6862 |
| llama | C_clean fixed | 15 | 0.7530 | 0.9989 | 0.9942 | 0.9914 |
| llama | C_clean at R0-selected layer | 10 | 0.7083 | 0.9980 | 0.9869 | 0.9803 |
| llama | C_clean D-selected* | 13 | 0.7245 | 0.9995 | 0.9966 | 0.9949 |

*C_clean D-selected results and intervals are optimistic development estimates. R0 atomic selection also uses D; bootstrap does not repeat layer selection.

## Paired contrasts and retention

Pooled C_clean-minus-reference changes with 95% paired percentile intervals. Same-layer controls prevent layer differences from being mistaken for objective differences. Full pooled, equal-topic macro and per-topic contrasts, including every layer against fresh reduced LR, are in paired-contrasts.csv.

| Model | Comparison | Atomic Δ [CI] | AND Δ [CI] | OR Δ [CI] | OR mixed/FF Δ [CI] |
|---|---|---|---|---|---|
| qwen | fixed C_clean − same-layer R0 | -0.1980 [-0.2123, -0.1846] | 0.0656 [0.0503, 0.0812] | 0.1501 [0.1318, 0.1666] | 0.2163 [0.1910, 0.2385] |
| qwen | fixed C_clean − reduced LR | -0.1980 [-0.2123, -0.1846] | 0.0552 [0.0393, 0.0712] | 0.1398 [0.1218, 0.1556] | 0.2031 [0.1787, 0.2241] |
| qwen | D-selected* C_clean − same-layer R0 | -0.0699 [-0.0789, -0.0614] | 0.0098 [0.0054, 0.0157] | 0.0941 [0.0774, 0.1087] | 0.1406 [0.1155, 0.1623] |
| qwen | D-selected* C_clean − atomic-selected R0 | -0.0701 [-0.0792, -0.0616] | 0.1165 [0.0955, 0.1373] | 0.1584 [0.1378, 0.1753] | 0.2170 [0.1908, 0.2372] |
| qwen | D-selected* C_clean − reduced LR | -0.0701 [-0.0792, -0.0616] | 0.0564 [0.0403, 0.0728] | 0.1460 [0.1283, 0.1623] | 0.2123 [0.1886, 0.2338] |
| llama | fixed C_clean − same-layer R0 | -0.2501 [-0.2613, -0.2389] | 0.0390 [0.0303, 0.0478] | 0.2716 [0.2492, 0.2996] | 0.3838 [0.3560, 0.4164] |
| llama | fixed C_clean − reduced LR | -0.2501 [-0.2613, -0.2389] | 0.0238 [0.0166, 0.0317] | 0.2374 [0.2150, 0.2629] | 0.3424 [0.3129, 0.3757] |
| llama | D-selected* C_clean − same-layer R0 | -0.2695 [-0.2785, -0.2605] | 0.0328 [0.0242, 0.0418] | 0.2331 [0.2124, 0.2561] | 0.3445 [0.3151, 0.3752] |
| llama | D-selected* C_clean − atomic-selected R0 | -0.2695 [-0.2785, -0.2605] | 0.0523 [0.0408, 0.0638] | 0.2469 [0.2254, 0.2669] | 0.3472 [0.3211, 0.3713] |
| llama | D-selected* C_clean − reduced LR | -0.2695 [-0.2785, -0.2605] | 0.0241 [0.0168, 0.0319] | 0.2380 [0.2155, 0.2639] | 0.3434 [0.3134, 0.3770] |

Retention against fresh reduced LR requires Δ≥−.005 atomic and Δ≥−.02 AND; interval support requires the paired 95% lower bound to meet that same threshold. Values below are **point / interval-supported**; thresholds were not changed.

| Model | Head | Scope | Atomic retention | AND retention |
|---|---|---|---|---|
| qwen | R0 fixed | pooled | PASS / PASS | PASS / PASS |
| qwen | R0 fixed | topic_macro | PASS / PASS | PASS / PASS |
| qwen | R0 atomic-D selected | pooled | PASS / PASS | FAIL / FAIL |
| qwen | R0 atomic-D selected | topic_macro | PASS / PASS | FAIL / FAIL |
| qwen | C_clean fixed | pooled | FAIL / FAIL | PASS / PASS |
| qwen | C_clean fixed | topic_macro | FAIL / FAIL | PASS / PASS |
| qwen | C_clean at R0-selected layer | pooled | FAIL / FAIL | PASS / PASS |
| qwen | C_clean at R0-selected layer | topic_macro | FAIL / FAIL | PASS / PASS |
| qwen | C_clean D-selected* | pooled | FAIL / FAIL | PASS / PASS |
| qwen | C_clean D-selected* | topic_macro | FAIL / FAIL | PASS / PASS |
| llama | R0 fixed | pooled | PASS / PASS | PASS / PASS |
| llama | R0 fixed | topic_macro | PASS / PASS | PASS / PASS |
| llama | R0 atomic-D selected | pooled | PASS / PASS | FAIL / FAIL |
| llama | R0 atomic-D selected | topic_macro | PASS / PASS | PASS / PASS |
| llama | C_clean fixed | pooled | FAIL / FAIL | PASS / PASS |
| llama | C_clean fixed | topic_macro | FAIL / FAIL | PASS / PASS |
| llama | C_clean at R0-selected layer | pooled | FAIL / FAIL | PASS / PASS |
| llama | C_clean at R0-selected layer | topic_macro | FAIL / FAIL | PASS / PASS |
| llama | C_clean D-selected* | pooled | FAIL / FAIL | PASS / PASS |
| llama | C_clean D-selected* | topic_macro | FAIL / FAIL | PASS / PASS |

The frozen bootstrap uses seed 1729 and 2,000 shared draws across every head/model: topic-stratified atomic person multiplicities and compound endpoint-product weights. Undefined endpoints propagate through the all-five macro; no redraw, imputation or topic dropping. Intervals require 1,800 valid draws and are conditional on fitted/selected heads. All 3,416 metric intervals and 5,544 contrast intervals have all 2,000 draws valid; no undefined endpoint occurred in this run. The independent audit rebuilt all integer schedules, checked all point AUROCs with sklearn, checked five independently recomputed bootstrap draws for every endpoint/head/topic, and verified every saved percentile interval and paired contrast. It also exactly recomputed every saved row score from each reusable head and all 60 P15 preprocessing packets.

## Artifacts and reproduction

- [All-layer pooled, macro and per-topic AUROCs/intervals](all-layer-metrics.csv) and [curves](all_layer_curves.svg). Published curves use the full 0–1 AUROC axis; the original external figure clipped a few low scores below .45. This rendering change leaves all numerical results unchanged ([receipt](curve-receipt.json)).
- [Fixed/selected per-topic and summary results](fixed-selected-metrics.csv); [all paired contrasts/retention](paired-contrasts.csv).
- [Input acceptance](acceptance.json), [exact metadata rebuild](exact-metadata-rebuild.json), [17 focused implementation tests](implementation-tests.json), [result audit](result-audit.json), [fit attempts](fit-summary.json), [execution/code/input hashes](execution.json), [complete external artifact manifest](completion.json).
- External reusable heads, P15 statistics, ordered row scores, bootstrap arrays and tensors remain outside Git. Run root: `/Users/apple/truth-probing-backups/20261001T164224Z/r2-recoverability-results-20261002/run-3a5b7bb`. Completion binds each external artifact by SHA-256 and size; the backup remained unchanged.
- Configuration SHA-256: `60066e87d1caee3213fa351af9a9f0161be438de26b9a6a07738cdc1b3315e28`; input manifest SHA-256: `f28e75f5fc95cf3f019f5b47c68c5c2753435854280bfb122f0df22187954445`. Each result identifies producer `3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f`.
- [Runner/reproduction instructions](../../docs/checkpoint_r2/fresh_recoverability_v1.md). Re-run the read-only audit with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /private/tmp/clean-extraction-venv/bin/python -B scripts/checkpoint_r2_audit_recoverability.py CHECKOUT EXTERNAL_RUN NEW_AUDIT_JSON`.

Only the requested bare-D stage ran. Wording was excluded from fitting, selection and evaluation. Remaining atomic bank, S_atom_all, B25 selection/repair, constituent heads, behavior/chat and E remain for subsequent authorized panels. High bare-compound AUROC does not establish wording robustness, held-out E generalization, joint atomic/AND retention, or historical compatibility.
