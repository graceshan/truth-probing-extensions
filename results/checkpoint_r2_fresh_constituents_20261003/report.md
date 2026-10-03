# Fresh raw constituent-truth transfer: completed development panel

Both-model execution and independent saved-prediction audit passed. Every initial and final fit is valid. **Bare D supports strong separately decodable constituent truth in both models. Llama meets all six .05 paired transfer-loss planning bounds in both directions; Qwen has one unresolved conditional endpoint per direction. The panel does not support wording-general useful transfer or uniformly strong frozen decisions.** All raw inputs, pair identities, objectives and source-only selection rules stayed frozen. No E, behavior/chat, interventions, atomic-bank or B25 fitting ran.

Fitting producer: `2d28c0b2733a4433aa6ceb4de020bfb815c2f7db`. Configuration SHA-256: `21e99762ffb43ded228d579c9ec8757aef7326c04e40e822aa1850b68ebe8bde`. Exact base: `1187140c7f79b0b64439eff382d420f51d974a25`. The delivery SHA is the final pushed branch commit, reported with this packet; it cannot be embedded in its own hashed file.

Actual host: MacBook-Pro-7.lan; macOS-12.6-arm64-arm-64bit; Python 3.11.6 via `/private/tmp/clean-extraction-venv/bin/python`. One fitting process and one BLAS/OpenMP thread. Initial heads: 240/240 valid. Final heads: 12/12 valid, deduplicated only for identical exposures/settings. Failed fits: 0; warm optimizer retries: 0.

Production runtime after input validation: 1000.8 s; peak Darwin RSS: 3,066,249,216 bytes (2.86 GiB). Independent audit: 353.3 s; 770 score vectors reconstructed, maximum absolute difference 0.0; 252 independently checked objectives/gradients; 22995 principal/all-layer point metrics; 2,100,000 bootstrap values; 420 paired endpoint contrasts. No refitting in audit.

Outer UTC command spans are 2,448 s (40.80 min) for startup/input validation plus the fitting/evaluation command and 1,470 s (24.50 min) for the audit command; total tracked span is 3,918 s (65.30 min). These are broader than the separately recorded monotonic runner/audit durations. Unpartitioned startup/control time is retained in `execution-accounting.json`; its full cause is not inferred or silently discarded. Audit peak RSS was 5,146,460,160 bytes (4.79 GiB).

The first detached launch ended before result initialization and produced no fits. Its receipts remain alongside the tracked foreground execution; the termination cause is unestablished. There was one actual fitting execution, no repeated fit attempt or scientific change. Solver warm retries, if any, are the inherited numerical procedure, not new supervision samples.

## Frozen exposures and accepted representations

TC has 20 entities/topic. Internal validation has the frozen four A15 entities/topic and six pairs/topic (240 logical rows/operator), excluded from source fitting and P15. Source fitting uses the remaining 16/topic and frozen 100 pairs/topic (4,000 logical rows/operator). Final refits use all TC and the same frozen 100-of-190 recipe as C_clean. FIRST/SECOND refer to surface position, swapping under BA. Logical observations preserve identity/order even for shared physical texts.

Fresh batch-1 BF16-compute/FP16-storage all-layer features were accepted under complete-copy inventory `7d54672272be2cbb30cc20559868c2d39581b219761db09fea76bb40da07f2cb`; acceptance, every listed hash and producer verify-output passed again. Saved layer s is HF hidden_states[s+1], embeddings excluded, final layer post-RMSNorm. P15 preprocessing is hash-bound to fresh recoverability producer `3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f`, and reproduced exactly from fresh P15 before reuse. No historical heads/statistics/36 conditional matches were adopted; historical compatibility remains unverified.

## Source-only selected layers

Endpoint order in every vector: FIRST marginal; SECOND marginal; FIRST given SECOND false/true; SECOND given FIRST false/true. Selection maximizes the minimum of the six equal-topic macros, then mean within 1e-12, then lower saved layer. Target validation, D and wording never enter selection.

| Model | Source | Saved layer | Source-validation six-endpoint macro vector |
|---|---|---:|---|
| qwen | AND | 18 | 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000 |
| qwen | OR | 19 | 0.9993, 0.9962, 1.0000, 0.9972, 0.9972, 0.9986 |
| llama | AND | 12 | 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000 |
| llama | OR | 12 | 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000 |

## Bare D: three matched-layer six-endpoint comparisons

Equal-topic macro AUROCs; all six endpoints remain separate. Full pooled/topic/AB/BA and intervals are in `six-endpoints.csv` and `constituent-transfer.csv`. Source-fit all-layer D curves in `constituent-layer-curves.csv` are descriptive, from the 16/topic fits, and remain separate from the selected full-TC refits.

| Model | Source direction | Evaluation arm | Six-endpoint macro vector |
|---|---|---|---|
| qwen | AND | source_on_source | 0.9986, 0.9984, 0.9989, 0.9983, 0.9965, 0.9995 |
| qwen | AND | source_on_target | 0.9670, 0.9856, 0.9772, 0.9675, 0.9766, 0.9940 |
| qwen | AND | target_on_target | 0.9968, 0.9970, 0.9975, 0.9965, 0.9953, 0.9979 |
| qwen | OR | source_on_source | 0.9963, 0.9959, 0.9971, 0.9959, 0.9943, 0.9971 |
| qwen | OR | source_on_target | 0.9873, 0.9752, 0.9932, 0.9880, 0.9796, 0.9741 |
| qwen | OR | target_on_target | 0.9976, 0.9980, 0.9977, 0.9974, 0.9957, 0.9993 |
| llama | AND | source_on_source | 0.9998, 0.9996, 0.9998, 0.9999, 0.9989, 0.9999 |
| llama | AND | source_on_target | 0.9979, 0.9958, 0.9985, 0.9972, 0.9945, 0.9995 |
| llama | AND | target_on_target | 1.0000, 0.9995, 0.9999, 1.0000, 0.9990, 0.9998 |
| llama | OR | source_on_source | 1.0000, 0.9995, 0.9999, 1.0000, 0.9990, 0.9998 |
| llama | OR | source_on_target | 0.9913, 0.9979, 0.9966, 0.9948, 0.9973, 0.9993 |
| llama | OR | target_on_target | 0.9998, 0.9996, 0.9998, 0.9999, 0.9989, 0.9999 |

## Paired bare-D transfer losses

Target-trained minus source-trained on identical target states; equal-topic macros, pointwise 95% percentile intervals.

| Model | Direction | Endpoint | Loss [95% interval] | Valid draws | .05 planning-margin status |
|---|---|---|---|---:|---|
| qwen | AND→OR | first_marginal | 0.0298 [0.0152, 0.0446] | 2000/2000 | upper_bound_below_05 |
| qwen | AND→OR | second_marginal | 0.0114 [0.0045, 0.0218] | 2000/2000 | upper_bound_below_05 |
| qwen | AND→OR | first_given_second_0 | 0.0204 [0.0060, 0.0364] | 2000/2000 | upper_bound_below_05 |
| qwen | AND→OR | first_given_second_1 | 0.0290 [0.0082, 0.0577] | 2000/2000 | unresolved_crosses_05 |
| qwen | AND→OR | second_given_first_0 | 0.0186 [0.0058, 0.0395] | 2000/2000 | upper_bound_below_05 |
| qwen | AND→OR | second_given_first_1 | 0.0039 [-0.0017, 0.0130] | 2000/2000 | upper_bound_below_05 |
| qwen | OR→AND | first_marginal | 0.0103 [0.0036, 0.0201] | 2000/2000 | upper_bound_below_05 |
| qwen | OR→AND | second_marginal | 0.0228 [0.0119, 0.0372] | 2000/2000 | upper_bound_below_05 |
| qwen | OR→AND | first_given_second_0 | 0.0044 [-0.0020, 0.0134] | 2000/2000 | upper_bound_below_05 |
| qwen | OR→AND | first_given_second_1 | 0.0093 [0.0020, 0.0229] | 2000/2000 | upper_bound_below_05 |
| qwen | OR→AND | second_given_first_0 | 0.0161 [0.0034, 0.0367] | 2000/2000 | upper_bound_below_05 |
| qwen | OR→AND | second_given_first_1 | 0.0252 [0.0085, 0.0552] | 2000/2000 | unresolved_crosses_05 |
| llama | AND→OR | first_marginal | 0.0021 [0.0002, 0.0061] | 2000/2000 | upper_bound_below_05 |
| llama | AND→OR | second_marginal | 0.0037 [0.0012, 0.0071] | 2000/2000 | upper_bound_below_05 |
| llama | AND→OR | first_given_second_0 | 0.0014 [-0.0000, 0.0046] | 2000/2000 | upper_bound_below_05 |
| llama | AND→OR | first_given_second_1 | 0.0028 [0.0000, 0.0098] | 2000/2000 | upper_bound_below_05 |
| llama | AND→OR | second_given_first_0 | 0.0045 [0.0007, 0.0107] | 2000/2000 | upper_bound_below_05 |
| llama | AND→OR | second_given_first_1 | 0.0003 [0.0000, 0.0010] | 2000/2000 | upper_bound_below_05 |
| llama | OR→AND | first_marginal | 0.0086 [0.0040, 0.0139] | 2000/2000 | upper_bound_below_05 |
| llama | OR→AND | second_marginal | 0.0016 [-0.0001, 0.0069] | 2000/2000 | upper_bound_below_05 |
| llama | OR→AND | first_given_second_0 | 0.0031 [0.0002, 0.0086] | 2000/2000 | upper_bound_below_05 |
| llama | OR→AND | first_given_second_1 | 0.0051 [0.0004, 0.0138] | 2000/2000 | upper_bound_below_05 |
| llama | OR→AND | second_given_first_0 | 0.0016 [-0.0001, 0.0096] | 2000/2000 | upper_bound_below_05 |
| llama | OR→AND | second_given_first_1 | 0.0007 [-0.0000, 0.0041] | 2000/2000 | upper_bound_below_05 |

Bootstrap: reviewed shared topic-stratified PCG64(1729) person/entity schedule, pair endpoint-product weights, 2,000 draws, >=1,800 valid for intervals. Wordings inherit the same bare pair/fact/order multiplicities. Empty classes/topics remain undefined; all-five macros propagate undefined draws. No selection is repeated within draws. These are pointwise conditional intervals, without simultaneous or training-uncertainty coverage. .90 absolute competence and .05 transfer-loss margin are planning targets. Small gaps between weak source/target heads do not support useful transfer; conditional strength with weak marginals does not establish general constituent decoding.

## Wording, fixed decisions and external composition

All three established wordings use the unchanged locked heads. These templates were held out from fitting/selection, not unseen by the researcher. No sign flips, threshold fitting or recalibration. Fixed probability threshold .5 is logit >=0; exact ties are True. Min frozen logits for AND/max for OR and Boolean decisions are external composition, not evidence of causal use. Full per-head accuracy/BA, joint correctness, compound/boundary AUROC and Boolean accuracy/BA are in `fixed-decisions-composition.csv`.

| Model | Source | Arm | Condition | Minimum six macro AUROCs | FIRST / SECOND accuracy | Joint correctness | Composition AUROC | Boolean BA |
|---|---|---|---|---:|---|---:|---:|---:|
| llama | AND | source_on_source | D_bare | 0.9989 | 0.9945 / 0.9898 | 0.9843 | 0.9997 | 0.9944 |
| llama | AND | source_on_source | and_both_following_v1 | 0.9817 | 0.7479 / 0.9954 | 0.7438 | 0.9912 | 0.6078 |
| llama | AND | source_on_target | D_bare | 0.9945 | 0.8244 / 0.9680 | 0.7978 | 0.9847 | 0.9332 |
| llama | AND | source_on_target | or_at_least_one_v1 | 0.9470 | 0.6787 / 0.9928 | 0.6737 | 0.9881 | 0.9191 |
| llama | AND | source_on_target | or_explicit_or_both_v1 | 0.9133 | 0.5687 / 0.7094 | 0.4153 | 0.8651 | 0.7112 |
| llama | AND | target_on_target | D_bare | 0.9990 | 0.9971 / 0.9882 | 0.9853 | 0.9995 | 0.9883 |
| llama | AND | target_on_target | or_at_least_one_v1 | 0.9228 | 0.8024 / 0.9945 | 0.7981 | 0.9917 | 0.9381 |
| llama | AND | target_on_target | or_explicit_or_both_v1 | 0.9685 | 0.6828 / 0.8166 | 0.5837 | 0.8518 | 0.7506 |
| llama | OR | source_on_source | D_bare | 0.9990 | 0.9971 / 0.9882 | 0.9853 | 0.9995 | 0.9883 |
| llama | OR | source_on_source | or_at_least_one_v1 | 0.9228 | 0.8024 / 0.9945 | 0.7981 | 0.9917 | 0.9381 |
| llama | OR | source_on_source | or_explicit_or_both_v1 | 0.9685 | 0.6828 / 0.8166 | 0.5837 | 0.8518 | 0.7506 |
| llama | OR | source_on_target | D_bare | 0.9913 | 0.9199 / 0.9832 | 0.9035 | 0.9688 | 0.9535 |
| llama | OR | source_on_target | and_both_following_v1 | 0.9541 | 0.8869 / 0.9713 | 0.8619 | 0.9659 | 0.9099 |
| llama | OR | target_on_target | D_bare | 0.9989 | 0.9945 / 0.9898 | 0.9843 | 0.9997 | 0.9944 |
| llama | OR | target_on_target | and_both_following_v1 | 0.9817 | 0.7479 / 0.9954 | 0.7438 | 0.9912 | 0.6078 |
| qwen | AND | source_on_source | D_bare | 0.9965 | 0.9850 / 0.9802 | 0.9657 | 0.9985 | 0.9871 |
| qwen | AND | source_on_source | and_both_following_v1 | 0.9908 | 0.8329 / 0.9120 | 0.7468 | 0.9940 | 0.6695 |
| qwen | AND | source_on_target | D_bare | 0.9670 | 0.8391 / 0.9323 | 0.7770 | 0.9438 | 0.8654 |
| qwen | AND | source_on_target | or_at_least_one_v1 | 0.9265 | 0.5515 / 0.7997 | 0.4217 | 0.9763 | 0.7336 |
| qwen | AND | source_on_target | or_explicit_or_both_v1 | 0.8770 | 0.6822 / 0.6471 | 0.4614 | 0.8992 | 0.7047 |
| qwen | AND | target_on_target | D_bare | 0.9953 | 0.9786 / 0.9842 | 0.9633 | 0.9955 | 0.9740 |
| qwen | AND | target_on_target | or_at_least_one_v1 | 0.9750 | 0.6476 / 0.9571 | 0.6201 | 0.9832 | 0.8936 |
| qwen | AND | target_on_target | or_explicit_or_both_v1 | 0.9372 | 0.7077 / 0.6310 | 0.5073 | 0.8811 | 0.6278 |
| qwen | OR | source_on_source | D_bare | 0.9943 | 0.9792 / 0.9810 | 0.9607 | 0.9948 | 0.9714 |
| qwen | OR | source_on_source | or_at_least_one_v1 | 0.9550 | 0.6151 / 0.8813 | 0.5154 | 0.9677 | 0.8116 |
| qwen | OR | source_on_source | or_explicit_or_both_v1 | 0.9501 | 0.7078 / 0.7084 | 0.5517 | 0.8970 | 0.6936 |
| qwen | OR | source_on_target | D_bare | 0.9741 | 0.8832 / 0.9023 | 0.8073 | 0.9512 | 0.8891 |
| qwen | OR | source_on_target | and_both_following_v1 | 0.9821 | 0.9524 / 0.8867 | 0.8469 | 0.9837 | 0.8992 |
| qwen | OR | target_on_target | D_bare | 0.9957 | 0.9829 / 0.9781 | 0.9615 | 0.9979 | 0.9835 |
| qwen | OR | target_on_target | and_both_following_v1 | 0.9877 | 0.8759 / 0.9154 | 0.7922 | 0.9921 | 0.7579 |

## Wording-specific endpoint classifications

For Qwen AND→OR on `or_explicit_or_both_v1`, FIRST marginal macro AUROC is .8944 [.8768, .9168], and the paired target-minus-source loss is **.0768 [.0593, .0921]**, a material deficit against .05. Three transferred six-endpoint macro points on that wording are below .90; the minimum is .8770. On `or_at_least_one_v1`, FIRST-given-SECOND-true loss is .0485 [.0163, .0877], unresolved.

For Llama AND→OR on explicit-or-both, SECOND marginal loss is .0553 [.0429, .0678] and SECOND-given-FIRST-false is .0539 [.0345, .0746], both unresolved. Llama wording macro points remain >=.90, but some absolute wording endpoint bounds cross .90. Qwen also has uncertain absolute wording bounds in addition to the three sub-.90 points. All principal endpoint intervals have **2,000/2,000 valid draws**. The CSVs retain every endpoint/topic classification, including stronger reverse-wording results; no aggregate substitutes for these failures or uncertainties.

Frozen decisions expose a separate limitation: Llama AND-source on the held-out AND wording has minimum six-endpoint macro AUROC .9817 but Boolean BA .6078. Qwen AND→OR at-least-one has minimum .9265, FIRST accuracy .5515 and joint correctness .4217. Separate head offsets/scales can impair min/max or Boolean composition despite strong individual rankings. This panel applies the original composition procedure without fitting an offset/scale correction; poor composition is not absence of constituent information.

## Scientific decision and boundaries

- **qwen, AND-source:** Bare point competence is promising, but at least one transfer-loss interval remains unresolved against .05; useful transfer remains uncertain.
- **qwen, OR-source:** Bare point competence is promising, but at least one transfer-loss interval remains unresolved against .05; useful transfer remains uncertain.
- **llama, AND-source:** Bare points and all six transfer-loss bounds meet the planning pattern; absolute bounds and all wording conditions must still qualify the conclusion. The bare ranking result is supported, but wording uncertainty and fixed-decision losses limit broader claims; decoding/composition alone does not establish internal causal use.
- **llama, OR-source:** Bare points and all six transfer-loss bounds meet the planning pattern; absolute bounds and all wording conditions must still qualify the conclusion. The bare ranking result is supported, but wording uncertainty and fixed-decision losses limit broader claims; decoding/composition alone does not establish internal causal use.

The next warranted diagnostic is **one bounded saved-score study of operator/wording offset, scale and interference patterns**, using existing heads and observations, without refitting, sign flips, new thresholds or E access. It should distinguish a ranking loss (notably Qwen explicit-or-both) from a frozen decision/composition loss. Any later source-only calibration experiment needs its own prospective design and authorization and must not replace this original panel. A joint-readout fit is not the first remedy for the strong bare-D ranking transfer observed here. Matched behavior remains a separate explanatory panel; these findings do not yet warrant causal intervention preparation.

Across-condition useful transfer requires every separate endpoint and planned wording; no mean can substitute for a failure. No follow-up was launched. No causal, final-test, or historical compatibility claim follows.

## Rebuild and artifacts

External run: `/Users/apple/truth-probing-backups/20261001T164224Z/r2-constituent-results-20261003/run-2d28c0b`. External control/audit: `/Users/apple/truth-probing-backups/20261001T164224Z/r2-constituent-results-20261003`. Large heads, preprocessing, ordered scores and bootstrap arrays remain outside Git. `delivery.json` pins the external completion/manifest and committed compact files.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
/private/tmp/clean-extraction-venv/bin/python -B -m src.checkpoint_r2_fresh_constituents run --expected-commit 2d28c0b2733a4433aa6ceb4de020bfb815c2f7db --output NEW_EXTERNAL_RUN_DIRECTORY
/private/tmp/clean-extraction-venv/bin/python -B -m scripts.checkpoint_r2_audit_constituents --run /Users/apple/truth-probing-backups/20261001T164224Z/r2-constituent-results-20261003/run-2d28c0b --expected-producer 2d28c0b2733a4433aa6ceb4de020bfb815c2f7db --receipt NEW_AUDIT_RECEIPT.json
```

The fitting command requires a clean checkout at the fitting producer; the audit may run from the final delivery checkout because all implementation/config/data hashes remain pinned. It rebuilds saved predictions/principal metrics/paired contrasts without fitting. Prior raw, pilot, recoverability and B25 results and all unrelated files remain byte-preserved.
