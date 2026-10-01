# V4 full-versus-P15 capacity pilot

The bounded pilot completed **72 unique configurations**, with **zero failed or flagged fits** and **1,260 endpoint summaries**. Two exact v4 canonical full raw-LR fits were reused; **70 new configuration fits** ran. Fitting/scoring/evaluation took **194.05 seconds**, including **113.53 seconds** summed configuration fitting/checkpoint timing; initial input verification and transfers are outside that timer. Execution stayed on Darwin, `MacBook-Pro-7.lan`, with one process and one BLAS thread. Every configuration was checkpointed before evaluation.

The result **does not support freezing the 15/topic proposal for the whole method bank yet**. There are **14 review triggers**: two pooled, two macro and nine topic OR-boundary drops, plus one topic atomic drop. These trigger review, not an automatic A membership change.

## Scope, adoption and physical inputs

Branch `t2-capacity-pilot-20261001` starts from reviewed `4cae65c7dfc825f774226007a81994f5a08776c9`; normal merge `40147cfb3861df6944a47fb112c539c89ac6bb6d` incorporates exactly `fd2d7be3b910c3fdba7ff6b2c2891eb0ab1b7992`. `selection_repair_adoption_v4.json` binds source correction, representation evidence and scope to exact hashes. `tc_successor_status` and `tc_current_eligible` are authoritative. All v3 restrictions and the four new v4 IDs remain excluded. A15/A10 identities and every earlier overlay/output are preserved. No later correction was incorporated.

Full corrected train has **3,040** rows (1,520 per label); P15 has **2,778** (1,389 per label), excluding every A15 person and source variant. The actual historical seed-0 balanced sampler is recomputed and its ordered identities checked: full **1,000**, P15 **700**. Each topic contributes 200/140 balanced rows. Full and P15 source identity lists, row/source/statement hashes, person/entity keys, labels/forms and original tensor indices are bound in external membership and mapping artifacts; per-configuration exposure summaries and hashes are committed. There is no P/A freeze.

All fits use the same v4-eligible D set: **1,012 atomic rows**, **483 complete pairs / 7,728 bare rows**, **3,864 rows per operator**, and **2,898 rows per critical boundary**. Evaluation never filters from scores. Complete-group gating uses exact constituent facts and accepted eligibility, not historical truth labels. Coverage and training changes for the canonical refresh are separately reported in `selection_repair_canonical_sensitivity_v4.md`; v3-to-v4 D coverage is unchanged.

Canonical-layer exports were reused only after physical and identity revalidation. Twelve additional-layer exports (Qwen L18/L22, Llama L10/L20 × train/atomic-D/bare-D) total **361,883,136 bytes**. Source SHA-256, actual byte lengths, NPY headers (`<f2`, C-order), source stability, live sidecar/manifest/progress/completion hashes, exact original row/layer indices and local export integrity all passed on verified host **ef7f7534c328**. Strict SSH host verification remained enabled; commands were one-shot transfers initiated on macOS. Large artifacts stay outside Git.

The first seek-based pilot exporter was intentionally stopped while waiting on remote filesystem reads. Its partial archive/request and exit record remain preserved. The separately versioned v2 exporter hashes each source once sequentially while collecting only requested byte spans for both layers; synthetic tests cover spans crossing chunk boundaries and truncation. No authentication or tensor-identity failure occurred. Historical hash-bound exporter code is unchanged.

Bounded recorded representation adoption retains raw-statement, last-real-token, `hidden_states[1:]`, zero-based layer, BF16 extraction/float16 storage evidence. Historical producer-code hash conflicts and missing standalone tokenizer/config bytes remain unresolved. No historical model execution is claimed independently replayed; a fresh extraction needs separate compatibility verification.

## Prespecified methods and inference

Layers are Qwen **17,18,22** and Llama **10,15,20**, zero-based. At each, full and P15 fit six methods:

- Raw LR uses Qwen C=1 / Llama C=10, unchanged canonical L-BFGS and no standardization.
- DoM uses the unchanged unit true-minus-false mean direction on all cohort rows.
- Covariance-adjusted mass mean uses cohort within-class covariance divided by n, absolute pseudoinverse cutoff 1e-3 and rtol=0, without shrinkage.
- t_G uses the exact balanced cohort sample and dataset-centered OLS; inference is the raw activation dot product.
- TTPD uses the same balanced sample for its dataset means, t_G, unregularized polarity LR and two-coordinate truth LR. Raw inference projects on t_G and the polarity coefficient; no polarity intercept, form/operator feature or evaluation centering is added. Original 100/10,000 cold warning-retry settings are preserved, with independent gradient/finite checks.
- R0 uses the verified `selection_repair_objectives`: cohort-only float64 population scaling, denominator max(std,1e-6), constants always zero, mean BCE + .001||w||², free intercept, 2,000 initial / 10,000 total warm-retry budget and gradient target 1e-4. Full R0 is explicitly an A-exposed capacity diagnostic; its full-cohort statistics never enter P15.

Every fitting component is estimated from its own method's fitting cohort. All inference is label-blind and uses saved parameters. TTPD and R0 explicit readouts were compared against their fused affine form. Full-training configurations are **never primary P-only bank members**. C is only operative for raw LR; it is carried as a pinned model reference in the configuration grid, not used for the other objectives.

Raw LR iterations range **22–59**, R0 **21–36**, and the TTPD component fits **7–34**. No iterative fit needed a retry. Maximum final gradient infinity norms are LR **9.75669e-5**, R0 **9.88725e-5**, TTPD **9.82771e-5**. DoM, covariance-MM and t_G are finite closed-form fits; they have no iterative convergence criterion. Numerical/library settings and every attempt are in `fits.json`.

## All 72 configurations and withholding effects

`configuration_summary.csv` has one row per configuration, including actual rows/entities/persons/labels, convergence, iterations, runtime, reused-fit flag and parameter/membership hashes. The table below pairs its full and P15 rows (36 pairs = 72 configurations). All are valid. Differences are P15 minus full on identical retained examples. Full pooled, macro and per-topic endpoint values and pointwise paired intervals are in `metrics.csv` / `metrics.json`.

| Model | Layer | Method | Fit rows full/P15 | Atomic pooled Δ | OR boundary pooled Δ | OR boundary macro Δ |
|---|---:|---|---:|---:|---:|---:|
| llama | 10 | l2_logistic | 3040/2778 | +0.000008 | -0.020617 | -0.012343 |
| llama | 10 | difference_of_means | 3040/2778 | +0.002035 | -0.003250 | -0.004626 |
| llama | 10 | mass_mean_covariance | 3040/2778 | +0.000000 | +0.016129 | +0.010516 |
| llama | 10 | burger_t_g | 1000/700 | +0.000566 | -0.003024 | +0.002317 |
| llama | 10 | ttpd | 1000/700 | +0.000141 | -0.003220 | +0.001531 |
| llama | 10 | r0 | 3040/2778 | +0.000004 | -0.005732 | +0.007100 |
| llama | 15 | l2_logistic | 3040/2778 | -0.000004 | +0.009252 | +0.010233 |
| llama | 15 | difference_of_means | 3040/2778 | +0.003375 | +0.000325 | +0.000547 |
| llama | 15 | mass_mean_covariance | 3040/2778 | +0.000000 | -0.009598 | -0.023845 |
| llama | 15 | burger_t_g | 1000/700 | +0.002652 | +0.001250 | +0.002517 |
| llama | 15 | ttpd | 1000/700 | +0.000707 | +0.001113 | +0.001419 |
| llama | 15 | r0 | 3040/2778 | +0.000000 | +0.017998 | +0.019259 |
| llama | 20 | l2_logistic | 3040/2778 | +0.000000 | -0.004619 | +0.001391 |
| llama | 20 | difference_of_means | 3040/2778 | +0.003484 | -0.000520 | +0.000688 |
| llama | 20 | mass_mean_covariance | 3040/2778 | +0.000000 | -0.029132 | +0.003324 |
| llama | 20 | burger_t_g | 1000/700 | +0.002433 | +0.000784 | +0.002350 |
| llama | 20 | ttpd | 1000/700 | +0.000937 | -0.000204 | +0.000529 |
| llama | 20 | r0 | 3040/2778 | +0.000000 | +0.006768 | +0.008563 |
| qwen | 17 | l2_logistic | 3040/2778 | +0.000008 | +0.002169 | -0.004149 |
| qwen | 17 | difference_of_means | 3040/2778 | +0.000125 | +0.000171 | -0.000091 |
| qwen | 17 | mass_mean_covariance | 3040/2778 | -0.000109 | -0.087685 | -0.050736 |
| qwen | 17 | burger_t_g | 1000/700 | -0.000027 | -0.002788 | +0.000022 |
| qwen | 17 | ttpd | 1000/700 | -0.000762 | -0.005159 | -0.000213 |
| qwen | 17 | r0 | 3040/2778 | +0.000000 | +0.001793 | -0.012937 |
| qwen | 18 | l2_logistic | 3040/2778 | +0.000031 | +0.007729 | -0.000660 |
| qwen | 18 | difference_of_means | 3040/2778 | +0.000109 | +0.000281 | -0.000106 |
| qwen | 18 | mass_mean_covariance | 3040/2778 | +0.000273 | -0.167999 | -0.116746 |
| qwen | 18 | burger_t_g | 1000/700 | +0.000062 | -0.003592 | -0.000941 |
| qwen | 18 | ttpd | 1000/700 | -0.000344 | -0.003638 | -0.000589 |
| qwen | 18 | r0 | 3040/2778 | +0.000000 | +0.007553 | +0.013609 |
| qwen | 22 | l2_logistic | 3040/2778 | +0.000215 | +0.007104 | +0.022073 |
| qwen | 22 | difference_of_means | 3040/2778 | +0.000324 | +0.000032 | -0.000217 |
| qwen | 22 | mass_mean_covariance | 3040/2778 | +0.000074 | +0.073453 | +0.064794 |
| qwen | 22 | burger_t_g | 1000/700 | -0.000180 | -0.002975 | -0.000675 |
| qwen | 22 | ttpd | 1000/700 | -0.000281 | -0.002842 | -0.000968 |
| qwen | 22 | r0 | 3040/2778 | -0.000012 | +0.046950 | +0.055429 |

## Paired uncertainty and review triggers

All 1,260 summaries have **2,000 valid seed-1729 paired draws**. The established topic-stratified person/entity bootstrap uses identical full/P15 multiplicities, preserves topic membership, and multiplies endpoint multiplicities for compound pairs. Intervals are pointwise 95% percentile intervals conditional on fitted heads, with no refitting/redraw/imputation/topic dropping. They are not simultaneous or selection-adjusted intervals.

Strict triggers are atomic Δ < −.005 or OR mixed-versus-FF Δ < −.05. Triggering does not depend on whether the paired interval excludes zero. No pooled or macro atomic comparison triggers. Exact triggered results follow.

| Model/layer/method | Scope/topic | Endpoint | Full | P15 | Δ [95% CI] |
|---|---|---|---:|---:|---|
| llama L15 mass_mean_covariance | topic/element_symb | or_mixed_vs_ff_auroc | 0.724441 | 0.667824 | -0.056617 [-0.095941, -0.017575] |
| llama L15 ttpd | topic/inventors | atomic_auroc | 0.964803 | 0.955669 | -0.009133 [-0.016331, -0.003382] |
| qwen L17 mass_mean_covariance | pooled/all | or_mixed_vs_ff_auroc | 0.711500 | 0.623815 | -0.087685 [-0.108837, -0.065865] |
| qwen L17 mass_mean_covariance | topic/cities | or_mixed_vs_ff_auroc | 0.770176 | 0.703169 | -0.067007 [-0.098443, -0.034474] |
| qwen L17 mass_mean_covariance | topic/sp_en_trans | or_mixed_vs_ff_auroc | 0.661927 | 0.545631 | -0.116295 [-0.161992, -0.074654] |
| qwen L17 mass_mean_covariance | topic/element_symb | or_mixed_vs_ff_auroc | 0.836902 | 0.764757 | -0.072145 [-0.146491, -0.016766] |
| qwen L17 mass_mean_covariance | topic/animal_class | or_mixed_vs_ff_auroc | 0.849365 | 0.794800 | -0.054565 [-0.126565, +0.026934] |
| qwen L17 mass_mean_covariance | topic_macro/all | or_mixed_vs_ff_auroc | 0.734534 | 0.683798 | -0.050736 [-0.079237, -0.025385] |
| qwen L18 mass_mean_covariance | pooled/all | or_mixed_vs_ff_auroc | 0.773015 | 0.605017 | -0.167999 [-0.195793, -0.144532] |
| qwen L18 mass_mean_covariance | topic/cities | or_mixed_vs_ff_auroc | 0.820639 | 0.631538 | -0.189100 [-0.226219, -0.156170] |
| qwen L18 mass_mean_covariance | topic/sp_en_trans | or_mixed_vs_ff_auroc | 0.798848 | 0.637111 | -0.161738 [-0.244080, -0.092579] |
| qwen L18 mass_mean_covariance | topic/inventors | or_mixed_vs_ff_auroc | 0.719492 | 0.635516 | -0.083975 [-0.165821, -0.014398] |
| qwen L18 mass_mean_covariance | topic/element_symb | or_mixed_vs_ff_auroc | 0.559317 | 0.404417 | -0.154900 [-0.248287, -0.083126] |
| qwen L18 mass_mean_covariance | topic_macro/all | or_mixed_vs_ff_auroc | 0.708688 | 0.591941 | -0.116746 [-0.149748, -0.088574] |

The largest losses concentrate in covariance-MM, especially Qwen L17/L18. Both pooled intervals lie wholly below −.05. The retained covariance ranks also change (Qwen L17: 2,489→2,369; L18: 2,531→2,408); this is a diagnostic observation, not a causal explanation or permission to retune the fixed cutoff. Llama L15 TTPD's inventor atomic drop exceeds .005, though its interval spans that threshold. One triggered animal-topic covariance interval includes zero. LR and R0 have no prespecified drop triggers at any reported scope.

## 15-versus-10 recommendation and precision before freeze

**Do not freeze P15 for the whole bank yet. Recommend seeking a separately authorized nested-P10 comparison next.** Strong LR/R0 capacity is encouraging, but cannot cancel material covariance-MM losses for other members of the planned bank. This task stops at P15; it supplies no evidence that P10 fixes those losses.

The existing P10 proposal would restore **86** training rows and increase balanced exposure **700→800**, while reducing possible unordered within-topic A pairs **105→45**. That is a real adaptation-capacity tradeoff; future B, allocation and fold choices must reflect it. Neither A membership nor B is changed here. For balanced methods, the observed difference includes deterministic resampling changes as well as withheld entities; it is not an isolated causal effect of removing A rows.

Before design freeze, review the method/topic triggers and decide whether to authorize the nested comparison. Fix the decision/precision rule and account for the many pointwise comparisons. Among per-topic checks, six atomic and sixteen OR-boundary intervals span their respective drop thresholds; the widest OR-boundary interval is approximately .382. Thus stable pooled results do not certify every topic, and more bootstrap draws alone would not add development entities. The pilot is conditional on one fixed proposal and fitted heads, not uncertainty over alternate reservations or refits.

Then record exact P/A/fact/sampler membership, B and fold/pair rules, model/layer/representation and numeric precision, all method/optimizer settings and library versions. Do not tune covariance or choose a favorable layer from this pilot without a separately specified decision. Retain or resolve the producer/tokenizer/config limitations explicitly; fresh extraction remains a separate compatibility decision.

The v4 canonical outputs have been refreshed in their new package. Broader historical layer/method grids, publication figures, selection-dependent comparisons and narratives still await refresh; no broad pipeline was launched. No P10/P20 fits, full 600-candidate bank, compound control, repair study, P/A freeze, model extraction or final-test prediction was performed.

## Validation and reproduction

Independent sklearn AUROCs match all **1,260** points to maximum error **2.220446049250313e-16**. It rechecks all parameter/result hashes, exact exposure/sampler identities, convergence flags, paired intervals and thresholds without refitting or loading activations. Integrated validation passed 153 tests/60 subtests, plus 66 cache/alignment tests, six new streaming cases, four complete synthetic-export cases and two validator-failure cases. The 25-case pilot suite includes exact synthetic TTPD equivalence, cohort-local R0 constants, label-blind inference and export scope checks. Source-projection, earlier audit/projection and preservation validators reproduce their pinned bytes. `git diff --check` passes. Only new task files were added after the reviewed merge; all earlier files are unchanged.

Compact results: `results/t2_capacity_pilot_v1_20261001/`. Full per-fit parameters, exact memberships, eligibility, scores and bootstrap draws: `/Users/apple/projects/t2-capacity-pilot-v1-20261001/`. Additional verified exports: `/Users/apple/projects/t2-capacity-pilot-exports-v2-20261001/`. `external_artifacts.json`, `completion.json` and `binding.json` link every version and hash; incompatible resumes/stale mixtures are rejected. The first aborted export remains in its separate v1 directory.

```sh
/tmp/clean-extraction-venv/bin/python -B scripts/57_validate_capacity_pilot_v1.py \
 --prepared-root /Users/apple/projects/t2-canonical-sensitivity-v4-20261001/prepared \
 --output-root /Users/apple/projects/t2-capacity-pilot-v1-20261001
```

Python 3.11.6, NumPy 2.4.6, SciPy 1.17.1, sklearn 1.9.1, pandas 3.0.6. The historical sklearn penalty deprecation warning is retained rather than changing method semantics. The validator's pandas read-only-mask issue was fixed with a non-mutating intersection; saved fits and scores were unchanged.
