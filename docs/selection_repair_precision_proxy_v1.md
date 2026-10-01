# Saved-D precision evidence for the research-direction discussion

2026-10-01. This is pre-decision evidence from eight fixed surrogate contrasts, not a direction recommendation or a design freeze. No fitting, extraction, remote transfer, E scoring, membership change or protocol adoption occurred. All saved fits retain their reviewed **v4** provenance; P/A remains unfrozen.

The main OR mixed-versus-FF contrast has native-D 95% interval half-widths of **0.0075–0.0238 pooled**, **0.0125–0.0269 equal-topic macro**, and **0.0097–0.0853 per topic**. These quantify the precision of the specified saved-head contrasts. They do not establish precision or power for either proposed production contrast.

## Prespecified analysis and provenance

- Local execution: macOS 12.6 arm64, `MacBook-Pro-7.lan`, `/Users/apple/projects/truth-probing-t1-score-rebuild`.
- Clean starting HEAD: `20562da8bd238a39478aef908f9339f77f61fe46`; fetched origin; created `t2-design-review-20261001`. No source-branch pull or merge.
- Configuration [precision_proxy_v1.json](../config/clean_protocol/precision_proxy_v1.json) was committed at `cd112e9` before computing contrasts; SHA-256 `a5dfe794d68c2490bfa6fa35d383f98559c4d8cc8e4122e5c0ab7f7b90d21dfb`.
- Fixed saved layers, zero-based: Qwen L17 and Llama L15. Within each of P15 and P10: R0 minus raw LR, and difference-of-means (DoM) minus raw LR. Eight contrasts, using 12 saved score sets; no layer or method selection based on results.
- Both practical margins remain **0.02**: repair-minus-selection and clean-control-minus-selection. Neither production contrast is estimated here; those production fits do not exist.
- E preparation metadata read through `git show` at exact commit `c71b9a16b35f65d6e7f70952ab5f6747b7211399`, without checkout or merge. Only admitted atomic metadata, graph/degree/count metadata and manifests/configuration were read. No E activations, scores, predictions or performance metrics were accessed.

## Statistical procedure

Use the identical retained v4 D rows for every method, model and cohort: 1,012 atomic rows; 483 complete compound pairs / 7,728 dependent binary rows. Main OR mixed-versus-FF uses 2,898 rows (TF/FT positive, FF negative); overall bare AND uses 3,864 rows (TT positive, TF/FT/FF negative). Atomic uses its bound labels. Scores are matched by the exact hash-bound ordered row files; alternate method offsets are not used.

Each of 2,000 paired draws (PCG64, seed 1729) resamples persons within each atomic topic and entities within each compound topic. All variants of an atomic person receive one multiplicity; all rows of an existing compound pair receive the endpoint product `m_i*m_j`. The same schedules and example weights are used across methods, models and cohorts. No missing edges are fabricated. Pooled AUROC includes cross-topic score comparisons; macro is the arithmetic mean of the five within-topic AUROCs. For each draw, subtract the raw-LR AUROC from its matched method AUROC, then take the 2.5% and 97.5% linear percentiles.

Intervals require at least 1,800 valid draws. Undefined draws remain undefined, with no redraw, imputation or dropping of a topic from macro. All 168 native summaries and 120 computed count-proxy summaries have **2,000 valid / 0 undefined draws**. The 48 unsupported projection records are explicitly unattempted, with zero draws, not failed replicates. These are conditional, pointwise intervals, not independent or simultaneous comparisons; they exclude fitting, selection and A-pairing variability.

## Main endpoint: native-D contrasts

All entries below are AUROC differences, method minus raw LR, with paired 95% intervals. P15/P10 here name the already fitted cohort, not a new allocation.

| Cohort | Model | Contrast | Pooled difference [95% interval] | Macro difference [95% interval] |
|---|---|---|---|---|
| P15 | Qwen | R0 − LR | -0.0241 [-0.0335, -0.0151] | -0.0149 [-0.0288, -0.0019] |
| P15 | Qwen | DoM − LR | -0.0147 [-0.0330, +0.0019] | -0.0255 [-0.0505, +0.0012] |
| P15 | Llama | R0 − LR | -0.0395 [-0.0542, -0.0250] | -0.0154 [-0.0294, -0.0013] |
| P15 | Llama | DoM − LR | +0.0226 [+0.0013, +0.0469] | -0.0210 [-0.0425, +0.0008] |
| P10 | Qwen | R0 − LR | -0.0171 [-0.0246, -0.0096] | -0.0089 [-0.0223, +0.0027] |
| P10 | Qwen | DoM − LR | -0.0042 [-0.0244, +0.0142] | -0.0308 [-0.0576, -0.0038] |
| P10 | Llama | R0 − LR | -0.0531 [-0.0668, -0.0405] | -0.0158 [-0.0321, +0.0028] |
| P10 | Llama | DoM − LR | +0.0114 [-0.0103, +0.0373] | -0.0202 [-0.0383, +0.0006] |

Pooled and macro can differ in sign: for example, Llama P15 DoM has a positive pooled difference and a negative macro difference. They are different estimands; pooled results must not substitute for equal-topic retention. These surrogate effects do not select a research direction.

| Endpoint | Pooled half-width range | Macro half-width range | Per-topic half-width range |
|---|---|---|---|
| OR mixed vs FF (main) | 0.0075–0.0238 | 0.0125–0.0269 | 0.0097–0.0853 |
| Overall bare AND | 0.0029–0.0213 | 0.0025–0.0192 | 0.0027–0.0681 |
| Atomic | 0.0000–0.0103 | 0.0000–0.0118 | 0.0000–0.0590 |

Half-width means `(upper−lower)/2`; tables also retain the maximum distance of either bound from the observed contrast, so asymmetry is not concealed. For the main endpoint, half-width is at most 0.02 in 6/8 pooled, 5/8 macro and 10/40 per-topic contrasts. A half-width near or above 0.02 makes an improvement of that scale hard to distinguish from no improvement in a similarly variable contrast. Even a smaller half-width is not a power guarantee: the effect, score covariance, training variability and future graph can differ. The margins have not been changed.

Atomic R0 comparisons often have perfect within-topic rankings and zero observed bootstrap width; the P10 Llama pooled R0 contrast is also zero. This is an empirical ceiling with no observed ranking errors to resample, not evidence that future atomic uncertainty is zero. Tiny count-proxy widths around 1e-15 are numerical roundoff, not meaningful precision.

## Graph support and bounded E-count approximation

The predeclared count-only procedure transports centered paired D deviations: `delta_proxy[b] = delta_D + sqrt(n_D/n_E)*(delta_D[b]−delta_D)`. It preserves the observed D contrast center; it does not predict an E effect. For compounds it is allowed only per topic when both graphs have degree four. This uses an explicitly approximate first-order entity-count variance assumption, conditional on comparable entity/edge effects. It does not verify score-population exchangeability or exact graph topology. All observed pair/surface dependence remains in the D bootstrap.

| Topic | D entities / pairs | D degree / components | E entities / pairs | Compound projection | Main native per-topic half-width range |
|---|---|---|---|---|---|
| Animals | 16 / 32 | 4–4; 1 component(s) | 16 / 32 | width × 1.00000 | 0.0440–0.0809 |
| Cities | 149 / 298 | 4–4; 1 component(s) | 146 / 292 | width × 1.01022 | 0.0097–0.0253 |
| Elements | 18 / 36 | 4–4; 1 component(s) | 18 / 36 | width × 1.00000 | 0.0229–0.0375 |
| Inventors | 33 / 49 | 1–4; 3 component(s) | 11 / 22 | Unsupported | 0.0267–0.0853 |
| Spanish | 34 / 68 | 4–4; 1 component(s) | 34 / 68 | width × 1.00000 | 0.0128–0.0437 |

The 33-entity retained inventor D graph has degrees 1–4 and components of 16, 11 and 6 entities; E has 11 entities in one degree-four component. Count scaling cannot recover the missing pair structure or its score covariance. Inventor compound projection is therefore unsupported. Macro projection cannot drop inventors, and no pooled compound graph-transport estimator was prespecified: **neither compound macro nor pooled E precision is reported**. This is the main missing precision dependency for a complete E projection.

For the four structurally supported topics, the approximation leaves animals/elements/Spanish widths unchanged and increases cities widths by about 1.02%. Their graphs are connected; triangle counts and full degree diagnostics are in `support.json`. Matching these metadata does not independently establish that E score distributions or entity effects will match D.

Across native bootstrap draws, positive-weight existing pair counts ranged: animals 3–28 of 32, cities 86–157 of 298, elements 4–29 of 36, inventors 7–35 of 49, Spanish 12–49 of 68. Every topic retained at least one pair in every draw. Full support diagnostics and schedule hashes are in `bootstrap_binding.json`.

Atomic E counts were derived separately from all **974 admitted rows**, rather than borrowed from compound eligibility:

| Topic | D atomic rows / persons | E atomic rows / persons | E compound entities |
|---|---|---|---|
| Animals | 64 / 16 | 64 / 16 | 16 |
| Cities | 596 / 149 | 590 / 149 | 146 |
| Elements | 72 / 18 | 72 / 18 | 18 |
| Inventors | 134 / 43 | 110 / 43 | 11 |
| Spanish | 146 / 68 | 138 / 68 | 34 |

Both atomic partitions have 294 persons in total, but E has fewer variants for some cities, inventors and Spanish persons. Every admitted atomic person has both labels. The atomic count-only proxy therefore uses scale one within topics; the pooled proxy multiplies each D topic’s row weights by its E/D row-count ratio before computing AUROC, while preserving whole-person D resampling. This approximates E topic composition, not its different within-person variant profiles or new scores. Atomic macro/per-topic widths equal native widths; the eight pooled proxy widths range from numerical zero to 0.01034. Complete approximate intervals are in the result tables, labeled as D proxies rather than E performance.

E’s 450 pairs produce 7,200 binary rows under the degree-four recipe; these are dependent rows, not 7,200 independent observations. This analysis supplies no exact E power calculation and no definitive sufficiency claim.

## What this adds to a research-direction decision

It supplies an auditable scale for paired uncertainty using fixed saved heads and actual D dependence. It shows where pooled estimates look more precise than equal-topic or small-topic estimates, and identifies the inventor graph mismatch that prevents a complete E precision projection. Mac 1 can incorporate this evidence into the broader research packet without treating it as a direction choice.

## What it cannot establish

It does not estimate repair-minus-selection or clean-control-minus-selection, choose P15 versus P10, justify a production method/layer choice, or freeze P/A. It does not include training or selection uncertainty, A sample/pairing variability, new AND wording, unobserved E entity effects or future extraction differences. Native development precision is conditional on a repeatedly used D set, not an untouched confirmatory result. Historical producer-code conflicts and absent standalone tokenizer/model-config bytes remain; historical extraction has not been independently replayed. Future fresh extraction still requires separate compatibility verification.

## What to check after choosing a design

After an explicitly authorized design is chosen, refresh precision using the **actual B=25 D results** and the actual matched production contrasts, preserving common rows and cluster weights. Resolve the intended E graph/estimand, topic aggregation and practical success criteria at the unchanged 0.02 margins; account for remaining small-topic limits and any multiplicity requirements. Check representation/extraction compatibility for any fresh states and finalize memberships only through a separate decision. No such work is authorized by this report.

## Validation and artifacts

Focused synthetic checks cover weighted AUROC ties against a direct pair formula and sklearn, whole-person and endpoint-product resampling, deterministic shared draws, surface duplication invariance, identical-method zero contrasts, undefined-class handling, macro propagation of undefined topics, percentile interpolation, graph support, stale input rejection and the implemented centered projection. The suite passed **11 tests**. A tiny four-entity test fixture initially fell below the production valid-draw threshold; its synthetic-only threshold was corrected. The production policy and outputs were unchanged.

Independent validation checked **576,000 draw values** by trapezoidal ROC integration, all native contrasts against bound saved pilot metric draws, all percentile intervals, exact input/output hashes, row alignment and coverage. Maximum independent AUROC/draw error: `8.44e-15`; archived draw error: `4.44e-16`; interval error: `1.73e-18`. Computation took 24.90 seconds, using the existing numerical environment. No earlier tracked artifact was modified.

- [Complete tables: CSV](../results/t2_precision_proxy_v1_20261001/results.csv) and [JSON](../results/t2_precision_proxy_v1_20261001/results.json): 168 native, 120 approximate and 48 explicitly unsupported projection summaries, including individual D-proxy AUROCs, differences, intervals, half-widths and valid/undefined counts.
- [Input bindings](../results/t2_precision_proxy_v1_20261001/input_bindings.json), [support](../results/t2_precision_proxy_v1_20261001/support.json), [bootstrap receipt](../results/t2_precision_proxy_v1_20261001/bootstrap_binding.json), [completion](../results/t2_precision_proxy_v1_20261001/completion.json), and [independent validation](../results/t2_precision_proxy_v1_20261001/validation.json).
- Large paired-draw archive remains outside Git at `/Users/apple/projects/t2-design-review-precision-v1-20261001/paired_contrast_draws.npz`, bound by completion SHA-256. All historical full/P15/P10 artifacts remain unchanged.

Reproduce into new, non-existing output paths:

```sh
/tmp/clean-extraction-venv/bin/python -B scripts/64_precision_proxy_v1.py \
  --output <new-compact-output-directory> \
  --external-output <new-external-draw-directory>
/tmp/clean-extraction-venv/bin/python -B scripts/65_validate_precision_proxy_v1.py \
  --output <same-new-compact-output-directory>
/tmp/clean-extraction-venv/bin/python -B -m pytest -p no:cacheprovider \
  tests/test_selection_repair_precision_proxy_v1.py -q
```
