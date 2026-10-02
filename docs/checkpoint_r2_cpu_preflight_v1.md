# Revision 2 CPU preparation handoff

This is preparation on **`f1f0872efe5e8ece5911e9df1be377f32a514bb1`**, pending integration with mac 1's exact reviewed checkpoint SHA. The new branch/worktree is `checkpoint-r2-cpu-preflight-20261002`, at `/Users/apple/projects/checkpoint-r2-cpu-preflight-20261002`. No production bank, C_clean, B25 repair, behavior, chat or E evaluation was run. P/A remains unfrozen here. Historical v4 fits have not been relabeled as v5 production fits.

The supplied `Truth_Probing_Next_Checkpoint_Plan_Revision_2.docx` was read in full, including the six-endpoint trap, separate source/full-T_C curves, all original controls, sample-level inference and bridge cap. Its byte hash and location are in the [preparation configuration](../config/clean_protocol/checkpoint_r2_cpu_preflight_v1.json), committed at `55e4893` before benchmarks. No applicable AGENTS.md was found; the repository README and the objective, pilot and uncertainty documentation/code were reviewed. Historical README results are not treated as new checkpoint evidence.

Local baseline: Darwin/macOS 12.6 arm64, `MacBook-Pro-7.lan`; clean `t2-design-review-20261001` at the requested base, synchronized before work. `git fetch origin` completed. A separate worktree was created from exactly that SHA, with no pull or merge of moving tips. The original two worktrees and completed branches/artifacts were not modified. This machine has 32 GiB RAM and 10 logical CPUs; about 556 GiB storage was free. All timing jobs were local, one process and one BLAS thread at a time.

## Implemented selection helpers

[checkpoint_r2_selection_v1.py](../src/checkpoint_r2_selection_v1.py) is a pure numerical/selection module; it performs no file access or fitting.

**S_atom_all.** `AtomicCandidate` accepts only candidate identity, fit validity, pooled atomic AUROC and the five topic atomic AUROCs. `atomic_eligibility` requires a valid fit, defined atomic endpoints and pooled atomic AUROC at least reduced LR minus .005. `select_atom_all` requires exact equality with the caller's S_all retention-eligible candidate IDs; an independently narrowed bank cannot pass. It maximizes pooled atomic AUROC; retains candidates within 1e-12 of that maximum; maximizes the arithmetic mean of the five topic AUROCs within that set; then uses a fresh PCG64(20261005) permutation of sorted final canonical IDs per model. The first permuted ID wins. Eligibility reasons, both tie sets and the entire randomized tie order are returned. Empty banks are visibly unresolved. No compound metric, score, label or AND gate is accepted by the API.

The canonical ID format is `model/P15/Lxx/method`, with `/C=<g-format value>` for raw LR. It is fixed in the 600-row inventory; changing ID spelling can change a seeded tie winner and must be reconciled explicitly in the common checkpoint. `select_original_atomic_r0` is a separate reference using the original maximum pooled atomic AUROC and **exact** tie/lower-layer rule. S_atom_all neither replaces nor mutates it.

**Constituent source-only selection.** `surface_labels` swaps canonical A/B truths under BA order and checks explicit surface labels when present. `constituent_endpoints` computes first and second marginal AUROCs plus first conditional on second=0/1 and second conditional on first=0/1, each within all five topics; each endpoint is then averaged equally over topics. Pooled values are recorded separately. `select_constituent_layer` ranks by the minimum of the six macro endpoints, then their mean within 1e-12 of the best minimum, then lower layer within the second 1e-12 tie set. It returns all source-validation curves, per-topic/per-endpoint values, invalid-layer reasons and tie sets. Undefined topics invalidate the layer; no topic is silently dropped.

The selector requires the `T_C_source_validation` role, a source-validation binding and rows containing only the declared source operator. It accepts no target or D scores. The separate evaluation primitive can later evaluate frozen heads on separately bound target data. Role strings alone cannot certify provenance: audited source-fit/validation membership, alias disjointness, exact pairs and ordered score bindings remain production gates.

Synthetic checks reproduce the Revision 2 trap: `score_1 = truth_1 + 10*truth_2` and its symmetric counterpart give six macro endpoints **[.75, .75, 1, 1, 1, 1]**. The selector prefers the fully competent layer rather than accepting conditional AUROC alone. Tests also cover candidate order invariance, reproducible model-local ties, non-chained tolerances, exact original R0 ties, secondary mean before lower layer, invalid fits, missing topics/classes, source/target separation and BA labels.

## Reusable implementation and input bindings

[reusable_components_and_inputs.json](../results/checkpoint_r2_cpu_preflight_v1_20261002/reusable_components_and_inputs.json) records exact SHA-256/size bindings for eight reusable components and their dependencies. [conditional_reuse.json](../results/checkpoint_r2_cpu_preflight_v1_20261002/conditional_reuse.json) records each potentially reusable fit's membership, original-row, feature, label, source-export and parameter hashes.

| Component | Pinned behavior / exposure | Reuse gate |
|---|---|---|
| Shared objective and preprocessing | Float64 P-only means/population std, floor 1e-6, constant coordinates always zero; R0 full P mean; repair/isolated separate .5/.5 means; .001 weight penalty; unpenalized intercept | Exact P and A identities, compatible states, same code/settings; held-out fold pairs excluded upstream |
| Shared optimizer | Zero initialization, full-batch L-BFGS, initial 2000 iterations, actual gradient infinity norm ≤1e-4; flagged first attempt warm-retries to 10000 total; both attempts retained | No loss, regularization, preprocessing or convergence changes |
| Raw LR | Raw features, five declared Cs; existing sklearn 2000/10000 cold warning-retry policy and independent finite/gradient checks | Exact rows, C, layer, representation and library versions |
| DoM / covariance-MM | All 2778 P15 rows; covariance is within-class, denominator n, absolute pseudoinverse cutoff .001, no shrinkage/retuning | Own fitting cohort only; preserve historical settings despite withholding damage |
| t_G / TTPD | Exact 700-row balanced P15 sampler, 140/topic; dataset-local training centering and raw inference; TTPD's two unregularized LR components and existing retry rules | Exact sampler identities, not merely row count; no evaluation-derived transforms |
| Cache maps | V4 source-row/hash/statement identities to original tensor indices; canonical and additional pilot layer exports | Physical identity already verified; fresh compatibility still requires mac 1's bridge |
| Bootstrap | Seed1729, 2000 topic-stratified paired draws; compound endpoint products and atomic person multiplicities; 1800-valid minimum | Identical retained examples, score order, weights and draw IDs across compared procedures |
| Historical saved fits/scores | Full/P15/P10 and precision lineage preserved, no new research metrics | Only 36 P15 configurations potentially match the new bank; no current production adoption |

The exact objective implementation remains unchanged. It is reusable as the reviewed numerical core; production C_clean/constituent assembly still needs a separately validated mean-block adapter using fixed P15 statistics, rather than fitting preprocessing on T_C. No such production fitter is claimed complete here.

Locally verified exports cover Qwen **17,18,22** and Llama **10,15,20**. Twelve additional-layer train/atomic-D/bare-D export files were rechecked by hash and NPY header only; no new D scores were calculated. Canonical training exports were additionally materialized for **P15 only** and matched exactly to the saved P15 feature/label hashes before timing. All original source indices and receipt chains were checked; alternate export offsets were not substituted for tensor indices.

The pinned base lacks accepted checkpoint packages for the complete 28/32-layer bank, allocated B25 pairs/facts/folds, mixed T_C and source-only constituent splits, every wording condition and Prompt1 chat inputs. This means “not accepted and bound on this preparation base,” not a claim that no historical overlapping state could exist elsewhere. No large files were fetched.

## Candidate and execution inventory

[bank_candidates.csv](../results/checkpoint_r2_cpu_preflight_v1_20261002/bank_candidates.csv) enumerates all **600 unique IDs**: 28 Qwen layers ×10=280; 32 Llama layers ×10=320. Every layer contains five raw-LR Cs (.001,.01,.1,1,10), DoM, covariance-MM, t_G, TTPD and R0. All layers are zero-based; weak methods remain included. All 600 are currently marked unrun/pending integration.

There are **36 conditional historical P15 matches**: six methods at each of three layers per model. Raw LR matches only Qwen C=1 and Llama C=10. Their saved fit/parameter files were hash-verified and are enumerated individually. If all membership, representation and settings gates pass, 564 bank configurations need new fitting. Otherwise the affected configurations must be rebuilt. No count-only compatibility claim is made.

The other 72 completed pilot configurations are excluded: 36 full-training/A-exposed diagnostics and 36 P10 fits with the wrong cohort. The two reused full canonical LR references were already included in the completed full count. The six timing fits in this task saved no parameters and create **zero new reusable candidates**.

| Family | Planned unique configurations / heads | Current status |
|---|---:|---|
| P15 bank | 600; at most 36 conditional reuses | Exact grid/fit inventory prepared; all-layer runner and gates pending |
| B25 repair folds | **900 = 60 layers ×3 samples ×5 folds** | All 900 identities enumerated; no fits |
| R_all / fixed / S_all-layer repair refits | At most 18 (6 per role) | Deduplicate equal layer + exact sample/objective/representation identities |
| Fixed / R_all-layer atomic-only adaptation | At most 12 | Same-layer identical fact/head controls deduplicate |
| C_clean | 60 | No viability fit run; required adapter/data/gates pending |
| Procedure-layer C_clean | **0 extra fits** | Reference existing per-model/layer C_clean at S_all, R_all, S_atom_all layers |
| Constituent source-fitted heads | 240 =60 layers ×2 operators ×2 heads | 16 entities/topic; source-validation selector implemented, fitting pending |
| Selected/matched full-T_C constituent refits | At most 16 heads | 20 entities/topic; deduplicate coincident selected layers |
| Prompt1 chat R0/C_clean | At most 8 | Future CPU inventory only; no execution |

Raw core upper inventory is **1846** configurations/heads, or **1810 new** if all 36 bank reuses are accepted. With up to eight chat heads this is 1854 total / 1818 new. These count candidate/head configurations, not optimizer attempts; TTPD internally includes two LR fits, and retries are separately recorded. C_clean uses 8000 mixed rows; each constituent source/refit operator uses 4000 rows, with different entity/pair populations. The five-fold repair fitting block has 20 pairs /320 rows, followed by 25-pair /400-row full-allocation refits.

[execution_inventory.json](../results/checkpoint_r2_cpu_preflight_v1_20261002/execution_inventory.json) separates planned, implemented, executable and missing components. The original S_all/R_all full eligibility/fallback orchestration, entity-disjoint fold assembly and bank fitting are **not implemented as production runners on this preparation base**. New pure selectors and statistical primitives do not establish those guarantees.

## Result schemas and inference

[result_schema.json](../results/checkpoint_r2_cpu_preflight_v1_20261002/result_schema.json) specifies all required panels/files and metadata fields. Explicit [unrun templates](../results/checkpoint_r2_cpu_preflight_v1_20261002/unrun_result_templates.csv) cover recoverability, original and added controls, each seed and its metric mean, the source/target transfer matrix and separate all-layer curves. Null metrics carry an unrun reason; they are not omitted results.

Each eventual record binds correction, representation, fit/exposure, model, layer, seed, evaluation rows/weights, selection and draw identities. Source-validation and source-fitted all-layer D curves are labeled **16/topic** and stay separate from selected/matched **20/topic full-T_C refits**. Six marginal/conditional values remain visible; no endpoint average can replace that vector. Schema obligations also cover frozen decisions, joint correctness, min/max/Boolean composition and matched-layer target-minus-source transfer losses.

All original controls remain, alongside S_atom_all and procedure-layer C_clean. S_atom_all is selected once per model and referenced across seeds. R_all-layer controls disclose compound-informed selection. D-selected C_clean and the bank oracle remain optimistic development references. Pooled, equal-topic and per-topic results, retention, all wording conditions, failures and exclusions are mandatory.

`average_sample_metrics` averages the three sample-specific metric values **inside each shared bootstrap draw**, preserving undefined values. It never pools head scores. A synthetic check demonstrates that pooled scores from individually perfect but offset heads can give a different result. Full paired B25 inference, cross-model repair effects and actual B25 precision refresh remain future integration work. Pointwise intervals condition on observed fits; three samples do not cover all training uncertainty.

## Measured CPU diagnostics and cost scenario

The frozen runner executes small synthetic cases first, then full-width synthetic timing cases, then at most six verified historical P15-only timing fits. All **18/18** completed validly. Total wall time including fresh process startup and input verification was **84.12 seconds**. Existing Python3.11.6 / NumPy2.4.6 / SciPy1.17.1 / sklearn1.9.1 were used, with one BLAS thread and one process. Deprecation warnings from unchanged historical sklearn settings were preserved; they were not convergence failures.

Representative matrices have 2778 rows and width 3584 (Qwen) or 4096 (Llama), materialized float64. Synthetic compound-objective timing adds 320 independent synthetic rows, using unchanged half-block loss and P-only preprocessing. These are numerical timing fixtures, **not B25 research fits**. Historical fits use only verified v4 P15 rows, canonical layers and pinned C; no D/E metric was computed. Parameters were discarded.

| Input / method | Qwen L17 seconds | Llama L15 seconds |
|---|---:|---:|
| Historical P15 raw LR (C=1 /10) | .208 | .140 |
| Historical P15 R0 | .431 | .638 |
| Historical P15 covariance-MM | 6.797 | 10.115 |
| Synthetic raw LR | .055 | .077 |
| Synthetic R0 | .413 | .506 |
| Synthetic covariance-MM | 17.920 | 28.807 |
| Synthetic compound objective (2778+320 rows) | .292 | .405 |

Historical R0 converged in 21 / 28 iterations, final gradient infinity norms 9.89e-5 / 9.41e-5. Historical raw LR gradients were 8.78e-5 / 6.30e-5. Synthetic compound timing converged in 15 / 17 iterations, gradients 7.49e-5 / 6.91e-5. All iterative jobs met the unchanged finite/gradient criteria; no retry was needed. Covariance fits are closed-form with the fixed absolute cutoff. Complete attempts, preprocessing, dimensions, thread libraries, warnings and input hashes are in [timing_summary.json](../results/checkpoint_r2_cpu_preflight_v1_20261002/timing_summary.json) and [timings.csv](../results/checkpoint_r2_cpu_preflight_v1_20261002/timings.csv).

Largest observed fresh-process peak RSS was **966.5 MiB** (historical Llama covariance), including imports, input validation, float64 arrays and workspaces. RSS uses Darwin's byte-valued `ru_maxrss`; it is process peak, not an isolated allocation delta. Stream layers and checkpoint each fit. An 8-GiB provisional per-worker ceiling for larger blocks is a planning budget, not a measured production peak; keep one worker until actual T_C workloads are checked.

[cpu_cost_scenarios.csv](../results/checkpoint_r2_cpu_preflight_v1_20261002/cpu_cost_scenarios.csv) accounts for all 1810 possible new raw-core fits, using **fresh representative measurements** rather than extrapolating only the earlier 36-fit pilot. The conditional serial scenario is **19.5–171.8 fitting minutes**. The bank's high scenario is 3× the larger current synthetic/historical rate; prior DoM/t_G/TTPD timings supplement methods not rebenchmarked. Non-bank fits use shape scaling and a 1×–10× conditioning/iteration sensitivity factor; actual T_C and B25 blocks have not been fitted. Eight optional chat CPU heads add a small shape-scaled allowance, without assuming any chat states are available.

An explicitly unmeasured **30–120 minute** reserve for I/O, scoring and 2000-draw table assembly gives approximately **0.83–4.89 hours** for the CPU packet including that optional chat-fit allowance. These are bounded planning scenarios, **not measured production runtime, statistical confidence bounds or a guarantee**. They assume reuse gates pass, comparable numerical software/hardware, layer streaming and no severe optimizer tail; GPU extraction/behavior likelihood/generation and queue/download time are excluded. The 10000-iteration cap can greatly exceed this range. Actual all-C/layer conditioning, T_C viability, repair folds and inference costs must be checked at the early review; exceeding the scenario calls for a cost/scope review, not a relaxed convergence rule or silent fit removal.

## Required mac 1 handoff before execution

1. Supply the **exact reviewed common checkpoint SHA** and ancestry, including Revision 2 configuration/amendments and the frozen candidate ID/tie convention. This preparation branch has not integrated a mac1 tip.
2. Verify v5 identity equality for train/D/A15/T_C against reviewed v4 inputs, not just 3040/1012/75/100 counts. Bind P15's 2778 rows, exact 700-row sampler and all A source-variant exclusions. Record the A15 method-specific withholding-damage scope amendment and explicit membership freeze separately.
3. Supply the independent representation bridge receipt: exact model/tokenizer revision, renderer/special-token policy, semantic position, padding/masks, zero-based layer indices, dtype/hook/device/software settings, token/position/dimension checks, per-layer state/score errors and batching invariance. Freeze tolerances before historical comparison. Respect the 2-hour active / 1-GPU-hour-per-model cap. Missing historical producer execution and standalone tokenizer/config evidence remain limitations; file hashes alone do not remove them.
4. Supply compatible, hash-bound all-layer exports and exact source-row/layer mappings. If the bridge cannot support reuse, record a coherent regeneration plan before any mixing of historical P and fresh compound states.
5. Supply B25 seeds 11/23/37 exact pairs/facts/folds and the T_C source split: four A15 validation persons/topic using PCG64(20261003), sixteen fitting persons/topic, 100/120 sampled fitting pairs/topic, all six validation pairs/topic, fixed full-T_C 100 pairs/topic. Enforce alias/person disjointness and source-only selection. No target/D tuning.
6. Separately authorize production execution after the common base is reviewed. Initial C_clean viability remains **unrun** in this preparation, as explicitly required. Behavior/chat and E remain outside current execution scope.

A passed bridge only licenses its verified contract. Old within-cache fits retain their original provenance even if reused by reference. The .02 practical margins, .005 atomic retention and .02 AND retention are unchanged; this preparation makes no research-direction decision.

## Validation and rerun commands

Run from the isolated preparation worktree using the existing environment:

```sh
/tmp/clean-extraction-venv/bin/python -B -m pytest -p no:cacheprovider \
  tests/test_checkpoint_r2_selection_v1.py \
  tests/test_selection_repair_objectives.py \
  tests/test_selection_repair_capacity_pilot_v1.py -q
/tmp/clean-extraction-venv/bin/python -B scripts/67_checkpoint_r2_preparation_checks.py \
  --output results/checkpoint_r2_cpu_preflight_v1_20261002 \
  --timings /Users/apple/projects/checkpoint-r2-cpu-timings-v1-20261002
git diff --check
```

The focused suite passed **62 tests**; two existing sklearn deprecation warnings remain. Inventory checks verified all 600 candidate IDs, 900 fold identities, 36 saved parameter/fit bindings, all 18 timing checkpoints, one-thread execution and pinned historical source files. All production statuses remain unrun. A first test invocation used a nonexistent historical test filename and ran no tests; the corrected command above passed. No historical implementation or result was edited.

To rebuild inventories or rerun the same bounded diagnostics, use **new output paths** (existing outputs are refused):

```sh
/tmp/clean-extraction-venv/bin/python -B scripts/66_checkpoint_r2_cpu_preflight.py inventory \
  --output /tmp/new-r2-inventory
/tmp/clean-extraction-venv/bin/python -B scripts/66_checkpoint_r2_cpu_preflight.py benchmark \
  --output /Users/apple/projects/new-r2-timing-directory
/tmp/clean-extraction-venv/bin/python -B scripts/67_checkpoint_r2_preparation_checks.py \
  --output /tmp/new-r2-inventory --timings /Users/apple/projects/new-r2-timing-directory
```

The fresh-inventory path rebuilds the grid/reuse/fold inventory and schemas. The checked-in reusable-component receipt additionally pins the already verified export headers and historical inputs; those bindings are rechecked when validating the shipped output directory. Neither command starts the production bank or any real compound fit.
