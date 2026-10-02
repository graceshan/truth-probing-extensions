# Nested P10 capacity fallback — completed 2026-10-01

**P10 does not sufficiently remove the material withholding damage. Do not adopt A10 as a successful capacity fix or freeze P/A automatically.** All 36 additional configurations converged and all requested comparisons were computed. Five original triggers clear, five improve but persist, and four worsen and persist. Four new triggers appear versus full. The resulting **13 P10-versus-full** and **11 P10-versus-P15** triggers include severe covariance-MM losses. Stable LR/R0 results do not justify removing methods or choosing favorable layers.

## Integration, scope and retained contracts

Execution remained on Darwin, `MacBook-Pro-7.lan`, in `/Users/apple/projects/truth-probing-t1-score-rebuild`. The clean reviewed starting commit was `de776f1c86e4ddc464f79feac391704cbbaf0dc0`. Origin was fetched and `t2-capacity-pilot-p10-20261001` created from that exact checkpoint; no earlier P10 task/checkpoint existed. No merge/pull from `clean-eval-protocol` or later source correction was performed. Successor configuration `capacity_pilot_p10_v2.json` authorizes only these 36 P10 configurations, while all hash-bound v1 code, configuration, full/P15 outputs and other worktrees remain unchanged.

The retained training/D contract is **v4**, correction `fd2d7be3b910c3fdba7ff6b2c2891eb0ab1b7992`, with `tc_successor_status` and `tc_current_eligible`. The recorded E-only v5 handoff `bb8bc51a864a5702f06ef23f923e11af97710c09` does not change these pinned inputs or relabel these fits. No E activation/score data or E predictions, new model extraction or remote transfer was needed. Recorded representation adoption remains bounded to existing caches; unresolved historical producer-code conflicts and absent standalone tokenizer/config bytes remain. Fresh extraction still requires separate compatibility verification.

## Exact membership and fitting exposure

A10 has **10 entities/topic, 50 total**, with unchanged proposal bytes and exact fact records nested within the unchanged A15 proposal (75 total). Every A10 person/source variant is excluded from P10. P15 is an exact subset of P10. Corrected full/P15/P10 memberships are **3,040 / 2,778 / 2,864** rows. P10 has **1,432 rows per label**, 849 original entity IDs and 848 resolved person keys; the distinction preserves inventor alias handling.

The actual saved seed-0 sampler was recomputed in original source order: **800 rows**, 160/topic, 80/80 labels per topic. It shares only **422** identities with the P15 sample: **278 leave and 378 enter**. Thus its net +100 exposure is not the same as adding 100 rows to P15. Exact ordered identities/hashes and all source-variant checks are in `membership.json`; per-fit exposure is in `configuration_summary.csv` and `fits.json`.

| Topic | P10 rows | Labels 0/1 | Person keys | Balanced rows |
|---|---:|---|---:|---:|
| animal_class | 160 | 80/80 | 42 | 160 |
| cities | 1760 | 880/880 | 440 | 160 |
| element_symb | 188 | 94/94 | 47 | 160 |
| inventors | 352 | 176/176 | 121 | 160 |
| sp_en_trans | 404 | 202/202 | 198 | 160 |

DoM, raw LR, covariance-MM and R0 each fit all 2,864 P10 rows. t_G/TTPD fit exactly the 800 sampled rows, including their own dataset means, projections and polarity/truth heads. Every preprocessing/covariance component is cohort-local; P10 never borrows full/P15 statistics. Original balanced sample counts remain full=1,000 and P15=700.

## All 36 new configurations and convergence

The unchanged grid is Qwen zero-based saved layers **17,18,22** and Llama **10,15,20**, each with raw LR, DoM, covariance-MM, t_G, TTPD and R0. Raw LR uses Qwen C=1 / Llama C=10. The successor calls the hash-bound v1 method implementation unchanged, including the fixed **1e-3 absolute covariance pseudoinverse cutoff**, no shrinkage or retuning, exact t_G/TTPD sampling and optimizer settings, and the verified R0 objective/preprocessing. No method was removed and no layer selected from these results.

There are **36 new configurations and 108 unique pilot configurations cumulatively**. Full/P15 heads were reused, with no refits. All 36 are finite and valid; none required a retry. LR iterations range **28–60**, maximum gradient infinity norm **9.7628518e-5**; R0 **23–39**, maximum **9.3164676e-5**; TTPD components **10–34**, maximum **7.3454531e-5**. DoM, covariance-MM and t_G are finite closed-form fits. Each fit and parameter archive was checkpointed separately, and flagged/failed checkpoints would remain recorded and excluded from scoring.

Run fitting/replay/scoring/evaluation took **201.47 seconds**, with **55.22 seconds** summed new fit time, one process and one BLAS thread. Initial predecessor verification and final receipt audits are outside the timed core. Python 3.11.6, NumPy 2.4.6, SciPy 1.17.1, sklearn 1.9.1 and pandas 3.0.6 match the completed pilot. The inherited sklearn penalty deprecation warnings remain recorded without changing numerical settings.

`configuration_summary.csv` lists every new fit. Complete point estimates and both paired differences are in `complete_comparisons.csv` (1,260 endpoint/scope rows), `metrics.json`/`metrics.csv` (2,520 comparison rows), and the readable `comparison_tables.md` (all five pooled and macro endpoints). Per-topic comparisons are fully retained, not selected for presentation.

## Reuse, D alignment and uncertainty

All locations were resolved through the completed pilot's committed artifact manifests. Before fitting each layer, source/export/header/companion receipts, exact ordered row/layer identities and float64 feature hashes were rechecked. Every one of the 72 saved full/P15 heads reproduced its saved atomic/bare scores **bit for bit**. No transfer or extractor ran.

Both P10 comparisons use the identical retained **1,012 atomic D rows**, **483 complete compound pairs / 7,728 bare rows**, **3,864 rows per operator**, and **2,898 per critical boundary**. Eligibility, labels, topics and order match the completed D files byte for byte. Atomic AUROC, AND/OR AUROCs and both truth-cell boundaries are reported pooled, equal-topic macro and per topic.

All **2,520** comparisons have **2,000 valid paired draws**. The seed-1729 topic-stratified person/entity schedule is unchanged: atomic person multiplicities and compound endpoint-product multiplicities. Every baseline draw vector equals its saved predecessor vector bit for bit; P10's vector is also identical across the two comparisons. These pointwise 95% percentile intervals are conditional on fixed fitted heads. Shared entities, models, methods and endpoints make the comparisons dependent; they are not independent or simultaneous tests. No redraw, imputation, topic dropping or refitting was used.

## All original 14 triggers

Thresholds are unchanged: atomic Δ < −.005 and OR mixed-versus-FF Δ < −.05. “Improved/worsened” describes point movement from P15 to P10; “cleared/persistent” separately describes the P10-versus-full threshold. A crossing does not certify absence of damage or statistical equivalence.

| Original identity | P15−full | P10−full | P10−P15 | Movement / threshold |
|---|---:|---:|---:|---|
| llama L15 mass_mean_covariance or_mixed_vs_ff_auroc topic/element_symb | -0.056617 | -0.030768 | +0.025849 | improved / cleared |
| llama L15 ttpd atomic_auroc topic/inventors | -0.009133 | +0.006460 | +0.015594 | improved / cleared |
| qwen L17 mass_mean_covariance or_mixed_vs_ff_auroc pooled/all | -0.087685 | -0.079995 | +0.007689 | improved / persistent |
| qwen L17 mass_mean_covariance or_mixed_vs_ff_auroc topic/cities | -0.067007 | -0.041357 | +0.025651 | improved / cleared |
| qwen L17 mass_mean_covariance or_mixed_vs_ff_auroc topic/sp_en_trans | -0.116295 | -0.061067 | +0.055228 | improved / persistent |
| qwen L17 mass_mean_covariance or_mixed_vs_ff_auroc topic/element_symb | -0.072145 | -0.038677 | +0.033468 | improved / cleared |
| qwen L17 mass_mean_covariance or_mixed_vs_ff_auroc topic/animal_class | -0.054565 | -0.051880 | +0.002686 | improved / persistent |
| qwen L17 mass_mean_covariance or_mixed_vs_ff_auroc topic_macro/all | -0.050736 | -0.032703 | +0.018034 | improved / cleared |
| qwen L18 mass_mean_covariance or_mixed_vs_ff_auroc pooled/all | -0.167999 | -0.193709 | -0.025710 | worsened / persistent |
| qwen L18 mass_mean_covariance or_mixed_vs_ff_auroc topic/cities | -0.189100 | -0.251374 | -0.062273 | worsened / persistent |
| qwen L18 mass_mean_covariance or_mixed_vs_ff_auroc topic/sp_en_trans | -0.161738 | -0.147599 | +0.014138 | improved / persistent |
| qwen L18 mass_mean_covariance or_mixed_vs_ff_auroc topic/inventors | -0.083975 | -0.097043 | -0.013067 | worsened / persistent |
| qwen L18 mass_mean_covariance or_mixed_vs_ff_auroc topic/element_symb | -0.154900 | -0.105613 | +0.049286 | improved / persistent |
| qwen L18 mass_mean_covariance or_mixed_vs_ff_auroc topic_macro/all | -0.116746 | -0.136537 | -0.019791 | worsened / persistent |

The main pooled covariance-MM OR-boundary concern remains:

| Qwen layer | Full AUROC | P15 AUROC | P10 AUROC | P10−full [95% CI] | P10−P15 [95% CI] |
|---|---:|---:|---:|---|---|
| 17 | 0.711500 | 0.623815 | 0.631505 | -0.079995 [-0.099249, -0.060780] | +0.007689 [-0.002846, +0.019438] |
| 18 | 0.773015 | 0.605017 | 0.579307 | -0.193709 [-0.226297, -0.167913] | -0.025710 [-0.043341, -0.009461] |
| 22 | 0.634228 | 0.707681 | 0.606637 | -0.027591 [-0.052127, -0.004562] | -0.101045 [-0.124359, -0.080207] |

L17 improves slightly but remains beyond the full-comparison threshold; its paired interval is entirely below −.05. L18 worsens further, also with its full-comparison interval entirely below −.05. L22 loses P15's gain: its full comparison stays above the pooled trigger threshold, but its P10-minus-P15 drop exceeds .05. More fitting rows therefore do not produce monotonic gains under the fixed covariance rule. This is an observed result, not proof of a particular covariance mechanism or permission to retune the cutoff.

## New triggers and remaining concerns

Four new P10-versus-full triggers appear beyond the nine persistent originals:

| Model/layer/method/topic | Endpoint | P10−full [95% CI] |
|---|---|---|
| qwen L18 mass_mean_covariance animal_class | or_mixed_vs_ff_auroc | -0.081055 [-0.139463, -0.034807] |
| qwen L22 mass_mean_covariance cities | or_mixed_vs_ff_auroc | -0.059658 [-0.095847, -0.024774] |
| qwen L22 mass_mean_covariance inventors | or_mixed_vs_ff_auroc | -0.066170 [-0.103572, -0.028425] |
| qwen L22 burger_t_g inventors | atomic_auroc | -0.006906 [-0.014692, +0.000595] |

There are also eleven P10-minus-P15 triggers, which answer a different comparison:

| Model/layer/method/scope | Endpoint | P10−P15 [95% CI] |
|---|---|---|
| llama L15 difference_of_means topic/sp_en_trans | atomic_auroc | -0.007506 [-0.014878, -0.002192] |
| llama L20 difference_of_means topic/sp_en_trans | atomic_auroc | -0.009570 [-0.017551, -0.003732] |
| llama L20 ttpd topic/sp_en_trans | atomic_auroc | -0.005630 [-0.012388, -0.001244] |
| qwen L18 mass_mean_covariance topic/cities | or_mixed_vs_ff_auroc | -0.062273 [-0.084991, -0.042632] |
| qwen L18 mass_mean_covariance topic/animal_class | or_mixed_vs_ff_auroc | -0.087036 [-0.147857, -0.035873] |
| qwen L22 mass_mean_covariance pooled/all | or_mixed_vs_ff_auroc | -0.101045 [-0.124359, -0.080207] |
| qwen L22 mass_mean_covariance topic/cities | or_mixed_vs_ff_auroc | -0.141483 [-0.173629, -0.109718] |
| qwen L22 mass_mean_covariance topic/inventors | or_mixed_vs_ff_auroc | -0.064348 [-0.127078, -0.003877] |
| qwen L22 mass_mean_covariance topic/animal_class | or_mixed_vs_ff_auroc | -0.063721 [-0.177620, +0.093222] |
| qwen L22 mass_mean_covariance topic_macro/all | or_mixed_vs_ff_auroc | -0.057180 [-0.093143, -0.016455] |
| qwen L22 ttpd topic/inventors | atomic_auroc | -0.009356 [-0.021625, +0.001008] |

No pooled or macro atomic comparison triggers, and LR/R0 trigger neither threshold at any reported scope. Remaining issues include Qwen covariance-MM across layers/topics, a new inventor t_G atomic loss versus full, and Spanish DoM/TTPD atomic losses versus P15. Some triggered topic intervals include zero or span the trigger threshold. Balanced-method changes reflect 278 removed/378 added sampled rows, not only restoration of A15-minus-A10 source rows; this is not an isolated causal withholding estimate.

## Recommendation and adaptation tradeoff

**Do not adopt A10 on the premise that the fallback solved the capacity issue. Keep both proposals unfrozen and retain the current A15 reference until design review.** P10 clears five original point triggers and improves several endpoints, but the largest material loss worsens, nine original triggers persist and four new ones appear. Reducing a trigger count from 14 to 13 is not sufficient evidence. The favorable LR/R0 results cannot silently remove other planned methods or choose a better-looking layer. No further experiment or retuning is launched here.

P10 restores 86 all-row training examples and increases balanced exposure by 100, but A shrinks from 75 to 50 entities. The possible within-topic unordered pairs fall from 105 to 45; these are dependent candidate pairs, not independent observations.

**If A10 is explicitly adopted later, B=25 remains feasible but uses all A entities.** Under the disjoint-endpoint allocation, five pairs/topic consume all ten A entities/topic: 25 pairs consume all 50 entities. Its three samples vary only in pairing, not entity or affirmative-fact membership. **B=50 must be dropped under A10.** Identical fixed-layer atomic-only controls must be deduplicated by their complete fitting identity—cohort/facts/weights, preprocessing, layer, objective and optimizer settings. Pairing-only sample labels do not create three distinct atomic-only fits or independent evidence. This is a configuration requirement, not an authorization to fit those controls now.

Before design freeze, specify the acceptable method/topic damage and precision criterion, exact A/P/fact and sampler identities, B, pairing/fold rules and control deduplication. Pointwise conditional intervals are not a simultaneous design-acceptance test. Versus full, five per-topic atomic and sixteen OR-boundary intervals span their drop thresholds; versus P15 the corresponding counts are fourteen and twelve. The widest topic OR-boundary interval versus full is about .288. Additional bootstrap draws do not add independent entities. The E-only handoff's 450-pair/7,200-row proposal likewise cannot be treated as 7,200 independent observations; no E precision analysis was run here.

Numeric precision, model/revision, layer indexing, method settings and the bounded representation evidence limitations remain explicit requirements. Any later protocol revision requires its own authorization and provenance; this result does not freeze A/P, authorize P20, the 600-candidate production bank, compound controls, repair fits or E predictions.

## Validation, preservation and reproduction

Independent sklearn AUROCs validate all 2,520 comparison records, with maximum error **2.220446049250313e-16**. The validator reconstructs every interval, checks bitwise baseline draw alignment and shared P10 draws, and independently evaluates **3,780 weighted-AUROC draw checks** (replicates 0,73,1999 across all 1,260 endpoint/scope settings), also with maximum error **2.220446049250313e-16**. It verifies exact memberships/sampler, all 36 configurations, all 14 original triggers, parameter/result hashes and convergence, without loading activations or fitting. The predecessor's full 72-configuration validation also passes.

Focused validation passed **71 tests**, followed by **five independent-validator/trigger-accounting tests**. Synthetic failures check altered counts/identities/settings, reordered sampler/draws, and library success with bad gradients. The v4 source-projection builder reproduced its pinned bytes. `git diff --check` passes, and no file existing at reviewed `de776f1` was modified or deleted. Full/P15 code/config/output hashes remain unchanged.

Compact outputs are in `results/t2_capacity_pilot_p10_v2_20261001/`. All P10 parameter archives, full membership/D metadata, scores and bootstrap arrays remain outside Git in `/Users/apple/projects/t2-capacity-pilot-p10-v2-20261001/`, individually hash-bound in `completion.json`. Existing exports/full/P15 roots remain referenced by their completed manifests. Resume requires exact configuration, source, membership and predecessor hashes; completed outputs refuse overwrite.

```sh
/tmp/clean-extraction-venv/bin/python -B scripts/59_validate_capacity_p10_v2.py \
 --output-root /Users/apple/projects/t2-capacity-pilot-p10-v2-20261001
```
