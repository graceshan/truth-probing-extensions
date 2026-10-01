# Research discussion evidence index — 1 October 2026

Companion to the [discussion brief](research_discussion_brief_20261001.md). This is documentation and existing-result reconciliation, not a new scientific analysis or an endorsed research direction.

## Snapshot, status and reading rules

- **C (mac 1):** `c71b9a16b35f65d6e7f70952ab5f6747b7211399`, the exact branch base. All links below use this immutable Git snapshot unless marked P10.
- **P (mac 2, read only):** `20562da8bd238a39478aef908f9339f77f61fe46`. P10 reports, configurations and aggregates were read with `git show P:path`. No P10 files were merged or copied into this branch.
- **Packet branch:** `t2-research-discussion-packet-20261001`, created from C after confirming Darwin, repository `/Users/apple/projects/truth-probing-extensions`, starting branch `clean-eval-protocol`, exact HEAD and the known untracked-file list. Origin was fetched; no pull or merge was performed. No other worktree was changed.
- **Completed finding** means a saved, provenance-bound result exists. **Prepared data** means factual/data readiness, not model performance. **Proposed experiment** means its scientific comparison has not run. **Unresolved** is an evidence or design question, not a negative finding.
- Narrative recommendations are historical recommendations. They do not constitute an approved allocation. Current fitting evidence stays attributed to v4; v5 changes E only. Current E status uses `e_successor_status` / `e_current_eligible`; prior columns are historical. D/train fitting uses the reviewed v4 contract.

## Claim-to-evidence map

| Brief claim/table | Evidence below | Status |
| --- | --- | --- |
| Historical benchmark and score reproducibility | R1–R2 | Completed recovery/reproduction; upstream limitations retained |
| Original-label/alias concerns and conservative corrections | R3–R4 | Completed bounded audits; not exhaustive truth certification |
| Historical versus corrected canonical AUROC table; wording/minmax findings | R2, R5 | Completed development results with distinct coverage |
| Full/P15/P10 exposure and method-specific losses | R6–R7 | Completed bounded pilots, no allocation pass |
| Selection-versus-repair and matched control objectives | R8 | Proposed production comparison; numerical components validated |
| T_C capacity | R9 | Completed fact pairs; control experiment unrun |
| E audit and prepared benchmark | R10 | Prepared facts/data, no E performance |
| Held-out AND/OR wording | R11 | Prepared data, not a tested transfer result |
| Broader conclusions awaiting correction | R12 | Explicit refresh gaps |
| Saved-score precision proxy | R13 | Pending separate mac 2 handoff |

<a id="r1"></a>
## R1. Historical benchmark recovery and reproduction

**Claim:** the recovered producer and audited-negative history reproduce the original D benchmark; recovery did not validate every factual label anew.

- Report: [T1 benchmark recovery](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/t1_benchmark_recovery.md).
- Committed artifacts: [benchmark metadata](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/development_validation_v1/metadata.json), [original compound table](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/development_validation_v1/development_validation_compounds.csv), [original pair table](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/development_validation_v1/development_validation_pairs.csv).
- Exact configuration: [development benchmark config](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/entity_disjoint_development_validation_v1.json); [generalization protocol and final-scoring guard](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/generalization_protocols.json).
- Producer: [recovered validation generator](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/src/validation_compound_benchmark.py). Negative evidence: [final historical reviewed registry](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/validated_negatives/v1/validation_reviewed_batch_001_inventor_manual_external_review_v5/registry.csv) and [registry provenance](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/validated_negatives/v1/validation_reviewed_batch_001_inventor_manual_external_review_v5/metadata.json).

The recorded scratch reproduction matched all benchmark output bytes: **262 entities, 524 pairs, 8,384 rows**, degree four and all 16 variants/pair. The compound CSV SHA256 is `96e96515e780086065694155e3061bdfe6f4bb80af062a63c2deecd6bfad6f94`. The recovery report locates external backup/reproduction receipts; it is not a new model-extraction replay. Registry `source_supported_false` and externally reviewed judgments are different evidence tiers.

<a id="r2"></a>
## R2. Rebuilt historical development results: pre-cleanup only

- Report: [historical rebuild](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/t1_score_rebuild.md).
- Quoted values: [committed main results](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t1_development_rebuild_20261001/main_results.csv).
- Reproduction, discrepancy and external-output receipts: [structured summary](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t1_development_rebuild_20261001/summary.json) and [full summary narrative](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t1_development_rebuild_20261001/summary.md).
- Frozen configurations: [Qwen LR transfer](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/pinned_qwen25_lr_transfer_v1.json), [Qwen method transfer](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/pinned_qwen25_method_transfer_v1.json), [input controls](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/priority2_input_controls_v1.json), [Llama replication](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/llama31_transfer_v1.json).

**Table lookup:** for the brief's historical rows, select `family=qwen_lr` with empty condition, or `family=llama, condition_id=raw_reference`; then `metric=and_auroc`, `or_auroc`, `or_mixed_vs_ff_auroc`, using `estimate`. The historical method-layer result mentioned qualitatively is explicitly indexed by `analysis_group=faithful_method_selected_layer` in the source table; do not transfer that layer selection into a corrected production bank.

The saved reconstruction reports **3,990 metric rows and 7,980,000 numeric bootstrap values** matching exactly. These are checks of existing saved results; this packet did not recompute scores or bootstrap draws. `summary.json` binds the full external `metrics.csv`, verification and diagnostic files by SHA256. Those larger files are not confused with the committed `main_results.csv` subset.

**Limits:** historical atomic per-example validation scores were absent from that recovery package; its atomic AUROCs are verified saved aggregates, not a row-score/CI reconstruction. Activation tensors were excluded from the score-recovery package, and some standalone Llama tokenizer/config bytes were unavailable. This does not mean no usable cache exists elsewhere: later bounded physical/cache checks are separate evidence (R4–R5). Producer-code conflicts, particularly historical pinned Qwen and Llama code, remain unresolved. Score replay is not independent verification of which extraction code ran.

<a id="r3"></a>
## R3. Source-label and alias evidence

- Initial identity/source audit: [T2A report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/selection_repair_t2a_source_audit.md), [row manifest](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/source_audit_v1/row_manifest.csv), [audit manifest](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/source_audit_v1/audit_manifest.json), [audit config](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/selection_repair_v1/source_audit.json), [identity evidence](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/selection_repair_v1/identity_evidence.json).
- Fixed inventor sample: [T2B report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/source_review_v1/review_report.md), [all review records](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/source_review_v1/reviews.json), [evidence](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/source_review_v1/evidence.json), [frozen review policy](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/source_review_v1/review_policy.md), [package receipt](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/source_review_v1/package_manifest.json).
- Bounded follow-up: [T2C report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/selection_repair_t2c_candidate_overlay.md), [v1 policy](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/candidate_overlay_v1/policy.json), [v1 receipt](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/candidate_overlay_v1/candidate_receipt.json).
- Corroboration/closure: [closure report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/selection_repair_source_closure_v1.md), [closure policy](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/source_closure_v1/policy.json), [closure receipt](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/source_closure_v1/closure_receipt.json), [closure recommendation](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/source_closure_v1/recommendation.json).

The fixed **30** inventor claims yielded **4 supported false, 2 supported true, 24 unresolved**. Read `reviews.json:judgment`; do not use only the resolved subset as a denominator or infer an error rate from the targeted follow-up. Confirmed sample conflicts included Nansen/UK and Fleming/France. The later seven-claim closure corroborated five label conflicts while retaining Walton/US and Bengio/US as unresolved. All original judgments and source labels remain historical; conservative restrictions change admission rather than silently flipping labels.

Identity evidence links Simjian aliases across train/D and quarantines their source rows; Farnsworth's train aliases retain their original entity IDs with a shared person key. This is reviewed alias handling, not exhaustive entity resolution. Residence judgments require residential evidence rather than nationality or isolated visits. Spanish fact-pair audits check both translations for the same word, polysemy, dialect, spelling/diacritics and direction; negation supplies no second independent affirmative fact (R9–R10). New factual re-audit was outside this packet.

<a id="r4"></a>
## R4. Correction lineage and bounded representation adoption

| Base | What changed / current interpretation | Exact artifact and policy |
| --- | --- | --- |
| v2 | Conservative closure; corroborated conflicts excluded, unresolved claims quarantined; no original label edits | [v2 row manifest](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/candidate_overlay_v2/row_manifest.csv), [closure policy](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/source_closure_v1/policy.json) |
| v3 | Previously unresolved Saffir/UK and Mège-Mouriès/US source claims propagated to paired quarantine | [projection report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/capacity_pilot_correction_projection.md), [v3 correction receipt](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/candidate_overlay_v3/correction_receipt.json) |
| v4 | Giffard/US and Slavyanov/Russia source pairs quarantined during T_C top-up; no new confirmed error | [top-up report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/tc_topup_fact_audit.md), [v4 correction receipt](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/candidate_overlay_v4/correction_receipt.json), [v4 source/fact manifest](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/candidate_overlay_v4/fact_inventory.csv) |
| v5 | E-only quarantine of unresolved exact claims and paired negations; non-E input identities unchanged | [E audit report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/final_partition_fact_audit.md), [v5 receipt](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/candidate_overlay_v5/correction_receipt.json), [semantic comparison to v4](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/final_partition_projection_v1/semantic_comparison_to_v4.json) |

The current admitted source counts are **train 3,040; D 1,012; E 974**. Check v5 `correction_receipt.json:admitted_by_split` and the exact projected memberships rather than summing overlapping person/label categories. v5's E restrictions do not relabel the v4 fits as newly performed under v5.

Representation evidence: [cache preflight](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/t2_cache_preflight.md), [physical follow-up compatibility](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_cache_physical_followup_20261001T183553Z/compatibility.json), [session verification](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_cache_physical_followup_20261001T183553Z/session_validation.json), [mapping report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_protocol_integration_20261001/candidate_maps_v1/mapping_report.json), [v4 adoption contract](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/selection_repair_adoption_v4.json). This supports the explicitly bounded existing-cache analyses, not unrestricted fresh extraction or an independent replay of all historical producers. Exact source/sidecar/model/layer identities matter; equal totals alone are insufficient.

<a id="r5"></a>
## R5. Corrected canonical v4 results and uncertainty

- Report: [canonical v4 refresh](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/selection_repair_canonical_sensitivity_v4.md).
- Numerical artifacts: [metrics CSV](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_canonical_sensitivity_v4_20261001/metrics.csv), [metrics JSON](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_canonical_sensitivity_v4_20261001/metrics.json).
- Exact configuration/adoption: [canonical v4 config](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/canonical_sensitivity_v4.json), [v4 adoption](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/selection_repair_adoption_v4.json).
- Validation/provenance: [independent validation](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_canonical_sensitivity_v4_20261001/independent_validation.json), [historical replay](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_canonical_sensitivity_v4_20261001/historical_replay.json), [input inventory](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_canonical_sensitivity_v4_20261001/input_inventory.json), [refresh decisions](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_canonical_sensitivity_v4_20261001/refresh_decisions.json), [completion receipt](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_canonical_sensitivity_v4_20261001/sensitivity_receipt.json).

**Brief lookup:** `scope=pooled, topic=all`; `condition=raw_reference`, `atomic_D`, `or_explicit_or_both_v1`, or `isolated_external_minmax_v1`; read `corrected_retained_auroc` for the corresponding `model` and `metric`. The canonical heads are Qwen saved L17/C=1 and Llama saved L15/C=10, full corrected training. They are A-exposed diagnostics.

The exact old-head/new-head distinction matters. Raw OR pooled AUROC illustrates it:

| Model | Historical full coverage | Historical retained | Corrected retained | Training Δ | Coverage Δ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 0.813332 | 0.835143 | 0.852700 | +0.017556 | +0.021811 |
| Llama | 0.763750 | 0.768046 | 0.760361 | −0.007685 | +0.004296 |

Columns are the identically named fields in `metrics.csv` (`historical_full_auroc`, `historical_retained_auroc`, `corrected_retained_auroc`, `training_delta`, `coverage_delta`), independently rounded. This is why subtracting the brief's two evidence stages is not a matched training-effect estimate.

The corrected evaluation retains **1,012 atomic D rows; 483 complete compound pairs; 7,728 bare rows; 3,864/operator; 2,898/critical boundary**. All **154** canonical summaries use the declared **2,000** seed-1729 paired person/entity draws. Compound weights use endpoint multiplicity products. Reported 95% percentile intervals are pointwise and conditional on fixed heads; they do not estimate retraining, selector or reservation uncertainty. “Critical boundary” excludes the easy extra truth cell: AND TT versus TF/FT; OR TF/FT versus FF. Pooled, equal-topic macro and topic results are all retained.

<a id="r6"></a>
## R6. Full/P15 pilot: scope, losses and exposure

- Report: [full/P15 pilot](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/selection_repair_capacity_pilot_v1.md).
- Configuration: [capacity pilot v1](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/capacity_pilot_v1.json).
- Numbers: [all endpoint summaries](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_capacity_pilot_v1_20261001/metrics.csv), [per-configuration exposure](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_capacity_pilot_v1_20261001/configuration_summary.csv), [triggers](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_capacity_pilot_v1_20261001/threshold_triggers.json).
- Completion/validation: [completion](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_capacity_pilot_v1_20261001/completion.json), [independent validation](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_capacity_pilot_v1_20261001/independent_validation.json), [validation and preservation](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_capacity_pilot_v1_20261001/validation_and_preservation.json), [historical recommendation](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_capacity_pilot_v1_20261001/recommendation.json).
- Exposure identities: [corrected full membership](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/capacity_pilot_projection_v3/corrected_outer_training.csv), [full balanced sample](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/capacity_pilot_projection_v3/corrected_outer_training_balanced.csv), [P15 membership](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/capacity_pilot_projection_v3/P15_admitted_membership.csv), [P15 balanced sample](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/capacity_pilot_projection_v3/P15_balanced_exposure.csv).

**72 configurations** = both full/P15 cohorts × two models × three layers/model × six methods; all valid, no failed/flagged fits. Qwen layers are 17/18/22, Llama 10/15/20, zero-based saved indices. LR/DoM/covariance-MM/R0 use all cohort rows; t_G/TTPD use their recomputed balanced cohort rows and cohort-local fitted components. Covariance-MM keeps its fixed absolute pseudoinverse cutoff; no retrospective tuning or method deletion is justified by these results.

The **14** triggers are **13 OR-boundary** plus **one atomic** result. Filter `triggered=True`; the atomic case is Llama L15 TTPD, inventors, P15−full **−0.009133**. The brief's covariance table can be read from this CSV's `full_auroc`/`P15_auroc` or the joined P10 artifact (R7). These pilot values are method-specific, not the canonical LR values in R5.

**Sampler reconciliation:** computing set intersections of the committed source-row ID lists gives full→P15 **426 shared, 574 leaving, 274 entering**. P15 has fewer rows, but its balanced sample is not a strict subset of the full balanced sample. The earlier v3→v4 correction also changed **42 outgoing/42 incoming** P15 sampled IDs and **36/36** P10 sampled IDs despite unchanged sample totals; see [version-change identity lists](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/capacity_pilot_projection_v3/balanced_membership_changes.json) and [canonical sampler bindings](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_canonical_sensitivity_v4_20261001/sampler_bindings.json). These are different contrasts from P15→P10.

<a id="r7"></a>
## R7. Completed nested P10 pilot—exact mac 2 commit, not merged

Every link in this section is pinned to **P**, not a moving branch:

- [P10 report](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/docs/selection_repair_capacity_p10_v2.md); [P10 successor configuration](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/config/clean_protocol/capacity_pilot_p10_v2.json).
- [Complete joined full/P15/P10 comparisons](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/results/t2_capacity_pilot_p10_v2_20261001/complete_comparisons.csv), [all comparison metrics](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/results/t2_capacity_pilot_p10_v2_20261001/metrics.csv), [readable comparison tables](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/results/t2_capacity_pilot_p10_v2_20261001/comparison_tables.md), [configuration exposure](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/results/t2_capacity_pilot_p10_v2_20261001/configuration_summary.csv).
- [Trigger follow-up](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/results/t2_capacity_pilot_p10_v2_20261001/trigger_followup.json), [exact membership and sampler](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/results/t2_capacity_pilot_p10_v2_20261001/membership.json), [recommendation and tradeoffs](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/results/t2_capacity_pilot_p10_v2_20261001/recommendation.json).
- [Completion and external artifact hashes](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/results/t2_capacity_pilot_p10_v2_20261001/completion.json), [independent validation](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/results/t2_capacity_pilot_p10_v2_20261001/independent_validation.json), [predecessor replay](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/results/t2_capacity_pilot_p10_v2_20261001/predecessor_replay.json), [preservation](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/results/t2_capacity_pilot_p10_v2_20261001/validation_and_preservation.json).

**36 additional P10 configurations**, **108 unique cumulatively**, all converged. Full/P15 heads were reused, not refit. The joined table contains **1,260** endpoint/scope settings; the two comparisons give **2,520** rows in `metrics.csv`. The saved validation records baseline score/draw agreement with predecessors; this packet reads existing aggregate artifacts and does not reopen prediction arrays.

**Brief table lookup:** in `complete_comparisons.csv`, filter `model=qwen, method=mass_mean_covariance, metric=or_mixed_vs_ff_auroc, scope=pooled, topic=all` and layers 17/18/22. Read `full_auroc`, `P15_auroc`, `P10_auroc`; the L18 interval comes from `P10_minus_full_ci_low/high`. Each rounded value in the brief was checked against these raw fields, not copied solely from prose.

Trigger arithmetic is verifiable from the joined table: **14** original; **5** cleared, **5** improved but persisted, **4** worsened and persisted; **4** new P10-versus-full triggers yield **13** total; **11** trigger versus P15. The two comparisons answer different questions. Neither LR nor R0 triggers at any scope, which is not equivalence or approval for excluding other methods.

Other method-specific examples, kept visible rather than attributing everything to covariance-MM:

| Method/model/layer | Endpoint, topic | Comparison | Δ AUROC |
| --- | --- | --- | ---: |
| Llama TTPD L15 | Atomic, inventors | P15−full | −0.009133 |
| Qwen t_G L22 | Atomic, inventors | P10−full | −0.006906 |
| Llama DoM L15 | Atomic, Spanish | P10−P15 | −0.007506 |
| Llama DoM L20 | Atomic, Spanish | P10−P15 | −0.009570 |
| Llama TTPD L20 | Atomic, Spanish | P10−P15 | −0.005630 |

Source: joined CSV, `metric=atomic_auroc, scope=topic`, methods `ttpd`, `burger_t_g`, `difference_of_means`. See full intervals there; some topic intervals include zero or span the review threshold. No pooled/macro atomic trigger establishes that every topic is stable.

P10 is an all-row superset of P15, but the balanced samples share only **422** IDs: **278 leave, 378 enter**, producing **800** versus **700** rows. `recommendation.json:balanced_P10_vs_P15` agrees with direct set comparison of the committed projection CSVs. **A10=50**, **A15=75**, with A10 nested; possible unordered pairs/topic are **45** and **105** respectively. These are combinatorial candidates, not independent observations. The proposals themselves are [A10](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/capacity_pilot_projection_v3/A10_proposal.json) and [A15](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/capacity_pilot_projection_v3/A15_proposal.json) at C, unchanged through the P10 contract.

The earlier recommendation to retain A15 as a reference is explicitly historical, not an approved preference in this packet. Neither gate passed. Under the proposed disjoint-endpoint B=25 allocation, A10 uses all its entities, so repeated samples change pairings only. Identical atomic-only controls need deduplication by exact fitting identity. B=50 needs 20 A entities/topic under that rule; neither A10 nor A15 supplies that allocation. The completed T_C target does not enlarge A.

<a id="r8"></a>
## R8. Proposed selection-versus-repair experiment

- Mathematical specification and numerical evidence: [objective report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/selection_repair_objectives.md), [implementation](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/src/selection_repair_objectives.py), [synthetic tests](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/tests/test_selection_repair_objectives.py).
- Role/allocation history: [source-closure role correction](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/selection_repair_source_closure_v1.md), [prospective A fact audit](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/capacity_pilot_fact_audit.md), [frozen audit policy](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/capacity_pilot_audit_v1/policy.json), [capacity config and prohibitions](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/capacity_pilot_v1.json).

The scientific proposal contrasts selecting a readout from the planned atomic bank using adaptation evidence with changing a shared readout through compound supervision. A matched atomic-only adaptation control asks whether any gain requires compound inputs rather than additional supervised facts. This describes the question; it does not select the selector, endpoint, folds, allocation or budget.

The committed numerical objectives are R0 `L(P)+0.001||w||²`; compound repair `0.5L(P)+0.5L(A_compound)+0.001||w||²`; atomic-only adaptation replaces the compound block with the same allocated isolated affirmative facts. Preprocessing is P-only; the intercept is unpenalized; operators share one head. The numerical module has synthetic validation. R0's bounded pilot execution does not imply that the compound/atomic adaptation comparison ran.

**Missing production evidence:** no completed full 600-candidate production bank, selector-versus-repair result, production T_C comparison, or frozen end-to-end allocation/CV specification is supplied at these checkpoints. The pilot references the intended 600-candidate bank but is not an executable specification or a census of it. A complete standalone October plan is not included here; the Section 9 quotation was supplied explicitly in the wording task. Do not invent unrecorded plan details.

<a id="r9"></a>
## R9. T_C factual capacity—prepared control pool

- [Top-up report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/tc_topup_fact_audit.md), [completed T_C manifest](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/tc_topup_audit_v1/combined_TC_completed_pairs.json), [capacity table](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/tc_topup_audit_v1/capacity_counts.csv), [recommendation/status](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/tc_topup_audit_v1/recommendation.json).
- Policy/provenance: [pre-judgment lock](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/tc_topup_audit_v1/pre_judgment_lock.json), [traversal receipt](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/tc_topup_audit_v1/traversal_receipt.json), [validation receipt](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/tc_topup_audit_v1/validation_receipt.json).

**20 completed keys/topic, 100 total; 190 possible unordered pairs/topic**, before a **planned 100/topic control cap**. Cumulative evidence accounting is **111 attempts, 100 completed, 11 unresolved**; no successful top-up remains needed for that target. T_C may overlap P and A, but not D/E. Extra T_C keys do not enlarge A or authorize B=50. This is evidence capacity, not a fitted compound-control result.

<a id="r10"></a>
## R10. E factual audit and production data preparation

- [E audit report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/final_partition_fact_audit.md), [capacity counts](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/final_partition_fact_audit_v1/capacity_counts.csv), [completed exact fact pairs](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/final_partition_fact_audit_v1/completed_E_fact_pairs.json), [all reviews](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/final_partition_fact_audit_v1/reviews.json), [unresolved cases](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/final_partition_fact_audit_v1/unresolved_cases.json).
- Audit policy/configuration: [pre-judgment lock](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/final_partition_fact_audit_v1/pre_judgment_lock.json), [recommendation/status](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/final_partition_fact_audit_v1/recommendation.json), [v5 correction receipt](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/selection_repair_v1/candidate_overlay_v5/correction_receipt.json).
- E data preparation: [E-v1 report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/final_e_preparation.md), [preparation config](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/final_e_preparation_v1.json), [E-v1 manifest](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v1/manifest.json), [E-v1 validation](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v1/validation_results.json). Current wording successor is R11; E-v1's old “undefined AND” note is superseded.

| Topic | Completed E entities | Degree-four pairs | Bare rows | Unresolved audited entities |
| --- | ---: | ---: | ---: | ---: |
| Animals | 16 | 32 | 512 | 0 |
| Cities | 146 | 292 | 4,672 | 3 |
| Elements | 18 | 36 | 576 | 0 |
| Inventors | 11 | 22 | 352 | 32 |
| Spanish | 34 | 68 | 1,088 | 0 |
| **Total** | **225** | **450** | **7,200** | **35** |

Source: `capacity_counts.csv` (`usable_entities`, `recipe_pairs`, `expected_binary_rows`, `unresolved_entities`). The inventory originally contained **296** entities: **36** initially excluded, **260** attempted, **225** completed and **35** unresolved. Negative attempts total **261**, including a rejected true candidate; the frozen replacement/stopping rule and all attempts persist. v5 quarantines **14 source claims / 28 paired rows**, all E, without establishing new confirmed label errors. E's **974** admitted atomic rows form a distinct inventory and do not imply 974 independently audited compound facts.

The degree-four recipe is now data, not merely projected capacity. No E model performance or reasoning result follows from it. Cities' dominance and the much smaller inventor/animal/element pools matter for pooled-versus-macro interpretations and precision.

<a id="r11"></a>
## R11. Completed held-out wording coverage and extraction inventories

- [Section 9 correction/report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/heldout_and_wording.md), [exact AND specification](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/heldout_and_wording_v1.json), [D/E preparation configuration](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/heldout_and_preparation_v1.json), [pure renderer](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/src/heldout_and_wording.py).
- E current package: [manifest](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v2/manifest.json), [counts](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v2/counts.csv), [condition index](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v2/condition_index.csv), [extraction bindings](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v2/extraction_bindings.csv), [validation](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v2/validation_results.json).
- D exact cohort: [manifest](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/development_heldout_and_v1/manifest.json), [counts](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/compounds/entity_disjoint/development_heldout_and_v1/counts.csv), [authoritative metadata recovery receipt](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/heldout_and_inputs_v1/recovery_receipt.json), [v4 pair eligibility](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/data/clean_protocol/heldout_and_inputs_v1/development_pair_eligibility.csv).

The plan specified `"Both of the following are true: " + first + " " + second`; it is implemented with complete original sentences and AB/BA surface order. The distinct `and_both_following_v1` condition is held out from compound fitting and selection. It changes AND and reuses bare OR. The established OR conditions change OR and reuse bare AND. No joint condition or extra fact is silently introduced.

| Prepared data | D | E |
| --- | ---: | ---: |
| Exact retained pairs | 483 | 450 |
| New AND wording rows | 3,864 | 3,600 |
| Logical extraction bindings | 3,864 (new AND only) | 19,424 (complete E inventory) |
| Unique texts | 3,864 | 19,190 |

E contains **7,200 bare rows**, **3,600 rows for each established OR wording**, **3,600 AND wording rows**, **450 isolated facts**, and **974 atomic rows**. The **234** repeated-text bindings are reversible; source identity is not replaced by text identity. These totals are data sizes, not independent sample sizes. E-v1 and all existing bare/OR examples remain unchanged. D wording uses the exact current cohort; its pairs/constituents were not redrawn.

No authoritative D metadata dependency is missing. Actual fresh extraction still needs a separately authorized entry point, pinned model/tokenizer/representation and compatibility verification. No extraction or scoring was launched by preparation; `final_evaluation_enabled=false`, P/A unfrozen. No performance on the new AND condition is supplied.

<a id="r12"></a>
## R12. What has and has not received a full post-cleanup refresh

The authoritative refresh scope is [v4 report](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/selection_repair_canonical_sensitivity_v4.md) and [refresh-decision artifact](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/results/t2_canonical_sensitivity_v4_20261001/refresh_decisions.json), under [canonical config](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/config/clean_protocol/canonical_sensitivity_v4.json).

| Analysis | Current evidence status |
| --- | --- |
| Canonical raw LR, raw AND/OR, critical boundaries, “or both”, isolated min/max | Refreshed in the new v4 canonical package, on matched current D |
| Six-method selected pilot layers, full/P15/P10 | Completed bounded pilots; not a refresh of every layer or the production bank |
| Original full layer/method grids and selection-dependent method comparisons | Not fully refreshed; historical selected-layer gains cannot be promoted as corrected results |
| Historical “at least one”, juxtaposition and broader surface/canonical diagnostics | Historical results remain; no full corrected refresh established by the bounded canonical package |
| Published transfer tables/figures, cross-model rankings and narrative conclusions | Need reconciliation with corrected baselines/coverage; some canonical contrasts updated, not all downstream analyses |
| Compound repair versus selection and matched atomic-only controls | Production scientific comparison not run |
| T_C experiment; E performance; new held-out AND performance | Facts/data prepared only; no supplied performance finding |

Interpret the older full/P15 report's “no P10 fits” as true **at that checkpoint**; R7 supersedes that work-status statement. Similarly, the older recommendation to investigate P10 has been answered by the completed P10 pilot, not converted into an approved A10 reservation. The v5-only handoff's prospective E counts have become prepared data, still not evaluated results. Historical files remain unchanged so these temporal distinctions remain reviewable.

<a id="r13"></a>
## R13. Pending precision handoff—no results yet in this packet

**Owner/status:** mac 2 is preparing a saved-score precision proxy, per the current task instruction. No proxy artifact/commit was supplied for this packet; no later mac 2 work was inspected. There is deliberately no numeric proxy result, adequacy verdict, power estimate or inferred recommendation here.

The question: under a declared saved-score/cohort proxy and the planned topic/entity/pair geometry, how precisely can relevant effects or acceptable losses be distinguished? Relevant existing context is [E dependence/precision handoff](https://github.com/graceshan/truth-probing-extensions/blob/c71b9a16b35f65d6e7f70952ab5f6747b7211399/docs/selection_repair_v5_e_only_handoff_20261001.md), [P10 precision caveats](https://github.com/graceshan/truth-probing-extensions/blob/20562da8bd238a39478aef908f9339f77f61fe46/docs/selection_repair_capacity_p10_v2.md), and the current D/E manifests in R11. Those are motivation/input provenance, **not proxy results**.

The incoming report should identify exact saved-score and corrected-cohort hashes; whether it proxies the E topic sizes, D comparisons, or another declared target; how shared endpoints/truth variants and topic weighting are handled; fixed-head versus refitting/selection uncertainty; and a predeclared precision/effect criterion. A saved-score proxy may inform design feasibility, but it cannot measure actual E performance or prove that an intervention/repair will improve it. Additional bootstrap draws do not add entities.

## Discussion alternatives and evidential boundaries

The brief's alternatives are proposed research questions, not empirically ranked interventions. Selection/repair is motivated by R5/R8 but has no completed comparison. Wording/geometry is motivated by R2/R5 and prepared R11 data. Broader replication addresses R12's missing refresh. Behavioral/causal reasoning work would require evidence not present in these readout studies. Measurement/evidence work responds to R3/R7/R10/R13. No option is approved, fitted or implemented by this packet.

## Reconciliation and preservation record

Numeric transcription was checked against structured committed artifacts: the historical `main_results.csv`; canonical v4 `metrics.csv`; full/P15 `metrics.csv`; P10 `complete_comparisons.csv` and recommendation at P; exact membership/sampler lists; T_C/E recommendation/count manifests; and D/E wording manifests. AUROCs in the brief are rounded to six decimal places. Set-overlap counts in R6/R7 are documented arithmetic on existing source-row identities, not new fitted results. All immutable links were resolved with Git object lookup; P10 provenance never uses a moving branch URL. No raw activation or prediction array was read, no numerical model was fit, and no factual source was re-adjudicated.

The packet changes only its two Markdown documents. The following **13** known unrelated untracked files were recorded by SHA256 before writing and verified unchanged before delivery. Their content was not used as scientific evidence for this packet.

| File | SHA256 |
| --- | --- |
| `config/clean_protocol/compound_transfer.json` | `90b30b7ee2e319489676ada9060872e723c400ae9a90c53244a12ee41c14cb41` |
| `docs/clean_compound_transfer.md` | `d90475248fe32c65f3ae84499167c26edf8fb6a86e3331dd9eba9810c106f6c4` |
| `paper_figure_results.zip` | `c28df097e798d245ff22bfa84cff6e18bccd934c8e4be3c730e49982e1f4ea9b` |
| `paper_round2_checks.zip` | `add535f691f0028fc23ac131ddce9ad1be7f50207c2c829b4ba8ea7e9b54681a` |
| `scripts/32_evaluate_clean_compound_transfer.py` | `11b2badf6b601d34547128df40b00dcb01c6f0dc14aa32159887c344cc1f8ef1` |
| `src/compound_transfer.py` | `7a051cc394cfd4898606e88d5fc81442045b917c17701fef0f33e99b55259fea` |
| `src/transfer_bootstrap.py` | `8cdabdfea19db1db3813a0895df41fee0219a90af168920928740f4dc019e42c` |
| `src/transfer_figures.py` | `e53f3a917be48fc95d91ececc5cfd4279ea2a2a568a789a65e8d7fde4293450e` |
| `src/transfer_frozen.py` | `04f80109a55f7bf44ab85c192065d18071f560dfce41d854dadd5e2f79209e1e` |
| `src/transfer_integrity.py` | `80eda3ac4026e0a730e29259986b7787d5f814bc854cb2d297d07ba497c8c7d6` |
| `src/transfer_metrics.py` | `2556203dedad4c78cd4aebb4c73e3e7feded5e22892cae63d92780bc3c060155` |
| `tests/test_compound_transfer.py` | `7afb2724af1df73cc59e5b62cd1ca3bea7597e326f672d63a590e5fbd419212c` |
| `tests/transfer_fixtures.py` | `48f679c24eb6932d2170aa3d32d071c7981e68bc2709ab3c71b178d64bfd0318` |

Open evidence items are explicit rather than silently filled: the pending precision proxy; unresolved historical extraction-producer/tokenizer/config provenance; facts that remain unresolved or only source-label-supported; full post-cleanup historical-grid/selection refresh; the unexecuted production selection/repair/control comparison; and the unrun E/held-out-AND evaluation. Full score/parameter/draw archives are external and hash-bound by the linked completion receipts; this documentation review does not claim to revalidate those binary archives or rerun their analyses.
