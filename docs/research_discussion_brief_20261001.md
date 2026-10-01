# Research discussion brief — 1 October 2026

**Purpose:** decide together which scientific question is most informative next. This packet recommends no direction, allocation or protocol freeze. Evidence is pinned to mac 1 `c71b9a1` and the completed mac 2 P10 checkpoint `20562da`; the latter was read without merging. The [evidence index](research_discussion_evidence_index_20261001.md) separates completed results, prepared data, proposals and unresolved evidence.

## What question are we trying to answer?

An atomic truth probe can discriminate true from false individual statements very well, yet transfer unevenly to conjunctions and disjunctions. Is the limiting factor the choice of an existing readout, the need to adapt that readout to compound inputs, wording and position sensitivity, or the benchmark and measurement design? Distinguishing these explanations is valuable before treating probe transfer as a claim about a model's reasoning. Here P is prospective probe-fitting training data; A is a proposed withheld adaptation/selection set; D is development; E is final evaluation; T_C is the separate compound-control pool. P/A remains unfrozen. [R5–R10](research_discussion_evidence_index_20261001.md#r5)

## What the completed results establish

The historical development benchmark and saved-score analyses were recovered reproducibly. That establishes reproducibility of the recorded data/results, not correctness of every fact or an independent replay of model extraction. Later source and alias cleanup changed both training membership and evaluation eligibility. Historical results must therefore remain labeled **pre-cleanup**, rather than being silently reused as corrected conclusions. [R1–R4](research_discussion_evidence_index_20261001.md#r1)

**Canonical raw-LR pooled AUROC** (OR boundary = TF/FT versus FF):

| Evidence stage | Model | AND | OR | OR boundary |
| --- | --- | ---: | ---: | ---: |
| Historical, original D coverage | Qwen | 0.902752 | 0.813332 | 0.746819 |
| Corrected v4, retained D | Qwen | 0.935923 | 0.852700 | 0.785275 |
| Historical, original D coverage | Llama | 0.985295 | 0.763750 | 0.660797 |
| Corrected v4, retained D | Llama | 0.973523 | 0.760361 | 0.650065 |

Sources: [R2 and R5](research_discussion_evidence_index_20261001.md#r2). The stage-to-stage differences combine training cleanup and changed coverage; they are **not** matched training effects. The v4 artifacts separately compare historical and corrected heads on identical retained examples. Pooled and equal-topic macro results can move differently.

Corrected atomic D AUROC is 0.999836 for Qwen and 1.000000 for Llama. Nevertheless, compound OR remains weaker than AND. Explicit “or both” wording lowers pooled OR AUROC to 0.771456 and 0.675119 respectively. External composition of isolated scores gives Qwen AND/OR AUROCs 0.999806/0.999817 and Llama 1.000000/1.000000. This demonstrates highly usable constituent information **when decomposition and min/max composition are supplied externally**; it does not show that the model internally applies Boolean rules to compound sentences. AUROC is a ranking measure, not proof of calibration or correct generated answers. [R5](research_discussion_evidence_index_20261001.md#r5)

These findings support a transfer/representation problem for the tested readouts. They do not establish the absence of a better existing direction, prove that adaptation will help, or isolate a causal reasoning mechanism. Historical selected-layer method gains are suggestive, but their full post-cleanup comparison is still missing. [R2, R12](research_discussion_evidence_index_20261001.md#r12)

## How much does withholding A change the starting point?

The bounded pilots cover six methods at three fixed layers per model: raw LR, difference of means (DoM), covariance-adjusted mass mean (covariance-MM), t_G, TTPD and R0. Full/P15 contributed 72 configurations; P10 added 36, reusing the completed full/P15 heads. This is **not the full planned 600-candidate bank**, and its losses cannot establish failure of that entire bank. Full fits are A-exposed diagnostics, not eligible primary P-only bank members. [R6–R7](research_discussion_evidence_index_20261001.md#r6)

| Cohort / prospective A | All-row fitting exposure | t_G/TTPD balanced exposure | A entities | Possible A pairs/topic |
| --- | ---: | ---: | ---: | ---: |
| Full diagnostic / no withholding | 3,040 | 1,000 | — | — |
| P15 / A15 | 2,778 | 700 | 75 | 105 |
| P10 / nested A10 | 2,864 | 800 | 50 | 45 |

All methods used identical retained D coverage. The review thresholds were strict drops greater than .005 atomic AUROC or .05 OR-boundary AUROC, evaluated separately at pooled, macro and topic scopes. P15 produced 14 triggers. P10 cleared five original triggers, improved five without clearing them, and worsened four; four new triggers gave 13 versus full. There were also 11 P10-versus-P15 triggers. Trigger counts are dependent review flags, not a capacity-gate pass rate. Neither proposal has passed the capacity gate. [R6–R7](research_discussion_evidence_index_20261001.md#r7)

**Illustrative method-specific issue: Qwen covariance-MM, pooled OR-boundary AUROC.**

| Saved layer | Full | P15 | P10 |
| --- | ---: | ---: | ---: |
| 17 | 0.711500 | 0.623815 | 0.631505 |
| 18 | 0.773015 | 0.605017 | 0.579307 |
| 22 | 0.634228 | 0.707681 | 0.606637 |

At L18, P10−full is −0.193709, with paired 95% interval [−0.226297, −0.167913]. LR/R0 triggered neither threshold at any reported scope. Other concerns include Llama TTPD's inventor atomic loss for P15, Qwen t_G's inventor loss for P10 versus full, and Spanish DoM/TTPD losses for P10 versus P15. These are method/topic-specific results; favorable LR/R0 behavior neither cancels other losses nor authorizes dropping methods. [R6–R7](research_discussion_evidence_index_20261001.md#r7)

A10 restores 86 all-row training examples, but its balanced sample is not P15 plus 100 rows: 422 identities are shared, 278 leave and 378 enter. Thus balanced-method differences mix withholding with resampling. Intervals are conditional on fixed fitted heads and pointwise, not simultaneous, selection-adjusted or uncertainty over alternative reservations. [R7](research_discussion_evidence_index_20261001.md#r7)

**A15 and A10 remain genuine alternatives.** A15 retains more adaptation entities and pairing/subsample diversity, at greater withholding cost. A10 reduces that cost but does not reliably remove the observed damage. Under the proposed disjoint-endpoint allocation, B=25 uses all A10 entities; pairing-only repeats do not create independent atomic-only controls. B=50 is unavailable under A10 and would require a larger A than A15 too. The earlier preference for A15 is historical advice, not approval; retaining either proposal as a reference is not adoption. [R7–R8](research_discussion_evidence_index_20261001.md#r8)

## Evidence quality and data readiness

Cleanup preserved original labels and partitions, quarantined unresolved exact claims and paired negations, excluded corroborated conflicts, and handled reviewed aliases at person level. The fixed inventor sample had 4 supported-false, 2 supported-true and 24 unresolved outcomes; the unresolved cases cannot be counted as correct negatives. Simjian's cross-partition aliases remain quarantined; Farnsworth's train aliases share a person key. Residence/polysemy/geography judgments remain bounded, defeasible evidence reviews—not exhaustive human certification. Corrected D also retains some source-label-supported facts. [R3–R4](research_discussion_evidence_index_20261001.md#r3)

T_C has 20 completed entities/topic, allowing 190 candidate unordered pairs/topic before the planned control cap; it may overlap P/A and excludes D/E. E has 225 completed entities: animals 16, cities 146, elements 18, inventors 11, Spanish 34. Its 450 pairs/7,200 bare rows are prepared, not evaluated. The audit retains 35 unresolved entities and all earlier attempts. Cities dominate pooled coverage; small-topic precision remains a concern. Neither binary rows nor overlapping pairs are independent observations. [R9–R10](research_discussion_evidence_index_20261001.md#r9)

The Section 9 AND wording was specified in the plan and is now implemented, correcting the earlier gap report. The held-out renderer has 3,864 D and 3,600 E rows. Existing bare and OR wording remain unchanged; wording data preparation is not a performance finding. E's v5 restrictions do not change v4 train/D/A/T_C or sampler inputs. [R10–R11](research_discussion_evidence_index_20261001.md#r11)

## Candidate directions for discussion—not decisions

| Direction | Scientific value | Limitations and additional work |
| --- | --- | --- |
| Selection versus repair | Test whether compound supervision chooses an already useful readout or improves a shared linear head beyond selection. | The production comparison has not run. Resolve allocation/precision, bank specification, selection/folds and matched atomic-only controls; prevent adaptation/evaluation leakage. |
| Wording and readout robustness | Test whether apparent logical transfer depends on surface form, position, method or covariance estimation. | Requires a prospectively specified comparison and compatible fresh extraction where needed; it diagnoses readout behavior, not causal reasoning. |
| Broader corrected replication | Establish which historical layer/method/model conclusions survive cleanup and whether pilot failures generalize. | Full grids and selection-dependent conclusions remain unrefreshed; more comparisons require multiplicity and selection discipline. |
| Behavioral or causal reasoning study | Connect internal readouts to model decisions, or test mechanisms with controlled interventions. | Requires a new behavioral/causal design; successful decoding alone supplies no such evidence. |
| Measurement and evidence study | Determine what the present topics and factual support can distinguish before increasing experimental complexity. | Precision and evidence restrictions may limit generality; further work would need separately agreed scope rather than score-driven fact replacement. |

The proposed repair objective combines P and A-compound losses; the atomic-only control uses the same allocated facts, with P-only preprocessing and matched weighting. Numerical objectives have synthetic validation, and R0 appears in the bounded pilot. That is not an executed selector-versus-repair experiment, compound-control fit, or production-bank result. [R8, R12](research_discussion_evidence_index_20261001.md#r8)

## Pending precision handoff and collaborator questions

**Pending: mac 2's saved-score precision proxy. No result is included or inferred.** It should clarify whether a declared effect or acceptable loss can be resolved at relevant topic/pair sizes using saved-score variability and shared-entity dependence. It cannot by itself establish actual E performance or uncertainty from refitting/selection. The handoff needs exact score/cohort provenance, graph/size assumptions, endpoints, interval/detectability criterion and limitations. This packet does not wait for it. [R13](research_discussion_evidence_index_20261001.md#r13)

Questions to settle together:

- Is the main claim about readout transfer, supervision needed for adaptation, robustness, or model reasoning—and does selection versus repair best distinguish the live hypotheses?
- What method/topic damage and precision would be acceptable, rather than declaring success from pooled stability or trigger counts?
- What scientific tradeoff justifies A15, A10, another explicitly designed allocation, or no allocation decision yet?
- Which historical findings need correction before motivating a new experiment, and which evidence/representation limitations are acceptable?
- If proceeding with selection/repair, how will matched controls, pairing-only duplicates, selection bias and untouched evaluation be handled?
- What precision-proxy outcome would change the proposed question or design, and what would remain unknowable without new evidence?
