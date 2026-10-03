# October 3 Revision 2: bounded saved-score diagnosis and input coverage

This task starts at exactly `0a256d96b0fbf41f7ce678693da15cf230685778`, after fetching origin. That delivery contains the completed B25 successor `1187140c7f79b0b64439eff382d420f51d974a25`, B25 producer `5e1bf47a457558cbc0735bd8f9f13ca029cbf3a4`, and constituent producer `2d28c0b2733a4433aa6ceb4de020bfb815c2f7db`.

The approved October 3 document is the copy marked Revision 2: `Truth_Probing_Next_Decision_Checkpoint_2026-10-03 (1).docx`, SHA-256 `f81c2209c03eacf5c6fe5e14729d68069035ea3a08de3d2b9eb273b070c4cae2`. Its text is preserved in `approved_october3_revision2.txt`. The unsuffixed October 3 file and older October 2 Revision 2 are different documents.

## Frozen scope and calculation

Only saved D scores enter the scientific analysis. Committed delivery manifests resolve exact external runs. Hashes, model/snapshot, saved layer, sample/head, text/source/person/fact bindings, canonical truth, surface truth and AB/BA order are checked before joining. Accepted fresh activation files are rehashed as opaque bytes for representation identity; their tensors are never loaded or analyzed. Historical tensors stay unadmitted.

For each R_all/S_all sample and each existing OR wording, the 25 cells compare mixed TF/FT positives from topic i to FF negatives from topic j. Pair wins plus half ties form cell AUROC. Cell weight is positive topic mass times negative topic mass divided by total positive times negative mass. Diagonal/off-diagonal contributions and masses, their normalized AUROCs, pooled AUROC and fixed five-topic macro are reported separately. Means across samples are means of metrics, never concatenated scores. Wording-minus-bare and bare R_all-minus-S_all differences are paired at the metric/draw level.

The exact existing 2,000 entity bootstrap schedules are verified across panels, regenerated independently, and mapped to wording by matched endpoints. Weights are recalculated in every draw. Undefined zero-mass cells contribute zero; a zero total class mass makes pooled AUROC undefined. Any undefined diagonal makes the fixed five-topic macro undefined. No redrawing, imputation or survivor-topic means. All intervals are pointwise 95% percentile intervals conditional on the existing fitted heads and support; intervals require at least 1,800 defined draws. Undefined counts are retained.

Matched wording tables preserve topic, canonical truth cell, surface order, label, other-constituent truth and exact head. Mean/median/population SD/IQR/range describe deltas; average tied-rank changes, Spearman, Kendall and sign changes describe ordering. Plots are a fixed bounded set: both R_all mean matrices and matched R_all seed-11 scores, plus selected constituent or-both scores. Seed 11 is a plotting example, not a headline winner; all three samples remain tabulated.

Constituent endpoint AUROCs and uncertainty are reconstructed on identical support, with source-only selected full-TC heads in both transfer directions and same-source comparisons. Confusion/joint correctness/Boolean tables retain TT/TF/FT/FF and AB/BA denominators. Decisions use only logit >= 0, ties True. Balanced accuracy is undefined without both classes. Min and max ranking use the same rows. Positive scaling preserves each head's ranking and signs, but relative scales can change min/max ranking. Conditional deltas alone do not establish interference: a positive affine change can cause them while preserving ordering.

The coverage inventory verifies the exact admitted full 3,040 and reduced 2,778 bindings, the 262 additions, and compatible reuse through reviewed identity/text/token/representation metadata. Similar statements cannot substitute. A missing-input queue retains complete source rows, exact text, both model contracts, required numerical conventions and hash anchors. Missing token IDs are left pending for mac 1's separately authorized extraction preparation. No inference is authorized here.

The deterministic balanced sampler is regenerated with original topic order and RandomState(0), checked against the reviewed full-training projection, and recorded as exactly 1,000 rows. Future preprocessing must be cohort-local to all 3,040 admitted rows for standardized methods, retaining existing method objectives and raw LR C scaling. Full training introduces A15 identity exposure; D stays disjoint. The configuration preserves all later required contrasts, retention, bank references, per-topic and wording reporting. Full-bank fitting is deferred.

## Reproduction

Use the producer SHA reported in the delivery, on macOS, with the verified external artifacts available at the manifest locations. The analysis requires the existing `/private/tmp/clean-extraction-venv/bin/python` environment (resolved Python 3.11.6); numerical packages and threadpools are recorded in execution receipts.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
python -B -m unittest tests.test_checkpoint_next_diagnosis -v
python -B -m src.checkpoint_next_score_diagnosis --expected-producer PRODUCER_SHA --output /absolute/fresh/output/directory
python -B -m scripts.checkpoint_next_audit_scores --run /absolute/fresh/output/directory
```

Substitute the exact interpreter above for `python`. Run the analysis and independent audit sequentially, one process/one numerical thread. The output directory must be new. No probe fitting, calibration, threshold search, label/membership changes, new facts, SSH/GPU, behavior/chat or E predictions. Original reports and arrays remain unchanged.

The independent audit uses separate sklearn pair-cell AUROCs, confusion matrices, balanced accuracy and min/max point metrics, plus a distinct stable rank-group pooled/macro bootstrap calculation and independent sampler reconstruction. It is an independent calculation on the same physical machine. Large bootstrap arrays and row score tables remain outside Git; compact reports, tables, manifest and queue are delivered on this branch only.
