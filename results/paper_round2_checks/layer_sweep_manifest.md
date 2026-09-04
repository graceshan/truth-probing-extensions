# Qwen2.5 all-layer sweep manifest

## Sources

- Result table: `results/r1/mechanism/all_layers_atomic_transfer.csv`
- Summary: `results/r1/mechanism/all_layers_atomic_transfer_summary.json`
- Source script: `scripts/mechanism/12_all_layers_atomic_transfer.py`
- Atomic gate metrics: `results/qwen_union_probe_gate/per_layer_metrics.csv`
- Existing atomic-only gate figure copied from: `figures/qwen_union_probe_gate_layer_auroc.png`

The existing PNG shows the atomic gate AUROCs. The cached all-layer compound-transfer table
has no separate pre-existing compound-sweep PNG in the repository, so no extraction or
model computation was rerun.

## Available columns

`layer`, `atomic_heldout_auroc`, `and_truth_auroc`, `and_TT_vs_mixed_auroc`, `and_mixed_vs_FF_auroc`, `and_TT_vs_FF_auroc`, `and_TF_vs_FT_auroc`, `and_dprime_TT_mixed`, `and_dprime_mixed_FF`, `and_mu_TT`, `and_mu_mixed`, `and_mu_FF`, `and_upper_gap`, `and_lower_gap`, `and_total_range`, `and_upper_gap_fraction`, `and_lower_gap_fraction`, `and_Delta_norm`, `and_pooled_sd`, `and_range_over_sd`, `or_truth_auroc`, `or_TT_vs_mixed_auroc`, `or_mixed_vs_FF_auroc`, `or_TT_vs_FF_auroc`, `or_TF_vs_FT_auroc`, `or_dprime_TT_mixed`, `or_dprime_mixed_FF`, `or_mu_TT`, `or_mu_mixed`, `or_mu_FF`, `or_upper_gap`, `or_lower_gap`, `or_total_range`, `or_upper_gap_fraction`, `or_lower_gap_fraction`, `or_Delta_norm`, `or_pooled_sd`, `or_range_over_sd`, `and_minus_or_auroc_gap`, `OR_to_AND_range_ratio`, `OR_range_compression`, `OR_to_AND_pooled_sd_ratio`, `OR_to_AND_range_over_sd_ratio`

## Verified results

- Atomic-AUROC >= 0.99 plateau: layers `[9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27]`; this is exactly layers 9–27: **True**.
- AND > OR transfer AUROC over all layers: **25/28**.
- AND > OR over the >=0.99 plateau: **19/19**.
- OR TT–mixed > OR mixed–FF over all layers: **25/28**.
- OR TT–mixed > OR mixed–FF over the >=0.99 plateau: **19/19**.
- Plateau median AND transfer AUROC: **0.910433333333**.
- Plateau median OR transfer AUROC: **0.704000000000**.
- Layer 22 AND transfer AUROC: **0.945866666667**.
- Layer 22 OR transfer AUROC: **0.679083333333**.
- Layer 22 OR TT–mixed AUROC: **0.833600000000**.
- Layer 22 OR mixed–FF AUROC: **0.580275000000**.
