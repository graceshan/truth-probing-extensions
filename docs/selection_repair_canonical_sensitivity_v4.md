# V4 bounded canonical development refresh

The user adopted exactly `fd2d7be3b910c3fdba7ff6b2c2891eb0ab1b7992` for this within-cache refresh and the separate full/P15 capacity pilot. `tc_successor_status` controls source admission and `tc_current_eligible` controls fact eligibility. The explicit decision and all predecessor, correction, manifest, projection, representation and physical receipt hashes are in `config/clean_protocol/selection_repair_adoption_v4.json`. This is not human factual verification or a general representation contract.

Execution is local Darwin, `MacBook-Pro-7.lan`. Normal merge `40147cfb3861df6944a47fb112c539c89ac6bb6d` has parents reviewed `4cae65c7dfc825f774226007a81994f5a08776c9` and exactly the correction above. No later correction was merged. All historical artifacts and implementations remain unchanged.

## Membership and bounded representation

Both models map **3,040 full outer-training and 1,012 D rows** through original source identity/hash, statement hash, person/partition and authoritative tensor-sidecar indices. No missing, ambiguous, duplicate or mismatched mapping remains. V3 restrictions are retained, and the new `inventors:155`, `inventors:380`, `neg_inventors:155`, and `neg_inventors:380` quarantines are absent. Both fits use 1,520 examples per label. A15/A10 proposal bytes remain unchanged. P15/P10 counts are 2,778/2,864; their balanced exposures remain 700/800, but the v3-to-v4 sampled identity replacements are respectively **42/42 and 36/36**.

The accepted recorded representation remains raw statements, last real token, `hidden_states[1:]`, zero-based saved layer indices, BF16 extraction and float16 storage. Recorded models/revisions are Qwen2.5-7B-Instruct `a09a35458c702b33eeacc393d103063234e8bc28` and Llama3.1-8B-Instruct `0e9e39f249a16976918f6564b8830bc894c89659`. Producer-code conflicts for historical pinned Qwen and Llama files remain unresolved. Standalone historical tokenizer/model config bytes are absent. Saved-probe score replay does not independently replay model extraction; fresh extraction requires separate compatibility verification. Adoption is bounded to these existing caches and authorized development analyses.

Before scoring, the versioned preparation fixes all eleven endpoints and eligibility masks. Atomic D retains **1,012/1,040** rows; bare compounds retain **483/524 complete pairs**, **7,728/8,384** rows. Both operators retain 3,864 rows and each critical boundary 2,898. Isolated fact eligibility is 510/524. These masks are identical to v3. All exclusions are in inventors; compound admission requires accepted exact constituent identities and complete pair groups, never the old compound label alone.

## Fitting and uncertainty

Qwen saved L17/C=1 converged in **53 iterations**, gradient infinity norm **7.1886345e-5**. Llama saved L15/C=10 converged in **23 iterations**, gradient **9.7566826e-5**. Neither needed a retry. Both use the unchanged canonical raw float64 LR, free intercept, no standardization, sklearn L-BFGS max_iter 2,000/tol 1e-4 and one BLAS thread. They are full-training A-exposed diagnostics, outside any primary P-only bank.

Both historical heads first passed coefficient/settings/hash checks, full atomic AUROC replay, and row-level saved-score replay. All **154 summaries** were computed. Each comparison shares the same retained examples and the established **2,000 seed-1729 paired person/entity bootstrap draws**, topic-stratified with compound endpoint-product multiplicities. Intervals are pointwise 95% percentile intervals conditional on fitted heads; no redraw, imputation or topic dropping. Training delta compares corrected and historical heads on matched retained examples. Coverage delta compares the same historical head on retained versus original examples.

## Updated canonical results

Full pooled, topic-macro and per-topic values and intervals are in `results/t2_canonical_sensitivity_v4_20261001/metrics.json` and `metrics.csv`. These versioned tables refresh the canonical LR, raw compound, critical boundary, or-both and isolated min/max outputs without overwriting historical tables. `v3_checkpoint_comparison.json` records point changes relative to v3; it is distinct from the historical-probe paired sensitivity.

### Pooled

| Model | Condition / endpoint | Historical matched | V4 matched | Training Δ [95% CI] | Coverage Δ |
|---|---|---:|---:|---|---:|
| llama | atomic_D / atomic_auroc | 0.999992 | 1.000000 | +0.000008 [+0.000000, +0.000036] | +0.000070 |
| llama | raw_reference / and_auroc | 0.986042 | 0.973523 | -0.012519 [-0.016373, -0.008943] | +0.000747 |
| llama | raw_reference / or_auroc | 0.768046 | 0.760361 | -0.007685 [-0.014672, -0.001673] | +0.004296 |
| llama | raw_reference / and_tt_vs_mixed_auroc | 0.979063 | 0.960310 | -0.018753 [-0.024535, -0.013413] | +0.001109 |
| llama | raw_reference / or_mixed_vs_ff_auroc | 0.662691 | 0.650065 | -0.012625 [-0.021237, -0.004938] | +0.001893 |
| llama | or_explicit_or_both_v1 / or_auroc | 0.686827 | 0.675119 | -0.011708 [-0.018931, -0.005061] | -0.000995 |
| llama | or_explicit_or_both_v1 / or_mixed_vs_ff_auroc | 0.571256 | 0.563408 | -0.007848 [-0.016038, -0.000369] | -0.008055 |
| llama | isolated_external_minmax_v1 / and_auroc | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| llama | isolated_external_minmax_v1 / or_auroc | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| llama | isolated_external_minmax_v1 / and_tt_vs_mixed_auroc | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| llama | isolated_external_minmax_v1 / or_mixed_vs_ff_auroc | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| qwen | atomic_D / atomic_auroc | 0.999855 | 0.999836 | -0.000020 [-0.000130, +0.000051] | +0.000203 |
| qwen | raw_reference / and_auroc | 0.916691 | 0.935923 | +0.019232 [+0.003624, +0.032643] | +0.013938 |
| qwen | raw_reference / or_auroc | 0.835143 | 0.852700 | +0.017556 [+0.006261, +0.029961] | +0.021811 |
| qwen | raw_reference / and_tt_vs_mixed_auroc | 0.890815 | 0.909961 | +0.019146 [-0.000976, +0.035803] | +0.012574 |
| qwen | raw_reference / or_mixed_vs_ff_auroc | 0.770180 | 0.785275 | +0.015096 [+0.000772, +0.030076] | +0.023361 |
| qwen | or_explicit_or_both_v1 / or_auroc | 0.725252 | 0.771456 | +0.046203 [+0.030354, +0.062393] | +0.013781 |
| qwen | or_explicit_or_both_v1 / or_mixed_vs_ff_auroc | 0.662508 | 0.704886 | +0.042378 [+0.026519, +0.059017] | +0.011507 |
| qwen | isolated_external_minmax_v1 / and_auroc | 0.999829 | 0.999806 | -0.000023 [-0.000181, +0.000067] | +0.000283 |
| qwen | isolated_external_minmax_v1 / or_auroc | 0.999840 | 0.999817 | -0.000023 [-0.000181, +0.000067] | +0.000333 |
| qwen | isolated_external_minmax_v1 / and_tt_vs_mixed_auroc | 0.999751 | 0.999717 | -0.000034 [-0.000271, +0.000100] | +0.000418 |
| qwen | isolated_external_minmax_v1 / or_mixed_vs_ff_auroc | 0.999760 | 0.999726 | -0.000034 [-0.000271, +0.000100] | +0.000470 |

### Equal-topic macro

| Model | Condition / endpoint | Historical matched | V4 matched | Training Δ [95% CI] | Coverage Δ |
|---|---|---:|---:|---|---:|
| llama | atomic_D / atomic_auroc | 0.999955 | 1.000000 | +0.000045 [+0.000000, +0.000284] | +0.000382 |
| llama | raw_reference / and_auroc | 0.989651 | 0.990181 | +0.000529 [-0.002473, +0.003334] | -0.000278 |
| llama | raw_reference / or_auroc | 0.807865 | 0.815050 | +0.007184 [-0.004935, +0.020428] | +0.002212 |
| llama | raw_reference / and_tt_vs_mixed_auroc | 0.984477 | 0.985271 | +0.000794 [-0.003710, +0.005002] | -0.000433 |
| llama | raw_reference / or_mixed_vs_ff_auroc | 0.715983 | 0.726023 | +0.010040 [-0.007393, +0.029592] | +0.001797 |
| llama | or_explicit_or_both_v1 / or_auroc | 0.781087 | 0.788036 | +0.006949 [-0.001787, +0.016502] | +0.004310 |
| llama | or_explicit_or_both_v1 / or_mixed_vs_ff_auroc | 0.700839 | 0.708688 | +0.007849 [-0.003659, +0.021571] | +0.006447 |
| llama | isolated_external_minmax_v1 / and_auroc | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| llama | isolated_external_minmax_v1 / or_auroc | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| llama | isolated_external_minmax_v1 / and_tt_vs_mixed_auroc | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| llama | isolated_external_minmax_v1 / or_mixed_vs_ff_auroc | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| qwen | atomic_D / atomic_auroc | 0.999984 | 0.999931 | -0.000054 [-0.000238, +0.000000] | +0.000884 |
| qwen | raw_reference / and_auroc | 0.906169 | 0.908041 | +0.001873 [-0.013366, +0.017994] | -0.005511 |
| qwen | raw_reference / or_auroc | 0.833426 | 0.851580 | +0.018155 [+0.003775, +0.034856] | +0.003157 |
| qwen | raw_reference / and_tt_vs_mixed_auroc | 0.881170 | 0.880719 | -0.000451 [-0.020514, +0.017933] | -0.008762 |
| qwen | raw_reference / or_mixed_vs_ff_auroc | 0.766341 | 0.788179 | +0.021838 [+0.001677, +0.046025] | +0.001421 |
| qwen | or_explicit_or_both_v1 / or_auroc | 0.755789 | 0.776931 | +0.021142 [+0.005032, +0.037773] | +0.004653 |
| qwen | or_explicit_or_both_v1 / or_mixed_vs_ff_auroc | 0.682891 | 0.705397 | +0.022506 [+0.003674, +0.042462] | +0.003640 |
| qwen | isolated_external_minmax_v1 / and_auroc | 0.999988 | 0.999988 | +0.000000 [+0.000000, +0.000000] | +0.001086 |
| qwen | isolated_external_minmax_v1 / or_auroc | 0.999988 | 0.999988 | +0.000000 [+0.000000, +0.000000] | +0.001235 |
| qwen | isolated_external_minmax_v1 / and_tt_vs_mixed_auroc | 0.999982 | 0.999982 | +0.000000 [+0.000000, +0.000000] | +0.001630 |
| qwen | isolated_external_minmax_v1 / or_mixed_vs_ff_auroc | 0.999982 | 0.999982 | +0.000000 [+0.000000, +0.000000] | +0.001753 |

## Remaining refresh and adoption decisions

The >0.002 rule flags **20 training-cleanup summaries** and **16 evaluation-coverage summaries**. Two ranking/contrast triggers also fire (see exact records in `refresh_decisions.json`). These are comparisons to the historical baseline, not only the four-row v3-to-v4 change.

Broader historical analyses still awaiting refresh include original full layer/method grids and selection-dependent analyses, published compound-transfer tables/figures and narrative conclusions, cross-model rankings, and Priority-2 or-both/minmax interpretations outside this new canonical result package. The bounded pilot refreshes only its six specified layers and methods; it does not refresh the full candidate bank. No broader historical pipeline was launched.

P/A remains unfrozen. No P10 fallback, P20 condition, production bank, compound control or repair fit is authorized by these results. Precision, configuration and cohort decisions remain for design review.

## Validation and artifacts

Independent sklearn AUROC recomputation checks all 154 summary points (maximum absolute error **2.220446049250313e-16**), all stored paired intervals, artifact hashes, fitting membership and refresh thresholds. Compact outputs are committed under `results/t2_canonical_sensitivity_v4_20261001/`; full preparation/maps and scores/bootstrap arrays remain in `/Users/apple/projects/t2-canonical-sensitivity-v4-20261001/`. V3 remains intact as its own checkpoint.

```sh
/tmp/clean-extraction-venv/bin/python -B scripts/54_validate_canonical_sensitivity_v4.py \
 --payload-root /Users/apple/truth-probing-backups/20261001T164224Z/restore_t1_staging_20261001T164224Z_kgkt5sy7 \
 --prepared-root /Users/apple/projects/t2-canonical-sensitivity-v4-20261001/prepared \
 --results-root /Users/apple/projects/t2-canonical-sensitivity-v4-20261001/results
```
