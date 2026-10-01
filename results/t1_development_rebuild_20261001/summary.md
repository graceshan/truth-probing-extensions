# T1 historical development-results rebuild

DEVELOPMENT, pre-label/alias-cleanup; historical definitions; no correction overlay

Verified 253 payload files; 8384 benchmark rows / 524 pairs / 262 entities.
Rebuilt 3990 metric rows. Discrepancies beyond absolute tolerance 1e-12: 0.
All four recorded schedule hashes and saved pair-weight arrays matched. Effective replicate counts are reported per result.

## Main score results

| family | condition | metric | estimate | ci_low | ci_high | valid_replicates |
|---|---|---|---|---|---|---|
| qwen_lr | raw_reference | and_auroc | 0.902752183 | 0.884839196 | 0.920869780 | 2000 |
| qwen_lr | raw_reference | or_auroc | 0.813332192 | 0.792112837 | 0.835445042 | 2000 |
| qwen_lr | raw_reference | or_mixed_vs_ff_auroc | 0.746818731 | 0.724173065 | 0.773158969 | 2000 |
| methods | faithful_common_layer/l2_logistic | and_auroc | 0.902752183 | 0.884839196 | 0.920869780 | 2000 |
| methods | faithful_common_layer/l2_logistic | or_auroc | 0.813332192 | 0.792112837 | 0.835445042 | 2000 |
| methods | faithful_common_layer/l2_logistic | or_mixed_vs_ff_auroc | 0.746818731 | 0.724173065 | 0.773158969 | 2000 |
| methods | faithful_common_layer/difference_of_means | and_auroc | 0.907857630 | 0.882200980 | 0.931151360 | 2000 |
| methods | faithful_common_layer/difference_of_means | or_auroc | 0.835116082 | 0.818674376 | 0.852054995 | 2000 |
| methods | faithful_common_layer/difference_of_means | or_mixed_vs_ff_auroc | 0.764095369 | 0.743680644 | 0.786622170 | 2000 |
| methods | faithful_common_layer/mass_mean_covariance | and_auroc | 0.762451015 | 0.737807693 | 0.792800788 | 2000 |
| methods | faithful_common_layer/mass_mean_covariance | or_auroc | 0.687110612 | 0.664714413 | 0.709192335 | 2000 |
| methods | faithful_common_layer/mass_mean_covariance | or_mixed_vs_ff_auroc | 0.617723326 | 0.596432946 | 0.639463927 | 2000 |
| methods | faithful_common_layer/burger_t_g | and_auroc | 0.902141241 | 0.875666703 | 0.925837393 | 2000 |
| methods | faithful_common_layer/burger_t_g | or_auroc | 0.830394256 | 0.813847357 | 0.847935092 | 2000 |
| methods | faithful_common_layer/burger_t_g | or_mixed_vs_ff_auroc | 0.759039847 | 0.738587725 | 0.781411001 | 2000 |
| methods | faithful_common_layer/ttpd | and_auroc | 0.893507019 | 0.866156152 | 0.918042472 | 2000 |
| methods | faithful_common_layer/ttpd | or_auroc | 0.828394810 | 0.811837398 | 0.846130437 | 2000 |
| methods | faithful_common_layer/ttpd | or_mixed_vs_ff_auroc | 0.756464513 | 0.736046569 | 0.779520815 | 2000 |
| methods | faithful_method_selected_layer/l2_logistic | and_auroc | 0.902752183 | 0.884839196 | 0.920869780 | 2000 |
| methods | faithful_method_selected_layer/l2_logistic | or_auroc | 0.813332192 | 0.792112837 | 0.835445042 | 2000 |
| methods | faithful_method_selected_layer/l2_logistic | or_mixed_vs_ff_auroc | 0.746818731 | 0.724173065 | 0.773158969 | 2000 |
| methods | faithful_method_selected_layer/difference_of_means | and_auroc | 0.991409482 | 0.985671695 | 0.995783377 | 2000 |
| methods | faithful_method_selected_layer/difference_of_means | or_auroc | 0.897069785 | 0.882336210 | 0.913084149 | 2000 |
| methods | faithful_method_selected_layer/difference_of_means | or_mixed_vs_ff_auroc | 0.846238382 | 0.824350499 | 0.869995379 | 2000 |
| methods | faithful_method_selected_layer/mass_mean_covariance | and_auroc | 0.762451015 | 0.737807693 | 0.792800788 | 2000 |
| methods | faithful_method_selected_layer/mass_mean_covariance | or_auroc | 0.687110612 | 0.664714413 | 0.709192335 | 2000 |
| methods | faithful_method_selected_layer/mass_mean_covariance | or_mixed_vs_ff_auroc | 0.617723326 | 0.596432946 | 0.639463927 | 2000 |
| methods | faithful_method_selected_layer/burger_t_g | and_auroc | 0.992431361 | 0.986846966 | 0.996449103 | 2000 |
| methods | faithful_method_selected_layer/burger_t_g | or_auroc | 0.890492371 | 0.876044996 | 0.907042896 | 2000 |
| methods | faithful_method_selected_layer/burger_t_g | or_mixed_vs_ff_auroc | 0.836269831 | 0.814852255 | 0.860941714 | 2000 |
| methods | faithful_method_selected_layer/ttpd | and_auroc | 0.992284468 | 0.986808616 | 0.996322257 | 2000 |
| methods | faithful_method_selected_layer/ttpd | or_auroc | 0.889370338 | 0.874794690 | 0.906047891 | 2000 |
| methods | faithful_method_selected_layer/ttpd | or_mixed_vs_ff_auroc | 0.834523046 | 0.812940220 | 0.859439640 | 2000 |
| methods | matched_1000_common_layer/l2_logistic | and_auroc | 0.883125255 | 0.860834618 | 0.906009635 | 2000 |
| methods | matched_1000_common_layer/l2_logistic | or_auroc | 0.800508481 | 0.782366791 | 0.821251912 | 2000 |
| methods | matched_1000_common_layer/l2_logistic | or_mixed_vs_ff_auroc | 0.728882896 | 0.709062576 | 0.752656742 | 2000 |
| methods | matched_1000_common_layer/difference_of_means | and_auroc | 0.903498788 | 0.877338319 | 0.927169559 | 2000 |
| methods | matched_1000_common_layer/difference_of_means | or_auroc | 0.830377564 | 0.814002431 | 0.847945740 | 2000 |
| methods | matched_1000_common_layer/difference_of_means | or_mixed_vs_ff_auroc | 0.758867308 | 0.738427709 | 0.780990802 | 2000 |
| methods | matched_1000_common_layer/mass_mean_covariance | and_auroc | 0.858296974 | 0.835794808 | 0.881408840 | 2000 |
| methods | matched_1000_common_layer/mass_mean_covariance | or_auroc | 0.730290521 | 0.710340810 | 0.754307222 | 2000 |
| methods | matched_1000_common_layer/mass_mean_covariance | or_mixed_vs_ff_auroc | 0.673868528 | 0.655147168 | 0.697589751 | 2000 |
| methods | matched_1000_common_layer/burger_t_g | and_auroc | 0.902141241 | 0.875666703 | 0.925837393 | 2000 |
| methods | matched_1000_common_layer/burger_t_g | or_auroc | 0.830394256 | 0.813847357 | 0.847935092 | 2000 |
| methods | matched_1000_common_layer/burger_t_g | or_mixed_vs_ff_auroc | 0.759039847 | 0.738587725 | 0.781411001 | 2000 |
| methods | matched_1000_common_layer/ttpd | and_auroc | 0.893507019 | 0.866156152 | 0.918042472 | 2000 |
| methods | matched_1000_common_layer/ttpd | or_auroc | 0.828394810 | 0.811837398 | 0.846130437 | 2000 |
| methods | matched_1000_common_layer/ttpd | or_mixed_vs_ff_auroc | 0.756464513 | 0.736046569 | 0.779520815 | 2000 |
| controls | raw_reference | and_auroc | 0.902752183 | 0.884839196 | 0.920869780 | 2000 |
| controls | raw_reference | or_auroc | 0.813332192 | 0.792112837 | 0.835445042 | 2000 |
| controls | raw_reference | or_mixed_vs_ff_auroc | 0.746818731 | 0.724173065 | 0.773158969 | 2000 |
| controls | or_explicit_or_both_v1 | and_auroc | 0.902752183 | 0.884839196 | 0.920869780 | 2000 |
| controls | or_explicit_or_both_v1 | or_auroc | 0.711471505 | 0.687110269 | 0.737244049 | 2000 |
| controls | or_explicit_or_both_v1 | or_mixed_vs_ff_auroc | 0.651000998 | 0.625925439 | 0.676774127 | 2000 |
| controls | or_at_least_one_v1 | and_auroc | 0.902752183 | 0.884839196 | 0.920869780 | 2000 |
| controls | or_at_least_one_v1 | or_auroc | 0.665577108 | 0.633416124 | 0.693034217 | 2000 |
| controls | or_at_least_one_v1 | or_mixed_vs_ff_auroc | 0.569022329 | 0.537925402 | 0.597452193 | 2000 |
| controls | isolated_external_minmax_v1 | and_auroc | 0.999545967 | 0.998444032 | 1.000000000 | 2000 |
| controls | isolated_external_minmax_v1 | or_auroc | 0.999507119 | 0.998248834 | 1.000000000 | 2000 |
| controls | isolated_external_minmax_v1 | or_mixed_vs_ff_auroc | 0.999289814 | 0.997520264 | 1.000000000 | 2000 |
| llama | raw_reference | and_auroc | 0.985294600 | 0.979477533 | 0.990379270 | 2000 |
| llama | raw_reference | or_auroc | 0.763749988 | 0.740915115 | 0.784924387 | 2000 |
| llama | raw_reference | or_mixed_vs_ff_auroc | 0.660797466 | 0.631457202 | 0.688813789 | 2000 |
| llama | or_explicit_or_both_v1 | and_auroc | 0.985294600 | 0.979477533 | 0.990379270 | 2000 |
| llama | or_explicit_or_both_v1 | or_auroc | 0.687822012 | 0.664926238 | 0.710941043 | 2000 |
| llama | or_explicit_or_both_v1 | or_mixed_vs_ff_auroc | 0.579310919 | 0.551150809 | 0.605209035 | 2000 |
| llama | isolated_external_minmax_v1 | and_auroc | 1.000000000 | 1.000000000 | 1.000000000 | 2000 |
| llama | isolated_external_minmax_v1 | or_auroc | 1.000000000 | 1.000000000 | 1.000000000 | 2000 |
| llama | isolated_external_minmax_v1 | or_mixed_vs_ff_auroc | 1.000000000 | 1.000000000 | 1.000000000 | 2000 |

## Atomic validation (saved aggregates only)

| family | condition | layer | estimate | evidence |
|---|---|---|---|---|
| qwen_lr | selected_l2_logistic | 17 | 0.999652367 | saved_aggregate_only |
| llama | selected_l2_logistic | 15 | 0.999922337 | saved_aggregate_only |
| methods | faithful/primary_fixed_model_layer/burger_t_g | 17 | 0.996749260 | saved_aggregate_only |
| methods | faithful/primary_fixed_model_layer/difference_of_means | 17 | 0.994903846 | saved_aggregate_only |
| methods | faithful/primary_fixed_model_layer/l2_logistic | 17 | 0.999652367 | saved_aggregate_only |
| methods | faithful/primary_fixed_model_layer/mass_mean_covariance | 17 | 0.999227071 | saved_aggregate_only |
| methods | faithful/primary_fixed_model_layer/ttpd | 17 | 0.996087278 | saved_aggregate_only |
| methods | faithful/secondary_method_selected_layer/burger_t_g | 18 | 0.998372781 | saved_aggregate_only |
| methods | faithful/secondary_method_selected_layer/difference_of_means | 18 | 0.997636834 | saved_aggregate_only |
| methods | faithful/secondary_method_selected_layer/l2_logistic | 17 | 0.999652367 | saved_aggregate_only |
| methods | faithful/secondary_method_selected_layer/mass_mean_covariance | 17 | 0.999227071 | saved_aggregate_only |
| methods | faithful/secondary_method_selected_layer/ttpd | 18 | 0.997914201 | saved_aggregate_only |
| methods | matched/primary_fixed_model_layer/burger_t_g | 17 | 0.996749260 | saved_aggregate_only |
| methods | matched/primary_fixed_model_layer/difference_of_means | 17 | 0.995702663 | saved_aggregate_only |
| methods | matched/primary_fixed_model_layer/l2_logistic | 17 | 0.999530325 | saved_aggregate_only |
| methods | matched/primary_fixed_model_layer/mass_mean_covariance | 17 | 0.999578402 | saved_aggregate_only |
| methods | matched/primary_fixed_model_layer/ttpd | 17 | 0.996087278 | saved_aggregate_only |

## OR constituent diagnostics (surface versus canonical)

| family | labeling | ordering | metric | estimate | ci_low | ci_high | valid_replicates |
|---|---|---|---|---|---|---|---|
| qwen_lr | surface | all | mixed_first_true_vs_second_true | 0.312076620 | 0.274315429 | 0.346501868 | 2000 |
| qwen_lr | surface | all | tf_vs_ff | 0.685099025 | 0.659462188 | 0.715075480 | 2000 |
| qwen_lr | surface | all | ft_vs_ff | 0.808538437 | 0.781540714 | 0.837913598 | 2000 |
| qwen_lr | surface | AB | mixed_first_true_vs_second_true | 0.336558913 | 0.286226987 | 0.385520131 | 2000 |
| qwen_lr | surface | AB | tf_vs_ff | 0.687110308 | 0.653522347 | 0.724513039 | 2000 |
| qwen_lr | surface | AB | ft_vs_ff | 0.794428501 | 0.757780252 | 0.836460991 | 2000 |
| qwen_lr | surface | BA | mixed_first_true_vs_second_true | 0.287953062 | 0.239890421 | 0.339370716 | 2000 |
| qwen_lr | surface | BA | tf_vs_ff | 0.684102034 | 0.649190454 | 0.720727444 | 2000 |
| qwen_lr | surface | BA | ft_vs_ff | 0.822435318 | 0.784962933 | 0.858037185 | 2000 |
| qwen_lr | canonical | all | canonical_tf_vs_ft | 0.525397340 | 0.490191651 | 0.558425682 | 2000 |
| qwen_lr | canonical | all | tf_vs_ff | 0.756866041 | 0.728776382 | 0.787239953 | 2000 |
| qwen_lr | canonical | all | ft_vs_ff | 0.736771422 | 0.710562416 | 0.765920932 | 2000 |
| qwen_lr | canonical | AB | canonical_tf_vs_ft | 0.336558913 | 0.286226987 | 0.385520131 | 2000 |
| qwen_lr | canonical | AB | tf_vs_ff | 0.687110308 | 0.653522347 | 0.724513039 | 2000 |
| qwen_lr | canonical | AB | ft_vs_ff | 0.794428501 | 0.757780252 | 0.836460991 | 2000 |
| qwen_lr | canonical | BA | canonical_tf_vs_ft | 0.712046938 | 0.660629284 | 0.760109579 | 2000 |
| qwen_lr | canonical | BA | tf_vs_ff | 0.822435318 | 0.784962933 | 0.858037185 | 2000 |
| qwen_lr | canonical | BA | ft_vs_ff | 0.684102034 | 0.649190454 | 0.720727444 | 2000 |
| llama | surface | all | mixed_first_true_vs_second_true | 0.396408463 | 0.360616838 | 0.441124893 | 2000 |
| llama | surface | all | tf_vs_ff | 0.605180351 | 0.569306219 | 0.642033685 | 2000 |
| llama | surface | all | ft_vs_ff | 0.716414581 | 0.678854037 | 0.748076067 | 2000 |
| llama | surface | AB | mixed_first_true_vs_second_true | 0.398101072 | 0.348753294 | 0.457395256 | 2000 |
| llama | surface | AB | tf_vs_ff | 0.609831158 | 0.567990770 | 0.652846692 | 2000 |
| llama | surface | AB | ft_vs_ff | 0.719687810 | 0.674676441 | 0.759973072 | 2000 |
| llama | surface | BA | mixed_first_true_vs_second_true | 0.394794155 | 0.351288007 | 0.450727075 | 2000 |
| llama | surface | BA | tf_vs_ff | 0.600540470 | 0.558859440 | 0.648391477 | 2000 |
| llama | surface | BA | ft_vs_ff | 0.713183235 | 0.668395627 | 0.751670075 | 2000 |
| llama | canonical | all | canonical_tf_vs_ft | 0.501200943 | 0.468945401 | 0.532941411 | 2000 |
| llama | canonical | all | tf_vs_ff | 0.661129341 | 0.627608457 | 0.692322446 | 2000 |
| llama | canonical | all | ft_vs_ff | 0.660465591 | 0.625874801 | 0.694219947 | 2000 |
| llama | canonical | AB | canonical_tf_vs_ft | 0.398101072 | 0.348753294 | 0.457395256 | 2000 |
| llama | canonical | AB | tf_vs_ff | 0.609831158 | 0.567990770 | 0.652846692 | 2000 |
| llama | canonical | AB | ft_vs_ff | 0.719687810 | 0.674676441 | 0.759973072 | 2000 |
| llama | canonical | BA | canonical_tf_vs_ft | 0.605205845 | 0.549272925 | 0.648711993 | 2000 |
| llama | canonical | BA | tf_vs_ff | 0.713183235 | 0.668395627 | 0.751670075 | 2000 |
| llama | canonical | BA | ft_vs_ff | 0.600540470 | 0.558859440 | 0.648391477 | 2000 |

## Evidence limits

Atomic validation row scores are not packaged. Values are hash-verified structured saved aggregates cross-checked against validation_metrics.csv; no atomic CI is reconstructed.

The recorded pinned_atomic_probes.py working-file digest conflicts with the recorded producer/Git digest. It remains unresolved. See discrepancies.json.
Activation tensors are excluded. Upstream extraction and fitting are not revalidated. No final-test rows or correction overlay were used.

Surface-position results are in surface_diagnostics.csv/json. Canonical TF becomes surface FT under BA; canonical-cell results remain separately labeled.

Full input identities, available and missing upstream byte records, provenance differences, and checks are in verification.json, input_hashes.json, and discrepancies.json.
