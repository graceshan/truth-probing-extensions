# Atomic validation method comparison

TRAIN-only fits; VALIDATION-only selection. No test or compound activations/labels were used.

All methods use the same entity split. LR/mean difference/covariance-MM use 3,144 training rows; Bürger t_G/TTPD follow the original paired, equal-dataset sampling with 1,000 training rows (seed 0). All methods evaluate the same 1,040 validation rows per model.

MM below means the original Marks–Tegmark covariance-adjusted iid variant, not Bürger's unadjusted mean-difference-plus-logistic-bias baseline. TTPD includes both unregularized LR stages.

## Qwen/Qwen2.5-7B-Instruct

Primary: fixed model-selected layer 17

| Method | Layer | Overall | Affirmative | Negated | Cities | Spanish | Inventors | Elements | Animals | Topic macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 LR | 17 | 0.999667 | 0.999763 | 0.999615 | 0.999910 | 1.000000 | 0.995428 | 1.000000 | 1.000000 | 0.999067 |
| Difference of means | 17 | 0.994867 | 0.999349 | 0.990633 | 0.999887 | 0.990430 | 0.976833 | 0.922840 | 1.000000 | 0.977998 |
| Covariance MM | 17 | 0.999216 | 0.998816 | 0.999408 | 0.999899 | 1.000000 | 0.992684 | 1.000000 | 0.997070 | 0.997931 |
| Bürger t_G | 17 | 0.996727 | 0.999334 | 0.993903 | 0.999809 | 0.997185 | 0.983539 | 0.957562 | 1.000000 | 0.987619 |
| Full TTPD | 17 | 0.996268 | 0.999319 | 0.993222 | 0.998367 | 0.996622 | 0.982777 | 0.983796 | 1.000000 | 0.992313 |

Secondary: each method's validation-selected layer

| Method | Layer | Overall | Affirmative | Negated | Cities | Spanish | Inventors | Elements | Animals | Topic macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 LR | 17 | 0.999667 | 0.999763 | 0.999615 | 0.999910 | 1.000000 | 0.995428 | 1.000000 | 1.000000 | 0.999067 |
| Difference of means | 18 | 0.997626 | 0.999438 | 0.996182 | 0.999899 | 0.998123 | 0.978509 | 0.992284 | 1.000000 | 0.993763 |
| Covariance MM | 18 | 0.999368 | 0.999023 | 0.999674 | 0.999662 | 1.000000 | 0.994208 | 1.000000 | 1.000000 | 0.998774 |
| Bürger t_G | 18 | 0.998388 | 0.999512 | 0.996804 | 0.999809 | 0.999812 | 0.985216 | 0.997685 | 1.000000 | 0.996504 |
| Full TTPD | 18 | 0.997940 | 0.999526 | 0.996508 | 0.999437 | 1.000000 | 0.986283 | 0.997685 | 1.000000 | 0.996681 |

## Qwen/Qwen3-8B

Primary: fixed model-selected layer 28

| Method | Layer | Overall | Affirmative | Negated | Cities | Spanish | Inventors | Elements | Animals | Topic macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 LR | 28 | 0.999700 | 0.999734 | 0.999704 | 0.999989 | 1.000000 | 0.993446 | 1.000000 | 1.000000 | 0.998687 |
| Difference of means | 28 | 0.979449 | 0.989064 | 0.969042 | 0.999088 | 0.939013 | 0.828532 | 0.981481 | 0.999023 | 0.949428 |
| Covariance MM | 28 | 0.983606 | 0.978957 | 0.988013 | 0.999414 | 0.995121 | 0.908093 | 0.964506 | 0.993164 | 0.972060 |
| Bürger t_G | 28 | 0.979774 | 0.984314 | 0.980363 | 0.999527 | 0.972415 | 0.833257 | 0.992284 | 1.000000 | 0.959497 |
| Full TTPD | 28 | 0.982441 | 0.983248 | 0.978616 | 0.999212 | 0.977857 | 0.848041 | 0.989198 | 1.000000 | 0.962862 |

Secondary: each method's validation-selected layer

| Method | Layer | Overall | Affirmative | Negated | Cities | Spanish | Inventors | Elements | Animals | Topic macro |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| L2 LR | 28 | 0.999700 | 0.999734 | 0.999704 | 0.999989 | 1.000000 | 0.993446 | 1.000000 | 1.000000 | 0.998687 |
| Difference of means | 21 | 0.997685 | 0.998861 | 0.998609 | 1.000000 | 0.998123 | 0.959000 | 0.999228 | 1.000000 | 0.991270 |
| Covariance MM | 19 | 0.998110 | 0.997765 | 0.998476 | 1.000000 | 0.999625 | 0.981100 | 1.000000 | 1.000000 | 0.996145 |
| Bürger t_G | 21 | 0.997748 | 0.998639 | 0.998698 | 1.000000 | 0.999062 | 0.958848 | 1.000000 | 1.000000 | 0.991582 |
| Full TTPD | 21 | 0.998391 | 0.998683 | 0.998417 | 0.999899 | 1.000000 | 0.974089 | 0.999228 | 1.000000 | 0.994643 |

The numerical fits used the source snapshot retained under `fit_implementation_snapshot/`. A metadata-only finalization converted dataset-name object arrays to Unicode for safe pickle-disabled loading. Every numerical parameter array was verified unchanged, and all selected validation diagnostics were reproduced from the finalized archives. Original fitting provenance and old/new archive hashes are retained.

No validation-based sign flips or train+validation refits were performed. Secondary selection uses pooled validation AUROC, with lower-layer tie breaking. L2 reuses the existing all-layer/C validation winner. These are validation results, not test or compound-transfer estimates.
