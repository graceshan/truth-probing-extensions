# Matched-training-data atomic control

All five methods fit the same ordered 1,000 TRAIN rows and evaluate the same 1,040 VALIDATION rows. The training rows are copied from the faithful Bürger sample, without resampling. No atomic test labels/activation values or compound data were accessed.

LR is newly fit with the existing C=0.01 and solver settings; C is not retuned. Layers are fixed at Qwen2.5 saved index 17 and Qwen3 index 28. No secondary layer sweep was performed. Signs use TRAIN/formulas only. There are no validation sign flips or train+validation refits.

## Qwen/Qwen2.5-7B-Instruct — saved layer 17

| Method | Overall | Affirmative | Negated | Cities | Spanish | Inventors | Elements | Animals | Topic macro | Δ overall vs faithful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| l2_logistic | 0.999582 | 0.999763 | 0.999408 | 0.999977 | 1.000000 | 0.993599 | 1.000000 | 1.000000 | 0.998715 | -0.000085 |
| difference_of_means | 0.995688 | 0.999423 | 0.994377 | 0.999854 | 0.996998 | 0.978205 | 0.947531 | 1.000000 | 0.984517 | +0.000821 |
| mass_mean_covariance | 0.999638 | 0.999482 | 0.999763 | 0.999932 | 1.000000 | 0.995428 | 1.000000 | 1.000000 | 0.999072 | +0.000422 |
| burger_t_g | 0.996727 | 0.999334 | 0.993903 | 0.999809 | 0.997185 | 0.983539 | 0.957562 | 1.000000 | 0.987619 | +0.000000 |
| ttpd | 0.996268 | 0.999319 | 0.993222 | 0.998367 | 0.996622 | 0.982777 | 0.983796 | 1.000000 | 0.992313 | +0.000000 |

## Qwen/Qwen3-8B — saved layer 28

| Method | Overall | Affirmative | Negated | Cities | Spanish | Inventors | Elements | Animals | Topic macro | Δ overall vs faithful |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| l2_logistic | 0.999201 | 0.998949 | 0.999541 | 0.999977 | 1.000000 | 0.983539 | 1.000000 | 1.000000 | 0.996703 | -0.000499 |
| difference_of_means | 0.975636 | 0.983766 | 0.979371 | 0.999223 | 0.968662 | 0.815882 | 0.993827 | 1.000000 | 0.955519 | -0.003813 |
| mass_mean_covariance | 0.998195 | 0.997765 | 0.998579 | 0.999955 | 1.000000 | 0.976985 | 1.000000 | 1.000000 | 0.995388 | +0.014589 |
| burger_t_g | 0.979774 | 0.984314 | 0.980363 | 0.999527 | 0.972415 | 0.833257 | 0.992284 | 1.000000 | 0.959497 | +0.000000 |
| ttpd | 0.982441 | 0.983248 | 0.978616 | 0.999212 | 0.977857 | 0.848041 | 0.989198 | 1.000000 | 0.962862 | +0.000000 |

The t_G and TTPD parameters and diagnostics reproduce the faithful fixed-layer results exactly. Their training data already matched this construction. The other methods were refit on this subset.

Shared ordered training-record SHA-256: `cc276dad34edf412285fde8e32be40ce152dae9ced78b2b4c20754f5fd92a189`.

Each model directory includes training_rows.csv (exact source copy), validation_rows.csv, row_ids.json, parameters.npz, validation_metrics.csv, summary.json and provenance.json. The provenance records both ordered row-ID and full-record hashes, cache-slice hashes, method settings and source/version hashes. verification.json confirms every faithful result file remained byte-for-byte unchanged.

AUROCs are validation results. The diagnostic CSV also retains zero-threshold accuracy from the shared helper; that threshold is not calibrated for raw projection methods.
