# V3 canonical LR sensitivity — completed 2026-10-01

The two corrected full-outer-training canonical references completed against
**candidate_overlay_v3**, exact reviewed correction
`2ffbe79cdb22eb772c1b842ab9f0e9d8d11a8e8c`. All 22 model/endpoints (154 pooled,
five-topic and macro summaries) were computed. **Refresh is required**; this
run did not launch downstream refresh work.

## Integration and bounded adoption

Execution remained on Darwin, `MacBook-Pro-7.lan`, in
`/Users/apple/projects/truth-probing-t1-score-rebuild`, branch
`t2-canonical-sensitivity-20261001`. Commit `2f663b9` checkpointed the v2 work
and physical receipts before integration. Normal merge `fd346987` has parents
`2f663b9` and exactly `2ffbe79cdb22eb772c1b842ab9f0e9d8d11a8e8c`. No later
branch tip was pulled or merged; `clean-eval-protocol` was not switched to or pushed.

`selection_repair_adoption_v3.json` records the user's acceptance and resumption
for these two within-cache development references. `audit_successor_status`
controls row admission; `current_eligible` controls projected fact eligibility.
Historical status/eligibility fields remain unchanged and do not grant current
admission. The separately versioned adapter uses these current fields explicitly.
The v2 adoption, hold, mappings, masks and locked implementation remain historical.
**There were no v2 fits or scores to reuse.**

The recorded raw-statement representation is accepted only within these caches.
Recorded Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28` and Llama3.1-8B-Instruct revision
`0e9e39f249a16976918f6564b8830bc894c89659` retain the saved raw-statement,
last-real-token, `hidden_states[1:]` layer convention, BF16 extraction and float16
storage evidence. Historical producer-code hash conflicts and absent standalone
tokenizer/model config bytes remain unresolved. Matching saved-probe replay here
is not independent replay of the historical model extraction. This adoption is
not human factual verification or authorization for a fresh extraction.

## Membership, physical binding and coverage

Each model maps all **3,044 train and 1,012 D** admitted rows, with no missing,
ambiguous, duplicate source/index, or mismatched binding. Original source-row
identity/hash, exact statement hash, person key and partition remain attached.
The four quarantined IDs are absent: `inventors:23`, `inventors:157`,
`neg_inventors:23`, `neg_inventors:157`. Duplicate statement text is not used to
collapse source identities. Tensor indices come from authoritative sidecars;
separate export indices select the corresponding verified saved-layer rows.

All seven source tensors previously passed sizes, SHA-256, NPY shape/dtype/order,
stable-read and live companion checks on verified host `ef7f7534c328`. Their
selected-layer exports were reverified locally and rebound to v3. No new SSH
session or tensor transfer was needed for v3. Both atomic source caches still
have 3,144 train / 1,040 D rows; the corrected references use 3,044, not the full
physical tensor. This compares the full accepted cleanup (100 training rows)
against the historical probes, not the isolated effect of v2-to-v3's four rows.

The successor projection validator and actual sampler recomputation passed:

| Prospective membership | Rows | Balanced exposure | Removed / added sample identities |
|---|---:|---:|---:|
| P15 | 2,782 | 700 | 72 / 72 |
| P10 | 2,868 | 800 | 68 / 68 |

These are compatibility checks on prospective projections. **P/A remains unfrozen**;
neither reference is a P-only production-bank head.

Score-independent masks were frozen before replay or fitting. Atomic D retains
1,012/1,040 rows (28 excluded); compounds retain 483/524 complete 16-row pairs
(41 pairs / 656 bare rows excluded). All exclusions are in inventors: D retains
134/162 rows, pairs 49/90, isolated facts 76/90. Across all topics, 510/524 isolated
facts are eligible. Complete-pair gating also removes supported constituent
rows when another fact in that pair is ineligible. Each AND/OR endpoint retains
3,864/4,192 rows; each critical boundary retains 2,898/3,144. No compound was
retained merely because its old truth label was present.

## Fitting and numerical verification

Raw float64 canonical sklearn LR was used with no feature standardization or
extra class weights, free intercept, L-BFGS, initial max_iter=2,000, tol=1e-4,
and one BLAS thread. The unchanged canonical warning-retry policy was available;
neither fit needed a retry. Each trained on 1,522 negative and 1,522 positive rows.

| Model | Zero-based saved layer / C | Iterations | Final gradient infinity norm | Valid |
|---|---|---:|---:|---|
| Qwen | 17 / 1 | 49 | 4.81827e-5 | yes |
| Llama | 15 / 10 | 18 | 8.28007e-5 | yes |

Python 3.11.6, NumPy 2.4.6, SciPy 1.17.1, sklearn 1.9.1, pandas 3.0.6.
The sklearn penalty deprecation warning is recorded; the pinned mathematical
procedure was preserved. Shared R0/repair objectives and historical evaluators
were not changed.

Before either corrected fit, both historical probes passed settings/hash checks,
atomic full-D AUROC replay, and saved row-score replay. Maximum absolute score
error was 2.48690e-14. Qwen matched 8,384 bare, 4,192 or-both and 524 isolated
scores; Llama matched all 13,100 selected development scores. Historical atomic
AUROCs exactly matched the saved aggregates, using row-level verified states.

## AUROC results

`Old full` is the historical probe on original evaluation coverage. `Old matched`
and `V3 matched` use identical retained rows. Training delta is V3 matched minus
Old matched; coverage delta is Old matched minus Old full. These are separate
comparisons. Intervals are pointwise paired 95% bootstrap intervals for training
delta, conditional on the fitted heads; 2,000 shared seed-1729 draws were valid
for every summary. No redraw, imputation or topic dropping was used.

### Pooled

| Model | Endpoint | Old full | Old matched | V3 matched | Training Δ [95% CI] | Coverage Δ |
|---|---|---:|---:|---:|---|---:|
| Qwen | Atomic D atomic | 0.999652 | 0.999855 | 0.999836 | -0.000020 [-0.000120, +0.000047] | +0.000203 |
| Qwen | Bare AND | 0.902752 | 0.916691 | 0.935726 | +0.019035 [+0.002971, +0.032994] | +0.013938 |
| Qwen | Bare OR | 0.813332 | 0.835143 | 0.851653 | +0.016510 [+0.005085, +0.029293] | +0.021811 |
| Qwen | Bare AND TT–mixed | 0.878241 | 0.890815 | 0.909815 | +0.019000 [-0.001290, +0.036165] | +0.012574 |
| Qwen | Bare OR mixed–FF | 0.746819 | 0.770180 | 0.783801 | +0.013622 [-0.000950, +0.028735] | +0.023361 |
| Qwen | Or-both OR | 0.711472 | 0.725252 | 0.773551 | +0.048298 [+0.032857, +0.064005] | +0.013781 |
| Qwen | Or-both OR mixed–FF | 0.651001 | 0.662508 | 0.707059 | +0.044550 [+0.028513, +0.060644] | +0.011507 |
| Qwen | Isolated min/max AND | 0.999546 | 0.999829 | 0.999834 | +0.000006 [-0.000109, +0.000157] | +0.000283 |
| Qwen | Isolated min/max OR | 0.999507 | 0.999840 | 0.999834 | -0.000006 [-0.000112, +0.000089] | +0.000333 |
| Qwen | Isolated min/max AND TT–mixed | 0.999334 | 0.999751 | 0.999751 | +0.000000 [-0.000164, +0.000199] | +0.000418 |
| Qwen | Isolated min/max OR mixed–FF | 0.999290 | 0.999760 | 0.999751 | -0.000009 [-0.000168, +0.000134] | +0.000470 |
| Llama | Atomic D atomic | 0.999922 | 0.999992 | 1.000000 | +0.000008 [+0.000000, +0.000036] | +0.000070 |
| Llama | Bare AND | 0.985295 | 0.986042 | 0.971306 | -0.014736 [-0.018999, -0.010636] | +0.000747 |
| Llama | Bare OR | 0.763750 | 0.768046 | 0.764814 | -0.003231 [-0.009950, +0.002769] | +0.004296 |
| Llama | Bare AND TT–mixed | 0.977954 | 0.979063 | 0.956985 | -0.022078 [-0.028477, -0.015948] | +0.001109 |
| Llama | Bare OR mixed–FF | 0.660797 | 0.662691 | 0.655989 | -0.006701 [-0.015282, +0.001120] | +0.001893 |
| Llama | Or-both OR | 0.687822 | 0.686827 | 0.670177 | -0.016650 [-0.023969, -0.009387] | -0.000995 |
| Llama | Or-both OR mixed–FF | 0.579311 | 0.571256 | 0.560340 | -0.010916 [-0.019462, -0.002949] | -0.008055 |
| Llama | Isolated min/max AND | 1.000000 | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| Llama | Isolated min/max OR | 1.000000 | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| Llama | Isolated min/max AND TT–mixed | 1.000000 | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| Llama | Isolated min/max OR mixed–FF | 1.000000 | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |

### Equal-topic macro

| Model | Endpoint | Old full | Old matched | V3 matched | Training Δ [95% CI] | Coverage Δ |
|---|---|---:|---:|---:|---|---:|
| Qwen | Atomic D atomic | 0.999100 | 0.999984 | 0.999975 | -0.000009 [-0.000034, +0.000000] | +0.000884 |
| Qwen | Bare AND | 0.911680 | 0.906169 | 0.908550 | +0.002382 [-0.012323, +0.017817] | -0.005511 |
| Qwen | Bare OR | 0.830269 | 0.833426 | 0.851443 | +0.018018 [+0.003436, +0.034246] | +0.003157 |
| Qwen | Bare AND TT–mixed | 0.889932 | 0.881170 | 0.881439 | +0.000269 [-0.019205, +0.018173] | -0.008762 |
| Qwen | Bare OR mixed–FF | 0.764920 | 0.766341 | 0.788303 | +0.021962 [+0.001845, +0.045590] | +0.001421 |
| Qwen | Or-both OR | 0.751136 | 0.755789 | 0.778497 | +0.022708 [+0.006131, +0.039481] | +0.004653 |
| Qwen | Or-both OR mixed–FF | 0.679252 | 0.682891 | 0.707412 | +0.024521 [+0.004775, +0.043963] | +0.003640 |
| Qwen | Isolated min/max AND | 0.998902 | 0.999988 | 0.999988 | +0.000000 [+0.000000, +0.000000] | +0.001086 |
| Qwen | Isolated min/max OR | 0.998753 | 0.999988 | 0.999988 | +0.000000 [+0.000000, +0.000000] | +0.001235 |
| Qwen | Isolated min/max AND TT–mixed | 0.998352 | 0.999982 | 0.999982 | +0.000000 [+0.000000, +0.000000] | +0.001630 |
| Qwen | Isolated min/max OR mixed–FF | 0.998229 | 0.999982 | 0.999982 | +0.000000 [+0.000000, +0.000000] | +0.001753 |
| Llama | Atomic D atomic | 0.999573 | 0.999955 | 1.000000 | +0.000045 [+0.000000, +0.000284] | +0.000382 |
| Llama | Bare AND | 0.989930 | 0.989651 | 0.988814 | -0.000837 [-0.003837, +0.002351] | -0.000278 |
| Llama | Bare OR | 0.805653 | 0.807865 | 0.819096 | +0.011231 [-0.000946, +0.025407] | +0.002212 |
| Llama | Bare AND TT–mixed | 0.984910 | 0.984477 | 0.983221 | -0.001256 [-0.005756, +0.003527] | -0.000433 |
| Llama | Bare OR mixed–FF | 0.714187 | 0.715983 | 0.731664 | +0.015681 [-0.001652, +0.036769] | +0.001797 |
| Llama | Or-both OR | 0.776777 | 0.781087 | 0.789349 | +0.008262 [-0.000362, +0.018238] | +0.004310 |
| Llama | Or-both OR mixed–FF | 0.694392 | 0.700839 | 0.711507 | +0.010668 [-0.000795, +0.025172] | +0.006447 |
| Llama | Isolated min/max AND | 1.000000 | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| Llama | Isolated min/max OR | 1.000000 | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| Llama | Isolated min/max AND TT–mixed | 1.000000 | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |
| Llama | Isolated min/max OR mixed–FF | 1.000000 | 1.000000 | 1.000000 | +0.000000 [+0.000000, +0.000000] | +0.000000 |

## Refresh decisions and limitations

The strict >0.002 rule fired for **21 training-cleanup summaries** and separately
for **16 evaluation-coverage summaries**. Both models have affected compound
endpoints. Four ranking/contrast interpretation triggers also fired:

- Atomic topic-macro model ranking reverses to Llama > Qwen; this is a tiny
  point-ranking change, not a claim of a significant difference.
- Llama's macro bare-OR versus or-both-OR paired interval now excludes zero.
- The macro bare-OR Llama-minus-Qwen paired interval now excludes zero.
- Qwen's pooled isolated AND/OR point comparison changes from a tiny difference
  to a tie; this is not an asserted performance gain.

Affected downstream work: canonical LR compound transfer tables/figures and
critical-boundary summaries; Priority-2 raw/or-both/minmax comparisons;
Qwen/Llama canonical-reference comparisons and interpretations; and method
comparisons using these historical baselines or development eligibility masks.
The trigger does not require a delta interval to exclude zero. Pooled and macro
movements can differ in direction, as shown above. No broader refresh was launched.

No endpoint is missing a verifiable input in this run. The producer-code and
standalone tokenizer/config limitations remain, and no general representation
contract or fresh-extraction compatibility is established. No final-test
predictions, model extraction, production-bank or repair fit, or A/P reservation
was performed.

## Artifacts and validation

Compact results, both corrected probe archives, adoption/mapping and preparation
receipts, sampler bindings, full 154-summary metrics, fit diagnostics, historical
replay and refresh decisions are in
`results/t2_canonical_sensitivity_v3_20261001/`.
Full v3 mappings/masks and large score/bootstrap artifacts remain under
`/Users/apple/projects/t2-canonical-sensitivity-v3-20261001/`.
The immutable v2 physical exports/preparation remain under
`/Users/apple/projects/t2-canonical-sensitivity-artifacts-20261001/`.
All external artifacts are hash-bound by the compact receipts.

The integrated suite passed **192 tests and 29 subtests**; two additional
independent-validator tampering tests passed. The deliberate nonfinite-fit test
emits one expected RuntimeWarning. Successor projection, historical pilot,
closure, candidate and T2B validators passed; T2A source-audit tests also passed.
Independent sklearn AUROC recomputation checked all 154 summaries to maximum
absolute error 2.22045e-16, recomputed every stored CI from paired draws, and
verified result hashes, membership counts and refresh thresholds. No refit or
activation loading was used in that validation. `git diff --check` passes.

```sh
/tmp/clean-extraction-venv/bin/python -B scripts/52_validate_canonical_sensitivity_v3.py \
  --payload-root /Users/apple/truth-probing-backups/20261001T164224Z/restore_t1_staging_20261001T164224Z_kgkt5sy7 \
  --prepared-root /Users/apple/projects/t2-canonical-sensitivity-v3-20261001/prepared \
  --results-root /Users/apple/projects/t2-canonical-sensitivity-v3-20261001/results
```

`51_canonical_sensitivity_v3.py prepare` creates a new versioned preparation from
explicit parent exports. Its `evaluate` command requires new result and score
directories and refuses overwrite. Adoption, preparation, source hashes and
membership versions must agree before scoring or fitting. The old v2 runner
continues to reject execution; only the separately adopted v3 entry point resumes
this bounded sensitivity.
