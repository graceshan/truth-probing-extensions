# Independent Final-E scientific audit, October 5, 2026

**PASS — the accepted evaluation reconstructed exactly and passed the pinned separate principal numerical checks. No discrepancy was found. Both models support the locked benchmark improvement-with-retention claim, with practical gain, equal-topic gain and full-training robustness supported by their separate checks.** These are pointwise conditional results, not simultaneous or joint 95% confidence.

Reviewer: mac 2. Environment: the shared local Mac (`MacBook-Pro-7`, Darwin/macOS 12.6 arm64), reviewed Python 3.11.6 at `/opt/homebrew/opt/python@3.11/bin/python3.11`. mac 1 and mac 2 are sessions on the same hardware; this is independent calculation, not hardware replication. RunPod was not contacted.

## Exact audit identities and approval

| Item | Verified identity |
|---|---|
| Evaluation delivery / publication base | `75030617daa1987d22d7d550714bf78bc2b0c48e` |
| Clean detached runnable producer | `b2c6dde71e82f5f0af308761bf19ed3341670587` |
| Producer tree | `93972950bca996dbfa9794dedf07934868b42e98` |
| Final-lock SHA-256 | `88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e` |
| Unchanged audit-authorization SHA-256 | `41af4c83bafc4d216b5df836cb6c1c018ecdc3c6659d41d324227aad2bfa6b62` |
| Source completion SHA-256 | `632ed93f295027701e1d798a94c758e332c747e071aee410ebb6972114d408bc` |
| Reconstruction completion SHA-256 | `03c451b2bb6c013a442f833923b8960d2b64cdef57f0015ec3bfa21c60591fb4` |
| Audit operation receipt SHA-256 | `92a540227efea40fa338c94377229e106963bd34625ab3f8a25eef29967e6337` |
| Companion audit receipt SHA-256 | `bf7f8abe6d1dbf67209f4f2b296c2f367742d4f00b5cd7d5e5a4d6e2ce87b60e` |
| Accepted raw campaign manifest SHA-256 | `8387b3e319b6c3d9ca9c1037f6f1b46166b6cf536f11bfe494f3ee016291d870` |
| Artifact-roots SHA-256 | `a072433410e825bd3c288f304187c901d8da70459f5359212517df4446485645` |

The owner approved evaluation followed by independent audit on October 5 with “Yes I approve”; the published evaluation provenance and unchanged scoped audit control record this approval. The user subsequently supplied/confirmed the exact evaluation delivery and source-completion pin. No renewed approval or additional mac 1 chat message was required. The earlier missing-handoff blocker receipt at `PACKAGE/audit-prerequisite-inspection-20261005-v1/blocked.json` is preserved byte-for-byte as historical evidence; it was a prerequisite stop, not a failed numerical audit.

Read the published evaluation handoff at the exact delivery, applicable repository/review instructions and the frozen runner/reference functions. No applicable `AGENTS.md` was found in the inspected repository/ancestors or producer. Verified the published evaluation ref and fetched its exact commit; created the separate additive branch `checkpoint-final-e-scientific-audit-20261005` at that delivery. The publication checkout was not used for execution. No moving-branch pull, reset, merge, force push or modification to mac 1's worktree occurred.

## Execution and receipts

Durable package (`PACKAGE`):
`/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1`.

Scientific root (`SCIENCE`): `PACKAGE/scientific-evaluation-20261005-v1`.

Exactly one audit invocation ran from `PACKAGE/execution-code`, with `PYTHONDONTWRITEBYTECODE=1` and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS`, `NUMEXPR_NUM_THREADS` all set to 1:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m src.final_e_runner audit \
  --model both --output "$PACKAGE/handoff/campaign" \
  --roots "$SCIENCE/control/artifact-roots.json" \
  --source-report "$SCIENCE/evaluation" \
  --report "$SCIENCE/audit-reconstruction" \
  --operation-receipt "$SCIENCE/operation-receipts/audit.json" \
  --authorization "$SCIENCE/control/local-both-audit.json" \
  --authorization-sha256 41af4c83bafc4d216b5df836cb6c1c018ecdc3c6659d41d324227aad2bfa6b62 \
  --lock "$PACKAGE/transport/final_lock_20261004.json" \
  --lock-sha256 88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e
```

Audit output, operation receipt and companion receipt paths were absent before launch. Preflight checked the recorded approval against the owner's supplied instructions. The unchanged gate verified required approval fields, exact lock/control hashes, producer, six immutable pins, fully resolved scientific lock and scoped phase/model/output/campaign authorization; field validation alone is not proof of authentic user approval. The original controls were reused without regeneration or modification. Preflight independently rechecked all 180 unique frozen fit/parameter/preprocessing/selection references, exact roots mapping, source manifests, successful evaluation operation/process receipts and source-completion inventory. The runner performed its mandatory full accepted-raw/settled-accounting verification.

The runner acquired `/tmp/truth-probing-cpu-exclusive.lock` itself; the launcher did not hold a second lock. Local job inspection found no other substantial numerical job before launch. During execution `lsof` confirmed that audit worker PID 76169 held the shared lock. The campaign lock and unchanged **10,800-second CPU-job cap** were also retained. There was no retry, repair or model loading.

| Measurement | Actual value |
|---|---:|
| Audit operation wall time | 770.077650 seconds |
| Launcher child wait wall time | 770.307953 seconds |
| Peak child RSS, macOS bytes | 3,851,239,424 (3.85 GB) |
| Preflight verification wall time | 8.383460 seconds |
| Post-audit verification wall time | 8.103676 seconds |

The operation returned completed, error null, exit zero, model_loaded false. These are measured durations, distinct from the stopping cap and historical GPU campaign charges.

## Agreement and actual independent coverage

Complete frozen replay compared every record in all five report tables, then compared the bindings and every group of every frozen-head prediction archive with `np.array_equal`:

| Complete reconstructed table | Rows |
|---|---:|
| Scalar metrics | 28,560 |
| Scalar contrasts | 38,556 |
| Constituent metrics | 9,450 |
| Constituent contrasts | 1,050 |
| Paired wording comparisons | 6,132 |
| **Total** | **83,748** |

All **85 head prediction archives matched exactly**, with no fitting or selection. The source and reconstructed completion hashes differ because their audit/evaluation access envelopes and measured times differ; that difference is expected and does not signify a scientific discrepancy. The companion receipt binds both completion hashes and records exact report/prediction agreement.

The separate pinned path passed **393,650 numerical/decision checks with maximum absolute difference 0.0**. It uses the previously reviewed full-bank sorted-pair weighted-AUROC implementation and explicit constituent target/confusion/Boolean/min-max formulas, not the production E summaries/interval implementation. Its unchanged numerical acceptance tolerance is `2e-12`; no tolerance was changed.

Actual selected record coverage, counted from the pinned reference predicates and existing source report:

| Separate-path record scope | Selected rows |
|---|---:|
| Scalar metric records | 720 |
| Scalar paired-contrast records | 1,000 |
| Paired wording records | 438 |
| Constituent metric records | 2,700 |
| Constituent paired-contrast records | 300 |

Scalar checks cover the six arms `R_all`, `S_all`, `S_atom_all`, `reduced_LR`, `S_full`, `S_atom_full`: atomic AUROC; bare AND AUROC and OR TF/FT-versus-FF AUROC; the two OR wording boundaries; pooled/fixed-five macro points and cached draws; sample means, paired contrasts, model-difference arithmetic, interval/count and retention decisions. Paired wording checks cover OR-boundary pooled/macro comparisons for every scalar head, including the unchanged OR support in the AND-wording condition.

Constituent checks explicitly derive FIRST/SECOND targets for AB/BA order, confusion/accuracy, balanced accuracy, joint correctness, Boolean composition and min/max ranking. Pooled/macro all-order records receive point/interval checks; AB/BA records receive point/classification checks. Pooled/macro source-on-target versus target-on-target transfer contrasts and their .90/.05 classifications are independently checked. Individual-topic fields and every other report field do **not** all have a second implementation. Complete replay separately covers the full 83,748-row report. The companion receipt's scope is preserved verbatim.

## Interpretation under the unchanged lock

The original primary remains `R_all − S_all`, pooled bare OR TF/FT versus FF AUROC, TT excluded, separately by model. Samples 11/23/37 are averaged at the metric/contrast level inside each of the 2,000 shared bootstrap draws; raw fitted-head scores are never combined. Both lower bounds exceed zero and the separate .02 practical-gain threshold:

| Locked comparison | Qwen difference [95% interval] | Llama difference [95% interval] |
|---|---:|---:|
| Original primary, pooled | .138637 [.118394, .155166] | .177017 [.149099, .201004] |
| Original primary, fixed-five macro | .151129 [.125737, .172078] | .157926 [.122276, .186238] |
| R_all − S_full, pooled sensitivity | .123162 [.101639, .141084] | .165781 [.140790, .187191] |
| R_all − S_full, macro sensitivity | .127979 [.095949, .160246] | .155114 [.118500, .183739] |

Positive macro intervals separately support the additional equal-topic claim. Macro remains a qualification, not a co-primary, and need not exceed .02; every topic/sample need not be significant. `R_all − S_full` separately supports restored-full-atomic-training robustness and its practical gain, without replacing the successful original primary. Exposure, preprocessing and balanced sampling change together; this is not a pure withholding intervention. `S_full` still uses compound labels and full-training repair was not tested.

Retention is `R_all − frozen reduced-training canonical LR`, evaluated on E. All four required checks per model pass their inclusive negative-margin lower-bound criteria:

| Endpoint / scope / margin | Qwen difference [95% interval] | Llama difference [95% interval] |
|---|---:|---:|
| Atomic pooled / −.005 | −.000104 [−.000284, −.000007] | .000031 [−.000040, .000151] |
| Atomic macro / −.005 | −.000356 [−.001213, −.000002] | .000197 [−.000247, .001802] |
| AND pooled / −.02 | .030830 [.020520, .042241] | .032782 [.025637, .041342] |
| AND macro / −.02 | .044505 [.023995, .071964] | .008153 [.003337, .013167] |

Negative Qwen atomic differences remain negative; support within .005 is not a no-loss claim. A lower bound equal to the loss boundary supports retention; a lower bound below it with an upper bound on/above it is unresolved; an entire interval strictly below it supports loss. Undefined intervals are unresolved. Point retention alone cannot substitute for interval support.

Both models therefore satisfy the locked basic improvement-with-retention criteria, supporting a replicated benchmark claim: under the fixed structured B25 compound allocation, learning a compound-supervised readout adds held-out-entity OR-boundary discrimination beyond compound-guided selection of an atomic-trained readout, with separately supported atomic and AND AUROC retention. Practical magnitude, equal-topic weighting and full-training robustness are separately supported. Different significance labels would not establish a model interaction; no new interaction claim is introduced.

All frozen secondary comparisons remain in the report. `S_full − S_atom_full` has positive pooled/macro intervals in both models, supporting its own selection-label-benefit comparison; `R_all − S_atom_full` is also preserved. Neither is substituted for the original primary. D-selected bank references remain descriptive; no E comparator search occurred.

Unfavorable results are preserved: 8,049 negative scalar contrast points, 217 negative constituent contrast points and 2,475 negative paired-wording points. Existing constituent classifications include 173 point-below-.90 records, 2,458 point-at-least-.90/bound-uncertain records, 60 unresolved .05 transfer records and 20 material-deficit-lower-bound-above-.05 records. These are counts of dependent report rows, **not counts of independent findings or people**; no new scientific analysis was fitted or selected. No undefined point/interval rows occurred in this report, and none were dropped or repaired.

Inference remains 95% pointwise percentile intervals conditional on frozen fits/selectors/support: separate atomic and compound entity frames, PCG64(1729), 2,000 draws, within-frame sharing, endpoint-multiplicity products for compounds, all-five-topic macro and at least 1,800 finite draws. Passing several checks creates no simultaneous or joint atomic–compound 95% coverage. Held-out-entity evidence is within this benchmark, not unseen-topic/unrestricted generalization or unfamiliarity to the pretrained language model. AUROC retention does not establish calibration/behavior; decodability does not establish causal use. Behavior/chat/calibration/causal work and loss-deletion fits remain deferred; the atomic-loss-causes-retention claim is omitted.

## Preservation and delivered evidence

Post-audit verification independently rehashed exact full coverage of the source report (**384 files including completion; 262,218,236 bytes**), reconstruction (**384 files including completion; 262,217,873 bytes**) and accepted raw campaign (**1,530 files; 9,092,446,740 bytes**). Source completion, original controls/receipts, blocker receipt, lock/six pins and all 180 frozen referenced artifacts remain unchanged. All 38 pre-existing worktree HEADs/statuses and the 15 standalone unrelated/DOCX file pins match the baseline. The execution producer is still clean, detached and unchanged. The raw ledger/attempts are included in the unchanged full raw inventory; audit CPU time was not reclassified into GPU accounting.

New small evidence is under `results/final_e_scientific_audit_20261005/`: preflight, launch/process, operation and companion audit receipts, post-verification, exact reconstruction completion inventory, scope coverage, locked interpretation records/principal CSV, historical blocker copy, handoff and inventory. `run_once.py` documents the fresh-path checks and exact single launch; `verify_delivery.py` documents post-preservation and locked interval interpretation only. The source tables, 85 score archives and all large bootstrap/activation arrays remain outside Git in the durable package. No credentials or model weights are committed.

No source evaluation, candidate/final lock, control, execution module, head, preprocessing, selector, sample, threshold, margin or inference procedure was changed. No fitting, reselection, calibration, new scientific endpoint/analysis, repair, retry, D reproduction, raw extraction, budget reset/adoption, RunPod contact or merge occurred. The old blocker and failed historical attempts remain preserved. This delivery records successful completion of the already-approved audit and stops; it does not authorize additional scientific work.
