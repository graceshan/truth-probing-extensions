# Final-E scientific evaluation, October 5, 2026

**Mac 1 evaluation completed; scientific acceptance is pending Mac 2 independent audit.** Under the frozen interpretation rules, both models support positive and practically meaningful original-primary gain, atomic/AND retention at pooled and macro scopes, positive equal-topic macro gain, and restored-full-training sensitivity. These are separate pointwise results, with no joint or simultaneous confidence claim.

The user/project owner approved evaluation followed by independent audit on 2026-10-05, replying **“Yes I approve”** to: “Do you approve these two scientific phases using the accepted handoff and unchanged producer, lock and interpretation rules, with the existing three-hour cap per CPU job and no refitting or reselection?” The new provenance receipt records that question, reply, date, identity and reviewed handoff. Earlier statements that scientific access was unauthorized remain preserved as historical records. Audit execution is left to Mac 2.

## Identities and execution

Publication branch: `checkpoint-final-e-scientific-evaluation-20261005`, based on reviewed raw handoff `e93f3aeb3cc612af03b2704d195bf17406573459`. Publication does not change the execution producer or merge a branch.

| Item | Identity |
|---|---|
| Clean detached execution producer | `b2c6dde71e82f5f0af308761bf19ed3341670587` |
| Execution tree | `93972950bca996dbfa9794dedf07934868b42e98` |
| Final-lock SHA-256 | `88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e` |
| Accepted campaign manifest SHA-256 | `8387b3e319b6c3d9ca9c1037f6f1b46166b6cf536f11bfe494f3ee016291d870` |
| Roots SHA-256 | `a072433410e825bd3c288f304187c901d8da70459f5359212517df4446485645` |
| Evaluate authorization SHA-256 | `7bab7820cd255240ab038ba19c6d78f849c69d316cb85eaa71ae8b169d9c448d` |
| Later audit authorization SHA-256 | `41af4c83bafc4d216b5df836cb6c1c018ecdc3c6659d41d324227aad2bfa6b62` |
| New scientific authorization index SHA-256 | `d9e37364fc688271f86d685e7a757b3424a2da85266eff1354c9d5829eab6d09` |
| Report completion SHA-256 | `632ed93f295027701e1d798a94c758e332c747e071aee410ebb6972114d408bc` |
| Evaluation operation receipt SHA-256 | `d5cfc010293d2207a0a013b7f36ff54af05a1367e294b0275ad772f3b658fb1a` |

Exactly one gated evaluation invocation completed with exit zero and no error. The frozen runner acquired its campaign and shared CPU locks itself, with one numerical thread and the existing 10,800-second cap. Established interpreter: `/opt/homebrew/opt/python@3.11/bin/python3.11`. The runner measured **569.806290 seconds**, including required raw verification; report calculation measured **537.667287 seconds**. The launcher measured **570.033536 seconds** and peak child RSS **3,880,812,544 bytes** (3.88 GB). No model was loaded. No refit, reselection, threshold/preprocessing change, scientific redesign, retry, D reproduction, RunPod contact or audit execution occurred.

The exact argv, environment, PID, exit and resource measurements are in `results/final_e_scientific_evaluation_20261005/evaluation-launch.json`, `evaluation-process.json` and `evaluation.json`. The new controls were generated using the user's exact construction, including independent hash assertions and exclusive creation. Both passed the existing access interface administratively; only `evaluate` was dispatched.

## Locked results, pending independent audit

The original primary is **R_all minus S_all**, pooled bare OR TF/FT-versus-FF AUROC, excluding TT, separately by model. Results are the ordinary mean of per-sample metrics for samples 11, 23 and 37, including within each bootstrap draw. Scores across fitted heads were never averaged or concatenated. Intervals below are 95% pointwise conditional paired percentile intervals; each displayed comparison has 2,000 finite draws.

| Comparison / scope | Qwen difference [interval] | Llama difference [interval] |
|---|---:|---:|
| Original primary, pooled | .138637 [.118394, .155166] | .177017 [.149099, .201004] |
| Original primary, five-topic macro | .151129 [.125737, .172078] | .157926 [.122276, .186238] |
| R_all − S_full sensitivity, pooled | .123162 [.101639, .141084] | .165781 [.140790, .187191] |
| R_all − S_full sensitivity, macro | .127979 [.095949, .160246] | .155114 [.118500, .183739] |
| S_full − S_atom_full, pooled | .100898 [.082369, .119632] | .209221 [.168973, .256574] |
| S_full − S_atom_full, macro | .088179 [.049099, .119028] | .106133 [.057715, .156937] |
| R_all − S_atom_full, pooled | .224060 [.199128, .243863] | .375002 [.348269, .407475] |
| R_all − S_atom_full, macro | .216158 [.180450, .240504] | .261247 [.221970, .297885] |

Primary pooled AUROCs: Qwen R_all .983280, S_all .844643, S_full .860117; Llama R_all .988610, S_all .811593, S_full .822829. Both original-primary paired lower bounds exceed zero and .02. Positive macro lower bounds separately support an equal-topic-weighting claim; macro was not made co-primary, and individual-topic/sample significance is not required. Positive R_all − S_full sensitivity supports survival of the learning advantage after restored full atomic training, with its practical threshold also met. This sensitivity does not replace the original primary. S_full remains compound-guided selection, with advantaged atomic exposure; exposure, preprocessing and balanced sampling change together, and full-training repair was not tested.

Retention compares R_all against the frozen reduced-training canonical LR on E:

| Endpoint / scope / negative margin | Qwen difference [interval] | Llama difference [interval] | Locked assessment |
|---|---:|---:|---|
| Atomic pooled / −.005 | −.000104 [−.000284, −.000007] | .000031 [−.000040, .000151] | Supported, both |
| Atomic macro / −.005 | −.000356 [−.001213, −.000002] | .000197 [−.000247, .001802] | Supported, both |
| AND pooled / −.02 | .030830 [.020520, .042241] | .032782 [.025637, .041342] | Supported, both |
| AND macro / −.02 | .044505 [.023995, .071964] | .008153 [.003337, .013167] | Supported, both |

Qwen's atomic differences are negative and preserved in the report; the intervals support retention within the declared .005 loss margin, not zero loss. Lower bounds at or above the negative margin support retention. Undefined intervals or intervals touching/crossing the boundary from below are unresolved; an entire interval strictly below it supports failure. Secondary endpoints cannot rescue an unsuccessful primary. Different significance labels do not establish a model difference.

Subject to audit, both models meet the locked model-support criteria, supporting the replicated benchmark claim: learning a readout under the fixed B25 structured compound allocation adds OR-boundary discrimination beyond selecting an atomic-trained readout, with separately supported atomic and AND retention. This concerns held-out entities within this benchmark, not unseen topics, model reasoning failure or causal use of decoded information. Retention concerns AUROC, not calibration or behavior. No claim is made that the atomic-loss term causes retention, and no loss-deletion fit was performed.

Inference remains 2,000 PCG64(1729) topic-stratified entity draws, separate atomic and compound frames, shared draws within each frame, endpoint-multiplicity products for compound rows, fixed five-topic macro and the 1,800-finite-draw requirement. All frozen wording, topic, constituent and control analyses remain in the complete external report, including unfavorable results. No undefined or insufficient-interval rows occurred in this evaluation; none were removed or repaired. Passing multiple checks creates no simultaneous 95% coverage.

## Verification, complete report and preservation

Preflight verified the actual safe-stop receipt byte-for-byte against the reviewed delivery, its four supporting receipt hashes and all three independent handoff manifest pins. It verified the producer/tree, all six immutable scientific hashes, source manifests and **all 180 unique referenced fit, parameter, preprocessing and selection-evidence artifacts**, including every selected parameter archive. Membership and pins were checked against the existing registry and reviewed source manifests; the registry was neither rebuilt nor reselected. All **85 heads, 156 scalar aliases and 12 constituent cases** were preserved.

Fresh post-evaluation verification checked every file in the completion inventory (383 files, 262,152,682 bytes, plus the completion file), exact raw campaign coverage and full SHA-256 checksums (1,530 files, 9,092,446,740 bytes), all 37 source controls and all 20 source operation receipts. It rechecked the saved artifacts, scientific lock, six pins, historical transport/local receipts and 15 unrelated untracked files. The detached producer remained clean and unchanged. Raw metadata and settled accounting were not modified: cumulative worker wall 2,946.696421564091 seconds; Qwen 1,043.0739877090673 and Llama 1,128.7931210140814 model seconds; running=false, zero unclean gap.

The original root remains on `t2-research-discussion-packet-20261001` at `d953a8f489cb3286e67f494069e69bdfb4ccfd48`, with its historical untracked files preserved. All existing worktrees, external artifacts, failed attempts, receipts, ledgers, previous controls/indexes and the mistaken-audit-prompt blocker receipt remain preserved. Job inspection found the CPU slot available and no active unrelated substantial numerical job; no process was interrupted. Verification hashing used the shared lock separately after the evaluator released it.

| Complete table | Rows | External filename |
|---|---:|---|
| Scalar metrics | 28,560 | `scalar-metrics.json` |
| Scalar contrasts | 38,556 | `scalar-contrasts.json` |
| Constituent metrics | 9,450 | `constituent-metrics.json` |
| Constituent contrasts | 1,050 | `constituent-contrasts.json` |
| Wording paired | 6,132 | `wording-paired.json` |

All 83,748 result rows and all score/bootstrap arrays remain in the durable external report and are bound by the completion inventory. Git contains the 24-row principal table, exact interpretation records, small receipts/controls and full completion inventory; large arrays and full JSON result tables remain external. Independent reconstruction and principal numerical checks have **not** been run by Mac 1.

## Mac 2 handoff

Use `results/final_e_scientific_evaluation_20261005/evaluation-handoff.json` and the exact published delivery SHA supplied with this delivery. Verify that delivery, the report-completion pin above and every handoff identity before audit. Reuse the already-approved audit control unchanged. The audit reconstruction and receipt paths remain absent; do not overwrite any collision. Mac 1 and Mac 2 share one physical Mac: the audit runner acquires the shared CPU lock itself, so do not wrap it in a second hold. Retain one numerical thread and the 10,800-second cap. No new approval is needed.

Durable package:
`/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1`.

Scientific root:
`/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/scientific-evaluation-20261005-v1`.

The report is `SCIENCE/evaluation`, roots and approved controls are `SCIENCE/control`, and receipts/logs are `SCIENCE/operation-receipts`. The later audit output is `SCIENCE/audit-reconstruction` and its operation receipt is `SCIENCE/operation-receipts/audit.json`. The execution checkout remains `PACKAGE/execution-code`; raw input remains `PACKAGE/handoff/campaign`; lock remains `PACKAGE/transport/final_lock_20261004.json`.

The exact successful evaluation command, run from the detached producer with all numerical thread variables set to 1, was:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m src.final_e_runner evaluate \
  --model both --output "$PACKAGE/handoff/campaign" \
  --roots "$SCIENCE/control/artifact-roots.json" --report "$SCIENCE/evaluation" \
  --operation-receipt "$SCIENCE/operation-receipts/evaluation.json" \
  --authorization "$SCIENCE/control/local-both-evaluate.json" \
  --authorization-sha256 7bab7820cd255240ab038ba19c6d78f849c69d316cb85eaa71ae8b169d9c448d \
  --lock "$PACKAGE/transport/final_lock_20261004.json" \
  --lock-sha256 88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e
```

For Mac 2 only, the approved frozen command uses `audit --model both`, the same raw output/roots/lock, `--source-report "$SCIENCE/evaluation"`, `--report "$SCIENCE/audit-reconstruction"`, `--operation-receipt "$SCIENCE/operation-receipts/audit.json"`, and `--authorization "$SCIENCE/control/local-both-audit.json" --authorization-sha256 41af4c83bafc4d216b5df836cb6c1c018ecdc3c6659d41d324227aad2bfa6b62`. Require complete reconstruction, frozen-head prediction agreement and the pinned separate principal numerical checks; stop on any discrepancy without retry or repair. Preserve the companion audit receipt and state its actual coverage. Mac 1 stops after publication; no audit, merge or further scientific work follows here.
