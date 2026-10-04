# Final-E raw acquisition and durable Mac handoff, October 4

**Accepted raw handoff; scientific evaluation, scoring, metrics and scientific audit remain unauthorized.** The reviewed control-copy recovery succeeded, the eleven approved phases completed sequentially, and campaign workers are stopped. The user can stop RunPod. No new remote task follows the safe-stop checkpoint.

## Recovery and exact code identities

Base delivery: `e41184ff8c7be8ce6079df2533426582e6230f2b`. Results branch: `checkpoint-final-e-raw-transport-recovery-20261004`. This is an additive results delivery; it does not change the execution producer or merge any branch.

All 28 existing remote regular control files were rehashed against their independently pinned local sources and published failure inventory, with exact file coverage, bytes and sizes and no unexpected entries. They were reused in place. Four missing bridge/inventory/accounting files were copied with `rsync -r --ignore-existing`; all four commands returned zero and every destination was independently rehashed. The original archive-mode exit-23 failure remains failed and preserved. No ownership, group, permission, timestamp replay or deletion option was used. The same compatible transport options were used for durable artifact copies.

The bridge and unchanged original bundle passed normal prerequisite verification, fresh-ref imports and full strict object-integrity checks. The recipient retained its original HEAD/worktree. The created execution checkout was clean and detached. Actual import evidence is in `results/final_e_raw_recovery_20261004/recipient-import.json` and the durable control copy.

| Identity | Exact value |
|---|---|
| Producer | `b2c6dde71e82f5f0af308761bf19ed3341670587` |
| Producer tree | `93972950bca996dbfa9794dedf07934868b42e98` |
| Final-lock SHA-256 | `88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e` |
| Bridge SHA-256 | `0fb71490a922b3bf5bb29ae233a3c60d1129b9b58ae2405601f6c1823ea67341` (1,578,104 bytes) |
| Original bundle SHA-256 | `da28abe3c64e50bffee3047003a621ee29b60dd46ceafd3052157ae796b7a1aa` (176,023 bytes) |
| Remote approved-index SHA-256 | `86ccbb56dc882ec42c7865b1b590c4a42703c8ac3c286d74352eb3c3dd33c908` |
| Local approved-index SHA-256 | `8e3ee5103379a8ccd3805439deb1ffeb3fbf9fddb2bbcf3fbf1573bd187362e5` |

The six immutable scientific pins, final lock, original inventories and all eleven approved records remain byte-identical. Controls were reused, never regenerated. All eleven passed the existing access interface without changing gates. Scientific and execution approval provenance remains the actual user instructions recorded in the preserved controls.

## Acquisition and measured resources

Qwen extract → token_audit → accept → verify completed before Llama extract → token_audit → accept → verify. Both-model transfer publication completed after accepted models and settled accounting. Commands, exact pins and scope/output paths are in `executed-phases.json`; supervisor commands are preserved in `supervised-sequence.sh`. Both supervised attempts returned zero, without interruption or cap hit.

| Model | Coverage / shards | Worker wall seconds | Forward/write/verification seconds | CUDA peak allocated / reserved bytes |
|---|---|---:|---:|---:|
| qwen | 19,190 unique texts; 19,424 logical bindings; 150 shards | 1273.916462 | 863.130429 | 15,298,184,704 / 15,424,552,960 |
| llama | 19,190 unique texts; 19,424 logical bindings; 150 shards | 1373.099962 | 936.055650 | 16,085,015,040 / 16,234,053,632 |

Both live attempts passed the frozen runtime/dependency/freeze/driver, offline snapshot/tokenizer, native tokenization, filesystem and actual reservation/fsync/readback checks before inference. Full per-stage costs and receipt pins are in `measurements.json`. No extra model test, runtime replacement, package installation, snapshot download, acquisition retry or procedure modification occurred.

qwen: actual allocation/readback reserve **28,437,984,635 bytes**, measured in 277.289429 seconds.
llama: actual allocation/readback reserve **28,447,937,454 bytes**, measured in 268.245989 seconds.

Storage evidence establishes a bounded reservation only; full production quota and power-loss durability remain unestablished. Owner-only protection on the non-secret artifact mount remains untested. The package does not claim broader filesystem guarantees.

## Accounting and caps

Pre-adoption reconciliation reverified 29 remote metadata records against known local pins and four unique finished carried attempts. No intervening additional, missing or inconsistent record was found. Normal adoption occurred only through the authorized launcher. The carried baseline remains Qwen 160.83801023702836 seconds and Llama 123.82646256202133 seconds.

There were exactly two new acquisition attempts, both successful, and zero new failed acquisition attempts. The prior control-copy failure and failed packaging attempts remain preserved; they did not become acquisition attempts. Full attempt identities, supervisor settlements and the unchanged adoption evidence are in `accounting.json`.

Final settled cumulative extraction-worker wall: **2946.696421564091 seconds**. Cumulative model charges: Qwen **1043.0739877090673**, Llama **1128.7931210140814** seconds, including carry-forward. The ledger has `running=false`, `model=null` and zero unclean-gap charge.

Caps remain 7,200 seconds/model, 18,000 cumulative extraction-worker seconds including carried charges, and 1,800 seconds/setup and non-extraction operations. The separate 10,800-second scientific CPU-job cap was not used for transfer/loader operations. No ledger or charge was reset.

## Durable transfer, acceptance and safe-stop

Campaign, operation-receipt and control manifests remain distinct. Source hashes were captured independently through the strictly verified connection before copying. Control coverage includes the bridge, exact inventory, actual recipient-import receipt and recovery evidence. Sealed source evidence was preserved; later worker observations were written separately.

| Manifest | SHA-256 | Files / file bytes |
|---|---|---|
| campaign-checksums.json | `8387b3e319b6c3d9ca9c1037f6f1b46166b6cf536f11bfe494f3ee016291d870` | 1,530 / 9,092,446,740 |
| operation-receipt-checksums.json | `f759df0f3dcb59067bc014e4f7fa39d224fdc9686538ac07432eeaaf55360896` | 20 / 13,997 |
| control-checksums.json | `654320071ca6d09760bf68736dffd260932fe53452901236a42d939178dc89ab` | 37 / 1,899,044 |

Durable copy measured **9,094,359,781 campaign/control/receipt file bytes**, plus **244,503 manifest bytes**, over **611.059227 measured monotonic seconds**, approximately **14.883 MB/s** for campaign/control/receipt bytes. Logs retain per-copy duration and successful return codes; no automatic transfer retry occurred.

Durable package root: `/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1`.
- Raw campaign: `/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/handoff/campaign`.
- Source controls: `/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/handoff/remote-control`.
- Source operation receipts: `/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/handoff/remote-operation-receipts`.
- All three manifests and independent source pins: `/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/handoff`.
- Local checksum, full-loader and safe-stop receipts: `/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/local-operation-receipts`.
- Recovery/import/accounting transport observations: `/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/recovery-receipts-20261004-v1`.

Independent manifest pins, exact control/receipt coverage, file types, containment, sizes and checksums passed. Gated local `verify-transfer --model both` then `verify --model both` passed from the exact clean detached Mac producer with the existing local approved scopes. Both retained their 1,800-second operation caps. Mac verification used one numerical thread and the shared exclusive CPU lock after job inspection. No unrelated process was interrupted.

The final observation confirmed both recorded supervisors absent, no campaign runners/workers or CUDA compute processes, unchanged source manifests, and settled accounting identical to the durable Mac ledger. The safe-stop receipt binds all supporting hashes. **Raw acquisition and durable Mac handoff are accepted; campaign workers are stopped; RunPod can be stopped by the user. Scientific evaluation and audit remain unauthorized.**

## Preservation and stop

Historical controls, drafts, lock/scientific inputs, bundles/inventories, original exit-23 command/log/receipt, failed packaging attempts, accounting records, worktrees, external artifacts and unrelated files were preserved. Historical file pins were checked again; the original continuation worktree and local detached producer remain clean. No completed fits, saved-D reproduction, wording diagnosis or scientific numerical audit was rerun. Private connection details, credentials, model data and large arrays are outside Git.

A local job-inspection diagnostic returned 1 because its filter counted its own PID (61725) at 5.6% CPU. The corrected observation found no unrelated substantial numerical job. Its process-label diagnostic also matched its own parent shell; that observation was not a campaign-worker finding. Both gated verification operations ran once under the shared exclusive lock and passed; no scientific or access check was weakened, and no phase was retried. The diagnostic evidence remains preserved. No unresolved raw-handoff blocker remains. Publication ends this task; no merge, scientific evaluation, scoring, metrics or scientific audit follows.
