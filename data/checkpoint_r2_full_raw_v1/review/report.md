# R2 completed fresh-pilot independent review — 2026-10-02

**PASS within scope.** Evidence supports complete fresh raw extraction at **batch
size 1** for both models. No completed-handoff blocker was found. Full extraction
still requires separate authorization and verified usable full storage quota for
the retained-copy plan. This review authorizes no execution.

Origin was fetched, then this isolated branch,
`checkpoint-r2-fresh-pilot-review-20261002`, was created exactly from
`3f9747f12b46dcb956082c22cbbfd2145c894b9d`. All checks ran at that exact HEAD before
committing the review. Earlier worktrees were not pulled or merged; code,
tolerances and batch settings were unchanged.

Actual host: **MacBook-Pro-7.lan**, macOS 12.6 arm64. Invoked
`/private/tmp/clean-extraction-venv/bin/python`; actual `sys.executable` was
`/opt/homebrew/opt/python@3.11/bin/python3.11`, Python **3.11.6**, NumPy **1.26.2**.
Resolved interpreter/prefix are recorded in the receipt. This CPU loader
environment differs from the recorded GPU runtime. Earlier Mac session records
identify the same host: describe this as **independent review/re-execution**,
without claiming second physical-machine replication. No SSH, GPU execution,
model loading, fitting or scoring occurred.

Source root:
`/Users/apple/projects/checkpoint-r2-fresh-pilot-handoff-20261002-3f9747f-volume`.
Archive SHA-256 matches the supplied pin:
`d85fcbb5bf960575bf8251abb861aa049d5e0e12a728ad1d3f60aab79137e43d`.
All **202 members** were inspected before extraction: unique relative paths under
`handoff-completed`, regular files/directories only, without traversal or links.
Extraction used a fresh external directory; existing handoff files were preserved.

All **143 checksummed files** passed sizes/hashes; the only additional file is the
checksum manifest itself, SHA
`694af40a6cf0ac76fbf51cff3e055d15447f680f2a18ccdd7b74c968af8a936a`.
Of **153 pilot inventory entries**, **127** were directly verified and **26** are
larger activation files intentionally absent from the transfer. The **17 preserved
setup files** also pass their separate manifest. The operator's previous-handoff
pointer matches the earlier initial-handoff/pending-authentication receipts; it
is distinct from the current volume setup manifest.

`verify-handoff-shards.sh` was read, hashed, executed at exact HEAD and rehashed
unchanged. SHA:
`00a025e313e87aefbf3c93fe01c3718a2fdf8fd48c73c6fe160e998276f6ee31`.
Both independently supplied shard receipt pins and downstream loader checks pass.
No external expected script SHA was supplied or found; this checksum identifies
the inspected script, which is outside the archive manifest. Additional loader
checks against frozen expected rows/models and token semantics pass for finite
FP16 arrays: Qwen **[8,28,3584]**, Llama **[8,32,4096]**.

Frozen contract/input hashes and deterministic manifest rebuild pass:
**80 train-only calibration + 240 disjoint verification texts/model**, with all
ordered source bindings preserved. All stage receipts/token packets agree on
native special tokens, masks, positions and final-nonpadding readout. Saved layer
s = HF hidden_states[s+1], embeddings excluded, final layer post-final-RMSNorm.
Actual block/norm checking is supported by the exact runner path; no inference
was repeated here.

Reported model/tokenizer revisions match: Qwen/Qwen2.5-7B-Instruct
`a09a35458c702b33eeacc393d103063234e8bc28`; meta-llama/Llama-3.1-8B-Instruct
`0e9e39f249a16976918f6564b8830bc894c89659`. Pinned config/tokenizer hashes,
weight/index identities and backend implementation hashes are bound in the
receipt; weights were not transferred or rehashed here. Reported runtime/freeze
matches Python **3.12.3**, torch **2.8.0+cu128**, transformers **4.56.2**,
tokenizers **0.22.1**, numpy **2.1.3**, safetensors **0.6.2**, huggingface-hub
**0.34.4**, NVIDIA A40, CUDA 12.8, cuDNN 91002, driver 595.91.07. BF16/FP16,
math SDPA and disabled TF32/flash/memory-efficient SDPA match; pip check passes.

Per-layer calibration formulas independently recompute: multiplier 3,
max/RMS floors 1e-5/1e-6, relative RMS floor 1e-6. Lock hashes match accepted
policies; widening is prohibited. Exact code publishes/rechecks the immutable
lock before/after verification. Final bindings establish consistency; flattened
archive timestamps do not independently prove chronology.

Batches **2/4/8 fail both hard ceilings**: relative RMS **0.01**, scaled maximum
(max absolute error/reference RMS) **0.10**. Maxima across layers below apply to
both computation and stored FP16.

| Model | Batch | Relative RMS | Scaled maximum |
| --- | ---: | ---: | ---: |
| Qwen | 2 | 0.022149053 | 1.733053920 |
| Qwen | 4 | 0.024390497 | 1.387055523 |
| Qwen | 8 | 0.023877325 | 1.104535713 |
| Llama | 2 | 0.013337896 | 0.629606839 |
| Llama | 4 | 0.017015538 | 1.587365095 |
| Llama | 8 | 0.016405148 | 2.184264259 |

Only batch 1 has calibrated limits/verification. Independent **240-row batch-1
verification reports zero computation/stored-FP16 error** and passes all fixed
limits/ceilings. Reference/repeat stored tensor hashes agree. No new configurations
or excluded-batch verification were attempted. Supervisor/operator exits and caps
pass.

Timing recomputation uses **320 / end-to-end seconds**, sample SD with n−1, and
raw hours **35,843 × seconds / 320 / 3,600**, averaged across repeats. Three timing
passes repeat 320 inputs; they are not 960 independent observations.

| Model | Repeat seconds | Mean ± sample SD inputs/s | Raw hours mean [observed range] |
| --- | --- | --- | --- |
| Qwen | 31.783425, 19.784934, 27.142763 | 12.677193 ± 3.148194 | 0.816332 [0.615583,0.988900] |
| Llama | 23.846541, 26.659308, 25.714270 | 12.622300 ± 0.724474 | 0.790497 [0.741954,0.829470] |

Combined raw forecast: **1.606828828 h**, summed observed extrema
**[1.357537292,1.818370527] h**. Load/integrity adds **0.028952029 h**;
download/hash **0.164865931 h**, provisioning **0.167777778 h** when applicable.
Campaign supervisor **0.309055361 h** and safe-stop container age **1.017069444 h**
recompute. Exact provider billing/price are unavailable. Ranges are extrapolations,
not confidence bounds; full-input lengths/composition/rate tails are unmeasured.
The 4,334,176-byte transfer rate recomputes to 6,254,487.720 bytes/s, implying
**0.736798103 h** for full tensor payload, without proving sustained transfer speed.

All stage output-byte calculations pass, including 128-byte NPY headers and
metadata. 320-row output bytes: **65,201,736 Qwen**, **84,879,906 Llama**.
Inventory **879,173,971** + four control/post-inventory files **66,905** = retained
pilot **879,240,876**. Raw inventory counts **35,843 texts/41,490 bindings/model**.

| Storage component | Bytes |
| --- | ---: |
| Weights/tokenizers | 31,312,510,114 |
| Two complete tensor copies | 33,179,721,728 |
| Estimated metadata per complete copy (two counted) | 220,690,055.39375 |
| Largest temporary weight file | 4,999,802,720 |
| One temporary shard payload | 83,886,080 |
| Retained pilot | 879,240,876 |
| Estimated total | **70,896,541,628.7875** |

One combined tensor copy is **16,589,860,864 bytes**. Metadata is extrapolated
from measured shard overhead. Recorded allocation/readback is **51,539,607,552
bytes (48 GiB)**; full quota remains unverified. Shared-pool statvfs and declared
100 GB volume do not prove usable capacity for the 70.90 GB plan.

**Directly inspected feature arrays are only the transferred eight-row shards
per model.** Full-pilot numerical findings are consistency checks on recorded
per-layer metrics/receipts, not recomputation from absent tensors or new inference.
This supports batch-1 engineering readiness within scope, without establishing
historical representation compatibility or research results.

Committed small receipts: [checks](../../results/checkpoint_r2_fresh_pilot_review_20261002/checks.json),
[arithmetic](../../results/checkpoint_r2_fresh_pilot_review_20261002/arithmetic.json),
[dry run](../../results/checkpoint_r2_fresh_pilot_review_20261002/dry-run.json).
Full audit receipt/script, member listing and extracted tensors stay outside Git
at `/Users/apple/truth-probing-backups/20261001T164224Z/r2-review-evidence-20261002`,
with identities/paths recorded in the check receipt.
