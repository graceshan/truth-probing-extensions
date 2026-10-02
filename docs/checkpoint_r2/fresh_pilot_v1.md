# R2 bounded fresh-extraction pilot

The fresh pilot is approved. This successor implements only the bounded fresh
representation/cost decision. Adopted A15/B25, source identities, partitions,
raw input construction, model/tokenizer snapshots and layer semantics remain
frozen. Historical packages, bridge/comparator and zero compatibility tolerances
are unchanged. No historical fitted head, preprocessing parameter or conditional
reuse match enters this pipeline. This is not a research result or historical
representation compatibility claim.

Local session: Darwin/macOS 12.6. Origin fetched without pull or merge. Durable
worktree `/Users/apple/projects/checkpoint-r2-fresh-pilot-20261002`, branch
`checkpoint-r2-fresh-pilot-20261002`, exact base
`abc7b8f33c94d323cfbdca83c6790b68894f569f`. All changes add new paths.

## Frozen selection and numerical procedure

Contract: `config/checkpoint_r2/fresh_pilot_v1.json`, SHA-256
`c64b523a4a56e2f1c5e1dc77036ec2f97a7547e17c4d488088809dc307cdfe00`.
Manifest: `data/checkpoint_r2_fresh_pilot_v1/manifest.json`, SHA-256
`e5356a32544acb643784f995be0ae0f493478ae97f94825d5bbe50bc859546f0`.
The runner rejects changed contract or frozen input hashes and reconstructs the
manifest deterministically before launching. Seed **20261002** is used only for
SHA-256 tie ordering within equal-population UTF-8-length bins. No outputs,
scores or truth labels inform selection. UTF-8 length is a selection proxy;
actual tokenizer lengths are recorded remotely and are never inferred from bytes.

Both models use the same **320 distinct raw texts**, with all ordered inventory
bindings retained for identical text. Each of the five topics supplies:

| Subset | Per-topic quotas | Total |
| --- | --- | ---: |
| Train-only calibration | 4 P15 atomic; 3 each T_C AND/AB, AND/BA, OR/AB, OR/BA | 80 |
| Independent verification | 6 P15 atomic; 6 atomic D; 3 each D bare AND/AB, AND/BA, OR/AB, OR/BA; 8 each of the three wording variants | 240 |

Subsets are disjoint by exact text ID, not an entity-disjoint research split.
They serve numerical calibration and independent numerical verification only.
Repeated research groups remain bindings; timing repeats remain repeats.
E, behavior and chat inputs are never selected.

The grid is **1/2/4/8**, contiguous manifest order, right padding for batches >1;
batch 1 is unpadded. Native special tokens are included, raw text has no chat
framing, truncation is prohibited, explicit position IDs are mask cumsum minus
one with zero padded positions, readout is the final nonpadding token. Token
sequences with special tokens removed must equal native no-special-token
encoding; separate single/batched tokenization APIs must agree. IDs, source
text hashes, ordered group bindings, masks, semantic positions, all-layer shape,
BF16 computation and finite FP16 storage are mandatory exact checks.

Saved layer s = HF `hidden_states[s+1]`. Embedding is excluded. Intermediate
states match decoder block residual outputs; final state matches final RMSNorm
output. First inference checks actual block/final-norm hooks, then removes them.
Qwen has 28 x 3,584 features and Llama 32 x 4,096; all are saved as FP16.

After three separately reported warm-ups (eight calibration inputs each), each
model produces a batch-1 calibration reference and grid comparisons, including
a batch-1 repeat. Per layer it reports maximum absolute error, RMS error and
RMS(delta)/max(RMS(reference),1e-6). Computation is exact BF16 widened to FP32;
FP16 comparisons and each side's cast error are reported separately. Compute
arrays are held in memory for comparison; retained feature tensors are FP16.

Calibration is bounded by relative RMS <= .01 and maximum absolute/reference
RMS <= .10. Every layer's calibrated max/RMS limits are max(3 times calibration
error, reference RMS times 1e-5/1e-6 respectively); relative RMS uses max(3 times
calibration relative RMS,1e-6). Factor three provides a prospective allowance for
sample/kernel variation; small fixed floors avoid a universal bitwise condition
when a calibration estimate is zero. Hard ceilings prevent a large calibration
error authorizing large verification errors. These are engineering tolerances
for fresh batching, not scientifically established equivalence thresholds.

An immutable `locked-calibration.json` records evidence, limits, procedure and
both subsets **before any verification inference**. Independent verification
must meet every calibrated limit and the hard ceilings for compute and stored
features. Limits cannot widen. Calibration-ceiling/OOM candidates are excluded;
each declared batch is tried at most once per subset. Any independent batching
failure selects verified batch 1. Independent batch-1 failure stops for a specific
correctness fix. Otherwise choose the largest verified batch. There is no
padding/backend/grid search after failure.

## Runtime, storage and bounds

The fresh supported baseline is Python **3.12.x**, PyTorch **2.8.0+cu128**,
Transformers **4.56.2**, tokenizers **0.22.1**, NumPy **2.1.3**, safetensors
**0.6.2**, huggingface-hub **0.34.4**. It requires a visible CUDA BF16 GPU,
not a particular historical GPU. SDPA's **math** backend is explicit; flash and
memory-efficient backends are disabled. TF32 is disabled and FP32 matmul
precision is highest. This bounded baseline prioritizes a supported declared
execution path; a faster backend is a separate future contract, not tuning in
this pilot. If the supplied GPU cannot support this baseline, resolve that
specific runtime prerequisite locally and push a reviewed successor before
inference; never silently execute a different configuration.

The new virtual environment records pip freeze and checks dependencies. Before
inference the worker verifies actual package pins and saves exact Python,
Torch build, CUDA/cuDNN, driver, GPU name/UUID/VRAM/capability, attention policy,
actual execution flags, implementation hashes, tokenizer backend hash and token
policy. Resolved model/tokenizer files and every safetensors shard/index are
hashed and recorded. Both revisions are checked by model metadata; authenticated
access is separately established by force-downloading Llama's protected
`config.json` at its pinned revision, checking its resolved snapshot and frozen
hash, and publishing `llama-protected-file-access.json` **before either model's
large weight download**. Public `model_info` success alone cannot pass this gate.
The receipt contains only model/file identity, checksum, size, timing and outcome;
a denied request retains error type/HTTP status without server text or headers.
The authentication gate is regression-tested against pinned Hub **0.34.4**.

Pinned model identities:

| Model | Revision |
| --- | --- |
| Qwen/Qwen2.5-7B-Instruct | a09a35458c702b33eeacc393d103063234e8bc28 |
| meta-llama/Llama-3.1-8B-Instruct | 0e9e39f249a16976918f6564b8830bc894c89659 |

The cache and output are new external directories. No old snapshots are repaired
or overwritten. Llama gated access uses the operator's existing authorization;
a token is never included in logs, command arguments or receipts. Before assigning
the new `HF_HOME`, the wrapper resolves and exports the **original absolute
`HF_TOKEN_PATH`**. Existing active-token files stay in their external credential
location; inherited `HF_TOKEN` (or the legacy environment alias) continues to
work. Credentials are not copied to output/cache directories or Git. The wrapper
disables shell tracing before handling authentication, including when invoked
with `bash -x`.

If no authorized credential is available, use this exact procedure in an
interactive **Bash terminal on the GPU host, before the first pilot attempt**.
Use a read token for the Hugging Face account already approved for the gated
Llama repository; model access approval must be granted before execution.
Replace `PILOT_COMMIT` with this delivery's exact pushed successor SHA and use a
new root whose durable parent exists:

```bash
set +x
IFS= read -r -s -p 'HF read token with approved Llama access: ' HF_TOKEN
printf '\n'
export HF_TOKEN
bash scripts/checkpoint_r2_fresh_gpu_setup.sh /durable/r2-fresh-run-20261002 PILOT_COMMIT
unset HF_TOKEN
```

The token is entered at the hidden prompt, stays in process environment memory,
and is never typed into a command or saved by this procedure. Do not paste it
into chat, supply `--token`, or enable shell tracing. After a failed campaign,
inspect its retained accounting and authentication outcome before any reviewed
retry; this procedure does not reset execution limits.

The writer publishes immutable NPY/rows/tokens/receipt files using fsynced
partials, exclusive hard links and directory fsync. A new downstream loader
`src.checkpoint_r2_fresh_store.load_features` validates hashes, payload values,
ordered composite row bindings, model/provenance, dtype and every layer axis.
Every stage writes a contiguous shard of <=320 rows and reopens/checks its exact
values; production timing uses this same path. There is no legacy-fit loader.

Before download/inference a practical **4 GiB incompressible write/hash/readback**
tests the actual destination plus exclusive publication, overwrite refusal,
orphan removal, atomic replace and directory fsync. Reported virtual free space
is informational. This proves only bounded write/readback and process-level
recovery operations, not full extraction quota or power-loss durability. A full
production allocation/quota check remains a prerequisite to later authorization.
Model download capacity is demonstrated by the actual bounded pinned downloads.

Three steady-state timing passes traverse all 320 inputs with the chosen policy.
Each includes tokenization, all-layer GPU execution, CPU transfer, contiguous
FP16 serialization/publication, hashes and downstream reopen/value checks.
Per-batch forward/transfer durations and per-shard writing/check durations are
reported separately; end-to-end rates, mean, sample standard deviation and the
three observed rates are retained. Warm-ups, setup/download and repeated timing
inputs are separate from unique observation count. Peak allocated/reserved VRAM
covers loading and pilot execution. CUDA peak tracking resets **once per model,
immediately after selecting cuda:0 and before any model loading**; it never
resets after loading. Transient device-transfer/loading peaks remain visible
through warm-up, calibration, verification and timing. An eight-row `transfer-ready/` shard per
model carries real values and independently supplied receipt checksum for mac 2.

Qwen and Llama run sequentially. The independent process-group watchdog enforces
**3,600 seconds per model** from model-load start through teardown (conservative
GPU wall accounting, including host work) and **7,200 active seconds combined**.
Setup/download are separate; a one-hour setup-phase timeout bounds hangs without
increasing active/GPU allowances. Total instance-billable wall includes setup and
download, with provisioning timestamps recorded by the setup script; queue and
provider-specific billing boundaries remain external observations. The campaign
sentinel refuses automatic reruns and budget resets after any attempt. Partial,
failure, last budget and supervisor receipts remain available. Stop and review a
failure/cap; changing output parent is not permission to reset the budget.

## Inventory and projection

Frozen raw inventory is **35,843 unique inputs/model**, **41,490 logical bindings/model**.
Qwen FP16 payload = **7,193,833,472 bytes**; Llama = **9,396,027,392 bytes**;
combined = **16,589,860,864 bytes**, or **16.59 decimal GB / 15.45 GiB**.
All raw layers are included; behavior (1,900 unique messages/model) and chat
(11,578/model) are separate and excluded from this projection.

For each model and measured timing repeat, projected raw hours =
35,843 / measured end-to-end inputs-per-second / 3,600. Report the mean hours
and observed range, not a guaranteed bound. Balanced topic/length pilot coverage
supports extrapolation; full-inventory tokenizer-length/composition differences
and rate tails remain a limitation. The timing already includes per-shard
integrity/reopen verification. Include actual download/hash and model-load time
separately. Production storage planning includes weights and tokenizer/cache,
largest temporary weight download, one temporary <=320-row shard, metadata,
one complete output copy, one retained destination copy and actual retained
pilot bytes. Each complete copy costs the 16.59 GB combined tensor payload plus
metadata; transferring one copy moves that many bytes. Transfer time must use
observed sustained transfer rate, not GPU inference throughput.

Use the supplied hourly price if available; otherwise report GPU-hours and
`price * (raw hours + setup/download + billed transfer/retention + provisioning)`.
No hourly price was supplied in the local task. No dollar estimate or measured
GPU throughput is fabricated.

## Current status and execution commands

Local implementation and focused/regression tests pass. The initial receipts in
`results/checkpoint_r2_fresh_pilot_v1_20261002/` remain the preserved **ac89d894**
delivery snapshot, including its original per-file hashes. The focused
successor starts exactly at `ac89d894ebdc0c596bf364cde2177aa0264a0a7b`; it changes
only authentication/access gating, loading-peak tracking, the setup wrapper and
handoff, with regression coverage. Frozen contract/input bytes, tolerances and
research design are unchanged. Current fix/test/preservation receipts are in
`results/checkpoint_r2_fresh_pilot_auth_peak_fix_20261002/`. These receipts
describe local metadata and synthetic validation only. No new GPU SSH connection was supplied. The old
address was not contacted or assumed to have acquired a GPU.

| Model | Unique pilot inputs | GPU/runtime | Accepted policy | Real correctness / throughput / VRAM / bytes / projected cost |
| --- | ---: | --- | --- | --- |
| Qwen | 320 frozen | Pending supplied connection | Pending verification | Not executed / unmeasured |
| Llama | 320 frozen | Pending supplied connection | Pending verification | Not executed / unmeasured |

Recommendation now: **hold complete extraction authorization until the real
bounded pilot supplies correctness and cost evidence**. On success review its
measured per-model costs; on failure fix the recorded correctness/runtime issue;
if measured cost exceeds the supplied budget, reconsider scope. No full extraction
is launched automatically.

Local rerun (use a fresh dry-run receipt path):

```bash
cd /Users/apple/projects/checkpoint-r2-fresh-pilot-20261002
/tmp/clean-extraction-venv/bin/python -B -m src.checkpoint_r2_fresh_pilot dry-run \
  --output /private/tmp/r2-fresh-review-new.json
/tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_checkpoint_r2_fresh_pilot.py tests/test_checkpoint_r2_fresh_runtime_fixes.py \
  tests/test_checkpoint_r2_bridge_runner.py \
  tests/test_checkpoint_r2_common_v1.py tests/test_checkpoint_r2_inputs.py \
  tests/test_checkpoint_r2_selection_v1.py tests/test_clean_atomic_extraction.py \
  tests/test_pinned_compound_extraction.py
```

After the new SSH connection is provided, verify its identity, then provision a
new remote checkout using the pushed commit reported with this delivery. Do not
pull, reuse an active checkout or change existing artifacts. For example, from a
remote repository whose origin is already authenticated:

```bash
# Replace PILOT_COMMIT with this delivery's exact pushed 40-character SHA.
git fetch origin checkpoint-r2-fresh-pilot-20261002
git worktree add --detach /durable/r2-fresh-code-20261002 PILOT_COMMIT
cd /durable/r2-fresh-code-20261002
# NEW_ROOT must not exist; its parent must be durable and have adequate cache space.
# Optional third argument is the actual supplied hourly dollar price.
bash scripts/checkpoint_r2_fresh_gpu_setup.sh /durable/r2-fresh-run-20261002 PILOT_COMMIT
```

The wrapper installs the pinned environment into NEW_ROOT, verifies dependencies,
and executes the exact pushed code. The runner itself records GPU/access/storage
once for this new host. If environment is already provisioned by this wrapper
and execution has never begun, its direct invocation is:

```bash
/durable/r2-fresh-run-20261002/venv/bin/python -B -m src.checkpoint_r2_fresh_pilot run \
  --expected-commit PILOT_COMMIT \
  --output /durable/r2-fresh-run-20261002/pilot \
  --cache /durable/r2-fresh-run-20261002/model-cache
```

This is not a retry command after an attempt. Any retry requires the preserved
campaign accounting and review, never a fresh budget. Runtime and manifest
receipts must exist before first inference. After completion, transfer both
`pilot/MODEL/transfer-ready/` directories and the independently obtained checksums
from `MODEL/result.json` or `artifact-inventory.json`. Large tensors stay outside Git.
On mac 2, from this same pushed revision:

```bash
python -B -m src.checkpoint_r2_fresh_pilot check-shard \
  --output /durable/received/qwen/transfer-ready \
  --expected-receipt-sha256 RECEIPT_SHA256_FROM_REMOTE_RESULT
```

Run the corresponding Llama check separately. These commands open actual NPY
values through the downstream loader. An independent machine's successful check
is reported only after that machine actually executes it.
