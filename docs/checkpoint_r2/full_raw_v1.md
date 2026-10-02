# R2 complete fresh raw extraction: preparation for review

Prepared locally on macOS in the isolated `checkpoint-r2-full-raw-prep-20261002`
branch, based exactly on `3f9747f12b46dcb956082c22cbbfd2145c894b9d`.
Origin was fetched; no pull or merge occurred. The stopped GPU Pod was not
contacted, restarted or used. Full extraction remains unlaunched. No fitting,
behavior/chat evaluation, E scoring, historical compatibility claim or research
result is supplied by this preparation.

## Evidence and representation

Completed handoff: `/Users/apple/projects/checkpoint-r2-fresh-pilot-handoff-20261002-3f9747f-volume`.
Pinned archive SHA-256: `d85fcbb5bf960575bf8251abb861aa049d5e0e12a728ad1d3f60aab79137e43d`.
Mac 2 review: **`1dbb5484d1055e0cf80993e666171cfa1743e2a2`**, fetched/read at that exact
commit without merging. Its report and three receipts are copied byte-for-byte
under `data/checkpoint_r2_full_raw_v1/review/`, with hashes in the successor
contract. Verdict: PASS_WITH_SCOPE_LIMITATIONS, no completed-handoff blockers.
The report explicitly describes independent CPU review/re-execution on the same
reported physical hostname; second physical-machine replication is not claimed.
The accepted review is the loader prerequisite; no additional hostname gate is
imposed on the user's accepted review.

`config/checkpoint_r2/full_raw_v1.json` SHA-256:
`ed15c662289d587cc9328bc43d980b42afe3d39e4dc352e334e0f504ebcd4ec1`.
Complete reconstructed manifest SHA-256:
`f28e75f5fc95cf3f019f5b47c68c5c2753435854280bfb122f0df22187954445`.
Executable source hashes and validation receipts are in
`results/checkpoint_r2_full_raw_prep_20261002/delivery_manifest.json`.

`src/checkpoint_r2_full_inputs.py` reconstructs actual CSV coverage: **35,843
unique texts and 41,490 logical source/group bindings per model**. It verifies
exact source text, text hash, source/group identities, fact order, partition and
labels as metadata bindings. Unique texts follow first inventory occurrence;
every binding retains its original CSV offset and source metadata hash.
The global ordered binding list is preserved alongside each row's binding list.
Derived counts reconcile with the frozen resource inventory and payload sizes.
All 2,000 behavior and 11,578 chat bindings are excluded; unknown/E raw groups,
E/test partitions and extra/missing identities fail closed. Frozen A15/B25 and
research manifests are preserved.

Both models always execute **batch 1, unpadded**. The preserved pilot backend,
semantic checker, writer and loader are reused without edits. Native special
tokens, raw text without chat framing, explicit positions, last nonpadding
readout, no truncation, BF16 computation and FP16 storage remain fixed. Saved
layer s = HF hidden_states[s+1]; embeddings excluded; final layer post-RMSNorm.
The decoder-block/final-norm hooks check actual layer semantics on the first new
forward of each model loading interval. There is no batching/tolerance search,
probe adoption, historical preprocessing or reuse of pilot feature observations.

First use rechecks exact Python 3.12.3, the recorded complete pip freeze (including
pandas 2.3.3), CUDA 12.8/cuDNN 91002, A40/device memory/capability and driver
595.91.07. Backend class/source/tokenizer hashes and execution flags must equal
the pilot receipt. Device UUID/hostnames are newly recorded. Any execution
runtime drift stops for review. These restrictions preserve this validated
baseline; a different supported GPU/runtime requires a separate amendment.

The pilot's Qwen/Llama snapshot receipts pin every model/tokenizer/index/weight
file. The successor hashes their real offline cache files and checks file sets,
revision, index membership and resolved cache containment before loading.
**No download function is called.** Missing/corrupt cache stops; no automatic
repair or replacement occurs. Validated offline reuse requires no new token.
`FreshBackend` rehashes again before loading and uses `local_files_only=True`.
Peak-memory tracking includes loading and execution, separately for each attempt.

## Output and recovery

Suggested fresh output: `/workspace/checkpoint-r2-full-raw-20261002`.
It is outside the checkout, pilot runtime/output and model cache. The immutable
external input manifest contains all texts/bindings (approximately 87 MB); only
its summary/hash is stored in Git. Each model has **113 bounded shards**, 112
with 320 rows and the final shard with 3 rows:
`MODEL/shards/00000` through `MODEL/shards/00112`.

Each shard contains FP16 `activations.npy`, ordered `rows.json`, exact physical
`tokens.json`, the existing loader `receipt.json`, and a successor `completed.json`.
Publication uses the pilot's fsynced exclusive hard-link writer. A completion
marker follows successful reopening and token/row checks, and binds the receipt
hash. Final portable relative-path indexes bind all shards, markers, axes and
exact complete counts. `verify-output` reopens the complete outputs after copying and checks the published
manifest, completion scope, model identities and absence of extra/missing shards.
Second-copy publication itself is a separate operation, not performed by `run`.

On explicit `--resume`, every published shard is rehashed and reopened with the
expected frozen rows, model, provenance and token-plan hash before reuse.
A valid receipt without a marker is verified and adopted without another forward.
A directory without a published receipt is atomically moved to uniquely named
`quarantine/MODEL/`; its partial bytes are preserved, then that shard is recomputed.
Published corruption, changed runtime/provenance, extra shards or wrong bindings
stop without replacing artifacts. Valid outputs and immutable indexes are never
overwritten. Atomic budget/progress pointers are the only replaceable metadata.
A cross-process filesystem lock prevents concurrent attempts.

`full-budget.json` and `attempts/NNNNN/` belong only to this full campaign. They
never read, reset or change the completed pilot's budget. Normal SIGINT/SIGTERM
kills the worker group, records the elapsed interval and leaves recoverable
shards. Linux parent-death signaling also kills the worker if the supervisor dies.
After an unclean supervisor/power loss, time since the last durable heartbeat is
conservatively charged (including downtime) to total time and any inflight model;
no budget reset occurs. Long downtime can exhaust the allowance and require
explicit accounting/limit review. Clock reversal or corrupted budgets stop.
There is no automatic retry or automatic creation of a replacement campaign.

## Storage reconciliation and verification

The accepted review's historical planning total is **70,896,541,628.7875 bytes**:
31,312,510,114 weights/tokenizers, two 16,589,860,864-byte tensor copies,
220,690,055.39375 estimated metadata bytes per copy, 4,999,802,720 temporary
weight bytes, 83,886,080 temporary shard bytes and 879,240,876 pilot bytes.
That total was an estimate; the earlier 48 GiB reservation did not establish
full usable quota. Existing environment/checkouts/handoff copies must also be
included, and the old estimate is not treated as a quota certificate.

This successor counts intact existing weights **once** and requires **zero new
weight downloads/temporary weight files**. Its local planning reservation is
**40,779,800,576 additional bytes** (40.78 GB): two complete tensor copies plus
1 GiB metadata allowance per copy, one temporary shard and a 5 GiB safety margin.
Known weights plus retained pilot are **32,191,750,990 bytes**; adding that
reservation gives **72,971,551,566 bytes**, before the already-retained runtime,
checkouts, copied handoffs and any other volume files. The live scan resolves
that remaining accounting. No current actual full-volume capacity is claimed.

After restart, `storage-check` scans all regular files on the actual mount,
deduplicating hard links by device/inode and excluding snapshot symlink aliases.
It separately checks cached weights against the pinned receipts. CPU tokenization
of all 35,843 texts/model records exact per-shard packet hashes and metadata sizes
without model inference. Reserved receipt/commit/global overhead bounds complete
copies and a temporary shard. Actual shard bytes are checked against the bound.
For resume, only loader-verified completed primary bytes reduce the additional
reservation; quarantine/partial bytes remain retained usage, never completed output.

The concrete quota test **writes the entire calculated additional reservation
as incompressible bytes while all existing artifacts remain**, fsyncs it and
hashes its full readback using the unchanged pilot storage probe. It also tests
exclusive publication, overwrite refusal, orphan recovery, atomic replacement and
directory fsync, plus chmod, Git initialization and cross-process flock exclusion.
Only this temporary probe is removed. A receipt binds actual mount identity,
retained usage, token plans, destination and successful byte count. The complete
second copy is conservatively reserved on the same volume. An external destination
can reduce that requirement only under a separately reviewed storage plan.

Launch requires that receipt within 24 hours, the same mount/cache/manifest, no
regression in verified primary storage and no unplanned volume growth beyond
small control-receipt overhead. Keep the volume exclusive between the check and
launch. Pool statvfs reports remain informational. This is point-in-time capacity
and process-level recovery evidence, not a power-loss durability guarantee.

## Proposed execution limits

Measured mean raw forecast is **1.606828828 GPU-hours** (Qwen .816332, Llama
.790497); summed observed extrema [1.357537292,1.818370527] are not confidence
bounds. Proposed independent full limits:

| Limit | Proposed cap |
| --- | ---: |
| Each model, load through teardown including writing/recovery | 5,400 s / 90 min |
| Combined full worker wall, cumulative across attempts | 12,600 s / 3.5 h |
| Any non-model setup phase inside the worker | 1,800 s / 30 min |

90 minutes/model exceeds the slowest pilot projection (.989/.829 h) by 25% plus
roughly 16/28 minutes for loading, cold reads and recovery. The combined limit
adds 30 minutes beyond two model allowances for host validation. An independent
supervisor enforces these fixed limits and kills the worker group at a cap.
First-use provisioning/cache/storage checks outside the worker are billed
separately and must not be mistaken for kernel time or reset the pilot limits.
No hourly price is supplied: cost = price * actual billed instance-hours.
The small 4.33 MB handoff does not establish sustained full 16.59 GB transfer speed.

## First use after restart and reviewed commands

These remote commands are **prepared for later explicit authorization, not executed**.
Use the user's newly confirmed GPU SSH connection after restart; do not assume an
old address is still the Pod. Keep local development on macOS. Before installation
or downloads, inspect `findmnt --target /workspace`, actual GPU visibility and the
preserved mount contents. The cached weights are read-only; no downloads are
planned. Fetch the exact delivery SHA into a new detached checkout, never pull
or merge. The `--expected-commit` gate also requires a clean checkout.

From that new exact checkout on the remote host (replace FULL_COMMIT with this
delivery's pushed 40-character SHA):

```bash
pilot_runtime=/workspace/checkpoint-r2-fresh-pilot-20261002-3f9747f-volume/runtime
full_control=/workspace/checkpoint-r2-full-raw-control-20261002
full_output=/workspace/checkpoint-r2-full-raw-20261002
full_commit=FULL_COMMIT
mkdir "$full_control"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
"$pilot_runtime/venv/bin/python" -m pip check
"$pilot_runtime/venv/bin/python" -B -m src.checkpoint_r2_full_raw dry-run --output "$full_control/metadata.json"
"$pilot_runtime/venv/bin/python" -B -m src.checkpoint_r2_full_raw preflight --pilot-cache "$pilot_runtime/model-cache" --output "$full_control/runtime.json"
"$pilot_runtime/venv/bin/python" -B -m src.checkpoint_r2_full_raw storage-check --pilot-cache "$pilot_runtime/model-cache" --volume /workspace --destination "$full_output" --output "$full_control/storage.json"
```

If the environment/snapshot preflight fails, stop for that specific prerequisite;
do not edit the preserved environment/cache or silently download replacements.
After the later explicit full-run authorization and successful capacity check:

```bash
"$pilot_runtime/venv/bin/python" -B -m src.checkpoint_r2_full_raw run \
  --acknowledge-reviewed-full-raw --expected-commit "$full_commit" \
  --pilot-cache "$pilot_runtime/model-cache" --storage-receipt "$full_control/storage.json" \
  --output "$full_output"
```

For an interruption, inspect the retained failure/supervisor/budget receipts and
monitor any surviving process first. Recompute `storage-check` into a new receipt
if conditions or age changed; the same destination is supported. Use the same
run command with `--resume`, the same output/campaign and any new storage receipt.
Do not increase limits or switch to a fresh output to reset spent time.

```bash
"$pilot_runtime/venv/bin/python" -B -m src.checkpoint_r2_full_raw verify-output --output "$full_output"
```

Local validation completed: **95 passed, 2 deselected** across the successor/pilot,
frozen input and auth/peak suites. The two excluded tiny BF16 architecture tests
require Torch; this CPU-only environment intentionally did not run them. Both
actual eight-row handoff shards reopen with expected hashes, rows, FP16 axes and
token semantics. The 1,194 base-tracked files, supplied DOCX, 13 unrelated files
and all 143 checksummed completed-handoff files were verified unchanged.
Receipts are in `results/checkpoint_r2_full_raw_prep_20261002/`.

Local metadata and focused test commands:

```bash
/private/tmp/r2-full-raw-prep-venv/bin/python -B -m src.checkpoint_r2_full_raw dry-run --output /private/tmp/r2-full-new-review.json
/private/tmp/r2-full-raw-prep-venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_checkpoint_r2_full_raw.py
```

The inherited input-package byte rebuild records NumPy 2.4.6, whereas the GPU
representation runtime remains NumPy 2.1.3. Validate that historical producer
receipt using its CPU NumPy version without changing frozen files or GPU pins:

```bash
PYTHONPATH=/private/tmp/r2-input-rebuild-libs /private/tmp/r2-full-raw-prep-venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_checkpoint_r2_inputs.py
```
