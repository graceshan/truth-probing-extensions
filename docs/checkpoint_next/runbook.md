# Prepared commands — do not execute during preparation

These commands reference the frozen producer commit below. The delivery may add documentation after this producer; execute the producer, not a moving branch. No command in this runbook contacted the Pod or ran research workloads during preparation. A later execution request must name this producer and the bounded phases. The authorization template is intentionally invalid (`reviewed: false`); this preparation does not grant execution approval.

```bash
PRODUCER=f633a03f317c68e310515113634948bf892bd962
BASE=0a256d96b0fbf41f7ce678693da15cf230685778
```

## Local first-use and code transfer

On macOS, preserve this checkout and prepare a new bundle outside it. Use the exact SHA above even if the branch advances. No pull, merge, reset, clean, stash or replacement of results is needed.

```bash
set -euo pipefail
PREP=/Users/apple/projects/truth-probing-extensions/.worktrees/checkpoint-next-behavior-prep-20261003
MAC_PY=/opt/homebrew/opt/python@3.11/bin/python3.11
cd "$PREP"
hostname
"$MAC_PY" -c 'import sys,platform; print(sys.executable); print(sys.version); print(platform.platform())'
git cat-file -e "$PRODUCER^{commit}"
git merge-base --is-ancestor "$BASE" "$PRODUCER"
git diff --name-status "$BASE" "$PRODUCER"
BUNDLE="/private/tmp/checkpoint-next-$PRODUCER.bundle"
test ! -e "$BUNDLE"
git bundle create "$BUNDLE" checkpoint-next-behavior-prep-20261003
git bundle verify "$BUNDLE"
shasum -a 256 "$BUNDLE"
```

After a later explicit Pod-use authorization, obtain the **current** host, port and user from the current Pod management record. Do not reuse an old endpoint merely because it appeared in an earlier report. Resolve SSH credentials through the user's existing secure terminal setup; do not paste keys/tokens into reports or authorization JSON.

```bash
read -r -p 'Current verified Pod host: ' POD_HOST
read -r -p 'Current verified SSH port: ' POD_PORT
read -r -p 'Current verified SSH user: ' POD_USER
ssh -p "$POD_PORT" "$POD_USER@$POD_HOST" 'hostname; uname -s; nvidia-smi'
scp -P "$POD_PORT" "$BUNDLE" "$POD_USER@$POD_HOST:/workspace/"
ssh -p "$POD_PORT" "$POD_USER@$POD_HOST"
```

The initial identity command is an execution-stage check, not a preparation-stage permission to use the Pod. Verify the management-record identity as well as host/GPU. Stop if the endpoint or accepted runtime is wrong.

On the Pod, set the same literal `PRODUCER` above. The paths below refer to the accepted fresh runtime/cache; their continued existence must be checked after restart. No downloads or installations are automatic.

```bash
set -euo pipefail
GPU_PY=/workspace/checkpoint-r2-fresh-pilot-20261002-3f9747f-volume/runtime/venv/bin/python
CACHE=/workspace/checkpoint-r2-fresh-pilot-20261002-3f9747f-volume/runtime/model-cache
CODE="/workspace/checkpoint-next-code-$PRODUCER"
GPU_RUN="/workspace/checkpoint-next-output-$PRODUCER"
REVIEW="/workspace/checkpoint-next-reviewed-$PRODUCER"
test -x "$GPU_PY"
test -d "$CACHE"
test ! -e "$CODE"
test ! -e "$GPU_RUN"
git clone --no-checkout "/workspace/checkpoint-next-$PRODUCER.bundle" "$CODE"
cd "$CODE"
git checkout --detach "$PRODUCER"
test "$(git rev-parse HEAD)" = "$PRODUCER"
test -z "$(git status --porcelain --untracked-files=no)"
"$GPU_PY" -c 'import sys,platform; print(sys.executable); print(sys.version); print(platform.node())'
"$GPU_PY" -m pip freeze --all
nvidia-smi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir "$GPU_RUN"
"$GPU_PY" -m src.checkpoint_next_runner prepare --output "$GPU_RUN/metadata"
```

The prepared worker verifies cached snapshot files, index, revisions and hashes, complete accepted runtime/GPU/driver/dependency freeze, backend/tokenizer source hashes, flags and saved-layer semantics before proceeding. This enforces the accepted fresh contract, not the original protocol's obsolete candidate setup. Do not repair a missing cache by downloading weights in this task. Setup is separately limited to 1,800 seconds/phase.

## mac 2 missing-input interface

mac 2 supplies two JSON arrays: exact admitted `outer_train` membership objects (unaltered fields) and loader-verified available raw records. Available records must contain exact `statement`, `statement_sha256`, nonempty `token_ids`, accepted `representation_sha256`, and `loader_verified: true`. An externally verified loader export is required; filling in that flag manually is not verification. Record and audit its source shard hashes. Historical tensors are ineligible.

On an exact-producer local checkout, these are metadata-only commands. They do not extract anything. The expected representation hash must come from the accepted fresh manifest/loader export, not a guessed or relabeled hash.

```bash
"$MAC_PY" -m src.checkpoint_next_runner missing-raw \
  --required /private/tmp/mac2-reviewed-required-outer-train.json \
  --available /private/tmp/mac2-loader-verified-fresh-raw.json \
  --representation-sha256 "$ACCEPTED_FRESH_REPRESENTATION_SHA256" \
  --output "/private/tmp/mac2-missing-raw-$PRODUCER.json"
shasum -a 256 "/private/tmp/mac2-missing-raw-$PRODUCER.json"
```

The package's 131 text-only misses are a planning crosswalk, not the final verified missing inventory. Before authorizing behavior/chat, supply a per-model reviewed ledger file with `model`, `active_gpu_seconds`, and status `completed` or `no_missing_inputs_verified`, plus reviewed inventory/transfer evidence. Authorizations bind its path, SHA-256 and exact seconds. Any bounded missing-raw extraction needs a later separately named reviewed task, runs under the accepted fresh contract, and consumes the same 7,200-second/model allowance. Do not invoke the old full-raw runner on an entire bank just to fill gaps.

## Training preflight and forecast review

Use externally reviewed authorization files, copied securely into `$REVIEW` outside the checkout. For `preflight`, each file must specify exact producer/config/inventory/model, `phase: preflight`, `reviewed: true`, `E: false`, and the reviewed missing-raw ledger path/hash/seconds. There is no valid authorization in this delivery.

The preflight is bounded to 40 fixed **training** inputs/model: shortest and longest UTF-8 source texts for P15 and fixed T_C within each topic under both prompts. It tests full candidate sequence sums, prompt boundaries, strict greedy parsing, repeated behavior, and repeated Prompt 1 last-input states. Full workload tokenization is a metadata pass and does not run D model forwards. Workload counts are 1,900 scoring/generation inputs and 11,578 chat forwards/model; no chat/behavior hidden-state reuse is assumed.

Run models sequentially. Each command is supervised; its setup/active phases have separate limits, and preflight use is charged to the campaign's cumulative active budget.

```bash
"$GPU_PY" -m src.checkpoint_next_runner preflight \
  --model qwen --cache "$CACHE" --output "$GPU_RUN" \
  --authorization "$REVIEW/qwen-preflight.json"
"$GPU_PY" -m src.checkpoint_next_runner verify \
  --phase preflight --model qwen --output "$GPU_RUN"
"$GPU_PY" -m src.checkpoint_next_runner preflight \
  --model llama --cache "$CACHE" --output "$GPU_RUN" \
  --authorization "$REVIEW/llama-preflight.json"
"$GPU_PY" -m src.checkpoint_next_runner verify \
  --phase preflight --model llama --output "$GPU_RUN"
"$GPU_PY" -m json.tool "$GPU_RUN/preflight/qwen/completion.json"
"$GPU_PY" -m json.tool "$GPU_RUN/preflight/llama/completion.json"
```

Review separate path timings, exact token boundaries/IDs, native termination/template behavior, layer/repeat checks, workload token-length coverage, GPU memory/runtime receipt, conservative maximum-training-cost forecast, planned two-copy storage, retained cache/raw usage and remaining budget. Maxima extrapolation is a forecast, not a measured production runtime guarantee. If length coverage or the remaining two-hour/model allowance is inadequate, stop; review a smaller concrete scope or amended cap before starting. The old successful raw extraction validates none of these new paths.

## Storage preflight and production launch

Each reviewed `storage_preflight` authorization additionally binds `preflight_sha256`. Review retained full-volume usage and quota. The existing filesystem/writer check must allocate and hash the complete planned reservation while all retained artifacts remain. It removes only its own probe. The new reservation directory itself must not exist. Use a setup timeout of 1,800 seconds, for example the standard-library wrapper below; do not treat a successful allocation as a durable quota guarantee.

```bash
export GPU_PY CODE GPU_RUN REVIEW
"$GPU_PY" - <<'PY'
import os,subprocess
for model in ('qwen','llama'):
    subprocess.run([os.environ['GPU_PY'],os.environ['CODE']+'/scripts/checkpoint_next_storage.py',
        '--model',model,'--output',os.environ['GPU_RUN']+'-reservation-'+model,
        '--authorization',os.environ['REVIEW']+'/'+model+'-storage.json',
        '--preflight',os.environ['GPU_RUN']+'/preflight/'+model+'/completion.json'],
        check=True,timeout=1800)
PY
```

Review and bind the resulting `reservation.json` path/SHA-256 in `launch` authorizations, plus `preflight_sha256`, `forecast_reviewed: true`, `storage_reviewed: true`. The launch checks identity, reservation size, mount device and freshness (24 hours). Recheck storage after every restart; preserve old checks and use a new reservation directory/receipt. Capacity races and write errors stop the affected task.

```bash
"$GPU_PY" -m src.checkpoint_next_runner launch \
  --model qwen --cache "$CACHE" --output "$GPU_RUN" \
  --authorization "$REVIEW/qwen-launch.json"
"$GPU_PY" -m src.checkpoint_next_runner verify \
  --phase launch --model qwen --output "$GPU_RUN"
"$GPU_PY" -m src.checkpoint_next_runner launch \
  --model llama --cache "$CACHE" --output "$GPU_RUN" \
  --authorization "$REVIEW/llama-launch.json"
"$GPU_PY" -m src.checkpoint_next_runner verify \
  --phase launch --model llama --output "$GPU_RUN"
```

## Resume, transfer and independent verification

Resume uses the same launch command, output directory, producer, config, inventory, ledger and authorization. First confirm the prior supervisor/worker has stopped, confirm current endpoint/runtime, and repeat the storage check if restarted/stale. Do not create a replacement campaign to reset caps. Completed shards are reopened and rehashed; unpublished partial directories move intact to quarantine; committed corruption stops. A fully completed phase verifies and returns without loading models. SIGINT/SIGTERM stop the worker; Linux parent-death signaling prevents an orphan GPU worker. An unclean active attempt is conservatively charged through recovery, including downtime; clock reversal or exhausted allowance requires accounting review.

For a transfer, stop writers and verify both models first. Create a manifest outside `$GPU_RUN` so it does not hash itself. All paths in the manifest are relative. Use an entirely new local destination; never overlay an existing result directory. Transfer the output tree, attempts, quarantined partials and manifest as non-secret research artifacts; do not include the model cache, private credentials or token files.

```bash
cd "$CODE"
export GPU_RUN
"$GPU_PY" - <<'PY'
import hashlib,json,os,pathlib
root=pathlib.Path(os.environ['GPU_RUN'])
files={}
for p in sorted(root.rglob('*')):
    if p.is_file():
        with p.open('rb') as f: files[str(p.relative_to(root))]=hashlib.file_digest(f,'sha256').hexdigest()
out=root.with_name(root.name+'-transfer.json')
with out.open('x') as f: json.dump(files,f,sort_keys=True,indent=2)
PY
```

On the Mac, set the current verified connection variables again if necessary. Code in `$VERIFY_CODE` must be an isolated exact-producer checkout; the delivery branch may contain documentation added after the producer. Preserve the existing preparation worktree by creating a second checkout only when later execution verification is authorized.

```bash
LOCAL_RUN="/Users/apple/projects/truth-probing-extensions/.worktrees/checkpoint-next-handoff-$PRODUCER"
VERIFY_CODE="/Users/apple/projects/truth-probing-extensions/.worktrees/checkpoint-next-verify-$PRODUCER"
test ! -e "$LOCAL_RUN"
test ! -e "$VERIFY_CODE"
git -C "$PREP" worktree add --detach "$VERIFY_CODE" "$PRODUCER"
scp -r -P "$POD_PORT" "$POD_USER@$POD_HOST:/workspace/checkpoint-next-output-$PRODUCER" "$LOCAL_RUN"
scp -P "$POD_PORT" "$POD_USER@$POD_HOST:/workspace/checkpoint-next-output-$PRODUCER-transfer.json" "$LOCAL_RUN-transfer.json"
export LOCAL_RUN
"$MAC_PY" - <<'PY'
import hashlib,json,os,pathlib
root=pathlib.Path(os.environ['LOCAL_RUN'])
expected=json.loads(root.with_name(root.name+'-transfer.json').read_text())
actual={}
for p in sorted(root.rglob('*')):
    if p.is_file():
        with p.open('rb') as f: actual[str(p.relative_to(root))]=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==expected, 'Transfer hash/coverage mismatch'
PY
cd "$VERIFY_CODE"
"$MAC_PY" -m src.checkpoint_next_runner verify --phase launch --model qwen --output "$LOCAL_RUN"
"$MAC_PY" -m src.checkpoint_next_runner verify --phase launch --model llama --output "$LOCAL_RUN"
```

The second physical copy and complete reopen are required before removing any temporary execution outputs. This preparation authorizes no deletion. Preserve and report failures; do not silently omit rows or methods. Independently reconstruct tokenization/candidate boundaries with the verified cached tokenizer on the GPU host before scientific acceptance; local schema/hash checks cannot independently prove model inference semantics.

## CPU fits and reporting after the CPU slot is free

Prepare exactly eight audited raw comparator manifest exports keyed by `qwen|llama/R0|C_clean/L17|L18|L15|L12` as appropriate. Each record needs `id`, model, family, saved layer, fitting identities hash, D support hash, `preprocessing_membership: original_P15`, P15 identity hash, and `regime: raw`. Bind this export's hash in a separately reviewed `chat_cpu_fit` authorization with `heads: 8`, `CPU_slot_exclusive: true`, exact producer/config/inventory, `E: false`, and a reviewed positive `cpu_seconds`. The three-hour full-bank CPU reference is not a chat fit allowance. There is no new chat layer search or full bank.

The output directory must be new. A fit failure stops; successful partial fits and attempts remain. The CPU runner intentionally requires a fresh separately reviewed output for any retry; no automatic refit, relaxed acceptance or overwriting is allowed. All eight fits need independent score/gradient reconstruction before scientific acceptance.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
CHAT_FITS="/private/tmp/checkpoint-next-chat-fits-$PRODUCER"
"$MAC_PY" scripts/checkpoint_next_chat_fit.py \
  --input "$LOCAL_RUN" --output "$CHAT_FITS" \
  --raw-matches /private/tmp/reviewed-eight-raw-comparators.json \
  --authorization /private/tmp/reviewed-eight-chat-cpu-authorization.json
"$MAC_PY" scripts/checkpoint_next_report.py \
  --input "$LOCAL_RUN" --model qwen \
  --probes /private/tmp/reviewed-qwen-frozen-full-D-probes.json \
  --output "/private/tmp/checkpoint-next-qwen-report-$PRODUCER.json"
"$MAC_PY" scripts/checkpoint_next_report.py \
  --input "$LOCAL_RUN" --model llama \
  --probes /private/tmp/reviewed-llama-frozen-full-D-probes.json \
  --output "/private/tmp/checkpoint-next-llama-report-$PRODUCER.json"
```

A frozen full-D probe export has `model`, `head_id`, `logical_ids` (all 7,728 bare-D example IDs), finite `scores`, `decision_rule` and `frozen_rule_source_sha256`. Set `decision_rule: null` for an AUROC-only arm. Existing zero-logit decisions use `zero_logit_ties_true`; no D-fitted threshold is accepted. Export individual seeds/heads separately; do not pool different heads' scores. Raw/chat comparisons must match family, layer, fitting identities and D support, while retaining the original P15 exposure. Reporting is a single CPU job; serialize with fits/audits and mac 2's worker.

## Stop commands and stop conditions

For a foreground supervised phase use Ctrl-C. For a background phase record the actual supervisor PID from the launching shell (`SUPERVISOR_PID=$!`); never use an indiscriminate process-name kill.

```bash
kill -TERM "$SUPERVISOR_PID"
ps -p "$SUPERVISOR_PID" -o pid=,command=
nvidia-smi
```

Inspect the durable attempt's `end.json`, logs and preserved shards before resume. At hard setup/active caps the supervisor kills its child process group and records the stop. Stop the affected task on any hash/binding/runtime/coverage mismatch, formatting-preflight inconsistency, missing offline snapshot, quota/write failure, optimizer failure or cap hit. Do not widen tolerances, change prompts/parsers/layers/identities, drop methods, or reset budgets. The Pod's own stop action requires later Pod-management authorization; no stop/restart API is called by this package.

E stays closed throughout every command. These controls do not establish prompted constituent availability or causal use. Actual timings, quotas, transfer speed and storage become measured only after their respective reviewed phases run.
