# October 3 integration successor — prepared commands only

This supersedes the **execution commands** in the historical behavior RUNBOOK.md. Preserve that document as history. No command below authorizes execution. No Pod contact, inference, extraction, head fitting, E access or push occurred in this preparation.

```bash
PRODUCER=9b0ac1d84b35d7faf2148d43664569aa867cba52
BASE=0a256d96b0fbf41f7ce678693da15cf230685778
BRANCH=checkpoint-next-integration-fix-20261003
MAC_PY=/opt/homebrew/opt/python@3.11/bin/python3.11
```

The integration merge is `5ef04a7b45d4985fd41bc777e630474fd5ae68a2`, with ordered parents `f63a9a9348892b81c0ab60261033a2b8600f26a2` and `bca07a055c70f7debb6389cf68feb6d2ab7c543f`. Execute the literal producer, not the later documentation delivery tip or a moving branch. The packaged incremental bundle includes both histories after BASE and requires that exact base and its ancestors. Review ZIP README and delivery.json identify the delivery tip and archive hashes. No successful push is required.

## First use and bundle transfer

After later explicit execution authorization, obtain the **current** host, port and account from the current Pod management record. Do not infer them from old endpoints. Use the user's existing secure credentials, outside artifacts. These are future commands only:

```bash
read -r -p 'Current verified host: ' POD_HOST
read -r -p 'Current verified SSH port: ' POD_PORT
read -r -p 'Current verified account: ' POD_USER
read -r -p 'Absolute reviewed integration.bundle path: ' REVIEW_BUNDLE
shasum -a 256 "$REVIEW_BUNDLE"
ssh -p "$POD_PORT" "$POD_USER@$POD_HOST" 'hostname; uname -s; nvidia-smi'
scp -P "$POD_PORT" "$REVIEW_BUNDLE" "$POD_USER@$POD_HOST:/workspace/integration.bundle"
ssh -p "$POD_PORT" "$POD_USER@$POD_HOST"
```

Set the literal PRODUCER/BASE/BRANCH above in the execution shell. Use a new code directory and the existing accepted local Git repository (or separately verified base bundle) as APPROVED_BASE_SOURCE. Do not modify that source. A missing prerequisite is a blocker, not a reason to pull a branch.

```bash
set -euo pipefail
read -r -p 'Existing local repository or verified base bundle containing exact BASE: ' APPROVED_BASE_SOURCE
CODE="/workspace/checkpoint-next-integration-code-$PRODUCER"
RUN="/workspace/checkpoint-next-integration-output-$PRODUCER"
REVIEW="/workspace/checkpoint-next-integration-authorization-$PRODUCER"
GPU_PY=/workspace/checkpoint-r2-fresh-pilot-20261002-3f9747f-volume/runtime/venv/bin/python
CACHE=/workspace/checkpoint-r2-fresh-pilot-20261002-3f9747f-volume/runtime/model-cache
REVIEW_BUNDLE=/workspace/integration.bundle
test ! -e "$CODE"
test ! -e "$RUN"
test -x "$GPU_PY"
test -d "$CACHE"
git init "$CODE"
git -C "$CODE" fetch "$APPROVED_BASE_SOURCE" "$BASE"
git -C "$CODE" cat-file -e "$BASE^{commit}"
git -C "$CODE" bundle verify "$REVIEW_BUNDLE"
git -C "$CODE" bundle list-heads "$REVIEW_BUNDLE"
git -C "$CODE" fetch "$REVIEW_BUNDLE" "refs/heads/$BRANCH:refs/remotes/review/$BRANCH"
git -C "$CODE" checkout --detach "$PRODUCER"
cd "$CODE"
test "$(git rev-parse HEAD)" = "$PRODUCER"
test -z "$(git status --porcelain --untracked-files=no)"
git merge-base --is-ancestor "$BASE" HEAD
hostname
"$GPU_PY" -c 'import sys,platform; print(sys.executable); print(sys.version); print(platform.platform())'
"$GPU_PY" -m pip freeze --all
nvidia-smi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
mkdir "$RUN" "$REVIEW"
"$GPU_PY" -m src.checkpoint_next_runner prepare --output "$RUN/metadata"
"$GPU_PY" -m src.checkpoint_next_runner missing-raw --output "$RUN/missing-metadata.json"
```

Confirm the bundle branch tip against delivery.json before fetching it. No installation/download fallback is provided. The worker verifies the accepted Python 3.12 runtime, complete dependency/GPU/driver evidence, torch 2.8.0+cu128, transformers 4.56.2, exact offline Qwen/Llama snapshots and implementation hashes. Qwen revision is `a09a35458c702b33eeacc393d103063234e8bc28`. Llama is `0e9e39f249a16976918f6564b8830bc894c89659`. The executable configuration is authoritative. Stop on missing/wrong runtime, GPU, cache or snapshots; do not restart or download automatically.

## Authorizations and the shared campaign

Generate invalid templates for external review. Keep each output new. No reviewed authorization is supplied by preparation.

```bash
for model in qwen llama; do
  for phase in missing_raw preflight storage_preflight launch; do
    "$GPU_PY" scripts/checkpoint_next_authorization_template.py \
      --model "$model" --phase "$phase" --output "$REVIEW/$model-$phase-template.json"
  done
done
```

A later reviewer supplies separate approved `MODEL-PHASE.json` files with exact producer, original config hash, integration config hash, inventory hash, model, named phase, `reviewed: true`, and `E: false`. Preflight/storage/launch must bind the actual supplement `acceptance.json` hash in `missing_raw_acceptance_sha256`. This replaces the old caller-supplied missing-raw seconds/loader-verified flags. Do not copy the historical authorization template or hand-fill completion receipts.

All missing raw, behavior preflight and behavior/chat launch attempts use **the same RUN directory and per-model lock/ledger**. The 7,200-second/model stopping allowance includes loading, execution, writing and teardown. Setup snapshot/runtime checks are limited separately to 1,800 seconds/phase. Failed/interrupted attempts still count; never create a new campaign to reset budget. These are caps, not guarantees of completion. The full-bank CPU reference cap remains 10,800 seconds and is not assigned to chat fitting.

## Exact missing raw extraction, token audit and acceptance

Execute sequentially only after the missing_raw authorizations are approved. This is 131 texts/model, batch 1 unpadded, BF16 computation, FP16 storage, all 28 Qwen /32 Llama saved layers, last input state, raw native special tokens. The committed queue order/hash is fixed. Existing full-raw outputs are read-only; only the new RUN/missing_raw subtree receives supplements. Before forwards, the runner binds all 131 native packets and performs real bounded two-copy allocation/publication checks. These reservations do not prove persistent quota; retained caches and originals must remain present during the check.

```bash
for model in qwen llama; do
  "$GPU_PY" -m src.checkpoint_next_runner extract-missing \
    --model "$model" --cache "$CACHE" --output "$RUN" \
    --authorization "$REVIEW/$model-missing_raw.json"
  "$GPU_PY" -m src.checkpoint_next_runner audit-missing-tokens \
    --model "$model" --cache "$CACHE" --output "$RUN"
  "$GPU_PY" -m src.checkpoint_next_runner accept-missing --model "$model" --output "$RUN"
  "$GPU_PY" -m src.checkpoint_next_runner verify --phase missing_raw --model "$model" --output "$RUN"
  sha256sum "$RUN/missing_raw/$model/acceptance.json"
done
```

The independent audit process reconstructs tokens from the pinned offline tokenizer without loading the model. Acceptance reopens every all-layer shard, verifies dtype, axes, hashes, exact text/member/token bindings, completion/memory/timing evidence and settled attempt ledger. It cannot be substituted by a fabricated flag. Transfer the verified campaign at this boundary to unblock mac 2; do not regenerate or expand the queue. The earlier 262 additional bindings mean 131 existing-text reuses plus 131 missing texts/model.

## Bounded training preflight and forecast review

```bash
for model in qwen llama; do
  "$GPU_PY" -m src.checkpoint_next_runner preflight \
    --model "$model" --cache "$CACHE" --output "$RUN" \
    --authorization "$REVIEW/$model-preflight.json"
  "$GPU_PY" -m src.checkpoint_next_runner verify --phase preflight --model "$model" --output "$RUN"
  "$GPU_PY" -m json.tool "$RUN/preflight/$model/completion.json"
done
```

Forty fixed training inputs/model exercise the new paths; full planned D tokenization is metadata only. Independently compare each candidate token's full-sequence likelihood term with a direct prefix-forward term (prospective absolute tolerance 0.05 log-probability for BF16 full/prefix kernels, no relative tolerance or post-result relaxation). Compare native greedy IDs with manual argmax decoding exactly, inspect actual prepared input/mask/position lengths at every step, and repeat behavior/chat forwards. Record traces and native termination. Local synthetic tests cover eight nonterminating steps and early termination; they do not validate either pinned research model. Existing raw extraction success is insufficient.

Review loading, tokenization, likelihood, generation, chat extraction, write/verify and teardown times, peak allocated/reserved CUDA bytes captured before teardown, unattributed walltime, memory headroom, tokens and native templates. Each attempt has `execution-evidence.json`; the completion pins its SHA. Hard process kills may prevent final evidence: retain that as missing evidence and the watchdog's charged time, never zero it out.

The runtime forecast uses maximum fixed-training costs and charges eight times the largest observed full generation call to account for the cap even if the observed answer ended early. This is deliberately a forecast, not a measured upper bound; later positions/lengths, future write/verify and teardown costs need an explicit reviewed nonnegative `unmeasured_allowance_seconds` in launch authorization. The launcher requires forecast plus allowance to fit the **remaining shared ledger** budget. Training token-length coverage must also pass. Stop and review a bounded amended scope/cap if it does not fit. No automatic smaller panel or larger budget is allowed.

## Storage and launch

Storage authorizations bind `preflight_sha256`; launch authorizations additionally bind preflight, reservation path/hash, `forecast_reviewed`, `storage_reviewed` and the allowance above. Keep the complete existing raw/model cache and missing supplement in place. Use a new reservation directory for each check.

```bash
export GPU_PY CODE RUN REVIEW
"$GPU_PY" - <<'PY'
import os,subprocess
for model in ('qwen','llama'):
    subprocess.run([os.environ['GPU_PY'],os.environ['CODE']+'/scripts/checkpoint_next_storage.py',
        '--model',model,'--campaign',os.environ['RUN'],
        '--output',os.environ['RUN']+'-reservation-'+model,
        '--authorization',os.environ['REVIEW']+'/'+model+'-storage_preflight.json',
        '--preflight',os.environ['RUN']+'/preflight/'+model+'/completion.json'],check=True,timeout=1800)
PY
for model in qwen llama; do
  "$GPU_PY" -m src.checkpoint_next_runner launch \
    --model "$model" --cache "$CACHE" --output "$RUN" \
    --authorization "$REVIEW/$model-launch.json"
  "$GPU_PY" -m src.checkpoint_next_runner verify --phase launch --model "$model" --output "$RUN"
done
```

Each model has 1,900 unique behavior inputs across both prompts, 11,578 chat inputs and 800 exact overlaps: 12,678 unique prompted inputs, with distinct scoring/generation/chat forwards as planned. Chat uses only Prompt 1 and layers Qwen 17/18, Llama 15/12. Both prompt strings, parser, masks, zero threshold and True ties remain frozen. Record 200 logical isolated bindings per prompt, not a reweighted 150-fact sample.

## Resume and stop

Resume by rerunning the exact phase command with the same producer, RUN and authorization after confirming the previous supervisor has stopped. Do not run multiple supervisors for one model. Revalidate current host/runtime/cache and redo a storage check with a new receipt after restart or after its 24-hour freshness window. Preserve all prior receipts. Completed shards reopen and verify; incomplete shards move intact into quarantine; committed corruption stops. Completed phases verify without model loading. For missing raw, run token audit and acceptance after its first extraction before using its completed-phase verification.

Use Ctrl-C for a foreground supervisor. For background execution record only its actual PID (`SUPERVISOR_PID=$!`), then:

```bash
kill -TERM "$SUPERVISOR_PID"
ps -p "$SUPERVISOR_PID" -o pid=,command=
nvidia-smi
```

The supervisor records stop status and charges active time; Linux parent-death handling prevents orphan workers. Interrupted active attempts conservatively charge through recovery, including downtime. A hard kill can leave incomplete memory/timing records; report the gap and retain all files. No Pod stop/restart API is provided. Stop on mismatched identity/hash/token/runtime, coverage gap, failed optimizer, cap/quota/write error or failed preflight. Do not relax objectives, iterations, tolerances, layers, rules, or scientific inputs.

## Transfer and independent local verification

Stop writers before taking a snapshot; transfer to a new destination. Include the entire campaign, attempts and quarantine, not cache/weights/credentials/environments. Missing-only and final transfers use different destinations/manifests. On the producer checkout:

```bash
export RUN
"$GPU_PY" - <<'PY'
import hashlib,json,os,pathlib
root=pathlib.Path(os.environ['RUN'])
files={}
for p in sorted(root.rglob('*')):
    if p.is_file():
        with p.open('rb') as f:files[str(p.relative_to(root))]=hashlib.file_digest(f,'sha256').hexdigest()
with root.with_name(root.name+'-transfer.json').open('x') as f:json.dump(files,f,indent=2,sort_keys=True)
PY
```

On macOS, reconstruct a new exact-producer verification checkout using the bundle/base procedure above (with macOS paths). Set VERIFY_CODE to it, keep the original worktrees untouched. Set current connection variables as already verified; use a new LOCAL_RUN:

```bash
read -r -p 'New absolute local handoff destination: ' LOCAL_RUN
read -r -p 'Exact-producer local verification checkout: ' VERIFY_CODE
test ! -e "$LOCAL_RUN"
scp -r -P "$POD_PORT" "$POD_USER@$POD_HOST:$RUN" "$LOCAL_RUN"
scp -P "$POD_PORT" "$POD_USER@$POD_HOST:$RUN-transfer.json" "$LOCAL_RUN-transfer.json"
export LOCAL_RUN
"$MAC_PY" - <<'PY'
import hashlib,json,os,pathlib
root=pathlib.Path(os.environ['LOCAL_RUN'])
expected=json.loads(root.with_name(root.name+'-transfer.json').read_text())
actual={}
for p in sorted(root.rglob('*')):
    if p.is_file():
        with p.open('rb') as f:actual[str(p.relative_to(root))]=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==expected,'transfer file/hash mismatch'
PY
cd "$VERIFY_CODE"
test "$(git rev-parse HEAD)" = "$PRODUCER"
for model in qwen llama; do
  "$MAC_PY" -m src.checkpoint_next_runner verify --phase missing_raw --model "$model" --output "$LOCAL_RUN"
  "$MAC_PY" -m src.checkpoint_next_runner verify --phase preflight --model "$model" --output "$LOCAL_RUN"
  "$MAC_PY" -m src.checkpoint_next_runner verify --phase launch --model "$model" --output "$LOCAL_RUN"
done
```

For a missing-only transfer omit the last two phase checks. Require the separately reconstructed token-audit receipt and actual all-layer reopen on this second copy before downstream use. Schema/hash consistency is not second-machine model replication. This preparation authorizes no deletion of either copy or originals.

## Eight chat heads and reporting, after a separate CPU-slot review

Prepare a reviewed JSON export of eight raw heads with `id`, `model`, `family`, `layer`, `fitting_identities_sha256`, `D_support_sha256` (the frozen 800 IDs), `preprocessing_membership: original_P15`, `P15_identities_sha256`, `regime: raw`, frozen `decision_rule`, its real `frozen_rule_source_sha256`, and full-D `logical_ids`/`scores` (7,728). Verify export provenance against existing raw deliveries; do not invent scores or thresholds. Chat fits require separate eight-head authorization with exact producer/config/inventory, raw export SHA, `CPU_slot_exclusive: true`, `heads: 8`, `E: false` and a reviewed positive `cpu_seconds`. CPU work stays serialized with mac 2. No full-training preprocessing or layer search is introduced.

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
read -r -p 'Reviewed eight-head raw export: ' RAW_MATCHES
read -r -p 'Reviewed chat CPU authorization: ' CHAT_AUTH
CHAT_FITS="/private/tmp/checkpoint-next-chat-fits-$PRODUCER"
test ! -e "$CHAT_FITS"
"$MAC_PY" scripts/checkpoint_next_chat_fit.py --input "$LOCAL_RUN" --output "$CHAT_FITS" \
  --raw-matches "$RAW_MATCHES" --authorization "$CHAT_AUTH"
for model in qwen llama; do
  "$MAC_PY" scripts/checkpoint_next_report.py --input "$LOCAL_RUN" --model "$model" \
    --chat-fits "$CHAT_FITS" --raw-matches "$RAW_MATCHES" \
    --output "/private/tmp/checkpoint-next-$model-report-$PRODUCER.json"
done
```

The adapter verifies all eight receipt/parameter/score hashes, successful fitting status, frozen head identities, chat-P15 preprocessing and exact 800-row support. Full-D raw results remain separate; no full-D chat scores exist. Optional `--probes` exports retain AUROC-only arms with null decision rules; never supply invented thresholds. The report has isolated accuracy/True rates/formatting/valid errors/disagreement and label-signed margins under each model/prompt/topic. The all-four-isolated-correct mask is unchanged.

Flagged optimizer returns publish parameters and every attempt's full initial/final vectors and diagnostics before raising. Such fits have no usable prediction output. A hard process timeout can prevent an unfinished optimizer returning diagnostics; preserve that incomplete task, report the limitation and never count it as a successful head. No retry fits are automatic. These named Prompt 1 controls establish neither prompted constituent availability nor causal use.
