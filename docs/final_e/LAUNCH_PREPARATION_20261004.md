# Raw acquisition and durable handoff preparation, October 4

**Preparation only. No RunPod contact, budget adoption, E extraction/output access or approved phase authorization is supplied.** Scientific approval is complete and is not requested again. Execute nothing below until actual operational instructions and separately reviewed authorization pins are supplied. All commands are prospective.

## Identity and verified local facts

| Item | Pinned value |
|---|---|
| Preparation base | `f12d9faf65ff5e31efbbcfbdb71d110a2ccbeb1c` |
| Final lock | `config/final_e/final_lock_20261004.json` |
| Final-lock SHA-256 | `88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e` |
| Exact runnable producer | `b2c6dde71e82f5f0af308761bf19ed3341670587` |
| Proposed campaign | `final-e-raw-20261004-b2c6dde-88b114dd-v1` |
| Incremental bundle prerequisite | `754ff1d7d362cf76ddb2bca681b32b2aeb7118f5` |
| Bundle | `producer.bundle`, 176,023 bytes; SHA-256 `da28abe3c64e50bffee3047003a621ee29b60dd46ceafd3052157ae796b7a1aa` |

The final lock and all six immutable pins are unchanged. A new clean detached local checkout holds the exact producer. The external lock, roots mapping, eleven closed drafts and frozen baseline-accounting evidence are copied into a durable external transport directory. `git bundle verify` passes locally; its only advertised head is `HEAD` at the producer and its only prerequisite is the full-bank delivery above. It contains three incremental commits; changed object paths were checked for activation/head archives, weights and caches. Large arrays, weights and caches are not bundled or committed.

Machine-readable absolute paths, eleven scopes and complete prospective runner argv are in `config/final_e/launch_preparation_20261004/paths_and_scopes.json`. Bundle inventory, current accounting, saved-head availability, closure and preservation evidence are in `results/final_e_launch_prep_20261004/`. Existing source manifests and selected-head file availability are checked; parameter archives are not reopened or numerically audited. Consult the receipt for the exact level of each check. No completed fit/activation/numerical audit is rerun.

## Paths and historical expectations

Use these root abbreviations only to make the commands readable; the inventory contains every full absolute path:

```bash
E_REMOTE_ROOT=/workspace/final-e-raw-20261004-b2c6dde-88b114dd-v1
E_MAC_ROOT=/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1
```

| Purpose | Remote path | Mac path |
|---|---|---|
| Campaign output | `$E_REMOTE_ROOT/campaign` | `$E_MAC_ROOT/handoff/campaign` |
| Detached producer | `$E_REMOTE_ROOT/execution-code` | `$E_MAC_ROOT/execution-code` |
| Lock | `$E_REMOTE_ROOT/control/final_lock_20261004.json` | `$E_MAC_ROOT/transport/final_lock_20261004.json` |
| Later approved authorizations | `$E_REMOTE_ROOT/control/authorizations/` | `$E_MAC_ROOT/control/authorizations/` |
| Later approved index | `$E_REMOTE_ROOT/control/approved-authorization-index.json` | `$E_MAC_ROOT/control/approved-authorization-index.json` |
| Operation receipts | `$E_REMOTE_ROOT/operation-receipts/` | `$E_MAC_ROOT/local-operation-receipts/` |
| Source campaign manifest | `$E_REMOTE_ROOT/manifests/campaign-checksums.json` | `$E_MAC_ROOT/handoff/campaign-checksums.json` |
| Source receipt manifest | `$E_REMOTE_ROOT/manifests/operation-receipt-checksums.json` | `$E_MAC_ROOT/handoff/operation-receipt-checksums.json` |
| Source control manifest | `$E_REMOTE_ROOT/manifests/control-checksums.json` | `$E_MAC_ROOT/handoff/control-checksums.json` |
| Durable source-receipt copy | — | `$E_MAC_ROOT/handoff/remote-operation-receipts/` |
| Durable source-control copy | — | `$E_MAC_ROOT/handoff/remote-control/` |

The Mac transport directory already exists. The Mac handoff/campaign, approved control files and verification-receipt destinations remain fresh/uncreated. Remote paths are proposals, not observations. Stop on a collision; preserve any existing campaign and choose reviewed new paths rather than overwrite it. Never put locks or authorizations into either producer checkout.

Historical accepted expectations, awaiting live verification:

- Persistent mount: `/workspace`, from the saved raw-only readiness filesystem receipt; current mount identity/durability/quota are unverified.
- Remote interpreter: `/workspace/checkpoint-r2-fresh-pilot-20261002-3f9747f-volume/runtime/venv/bin/python`, Python 3.12.3.
- Offline cache: `/workspace/checkpoint-r2-fresh-pilot-20261002-3f9747f-volume/runtime/model-cache`.
- NVIDIA A40; Python/package/GPU/driver/freeze checks must match the accepted records. The pinned contract includes PyTorch 2.8.0+cu128, Transformers 4.56.2 and tokenizers 0.22.1.
- Mac interpreter: `/opt/homebrew/opt/python@3.11/bin/python3.11`, locally observed Python 3.11.6.

Sources: `data/checkpoint_r2_full_raw_v1/pilot/host-runtime.json`, the two snapshot receipts, and the historical filesystem receipt pinned by this package. No old SSH endpoint is reused. Current endpoint, port, SSH target, existing credential/key location, independently verified host-key fingerprint and recipient Git repository are outstanding, supplied privately for the actual Pod session. No private connection values are committed. Credentials stay outside the non-secret research-artifact volume; historical owner-only enforcement was untested.

## Later code import and control transport

The documented shared ancestor is available locally. Its presence on the future recipient is **pending**. After separate connection/operational authorization, identify an existing recipient repository containing that commit. If absent, stop and prepare a small bridge from a genuinely available shared ancestor; do not fetch/bundle all history blindly.

Copy the reviewed bundle, exact lock and baseline-accounting evidence to the external remote control directory; preserve source copies and drafts. Later approved records/index must be distinct from preserved DRAFT snapshots and supplied by actual instruction. Before import, compare the bundle's remotely measured SHA-256 to the inventory pin, then:

```bash
: "${E_RECIPIENT_REPO:?existing recipient repository must be identified}"
git -C "$E_RECIPIENT_REPO" cat-file -e 754ff1d7d362cf76ddb2bca681b32b2aeb7118f5^{commit}
git -C "$E_RECIPIENT_REPO" bundle verify "$E_REMOTE_ROOT/control/producer.bundle"
git bundle list-heads "$E_REMOTE_ROOT/control/producer.bundle"
# Require exactly HEAD at b2c6dde71e82f5f0af308761bf19ed3341670587.
# Require the destination ref/path to be absent; preserve a collision.
git -C "$E_RECIPIENT_REPO" fetch "$E_REMOTE_ROOT/control/producer.bundle" \
  HEAD:refs/review/final-e-raw-20261004-b2c6dde-88b114dd-v1
git -C "$E_RECIPIENT_REPO" worktree add --detach \
  "$E_REMOTE_ROOT/execution-code" b2c6dde71e82f5f0af308761bf19ed3341670587
test "$(git -C "$E_REMOTE_ROOT/execution-code" rev-parse HEAD)" = b2c6dde71e82f5f0af308761bf19ed3341670587
test -z "$(git -C "$E_REMOTE_ROOT/execution-code" status --porcelain)"
test ! -e "$E_REMOTE_ROOT/campaign"
mkdir "$E_REMOTE_ROOT/operation-receipts"
mkdir "$E_REMOTE_ROOT/manifests"
```

Capture HEAD, tree SHA, detached/clean status, source bundle/lock hashes and the actual approved-record/index pins as code/control evidence before extraction. These small evidence files belong under remote `control`, never in code. Importing code or a finalized scientific lock does not authorize execution.

To reproduce the bundle locally, choose an absent external destination and run from the clean detached producer; use one pack thread and the shared CPU lock. The receipt records the delivered bytes. Reproduction produces a new transport artifact whose own bytes/head/prerequisite must be inspected and recorded:

```bash
: "${E_NEW_BUNDLE:?fresh external reproduction path required}"
test ! -e "$E_NEW_BUNDLE"
git -c pack.threads=1 -C "$E_MAC_ROOT/execution-code" bundle create \
  "$E_NEW_BUNDLE" HEAD ^754ff1d7d362cf76ddb2bca681b32b2aeb7118f5
git -C "$E_MAC_ROOT/execution-code" bundle verify "$E_NEW_BUNDLE"
git bundle list-heads "$E_NEW_BUNDLE"
shasum -a 256 "$E_NEW_BUNDLE"
```

## Closed scopes and future command forms

All eleven committed authorization drafts retain `approved=false`, `instruction=null`, `reviewer=null`, and `status=DRAFT`. Each binds the exact campaign, producer, lock hash, one phase/model and one absolute output. Remote and local paths differ. There are no wildcard, `evaluate` or scientific `audit` drafts.

| Environment | Model | Separate authorization phases | Runner commands |
|---|---|---|---|
| Remote | Qwen | extract, token_audit, accept, verify | Same names |
| Remote | Llama | extract, token_audit, accept, verify | Same names |
| Remote | both | transfer | transfer |
| Mac | both | transfer | verify-transfer |
| Mac | both | verify | verify |

For the later authorized remote attempt, perform the commands from the detached remote producer. These environment settings and expected runtime/cache paths do not certify their current existence:

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
E_PY=/workspace/checkpoint-r2-fresh-pilot-20261002-3f9747f-volume/runtime/venv/bin/python
E_CACHE=/workspace/checkpoint-r2-fresh-pilot-20261002-3f9747f-volume/runtime/model-cache
E_VOLUME=/workspace
E_CODE="$E_REMOTE_ROOT/execution-code"
E_OUTPUT="$E_REMOTE_ROOT/campaign"
E_LOCK="$E_REMOTE_ROOT/control/final_lock_20261004.json"
E_LOCK_SHA=88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e
E_RECEIPTS="$E_REMOTE_ROOT/operation-receipts"
E_AUTH_INDEX="$E_REMOTE_ROOT/control/approved-authorization-index.json"
: "${E_AUTH_INDEX_SHA:?separately reviewed approved-index hash required}"
cd "$E_CODE"
```

The later approved index has entries `"model:phase": {"file": "absolute path", "sha256": "approved record hash"}`. Its own hash must be independently supplied. It is not created or approved by this package. This helper reads those pins; it never approves records or substitutes draft hashes:

```bash
resolve_auth() {
  local key="$1" pair
  pair=$("$E_PY" - "$E_AUTH_INDEX" "$E_AUTH_INDEX_SHA" "$key" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]);assert hashlib.sha256(p.read_bytes()).hexdigest()==sys.argv[2]
entry=json.loads(p.read_bytes())[sys.argv[3]];f=pathlib.Path(entry['file'])
assert f.is_absolute() and len(entry['sha256'])==64
assert hashlib.sha256(f.read_bytes()).hexdigest()==entry['sha256']
a=json.loads(f.read_bytes());assert a['approved'] is True and a['reviewer'] and a['instruction']
print(str(f));print(entry['sha256'])
PY
  ) || return
  E_AUTH_FILE=${pair%%$'\n'*}
  E_AUTH_SHA=${pair#*$'\n'}
}
e_phase() {
  local command="$1" model="$2" phase="$1"
  shift 2
  if [ "$command" = verify-transfer ]; then phase=transfer; fi
  resolve_auth "$model:$phase" || return
  "$E_PY" -m src.final_e_runner "$command" --model "$model" --output "$E_OUTPUT" \
    --lock "$E_LOCK" --lock-sha256 "$E_LOCK_SHA" \
    --authorization "$E_AUTH_FILE" --authorization-sha256 "$E_AUTH_SHA" "$@"
}
```

Run sequentially, stopping on the first failure. Extraction is the supervised public runner; the recorded background PID is its Python supervisor, followed immediately by `wait`. Do not pipe it through another process or launch Llama concurrently:

```bash
resolve_auth qwen:extract
"$E_PY" -m src.final_e_runner extract --model qwen --output "$E_OUTPUT" \
  --lock "$E_LOCK" --lock-sha256 "$E_LOCK_SHA" \
  --authorization "$E_AUTH_FILE" --authorization-sha256 "$E_AUTH_SHA" \
  --cache "$E_CACHE" --volume "$E_VOLUME" > "$E_RECEIPTS/qwen-extract.log" 2>&1 &
E_SUPERVISOR_PID=$!
printf '%s\n' "$E_SUPERVISOR_PID" > "$E_RECEIPTS/qwen-supervisor.pid"
wait "$E_SUPERVISOR_PID"
e_phase token_audit qwen --cache "$E_CACHE" --operation-receipt "$E_RECEIPTS/qwen-token_audit.json"
e_phase accept qwen --operation-receipt "$E_RECEIPTS/qwen-accept.json"
e_phase verify qwen --operation-receipt "$E_RECEIPTS/qwen-verify.json"

resolve_auth llama:extract
"$E_PY" -m src.final_e_runner extract --model llama --output "$E_OUTPUT" \
  --lock "$E_LOCK" --lock-sha256 "$E_LOCK_SHA" \
  --authorization "$E_AUTH_FILE" --authorization-sha256 "$E_AUTH_SHA" \
  --cache "$E_CACHE" --volume "$E_VOLUME" > "$E_RECEIPTS/llama-extract.log" 2>&1 &
E_SUPERVISOR_PID=$!
printf '%s\n' "$E_SUPERVISOR_PID" > "$E_RECEIPTS/llama-supervisor.pid"
wait "$E_SUPERVISOR_PID"
e_phase token_audit llama --cache "$E_CACHE" --operation-receipt "$E_RECEIPTS/llama-token_audit.json"
e_phase accept llama --operation-receipt "$E_RECEIPTS/llama-accept.json"
e_phase verify llama --operation-receipt "$E_RECEIPTS/llama-verify.json"
e_phase transfer both --manifest "$E_REMOTE_ROOT/manifests/campaign-checksums.json" \
  --operation-receipt "$E_RECEIPTS/both-transfer.json"
```

There is no standalone invented preflight or uncharged model test. The authorized extractor performs the accepted runtime/dependency/driver and pinned offline-snapshot checks, native tokenization, filesystem checks and actual allocation/fsync/readback of the complete two-copy/temporary-shard reserve before inference. Then model loading, forward/writing/verification and teardown are measured/charged. Token audit is a separate tokenizer-only process. Each model requires 19,190 texts, 19,424 logical bindings, 150 complete shards, index/completion-marker pins, independent token audit and acceptance. The transfer runner rechecks both accepted models and settled accounting.

## Accounting and failure policy

Frozen carried charges remain Qwen **160.83801023702836** and Llama **123.82646256202133** seconds. Caps remain 7,200 seconds/model, 18,000 cumulative extraction-worker seconds including carried charges, 1,800 seconds/setup phase, and separately 10,800 seconds/CPU job. The total is a stopping limit, not project elapsed time or a forecast. The historical raw contract's older cap fields do not replace the final-lock caps.

Local receipt reconciliation does not adopt accounting or certify current remote charges. The later authorized extraction launcher performs adoption through `final_e_raw.launch` after the access gate. New/inconsistent charges must be reviewed first; never edit/reset carryforward or substitute a fresh ledger. Current runtime, token lengths, CUDA peaks, persistent storage/quota and transfer throughput remain unmeasured. Free-space reports and historical success do not establish current readiness.

Stop on the first mismatch, cap hit or failure. Preserve all failed attempts, partials, ledgers and supervisor/operation receipts. No automatic retry, package installation, snapshot download, runtime replacement, tolerance change or producer modification. A normal stop uses TERM only on the recorded, identity-checked exact supervisor PID; wait for its exit and settled receipt. Do not use broad kill commands. Resume requires failure review and a new scoped instruction/authorization with the same campaign, producer and lock. Hard supervisor loss or missing/inconsistent settlement blocks resume pending accounting review. A completed model index forbids re-extraction.

Success-transfer is unavailable after an incomplete/failed acquisition. Preserve the failure evidence and obtain a separately scoped failure-handoff instruction if necessary; do not represent a partial copy as accepted raw.

## Durable transfer and independent manifest pins

After both remote models are accepted and transfer-publication has completed, verify recorded supervisor PIDs have exited, no campaign runner/worker remains, NVIDIA compute applications are stopped, and `full-budget.json` plus every supervisor settlement agree. Capture the stopped-worker/code/accounting observation in remote control evidence. If any process/receipt state is ambiguous, stop; do not shut down the Pod or create a successful handoff claim.

The campaign manifest is created only by the authorized runner. Create two additional byte manifests outside their source directories: one for all external operation receipts/logs/PID files, one for complete control evidence (bundle, lock, baseline receipts, actual authorizations/index and code/stopped-worker observations). This generic helper is for **control/receipt directories only**, not a substitute for gated campaign acceptance:

```bash
byte_manifest() {
  "$E_PY" - "$1" "$2" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).resolve();out=pathlib.Path(sys.argv[2]).resolve()
assert not out.is_relative_to(root) and not out.exists()
files={}
for p in sorted(root.rglob('*')):
    assert not p.is_symlink()
    if p.is_file():
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        files[str(p.relative_to(root))]={'bytes':p.stat().st_size,'sha256':h.hexdigest()}
with out.open('x') as f:json.dump({'files':files},f,indent=2,sort_keys=True)
PY
}
byte_manifest "$E_RECEIPTS" "$E_REMOTE_ROOT/manifests/operation-receipt-checksums.json"
byte_manifest "$E_REMOTE_ROOT/control" "$E_REMOTE_ROOT/manifests/control-checksums.json"
sha256sum "$E_REMOTE_ROOT/manifests/campaign-checksums.json" \
  "$E_REMOTE_ROOT/manifests/operation-receipt-checksums.json" \
  "$E_REMOTE_ROOT/manifests/control-checksums.json"
```

From the Mac, capture those three hashes independently through the current verified SSH connection, **before** copying their manifests. Use privately supplied current SSH values and a verified known-hosts file; no historical endpoint is implied. Store the captured output separately, then transfer to new durable destinations. Record copy start/end/bytes and measured throughput; do not make a throughput forecast from old receipts.

```bash
set -euo pipefail
# Set both root variables from the absolute-path block above in this Mac shell.
test ! -e "$E_MAC_ROOT/handoff"
mkdir "$E_MAC_ROOT/handoff"
: "${E_SSH_TARGET:?current privately supplied connection required}"
: "${E_SSH_PORT:?current port required}"
: "${E_SSH_KEY:?existing private key location required}"
: "${E_KNOWN_HOSTS:?independently verified current host key required}"
E_SSH=(ssh -p "$E_SSH_PORT" -i "$E_SSH_KEY" -o StrictHostKeyChecking=yes -o "UserKnownHostsFile=$E_KNOWN_HOSTS")
"${E_SSH[@]}" "$E_SSH_TARGET" \
  "sha256sum '$E_REMOTE_ROOT/manifests/campaign-checksums.json' '$E_REMOTE_ROOT/manifests/operation-receipt-checksums.json' '$E_REMOTE_ROOT/manifests/control-checksums.json'" \
  > "$E_MAC_ROOT/handoff/independently-captured-remote-manifest-pins.txt"
E_RSYNC_SSH="ssh -p $E_SSH_PORT -i $E_SSH_KEY -o StrictHostKeyChecking=yes -o UserKnownHostsFile=$E_KNOWN_HOSTS"
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$E_MAC_ROOT/handoff/copy-start-utc.txt"
rsync -a --stats -e "$E_RSYNC_SSH" "$E_SSH_TARGET:$E_REMOTE_ROOT/campaign/" "$E_MAC_ROOT/handoff/campaign/" > "$E_MAC_ROOT/handoff/copy-campaign.log" 2>&1
rsync -a --stats -e "$E_RSYNC_SSH" "$E_SSH_TARGET:$E_REMOTE_ROOT/operation-receipts/" "$E_MAC_ROOT/handoff/remote-operation-receipts/" > "$E_MAC_ROOT/handoff/copy-receipts.log" 2>&1
rsync -a --stats -e "$E_RSYNC_SSH" "$E_SSH_TARGET:$E_REMOTE_ROOT/control/" "$E_MAC_ROOT/handoff/remote-control/" > "$E_MAC_ROOT/handoff/copy-control.log" 2>&1
rsync -a --stats -e "$E_RSYNC_SSH" "$E_SSH_TARGET:$E_REMOTE_ROOT/manifests/" "$E_MAC_ROOT/handoff/" > "$E_MAC_ROOT/handoff/copy-manifests.log" 2>&1
date -u '+%Y-%m-%dT%H:%M:%SZ' > "$E_MAC_ROOT/handoff/copy-end-utc.txt"
```

Preserve remote/source copies and resumable failed transfer logs. If a copy fails, do not continue to verification/safe-stop; an explicit reviewed transfer-continuation instruction must preserve the same destinations and identities. Verify each downloaded manifest's hash against its independently captured remote pin. Verify exact file coverage, relative-path containment, no symlinks, sizes and streamed SHA-256 for the copied receipt/control directories using their respective manifests; this does not establish loader acceptance. Compare copied lock/bundle to the package pins and source-code identity to the exact producer; retain all actual approval/accounting evidence.

The following prospective Mac check verifies all three manifest pins and exact receipt/control coverage. Run with one numerical thread and the shared CPU lock after other substantial local jobs finish. It hashes only receipt/control evidence here; the gated campaign checksum and loader checks follow separately.

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 - "$E_MAC_ROOT" <<'PY'
import fcntl,hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);handoff=root/'handoff'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
with open('/tmp/truth-probing-cpu-exclusive.lock','a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
    pins={}
    for line in (handoff/'independently-captured-remote-manifest-pins.txt').read_text().splitlines():
        digest,name=line.split(None,1);name=pathlib.Path(name.strip()).name
        assert len(digest)==64 and name not in pins
        pins[name]=digest
    assert set(pins)=={'campaign-checksums.json','operation-receipt-checksums.json','control-checksums.json'}
    for name,digest in pins.items():assert sha(handoff/name)==digest
    for name,folder in [('operation-receipt-checksums.json','remote-operation-receipts'),('control-checksums.json','remote-control')]:
        base=handoff/folder;files=json.loads((handoff/name).read_text())['files']
        actual={str(p.relative_to(base)) for p in base.rglob('*') if p.is_file()}
        assert actual==set(files)
        for rel,expected in files.items():
            p=base/rel
            assert p.resolve().is_relative_to(base.resolve())
            assert not any(q.is_symlink() for q in [p,*p.parents] if q.is_relative_to(base))
            assert p.stat().st_size==expected['bytes'] and sha(p)==expected['sha256']
    assert sha(handoff/'remote-control/final_lock_20261004.json')=='88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e'
    assert sha(handoff/'remote-control/producer.bundle')=='da28abe3c64e50bffee3047003a621ee29b60dd46ceafd3052157ae796b7a1aa'
    receipt=root/'local-operation-receipts/control-receipt-checksums.json'
    receipt.parent.mkdir(exist_ok=True)
    with receipt.open('x') as f:json.dump({'status':'source_manifest_pins_and_receipt_control_coverage_verified','pins':pins},f,indent=2)
PY
```

For the later Mac checks, use the clean detached Mac producer and separately approved Mac scopes. Define `resolve_auth` from the hash-checking helper above in the Mac shell (it does not persist from the remote shell). Set `E_PY` to the Mac interpreter; `E_OUTPUT` to `$E_MAC_ROOT/handoff/campaign`; `E_LOCK` to `$E_MAC_ROOT/transport/final_lock_20261004.json`; `E_AUTH_INDEX` to `$E_MAC_ROOT/control/approved-authorization-index.json`, with its independently supplied Mac index pin. The lock hash and campaign identity remain unchanged. Set one numerical thread and acquire the shared exclusive CPU lock for both checksum and full-loader checks, after inspecting other substantial jobs; do not compete with the unrelated correlation job.

```bash
set -euo pipefail
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
E_LOCK_SHA=88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e
E_PY=/opt/homebrew/opt/python@3.11/bin/python3.11
E_CODE="$E_MAC_ROOT/execution-code"
E_OUTPUT="$E_MAC_ROOT/handoff/campaign"
E_LOCK="$E_MAC_ROOT/transport/final_lock_20261004.json"
E_AUTH_INDEX="$E_MAC_ROOT/control/approved-authorization-index.json"
: "${E_AUTH_INDEX_SHA:?separately reviewed Mac approved-index hash required}"
test -d "$E_MAC_ROOT/local-operation-receipts"
# This fresh directory was created by the manifest/control check above.
cd "$E_CODE"
locked_phase() {
  local command="$1" phase="$1"
  shift
  if [ "$command" = verify-transfer ]; then phase=transfer; fi
  resolve_auth "both:$phase" || return
  "$E_PY" - "$E_PY" -m src.final_e_runner "$command" --model both --output "$E_OUTPUT" \
    --lock "$E_LOCK" --lock-sha256 "$E_LOCK_SHA" \
    --authorization "$E_AUTH_FILE" --authorization-sha256 "$E_AUTH_SHA" "$@" <<'PY'
import fcntl,subprocess,sys
with open('/tmp/truth-probing-cpu-exclusive.lock','a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
    subprocess.run(sys.argv[1:],check=True)
PY
}
locked_phase verify-transfer --manifest "$E_MAC_ROOT/handoff/campaign-checksums.json" \
  --operation-receipt "$E_MAC_ROOT/local-operation-receipts/campaign-checksums.json"
locked_phase verify --operation-receipt "$E_MAC_ROOT/local-operation-receipts/full-loader.json"
```

Checksum verification is followed by full reopened-shard/token/index/completion/provenance/acceptance/accounting loader verification for both models. The Mac records are external and fresh. Both operations retain the existing 1,800-second setup cap; the separate 10,800-second CPU cap is for later scientific CPU jobs and is not substituted here. Capture final source-worker stopped state before release; do not mutate sealed source receipt/control trees—new observations go in new external evidence with their own pins.

**Future safe-stop checkpoint:** only after complete durable campaign/receipt/control transfer, independent manifest-pin checks, full local loader acceptance, exact code/lock/authorization availability and confirmed stopped remote workers, record `$E_MAC_ROOT/local-operation-receipts/safe-stop-handoff.json` with those receipt/hash references and announce: “Raw acquisition and durable Mac handoff are accepted; workers are stopped; RunPod can be stopped by the user. Scientific evaluation and audit remain unauthorized.” Hand shutdown back to the user; start no new remote task. This preparation does not announce that checkpoint.

## Validation, remaining inputs and stop

The lightweight package validator verifies lock/six pins, transport inventory/bundle, paths, statically parsed CLI arguments and all eleven draft rejections through the existing access gate. It never invokes runner commands or raw functions, loads models, opens E arrays, reruns numerical suites or runs historical validators. Fresh evidence is separate from all prior receipts. Choose an absent, unique recheck directory below; the validator refuses to overwrite receipts. The active correlation job is preserved; no substantial numerical job is launched.

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.11/bin/python3.11 \
  results/final_e_launch_prep_20261004/validate_preparation.py \
  /private/tmp/final-e-launch-prep-recheck-UNIQUE
```

Outstanding: actual connection/host key and recipient prerequisite; current mount/runtime/cache/snapshot availability; actual token plans/lengths, reservation/fsync/readback/quota and CUDA measurements; no intervening or inconsistent charges; genuine phase instructions/approved records and independently supplied pins; measured transfer throughput and accepted durable handoff. A missing item is a specific operational gap, not grounds to alter the approved science or claim readiness.

Proposed later authorization scope is **raw acquisition and durable handoff only**: the eleven named remote/Mac phases at their exact paths, supervised sequential extraction with existing caps/checks, independent token audit/acceptance/verification, manifest publication and verified durable transfer, full local loader checks and safe-stop/shutdown handoff. It excludes evaluation, scientific audit, predictions/metrics, fitting, calibration, scientific redesign and automatic retries. No such authorization is issued now.

Publish only this additive preparation package on `checkpoint-final-e-launch-prep-20261004`, verify remote SHA and stop. Do not merge or execute after publication.
