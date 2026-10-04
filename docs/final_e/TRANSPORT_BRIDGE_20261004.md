# Final-E Git transport supplement, October 4

**Local code-transport preparation only; stop for supplement review.** No Pod contact, remote import/mutation, budget adoption, E phase, model loading or numerical audit occurred. The scientific lock and eleven-phase execution approvals remain recorded; this supplement changes transport only.

## Delivered bridge and unchanged identities

| Item | Identity |
|---|---|
| Supplement base / blocker delivery | `adf7ef9a4aaa6b38757757be323278a792b6a61b` |
| Preparation delivery | `9939c7a26c5ec198f3b153d53567e5ec655c3cc3` |
| Sole bridge prerequisite | `6865bfa43c52d9c5d7d3c96df90351ad8f895aea` |
| Sole advertised bridge target | `754ff1d7d362cf76ddb2bca681b32b2aeb7118f5`, ref `refs/heads/final-e-bridge-20261004` |
| Bridge size / SHA-256 | **1,578,104 bytes**, `0fb71490a922b3bf5bb29ae233a3c60d1129b9b58ae2405601f6c1823ea67341` |
| Unchanged original producer bundle | **176,023 bytes**, `da28abe3c64e50bffee3047003a621ee29b60dd46ceafd3052157ae796b7a1aa` |
| Unchanged producer | `b2c6dde71e82f5f0af308761bf19ed3341670587` |
| Exact producer tree | `93972950bca996dbfa9794dedf07934868b42e98` |
| Unchanged final-lock SHA-256 | `88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e` |
| Unchanged remote authorization-index SHA-256 | `86ccbb56dc882ec42c7865b1b590c4a42703c8ac3c286d74352eb3c3dd33c908` |
| Unchanged local authorization-index SHA-256 | `8e3ee5103379a8ccd3805439deb1ffeb3fbf9fddb2bbcf3fbf1573bd187362e5` |

Durable supplement directory:
`/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/transport-supplement-20261004-6865bfa-to-754ff1d-v3`

It contains `bridge-6865bfa-to-754ff1d.bundle`, `bridge-inventory.json`, `object-inspection.json` and `two-stage-import-proof.json`. The bundle stays outside Git. Small evidence copies are in `results/final_e_transport_bridge_20261004/`.

The recorded raw-only recipient path was recovered from the private inspection evidence:
`/workspace/checkpoint-next-raw-only-code-6865bfa43c52d9c5d7d3c96df90351ad8f895aea`.
Its observed HEAD was the bridge prerequisite; this is historical evidence, not a current recipient observation. No SSH endpoint, key location or private connection value is published.

## Construction and independent local proof

Local ancestry verification confirmed ten commits from the prerequisite to the target. Only the 158 newly reachable Git objects were inspected. The largest blob is the existing 2,044,433-byte `adapter-bindings.json` manifest. No new activation/parameter archives, model weights, caches, credential files or private-key/token patterns were found. The bridge carries the missing ten-commit history, without unrelated branches.

Native `git bundle create` enumerates both boundary prerequisites: `6865bfa…` and `fc083594dc17d1fc38f7b52dd8030b0a7f2cb67d`. The latter is already an ancestor of the former. The delivered v2-format bundle removes only that implied ancestor's prerequisite header line, after proving ancestry; its advertised target and Git pack payload are unchanged. Complete prerequisite history is required. Normalization details and pack hash are recorded in the inventory. Earlier packaging attempts and their failure receipts remain external and are not the delivered bridge.

The test recipient was initialized empty and seeded by ordinary local fetch with only history reachable from `6865bfa…`. Its 2,233 initial objects exactly equal that reachable set. It had no alternates, remote configuration, shared/hardlinked object storage or hidden later commits.

| Checkpoint | `6865bfa` | `754ff1d` | `b2c6dde` |
|---|---|---|---|
| Before imports | Present | Absent | Absent |
| After bridge import | Present | Present | Absent |
| After unchanged original bundle import | Present | Present | Present |

Both bundles passed `git bundle verify` in that recipient. Imports used fresh refs and object checking; full strict `git fsck` passed after each stage. The final producer commit bytes and tree match the source exactly. A clean detached test checkout verified all six immutable scientific pins against the unchanged external final lock. Commands, return codes and identities are in the import proof. Git work ran with one pack thread under the shared exclusive CPU lock; measured proof duration was 26.85 seconds.

All original transport files/inventory, eleven approved controls, both indexes, draft/lock/scientific files, historical receipts, accounting metadata, unrelated files and prior worktrees were preserved. The execution checkout was not modified. No carried charge changed or budget was adopted.

## Later recipient procedure — pending, not executed

First review this exact supplement. For a later authorized attempt, re-establish the actual current Pod connection and trusted host key. A stopped/restarted Pod may have changed endpoints or host identity. Do not reuse an old connection by assumption. Recheck the actual repository, complete prerequisite history, mount and destination collisions before import.

Copy the bridge and its exact inventory into new external remote control paths below. Preserve `control/producer.bundle` and the original transport inventory unchanged. Include the bridge, inventory and subsequent import receipt in the later **control manifest**, in addition to all existing controls/accounting/code evidence. Keep the separately published campaign and operation-receipt manifests distinct.

The following commands are prospective, do not run as part of supplement preparation. Use `set -euo pipefail`; stop on any mismatch or existing destination/ref. No reset, moving-branch pull, merge, force update or execution-checkout modification is needed.

```bash
set -euo pipefail
E_RECIPIENT_REPO=/workspace/checkpoint-next-raw-only-code-6865bfa43c52d9c5d7d3c96df90351ad8f895aea
E_REMOTE_ROOT=/workspace/final-e-raw-20261004-b2c6dde-88b114dd-v1
E_BRIDGE="$E_REMOTE_ROOT/control/transport-supplement/bridge-6865bfa-to-754ff1d.bundle"
E_BRIDGE_INVENTORY="$E_REMOTE_ROOT/control/transport-supplement/bridge-inventory.json"
E_ORIGINAL="$E_REMOTE_ROOT/control/producer.bundle"
E_BRIDGE_BASE=6865bfa43c52d9c5d7d3c96df90351ad8f895aea
E_BRIDGE_TARGET=754ff1d7d362cf76ddb2bca681b32b2aeb7118f5
E_PRODUCER=b2c6dde71e82f5f0af308761bf19ed3341670587
E_BRIDGE_REF=refs/review/final-e-bridge-20261004-754ff1d
E_PRODUCER_REF=refs/review/final-e-raw-20261004-b2c6dde-88b114dd-v1

# Preserve any existing worktree state; import only through new refs.
git -C "$E_RECIPIENT_REPO" status --porcelain
test "$(git -C "$E_RECIPIENT_REPO" rev-parse HEAD)" = "$E_BRIDGE_BASE"
test "$(git -C "$E_RECIPIENT_REPO" rev-parse --is-shallow-repository)" = false
git -C "$E_RECIPIENT_REPO" cat-file -e "$E_BRIDGE_BASE^{commit}"
git -C "$E_RECIPIENT_REPO" merge-base --is-ancestor \
  fc083594dc17d1fc38f7b52dd8030b0a7f2cb67d "$E_BRIDGE_BASE"
! git -C "$E_RECIPIENT_REPO" show-ref --verify --quiet "$E_BRIDGE_REF"
! git -C "$E_RECIPIENT_REPO" show-ref --verify --quiet "$E_PRODUCER_REF"
test ! -e "$E_REMOTE_ROOT/execution-code"

printf '%s  %s\n' \
  0fb71490a922b3bf5bb29ae233a3c60d1129b9b58ae2405601f6c1823ea67341 "$E_BRIDGE" \
  da28abe3c64e50bffee3047003a621ee29b60dd46ceafd3052157ae796b7a1aa "$E_ORIGINAL" \
  | sha256sum --check --strict
E_PY=/workspace/checkpoint-r2-fresh-pilot-20261002-3f9747f-volume/runtime/venv/bin/python
"$E_PY" - "$E_BRIDGE" "$E_BRIDGE_INVENTORY" <<'PY'
import hashlib,json,pathlib,sys
bundle=pathlib.Path(sys.argv[1]);inventory=pathlib.Path(sys.argv[2])
assert bundle.stat().st_size==1578104
assert hashlib.sha256(inventory.read_bytes()).hexdigest()=='2448da945e2ad60519e12ea80073bac2a5e4a52381c807b0e241ae76eaa9069d'
head=bundle.read_bytes().split(b'\n\n',1)[0].decode().splitlines()
assert [s.split()[0][1:] for s in head if s.startswith('-')]==['6865bfa43c52d9c5d7d3c96df90351ad8f895aea']
assert [s for s in head if s and not s.startswith(('#','-'))]==['754ff1d7d362cf76ddb2bca681b32b2aeb7118f5 refs/heads/final-e-bridge-20261004']
assert json.loads(inventory.read_bytes())['bundle']['sha256']=='0fb71490a922b3bf5bb29ae233a3c60d1129b9b58ae2405601f6c1823ea67341'
PY
git -C "$E_RECIPIENT_REPO" bundle verify "$E_BRIDGE"
test "$(git bundle list-heads "$E_BRIDGE")" = \
  "$E_BRIDGE_TARGET refs/heads/final-e-bridge-20261004"
# Record measured bytes/hash/header and verification output before importing.
git -c pack.threads=1 -c transfer.fsckObjects=true -C "$E_RECIPIENT_REPO" \
  fetch --no-tags --no-write-fetch-head "$E_BRIDGE" \
  "refs/heads/final-e-bridge-20261004:$E_BRIDGE_REF"
test "$(git -C "$E_RECIPIENT_REPO" rev-parse "$E_BRIDGE_REF")" = "$E_BRIDGE_TARGET"
git -C "$E_RECIPIENT_REPO" cat-file -e "$E_BRIDGE_TARGET^{commit}"

git -C "$E_RECIPIENT_REPO" bundle verify "$E_ORIGINAL"
test "$(git bundle list-heads "$E_ORIGINAL")" = "$E_PRODUCER HEAD"
git -c pack.threads=1 -c transfer.fsckObjects=true -C "$E_RECIPIENT_REPO" \
  fetch --no-tags --no-write-fetch-head "$E_ORIGINAL" "HEAD:$E_PRODUCER_REF"
test "$(git -C "$E_RECIPIENT_REPO" rev-parse "$E_PRODUCER_REF")" = "$E_PRODUCER"
test "$(git -C "$E_RECIPIENT_REPO" rev-parse "$E_PRODUCER^{tree}")" = \
  93972950bca996dbfa9794dedf07934868b42e98
git -C "$E_RECIPIENT_REPO" worktree add --detach "$E_REMOTE_ROOT/execution-code" "$E_PRODUCER"
test "$(git -C "$E_REMOTE_ROOT/execution-code" rev-parse HEAD)" = "$E_PRODUCER"
test -z "$(git -C "$E_REMOTE_ROOT/execution-code" status --porcelain)"
! git -C "$E_REMOTE_ROOT/execution-code" symbolic-ref -q HEAD
test "$(git -C "$E_RECIPIENT_REPO" rev-parse HEAD)" = "$E_BRIDGE_BASE"
```

Verify the final lock, all six immutable pins, the remote index and every remote control hash before any phase. The local index and its two controls remain separately pinned at their original Mac paths. Locks/authorizations stay outside both producer checkouts. Use the existing launch runbook's gate and command forms; no E command is added here.

After the exact reviewed controls have been transported at their original paths, this prospective standard-library check verifies their identities without dispatching any phase:

```bash
"$E_PY" - "$E_REMOTE_ROOT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);code=root/'execution-code';control=root/'control'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
lockfile=control/'final_lock_20261004.json'
assert sha(lockfile)=='88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e'
lock=json.loads(lockfile.read_bytes())
assert lock['producer_sha']=='b2c6dde71e82f5f0af308761bf19ed3341670587'
assert len(lock['immutable_contract'])==6
for rel,digest in lock['immutable_contract'].items():assert sha(code/rel)==digest
indexfile=control/'approved-authorization-index.json'
assert sha(indexfile)=='86ccbb56dc882ec42c7865b1b590c4a42703c8ac3c286d74352eb3c3dd33c908'
index=json.loads(indexfile.read_bytes())
assert set(index)=={m+':'+p for m in ('qwen','llama') for p in ('extract','token_audit','accept','verify')}|{'both:transfer'}
for key,pin in index.items():
    model,phase=key.split(':');path=pathlib.Path(pin['file'])
    assert path.is_absolute() and path.is_relative_to(control/'authorizations')
    assert sha(path)==pin['sha256']
    a=json.loads(path.read_bytes())
    assert a['approved'] is True and a['status']=='APPROVED'
    assert a['phase']==phase and a['model']==model
    assert a['producer_sha']==lock['producer_sha'] and a['lock_sha256']==sha(lockfile)
    assert a['campaign_id']=='final-e-raw-20261004-b2c6dde-88b114dd-v1'
    assert a['output']==str(root/'campaign')
print('producer, lock, six immutable pins and nine remote approved controls verified; no dispatch')
PY
```

Write a fresh external `control/transport-supplement/recipient-import.json` recording actual repository identity, pre-import state, commands/outcomes, both measured bundle hashes/heads/prerequisites, bridge-inventory hash, newly created refs, exact producer/tree, clean/detached status, immutable/control pins and preserved source HEAD/worktree state. It must reflect observations; it is not supplied or approved by this preparation. Source evidence is sealed only after the later authorized execution/handoff procedure.

**Remaining checks:** supplement review; actual recipient/connection/history and bridge import; complete runtime/dependency/snapshot verification; real allocation/fsync/readback storage reservations and quota; native tokenization/CUDA measurements; current accounting reconciliation and normal adoption; supervised phases and durable loader-accepted handoff. This transport proof certifies none of those live checks. The approved campaign, producer, lock, caps and eleven control scopes remain unchanged. No new scientific approval or budget reset is needed.

Publish the additive supplement on `checkpoint-final-e-transport-bridge-20261004` and stop. Do not import on RunPod or launch E before this supplement has been reviewed.
