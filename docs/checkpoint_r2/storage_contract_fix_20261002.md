# R2 storage permission-contract correction for review

Isolated branch `checkpoint-r2-storage-contract-fix-20261002`, based exactly on
`7933063036f5d57c83f41b8ea46e560c1f10cd2e`; origin fetched without pull or merge.
This task ran no extraction, numerical pilot, fitting, E evaluation or full-capacity
allocation. Historical code, frozen configuration and original worktrees are preserved.

## Evidence and diagnosis

Verified all 16 failure receipt files at
`/Users/apple/truth-probing-backups/20261002T175950Z-r2-full-raw-7933063/control`.
The unchanged receipt-manifest SHA-256 is
`90a99a36a86f25cc6d0aa2558f3fe5423eb16b9d9c96db6214df633ef74d7f51`.
Compared the successful pilot's `handoff-setup/preflight.json` and writer receipt
under `/Users/apple/projects/checkpoint-r2-fresh-pilot-handoff-20261002-3f9747f-volume`.
Both observations identify `/workspace`, source
`mfs#ca-mtl-1.runpod.net:9421[/podvolumes/urjz17umkbo6/abvvcizoq2pw0g]`, filesystem `fuse`.
There is no evidence that this mount source changed. The pilot receipt says chmod
passed, but records no resulting mode bits. It cannot establish exact 0600 reporting.

The failed successor's temporary file reported 0666 before and after successful
`chmod(0600)`. A small probe on the already-running Pod reproduced this. This is
an observed mode-reporting mismatch; the underlying FUSE/provider cause and actual
access enforcement remain unknown. No test under another identity was performed.
Mode bits alone are not evidence of owner-only confidentiality or its absence.

## Narrow correction

`filesystem_prerequisites()` now requires explicit
`artifact_contract='r2-non-secret-research-artifacts-v1'`. The full raw storage
caller supplies that contract for public/non-secret research files, including
the campaign lock. Requests for private/credential/unknown contracts are rejected
before creating a probe; no default policy silently permits mode emulation.

The successful chmod call, requested/observed modes, mount/device evidence and
exact-mode reporting status are recorded separately. A returned mode mismatch is
`unsupported` reporting, rather than a fatal owner-only-mode requirement.
`access_enforcement` and `owner_only_protection` remain `untested`, even when
reported mode is 0600. All chmod errors propagate; this correction handles only
a successful call with differing reported bits, not arbitrary filesystem failures.

The real campaign opens a non-secret lock with append mode and uses exclusive
nonblocking `flock`; it never depended on chmod of that lock. The probe now matches
that open mode. A second process must receive only `BlockingIOError` while held,
and must acquire after release. Child errors cannot masquerade as lock exclusion.
Git initialization remains mandatory; the actual chosen checkout must still pass
the unchanged exact-commit/clean-checkout execution gate.

The receipt validator requires the explicit contract and passed Git/lock results,
rejects false protection/mode claims, and is required when a storage receipt is
consumed. Legacy list-only prerequisite receipts are not accepted. Existing
receipts are not edited or migrated.

Credential handling is unchanged. Tokens stay outside research outputs, cache
receipts and Git, in the existing external credential location. Credentials or
other private files require a separately protected location with independently
enforced access restrictions; this non-secret receipt never certifies that.
No credential was moved, read, probed or logged. Full extraction reuses snapshots
offline and does not need to relocate authentication.

The writer, loader and capacity arithmetic are unchanged. Cross-process exclusion,
exclusive publication, overwrite refusal, file/directory fsync, partial recovery,
corruption rejection and hash readback still block on failure. The prerequisite
receipt accurately labels writer/full-capacity checks `untested` at this stage;
the mandatory subsequent `storage_probe` must pass the entire calculated additional
reservation, including the conservative second copy, before a storage proof exists.

## Validation and preserved scope

**62 tests passed**: the 25 new storage-contract regression cases and 37 existing
full-raw tests. Coverage includes native and emulated modes, explicit scope,
chmod errors, failed locking/Git, misleading capability receipts, and publication,
overwrite, fsync, readback and recovery failures through a tiny synthetic storage
plan. Existing full-raw tests retain coverage/counts, corruption rejection and
interruption/accounting behavior. No model ran.

The reachable Pod's isolated diagnostic directory used a **4,096-byte** writer
reservation and **27,376 temporary logical bytes** including its tiny Git checkout.
It passed lock exclusion/release, a clean detached Git checkout, and the unchanged
writer's full set of process-level integrity operations. A second bounded probe
confirmed the corrected helper by source SHA-256:
`83af632f3977b3d5740c2a0c2e0507925f9bf3c2caa954224e9bca15b45cb222`.
Both temporary directories were removed; existing data and mount options were
unchanged. These probes do not prove production quota or power-loss durability.
Receipts: `results/checkpoint_r2_storage_contract_fix_20261002/`.

Frozen config SHA-256 remains
`ed15c662289d587cc9328bc43d980b42afe3d39e4dc352e334e0f504ebcd4ec1`.
The 35,843 texts / 41,490 bindings per model, batch 1 unpadded, model/runtime pins,
BF16/FP16 conventions, tolerances, 113 shards/model and 5,400/12,600/1,800-second
caps are unchanged. Recovery and full-campaign accounting code is unchanged.
The accepted pilot review remains `1dbb5484d1055e0cf80993e666171cfa1743e2a2`.

## Before authorized extraction resumes

Review this successor; use its exact SHA in a new detached remote checkout.
Preserve the failed control directory and receipt files. Recheck GPU/runtime,
all offline snapshot hashes and metadata with the existing environment. Run the
successor's full `storage-check` into a fresh receipt file at the original planned
output destination; actual full capacity remains unverified. Do not reuse the
4 KiB diagnostic as a quota proof. If every prerequisite passes, use the already
authorized durable full run with the successor SHA and unchanged limits. Inspect
any existing campaign first and retain its original budget on resume. The failed
attempt created no full campaign, model outputs or full-execution spend.
