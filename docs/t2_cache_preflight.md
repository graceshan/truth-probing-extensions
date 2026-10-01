# T2 atomic activation-cache preflight

This is a read-only preflight from integration commit
`dae31152eb35f91a28130acaef1c9630a681ef3f`, on the separate
`t2-cache-preflight-20261001` branch. It does not consume provisional admission,
exclusion, person-key, or correction decisions. It creates no final row-selection
map, A/P reservation, fits, predictions, benchmark, or historical artifact edits.
Mac 1's working tree and the integrated T1 branch are preserved.

The reuse contract is the final section of
`docs/selection_repair_t2a_source_audit.md`: original dataset, source-row index,
source-file hash, statement hash, exact sidecar, and representation must bind.
Matching filenames or row counts alone is insufficient.

## Result of this preflight

All four caches pass **local original-source and recorded representation checks**.
**Physical tensor verification is pending:** the specified Runpod connection
rejected authentication with `Permission denied (publickey,password)`. No remote
NumPy header, current tensor size, or actual tensor digest was obtained. Recorded
tensor identities are expectations, not inferred successful measurements.

| Cache | Rows | Recorded shape | Recorded bytes | Local binding | Physical identity |
|---|---:|---|---:|---|---|
| Qwen train | 3,144 | 3144 × 28 × 3584 | 631,013,504 | verified | pending |
| Qwen validation | 1,040 | 1040 × 28 × 3584 | 208,732,288 | verified | pending |
| Llama train | 3,144 | 3144 × 32 × 4096 | 824,180,864 | verified | pending |
| Llama validation | 1,040 | 1040 × 32 × 4096 | 272,629,888 | verified | pending |

The compact report and machine-readable manifest are in
`results/t2_cache_preflight_20261001/compatibility.md` and `compatibility.json`.
The JSON includes expected tensor SHA-256 digests, inventory-mapped evidence,
ordered-row digests, representation and tokenizer identities, the remote attempt,
missing evidence, and concrete reuse/extraction requirements.

## What was checked locally

- The restored inventory matches the digest established by the verified macOS
  restoration. Every consumed package file is size/hash checked through its
  inventory mapping. Historical `/root` and `/workspace` strings are not opened
  as local paths.
- Ten original source CSV hashes and the original entity-partition manifest hash
  match their historical metadata. Exact original statements, original labels,
  dataset IDs, zero-based source indices, entity IDs, and train/validation
  assignments agree for every cache row. Coverage is checked against the complete
  original train/validation source-row sets; no admission filter is applied.
- The extraction-bound allowed-row export has SHA-256
  `4eeb70add42e410775dd6389297a046f91506c90b1a7ffbd40a55a75bd0599d7`.
  The sidecars reconstruct its ordered partition and combined record digests.
  Qwen and Llama have exactly the same rows in the same order. Source-row IDs are
  globally distinct and original train/validation entities are disjoint.
- Both validation caches contain **six duplicate-statement groups / 12 rows**.
  These remain distinct by `(dataset, source_row_index)`; train has no duplicate
  statement groups. Text alone is never a join key. A preliminary correspondence
  digest binds cache positions to original source IDs/hashes, without exporting
  a selection map or using person keys.
- Qwen completion receipts, metadata identities, the frozen atomic input spec,
  partition digests and recorded smoke statement hashes agree. Llama sidecar,
  frozen tensor record, manifest identity, finalization receipt, ordered source
  IDs/statements, and every contiguous extraction-chunk identity agree. Historical
  smoke receipts are checked as records; smoke inference is not repeated.
- Recorded extraction contracts agree with the frozen specifications: exact raw
  statement, special tokens, no chat template or truncation, unpadded batch one,
  BF16 compute, last real token, explicit position IDs, saved HF hidden states
  1 through 28/32 (embedding excluded), and final post-RMSNorm state. Both caches
  require C-order little-endian float16 arrays. Tensor headers remain unverified
  until SSH succeeds.

Qwen is `Qwen/Qwen2.5-7B-Instruct`, revision
`a09a35458c702b33eeacc393d103063234e8bc28`, 28 layers, width 3584. Llama is
`meta-llama/Llama-3.1-8B-Instruct`, revision
`0e9e39f249a16976918f6564b8830bc894c89659`, 32 layers, width 4096. Their respective
tokenizer revisions agree with the model revisions. Recorded tokenizer class,
backend/config digests and available special-token identities are retained in
the manifest. Standalone historical tokenizer/model config files are absent
from the restored package; this is not live tokenizer or model replay.

## Ordering hazard and unresolved provenance

The two packaged method-suite `allowed_atomic_rows.csv` exports have SHA-256
`76c2e5f3da1958205dd7a14e131113e3bd6d9a0e4f01966dd0e0f07d2aef85ed`.
Their 4,184 records match the extraction-bound original **by source-row ID**, but
**4,152 positions differ**. Do not treat their row offsets as activation indices.
Use the extraction-bound sidecar order and explicit source-row joins.

The `pinned_atomic_probes.py` conflict remains unresolved:

- Recorded working-file digest:
  `6d1991fc639e2d664420122990cbb68653519184745cb24a812364af3d1e0466`
- Recorded producer/Git digest:
  `0eef476667172ae8455bccdefc5d1ef7dde3a4549f88aaa7d4e80ac36f052918`

The Llama extraction manifest also records earlier hashes for
`src/clean_compounds.py` and `src/llama_replication_probes.py` than this checkout.
Both recorded/current digests are preserved in the manifest. These are execution
provenance limitations, separate from row bindings and physical tensor identity.
Even matching tensor bytes would not establish which historical implementation
executed. No conflict is silently resolved.

## Read-only remote verification

Run from the preflight worktree, using a new external output directory:

```sh
/tmp/clean-extraction-venv/bin/python -B scripts/47_preflight_atomic_caches.py \
  --payload-root /Users/apple/truth-probing-backups/20261001T164224Z/restore_t1_staging_20261001T164224Z_kgkt5sy7 \
  --output-root /tmp/t2-cache-preflight-new-run \
  --ssh
```

Omit `--ssh` for local-only checks; physical verification remains pending. The
SSH child process uses `root@69.30.85.22`, port `22114`, and the supplied
`~/.ssh/id_ed25519` identity. It is noninteractive, verifies the existing host key,
does not print key contents, and does not install code remotely. The Codex session
stays on macOS. Authentication or connection failure is saved as **pending**.

The remote stdlib-only program permits exactly four train/validation tensor paths
under `/workspace/truth-probing-artifacts`. It verifies live companion sidecars,
manifests and receipts, streams SHA-256 over each tensor in place, parses only its
NumPy header, and checks byte length, shape, dtype, storage order, and file stability
during reading. It rechecks companions after the tensor read. It never loads
activation values, unpickles, copies tensors, accesses final-test activations,
or writes remote files. NumPy need not be installed on the remote host.

## Conditions before corrected fits

1. Restore authorized SSH access and obtain matching physical tensor and live
   companion identities. Authentication failure alone does not require fresh
   extraction; it leaves reuse unverified.
2. Obtain the subsequently approved audit version. Its admitted source rows,
   original partitions, corrected targets, person keys and leakage decisions
   control the future selection. Do not promote this preliminary correspondence
   into a final map or infer audit approval from cache compatibility.
3. Explicitly adopt the checked representation contract for the new fit. Join
   admitted rows by dataset/index/source SHA/statement SHA to sidecar positions,
   preserve duplicate-text identities, and check coverage without silent drops.
4. Unchanged text can reuse its verified activation after approved label or
   person-key changes, subject to the approved partition/leakage policy. Labels
   are not an input to the recorded extraction; historical labels stay untouched.

Fresh extraction is required for required rows absent from a verifiable cache,
changed text, a changed target model/tokenizer/preprocessing/readout/layer/dtype
contract, or unresolved tensor/sidecar corruption after attempts to retrieve the
byte-verified original. A future policy demanding stronger representation evidence
may also require fresh extraction. This preflight neither reserves A/P rows nor
authorizes corrected fitting while the factual audit remains pending.

## Focused tests

```sh
/tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_t2_cache_preflight.py
```

All 11 tests passed. Synthetic tests cover duplicate-text disambiguation, exact
statement/label/entity/partition bindings, missing coverage, source IDs, NPY
header-only parsing, dtype/order/length checks, hashes rather than shape-only
acceptance, the remote train/validation allowlist, changed sidecars, links, and
authentication failure remaining pending. No real tensor or live SSH is used by
the tests.
