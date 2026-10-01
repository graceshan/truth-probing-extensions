# Provisional candidate-to-cache mapping — 2026-10-01

The integration branch begins at reviewed objective commit
`5429211251ead82cbc78ab650918ca967d92e09e`. Normal merge
`94399c3d0312d611cae19f391fa72f33effc72a9` has exactly that first parent and
reviewed source-overlay commit `b8e27508f20bab896a46f448081f43ed32d919c4` as its
second parent. Later source-review commits are outside this iteration.

The mapping is **provisional and unusable for production**. It proposes cache
correspondence for the pinned correction candidate; it does not adopt that
candidate or a representation. Train means corrected outer training, not P.
No A/P allocation, final selection, fitting, extraction, SSH session, tensor
read/copy, or final-test predictions were performed. Existing objective
mathematics, historical evaluators, cache checkers and historical reports are
unchanged.

## Portable preservation checks

The normal candidate test creates 13 synthetic unrelated files in a temporary
directory, exercises candidate output creation/checking, and verifies those files
are unchanged. It no longer requires mac 1's untracked files. All existing
tracked historical-input validations remain enabled. The historical
`preservation_baseline.json`, its 13 hashes, and the entire candidate overlay
remain byte-identical to the reviewed source commit.

An explicit local evidence check is available separately:

```sh
python -B scripts/check_selection_repair_local_preservation.py --root /path/to/checkout
```

This reports matched, missing and mismatched historical untracked files and exits
nonzero for either failure. It requires an explicit root and is not invoked by
normal tests or mapping. Its behavior is tested with synthetic files; none of mac
1's untracked files is copied, committed or modified.

## Identity and version policy

`config/clean_protocol/selection_repair_mapping_v1.json` pins the exact candidate
receipt, physical compatibility receipt and session receipt by size and SHA-256.
The candidate receipt binds all locked historical inputs and all candidate
outputs. The physical receipt pins the restored inventory and four caches'
companion files. The session receipt must identify the same physical receipt and
record the expected remote host. Hash/size changes, an unverified receipt,
foreign cache paths and mixed companion identities are rejected.

The mapper hashes metadata only. It opens no `.npy`, checkpoint or score package,
imports no model/numerical library, and has no SSH capability. Restored reads are
limited to `inventory.json` and the ten unique allowlisted companion files.
Historical source/registry files are hashed for the candidate lock; source rows
are joined from the original and candidate manifests, without text-only lookup.

Candidate admission uses **`candidate_status` exclusively**. The older `status`
column is preserved as historical information. Expected counts are derived from
the full pinned manifest, reconciled with the candidate receipt and its admitted
subset, never guessed from cache length.

Each row binds through `(dataset, zero-based source index)`, original source-file
SHA-256, canonical original-source-row JSON SHA-256, and exact statement SHA-256.
The original T2A manifest supplies the source-row hash; the sidecar has no such
column. The chain verifies unchanged original columns and source-file identities,
then checks exact sidecar text, label, entity, form, topic and partition. Person
keys and original partition roles are carried unchanged from the candidate.

Tensor indices are zero-based positions in each authoritative `metadata.csv`,
whose bytes match the saved live companion receipt. Recorded ordered source-ID
and statement digests are rechecked. Alternate method-export offsets are never
used. Duplicate statements retain distinct dataset/index identities. Missing,
ambiguous, duplicate-identity, unexpected or mismatched bindings are reported;
an alignment failure produces diagnostics and no partial map files. Stale input
hashes fail before output creation. Existing outputs are never overwritten.
`--check-only` rebuilds metadata correspondence and requires every saved byte to
match, including the map/proposal hashes. Map rows themselves carry a provisional
status and an unallocated role.

## Coverage for the pinned candidate

See [mapping_report.json](../results/t2_protocol_integration_20261001/candidate_maps_v1/mapping_report.json)
and its four sibling `*.provisional.csv` files.

| Cache | Candidate expected | Mapped | Full original sidecar | Missing / ambiguous / duplicate-ID / mismatched |
|---|---:|---:|---:|---|
| Qwen train | 3,048 | 3,048 | 3,144 | 0 / 0 / 0 / 0 |
| Qwen validation (D) | 1,012 | 1,012 | 1,040 | 0 / 0 / 0 / 0 |
| Llama train | 3,048 | 3,048 | 3,144 | 0 / 0 / 0 / 0 |
| Llama validation (D) | 1,012 | 1,012 | 1,040 | 0 / 0 / 0 / 0 |

Each D map contains six duplicate-statement groups / 12 distinct source rows;
train has none. Qwen and Llama agree exactly on source identity, sidecar index,
person key and partition. The report explicitly enumerates duplicate-statement
source identities. No E cache map is produced.

## Representation proposal and remaining decisions

[representation_proposal.json](../results/t2_protocol_integration_20261001/candidate_maps_v1/representation_proposal.json)
records exact model/config/tokenizer evidence, representation fingerprints, and
per-cache sidecar/tensor/companion hashes. Its adoption fields are false.

- Qwen: `Qwen/Qwen2.5-7B-Instruct`, model/tokenizer revision
  `a09a35458c702b33eeacc393d103063234e8bc28`, 28 saved layers, width 3584.
- Llama: `meta-llama/Llama-3.1-8B-Instruct`, model/tokenizer revision
  `0e9e39f249a16976918f6564b8830bc894c89659`, 32 saved layers, width 4096.
- Recorded rendering: exact raw statement, special tokens enabled, no chat
  template or truncation, unpadded batch one, explicit semantic position IDs,
  BF16 compute and last-real-token readout. Tensor axis index j corresponds to
  HF `hidden_states[j+1]`; embeddings are excluded and the final saved layer is
  post-final-RMSNorm. Saved arrays are little-endian float16, C order.
- Verified physical facts come from the saved `2026-10-01T18:36:28.140426+00:00`
  receipt: tensor hashes, sizes, headers, stable reads and live companion hashes
  matched. This task rechecks local metadata against that receipt; it performs
  no fresh remote measurement or execution replay.
- Embedded config and tokenizer identities are recorded evidence. Standalone
  historical tokenizer/model config bytes are absent from the restored package;
  no live tokenizer/model replay establishes them anew.
- `pinned_atomic_probes.py` remains unresolved: working-file digest
  `6d1991fc639e2d664420122990cbb68653519184745cb24a812364af3d1e0466` versus
  producer/Git digest
  `0eef476667172ae8455bccdefc5d1ef7dde3a4549f88aaa7d4e80ac36f052918`.
  Recorded Llama hashes for `src/clean_compounds.py` and
  `src/llama_replication_probes.py` also differ from the checkout. All identities
  remain in the proposal. Matching tensors do not establish which historical
  implementation executed.

Production work requires explicit adoption of this exact correction-overlay
receipt and the representation proposal, including a decision on missing
provenance. A future approved assembly must separately enforce audited facts,
held-out pair removal and any allocation/leakage policy. This mapper has no
production loader or adoption override; `require_production_ready` always
rejects these proposal-only artifacts. It cannot infer approval from coverage.

## Reproduction and focused checks

```sh
/tmp/clean-extraction-venv/bin/python -B scripts/48_map_selection_repair_candidates.py \
  --payload-root /path/to/restored/payload --output-root /tmp/new-candidate-map
/tmp/clean-extraction-venv/bin/python -B scripts/48_map_selection_repair_candidates.py \
  --payload-root /path/to/restored/payload \
  --output-root results/t2_protocol_integration_20261001/candidate_maps_v1 --check-only
/tmp/clean-extraction-venv/bin/python -B scripts/46_audit_selection_repair_sources.py
/tmp/clean-extraction-venv/bin/python -B data/clean_protocol/selection_repair_v1/source_review_v1/validate.py
/tmp/clean-extraction-venv/bin/python -B scripts/47_build_selection_repair_candidate.py --check-only
/tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_selection_repair_candidate_mapping.py \
  tests/test_selection_repair_candidate_overlay.py \
  tests/test_selection_repair_source_audit.py \
  data/clean_protocol/selection_repair_v1/source_review_v1/test_validation.py \
  tests/test_selection_repair_objectives.py tests/test_t2_cache_preflight.py
```

The focused suite covers source-row and statement hashes, candidate-only admission,
reordered authoritative sidecars, duplicate statements versus duplicate IDs,
missing/ambiguous/mismatched bindings, historical identity/partition preservation,
receipt version mixing, invalid physical status/shape/dtype/order/stability,
companion mismatch, prohibited cache paths, output overwrite/staleness, and
production rejection. Existing objective tests use only synthetic arrays; existing
cache-checker tests use only synthetic NPY fixtures, with no live SSH.
