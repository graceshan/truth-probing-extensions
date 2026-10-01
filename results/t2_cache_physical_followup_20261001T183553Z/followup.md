# T2 physical cache follow-up — 2026-10-01

**Physical identity verification passed for all four allowlisted train/validation caches.** Source-audit and representation decisions remain pending.

Checker completed at `2026-10-01T18:36:28.140426+00:00`. Run from `/Users/apple/projects/truth-probing-t1-score-rebuild` on branch `t2-cache-preflight-20261001`, HEAD and reviewed ancestor `ff10b8e4c5dd185d7dda3579f6d0f98f4af76ae5`. The tracked tree was clean; `git fetch origin` and `git pull --ff-only origin t2-cache-preflight-20261001` succeeded with no divergence or update.

Local host: `MacBook-Pro-7.lan`, Darwin/macOS 12.6 (21G115). The existing Python 3.11.6 environment was available. A one-shot SSH command authenticated and returned `ef7f7534c328` before any artifact checks. The checker then completed a separate one-shot SSH command and returned to macOS. Strict host-key verification, batch mode, and the specified identity remained enabled.

The sandbox could not access the local agent socket (`Operation not permitted`). Outside the sandbox, `ssh-add -l` showed the existing loaded ED25519 key; its fingerprint matched `id_ed25519.pub`. No credentials or SSH configuration changed. Authentication is no longer blocked.

## Actual measurements

| Cache | Result | Actual shape | Actual bytes | Actual SHA-256 |
|---|---|---|---:|---|
| qwen/train | verified | 3144 × 28 × 3584 | 631,013,504 | `1fa45a5e821bf3030260766658944b2ea38d7afe68b7a41486712652486c1b11` |
| qwen/validation | verified | 1040 × 28 × 3584 | 208,732,288 | `574a4bff7706383d615b304bb62512a402f422a09502e118173085030bd354ab` |
| llama/train | verified | 3144 × 32 × 4096 | 824,180,864 | `af7a0b6295bdbe63094ac7b5f627dcfd1f17e11bafbab274f4fd4da9ce50f22a` |
| llama/validation | verified | 1040 × 32 × 4096 | 272,629,888 | `7cbfd58bc53af5806926f4ced8725237cdc74cafd653d5e20e30c9dc46a6bb4f` |

All measured sizes and SHA-256 hashes equal the original recorded expectations. Every NPY header declares little-endian float16 (`<f2`), C order, and the expected shape; computed payload length equals actual file length. All reads were stable. Qwen uses NPY 2.0 and Llama NPY 1.0, each with a 128-byte header.

All 12 companion/cache checks passed (10 unique files): Qwen train/validation `metadata.csv` plus their shared `completion.json` and `extraction_manifest.json`; Llama train/validation each has `metadata.csv`, `extraction_manifest.json`, and `progress.json`. Their live sizes and SHA-256 hashes matched recorded expectations, reads were stable, and the verifier rechecked companions after each tensor hash. Full paths, hashes, and actual measurements are retained in `compatibility.json`.

## Validation and preserved evidence

Independent receipt checks confirmed the exact four-cache set, expected row counts, sizes, hashes, shapes, dtypes, order, stability, companion identities, and Markdown/JSON status consistency. Local row bindings, input receipts, code hashes, provenance conflicts, and reuse conditions match the original report. The original pending report files match their reviewed Git versions byte for byte. `session_validation.json` records this validation and report digests. No checker code changed, so focused synthetic tests were not rerun.

New external receipts: `/Users/apple/projects/t2-cache-physical-followup-20261001T183553Z`. The checked-in `compatibility.json` is byte-identical to the generated receipt. The checked-in `compatibility.md` removes one trailing space from the generated remote-status line; its contents otherwise match. Original reports in `results/t2_cache_preflight_20261001/` remain unchanged.

## Remaining limitations and decisions

- `pinned_atomic_probes.py` remains unresolved: working-file SHA-256 `6d1991fc639e2d664420122990cbb68653519184745cb24a812364af3d1e0466` versus recorded producer/Git SHA-256 `0eef476667172ae8455bccdefc5d1ef7dde3a4549f88aaa7d4e80ac36f052918`. Tensor equality does not establish which implementation executed.
- Recorded Llama extraction hashes for `src/clean_compounds.py` and `src/llama_replication_probes.py` differ from this checkout; both identities remain in the JSON report.
- Standalone historical tokenizer/model config bytes are absent from the restored package. Embedded identity records agree, but no live tokenizer/model replay was performed.
- The two alternate allowed-row exports differ from extraction order at 4,152 positions. Join by dataset, original source index, source-file hash, and exact statement hash. Preserve the six duplicate-statement groups / 12 rows in each validation cache as distinct source identities.
- Approved source-audit decisions, final admitted rows, person keys, corrected targets and split/leakage policy remain pending. The target representation contract has not been adopted. Physical verification alone authorizes none of these decisions.

No source corrections, final admitted-row map, A/P reservation, fitting, inference, or extraction were performed. Only headers and streaming hashes were read remotely; activation values were not inspected, tensors were not copied, and final-test caches were not accessed. Mac 1’s worktree and historical artifacts were not modified.
