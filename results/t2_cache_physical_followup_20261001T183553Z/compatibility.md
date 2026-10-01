# T2 atomic cache compatibility preflight

Preflight only; original source-row correspondence, independent of pending factual correction decisions

| Cache | Source-row/representation binding | Physical tensor | Rows × layers × width | Bytes |
|---|---|---|---|---|
| qwen/train | verified | verified | 3144 × 28 × 3584 | 631013504 |
| qwen/validation | verified | verified | 1040 × 28 × 3584 | 208732288 |
| llama/train | verified | verified | 3144 × 32 × 4096 | 824180864 |
| llama/validation | verified | verified | 1040 × 32 × 4096 | 272629888 |

Sizes/shapes in the table are recorded expectations until physical verification succeeds. All expect little-endian C-order float16.

Remote status: verified.

Local checks bind every cache row to exact original text, label, source-file hash, zero-based source index, and original partition. Qwen and Llama row orders match.
Duplicate text remains keyed by distinct source-row IDs. No correction decisions or person keys were loaded.

## Missing evidence and conflicts

- No missing physical tensor identities.
- Standalone historical tokenizer/model config bytes are not in the restored package; embedded identity records agree but live tokenizer replay was not performed.
- The final approved factual audit, admitted rows, person keys, corrected split/grouping policy and corrected-fit target representation are not yet supplied.
- `pinned_atomic_probes.py`: working-file `6d1991fc639e2d664420122990cbb68653519184745cb24a812364af3d1e0466` versus recorded producer/Git `0eef476667172ae8455bccdefc5d1ef7dde3a4549f88aaa7d4e80ac36f052918`; unresolved. Matching tensors would not resolve execution provenance.
- The two packaged method-suite allowed-row exports contain the same rows but differ in ordering from the extraction-bound original export; never use their row positions as tensor indices.
- Other recorded extraction source-hash differences are preserved in the JSON report, not silently accepted as execution proof.

## Reuse conditions

- Physical bytes, NPY header and live sidecar/manifest hashes must match before using a tensor.
- Use original dataset + zero-based source row index + source-file hash + exact statement hash; never text alone or alternate export order.
- Final admitted rows and person keys must come from the subsequently approved audit version; this preflight makes no corrected selection or A/P reservation.
- The intended fit must explicitly adopt the verified historical representation, tokenizer, token position, layers and dtype.
- Label or person-key corrections alone do not change activations of unchanged exact text; apply approved targets and leakage exclusions later without altering historical labels.

## When fresh extraction is required

- An approved required source row/exact statement is absent, changed or not uniquely bound to a cache row.
- Model/revision, tokenizer/preprocessing, readout token, saved layer convention, numerical extraction format or required dtype differ from the accepted target contract.
- Tensor/sidecar integrity conflicts or missing bytes cannot be resolved by retrieving a byte-verified original cache.
- The approved future policy requires representation evidence beyond what these historical records establish.

No physical-verification success or final corrected-fit authorization is inferred from matching counts or filenames.
