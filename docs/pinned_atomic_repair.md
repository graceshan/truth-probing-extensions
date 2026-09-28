# Pinned Qwen2.5 atomic repair

`scripts/33_extract_clean_atomic_activations.py` is an isolated repair entry point.
Its only dataset input is
`results/clean_protocol/atomic_method_suite_v1/qwen25_7b/allowed_atomic_rows.csv`.
It verifies the approved ordered-record digest
`1faf186cb28b48311e85fd41921316fad9d06b3d28c9756aa5e8b381a950f1d6`,
3,144 train rows and 1,040 validation rows, unique dataset/original-row identities,
and disjoint train/validation entities. Original statement text and all eight
metadata columns are preserved. Any other split fails. There is no input-path
override, mixed-source scan, original entity-manifest read, historical cache read,
test artifact access, or call into an existing atomic data/probe loader.

Model and tokenizer are pinned to Qwen/Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`. Smoke/extract enforce Torch 2.11.0
(CUDA build suffix allowed), Transformers 5.12.1, CUDA BF16, and SDPA. Every
forward uses exactly one unpadded raw statement, special tokens enabled,
no chat template, no truncation, explicit semantic position IDs, and
`use_cache=False`. Production readouts use `logits_to_keep=1`. Hidden states
are saved at the last real token as HF entries 1 through 28, excluding embeddings;
the last entry is after final RMSNorm. Storage is float16. No layer or C is selected
or inherited from the historical probe.

## Modes

From the repository root:

```bash
python scripts/33_extract_clean_atomic_activations.py --mode plan
python scripts/33_extract_clean_atomic_activations.py --mode smoke
```

Plan uses only the standard library, verifies the export and destination, and
prints the fixed contract and shapes. It does not load weights or write files.

Smoke loads full pretrained weights, selecting ten allowed rows: for each topic,
one train affirmative and one validation negated row, minimizing the SHA-256 of
`[dataset,row_index,statement]`. This selection depends on no activation or score.
Each row has three separate unpadded forwards: production, production repeat,
and an independent direct forward with full logits and direct HF `[1:]`/last-token
indexing. All compute readouts must be exactly equal, and their finite float16
bytes must match. An in-memory NPY roundtrip must preserve those bytes. There is
no tolerance fitting, padding comparison, historical reference, or cross-runtime
determinism claim. The sample need not be numerically equivalent to old caches.

Unique reports (including failed comparison/forward diagnostics) are written to
`results/clean_protocol/atomic_repair_smoke/`. Smoke creates no activation arrays
or output cache directory. Input/version/revision failures stop before the smoke.

Bulk extraction is a separate explicit `--mode extract` invocation, not part of
implementation validation. It loads once and reruns the same smoke before creating
the output directory. A prior report cannot bypass this gate.

## Outputs and provenance

The default new directory is
`acts/clean_protocol/atomic/qwen25_a09a354_bs1_bf16_v1/`:

```text
train/activations.npy       # [3144, 28, 3584], float16
train/metadata.csv
validation/activations.npy  # [1040, 28, 3584], float16
validation/metadata.csv
extraction_manifest.json
completion.json
```

Metadata order is activation row order; `row_index` retains the original source
index, not the new cache index. Only those two partitions are created. Manifests
record the approved input digest and byte hash, partition record hashes, exact
model/tokenizer revisions and configuration/backend hashes, model/output-capture
implementation hashes, repository source hashes, runtime/numerical settings, and
the fresh smoke report. Git HEAD is contextual only; source hashes describe the
actual implementation. `completion.json` binds the manifest and every output
array/sidecar with full file hashes and sizes. Whole-array hashing is permitted
because these new arrays contain only train/validation rows.

An existing output directory, even empty or incomplete, is refused. Symlinks and
paths outside the versioned atomic directory naming scheme are rejected. Creation
is exclusive, with partial NPY files and a completion receipt written last. A
consumer must require and verify the completion receipt. No resume is implemented;
after an interrupted run, retain it and select a fresh version, for example
`--output acts/clean_protocol/atomic/qwen25_a09a354_bs1_bf16_v2`.

The historical compound compatibility gate and LR selection implementation are
unchanged. Repaired caches do not authorize compound extraction or reuse of old
probe coefficients/layer/C.

Synthetic verification (no real datasets or pretrained weights):

```bash
python -m pytest -q tests/test_clean_atomic_extraction.py
```
