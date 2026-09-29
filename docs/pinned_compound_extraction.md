# Canonical pinned Qwen2.5 DEVELOPMENT / VALIDATION extraction

This is a separate representation-specific path. It does not import an LR or
method loader, read a selected probe, fit/score a probe, compute compound task
metrics, or consult historical Qwen2.5 activations/compatibility pins. Script 28,
the historical compatibility gates, and the Qwen3 path are unchanged.

Commands from the repository root (the implementation task did not run these
against real weights or caches):

```bash
python scripts/35_extract_pinned_compound_activations.py --mode plan
python scripts/35_extract_pinned_compound_activations.py --mode smoke
python scripts/35_extract_pinned_compound_activations.py --mode extract
```

Resume an interrupted extraction with the same configuration:

```bash
python scripts/35_extract_pinned_compound_activations.py --mode extract --resume
```

The fixed output is
`acts/clean_protocol/entity_disjoint/development_validation_v1/qwen2_5_7b_a09a354_bs1_bf16_v1/`.
Only a fresh representation-specific `..._vN` directory may be supplied through
`--output`. There is no input/model/revision/dtype/batch-size override. An existing
directory requires explicit resume; historical output names and symlinks fail.

## Representation and repaired cache binding

The data inputs are exclusively the six files in
`acts/clean_protocol/atomic/qwen25_a09a354_bs1_bf16_v1/` and
`data/clean_protocol/compounds/entity_disjoint/development_validation_v1/development_validation_compounds.csv`.
The compound CSV must match the frozen SHA-256
`96e96515e780086065694155e3061bdfe6f4bb80af062a63c2deecd6bfad6f94`,
contain exactly 8,384 rows, and contain only development/validation splits.
No atomic original source, entity partition manifest, historical array, allowed-row
source export, test artifact, or downstream method artifact is opened.

The existing strict repaired-cache validator has been factored into
`src/repaired_atomic_cache.py`. Both the LR adapter and compound orchestrator reuse
it. It verifies completion, manifest hash, all four output hashes/sizes, array
shapes/dtypes, metadata schema/alignment/identities/splits, and the approved atomic
input digest. It verifies the pinned revisions, extraction configuration, and
recorded successful repaired smoke. Full-file hashes stream only the permitted
train/validation arrays; replay materializes only its ten selected rows.

The representation fingerprint is SHA-256 of sorted-key, compact, UTF-8 JSON
(`ensure_ascii=False`) containing this explicit allowlist:

- Schema version, model identifier, model revision, tokenizer revision.
- BF16 compute, SDPA, batch size one, no padding.
- Exact raw-text input, special tokens enabled, no chat template, no truncation.
- Explicit semantic position IDs, `use_cache=False`, hidden-state output enabled.
- Last-real-token readout and float16 storage.
- 28 layers, width 3,584, and the complete existing `layer_convention(28)` mapping:
  HF entries 1–28, embedding excluded, final saved layer post-final-RMSNorm,
  last-token indexing algorithm, and semantic-position algorithm.

Torch/Transformers versions are enforced separately (2.11.0 and 5.12.1). The
fingerprint excludes runtime versions, hardware, timestamps, paths, Git metadata,
and all method/probe selections. The repaired descriptor and current descriptor
must produce the same fingerprint. Repaired manifests created before this feature
need no rewrite: their descriptor is derived from verified extraction fields.
If a repaired manifest contains a fingerprint, it must agree with that derivation.

The compound manifest records the descriptor/fingerprint and a separate repaired
cache identity: SHA-256 of the six fixed relative filenames' verified hash/size
records. It separately records code hashes, runtime/hardware settings, timestamps,
model/config/tokenizer identities, benchmark hashes, and output location. Fresh
model/tokenizer revisions must resolve to
`a09a35458c702b33eeacc393d103063234e8bc28`; model config-file, tokenizer config-file,
and tokenizer-backend hashes must also match the repaired producer.

## Exact gates

Plan verifies static inputs and representation binding, prints the expected
`[8384, 28, 3584]` shape and exact replay identities, loads no model, and writes
nothing. Smoke and extract load one pinned Qwen2.5 model in BF16/SDPA and require:

1. **Repaired representation binding.** Validate the repaired completion/manifest,
   all cache hashes, canonical convention, matching fingerprint, and fresh resolved
   model/tokenizer identities.
2. **Exact repaired atomic replay.** In topic order `cities`, `sp_en_trans`,
   `inventors`, `element_symb`, `animal_class`, select one TRAIN affirmative and one
   VALIDATION negated row. Within each stratum, select the minimum SHA-256 of UTF-8
   compact JSON `[dataset, original_row_index, exact_statement]`. Record the
   corresponding zero-based row in the repaired split array, original row index,
   entity, split, form, and exact statement. The ordered sample hash uses the same
   sorted-key encoding as the representation fingerprint. Each row is freshly
   forwarded through the **same readout function used by bulk compound extraction**.
   Compare all 28 layers to its exact repaired cached row after float16 conversion.
   Require exact saved-byte equality. Report global and per-row/per-layer max/mean
   absolute differences and byte equality; no tolerance, backend, or dtype fallback.
3. **Unpadded compound smoke.** Select the four smallest SHA-256 values of compact
   JSON `[example_id, exact_statement]` from the approved compound benchmark. For
   each, run production readout, production repeat, and the repaired extractor's
   independent direct HF forward. The direct reference computes full logits and
   directly indexes HF `[1:]` at the final unpadded token. Require finite values,
   exact compute equality and exact saved-float16 byte equality on every saved
   layer. All forwards use one sequence, no padding, explicit semantic positions,
   raw text/special tokens, no truncation/chat template, and `use_cache=False`.

Thus smoke has 10 atomic forwards plus 12 compound forwards, all batch size one.
Unique JSON diagnostics under `results/clean_protocol/pinned_compound_extraction/`
record successful or failed replay/smoke gates. Any failed gate blocks extraction.
No padding or historical comparison participates. Smoke creates no compound array.

## Durable extraction and resume

Extract reruns all three gates in the current model process, including on resume;
a saved passing report alone grants no permission to skip a gate. Only afterwards
does a thin pinned loop call the unchanged `ExtractionWriter`. It writes one
compound row at a time, with all layers, finite float16 values, and the byte-exact
full benchmark metadata. It preserves the existing lock, fsync, chunk hashes,
contiguous progress journal, final file hash, overwrite refusal and resume checks.

Outputs are `metadata.csv`, `extraction_manifest.json`, `progress.json`, and
`activations.npy` (or `activations.partial.npy` while incomplete), plus the lock.
The representation fingerprint is distinct from the stricter resume identity.
Resume also binds this particular repaired cache, benchmark, sample identities,
runtime/hardware, resolved model metadata, and actual implementation hashes.
Fresh diagnostic timestamps/paths and repeated gate measurements do not break
resume; changed data/configuration/code/runtime does. Committed rows are verified
before any writable array is opened and are not re-extracted.

All layers remain available to LR, difference of means, covariance-MM, t_G, TTPD,
layer sweeps and later representation analysis. No LR layer, C, AUROC, coefficient
archive, or other method artifact is needed. Later scoring code must bind each
method's training representation to this fingerprint; this extractor does not
implement or bypass that future scoring check.

Synthetic regression commands:

```bash
python -m pytest -q tests/test_pinned_compound_extraction.py tests/test_clean_atomic_extraction.py tests/test_clean_extraction.py tests/test_atomic_activation_compatibility.py
python -m unittest discover -s tests -p 'test_*atomic_probes.py'
```
