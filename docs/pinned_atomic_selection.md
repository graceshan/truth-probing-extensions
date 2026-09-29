# Pinned Qwen2.5 LR selection

The pinned entry point reads only
`acts/clean_protocol/atomic/qwen25_a09a354_bs1_bf16_v1/` as data. There is no
model/config/input-path override or test-evaluation mode. It never constructs
the historical `AtomicData` loader or follows paths in extraction provenance.

```bash
python scripts/34_select_pinned_atomic_probe.py --check-only
python scripts/34_select_pinned_atomic_probe.py
```

Check-only validates the completion receipt, manifest, exact output-file allowlist,
every output hash/size, array headers/shapes/dtypes, metadata schema/counts/splits,
unique original row identities, disjoint entities, and binary labels with both
classes in each split. It verifies the fixed model/tokenizer revision, approved
input digest, extraction runtime and complete unpadded BF16/SDPA convention, and
passing unpadded smoke evidence. It also checks partition-record hashes and
reconstructs the original export digest from the two metadata files in original
topic/form/source-row order. No original row export or mixed source is reopened.

Hash verification streams the full new activation files as bytes. Check-only
does not materialize numerical activation matrices, fit, score, or write results.
Source-code hashing is separate from data access.

Normal mode performs the same checks before any fit, then directly calls the
unchanged `src.clean_atomic_probes.fit_and_save`. The adapter implements its
train/validation partition interface and converts only the requested layer to
float64. The shared constants, diagnostics, convergence retry, selection rule,
and saving logic are not copied or modified. The zero test-access counter exists
only to satisfy the shared saver assertion; no test partition can be obtained.

All 28 layers and C values `[0.001, 0.01, 0.1, 1.0, 10.0]` are fitted on TRAIN
only: 140 grid entries, with the existing 2,000-to-10,000-iteration convergence
retry policy. Selection uses pooled VALIDATION AUROC, then smaller C, then lower
layer. The train-fitted winning coefficients are saved without a train+validation
refit. Neither historical layer 17 nor C=0.01 is presumed to win.

Results are exclusively reserved under
`results/clean_protocol/atomic_probes_pinned_v1/qwen25_7b/`. Existing directories
are refused before fitting. Outputs retain the shared format:
`selection.json`, `selected_probe.npz`, `validation_metrics.csv`, `split_counts.csv`.
The selection's `structural.repaired_cache_files` records SHA-256 and size for
the extraction manifest, completion receipt, and both activation/metadata pairs;
`structural.adapter_code_sha256` binds the adapter, pinned CLI, and imported
extraction-contract module. Existing shared
selection provenance records library versions, configuration, and selection code.

Synthetic adapter and shared-selection tests (no real cache selection):

```bash
python -m unittest discover -s tests -p 'test_*atomic_probes.py'
```
