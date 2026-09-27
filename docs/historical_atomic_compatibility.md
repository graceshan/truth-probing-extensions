# Historical atomic activation compatibility gate

`scripts/28_extract_clean_activations.py --mode smoke` now runs both the
full-weight padding smoke and a historical atomic representation smoke. It
never extracts the compound benchmark. No probes or labels are loaded.

The historical sample contains ten rows: one train affirmative and one
validation negated statement from each of cities, Spanish translations,
inventors, element symbols and animal classes. Selection takes the smallest
SHA-256 of `[dataset, row_index, exact_statement]` within each designated split.
It is deterministic, independent of labels/scores/activation values, and uses
the same rows for both models when the sidecars align.

Only statement columns are loaded from primary atomic source CSVs and
activation sidecars. Their exact statement order must agree. Entity identities
and split assignments come from the entity partition manifest; labels are not
requested or retained. The activation files are opened read-only with a memmap;
only the ten selected train/validation rows are copied. No whole activation
file hash is computed, because that would read test activation values. The
report hashes each selected historical activation row instead. It records all
sample identities, statement text, source/sidecar paths, row indices, splits,
ordered statement hashes, and split-manifest hash.

Each selected statement is re-extracted singly with no padding, raw text,
`add_special_tokens=True`, no chat template, and the unchanged HF `[1:]` saved
layer convention. All saved layers are compared. Diagnostics report global,
per-layer and per-statement max/mean absolute differences before and after
converting the current readout to the historical float16 dtype, plus an exact
byte-identity result after conversion.

No historical tolerance is chosen to obtain a pass. The runner also measures
current single-versus-right/left-padding differences on these **same ten
statements**, using the configured batch size (at least two for the diagnostic).
This noise measurement is independent of the cached historical values. The
historical gate requires either exact float16 byte identity or, on **every**
saved layer, historical max **and** mean error no larger than the observed
current padding max/mean envelope. There is no multiplier, additive tolerance,
or automatic relaxation. The same-sample padding smoke must itself pass the
existing padding gate. General synthetic-statement padding results are included
alongside this comparison.

This is deliberately conservative: harmless kernel differences can still
require review. A mismatch is reported as a failure, no passing receipt is
written, and extraction stops. The runner never re-extracts atomic caches,
fits/scores probes, or widens tolerances. Passing ten sampled rows establishes
sampled representation compatibility, not proof of identical historical weights.

`--revision` accepts an immutable 40-character HF commit SHA, not `main`.
The model's resolved commit is read from its loaded config. The tokenizer's
resolved commit is independently checked from HF's cached
`tokenizer_config.json` snapshot path, because this Transformers tokenizer does
not expose `_commit_hash` in its initialization metadata. Its config content
hash and backend hash are recorded too. Both are loaded from their explicit
immutable commits.

Only after both gates pass, smoke mode creates:

```text
results/clean_protocol/extraction_diagnostics/compatibility_pins/qwen2_5_7b.json
results/clean_protocol/extraction_diagnostics/compatibility_pins/qwen3_8b.json
```

Each receipt records model/tokenizer commits, the full diagnostics path and hash,
and the extraction identity hash, which binds model/config/tokenizer, code,
numerics, benchmark and selected historical row provenance. Existing receipts
are never overwritten with different evidence identities. A repeated successful
smoke of the same identity may retain the original receipt. Keep the referenced
diagnostics JSON with the receipt.

Full extraction refuses to load without a valid passing receipt. It reads the
model/tokenizer revision from that receipt, verifies its evidence, loads those
exact snapshots, and reruns both gates before allocating any compound activation
array. A conflicting `--revision` or any changed configuration/sample/code hash
fails. Both resolved commits, requested pins, sample provenance and fresh gate
results appear in `extraction_manifest.json`. The Python extraction entry point
also requires both gates and matching revision/sample identities.

Run these GPU commands from the repository root, with the historical primary
atomic NPY files and CSV sidecars present at their original configured paths:

```bash
python scripts/28_extract_clean_activations.py --model qwen2_5_7b --mode smoke --revision a09a35458c702b33eeacc393d103063234e8bc28 --device cuda --dtype bfloat16 --batch-size 4 --padding-side right --attention sdpa
python scripts/28_extract_clean_activations.py --model qwen3_8b --mode smoke --revision b968826d9c46dd6066d109eabc6255188de91218 --device cuda --dtype bfloat16 --batch-size 4 --padding-side right --attention sdpa
```

These are candidate current snapshot revisions, not claims about the unknown
historical revisions. No passing receipts are supplied by the implementation
task; they can only be produced by successful full-weight smoke runs.

Local validation uses synthetic caches with forbidden rows guarded against
access and label columns excluded. Run only the extraction-specific suites:

```bash
python -m pytest -q tests/test_clean_extraction.py tests/test_atomic_activation_compatibility.py
```

The real sample selection/read-only preflight succeeded for both models. It
selected the same dataset/row indices for each: cities/423, neg_cities/1078,
sp_en_trans/31, neg_sp_en_trans/108, inventors/229, neg_inventors/393,
element_symb/82, neg_element_symb/1, animal_class/106, neg_animal_class/149.
The current environment still has no CUDA/model weights, so full-weight
historical compatibility is **not yet measured or passed**.
