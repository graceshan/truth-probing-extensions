# Clean compound generation

`src/clean_compounds.py` consumes the shared entity manifest; it never assigns
splits. Only manifest-authorized, compound-usable entities in the requested
topic and split can form a pair. Every pair is checked again at generation.
Duplicate entity identities, including identities listed in multiple splits,
invalid entity IDs, manifest hash mismatches, and source hash mismatches fail.
The exploratory generator and its outputs are separate and unchanged.

No real benchmark generation is part of the architecture smoke test. Run only:

```bash
python3 -B -m unittest discover -s tests -p 'test_clean_compounds.py'
python3 -B tests/smoke_clean_compounds.py
```

Both commands use invented sources and explicit synthetic manifest membership.
The smoke runner invokes the CLI on one synthetic train pair and removes its
temporary outputs. Synthetic validation/test partitions are used in unit tests
to verify enforcement; real validation/test compounds must remain ungenerated
until the experiment suite is locked and their generation is authorized.

## Configuration and output

`scripts/18_generate_clean_compounds.py` requires `--config PATH`. It has no
default dataset, split, or pair count. Paths are relative to that configuration
file. This example describes the synthetic fixture configuration:

```json
{
  "schema_version": 1,
  "entity_manifest": "manifest.csv",
  "entity_manifest_metadata": "manifest_metadata.json",
  "source_dir": "sources",
  "split": "train",
  "pair_sampling_seed": 0,
  "generation_seed": 0,
  "pairs_per_topic": {"cities": 1},
  "template_version": "clean-binary-v1",
  "output_dir": "clean_protocol/smoke"
}
```

Within the repository, outputs must be under `data/clean_protocol/`. Existing
differing files cannot be overwritten; identical reruns verify without writing.
`compounds.csv` is sorted by example ID, using UTF-8, LF line endings, and
`True`/`False` for Boolean fields. The Python API uses actual Boolean values.
Consumers must parse Boolean columns explicitly, not apply `bool()` to strings.
`metadata.json` records manifest hash/version and metadata hash, requested split,
both seeds, requested pair counts, template version, source hashes, row count,
output hash, and generation assertions. The CLI verifies byte-identical rebuilds
before saving and prints aggregate metadata only.

## Construction and stable IDs

All IDs use full SHA-256 of compact UTF-8 JSON (`ensure_ascii=False`, separators
`,` and `:`), prefixed with `fact_`, `pair_`, or `example_` respectively:

- Fact: `["fact-v1", topic, entity_id, exact_atomic_statement, truth]`.
- Unordered pair: `["pair-v1", topic, sorted_entity_ids]`.
- Example: `["example-v1", pair_id, split, canonical_fact_a_id,
  canonical_fact_b_id, operator, ordering, template_id, rendered_statement]`.

Canonical A/B is ascending entity-ID order, independent of input order. Pair IDs
are invariant across truth patterns, operators, and AB/BA surface ordering.
Fact IDs identify the true or false assertion, not just the underlying entity.
Example IDs distinguish different surface forms. IDs do not contain seeds or
row indices. Identical content retains its IDs when resampled; new false-object
content gets new fact/example IDs. Template changes require a version change.

Pair sampling ranks all unordered unique candidates by SHA-256 of
`["pair-sampling-v1", pair_sampling_seed, topic, split, sorted_entity_ids]`,
breaking hash ties by the pair's IDs. Select the requested smallest ranks.
Entity reuse within a split is allowed. Capacity `n*(n-1)/2` is checked before
candidate enumeration; there is no rejection loop. Enumeration costs O(n²)
candidate visits with bounded selection memory. Counts are configurable per topic.

True facts preserve exact affirmative source statements. False facts replace
the correct object with a different object from usable entities in the same
topic/split, selected by SHA-256 rank of `["wrong-object-v1", generation_seed,
topic, entity_id, object]`. A missing/ambiguous true mapping or lack of a distinct
wrong object raises an error. Wrong-object labels retain the exploratory source
assumption that the supplied unique correct object defines truth; they are not
an independent factual verification. No facts are constructed for other splits.

Standard R1 emits four Boolean truth patterns × AND/OR × AB/BA = 16 rows/pair.
`boolean_truth(operator, values)` centralizes truth evaluation, supports nonempty
n-ary Boolean sequences, and defines XOR as odd parity (equivalent to binary
exclusive OR for two inputs). Binary XOR rendering is available separately;
standard R1 includes only AND and OR. Empty inputs and non-Boolean values fail.

## Output schema

| Columns | Meaning |
|---|---|
| `example_id`, `statement`, `topic`, `split` | Stable example identity, surface text, topic, manifest split |
| `entity_a_id`, `entity_b_id` | Canonical A/B entity IDs |
| `fact_a_id`, `fact_b_id`, `fact_a_statement`, `fact_b_statement` | Canonical atomic assertions and exact text |
| `pair_id` | Unordered entity-pair identity |
| `canonical_truth_a`, `canonical_truth_b` | Canonical Boolean labels, unchanged by surface ordering |
| `surface_first_entity_id`, `surface_second_entity_id` | Actual surface entity order |
| `surface_first_truth`, `surface_second_truth` | Actual surface truth order |
| `operator`, `ordering`, `template_id` | AND/OR/XOR, AB/BA, versioned topic/operator template |
| `compound_label`, `generation_seed` | Centralized Boolean result and false-fact generation seed |

For canonical `(True, False)` with `BA`, canonical labels stay `(True, False)`
and surface labels become `(False, True)`. Downstream code should use these
Boolean fields and the shared helper, never infer truth from cell-name strings.
