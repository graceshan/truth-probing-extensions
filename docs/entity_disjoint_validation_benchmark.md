# Clean entity-disjoint development validation benchmark

This is DEVELOPMENT / VALIDATION, not a final-test estimate. The command uses only
compound-usable validation entities from the shared manifest and the reviewed v5
negative registry. No model extraction or scoring is performed.

Run from the repository root:

```bash
python3 scripts/27_generate_validation_compound_benchmark.py --config config/clean_protocol/entity_disjoint_development_validation_v1.json
```

Pair seed 0 is fixed in configuration before generation. Algorithm
`validation-degree4-ring-v1` sorts entities by SHA-256 of compact UTF-8 JSON
`["validation-degree4-ring-v1", seed, topic, "validation", entity_id]`, breaking
hash ties by entity ID. Each entity connects to circular offsets 1 and 2. For
n >= 5, this constructs exactly 2n distinct unordered edges and degree 4 at every
entity. This is a seeded circulant graph, not a uniform sample of all regular
graphs. This version rejects other requested counts. Pair IDs retain `pair-v1`.

The exact pair counts are cities 298, sp_en_trans 68, inventors 90, element_symb 36,
and animal_class 32: 524 pairs and 8,384 binary rows. Every pair has four canonical
truth assignments, AND/OR, and AB/BA orders. Canonical truth and surface truth are
stored separately. The existing `clean-binary-v1` templates and stable IDs are reused.

Each false fact is checked against the lowest-ranked accepted candidate in the
pinned reviewed registry, including single-country inventor semantics. Missing
accepted negatives abort generation instead of dropping entities. Source rows and
registry judgments are never modified. Existing loaders hash mixed-source files
for integrity and filter by manifest membership before interpreting test facts;
no test candidate queues are opened, and no test examples are constructed.

Outputs live in `data/clean_protocol/compounds/entity_disjoint/development_validation_v1/`:
the labeled example table, pair table, per-entity degrees, selected-negative IDs
and statuses, and provenance metadata containing source/config/code hashes,
pair IDs, degree summaries, schema, and integrity checks. The command reproduces
the output bytes before saving, then verifies the saved bytes. Existing outputs
cannot be overwritten with different content.

The example schema extends the existing clean binary schema with `protocol`,
`evaluation_phase`, and `benchmark_label`. This development benchmark is entity
disjoint from train-entity fitting, but validation entities were used for atomic
layer/C selection; it must not be described as untouched final evaluation.
