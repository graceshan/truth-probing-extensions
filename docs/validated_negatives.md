# Validated negative infrastructure

This is an opt-in factual-validation layer for the clean generator. It does not
change the default generator, exploratory pipeline, frozen LR baseline, or final
exposure settings. There is no model scoring, web research, or automatic external
validation in this workflow.

## Candidate queues and identity

`config/clean_protocol/validated_negatives.json` pins the existing exhaustive
source audit and uses negative-candidate seed **0**. For each compound-usable
entity in the requested development partition, the candidate pool is the union
of all recorded true objects of usable entities in the same topic and shared
partition. All objects recorded true for the target entity are removed, including
objects with contradictory labels. Multiple true objects are retained when
performing this exclusion. The existing generator's unique-true-object check is
unchanged; a registry does not resolve source contradictions.

Candidates sort by ascending SHA-256 of the UTF-8 encoding of compact JSON:

```text
["negative-candidate-v1", negative_candidate_seed, topic, entity_id, exact_candidate_object]
```

Serialization uses `ensure_ascii=False` and separators `(',', ':')`, without
normalizing object text. Exact object text breaks hash ties. Ranks start at 1.
Neither review status nor data row order influences ranking. `ranking_hash` and
`candidate_rank` are stored with every proposition; changing an acceptance status
does not renumber the queue. A changed audit hash requires an explicit updated
configuration, even when reordering audit rows would produce the same ranks.

`fact_id` reuses the clean generator's existing identity:
`stable_id("fact", [topic, entity_id, rendered_atomic_statement, False])`.
The seed never enters fact identity. `proposition_key` preserves the exact
`[topic, entity, object]` tuple from the source audit.

## Registry and evidence

`registry.csv` has one row per candidate proposition:

```text
topic, split, entity_id, entity, candidate_object, candidate_rank, ranking_hash,
fact_id, statement, proposition_key, source_classification, validation_status,
evidence_source, evidence_note, reviewer_or_method, validation_version
```

Only exact source-audited false propositions initialize as
`source_supported_false`, with their source-file/row references. Other candidates
initialize as `unverified`. Unverified candidates are not established false.
An explicit review can mark a proposition `externally_validated_false` or
`rejected_true_or_ambiguous`; both require evidence source, evidence note, reviewer
or method, and a validation version. This code checks evidence presence and
provenance; it does not independently establish the truth of a reviewer's claim.
Evidence must be factual, never based on probe/model scores or performance.

The evidence-only review JSON is a list of objects with **exactly** these keys:

```text
fact_id, validation_status, evidence_source, evidence_note,
reviewer_or_method, validation_version
```

Extra fields, including score/model columns, are rejected. An unverified source
classification cannot be promoted to source-supported by an external review.
Registry loading rebuilds the pinned queue and rejects missing candidates or
modified ranks, identities, statements, or source classifications. Source-backed
evidence must match the audit. Review updates are written to separate snapshots,
with the parent registry hash; existing snapshots cannot be overwritten.

`metadata.json` records the seed, ranking algorithm/version, source audit file
hashes and combined audit hash, entity manifest hash/version, registry and
initial validation versions, registry content hash, row count, and aggregate
availability. Per-row validation versions record subsequent review versions.

## Development-only commands

Initialize only train or validation, writing separate registries:

```sh
python3 -B scripts/23_manage_negative_registry.py initialize --split train
python3 -B scripts/23_manage_negative_registry.py initialize --split validation
```

The default location is `data/clean_protocol/validated_negatives/v1/<split>/`.
Initialization prints aggregate counts only and renders no compound examples.
It transfers the source audit's existing classifications without manually
validating any unverified facts.

For a future authorized development review, `export` requires a new output
directory, and `review` additionally requires `--reviews` and
`--validation-version`. `--registry-dir` selects an existing snapshot;
`--output-dir` selects a separate destination under
`data/clean_protocol/validated_negatives/`. No review of real unverified
candidates was performed during implementation.

All initialize/export/review commands reject `--split test` before opening any
configuration, registry, candidate, or review file. There is no finalization
flag. Programmatic registry construction, loading, lookup, export and validated
generation also reject test exposure. Test queues are deferred entirely.

The frozen audit CSVs contain multiple splits. Their bytes are integrity-hashed
and CSV-decoded, but the split filter runs before interpreting candidate objects,
knowledge sets, or classifications. Test facts are never indexed, ranked,
classified, exported, displayed, or validated by this workflow. The shared
manifest may be read for structural identity checks.

## Clean generator contract

Add this explicit section to a future clean generation configuration:

```json
{
  "validated_negatives": {
    "config": "config/clean_protocol/validated_negatives.json",
    "registry_dir": "data/clean_protocol/validated_negatives/v1/train"
  }
}
```

The registry must match the generation split and shared manifest. Programmatic
callers can pass a loaded registry as `negative_registry=` to `CompoundGenerator`.
The default, without this option, remains byte-identical on the synthetic fixture.

For each entity, choose the lowest-ranked candidate whose status is
`source_supported_false` or `externally_validated_false`. Rejected and unverified
candidates are skipped without changing ranks. An accepted rank 2 can therefore
replace a rejected rank 1; an accepted rank 3 cannot override accepted rank 2.

An entity with no accepted candidate is marked `no accepted negative` in
`generator.unavailable_entities`. Its true fact remains accessible, but requesting
its false fact raises an error. Standard R1 pair sampling excludes unavailable
entities and checks capacity before loading facts. No unverified fallback is
permitted. Generation metadata records the registry hash/provenance and unavailable
entities, binding emitted negatives to the reviewed snapshot.

## Tests

```sh
python3 -B -m unittest discover -s tests -p 'test_validated_negatives.py' -v
python3 -B -m unittest discover -s tests -p 'test_clean_compounds.py' -v
```

Tests use invented evidence only. They cover deterministic queues, shuffle
invariance, effective seed with seed-independent fact IDs, exclusion of all
known-true objects, source initialization, rejection fall-through, unverified
unavailability, lowest-accepted selection, registry integrity, reduced pair
capacity, unchanged true facts, test-workflow guards, forbidden score/model
columns, byte-identical default generation, and unchanged exploratory files.
