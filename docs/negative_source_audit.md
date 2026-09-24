# Source-data negative audit

Run `python3 -B scripts/19_audit_negative_sources.py`. Outputs are saved to
`data/clean_protocol/audits/source_negatives_v1/`; differing existing outputs
cannot be overwritten. Tests: `python3 -B -m unittest discover -s tests -p
'test_negative_audit.py'` (enter the test command on one line).

The audit parses all rows in the five affirmative atomic sources used by the
compound generator. It uses no external knowledge, no negation-derived labels,
and does not call benchmark generation or select a seeded false fact.

`propositions.csv` preserves every parsed source occurrence, including duplicates,
with exact entity/object strings, topic-scoped entity ID, source statement and
label, dataset/file, and one-based data-row reference (excluding the CSV header).
The proposition key is compact UTF-8 JSON `[topic, entity, object_value]`, without
case, whitespace, or Unicode normalization. `entity_knowledge.csv` retains all
distinct true and false objects, including contradictory memberships and entities
without true rows. No inconsistency is silently resolved.

`issues.json` reports contradictory proposition labels, duplicate propositions,
multiple true objects, parse failures, invalid labels, manifest coverage gaps,
and provenance mismatches. Duplicates are counted both as proposition groups and
as extra occurrences in `summary.csv`. A contradiction remains an explicit issue;
classification precedence never treats a known-true object as a supported false.

`candidate_audit.csv` enumerates alternatives from the union of true objects of
compound-usable entities within each manifest topic/split, excluding the target's
unique true object exactly as the current generator does. These are potential
choices across generation seeds, not sampled negative facts. Classifications:

- `invalid_known_true`: in the entity's known-true object set, even if also labeled false.
- `supported_false`: explicitly labeled false for the exact proposition and not known true.
- `unverified_negative`: neither known true nor explicitly labeled false. This is
  not an established false fact.

If an entity has zero or multiple true objects, the current generator would
reject that topic/split. The audit instead preserves all true objects, reviews
the broader object-pool alternatives, and marks `generator_stratum_blocked=True`.
It does not pretend those blocked rows are currently selectable. Metadata reports
every stratum's block status. Parse/provenance gaps are also explicit; results
must not be considered complete when the corresponding metadata flag is false.

`summary.csv` counts all atomic/manifest entities and also reports the usable
subset. `entities_zero_supported_false` includes unusable entities, which have no
generator candidates. `usable_entities_zero_supported_false` isolates the usable
subset. Candidate totals sum the three mutually exclusive classifications.
An explicit false source proposition outside the split-local object pool does
not count as a currently available candidate.

`train_examples.csv` contains up to two deterministic examples per topic and
classification, restricted to train-assigned entities. It contains proposition
tuples, not compound sentences. Held-out candidates are processed automatically
for the exhaustive audit; no held-out example display, sampling, scoring, or
compound benchmark generation is performed. `metadata.json` records source and
manifest hashes, scope, diagnostic counts, aggregate summaries, and stratum status.

The clean generator remains unchanged: it requires one true object, excludes
that object from its split-local pool, and chooses a fixed alternative by hash
rank using the generation seed. It never checks explicit false source labels.
