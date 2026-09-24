# Shared entity partition manifest

Build or verify from the repository root:

```bash
python3 -B scripts/17_build_entity_partitions.py
python3 -B -m unittest discover -s tests -p 'test_entity_partitions.py'
```

Configuration: `config/clean_protocol/entity_split.json` (seed `0`).
`manifest.csv` maps `(topic, exact parsed entity)` to one split and records
`entity_id` and `compound_usable`. All entities in the affirmative/negated atomic
union are retained, including those without a true affirmative source row.
Usability requires a unique true object supplied by the affirmative source;
it does not infer objects from negated rows or independently verify world facts.

Within each topic and usability stratum, sort by SHA-256 of UTF-8 compact JSON
`[seed, topic, entity]` (`ensure_ascii=False`), breaking ties by exact entity.
Validation and test each receive `floor(n/5)` entities; train receives the
remainder. Allocate in train, validation, test order. Small unusable strata may
therefore all enter train. Entity IDs are `entity_` plus SHA-256 of the same JSON
encoding of `[topic, entity]`, independent of seed. No string normalization is
performed. Consumers must join by topic and exact entity, never split rows or
pairs independently; compound generation may use only usable entities within
one partition.

`counts.csv` contains aggregate entity counts and usable unordered pair
capacities (`n * (n - 1) / 2`, allowing entity reuse within a partition).
`metadata.json` records configuration, source hashes, algorithm, manifest hash,
and checks. Identical reruns verify existing files without rewriting them;
differing outputs are refused. A changed protocol needs a new output location.

These are entity assignments only. No compound rows or final evaluation
examples are generated. Final-test assignments must not be used to select
examples for inspection or scoring during method development. The frozen
exploratory benchmark remains unchanged; existing exploratory results do not
become untouched final-test estimates through this manifest.
