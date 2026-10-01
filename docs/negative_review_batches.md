# Sequential validation factual-review batches

`scripts/24_prepare_negative_review_batch.py` exports exactly one lowest-ranked
currently-unverified candidate for each compound-usable validation entity with
no `source_supported_false` or `externally_validated_false` candidate anywhere
in its queue. It does not change statuses, add factual judgments, or consume
model/probe outputs. Train and test requests are rejected before factual reads.

The first batch is stored at
`data/clean_protocol/validated_negatives/v1/review_batches/validation_batch_001/`:

- `review_batch.csv`: sorted by `topic, entity_id`, with entity identity, candidate
  rank/object/complete statement, all recorded known true objects as a JSON array,
  source classification, fact ID, and current validation status.
- `summary.csv`: per-topic entity and exported-candidate counts, including any
  exhausted queues separately from entities with accepted negatives.
- `metadata.json`: registry, audit, manifest and batch hashes, seed/ranking
  version, selection rule, and previous-batch hash when applicable.

Known true objects are copied from the pinned source audit. They are context for
review, not a new judgment about the candidate. Only validation knowledge is
interpreted; the existing mixed-split audit's bytes are integrity-hashed, and
other splits are discarded before interpreting their factual fields.

First-batch command:

```sh
python3 -B scripts/24_prepare_negative_review_batch.py first \
  --output-dir data/clean_protocol/validated_negatives/v1/review_batches/validation_batch_001
```

After evidence-backed reviewed statuses have been imported into a **new validation
registry snapshot** using the existing registry review command, use:

```sh
python3 -B scripts/24_prepare_negative_review_batch.py next \
  --registry-dir path/to/reviewed_validation_registry_snapshot \
  --previous-batch data/clean_protocol/validated_negatives/v1/review_batches/validation_batch_001 \
  --output-dir data/clean_protocol/validated_negatives/v1/review_batches/validation_batch_002
```

Replace the registry path with the imported snapshot directory. `next` requires
the previous batch and verifies its content hash and queue provenance. An entity
with any accepted negative disappears. An unreviewed candidate stays unchanged.
A rejected candidate advances exactly to rank +1; if that was the final rank,
the entity is reported as exhausted. Imported out-of-order rejections that would
skip the next rank fail explicitly. Newly unavailable entities are included at
their lowest unverified candidate. Re-running with the same registry and previous
batch gives byte-identical output; output directories cannot overwrite differing
existing files. Use `next` for progression, not a fresh `first` invocation.

This task exported no candidates above rank 1 and imported no reviews. Status
changes in progression tests use synthetic fixtures only.

```sh
python3 -B -m unittest discover -s tests -p 'test_negative_review_batches.py' -v
```
