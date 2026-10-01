# T2A: provisional source-label and person-identity overlay

This is `selection_repair_v1/source_audit_v1`, based on recovery commit
`21996d9ba5b6b217a07a06d33c3d56db908c3b41`. It audits original source facts and
identities across outer train, validation (D), and test (E), under the explicit
T2A authorization. It does not inspect predictions, open activations, fit models,
or generate compound benchmarks. Historical development-only guards, entity
partitions, source labels, registries, and recovered code remain unchanged.

**All manifests are provisional.** “Admitted” means retained by these automated
checks, not independently verified true/false. Do not use the overlay to proceed
to selection or extraction until the pending factual review has passed. There
is no A/P allocation or sensitivity fitting here.

## Reproduce or verify

From the repository root, using the existing environment:

```bash
PYTHONDONTWRITEBYTECODE=1 /tmp/clean-extraction-venv/bin/python -B scripts/46_audit_selection_repair_sources.py
PYTHONDONTWRITEBYTECODE=1 /tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_selection_repair_source_audit.py tests/test_generalization_protocols.py tests/test_validated_negatives.py tests/test_negative_review_batches.py tests/test_inventor_country_semantics.py tests/test_inventor_source_consistency.py tests/test_validation_compound_benchmark.py tests/test_clean_compounds.py tests/test_negative_audit.py tests/test_entity_partitions.py
git diff --check
```

The entry point accepts no overrides. All existing outputs must match byte for
byte on rerun; any differing output or changed input provenance fails without
an overwrite. NumPy version is included in the receipt; use the recorded version
for byte-identical verification. Initial validation: 93 tests and 42 subtests
passed. Tests use synthetic fixtures and authorized source-fact reads; historical
benchmark tests exercise their existing development/synthetic contracts.

## Identity and evidence policy

Configuration and evidence decisions live in
`config/clean_protocol/selection_repair_v1/`. Source CSV hashes are checked against
original partition metadata. Dataset names and zero-based row indices identify
original rows; `source_row` also records one-based CSV data-row numbers. Each row
retains its original exact statement, label, entity ID, partition, original-column
JSON, source SHA-256, statement UTF-8 SHA-256, and canonical original-row JSON
SHA-256. Duplicate text remains separate original rows. Affirmative/negated row
pairs are checked for exact negation text, opposite label, identical entity and
partition at each original row index.

Country normalization splits slash lists, collapses component whitespace, and
uses the explicit case-sensitive mapping in `source_audit.json`. The U.S./U.K.
abbreviations expand to United States/United Kingdom; articles are removed from
the Netherlands and the Soviet Union. All other observed components have explicit
identity mappings. England/Scotland are **not** collapsed into the U.K.; historical
states are not equated with successor states. Unknown components fail closed.

Positive country support comes from affirmative source rows labeled 1, aggregated
by confirmed person identity, or explicitly adjudicated `supported_true` evidence.
This version adds no external residence claims. A false affirmative with **any**
normalized component overlapping supported true is provisionally excluded with
its paired negation. For slash-valued statements, this is a conservative conflict
exclusion, not a relabeling or proof of the entire compound country claim. An
incomplete positive set cannot establish falsity through nonmembership.

The original 43 reviewed-negative files and 19 inventor-audit files are hashed
opaquely in the receipt and remain untouched. Accepted false judgments, rejected
or ambiguous judgments, and supporting evidence retain their historical roles;
none is silently promoted to positive residence evidence. The Benjamin Franklin /
France historical `rejected_true_or_ambiguous` review remains unresolved here:
its Library of Congress note needs independent factual adjudication. It is a
historical generated candidate, not an identified original false-source row.

The two explicit identity decisions use retrieved biography titles and introductory
full names, with URLs, observed revisions, response hashes and short excerpts in
`identity_evidence.json`:

- [Luther Simjian](https://en.wikipedia.org/wiki/Luther_Simjian) identifies Luther
  George Simjian. These original entities span train and D; all 8 source rows
  across both partitions are quarantined. No row changes partitions.
- [Philo Farnsworth](https://en.wikipedia.org/wiki/Philo_Farnsworth) identifies
  Philo Taylor Farnsworth. Both entities are in train; all their retained rows
  share a common person key. Their exact source entity IDs remain intact.

These are single-secondary-source identity decisions, not independently
corroborated primary biographies or complete residence audits. Initial attempts
to retrieve an invent.org Simjian page and a Britannica Farnsworth page returned
404 and 403 respectively; they supply no evidence. Only the successful biography
responses support these decisions. Response hashes identify the fetched HTML;
short excerpts, not the full HTML, are archived in this task.

Unicode/case-normalized matches and first/last-token matches only nominate further
candidates; they never merge people. No additional candidates were found by this
limited rule. That is not a claim of an exhaustive alias audit. Unresolved
candidates, when found, remain separate and appear in `alias_decisions`; all
30 sampled residence judgments are still unresolved/pending.

## Measured counts

These counts are derived from manifests; no historical training-size constants
are used in the implementation. All exclusions are in the inventor topic.

| Partition | Before | Retained | Excluded | Retained entities | Retained person/entity keys |
|---|---:|---:|---:|---:|---:|
| Train | 3,144 | 3,128 | 16 | 907 | 906 |
| D | 1,040 | 1,036 | 4 | 295 | 295 |
| E | 1,028 | 1,022 | 6 | 296 | 296 |
| Total | 5,212 | 5,186 | 26 | 1,498 | 1,497 |

Before exclusion there are 1,500 original entities and 1,498 person/entity keys
across all partitions (the Simjian person key spans two partitions). Within
inventors: 812 rows become 786; 231 entities become 229, and 229 person keys become
228. `counts.csv` includes before/admitted/excluded counts for every topic,
partition and label, including zero cells and totals. Unique-key counts across
labels are not additive.

The 10 country-overlap false affirmatives are:

- Train: Igor Sikorsky, Gideon Sundback, Robert Moog, Maria Telkes, Hedy Lamarr,
  Mary Phelps Jacob.
- D: Luther Simjian.
- E (source facts only): Blaise Pascal, John von Neumann, Elisha Gray.

Their 20 paired rows overlap the 8-row Simjian quarantine by 2 rows, giving
26 distinct exclusions. Reasons and exact positive-support source-row IDs are
recorded on both forms in `excluded_rows.csv`. Non-overlap slash-valued source
records are not automatically declared false or excluded merely for formatting.

## Fixed review sample

`review_sampling_frame.csv` contains all **166** retained inventor false
affirmatives across train/D/E, ordered by `(dataset, integer row_index)`.
`numpy.random.Generator(numpy.random.PCG64(20261001)).choice(166, size=30,
replace=False)` draws the sample once, with queue order preserving draw order.
No stratification, quota, replacement, redraw, or factual-result-based filtering
is applied. This draw contains **20 train, 5 D, 5 E** rows.

The review queue is:

`data/clean_protocol/selection_repair_v1/source_audit_v1/review_queue_30.csv`

All 30 `review_status` values are `pending`, with empty evidence/decision fields.
Future review adjudications should be separate versioned evidence; do not mutate
or resample this frozen queue. Review failure requires an explicit subsequent
protocol decision, not automatic promotion of provisional manifests.

## Artifacts and future reuse

Under `data/clean_protocol/selection_repair_v1/source_audit_v1/`:

- `row_manifest.csv`: every original row, overlay status and reasons.
- `admitted_rows.csv`, `excluded_rows.csv`: exact subsets of that manifest.
- `counts.csv`: manifest-derived counts by topic, partition, label and unique key.
- `review_sampling_frame.csv`, `review_queue_30.csv`: full frame and fixed draw.
- `audit_manifest.json`: input/code/evidence hashes and sizes, output hashes,
  alias decisions, unresolved facts, sampling policy, counts and scope receipt.

Original `(dataset, row_index, source_sha256, statement_sha256)` identities support
a later explicit activation index map. **Activation compatibility is not verified.**
Any reuse must separately check model/revision, token position, saved layer,
representation details and exact sidecars. This source-only audit exposes no
activation or prediction loader and changes no historical scientific output.
