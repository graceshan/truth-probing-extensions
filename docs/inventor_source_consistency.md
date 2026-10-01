# Inventor source-label consistency

The inventor classifier applies this precedence to every candidate:

1. Parse every recorded true object into country components. Any overlap with
   candidate components yields `invalid_known_true`, regardless of a false label.
2. Only with no true-component overlap may an exact raw false source proposition
   establish source support (`supported_false` classification and
   `source_supported_false` registry status for an eligible candidate).
3. Otherwise the proposition is unverified.

The single-country generation requirement is separate. A non-overlapping slash
proposition can have a false source label without being eligible for generation.
Raw source labels, statements, and object strings are retained in audit provenance;
they are never silently corrected. Invalid candidates are omitted from semantic
queues, not assigned an invented registry validation status.

`src/inventor_country_semantics.py` already implements this precedence. The new
audit and focused tests verify that false-label lookup cannot override truth.

## Exhaustive non-test source-support cohort

```sh
python3 -B scripts/26_audit_inventor_source_consistency.py
```

The cohort uses the preserved train registry and validation v2, before semantic
exclusions. This captures all previous source support, including cases already
removed in v3. No test queue is loaded.

| Split | Previously source-supported | Truth-overlap conflicts | Remain after truth check | Additional slash-only exclusions | Eligible survivors |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 105 | 6 | 99 | 4 | 95 |
| Validation | 32 | 1 | 31 | 0 | 31 |
| Total | 137 | 7 | 130 | 4 | 126 |

The seven source-label inconsistencies are:

| Split | Entity | Raw true source value | Previously false candidate | Overlap |
| --- | --- | --- | --- | --- |
| train | Gideon Sundback | Sweden/the U.S | the U.S | the U.S |
| train | Hedy Lamarr | Austria/the U.S | the U.S | the U.S |
| train | Igor Sikorsky | Ukraine/the U.S | the U.S | the U.S |
| train | Maria Telkes | Hungary/the U.S | Hungary | Hungary |
| train | Mary Phelps Jacob | the U.S | Ukraine/the U.S | the U.S |
| train | Robert Moog | the U.S | Austria/the U.S | the U.S |
| validation | Luther Simjian | Turkey/the U.S | the U.S | the U.S |

Luther is the only validation case, not the only non-test case. The four additional
non-overlapping slash exclusions are Konrad Zuse (`Poland/France`), James Starley
(`Sweden/the U.S`), Hans Geiger (`Austria/the U.S`), and Leonardo da Vinci
(`Poland/France`). They remain truth-consistent source-false propositions but
cannot supply single-country generated negatives.

Files under `data/clean_protocol/audits/inventor_source_consistency_v1/` include:

- `all_previously_supported.csv`: all 137 cases, with raw true/false source
  statements, original labels and source row references.
- `source_label_inconsistencies.csv`: all seven demotions, exact fact IDs and
  overlapping components.
- `nonoverlap_slash_exclusions.csv`: the four separate generation exclusions.
- `counts.csv`, `metadata.json`, `verification.json`, and `preservation_checks.json`.

These are audit classifications; preserved historical registry files are not
mutated. The current semantic configuration enforces the corrected rules.

## Current validation registry and complete review batch

The v3 registry exactly matches a fresh truth-consistent reconstruction from v2.
It already excludes Luther's inconsistent candidate. No row changes or v4 snapshot
are necessary. All 74 manual non-inventor judgments are unchanged.

The new complete inventor-only validation batch is:

`data/clean_protocol/validated_negatives/v1/review_batches/validation_inventors_source_consistency_001/review_batch.csv`

It contains all **14** currently unresolved validation inventors, including Luther
Simjian, with one current lowest-ranked eligible single-country unverified
candidate each. It supersedes the earlier 13-entity export for the upcoming review;
the earlier file remains preserved. No judgments are imported during export.

```sh
python3 -B -m unittest discover -s tests -p 'test_inventor_source_consistency.py' -v
```

Tests use synthetic cases only and cover strict precedence, complete non-test
cohorts, raw-label preservation, deterministic output and test-queue guards.
No compounds, model scores or test candidate queues are generated or inspected.
