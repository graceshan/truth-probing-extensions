# Inventor single-country negative semantics

The raw inventor relation is still the source statement `entity lived in object`.
Source strings and true rows are never rewritten or removed. For negative
eligibility, `src/inventor_country_semantics.py` splits an object on `/`, trims
and collapses whitespace within each component, and preserves case, punctuation,
articles and country spelling. It does not infer aliases such as `US` = `the U.S`
or `Turkey` = `Türkiye`. Empty components fail explicitly.

A candidate is `invalid_known_true` whenever any candidate component overlaps
any component of any recorded true object for that entity. This classification
takes precedence over a source false label. Independently, every slash-valued
candidate is ineligible for clean false generation, even when no true component
overlaps. A false slash-valued source proposition does not establish its individual
components as false.

The new inventor pool consists of atomic country components from all usable
entities' true objects in the **same shared split**. This adds `Poland` and `Turkey`
to the validation pool while removing `Poland/Germany` and `Turkey/the U.S` as
candidate values. An entity's known-true components are removed before ranking.
Seed 0 and the existing `negative-candidate-v1` SHA-256 inputs are unchanged.
Ranks are recomputed densely over eligible candidates. Surviving facts retain
their IDs, ranking hashes, and review status; inventor ranks can change because
the candidate pool changed. Every non-inventor row is copied exactly.

Use `config/clean_protocol/validated_negatives_inventor_single_country.json` to
load or review the new registry. Its registry format/provenance version is
`validated-negative-registry-v2`, with semantic version
`inventor-single-country-v1`. The original configuration remains available for
reading the preserved historical snapshots.

The clean generator applies single-country eligibility to inventor negatives.
If a legacy validated registry selects a slash value or a known-true component,
generation raises an error directing the caller to rebuild the registry. The
default clean candidate pool also uses single-country components. Exploratory
code is unchanged. True atomic facts preserve their original multi-country text.

## Rebuilt artifacts

Run the validation-only rebuild and audit:

```sh
python3 -B scripts/25_rebuild_inventor_negatives.py
```

It writes new artifacts and refuses to overwrite differing existing ones:

- Registry:
  `data/clean_protocol/validated_negatives/v1/validation_reviewed_batch_001_inventor_single_country_v3/`
- Audit: `data/clean_protocol/audits/inventor_single_country_v1/`
- Review batch:
  `data/clean_protocol/validated_negatives/v1/review_batches/validation_inventors_single_country_001/`

The audit records all train/validation inventor atomic object occurrences,
including raw statements, labels, parsed components and source references. Test
entities are filtered by manifest membership before their object fields are
parsed. Their facts and candidate queues are not inspected or exported. The audit
of candidate changes covers the specified validation registry, not a regenerated
test or train registry. Existing mixed-source bytes are hashed for integrity.

`slash_values.csv` lists every slash value in this non-test scope and distinguishes
true-source, false-source and old validation-candidate occurrences.
`excluded_candidates.csv` lists every removed candidate, with reasons and former
status. `newly_invalid_known_true.csv` lists all newly detected true-component
overlaps. `source_semantic_conflicts.csv` preserves all source false propositions
that overlap recorded truth components. `new_candidate_audit.csv` and
`new_candidates.csv` record the rebuilt pool and additions. `rank_one_changes.csv`
lists every validation inventor whose first candidate changes.

## Results and newly exposed inconsistency

The validation inventor queue changes from 405 to 403 propositions: 90 old
candidates are removed and 88 single-country candidates are added. Of the old
candidates, 27 are newly classified `invalid_known_true`; 17 entities change
their rank-1 candidate. All 74 prior manual judgments and all non-inventor rows
are preserved exactly. No manual inventor judgments are imported.

The three named cases are all excluded:

- Henry Ford / `Turkey/the U.S`: both slash-ineligible and `invalid_known_true`.
- Hans von Ohain / `Turkey/the U.S`: slash-ineligible; no recorded true overlap.
- George Washington Carver / `Poland/Germany`: slash-ineligible; no recorded true overlap.

An additional conflict is exposed for **Luther Simjian**: true source object
`Turkey/the U.S` overlaps the previously `source_supported_false` candidate
`the U.S`. That candidate is invalid under the new rule and is removed, with its
old evidence retained in the audit and v2 snapshot. Luther Simjian now lacks an
accepted negative, bringing total unresolved validation inventors to **14**.

The requested fresh review batch remains scoped to the **original 13 unresolved
inventor entities from batch 001**, one current lowest-ranked unverified candidate
each. It does not silently include Luther Simjian or any other topic. The extra
unresolved entity is explicitly reported in the audit and verification metadata.
This is a fresh semantic queue version, not a rejection-based advancement of the
old batch; old batch ranks must not be used as review targets for the new queue.

## Validation

```sh
python3 -B -m unittest discover -s tests -p 'test_inventor_country_semantics.py' -v
python3 -B -m unittest discover -s tests -p 'test_negative_review_batches.py' -v
```

Tests use synthetic atomic constituents and review statuses. They cover component
overlap, formatting-only normalization, source-label conflicts, raw true-row
preservation, single-country eligibility, unchanged non-inventor judgments,
deterministic migration, registry reload, shuffle-invariant ranks, inventor-only
batch export and test guards. No compounds or model scores are generated.
