# T2C: bounded source follow-up and pending candidate overlay

This source-only task starts from `68f861cac10d0e631fc215e1f09210f6d5e9ca90`
on `clean-eval-protocol`, macOS, in
`/Users/apple/projects/truth-probing-extensions`. Origin was verified as
`graceshan/truth-probing-extensions`; fetch and fast-forward-only pull found no
divergence. No other worktree was modified. The 13 unrelated untracked files
are individually hashed in `candidate_overlay_v1/preservation_baseline.json`.

**This is a candidate pending review, not the production default.** No source
CSV, historical judgment, T2A manifest, T2B sample, partition, or source label
is changed. The user authorized this new overlay after T2B's proposed-only
stage; that earlier package and its restrictions remain historically intact.
No activations, models, fitting, extraction, compound generation, predictions,
or scientific metrics were accessed or run. Source-only E facts are in scope.

## Policy, scope and evidence

All new artifacts are under
`data/clean_protocol/selection_repair_v1/candidate_overlay_v1/`.
`policy.json`, `targeted_queue.json` and `screening_inventory.json` were saved
and hashed in `pre_adjudication_freeze.json` before the new adjudications.
Residence semantics remain those in T2B `review_policy.md`: established homes
or residential study/employment postings, not nationality, birthplace, visits
or touring. There is no new duration cutoff, slash operator, border mapping,
or inference of falsity from an incomplete residence set.

After carrying T2A restrictions and the 52 T2B recommended rows, the bounded
screen enumerates all **140 retained inventor false affirmatives**: 83 train,
30 D, 27 E. It uses exact subjects in the existing evidence, eleven fixed
residence/identity leads, and declared semantic triggers. Of these 140, 22
enter the target queue and 118 are marked **screened, no trigger found in
bounded material**, which is not a supported-false judgment. The pre-review
screen flags 136/140 as lacking a completed substantive residence review;
the other four retain their earlier T2B supported-false judgments. No claim
is made to have exhaustively researched every inventor's lifetime.

The complete target queue has **42 affirmative claims**, including both
original labels: 27 train, 7 D, 8 E. Its overlapping triggers are 16 slash
claims, 4 joint-person claims, 2 explicit historical-polity claims, 5 fixed
pre-US geography sentinels, 11 residence/identity leads, and 4 companion true
claims supported by previously inspected biographies. This historical-geography
watchlist is bounded, not a census of all border ambiguities.

All 42 receive separate follow-up reviews, with evidence URLs, titles,
retrieval dates, response hashes, inspected excerpts/locators, reasoning and
limitations. Failed or inaccessible retrievals remain in `retrieval_log.json`;
they do not support judgments. Reused T2B evidence is referenced by exact
package hash and evidence ID. Reviewer: **Codex-assisted external review,
not human verification**. Full web responses are not copied into Git;
short locatable excerpts and retrieval hashes are preserved.

## Findings and limitations

The target outcomes are **13 supported true, 29 unresolved, 0 newly supported
false**. Seven supported-true outcomes contradict original false labels:

| Original affirmative ID | Partition | Claim | Inspected support | Limitation |
|---|---|---|---|---|
| inventors:2 | Train | Walton / US | Explicit two-year factory-startup posting | Secondary biography |
| inventors:68 | D | Marconi / UK | English Heritage London plaque: lived here 1896–1897 | Address-specific evidence |
| inventors:139 | Train | Kapany / UK | Imperial London doctoral study 1952–55 | Secondary biography; institutional lead unavailable |
| inventors:233 | D | Göbel / US | Explicit New York settlement; contemporary Henry Goebel patent | Alias/residence narrative uses secondary biography |
| inventors:234 | Train | Bengio / US | MIT and Bell Labs postdoctoral appointments | Secondary biography; ACM page inaccessible |
| inventors:240 | E | Ericsson / US | Explicit New York settlement | Secondary settlement account; institutional US-work corroboration is narrower |
| inventors:377 | Train | Brill / US | NIHF California employment plus USC study | Posting-specific evidence |

These seven pairs are excluded in the **candidate** only. Secondary-only
findings need independent corroboration before production promotion. The
failure pattern remains omitted overseas residential work/study or migration,
not a license to infer every foreign job or visit was residential.

The six agreeing true-label reviews concern Porro/Italy, both Wright brothers/US,
Basov/Soviet Union, Edison/US, both Brin and Page/US, and Maxwell/UK. Joint-person
support examines both people. Italy and the Soviet Union are supported by
postings during the named polity's existence; this does not introduce a general
modern-border equivalence.

All 29 unresolved targets and their negations are quarantined. They include
every targeted slash statement; ambiguous historical geography; the surname-only
Kwolek identity; Daimler's French-posting lead with incomplete historical-location
corroboration; Theremin's German tour; Goodyear's UK stays; the joint Lumière
claim; and the Brin/Page UK claim. No normalized-name merge is added. The
Huygens Netherlands positive remains unresolved under historical semantics;
the original T2B Scotland-negative outcome is preserved independently.

Franklin/France remains an **ineligible historical negative candidate**, fact
`fact_49c7c4369a56220424ffb4d35bab8ec46e341f69329209a10d008b109a4383c4`.
Its historical `rejected_true_or_ambiguous` record and later LOC-supported
positive adjudication remain separate. It has no fabricated source-row pair
and contributes nothing to the fixed sample denominator.

The original sample is still exactly 30 draws in the same order: **4 supported
false, 2 supported true, 24 unresolved**. Follow-up outcomes never replace,
redraw or inflate that sample, and no new error-rate estimate is inferred from
the deliberately targeted queue.

## Candidate overlay and capacity

`row_manifest.csv` preserves every original T2A column verbatim and appends
candidate status, reasons, evidence references and pending status. Separate
admitted, excluded and quarantined tables are exact subsets. Paired handling is
mandatory and overlapping restrictions are deduplicated. Simjian's eight rows
remain quarantined across train/D; Farnsworth's two original train entity IDs
retain their shared person key. No records move partitions.

The 26 T2A restrictions become 18 excluded plus 8 Simjian quarantined rows,
including the original two-row overlap. T2B adds 4 excluded and 48 quarantined;
T2C adds 14 excluded and 58 quarantined. All original reasons survive.

| Partition | Original rows | Admitted | Excluded | Quarantined | Admitted entities | Admitted person keys |
|---|---:|---:|---:|---:|---:|---:|
| Train | 3,144 | 3,048 | 24 | 72 | 900 | 899 |
| D | 1,040 | 1,012 | 4 | 24 | 294 | 294 |
| E | 1,028 | 1,002 | 8 | 18 | 295 | 295 |
| Total | 5,212 | 5,062 | 36 | 114 | 1,489 | 1,488 |

The following capacity bounds count **person/entity keys**, not available
compound pairs. `Source pair` means at least one retained positive and negative
affirmative source fact for the same key. Negations never create a missing
independent affirmative fact. `Positive bound` is conditional on retained
positive source labels being valid and obtaining an audited negative; it does
not authorize new facts or promise review will pass. `Historical negative pair`
combines a retained source-positive with an eligible historical externally
reviewed negative, without calling the positive independently audited.

| Topic | Partition | Original/admitted/excluded/quarantined rows | Positive bound | Source pair | Historical negative pair | Audited positive+negative pair |
|---|---|---|---:|---:|---:|---:|
| cities | Train | 1800 / 1800 / 0 / 0 | 450 | 450 | 0 | 0 |
| cities | D | 596 / 596 / 0 / 0 | 149 | 149 | 24 | 0 |
| cities | E | 596 / 596 / 0 / 0 | 149 | 149 | 0 | 0 |
| sp_en_trans | Train | 424 / 424 / 0 / 0 | 106 | 4 | 0 | 0 |
| sp_en_trans | D | 146 / 146 / 0 / 0 | 34 | 1 | 34 | 0 |
| sp_en_trans | E | 138 / 138 / 0 / 0 | 34 | 0 | 0 | 0 |
| inventors | Train | 492 / 396 / 24 / 72 | 126 | 65 | 0 | 2 |
| inventors | D | 162 / 134 / 4 / 24 | 43 | 24 | 11 | 0 |
| inventors | E | 158 / 132 / 8 / 18 | 43 | 22 | 0 | 1 |
| element_symb | Train | 228 / 228 / 0 / 0 | 57 | 57 | 0 | 0 |
| element_symb | D | 72 / 72 / 0 / 0 | 18 | 18 | 13 | 0 |
| element_symb | E | 72 / 72 / 0 / 0 | 18 | 18 | 0 | 0 |
| animal_class | Train | 200 / 200 / 0 / 0 | 48 | 48 | 0 | 0 |
| animal_class | D | 64 / 64 / 0 / 0 | 16 | 16 | 3 | 0 |
| animal_class | E | 64 / 64 / 0 / 0 | 16 | 16 | 0 | 0 |

The three fully audited source pairs are Edison and Maxwell in train, and Porro
in E. Here “audited” means evidence-reviewed in these Codex-assisted packages;
it is not human certification. All remain subject to candidate review.

`capacity.csv` provides the full 15 topic/partition rows, fact counts, original
and admitted/excluded/quarantined distinct entities/person keys, and explicit
10/15/20 threshold statuses. `counts.csv` breaks every stage down by label and
form, including zero cells. Unique-key counts across statuses or labels overlap
and must not be added. `capacity_by_person.csv` links the counts to original
entity IDs. `fact_inventory.csv` identifies the exact statements and evidence
tier behind capacity, including explicit ineligibility reasons.

Historical registry facts retain their original validation status. Unverified
rows are tallied in `historical_registry_counts.csv` but provide no negative
capacity. `source_supported_false` remains source-label support, not an external
audit. Existing train and latest D registries are inventoried; no E registry
is generated. Generated historical inventor negatives are conservatively
unavailable for any person with a restricted source claim until revalidation;
exact source facts with separate accepted evidence are assessed individually.
Rejected claims, including Franklin/France, never count as available negatives.

## Remaining decisions and factual work

- **10, 15, 20 train adaptation entities per topic:** all fit within the retained
  positive-entity bounds, but **none is currently supported as a fully audited
  five-topic allocation**, even at 10. This is conditional capacity, not a
  reservation or selection of A/P entities.
- D/E bounds permit 10 or 15 per topic conditional on review. **20 per topic
  is structurally unavailable** for animal_class (16 positive-bearing keys)
  and element_symb (18), without changing this retained positive pool. No such
  change is made or proposed automatically.
- For a hypothetical n-person train requirement, each topic needs n audited
  positive and n audited negative facts on the same n keys. The train inventor
  ledger supplies only two completed pairs; remaining topics have none. Review
  should validate country/residence/identity meanings, city country facts,
  element symbols, animal categories and Spanish translations, including
  negatives, rather than treating source labels as independent audits.
  Thus 10/15/20 train inventor keys require at least 8/13/18 additional
  completed pair audits; train's other topics require 10/15/20. E inventors
  would need 9/14/19 additional completed pair audits, subject to retaining
  Porro's review. D inventors currently need the full requested number of
  completed pairs, although 11 keys have historical reviewed negatives.
  These are outstanding pair-level review counts, not permission to select
  those keys or a guarantee that reviews will succeed.
- **Spanish:** train/D/E have 208/68/68 original entity keys, but only
  106/34/34 have retained positive affirmatives. Only 4/1/0 have source-paired
  positive and negative affirmatives on the same key. D's 34 historical
  reviewed negatives are useful evidence, but their positive translations
  still require review. Train and E need exact same-word negative evidence
  for prospective selected keys. False-only terms cannot be paired with
  unrelated true terms; polysemy, spelling/diacritics, dialect and translation
  direction need explicit factual treatment. No new translations are created.
- A, T_C and E factual-review budgets depend on an as-yet-unapproved role plan.
  These tables leave outer partitions intact and do not assign roles. Joint
  disjoint A/T_C feasibility cannot be inferred without both requested sizes
  and passing same-entity positive/negative audits. Never move D/E into train.
- Review the secondary-source findings, resolve or retain quarantine for
  ambiguous residence/geography/identity claims, and approve the candidate
  before any experiment. No activation compatibility is claimed: any later
  reuse requires exact source-row/statement index mapping plus model/revision,
  token-position, layer and sidecar checks.

## Reproduce and validate

```bash
/tmp/clean-extraction-venv/bin/python -B scripts/47_build_selection_repair_candidate.py
/tmp/clean-extraction-venv/bin/python -B scripts/47_build_selection_repair_candidate.py --check-only
/tmp/clean-extraction-venv/bin/python -B scripts/46_audit_selection_repair_sources.py
/tmp/clean-extraction-venv/bin/python -B data/clean_protocol/selection_repair_v1/source_review_v1/validate.py
PYTHONDONTWRITEBYTECODE=1 /tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_selection_repair_candidate_overlay.py tests/test_selection_repair_source_audit.py data/clean_protocol/selection_repair_v1/source_review_v1/test_validation.py
git diff --check
```

The candidate builder accepts no scientific/path overrides, verifies locked
inputs before processing, and refuses differing existing outputs before writing
anything. Reruns are byte-identical. The input lock covers original sources,
partition/audit inputs, all reviewed-negative history and inventor audit history,
and T2A/T2B packages. The candidate receipt binds inputs, evidence, code and
derived output hashes. Initial focused validation: **48 tests plus 17 subtests
passed**, including deterministic reruns, mutation failures, sample preservation,
paired restrictions, alias separation, capacity tiers and all 13 unrelated
file hashes. Both historical validators pass without rewriting their outputs.
