# Prospective capacity pilot: 15/topic A, nested 10/topic fallback

This fact-only package proposes **15 completed same-entity pairs per topic**
for the capacity pilot. It does **not** freeze A/P, generate production
compounds, fit probes, access activations, inspect predictions, or adopt a cache
mapping. The user accepted v2's conservative correction policy for this stage;
historical overlay/recommendation/receipt bytes and mac 2's separate records
remain unchanged.

The package is
`data/clean_protocol/selection_repair_v1/capacity_pilot_audit_v1/`.
`A15_proposal.json` contains all 75 exact positive/negative fact pairs, identity
bindings, evidence references, judgments and provenance. `A10_proposal.json`
contains the first ten completed pairs in each topic's same order. These are
reviewable pilot proposals, not final reservations or permission to fit.

## Git and ordering provenance

mac 1 reported Darwin, branch `clean-eval-protocol`, reviewed starting HEAD
`827f01460802afde5dd82da424159f284812623c`, and exactly the 13 expected unrelated
untracked files. Fetch found origin synchronized, so no pull was needed. The
normal merge incorporated exactly `455016040208d0304b4b7547bb96dc626c979ac5`,
preserving both histories. No subsequent mac 2 commits were incorporated.

Before new judgments, `pre_review_lock.json` and `candidate_order.json` recorded
the input hashes, eligible population, full draw order and RNG. Each topic
starts with ascending Unicode-codepoint reviewed `person_key`, then uses a
fresh **CPython `random.Random(20261002)` (MT19937) `shuffle`**. The same seed
is restarted per topic. Topics are sorted lexicographically. Source row order,
scores, predictions and subsequent factual outcomes do not enter key ordering.

Eligibility requires a v2-admitted, label-1 affirmative in outer train, with
all original rows for the reviewed person key confined to train. Quarantined
claims never supply facts. Farnsworth's two entity IDs share the reviewed
person key; Simjian's cross-partition identity is ineligible. The full fixed
pool has 450 cities, 106 Spanish words, 126 inventor people, 57 elements and
48 animal keys. Positive-bearing eligibility is not an independent factual
certificate. False-only training entities remain in the original/P reference.

`pre_judgment_receipt.json` additionally locks the proposed negative queue and
fact-selection rule. We use the lowest numerical source row index for each
available affirmative polarity. If no admitted same-person false affirmative
exists, we use the first candidate from the recovered reviewed-negative
infrastructure: `validated_negatives.initialize`, train only,
`inventor-single-country-v1`, negative seed 0. Its SHA-256 ranking, exact fact
IDs, objects, statements and original unverified statuses are preserved in
`negative_proposals.json`. No wrong-object fallback or historical registry
status rewrite occurs. A proposed negative counts only after adjudication.

## Audit outcomes

| Topic | Attempted | Usable | Rejected | Unresolved | Last pilot rank |
|---|---:|---:|---:|---:|---:|
| Cities | 15 | 15 | 0 | 0 | 15 |
| Spanish | 15 | 15 | 0 | 0 | 15 |
| Inventors | 23 | 15 | 0 | 8 | 23 |
| Elements | 15 | 15 | 0 | 0 | 15 |
| Animals | 15 | 15 | 0 | 0 | 15 |
| **Total** | **83** | **75** | **0** | **8** | |

`reviews.json` retains every consecutive pilot attempt. The unresolved inventor
keys are Konrad Zuse (rank 1), John Wesley Hyatt (8), Felix Hoffmann (9),
Herbert Saffir (12), Edwin Link (14), Philo Farnsworth/Philo Taylor Farnsworth
(17), Hippolyte Mège-Mouriès (19), and Robert Chesebrough (20). Their positive
facts are supported, but residential gaps prevent acceptance of their exact
negative propositions. They are not redrawn or hidden by later successes.
No unresolved label is declared a proven source error.

The inventor proposal, in completion order, is Trevithick, Alferov, Kelly,
Handler, Benz, Yamanaka, Geiger, Oersted, Edison, Theremin, Bombardier, Morgan,
Appert, Fry and Galdikas. The fallback is the first ten, ending with Theremin.

These are **Codex-assisted external reviews, not human certification**.
`evidence.json` records actual retrieved text excerpts/locators, titles, URLs,
retrieval dates, response hashes and explicit fact bindings. Failed and
irrelevant retrievals remain in `retrieval_log.json`. Successful response
bytes were inspected locally; full third-party pages are not committed.
Reused evidence and completed judgments point to their historical packages
with file hashes and original IDs. Edison's completed pair is reused.

The factual standard distinguishes a source label from an external judgment:

- City references locate the identified city, with explicit spelling/identity
  treatment for Cần Thơ, Yaoundé, Antwerpen and Wanzhou's urban core. A different
  country's namesake is not substituted for the source entity.
- NIH PubChem's element table binds each correct symbol and assigns the proposed
  false symbol to another element. Symbol case is preserved.
- Zoological references bind the animal to its ordinary category and scientific
  classification. Seagull/gull and peacock/peafowl aliases are explicit. Exact
  source text, including its grammatical mistakes, is retained.
- All 15 Spanish negatives are independently adjudicated recovered-pipeline
  candidates for the **same word**. Cambridge's GLOBAL/PASSWORD entries and
  SpanishDictionary.com entries were checked for senses, idioms, spelling,
  diacritics, relevant regional uses and Spanish-to-English direction. Examples
  include arena's sand/arena/stone senses, cuerpo's body/corps meanings, Mexican
  telephone *bueno*, Latin American *viejo* and Southern Cone *grupo*. A valid
  positive sense is not treated as the only possible translation. No D/E fact
  is transferred into training, and no negation supplies the second fact.
- Inventor negatives use sustained home/education/employment chronologies and
  the overseas episodes they describe. Nationality, patents, company exports,
  modern-border substitutions and absence from a partial country list are
  insufficient. Alferov's account explicitly distinguishes a short UK visit
  from his six-month Illinois posting. Yamanaka's autobiography is supplemented
  by Kyoto's current CV. Bombardier's museum provides a detailed chronology.
  These residence judgments remain defeasible; they are not universal proofs
  that undocumented stays were impossible. Restricted historical claims remain
  restricted even when a different exact negative passes a new review.

## Prospective P membership and exposure

For each proposal, remove **every original train source variant** of each A
person/entity, including aliases, both source labels, affirmatives and their
negations. Intersect the remainder with v2 admission to obtain prospective P.
The package saves the full removed-variant list, original-source remainder,
admitted P membership and actual sampler-selected rows for both sizes.

The original reference contains 3,144 rows; v2 admits 3,048. Their separate
reference CSVs and balanced-sampler selections are retained. At A15, 266
original source rows are removed, of which 262 are admitted by v2. At A10,
180 original rows are removed, of which 176 are admitted. The four-row
difference is already-restricted Geiger/Theremin material, still listed in the
person-wide exclusions. No historical source labels or partitions change.

In the tables, each P label count is for **each** of labels 0 and 1. Sampled
rows are the actual result of the existing `balanced_burger_indices`, seed 0,
in original v2 source order. The sampler selects matching affirmative/negated
rows; it does not expose every available P entity. Sampling metadata and full
identity lists are included, with NumPy/pandas versions.

| Topic | Original rows | v2 admitted rows | A15 P rows | Each label | P entities / people | Sampled rows | Sampled people |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cities | 1,800 | 1,800 | 1,740 | 870 | 435 / 435 | 140 | 68 |
| Spanish | 424 | 424 | 394 | 197 | 193 / 193 | 140 | 69 |
| Inventors | 492 | 396 | 344 | 172 | 118 / 117 | 140 | 61 |
| Elements | 228 | 228 | 168 | 84 | 42 / 42 | 140 | 42 |
| Animals | 200 | 200 | 140 | 70 | 37 / 37 | 140 | 37 |
| **Total** | **3,144** | **3,048** | **2,786** | **1,393** | **825 / 824** | **700** | **277** |

| Topic | A10 P rows | Each label | P entities / people | Sampled rows | Sampled people |
|---|---:|---:|---:|---:|---:|
| Cities | 1,760 | 880 | 440 / 440 | 160 | 74 |
| Spanish | 404 | 202 | 198 / 198 | 160 | 79 |
| Inventors | 360 | 180 | 123 / 122 | 160 | 73 |
| Elements | 188 | 94 | 47 / 47 | 160 | 46 |
| Animals | 160 | 80 | 42 / 42 | 160 | 42 |
| **Total** | **2,872** | **1,436** | **850 / 849** | **800** | **314** |

Both unreserved references expose 200 rows/topic (1,000 total). The 15/topic
pilot exposes 70 affirmative plus 70 negated rows/topic, 70 of each final
label. The fallback exposes 80 plus 80, 80 of each label. These are distinct
selected source rows for one sampler draw, not optimizer-step counts.
`exposure_counts.csv` also reports all original/remainder counts, labels,
forms and distinct entity/person keys without assuming four rows per key.

## T_C and corrections

T_C may overlap either proposed P or A, always within outer training.
`tc_recovered_pairs.json` reuses Maxwell's already completed source pair at
reservation rank 45 **only for T_C availability**; it does not skip the A queue
or enlarge A. T_C therefore has 15 completed keys in each non-inventor topic
and 16 inventors: **76 completed pairs** in total.

Reaching 20/topic requires at least **24 additional completed pairs**: five
each for cities, Spanish, elements and animals, and four inventors. The separate
`TC_topup_review_queue.csv` preserves the fixed ranks, exact positive and
negative candidates, negative provenance and current review state for all
remaining eligible keys. Unresolved inventor attempts can receive targeted
negative follow-up; other keys need both fact reviews. The queue is a workload,
not a promise that its first 24 reviews will pass. Initial references fetched
for inventor ranks 24 and 26 are retained as unadjudicated leads, outside the
83 completed pilot attempts. No new top-up pair is accepted from those leads.

`correction_proposals.json` records **no newly established error in an admitted
atomic source row**, hence no new affected source/negation pair and no required
advance of the v2 correction base from this audit. This does not certify every
remaining P label. The historical corrections/quarantines are unchanged;
mac 2 retains ownership of adoption/cache-mapping and sensitivity records.

## Reproduction and validation

```sh
/tmp/clean-extraction-venv/bin/python -B scripts/49_build_capacity_pilot_audit.py
PYTHONDONTWRITEBYTECODE=1 /tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_capacity_pilot_audit.py tests/test_selection_repair_candidate_overlay.py tests/test_selection_repair_source_closure.py tests/test_selection_repair_source_audit.py data/clean_protocol/selection_repair_v1/source_review_v1/test_validation.py
/tmp/clean-extraction-venv/bin/python -B scripts/check_selection_repair_local_preservation.py --root /Users/apple/projects/truth-probing-extensions
git diff --check
```

The default command checks byte-identical reproduction. `--write` can create
missing derived outputs but refuses to overwrite differing existing files.
Judgments/evidence are editorial inputs; reproduction does not generate new
factual judgments or make network requests. The validator rebuilds the fixed
order and recovered negative queues and verifies hashes, exact evidence/fact
bindings, source-pair status symmetry, historical judgment reuse, sequential
stopping, nested fallback, whole-person exclusions and actual sampler counts.
Tests reject cross-word positives, changed pipeline ranks, wrong evidence,
missing evidence, promoted unresolved pairs, skipped candidates, substituted
negations and broken paired source handling. The local-only 13-file preservation
check is separate from portable tests.

Focused validation passed **82 tests plus 17 subtests**. All 13 unrelated
untracked files match their historical hashes. The input lock checks all
previously tracked data, configuration and documentation files without
rewriting them. No production compounds, activation access or probe fitting
are part of validation.
