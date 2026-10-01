# T_C top-up audit and v4 correction projection

The frozen queue produced 24 new completed same-entity affirmative fact pairs
from 27 attempts. Combined with the 76 preserved pairs, T_C now has **20 usable
entities per topic**, or **190 possible unordered pairs per topic** before the
planned 100/topic control cap. No compounds were generated. A15 remains 75 keys
and its nested A10 remains 50; P/A is unfrozen and this does not commit to B=50.

**Correction handoff for mac 2:** advance from `candidate_overlay_v3` to
`candidate_overlay_v4` and use `capacity_pilot_projection_v3` before any later
authorized fitting. The authoritative source status is `tc_successor_status`;
the authoritative fact-eligibility field is `tc_current_eligible`. Earlier status
and eligibility columns are preserved historical references. No adoption,
cache-mapping or sensitivity record was edited.

## New source restrictions

| Exact affirmative | Source and paired negation | Judgment and action |
| --- | --- | --- |
| Henri Giffard lived in the U.S. | `inventors:155`, `neg_inventors:155` | Unresolved: recovered biography gives French schooling and invention milestones, including a London episode, but insufficient residential chronology to support US nonresidence. Quarantine both rows. |
| Nikolay Slavyanov lived in Russia. | `inventors:380`, `neg_inventors:380` | Unresolved: study and employment occurred in the Russian Empire; the bare source object does not resolve historical-polity versus successor semantics. Quarantine both rows without modern-border substitution. |

No new confirmed source-label error was established. Original labels, identities,
partitions and every earlier overlay byte remain unchanged. These are exact-claim
restrictions, not person-wide bans: Giffard's France pair remains admitted.
The registry duplicate of Giffard's US claim,
`fact_19cfa5eef580ef4604ebd25d7bb0b7346307c99c892a38382b9dec0e14c8cf24`,
is also ineligible. Exact identity/statement-hash matching applies the same policy
to any alternate source or registry reference. Walton's generated Poland claim
remains unresolved and unusable; it causes no additional source restriction.

## Coverage and traversal

`tc_topup_audit_v1/pre_judgment_lock.json` was written before new judgments.
It binds the current v3 base, the frozen projection-v2 queue, 119 tracked audit
inputs and 13 unrelated untracked-file hashes. The earlier input locks remain
intact and bind their own historical dependencies. `traversal_queue.csv` is a
byte-identical copy of the supplied queue. No RNG was rerun: this retains the
original CPython MT19937 ordering seeded with 20261002 from canonical sorted
person keys, including reviewed inventor aliases and original outer partitions.

The rule is lexical topic order and increasing original rank, skipping all prior
attempts and completed keys. Each unreviewed candidate uses its exact queued
positive and negative; stop immediately at 20 completed keys per topic. No
candidate was redrawn or given a replacement fact. All eight prior unresolved
attempts remain recorded and were not reopened. Maxwell's previously recovered
rank-45 pair remains part of the original 76, not a new attempt.

| Topic | New attempts | New completed | New unresolved | New rejected | Final usable | Possible pairs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Animals | 5 | 5 | 0 | 0 | 20 | 190 |
| Cities | 5 | 5 | 0 | 0 | 20 | 190 |
| Elements | 5 | 5 | 0 | 0 | 20 | 190 |
| Inventors | 7 | 4 | 3 | 0 | 20 | 190 |
| Spanish | 5 | 5 | 0 | 0 | 20 | 190 |
| Total | 27 | 24 | 3 | 0 | 100 | 950 |

Cumulative accounting is 111 attempted entities: 100 completed, 11 unresolved,
zero rejected. Remaining successful top-ups toward 20/topic: **zero**. The
unresolved records are retained; capacity completion does not resolve them.

Animals, cities, elements and Spanish stop at rank 20. Inventors stop at rank 30:
Giffard (24) unresolved, Fleming (25) usable, Copeman (26) usable, Blanco (27)
usable, Walton (28) unresolved, Slavyanov (29) unresolved, Strauss (30) usable.
A preliminary Torvalds retrieval beyond the stop is logged as unadjudicated and
contributes no judgment or completed pair.

## Evidence and eligibility

Each pair contains exact affirmative statement text, original source or frozen
recovered-pipeline provenance, person/entity identity, statement SHA-256,
judgment, reasoning and explicit evidence bindings. Source labels and negations
are never treated as factual verification. Judgments are Codex-assisted external
reviews, not human certification, and remain defeasible.

The element reviews reuse the recovered NIH PubChem periodic-table response,
checking both the true symbol and the owner of the proposed false symbol.
Fleming reuses the reviewed colleague memoir that bounds the Boulogne posting;
his Scottish/London life chronology supports the new negative independently of
an anachronism-only reading of “Ancient Greece.” Copeman reuses the earlier
unadjudicated reference lead with fuller local/family biography. Walton's
previous US-posting evidence is preserved and reused without reopening that
historical judgment. Unsuccessful retrievals and discovery pages are logged and
provide no acceptance evidence.

Both translations for each Spanish headword were checked in Cambridge's
GLOBAL/PASSWORD entries and SpanishDictionary's editorial entries, including
polysemy, register, regional senses, exact spelling/diacritics and Spanish→English
direction. In particular, `papel` has paper/document/role/performance/share senses;
`papel de regalo` is wrapping paper and does not make bare `papel` mean “present.”
`presentar` in an example does not transfer its meaning to `papel`. No D/E fact or
positive from a different word was borrowed.

`combined_TC_completed_pairs.json` preserves the original 76 records as its
unchanged prefix, followed by the 24 completed top-ups. `combined_evidence.json`
preserves all original evidence entries unchanged. New unresolved facts and
paired restrictions propagate to source, registry, audited-fact and queue-status
records. A supported fact in an unresolved pair does not make that pair eligible.

## Current prospective fitting exposure

The v4 projection derives **3,040** admitted outer-training rows, **2,778** P15
rows and **2,864** P10 rows. Validation/test remain 1,012/1,002 admitted rows.
Every source variant of each proposed A person is removed from prospective P,
including aliases and negations. The original outer-training reference is
preserved byte-for-byte. T_C may overlap both A and P; each topic has 15 T_C keys
in A15 and five in prospective P15.

| Topic | Corrected train | P15 rows | P15 sampled | P15 sampled person keys | P10 rows | P10 sampled | P10 sampled person keys |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Animals | 200 | 140 | 140 | 37 | 160 | 160 | 42 |
| Cities | 1,800 | 1,740 | 140 | 68 | 1,760 | 160 | 74 |
| Elements | 228 | 168 | 140 | 42 | 188 | 160 | 46 |
| Inventors | 388 | 336 | 140 | 59 | 352 | 160 | 71 |
| Spanish | 424 | 394 | 140 | 69 | 404 | 160 | 79 |
| Total | 3,040 | 2,778 | 700 | 275 | 2,864 | 800 | 312 |

All listed memberships have equal label-0/label-1 and affirmative/negated counts.
Each topic's sampled exposure is 70/70 labels at A15 or 80/80 at A10. The actual
`balanced_burger_indices` sampler was rerun with seed 0 and original source order;
its selected membership changes even though totals stay 700/800. Per-topic rows,
labels, entities, person keys, source-variant exclusions and before/after sampled
IDs and hashes are recorded in the successor projection.

## Reproduction and regression checks

Run in the repository's Python environment with NumPy, pandas and pytest:

```sh
python scripts/51_complete_tc_topup_audit.py
python scripts/49_build_capacity_pilot_audit.py
python scripts/50_project_capacity_pilot_corrections.py
python -m pytest tests/test_tc_topup_audit.py tests/test_capacity_pilot_audit.py tests/test_capacity_pilot_projection.py tests/test_selection_repair_candidate_overlay.py tests/test_selection_repair_source_closure.py tests/test_selection_repair_source_audit.py -q
python scripts/check_selection_repair_local_preservation.py --root "$PWD"
```

The new builder compares every generated byte with the checked-in package;
`--write` only creates missing, matching outputs and refuses to overwrite differing
ones. Historical builders reproduce their original bases. Regression checks
inject new unresolved positive and negative source claims, a generated exact
source duplicate, confirmed label conflicts, changed identities/hashes/evidence,
rank skips, reopened attempts and reviews past the stopping point. Every current
fitting membership rejects restricted source pairs, and eligibility checks reject
registry bypasses. Tests also preserve A15/A10 bytes, all 76 earlier pairs and
evidence, original queue columns, balanced labels and deterministic counts.
