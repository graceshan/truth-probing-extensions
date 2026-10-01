# Final-partition fact audit and E-only correction v5

The score-blind, bounded E audit is complete against the locked queue. It yields
**225 completed same-entity affirmative true/false pairs**, with **35 unresolved
entities excluded from compound eligibility**. This is factual coverage, not a
final benchmark result. No production compounds, fitting, activations or
predictions were accessed or produced. `final_evaluation_enabled` remains false;
P/A remains unfrozen.

Mac 1 was Darwin at `fd2d7be3b910c3fdba7ff6b2c2891eb0ab1b7992` on
`clean-eval-protocol`. Fetch found origin synchronized, so no pull was needed.
The exact reviewed sensitivity commit
`4cae65c7dfc825f774226007a81994f5a08776c9` was merged normally as `61e2197`.
No later sensitivity-branch work was merged. Historical packages and all 13
unrelated untracked files are preserved.

## Current base and downstream handoff

Use `candidate_overlay_v5/e_successor_status` and `e_current_eligible` for the
successor. Earlier columns, including `tc_successor_status` and
`tc_current_eligible`, record the historical v4 state. The successor membership
projection is `final_partition_projection_v1`; the original
`capacity_pilot_projection_v3` and every earlier overlay remain unchanged.

**Only E admission changed: 28 rows, or 14 exact affirmative/negation pairs,
are newly quarantined. No source label was confirmed wrong.** Original labels,
statements, identities, reviewed person keys and partitions remain unchanged.
Quarantine is claim-specific, not a person-wide ban. For example, Jeffreys's
France claim is quarantined but his supported UK source pair remains admitted.

| Membership | v4 | v5 |
| --- | ---: | ---: |
| Admitted outer train | 3,040 | 3,040 |
| Admitted D / validation | 1,012 | 1,012 |
| Admitted E / test | 1,002 | 974 |
| Prospective P15 | 2,778 | 2,778 |
| P15 balanced selection | 700 | 700 |
| Prospective P10 | 2,864 | 2,864 |
| P10 balanced selection | 800 | 800 |
| Train balanced selection | 1,000 | 1,000 |

The projection explicitly compares semantic row signatures (identity, label,
exact text/hash, person key, partition and current admission), not just totals.
P membership and actual seed-0 balanced sampler selections match v4.
A15's 75 keys, nested A10's 50 keys and all 100 T_C completed pairs are copied
byte-for-byte. The original outer-training reference is also byte-identical.

**Fits and D evaluation inputs already bound to v4 remain reusable with respect
to this E-only audit.** The imported reviewed sensitivity package used v3, not
v4: v4 had already quarantined `inventors:155`, `neg_inventors:155`,
`inventors:380` and `neg_inventors:380` in training. This task does not silently
rebind those historical sensitivity fits. Mac 2 still owns the corresponding
adoption/cache-mapping and refresh decisions; v5 adds no further non-E change.

## Inventory, fixed policy and actual coverage

`final_partition_fact_audit_v1/pre_judgment_lock.json` was saved before new
judgments, binding the v4/projection-v3 inputs, original E inventory, ordered
entity and negative queues, source/config/producer dependencies and the 13
untracked-file hashes. The original E manifest contains 296 entities. Thirty-four
Spanish entities have no usable positive and two inventor positives were already
quarantined (John von Neumann and Alexander Graham Bell); they remain explicitly
inventoried and excluded. The new audit visits all 260 eligible entities.

This separate factual path preserves the recovered development-only registry and
final-scoring guards. It uses the recovered `negative-candidate-v1` hash ranking,
seed **0**, exact atomic templates and `inventor-single-country-v1` semantics.
There is no random redraw. Entities sort by topic then original entity ID, with
reviewed person-key and cross-partition checks. Positives are the lowest numeric
admitted affirmative positive source rows. The same-topic, same-E object pool is
formed from all recorded true objects of the v4 eligible entities; all target
known-true objects are removed. Inventor objects split into country components,
without inferred country aliases. Pool membership/source labels propose claims;
they do not verify those claims.

The frozen negative queue contains 9,101 candidates. For each entity, skip exact
previous restrictions and review the lowest-ranked eligible candidate. A
supported-true negative proposal is rejected and traversal advances to the next
rank. A supported-false candidate stops traversal; an unresolved candidate also
stops that entity, retaining its unresolved outcome rather than seeking an easier
replacement. A positive that remains unresolved cannot produce a completed pair.
There is no A or T_C quota on E, no reopening of old unresolved claims, and no
transfer of training/D facts to E.

| Topic | Original E entities | Inventory exclusions | Attempted entities | Negative attempts | Unresolved entities | Usable fact pairs | All unordered pairs | Degree-four recipe pairs | Expected binary rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Animals | 16 | 0 | 16 | 16 | 0 | 16 | 120 | 32 | 512 |
| Cities | 149 | 0 | 149 | 149 | 3 | 146 | 10,585 | 292 | 4,672 |
| Elements | 18 | 0 | 18 | 18 | 0 | 18 | 153 | 36 | 576 |
| Inventors | 45 | 2 | 43 | 44 | 32 | 11 | 55 | 22 | 352 |
| Spanish | 68 | 34 | 34 | 34 | 0 | 34 | 561 | 68 | 1,088 |
| Total | 296 | 36 | 260 | 261 | 35 | 225 | 11,474 | 450 | 7,200 |

There are 521 fact judgments, including 41 unresolved judgments. One proposed
negative is supported true and rejected: John Ericsson / England, rank 2.
His rank-1 US claim was already restricted and was not reopened. Rank 3,
Netherlands, is supported false using the sustained Sweden–England–New York
residential chronology and recovered historical home records. Every queue item,
including unattempted candidates, retains its original rank and current eligibility.

The recovered development recipe is `validation-degree4-ring-v1`, seed 0:
hash-order eligible entity IDs, connect circular offsets 1 and 2, and canonicalize
unordered endpoints. For n >= 5 this yields 2n edges, degree four everywhere;
four truth assignments × AND/OR × AB/BA would give 16 rows per edge. The capacity
checker substitutes `test` in the hash input and verifies this graph in memory.
It writes no production pair/example table and does not change the recovered
validation generator. **450 pairs / 7,200 rows is conditional capacity**, not an
already authorized final benchmark. A production test adapter/configuration and
acceptance of the unresolved exclusions still require a separate decision.

## Evidence and unresolved cases

Evidence includes exact statements and hashes, entity/person identity,
provenance, locatable excerpts, response hashes, reasoning and per-fact bindings.
The package reuses **27 evidence entries**: 18 row-specific selections from the
previously recovered NIH PubChem response, plus nine inventor reference records
(including a primary patent and historical biography). Reuse keeps the original
record and package hash; new fact bindings do not rewrite old evidence.

Cities use identified locality/country evidence, with spelling and disambiguation
follow-ups. The recovered city template expresses geographic location: urban
neighborhoods and administrative districts are described explicitly, without
claiming a new legal incorporation status. The unresolved cities are:

- **Shivaji Nagar:** the retrieved Bengaluru locality cannot uniquely bind the
  source's undifferentiated name among namesakes.
- **Kampung Baru Subang:** the retrieved Subang township record does not establish
  the exact named locality. Neither nearby-place evidence nor the source label
  supplies a completed fact pair.
- **Muzaffarabad:** the reference specifies Pakistan-administered disputed
  Kashmir; the bare positive does not distinguish administration from
  sovereignty. The independent Eritrea negative is supported, but cannot resolve
  the positive wording.

Animal judgments use the exact animal's classification. Element judgments verify
both the true symbol and the owner of the proposed false symbol in the recovered
PubChem table, with case-sensitive symbols. Spanish reviews inspect Cambridge
GLOBAL/PASSWORD and SpanishDictionary editorial senses for both proposed
translations, preserving spelling, diacritics, Spanish→English direction,
polysemy, regional/register usage and inflected senses. In particular, *huevo*
in *costar un huevo* does not mean bare “arm”; *lista* includes its noun senses
and the feminine adjective of *listo*; *plata* includes Latin American money;
none licenses the queued false translation. Corpus examples are distinguished
from dictionary senses.

Inventor acceptance requires explicit residence, settlement, or documented
residential study/work. Nationality, patent ownership, company location, a visit
or a biography's omission of a country is insufficient. Supported negatives use
sustained residential chronologies; historical-country negatives are not accepted
solely because the named state did not yet exist. The 32 unresolved inventor
cases retain their particular gaps in `unresolved_cases.json`; no successful
replacement is hidden behind the capacity counts. Six positive claims need
historical-polity clarification: Hargreaves, Newton, Gutenberg, Siemens, Schwarz
and Kay. Other unresolved cases primarily lack sufficient negative residence
coverage. New secondary-only judgments remain defeasible Codex-assisted reviews,
not human certification or universal proofs of nonresidence.

## Exact new source quarantines

Each ID below and **its corresponding `neg_` dataset ID with the same index**
is quarantined; all are in the test partition. The correction receipt enumerates
all 28 IDs explicitly and binds their original hashes.

| Affirmative source ID | Exact statement | Issue |
| --- | --- | --- |
| `cities:246` | The city of Kampung Baru Subang is in Malaysia. | Exact locality binding unresolved |
| `cities:874` | The city of Shivaji Nagar is in India. | Namesake binding unresolved |
| `cities:1266` | The city of Muzaffarabad is in Pakistan. | Administration/sovereignty wording unresolved |
| `inventors:43` | George Eastman lived in the U.K. | Residence chronology incomplete |
| `inventors:90` | Nikolai Basov lived in the U.S. | Residence chronology incomplete |
| `inventors:173` | Sir Isaac Newton lived in the U.K. | Historical polity unresolved |
| `inventors:183` | Alec Jeffreys lived in France. | Residence chronology incomplete |
| `inventors:190` | Yoshiro Nakamatsu lived in the U.S. | Residence chronology incomplete |
| `inventors:197` | Johannes Gutenberg lived in Germany. | Historical polity unresolved |
| `inventors:210` | Grace Hopper lived in the U.K. | Residence chronology incomplete |
| `inventors:227` | Carl Wilhelm Siemens lived in Germany. | Historical polity unresolved |
| `inventors:315` | John Kay lived in the U.K. | Historical polity / identity-specific chronology |
| `inventors:333` | James Hargreaves lived in the U.K. | Historical polity unresolved |
| `inventors:334` | David Schwarz lived in Croatia. | Historical polity unresolved |

The successor checks exact topic/person/statement hashes and text against all
source and historical registry records. The 14 changed affirmative inventory
records are source-origin; no extra historical registry-origin record changes.
The complete E negative queue receives current eligibility separately, preventing
an alternate generated reference from bypassing a source restriction. Source
negations receive the same admission action; source labels are never rewritten.

## Reproduction and verification

The saved reports and manifests live under
`data/clean_protocol/selection_repair_v1/final_partition_fact_audit_v1/`.
`completed_E_fact_pairs.json` contains only completed pairs;
`original_E_inventory.json`, `negative_candidate_queue.json`, `reviews.json`,
`evidence.json` and `retrieval_log.json` retain the full audit trail.
`final_partition_projection_v1/semantic_comparison_to_v4.json` provides the
explicit non-E reuse comparison.

```sh
python scripts/53_audit_final_partition_facts.py
python scripts/51_complete_tc_topup_audit.py
python scripts/50_project_capacity_pilot_corrections.py
python scripts/49_build_capacity_pilot_audit.py
python scripts/check_selection_repair_local_preservation.py --root "$PWD"
python -m pytest tests/test_final_partition_fact_audit.py tests/test_tc_topup_audit.py tests/test_capacity_pilot_audit.py tests/test_capacity_pilot_projection.py tests/test_selection_repair_candidate_overlay.py tests/test_selection_repair_source_closure.py tests/test_selection_repair_source_audit.py tests/test_validated_negatives.py tests/test_inventor_country_semantics.py tests/test_clean_compounds.py tests/test_validation_compound_benchmark.py -q
```

The new builder verifies every output byte and input hash; `--write` creates
missing outputs only and refuses differing existing files. Regression checks
cover new unresolved positives/negatives, paired exclusion from every membership,
generated exact duplicates, registry eligibility, confirmed-conflict paired
actions, identity/hash/evidence mutations, queue skips and reopening, Spanish
sense/direction metadata, unchanged A/T_C and sampler membership, deterministic
ordering and complete-pair-only graph capacity. Existing final-exposure guards
remain active. All final production and evaluation decisions remain outstanding.
