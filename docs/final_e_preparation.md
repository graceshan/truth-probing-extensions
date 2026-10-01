# Production E data preparation v1

This package adopts the reviewed v5 score-independent E exclusions and prepares
benchmark **data only** from the 225 completed same-entity true/false fact pairs.
`candidate_overlay_v5/e_successor_status` and `e_current_eligible` are authoritative;
`final_partition_projection_v1` supplies current membership. Historical status
columns are not interpreted as current. No unresolved case is reopened and no
fact is regenerated, replaced or chosen using scores. The original audit,
including its 36 inventory exclusions, 35 unresolved entities, 261 attempted
negative candidates and one rejected candidate, remains unchanged and hash-bound.

`final_evaluation_enabled=false`; P/A remains unfrozen. The data-preparation CLI
has no model, probe, activation, prediction, fitting or scoring operation. It does
not enable or bypass the recovered evaluation or negative-registry guards. No
extraction was launched. The correction base remains v5.

## Recipe and exact statements

`final-e-degree4-ring-v1` is the separately versioned test-data adapter of
`validation-degree4-ring-v1`. It retains the original hash namespace intentionally:
order entities by SHA256 of compact UTF-8 JSON
`["validation-degree4-ring-v1", 0, topic, "test", entity_id]`, then entity ID as a
tie-breaker. Connect circular offsets 1 and 2, canonicalize endpoints in lexical
entity-ID order, and sort the unique edges. Every admitted compound entity has
degree four. Pair IDs reuse the recovered `pair-v1` identity, independent of
surface order; they cannot overlap D pairs because entity/person sets are disjoint.

Each pair emits all four canonical truth cells × AND/OR × AB/BA: 16 rows.
The existing `clean-binary-v1` renderer and Boolean semantics are reused. The
canonical A/B facts remain fixed when surface order changes. Period removal and
lowercasing a second-clause leading `The` are exactly the established renderer's
behavior; the full audited constituent sentences remain unchanged in their own
columns and in `isolated_facts.json`.

Every selected fact carries its exact text/hash, entity, reviewed person key,
label, judgment, evidence IDs, original source or negative-candidate reference,
and provenance. Positive `fact-v1` IDs are derived from those exact records;
negative IDs must equal the audited IDs. `constituent_evidence.json` copies the
selected evidence records without changing them. A source reference and all
alternate exact statement/person references must remain eligible, including the
paired source negation. Only completed pairs enter compound generation.

## Counts for the precision check

| Topic | Entities | Unordered pairs | Bare rows | Each OR wording | Isolated facts | Atomic E rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Animals | 16 | 32 | 512 | 256 | 32 | 64 |
| Cities | 146 | 292 | 4,672 | 2,336 | 292 | 590 |
| Elements | 18 | 36 | 576 | 288 | 36 | 72 |
| Inventors | 11 | 22 | 352 | 176 | 22 | 110 |
| Spanish | 34 | 68 | 1,088 | 544 | 68 | 138 |
| **Total** | **225** | **450** | **7,200** | **3,600** | **450** | **974** |

These are data counts, **not independent statistical sample sizes**. Sixteen bare
rows share each pair; four pairs share each entity; wording conditions reuse the
same facts and pairs. Atomic E membership is a separate v5-admitted inventory,
not a claim that every atomic entity has an independently completed compound pair.

## Established wording and remaining definition gap

The exact definitions are recovered from
`src/priority2_input_controls.py:TEMPLATES` and `variants`:

- `or_explicit_or_both_v1`:
  `render_binary(first, second, 'OR').removesuffix('.') + ', or both.'`
- `or_at_least_one_v1`:
  `'At least one of the following is true: ' + first + ' ' + second`

Both retain the original surface order and canonical truth cell. Each produces
3,600 OR rows. `condition_index.csv` maps each of the three full conditions to
7,200 rows: bare uses its original table; each wording condition reuses the exact
3,600 bare AND rows and selects its 3,600 new OR rows. This reuse does not create
additional extraction requests. A test evaluates the recovered pure rendering
expression as an independent oracle for every generated wording row, without
importing its scoring/cache dependencies into production preparation.

The reviewed repository contains **no distinct held-out AND renderer or any
additional October 2–3 wording definition**. Such wording remains pending an exact
definition; none is invented. The two generated OR controls are established
wording controls, not a claim that new previously unseen template definitions
have been supplied. Juxtaposition and XOR are not added to this binary benchmark.

## Extraction inventory and readiness

Artifacts live at
`data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v1/`:

- `bare_compounds.csv`, `wording_compounds.csv`, `condition_index.csv`;
- `pairs.csv`, `entity_degrees.csv`, `pair_hash_order.json`, `counts.csv`;
- `isolated_facts.json`, `constituent_evidence.json`, `admitted_atomic_E.csv`;
- `extraction_bindings.csv`, `extraction_texts.csv`;
- `configuration.json`, `non_E_preservation.json`, `manifest.json`.

There are **15,824 logical extraction items**: 7,200 bare + 7,200 new wording +
450 isolated + 974 atomic. They map reversibly to **15,590 unique exact texts**,
saving 234 repeated text computations under an identical future representation
contract. Each binding retains its kind, logical identity, source-row identity or
fact/pair references, artifact and zero-based data-record index. CSV headers do
not count as records. Text IDs are exact UTF-8 SHA256 identities. They never replace
source-row IDs: identical text can have distinct entity, label or provenance
bindings, which must remain separate downstream.

Deduplication is valid only within the same model revision, tokenizer,
representation, prompt and readout contract. No activation cache is selected,
matched or reused here. Bare, established OR wording, isolated and atomic input
data are prepared and validated for a separately authorized extraction workflow.
This inventory does not make the existing development-only extractor E-capable,
and it does not open final scoring. Additional undefined wording remains a gap.

## Preservation and downstream handoff

The local session is Darwin on `clean-eval-protocol`, starting at reviewed commit
`bb8bc51a864a5702f06ef23f923e11af97710c09`. Fetch found origin synchronized; no pull
or merge was needed. All 13 unrelated untracked files remain byte-identical.
Only this versioned adapter, config, tests, report and generated package are new.

The adapter explicitly compares source semantic identities, labels, statements,
person keys, partitions and admission between v4 and v5 for train/D. It verifies
current projected membership and compares actual P15/P10/train balanced selections
with the v4 projection. A15, A10, T_C and the original outer-training reference
are byte-identical. E person keys are disjoint from every train/D person and
from A/T_C. Train remains 3,040 admitted rows, D 1,012, P15 2,778 (700 selected),
P10 2,864 (800 selected), and train balanced selection 1,000.

**No non-E fitting or evaluation inputs change in this task.** V4/v5-bound train/D
fits remain reusable. The historical imported sensitivity package was bound to
v3; mac 2 still owns the already-required v4 adoption/cache-mapping decisions.
This data-only adapter introduces no further invalidation and changes no mac 2
adoption or sensitivity artifact.

## Validation and reproduction

```sh
/tmp/clean-extraction-venv/bin/python scripts/54_prepare_final_e_benchmark.py --write
/tmp/clean-extraction-venv/bin/python scripts/54_prepare_final_e_benchmark.py
/tmp/clean-extraction-venv/bin/python scripts/53_audit_final_partition_facts.py
/tmp/clean-extraction-venv/bin/python scripts/check_selection_repair_local_preservation.py --root "$PWD"
/tmp/clean-extraction-venv/bin/python -m pytest tests/test_final_e_preparation.py tests/test_final_partition_fact_audit.py tests/test_tc_topup_audit.py tests/test_capacity_pilot_audit.py tests/test_capacity_pilot_projection.py tests/test_selection_repair_candidate_overlay.py tests/test_selection_repair_source_closure.py tests/test_selection_repair_source_audit.py tests/test_validated_negatives.py tests/test_inventor_country_semantics.py tests/test_clean_compounds.py tests/test_validation_compound_benchmark.py -q
```

`--write` creates missing files only and preflights the entire package before any
write; differing existing bytes are refused. Normal invocation verifies all saved
bytes. The config pins input hashes; the generated manifest binds config, input,
producer/dependency/test code and every generated output. Historical packages are
never regenerated against a different base.

The focused suite passed **148 tests and 85 subtests**. Saved data regenerated
byte-for-byte; the historical E audit also regenerated successfully. All 13
unrelated-file hashes matched.

Tests cover exact evidence/fact identity, unresolved constituents, wrong
split/person binding, restricted alternate source references, paired quarantine,
wrong Boolean labels, changed surface order/text, unique IDs, full truth cells,
degree four, exact established wording, reversible text bindings, lossless
serialization, immutable-write preflight, and deterministic generation. A guarded
subprocess verifies preparation reads no activation/prediction artifacts and
imports no Torch, Transformers or sklearn. Existing test-scoring guards remain
closed. Validation outcomes are recorded in `validation_results.json` alongside
the generated package.
