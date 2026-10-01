# Section 9 held-out AND wording: D and E preparation

The October plan **did specify** “Both of the following are true: A. B.” in
Section 9. The earlier E-v1 report incorrectly described this wording as
undefined. This successor implements the user-supplied exact Section 9 definition;
the historical E-v1 report, code, configuration, manifests and outputs remain
unchanged for reproducibility.

The frozen specification is `config/clean_protocol/heldout_and_wording_v1.json`.
Its distinct condition ID is `and_both_following_v1`, with this exact renderer:

```python
"Both of the following are true: " + first + " " + second
```

`first` and `second` are the complete audited constituent sentences in the
existing AB or BA surface order. The renderer preserves their capitalization,
punctuation and text. It adds no quotation marks, clause labels, periods or facts.
`src/heldout_and_wording.py` is a pure standard-library module, shared identically
by D and E. Specification provenance records the authoritative user-supplied
Section 9 quotation and exact renderer; it does not claim an additional locally
available plan document was read.

This condition is explicitly **held out from compound fitting and selection**.
All packages retain `final_evaluation_enabled=false` and `P_A_frozen=false`.
No extraction, fitting, scoring or prediction inspection occurred.

## Coverage

| Topic | D entities | D pairs | D new AND rows | E entities | E pairs | E new AND rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Animals | 16 | 32 | 256 | 16 | 32 | 256 |
| Cities | 149 | 298 | 2,384 | 146 | 292 | 2,336 |
| Elements | 18 | 36 | 288 | 18 | 36 | 288 |
| Inventors | 33 | 49 | 392 | 11 | 22 | 176 |
| Spanish | 34 | 68 | 544 | 34 | 68 | 544 |
| **Total** | **250** | **483** | **3,864** | **225** | **450** | **3,600** |

Each retained pair contributes four truth cells × two surface orders. AND labels,
pair IDs, constituent fact IDs, canonical truths and AB/BA surface identities are
unchanged. New wording example IDs bind the condition, split, original bare
example ID and exact rendered text. No pairs or facts are replaced or regenerated.

The counts are data counts, not independent statistical sample sizes. Rows share
pairs, entities and facts; wording conditions share the original bare identities.

## Exact D cohort and factual eligibility

Five authoritative metadata-only inputs were recovered locally, verified against
the reviewed v4 preparation receipts and copied immutably into
`data/clean_protocol/heldout_and_inputs_v1/`:

- original Qwen D compound metadata;
- recovered isolated facts and constituent map;
- v4 development fact eligibility and pair eligibility tables.

`recovery_receipt.json` records the original locations, exact hashes, byte sizes
and authoritative receipts. Only these metadata files were read from the
recovered archives/preparation directory. No activation, probe or prediction
files were loaded. Reproduction now uses the checked-in copies and has no
external D metadata dependency.

The new adapter reproduces the reviewed v4 `current_fact_eligibility` contract
using `tc_successor_status` and `tc_current_eligible`, then requires exact equality
to the hash-pinned v4 fact and pair tables. A positive needs an admitted exact
source claim. A negative needs its exact eligible registry entry; source labels
alone do not accept a generated negative. Any restricted or conflicting exact
source claim vetoes eligibility. The original constituent map and fact keys must
match every original row before filtering.

The retained cohort is exactly **483 of the original 524 pairs**. A pair survives
only if all four constituent facts pass; all 16 original bare variants stay
together. No fresh degree-four graph is constructed. In particular, the retained
inventor graph is the reviewed 49-pair subset, not a new degree-four graph on
33 inventors. D evidence retains the accepted v4 limitation: retained source-label
support is not newly represented as independent human verification.

## Packages and operator reuse

Both packages are under
`data/clean_protocol/compounds/entity_disjoint/`:

- `final_e_preparation_v2/`: complete E successor, including the new
  `and_both_following_v1.csv`. Existing `bare_compounds.csv`, OR-only
  `wording_compounds.csv`, pair/degrees/order files, isolated facts, evidence and
  admitted atomic E rows are copied byte-for-byte from E v1. The predecessor
  manifest remains explicitly historical as `predecessor_manifest.json`.
- `development_heldout_and_v1/`: the exact retained 7,728 bare D rows as metadata,
  the 3,864 new AND wording rows, constituent records and v4 eligibility tables.

Each package includes a frozen wording specification, preparation configuration,
per-topic counts, condition index, reversible extraction inventory, preservation
comparison and manifest with input/code/output hashes.

| Condition | Changed operator | Reused bare operator |
| --- | --- | --- |
| `bare` | Neither | AND and OR |
| `or_explicit_or_both_v1` | OR | AND |
| `or_at_least_one_v1` | OR | AND |
| `and_both_following_v1` | AND | OR |

E's condition index preserves its original three full 7,200-row conditions and
adds the AND condition, also 7,200 rows. D's new condition index covers its exact
7,728 retained bare identities. Reused bare OR rows are references, not new
wording or new extraction requests. These are separate one-operator conditions;
no unrequested joint AND+OR wording condition is introduced.

## Extraction inventory

| Inventory scope | Logical bindings | Unique exact texts | Duplicate bindings |
| --- | ---: | ---: | ---: |
| Complete E v2 | 19,424 | 19,190 | 234 |
| New D AND wording only | 3,864 | 3,864 | 0 |

The E total is 7,200 bare + 7,200 existing OR wordings + 3,600 new AND wordings +
450 isolated facts + 974 atomic rows. Every old E logical binding is preserved;
new rows append reversible references. Counts and savings are computed from exact
text hashes, not hard-coded. D's inventory covers the new wording only; original
bare rows remain available through the condition index for reuse.

Each binding resolves to an artifact and zero-based record index, exact logical
ID, text hash, pair and constituent IDs. Source-row and text identities remain
distinct. Deduplication is valid only within an identical future model, tokenizer,
prompt and representation contract. No actual cache reuse is performed here.

The missing-template issue is now resolved and **no authoritative D data input is
missing**. Concrete future extraction dependencies remain the separately
authorized extraction entry point, a pinned model/tokenizer/representation
contract, and the required fresh-extraction compatibility checks. The existing
development-only extractor is not silently made E-capable. Extraction was not
launched; final scoring stays closed.

## Integration, preservation and validation

The local session was Darwin in the mac 1 repository on `clean-eval-protocol`, at
reviewed HEAD `9ccc996efcb687365e3fd8ee977c143e538789f2`. Fetch found origin
synchronized. The exact reviewed commit
`de776f1c86e4ddc464f79feac391704cbbaf0dc0` was merged normally as `509daf8`.
No subsequent P10 work was merged, and no other worktree was modified.

All E-v1 hash-bound files remain unchanged. Train/D/A/T_C memberships and actual
sampler selections are unchanged from the reviewed v4/v5 inputs. E continues to
use v5 admission and the same 225 completed pairs. D retains the v4 483-pair
contract. No correction-base advance or downstream fit invalidation is introduced.
The 13 unrelated untracked files remain byte-identical.

The focused suite passed **49 tests and 50 subtests**. It covers exact specification
rendering, both splits, all truth/order cells, original AND labels and identities,
wrong-operator/text/label rejection, ineligible fact rejection, exact v4 eligibility
reconstruction, unique wording IDs, reversible extraction bindings, deterministic
regeneration and immutable-write preflight. A guarded subprocess verifies the
adapter imports no Torch/Transformers/sklearn and reads no activation, probe or
prediction data. The pure renderer itself imports no NumPy or pandas either.

```sh
/tmp/clean-extraction-venv/bin/python scripts/58_prepare_heldout_and_wording.py --write
/tmp/clean-extraction-venv/bin/python scripts/58_prepare_heldout_and_wording.py
/tmp/clean-extraction-venv/bin/python scripts/54_prepare_final_e_benchmark.py
/tmp/clean-extraction-venv/bin/python scripts/check_selection_repair_local_preservation.py --root "$PWD"
/tmp/clean-extraction-venv/bin/python -m pytest tests/test_heldout_and_wording.py tests/test_final_e_preparation.py tests/test_final_partition_fact_audit.py tests/test_clean_compounds.py tests/test_validation_compound_benchmark.py -q
```

`--write` creates missing files only after preflighting every output; it refuses
any differing existing bytes. Normal invocation verifies saved outputs exactly.
The separate `validation_results.json` in E v2 records executed checks and binds
the D/E manifests and this report.
