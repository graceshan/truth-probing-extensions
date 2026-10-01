# Machine-readable generalization protocols

`config/clean_protocol/generalization_protocols.json` defines three protocol
families and seven named regimes. `src/generalization_protocols.py` resolves and
validates their data-exposure contracts against the shared manifest, pinned by
hash and version. It never creates entity assignments, compounds, probes, or
metrics. These are definitions for future runs; no new experiment is instantiated.

| Regime | Atomic fitting | Layer/C selection | Compound fitting / calibration | Development evaluation | Future final evaluation (disabled) |
| --- | --- | --- | --- | --- | --- |
| `new_pairings` | All topics, train entities | All topics, validation entities | All topics, train entities; separate pair groups | All topics, train entities; unseen pairs | Train entities; separately reserved unseen pairs |
| `entity_disjoint` | All topics, train entities | All topics, validation entities | All topics, train entities; separate pair groups | All topics, validation entities | All topics, test entities |
| `holdout_cities` | Other four topics, train | Other four topics, validation | Other four topics, train | Cities, validation | Cities, test |
| `holdout_sp_en_trans` | Other four topics, train | Other four topics, validation | Other four topics, train | Spanish translation, validation | Spanish translation, test |
| `holdout_inventors` | Other four topics, train | Other four topics, validation | Other four topics, train | Inventors, validation | Inventors, test |
| `holdout_element_symb` | Other four topics, train | Other four topics, validation | Other four topics, train | Element symbols, validation | Element symbols, test |
| `holdout_animal_class` | Other four topics, train | Other four topics, validation | Other four topics, train | Animal classes, validation | Animal classes, test |

All scopes are permissions, not a requirement to use every eligible entity.
Calibration is restricted to train entities in this version. It uses separate
pair groups from compound fitting and evaluation. Compound fitting and calibration
may be absent; their ledger roles must then be explicit empty lists.

## Claims and separation

**New pairings is pair-disjoint.** It asks whether new combinations generalize
when their constituents may already have appeared individually. Evaluation
entities may overlap atomic fitting, compound fitting, and calibration. The
unordered evaluation pair IDs must not overlap compound fitting or calibration.
The train entity pool is deliberately reused; pair-role assignments do not change
the shared entity split. All variants of a pair belong to exactly one compound
role, including across different operators, orders, truth cells, and templates.
The validator checks this grouping, rather than assuming a fixed variant count.

**Entity disjoint** asks whether an atomic-trained readout transfers to compounds
of entities absent from atomic fitting. No evaluation entity may occur in atomic
fitting, compound fitting, or calibration. Existing topic-scoped IDs and manifest
partitions enforce this. Development evaluation uses validation entities and may
therefore share entities with atomic layer/C validation; it is a development
diagnostic, not an untouched final estimate. Future final evaluation uses test
entities, which are absent from every fitting and selection role.

**Strict leave-one-topic-out** excludes the held-out topic from all four
development roles, including atomic layer/C selection. Evaluation is restricted
to that topic. Held-out labels must not affect layer/C, model, method, template,
or calibration choices. Repeatedly adapting a method to the held-out evaluation
would invalidate the strict claim even if the immediate fitting ledger passed.
Every strict regime needs its own fitting and selection run on the other four
topics. The frozen all-topic clean LR baseline is not a strict-topic readout and
is preserved unchanged.

**Compound-only topic holdout** can use the held-out topic during atomic fitting
or layer/C selection and only omit it from compound training. That is a different
claim and is rejected by these strict regimes. No such regime is labeled strict
in the configuration.

The earlier entity-first requirement continues to govern the identity and pair
construction layer. The explicitly named `new_pairings` control reuses train
entities and makes only the narrower pair-disjoint claim.

## Exposure API and checks

Run the definition/manifest check (prints JSON, writes nothing):

```sh
python3 -B scripts/22_check_generalization_protocols.py
```

Validate a complete, label-free **development** exposure ledger:

```sh
python3 -B scripts/22_check_generalization_protocols.py \
  --regime holdout_cities --exposures path/to/exposure_metadata.json
```

The ledger is a JSON object with all five keys: `atomic_probe_fitting`,
`layer_C_selection`, `compound_fitting`, `calibration`, `evaluation`.
Each value is a list. Atomic records have `topic`, `split`, `entity_id`.
Compound records have `topic`, `split`, `entity_a_id`, `entity_b_id`, `pair_id`,
and optionally `example_id` for checking variant-row exposure. The records are
metadata projections; statements, labels, activations, and scores are rejected.
Atomic records may include `compound_usable=False` entities. Compound records
must contain two distinct usable entities from the same topic and partition.

The validator checks every entity against the stored manifest, recomputes the
existing unordered pair ID, checks allowed topics and splits, and checks pair
ownership across all compound roles. Entity-disjoint regimes additionally check
evaluation entity-set intersections. It returns counts and deterministic
protocol, manifest, and exposure hashes. Deliberate violations raise `ValueError`.

The ledger must include **all actual exposure**, including earlier tuning and
reused readout provenance. The validator cannot discover omitted history or prove
that an unreported label was never inspected. Successful validation does not
authorize generation or scoring. Future experiment runners must supply and check
the ledger before fitting or evaluation; existing runners are unchanged.

## Final-test lock

Final scopes are machine-readable design declarations only.
`final_evaluation_enabled` is false; final exposure validation is rejected, and
the CLI offers no final-evaluation flag. No final examples, facts, or pair lists
are instantiated by the summary command. It reads only the protocol definition
and shared entity manifest.

For new pairings, eventual final pairs must also be unseen during development
evaluation and every method-selection step. That separate final pair reservation
and its history checks will be implemented only after the experiment suite is
locked. Existing development pairs must never become an untouched final set.

Run identity-only synthetic leakage tests:

```sh
python3 -B -m unittest discover -s tests -p 'test_generalization_protocols.py' -v
```

Tests cover all seven regimes, pair reuse across roles, variant grouping, entity
leakage, held-out topic exposure in every development role, compound-only versus
strict topic holdout, forged identities/splits/pair IDs, cross-split pairs,
unusable compound entities, duplicate manifest entities, forbidden payload fields,
determinism, and the final-evaluation lock. No final evaluation examples or test
decision scores are needed.
