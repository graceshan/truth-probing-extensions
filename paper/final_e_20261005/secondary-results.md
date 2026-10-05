# Secondary-results inventory

All frozen rows remain authoritative, including unfavorable and inconclusive outcomes. The [machine-readable catalog](evidence/secondary-catalog.json) covers every source row by scientific group, retaining model/head, arms, conditions, endpoints, sample, topic, scope and order coverage. Counts describe dependent report rows, not independent findings. The [exact extracted rows](evidence/extracted-rows.json) include all displayed rows, every sample-mean pooled/macro scalar contrast, and all unfavorable/uncertain constituent classifications. No new metric or interval is computed.

- [scalar-metrics: complete source table](/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/scientific-evaluation-20261005-v1/evaluation/scalar-metrics.json) — 28,560 rows; SHA-256 `c38e88da29dda7ec3a6b32dc34a79ccb3056f03c8d8deb451a3bfa519293dcdc`.
- [scalar-contrasts: complete source table](/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/scientific-evaluation-20261005-v1/evaluation/scalar-contrasts.json) — 38,556 rows; SHA-256 `87a2be3006ed666483a0765bbc16aff92be807063e3ed605033a95d0a9032b5c`.
- [constituent-metrics: complete source table](/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/scientific-evaluation-20261005-v1/evaluation/constituent-metrics.json) — 9,450 rows; SHA-256 `4c75010da3d09adcf6e9e019173d6880344bfa561deab5b9c7d4c69cc22a7c1e`.
- [constituent-contrasts: complete source table](/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/scientific-evaluation-20261005-v1/evaluation/constituent-contrasts.json) — 1,050 rows; SHA-256 `b0dac84c7e1b7ec20c857734a87f46827ebf7ec99034c15da7a1caef87b5c883`.
- [wording-paired: complete source table](/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/scientific-evaluation-20261005-v1/evaluation/wording-paired.json) — 6,132 rows; SHA-256 `51e77c17d7459304530eb0fb50e68e883e1708d466f551b5ed06480142c4e0b8`.

[Wording table](tables/wording.pdf): all six frozen R_all heads, both OR wordings, pooled OR boundary; wording minus bare. Every point is negative. Eleven of twelve intervals are wholly negative; Llama sample 37, at-least-one wording, crosses zero. This is not wording invariance. The complete source also retains AND wording, all other heads, macro and topic rows, and unchanged-operator comparisons.

[Constituent table](tables/constituent.pdf): all six constituent AUROC endpoints for both models, AND-to-OR transfer at the saved source-selected layer, explicit-or-both wording, pooled, all surface orders. Positive target-minus-source differences mean worse transfer. Some exceed the frozen .05 deficit criterion; this precludes an unqualified transfer-success claim. The complete sources retain both directions, every wording, topic, AB/BA order, marginal/conditional metrics, fixed-threshold decisions, joint correctness, Boolean composition and min/max ranking.

[Separate full-bank table](tables/secondary.pdf): S_full minus S_atom_full and R_all minus S_atom_full retain their own evidence roles. D-selected references remain descriptive; they are not E-selected maxima.

Frozen scalar comparisons (complete conditions, metrics, samples and topic/order scope live in the linked catalog):

**b25**

- `R_all − S_all`
- `S_all − S_atom_all`
- `R_all − atomic_only_at_R_all`
- `R_all − R0_at_R_all`
- `R_fixed − S_fixed`
- `R_fixed − atomic_only_fixed`
- `R_fixed − R0_fixed`
- `repair_at_S_all − S_all`
- `S_all − reduced_LR`
- `R_all − reduced_LR`
- `S_fixed − reduced_LR`
- `R_fixed − reduced_LR`
- `LR_only_selection − S_all`
- `S_all − original_atomic_selected_R0`
- `R_all − bank_oracle`
- `C_clean_fixed − S_all`
- `C_clean_D_selected − S_all`
- `C_clean_at_S_all − S_all`
- `C_clean_at_R_all − R_all`
- `C_clean_at_S_atom_all − S_atom_all`
- `C_clean_D_selected − reduced_LR`
- `C_clean_fixed − reduced_LR`
- `C_clean_at_S_all − reduced_LR`
- `C_clean_at_R_all − reduced_LR`
- `R0_at_R_all − reduced_LR`
- `S_atom_all − reduced_LR`
- `original_atomic_selected_R0 − reduced_LR`
- `atomic_only_fixed − reduced_LR`
- `atomic_only_at_R_all − reduced_LR`
- `R_all_equal_boundary_sensitivity − S_all_equal_boundary_sensitivity`

**full_bank**

- `R_all − S_full`
- `S_full − S_atom_full`
- `R_all − S_atom_full`
- `S_full − S_all`
- `S_full − reduced_LR`
- `S_atom_full − reduced_LR`
- `R_all − reduced_LR`
- `full_canonical_LR − reduced_LR`
- `full_bank_eligible − reduced_LR`
- `full_bank_unrestricted − reduced_LR`

Controls include fixed-layer selection/repair, LR-only selection, repair at the selected atomic layer, R0 and atomic-only adaptation, C_clean and the D-selected bank references. These cannot establish that the atomic-loss term causes retention or that information lies outside the collective bank span. Equal-boundary sensitivity does not replace the original pooled primary. No E-driven comparator search occurred.

Existing constituent-metrics classifications: `lower_bound_at_least_90`: 1,149 rows; `point_at_least_90_bound_uncertain`: 2,458 rows; `point_below_90`: 173 rows.

Existing constituent-contrasts classifications: `material_deficit_lower_bound_above_05`: 20 rows; `unresolved_crosses_05`: 60 rows; `upper_bound_below_05`: 340 rows.

Negative points in the complete sources: 8,049 scalar-contrasts rows, 217 constituent-contrasts rows, 2,475 wording-paired rows. These signs alone are not significance or failure classifications. Classification fields are absent where the frozen report did not define them; absence is not failure. AB/BA constituent records can be point-only; no interval is invented.
