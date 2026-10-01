# T1 historical development-results rebuild

This command evaluates the four recovered score families on the benchmark at
recovery commit `21996d9ba5b6b217a07a06d33c3d56db908c3b41`. Every result is labeled
**DEVELOPMENT, pre-label/alias-cleanup**. It applies no correction overlay, changes
no historical definitions, and performs no fitting, model inference, extraction,
or final-test inspection. Existing source evaluators and historical artifacts are
unchanged. The work belongs on `t1-score-rebuild-20261001`, not the main worktree.

## One command

Use Python 3.11+ with NumPy, pandas, scikit-learn (an existing import dependency),
and threadpoolctl. No torch, transformers, GPU, model weights, or activation tensors
are needed. The verified macOS run used `/tmp/clean-extraction-venv/bin/python`.
Exact numerical library versions and executed code hashes appear in the report.

```sh
/tmp/clean-extraction-venv/bin/python -B \
  scripts/46_rebuild_t1_development_results.py \
  --repo-root /Users/apple/projects/truth-probing-t1-score-rebuild \
  --payload-root /Users/apple/truth-probing-backups/20261001T164224Z/restore_t1_staging_20261001T164224Z_kgkt5sy7 \
  --output-root /Users/apple/projects/t1-development-results-20261001
```

Run from the rebuild worktree. Choose a **new** output directory outside both the
checkout and backup directory; existing output directories are rejected. Input
roots may be relocated explicitly. The packaged `original_path` values are lookup
keys in `inventory.json`, never paths accessed on the Mac. No provenance document
or recorded expected hash is rewritten. Output creation failure or invalid
required evidence stops the command; computation failures leave `failure.json`.

The inventory digest is pinned to
`836aeb833ba32ea65a3068823971a66280b71bdd81e448492e5baaa410e5c744`, established by the
prior restoration check against the independently verified archive. That check
found 253 payload files plus the inventory. The command rehashes all 253 files,
rejects missing/unexpected files and linked paths, and checks that all input bytes
remain unchanged after evaluation. It does not redownload or re-extract anything.

## Verification and evidence boundaries

| Family | Inventory source subtree under `truth-probing-artifacts` | Row scores |
|---|---|---|
| Qwen LR | `pinned_transfer_qwen25_v1/scoring` | `row_scores.csv` |
| Atomic methods | `pinned_method_transfer_qwen25_v1` | `method_scores.csv` |
| Input controls | `priority2_input_controls_v1/scoring` | `condition_scores.csv`, `isolated_scores.csv` |
| Llama | `llama31_replication_v1/analysis/scores` | `scores.csv` |

All four frozen specs must match their scoring and evaluation manifest hashes.
Available recorded file sizes and hashes, score schemas, finite values, ordered
IDs, condition and layer assignments, exact coverage, and one-to-one joins are
checked. A missing or extra ID never disappears through an inner join. The
recovered benchmark and packaged compound metadata must have identical recorded
bytes: SHA-256
`96e96515e780086065694155e3061bdfe6f4bb80af062a63c2deecd6bfad6f94`.
Its eight producer-code hashes must match the recovered checkout. Metadata checks
cover labels, Boolean semantics, 8,384 rows, 524 pairs, 262 entities, degree four,
five topics, 16 variants per pair, fact identity, and development-only scope.

Wording variants, isolated fact IDs and labels, and constituent mappings are
regenerated in memory with the recovered code and checked against the saved file
hashes. Isolated constituent joins bind fact ID, entity, topic, statement, and
truth. Min/max composition uses `min(s_A,s_B)` for AND and `max(s_A,s_B)` for OR;
Boolean composition uses the frozen `>= 0` threshold and receives no AUROC.
Juxtaposition remains a geometry diagnostic with no formal compound truth label.

Atomic validation scores per example are absent. The selected LR and method
validation metrics come only from verified `selection.json` / `summary.json`
records, cross-checked against `validation_metrics.csv`. They are explicitly
marked `saved_aggregate_only`, with no rebuilt atomic CI. Parameter archives
contain parameters, not a replacement for missing atomic row scores. No number
is taken from narrative text and no activation tensor is required to evaluate
finalized compound scores.

`verification.json` distinguishes available upstream hash identities from missing
upstream bytes. It does not claim to rerun extraction, smoke tests, fitting,
atomic selection, model/tokenizer execution, or parameter-to-score replay. The
seven excluded activation tensors and any other unavailable upstream dependencies
are reported. Recovered code/registry preservation by mac 1 is separate.

## Historical provenance conflicts

The packaged Qwen `representation_binding.json` records these incompatible
identities for `src/pinned_atomic_probes.py`:

- Working-file SHA-256: `6d1991fc639e2d664420122990cbb68653519184745cb24a812364af3d1e0466`
- Recorded producer and Git SHA-256: `0eef476667172ae8455bccdefc5d1ef7dde3a4549f88aaa7d4e80ac36f052918`

The rebuild records both, the current checkout hash, and an **unresolved** status.
It does not choose one as the implementation that executed historically. All
available scoring/evaluation source hashes are compared with the checkout;
differences, including the recovered benchmark generator versus older scoring
provenance, are retained separately from numerical discrepancies. Verifying
finalized score bytes does not establish complete upstream provenance.

## Bootstrap and surface diagnostics

The original pure CPU evaluators are reused for the full historical metric tables:
AND/OR AUROC, mixed-versus-FF OR boundary, threshold and geometry measures,
method comparisons, wording controls, juxtaposition, and constituent composition.
The frozen bootstrap is unchanged: 2,000 replicates, seed 1729, NumPy PCG64,
sorted topic/entity/pair order, endpoint resampling within each topic, and pair
weight equal to the product of endpoint multiplicities. Every condition uses
the same schedule. Contrasts subtract matched replicate values, never independent
intervals or independently resampled results.

Undefined replicates are retained as invalid, without redrawing or imputation.
A macro replicate requires all five topics. Linear pointwise percentile 95% CIs
require at least 1,800 valid replicates; every row reports total, valid, and invalid
counts and CI status. AUROC ties receive half credit. All four recorded schedule
hashes, saved pair weights, metric estimates, CIs, replicate validity, and numeric
draw arrays are compared. Numerical discrepancies beyond `1e-12` absolute
tolerance are listed; maximum differences and smaller floating-point differences
are also reported. Historical files are never overwritten.

The original Qwen bootstrap archive stores metric IDs as an object array. It is
never unpickled. Its numeric arrays are checked against the hash-verified companion
`bootstrap_summary.csv` column order, as declared by the historical writer; this
binding limitation is explicit in the comparison report. New archives use string
arrays and can be read with `allow_pickle=False`.

New diagnostics compute surface truth after ordering: AB uses `(a,b)`, BA uses
`(b,a)`. Thus canonical TF becomes surface FT under BA, and vice versa. Surface
and canonical analyses are separately labeled, with AB, BA, and pooled-order
results. These diagnostics do not alter historical formal metrics or relabel
saved outputs; they correct only the interpretation of surface position.

## Outputs and tests

- `summary.md`: readable main, atomic-validation, and surface/canonical tables.
- `metrics.csv/json`: all rebuilt historical metrics and effective replicates.
- `atomic_validation.csv/json`: structured saved aggregates with evidence labels.
- `surface_diagnostics.csv/json`: corrected surface diagnostics and canonical controls.
- `*_bootstrap_draws.npz`, `bootstrap_schedule.npz`: numeric draws, validity masks,
  pair weights, and endpoint multiplicities, retained outside Git.
- `input_hashes.json`, `verification.json`, `discrepancies.json`: identities, checks,
  gaps, runtime/code provenance, and differences against preserved historical outputs.

```sh
/tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_t1_score_rebuild.py
```

The focused tests use synthetic data for BA reversal, independent surface versus
canonical diagnostics, product endpoint weights, shared paired contrasts,
undefined replicate handling, macro validity, exact joins, constituent identity
and truth, and path safety. Raw scores, backup archives, and large numeric outputs
remain outside Git; only code, tests, this documentation, and small summaries are
committed on the rebuild branch.

## Verified macOS run, 2026-10-01

All 3,990 historical metric rows and all 7,980,000 numeric bootstrap values
matched the saved outputs **exactly**, including CI bounds and validity masks.
Every historical result had 2,000 valid replicates. All four saved schedule hashes
and pair-weight arrays matched
`60e1de0274d56a3707782735d50aabe25c264aa37dd54fa66c228ad295ca920f`.
The run produced 1,380 new surface/canonical diagnostic rows and retained 170
atomic-validation aggregate entries with explicit evidence labels. All 15 focused
tests passed.

| Condition | AND AUROC | OR AUROC | OR mixed vs FF AUROC |
|---|---:|---:|---:|
| Qwen LR | 0.902752 | 0.813332 | 0.746819 |
| Qwen difference of means, common layer | 0.907858 | 0.835116 | 0.764095 |
| Qwen difference of means, selected layer | 0.991409 | 0.897070 | 0.846238 |
| Qwen OR “or both” wording | 0.902752 | 0.711472 | 0.651001 |
| Qwen OR “at least one” wording | 0.902752 | 0.665577 | 0.569022 |
| Qwen isolated min/max composition | 0.999546 | 0.999507 | 0.999290 |
| Llama LR | 0.985295 | 0.763750 | 0.660797 |
| Llama OR “or both” wording | 0.985295 | 0.687822 | 0.579311 |
| Llama isolated min/max composition | 1.000000 | 1.000000 | 1.000000 |

Saved atomic validation AUROC is 0.999652367 for Qwen LR and 0.999922337 for
Llama LR; these are not row-score reproductions. In the Qwen raw OR mixed-cell
diagnostic, surface TF-versus-FT AUROC is 0.312077, versus canonical TF-versus-FT
0.525397. Under BA ordering alone they are 0.287953 and 0.712047 respectively,
demonstrating why canonical cell names cannot be used as surface-position labels.

Seven source-hash differences are references to the older `clean_compounds.py`
digest `b46510ed…` versus the recovered benchmark producer `7f4c5643…`; these are
reported separately from the unresolved `pinned_atomic_probes.py` conflict.
Unavailable upstream byte records are the seven excluded activation tensors and
three Llama config/tokenizer files. Their absence does not prevent verification
of finalized scores; it does limit upstream replay claims.

Compact checked-in receipts and tables are in
`results/t1_development_rebuild_20261001/`. Full results and draw archives are at
`/Users/apple/projects/t1-development-results-20261001/`; the compact receipt
records their SHA-256 digests. Pick a fresh output directory when repeating the
documented command.
