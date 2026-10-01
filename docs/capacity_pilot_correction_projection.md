# Successor correction base and fixed-A capacity projection

The existing unresolved-source policy requires quarantining **two affirmative
source claims and their paired negations** before fitting. No new confirmed
label error was established. This successor supersedes the earlier audit's
recommendation that its findings required no correction-base advance; the
historical recommendation itself remains unchanged.

Starting state: Darwin/mac 1, `clean-eval-protocol`, reviewed commit
`9bbb2d253342261e1d0612ec9b121b8c72390725`. Fetch found origin synchronized;
no pull or merge was needed. All 759 previously tracked files and the 13
unrelated untracked files were hashed before successor work.

## Review propagation

All **83 review outcomes / 166 facts** were inspected with exact entity/person,
statement text, label and SHA-256 binding checks against the original v2 base.
Of eight unresolved negatives, six are generated candidates and two are admitted
atomic source facts. The six generated candidates remain unaccepted; their
unresolved judgments do not justify removing people or unrelated source claims.
No biography searches or new factual adjudications were performed.

| Affirmative | Exact unresolved statement | Paired negation | Statement SHA-256 (affirmative) |
|---|---|---|---|
| `inventors:23` | Herbert Saffir lived in the U.K. | `neg_inventors:23` | `b9a84f71b7e34f20906d0c786d03fec664241155d7362119bbbbef29bc67ac14` |
| `inventors:157` | Hippolyte Mège-Mouriès lived in the U.S. | `neg_inventors:157` | `87844d73d1bc8759331614861dba43b6da92bdadadc5d1837c28c291163bafa8` |

Only these four rows move from admitted to quarantined. Labels remain 0 for the
affirmatives and 1 for the negations. Original statements, hashes, identities,
partitions, evidence and every earlier overlay column remain unchanged.
Saffir's supported US pair (`inventors:58`, `neg_inventors:58`) and Mège-Mouriès's
supported France pair (`inventors:283`, `neg_inventors:283`) remain admitted.

The successor artifacts are under
`data/clean_protocol/selection_repair_v1/`:

- `candidate_overlay_v3/`: full manifest, admitted/excluded/quarantined subsets,
  exact status changes, source-row eligibility, projected historical fact
  inventory and correction receipt. **`audit_successor_status` is authoritative**;
  `successor_status` is the preserved historical v2 value.
- `capacity_pilot_projection_v2/`: all review-outcome bindings, corrected train
  and P memberships, actual balanced selections, membership deltas, counts,
  audited-fact eligibility, annotated T_C queue, recommendation and receipt.

The two exact `source_supported_false` registry duplicates in the historical
fact inventory also become currently ineligible. Historical status and
`eligible` columns are retained; the projected `current_eligible` column is
authoritative. This closes the alternative route by which source-label support
could have reintroduced either unresolved statement. Source-row eligibility
also marks the paired negations ineligible. All eight unresolved reviewed
facts remain ineligible for audited fact use. T_C review-queue candidates keep
their text, rank and provenance; the two source claims are explicitly annotated
as quarantined and requiring resolution, without proposing replacements.

## Recomputed membership

These counts are derived from the four-row status transition and frozen A
proposals, not constants in the projection builder.

| Current membership | Total rows | Inventor rows | Inventor label 0 / 1 | Inventor entities / people |
|---|---:|---:|---:|---:|
| Corrected outer training | **3,044** | 392 | 196 / 196 | 133 / 132 |
| Prospective P15 | **2,782** | 340 | 170 / 170 | 118 / 117 |
| Prospective P10 | **2,868** | 356 | 178 / 178 | 123 / 122 |

All other topic row counts are unchanged. D and E admission remains 1,012 and
1,002 rows respectively. The original 3,144-row outer-training reference remains
in `capacity_pilot_audit_v1`, hash-bound by the successor receipt, and is not a
current fitting membership.

The existing balanced sampler is rerun with seed 0 in original source order.
Corrected outer training exposes 1,000 rows. P15 still exposes **700 rows**
(140/topic; 70 per label/topic); P10 still exposes **800 rows** (160/topic;
80 per label/topic). Their identities were recomputed:

- P15: 72 old rows leave and 72 enter; all four quarantined rows leave.
  Inventor exposure remains 61 people.
- P10: 68 old rows leave and 68 enter. Saffir's two affected rows were in the
  old sample; neither Mège-Mouriès affected row was. Inventor exposure changes
  from 73 to 67 people.

Exact membership changes and old/new hashes are recorded in
`balanced_membership_changes.json`; per-topic labels, forms, entities and people
are in `exposure_counts.csv`. Quarantine may change the random selection beyond
the four removed rows even when total exposure stays constant.

**A15's 75 keys and nested A10's 50 keys are unchanged**, including every fact
and evidence binding; their proposal files are byte-identical copies. Completed
T_C pairs are also byte-identical. The frozen candidate order, judgments,
historical negative proposals/queue provenance and all v1 artifacts remain
unchanged. No candidate or fact is redrawn, replaced or reranked, and the
historical negative queue is never rebuilt against v3.

Use v3 and this successor projection before any future authorized fitting.
The original v1 package remains a reproducible historical projection against v2.
**P/A remains unfrozen.** This propagation task performs no fitting, activation
access, prediction inspection or compound generation and does not modify mac 2's
adoption/cache-mapping records.

## Validation

```sh
/tmp/clean-extraction-venv/bin/python -B scripts/50_project_capacity_pilot_corrections.py
/tmp/clean-extraction-venv/bin/python -B scripts/49_build_capacity_pilot_audit.py
PYTHONDONTWRITEBYTECODE=1 /tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_capacity_pilot_projection.py tests/test_capacity_pilot_audit.py tests/test_selection_repair_source_closure.py tests/test_selection_repair_candidate_overlay.py tests/test_selection_repair_source_audit.py data/clean_protocol/selection_repair_v1/source_review_v1/test_validation.py
/tmp/clean-extraction-venv/bin/python -B scripts/check_selection_repair_local_preservation.py --root /Users/apple/projects/truth-probing-extensions
git diff --check
```

Result: **91 tests and 29 subtests passed**. Both successor and historical
regeneration are byte-identical. All historical-file hashes and all 13 local
untracked-file hashes pass. The new regression marks a different admitted
source fact unresolved, discovers its paired restriction dynamically, and
rejects either surviving polarity in every current fitting-membership type.
Additional tests cover source/registry eligibility propagation, positive-row
retention, statement/identity mutations, unchanged A/T_C and review-queue bytes,
derived counts and actual sampler changes.

The successor command defaults to check-only. `--write` creates missing outputs
and refuses to overwrite differing existing files. It reads the existing
judgments and frozen queue; it does not initialize another queue or generate
new judgments. Current membership and fact-eligibility outputs are validated
before their hashes are placed in the receipts.
