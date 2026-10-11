# Post-hoc reduced-bank maximum on E — executed and audited

**PASS.** All 600 frozen original reduced/P15 candidates were scored: 280 Qwen and 320 Llama. Original atomic-D eligibility retains 144 and 182, respectively. Eligible and unrestricted maxima coincide. No tested candidate matches or exceeds the unchanged joint-training mean on E.

This is explicitly authorized **post-hoc** analysis. E labels identify the winner. The maximum is an optimistic descriptive reference on this observed test set, not an independently evaluated selection procedure or a bound on future-population performance. Its scope is the finite tested bank, not every possible atomic-trained probe. The original selected heads, primary comparisons, scientific lock and manuscript are unchanged. Compound D historically informed exploratory development; a separate October 10 analysis already examined E. This specification was saved before this execution and makes no prospective or first-use claim.

## Results

Endpoint: bare pooled OR TF/FT versus FF, excluding TT, using the accepted E identities and score directions. There are 450 pairs, 2,700 boundary rows (1,800 positive and 900 negative), and both surface orders. Original atomic-D eligibility is unchanged; atomic E is not used. Each score tie receives half credit.

| Model | Maximum scope | S_all | E maximum | R_all | Maximum − S_all | R_all − maximum |
| --- | --- | --- | --- | --- | --- | --- |
| qwen | eligible | 0.844643 | 0.866289 | 0.983280 | 0.021646 | 0.116991 |
| qwen | unrestricted | 0.844643 | 0.866289 | 0.983280 | 0.021646 | 0.116991 |
| llama | eligible | 0.811593 | 0.834967 | 0.988610 | 0.023375 | 0.153643 |
| llama | unrestricted | 0.811593 | 0.834967 | 0.988610 | 0.023375 | 0.153643 |

Both maxima use the same unique winner in each model. Exact maximizers and maximum-relative 1e-12 tolerance ties were enumerated separately; they coincide here. No winner was chosen by a random tie-break.

| Model | Scope | Candidate ID | Method | Saved layer | C | Exact maximum |
| --- | --- | --- | --- | --- | --- | --- |
| qwen | eligible | qwen/P15/L18/l2_logistic/C=10 | l2_logistic | 18 | 10.0 | True |
| qwen | unrestricted | qwen/P15/L18/l2_logistic/C=10 | l2_logistic | 18 | 10.0 | True |
| llama | eligible | llama/P15/L09/l2_logistic/C=1 | l2_logistic | 9 | 1.0 | True |
| llama | unrestricted | llama/P15/L09/l2_logistic/C=1 | l2_logistic | 9 | 1.0 | True |

Saved layers are zero-based (`hidden_states[saved_layer + 1]`). The winning Qwen C=10 and Llama C=1 probes are the original raw-feature L2 logistic fits on 2,778 P15 atomic rows, with an unpenalized intercept and objective mean BCE plus squared weights divided by 2Cn. All original fitted method-specific preprocessing is retained. `head_provenance.json` preserves complete settings, original fitting producers and fit/parameter references for every candidate.

## Frozen comparison procedures

| Model | Procedure | Allocation 11 | Allocation 23 | Allocation 37 | Mean AUROC |
| --- | --- | --- | --- | --- | --- |
| qwen | S_all | 0.801351 | 0.866289 | 0.866289 | 0.844643 |
| qwen | R_all | 0.976946 | 0.984270 | 0.988623 | 0.983280 |
| llama | S_all | 0.811593 | 0.811593 | 0.811593 | 0.811593 |
| llama | R_all | 0.983474 | 0.992389 | 0.989967 | 0.988610 |

S_all is original reduced-bank Compound selection using A; R_all is Joint training. Means are the ordinary means of three separately calculated AUROCs. Predictions are neither averaged nor concatenated. Their accepted E scores and metrics were independently reconstructed. Maximum-based differences above are descriptive point differences only: no fixed-winner confidence interval is attached, and no bootstrap analysis was run.

## Verification and independent audit

- Complete grid verified against the B25 bank manifest, saved fits, exact parameter hashes and original D eligibility. The independent audit separately reconstructed the 10-settings-per-layer grid and atomic-D gate.
- Accepted E loader, model provenance, FP16 storage, saved-layer indexing, text identities and logical row ordering verified. Every candidate is bound to the same 2,700-row identity/label/order hash. All predictions are finite.
- All 23 overlapping reduced-bank heads reproduce the accepted scores **exactly on all 7,200 bare rows**, including the boundary subset. 60 accepted endpoint rows were checked. The independent audit rechecked all 23 boundary score vectors against the original accepted predictions.
- All 600 AUROCs independently recomputed from saved scores using a Mann–Whitney average-rank calculation, separate from the production weighted pair-count implementation. Maximum absolute discrepancy: 0.
- Both maxima, every exact/tolerance winner, all 12 frozen allocation AUROCs, four means, and every reported descriptive gap independently reconstructed.
- Six focused tests passed: exact score ties, exact/tolerance maximum ties, eligibility filtering, missing/duplicate/substituted/invalid candidates, row/label misalignment, and nonfinite or unscored candidates. No unrelated suite or complete historical evaluation was rerun.
- No fitting, cross-validation, re-extraction, GPU work, full-bank evaluation, wording sweep, score flipping, recalibration, ensembles, or new E thresholds.

## Lineage and source hashes

Reviewed base: `a8b4cd326daa243338dc7941c19d98ce75f5b77f`.
Analysis code commit used: `cf03ee50e18bf7535eb6eec9e95a191fc7438f7e`.
B25 delivery: `1187140c7f79b0b64439eff382d420f51d974a25`; completed successor producer: `5e1bf47a457558cbc0735bd8f9f13ca029cbf3a4`.
Original E extraction producer: `b2c6dde71e82f5f0af308761bf19ed3341670587`.
Unchanged scientific-lock SHA-256: `88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e`.
Training-representation manifest SHA-256: `f28e75f5fc95cf3f019f5b47c68c5c2753435854280bfb122f0df22187954445`.
Original parameter producers are preserved: 62 heads from `3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f`, 23 from `b25581ca9c22d93c8c988c93c0c5491219d6cfec`, and 515 from `5e1bf47a457558cbc0735bd8f9f13ca029cbf3a4`.

The reviewed paper README led to the accepted scientific evaluation and independent audit handoffs. Both completion hashes and the successful audit companion were checked. `input_hashes.json` records source manifests, heads, preprocessing archives, original scores, raw-loader receipts, representation files and code hashes. `support_receipt.json` pins the E support. Historical access records identify accepted inputs; the present user request and `analysis_spec.json` authorize this separate analysis.

## Runtime and reproduction

Local macOS, one analysis process, one BLAS/OpenMP thread, protected by `/tmp/truth-probing-cpu-exclusive.lock`. Final scoring/verification wall time: **89.753 seconds**; process peak resident memory: **7,120,601,088 bytes (6.632 GiB)**. These are measured by a monotonic clock and macOS `getrusage`, respectively. Independent audit: 10.379 seconds and 469,499,904 peak RSS bytes. These measurements cover each script's analysis/audit function and whole-process peak memory, not Git publishing.

The preliminary run completed scientific scoring but had a shadowed elapsed-time variable. Its outputs remain intact in the external run-v1 directory, and `superseded_run.json` records why its timing was rejected. The corrected implementation was committed and the unchanged scientific calculation was executed in fresh run-v2. `rerun_consistency.json` checks that all scores and numerical results are unchanged.

Large arrays remain outside Git at `/Users/apple/truth-probing-backups/posthoc-reduced-bank-e-20261011/run-v2`. The directory is immutable for this run; a reproduction must choose a new external destination and a fresh small-output directory containing a copy of `analysis_spec.json`. From the new worktree:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -B -m unittest tests.test_posthoc_reduced_bank_e -v
/opt/homebrew/opt/python@3.11/bin/python3.11 -B scripts/posthoc_reduced_bank_e.py --external /Users/apple/truth-probing-backups/posthoc-reduced-bank-e-20261011/run-v2
/opt/homebrew/opt/python@3.11/bin/python3.11 -B scripts/audit_posthoc_reduced_bank_e.py
/opt/homebrew/opt/python@3.11/bin/python3.11 -B scripts/report_posthoc_reduced_bank_e.py
```

The listed scoring command is the command actually used; it deliberately refuses to overwrite its outputs. Relocated audit/report runs accept `--results`. The original manifests contain absolute artifact roots; all relocated inputs must still match their original hashes. The final delivery SHA is supplied with the verified remote handoff rather than self-embedded in this report.

## Files

- `candidate_results.csv`: all 600 candidates, methods/layers/C/settings, original eligibility, AUROC, producer/hash identities and scoring status.
- `summary.csv`, `winning_candidates.csv`, `fixed_procedures.csv`: both maxima, all ties, descriptive gaps and each allocation.
- `analysis_spec.json`, `workspace.json`, `execution.json`, `audit.json`, `focused_tests.json`, `input_hashes.json`, `head_provenance.json`: specification, source lineage, measured runtime and verification evidence.
- `controls.tex`: proposed Controls paragraph. No manuscript file was edited.
