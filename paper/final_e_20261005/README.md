# Final-E paper package — 5 October 2026

This additive package presents the accepted Final-E results for the planned NAACL paper. Start with [results.md](results.md), [methods.md](methods.md), [limitations.md](limitations.md) and the [claim-to-evidence map](claim-evidence.md). The historical manuscript and repository documentation are unchanged. The worktree branch is `checkpoint-final-e-paper-packet-20261005`, based on the accepted scientific audit. No merge is requested or performed.

The two main figures are [primary gain](figures/primary-gain.pdf) and [retention](figures/retention.pdf), with editable SVG and PNG previews beside them. Five paper tables are supplied in PDF/SVG/PNG plus editable CSV/LaTeX/Markdown:

- [Main OR comparison](tables/main.pdf): R_all, S_all and S_full AUROCs; original-primary and separately labeled sensitivity gains.
- [Retention](tables/retention.pdf): atomic and AND endpoints, pooled/macro, margins and frozen classifications.
- [Separate full-bank comparisons](tables/secondary.pdf): S_full − S_atom_full and R_all − S_atom_full.
- [Wording sensitivity](tables/wording.pdf): both OR variants for all six R_all heads, including the unfavorable outcomes.
- [Constituent transfer](tables/constituent.pdf): all six pooled constituent AUROC endpoints for both models under explicit OR wording, with frozen transfer classifications.

The [secondary inventory](secondary-results.md) covers all frozen comparisons and links all five complete audited tables. Its [catalog](evidence/secondary-catalog.json) groups all 83,748 source rows without treating dependent rows as independent findings. [Extracted rows](evidence/extracted-rows.json) preserve exact values and complete identities, including topic and surface order where applicable. [Display bindings](evidence/display-map.json) connect table cells and figure data to source-file SHA-256 and zero-based record indices through those extracted rows. [Threshold sources](evidence/threshold-sources.json) pin margins and protocol facts. All displayed numbers are copied from the source; six-decimal formatting never drives a decision.

## Accepted delivery chain and pins

Repository: `https://github.com/graceshan/truth-probing-extensions`.

- Scientific audit delivery: `3dcd696ad17358ae8a89e458cb79a86a98792291`; [audit report](../../docs/final_e/SCIENTIFIC_AUDIT_20261005.md), [handoff](../../results/final_e_scientific_audit_20261005/audit-handoff.json).
- Scientific evaluation delivery: `75030617daa1987d22d7d550714bf78bc2b0c48e`; [evaluation report](../../docs/final_e/SCIENTIFIC_EVALUATION_20261005.md), [handoff](../../results/final_e_scientific_evaluation_20261005/evaluation-handoff.json).
- Accepted raw handoff: `e93f3aeb3cc612af03b2704d195bf17406573459`.
- Frozen producer: `b2c6dde71e82f5f0af308761bf19ed3341670587`; tree `93972950bca996dbfa9794dedf07934868b42e98`.
- Final-lock SHA-256: `88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e`.
- Source evaluation completion SHA-256: `632ed93f295027701e1d798a94c758e332c747e071aee410ebb6972114d408bc`.
- Audit reconstruction completion SHA-256: `03c451b2bb6c013a442f833923b8960d2b64cdef57f0015ec3bfa21c60591fb4`.
- Companion audit receipt SHA-256: `bf7f8abe6d1dbf67209f4f2b296c2f367742d4f00b5cd7d5e5a4d6e2ce87b60e`.

The companion receipt establishes complete replay acceptance and the separate principal checks. This package does not rerun either. The evaluation and audit principal CSVs have two renamed secondary display roles; the builder matches their full scientific identity (including panel and arms), rather than role text. All 24 principal rows agree exactly with the pinned source.

The final publication SHA and inventory hash are supplied in the published handoff. The commit message records the inventory hash; neither README nor inventory contains its own commit hash. The [inventory](inventory.json) records sizes and SHA-256 for every package file except itself. This avoids self-reference.

## Build and verification

Run from the root of this publication worktree. The exact build command used is:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -B paper/final_e_20261005/build.py
```

The established Python is 3.11.6; the formatting dependencies are the existing Matplotlib and NumPy environment (exact versions recorded in [verification.md](verification.md)). No installation or scientific-runtime change is needed. The builder sets OMP, OpenBLAS, MKL, vecLib and NumExpr thread counts to one before plotting imports, and acquires `/tmp/truth-probing-cpu-exclusive.lock` without waiting. It exits if another session holds the lock. Mac 1 and Mac 2 share one physical Mac. Do not wrap the builder in a second hold on that lock.

The default evaluation directory is the actual absolute path:

```text
/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1/scientific-evaluation-20261005-v1/evaluation
```

The parent durable package retains `execution-code`, `handoff/campaign`, `transport/final_lock_20261004.json`, scientific controls, operation receipts, source tables and arrays. The builder reads only the five existing JSON result tables and completion, the transport lock and small published protocol/audit files. It does not import scientific modules, call evaluate/audit, open prediction/bootstrap arrays, load models, access activations or contact RunPod. It rejects a hash mismatch or ambiguous row identity instead of repairing the analysis.

To preserve the candidate while rebuilding generated outputs, use a new output directory:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -B paper/final_e_20261005/build.py --output /tmp/final-e-paper-review-render
/opt/homebrew/opt/python@3.11/bin/python3.11 -B paper/final_e_20261005/verify.py --package /tmp/final-e-paper-review-render
```

The output directory receives generated figures, tables, exact extraction, catalog, claim map, secondary inventory and results prose. Authored methods, limitations, README and verification notes remain in this package. The builder overwrites only its named generated outputs; it does not remove unrelated output files. For paper integration, LaTeX tables use `graphicx` for `\resizebox`; the rendered table PDFs are generated from the same cell strings by Matplotlib. No manuscript template is overwritten.

Presentation and package checks:

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -B paper/final_e_20261005/verify.py
/opt/homebrew/opt/python@3.11/bin/python3.11 -B paper/final_e_20261005/inventory.py --check
```

`inventory.py` without `--check` deliberately refreshes the inventory and is for authoring a new delivery only. Do not run it to conceal a discrepancy in a pinned candidate. Both build and presentation verification use the shared CPU lock. These checks inspect copied rows, identities, formatting, categorical decisions, source hashes, coverage and asset structure; they do not calculate new scientific results.

Committed assets are the small formatting scripts, protocol pins, prose, claim map, extracted JSON rows, exhaustive comparison catalog, rendered figures and tables, and presentation evidence. Complete scientific tables and large activation/prediction/bootstrap arrays remain externally retained; the source completion inventory already committed in the accepted evaluation delivery binds them. The 5,508 extracted rows are a review aid, not a replacement for the complete external report. Read [verification.md](verification.md) for completed checks and presentation limits.
