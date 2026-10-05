# Mac 2 Final-E paper-package review — 5 October 2026

**PASS for the pinned paper package. No scientific-source, claim, identity, numerical-copy or presentation discrepancy requiring correction was found. No unresolved finding.** This is an additive paper review of the exact candidate below, not a new scientific evaluation or audit. The candidate is preserved byte-for-byte. The accepted scientific conclusions remain conditional on the unchanged lock.

Review branch: `checkpoint-final-e-paper-review-20261005`. Parent/base: exact candidate `c30846d90e0695f1f55f29f2bb92d5a1a754ad8a`. Only `paper/final_e_review_20261005/` is added. No merge is performed. The user explicitly approved review and publication; the published identities supplied the handoff without a separate Mac 1 message.

## Exact identities

| Item | Pin |
|---|---|
| Candidate branch | `checkpoint-final-e-paper-packet-20261005` |
| Candidate delivery | `c30846d90e0695f1f55f29f2bb92d5a1a754ad8a` |
| Candidate inventory SHA-256 | `4646dcc19e0eeb90aee535addff3b0b7a6f952b19aadec059d323a3f0176e9b8` |
| Accepted scientific audit | `3dcd696ad17358ae8a89e458cb79a86a98792291` |
| Accepted scientific evaluation | `75030617daa1987d22d7d550714bf78bc2b0c48e` |
| Accepted raw handoff | `e93f3aeb3cc612af03b2704d195bf17406573459` |
| Frozen producer | `b2c6dde71e82f5f0af308761bf19ed3341670587` |
| Producer tree | `93972950bca996dbfa9794dedf07934868b42e98` |
| Final-lock SHA-256 | `88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e` |
| Source evaluation completion SHA-256 | `632ed93f295027701e1d798a94c758e332c747e071aee410ebb6972114d408bc` |
| Audit reconstruction completion SHA-256 | `03c451b2bb6c013a442f833923b8960d2b64cdef57f0015ec3bfa21c60591fb4` |
| Companion audit receipt SHA-256 | `bf7f8abe6d1dbf67209f4f2b296c2f367742d4f00b5cd7d5e5a4d6e2ce87b60e` |

Repository: <https://github.com/graceshan/truth-probing-extensions>. An actual `git ls-remote` check, rather than a cached tracking ref, matched the candidate delivery before review. No local or remote paper-review branch existed. The isolated review worktree is `/Users/apple/truth-probing-backups/20261001T164224Z/checkpoint-final-e-paper-review-20261005`.

The durable root is `/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1`; the source table directory is its `scientific-evaluation-20261005-v1/evaluation/`. Both the durable `transport/final_lock_20261004.json` and committed `config/final_e/final_lock_20261004.json` match the lock pin.

## Actual review coverage

Read the candidate [README](../final_e_20261005/README.md), [methods](../final_e_20261005/methods.md), [results](../final_e_20261005/results.md), [limitations](../final_e_20261005/limitations.md), both claim-evidence files, [secondary inventory](../final_e_20261005/secondary-results.md), and [verification](../final_e_20261005/verification.md). Read the [accepted audit report](../../docs/final_e/SCIENTIFIC_AUDIT_20261005.md), handoff, operation/process and companion receipts. Relevant published audit bytes were compared with the exact accepted audit commit. Durable completion, companion, operation, authorization and lock pins were checked without executing their scientific operations.

The candidate inventory covers **57 files, 11,441,426 bytes**, excluding only itself. Every size and SHA-256 matched, and actual filesystem coverage matched the inventory. All **58 package files including inventory** also matched the exact candidate Git objects. The candidate worktree remained clean at its original delivery.

The source completion was hashed first. All five source tables then matched their completion-inventory sizes and SHA-256 values:

| Complete source | Rows | SHA-256 |
|---|---:|---|
| scalar-metrics.json | 28,560 | `c38e88da29dda7ec3a6b32dc34a79ccb3056f03c8d8deb451a3bfa519293dcdc` |
| scalar-contrasts.json | 38,556 | `87a2be3006ed666483a0765bbc16aff92be807063e3ed605033a95d0a9032b5c` |
| constituent-metrics.json | 9,450 | `4c75010da3d09adcf6e9e019173d6880344bfa561deab5b9c7d4c69cc22a7c1e` |
| constituent-contrasts.json | 1,050 | `b0dac84c7e1b7ec20c857734a87f46827ebf7ec99034c15da7a1caef87b5c883` |
| wording-paired.json | 6,132 | `51e77c17d7459304530eb0fb50e68e883e1708d466f551b5ed06480142c4e0b8` |

The candidate verifier checked **5,508 exact extracted rows, 72 table/figure display bindings, 44 table rows, 12 claim entries and 21 asset files**. The independent review checker additionally verified complete identity-key sets and uniqueness, exact membership of all **3,873 catalog groups covering 83,748 rows**, their sample/scope/topic/order sets, classification counts and negative-point counts. These are inventories of dependent records, not new scientific estimates or independent findings. Every local Markdown link resolves, including the five complete external source tables.

All **24 principal CSV rows** agree across evaluation, audit and original tables by model, panel, arms, condition, endpoint, sample, scope and topic. The audit CSV omits topic for its pooled/macro rows; these are explicitly matched to `topic=all`. The two renamed roles cover eight rows: `full_training_selection_label_benefit` → `secondary_selection_comparison`, and `full_training_atomic_only_comparator` → `secondary_atomic_only_comparison`. No comparison was identified by role text alone. Exact identities and role pairs are retained in [independent-check.json](evidence/independent-check.json).

## Claim and protocol review

All twelve proposed claims pass against their full-precision pinned rows and the unchanged interpretation contract. The candidate's [claim map](../final_e_20261005/claim-evidence.json) and [extracted rows](../final_e_20261005/evidence/extracted-rows.json) provide the complete scientific identities; the review does not replace them with rounded summaries.

| Claims / candidate location | Reviewed evidence and qualification | Result |
|---|---|---|
| C1–C2; results paragraphs 1–2 | Original primary remains model-specific, pooled bare OR TF/FT versus FF, TT excluded, R_all − S_all; rows `scalar-contrasts:378` (Qwen) and `:14658` (Llama). Positive gain and strict .02 practical threshold are separate. | PASS |
| C3; results paragraph 2 | Fixed-five-topic macro rows `scalar-contrasts:384` and `:14664`; positive macro intervals qualify the primary, without co-primary or every-topic claims. | PASS |
| C4–C5; results paragraph 3 and retention assets | R_all − reduced_LR, atomic_E/atomic_auroc and bare/AND_auroc, samples `mean_11_23_37`, pooled/macro, topic all. All eight lower bounds meet inclusive −.005/−.02 margins. Qwen atomic rows `:4641` and `:4647` retain negative points and wholly negative intervals. Retention does not mean zero loss. | PASS |
| C6; results paragraph 4 | R_all − S_full in full_bank, bare OR boundary; rows `:28938`, `:28944`, `:33698`, `:33704`. Pooled practical support and macro support are separate. Exposure, preprocessing and balanced sampling change together; S_full uses compound labels; full-training repair untested. | PASS |
| C7–C8; secondary table | S_full − S_atom_full and R_all − S_atom_full remain distinct full_bank roles, both scopes/models. All eight intervals positive; no substitution for primary or pure exposure-effect claim. | PASS |
| C9; secondary inventory | D-selected bank and C_clean references remain descriptive; registry categories agree with the listed heads. No E-selected maximum or collective-bank-span claim. | PASS |
| C10; wording table and limitations | All six R_all heads, both alternate OR conditions, pooled OR boundary, wording minus bare. Twelve negative points; eleven wholly negative intervals. `wording-paired:1050`, Llama B25_37 at-least-one, crosses zero. No averaging of interval endpoints. | PASS |
| C11; constituent table and limitations | AND-to-OR, explicit-or-both, pooled/all-order, Qwen layer 18 and Llama layer 12; target-trained minus source-trained. All six endpoints/model retained: seven material-deficit, two unresolved, three upper-bound-below-.05 classifications. Other directions/topics/orders and .90 outcomes remain in complete sources. | PASS |
| C12; closing results and limitations | Conjunction of original primary and separate atomic/AND retention, with no simultaneous or joint 95% coverage. Held-out entities within this benchmark, not unseen-topic generalization, calibration, behavior or causal use. | PASS |

The metric/contrast means are those already reported for samples 11/23/37 within every bootstrap draw. Neither fitted-head scores nor interval endpoints were averaged here. The inference description preserves 2,000 PCG64(1729) topic-stratified entity draws, separate atomic/compound frames, shared within-frame draws, endpoint-multiplicity products for compounds, ordinary all-five-topic macro, linear percentile intervals and the 1,800-finite-draw requirement. Undefined values remain unresolved. No new uncertainty calculation was performed.

Methods model identities/revisions match the pinned raw contract: **Qwen/Qwen2.5-7B-Instruct** at `a09a35458c702b33eeacc393d103063234e8bc28` and **meta-llama/Llama-3.1-8B-Instruct** at `0e9e39f249a16976918f6564b8830bc894c89659`. Layer counts/widths are 28/3,584 and 32/4,096. Last nonpadding token, raw statement input, special tokens, no chat template/truncation, BF16 computation, FP16 storage, batch one, `hidden_states[1:]` and final RMSNorm descriptions agree.

Pinned protocol/full-bank configurations and E manifest support the stated 2,778 reduced/3,040 full atomic fitting rows, 700/1,000 balanced exposure recipes, 1,012 atomic development rows, 75 reserved entities, five pairs/topic, 25 pair blocks and 400 dependent rows/sample. The E manifest has 450 pairs, 225 compound entities, 7,200 bare rows, 3,600/operator, 2,700 OR-boundary rows and 974 atomic rows; the lock supplies the separate 294-entity atomic frame. Topic counts and unequal support are disclosed. The 280/320 bank counts, LR C grid, P-only population standardization/floor/masking, canonical layers/C, repair objective, unpenalized intercept, five disjoint-fold AUROC selection, refit/fallback and tie rules agree with the configurations. R_all layers match the registry: Qwen 20/22/18, Llama 14/13/13. Constituent source-only selection and matched-layer references agree with the registry and protocol.

Limitations preserve fixed-budget rather than sample-efficiency claims, dependent observations, training/selection uncertainty exclusions, descriptive references, unfavorable wording and constituent evidence, and the absence of atomic-loss-causes-retention or causal-use claims. Complete audit replay of 83,748 rows/85 prediction archives is correctly distinguished from the narrower separate implementation's 393,650 checks; not every field has a second implementation. These audit results were accepted provenance, not rerun. Both sessions used the same physical Mac.

## Export inspection and reproducibility

Viewed **each of the seven candidate PNG previews and each of the seven actual PDFs**, the latter freshly rasterized by native PDFKit into a separate temporary directory. Checked titles, captions, table cells, axes, intervals, comparison directions, margins and page boundaries. No clipped or missing content was found. The retention figure labels its different x scales. Qwen negative atomic points are visible; the extremely small negative upper-bound gaps from zero should be read numerically from the accompanying table, not inferred from pixels.

All **44 CSV rows** were checked against extracted text from the actual PDFs, SVG text comments and LaTeX fragments. PDFKit returns the model column separately for the wording and constituent tables; its ordered model sequence was checked separately and its spatial association visually inspected. Every SVG parses and every internal resource/clip reference resolves, with no external dependency. Figure geometry and source bindings were inspected and all generated files were independently rebuilt.

After reading all of `build.py`, ran it only with a fresh `/tmp/final-e-mac2-paper-review-render` output directory. **All 45 generated files were byte-identical to the candidate**, including seven PDF/SVG/PNG sets, all editable table files, extraction/catalog/bindings, results and claims. No candidate inventory was refreshed. [Visual evidence](evidence/visual-review.json) pins both candidate assets and actual PDF rasterizations; [independent evidence](evidence/independent-check.json) records each rebuilt file's hash.

## Commands, preservation and limits

From the isolated review worktree, used the established `/opt/homebrew/opt/python@3.11/bin/python3.11` with `-B`:

```bash
python3.11 -B paper/final_e_20261005/inventory.py --check
python3.11 -B paper/final_e_20261005/verify.py --receipt paper/final_e_review_20261005/evidence/presentation-check.json
python3.11 -B paper/final_e_20261005/build.py --output /tmp/final-e-mac2-paper-review-render
swift -module-cache-path /tmp/final-e-mac2-review-swift-cache paper/final_e_20261005/render_pdf_previews.swift paper/final_e_20261005 /tmp/final-e-mac2-review-pdf-previews
swift -module-cache-path /tmp/final-e-mac2-review-swift-cache paper/final_e_review_20261005/pdf_text.swift paper/final_e_20261005 paper/final_e_review_20261005/evidence/pdf-text.json
python3.11 -B paper/final_e_review_20261005/check_review.py
```

The builder/verifier and each review rendering/check process acquired `/tmp/truth-probing-cpu-exclusive.lock` themselves, without double-locking. Numerical thread variables were one. No substantial scientific job was active at the initial relevant-process inspection; unrelated applications were left running. No applicable AGENTS.md was found in inspected ancestors or repository/worktree. The main checkout's original branch, HEAD and historical dirty/untracked status were preserved. The producer remained clean at its exact SHA/tree. [Workspace evidence](evidence/workspace-preservation.json) records these checks.

No fits, selection, new metrics/contrasts/bootstrap calculations, D reproduction, scientific evaluate/audit command, raw extraction, model loading or RunPod contact occurred. No raw campaign or scientific arrays were opened or rehashed. Scientific sources, controls, receipts, lock and execution code were read-only. No switch of the user's checkout, reset, clean, stash, merge, force push or interruption of unrelated work occurred.

This PASS certifies the inspected package and its presentation against accepted outputs. Complete external table links work on the supplied shared Mac; they are absolute local paths, not public archival URLs. No conference manuscript/template was supplied. The LaTeX fragments were value-checked, not compiled into a final manuscript; final column-size readability and template integration remain outside this review. SVG structure and byte-identical rebuilding were checked, but no separate SVG rendering engine was used. These are coverage limits, not unresolved defects in the candidate.

The review [inventory](inventory.json) covers every review file except itself. Its SHA-256 and the exact published review delivery are supplied in the publication handoff, avoiding self-referential hashes. Publication is on the separate review branch only; the remote SHA is independently checked after push.
