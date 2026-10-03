# October 3 Revision 2 behavior/chat preparation

Preparation is complete; GPU/CPU research execution is pending review. The exact runnable producer is `f633a03f317c68e310515113634948bf892bd962`; the delivery adds its pinned command runbook after that producer without changing executable code or configuration. This branch starts exactly at `0a256d96b0fbf41f7ce678693da15cf230685778`. Origin was fetched before the isolated worktree was created. No moving branch was pulled or merged. Completed B25 and constituent lineage was checked through `1187140`, `2d28c0b` and the exact base. Committed completion/audit receipts report completed B25 with no fit failures and 252 valid constituent fits with a separate saved-prediction audit. Those historical results were neither refitted nor newly audited here.

Preparation ran on **MacBook-Pro-7.local**, macOS 12.6 arm64, `/opt/homebrew/opt/python@3.11/bin/python3.11`, Python 3.11.6. Local checks were serialized, with numerical thread counts set to one. The reported host is the actual host, not an assumed session label or second-machine replication.

The attached October 3 **Revision 2** document was read before implementation and copied unchanged to `docs/checkpoint_next/Truth_Probing_Next_Decision_Checkpoint_2026-10-03.docx`. SHA-256: `f81c2209c03eacf5c6fe5e14729d68069035ea3a08de3d2b9eb273b070c4cae2`. The older October 2 Revision 2 document is not this task's authority. The October 3 text is retained alongside the unchanged DOCX; the DOCX remains authoritative.

## Frozen inventories

| Quantity | Per model, except where specified |
| --- | ---: |
| D pairs | 50, ten per topic |
| D compound bindings | 800, all 16 cells/operator/order leaves per pair |
| Isolated bindings | 200 |
| Unique isolated facts/texts | 150 |
| Two-prompt behavior logical bindings | 2,000 |
| Two-prompt behavior unique prompted inputs | 1,900 |
| Prompt 1 original P15 | 2,778 |
| Prompt 1 fixed T_C | 8,000 |
| Prompt 1 D compounds | 800 |
| Chat logical/unique prompted inputs | 11,578 |
| Behavior/chat exact prompted-text overlap | 800 |
| Union unique prompted inputs | 12,678 |
| Behavior logical bindings across two models | 4,000 |
| Chat logical bindings across two models | 23,156 |
| Training preflight inputs | 40 |

Text deduplication retains every logical binding, including repeated isolated facts across pairs. Prompt indices remain separate. All exact text, label, source identity and pair bindings are independently compared with the inherited frozen behavior/chat extraction inventory. Prompt/token/state reuse is not established merely by a text match. The implementation makes dedicated chat forwards; it does not silently reuse continuation-scoring states. Saving two layers reduces storage, not forward count.

`data/checkpoint_next_behavior_prep_20261003/inventory.json.gz` is a deterministic, lossless full JSON inventory, including ordered jobs, all bindings, eight recipes and training preflight rows. `behavior-bindings.csv` makes the 800/200 support reviewable; accounting, hashes, recipes and preflight inputs are separate JSON files. Decompress with Python's standard `gzip` module. `inventory()` reconstructs the exact artifact; `frozen()` checks equality.

## Implemented contracts

Both approved prompts are exact inherited strings. Rendering uses user-only native chat framing and the native answer-generation prefix. Qwen's pinned native template implicitly adds a default system prefix; the adapter removes only that complete prefix, checks the exact native token-prefix decomposition, and preserves the native user/assistant segments. Llama uses the native BOS/header structure. This convention is explicit and awaits the new-path real-tokenizer preflight.

Likelihood scoring verifies that tokenizing the complete prompt plus `True` or `False` preserves every prompt token. It scores all candidate tokens at the preceding causal logits, sums token log-probabilities, adds no whitespace, normalizes no lengths, and scores no termination token. The zero-margin rule includes ties as True. Generation is greedy, one beam, at most eight new tokens, with termination IDs from the accepted cached model config/tokenizer. Only termination suffix tokens and outer whitespace are removed for parsing; exact case-sensitive `True`/`False` is required. Raw output, candidate IDs, generation IDs, margins and format failures are saved.

Full support is primary. A secondary model/prompt-specific pair mask requires all four isolated likelihood decisions correct and retains all 16 compounds. It never examines compound correctness. Isolated raw and label-signed margins retain binding and unique-fact denominators; no confidence filter is introduced. Generation format failures count as incorrect in primary accuracy but are reported separately from valid factual errors. Valid-answer disagreement uses valid generated answers as its denominator.

Exactly eight chat recipes are frozen: R0/C_clean at Qwen saved layers 17/18 and Llama 15/12. Extraction saves HF `hidden_states[layer+1]` at the last input token before the answer. Actual decoder/norm hook checks reuse reviewed extraction semantics. Preprocessing is fit on original chat-P15 only, with the reviewed population standard deviation, floor and constant-coordinate mask. R0 uses mean P15 BCE plus `.001 ||w||²`; C_clean uses mean fixed T_C BCE plus the same penalty and free intercept. Equal T_C pair/cell/order weights are verified to equal its 8,000-row mean. Existing solver acceptance is reused. No full-training sensitivity inputs enter these fits, and no new layer selector is present.

Reporting prepares full/masked pooled, topic, macro, cell/order, True-rate, format/error and likelihood/generation disagreement tables. AND/OR/OR-boundary AUROC is kept separate from forced-choice decisions. The same frozen 50-pair support is required for probe/behavior comparisons; full-D probe ranking remains a separate panel. Existing zero-logit decisions support joint error tables; AUROC-only arms get no invented threshold. Paired topic-stratified entity bootstrap functions are reused with 2,000 planned draws, endpoint multiplicity products, fixed observed masks and undefined draws/topics retained. Local tests use only 20 synthetic draws.

These chat controls establish only the named Prompt 1 readouts. They do not establish prompted constituent availability, absence at other layers, or causal use. Strong output-margin ranking with weak decisions requires answer-bias/threshold diagnosis before a composition claim. Formatting failure is not itself a factual error.

## Validation and preservation

All **15 focused synthetic tests passed** in 2.954 seconds. The focused standard-library unittest suite covers continuation indexing/sums, native prefix removal and boundary failure, strict parsing/caps, all inventory counts, original P15-only objective preparation, exact missing-input bindings, immutable publication/reopen, resume and corruption stops, partial preservation, conservative interruption accounting, isolated-only masks, explicit denominators, no thresholds for AUROC-only arms, raw/chat identity checks and empty-mask undefined endpoints. No research head is fitted by these tests.

The metadata dry run, full inventory reconstruction, source hashes and Python compilation passed. Exact validation evidence and test output are in `results/checkpoint_next_behavior_prep_20261003/`. Existing reviewed source files are reused without edits. The original worktree's HEAD and tracked status remain unchanged; hashes for all 15 pre-existing untracked files (13 unrelated files plus two supplied documents) match the initial snapshot. Historical results/manifests/worktrees were preserved. Only this successor branch is delivered; no merge into `clean-eval-protocol` occurs.

## Remaining execution prerequisites

1. Review this exact producer and provide a later explicit execution authorization. No valid launch authorization is included here.
2. Identify the current Pod endpoint through its current management record. The Pod has not been contacted, restarted or used. Old SSH endpoints are not assumed current.
3. Verify the existing offline snapshots, actual GPU/driver, Python 3.12.3 and full accepted dependency freeze. The accepted reference is `config/checkpoint_r2/full_raw_v1.json` and its pilot receipts: Torch 2.8.0+cu128, Transformers 4.56.2, batch 1 unpadded, BF16/math SDPA, no TF32, FP16 saved states. Obsolete setup fields in the original protocol are not used. Missing/corrupt caches or runtime drift stop for review, without download or substitution.
4. Execute the new bounded training-only preflight: P15 and fixed T_C shortest/longest UTF-8 texts per topic under both prompts, 40 inputs/model. Benchmark tokenization, both full-sequence likelihoods, greedy generation and Prompt 1 chat independently. Repeat behavior/chat measurements exactly. Full workload tokenization is metadata-only and must verify every continuation boundary before inference. Forecast uses measured maximum training path costs times exact planned inputs and requires training token-length coverage. An inadequate forecast or coverage stops for a smaller reviewed scope or revised cap.
5. Review quota/current mount and run the existing allocation/publication check while retained weights, raw outputs and other artifacts remain. Reserve primary and second copies, temporary shard and safety. Recheck after restart. Statvfs, tensor arithmetic and a successful historical raw extraction are not storage/runtime guarantees.
6. Obtain mac 2's loader-verified raw-input/token/representation inventory and its missing-raw resource ledger. The metadata crosswalk finds 262 additional bindings/texts beyond P15 but only 131 texts absent from the committed raw text inventory, because some additional texts already occur in other raw groups. Of 3,040 outer-training bindings, 2,909 have text-only candidate matches. These are not yet verified reuse decisions. The interface preserves exact admitted source rows and groups absent bindings by unique text; it rejects unverified available representations. It does not extract them or claim 262 new forwards.
7. Include missing-raw GPU use in the same two-hour/model ledger. GPU attempts/resumes conservatively charge interrupted wall time, including downtime; no cap reset. The prepared ceilings are 7,200 active GPU seconds/model and 1,800 setup seconds/phase. The three-hour full-bank CPU cap is retained as a reference for mac 2, not silently assigned to chat fitting. Chat fitting needs a separately reviewed CPU stopping limit and an exclusive CPU slot after mac 2's worker finishes.
8. Transfer/reopen exact output and token/shard hashes, then obtain the eight raw comparator exports and frozen full-D probe score exports. Reconstruct saved chat scores and verify optimizer gradients independently before scientific acceptance. The prepared CPU runner fits exactly eight heads; synthetic tests do not establish actual convergence or numerical validity on model states.

Runtime, full serialized storage, peak GPU memory, quota, transfer speed and chat CPU cost are **unmeasured**. Only arithmetic FP16 chat payloads are known: Qwen 165,982,208 bytes and Llama 189,693,952 bytes, one copy each, before metadata/temporary/second-copy costs. These are not complete storage forecasts.

No weights were downloaded, real model inference run, research heads fitted, or E inputs/predictions inspected or scored. Full-bank fitting, saved-score diagnosis, related work and the conditional six-fit ablation belong to separate checkpoint work; none is reported completed here. First-use through stop commands are in `runbook.md` and are prepared for later execution only.
