# October 3 Revision 2 integration preparation

Local preparation is complete on `checkpoint-next-integration-fix-20261003`. The final executable producer is `9b0ac1d84b35d7faf2148d43664569aa867cba52`. The documentation delivery is its descendant; the external review package's delivery.json and README record its exact SHA. Execution commands are in INTEGRATION_RUNBOOK_20261003.md and have not been executed.

The exact normal merge `5ef04a7b45d4985fd41bc777e630474fd5ae68a2` has ordered parents:

1. Behavior delivery `f63a9a9348892b81c0ab60261033a2b8600f26a2`.
2. Reviewed diagnosis delivery `bca07a055c70f7debb6389cf68feb6d2ab7c543f`.

Their verified common base is `0a256d96b0fbf41f7ce678693da15cf230685778`, retaining the completed B25/constituent lineage. The intermediate implementation commit `d0416bc28cf4aaef7ad3da6ec1ae9d07227ed9e2` is preserved; the final producer adds raw-export content binding. No commits were rewritten. Both original branch tips and worktrees remain unchanged. No push was attempted.

## Corrections

Generation omits explicit position_ids only in generate(), allowing pinned GenerationMixin to derive growing positions from the mask. Direct likelihood/chat forwards keep explicit positions. Greedy decoding, no cache, native termination, eight-token cap, both prompts, strict parser and numerical settings remain frozen. The prospective 40-input training preflight independently checks causal full-sequence terms against prefix forwards, manual greedy IDs against generation, and actual prepared input/mask/position lengths at every decoding step. Repeatability is additional evidence. The prospective BF16 absolute log-probability tolerance is 0.05; IDs and position traces must match exactly.

Isolated reports now include accuracy, True rates, format failures, valid factual errors and valid-answer disagreement with explicit denominators, by model/prompt/topic. They retain all 200 logical bindings and disclose 150 unique facts, retain label-signed margins, and leave the all-four-isolated-correct pair mask unchanged. There is no confidence or compound-correctness filter.

The reporting adapter accepts exactly eight successful chat fits, verifies receipt/parameter/score hashes, binds the raw export content used at fitting, and matches family, layer, P15/fitting identities and the exact 800 D rows. The 7,728-row full-D raw arm stays separate. No full-D chat estimate, D threshold, calibration, objective change or layer search was added. AUROC-only arms retain no decision table.

Execution attempts record loading, tokenization, likelihood, generation, chat/raw extraction, preflight checks, write/verify and teardown durations, plus unattributed walltime. CUDA allocated/reserved peaks share the reviewed loader's reset before model loading and are published before teardown. Completion binds execution evidence. Recoveries do not relabel reused work as newly measured. A hard process kill can prevent final evidence; its watchdog charge and incomplete artifacts are retained and must be reported.

Flagged chat fits publish parameters, status and every optimizer attempt's initial/final vectors and diagnostics before stopping. Failed fits cannot predict. No objective, tolerance or iteration budget changed. An optimizer killed before returning cannot supply an unfinished diagnostic record and remains incomplete, never successful.

The missing-raw runner consumes the committed queue unchanged. It reconstructs it against the accepted raw manifest, frozen outer_train and committed 6,080-row coverage evidence. Extraction uses reviewed offline loading, batch 1 unpadded BF16/FP16, all saved layers and native raw token conventions. Immutable reviewed shard publication/load, token-plan/forward equality, quarantine, corruption stops, separate tokenizer-only reconstruction and independent acceptance are prepared. Existing raw outputs are never overwritten. No loader-verified flag or completed real ledger was created in preparation.

One campaign lock/attempt ledger charges missing raw, loading, preflight and behavior/chat against 7,200 seconds/model. Setup remains capped at 1,800 seconds/phase. The generation forecast includes the eight-token cap even when measured answers terminate early; it remains a forecast, with explicit reviewed allowances and a remaining-budget check. The full-bank 10,800-second CPU reference is retained and does not authorize chat fits.

## Frozen inventories

| Inventory | Per model unless stated |
| --- | ---: |
| D pairs / compound rows | 50 / 800 |
| Isolated logical bindings / unique facts per prompt | 200 / 150 |
| Behavior bindings across both prompts | 2,000 |
| Unique behavior prompted inputs | 1,900 |
| Prompt 1 chat P15 / T_C / D | 2,778 / 8,000 / 800 |
| Unique chat inputs | 11,578 |
| Exact behavior/chat prompted overlap | 800 |
| Unique prompted union | 12,678 |
| Fixed training preflight inputs | 40 |
| Heads across both models | 8 |
| Outer train / original P15 / additional bindings | 3,040 / 2,778 / 262 |
| Additional existing-text reuses / missing texts | 131 / 131 |
| Missing model/input combinations across both models | 262 |
| Separate full-D raw compound rows | 7,728 |

Exact missing FP16 payloads are Qwen 26,292,224 and Llama 34,340,864 bytes. Exact two-layer chat payloads are Qwen 165,982,208 and Llama 189,693,952 bytes. Token metadata sizes, reserves, second copies, retained cache/raw occupancy and usable quota are separate; payload arithmetic is not a quota guarantee. No numerical hidden-state reuse across paths is presumed.

## Validation and preservation

Actual host: `MacBook-Pro-7.local`, macOS. Actual interpreter: `/opt/homebrew/opt/python@3.11/bin/python3.11`, Python 3.11.6. All local checks ran serially with one CPU process and BLAS/OpenMP threads set to one. Torch and transformers are absent locally; the generation regression uses explicit synthetic logits and a GenerationMixin-policy fixture, not research weights.

All **27 focused tests passed** in the final recorded run (19.524 seconds reported by unittest). The original 15 behavior tests remain intact. New checks exercise eight nonterminating steps and early native EOS, reject the old fixed-position policy at step two, independently verify score indexing/greedy sequences, preserve failed optimizer attempts, enforce isolated weighting and eight-head identities, keep 800/full-D supports distinct, and exercise actual reviewed shard write/reopen, 131-row resume, quarantine and corruption rejection. Memory evidence is checked before teardown. The shared ledger test charges all three stages. Deterministic inventory and queue reconstructions, scientific hash checks, Python syntax and git diff whitespace checks passed. No saved-score diagnosis was rerun.

The preservation receipt rehashes all 15 original untracked files (13 unrelated plus both distinct checkpoint documents), confirms the original root HEAD `d953a8f489cb3286e67f494069e69bdfb4ccfd48`, both delivery branch tips, unchanged diagnosis files, and unchanged behavior freeze. Historical result receipts remain in place. The October 3 document was read and checked against its title and scope; it was not confused with October 2 Revision 2.

Key unchanged hashes:

| Artifact | SHA-256 |
| --- | --- |
| Approved October 3 DOCX | `f81c2209c03eacf5c6fe5e14729d68069035ea3a08de3d2b9eb273b070c4cae2` |
| Missing-input queue | `069382d42e56109709a4842690c7cb5503031ce15e21b528b0ba497b53e97b6f` |
| Behavior inventory (canonical digest) | `a253e40470c6cb46fb6f8d4fca84160204511fb105b6f124179f22c6c74f722a` |
| Accepted raw contract | `ed15c662289d587cc9328bc43d980b42afe3d39e4dc352e334e0f504ebcd4ec1` |
| Accepted raw manifest (reviewed canonical hash) | `f28e75f5fc95cf3f019f5b47c68c5c2753435854280bfb122f0df22187954445` |
| Ordered outer_train membership (reviewed canonical hash) | `a2b0344d6ca9e5a0d8190a45738b25b3ba3b0d2854aa1aa64f9282b9ea175d61` |

The complete unchanged file-hash inventory is `results/checkpoint_next_integration_fix_20261003/scientific-input-hashes.json`. Compact behavior digests and indented reviewed raw digests retain their distinct original encodings.

## Remaining prerequisites

No Pod was contacted/restarted; no weights downloaded; no model inference, missing extraction, research head fit or E access occurred. There are no completed behavior/chat or missing-raw execution receipts in this delivery. Existing diagnosis evidence is inherited unchanged.

Later execution requires exact-producer authorization, current endpoint/runtime/cache verification, real pinned-token boundary audits, the new-path training checks, measured CUDA memory and timings, actual quota/allocation checks, cap/forecast review including unmeasured allowances, verified supplement acceptance and transfer, and independent downstream reopen. The runtime, GPU peaks, generation lengths/costs, storage quota and transfer speed for these paths are currently **unmeasured**. Synthetic success does not establish pinned-model success. Chat fitting additionally requires an available serialized CPU slot, reviewed raw comparator export and separate bounded authorization, followed by optimizer/score review.

These Prompt 1 controls do not establish prompted constituent availability or causal use. Weakness at two layers does not prove absence. Strong output-margin AUROC, weak forced-choice decisions, valid factual errors and formatting failures stay distinct. Full-bank fitting, any explanatory experiment and E remain outside this delivery.
