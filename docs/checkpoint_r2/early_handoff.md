# Revision 2 early integration handoff

The approved A15 checkpoint inputs are frozen and locally validated. The exact reviewed lineages are merged. **Fresh extraction compatibility is not established:** SSH succeeds, but the documented remote host has no visible GPU and lacks the required Transformers runtime. No extraction, activation-value read, research score read, production fit, behavior generation or E prediction occurred.

The integrated base is `0147f70197e6effcd30849d66b13493d6ef0934e`, with parents from the exact approved mac 1 and reviewed mac 2 lineages. The delivery commit is the commit containing this report (`git log -1 --format=%H -- docs/checkpoint_r2/early_handoff.md`). [Protocol/configuration](protocol_amendment.md), [input/code SHA-256 manifest](../../data/checkpoint_r2_v1/manifest.json), [exposure hashes](../../results/checkpoint_r2_preflight_v1/exposure_manifest.json) and [validation receipt](../../results/checkpoint_r2_preflight_v1/validation.json) provide the reproducible handoff. No mac 2 checkpoint preparation code is represented as integrated.

Local context: macOS/Darwin, `MacBook-Pro-7.lan`; original worktree remains on `t2-research-discussion-packet-20261001` at `d953a8f`. Its 13 unrelated untracked files and the supplied Revision 2 DOCX are recorded separately and preserved byte-for-byte. All task changes live on `checkpoint-r2-integration-20261002` in `/private/tmp/checkpoint-r2-integration-20261002`. No other worktree was modified. The final preservation receipt binds the original files and checks historical tracked paths against the merged base.

| Check | Outcome |
|---|---|
| Ordered v4→v5 train/D, P15, A15, T_C and sampler equivalence | Passed; counts 3,040 / 1,012 / 2,778 / 75 / 100 / 700 |
| Exact fact/evidence/person bindings, duplicate restrictions and paired quarantine | Passed |
| Three B25 samples, five folds, source split and fixed pair lists | Passed; 25 pairs/sample; 500 control/refit, 500 source-fit and 30 source-validation pairs |
| D recovery and held-out wording preservation | Passed; 483 retained pairs, 7,728 bare rows |
| Historical bridge metadata mapping | Passed; 380 exact model/stage/row bindings |
| SSH, remote hostname and expected tensor-file presence | Passed; `ef7f7534c328`, Linux; six expected files at recorded sizes |
| Remote GPU/runtime prerequisite | Failed readiness: CUDA false, no NVIDIA devices or `nvidia-smi`; Transformers absent, Torch 2.8.0 versus recorded 2.11.0 |
| Fresh hidden states, frozen-head deltas, repeatability, batch/padding invariance | Pending; extraction never started |
| Historical per-example token replay and unrecorded execution details | Unverifiable from available records |
| Offline current Qwen tokenization and exact answer boundary | Passed; Llama tokenizer unavailable locally/remotely |

[Bridge receipt](../../results/checkpoint_r2_preflight_v1/bridge_receipt.json), [remote observation](../../results/checkpoint_r2_preflight_v1/remote_preflight.json), [historical mappings/contracts](../../results/checkpoint_r2_preflight_v1/historical_bridge_bindings.json) and [frozen identities](../../data/checkpoint_r2_v1/bridge_identities.json) separate these findings. File presence and previous physical hashes establish identity evidence; they do not establish fresh representation compatibility. This is neither SSH authentication failure nor a cache-corruption finding.

The independent bridge has 180 inputs/model: 20 short/long admitted P15/atomic-D rows plus ten complete retained D pairs (160 rows), covering all topics, operators, truth cells and orders. Ten disjoint train-row IDs/model form calibration. Exact historical row offsets are joined by identity/text, not assumed from source order. Canonical historical Qwen L17/C1 and Llama L15/C10 heads are hash-bound for compatibility diagnostics only; they are not new checkpoint candidate fits.

The current contract records pinned model/tokenizer revisions, BF16 compute, FP16 saved states, raw exact statements, special tokens, no truncation/chat, explicit semantic positions, SDPA, batch one/no padding, `use_cache=False`, HF hidden_states[1:] and the last real token. Saved layers are zero-based; the final layer is post-final RMSNorm. The historical receipts record NVIDIA A40/CUDA 12.8, Torch 2.11.0 and Transformers 5.12.1. Qwen records TF32 off, highest float32 matmul precision, all three SDPA backends enabled and deterministic algorithms false. Missing historical settings remain unknown; do not infer a deterministic historical execution merely from a successful repeat today.

The frozen acceptance rule is exact token/position/mask/shape equality and zero hidden-state and frozen-score discrepancy **only when identical deterministic execution is independently verifiable**. The pure comparator reports per-layer absolute errors and relative errors; it does not calculate AUROC. Identical arrays with unverified historical execution remain “unverifiable.” No approximate tolerance is approved. Any nonzero tolerance requires relevant training-only cross-configuration calibration and a separately frozen amendment before inspecting independent historical discrepancies. Single-configuration repeats establish repeatability only.

Future calibration should repeat the unpadded baseline and test batch-two left/right padding with explicit semantic positions on the frozen train identities; cover actual hardware/kernel/software differences when both configurations exist. The tracked extraction implementations are `src/clean_atomic_extraction.py`, `src/pinned_compound_extraction.py` and `src/llama_replication_extraction.py`. Their old production entry points target historical cohorts and must **not** be launched on this new manifest. A bounded GPU producer for these frozen IDs still needs wiring and review after runtime restoration. The pure comparator and historical index maps are ready; no production extraction runner is claimed ready.

The diagnostic cap remains two hours active investigation and one GPU hour/model, stopping at the applicable limit. Current GPU time and numerical-comparison investigation time are zero. The timed remote setup check took 5.656 seconds; two preliminary read-only capability checks preceded it. No download or queue time occurred. Runtime failure stopped this preflight; it did not trigger broad regeneration.

No historical representation is newly certified for mixing with fresh states. Existing within-cache results retain their recorded historical scope. If compatibility remains unverifiable, regenerate the smallest complete affected comparison for each affected model: P15/preprocessing, atomic D, allocated A, required T_C, compound D and required wording, then rebuild dependent heads. Do not combine new compound states with incompatible old P states. E, unrelated historical families and chat are outside that fallback unless separately required. The full checkpoint raw union below is an upper scope; B25/control without the constituent source diagnostic needs at most 31,987 unique raw inputs/model under the same inventory contract. Neither fallback is launched.

| Inventory, per model | Logical bindings | Unique inputs | Current Qwen input tokens | Saved-state storage |
|---|---:|---:|---:|---|
| Raw checkpoint union | 41,490 | 35,843 | 723,409 | Qwen 7,193,833,472 bytes; Llama 9,396,027,392 bytes, all layers FP16 |
| Behavior, both prompts | 2,000 | 1,900 | 151,466 | No hidden-state storage required unless sharing Prompt 1 passes |
| Prompt 1 matched chat | 11,578 | 11,578 | 894,898 | Qwen 82,991,104–165,982,208 bytes; Llama 94,846,976–189,693,952 bytes, one/two layers |

The behavior count is 4,000 logical model/prompt/input bindings across both models, reduced to 3,800 distinct combinations by exact text identity. Qwen's two forced-choice candidates require 3,800 continuation tokens, or 306,732 total candidate-scoring tokens without prefix reuse. Its greedy cap is 15,200 generated tokens; the two-model cap is 30,400. Candidate scoring and generation are additional work, not included in an input-only forward estimate. Output IDs at four bytes/token would require at most 121,600 bytes for generated tokens across both models, plus candidate likelihoods, strings, manifests and format diagnostics; no outputs yet exist.

The chat logical total is 23,156 across both models. There are 800 identical Prompt 1 compound messages/model shared with behavior; actual forward reuse requires consistent instrumentation and is not automatically credited. Each record retains its kind, group, source-row/example/fact/pair identity and exact text hash. Deduplication concerns computation only, not exposure or independent statistical support.

[Resource scenarios](../../data/checkpoint_r2_v1/resource_estimates.json) and [Qwen tokenizer receipt](../../results/checkpoint_r2_preflight_v1/qwen_tokens/receipt.json) distinguish estimates from measurements. At **hypothetical** 5/20/50 inputs per second, raw extraction takes 7,168.6/1,792.15/716.86 seconds/model; chat takes 2,315.6/578.9/231.56 seconds/model. These are sensitivity calculations, not observed throughput or scheduling promises. Behavior's corresponding 380/95/38 seconds/model describe prompt processing only and exclude its two candidate continuations and generation. Llama exact tokens remain pending; the UTF-8-based planning ranges are explicitly heuristic, not tokenizer measurements. Runtime must be measured on the actual GPU before launch.

Both all-layer raw outputs total 16,589,860,864 bytes before headers/manifests. A staging-plus-durable-copy allowance doubles that, before model weights, temporary workspace and chat output. The remote filesystem reports one PiB free through its volume interface; that is not a verified account quota or practical storage allocation. Confirm quota, durable destination and weight availability. No GPU benchmark, download or broad tensor transfer was performed.

Mac 2 must fetch the pushed integration SHA, reconcile its preparation against these exact manifests/configuration, and hand back a reviewed code SHA for objective/eligibility/bank-reuse/result-schema integration. Representative CPU costs, convergence and initial C_clean viability remain mac 2 dependencies. This task did not run them or inspect ongoing work. Human confidence review also remains disclosed as pending, not a new indefinite factual audit.

The next executable local step is mac 2's manifest/config cross-check and reviewed integration. The next GPU step requires a GPU-enabled authenticated host, pinned model/tokenizer weights and recorded runtime, verified storage, then a **bounded** producer wired only to the frozen calibration/bridge IDs. Run the staged repeatability/compatibility checks under the cap, preserve unresolved provenance, and hold the early review. Do not launch production, fallback regeneration or E scoring automatically.

Run local checks from the checkpoint worktree:

```bash
cd /private/tmp/checkpoint-r2-integration-20261002
/tmp/clean-extraction-venv/bin/python -B -m src.checkpoint_r2_inputs
/tmp/clean-extraction-venv/bin/python -B -m src.heldout_and_preparation
/tmp/clean-extraction-venv/bin/python -B -m src.final_partition_fact_audit
/tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider tests/test_checkpoint_r2_inputs.py tests/test_selection_repair_capacity_p10_v2.py tests/test_selection_repair_capacity_p10_validation_v2.py tests/test_selection_repair_precision_proxy_v1.py tests/test_heldout_and_wording.py tests/test_selection_repair_objectives.py
git diff --check
```

These yielded 86 passing tests plus 10 subtests, deterministic input regeneration, and successful historical v5/D/E data-only validators. The E validator performs no predictions or scoring. Use `--write` on the input module only to materialize a missing package; existing differing bytes are rejected. The read-only capability recheck writes a **new** receipt:

```bash
/tmp/clean-extraction-venv/bin/python -B scripts/checkpoint_r2_remote_preflight.py --output /tmp/checkpoint-r2-remote-recheck.json
/tmp/clean-extraction-venv/bin/python -B scripts/checkpoint_r2_bind_historical_bridge.py --payload-root /Users/apple/truth-probing-backups/20261001T164224Z/restore_t1_staging_20261001T164224Z_kgkt5sy7/files/workspace/truth-probing-artifacts --output /tmp/checkpoint-r2-historical-recheck.json
/tmp/clean-extraction-venv/bin/python -B scripts/checkpoint_r2_token_inventory.py --tokenizer /Users/apple/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28 --output /tmp/checkpoint-r2-qwen-recheck
```

Use fresh output paths on subsequent invocations; these tools refuse overwrites. The tokenizer-only preparation loads no model weights and produces no likelihoods. It preserves the native Qwen template's default system preamble without adding an application system message. The complete scientific checkpoint, actual B25 precision, all fit panels, behavioral diagnosis and research decision remain unrun.
