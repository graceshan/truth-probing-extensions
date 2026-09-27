# Clean development/validation activation extraction

Update: full extraction now also requires the
[historical atomic compatibility gate](historical_atomic_compatibility.md).
`--mode smoke` runs both gates and writes a passing immutable revision receipt.
Run those GPU smoke commands first; full extraction loads the exact revisions
from that receipt and reruns both gates. The 72-test results below describe the
initial implementation; the added compatibility suite is reported separately.

Implemented extraction correctness and provenance only. The 8,384-row pass has
**not** been run. No probe was fitted or scored, no test data was accessed, and
no existing activation checkpoint was modified. The default CLI mode is a
read-only plan; creating arrays requires `--mode extract`.

The local suite passes **72 tests**. Pretrained Qwen2.5-7B/Qwen3-8B GPU smoke
tests remain **unrun**: this machine has no CUDA and neither model's weights.
The local smoke results below use the actual pinned HF architectures, with
random weights and reduced width. They establish implementation behavior, not
full-weight numerical equivalence. Each future extraction command first loads
its model once, runs the tiny full-weight smoke, and refuses to allocate the
benchmark array if that smoke fails.

Audit evidence and machine-readable numerical results are under
`results/clean_protocol/extraction_diagnostics/`. The source audit records
hashes of the pre-edit extractor, installed model implementations, HF output
capture implementation, model configs, benchmark and frozen selection files.

1. **Historical extraction audit, completed before editing.**

   `src/extract.py` previously tokenized raw statements with
   `padding=batch_size > 1`, omitted explicit position IDs, selected
   `attention_mask.sum(dim=1)-1`, and saved `out.hidden_states[1:]` as float16.
   Its default is `batch_size=1`; no padding is introduced in that mode.
   `git show 409759f:src/extract.py` records the same convention and explicitly
   describes single-statement extraction as the default GPU workflow.

   Qwen3's `acts/qwen3_8b/extraction_summary.json` explicitly records
   `batch_size=1`, `add_special_tokens=true`, no chat template, and embedding
   removal. That recorded run is unaffected by the padding bug.

   Qwen2.5's historical code supports the same unpadded default, which is
   unaffected. **No per-run Qwen2.5 manifest was found, so batch size 1 for every
   existing checkpoint cannot be independently certified.** The old Colab
   notebook uses a stale extractor signature and is not proof of the production
   invocation. Current cached Qwen2.5 tokenization defaults to right padding,
   but that also does not prove historical run settings.

   Neither historical extraction record establishes the exact original model
   commit or all runtime settings. The new pinned revisions provide repeatable
   future extraction; they do not retroactively prove the old weight revisions.

2. **Installed/pinned implementation and unchanged layer mapping.**

   Inspected `transformers==5.12.1`, with `torch==2.11.0` installed in a temporary
   audit environment. Qwen2.5 uses `models/qwen2/modeling_qwen2.py`; Qwen3 uses
   `models/qwen3/modeling_qwen3.py`. Both models use `@capture_outputs` and
   `_can_record_outputs["hidden_states"]` targeting their decoder layer class.

   `utils/output_capturing.py:112` captures the first block input (the embedding
   output); it then collects block outputs. At lines 259–267, the default
   `tie_last_hidden_states=True` replaces the final collected block output with
   `outputs.last_hidden_state`. Qwen2 line 409 and Qwen3 line 434 apply the
   model's final RMSNorm before returning that last hidden state. Hook-based
   tests check every entry, including the final pre/post-normalization distinction.

   | Model | Blocks | Hidden size | Saved axis | Frozen selected layer |
   | --- | ---: | ---: | --- | --- |
   | Qwen/Qwen2.5-7B-Instruct | 28 | 3,584 | HF entries 1..28 | saved 17 = HF 18 = block 18 output |
   | Qwen/Qwen3-8B | 36 | 4,096 | HF entries 1..36 | saved 28 = HF 29 = block 29 output |

   HF entry 0 is the embedding state and is dropped. Saved layer 0 is HF entry
   1, the first block's output. Intermediate saved entries are residual outputs
   before the next block. **Saved layer 27 for Qwen2.5 and 35 for Qwen3 are after
   final RMSNorm.** The old docstring's blanket `resid_post` claim was inaccurate
   for that final entry; the array convention itself is unchanged.

   The frozen `atomic_probes_converged/{qwen25_7b,qwen3_8b}/selection.json`
   files select 17 and 28. `src/clean_atomic_probes.py::Partition.matrix` indexes
   `acts[rows, layer, :]` directly, with no layer offset or renormalization.
   No probe fitting, coefficient changes, or score computations are involved.

3. **Token and position algorithms.**

   The old count-based index is correct only for a contiguous real-token prefix.
   `[0,0,1,1,1]` produced index 2 (the first real token), instead of index 4.
   With more left padding it can select a padding token outright.

   `last_real_token_indices` now computes
   `max(where(mask == 1, arange(sequence_length), -1))` independently for each
   row. It rejects empty tensors, nonbinary masks, and all-padding rows.

   Qwen2 lines 372–375 and Qwen3 lines 397–400 default to
   `arange(padded_sequence_length) + past_seen_tokens`, broadcast over rows.
   Their direct forwards do not derive real-token numbering from the mask.
   Extraction does not use `generate()` or its input-preparation logic.

   Clean extraction explicitly supplies
   `(mask.long().cumsum(-1)-1).masked_fill(mask == 0, 0)` as `position_ids`.
   Every real sequence receives 0..length-1, matching the unpadded reference;
   padding positions receive zero and remain masked. Explicit IDs are needed
   to guarantee those semantics for left padding. A uniform RoPE position
   shift can sometimes leave default-padded activations nearly invariant, but
   that is not a substitute for matching the reference token positions.

   Tokenization uses raw `metadata.statement`, `add_special_tokens=True`,
   `return_tensors="pt"`, `return_attention_mask=True`, `truncation=False`, and
   `padding=(batch_size > 1)`. No max-length truncation is applied; over-context
   inputs raise. No chat template is used, so Qwen3 thinking mode is not applicable.
   Clean calls disable the unused KV cache and compute only one logit position
   using `logits_to_keep=1`; every hidden-state layer/token is still returned.
   Tests independently compare against the old full-logit/default-cache path.

4. **Smoke evidence and tolerances.**

   Four synthetic statements of different token lengths are processed singly
   without explicit IDs (legacy reference), singly with explicit IDs, and in
   right- and left-padded batches. Every saved layer is compared, both before
   storage conversion and after float16 conversion. JSON reports include max
   and mean absolute error globally, by layer, and by statement.

   Local models use 28/36 layers, hidden size 32, a synthetic WordLevel tokenizer,
   seed 123, CPU, and the installed Qwen2/Qwen3 implementation. Both SDPA and
   eager attention were tested in FP32 and BF16. These are **not pretrained
   model measurements**. For the SDPA FP32 compute readouts:

   | Architecture | Padding | Max absolute difference | Mean absolute difference |
   | --- | --- | ---: | ---: |
   | Qwen2 | right | 1.0728836e-6 | 1.0848432e-8 |
   | Qwen2 | left | 9.5367432e-7 | 1.1995496e-8 |
   | Qwen3 | right | 1.0728836e-6 | 1.3286475e-8 |
   | Qwen3 | left | 1.1920929e-6 | 1.3146838e-8 |

   SDPA FP32 readouts after float16 storage conversion:

   | Architecture | Padding | Max absolute difference | Mean absolute difference |
   | --- | --- | ---: | ---: |
   | Qwen2 | right | 3.0517578e-5 | 8.5315799e-9 |
   | Qwen2 | left | 3.0517578e-5 | 8.5482110e-9 |
   | Qwen3 | right | 1.5258789e-5 | 3.9063810e-9 |
   | Qwen3 | left | 9.5367432e-7 | 5.4327148e-10 |

   BF16 CPU comparisons are exactly zero for both padding sides and attention
   implementations. The worst FP32 eager compute difference is 1.5497208e-6;
   the largest eager mean difference is 1.2869671e-8. All clean unpadded
   comparisons against the legacy path are exactly equal in these tests.

   Tolerances use each reference layer's RMS scale, preventing the final norm's
   different scale from hiding intermediate-layer errors. The max/mean limits
   are `1e-6 + max_fraction*RMS` and `1e-7 + mean_fraction*RMS`:

   | Compute dtype | Max fraction | Mean fraction |
   | --- | ---: | ---: |
   | FP32 | 1e-4 | 1e-5 |
   | FP16 | 5e-3 | 1e-3 |
   | BF16 | 2e-2 | 3e-3 |

   BF16 unit roundoff is about 0.0039; the limits allow modest accumulation from
   different matrix-kernel shapes without accepting wholesale token mistakes.
   Float16 storage checks use at least the FP16 limits because values straddling
   a rounding boundary can differ by a float16 ULP even with tiny FP32 error.
   Both max and mean limits must pass on **every** layer. These are conservative
   initial GPU gates, not measured GPU noise bounds. A failing GPU run saves
   its diagnostics and stops; do not widen the tolerance without diagnosis.

5. **Row alignment, provenance and resume.**

   The sidecar is the entire source CSV copied byte-for-byte, preserving all
   25 columns, row order, quoting, and field text. Extraction takes statements
   from the exact row slice that is committed with that activation chunk.
   Required columns include all ten requested identity/label fields; duplicate
   IDs, empty required fields and non-development/validation splits fail.

   SHA-256 hashes cover benchmark bytes, ordered example IDs, ordered statements,
   and sidecar bytes. Ordered hashes encode a compact UTF-8 JSON array with
   `ensure_ascii=False`, so delimiters and newlines are unambiguous. The current
   benchmark and expected byte-identical sidecar hash is:
   `96e96515e780086065694155e3061bdfe6f4bb80af062a63c2deecd6bfad6f94`.

   `extraction_manifest.json` records the requested model/tokenizer IDs and
   immutable revision pins, resolved commits when exposed, config and config
   hash, model identity hash, tokenizer backend hash/class/vocabulary/special
   tokens/padding/truncation settings, exact prompt and call options, complete
   layer mapping, normalization/index/position algorithms, compute/storage
   dtypes, device/GPU/versions/attention backend/numeric settings, data hashes
   and expected shape, source-code hashes, timestamp, and smoke report. Code
   provenance records Git HEAD and dirty status; `git_commit_sha` is null for a
   dirty tree, while implementation hashes still identify the executed code.
   Authentication kwargs and environment variables are never serialized.

   A nonblocking OS lock prevents concurrent writers. New runs refuse nonempty
   output directories. `activations.partial.npy` is a float16 NPY memmap.
   Each sequential batch is flushed and fsynced before an atomic `progress.json`
   update commits its contiguous range, activation hash and ordered row hashes.
   Completed chunks are not rewritten. Resume validates the complete extraction
   identity, sidecar, array header, all committed chunk hashes and row coverage.
   Changing inputs, model/config, tokenizer, dtype, batch size, backend or
   implementation fails. Timestamps and unrelated Git changes alone do not.

   A crash between data flush and journal commit can require recomputing that
   **uncommitted** tail. Exactly-once refers to committed row coverage: no gap,
   overlap or duplicate committed row is accepted. Finalization rechecks source,
   sidecar, all chunks and full coverage, renames to `activations.npy`, and
   records a full-file hash in `progress.json`. A crash after the rename but
   before the final journal update is recoverable. A completed checkpoint is
   verified before reuse. Keep all four data/provenance files together.

6. **Planned output and resource estimates.**

   Both model directories are under
   `acts/clean_protocol/entity_disjoint/development_validation_v1/`:

   | Subdirectory | Final shape | Float16 payload | GiB |
   | --- | --- | ---: | ---: |
   | `qwen2_5_7b/` | `[8384, 28, 3584]` | 1,682,702,336 bytes | 1.56714 |
   | `qwen3_8b/` | `[8384, 36, 4096]` | 2,472,542,208 bytes | 2.30273 |

   Add a small NPY header (normally 128 bytes), `metadata.csv`,
   `extraction_manifest.json`, `progress.json`, and `.writer.lock`. The partial
   array is renamed on completion, not duplicated. No benchmark activation
   arrays or model output directories were created during this task.

   Recommend starting at **batch size 4 for each model on a 24 GiB or larger
   CUDA GPU**, with BF16 and SDPA, one model loaded at a time. This is a
   conservative recommendation, not a measured GPU memory/performance result.
   The cached Qwen2.5 tokenizer gives 12–29 tokens for this benchmark. GPU
   memory availability and Qwen3 token lengths should still be checked on the
   target host; higher batches need their own smoke/memory validation.

7. **Commands for later use — not launched here.**

   Install the repository's pinned requirements in the GPU environment first.
   The CLI enforces Torch 2.11.0 and Transformers 5.12.1. Model/tokenizer commits
   are pinned in `MODELS`: Qwen2.5
   `a09a35458c702b33eeacc393d103063234e8bc28`, Qwen3
   `b968826d9c46dd6066d109eabc6255188de91218`.

   Full-weight smoke only, with no benchmark arrays:

   ```bash
   python scripts/28_extract_clean_activations.py --model qwen2_5_7b --mode smoke --device cuda --dtype bfloat16 --batch-size 4 --padding-side right --attention sdpa
   python scripts/28_extract_clean_activations.py --model qwen3_8b --mode smoke --device cuda --dtype bfloat16 --batch-size 4 --padding-side right --attention sdpa
   ```

   Future full extraction, with an integrated smoke gate before array allocation:

   ```bash
   python scripts/28_extract_clean_activations.py --model qwen2_5_7b --mode extract --device cuda --dtype bfloat16 --batch-size 4 --padding-side right --attention sdpa
   python scripts/28_extract_clean_activations.py --model qwen3_8b --mode extract --device cuda --dtype bfloat16 --batch-size 4 --padding-side right --attention sdpa
   ```

   For an interrupted run, repeat its exact command with `--resume`. Do not
   change the batch size, dtype, padding side or attention implementation.
   Omitting `--mode` performs only the plan and does not load weights.

   Local extraction tests (install `pytest` separately if needed):

   ```bash
   python -m pytest -q tests/test_clean_extraction.py
   ```

   The executed suite includes the twelve requested categories plus invalid
   masks, position-ID interception, every-layer norm mapping, independent legacy
   reference equality, chunk corruption, overlapping journal ranges, writer
   locking, partial finalization, crash recovery, manifest schema, credential
   exclusion and first-uncompleted-row resume. No broader probe tests were run,
   because they would fit or score probes outside this task's authorized scope.
