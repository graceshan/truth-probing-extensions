# Revision 2 bounded compatibility runner

Implemented on `checkpoint-r2-bridge-runner-20261002` from exactly common commit `a693f0cec29c9d896787ed5dd696aea3053ee5c3`. The persistent worktree is `/Users/apple/projects/checkpoint-r2-bridge-runner-20261002`. This delivery implements and synthetically validates the bounded runner; it does not run real model inference. Production, historical-fit adoption and E scoring remain disabled.

The entry point is `python -B -m src.checkpoint_r2_bridge_runner`. [Runner configuration](../../config/checkpoint_r2/bridge_runner_v1.json) pins the common contract, original protocol, frozen identity list, historical bindings, comparator and Qwen tokenizer receipt. No common-base file is changed. The original comparator remains SHA-256 `b84b14c6b4ee9ed8dcff596429cd47538151c60c7c5e68b6736d8a056c1efa76`; all its exact/zero criteria remain unchanged.

The default `dry-run` mode verifies metadata only. The separate `run` mode requires an explicit `--acknowledge-bounded-inference` flag and later user authorization. There are no options for custom IDs, split, layer subset, model revision, precision, batches, tolerances or production execution. It processes both pinned models in sequence. It does not resume or overwrite a prior output directory.

The input contract contains 180 independent rows and ten separate train-only calibration rows per model. It verifies all 380 ordered model/stage/row bindings, exact texts, IDs, source splits, unique metadata offsets and source-row/tensor paths. It rejects E, changed hashes, reordered/extra IDs, ambiguous bindings and incomplete layer coverage. External sidecars and diagnostic head archives must match their frozen hashes. Actual bounded runs additionally hash each allowlisted tensor, verify its shape/dtype and materialize only the named rows using read-only memory mapping. Full-file hashing is integrity IO, not extraction or evaluation of additional rows.

The two frozen historical LR heads are Qwen saved layer 17/C=1 and Llama saved layer 15/C=10. Their archived coefficients, intercept, classes and layer/C are checked; preprocessing is explicitly the historical **identity/no normalization** operation, bound separately by hash. This does not substitute checkpoint P-standardization or adopt those historical fits. Both historical and fresh selected states receive the same frozen affine diagnostic. Exact per-ID scores and differences are retained solely as compatibility diagnostics; there is no refit, label-based metric or AUROC path.

The backend loads only locally available pinned snapshots:

| Setting | Qwen | Llama |
|---|---|---|
| Model | Qwen/Qwen2.5-7B-Instruct | meta-llama/Llama-3.1-8B-Instruct |
| Revision | `a09a35458c702b33eeacc393d103063234e8bc28` | `0e9e39f249a16976918f6564b8830bc894c89659` |
| Saved layers / width | 28 / 3,584 | 32 / 4,096 |
| Recorded tokenizer class | Qwen2Tokenizer | TokenizersBackend |

Python 3.12.3, Torch 2.11.0, Transformers 5.12.1, NumPy 2.4.6, CUDA 12.8, cuDNN 91900 and NVIDIA A40 are the explicit initial execution configuration. No version/GPU fallback is permitted. The backend requires pinned config/tokenizer file hashes, tokenizer class/backend hash, exact resolved snapshot revision and safetensors index/shards. It records weight-file hashes and actual dependency versions, driver, hardware, model implementation hash and execution flags. Auxiliary dependency versions are not proof of historical execution; no historical lock is invented.

Extraction uses exact raw text with special tokens, no chat/truncation, SDPA, BF16 compute, FP16 saved states, `use_cache=False`, `logits_to_keep=1` and all HF hidden states `[1:]`. Embeddings are excluded; the final saved state is post-final RMSNorm. The selected token is the last real token, with attention-mask-derived semantic position IDs and padded positions zero. Every forward records actual token IDs, masks, position IDs, physical and semantic readout positions, layer convention, precision and batch/padding settings. Compute states are saved as float32, an exact widening of BF16, so repeatability can check compute as well as FP16 storage.

Fresh flags explicitly follow the recorded Qwen settings: TF32 off, highest float32 matmul precision, all three SDPA backends enabled, deterministic-algorithms flag false. Seed zero and one CPU/interop thread are explicit fresh diagnostic settings. Applying this declared configuration to Llama does not establish its unrecorded historical kernel/determinism settings. Unknown historical execution remains unknown even if a future numerical comparison matches.

The fixed stages are calibration baseline (ten unpadded single inputs), same-configuration repeat, batch-two right padding, batch-two left padding, then the 180 unpadded independent inputs. Pairing for padding is consecutive rows in the frozen calibration order. All-padding or noncontiguous masks, incorrect semantic positions, wrong readout positions and pad-token mismatches fail validation. The comparator receives a validated logical token view with padding removed; the original physical views remain in receipts. This explicit normalization tests semantic batch/padding invariance without pretending that the physical padded tensors are identical.

Repeatability/invariance requires identical compute arrays and all strict comparator checks. A calibration failure stops before the independent stage; there is no setting search or tolerance widening. These checks establish only repeatability or the named batch/padding invariance. They do not establish historical cross-configuration compatibility. Only the frozen current configuration is attempted; missing historical hardware/software configurations are not fabricated as a calibration experiment.

Historical per-example token/mask/position records are absent. The historical comparator packet therefore contains explicit `null` sentinels, never tokens copied from fresh extraction. Its contract is the separately recorded historical manifest, not the fresh runtime. The original comparator result is retained verbatim except for JSON-safe serialization. False checks caused by absent records are labeled unavailable evidence, not a demonstrated token discrepancy. Exact numerical agreement yields the runner-level verdict **unverifiable**, never verified reuse. A numerical mismatch is reported as such alongside the provenance gap. There is no CLI option to assert historical execution was verified. A reviewed successor evidence contract would be necessary to change that status.

Two hours of active investigation and one GPU hour per model remain the limits. GPU charging is deliberately conservative: wall time from backend initialization through model teardown, including setup and intervening host work, not just measured kernels. Model preparation/historical integrity IO is separately recorded as setup; active comparison time accumulates across both models. Download and queue time are zero in this offline runner. A separate 30-minute setup timeout prevents indefinite blocked loading and does not enlarge either study cap. An independent supervisor polls every 0.1 seconds and kills the worker process group at the applicable cap; it also kills the child if the supervisor is interrupted. This is bounded software supervision, not a hard real-time OS guarantee.

Fresh output requires an external durable directory, outside the checkout, temporary directories and historical artifact root. The parent directory must already exist. Receipts are published from fsynced temporary files via exclusive atomic links; arrays are published only after complete writing. Immutable numbered events, `failure.json`, `worker_exit.json`, or `watchdog_termination.json` retain partial/failure outcomes. Only this run's `watchdog.json` pointer is replaced atomically. Abandoned `.partial` files are not completed outputs. The destination must support these POSIX publication/fsync semantics; an unsupported object/FUSE mount fails rather than silently weakening them. Validate the durable destination before a later run.

JSON uses `allow_nan=False`. Undefined infinite/NaN diagnostics become `null`, with paths listed in `undefined_nonfinite_paths`; such a document has a `diagnostics` wrapper. This preserves the underlying failed checks and never converts an undefined diagnostic to zero or a pass.

The complete successful bridge would save 190 historical FP16 rows and 220 fresh rows in both FP16 and compute-as-float32 per model. Tensor payload is **170,598,400 bytes Qwen + 222,822,400 bytes Llama = 393,420,800 bytes**, before NPY headers, metadata, scores and event overhead. Reserve at least **1 GiB** of verified durable output space, separately from model weights, historical tensors and temporary setup space. There are 210 forward calls/model (40 calibration sequence evaluations across 30 calls, plus 180 independent calls). No measured GPU throughput or completed bridge results are claimed.

The [successful fresh read-only readiness observation](../../results/checkpoint_r2_bridge_runner_v1/readiness_retry.json) used the existing authorized `root@69.30.85.22:22114` connection with strict host-key checking. The Codex session remained on local macOS, `MacBook-Pro-7.lan`; the remote shell was Linux host `ef7f7534c328`.

| Readiness item | Observation |
|---|---|
| GPU / VRAM | CUDA unavailable, zero devices, no `/dev/nvidia*`, no `nvidia-smi`; VRAM not measurable |
| Driver | `/proc/driver/nvidia/version` reports 570.195.03; this does not establish GPU visibility |
| Packages | Python 3.12.3; Torch 2.8.0+cu128; NumPy 2.1.2; Transformers/tokenizers/safetensors/huggingface-hub absent |
| Compiled runtime | CUDA 12.8, cuDNN 91002; differs from the pinned setup |
| Pinned model directories | Found under searched `/workspace` HF cache roots, but config/tokenizer files and weight indices are zero-byte, hash-mismatching/unreadable; not usable model availability |
| Historical paths | Six expected tensors and their recorded sidecars/heads are present; this readiness check did not reread tensor values or full tensor hashes |
| Storage | `fuse.geesefs`, virtual 1-PiB free report, access flag writable; quota utility absent; practical quota, durability and atomic-publication support unverified |

The first probe encountered an empty weight-index JSON and returned no completed runtime report. Its [failure receipt](../../results/checkpoint_r2_bridge_runner_v1/readiness.json) and checker source are retained. The checker was corrected to record malformed cache entries instead of aborting the whole report; the read-only retry produced the table above. No package installation, download, remote file write, address change or model inference occurred.

GPU visibility, packages, valid pinned model snapshots and a durable writable destination are recoverable prerequisites. Missing historical per-example tokens, unrecorded execution flags and the previously documented producer-code provenance conflicts are different evidence gaps. A working GPU would not automatically resolve them. The runner never admits a historical fit or claims verified compatibility from tensor agreement alone.

Local validation passed **166 tests**, including 31 new runner cases and relevant existing common/input/selector/atomic/compound extraction suites. After the conservative lifetime cap, score-receipt and CLI refinements, 33 runner tests passed. The additional CLI cases verify macOS temporary-path handling in metadata-only mode and explicit inference acknowledgment. Tests use synthetic backends and small tensor fixtures, including actual BF16 tensor readout operations; no real model was loaded. They cover frozen hashes/identity ordering, E/extra/ambiguous rejection, all-layer readout, both padding sides, numerical mismatches, missing history, head/preprocessing hashes, valid JSON for undefined relative errors, failure receipts, exclusive output publication and process-level timeout killing.

The [final metadata-only dry run](../../results/checkpoint_r2_bridge_runner_v1/metadata_dry_run_final.json) validated 380 ordered bindings, six metadata files and 13 historical metadata/head hashes. It read no activation values. A repeated dry run was byte-identical. The original common validator passed with all memberships and 1,149 lineage files preserved; the final preservation receipt additionally checks every file at the exact common SHA plus the supplied DOCX and 13 unrelated original-worktree files. The earlier development dry-run receipt is retained separately; `metadata_dry_run_final.json` binds the final runner configuration.

Run the local checks with fresh receipt paths:

```bash
cd /Users/apple/projects/checkpoint-r2-bridge-runner-20261002
/tmp/clean-extraction-venv/bin/python -B -m src.checkpoint_r2_bridge_runner --mode dry-run \
  --historical-root /Users/apple/truth-probing-backups/20261001T164224Z/restore_t1_staging_20261001T164224Z_kgkt5sy7/files/workspace/truth-probing-artifacts \
  --output /tmp/r2-bridge-metadata-review.json
/tmp/clean-extraction-venv/bin/python -B -m pytest -q -p no:cacheprovider \
  tests/test_checkpoint_r2_bridge_runner.py tests/test_checkpoint_r2_common_v1.py \
  tests/test_checkpoint_r2_inputs.py tests/test_checkpoint_r2_selection_v1.py \
  tests/test_clean_atomic_extraction.py tests/test_pinned_compound_extraction.py
/tmp/clean-extraction-venv/bin/python -B scripts/68_validate_checkpoint_r2_common.py \
  --original-worktree /Users/apple/projects/truth-probing-extensions
# Optional later read-only refresh, still run from the local Mac:
/tmp/clean-extraction-venv/bin/python -B scripts/checkpoint_r2_bridge_readiness.py \
  --output /tmp/r2-bridge-readiness-review.json
git diff --check
```

The following setup/run commands are **for review only and were not executed**. First restore a GPU-enabled A40 runtime with the exact versions above and a reviewed dependency lock. No complete historical wheel/dependency lock is available in this package; do not replace that gap with an unpinned `pip install -U`. An operator-provided hash-locked offline wheel bundle could be installed only after separate authorization:

```bash
# GPU host only, after its identity, approved lock and storage are verified:
/opt/bridge-r2/bin/python -m pip install --no-index --require-hashes \
  --find-links /approved/bridge-r2-wheels -r /approved/bridge-r2-runtime.lock
# The /approved paths are external prerequisites, not files supplied by this task.
```

Restore or download only the pinned revisions into a **fresh** approved cache, after separate authorization. Do not repair or overwrite the observed empty snapshots in place. For a provisioned `hf` CLI, the revision-specific commands are:

```bash
hf download Qwen/Qwen2.5-7B-Instruct \
  --revision a09a35458c702b33eeacc393d103063234e8bc28 --cache-dir /approved/bridge-r2-hf-cache
hf download meta-llama/Llama-3.1-8B-Instruct \
  --revision 0e9e39f249a16976918f6564b8830bc894c89659 --cache-dir /approved/bridge-r2-hf-cache
```

Llama access authorization, exact runtime provisioning, practical quota and durable POSIX output remain prerequisites. Provision the reviewed runner commit on the GPU host without changing any active checkout. Then review the following bounded invocation; it performs real bridge inference only when later authorized:

```bash
# GPU host, from the checkout of this runner's reviewed delivery commit:
export HF_HUB_CACHE=/approved/bridge-r2-hf-cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
/opt/bridge-r2/bin/python -B -m src.checkpoint_r2_bridge_runner --mode dry-run \
  --historical-root /workspace/truth-probing-artifacts \
  --output /approved/bridge-r2-runs/metadata-before-run.json
/opt/bridge-r2/bin/python -B -m src.checkpoint_r2_bridge_runner --mode run \
  --acknowledge-bounded-inference \
  --historical-root /workspace/truth-probing-artifacts \
  --output /approved/bridge-r2-runs/reviewed-run-001
```

The parent `/approved/bridge-r2-runs` must be an existing verified durable destination and the run directory must not exist. No command above enables production, historical-fit adoption or E scoring. Do not rerun a failed/capped bridge into a fresh directory to reset its investigation budget; review its failure/remaining budget first. This task ends at implementation, local synthetic validation and read-only readiness.
