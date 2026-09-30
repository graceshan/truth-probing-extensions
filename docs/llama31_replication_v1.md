# Llama-3.1 cross-family replication v1

This implementation adds the first independent model-family replication to the
clean protocol. It does not inspect new scientific results. The Qwen findings
supplied in the task motivate the replication; they are not inputs to Llama
layer/C selection. All validation here uses synthetic data and a tiny randomly
initialized Llama, without downloading pretrained weights.

## Design and reuse

Use **meta-llama/Llama-3.1-8B-Instruct**. This matches the instruction-tuned status
of canonical Qwen2.5; both receive raw statements without chat wrappers. There is
no identified methodological reason to switch to the base checkpoint. Access to
this gated repository and its license must be available on the production host;
an access failure stops the run and must not cause a silent base-model fallback.

The smallest safe extension is a model-specific adapter, not edits to Qwen's
frozen pipeline. Existing `clean_extraction.ExtractionWriter` handles durable
arrays, sidecars and contiguous journals. Existing hidden-state indexing and
last-real-token helpers are reused. `clean_atomic_probes.fit_and_save` and its
selection/convergence primitives are reused unchanged, with a separate loader
that has only TRAIN and VALIDATION. `MetricPlan`, `EntityBootstrap`, `AUC`,
`interval`, structural metadata checks and isolated composition formulas are
reused unchanged. No historical compatibility gate, historical activation cache,
Qwen probe, or Qwen scientific score file is a Llama input.

The existing generic extraction script supports Qwen/Qwen3 and assumes historical
compatibility and padding checks. It is therefore unsuitable as Llama's entry
point. The new adapter reuses its writer, not that orchestration. The old generic
atomic loader has a gated test interface; the Llama loader does not expose it.

New files are the five `src/llama_replication_*.py` modules, the single staged
`scripts/45_llama_replication.py` entry point, synthetic tests, and this document.
Canonical Qwen code, specs, caches and results are unchanged.

## Revision pinning and representation

No Llama revision SHA has been guessed or resolved in this local task. The
metadata-only `pin` stage requires an explicit 40-character Hugging Face commit
SHA, checks that config and tokenizer files resolve to that exact snapshot, and
records their measured hashes/sizes. `main`, branch names and tags are rejected.
It downloads only model config/tokenizer resources, not weight shards. Review
and commit `config/clean_protocol/llama31_representation_v1.json` before any
full-weight smoke/extraction. All later commands obtain their revision from this
pin, with no revision override.

The model-specific representation records:

| Field | Contract |
|---|---|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Model/tokenizer revision | Same explicitly supplied immutable HF commit |
| Architecture | Discovered config must be `LlamaForCausalLM`, model type `llama` |
| Shape | Config must report 32 layers, width 4096 |
| Input | Exact existing raw statement; no chat template |
| Special tokens | `add_special_tokens=True`; actual BOS/EOS configuration and tokenizer backend hash recorded |
| Truncation/padding | Both disabled; reject context overflow or any padding |
| Batch | Exactly one statement |
| Arithmetic/storage | BF16 CUDA / float16 C-order NPY |
| Attention/cache | SDPA / `use_cache=False` |
| Positions | Explicit `arange(sequence_length)`, gated against native Llama positions |
| Readout | Greatest occupied token index |
| Layers | HF `hidden_states[1:]`; embedding excluded; last entry after final RMSNorm |
| Runtime | Torch 2.11.0, Transformers 5.12.1; actual versions, CUDA, GPU, Python, NumPy and implementation-source hash recorded |

Inspection of the installed Llama implementation confirmed that absent position
IDs use `arange(sequence_length) + past_seen_tokens`. With no cache, a single
unpadded row has exactly the intended 0..L-1 positions. This is verified again
on the live model: explicit positions/reduced logits must match native default
positions/full logits exactly, and repeated forwards must match exactly. A tiny
random Llama exercises this behavior locally; it does not replace the live GPU
gate. Model/tokenizer source identities are checked before loading weights.

The representation fingerprint is Llama-specific and measured, never the Qwen
fingerprint. Runtime/GPU details are in extraction manifests. Transfer extraction
must reproduce four saved Llama atomic readouts exactly in float16 and match the
same representation pin.

## Exact data reuse and storage

Atomic inputs use the existing approved `allowed_atomic_rows.csv` loader and
ordered input digest `1faf186cb28b48311e85fd41921316fad9d06b3d28c9756aa5e8b381a950f1d6`.
No original mixed sources or entity/test manifests are reopened. Expected arrays:

- TRAIN: `[3144,32,4096]`, float16, 0.768 GiB payload.
- VALIDATION: `[1040,32,4096]`, float16, 0.254 GiB payload.

There is no TEST partition interface. Slices are mapped read-only, only the
requested layer is materialized, numerical fitting inputs are float64, and
activation-row read counts are recorded.

Transfer inputs are byte-verified against the existing frozen Priority-2 spec:

| Condition | Rows |
|---|---:|
| `raw_reference` | 8,384 |
| `or_explicit_or_both_v1` | 4,192 |
| `juxtaposition_v1` | 4,192 |
| `isolated_constituents_v1` | 524 |
| Total | **17,292** |

Raw statements come from the canonical development/validation CSV. Other
statements are a filtered projection of the existing Priority-2 `statements.csv`;
no variant is regenerated or resampled. Constituent/variant maps and metadata
are reused byte-for-byte. All 524 isolated facts are freshly evaluated by Llama,
regardless of their atomic-cache coverage. No Qwen isolated scores are reused.
The at-least-one condition is not extracted or analyzed.

All 32 layers are initially saved for transfer, requiring 4.222 GiB. No layer
band is assumed: the all-layer atomic pilot and validation selection precede
transfer extraction. This conservative v1 needs about 5.244 GiB of activation
payload total, plus metadata, journals, probe outputs, and approximately 16 GB
of HF weight files. Allow at least 30 GB free disk, more for runtime packages
and duplicate HF downloads. BF16 weights alone use roughly 15 GiB of GPU memory;
a 24 GB GPU is a reasonable starting target for these short batch-one inputs,
with 40–48 GB providing more headroom. Validate actual peak memory in smoke.

There are 4,184 atomic plus 17,292 transfer forwards, in addition to small smoke
and replay gates. Runtime is hardware-dependent; extrapolate from smoke timing
rather than claiming a fixed duration. Batch-one forwards and the durable
per-row journal are the extraction bottlenecks. Atomic CPU selection performs
160 fits; L-BFGS is the main CPU cost. Shared endpoint-bootstrap evaluation runs
on CPU without weights.

## Atomic selection

The unchanged clean rule fits TRAIN only, uses all 32 saved layers and
`C = [0.001, 0.01, 0.1, 1, 10]`, and maximizes pooled atomic VALIDATION AUROC.
Exact ties prefer smaller C, then lower layer. L2 logistic includes an intercept,
no scaling/class weighting, and float64 solver inputs. The existing convergence
policy retries a convergence warning from scratch at 10,000 iterations, then
fails if still unconverged. No validation refit or sign flip is implemented.
Layer 17 is not assumed or preferred.

The completed probe must have the entire 160-configuration converged grid,
matching archive/metric hashes, consistent classes/parameters and selected
winner, exact atomic-cache identities, and zero test/compound access. The scorer
uses only frozen affine coefficients and intercept; it has no fit operation.

## Two-stage transfer and scientific policy

Preflight hashes complete files as opaque bytes, inspects headers and frozen
probe integrity, and parses only truth-free identity/statement/scope projections.
It materializes zero numerical activation matrices and computes zero scores.
The measured analysis spec contains exact input/probe/cache identities, selected
atomic layer/C assertions, Llama representation, code hashes, output schemas,
metric taxonomy, and the existing frozen statistical choices. Unknown top-level
spec keys and changed policy/source identities are rejected.

Freeze requires an unchanged passing preflight. Commit the exact spec before
scoring; the score command verifies its bytes against Git HEAD. No speculative
production spec or placeholder identities are checked in by this implementation.
All output roots are Llama-specific; finalized outputs cannot be overwritten or
resumed. Changing the policy later requires a new version.

Stage 1 writes `scores/scores.csv` with exactly:

```text
example_id,condition_id,frozen_probe_score
```

It scores only the selected layer, converts batches of 256 to float64, uses one
BLAS thread, performs no preprocessing/calibration, and writes 17-digit scores.
It publishes `scores/scoring_manifest.json` last, binding spec/hash/order/selected
probe and zero fitting, truth-column materialization and test access.

Stage 2 checks completion, spec and score hashes, exact ordered one-to-one score
coverage, and every canonical metadata hash before its first truth-bearing read.
It never opens activation arrays or coefficient archives. Primary/boundary
metrics use the unchanged canonical definitions, individual example rows and
half-credit AUROC ties. Raw, OR-both (with unchanged Llama raw AND), and external
min/max receive pooled, five-topic and equal-topic-macro tables. Boolean outputs
are separate threshold metrics; false is encoded as -1 and true as +1 when using
the existing `>=0` metric primitive. They include accuracy, balanced accuracy and
per-cell true-response fractions. Juxtaposition has no formal label and reports
only the four requested geometry AUROCs, including topic/macro versions.

All conditions share one endpoint schedule: 2,000 replicates, PCG64 seed 1729,
topic-stratified entity multiplicities, pair weight m_i*m_j, no topic-weight
renormalization, no redraw/imputation, 1,800 minimum valid replicates and 95%
pointwise percentile intervals with linear interpolation. The schedule hash
must equal the recorded Qwen schedule because entities and pairs are unchanged.
Paired changes subtract within the same replicate:

- OR-both minus raw: OR AUROC, AND-minus-OR, OR mixed-vs-FF and the three direct OR boundaries.
- External min/max minus raw: the five primary transfer metrics.

There is no method/template winner selection. External composition means **two
isolated model evaluations plus a known parse/external composition rule**, not
evidence for an internal min/max algorithm. Min/max uses raw frozen scores;
Boolean composition thresholds each isolated score at zero before AND/OR.

Evaluation stores `primary_metrics.csv/json`, `boundary_metrics.csv/json`,
`boolean_metrics.csv/json`, `juxtaposition_geometry.csv/json`,
`paired_contrasts.csv/json`, `bootstrap_summary.csv`, `bootstrap_draws.npz`,
`bootstrap_pair_weights.npz`, and a final `evaluation_manifest.json` under
`results/clean_protocol/llama31_transfer_v1/evaluation/`.

## Future production order (not run locally)

On the authorized artifact host, first make the canonical allowed atomic rows,
raw benchmark, and existing Priority-2 variant tables available at their fixed
repository-relative paths, without symlinks. Obtain an exact HF commit SHA for
the requested Instruct checkpoint. Do not substitute a floating revision.

```bash
python scripts/45_llama_replication.py --stage pin --mode pin --revision "$LLAMA_REVISION_SHA"
git add config/clean_protocol/llama31_representation_v1.json
git commit -m "Pin Llama-3.1 replication representation"
git push origin clean-eval-protocol

python scripts/45_llama_replication.py --stage atomic --mode plan
python scripts/45_llama_replication.py --stage atomic --mode smoke
python scripts/45_llama_replication.py --stage atomic --mode extract

python scripts/45_llama_replication.py --stage select --mode check-only
python scripts/45_llama_replication.py --stage select --mode fit

python scripts/45_llama_replication.py --stage transfer --mode plan
python scripts/45_llama_replication.py --stage transfer --mode smoke
python scripts/45_llama_replication.py --stage transfer --mode extract

python scripts/45_llama_replication.py --stage analysis --mode preflight
python scripts/45_llama_replication.py --stage analysis --mode freeze-spec
git diff --no-index /dev/null config/clean_protocol/llama31_transfer_v1.json
```

Review the measured spec before the next steps. No compound scores have been
computed at this point. The no-index diff intentionally returns 1 for a new file.

```bash
git add config/clean_protocol/llama31_transfer_v1.json
git commit -m "Freeze Llama-3.1 cross-family replication analysis"
git push origin clean-eval-protocol
python scripts/45_llama_replication.py --stage analysis --mode score
python scripts/45_llama_replication.py --stage analysis --mode evaluate
```

Smoke/extract commands require GPU weights and run the gates again; plan,
selection, preflight, freeze, scoring and evaluation are CPU stages. Stage order
is enforced by artifact verification; transfer requires completed atomic
selection. No script accesses final/test data, creates a split, changes Qwen
artifacts, retrains on chat, or adds another model family/Route A/B analysis.
