# Exact-producer full-bank CPU runbook

These are prepared future commands. **They were not executed.** This delivery authorizes local preparation only. Obtain separate authorization naming the exact CPU producer and the actual supplement handoff before any research fits. No SSH, Pod, GPU, model inference, repair or chat fitting command belongs here.

CPU producer: `7bba980b94a3ffa4b9127bbc50a351820ddc2629`.
Supplement extraction producer: `9b0ac1d84b35d7faf2148d43664569aa867cba52`.
Approved integration base/prerequisite: `fc083594dc17d1fc38f7b52dd8030b0a7f2cb67d`.

## Restore and inspect

The review bundle is incremental and requires the exact integration base and its ancestry in a repository. Verify/import it into a separate review repository containing that prerequisite, or use the existing local repository. No pull, successful push or moving branch is needed. Keep originals and other worktrees untouched.

```bash
git bundle verify /absolute/path/checkpoint-next-full-bank-prep-20261003.bundle
git bundle list-heads /absolute/path/checkpoint-next-full-bank-prep-20261003.bundle
```

For a separate repository containing the prerequisite:

```bash
git fetch /absolute/path/checkpoint-next-full-bank-prep-20261003.bundle refs/heads/checkpoint-next-full-bank-prep-20261003:refs/heads/checkpoint-next-full-bank-prep-20261003
```

Create a fresh isolated exact-producer checkout; the delivery adds documents/receipts after the producer and cannot substitute for the exact producer HEAD at launch:

```bash
git worktree add --detach /private/tmp/checkpoint-next-full-bank-cpu-7bba980 7bba980b94a3ffa4b9127bbc50a351820ddc2629
cd /private/tmp/checkpoint-next-full-bank-cpu-7bba980
test "$(git rev-parse HEAD)" = 7bba980b94a3ffa4b9127bbc50a351820ddc2629
git status --porcelain=v1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
CPU_PY=/opt/homebrew/opt/python@3.11/bin/python3.11
PRODUCER=7bba980b94a3ffa4b9127bbc50a351820ddc2629
PREP=/Users/apple/truth-probing-backups/20261001T164224Z/full-bank-prep-results-20261003/run-7bba980
```

CPU packages remain pinned: NumPy 1.26.2, SciPy 1.11.4, scikit-learn 1.3.2, pandas 3.0.3, threadpoolctl 3.2.0, matplotlib 3.8.2. Use the inherited accepted raw dataset paths and manifests without replacing them with historical tensors. A different machine/path layout needs a separately reviewed relocation binding; do not edit provenance paths silently.

## Verify the actual transferred supplement

Both models must supply their 131 unique texts. Supply the complete campaign transfer manifest, original extraction identity, completed extraction/shard/token audit/attempt evidence, and real acceptance files. A producer SHA relabel, manually supplied verified flag, partial one-model supplement or fabricated token audit is rejected. The CPU producer stays separate.

```bash
SUPPLEMENT=/absolute/path/to/the/new-verified-local-supplement-handoff
TRANSFER=/absolute/path/to/its-exact-transfer.json
AUTH=/absolute/path/to/separately-reviewed-full-bank-authorization.json
RUN=/absolute/path/to/a-new-full-bank-results-directory
export SUPPLEMENT TRANSFER
"$CPU_PY" - <<'PY'
import json,os
from src.checkpoint_next_full_bank_inputs import FullTrainingAdapter
adapter=FullTrainingAdapter(os.environ['SUPPLEMENT'],os.environ['TRANSFER'])
assert adapter.supplement_status=='transferred_independently_reopened'
print(json.dumps(adapter.supplement,indent=2))
PY
```

This is a read-only CPU reopen, not GPU extraction or fitting. The adapter must consume actual original-producer acceptance. It does not call the extraction runner's current-HEAD identity function. Future full-bank launch/resume also repeats complete existing fresh-byte verification.

The delivered `authorization.template.json` is deliberately unapproved. Its `identity` must bind this CPU producer, exact config, exact preparation receipt, original acceptance and the actual complete transfer-manifest hash. `reviewed`, `research_fit_authorized` and `CPU_slot_exclusive` become true only under separate future authorization. Counts remain `candidates:600`, `repair_fits:0`, `worker_cap_seconds:10800`, `E:false`. Do not treat elapsed time, files appearing in another session or a template as approval.

## Run

After actual input acceptance, output/resource review and confirmation that no other heavy CPU fit/audit is running:

```bash
"$CPU_PY" scripts/checkpoint_next_full_bank.py run \
  --producer "$PRODUCER" --prepared "$PREP" \
  --supplement "$SUPPLEMENT" --transfer-manifest "$TRANSFER" \
  --authorization "$AUTH" --output "$RUN"
```

The numerical worker runs with one CPU process and BLAS/OpenMP thread under a watchdog supervisor. A local exclusive CPU lock prevents two instances of this full-bank supervisor; coordinate the CPU slot with mac 1 because older workers may not use that lock. Record the actual supervisor PID if backgrounding. Inputs, producer, config and all per-fit artifacts bind resume. Outputs must be outside Git and in a fresh directory.

The same 600 candidates are fitted; no repair fold, repair head or chat head is fitted. R0 fitted preprocessing uses full admitted train; t_G/TTPD use the exact 1,000-row sampler. Raw LR's C scaling and all existing numerical acceptance conventions remain frozen. Candidate checkpoint/status files retain failed parameters and complete diagnostic attempts when returned. A process killed before an optimizer returns can only retain its incomplete checkpoint; it is never a successful fit.

The **10,800-second cap is cumulative CPU worker wall time** across attempts, including launch/resume verification, fitting, scoring and reporting. It is not a measured runtime forecast. No budget is replenished on restart. An unsettled attempt is conservatively charged through recovery, including downtime. Cap exhaustion, input/hash mismatch, failed optimizer, missing coverage or output errors stop the affected task. No methods are removed and no tolerances, labels, memberships, layers or caps are widened.

## Resume / stop

Confirm the old supervisor/worker has stopped, preserve every output, and use exactly the same producer, preparation receipt, transfer, authorization and RUN:

```bash
"$CPU_PY" scripts/checkpoint_next_full_bank.py resume \
  --producer "$PRODUCER" --prepared "$PREP" \
  --supplement "$SUPPLEMENT" --transfer-manifest "$TRANSFER" \
  --authorization "$AUTH" --output "$RUN"
```

Completed fits reopen and hash check. Incomplete task-owned fit folders are preserved intact in quarantine before reattempt; a previously finalized failed fit is not silently refitted. A persistent failure or exhausted cap needs review, not a retry loop. Every attempt writes all 600 states as valid/failed/incomplete/unrun. Keep failed attempts and quarantine in the eventual research delivery.

Ctrl-C stops a foreground supervisor. For a separately recorded background PID, send TERM to the actual supervisor and check termination. The supervisor stops its process group and records time/accounting. A hard supervisor kill may leave incomplete evidence; report it and retain all files. No remote stop/contact is part of this runbook.

## Independent audit and result review

Only after complete worker output exists and the serialized CPU slot is free:

```bash
"$CPU_PY" scripts/checkpoint_next_full_bank.py audit \
  --producer "$PRODUCER" --prepared "$PREP" \
  --supplement "$SUPPLEMENT" --transfer-manifest "$TRANSFER" \
  --authorization "$AUTH" --output "$RUN" \
  --audit-output "$RUN-independent-audit.json"
```

The audit runs separately and sequentially; it does not fit anything or consume a new full-bank worker allowance. Retain its actual runtime/resources. Its independent pair-mass calculation checks all 600 atomic and allocated-boundary points, new locked-head metric points/draws, undefined accounting, paired contrasts and within-draw means/intervals against frozen source draw pins. It does not claim second-machine replication or resample selectors.

Inspect `completion.json`, every candidate/attempt record, `locked.json`, `bootstrap-provenance.json`, `metrics.csv`, `contrasts.csv`, `bank-maxima.csv` and the independent audit. Require all 600 valid candidates before claiming a complete bank. Show failure/incomplete/unrun counts if blocked. Keep exact extraction and CPU producer identities separate in the eventual delivery.

Report R_all-S_full, S_full-S_atom_full, R_all-S_atom_full and S_full-original S_all for all seeds plus metric means. Retain original reduced-LR eligibility and full canonical LR as separate arms; eligible/unrestricted bank references, retention, wordings, pooled/macro/five-topic endpoints and conditional uncertainty stay visible. A identities occur in full atomic training: this is an advantaged comparator against existing reduced-P repair, not full-training repair or an equal-exposure replacement.

## Reproduce local preparation only

These commands reproduce metadata/synthetic work in fresh external directories and do not launch bank/chat fits:

```bash
"$CPU_PY" scripts/checkpoint_next_full_bank.py prepare \
  --producer "$PRODUCER" \
  --output /absolute/path/to/new-preparation-output \
  --export-output /absolute/path/to/new-eight-raw-exports
"$CPU_PY" -m unittest discover -s tests -p test_checkpoint_next_full_bank.py -v
"$CPU_PY" -m unittest discover -s tests -p test_checkpoint_next_integration.py -v
"$CPU_PY" -m unittest discover -s tests -p test_checkpoint_next_behavior.py -v
```

Preparation reopens only manifested existing raw scores for the eight exports; it does not calculate new probe scores or scientific performance. Source arrays remain unchanged. Keep exports outside Git. The compact export supports the later matched chat task without changing its original P15 preprocessing exposure; no chat research fit is authorized here.
