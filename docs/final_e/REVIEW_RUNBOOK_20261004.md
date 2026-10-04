# Successor review reproduction and future E adoption

Runnable successor: `b2c6dde71e82f5f0af308761bf19ed3341670587`.
Review base: `ce3600d3e858c56476f9e098d5c582c6c01376af`.
Original runnable producer `880345c2bea9982d91786516219da873ddb4405d` and its receipts remain preserved. Read the review delivery documents; run implementation from the exact successor producer. Publication does not authorize E.

## Reproduce this review locally without E access

Use a new detached checkout and fresh external destinations. These commands reproduce metadata, synthetic tests and saved-D validation only. The artifact-roots example contains the actual locally verified manifest locations; relocation may change paths, never file/content pins.

```bash
export E_REPO=/Users/apple/projects/truth-probing-extensions
export E_REVIEW_PRODUCER=b2c6dde71e82f5f0af308761bf19ed3341670587
export E_REVIEW_CODE=/Users/apple/truth-probing-backups/20261001T164224Z/final-e-review-reproduction-$E_REVIEW_PRODUCER
export E_REVIEW_PY=/opt/homebrew/opt/python@3.11/bin/python3.11
export E_REVIEW_OUT=/Users/apple/truth-probing-backups/20261001T164224Z/final-e-review-reproduction-evidence-new
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
hostname
uname -s
"$E_REVIEW_PY" -c 'import sys,platform; print(sys.executable,platform.python_version())'
git -C "$E_REPO" merge-base --is-ancestor ce3600d3e858c56476f9e098d5c582c6c01376af "$E_REVIEW_PRODUCER"
git -C "$E_REPO" worktree add --detach "$E_REVIEW_CODE" "$E_REVIEW_PRODUCER"
cd "$E_REVIEW_CODE"
test "$(git rev-parse HEAD)" = "$E_REVIEW_PRODUCER"
test -z "$(git status --porcelain)"
mkdir "$E_REVIEW_OUT"
```

Inspect the shared physical Mac for active heavy numerical work. Each following stage acquires `/tmp/truth-probing-cpu-exclusive.lock` nonblocking. Execute sequentially and stop if any fails. Do not launch a second reproduction while another session holds the slot.

```bash
"$E_REVIEW_PY" - <<'PY'
import fcntl,unittest
with open('/tmp/truth-probing-cpu-exclusive.lock','a') as slot:
    fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromName(m)
        for m in ('tests.test_final_e_locked_path','tests.test_final_e_review'))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(0 if result.wasSuccessful() else 1)
PY
"$E_REVIEW_PY" -m src.final_e_prepare \
  --roots config/final_e/artifact_roots.example.json \
  --output "$E_REVIEW_OUT/metadata"
"$E_REVIEW_PY" -m src.final_e_validation \
  --roots config/final_e/artifact_roots.example.json \
  --output "$E_REVIEW_OUT/D-reproduction" --recompute-draws
"$E_REVIEW_PY" -m scripts.review_final_e_numerical \
  --roots config/final_e/artifact_roots.example.json \
  --output "$E_REVIEW_OUT/D-independent"
```

These are the reviewed stage command forms. `*-stage.json` receipts record the exact actual commands, paths, producer, host, runtime and RSS measured in this delivery. No research fit or completed wording diagnosis is repeated. `final_e_prepare` writes metadata-only schedules and a **DRAFT** lock, not E predictions or a final lock.

## Later adoption remains conditional

The original `RUNBOOK_20261003.md` remains historical and unchanged. After explicit scientific-lock approval and later runtime/phase instructions, use its raw-only interfaces and secure handoff sequence with these successor differences:

1. Set the exact execution producer to `b2c6dde71e82f5f0af308761bf19ed3341670587` everywhere: detached checkout, finalized lock `producer_sha`, scoped phase authorizations, transferred code and independently recorded pins. Do not execute the original extraction producer for E. Neither CPU producer is an extraction identity for prior Qwen/Llama supplements.
2. Preserve the original draft's `immutable_contract` exactly. The registry, metadata, contrasts, bootstrap contracts and carry-forward configuration are unchanged. Do not finalize a lock in this review or substitute self-generated approval pins for actual reviewer instructions.
3. The access receipt now carries the explicit reviewed limits. Extraction attempts, acceptance, local verification and resume require consistent limits/producer/lock/campaign evidence. Existing caps stay 7,200 seconds/model, 1,800 seconds/setup and 10,800 seconds/CPU job; the total-wall cap is still a lock decision. Recheck actual intervening charges before adoption.
4. Acceptance receipts additionally pin budget adoption and the selected successful worker/supervisor attempt. Complete loader verification validates all prior settled supervisor records, monotonic charges and the current ledger. Transfer publication first requires both accepted models. Token audit, complete shard/index hashes and separately captured transfer/receipt manifest pins are still required.
5. A normal supervised TERM/failure can resume only after explicit failure review and new scoped instruction, using the same campaign/lock/producer and intact settled receipts. Hard supervisor loss or incomplete/inconsistent accounting stops at the gate pending explicit accounting review. Preserve ledger, partials and attempts; no fabricated settlement, reset or automatic retry is available.
6. The `audit` phase retains complete frozen-head/reported-record replay and also checks principal scalar and constituent numbers with separate formulas. It publishes both receipts under the original 10,800-second job cap. A reproducible but independently incorrect report fails audit. Preserve both report directories and the failed operation receipt; do not change procedures after E access.

No actual E execution commands were run. Before any later launch, separately verify the accepted offline snapshots/runtime, real persistent storage reserve, native token plans, current carried accounting and the finalized hash-bound lock/phase authorizations. Token lengths, CUDA peaks, quota and transfer throughput are unmeasured. Confirm complete local durable handoff and stopped remote workers before announcing the safe-stop checkpoint. Keep RunPod stopped during review.
