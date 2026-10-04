# Full-bank per-model adapter successor — preparation only

Branch `checkpoint-next-full-bank-adapter-successor-20261003` starts exactly at full-bank delivery `98ec4004168c2bc5d1906ce1d7cf95a93dde759f`. A normal merge imports exactly raw-only delivery `4e508a8f2a7d85cd8e540686be9b007b7fc29810`. Both are ancestors of runnable CPU producer **`b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8`**. The merged CPU HEAD is never an extraction producer. Delivery adds only this document and verification metadata after the runnable producer.

This local macOS preparation performed no Pod contact, model inference, research fit, behavior/chat execution, supplement adoption, calibration or E access. Publication does not authorize execution or constitute implementation approval. The original worktrees, scientific inputs, original deliveries and review ZIP remain preserved.

## Changes and provenance

`full_bank_adapter_successor_v2.json` is a new configuration. The original `full_bank_prep_v1.json` remains byte-for-byte unchanged and is itself pinned by v2. The successor registers the model assignments and hashes the changed implementation files. Its `successor_implementation_changes` records old and new hashes; all other inherited science and numerical dependency pins remain unchanged.

The handoff descriptor has separate `campaign`, external `transfer_manifest` and exact `verifier_checkout` fields for each model. Arbitrary producer overrides are unavailable. The adapter checks the complete identity before dispatch, verifies every transfer file, rejects symlinks, unsettled/orphan attempts and inconsistent carried budgets, and calls saved-shard verification in a clean exact-producer checkout. Git/Python environment overrides cannot redirect this verification. It does not call extraction, acceptance creation, independent token-audit generation or interruption recovery. Shards are loaded with each model's original extraction provenance. Verification rereads transfer hashes after the subprocess.

The resulting downstream receipt binds the complete identity; exact model specification and tokenizer/backend evidence; ordered token packets and independent audit; completion, acceptance and every shard file hash; separate transfer manifests; original and successor start/end hashes; measured timing/peak evidence; and adoption as a provenance/accounting link. It does not relabel either extraction identity. Qwen's partial failed preflight remains separate historical evidence, not a completed preflight acceptance.

| Input | Extraction producer | Verified state |
|---|---|---|
| Qwen | `9b0ac1d84b35d7faf2148d43664569aa867cba52` | Original transferred handoff reopened; 131 shards, 574 files |
| Llama | `6865bfa43c52d9c5d7d3c96df90351ad8f895aea` | **Pending actual completed successor handoff and transfer** |

Qwen acceptance SHA-256: `96c5d1f5c064e79a58b6704763d6dbca77e4f352d48b13ae3658d9ddc8ba539d`.
Original preflight transfer manifest SHA-256: `88918f7fcd28b37a8a593a844b1c8ab10c11e940e4144e493dc4c05809729ebc`.
Qwen verification took 6.3066 seconds, with parent peak RSS 38,912,000 bytes and child peak RSS 758,808,576 bytes (macOS `getrusage` measurements). No Llama acceptance or shard hashes are invented. Its verifier checkout exists, but contains no evidence of completed extraction.

Committed metadata under `results/checkpoint_next_full_bank_adapter_successor_20261003/` includes the full read-only Qwen downstream receipt, the descriptor with `llama: null`, verification measurements and preservation/test receipts. These are downstream verification receipts, not rewritten source extraction acceptances.

The 3,040 admitted full-training rows, original 2,778 P15 rows, exact balanced 1,000-row sampler, 2,909 existing fresh representations/model and 131 missing texts/model are unchanged. The 600-candidate grid, method preprocessing/objectives, raw LR C scaling, convergence/acceptance rules, reduced-training common LR eligibility reference and margin, B25 selectors/tie rules, contrasts, paired 2,000-draw schedules and within-draw metric averaging are unchanged. No repair head or 900 repair folds are prepared anew. A identities occur in full atomic training: this compares an advantaged full-training atomic bank against frozen reduced-P repair; it does not test full-training repair.

## Focused validation

47 metadata/synthetic tests passed in 23.387 seconds, one process and one BLAS/OpenMP thread. The new suite uses all 131 synthetic shards/model with the actual writer and raw loader, and explicitly synthetic numerical dimensions/backend evidence. Exact-checkout subprocess transport is mocked in those synthetic tests; the real Qwen test uses its unmodified original producer checkout. Cases cover mixed producers, wrong/swapped identities and verifiers, altered complete/row bindings, missing/corrupt shards, token plans and independent audits, changed acceptance/completion identities, failed-attempt carry, incomplete/orphan ledgers, nonfinite charges, inconsistent adoption/accounting, unexpected phases and transfer/symlink substitution. Existing tests retain full memberships, all 600 candidates, selectors and failed-fit serialization guards.

The initial test run had one failure and two errors in synthetic fixture setup/path mocks; these were corrected without changing scientific rules. Full details are in `validation.json`. `git diff --check`, frozen configuration pins, exact memberships and both reviewed ancestry checks passed. Research fits remain unrun; full-bank runtime is unmeasured.

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/opt/homebrew/opt/python@3.11/bin/python3.11 -m unittest \
  tests.test_checkpoint_next_full_bank_handoffs \
  tests.test_checkpoint_next_full_bank tests.test_checkpoint_next_raw_only
```

## Exact future acceptance commands

These commands are for later local acceptance of mac 1's actual completed transfer. No extraction/Pod command is provided. Keep Qwen's original immutable handoff and Llama's actual successor transfer separate. Do not use the raw-only delivery SHA or this CPU producer as an extraction identity.

Create a new exact CPU checkout from the successor repository and retain the two existing exact verification checkouts:

```bash
MAC_PY=/opt/homebrew/opt/python@3.11/bin/python3.11
PRODUCER=b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8
TASK_ROOT=/Users/apple/truth-probing-backups/20261001T164224Z
CPU_CODE="$TASK_ROOT/full-bank-cpu-$PRODUCER"
git worktree add --detach "$CPU_CODE" "$PRODUCER"
cd "$CPU_CODE"
test "$(git rev-parse HEAD)" = "$PRODUCER"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
QWEN_VERIFIER="$TASK_ROOT/full-bank-adapter-verifier-qwen-9b0ac1d"
LLAMA_VERIFIER="$TASK_ROOT/full-bank-adapter-verifier-llama-6865bfa"
ORIGINAL=/Users/apple/projects/truth-probing-extensions/.worktrees/checkpoint-next-gpu-execution-9b0ac1d/handoffs/preflight
ORIGINAL_MANIFEST=/Users/apple/projects/truth-probing-extensions/.worktrees/checkpoint-next-gpu-execution-9b0ac1d/transfers/preflight-transfer.json
```

If verifier checkouts are unavailable on another Mac, create fresh detached checkouts of the exact producer SHAs in the input table above; never move/modify the original worktrees. Set `LOCAL_LLAMA_RUN` and `LOCAL_LLAMA_MANIFEST` to mac 1's actual distinct completed transfer paths. Set `HANDOFFS` and `INPUT_ACCEPTANCE` to new durable JSON paths outside every worktree. The manifest must cover the successor's entire settled campaign including adoption, failed attempts, quarantine, audit and acceptance; its original 574-file source must remain independently available.

```bash
: "${LOCAL_LLAMA_RUN:?actual completed transferred campaign required}"
: "${LOCAL_LLAMA_MANIFEST:?actual external successor transfer manifest required}"
: "${HANDOFFS:?new descriptor path required}"
: "${INPUT_ACCEPTANCE:?new acceptance receipt path required}"
export ORIGINAL ORIGINAL_MANIFEST QWEN_VERIFIER LLAMA_VERIFIER
export LOCAL_LLAMA_RUN LOCAL_LLAMA_MANIFEST HANDOFFS
"$MAC_PY" - <<'PY'
import json,os
b=dict(schema='checkpoint-next-full-bank-handoffs-v2',
       qwen=dict(campaign=os.environ['ORIGINAL'],transfer_manifest=os.environ['ORIGINAL_MANIFEST'],
                 verifier_checkout=os.environ['QWEN_VERIFIER']),
       llama=dict(campaign=os.environ['LOCAL_LLAMA_RUN'],transfer_manifest=os.environ['LOCAL_LLAMA_MANIFEST'],
                  verifier_checkout=os.environ['LLAMA_VERIFIER']))
with open(os.environ['HANDOFFS'],'x') as f:json.dump(b,f,indent=2)
PY
"$MAC_PY" -m src.checkpoint_next_full_bank_handoffs \
  --handoffs "$HANDOFFS" --require-complete --output "$INPUT_ACCEPTANCE"
```

This refuses incomplete Llama evidence and performs no write inside either campaign. Omitting `--require-complete` with `llama: null` produces only a Qwen verification/pending receipt; it cannot authorize fitting.

## Exact future full-bank preparation/run/resume/audit

After actual mixed-producer acceptance, create new producer-bound preparation/export outputs (`PREPARED`, `EXPORTS`) outside the worktree. This step only prepares metadata and existing raw comparator exports; no research fit:

```bash
"$MAC_PY" scripts/checkpoint_next_full_bank.py prepare \
  --producer "$PRODUCER" --output "$PREPARED" --export-output "$EXPORTS"
```

The previous producer's preparation receipt remains preserved and cannot be relabeled as this producer. The new CLI replaces `--supplement`/`--transfer-manifest` with `--handoffs`/`--input-acceptance`; all future entry points propagate them. The CPU identity pins the descriptor, complete downstream acceptance, successor configuration, exact preparation and existing fresh acceptance. Build an **unreviewed** authorization template at a new `AUTH_TEMPLATE` path:

```bash
export PRODUCER HANDOFFS INPUT_ACCEPTANCE PREPARED AUTH_TEMPLATE
"$MAC_PY" - <<'PY'
import json,os
from types import SimpleNamespace
from scripts.checkpoint_next_full_bank import identity
from src.checkpoint_next_full_bank_inputs import configuration
a=SimpleNamespace(producer=os.environ['PRODUCER'],handoffs=os.environ['HANDOFFS'],
                  input_acceptance=os.environ['INPUT_ACCEPTANCE'],prepared=os.environ['PREPARED'])
v=dict(identity=identity(a,configuration()),reviewed=False,research_fit_authorized=False,
       CPU_slot_exclusive=False,candidates=600,repair_fits=0,worker_cap_seconds=10800,E=False)
with open(os.environ['AUTH_TEMPLATE'],'x') as f:json.dump(v,f,indent=2)
PY
```

A later explicit execution review must supply `AUTHORIZATION` with the required approval fields. This task supplies none. Set `BANK_OUTPUT` to a fresh external bank directory. After that separate authorization, the exact commands are:

```bash
"$MAC_PY" scripts/checkpoint_next_full_bank.py run \
  --producer "$PRODUCER" --output "$BANK_OUTPUT" --prepared "$PREPARED" \
  --handoffs "$HANDOFFS" --input-acceptance "$INPUT_ACCEPTANCE" \
  --authorization "$AUTHORIZATION"

"$MAC_PY" scripts/checkpoint_next_full_bank.py resume \
  --producer "$PRODUCER" --output "$BANK_OUTPUT" --prepared "$PREPARED" \
  --handoffs "$HANDOFFS" --input-acceptance "$INPUT_ACCEPTANCE" \
  --authorization "$AUTHORIZATION"

"$MAC_PY" scripts/checkpoint_next_full_bank.py audit \
  --producer "$PRODUCER" --output "$BANK_OUTPUT" --prepared "$PREPARED" \
  --handoffs "$HANDOFFS" --input-acceptance "$INPUT_ACCEPTANCE" \
  --authorization "$AUTHORIZATION" --audit-output "$AUDIT_OUTPUT"
```

Run uses a fresh output; resume uses the same immutable identity and cumulative ledger. The existing exclusive CPU slot, one-process/thread worker, **10,800-second cumulative worker cap** and all 600 valid/failed/incomplete/unrun records remain. A cap is not a runtime estimate. Failed candidates, parameters, diagnostics, attempts and incomplete quarantined files remain preserved; failure blocks selectors/scoring and cannot be hidden by omission or automatic refitting. Audit requires an actually completed bank and a new audit output. Selectors lock before wording evaluation; each fitted head stays separate, with metrics averaged within paired draws.
