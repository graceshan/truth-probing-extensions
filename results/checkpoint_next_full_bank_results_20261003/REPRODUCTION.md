# Exact executed commands and review paths

Execution producer: `b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8`. Execution checkout: `/Users/apple/truth-probing-backups/20261001T164224Z/full-bank-cpu-b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8`. External output root: `/Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z`. Results delivery branch: `checkpoint-next-full-bank-results-20261003`.

No implementation/configuration changes were made during execution. Read the pinned adapter runbook at delivery `1d629a84aee5217e8f422b0bc80ee0debacdf3c8`. A fresh run requires separate fresh outputs and explicit authorization; the completed campaign must not be replaced. For this review, inspect the completed hashes and audit receipts.

All commands used `/opt/homebrew/opt/python@3.11/bin/python3.11`, Python 3.11.6, on `MacBook-Pro-7.local`, macOS. The following environment was set for all numerical stages:

```bash
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
```

Acceptance, preparation and independent audit were held under `/private/tmp/truth-probing-cpu-exclusive.lock`; the frozen run supervisor held that same lock for its entire worker attempt. Process readiness receipts are included. No concurrent heavy fitting/bootstrap/audit job ran.

## Acceptance

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -m src.checkpoint_next_full_bank_handoffs --handoffs /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/handoffs.json --require-complete --output /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/input-acceptance.json
```

## Preparation

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-cpu-b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8/scripts/checkpoint_next_full_bank.py prepare --producer b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8 --output /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/prepared --export-output /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/exports
```

## Run

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-cpu-b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8/scripts/checkpoint_next_full_bank.py run --producer b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8 --output /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/bank --prepared /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/prepared --handoffs /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/handoffs.json --input-acceptance /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/input-acceptance.json --authorization /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/authorization.json
```

## Audit

```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-cpu-b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8/scripts/checkpoint_next_full_bank.py audit --producer b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8 --output /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/bank --prepared /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/prepared --handoffs /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/handoffs.json --input-acceptance /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/input-acceptance.json --authorization /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/authorization.json --audit-output /Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z/independent-audit.json
```

Run succeeded on its first worker attempt; no resume command was executed. The registered resume form remains in the pinned runbook and uses the same cap/accounting. Exact input and authorization pins are in `identity.json` and `authorization.json`; all output hashes are in `external-output-manifest.json`. Arrays, environments, credentials, model weights and unrelated files are excluded from Git.

The reproduction listing records what was executed. Independent audit used saved scores and parameters only; it did not fit probes or adjust any threshold, label, membership or sign.
