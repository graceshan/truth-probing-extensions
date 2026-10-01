#!/usr/bin/env python3
"""Authorized source-fact/identity audit across outer partitions, not evaluation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.selection_repair_source_audit import run

if __name__ == '__main__':
    if len(sys.argv) != 1:
        raise SystemExit('No overrides: use the versioned source audit configuration.')
    report = run(ROOT)
    print(json.dumps({k: report[k] for k in ('manifest_status','counts','sampling','alias_decisions')}, indent=2))
