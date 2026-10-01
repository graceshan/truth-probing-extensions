#!/usr/bin/env python3
"""Validate/build the seven-claim closure; no selection or experiment options."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.selection_repair_source_closure import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    r = run(ROOT, check_only=args.check_only)
    print(json.dumps({k: r[k] for k in ('status', 'classification_counts', 'admitted_rows',
        'admitted_by_partition', 'row_status_counts', 'changed_rows', 'admission_membership_changed')}, indent=2))
