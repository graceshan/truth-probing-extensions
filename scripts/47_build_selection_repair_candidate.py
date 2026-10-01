#!/usr/bin/env python3
"""Build/verify the pending T2C source-only candidate; no experiment entry point."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.selection_repair_candidate_overlay import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    receipt = run(ROOT, check_only=args.check_only)
    print(json.dumps({k: receipt[k] for k in ('status', 'original_rows', 'row_counts', 'sample_counts', 'targeted_counts')}, indent=2))
