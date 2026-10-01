#!/usr/bin/env python3
"""Build/check pinned provisional candidate maps from metadata only; no SSH or tensor I/O."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.selection_repair_candidate_mapping import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    report = run(ROOT, args.payload_root, args.output_root, args.check_only)
    print(json.dumps({'status': report['status'], 'production_usable': report['production_usable'],
                      'coverage': {key: {name: value[name] for name in
                                   ('expected_rows_from_candidate_manifest', 'mapped_rows', 'exact_coverage', 'issue_counts', 'duplicate_statement_rows')}
                                   for key, value in report['cache_coverage'].items()}}, indent=2))
    raise SystemExit(0 if report['metadata_alignment_passed'] else 1)
