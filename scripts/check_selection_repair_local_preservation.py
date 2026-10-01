#!/usr/bin/env python3
"""Explicit local check of mac 1's historical untracked files; not a suite prerequisite."""
import argparse
import hashlib
import json
from pathlib import Path


def check_preservation(root, baseline):
    result = {'matched': [], 'missing': [], 'mismatched': []}
    for relative, expected in baseline.items():
        path = root / relative
        if not path.is_file():
            result['missing'].append(relative)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            result['mismatched'].append(relative)
        else:
            result['matched'].append(relative)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True,
                        help='Explicit checkout containing the historical local files')
    args = parser.parse_args()
    baseline = args.root / 'data/clean_protocol/selection_repair_v1/candidate_overlay_v1/preservation_baseline.json'
    result = check_preservation(args.root, json.loads(baseline.read_text())['unrelated_untracked_sha256'])
    print(json.dumps(result, indent=2))
    raise SystemExit(bool(result['missing'] or result['mismatched']))
