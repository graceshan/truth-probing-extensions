#!/usr/bin/env python3
"""Fixed-path preflight, spec freeze, or label-blind pinned LR scoring."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pinned_compound_scoring import preflight, freeze_spec, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['preflight', 'freeze-spec', 'score'], required=True)
    args = parser.parse_args()
    print(json.dumps({'preflight': preflight, 'freeze-spec': freeze_spec, 'score': score}[args.mode](), indent=2))


if __name__ == '__main__':
    main()
