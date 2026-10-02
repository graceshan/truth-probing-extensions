#!/usr/bin/env python3
"""Validate the score-only precision artifacts without any fitting or E scoring."""
import argparse
import json
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.selection_repair_precision_validation_v1 import validate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(validate(ROOT, args.output.resolve()), indent=2))
