#!/usr/bin/env python3
"""Run the frozen score-only precision analysis; never fits or loads activations."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.selection_repair_precision_proxy_v1 import run

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--external-output', type=Path, required=True)
    args = parser.parse_args()
    results = run(ROOT, args.output.resolve(), args.external_output.resolve())
    print(f'Completed {len(results)} precision records from saved D scores; no fitting or E scoring.')
