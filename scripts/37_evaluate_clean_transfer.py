#!/usr/bin/env python3
"""Evaluate finalized pinned LR scores with the frozen v1 analysis specification."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.clean_transfer_evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(evaluate(), indent=2))


if __name__ == '__main__':
    main()
