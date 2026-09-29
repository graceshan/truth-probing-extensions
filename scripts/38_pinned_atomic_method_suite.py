#!/usr/bin/env python3
"""CPU-only canonical atomic methods; check-only never fits or loads matrices."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pinned_atomic_method_suite import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("faithful", "matched", "both"), default="faithful")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(suite=args.suite, check_only=args.check_only), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
