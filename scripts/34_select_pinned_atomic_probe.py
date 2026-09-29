#!/usr/bin/env python3
"""Check repaired Qwen2.5 caches or run unchanged clean TRAIN/VALIDATION LR selection."""
import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from src.pinned_atomic_probes import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="verify cache hashes/headers/metadata; no fitting")
    args = parser.parse_args()
    print(json.dumps(run(check_only=args.check_only), indent=2))


if __name__ == "__main__":
    main()
