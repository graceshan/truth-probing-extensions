#!/usr/bin/env python3
"""Plan, unpadded smoke, or explicit TRAIN/VALIDATION pinned Qwen2.5 repair."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.clean_atomic_extraction import OUTPUT, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "smoke", "extract"), default="plan")
    parser.add_argument("--output", type=Path, default=OUTPUT,
                        help="New acts/clean_protocol/atomic/qwen25_a09a354_bs1_bf16_vN directory")
    args = parser.parse_args()
    print(json.dumps(run(args.mode, output=args.output), indent=2))


if __name__ == "__main__":
    main()
