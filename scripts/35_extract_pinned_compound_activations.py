#!/usr/bin/env python3
"""Pinned Qwen2.5 DEVELOPMENT/VALIDATION compounds; no historical or probe gates."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pinned_compound_extraction import OUTPUT, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plan", "smoke", "extract"), default="plan")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.mode, output=args.output, resume=args.resume), indent=2))


if __name__ == "__main__":
    main()
