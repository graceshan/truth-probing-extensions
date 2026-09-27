#!/usr/bin/env python3
"""Fit/report clean atomic probe methods. No test or compound-data mode exists."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.atomic_method_suite import run_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("qwen25_7b", "qwen3_8b", "both"), default="both")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/clean_protocol/atomic_probe_methods.json").read_text())
    models = ("qwen25_7b", "qwen3_8b") if args.model == "both" else (args.model,)
    for model in models:
        result = run_model(ROOT, config, model, resume=args.resume, check_only=args.check_only)
        if args.check_only:
            print(json.dumps({"model": model, "fixed_layer": result["fixed_primary_layer"],
                              "burger_training_rows": result["burger_training_rows"],
                              "test_labels_loaded": result["data"]["test_labels_loaded"]}))


if __name__ == "__main__":
    main()
