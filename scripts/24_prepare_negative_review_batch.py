"""Export one current unverified candidate per unavailable validation entity."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clean_compounds import check
from src.entity_partitions import save_outputs
from src.negative_review_batches import build_review_batch, load_previous_batch, validation_only
from src.validated_negatives import load_registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("first", "next"))
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/clean_protocol/validated_negatives.json")
    parser.add_argument("--registry-dir", type=Path, help="existing validation registry, including imported-review snapshots")
    parser.add_argument("--previous-batch", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    validation_only(args.split)  # Reject train/test before reading configuration or facts.
    if (args.action == "next") != (args.previous_batch is not None):
        parser.error("next requires --previous-batch; first does not accept it")
    output = args.output_dir.resolve()
    allowed = ROOT / "data/clean_protocol/validated_negatives/v1/review_batches"
    check(output.is_relative_to(allowed), "output must be under the v1 review_batches directory")
    config = json.loads(args.config.read_text())
    directory = args.registry_dir or ROOT / config["output_dir"] / "validation"
    registry = load_registry(config, ROOT, "validation", directory)
    previous = load_previous_batch(args.previous_batch) if args.previous_batch else None
    files = build_review_batch(registry, ROOT / config["source_audit_dir"], previous=previous)
    save_outputs(output, files)
    metadata = json.loads(files["metadata.json"])
    print(json.dumps({"output_dir": str(output), "rows": metadata["row_count"], "summary": metadata["summary"],
                      "registry_modified": False, "test_candidates_exported": 0}, indent=2))


if __name__ == "__main__":
    main()
