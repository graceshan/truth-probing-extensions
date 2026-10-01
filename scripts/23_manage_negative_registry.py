"""Initialize, export or update development-only factual validation registries."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clean_compounds import check
from src.entity_partitions import save_outputs
from src.validated_negatives import (
    apply_reviews, build_registry, development_only, export_candidates, json_bytes, load_registry,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("initialize", "export", "review"))
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--config", type=Path, default=ROOT / "config/clean_protocol/validated_negatives.json")
    parser.add_argument("--registry-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--reviews", type=Path, help="evidence-only JSON review records")
    parser.add_argument("--validation-version", help="required for an explicit review update")
    args = parser.parse_args()
    development_only(args.split)  # Before config, registry, review or source files are opened.
    config = json.loads(args.config.read_text())
    directory = args.registry_dir or ROOT / config["output_dir"] / args.split
    if args.action != "initialize" and args.output_dir is None:
        parser.error("export/review requires a new --output-dir")
    output = (args.output_dir or directory).resolve()
    check(output.is_relative_to(ROOT / "data/clean_protocol/validated_negatives"),
          "registry output must remain under data/clean_protocol/validated_negatives")
    if args.action == "initialize":
        check(args.reviews is None and args.validation_version is None, "initialize does not accept review inputs")
        registry = build_registry(config, ROOT, args.split)
        files = registry.files()
    else:
        registry = load_registry(config, ROOT, args.split, directory)
        if args.action == "export":
            check(args.reviews is None and args.validation_version is None, "export does not accept review inputs")
            files = {"candidates.csv": export_candidates(registry, args.split),
                     "metadata.json": json_bytes({**registry.provenance, "registry_sha256": registry.digest})}
        else:
            check(args.reviews is not None and args.validation_version, "review requires evidence file and validation version")
            parent = registry.digest
            registry = apply_reviews(registry, json.loads(args.reviews.read_text()), args.validation_version)
            files = registry.files(parent_sha256=parent)
    save_outputs(output, files)
    print(json.dumps({"output_dir": str(output), "summary": registry.summary(),
                      "test_candidates_processed": 0, "scores_used": False}, indent=2))


if __name__ == "__main__":
    main()
