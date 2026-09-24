"""Generate one explicitly configured split; no default benchmark run."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clean_compounds import build_generation, check
from src.entity_partitions import save_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    output_dir = (config_path.parent / config["output_dir"]).resolve()
    if output_dir.is_relative_to(ROOT):
        check(output_dir.is_relative_to(ROOT / "data/clean_protocol"),
              "repository outputs must be under data/clean_protocol")
    files = build_generation(config, config_path.parent)
    check(files == build_generation(config, config_path.parent), "generation is not byte-deterministic")
    save_outputs(output_dir, files)
    metadata = json.loads(files["metadata.json"])
    print(json.dumps({"split": metadata["requested_split"], "rows": metadata["output_row_count"],
                      "assertions_passed": metadata["assertions_passed"],
                      "byte_identical_reproduction": True, "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
