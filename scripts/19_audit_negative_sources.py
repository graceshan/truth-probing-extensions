"""Audit source support for negatives without generating benchmark examples."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.entity_partitions import require, save_outputs
from src.negative_audit import build_audit


def main():
    manifest_dir = ROOT / "data/clean_protocol/entity_partitions"
    args = (ROOT / "data/tiu_datasets", manifest_dir / "manifest.csv", manifest_dir / "metadata.json")
    files, metadata = build_audit(*args)
    repeated, _ = build_audit(*args)
    require(files == repeated, "audit is not byte-deterministic")
    output = ROOT / "data/clean_protocol/audits/source_negatives_v1"
    save_outputs(output, files)
    print(files["summary.csv"].decode().strip())
    print(json.dumps({"issue_counts": metadata["issue_counts"],
                      "blocked_generator_strata": sum(s["current_generator_blocked"] for s in metadata["strata"]),
                      "deterministic": True, "output_dir": str(output)}, indent=2))


if __name__ == "__main__":
    main()
