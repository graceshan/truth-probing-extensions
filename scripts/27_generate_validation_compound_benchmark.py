"""Generate only entity-disjoint DEVELOPMENT / VALIDATION compounds, without scoring."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.clean_compounds import check
from src.entity_partitions import save_outputs
from src.validated_negatives import json_bytes
from src.validation_compound_benchmark import build_validation_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = (ROOT / config["output_dir"]).resolve()
    check(output.is_relative_to(ROOT / "data/clean_protocol/compounds/entity_disjoint"),
          "output must be in the clean entity-disjoint compound directory")
    files = build_validation_benchmark(config, ROOT)
    check(files == build_validation_benchmark(config, ROOT), "generation is not byte-identical")
    metadata = json.loads(files["metadata.json"])
    metadata["assertions_passed"].append("byte_identical_reproduction")
    metadata["byte_identical_reproduction"] = True
    files["metadata.json"] = json_bytes(metadata)
    save_outputs(output, files)
    check(all((output / name).read_bytes() == payload for name, payload in files.items()),
          "saved outputs differ from checked bytes")
    print(json.dumps({"benchmark_label": metadata["benchmark_label"], "output_dir": str(output),
                      "pairs": metadata["output_pair_count"], "rows": metadata["output_row_count"],
                      "entities": metadata["entity_count"], "degree_summaries": metadata["degree_summaries"],
                      "assertions_passed": metadata["assertions_passed"]}, indent=2))


if __name__ == "__main__":
    main()
