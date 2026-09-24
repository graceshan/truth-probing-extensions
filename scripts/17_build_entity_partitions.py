"""Build the clean entity manifest and print aggregate counts only."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.entity_partitions import build_manifest, require, save_outputs


def main():
    config = json.loads((ROOT / "config/clean_protocol/entity_split.json").read_text())
    files, counts = build_manifest(ROOT, config)
    reproduced, _ = build_manifest(ROOT, config)
    require(files == reproduced, "full same-seed rebuild differs")
    output_dir = (ROOT / config["output_dir"]).resolve()
    require(output_dir.is_relative_to((ROOT / "data/clean_protocol").resolve()),
            "outputs must stay in data/clean_protocol")
    save_outputs(output_dir, files)
    print(f"All assertions passed; seed={config['seed']}. Outputs: {config['output_dir']}")
    print("topic,split,entities,compound_usable,max_unordered_usable_pairs")
    for row in counts:
        print(",".join(str(v) for v in row.values()))


if __name__ == "__main__":
    main()
