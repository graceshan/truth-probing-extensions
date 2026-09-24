"""Run the CLI twice on one invented pair, then remove all temporary artifacts."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from compound_fixtures import synthetic_fixture
from src.clean_compounds import FIELDS


def main():
    with tempfile.TemporaryDirectory(prefix="clean-compound-smoke-") as directory:
        root = Path(directory)
        config, _, _ = synthetic_fixture(root)
        config["pairs_per_topic"] = {"cities": 1}
        path = root / "smoke_config.json"
        path.write_text(json.dumps(config))
        command = [sys.executable, "-B", str(ROOT / "scripts/18_generate_clean_compounds.py"),
                   "--config", str(path)]
        first = subprocess.run(command, check=True, capture_output=True, text=True)
        output = root / config["output_dir"]
        before = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
        subprocess.run(command, check=True, capture_output=True, text=True)
        after = {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in output.iterdir()}
        assert before == after
        print(first.stdout.strip())
        print("Synthetic-only smoke: 1 pair, 16 rows; rerun preserved output bytes and timestamps.")
        print("Output schema: " + ", ".join(FIELDS))


if __name__ == "__main__":
    main()
