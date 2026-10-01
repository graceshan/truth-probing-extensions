"""Print protocol definitions or validate a label-free development exposure ledger."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.generalization_protocols import ProtocolCatalog


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/clean_protocol/generalization_protocols.json")
    parser.add_argument("--regime", help="required when checking an exposure ledger")
    parser.add_argument("--exposures", type=Path, help="JSON object with all five exposure roles; identities only")
    args = parser.parse_args()
    if bool(args.regime) != bool(args.exposures):
        parser.error("--regime and --exposures must be supplied together")
    catalog = ProtocolCatalog(args.config, ROOT)
    result = (catalog.validate_exposures(args.regime, json.loads(args.exposures.read_text()))
              if args.exposures else catalog.summary())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
