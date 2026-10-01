#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.selection_repair_capacity_p10_validation_v2 import validate
p=argparse.ArgumentParser(description='Independent saved-score validation; no fitting or activation access')
p.add_argument('--output-root',type=Path,required=True)
a=p.parse_args();print(json.dumps(validate(ROOT,a.output_root),indent=2))
