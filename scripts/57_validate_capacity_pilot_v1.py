#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.selection_repair_capacity_validation_v1 import validate
p=argparse.ArgumentParser()
p.add_argument('--prepared-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True)
a=p.parse_args();print(json.dumps(validate(ROOT,a.prepared_root,a.output_root),indent=2))
