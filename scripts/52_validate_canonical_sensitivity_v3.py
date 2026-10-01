#!/usr/bin/env python3
"""Check v3 receipts and independently recompute all AUROC points/intervals from saved scores."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.selection_repair_sensitivity_v3_validation import validate
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('payload-root','prepared-root','results-root'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    print(json.dumps(validate(ROOT,args.payload_root,args.prepared_root,args.results_root),indent=2))
