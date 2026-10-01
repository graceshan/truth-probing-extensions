#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.selection_repair_capacity_pilot_v1 import run
p=argparse.ArgumentParser(description='Bounded 72-configuration full/P15 pilot; no fallback or bank')
for name in ('payload','prepared','artifact','exports','canonical-results','output'):p.add_argument('--'+name+'-root',type=Path,required=True)
p.add_argument('--resume',action='store_true')
a=p.parse_args()
r=run(ROOT,a.payload_root,a.prepared_root,a.artifact_root,a.exports_root,a.canonical_results_root,a.output_root,a.resume)
print(json.dumps({k:v for k,v in r.items() if k!='artifacts'},indent=2))
