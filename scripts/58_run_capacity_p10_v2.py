#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.selection_repair_capacity_p10_v2 import run
p=argparse.ArgumentParser(description='Exactly 36 nested P10 configurations, unchanged v4 contract')
p.add_argument('--payload-root',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);p.add_argument('--resume',action='store_true')
a=p.parse_args();r=run(ROOT,a.payload_root,a.output_root,a.resume)
print(json.dumps({k:v for k,v in r.items() if k!='artifacts'},indent=2))
