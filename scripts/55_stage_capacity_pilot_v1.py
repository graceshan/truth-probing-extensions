#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.selection_repair_capacity_staging_v1 import stage
p=argparse.ArgumentParser(description='One-shot additional-layer export for the v4 P15 pilot')
for name in ('payload','prepared','artifact','output'):p.add_argument('--'+name+'-root',type=Path,required=True)
a=p.parse_args()
stage(ROOT,a.payload_root,a.prepared_root,a.artifact_root,a.output_root)
