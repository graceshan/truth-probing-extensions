#!/usr/bin/env python3
"""Versioned v4 preparation and two canonical LR development references."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','evaluate'])
    parser.add_argument('--payload-root',type=Path,required=True)
    parser.add_argument('--prepared-root',type=Path,required=True)
    parser.add_argument('--artifact-root',type=Path,required=True)
    parser.add_argument('--parent-prepared-root',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--score-output-root',type=Path)
    args=parser.parse_args()
    if args.mode=='prepare':
        if args.parent_prepared_root is None:parser.error('prepare requires --parent-prepared-root')
        from src.selection_repair_sensitivity_v4 import prepare
        result=prepare(ROOT,args.payload_root,args.parent_prepared_root,args.artifact_root,args.prepared_root)
        print(json.dumps({k:v for k,v in result.items() if k!='metadata_inputs'},indent=2))
    else:
        if args.output_root is None or args.score_output_root is None:parser.error('evaluate requires output and score-output roots')
        from src.selection_repair_canonical_sensitivity_v4 import run
        result=run(ROOT,args.payload_root,args.prepared_root,args.artifact_root,args.output_root,args.score_output_root)
        print(json.dumps({k:result[k] for k in ('status','endpoint_results','missing_endpoints','runtime')},indent=2))
