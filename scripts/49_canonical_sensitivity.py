#!/usr/bin/env python3
"""Frozen v2 preparation, local export verification, or correction-held sensitivity."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','verify-exports','evaluate'])
    parser.add_argument('--payload-root',type=Path)
    parser.add_argument('--prepared-root',type=Path,required=True)
    parser.add_argument('--artifact-root',type=Path)
    parser.add_argument('--output-root',type=Path)
    args=parser.parse_args()
    if args.mode in ('prepare','evaluate') and args.payload_root is None:
        parser.error(args.mode+' requires --payload-root')
    if args.mode=='prepare':
        from src.selection_repair_sensitivity_inputs import prepare
        result=prepare(ROOT,args.payload_root,args.prepared_root)
        print(json.dumps({k:v for k,v in result.items() if k!='metadata_inputs'},indent=2))
    elif args.mode=='verify-exports':
        if args.artifact_root is None:parser.error('verify-exports requires --artifact-root')
        from src.selection_repair_sensitivity_verification import verify_exports
        print(json.dumps(verify_exports(ROOT,args.prepared_root,args.artifact_root),indent=2))
    else:
        if args.artifact_root is None or args.output_root is None:parser.error('evaluate requires --artifact-root and --output-root')
        from src.selection_repair_canonical_sensitivity import run
        result=run(ROOT,args.payload_root,args.prepared_root,args.artifact_root,args.output_root)
        print(json.dumps({k:result[k] for k in ('status','endpoint_results','missing_endpoints','runtime','baseline_checks')},indent=2))
