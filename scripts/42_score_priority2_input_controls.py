#!/usr/bin/env python3
"""Fixed-policy Priority-2 preflight, specification freeze, or truth-blind scoring."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.priority2_transfer import preflight,freeze_spec,score

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['preflight','freeze-spec','score'],required=True)
    args=parser.parse_args()
    print(json.dumps({'preflight':preflight,'freeze-spec':freeze_spec,'score':score}[args.mode](),indent=2))
