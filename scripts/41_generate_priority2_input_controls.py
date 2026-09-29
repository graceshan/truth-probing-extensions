#!/usr/bin/env python3
"""Generate deterministic same-fact controls, or report exact isolated coverage."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.priority2_input_controls import generate

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['audit','generate'],required=True)
    args=parser.parse_args()
    print(json.dumps(generate(audit_only=args.mode=='audit'),indent=2))
