#!/usr/bin/env python3
"""Pinned single-GPU extraction of all Priority-2 text rows; no model load in plan."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.priority2_extraction import run

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['plan','smoke','extract'],required=True)
    args=parser.parse_args()
    print(json.dumps(run(args.mode),indent=2))
