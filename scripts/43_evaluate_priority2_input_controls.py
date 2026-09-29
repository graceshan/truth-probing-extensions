#!/usr/bin/env python3
"""Evaluate finalized Priority-2 scores; no activation or probe access."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.priority2_evaluation import evaluate

if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(evaluate(),indent=2))
