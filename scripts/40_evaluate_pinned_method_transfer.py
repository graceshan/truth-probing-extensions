#!/usr/bin/env python3
"""Evaluate finalized multi-method scores using one frozen endpoint schedule."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.pinned_method_evaluation import evaluate


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(evaluate(), indent=2))
