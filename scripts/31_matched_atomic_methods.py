#!/usr/bin/env python3
"""Run the fixed-layer matched-data control for both models; never overwrite results."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.atomic_methods_matched import run

if __name__ == "__main__":
    run(ROOT)
