#!/usr/bin/env python3
"""Validate/reproduce the successor correction overlay and fixed-A projection."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.capacity_pilot_projection import main

if __name__ == '__main__':
    main()
