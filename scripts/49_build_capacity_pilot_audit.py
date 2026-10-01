#!/usr/bin/env python3
"""Validate/reproduce the separate fact-audit pilot proposal package."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.capacity_pilot_audit import main

if __name__ == '__main__':
    main()
