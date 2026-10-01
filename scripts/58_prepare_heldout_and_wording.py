#!/usr/bin/env python3
"""Prepare/verify immutable D and E Section 9 wording; data only."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.heldout_and_preparation import main
if __name__=='__main__':main()
