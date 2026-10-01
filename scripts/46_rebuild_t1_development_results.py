#!/usr/bin/env python3
"""Rebuild historical development results from a verified relocated score package."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.t1_score_rebuild import main

if __name__ == '__main__':
    main()
