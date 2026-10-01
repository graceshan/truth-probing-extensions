#!/usr/bin/env python3
"""Read-only atomic cache metadata and optional remote physical preflight."""
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.t2_cache_preflight import main

if __name__ == '__main__':
    main()
