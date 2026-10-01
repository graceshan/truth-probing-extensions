#!/usr/bin/env python3
"""Verify or materialize the immutable E fact-only audit projection."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.final_partition_fact_audit import main
if __name__ == '__main__':
    main()
