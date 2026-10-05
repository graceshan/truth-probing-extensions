#!/usr/bin/env python3
"""Inventory this additive review only; exclude only inventory.json itself."""
import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
args = parser.parse_args()
root = Path(__file__).resolve().parent
inventory = root / 'inventory.json'
files = {}
for p in sorted(root.rglob('*')):
    if p.is_file() and p != inventory:
        b = p.read_bytes()
        files[str(p.relative_to(root))] = {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}
record = {'schema': 'final-e-paper-review-inventory-v1', 'excluded': ['inventory.json'], 'files': files}
if args.check:
    assert json.loads(inventory.read_text()) == record, 'Review inventory mismatch'
else:
    inventory.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')
print(json.dumps({'status': 'verified' if args.check else 'created', 'files': len(files),
                  'bytes': sum(p['bytes'] for p in files.values()),
                  'sha256': hashlib.sha256(inventory.read_bytes()).hexdigest()}))
