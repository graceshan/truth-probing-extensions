#!/usr/bin/env python3
"""Create or verify the package file inventory, excluding only the inventory itself."""
import argparse, hashlib, json, pathlib
ap=argparse.ArgumentParser();ap.add_argument('--check',action='store_true');args=ap.parse_args()
root=pathlib.Path(__file__).resolve().parent
path=root/'inventory.json'
files={str(p.relative_to(root)):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(root.rglob('*')) if p.is_file() and p!=path}
record={'schema':'final-e-paper-file-inventory-v1','excluded':['inventory.json'],'files':files}
if args.check: assert json.loads(path.read_text())==record,'Package inventory mismatch'
else:path.write_text(json.dumps(record,indent=2,sort_keys=True)+'\n')
print(json.dumps({'path':'paper/final_e_20261005/inventory.json','files':len(files),'bytes':sum(r['bytes'] for r in files.values()),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'verified' if args.check else 'created'}))
