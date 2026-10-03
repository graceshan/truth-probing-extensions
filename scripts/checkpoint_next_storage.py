"""Future first-use storage allocation; requires exact reviewed preflight receipt.

This command is prepared, not run. Allocation probes use a new non-secret research
output directory and the existing audited writer/capacity function. No credentials.
"""
import argparse
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.checkpoint_next_runner import frozen, identity, authorized
from src.checkpoint_next_store import publish_json
from src.checkpoint_next_inventory import config, require, sha
from src.checkpoint_r2_fresh_store import storage_probe
from src.checkpoint_r2_full_support import filesystem_prerequisites, NON_SECRET_ARTIFACT_CONTRACT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--model',required=True,choices=['qwen','llama'])
    p.add_argument('--authorization',required=True);p.add_argument('--preflight',required=True);p.add_argument('--campaign',required=True)
    args=p.parse_args();x=frozen()
    auth_args=argparse.Namespace(**vars(args));auth_args.output=args.campaign
    a=authorized(auth_args,'storage_preflight',x)
    require(sha(args.preflight)==a['preflight_sha256'],'storage preflight receipt mismatch')
    pre=json.loads(Path(args.preflight).read_text())
    require(pre['identity']==identity(args.model,x) and pre['new_paths_measured'],'unverified new-path preflight')
    directory=Path(args.output).resolve();directory.mkdir(parents=True,exist_ok=False)
    capabilities=filesystem_prerequisites(directory,artifact_contract=NON_SECRET_ARTIFACT_CONTRACT)
    probe=storage_probe(directory,pre['storage_reserved_bytes'])
    publish_json(directory/'reservation.json',dict(identity=identity(args.model,x),reserved_bytes=pre['storage_reserved_bytes'],
        allocation_verified=True,capabilities=capabilities,probe=probe,device=directory.stat().st_dev,utc=time.time(),
        guarantee=False,scope='first-use allocation/publication check only; recheck capacity after restart; quota races/errors stop affected task'))


if __name__=='__main__':main()
