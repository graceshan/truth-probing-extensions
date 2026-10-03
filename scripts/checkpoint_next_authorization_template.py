"""Write an intentionally UNREVIEWED exact-producer template; never grants execution."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.checkpoint_next_runner import frozen,identity
from src.checkpoint_next_store import publish_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',choices=['qwen','llama'],required=True)
    p.add_argument('--phase',choices=['missing_raw','preflight','storage_preflight','launch','chat_cpu_fit'],required=True)
    p.add_argument('--output',required=True)
    a=p.parse_args();r=dict(identity(a.model,frozen()),phase=a.phase,reviewed=False,E=False)
    if a.phase!='missing_raw':r['missing_raw_acceptance_sha256']=None
    if a.phase in ('launch','storage_preflight'):r['preflight_sha256']=None
    if a.phase=='launch':r.update(forecast_reviewed=False,storage_reviewed=False,unmeasured_allowance_seconds=None,
        storage_receipt_path=None,storage_receipt_sha256=None)
    if a.phase=='chat_cpu_fit':r.update(heads=8,CPU_slot_exclusive=False,cpu_seconds=None,raw_matches_sha256=None)
    publish_json(a.output,r)


if __name__=='__main__':main()
