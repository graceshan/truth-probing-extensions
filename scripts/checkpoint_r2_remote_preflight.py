"""Read-only remote capability/metadata stat preflight; no activation values read."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import time

REMOTE = r'''
import glob, os, json, shutil, importlib.metadata, importlib.util
from pathlib import Path
r=Path('/workspace/truth-probing-artifacts')
paths=[
 'atomic_repair/qwen25_a09a354_bs1_bf16_v1/train/activations.npy',
 'atomic_repair/qwen25_a09a354_bs1_bf16_v1/validation/activations.npy',
 'pinned_compound_qwen2_5/qwen2_5_7b_a09a354_bs1_bf16_v1/activations.npy',
 'llama31_replication_v1/atomic/train/activations.npy',
 'llama31_replication_v1/atomic/validation/activations.npy',
 'llama31_replication_v1/transfer/activations.npy']
versions={}
for p in ('torch','transformers','tokenizers','numpy'):
 try:versions[p]=importlib.metadata.version(p)
 except importlib.metadata.PackageNotFoundError:versions[p]=None
cuda=False
if importlib.util.find_spec('torch') is not None:
 import torch
 cuda=torch.cuda.is_available()
print(json.dumps(dict(hostname=os.uname().nodename,system=os.uname().sysname,python=os.sys.version,
 nvidia_devices=glob.glob('/dev/nvidia*'),nvidia_smi=shutil.which('nvidia-smi'),cuda_available=cuda,
 versions=versions,workspace_free_bytes=shutil.disk_usage('/workspace').free,
 hf_model_cache_roots=glob.glob('/root/.cache/huggingface/hub/models--*'),
 artifacts={str(r/p):dict(exists=(r/p).is_file(),bytes=(r/p).stat().st_size if (r/p).is_file() else None) for p in paths}),indent=2))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():raise SystemExit('Refuse to overwrite preflight receipt')
    cmd=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-p','22114','-i','/Users/apple/.ssh/id_ed25519','root@69.30.85.22','python -']
    start=time.monotonic();result=subprocess.run(cmd,input=REMOTE,text=True,capture_output=True,timeout=45)
    elapsed=time.monotonic()-start
    receipt=dict(recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=cmd,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr,setup_seconds=elapsed,GPU_seconds=0,extraction_launched=False)
    if result.returncode==0:
        remote=json.loads(result.stdout);receipt['remote']=remote
        receipt['status']='ready_for_bounded_setup_review' if remote['cuda_available'] and remote['versions']['transformers'] else 'blocked_remote_GPU_or_runtime_unavailable'
    else:receipt['status']='pending_connection_or_authentication_not_cache_corruption'
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(receipt,indent=2)+'\n')
    print(receipt['status'])

if __name__=='__main__':main()
