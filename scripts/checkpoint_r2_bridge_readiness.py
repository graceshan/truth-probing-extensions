"""One authorized read-only SSH check; never install, download or run a model."""
import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.checkpoint_r2_bridge_runner import Plan, write_new, digest

REMOTE=r'''
import os, json, glob, shutil, platform, subprocess, importlib.metadata, hashlib
from pathlib import Path
cfg=json.loads(CONFIG_PLACEHOLDER)
def query(args):
 try:
  p=subprocess.run(args,text=True,capture_output=True,timeout=8)
  return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
 except (OSError,subprocess.TimeoutExpired) as e:return dict(unavailable=str(e))
versions={}
for name in ['torch','transformers','tokenizers','numpy','safetensors','huggingface-hub']:
 try:versions[name]=importlib.metadata.version(name)
 except importlib.metadata.PackageNotFoundError:versions[name]=None
cuda={}
if versions['torch']:
 import torch
 cuda=dict(visible=torch.cuda.is_available(),compiled_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),device_count=torch.cuda.device_count(),devices=[dict(name=torch.cuda.get_device_name(i),total_vram_bytes=torch.cuda.get_device_properties(i).total_memory,free_vram_bytes=torch.cuda.mem_get_info(i)[0]) for i in range(torch.cuda.device_count())])
roots={Path('/root/.cache/huggingface/hub'),Path('/workspace/.cache/huggingface/hub'),Path('/workspace/huggingface/hub')}
for key in ['HF_HUB_CACHE','HUGGINGFACE_HUB_CACHE']:
 if os.environ.get(key):roots.add(Path(os.environ[key]))
if os.environ.get('HF_HOME'):roots.add(Path(os.environ['HF_HOME'])/'hub')
models={}
for model,spec in cfg['models'].items():
 candidates=[]
 for root in sorted(roots):
  p=root/('models--'+spec['model_id'].replace('/','--'))/'snapshots'/spec['revision']
  if not p.is_dir():continue
  files={}
  for name,expected in spec['files_sha256'].items():
   f=p/name;actual=hashlib.sha256(f.read_bytes()).hexdigest() if f.is_file() else None
   files[name]=dict(present=f.is_file(),sha256=actual,expected_sha256=expected,matches=actual==expected)
  index=p/'model.safetensors.index.json';weights={}
  index_error=None
  if index.is_file():
   try:
    for name in sorted(set(json.loads(index.read_text())['weight_map'].values())):
     f=p/name;weights[name]=dict(present=f.is_file(),bytes=f.stat().st_size if f.is_file() else None)
   except (ValueError,KeyError,OSError,TypeError) as e:index_error=type(e).__name__+': '+str(e)
  candidates.append(dict(path=str(p),config_tokenizer=files,weight_index_present=index.is_file(),weight_index_error=index_error,weight_index_bytes=index.stat().st_size if index.is_file() else None,weights=weights))
 models[model]=dict(revision=spec['revision'],searched_cache_roots=list(map(str,sorted(roots))),snapshots=candidates,availability='present_in_searched_roots' if candidates else 'not_found_in_searched_roots')
root=Path('/workspace/truth-probing-artifacts');artifacts={}
paths=set(cfg['historical_file_sha256'])|set(cfg['tensors'])
for rel in sorted(paths):
 p=root/rel;artifacts[str(p)]=dict(present=p.is_file(),bytes=p.stat().st_size if p.is_file() else None)
volume=Path('/workspace');disk=shutil.disk_usage(volume)
print(json.dumps(dict(hostname=platform.node(),system=platform.system(),python=platform.python_version(),versions=versions,cuda=cuda,nvidia_devices=glob.glob('/dev/nvidia*'),nvidia_smi=query(['nvidia-smi','--query-gpu=name,driver_version,memory.total,memory.free','--format=csv']),nvidia_driver=query(['cat','/proc/driver/nvidia/version']),pinned_models=models,artifacts=artifacts,storage=dict(path=str(volume),free_bytes=disk.free,total_bytes=disk.total,write_access_flag=os.access(volume,os.W_OK),filesystem=query(['df','-T','/workspace']),mount=query(['findmnt','-no','SOURCE,FSTYPE,OPTIONS','-T','/workspace']),quota=query(['quota','-s']),practical_quota_verified=False,readiness_note='Read-only stat/access indicators only; virtual free space does not establish account quota, durable allocation or a successful write. No write probe performed.')),allow_nan=False))
'''


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise SystemExit('Refuse overwrite')
    plan=Plan();program=REMOTE.replace('CONFIG_PLACEHOLDER',repr(json.dumps(plan.config)))
    command=['ssh','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-p','22114','-i','/Users/apple/.ssh/id_ed25519','root@69.30.85.22','python -B -']
    start=time.monotonic()
    try:
        result=subprocess.run(command,input=program,text=True,capture_output=True,timeout=55)
        receipt=dict(returncode=result.returncode,stderr=result.stderr,remote=json.loads(result.stdout) if result.returncode==0 else None,stdout_on_failure=result.stdout if result.returncode else None)
    except subprocess.TimeoutExpired as e:receipt=dict(status='readiness_timeout',error=str(e))
    receipt.update(local_host=platform.node(),local_system=platform.system(),command=command,setup_seconds=time.monotonic()-start,inference_executed=False,remote_writes=False,downloads=False,code_sha256=digest(Path(__file__)))
    write_new(a.output,receipt);print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
