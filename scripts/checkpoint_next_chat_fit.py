"""Future eight-head CPU runner. Use only after GPU transfer verification and CPU-slot review."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.checkpoint_next_inventory import require, digest, sha
from src.checkpoint_next_runner import frozen,producer,verify
from src.checkpoint_next_reporting import match_raw_chat
from src.checkpoint_next_store import publish_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for field in ('input','output','authorization','raw-matches'):p.add_argument('--'+field,required=True)
    p.add_argument('--worker',action='store_true');a=p.parse_args();x=frozen();auth=json.loads(Path(a.authorization).read_text())
    require(auth['phase']=='chat_cpu_fit' and auth['reviewed'] and auth['producer_commit']==producer()
        and auth['heads']==8 and auth['CPU_slot_exclusive'] and auth['E'] is False,'reviewed eight-head CPU scope required')
    require(auth['config_sha256']==x['config_sha256'] and auth['inventory_sha256']==digest(x),'CPU authorization freeze')
    require(auth['raw_matches_sha256']==sha(a.raw_matches),'raw comparator export binding')
    output=Path(a.output)
    if not a.worker:
        require(not output.exists(),'CPU output must be new; preserve existing results')
        output.mkdir(parents=True)
        # Chat CPU cap is separately reviewed after measuring costs; the document's
        # 3-hour full-bank CPU cap is not silently reassigned to these chat fits.
        require(isinstance(auth['cpu_seconds'],(int,float)) and auth['cpu_seconds']>0,'reviewed chat CPU stopping limit required')
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1')
        command=[sys.executable,__file__]+sys.argv[1:]+['--worker']
        try:subprocess.run(command,check=True,timeout=auth['cpu_seconds'],env=env)
        except subprocess.TimeoutExpired:
            publish_json(output/'halt.json',dict(status='chat CPU cap hit',fits_retained=True));raise
        return
    import numpy as np
    from src.checkpoint_next_chat import prepare_blocks,fit_prepared
    from src.checkpoint_r2_fresh_store import publish
    raw=json.loads(Path(a.raw_matches).read_text())
    require(len(raw)==8 and len({r['id'] for r in raw})==8,'exactly eight raw matches required')
    matches={r['id']:r for r in raw};artifacts=[]
    for model in ('qwen','llama'):
        verification=verify(a.input,'launch',model)
        physical={}
        for path in sorted((Path(a.input)/'launch'/model/'shards').glob('*/records.json')):
            rows=json.loads(path.read_text())
            state=np.load(path.parent/'chat.npy',allow_pickle=False,mmap_mode='r') if (path.parent/'chat.npy').exists() else None
            for r in rows:
                if 'chat_index' in r:physical[r['id']]=np.array(state[r['chat_index']],copy=True)
        groups={g:[r for r in x['bindings'] if r['role']=='chat' and r['group']==g] for g in ('P15','TC_control_and_final_refit','behavior_D')}
        for head in (h for h in x['heads'] if h['model']==model):
            require(head['id'] in matches,'missing raw same-family/layer comparator')
            metadata=dict(head,preprocessing_membership='original_P15',P15_identities_sha256=digest([r['logical_id'] for r in groups['P15']]),regime='prompt1_chat')
            match_raw_chat(matches[head['id']],metadata)
            slot=head['layer']
            from src.checkpoint_next_inventory import config
            layer_index=config()['layers'][model].index(slot)
            features={g:np.stack([physical[r['job_id']][layer_index] for r in rs]).astype(np.float64) for g,rs in groups.items()}
            pre,block=prepare_blocks(head,groups['P15'],features['P15'],groups['TC_control_and_final_refit'],features['TC_control_and_final_refit'])
            start=time.monotonic();fit=fit_prepared(head,pre,block,auth,producer())
            folder=output/head['id'];folder.mkdir(parents=True,exist_ok=False)
            publish(folder/'parameters.npz',lambda f:np.savez(f,weights=fit.weights,intercept=fit.intercept,
                mean=pre.mean,population_std=pre.population_std,denominator=pre.denominator,constant_mask=pre.constant_mask))
            publish(folder/'scores.npz',lambda f:np.savez(f,**{g:fit.decision_function(v) for g,v in features.items()}))
            attempts=[]
            for attempt in fit.attempts:
                row=asdict(attempt);row.pop('initial_parameters');row.pop('final_parameters');attempts.append(row)
            receipt=dict(head=metadata,producer_commit=producer(),verification=verification,objective=head['objective'],
                attempts=attempts,status=fit.status,seconds=time.monotonic()-start,
                preprocessing_source='original chat-P15 only',parameters_sha256=sha(folder/'parameters.npz'),
                scores_sha256=sha(folder/'scores.npz'),score_identities={g:[r['logical_id'] for r in rs] for g,rs in groups.items()})
            publish_json(folder/'receipt.json',receipt);artifacts.append(dict(id=head['id'],receipt_sha256=sha(folder/'receipt.json')))
        del physical
    require(len(artifacts)==8,'eight valid fits required')
    publish_json(output/'completion.json',dict(status='completed',producer_commit=producer(),heads=artifacts,
        interpretation='Prompt 1 named readouts only; no prompted constituent availability or causal use'))


if __name__=='__main__':main()
