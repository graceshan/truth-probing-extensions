"""Immutable publication and strict recovery for behavior/selected-layer chat shards."""
import json
import os
import uuid
from pathlib import Path
from src.checkpoint_next_inventory import digest, require, sha


def publish_json(path, value):
    # Reviewed exclusive-link + fsync writer (never replaces an existing artifact).
    from src.checkpoint_r2_fresh_store import write_json
    write_json(Path(path), value)


def same_or_publish(path, value):
    path=Path(path)
    if path.exists():
        require(json.loads(path.read_text())==value, 'immutable metadata mismatch: '+str(path))
    else:
        publish_json(path,value)


def validate_record(r, job, layers):
    import math
    require(r['id']==job['id'] and r['job_sha256']==digest(job), 'record/job binding')
    p=r['packet'];n=len(p['prompt_ids'])
    require(n>0 and p['attention_mask']==[1]*n and p['position_ids']==list(range(n)) and
            p['readout_index']==n-1 and p['boundary_verified'] and p['system_messages']==0, 'token/readout packet')
    require(set(p['candidates'])=={'True','False'} and all(p['candidates'].values()), 'candidate coverage')
    require(not any(set(v)&set(p['special_token_ids']) for v in p['candidates'].values()), 'special token in scored continuation')
    require(('likelihood' in r)==job['behavior'] and ('generation' in r)==job['behavior'] and
            ('chat_index' in r)==job['chat'], 'measurement coverage')
    if job['behavior']:
        l=r['likelihood'];margin=l['true']['logprob']-l['false']['logprob']
        require(math.isfinite(margin) and l['margin']==margin and l['decision']==int(margin>=0), 'margin/rule')
        for a in ('True','False'):
            terms=l[a.lower()]['token_logprobs']
            require(len(terms)==len(p['candidates'][a]) and all(math.isfinite(t) and t<=0 for t in terms)
                    and l[a.lower()]['logprob']==math.fsum(terms), 'full continuation sum')
        g=r['generation']
        require(len(g['token_ids'])<=8 and g['canonical']==g['content'].strip(), 'generation cap/parser')
        cleaned=list(g['token_ids'])
        while cleaned and cleaned[-1] in g['termination_ids']:
            cleaned.pop()
        require(g['content_token_ids']==cleaned and set(g['termination_ids'])<=set(p['special_token_ids']), 'native termination suffix')
        valid=g['canonical'] in ('True','False') and not set(cleaned)&set(p['special_token_ids'])
        require(g['format_failure']==(not valid), 'format failure parser mismatch')
        require(g['format_failure']==(g['decision'] is None) and
                (g['format_failure'] or g['decision']==int(g['canonical']=='True') and g['canonical'] in ('True','False')), 'strict format rule')
    if job['chat']:
        require(r['saved_layers']==layers, 'selected chat layer binding')


def verify_shard(path, jobs, identity, layers, width):
    import numpy as np
    path=Path(path)
    commit=json.loads((path/'commit.json').read_text())
    require(commit['identity']==identity and commit['jobs_sha256']==digest(jobs), 'shard producer/config/input identity')
    names={'records.json'} | ({'chat.npy'} if any(j['chat'] for j in jobs) else set())
    require(set(commit['files'])==names, 'shard file coverage')
    for name,h in commit['files'].items():
        require(sha(path/name)==h,'corrupt committed shard: '+name)
    records=json.loads((path/'records.json').read_text())
    require(len(records)==len(jobs),'shard record count')
    for r,j in zip(records,jobs):
        validate_record(r,j,layers)
    if 'chat.npy' in names:
        x=np.load(path/'chat.npy',allow_pickle=False,mmap_mode='r')
        require(x.dtype==np.float16 and x.shape==(sum(j['chat'] for j in jobs),2,width) and np.isfinite(x).all(),'selected chat payload')
        require([r['chat_index'] for r in records if 'chat_index' in r]==list(range(len(x))),'chat vector order')
    return commit


def recover(path, jobs, identity, layers, width, quarantine):
    path=Path(path)
    if not path.exists():
        return False
    require(path.is_dir() and not path.is_symlink(),'unexpected shard path')
    if (path/'commit.json').exists():
        verify_shard(path,jobs,identity,layers,width)  # corruption blocks; never rerun/overwrite.
        return True
    # Preserve all interrupted outputs for diagnosis; no cleanup/reset.
    quarantine=Path(quarantine);quarantine.mkdir(parents=True,exist_ok=True)
    os.rename(path,quarantine/(path.name+'-'+uuid.uuid4().hex))
    from src.checkpoint_r2_fresh_store import fsync_dir
    fsync_dir(path.parent);fsync_dir(quarantine)
    return False


def save_shard(path, jobs, records, chat, identity, layers, width):
    import numpy as np
    from src.checkpoint_r2_fresh_store import publish
    path=Path(path);path.mkdir(parents=True,exist_ok=False)
    publish_json(path/'records.json',records)
    if chat:
        x=np.stack(chat)
        require(x.dtype==np.float16,'chat stored dtype')
        publish(path/'chat.npy',lambda f:np.save(f,x,allow_pickle=False))
    names=['records.json']+(['chat.npy'] if chat else [])
    publish_json(path/'commit.json',dict(schema='checkpoint-next-shard-v1',identity=identity,
        jobs_sha256=digest(jobs),files={n:sha(path/n) for n in names},saved_layers=layers))
    return verify_shard(path,jobs,identity,layers,width)
