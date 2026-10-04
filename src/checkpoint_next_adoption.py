"""Read-only adoption of the exact failed campaign; never relabel its artifacts.

An adoption is location-bound for execution, portable for read-only verification,
and valid only with the pinned original handoff and all settled attempt receipts.
"""
import argparse
import json
import math
from pathlib import Path
from src.checkpoint_next_inventory import ROOT, config, require, sha
from src.checkpoint_next_store import publish_json, verify_shard

POLICY=ROOT/'config/checkpoint_next/raw_only_fix_v1.json'


def policy():
    p=json.loads(POLICY.read_text())
    for name,h in p['scientific_files'].items():
        require(sha(ROOT/name)==h,'successor scientific contract changed: '+name)
    require(p['shared_model_seconds']==7200 and p['setup_seconds']==1800,'frozen successor caps/tolerance')
    return p


def phase_allowed(model, phase):
    p=policy()
    require(phase in p['allowed_phases'][model],
            'successor phase blocked: Qwen raw already accepted; behavior/chat deferred; no launch')


def verify_original(root, manifest, x):
    p=policy();root=Path(root).resolve();manifest=Path(manifest)
    require(sha(manifest)==p['original_manifest_sha256'],'unapproved original handoff manifest')
    files=json.loads(manifest.read_text())
    actual={str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}
    require(actual==set(files),'original handoff file coverage changed')
    for rel,h in files.items():
        f=root/rel
        require(not Path(rel).is_absolute() and '..' not in Path(rel).parts and
                f.resolve().is_relative_to(root) and not f.is_symlink(),'original handoff path')
        require(sha(f)==h,'original artifact changed: '+rel)
    charges={}
    for model,entries in p['original_attempts'].items():
        actual={f.parent.name for f in (root/'attempts'/model).glob('*/start.json')}
        require(actual=={e['attempt'] for e in entries},'original attempt set changed')
        charges[model]=0.
        for e in entries:
            folder=root/'attempts'/model/e['attempt']
            require(sha(folder/'start.json')==e['start_sha256'] and sha(folder/'end.json')==e['end_sha256'],
                    'original ledger hash')
            end=json.loads((folder/'end.json').read_text());charge=end['active_seconds']
            require(math.isfinite(charge) and charge==e['active_seconds'] and charge>=0,'original ledger charge')
            charges[model]+=charge
    from src.checkpoint_next_runner import _missing_acceptance
    acceptance=_missing_acceptance(root,'qwen',p['original_identities']['qwen'])
    require(sha(root/'missing_raw/qwen/acceptance.json')==p['original_acceptance_sha256'] and
            acceptance==json.loads((root/'missing_raw/qwen/acceptance.json').read_text()),'original accepted supplement')
    c=config();partial=root/'preflight/qwen'
    require(not (partial/'completion.json').exists(),'partial evidence unexpectedly completed')
    require({f.name for f in (partial/'shards').iterdir()}=={'00000'},'partial shard coverage')
    require([j['id'] for j in x['preflight'][:16]]==p['qwen_partial_ids'],'frozen progress order')
    verify_shard(partial/'shards/00000',x['preflight'][:16],p['original_identities']['qwen'],
                 c['layers']['qwen'],c['models']['qwen']['width'])
    return dict(original_producer=p['original_producer'],original_manifest_sha256=sha(manifest),
        original_identities=p['original_identities'],original_attempts=p['original_attempts'],carry_seconds=charges,
        qwen_acceptance_sha256=p['original_acceptance_sha256'],qwen_accepted_raw_shards=131,
        qwen_partial_ids=p['qwen_partial_ids'],
        partial_is_full_acceptance=False,llama_acceptance=None,scientific_files=p['scientific_files'])


def create(output, original, manifest, x):
    from src.checkpoint_next_runner import producer
    out=Path(output).resolve()
    require(not out.exists(),'adoption requires a new campaign; resume existing adoption instead')
    evidence=verify_original(original,manifest,x)
    # Identity is the successor commit, not a caller-selected arbitrary substitute.
    require(producer()!=policy()['original_producer'],'successor producer required')
    out.mkdir(parents=True)
    receipt=dict(schema='checkpoint-next-adoption-v1',successor_producer=producer(),
        policy_sha256=sha(POLICY),execution_output=str(out),original=evidence,
        full_preflight_acceptance=False)
    publish_json(out/'adoption.json',receipt)
    return receipt


def receipt(output):
    from src.checkpoint_next_runner import producer
    out=Path(output).resolve();r=json.loads((out/'adoption.json').read_text())
    require(r['schema']=='checkpoint-next-adoption-v1' and r['successor_producer']==producer() and
            r['policy_sha256']==sha(POLICY) and r['full_preflight_acceptance'] is False,'unsupported producer/adoption substitution')
    p=policy()
    require(r['original']['original_producer']==p['original_producer'] and
            r['original']['original_attempts']==p['original_attempts'] and
            r['original']['original_identities']==p['original_identities'] and
            r['original']['original_manifest_sha256']==p['original_manifest_sha256'] and
            r['original']['carry_seconds']=={m:sum(e['active_seconds'] for e in es) for m,es in p['original_attempts'].items()},
            'unsupported original producer/ledger substitution')
    return r


def verify(output, original, manifest, x, execution=False):
    out=Path(output).resolve();r=receipt(out)
    if execution:require(r['execution_output']==str(out),'execution campaign relocated; review required, no reset')
    require(r['original']==verify_original(original,manifest,x),'original adoption evidence changed')
    return r


def budget(output, model, adoption):
    from src.checkpoint_next_runner import attempts_elapsed
    carry=adoption['original']['carry_seconds'][model]
    own=attempts_elapsed(output,model)
    require(math.isfinite(own) and own>=0,'successor charge')
    total=carry+own
    return dict(original_seconds=carry,successor_seconds=own,charged_seconds=total,remaining_seconds=7200-total)


def main():
    from src.checkpoint_next_runner import frozen
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['create','verify']);p.add_argument('--output',required=True)
    p.add_argument('--original-campaign',required=True);p.add_argument('--original-manifest',required=True)
    a=p.parse_args();fn=create if a.action=='create' else verify
    print(json.dumps(fn(a.output,a.original_campaign,a.original_manifest,frozen()),indent=2))


if __name__=='__main__':main()
