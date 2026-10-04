"""Exact reviewed missing-raw supplement. Metadata preparation never loads tensors.

Extraction is a phase of checkpoint_next_runner, sharing its lock and durable
attempt ledger with behavior preflight/launch. Existing raw outputs are read-only.
"""
import csv
import json
import os
import time
import uuid
from pathlib import Path
from src.checkpoint_next_inventory import ROOT, require, sha, config
from src.checkpoint_r2_fresh_inputs import hash_value, text_hash
from src.checkpoint_next_store import publish_json, same_or_publish

SETTINGS=ROOT/'config/checkpoint_next/integration_fix_v1.json'
QUEUE=ROOT/'results/checkpoint_next_score_diagnosis_20261003/missing-input-queue.json'


def settings():
    c=json.loads(SETTINGS.read_text())
    for name,h in c['evidence_sha256'].items():require(sha(ROOT/name)==h,'integration evidence changed: '+name)
    return c


def validate_queue():
    settings();q=json.loads(QUEUE.read_text());c=config()
    from src.checkpoint_r2_full_inputs import reconstruct
    raw=json.loads((ROOT/'config/checkpoint_r2/full_raw_v1.json').read_text())
    base=ROOT/'data/checkpoint_r2_v1'
    with (base/'extraction_inventory.csv').open() as handle:raw_inventory=list(csv.DictReader(handle))
    with (base/'compound_bindings.csv').open() as handle:compounds=list(csv.DictReader(handle))
    manifest=reconstruct(raw_inventory,json.loads((base/'memberships.json').read_text()),
        json.loads((base/'facts.json').read_text()),compounds,set(raw['raw_groups']))
    manifest['fp16_payload_bytes']={m:manifest['unique_texts']*spec['layers']*spec['width']*2 for m,spec in raw['models'].items()}
    require(hash_value(manifest)==q['accepted_manifest_sha256'],'accepted raw manifest reconstruction')
    require(q['model_contracts']==c['models'] and q['numerical_contract']==c['execution'],'raw model/numerical contract')
    members=json.loads((ROOT/'data/checkpoint_r2_v1/memberships.json').read_text())
    train=members['outer_train'];require(hash_value(train)==q['outer_train_membership_sha256'],'outer train identity')
    existing={r['id'] for r in manifest['rows']};rebuilt={}
    for r in train:
        tid='text_'+text_hash(r['statement'])
        require(r['statement_sha256']==text_hash(r['statement']),'source text hash')
        if tid not in existing:
            rebuilt.setdefault(tid,dict(text_id=tid,statement=r['statement'],statement_sha256=r['statement_sha256'],
                topic=r['topic'],split='train',bindings=[]))['bindings'].append(r)
    require(list(rebuilt.values())==q['rows'],'exact queue order/membership mismatch')
    require(len(q['rows'])==q['unique_texts_per_model']==131 and q['model_input_combinations']==262,'queue counts')
    with (QUEUE.parent/'full-training-coverage.csv').open() as handle:
        coverage=list(csv.DictReader(handle))
    require(len(coverage)==6080,'coverage count')
    for model in ('qwen','llama'):
        rows=[r for r in coverage if r['model']==model]
        require([r['source_row_id'] for r in rows]==[r['source_row_id'] for r in train],'coverage exact ordered train')
        for r,b in zip(rows,train):
            missing=r['text_id'] in rebuilt
            require(r['text_id']=='text_'+b['statement_sha256'] and r['source_metadata_sha256']==hash_value(b)
                and r['status']==('missing_unique_text' if missing else 'exact_compatible_fresh_reuse'),'coverage source/status')
            require(bool(r['representation_receipt_sha256'])!=missing,'coverage receipt state')
    return q


def raw_rows():
    return [dict(id=r['text_id'],**{k:v for k,v in r.items() if k!='text_id'}) for r in validate_queue()['rows']]


def metadata():
    q=validate_queue();c=config()
    return dict(queue_sha256=sha(QUEUE),unique_texts_per_model=131,model_input_combinations=262,
        missing_bindings=131,additional_bindings=262,additional_reused_bindings=131,
        outer_train=3040,P15=2778,coverage_rows=6080,
        payload_bytes={m:131*s['layers']*s['width']*2 for m,s in c['models'].items()},
        execution_status='not executed',token_binding='pending pinned offline tokenizer',
        runtime_seconds=None,peak_cuda_bytes=None,storage_quota=None,
        shared_cap_seconds_per_model=7200,forecast='unmeasured; payload is exact, metadata/reserves are allowances')


def provenance(model, identity):
    return dict(identity,model=model,queue_sha256=sha(QUEUE),contract_sha256=sha(ROOT/'config/checkpoint_r2/full_raw_v1.json'),
                batch_size=1,padding='none',compute_dtype='bfloat16',stored_dtype='float16')


def verify_supplement(output, model, identity):
    """Independent loader checks actual supplement; never accepts a caller-supplied verified flag."""
    from src.checkpoint_r2_fresh_store import load_features
    from src.checkpoint_r2_fresh_pilot import semantic_view
    rows=raw_rows();root=Path(output)/'missing_raw'/model;spec=config()['models'][model]
    require(json.loads((root/'identity.json').read_text())==identity,'supplement identity')
    plan=json.loads((root/'token-plan.json').read_text());require(len(plan)==131,'token coverage')
    names={f'{i:05d}' for i in range(131)}
    require({p.name for p in (root/'shards').iterdir()}==names,'supplement exact coverage')
    receipts=[]
    for i,row in enumerate(rows):
        path=root/'shards'/f'{i:05d}'
        values,got,receipt=load_features(path,expected_rows=[row],expected_spec=spec,expected_provenance=provenance(model,identity))
        tokens=json.loads((path/'tokens.json').read_text())
        require(tokens==[plan[i]],'raw token plan/forward binding')
        semantic_view(tokens,[row]);p=tokens[0];n=len(p['token_ids'])
        require(p['token_ids']==p['unpadded_token_ids'] and p['attention_mask']==[1]*n and
            p['position_ids']==list(range(n)) and p['readout_index']==n-1 and p['semantic_readout_position']==n-1
            and p['add_special_tokens'] and not p['chat_template'] and not p['truncation'],'raw unpadded contract')
        receipts.append(dict(path=f'shards/{i:05d}',sha256=sha(path/'receipt.json'),tokens_sha256=hash_value(tokens)))
        del values
    return dict(identity=identity,queue_sha256=sha(QUEUE),coverage_evidence_sha256=sha(QUEUE.parent/'full-training-coverage.csv'),
        status='loader_verified',unique_texts=131,bindings=131,receipts=receipts,token_plan_sha256=sha(root/'token-plan.json'))


def extract(args, backend, identity, costs):
    """Called only by the authorized watchdog child; never writes accepted original raw."""
    from src.checkpoint_r2_full_support import single_packet
    from src.checkpoint_r2_fresh_store import write_shard, load_features, fsync_dir, storage_probe
    rows=raw_rows();root=Path(args.output)/'missing_raw'/args.model;root.mkdir(parents=True,exist_ok=True)
    same_or_publish(root/'identity.json',identity)
    t=time.monotonic();plan=[single_packet(backend.tokenizer,r,backend.settings['max_tokens']) for r in rows]
    costs['tokenization']=time.monotonic()-t
    t=time.monotonic();same_or_publish(root/'token-plan.json',plan)
    # Bounded allocation proof; retain original outputs and a second supplement copy.
    reserve=2*(131*backend.spec['layers']*backend.spec['width']*2+16*1024**2)
    storage=storage_probe(root,reserve)
    publish_json(Path(args.attempt)/'missing-storage.json',storage)
    costs['writing_verification']+=time.monotonic()-t
    shards=root/'shards';shards.mkdir(exist_ok=True)
    require(all(p.name in {f'{i:05d}' for i in range(131)} for p in shards.iterdir()),'unexpected raw shard')
    for i,row in enumerate(rows):
        path=shards/f'{i:05d}';t=time.monotonic()
        if path.exists():
            require(path.is_dir() and not path.is_symlink(),'raw shard path')
            if (path/'receipt.json').exists():
                load_features(path,expected_rows=[row],expected_spec=backend.spec,expected_provenance=provenance(args.model,identity))
                require(json.loads((path/'tokens.json').read_text())==[plan[i]],'resumed raw token binding')
                costs['writing_verification']+=time.monotonic()-t
                continue
            quarantine=root/'quarantine';quarantine.mkdir(exist_ok=True)
            os.rename(path,quarantine/(path.name+'-'+uuid.uuid4().hex));fsync_dir(shards);fsync_dir(quarantine)
        t=time.monotonic();values,tokens=backend.forward([row],1)
        costs['raw_extraction']+=time.monotonic()-t
        require(tokens==[plan[i]],'raw forward vs token plan')
        t=time.monotonic();write_shard(path,[row],values,backend.spec,provenance(args.model,identity),tokens)
        costs['writing_verification']+=time.monotonic()-t
    t=time.monotonic();receipt=verify_supplement(args.output,args.model,identity)
    costs['writing_verification']+=time.monotonic()-t
    return dict(identity=identity,phase='missing_raw',status='completed',jobs=131,verification=receipt,
                storage=storage,token_plan_sha256=sha(root/'token-plan.json'))


def audit_tokens(output,model,identity,cache):
    """Independent offline tokenizer-only process after extraction; no model loading."""
    import inspect
    from transformers import AutoTokenizer
    from src.checkpoint_r2_full_support import evidence,verify_snapshot,single_packet
    c=config();spec=c['models'][model]
    snapshot=verify_snapshot(evidence(c,model+'-snapshot.json'),spec,cache)
    tokenizer=AutoTokenizer.from_pretrained(snapshot['snapshot'],local_files_only=True,trust_remote_code=False,use_fast=True)
    expected=evidence(c,model+'/runtime-before-inference.json')['backend']
    require(text_hash(tokenizer.backend_tokenizer.to_str())==expected['tokenizer_backend_sha256'] and
            sha(inspect.getfile(type(tokenizer)))==expected['tokenizer_implementation_sha256'],'independent tokenizer implementation')
    rows=raw_rows();packets=[single_packet(tokenizer,r,c['execution']['max_tokens']) for r in rows]
    root=Path(output)/'missing_raw'/model
    require(packets==json.loads((root/'token-plan.json').read_text()),'independent native token reconstruction')
    return dict(identity=identity,queue_sha256=sha(QUEUE),token_plan_sha256=sha(root/'token-plan.json'),
                tokenizer_backend_sha256=expected['tokenizer_backend_sha256'],
                tokenizer_implementation_sha256=expected['tokenizer_implementation_sha256'],
                rows=131,status='independently_reconstructed',model_loaded=False)
