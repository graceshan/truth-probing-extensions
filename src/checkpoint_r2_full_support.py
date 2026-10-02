"""Offline snapshot reuse, prospective storage reservation, and independent handoff checks."""
import json
import os
import platform
import socket
import subprocess
import time
import tempfile
import fcntl
from pathlib import Path
from src.checkpoint_r2_fresh_inputs import ROOT, canonical, file_hash, hash_value, require, text_hash
from src.checkpoint_r2_fresh_store import load_features, pin, storage_probe, write_json
from src.checkpoint_r2_fresh_pilot import semantic_view


def evidence(cfg, name):
    p = ROOT / 'data/checkpoint_r2_full_raw_v1/pilot' / name
    require(file_hash(p) == cfg['evidence_sha256'][str(p.relative_to(ROOT))], 'pilot evidence hash')
    return json.loads(p.read_text())


def verify_snapshot(receipt, spec, cache):
    cache = Path(cache).resolve()
    path = cache / ('models--' + spec['model_id'].replace('/', '--')) / 'snapshots' / spec['revision']
    require(receipt['model_id'] == spec['model_id'] and receipt['revision'] == receipt['tokenizer_revision'] == spec['revision'], 'snapshot identity')
    actual, inodes = {}, {}
    require(path.is_dir(), 'verified offline snapshot missing; no automatic download')
    require(set(p.name for p in path.iterdir()) == set(receipt['files']), 'snapshot file set')
    for name, old in receipt['files'].items():
        require(Path(name).name == name, 'snapshot filename')
        p = path / name
        resolved = p.resolve(strict=True)
        require(resolved.is_relative_to(cache), 'snapshot escapes approved cache')
        identity = pin(p)
        require(identity == {k:old[k] for k in ('sha256', 'bytes')}, 'snapshot file corrupt: ' + name)
        actual[name] = dict(identity, path=str(p), resolved_path=str(resolved))
        st = resolved.stat(); inodes[(st.st_dev, st.st_ino)] = st.st_size
    index = json.loads((path/'model.safetensors.index.json').read_text())
    weights = set(index['weight_map'].values())
    require(all(Path(n).name == n and n.endswith('.safetensors') for n in weights), 'weight index names')
    require(set(actual) == weights | set(spec['files_sha256']) | {'model.safetensors.index.json'}, 'missing/unexpected weights')
    require(all(actual[n]['sha256'] == h for n,h in spec['files_sha256'].items()), 'frozen model/tokenizer files')
    return dict(model_id=spec['model_id'], revision=spec['revision'], tokenizer_revision=spec['revision'],
                snapshot=str(path), files=actual, downloaded_bytes=sum(inodes.values()), new_download_bytes=0,
                policy='read-only offline reuse of hash-verified pilot snapshot')


def verify_cached_models(cfg, cache):
    return {m:verify_snapshot(evidence(cfg,m+'-snapshot.json'),s,cache) for m,s in cfg['models'].items()}


def single_packet(tokenizer, row, max_tokens):
    tokens = tokenizer.encode(row['statement'], add_special_tokens=True, truncation=False)
    encoded = tokenizer([row['statement']], add_special_tokens=True, truncation=False, padding=False, return_special_tokens_mask=True)
    raw = tokenizer.encode(row['statement'], add_special_tokens=False, truncation=False)
    special = encoded['special_tokens_mask'][0]
    require(encoded['input_ids'] == [tokens] and 0 < len(tokens) <= max_tokens, 'tokenization/truncation bound')
    require([t for t,s in zip(tokens,special) if not s] == raw, 'native special tokens')
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    n = len(tokens)
    packet = dict(id=row['id'],statement_sha256=row['statement_sha256'],unpadded_token_ids=tokens,
        raw_token_ids=raw,unpadded_special_tokens_mask=special,token_ids=tokens,attention_mask=[1]*n,
        position_ids=list(range(n)),readout_index=n-1,semantic_readout_position=n-1,pad_token_id=pad,
        add_special_tokens=True,truncation=False,chat_template=False)
    semantic_view([packet],[row])
    return packet


def token_storage_plan(cfg, manifest, snapshots, tokenizer_factory=None):
    real_tokenizer = tokenizer_factory is None
    if real_tokenizer:
        from transformers import AutoTokenizer
        tokenizer_factory = lambda p:AutoTokenizer.from_pretrained(p,local_files_only=True,trust_remote_code=False,use_fast=True)
    models, complete = {}, len(canonical(manifest)) + cfg['storage']['global_receipt_reserve_bytes']
    largest = 0
    for m,spec in cfg['models'].items():
        tokenizer = tokenizer_factory(snapshots[m]['snapshot'])
        expected = evidence(cfg,m+'/runtime-before-inference.json')['backend']
        if real_tokenizer:
            require(text_hash(tokenizer.backend_tokenizer.to_str()) == expected['tokenizer_backend_sha256'], 'tokenizer implementation/backend changed')
        per_shard, total, lengths = [], 0, []
        for start in range(0,len(manifest['rows']),cfg['shard_rows']):
            rows=manifest['rows'][start:start+cfg['shard_rows']]
            packets=[single_packet(tokenizer,r,cfg['execution']['max_tokens']) for r in rows]
            lengths.extend(len(p['token_ids']) for p in packets)
            payload=len(rows)*spec['layers']*spec['width']*2
            reserved=(payload+128+len(canonical(rows))+len(canonical(packets))+
                      cfg['storage']['per_shard_receipt_reserve_bytes']+cfg['storage']['per_shard_commit_reserve_bytes'])
            per_shard.append(dict(start=start,rows=len(rows),tokens_sha256=hash_value(packets),reserved_bytes=reserved))
            total+=reserved;largest=max(largest,reserved)
        models[m]=dict(shards=per_shard,complete_shard_bytes=total,token_min=min(lengths),token_max=max(lengths))
        complete+=total
    return dict(models=models,complete_copy_reserved_bytes=complete,temporary_shard_reserved_bytes=largest,
                second_copy_reserved_bytes=complete,full_tokens_preflight=True)


def volume_inventory(volume):
    volume=Path(volume).resolve(); seen={}
    def fail(error):raise error
    for parent, dirs, names in os.walk(volume, followlinks=False,onerror=fail):
        for name in names:
            p=Path(parent)/name
            if p.is_symlink():continue  # real blob target is walked separately; never count symlink aliases twice
            if p.is_file():
                st=p.stat();seen[(st.st_dev,st.st_ino)]=st.st_size
    return dict(unique_regular_files=len(seen),retained_unique_file_bytes=sum(seen.values()),
                accounting='logical regular-file sizes, unique (device,inode); actual allocation test establishes usable capacity')


def filesystem_prerequisites(volume):
    with tempfile.TemporaryDirectory(prefix='.r2-full-fs-',dir=volume) as directory:
        root=Path(directory);lock=root/'lock';lock.write_bytes(b'lock');lock.chmod(0o600)
        require(lock.stat().st_mode & 0o777==0o600,'chmod prerequisite')
        subprocess.run(['git','init','--quiet',str(root/'git')],check=True)
        with lock.open('r') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            code="import fcntl,sys; f=open(sys.argv[1]); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
            check=subprocess.run([__import__('sys').executable,'-c',code,str(lock)],capture_output=True)
            require(check.returncode!=0 and b'BlockingIOError' in check.stderr,'filesystem lock exclusion prerequisite')
    return ['chmod','git_init','cross_process_flock_exclusion']


def additional_capacity(plan, cfg, valid_primary_bytes=0):
    return (max(0,plan['complete_copy_reserved_bytes']-valid_primary_bytes)+plan['second_copy_reserved_bytes']+
            plan['temporary_shard_reserved_bytes']+cfg['storage']['safety_margin_bytes'])


def storage_check(cfg, manifest, cache, volume, destination, receipt_path):
    volume=Path(volume).resolve();destination=Path(destination).resolve()
    require(platform.system()=='Linux' and os.path.ismount(volume), 'actual Linux persistent mount required')
    require(destination.is_relative_to(volume) and Path(cache).resolve().is_relative_to(volume), 'same-volume cache/output storage plan')
    require(not destination.is_relative_to(Path(cache).resolve().parent) and not destination.is_relative_to(ROOT), 'preserve pilot/cache checkout')
    from src.checkpoint_r2_full_raw import valid_primary_bytes
    valid_primary_bytes_count=valid_primary_bytes(destination,cfg,manifest)
    prerequisites=filesystem_prerequisites(volume)
    snapshots=verify_cached_models(cfg,cache)
    plan=token_storage_plan(cfg,manifest,snapshots)
    mount=subprocess.check_output(['findmnt','-n','-o','TARGET,SOURCE,FSTYPE','--target',str(volume)],text=True).strip()
    retained=volume_inventory(volume)
    needed=additional_capacity(plan,cfg,valid_primary_bytes_count)
    # Incompressible bytes coexist with ALL retained files; no virtual statvfs quota claim.
    probe=storage_probe(volume,needed)
    receipt=dict(schema='r2-full-raw-storage-v1',verified_at_unix=time.time(),volume=str(volume),mount=mount,
        destination=str(destination),manifest_sha256=hash_value(manifest),contract_sha256=file_hash(ROOT/'config/checkpoint_r2/full_raw_v1.json'),
        filesystem_prerequisites=prerequisites,cached_weights_counted_once_bytes=sum(s['downloaded_bytes'] for s in snapshots.values()),new_weight_download_bytes=0,
        retained=retained,plan=plan,valid_primary_bytes=valid_primary_bytes_count,additional_capacity_verified_bytes=needed,probe=probe,
        snapshots_sha256=hash_value(snapshots),second_copy_location_policy='reserved on same volume; copying is a separate operation',
        caveat='Point-in-time allocation/fsync/readback; keep volume exclusive until launch; power-loss durability is not proved')
    write_json(receipt_path,receipt)
    return receipt


def check_storage_receipt(receipt,cfg,manifest,cache,destination,snapshots):
    require(receipt['schema']=='r2-full-raw-storage-v1' and receipt['destination']==str(Path(destination).resolve()), 'storage destination identity')
    require(receipt['manifest_sha256']==hash_value(manifest) and receipt['contract_sha256']==file_hash(ROOT/'config/checkpoint_r2/full_raw_v1.json'), 'storage plan identity')
    require(0 <= time.time()-receipt['verified_at_unix'] <= 86400, 'storage receipt older than 24 hours; repeat actual check')
    volume=Path(receipt['volume'])
    require(os.path.ismount(volume) and Path(cache).resolve().is_relative_to(volume), 'persistent mount/cache identity')
    mount=subprocess.check_output(['findmnt','-n','-o','TARGET,SOURCE,FSTYPE','--target',str(volume)],text=True).strip()
    require(mount==receipt['mount'], 'mount changed since real allocation test')
    require(receipt['snapshots_sha256']==hash_value(snapshots), 'verified snapshot plan changed')
    from src.checkpoint_r2_full_raw import valid_primary_bytes
    current_primary=valid_primary_bytes(destination,cfg,manifest)
    require(current_primary>=receipt['valid_primary_bytes'], 'valid primary storage regressed; repeat allocation check')
    live=volume_inventory(volume)
    allowed_growth=current_primary-receipt['valid_primary_bytes']+cfg['storage']['global_receipt_reserve_bytes']
    require(live['retained_unique_file_bytes']<=receipt['retained']['retained_unique_file_bytes']+allowed_growth, 'unplanned retained volume growth; repeat allocation check')
    require(receipt['additional_capacity_verified_bytes'] >= additional_capacity(receipt['plan'],cfg,receipt['valid_primary_bytes']), 'insufficient demonstrated additional capacity')
    require(receipt['probe']['actual_write_and_readback_bytes']==receipt['additional_capacity_verified_bytes'], 'capacity readback mismatch')


def verify_handoff(cfg, handoff, output):
    handoff=Path(handoff)
    require(file_hash(handoff/'handoff-checksums.json')==cfg['pilot_handoff_manifest_sha256'], 'completed handoff manifest')
    from src.checkpoint_r2_fresh_pilot import plan
    _,pilot=plan()
    checks={}
    for m,expected in cfg['small_shard_receipt_sha256'].items():
        shard=handoff/'pilot'/m/'transfer-ready'
        require(file_hash(shard/'receipt.json')==expected, 'independent shard receipt hash')
        values,rows,receipt=load_features(shard,expected_rows=pilot['stages']['calibration'][:8],expected_spec=cfg['models'][m])
        semantic_view(json.loads((shard/'tokens.json').read_text()),rows)
        require(receipt['provenance']['commit']==cfg['base_commit'], 'pilot producer identity')
        checks[m]=dict(receipt_sha256=expected,payload_sha256=receipt['payload_sha256'],shape=list(values.shape),passed=True)
    result=dict(schema='r2-full-raw-mac2-v1',hostname=socket.gethostname(),platform=platform.system(),
        contract_sha256=file_hash(ROOT/'config/checkpoint_r2/full_raw_v1.json'),checks=checks,
        independent_mac2=(platform.system()=='Darwin' and socket.gethostname()!=cfg['preparation_hostname']))
    write_json(output,result)
    return result


def check_review(cfg, receipt=None):
    if receipt is None:
        p=ROOT/'data/checkpoint_r2_full_raw_v1/review/checks.json'
        require(file_hash(p)==cfg['review_sha256'][str(p.relative_to(ROOT))], 'mac2 review receipt changed')
        receipt=json.loads(p.read_text())
    require(cfg['review_commit']=='1dbb5484d1055e0cf80993e666171cfa1743e2a2'
            and receipt['reviewed_code_commit']==cfg['base_commit'] and receipt['status']=='PASS_WITH_SCOPE_LIMITATIONS', 'review identity/verdict')
    require(receipt['scope']['blockers_to_completed_handoff']==[], 'completed handoff review blocker')
    for m,spec in cfg['models'].items():
        model=receipt['models'][m];shard=model['transferred_shard']
        require(model['model_spec']==spec and shard['receipt']['sha256']==cfg['small_shard_receipt_sha256'][m]
                and shard['shape']==[8,spec['layers'],spec['width']] and shard['source_rows_and_token_semantics_verified']
                and shard['all_values_finite'] and shard['dtype']=='float16', 'reviewed loader result')
    return receipt
