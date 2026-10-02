"""Complete raw metadata and tiny synthetic tensors only; no GPU/model access."""
import copy
import csv
import json
import os
import signal
import sys
from pathlib import Path
import numpy as np
import pytest
from src.checkpoint_r2_fresh_inputs import ROOT, canonical, hash_value, text_hash
from src.checkpoint_r2_fresh_store import pin, write_shard, load_features
from src.checkpoint_r2_full_inputs import reconstruct
from src.checkpoint_r2_full_support import (additional_capacity, check_review, filesystem_prerequisites, NON_SECRET_ARTIFACT_CONTRACT,
    single_packet, token_storage_plan, verify_snapshot, volume_inventory)
from src.checkpoint_r2_full_raw import (cap_reason, extract_model, pending_shards, plan,
    recover_budget, runtime_matches, supervise, verify_completed)


@pytest.fixture(scope='module')
def frozen():return plan()


def test_complete_inventory_and_ordered_bindings(frozen):
    cfg,manifest=frozen
    inventory=list(csv.DictReader((ROOT/'data/checkpoint_r2_v1/extraction_inventory.csv').open()))
    raw=[r for r in inventory if r['kind']=='raw']
    assert manifest['unique_texts']==len({r['text_id'] for r in raw})==35843
    assert manifest['logical_bindings']==len(raw)==41490
    assert [{k:b[k] for k in raw[0]} for b in manifest['bindings']]==raw
    assert sorted([b for r in manifest['rows'] for b in r['bindings']],key=lambda b:b['inventory_offset'])==manifest['bindings']
    assert list(dict.fromkeys(r['text_id'] for r in raw))==[r['id'] for r in manifest['rows']]
    assert manifest['excluded_workloads']=={'behavior':2000,'chat':11578}
    assert sum(manifest['fp16_payload_bytes'].values())==16589860864
    assert any(len(r['bindings'])>1 for r in manifest['rows'])
    assert {r['split'] for r in manifest['rows']}=={'train','validation'}
    assert not any('E'==b['group'] or b['kind']!='raw' for b in manifest['bindings'])
    assert cfg['execution_limits']=={'model_seconds':5400,'total_seconds':12600,'setup_phase_seconds':1800}


def synthetic_source(identity,text='one two'):
    return dict(source_row_id=identity,statement=text,statement_sha256=text_hash(text),split='train',topic='cities',label='1')


def inventory_binding(source,kind='raw',group='P15'):
    return dict(kind=kind,group=group,logical_id=source['source_row_id'],pair_id='',fact_a_id='',fact_b_id='',
        label='1',text_id='text_'+text_hash(source['statement']),statement=source['statement'])


def test_dedup_keeps_every_source_binding_and_excludes_workloads():
    sources=[synthetic_source('a'),synthetic_source('b'),synthetic_source('c','different')]
    inventory=[inventory_binding(s) for s in sources]
    inventory += [inventory_binding(sources[0],kind=k,group='E') for k in ('behavior','chat')]
    m=reconstruct(inventory,{'P15':sources,'atomic_D':[]},{},[],{'P15'})
    assert m['unique_texts']==2 and m['logical_bindings']==3
    assert [b['logical_id'] for b in m['rows'][0]['bindings']]==['a','b']
    assert [b['inventory_offset'] for b in m['bindings']]==[0,1,2]


@pytest.mark.parametrize('change', ['E_group','test_split','wrong_text','wrong_hash','duplicate_binding','unknown_source','extra_fact'])
def test_raw_E_and_binding_errors_rejected(change):
    s=synthetic_source('a');b=inventory_binding(s);rows=[b]
    if change=='E_group':b['group']='E'
    if change=='test_split':s['split']='test'
    if change=='wrong_text':b['statement']='changed'
    if change=='wrong_hash':b['text_id']='text_wrong'
    if change=='duplicate_binding':rows.append(b)
    if change=='unknown_source':b['logical_id']='unknown'
    if change=='extra_fact':b['fact_a_id']='extra'
    with pytest.raises(ValueError):reconstruct(rows,{'P15':[s],'atomic_D':[]},{},[],{'P15'})


def test_unknown_E_workload_is_not_admitted():
    s=synthetic_source('a')
    with pytest.raises(ValueError,match='E workload'):
        reconstruct([inventory_binding(s,kind='E')],{'P15':[s],'atomic_D':[]},{},[],{'P15'})


class Tokenizer:
    pad_token_id=0;eos_token_id=2
    def encode(self,text,add_special_tokens=True,truncation=False):
        return ([1] if add_special_tokens else [])+[ord(c)+3 for c in text]
    def __call__(self,texts,**kwargs):
        tokens=[self.encode(s) for s in texts]
        return dict(input_ids=tokens,special_tokens_mask=[[1]+[0]*(len(t)-1) for t in tokens])


def tiny_rows(n=5):
    return [dict(id='text_'+text_hash(str(i)),statement=str(i),statement_sha256=text_hash(str(i)),split='train',topic='cities',
        bindings=[dict(inventory_offset=i,kind='raw',group='P15',logical_id=str(i))]) for i in range(n)]


SPEC=dict(model_id='synthetic',revision='0'*40,layers=2,width=3)
PROV=dict(commit='test',batch_size=1,padding='none')


def token_plan(rows,size=2):
    return [dict(tokens_sha256=hash_value([single_packet(Tokenizer(),r,1024) for r in rows[i:i+size]]),reserved_bytes=100000)
            for i in range(0,len(rows),size)]


class Backend:
    def __init__(self,fail_after=None):self.calls=[];self.fail_after=fail_after
    def forward(self,rows,batch_size):
        assert batch_size==len(rows)==1
        if self.fail_after is not None and len(self.calls)==self.fail_after:raise RuntimeError('synthetic interruption')
        self.calls.append(rows[0]['id'])
        features=np.full((1,2,3),float(rows[0]['statement'])+.25,dtype=np.float32)
        return features,[single_packet(Tokenizer(),rows[0],1024)]


def test_interrupted_extraction_skips_hash_verified_completed_shards(tmp_path):
    rows=tiny_rows();dest=tmp_path/'shards';quarantine=tmp_path/'quarantine';tokens=token_plan(rows)
    with pytest.raises(RuntimeError):extract_model(dest,rows,SPEC,PROV,Backend(fail_after=2),2,tokens,quarantine)
    before={p.name:pin(p) for p in (dest/'00000').iterdir()}
    backend=Backend();result=extract_model(dest,rows,SPEC,PROV,backend,2,tokens,quarantine)
    assert backend.calls==[r['id'] for r in rows[2:]]
    assert before=={p.name:pin(p) for p in (dest/'00000').iterdir()}
    assert sum(r['rows'] for r in result)==5 and sum(r['bindings'] for r in result)==5
    for i,start in enumerate(range(0,5,2)):
        values,restored,_=load_features(dest/f'{i:05d}')
        assert restored==rows[start:start+2]
        assert np.array_equal(values[:,0,0],np.array([float(r['statement'])+.25 for r in restored],dtype=np.float16))


def test_receipt_published_before_marker_is_reverified_and_adopted(tmp_path):
    rows=tiny_rows(2);dest=tmp_path/'shards';dest.mkdir()
    packet=[single_packet(Tokenizer(),r,1024) for r in rows]
    write_shard(dest/'00000',rows,np.ones((2,2,3),np.float32),SPEC,PROV,packet)
    backend=Backend()
    result=extract_model(dest,rows,SPEC,PROV,backend,2,token_plan(rows),tmp_path/'quarantine')
    assert not backend.calls and result[0]['rows']==2
    assert json.loads((dest/'00000/completed.json').read_text())['adopted_after_interruption']


def test_incomplete_outputs_are_preserved_in_quarantine(tmp_path):
    rows=tiny_rows(2);dest=tmp_path/'shards';partial=dest/'00000';partial.mkdir(parents=True)
    (partial/'activations.npy.unpublished.partial').write_bytes(b'preserve incomplete bytes')
    backend=Backend();extract_model(dest,rows,SPEC,PROV,backend,2,token_plan(rows),tmp_path/'quarantine')
    assert list((tmp_path/'quarantine').glob('*/activations.npy.unpublished.partial'))[0].read_bytes()==b'preserve incomplete bytes'
    assert len(backend.calls)==2


@pytest.mark.parametrize('name',['activations.npy','tokens.json','rows.json','receipt.json','completed.json'])
def test_published_corruption_stops_without_overwrite(tmp_path,name):
    rows=tiny_rows(2);dest=tmp_path/'shards';q=tmp_path/'quarantine'
    extract_model(dest,rows,SPEC,PROV,Backend(),2,token_plan(rows),q)
    p=dest/'00000'/name;p.write_bytes(p.read_bytes()+b'corruption');before=p.read_bytes()
    backend=Backend()
    with pytest.raises((ValueError,json.JSONDecodeError)):extract_model(dest,rows,SPEC,PROV,backend,2,token_plan(rows),q)
    assert p.read_bytes()==before and not backend.calls and not q.exists()


def test_reordered_bindings_and_extra_shards_rejected(tmp_path):
    rows=tiny_rows(2);dest=tmp_path/'shards';q=tmp_path/'quarantine'
    extract_model(dest,rows,SPEC,PROV,Backend(),2,token_plan(rows),q)
    altered=copy.deepcopy(rows);altered[0]['bindings'][0]['logical_id']='wrong'
    with pytest.raises(ValueError):pending_shards(dest,altered,SPEC,PROV,2,q)
    (dest/'99999').mkdir()
    with pytest.raises(ValueError,match='extra'):extract_model(dest,rows,SPEC,PROV,Backend(),2,token_plan(rows),q)


def test_predeclared_token_sequence_and_storage_bound_enforced(tmp_path):
    rows=tiny_rows(2);bad=token_plan(rows);bad[0]['tokens_sha256']='wrong'
    with pytest.raises(ValueError,match='tokenization differs'):extract_model(tmp_path/'a',rows,SPEC,PROV,Backend(),2,bad,tmp_path/'q')
    bad=token_plan(rows);bad[0]['reserved_bytes']=1
    with pytest.raises(ValueError,match='storage reservation'):extract_model(tmp_path/'b',rows,SPEC,PROV,Backend(),2,bad,tmp_path/'q')


def snapshot_fixture(tmp_path):
    spec=dict(SPEC,model_id='test/model',files_sha256={})
    cache=tmp_path/'cache';snap=cache/'models--test--model'/'snapshots'/spec['revision'];snap.mkdir(parents=True)
    blobs=cache/'blobs';blobs.mkdir();weight=blobs/'weight';weight.write_bytes(b'synthetic weights')
    for name in ('model-1.safetensors','model-2.safetensors'):(snap/name).symlink_to(weight)
    (snap/'config.json').write_text('{}')
    (snap/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'a':'model-1.safetensors','b':'model-2.safetensors'})))
    spec['files_sha256']={'config.json':pin(snap/'config.json')['sha256']}
    receipt=dict(model_id=spec['model_id'],revision=spec['revision'],tokenizer_revision=spec['revision'],files={p.name:pin(p) for p in snap.iterdir()})
    return cache,snap,spec,receipt,weight


def test_offline_snapshot_reuse_hashes_all_files_and_counts_aliases_once(tmp_path):
    cache,snap,spec,receipt,weight=snapshot_fixture(tmp_path)
    result=verify_snapshot(receipt,spec,cache)
    assert result['new_download_bytes']==0
    assert result['downloaded_bytes']==weight.stat().st_size+(snap/'config.json').stat().st_size+(snap/'model.safetensors.index.json').stat().st_size
    before=weight.read_bytes();weight.write_bytes(before+b'bad')
    with pytest.raises(ValueError,match='corrupt'):verify_snapshot(receipt,spec,cache)


def test_missing_snapshot_and_unrecorded_files_abort(tmp_path):
    cache,snap,spec,receipt,weight=snapshot_fixture(tmp_path)
    (snap/'extra.json').write_text('{}')
    with pytest.raises(ValueError,match='file set'):verify_snapshot(receipt,spec,cache)
    with pytest.raises(ValueError,match='missing'):verify_snapshot(receipt,spec,tmp_path/'missing')


def test_storage_counting_hardlinks_and_symlinks_once(tmp_path):
    p=tmp_path/'bytes';p.write_bytes(b'12345');os.link(p,tmp_path/'link');(tmp_path/'alias').symlink_to(p)
    assert volume_inventory(tmp_path)['retained_unique_file_bytes']==5
    cfg=dict(storage=dict(safety_margin_bytes=10))
    storage=dict(complete_copy_reserved_bytes=100,second_copy_reserved_bytes=100,temporary_shard_reserved_bytes=5)
    assert additional_capacity(storage,cfg)==215
    assert additional_capacity(storage,cfg,valid_primary_bytes=60)==155


def test_filesystem_lock_git_chmod_prerequisites(tmp_path):
    receipt=filesystem_prerequisites(tmp_path,artifact_contract=NON_SECRET_ARTIFACT_CONTRACT)
    assert receipt['permissions']['exact_mode_reporting']['status']=='passed'
    assert receipt['cross_process_flock_exclusion']['status']=='passed'
    assert receipt['git_initialization']['status']=='passed'
    assert not list(tmp_path.iterdir())


def test_full_accounting_preserves_spend_across_unclean_interruption():
    before=dict(total_seconds=100,model_seconds={'qwen':60,'llama':0},running=True,model='qwen',as_of_unix=10)
    after=recover_budget(before,30)
    assert after['total_seconds']==120 and after['model_seconds']['qwen']==80
    assert before['total_seconds']==100
    limits=dict(model_seconds=90,total_seconds=200,setup_phase_seconds=10)
    assert cap_reason(after,limits) is None
    assert cap_reason(after,limits,11)=='full_setup_phase_cap'
    after['model_seconds']['qwen']=90;assert cap_reason(after,limits)=='full_model_cap_qwen'
    with pytest.raises(ValueError):recover_budget(before,5)
    before['total_seconds']=-1
    with pytest.raises(ValueError,match='corrupt'):recover_budget(before,30)


def test_separate_watchdog_stops_and_retains_full_budget(tmp_path):
    attempt=tmp_path/'attempt';attempt.mkdir()
    result=supervise([sys.executable,'-c','import time; time.sleep(20)'],tmp_path,attempt,
                    dict(model_seconds=10,total_seconds=.3,setup_phase_seconds=10))
    assert result==124
    receipt=json.loads((attempt/'supervisor-exit.json').read_text())
    assert receipt['cap_reason']=='full_total_cap' and not receipt['budget_reset_allowed']
    assert json.loads((tmp_path/'full-budget.json').read_text())['total_seconds']>=.3
    second=tmp_path/'second';second.mkdir()
    with pytest.raises(ValueError,match='exhausted'):
        supervise([sys.executable,'-c','pass'],tmp_path,second,dict(model_seconds=10,total_seconds=.3,setup_phase_seconds=10))


def test_exact_mac2_review_is_pinned_and_corrupt_verdict_rejected(frozen):
    cfg,_=frozen
    receipt=check_review(cfg)
    assert cfg['review_commit']=='1dbb5484d1055e0cf80993e666171cfa1743e2a2'
    assert receipt['status']=='PASS_WITH_SCOPE_LIMITATIONS'
    receipt=copy.deepcopy(receipt);receipt['scope']['blockers_to_completed_handoff']=['corrupt']
    with pytest.raises(ValueError,match='review blocker'):check_review(cfg,receipt)


def test_runtime_change_requires_review():
    expected=dict(python='3.12.3',packages={},cuda='12.8',cudnn=1,gpu='A40',gpu_total_bytes=10,
        compute_capability=[8,6],pip_freeze=['x==1'],driver='A40, uuid, 595, 10')
    runtime_matches(expected,expected)
    changed=dict(expected,python='3.12.4')
    with pytest.raises(ValueError,match='runtime changed'):runtime_matches(changed,expected)


def test_full_token_plan_counts_actual_metadata_without_gpu(frozen):
    cfg,_=frozen;cfg=copy.deepcopy(cfg);cfg['shard_rows']=2
    rows=tiny_rows(3);manifest=dict(rows=rows)
    snapshots={m:dict(snapshot='synthetic') for m in cfg['models']}
    planned=token_storage_plan(cfg,manifest,snapshots,tokenizer_factory=lambda _:Tokenizer())
    assert planned['full_tokens_preflight']
    assert all([s['rows'] for s in planned['models'][m]['shards']]==[2,1] for m in cfg['models'])
    payload=sum(3*s['layers']*s['width']*2 for s in cfg['models'].values())
    assert planned['complete_copy_reserved_bytes']>payload
    assert planned['second_copy_reserved_bytes']==planned['complete_copy_reserved_bytes']


def test_worker_sequential_completion_and_resume_after_model_failure(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import src.checkpoint_r2_full_raw as full
    import src.checkpoint_r2_fresh_backend as fresh
    rows=tiny_rows(3);tokens=token_plan(rows)
    cfg=dict(models={'qwen':SPEC,'llama':SPEC},runtime_lock={},execution={},shard_rows=2)
    manifest=dict(rows=rows);campaign=dict(commit='test',contract_sha256='test',manifest_sha256=hash_value(manifest))
    output=tmp_path/'output';output.mkdir();attempt=tmp_path/'attempt';attempt.mkdir()
    storage=tmp_path/'storage.json';storage.write_text(json.dumps(dict(plan=dict(models={m:dict(shards=tokens) for m in cfg['models']}))))
    runtime=dict(python='3.12.3',packages={},cuda='12.8',cudnn=1,gpu='A40',gpu_total_bytes=10,compute_capability=[8,6],pip_freeze=[],driver='A40, uuid, 595, 10')
    monkeypatch.setattr(full.ctypes,'CDLL',lambda _:SimpleNamespace(prctl=lambda *args:0))
    monkeypatch.setattr(fresh,'runtime_receipt',lambda _:runtime)
    monkeypatch.setattr(full,'evidence',lambda cfg,name:runtime if name=='host-runtime.json' else dict(backend={'test':'identity'}))
    monkeypatch.setattr(full,'verify_cached_models',lambda *args:{m:dict(snapshot='synthetic') for m in cfg['models']})
    monkeypatch.setattr(full,'check_storage_receipt',lambda *args:None)
    load_calls=[];failed=[False];backends=[]
    class Fake(Backend):
        details={'test':'identity'}
        def __init__(self,*args):
            super().__init__();load_calls.append(len(load_calls));backends.append(self)
        def forward(self,rows,batch):
            if len(load_calls)==2 and not failed[0]:failed[0]=True;raise RuntimeError('interrupt Llama')
            return super().forward(rows,batch)
        def close(self):pass
        def peak(self):return dict(allocated=1,reserved=1)
    monkeypatch.setattr(fresh,'FreshBackend',Fake)
    args=SimpleNamespace(output=output,pilot_cache=tmp_path/'cache',storage_receipt=storage,parent_pid=os.getppid())
    with pytest.raises(RuntimeError):full.worker(args,cfg,manifest,campaign,attempt)
    assert (output/'qwen/index.json').exists() and not (output/'llama/index.json').exists()
    before=pin(output/'qwen/index.json');second=tmp_path/'second';second.mkdir()
    full.worker(args,cfg,manifest,campaign,second)
    assert pin(output/'qwen/index.json')==before and len(load_calls)==3
    assert (output/'completion.json').exists()
    for m in cfg['models']:
        index=json.loads((output/m/'index.json').read_text())
        assert index['unique_texts']==index['logical_bindings']==3
        assert [s['path'] for s in index['shards']]==['00000','00001']


def test_local_full_cli_cannot_launch_without_reviewed_remote_execution(tmp_path):
    import subprocess
    result=subprocess.run([sys.executable,'-B','-m','src.checkpoint_r2_full_raw','run','--output',str(tmp_path/'never-created')],cwd=ROOT,capture_output=True,text=True)
    assert result.returncode!=0 and 'later explicit review/authorization' in result.stderr
    assert not (tmp_path/'never-created').exists()


def test_resume_storage_only_credits_verified_primary_files(tmp_path):
    from src.checkpoint_r2_full_raw import valid_primary_bytes,shard_provenance,CONFIG_SHA256
    from src.checkpoint_r2_fresh_store import write_json
    rows=tiny_rows(2);manifest=dict(rows=rows);cfg=dict(models={'qwen':SPEC},shard_rows=2)
    campaign=dict(commit='test',contract_sha256=CONFIG_SHA256,manifest_sha256=hash_value(manifest))
    write_json(tmp_path/'campaign.json',campaign);write_json(tmp_path/'input-manifest.json',manifest)
    (tmp_path/'qwen').mkdir();write_json(tmp_path/'qwen/runtime-origin.json',{'synthetic':'origin'})
    prov=shard_provenance(campaign,'qwen',pin(tmp_path/'qwen/runtime-origin.json')['sha256'])
    extract_model(tmp_path/'qwen/shards',rows,SPEC,prov,Backend(),2,token_plan(rows),tmp_path/'quarantine')
    expected=(tmp_path/'input-manifest.json').stat().st_size+sum(p.stat().st_size for p in (tmp_path/'qwen/shards/00000').iterdir())
    (tmp_path/'quarantine').mkdir();(tmp_path/'quarantine/partial').write_bytes(b'not completed')
    assert valid_primary_bytes(tmp_path,cfg,manifest)==expected
    p=tmp_path/'input-manifest.json';p.write_bytes(p.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='manifest corrupt'):valid_primary_bytes(tmp_path,cfg,manifest)


@pytest.mark.parametrize('change',['none','extra_shard','manifest','completion','index_identity'])
def test_portable_final_audit_rejects_manifest_scope_and_extra_shards(tmp_path,change):
    from src.checkpoint_r2_full_raw import verify_output,shard_provenance,CONFIG_SHA256
    from src.checkpoint_r2_fresh_store import write_json
    rows=tiny_rows(3);manifest=dict(rows=rows,unique_texts=3,logical_bindings=3)
    cfg=dict(models={'qwen':SPEC},shard_rows=2)
    campaign=dict(schema='r2-full-raw-campaign-v1',commit='test',contract_sha256=CONFIG_SHA256,manifest_sha256=hash_value(manifest))
    write_json(tmp_path/'campaign.json',campaign);write_json(tmp_path/'input-manifest.json',manifest)
    (tmp_path/'qwen').mkdir();write_json(tmp_path/'qwen/runtime-origin.json',dict(synthetic='origin'))
    prov=shard_provenance(campaign,'qwen',pin(tmp_path/'qwen/runtime-origin.json')['sha256'])
    shards=extract_model(tmp_path/'qwen/shards',rows,SPEC,prov,Backend(),2,token_plan(rows),tmp_path/'quarantine')
    index=dict(model='wrong' if change=='index_identity' else 'qwen',unique_texts=3,logical_bindings=3,
        manifest_sha256=hash_value(manifest),shards=shards)
    write_json(tmp_path/'qwen/index.json',index)
    write_json(tmp_path/'completion.json',dict(schema='r2-full-raw-completion-v1',manifest_sha256=hash_value(manifest),
        indices={'qwen':pin(tmp_path/'qwen/index.json')},raw_only=True,scientific_evaluation=change=='completion'))
    if change=='extra_shard':(tmp_path/'qwen/shards/99999').mkdir()
    if change=='manifest':(tmp_path/'input-manifest.json').write_bytes(b'{}')
    if change=='none':assert verify_output(tmp_path,cfg,manifest)['status']=='full_output_hashes_and_coverage_verified'
    else:
        with pytest.raises(ValueError):verify_output(tmp_path,cfg,manifest)
