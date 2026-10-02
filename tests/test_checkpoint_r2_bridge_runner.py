import copy
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import SimpleNamespace
import numpy as np
import pytest
from src import checkpoint_r2_bridge_runner as r


@pytest.fixture
def plan():return r.Plan()


def test_actual_frozen_plan(plan):
    assert len(plan.identities['calibration'])==10
    assert len(plan.identities['independent'])==180
    assert len(plan.history['bindings'])==380
    assert plan.config['historical_evidence']['reuse_permitted'] is False
    assert r.digest(r.ROOT/'src/checkpoint_r2_bridge.py')=='b84b14c6b4ee9ed8dcff596429cd47538151c60c7c5e68b6736d8a056c1efa76'


@pytest.mark.parametrize('field',['extra','reordered','changed_text','E','ambiguous','wrong_stage'])
def test_reject_identity_binding_changes(plan,field):
    rows=copy.deepcopy(plan.identities['calibration']);b=copy.deepcopy(plan.bindings('qwen','calibration'))
    if field=='extra':b.append(b[0])
    if field=='reordered':b[0],b[1]=b[1],b[0]
    if field=='changed_text':rows[0]['statement']+=' changed'
    if field=='E':rows[0]['split']=b[0]['split']='test'
    if field=='ambiguous':b[1]['metadata_path']=b[0]['metadata_path'];b[1]['historical_index']=b[0]['historical_index']
    if field=='wrong_stage':b[0]['stage']='independent'
    with pytest.raises(ValueError):r.Plan.validate_ordered_bindings(rows,b,'qwen','calibration')


def test_frozen_hash_failure(tmp_path,plan):
    for relative in [r.CONFIG,*plan.config['frozen_sha256']]:
        p=tmp_path/relative;p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(r.ROOT/relative,p)
    (tmp_path/r.BRIDGE).write_text('{}')
    with pytest.raises(ValueError,match='frozen input'):r.Plan(tmp_path)


def synthetic_packet(rows,layers=28,width=3584,padding=None):
    lengths=[2 if row['id'].endswith('0') else 4 for row in rows];size=max(lengths);physical=[]
    for length in lengths:
        ids=list(range(1,length+1));count=size-length
        mask=[0]*count+[1]*length if padding=='left' else [1]*length+[0]*count
        tokens=[99]*count+ids if padding=='left' else ids+[99]*count
        positions=[sum(mask[:i+1])-1 if mask[i] else 0 for i in range(size)]
        readout=max(i for i,v in enumerate(mask) if v)
        physical.append(dict(token_ids=tokens,attention_mask=mask,position_ids=positions,readout_index=readout,semantic_readout_position=length-1,pad_token_id=99))
    values=np.stack([np.full((layers,width),len(row['statement']),dtype=np.float32) for row in rows])
    return dict(ids=[x['id'] for x in rows],statement_sha256=[r.text_hash(x['statement']) for x in rows],compute=values,physical=physical,execution={'synthetic':True,'padding':padding})


@pytest.fixture
def rows():return [dict(id='train:0',split='train',statement='Short.'),dict(id='train:1',split='train',statement='Longer input.')]


@pytest.mark.parametrize('padding',['left','right'])
def test_padding_semantic_and_readout_positions(rows,padding):
    packet=synthetic_packet(rows,padding=padding);logical=r.validate_packet(rows,packet,28,3584)
    assert logical['tokens']==[[1,2],[1,2,3,4]]
    assert logical['positions']==[[0,1],[0,1,2,3]]
    assert packet['physical'][0]['readout_index']==(3 if padding=='left' else 1)


@pytest.mark.parametrize('field',['layers','order','text','mask','position','readout','padding_token','nonfinite'])
def test_reject_backend_misbindings(rows,field):
    packet=synthetic_packet(rows,padding='right')
    if field=='layers':packet['compute']=packet['compute'][:,:27]
    if field=='order':packet['ids'].reverse()
    if field=='text':packet['statement_sha256'][0]='changed'
    if field=='mask':packet['physical'][0]['attention_mask']=[1,0,1,0]
    if field=='position':packet['physical'][0]['position_ids'][0]=1
    if field=='readout':packet['physical'][0]['readout_index']=3
    if field=='padding_token':packet['physical'][0]['token_ids'][3]=55
    if field=='nonfinite':packet['compute'][0,0,0]=np.inf
    with pytest.raises(ValueError):r.validate_packet(rows,packet,28,3584)


def test_actual_backend_readout_on_small_tensors():
    import torch
    from src.checkpoint_r2_bridge_backend import read_hidden_at_last
    mask=np.array([[0,1,1],[1,1,0]])
    embedding=torch.full((2,3,2),-99,dtype=torch.bfloat16)
    a=torch.arange(12,dtype=torch.bfloat16).reshape(2,3,2)
    b=a+20
    result,indices=read_hidden_at_last([embedding,a,b],mask,2,2)
    assert indices.tolist()==[2,1]
    np.testing.assert_array_equal(result,np.array([[[4,5],[24,25]],[[8,9],[28,29]]],np.float32))
    with pytest.raises(ValueError):read_hidden_at_last([embedding,a],mask,2,2)
    with pytest.raises(ValueError):read_hidden_at_last([embedding,a.float(),b],mask,2,2)


def test_undefined_relative_error_is_valid_JSON():
    value={'delta':float('inf'),'nested':[float('nan'),1.]}
    payload=r.strict_json(value);assert b'Infinity' not in payload and b'NaN' not in payload
    parsed=json.loads(payload);assert parsed['diagnostics']['delta'] is None and len(parsed['undefined_nonfinite_paths'])==2


def test_atomic_exclusive_receipts(tmp_path):
    path=tmp_path/'receipt.json';r.write_new(path,{'complete':True})
    before=path.read_bytes()
    with pytest.raises(FileExistsError):r.write_new(path,{'changed':True})
    assert path.read_bytes()==before and not list(tmp_path.glob('*.partial'))


def test_budget_setup_separate_and_both_caps():
    clock=[0.];b=r.Budget(lambda:clock[0]);clock[0]=100;b.transition('active');b.begin_gpu('qwen');clock[0]=101;b.end_gpu()
    assert b.snapshot()['setup_seconds']==100 and b.snapshot()['gpu_seconds']['qwen']==1
    b.gpu['qwen']=3600
    with pytest.raises(TimeoutError,match='GPU'):b.check()
    b.gpu['qwen']=0;clock[0]=7300
    with pytest.raises(TimeoutError,match='active'):b.check()


def test_watchdog_hard_kills_expired_child(tmp_path):
    b=r.Budget();b.transition('active');b.totals['active']=7201;r.Store(tmp_path).watch(b)
    code=r.supervise([sys.executable,'-c','import time; time.sleep(20)'],tmp_path)
    assert code==124 and json.loads((tmp_path/'watchdog_termination.json').read_text())['reason']=='two_hour_active_cap'


def test_frozen_head_identity_preprocessing_and_hash(tmp_path):
    path=tmp_path/'head.npz'
    np.savez(path,coef=np.array([[1.,2.]]),intercept=np.array([3.]),classes=np.array([0,1]),layer=np.array(1),C=np.array(1.),n_iter=np.array([2]),max_iter=np.array(10))
    spec={'probe':{'path':'head.npz','sha256':r.digest(path)},'fixed_layer':1,'fixed_C':1,'width':2,'preprocessing':{'operation':'identity'}}
    head=r.FrozenHead.load(tmp_path,spec)
    np.testing.assert_array_equal(head.score(np.array([[[99.,99.],[4.,5.]]],np.float16)),[[17.]])
    spec['probe']['sha256']='bad'
    with pytest.raises(ValueError,match='head hash'):r.FrozenHead.load(tmp_path,spec)


@pytest.fixture
def tiny_engine(tmp_path,rows):
    plan=SimpleNamespace(identities={'calibration':rows,'independent':[dict(x,id=x['id'].replace('train','dev'),split='validation') for x in rows]},history={'contracts':{m:{'recorded_manifests':{'missing_token_records':True}} for m in ('qwen','llama')}},config={'models':{}})
    for m,l,w in [('qwen',28,3584),('llama',32,4096)]:
        p=tmp_path/(m+'.npz');np.savez(p,coef=np.ones((1,w))/w,intercept=np.array([0.]),classes=np.array([0,1]),layer=np.array(0),C=np.array(1.),n_iter=np.array([1]),max_iter=np.array(10))
        plan.config['models'][m]={'layers':l,'width':w,'fixed_layer':0,'fixed_C':1,'probe':{'path':p.name,'sha256':r.digest(p)},'preprocessing':{'operation':'identity'}}
    plan.historical_metadata=lambda root:None
    class Backend:
        def __init__(self,m,c):self.spec=c['models'][m];self.runtime={'synthetic':True};self.calls=0
        def forward(self,rs,padding):self.calls+=1;return synthetic_packet(rs,self.spec['layers'],self.spec['width'],padding)
        def close(self):pass
    class History:
        def __init__(self,plan,root,model,budget,store):self.spec=plan.config['models'][model]
        def stage(self,stage):return synthetic_packet(plan.identities[stage],self.spec['layers'],self.spec['width'])['compute'].astype(np.float16)
    return plan,Backend,History


def test_complete_synthetic_run_missing_history_never_earns_reuse(tmp_path,tiny_engine):
    plan,backend,history=tiny_engine;out=tmp_path/'out';out.mkdir()
    r.execute_models(plan,tmp_path,r.Store(out),r.Budget(),backend,history)
    result=json.loads((out/'completion.json').read_text())
    assert result['status']=='completed_diagnostics_unverifiable_history' and result['verified_reuse'] is False
    for m in ('qwen','llama'):
        assert result['results'][m]['verdict']=='unverifiable'
        assert result['results'][m]['strict_comparator']['checks']['hidden_exact']
        assert not result['results'][m]['strict_comparator']['checks']['tokens']
    events=[json.loads(p.read_text()) for p in (out/'events').glob('*.json')]
    assert sum(e['kind']=='calibration_check' for e in events)==6
    assert all(e['all_exact_checks_passed'] for e in events if e['kind']=='calibration_check')
    for p in out.rglob('*.json'):json.loads(p.read_text(),parse_constant=lambda x:pytest.fail(x))


def test_failed_padding_stops_before_independent_with_partial_receipt(tmp_path,tiny_engine):
    plan,Backend,history=tiny_engine;out=tmp_path/'failure';out.mkdir()
    class Bad(Backend):
        def forward(self,rs,padding):
            p=super().forward(rs,padding)
            if padding:p['compute'][0,0,0]+=1
            return p
    with pytest.raises(ValueError,match='calibration mismatch'):r.execute_models(plan,tmp_path,r.Store(out),r.Budget(),Bad,history)
    failure=json.loads((out/'failure.json').read_text());assert not failure['verified_reuse']
    assert not list(out.glob('*/independent/*')) and not (out/'completion.json').exists()


def test_historical_small_tensor_fixture_hash_shape_and_order(tmp_path):
    a=np.arange(4*2*3,dtype=np.float16).reshape(4,2,3);p=tmp_path/'acts.npy';np.save(p,a)
    plan=SimpleNamespace(config={'tensors':{'acts.npy':{'model':'qwen','sha256':r.digest(p),'shape':[4,2,3]}}},bindings=lambda model,stage:[{'remote_tensor':'/workspace/truth-probing-artifacts/acts.npy','historical_index':i} for i in (3,0)])
    h=r.Historical(plan,tmp_path,'qwen',r.Budget(),r.Store(tmp_path/'receipts'))
    np.testing.assert_array_equal(h.stage('independent'),a[[3,0]])
    plan.config['tensors']['acts.npy']['shape']=[4,1,3]
    with pytest.raises(ValueError,match='shape'):r.Historical(plan,tmp_path,'qwen',r.Budget(),r.Store(tmp_path/'receipts2'))


def test_dry_mode_never_imports_backend_or_opens_arrays(plan,monkeypatch):
    monkeypatch.setattr(plan,'historical_metadata',lambda root:{'tensor_values_read':False})
    monkeypatch.setattr(np,'load',lambda *a,**k:pytest.fail('dry run opened array'))
    receipt=plan.dry_run('/unused');assert not receipt['real_inference_executed'] and receipt['counts']=={'calibration':10,'independent':180}


def test_actual_comparator_nonfinite_relative_diagnostic_serializes(plan,rows):
    head=r.FrozenHead(np.zeros((1,3584)),np.zeros(1),0,{'frozen':'synthetic'})
    packet=synthetic_packet(rows);packet['compute'][:]=65504.;packet['logical']=r.validate_packet(rows,packet,28,3584)
    fresh=r.packet_record(rows,[packet],head,'qwen')
    result=r.compare_historical(plan,'qwen','calibration',rows,fresh,np.zeros_like(fresh['hidden']),head)
    assert result['verdict'].startswith('failed_numerical')
    value=json.loads(r.strict_json(result),parse_constant=lambda x:pytest.fail(x))
    assert value['undefined_nonfinite_paths'] and value['diagnostics']['verified_reuse'] is False


def test_missing_artifact_generates_setup_failure_receipt(tmp_path,tiny_engine):
    plan,backend,history=tiny_engine;out=tmp_path/'missing';out.mkdir()
    (tmp_path/'qwen.npz').unlink()
    with pytest.raises(FileNotFoundError):r.execute_models(plan,tmp_path,r.Store(out),r.Budget(),backend,history)
    receipt=json.loads((out/'failure.json').read_text());assert receipt['budget']['gpu_seconds']=={'qwen':0,'llama':0}


def test_supervisor_setup_timeout_predicate():
    b=r.Budget();s=b.snapshot();assert r.limits_exceeded(s,s['as_of_monotonic']+1800)=='setup_timeout_separate_from_active_cap'
    assert r.limits_exceeded(s,s['as_of_monotonic']+1799) is None


def test_cli_metadata_dry_run_accepts_symlinked_temp_parent(tmp_path,monkeypatch):
    target=tmp_path/'target';target.mkdir();link=tmp_path/'link';link.symlink_to(target,target_is_directory=True)
    plan=SimpleNamespace(dry_run=lambda root:{'mode':'metadata_only','real_inference_executed':False})
    monkeypatch.setattr(r,'Plan',lambda:plan)
    monkeypatch.setattr(sys,'argv',['runner','--mode','dry-run','--historical-root',str(tmp_path/'historical'),'--output',str(link/'receipt.json')])
    r.main();assert json.loads((target/'receipt.json').read_text())['real_inference_executed'] is False


def test_cli_real_mode_requires_acknowledgment_before_execution(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'Plan',lambda:SimpleNamespace())
    monkeypatch.setattr(sys,'argv',['runner','--mode','run','--historical-root',str(tmp_path/'history'),'--output',str(tmp_path/'run')])
    with pytest.raises(ValueError,match='explicit later authorization'):r.main()
    assert not (tmp_path/'run').exists()
