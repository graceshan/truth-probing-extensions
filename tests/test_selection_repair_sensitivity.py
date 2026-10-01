"""Synthetic export/eligibility checks; no real tensors or live SSH."""
import copy
import io
import json
from pathlib import Path

import numpy as np
import pytest
from src import selection_repair_sensitivity_inputs as s
from src import selection_repair_cache_export_remote as r


def test_exact_fact_evidence_and_restrictions_control_eligibility():
    statement = 'The city of A is in B.'
    fact = dict(fact_key='key', fact_id=s.stable_id('fact',['cities','e',statement,True]),
                topic='cities',entity_id='e',statement=statement,truth='True',split='validation')
    source = dict(source_row_id='cities:2',topic='cities',entity_id='e',statement=statement,
                  split='validation',form='affirmative',label='1',person_key='person',successor_status='admitted')
    assert s.fact_eligibility([fact],[source],[])[0]['eligible']
    duplicate = dict(source,source_row_id='cities:7')
    receipt = s.fact_eligibility([fact],[source,duplicate],[])[0]
    assert json.loads(receipt['source_row_ids']) == ['cities:2','cities:7']
    source['successor_status']='quarantined'
    assert not s.fact_eligibility([fact],[source],[])[0]['eligible']
    false = dict(fact,truth='False',fact_id='negative')
    registry = dict(fact_ref='negative',split='validation',origin='locked_registry',topic='cities',entity_id='e',
                    statement=statement,label='0_candidate',eligible='True',support='historical_external_negative',
                    person_key='person',evidence_ref='reviewed')
    assert s.fact_eligibility([false],[],[registry])[0]['eligible']
    assert not s.fact_eligibility([false],[source],[registry])[0]['eligible']
    registry['statement']='different exact text'
    assert not s.fact_eligibility([false],[],[registry])[0]['eligible']


def test_pair_requires_every_constituent_and_complete_variant_group():
    rows=[]; mapping=[]
    for op in ['AND','OR']:
        for a in ['True','False']:
            for b in ['True','False']:
                for order in ['AB','BA']:
                    eid=str(len(rows));rows.append(dict(example_id=eid,pair_id='p',topic='cities',entity_a_id='a',entity_b_id='b',
                         operator=op,canonical_truth_a=a,canonical_truth_b=b,ordering=order))
                    mapping.append(dict(base_example_id=eid,fact_a_key='a'+a,fact_b_key='b'+b))
    facts=[dict(fact_key=entity+truth,eligible=True) for entity in ['a','b'] for truth in ['True','False']]
    assert s.pair_eligibility(rows,facts,mapping)[0]['retained_rows']==16
    facts[0]['eligible']=False
    assert s.pair_eligibility(rows,facts,mapping)[0]['retained_rows']==0
    with pytest.raises(ValueError,match='incomplete'):
        s.pair_eligibility(rows[:-1],facts,mapping[:-1])


def remote_fixture(tmp_path,monkeypatch):
    monkeypatch.setattr(r,'ROOT',tmp_path)
    monkeypatch.setattr(r,'DIRECTORIES',{'qwen_train':'q/train'})
    directory=tmp_path/'q/train';directory.mkdir(parents=True)
    values=np.arange(3*28*4,dtype=np.float16).reshape(3,28,4)
    np.save(directory/'activations.npy',values)
    files=[directory/'metadata.csv',directory.parent/'extraction_manifest.json',directory.parent/'completion.json']
    for file in files:file.write_text('synthetic metadata')
    request=dict(path=str(directory/'activations.npy'),**s.identity((directory/'activations.npy').read_bytes()),
                 shape=[3,28,4],layer=17,row_indices=[0,2],row_identity_sha256='synthetic',
                 companions={str(file):s.identity(file.read_bytes()) for file in files})
    return request,values


def test_export_exact_original_rows_and_zero_based_layer(tmp_path,monkeypatch):
    request,values=remote_fixture(tmp_path,monkeypatch)
    data,receipt=r.verify_and_export('qwen_train',request)
    np.testing.assert_array_equal(np.load(io.BytesIO(data),allow_pickle=False),values[[0,2],17,:])
    assert receipt['original_row_indices']==[0,2] and receipt['saved_layer_index']==17
    assert receipt['source_stable_through_export'] and receipt['export']['sha256']==s.sha(data)
    assert receipt['source']['header']['shape']==[3,28,4]


@pytest.mark.parametrize('field,value', [('layer',18),('row_indices',[2,0]),('row_indices',[0,0]),('row_indices',[3]),
                                        ('sha256','0'*64),('shape',[3,27,4])])
def test_export_rejects_identity_layer_row_failures(tmp_path,monkeypatch,field,value):
    request,_=remote_fixture(tmp_path,monkeypatch);request[field]=value
    with pytest.raises(ValueError):r.verify_and_export('qwen_train',request)


def test_export_rejects_test_path_and_changed_companion(tmp_path,monkeypatch):
    request,_=remote_fixture(tmp_path,monkeypatch)
    bad=copy.deepcopy(request);bad['path']=bad['path'].replace('/train/','/test/')
    with pytest.raises(ValueError,match='tensor path'):r.verify_and_export('qwen_train',bad)
    Path(next(iter(request['companions']))).write_text('changed')
    with pytest.raises(ValueError,match='companion identity'):r.verify_and_export('qwen_train',request)


def test_source_change_after_hash_rejected(tmp_path,monkeypatch):
    request,_=remote_fixture(tmp_path,monkeypatch)
    measured,token=r.measure(request['path'],tensor=True)
    with Path(request['path']).open('ab') as stream:stream.write(b'x')
    with pytest.raises(ValueError,match='changed after verification'):r.export_layer(request,measured,token)


def test_npy_order_dtype_and_length_are_enforced():
    for array in [np.zeros((2,3),dtype=np.float32),np.asfortranarray(np.zeros((2,3),dtype=np.float16))]:
        stream=io.BytesIO();np.save(stream,array);stream.seek(0)
        with pytest.raises(ValueError,match='C-order'):r.header(stream)


def test_canonical_raw_objective_gradient_and_no_standardization():
    from src.selection_repair_canonical_sensitivity import gradient_diagnostics, fit_reference
    from sklearn.linear_model import LogisticRegression
    from src.clean_atomic_probes import PROBE_CONFIG
    X=np.array([[20.,-2.],[4.,3.],[-3.,4.],[-20.,0.],[8.,-3.],[-6.,1.]],dtype=np.float64)
    y=np.array([1,0,1,0,1,0]);w=np.array([.13,-.27]);b=.4;C=10.
    check=gradient_diagnostics(X,y,w,b,C)
    theta=np.r_[w,b];grad=[]
    for i in range(3):
        delta=np.eye(3)[i]*1e-6
        plus=gradient_diagnostics(X,y,(theta+delta)[:-1],(theta+delta)[-1],C)['objective']
        minus=gradient_diagnostics(X,y,(theta-delta)[:-1],(theta-delta)[-1],C)['objective']
        grad.append((plus-minus)/2e-6)
    assert abs(max(abs(np.array(grad)))-check['gradient_infinity_norm'])<1e-8
    import warnings
    with warnings.catch_warnings(record=True):
        independent=LogisticRegression(C=C,**PROBE_CONFIG).fit(X,y)
    fitted,diagnostics=fit_reference(X,y,C,17)
    np.testing.assert_allclose(fitted.coef_,independent.coef_,atol=1e-12)
    assert diagnostics['valid_for_scoring']


def test_paired_identical_scores_zero_difference_and_coverage_separate():
    from src.selection_repair_canonical_sensitivity import compare_endpoint,atomic_weights
    topics=np.array(['a','a','a','a','b','b','b','b'])
    rows=[dict(topic=topics[i],person_key='p'+str(i//2)) for i in range(8)]
    keep=np.array([True,True,False,False,True,True,True,True])
    options=dict(replicates=20,minimum_valid=18,seed=1729)
    weights=atomic_weights(rows,keep,options)
    scores=np.array([1.,0.,-1.,2.,1.,0.,1.,0.]);y=np.tile([1,0],4)
    records,draws=compare_endpoint(scores,scores,y,np.ones(8,bool),keep,topics,weights,
                  dict(condition='atomic_D',metric='atomic_auroc'),'qwen',options,['a','b'])
    assert all(r['training_delta']==0 for r in records)
    assert records[0]['coverage_delta']>0 and records[0]['excluded_rows']==2
    for a,b in draws:np.testing.assert_array_equal(a,b)
    assert records[0]['paired_delta_ci']['ci_low']==records[0]['paired_delta_ci']['ci_high']==0


def test_refresh_threshold_is_strict_and_rank_change_is_independent():
    from src.selection_repair_canonical_sensitivity import refresh_triggers
    base=dict(model='qwen',condition='raw_reference',metric='and_auroc',scope='pooled',topic='all',status='computed',
              historical_retained_auroc=.7,corrected_retained_auroc=.702,training_delta=.002,coverage_delta=0)
    cfg={'refresh':{'summary_scopes':['pooled'],'absolute_auroc_movement_threshold':.002},
         'bootstrap':{'minimum_valid':18,'replicates':20}}
    d=[(np.full(20,.7),np.full(20,.702))]
    assert not refresh_triggers([base],d,cfg)[0]
    moved=dict(base,training_delta=.00201,corrected_retained_auroc=.70201)
    assert refresh_triggers([moved],d,cfg)[0][0]['kind']=='training_cleanup_auroc'
    right=dict(base,model='llama',historical_retained_auroc=.701,corrected_retained_auroc=.701,training_delta=0)
    triggers,_=refresh_triggers([base,right],d+[(np.full(20,.701),np.full(20,.701))],cfg)
    assert any(t['kind']=='ranking_or_contrast_interpretation_changed' for t in triggers)


def test_pending_correction_blocks_before_preparation_or_fit(monkeypatch,tmp_path):
    from src import selection_repair_canonical_sensitivity as evaluator
    from src import selection_repair_sensitivity_hold as hold
    root=Path(__file__).resolve().parents[1]
    def forbidden(*args,**kwargs):
        pytest.fail('dependent input or fit reached while correction pending')
    monkeypatch.setattr(s,'verify_preparation',forbidden)
    monkeypatch.setattr(evaluator,'fit_reference',forbidden)
    with pytest.raises(ValueError,match='Pending reviewed source correction'):
        evaluator.run(root,tmp_path/'payload',tmp_path/'prepared',tmp_path/'artifacts',tmp_path/'result')
    assert not (tmp_path/'result').exists()
    with pytest.raises(ValueError,match='Missing or changed'):
        hold.require_current_correction(tmp_path)
    pin=tmp_path/hold.HOLD_PATH;pin.parent.mkdir(parents=True);pin.write_text('{"status":"approved"}')
    with pytest.raises(ValueError,match='Missing or changed'):
        hold.require_current_correction(tmp_path)


def test_local_export_audit_and_receipt_mutations(tmp_path,monkeypatch):
    from src.selection_repair_sensitivity_verification import audit_cache
    request,_=remote_fixture(tmp_path,monkeypatch)
    data,record=r.verify_and_export('qwen_train',request)
    path=tmp_path/'export.npy';path.write_bytes(data)
    assert audit_cache(path,request,record)['local_export_stable']
    mutations=[('row',lambda x:x.update(original_row_indices=[1,2])),
               ('layer',lambda x:x.update(saved_layer_index=18)),
               ('source',lambda x:x['source'].update(sha256='0'*64)),
               ('shape',lambda x:x['source']['header'].update(shape=[3,27,4])),
               ('order',lambda x:x['source']['header'].update(fortran_order=True)),
               ('stable',lambda x:x.update(source_stable_through_export=False)),
               ('companion',lambda x:next(iter(x['companions'].values())).update(sha256='0'*64))]
    for _,mutate in mutations:
        bad=copy.deepcopy(record);mutate(bad)
        with pytest.raises(ValueError):audit_cache(path,request,bad)
    damaged=bytearray(data);damaged[-1]^=1;path.write_bytes(damaged)
    with pytest.raises(ValueError,match='local export identity'):audit_cache(path,request,record)


def test_frozen_preparation_rejects_adoption_config_and_file_changes(tmp_path,monkeypatch):
    prepared=tmp_path/'prepared';prepared.mkdir()
    config=tmp_path/s.CONFIG;config.parent.mkdir(parents=True);config.write_text('{}')
    code=tmp_path/'source.py';code.write_text('synthetic')
    manifest=prepared/'map.csv';manifest.write_text('synthetic membership')
    inventory=dict(config_sha256=s.sha(config.read_bytes()))
    (prepared/'input_inventory.json').write_bytes(s.json_bytes(inventory))
    receipt=dict(adoption_sha256='v2',artifacts={'map.csv':s.identity(manifest.read_bytes())},
                 source_sha256={'source.py':s.sha(code.read_bytes())})
    (prepared/'preparation_receipt.json').write_bytes(s.json_bytes(receipt))
    monkeypatch.setattr(s,'verify_adoption',lambda root:({'adoption_sha256':'v2'},{}))
    s.verify_preparation(tmp_path,prepared)
    for path,message in [(manifest,'stale frozen'),(code,'preparation code'),(config,'configuration changed')]:
        original=path.read_bytes();path.write_bytes(original+b' ')
        with pytest.raises(ValueError,match=message):s.verify_preparation(tmp_path,prepared)
        path.write_bytes(original)
    monkeypatch.setattr(s,'verify_adoption',lambda root:({'adoption_sha256':'successor'},{}))
    with pytest.raises(ValueError,match='mixed preparation/adoption'):s.verify_preparation(tmp_path,prepared)


def test_mixed_staging_receipts_rejected_before_export_reads(tmp_path,monkeypatch):
    from src import selection_repair_sensitivity_verification as v
    prepared=tmp_path/'prepared';prepared.mkdir()
    artifacts=tmp_path/'artifacts';artifacts.mkdir()
    (prepared/'preparation_receipt.json').write_text('{}')
    request=s.json_bytes(dict(caches={}))
    (prepared/'export_request.json').write_bytes(request)
    remote=s.json_bytes(dict(request_sha256=s.sha(request)))
    (artifacts/'export_receipt.json').write_bytes(remote)
    stage=dict(remote_receipt=s.identity(remote),preparation_receipt_sha256='wrong',export_request_sha256=s.sha(request))
    (artifacts/'staging_receipt.json').write_bytes(s.json_bytes(stage))
    monkeypatch.setattr(s,'verify_preparation',lambda *args:({},{}))
    with pytest.raises(ValueError,match='mixed staging/preparation'):v.verify_exports(tmp_path,prepared,artifacts)
    stage['remote_receipt']['sha256']='wrong'
    (artifacts/'staging_receipt.json').write_bytes(s.json_bytes(stage))
    with pytest.raises(ValueError,match='stale remote receipt'):v.verify_exports(tmp_path,prepared,artifacts)


def test_no_silent_topic_drop_when_truth_class_missing():
    from src.selection_repair_canonical_sensitivity import compare_endpoint
    labels=np.array([0,1,0,0]);scores=np.array([0.,1.,0.,1.])
    weights=np.ones((20,4));topics=np.array(['a','a','b','b'])
    records,_=compare_endpoint(scores,scores,labels,np.ones(4,bool),np.ones(4,bool),topics,weights,
                              dict(condition='atomic_D',metric='atomic_auroc'),'qwen',
                              dict(replicates=20,minimum_valid=18),['a','b'])
    assert records[0]['status']=='computed'
    assert records[2]['status']==records[3]['status']=='not_computed'
    assert records[3]['corrected_retained_auroc'] is None
    assert records[3]['paired_delta_ci']['valid_replicates']==0


def test_canonical_flagged_fit_not_valid_for_scoring(monkeypatch):
    from src import selection_repair_canonical_sensitivity as evaluator
    from types import SimpleNamespace
    X=np.array([[1.],[-1.]],dtype=np.float64);y=np.array([1,0])
    for coef,converged in [(0.,True),(float('nan'),True),(0.,False)]:
        probe=SimpleNamespace(coef_=np.array([[coef]]),intercept_=np.array([0.]))
        monkeypatch.setattr(evaluator.canonical_lr,'fit_converged_probe',
                            lambda *args,**kwargs:(probe,dict(final_converged=converged)))
        _,diagnostics=evaluator.fit_reference(X,y,1.,17)
        assert diagnostics['valid_for_scoring'] is False
