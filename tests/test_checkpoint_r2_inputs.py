import copy
import csv
import io
import json
from collections import Counter
import numpy as np
import pytest
from src import checkpoint_r2_inputs as m
from src.checkpoint_r2_bridge import compare_bridge


@pytest.fixture(scope='module')
def built():
    return m.build()


def test_determinism_and_immutable_outputs(built):
    first,_=built; second,_=m.build()
    assert first==second
    for name,payload in first.items():
        assert (m.ROOT/m.OUTPUT/name).read_bytes()==payload


def test_disjoint_folds_and_source_split(built):
    files,receipt=built
    entities=json.loads(files['entities.json']);pairs=json.loads(files['pairs.json']);groups=json.loads(files['pair_groups.json'])
    split=list(csv.DictReader(io.StringIO(files['constituent_split.csv'].decode())))
    validation={r['person_key'] for r in split if r['role']=='source_validation'}
    assert len(validation)==20 and validation<={r['person_key'] for r in entities if r['A15']}
    for p in groups['constituent_source_fit']:
        assert not {pairs[p['pair_id']]['person_a'],pairs[p['pair_id']]['person_b']}&validation
    for seed in (11,23,37):
        seen=set()
        for fold in range(5):
            ids={pairs[p['pair_id']][k] for p in groups[f'B25_{seed}'] if p['fold']==fold for k in ('person_a','person_b')}
            assert len(ids)==10 and not seen&ids
            seen|=ids
        assert len(seen)==50
    assert receipt['counts']['balanced']==700


def test_complete_cells_reversible_texts_and_behavior(built):
    files,_=built
    rows=list(csv.DictReader(io.StringIO(files['compound_bindings.csv'].decode())))
    groups={}
    for r in rows:groups.setdefault((r['group'],r['pair_id']),[]).append(r)
    for rs in groups.values():
        assert len(rs)==16
        assert len({(r['truth_a'],r['truth_b'],r['operator'],r['ordering']) for r in rs})==16
    inv=list(csv.DictReader(io.StringIO(files['extraction_inventory.csv'].decode())))
    assert len({(r['kind'],r['group'],r['logical_id'],r['pair_id']) for r in inv})==len(inv)
    for r in inv:assert r['text_id']=='text_'+m.sha(r['statement'].encode())
    behavior=json.loads(files['behavior_pairs.json']); assert len(behavior)==len(set(behavior))==50
    assert Counter(r['topic'] for r in rows if r['group']=='D_bare' and r['pair_id'] in behavior)==Counter({t:160 for t in m.TOPICS})


def example(built):
    files,_=built;facts=json.loads(files['facts.json'])
    row=next(csv.DictReader(io.StringIO(files['compound_bindings.csv'].decode())))
    for k in ('truth_a','truth_b','surface_first_truth','surface_second_truth','label'):row[k]=int(row[k])
    return row,facts


@pytest.mark.parametrize('mutation',['label','person','split','text','unresolved'])
def test_reject_invalid_constituent_or_compound(built,mutation):
    row,facts=example(built)
    if mutation=='label':row['label']=1-row['label']
    if mutation=='person':row['person_a']='another-person'
    if mutation=='split':facts[row['fact_a_id']]['split']='test'
    if mutation=='text':row['statement']='Substitute fact.'
    if mutation=='unresolved':facts[row['fact_a_id']]['eligible']=False
    with pytest.raises(ValueError):m.validate_row(row,facts)


def test_reject_exact_registry_reintroduction():
    e={'topic':'cities','person_key':'person'}
    f={'label':0,'judgment':'supported_false','statement':'Exact.','statement_sha256':m.sha(b'Exact.'),'origin':'generated','evidence_ids':['ev']}
    ledger=[dict(topic='cities',person_key='person',statement='Exact.',split='train',e_current_eligible='False')]
    evidence={'ev':{'fact_bindings':[dict(e,statement_sha256=f['statement_sha256'],judgment='supported_false')]}}
    with pytest.raises(ValueError,match='restricted registry'):m.validate_fact(f,e,[],ledger,evidence)


def bridge_fixture():
    return dict(ids=['train:1'],split='train',tokens=[[1,2]],positions=[[0,1]],masks=[[1,1]],hidden=np.zeros((1,28,3584),dtype=np.float16),scores=np.zeros((1,1)),contract={'model':'qwen'},probe_bindings={'sha256':'frozen-head-and-preprocessing'})


def test_unknown_history_never_passes_reuse():
    a=bridge_fixture();b=copy.deepcopy(a)
    r=compare_bridge(a['ids'],a,b)
    assert r['verdict']=='unverifiable' and not r['reuse_permitted']
    assert compare_bridge(a['ids'],a,b,True)['reuse_permitted']


@pytest.mark.parametrize('mutation',['token','position','mask','hidden','score','contract','probe'])
def test_bridge_rejects_discrepancy(mutation):
    a=bridge_fixture();b=copy.deepcopy(a)
    if mutation=='token':b['tokens'][0][0]=3
    if mutation=='position':b['positions'][0][0]=4
    if mutation=='mask':b['masks'][0][0]=0
    if mutation=='hidden':b['hidden'][0,0,0]=.01
    if mutation=='score':b['scores'][0,0]=1e-12
    if mutation=='contract':b['contract']['dtype']='different'
    if mutation=='probe':b['probe_bindings']['sha256']='different'
    assert compare_bridge(a['ids'],a,b,True)['verdict']=='failed'


def test_bridge_rejects_E_and_missing_layers():
    a=bridge_fixture();b=copy.deepcopy(a);b['split']='test'
    with pytest.raises(ValueError):compare_bridge(a['ids'],a,b,True)
    b=copy.deepcopy(a);b['hidden']=b['hidden'][:,:1]
    with pytest.raises(ValueError):compare_bridge(a['ids'],a,b,True)
