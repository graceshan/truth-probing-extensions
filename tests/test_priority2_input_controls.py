"""Priority-2 synthetic generation, fake-forward extraction and CPU transfer tests."""
import builtins
from contextlib import contextmanager
import copy
import io
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from src import clean_transfer_contracts as c
from src import method_transfer_contracts as reference
from src import priority2_input_controls as p
from src import priority2_extraction as ex
from src import priority2_transfer as t
from src import priority2_evaluation as e
from src import pinned_compound_extraction as pinned
from src import pinned_compound_scoring as lr_score
from src import clean_transfer_evaluation as lr_eval
from src import clean_atomic_extraction as atomic
from src.repaired_atomic_cache import RepairedAtomicCache
from clean_transfer_fixtures import artifacts, graph


def raw_facts(frame):
    f = frame.copy()
    for side in ['a','b']:
        f[f'fact_{side}_statement'] = [f'The {eid} has the value {"true" if truth else "false"}.'
            for eid,truth in zip(f[f'entity_{side}_id'],f[f'canonical_truth_{side}'])]
    f['statement'] = [p.render_binary(r.fact_a_statement if r.ordering == 'AB' else r.fact_b_statement,
        r.fact_b_statement if r.ordering == 'AB' else r.fact_a_statement,r.operator) for r in f.itertuples()]
    return f


def synthetic_cache(facts, *, covered=None):
    covered = set(facts.fact_key) if covered is None else set(covered)
    rows = [dict(dataset=r.topic,row_index=i,statement=r.statement,entity_id=r.entity_id,
                 topic=r.topic,form='affirmative',split='validation',label=int(r.truth))
            for i,r in enumerate(facts.itertuples()) if r.fact_key in covered]
    return SimpleNamespace(rows={'train':[],'validation':rows})


def test_full_canonical_counts_templates_and_determinism():
    frame,expected = graph(dict(cities=149,sp_en_trans=34,inventors=45,element_symb=18,animal_class=16))
    raw = raw_facts(frame)
    tables,facts,mapping = p.variants(raw)
    assert len(raw) == 8384 and len(facts) == 524 and len(mapping) == 8384
    assert {len(v) for v in tables.values()} == {4192}
    raw_or = raw[raw.operator == 'OR'].set_index('example_id')
    for condition in [p.BOTH,p.LEAST]:
        table = tables[condition]
        assert set(table.base_example_id) == set(raw_or.index)
        for r in table.itertuples():
            original = raw_or.loc[r.base_example_id]
            for key in p.IDENTITY: assert getattr(r,key) == original[key]
            assert r.compound_label == original.compound_label
            first,second = ((r.fact_a_statement,r.fact_b_statement) if r.ordering == 'AB' else (r.fact_b_statement,r.fact_a_statement))
            expected_text = original.statement[:-1]+', or both.' if condition == p.BOTH else 'At least one of the following is true: '+first+' '+second
            assert r.statement == expected_text
    juxtap = tables[p.JUX]
    assert 'operator' not in juxtap and 'compound_label' not in juxtap
    assert juxtap.base_tuple_id.is_unique
    assert set(juxtap.raw_or_example_id) == set(raw_or.index)
    assert set(juxtap.raw_and_example_id) == set(raw.loc[raw.operator == 'AND','example_id'])
    for r in juxtap.itertuples():
        first,second = (r.fact_a_statement,r.fact_b_statement) if r.ordering == 'AB' else (r.fact_b_statement,r.fact_a_statement)
        assert r.statement == first+' '+second
    cache = synthetic_cache(facts,covered=set())
    a,audit = p.build(raw,cache)
    b,_ = p.build(raw,cache)
    assert a == b and audit['missing'] == 524 and audit['extraction_required']


def test_coverage_exact_provenance_missing_and_ambiguity(monkeypatch):
    frame,expected = graph()
    monkeypatch.setattr(c,'ROWS',len(frame))
    monkeypatch.setattr(c,'BENCHMARK',expected)
    _,facts,_ = p.variants(raw_facts(frame))
    cache = synthetic_cache(facts)
    sources,missing,audit = p.coverage(facts,cache)
    assert not len(missing) and not audit['extraction_required'] and audit['covered'] == len(facts)
    assert set(sources.source_kind) == {'repaired'}
    cache.rows['validation'][0]['statement'] += ' '
    sources,missing,audit = p.coverage(facts,cache)
    assert len(missing) == 1 and audit['extraction_required']
    cache = synthetic_cache(facts)
    cache.rows['validation'][0]['entity_id'] = 'different_entity'
    assert len(p.coverage(facts,cache)[1]) == 1
    cache = synthetic_cache(facts)
    cache.rows['validation'][0]['label'] = 1-cache.rows['validation'][0]['label']
    assert len(p.coverage(facts,cache)[1]) == 1
    cache = synthetic_cache(facts)
    cache.rows['validation'].append(cache.rows['validation'][0].copy())
    with pytest.raises(ValueError,match='ambiguous'): p.coverage(facts,cache)
    raw = raw_facts(frame)
    raw.loc[0,'split'] = 'test'
    with pytest.raises(ValueError,match='scope'): p.variants(raw)


def test_composition_formulas_and_zero_boundary():
    a=np.array([-2.,0.,3.,2.,-1.,0.])
    b=np.array([1.,-1.,2.,-2.,-3.,0.])
    op=np.array(['AND','OR','AND','OR','AND','AND'])
    continuous,boolean=e.composition(a,b,op)
    np.testing.assert_array_equal(continuous,[-2.,0.,2.,2.,-3.,0.])
    np.testing.assert_array_equal(boolean,[False,True,True,True,False,True])
    with pytest.raises(ValueError): e.composition(a,b,['XOR']*6)


@pytest.fixture
def controls(artifacts,monkeypatch):
    root,frame,_=artifacts
    raw=raw_facts(frame)
    directory=root/c.COMPOUND
    raw.to_csv(directory/'metadata.csv',index=False)
    raw_path=root/p.RAW
    raw_path.parent.mkdir(parents=True)
    raw_path.write_bytes((directory/'metadata.csv').read_bytes())
    monkeypatch.setattr(c,'METADATA_SHA',c.file_hash(raw_path))
    manifest=c.read_json(directory/'extraction_manifest.json')
    manifest['data'].update(sidecar_sha256=c.METADATA_SHA,benchmark_sha256=c.METADATA_SHA)
    (directory/'extraction_manifest.json').write_bytes(c.canonical(manifest))
    progress=c.read_json(directory/'progress.json')
    progress['identity_sha256']=lr_score.extraction_identity(manifest)
    (directory/'progress.json').write_bytes(c.canonical(progress))
    template=c.spec_template
    def small_template(inputs,descriptor):
        spec=template(inputs,descriptor)
        spec['bootstrap'].update(replicates=40,minimum_valid=36)
        return spec
    monkeypatch.setattr(c,'spec_template',small_template)
    lr_score.preflight(root)
    lr_score.freeze_spec(root)
    lr_score.score(root)
    lr_eval.evaluate(root)
    monkeypatch.setattr(reference,'LR_SPEC_SHA',c.file_hash(root/c.SPEC))
    monkeypatch.setattr(reference,'LR_EVALUATION_SHA',c.file_hash(root/c.OUTPUT/'evaluation/evaluation_manifest.json'))
    monkeypatch.setattr(reference,'LR_PRIMARY_SHA',c.file_hash(root/c.OUTPUT/'evaluation/primary_metrics.csv'))
    monkeypatch.setattr(pinned,'WIDTH',3)
    cache=RepairedAtomicCache(root)
    expected={}
    for split,rows in cache.rows.items():
        values=np.load(root/c.ATOMIC/split/'activations.npy',allow_pickle=False)
        for i,r in enumerate(rows): expected[r['statement']]=values[i].copy()
    calls=[]
    def readout(model,tokenizer,statement):
        calls.append(statement)
        if statement not in expected:
            rng=np.random.default_rng(int(c.digest(statement.encode())[:8],16))
            expected[statement]=rng.normal(size=(28,3)).astype(np.float16)
        return torch.from_numpy(expected[statement].astype(np.float32))
    monkeypatch.setattr(atomic,'load_pinned_model',lambda:(None,None,cache.manifest['resolved_model']))
    monkeypatch.setattr(pinned,'pinned_readout',readout)
    monkeypatch.setattr(atomic,'reference_readout',readout)
    monkeypatch.setattr(pinned,'runtime_provenance',lambda:dict(torch='2.11.0',transformers='5.12.1'))
    return root,raw,cache,calls


@contextmanager
def guarded(monkeypatch,root,stage):
    with monkeypatch.context() as patch:
        read=pd.read_csv
        def checked_csv(path,*args,**kwargs):
            name=str(path)
            if stage=='score' and (name == str(root/p.RAW) or name == str(root/c.COMPOUND/'metadata.csv')):
                assert callable(kwargs.get('usecols'))
                assert all(not kwargs['usecols'](x) for x in ['canonical_truth_a','canonical_truth_b','compound_label','surface_first_truth','surface_second_truth'])
            if stage=='score' and name.startswith(str(root/p.DATA)):
                assert Path(name).name in ['statements.csv','scoring_index.csv','isolated_sources.csv','isolated_facts.csv']
                if Path(name).name=='isolated_facts.csv': assert kwargs.get('usecols') is not None
            return read(path,*args,**kwargs)
        patch.setattr(pd,'read_csv',checked_csv)
        for module in [builtins,io,os]:
            original=module.open
            def opened(path,*args,_original=original,**kwargs):
                if not isinstance(path,int):
                    name=Path(os.fsdecode(path)).absolute()
                    if name.is_relative_to(root):
                        assert not any(x in name.parts for x in ['test','atomic_test','compound_test'])
                        if stage=='evaluate':
                            assert not name.is_relative_to(root/c.ATOMIC) and not name.is_relative_to(root/c.PROBE) and not name.is_relative_to(root/p.ACTS)
                    elif name.is_relative_to(c.ROOT):
                        assert name.suffix in ['.py','.pyc'], name
                return _original(path,*args,**kwargs)
            patch.setattr(module,'open',opened)
        yield


def test_fake_extraction_truth_blind_scoring_and_evaluation(controls,monkeypatch):
    root,raw,cache,calls=controls
    preserved={path:path.read_bytes() for base in [root/c.COMPOUND,root/c.ATOMIC,root/c.PROBE,root/c.OUTPUT]
               for path in base.rglob('*') if path.is_file()}
    p.generate(root)
    assert ex.run('plan',root)['model_loaded'] is False and calls==[]
    plan=ex.run('extract',root)
    assert plan['shape']==[3*len(raw)//2+2*c.BENCHMARK['entities'],28,3]
    _,generated=p.verify_generation(root,cache)
    files,statements=ex.verify(root,cache,generated)
    assert ex.Statements(root/p.ACTS/'metadata.csv').frame.equals(statements)
    with guarded(monkeypatch,root,'score'):
        load=np.load
        def preflight_load(path,*args,**kwargs):
            assert str(path).endswith('selected_probe.npz')
            return load(path,*args,**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(np,'load',preflight_load)
            t.preflight(root)
            t.freeze_spec(root)
        with pytest.raises(ValueError,match='commit exact'): t.score(root)
        monkeypatch.setattr(t,'committed_spec',lambda root:None)
        t.score(root)
    scores=pd.read_csv(root/p.OUTPUT/'scores/condition_scores.csv',float_precision='round_trip')
    isolated=pd.read_csv(root/p.OUTPUT/'scores/isolated_scores.csv',float_precision='round_trip')
    assert list(scores)==t.SCORE_COLUMNS and list(isolated)==t.ISOLATED_COLUMNS
    with np.load(root/c.PROBE/'selected_probe.npz',allow_pickle=False) as probe:
        arrays=np.load(root/p.ACTS/'activations.npy',allow_pickle=False)
        expected=arrays[:,17,:].astype(float)@probe['coef'][0]+probe['intercept'][0]
        lookup=dict(zip(statements.example_id,expected))
        np.testing.assert_allclose(scores.frozen_probe_score,scores.example_id.map(lookup),rtol=1e-14,atol=1e-14)
        np.testing.assert_allclose(isolated.frozen_probe_score,isolated.fact_key.map(lookup),rtol=1e-14,atol=1e-14)
    with guarded(monkeypatch,root,'evaluate'):
        with monkeypatch.context() as patch:
            patch.setattr(np,'load',lambda *a,**k:pytest.fail('evaluator opened activation/archive'))
            e.evaluate(root)
    output=root/p.OUTPUT/'evaluation'
    formal=pd.read_csv(output/'formal_metrics.csv')
    geometry=pd.read_csv(output/'juxtaposition_geometry.csv')
    boolean=pd.read_csv(output/'boolean_metrics.csv')
    assert p.JUX not in set(formal.condition_id)
    assert set(geometry.category)=={'geometry'} and len(geometry)==4*7
    assert set(boolean.condition_id)=={p.BOOLEAN} and not boolean.metric.str.contains('auroc').any()
    with np.load(output/'bootstrap_draws.npz',allow_pickle=False) as z:
        lookup={key:i for i,key in enumerate(z['metric_ids'])}
        values=z['values']
        for contrast in t.contrasts():
            for metric in contrast['metrics']:
                suffix='pooled/all/'+metric
                np.testing.assert_equal(values[:,lookup[contrast['left']+'_minus_'+contrast['right']+'/'+suffix]],
                    values[:,lookup[contrast['left']+'/'+suffix]]-values[:,lookup[contrast['right']+'/'+suffix]])
        # Raw AND is identical under both OR wording conditions in every replicate.
        for condition in [p.BOTH,p.LEAST]:
            np.testing.assert_equal(values[:,lookup[condition+'/pooled/all/and_auroc']],values[:,lookup[p.RAW_ID+'/pooled/all/and_auroc']])
    assert all(path.read_bytes()==payload for path,payload in preserved.items())
    for call in [lambda:p.generate(root),lambda:ex.run('extract',root),lambda:t.preflight(root),lambda:t.freeze_spec(root),lambda:t.score(root),lambda:e.evaluate(root)]:
        with pytest.raises(ValueError,match='refusing'): call()


def test_extraction_gates_and_input_tamper(controls,monkeypatch):
    root,_,cache,calls=controls
    p.generate(root)
    real=pinned.replay_atomic
    monkeypatch.setattr(pinned,'replay_atomic',lambda *args:dict(passed=False))
    with pytest.raises(ValueError,match='replay'): ex.run('extract',root)
    assert not (root/p.ACTS).exists()
    monkeypatch.setattr(pinned,'replay_atomic',real)
    real_smoke=pinned.compound_smoke
    monkeypatch.setattr(pinned,'compound_smoke',lambda *a,**k:dict(passed=False,batch_size=1,padding=False))
    with pytest.raises(ValueError,match='smoke'): ex.run('extract',root)
    assert not (root/p.ACTS).exists()
    monkeypatch.setattr(pinned,'compound_smoke',real_smoke)
    path=root/p.DATA/'statements.csv'
    with path.open('ab') as handle: handle.write(b'altered')
    with pytest.raises(ValueError,match='generation hash'): ex.run('plan',root)


def test_isolated_readouts_reuse_exact_rows_and_no_test_interface(controls):
    root,_,_,_=controls
    with np.load(root/c.PROBE/'selected_probe.npz',allow_pickle=False) as z:
        probe=dict(layer=int(z['layer']),coef=z['coef'][0],intercept=float(z['intercept'][0]))
    sources=pd.DataFrame([dict(fact_key='one',source_kind='repaired',cache_split='train',cache_row_index='2',example_id=''),
                          dict(fact_key='two',source_kind='repaired',cache_split='validation',cache_row_index='3',example_id='')])
    result=t.isolated_readouts(root,sources,{},probe,256)
    for key,split,index in [('one','train',2),('two','validation',3)]:
        X=np.load(root/c.ATOMIC/split/'activations.npy',allow_pickle=False)
        assert result[key]==pytest.approx(X[index,17].astype(float)@probe['coef']+probe['intercept'])
    extra=pd.DataFrame([dict(fact_key='three',source_kind='extract',cache_split='',cache_row_index='',example_id='three')])
    mixed=t.isolated_readouts(root,pd.concat([sources,extra],ignore_index=True),{'three':-4.25},probe,1)
    assert mixed=={**result,'three':-4.25}
    sources.loc[0,'cache_split']='test'
    with pytest.raises(ValueError,match='coverage'): t.isolated_readouts(root,sources,{},probe,256)


def test_truth_free_statement_header_rejected_before_parsing(tmp_path,monkeypatch):
    path=tmp_path/'statements.csv'
    path.write_text('example_id,condition_id,statement,split,protocol,evaluation_phase,compound_label\nx,y,z,validation,entity_disjoint,development,1\n')
    monkeypatch.setattr(pd,'read_csv',lambda *a,**k:pytest.fail('parsed unexpected truth column'))
    with pytest.raises(ValueError,match='truth-free'): ex.Statements(path)


def test_manifest_spec_and_score_corruption_fail_closed(controls,monkeypatch):
    root,_,cache,_=controls
    p.generate(root)
    ex.run('extract',root)
    _,files=p.verify_generation(root,cache)
    manifest_path=root/p.ACTS/'extraction_manifest.json'
    original=manifest_path.read_bytes()
    manifest=c.read_json(manifest_path)
    manifest['smoke_test']['compound_smoke']['conditions'][p.JUX]['passed']=False
    manifest_path.write_bytes(c.canonical(manifest))
    with pytest.raises(ValueError,match='smoke'): ex.verify(root,cache,files)
    manifest_path.write_bytes(original)
    progress_path=root/p.ACTS/'progress.json'
    prior=progress_path.read_bytes()
    progress=c.read_json(progress_path)
    progress['next_row']-=1
    progress_path.write_bytes(c.canonical(progress))
    with pytest.raises(ValueError,match='incomplete'): ex.verify(root,cache,files)
    progress_path.write_bytes(prior)
    t.preflight(root)
    t.freeze_spec(root)
    spec=t.validate(c.read_json(root/p.SPEC))
    for mutate in [lambda s:s.update(unknown=True),lambda s:s['bootstrap'].update(seed=1),
                   lambda s:s['templates'].update({p.JUX:'A blah B.'}),lambda s:s['composition'].update(continuous_and='mean(s_A,s_B)'),
                   lambda s:s['condition_counts'].update({p.BOTH:4191}),lambda s:s['inputs']['extraction']['files']['activations.npy'].update(sha256='placeholder')]:
        candidate=copy.deepcopy(spec)
        mutate(candidate)
        with pytest.raises(ValueError): t.validate(candidate)
    monkeypatch.setattr(t,'committed_spec',lambda root:None)
    t.score(root)
    scorepath=root/p.OUTPUT/'scores/condition_scores.csv'
    scores=pd.read_csv(scorepath,float_precision='round_trip')
    scores.loc[1,'example_id']=scores.loc[0,'example_id']
    scores.to_csv(scorepath,index=False,float_format='%.17g')
    scoring_path=scorepath.parent/'scoring_manifest.json'
    manifest=c.read_json(scoring_path)
    manifest['outputs']['condition_scores.csv']=c.record(scorepath)
    scoring_path.write_bytes(c.canonical(manifest))
    with pytest.raises(ValueError,match='identity/cardinality'): e.load(root,spec)
