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
    assert a == b and audit['no_match'] == 524
    # Coverage never controls extraction cardinality or scientific source.
    for candidate in [cache, synthetic_cache(facts)]:
        payloads,report = p.build(raw,candidate)
        statements = pd.read_csv(io.BytesIO(payloads['statements.csv']),keep_default_na=False)
        sources = pd.read_csv(io.BytesIO(payloads['isolated_sources.csv']),keep_default_na=False)
        assert statements.groupby('condition_id').size().to_dict() == {
            p.BOTH:4192,p.LEAST:4192,p.JUX:4192,p.ISO:524}
        assert len(statements) == 13100
        isolated = statements[statements.condition_id == p.ISO]
        assert isolated.example_id.is_unique and isolated.statement.is_unique
        assert set(isolated.example_id) == set(facts.fact_key)
        assert isolated.set_index('example_id').statement.to_dict() == facts.set_index('fact_key').statement.to_dict()
        assert set(sources.source_kind) == {'fresh_extraction'}
        assert sources.cache_split.eq('').all() and sources.cache_row_index.eq('').all()
        assert sources.fact_key.equals(sources.example_id)
        assert report['fresh_isolated_rows'] == 524 and report['isolated_scoring_source'] == 'fresh_priority2_extraction'
        assert payloads['statements.csv'] == a['statements.csv']
        assert payloads['isolated_sources.csv'] == a['isolated_sources.csv']


def test_coverage_exact_provenance_missing_and_ambiguity(monkeypatch):
    frame,expected = graph()
    monkeypatch.setattr(c,'ROWS',len(frame))
    monkeypatch.setattr(c,'BENCHMARK',expected)
    _,facts,_ = p.variants(raw_facts(frame))
    cache = synthetic_cache(facts)
    sources,audit = p.coverage(facts,cache)
    assert audit['no_match'] == audit['duplicate_exact_match'] == 0 and audit['unique_exact_match'] == len(facts)
    assert set(sources.source_kind) == {'fresh_extraction'}
    cache.rows['validation'][0]['statement'] += ' '
    sources,audit = p.coverage(facts,cache)
    assert audit['no_match'] == 1
    cache = synthetic_cache(facts)
    cache.rows['validation'][0]['entity_id'] = 'different_entity'
    assert p.coverage(facts,cache)[1]['no_match'] == 1
    cache = synthetic_cache(facts)
    cache.rows['validation'][0]['topic'] = 'different_topic'
    assert p.coverage(facts,cache)[1]['no_match'] == 1
    cache = synthetic_cache(facts)
    cache.rows['validation'][0]['label'] = 1-cache.rows['validation'][0]['label']
    assert p.coverage(facts,cache)[1]['no_match'] == 1
    cache = synthetic_cache(facts)
    # Same structure as the real issue: identical true Spanish key, distinct
    # original row indices in repaired VALIDATION, both affirmative.
    index = next(i for i,r in enumerate(cache.rows['validation']) if r['topic']=='sp_en_trans' and r['label']==1)
    duplicate = {**cache.rows['validation'][index], 'row_index':10000}
    cache.rows['validation'].append(duplicate)
    sources,audit = p.coverage(facts,cache)
    assert audit['duplicate_exact_match'] == 1 and audit['unique_exact_match'] == len(facts)-1
    detail = next(r for r in audit['identities'] if r['classification']=='duplicate_exact_match')
    expected_candidates = [dict(cache_split='validation',cache_row_index=i,
        **{k:cache.rows['validation'][i][k] for k in ['dataset','row_index','form','label','statement','entity_id','topic']})
        for i in [index,len(cache.rows['validation'])-1]]
    assert detail['candidates'] == expected_candidates and detail['exact_match_count'] == 2
    assert sources.cache_split.eq('').all() and sources.cache_row_index.eq('').all()
    assert sources.example_id.equals(sources.fact_key)
    class NoTest(dict):
        def __getitem__(self,key):
            assert key in ['train','validation'], 'TEST accessed'
            return super().__getitem__(key)
    cache.rows = NoTest(cache.rows)
    payloads,_ = p.build(raw_facts(frame),cache)
    repeated,_ = p.build(raw_facts(frame),cache)
    assert payloads == repeated
    assert len(pd.read_csv(io.BytesIO(payloads['statements.csv']))) == sum(p.condition_counts().values())
    cache.rows['validation'][0]['split'] = 'test'
    with pytest.raises(ValueError,match='scope'): p.coverage(facts,cache)
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


def test_audit_does_not_generate_and_duplicate_does_not_block_generation(tmp_path,monkeypatch):
    frame,expected = graph()
    monkeypatch.setattr(c,'ROWS',len(frame))
    monkeypatch.setattr(c,'BENCHMARK',expected)
    raw = raw_facts(frame)
    _,facts,_ = p.variants(raw)
    cache = synthetic_cache(facts)
    cache.files = {}
    row = next(r for r in cache.rows['validation'] if r['topic']=='sp_en_trans' and r['label']==1)
    cache.rows['validation'].append({**row,'row_index':10000})
    monkeypatch.setattr(p,'RepairedAtomicCache',lambda root:cache)
    path = tmp_path/p.RAW
    path.parent.mkdir(parents=True)
    raw.to_csv(path,index=False)
    monkeypatch.setattr(c,'METADATA_SHA',c.file_hash(path))
    original = path.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(p,'build',lambda *a:pytest.fail('audit invoked generation'))
        audit = p.generate(tmp_path,audit_only=True)
    assert audit['duplicate_exact_match'] == 1
    assert audit['required'] == audit['fresh_isolated_rows'] == len(facts)
    assert audit['isolated_scoring_source'] == p.ISOLATED_SCORING_SOURCE
    assert audit['atomic_test_accessed'] is False and not (tmp_path/p.DATA).exists()
    result = p.generate(tmp_path)
    assert result['generated'] is True
    manifest,_ = p.verify_generation(tmp_path,cache)
    assert manifest['condition_counts'] == p.condition_counts()
    assert 'missing_facts' not in manifest and path.read_bytes() == original
    ex.Statements(tmp_path/p.DATA/'statements.csv')


def test_metric_and_bootstrap_policy_unchanged():
    lr = c.spec_template({}, {})
    spec = t.template(lr, {}, {}, {}, p.condition_counts())
    assert spec['bootstrap'] == lr['bootstrap']
    assert spec['statistics'] == lr['statistics']
    assert spec['formal_metrics'] == [d for d in lr['metrics'] if d['category'] in ['primary','boundary']]
    assert spec['boolean_metrics'] == [d for d in lr['metrics'] if d['category']=='threshold']
    assert spec['composition'] == dict(continuous_and='min(s_A,s_B)',continuous_or='max(s_A,s_B)',
        boolean_and='(s_A>=0) and (s_B>=0)',boolean_or='(s_A>=0) or (s_B>=0)',
        interpretation='two isolated model evaluations plus a known parse/external composition rule',boolean_auroc_reported=False)
    assert spec['condition_counts'] == {p.BOTH:4192,p.LEAST:4192,p.JUX:4192,p.ISO:524}


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
        def score_load(path,*args,**kwargs):
            assert not Path(path).is_relative_to(root/c.ATOMIC), 'isolated scoring opened repaired atomic array'
            return load(path,*args,**kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(np,'load',score_load)
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


def test_isolated_readouts_fresh_only_without_file_access(monkeypatch):
    monkeypatch.setattr(c,'BENCHMARK',dict(entities=1))
    sources=pd.DataFrame([dict(fact_key=key,source_kind='fresh_extraction',cache_split='',cache_row_index='',example_id=key)
                          for key in ['one','two']],columns=p.SOURCE_FIELDS)
    def forbidden(*args,**kwargs):
        pytest.fail('isolated score selection attempted file/activation access')
    monkeypatch.setattr(np,'load',forbidden)
    monkeypatch.setattr(builtins,'open',forbidden)
    monkeypatch.setattr(io,'open',forbidden)
    monkeypatch.setattr(os,'open',forbidden)
    extracted={'one':-4.25,'two':1.5}
    assert t.isolated_readouts(sources,extracted) == extracted
    for key,value in [('source_kind','repaired'),('source_kind','extract'),('cache_split','test'),
                      ('cache_split','validation'),('cache_row_index','3'),('example_id','compound_row')]:
        altered=sources.copy()
        altered.loc[0,key]=value
        with pytest.raises(ValueError,match='fresh isolated source'): t.isolated_readouts(altered,extracted)
    for scores in [{'one':1.}, {**extracted,'compound':3.}, {'one':np.nan,'two':1.}]:
        with pytest.raises(ValueError,match='coverage'): t.isolated_readouts(sources,scores)


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
                   lambda s:s['condition_counts'].update({p.BOTH:4191}),
                   lambda s:s['condition_counts'].update(missing_isolated=0),
                   lambda s:s.update(isolated_scoring_source='repaired'),
                   lambda s:s['inputs']['extraction']['files']['activations.npy'].update(sha256='placeholder')]:
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
