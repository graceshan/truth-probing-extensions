"""Synthetic method equivalence, no-leak inference and pinned metadata integration."""
import json
from pathlib import Path
import numpy as np
import pytest
from threadpoolctl import threadpool_limits
from src import selection_repair_capacity_methods_v1 as m
from src import atomic_probe_methods as old
from src import selection_repair_capacity_pilot_v1 as p
from src import selection_repair_capacity_staging_v1 as staging
from src import selection_repair_capacity_export_v1 as remote


def synthetic():
    rng=np.random.default_rng(52);X=rng.normal(size=(120,7));y=np.tile([0,1],60)
    X[:,0]+=y*1.4
    rows=[dict(form='affirmative' if i<60 else 'negated',dataset='a' if i<60 else 'neg_a') for i in range(120)]
    X[:,1]+=np.array([1 if i<60 else -1 for i in range(120)])
    return X,y,rows


@pytest.mark.parametrize('method',m.METHODS)
def test_synthetic_methods_inference_and_exact_fitting_cohort(method):
    X,y,rows=synthetic()
    with threadpool_limits(limits=1):
        params,details=m.fit(method,X,y,rows,1.,17)
        assert details['valid_for_scoring']
        scored=m.score(method,params,X)
        np.testing.assert_allclose(scored,X@params['coef']+params['intercept'],atol=1e-8)
        # Inference is independent of other evaluation rows and their labels/forms.
        assert m.score(method,params,X[:1])[0]==pytest.approx(scored[0],abs=1e-12)
        if method=='r0':
            np.testing.assert_array_equal(params['mean'],X.mean(0))
            np.testing.assert_array_equal(params['population_std'],X.std(0))
        if method=='ttpd':
            faithful=old.fit_ttpd(X,y,np.array([1 if r['form']=='affirmative' else -1 for r in rows]),np.array([r['dataset'] for r in rows]))
            np.testing.assert_array_equal(params['coef'],faithful.parameters['coef'])
            np.testing.assert_array_equal(scored,faithful.decision_function(X))


def test_r0_constant_coordinate_never_scores_shifted_evaluation():
    X,y,rows=synthetic();X[:,1]=.1
    with threadpool_limits(limits=1):params,diagnostics=m.fit('r0',X,y,rows,1,17)
    assert params['constant_mask'][1] and params['coef'][1]==0
    shifted=X.copy();shifted[:,1]=1e9
    np.testing.assert_array_equal(m.score('r0',params,shifted),m.score('r0',params,X))


def test_current_cohorts_exact_exclusion_and_sampler():
    root=Path(__file__).resolve().parents[1]
    cohorts,samples=p.cohorts(root)
    assert [len(cohorts[k]) for k in ('full','P15')]==[3040,2778]
    assert [len(samples[k]) for k in ('full','P15')]==[1000,700]
    a=json.loads((root/p.s.projection.PROJECTION/'A15_proposal.json').read_text())['completed_pairs']
    assert not {r['person_key'] for r in a}&{r['person_key'] for r in cohorts['P15']}
    for rows in samples.values():
        assert all(c['labels']['0']==c['labels']['1'] for c in p.counts(rows))


def test_grid_exact_and_no_fallback():
    root=Path(__file__).resolve().parents[1]
    config=json.loads((root/p.CONFIG).read_text());grid=p.configurations(config)
    assert len(grid)==len({tuple(r.values()) for r in grid})==72
    assert {r['cohort'] for r in grid}=={'full','P15'}
    assert remote.DIRECTORIES.keys()==staging.DIRECTORIES.keys()-{'qwen_controls'}


@pytest.mark.parametrize('key,layer',[('qwen_train',17),('llama_train',15),('qwen_raw',0),('llama_transfer',31),('qwen_controls',18)])
def test_additional_exporter_refuses_unscoped_layer_or_cache(key,layer):
    path=str(remote.ROOT/remote.DIRECTORIES.get(key,'invalid')/'activations.npy')
    with pytest.raises(ValueError):remote.verify_and_export(key,dict(path=path,layer=layer))


def test_remote_host_verified_before_request_or_file_access(monkeypatch):
    monkeypatch.setattr(remote.socket,'gethostname',lambda:'wrong')
    with pytest.raises(ValueError,match='hostname'):remote.main()

@pytest.mark.parametrize('chunk_size',[1,3,8,11,100])
def test_stream_export_exact_spans_across_chunk_boundaries(chunk_size):
    import io,hashlib
    from src import selection_repair_capacity_export_v2 as v2
    data=bytes(range(100));outputs={'a':io.BytesIO(),'b':io.BytesIO()}
    spans=[(0,7,'a'),(7,13,'b'),(19,33,'a'),(51,70,'b'),(98,100,'a')]
    size,digest=v2.stream_select(io.BytesIO(data),spans,outputs,chunk_size)
    assert size==100 and digest==hashlib.sha256(data).hexdigest()
    assert outputs['a'].getvalue()==data[:7]+data[19:33]+data[98:]
    assert outputs['b'].getvalue()==data[7:13]+data[51:70]


def test_stream_export_truncation_rejected():
    import io
    from src import selection_repair_capacity_export_v2 as v2
    with pytest.raises(ValueError,match='truncated'):
        v2.stream_select(io.BytesIO(b'abc'),[(0,5,'a')],{'a':io.BytesIO()},2)

@pytest.mark.parametrize('mutation',['none','test_row','wrong_layer','wrong_hash'])
def test_single_pass_export_complete_synthetic_source(tmp_path,monkeypatch,mutation):
    import io,hashlib,copy
    from src import selection_repair_capacity_export_v2 as v2
    monkeypatch.setattr(v2,'ROOT',tmp_path)
    directory=tmp_path/v2.DIRECTORIES['llama_transfer'];directory.mkdir(parents=True)
    tensor=np.arange(4*32*3,dtype=np.float16).reshape(4,32,3)
    np.save(directory/'activations.npy',tensor)
    metadata='example_id,split,condition_id,evaluation_phase\n'
    metadata+='a,validation,raw_reference,development\n'
    metadata+='b,validation,raw_reference,development\n'
    metadata+='c,'+('test' if mutation=='test_row' else 'validation')+',raw_reference,development\n'
    metadata+='d,validation,raw_reference,development\n'
    (directory/'metadata.csv').write_text(metadata)
    for name in ('extraction_manifest.json','progress.json'):(directory/name).write_text('{}')
    pin=lambda p:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    spec=dict(path=str(directory/'activations.npy'),**pin(directory/'activations.npy'),shape=[4,32,3],row_indices=[0,2],row_identity_sha256='synthetic',
              companions={str(directory/n):pin(directory/n) for n in ('metadata.csv','extraction_manifest.json','progress.json')})
    jobs={f'llama_transfer_L{layer}':dict(spec,layer=layer) for layer in [10,20]}
    if mutation=='wrong_layer':jobs['llama_transfer_L20']['layer']=21
    if mutation=='wrong_hash':
        for job in jobs.values():job['sha256']='0'*64
    if mutation!='none':
        with pytest.raises(ValueError):v2.export_source('llama_transfer',jobs)
        return
    result=v2.export_source('llama_transfer',jobs)
    for layer in [10,20]:
        data,record=result[f'llama_transfer_L{layer}']
        np.testing.assert_array_equal(np.load(io.BytesIO(data),allow_pickle=False),tensor[[0,2],layer])
        assert record['export']['sha256']==hashlib.sha256(data).hexdigest()
        assert record['original_row_indices']==[0,2] and record['source_stable_through_export']
