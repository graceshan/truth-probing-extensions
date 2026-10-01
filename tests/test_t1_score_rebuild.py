"""Synthetic-only ordering, join, and shared endpoint bootstrap regressions."""
import numpy as np
import pandas as pd
import pytest

from src import clean_transfer_contracts as c
from src.clean_transfer_evaluation import validate_metadata
from src.clean_transfer_statistics import EntityBootstrap, AUC, interval
from src.t1_score_package import exact_join, bind_constituents, safe_file
from src.t1_score_rebuild import surface_cells, diagnostics
from src.pinned_method_evaluation import compute
from clean_transfer_fixtures import graph


def fixture():
    raw,benchmark=graph()
    frame=validate_metadata(raw,benchmark)
    spec=c.spec_template({}, {})
    spec['benchmark']=benchmark
    spec['bootstrap'].update(replicates=40,minimum_valid=36)
    return frame,spec


def test_surface_tf_ft_reversal():
    f=pd.DataFrame(dict(ordering=['AB','BA','AB','BA'],canonical_truth_a=[True,True,False,False],
                        canonical_truth_b=[False,False,True,True]))
    assert surface_cells(f).tolist() == ['TF','FT','FT','TF']
    f['surface_first_truth']=[True,True,False,True]
    with pytest.raises(ValueError,match='surface label'): surface_cells(f)


def test_surface_diagnostic_detects_second_position_under_ba():
    f,spec=fixture()
    f['frozen_probe_score']=np.where(f.surface_second_truth,1.,-1.)
    table=diagnostics(f,spec['bootstrap'])
    def value(basis,ordering,metric):
        return table[(table.labeling == basis)&(table.ordering == ordering)&
                     (table.operator == 'OR')&(table.metric == metric)].estimate.item()
    metric='mixed_first_true_vs_second_true'
    assert value('surface','AB',metric) == value('surface','BA',metric) == 0.
    assert value('canonical','AB','canonical_tf_vs_ft') == 0.
    assert value('canonical','BA','canonical_tf_vs_ft') == 1.
    assert value('canonical','all','canonical_tf_vs_ft') == .5


@pytest.mark.parametrize('failure',['missing','extra','duplicate','blank','null','nonfinite'])
def test_invalid_score_join_is_rejected(failure):
    metadata=pd.DataFrame({'example_id':['a','b']})
    scores=pd.DataFrame({'example_id':['a','b'],'frozen_probe_score':[.1,.2]})
    if failure == 'missing': scores=scores.iloc[:1]
    elif failure == 'extra': scores.loc[2]=['c',.3]
    elif failure == 'duplicate': scores.loc[1,'example_id']='a'
    elif failure == 'blank': scores.loc[1,'example_id']=' '
    elif failure == 'null': scores.loc[1,'example_id']=None
    else: scores.loc[1,'frozen_probe_score']=np.inf
    with pytest.raises(ValueError): exact_join(metadata,scores)


def test_join_preserves_metadata_order_and_scores():
    metadata=pd.DataFrame({'example_id':['b','a']})
    scores=pd.DataFrame({'example_id':['a','b'],'frozen_probe_score':[1.,2.]})
    assert exact_join(metadata,scores).frozen_probe_score.tolist() == [2.,1.]


def test_constituent_labels_and_foreign_keys_are_bound():
    frame=pd.DataFrame(dict(example_id=['e'],fact_a_id=['a'],fact_b_id=['b'],entity_a_id=['A'],entity_b_id=['B'],
        fact_a_statement=['A.'],fact_b_statement=['B.'],topic=['x'],canonical_truth_a=[True],canonical_truth_b=[False]))
    facts=pd.DataFrame(dict(fact_key=['ka','kb'],fact_id=['a','b'],entity_id=['A','B'],statement=['A.','B.'],topic=['x','x'],truth=[True,False]))
    mapping=pd.DataFrame(dict(base_example_id=['e'],fact_a_key=['ka'],fact_b_key=['kb']))
    bind_constituents(frame,facts,mapping)
    bad=mapping.copy();bad.loc[0,'fact_b_key']='ka'
    with pytest.raises(ValueError,match='constituent identity'): bind_constituents(frame,facts,bad)
    bad=facts.copy();bad.loc[1,'truth']=True
    with pytest.raises(ValueError,match='constituent truth'): bind_constituents(frame,bad,mapping)


def test_endpoint_weights_are_products_and_row_order_invariant():
    f,spec=fixture(); options=spec['bootstrap']
    schedule=EntityBootstrap(f,options)
    shuffled=EntityBootstrap(f.sample(frac=1,random_state=7),options)
    assert schedule.sha256 == shuffled.sha256
    pairs=f[['pair_id','topic','entity_a_id','entity_b_id']].drop_duplicates().set_index('pair_id')
    for idx,pair_id in enumerate(schedule.pair_ids):
        pair=pairs.loc[pair_id]; entities=schedule.entities[pair.topic]; counts=schedule.multiplicities[pair.topic]
        np.testing.assert_array_equal(schedule.weights[:,idx],counts[:,entities.index(pair.entity_a_id)]*counts[:,entities.index(pair.entity_b_id)])
    # PCG64 multinomial stream, not a separate condition-specific RNG.
    rng=np.random.Generator(np.random.PCG64(options['seed']))
    for topic in sorted(schedule.entities):
        n=len(schedule.entities[topic])
        np.testing.assert_array_equal(schedule.multiplicities[topic],rng.multinomial(n,np.full(n,1/n),size=options['replicates']))


def test_identical_methods_have_zero_paired_contrast_on_same_draws():
    f,spec=fixture(); rng=np.random.default_rng(6)
    f['frozen_probe_score']=rng.normal(size=len(f))
    definitions=[x for x in spec['metrics'] if x['id'] in ['and_auroc','or_auroc','and_minus_or_auroc']]
    spec['metrics']=definitions
    spec['conditions']=[dict(analysis_group='synthetic',method=m,atomic_layer=0) for m in ['a','b']]
    spec['threshold_applicability']={}
    spec['lr_reference']={'schedule_sha256':EntityBootstrap(f,spec['bootstrap']).sha256}
    spec['paired_contrasts']=[dict(id='a-minus-b',left=['synthetic','a'],right=['synthetic','b'],metrics=['or_auroc'],role='synthetic')]
    scores=pd.concat([f[['example_id','frozen_probe_score']].assign(analysis_group='synthetic',method=m) for m in ['a','b']])
    table,draws,_=compute(scores,f.drop(columns='frozen_probe_score'),spec)
    contrast=table.index[table.category == 'paired_contrast']
    assert len(contrast) == 7
    np.testing.assert_array_equal(draws[:,contrast],0)
    assert table.loc[contrast,'valid_replicates'].eq(40).all()
    for scope,topic in [('pooled','all')]+[('topic',x) for x in spec['benchmark']['topics']]:
        ids=table.metric_id.tolist()
        a,b,delta=[ids.index('synthetic/a/'+scope+'/'+topic+'/'+m) for m in ['and_auroc','or_auroc','and_minus_or_auroc']]
        np.testing.assert_array_equal(draws[:,delta],draws[:,a]-draws[:,b])


def test_invalid_replicates_are_retained_and_paired_intersection_used():
    left=AUC([0,1],[0.,1.],[0,1]); right=AUC([1,2],[0.,1.],[0,1])
    weights=np.array([[1,1,1],[0,1,1],[1,1,0],[1,0,1]])
    paired=left(weights)-right(weights)
    assert np.isfinite(paired).tolist() == [True,False,False,False]
    report=interval(paired,{'minimum_valid':2})
    assert report['total_replicates'] == 4 and report['valid_replicates'] == 1
    assert report['ci_low'] is None and report['ci_status'] == 'insufficient_valid_replicates'


def test_macro_invalid_when_any_topic_undefined():
    f,spec=fixture(); f['frozen_probe_score']=np.where(f.compound_label,1.,-1.)
    from src.clean_transfer_statistics import MetricPlan
    from src.clean_transfer_evaluation import match_and_or
    plan=MetricPlan(f,match_and_or(f),spec)
    weights=np.ones((1,len(f))); weights[:,f.topic == 'cities']=0
    values=plan.evaluate(weights)[0]; ids=[x['metric_id'] for x in plan.records]
    assert np.isnan(values[ids.index('topic_macro/all/or_auroc')])
    assert np.isfinite(values[ids.index('pooled/all/or_auroc')])


def test_safe_file_rejects_traversal_and_links(tmp_path):
    path=tmp_path/'file';path.write_text('fixture')
    (tmp_path/'alias').symlink_to(path)
    for name in ['../file','/file','alias']:
        with pytest.raises(ValueError): safe_file(tmp_path,name)
