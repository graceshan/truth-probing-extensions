"""Nested membership, immutable policy and paired-alignment checks; no real fits."""
import copy,json
from pathlib import Path
import numpy as np
import pytest
from src import selection_repair_capacity_p10_v2 as p
ROOT=Path(__file__).resolve().parents[1]


def test_exact_p10_membership_and_nonadditive_sampler():
    rows,sampled,report=p.membership(ROOT)
    assert len(rows)==2864 and len(sampled)==800
    assert report['P15_subset_P10'] and report['A10_nested']
    assert report['all_A10_source_variants_excluded']
    overlap=report['sample_comparison_to_P15']
    assert overlap['shared']+len(overlap['removed'])==700
    assert overlap['shared']+len(overlap['added'])==800
    assert overlap['P15_sample_subset_P10'] is False and overlap['removed']
    assert sum(int(r['label']) for r in rows)==1432
    assert all(c['rows']==160 for c in report['sample_counts'][1:])


def test_reordered_p10_sampler_rejected(monkeypatch):
    original=p.s.projection.prior.balanced
    monkeypatch.setattr(p.s.projection.prior,'balanced',lambda rows:list(reversed(original(rows))) if len(rows)==2864 else original(rows))
    with pytest.raises(ValueError,match='sampler identity'):p.membership(ROOT)


def test_count_preserving_p10_identity_change_rejected(monkeypatch):
    original=p.s.parse_rows
    def changed(data):
        rows=original(data)
        if len(rows)==2864:rows[0]=dict(rows[0],source_row_id='inventors:155')
        return rows
    monkeypatch.setattr(p.s,'parse_rows',changed)
    with pytest.raises(ValueError,match='every A10 source variant'):p.membership(ROOT)


def test_only_36_new_configs_prescribed():
    config=p.verify_config(ROOT);grid=p.configurations(config)
    assert len(grid)==len({tuple(x.values()) for x in grid})==36
    assert {x['cohort'] for x in grid}=={'P10'}
    assert config['correction_commit']=='fd2d7be3b910c3fdba7ff6b2c2891eb0ab1b7992'
    assert not config['P_A_frozen']


@pytest.mark.parametrize('mutation',['C','layer','method','bootstrap','threshold','correction','count','frozen'])
def test_changed_policy_rejected(tmp_path,mutation):
    config=json.loads((ROOT/p.CONFIG).read_text())
    for pin in config['bindings'].values():
        dest=tmp_path/pin['path'];dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/pin['path']).read_bytes())
    if mutation=='C':config['models']['qwen']['C']=10
    if mutation=='layer':config['models']['llama']['layers'][0]=11
    if mutation=='method':config['methods'].pop()
    if mutation=='bootstrap':config['bootstrap']['seed']=1
    if mutation=='threshold':config['review_drop_thresholds']['atomic_auroc']=.05
    if mutation=='correction':config['correction_commit']='bb8bc51a864a5702f06ef23f923e11af97710c09'
    if mutation=='count':config['additional_configurations']=72
    if mutation=='frozen':config['P_A_frozen']=True
    dest=tmp_path/p.CONFIG;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(config))
    with pytest.raises(ValueError):p.verify_config(tmp_path)


def test_draw_order_and_values_must_match_exactly():
    a=np.array([.2,.3,np.nan,.5]);p.require_draw_alignment(a,a.copy())
    for b in (a[::-1],a+1e-14,np.array([.2,.3,.4,.5])):
        with pytest.raises(ValueError,match='paired draws'):p.require_draw_alignment(a,b)


@pytest.mark.parametrize('new,trajectory,status',[(-.03,'improved','cleared'),(-.06,'improved','persistent'),(-.1,'persistent','persistent'),(-.12,'worsened','persistent'),(None,'not_computed','not_computed')])
def test_threshold_persistence_separate_from_change(new,trajectory,status):
    assert p.classify_trigger(-.1,new,.05)==dict(trajectory=trajectory,threshold_status=status)
