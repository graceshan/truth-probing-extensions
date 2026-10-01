"""Successor admission and export-binding regressions; no research-state fitting."""
import copy
import json
from pathlib import Path

import pytest
from src import selection_repair_sensitivity_v3 as s


def fixture_rows():
    row=dict(source_row_id='x:1',dataset='x',row_index='1',source_sha256='source',source_row_sha256='row',
             statement_sha256='statement',entity_id='entity',person_key='person',split='train',partition_role='outer_train',
             label='0',successor_status='admitted',audit_successor_status='admitted')
    mapping={k:v for k,v in row.items() if k!='audit_successor_status'}
    mapping['tensor_row_index']='5'
    return row,mapping


def test_current_admission_controls_export_subset_despite_old_admission():
    row,mapping=fixture_rows()
    banned=dict(row,source_row_id='inventors:23',audit_successor_status='quarantined')
    old=dict(mapping,source_row_id='inventors:23',tensor_row_index='7')
    result=s.bind_mapping([row,banned],[mapping,old],[2,5,7],'train')
    assert len(result)==1 and result[0]['export_row_index']==1
    assert result[0]['tensor_row_index']=='5' and result[0][s.STATUS]=='admitted'
    banned[s.STATUS]='admitted'
    with pytest.raises(ValueError,match='quarantined source'):s.bind_mapping([row,banned],[mapping,old],[2,5,7],'train')


@pytest.mark.parametrize('mutation', ['source_hash','statement_hash','person','missing_export','duplicate_export','missing_row'])
def test_bad_successor_bindings_rejected(mutation):
    row,mapping=fixture_rows();indices=[2,5,7];maps=[mapping]
    if mutation=='source_hash':mapping['source_row_sha256']='wrong'
    if mutation=='statement_hash':mapping['statement_sha256']='wrong'
    if mutation=='person':mapping['person_key']='wrong'
    if mutation=='missing_export':indices=[2,7]
    if mutation=='duplicate_export':indices=[2,5,5]
    if mutation=='missing_row':maps=[]
    with pytest.raises(ValueError):s.bind_mapping([row],maps,indices,'train')


def test_projected_current_eligibility_overrides_historical_registry_value():
    fact=dict(fact_key='key',fact_id='negative',topic='cities',entity_id='e',statement='A is in B.',truth='False',split='validation')
    ledger=dict(fact_ref='negative',origin='locked_registry',split='validation',topic='cities',entity_id='e',
                statement=fact['statement'],label='0_candidate',eligible='True',current_eligible='False',
                support='historical_source_label_negative',person_key='p',evidence_ref='review')
    assert s.current_fact_eligibility([fact],[],[ledger])[0]['current_eligible'] is False
    ledger.update(eligible='False',current_eligible='True')
    assert s.current_fact_eligibility([fact],[],[ledger])[0]['current_eligible'] is True
    del ledger['current_eligible']
    with pytest.raises(ValueError,match='current fact'):s.current_fact_eligibility([fact],[],[ledger])


def test_current_source_quarantine_overrides_historical_status():
    statement='A is in B.'
    fact=dict(fact_key='key',fact_id=s.prior.stable_id('fact',['cities','e',statement,True]),topic='cities',
              entity_id='e',statement=statement,truth='True',split='validation')
    source=dict(source_row_id='cities:2',topic='cities',entity_id='e',statement=statement,split='validation',
                form='affirmative',label='1',person_key='p',successor_status='admitted',audit_successor_status='quarantined')
    assert s.current_fact_eligibility([fact],[source],[])[0]['current_eligible'] is False
    source.update(successor_status='quarantined',audit_successor_status='admitted')
    assert s.current_fact_eligibility([fact],[source],[])[0]['current_eligible'] is True


def test_historical_preparation_rejected_by_v3_boundary(tmp_path,monkeypatch):
    monkeypatch.setattr(s,'verify_adoption',lambda root:({'adoption_sha256':'v3'},{}))
    (tmp_path/'preparation_receipt.json').write_text(json.dumps(dict(adoption_sha256='v2')))
    with pytest.raises(ValueError,match='mixed/stale v3'):s.verify_preparation(tmp_path,tmp_path)


def test_projection_recomputes_changed_sampler_identities():
    root=Path(__file__).resolve().parents[1]
    result=s.sampler_bindings(root)
    assert result['memberships']['corrected_outer_training']['rows']==3044
    for n,rows,count in [('15',700,72),('10',800,68)]:
        sampler=result['sampler'][n]
        assert sampler['rows']==rows and len(sampler['removed'])==len(sampler['added'])==count
    assert result['P_A_frozen'] is False
