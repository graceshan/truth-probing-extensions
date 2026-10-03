import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.special import expit

from src import selection_repair_objectives as objective
from src.checkpoint_r2_constituent_adapter import exclusion_gate,FIT,VALID,FULL,ConstituentAdapter,check_production_packet
from src.checkpoint_r2_constituent_metrics import summaries,paired_contrast,transfer_status,definitions
from src.checkpoint_r2_fresh_constituents import predict_pair,fitting_spec,initial_inventory
from src.checkpoint_r2_b25_adapter import resolve
from src.checkpoint_r2_common_v1 import FrozenBindings
from src.checkpoint_r2_selection_v1 import TOPICS,ENDPOINTS,surface_labels,select_constituent_layer
from src.checkpoint_r2_fresh_inputs import ROOT,hash_value,text_hash


def fixture():
    frozen=FrozenBindings(ROOT)
    return frozen,pd.DataFrame(frozen.block(VALID,'AND')['rows'])


class ConstituentTests(unittest.TestCase):
    def test_surface_reversal(self):
        rows=pd.DataFrame(dict(canonical_truth_a=[1,1],canonical_truth_b=[0,0],ordering=['AB','BA']))
        first,second=surface_labels(rows)
        np.testing.assert_array_equal(first,[1,0]);np.testing.assert_array_equal(second,[0,1])
        rows['surface_first_truth']=[1,1]
        with self.assertRaises(ValueError):surface_labels(rows)

    def test_frozen_coverage_exclusions(self):
        import json
        frozen,rows=fixture();groups={g:[r for r in frozen._rows if r['group']==g] for g in (FIT,VALID,FULL)}
        p=json.loads((ROOT/'data/checkpoint_r2_v1/memberships.json').read_text())['P15']
        entities=json.loads((ROOT/'data/checkpoint_r2_v1/entities.json').read_text())
        rec=exclusion_gate(groups,p,entities);self.assertEqual(len(rec['validation_people']),20)
        bad=p+[dict(p[0],person_key=rec['validation_people'][0])]
        with self.assertRaises(ValueError):exclusion_gate(groups,bad,entities)
        altered=copy.deepcopy(groups);altered[FIT][0]['person_a']=rec['validation_people'][0]
        with self.assertRaises(ValueError):exclusion_gate(altered,p,entities)

    def test_logical_duplicate_texts_retained(self):
        sources=[dict(example_id='a',statement='same'),dict(example_id='b',statement='same')]
        rows=[dict(id='text_'+text_hash('same'))]
        bindings=[dict(group=FIT,logical_id=r['example_id'],source_metadata_sha256=hash_value(r),statement='same',text_id=rows[0]['id'],inventory_offset=i) for i,r in enumerate(sources)]
        result=resolve(FIT,sources,dict(rows=rows,bindings=bindings),'example_id')
        self.assertEqual(len(result),2);self.assertEqual(result[0]['text_row'],result[1]['text_row'])
        with self.assertRaises(ValueError):resolve(FIT,sources[::-1],dict(rows=rows,bindings=bindings),'example_id')

    def test_mean_constituent_objective_gradient(self):
        rng=np.random.default_rng(123);P=rng.normal(size=(13,5));X=rng.normal(size=(8,5));y=np.array([0,1]*4)
        pre=objective.fit_p_preprocessing(P);block=objective.prepare_block(X,y,pre);par=rng.normal(size=6)
        loss,grad=objective.objective_and_gradient(par,'r0',block)
        z=pre.transform(X)@par[:-1]+par[-1]
        expected=np.logaddexp(0,np.where(y==1,-z,z)).mean()+.001*np.dot(par[:-1],par[:-1])
        self.assertAlmostEqual(loss,expected,14)
        numerical=[]
        for i in range(6):
            delta=np.zeros(6);delta[i]=1e-6
            numerical.append((objective.objective_and_gradient(par+delta,'r0',block)[0]-objective.objective_and_gradient(par-delta,'r0',block)[0])/2e-6)
        np.testing.assert_allclose(grad,numerical,atol=1e-8,rtol=1e-7)
        double=objective.prepare_block(np.repeat(X,2,axis=0),np.repeat(y,2),pre)
        self.assertAlmostEqual(objective.objective_and_gradient(par,'r0',double)[0],loss,14)
        self.assertAlmostEqual(grad[-1],np.mean(expit(z)-y),14)

    def test_source_only_selector(self):
        _,rows=fixture();first,second=surface_labels(rows);good=np.column_stack((first,second)).astype(float)
        sel=select_constituent_layer(source_operator='AND',role='T_C_source_validation',validation_binding='hash',rows=rows,scores_by_layer={4:good,3:good,2:-good})
        self.assertEqual(sel['selected_layer'],3)
        for role in ('D_evaluation','full_T_C_refit','T_C_source_fit'):
            with self.assertRaises(ValueError):select_constituent_layer(source_operator='AND',role=role,validation_binding='hash',rows=rows,scores_by_layer={0:good})
        rows2=rows.copy();rows2.operator='OR'
        with self.assertRaises(ValueError):select_constituent_layer(source_operator='AND',role='T_C_source_validation',validation_binding='hash',rows=rows2,scores_by_layer={0:good})

    def test_missing_endpoint_and_invalid_fit(self):
        _,rows=fixture();first,second=surface_labels(rows);good=np.column_stack((first,second)).astype(float)
        sel=select_constituent_layer(source_operator='AND',role='T_C_source_validation',validation_binding='hash',rows=rows,scores_by_layer={0:np.full_like(good,np.nan),1:good})
        self.assertEqual(sel['selected_layer'],1);self.assertEqual(sel['curves'][0]['status'],'invalid_scores')
        reduced=rows[rows.topic!=TOPICS[0]]
        sel=select_constituent_layer(source_operator='AND',role='T_C_source_validation',validation_binding='hash',rows=reduced,scores_by_layer={0:good[rows.topic!=TOPICS[0]]})
        self.assertEqual(sel['status'],'no_valid_layer')
        with self.assertRaises(ValueError):predict_pair(Path('/nonexistent'),[dict(valid=False),dict(valid=True)],None,None)

    def test_fixed_tie_and_composition(self):
        _,frame=fixture();rows=frame.to_dict('records');scores=np.zeros((len(rows),2))
        records,_=summaries(rows,scores,'AND')
        pooled={r['metric']:r['point'] for r in records if r['scope']=='pooled' and r['surface_order']=='all'}
        self.assertEqual(pooled['first_accuracy'],.5);self.assertEqual(pooled['joint_correctness'],.25)
        self.assertEqual(pooled['boolean_composition_accuracy'],.25)
        or_rows=[dict(r,operator='OR',label=int(r['truth_a'] or r['truth_b'])) for r in rows]
        records,_=summaries(or_rows,scores,'OR')
        self.assertEqual(next(r['point'] for r in records if r['metric']=='boolean_composition_accuracy' and r['scope']=='pooled' and r['surface_order']=='all'),.75)

    def test_paired_schedule_and_undefined(self):
        _,frame=fixture();rows=frame.to_dict('records');f,s=surface_labels(frame);scores=np.column_stack([f,s])
        opts=dict(replicates=20,minimum_valid=18);weights=np.ones((20,len(rows)));weights[0]=0
        rec,draw=summaries(rows,scores,'AND',weights,opts,orders=False)
        contrast=paired_contrast(rec,rec,draw,draw,opts)
        self.assertTrue(all(r['valid_replicates']==19 and r['point']==0 and r['ci_low']==r['ci_high']==0 for r in contrast))
        with self.assertRaises(ValueError):paired_contrast(rec,list(reversed(rec)),draw,draw,opts)
        self.assertEqual(transfer_status(dict(ci_low=.06,ci_high=.07)),'material_deficit_lower_bound_above_05')
        self.assertEqual(transfer_status(dict(ci_low=.01,ci_high=.05)),'unresolved_crosses_05')

    def test_synthetic_guard_preserved(self):
        frozen,frame=fixture();block=frozen.block(VALID,'AND')
        packet=dict(metadata_sha256=block['metadata_sha256'],ordered_keys=block['ordered_keys'],score_kind='fresh_production_constituent_v1',scores=np.zeros((len(frame),2)))
        with self.assertRaises(ValueError):frozen.align(block,packet)

    def test_production_packet_rejects_reordering_and_representation(self):
        class Stub:
            cfg=dict(models=dict(qwen=dict(layers=28)))
            output=Path('/unused')
            def block(self,g,o):return dict(rows=[{},{}],bindings=[dict(group=g,logical_id='a'),dict(group=g,logical_id='b')])
        a=Stub();block=a.block(VALID,'AND')
        packet=dict(score_kind='fresh_production_constituent_v1',model='qwen',saved_layer=2,group=VALID,operator='AND',representation_index_sha256='representation',row_binding_sha256=hash_value(block['bindings']),ordered_keys=[[VALID,'a'],[VALID,'b']])
        with patch('src.checkpoint_r2_constituent_adapter.file_hash',return_value='representation'):
            check_production_packet(a,packet,np.zeros((2,2)))
            bad=dict(packet,ordered_keys=packet['ordered_keys'][::-1])
            with self.assertRaises(ValueError):check_production_packet(a,bad,np.zeros((2,2)))
            with self.assertRaises(ValueError):check_production_packet(a,dict(packet,representation_index_sha256='other'),np.zeros((2,2)))
            with self.assertRaises(ValueError):check_production_packet(a,dict(packet,score_kind='synthetic_validation'),np.zeros((2,2)))
            with self.assertRaises(ValueError):check_production_packet(a,packet,np.full((2,2),np.nan))

    def test_independent_weighted_rank_rebuild(self):
        from scripts.checkpoint_r2_audit_constituents import independent_auc_draws
        from sklearn.metrics import roc_auc_score
        y=np.array([0,1,1,0,0,1]);s=np.array([0.,0.,1.,2.,3.,3.])
        weights=np.array([[1,1,1,1,1,1],[2,1,0,3,1,2],[0,0,0,0,0,0]])
        result=independent_auc_draws(y,s,weights)
        for i in range(2):self.assertAlmostEqual(result[i],roc_auc_score(y,s,sample_weight=weights[i]),14)
        self.assertTrue(np.isnan(result[2]))

    def test_no_validation_fit_spec(self):
        adapter=object.__new__(ConstituentAdapter)
        with self.assertRaises(ValueError):fitting_spec(adapter,'qwen',0,'AND','FIRST',VALID)

    def test_240_inventory(self):
        class Stub:
            cfg=dict(models=dict(qwen=dict(layers=28),llama=dict(layers=32)))
            def block(self,g,o):return dict(rows=[dict(group=g,operator=o)],bindings=[dict(group=g,logical_id=o)])
        result=initial_inventory(Stub());self.assertEqual(len(result),240);self.assertEqual(len({r['id'] for r in result}),240)


if __name__=='__main__':unittest.main()
