"""Synthetic/metadata regressions. No E representations, inference or fitting."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.final_e_inventory import Metadata, adapt_compound, truth, ROOT, read
from src.final_e_access import draft, validate_gate, authorize
from src.final_e_registry import apply_head
from src.final_e_reporting import schedules, scalar_result, constituent_result
from src.checkpoint_r2_fresh_store import pin, write_json
from src.checkpoint_r2_fresh_inputs import hash_value
from src.clean_transfer_statistics import EntityBootstrap, interval
from src.checkpoint_r2_selection_v1 import average_sample_metrics, TOPICS
from src.checkpoint_r2_full_raw import recover_budget, cap_reason

OPTIONS=dict(replicates=20,seed=1729,minimum_valid=18)


class InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.m=Metadata()

    def test_exact_current_inventory(self):
        r=self.m.receipt
        self.assertEqual((r['compound_entities'],r['compound_pairs'],r['group_rows']['bare'],r['group_rows']['atomic_E']),(225,450,7200,974))
        self.assertEqual((r['logical_bindings'],r['unique_texts'],r['deduplication_savings']),(19424,19190,234))
        self.assertEqual((r['atomic_people'],r['compound_people'],r['shared_people']),(294,225,225))

    def test_duplicate_texts_keep_logical_identities(self):
        counts={}
        for b in self.m.bindings:counts.setdefault(b['text_id'],[]).append(b)
        duplicates=[v for v in counts.values() if len(v)>1]
        self.assertTrue(duplicates)
        for bs in duplicates:
            self.assertEqual(len(bs),len({b['binding_id'] for b in bs}))
        self.assertEqual(sum(len(bs)-1 for bs in duplicates),234)

    def test_surface_truth_and_order_rejected(self):
        row=dict(self.m.sources['bare_compounds.csv'][0]);pair=next(p for p in self.m.pairs if p['pair_id']==row['pair_id'])
        adapt_compound(row,pair)
        for k,v in [('surface_first_truth','True' if row['surface_first_truth']=='False' else 'False'),
                    ('compound_label','True' if row['compound_label']=='False' else 'False'),('ordering','invalid'),
                    ('surface_first_entity_id','wrong')]:
            with self.subTest(field=k),self.assertRaises(ValueError):adapt_compound(dict(row,**{k:v}),pair)

    def test_no_bool_string_coercion(self):
        self.assertEqual([truth(x) for x in ('True','False','0','1')],[1,0,0,1])
        with self.assertRaises(ValueError):truth('false')

    def test_E_schedule_shared_and_deterministic(self):
        a,ra=schedules(self.m,OPTIONS);b,rb=schedules(self.m,OPTIONS)
        self.assertEqual(ra,rb)
        self.assertTrue(np.array_equal(a['atomic_E'],b['atomic_E']))
        for g in self.m.rows:
            if g not in ('atomic_E','bare'):self.assertIs(a[g],a['bare'])
        self.assertEqual(a['bare'].shape,(20,7200));self.assertEqual(a['atomic_E'].shape,(20,974))

    def test_reordered_wording_rejected(self):
        other=SimpleNamespace(rows=dict(self.m.rows),receipt=self.m.receipt)
        other.rows['or_at_least_one_v1']=self.m.rows['or_at_least_one_v1'][::-1]
        with self.assertRaises(ValueError):schedules(other,OPTIONS)


class StatisticsTests(unittest.TestCase):
    def test_shared_entity_multiplicity_products(self):
        frame=pd.DataFrame([dict(pair_id='ab',topic='x',entity_a_id='a',entity_b_id='b'),
                            dict(pair_id='ac',topic='x',entity_a_id='a',entity_b_id='c'),
                            dict(pair_id='ab',topic='x',entity_a_id='a',entity_b_id='b')])
        s=EntityBootstrap(frame,OPTIONS);m=s.multiplicities['x'];entities=s.entities['x'];i={v:j for j,v in enumerate(entities)}
        self.assertTrue(np.array_equal(s.weights[:,0],m[:,i['a']]*m[:,i['b']]))
        self.assertTrue(np.array_equal(s.weights[:,1],m[:,i['a']]*m[:,i['c']]))
        self.assertTrue(np.array_equal(s.row_weights(0,20)[:,0],s.row_weights(0,20)[:,2]))

    def test_undefined_draws_not_redrawn(self):
        d=np.array([np.nan]*3+[.5]*17);r=interval(d,OPTIONS)
        self.assertEqual((r['valid_replicates'],r['invalid_replicates'],r['ci_low']),(17,3,None))

    def test_no_surviving_topic_macro(self):
        rows=[dict(label=y,topic=TOPICS[0]) for y in (0,1)]
        result=scalar_result({'atomic_fixture':rows},{'atomic_fixture':np.array([0.,1.])})
        self.assertIsNone(next(r['point'] for r in result['records'] if r['scope']=='topic_macro'))

    def test_metric_mean_not_score_mean_and_nan_propagates(self):
        actual=average_sample_metrics({11:np.array([1.,np.nan]),23:np.array([.5,.8]),37:np.array([.25,.6])})
        self.assertAlmostEqual(actual[0],7/12);self.assertTrue(np.isnan(actual[1]))

    def test_ties_true_and_constituent_targets(self):
        rows=[dict(label=a&b,truth_a=a,truth_b=b,operator='AND',ordering=order,topic=topic)
              for topic in TOPICS for order in ('AB','BA') for a in (0,1) for b in (0,1)]
        result=constituent_result(rows,np.zeros((len(rows),2)),'AND',None,OPTIONS)
        get=lambda metric:next(r['point'] for r in result['records'] if r['metric']==metric and r['scope']=='pooled' and r['surface_order']=='all')
        self.assertEqual(get('first_accuracy'),.5);self.assertEqual(get('joint_correctness'),.25)
        self.assertEqual(get('boolean_composition_accuracy'),.25);self.assertEqual(get('external_composition_auroc'),.5)

    def test_missing_score_coverage_rejected(self):
        with self.assertRaises(ValueError):scalar_result({'atomic_fixture':[dict(label=0,topic='cities')]},{})
        with self.assertRaises(ValueError):scalar_result({'atomic_fixture':[dict(label=0,topic='cities')]},{'atomic_fixture':np.array([])})


class FrozenHeadTests(unittest.TestCase):
    def fixture(self,root):
        width=3584;p=root/'params.npz'
        np.savez(p,coef=np.ones(width),intercept=0.)
        fit=dict(id='synthetic',producer_sha='2'*40,model='qwen',layer=2,scoring_method='l2_logistic',valid_for_scoring=True,parameters=pin(p))
        write_json(root/'fit.json',fit)
        ref=lambda name:dict(source='fixture',path=name,**pin(root/name))
        return dict(frozen_id='synthetic',parameter_producer='2'*40,model='qwen',saved_layer=2,method='l2_logistic',family='scalar',fit=ref('fit.json'),parameters=ref('params.npz'))

    def test_saved_head_no_fitting_and_tamper_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);h=self.fixture(p);roots={'fixture':td};X=np.ones((2,3584))
            self.assertTrue(np.array_equal(apply_head(h,X,roots),np.array([3584.,3584.])))
            for k,v in [('method','r0'),('saved_layer',3),('model','llama'),('parameter_producer','9'*40)]:
                with self.subTest(field=k),self.assertRaises(ValueError):apply_head(dict(h,**{k:v}),X,roots)
            with (p/'params.npz').open('ab') as f:f.write(b'altered')
            with self.assertRaises(ValueError):apply_head(h,X,roots)

    def test_altered_preprocessing_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td);h=self.fixture(p);width=3584
            np.savez(p/'pre.npz',mean=np.zeros(width),denominator=np.ones(width),constant_mask=np.zeros(width,bool))
            np.savez(p/'head.npz',weights=np.ones(width),intercept=0.)
            write_json(p/'pre.json',{'fit_group':'P15'})
            fit=read(p/'fit.json');fit.update(parameters=pin(p/'head.npz'),preprocessing_artifact=pin(p/'pre.npz'),valid=True)
            (p/'fit.json').unlink();write_json(p/'fit.json',fit)
            ref=lambda n:dict(source='fixture',path=n,**pin(p/n))
            h.update(family='constituent',method='constituent',fit=ref('fit.json'),parameters=ref('head.npz'),preprocessing=ref('pre.npz'),preprocessing_metadata=ref('pre.json'))
            self.assertEqual(apply_head(h,np.ones((1,width)),{'fixture':td})[0],width)
            np.savez(p/'other.npz',mean=np.ones(width),denominator=np.ones(width),constant_mask=np.zeros(width,bool))
            h['preprocessing']=ref('other.npz')
            with self.assertRaises(ValueError):apply_head(h,np.ones((1,width)),{'fixture':td})


class ClosedGateTests(unittest.TestCase):
    def test_no_authorization_blocks_before_output_IO(self):
        args=SimpleNamespace(lock=None,lock_sha256=None,authorization=None,authorization_sha256=None)
        with patch('src.final_e_access.file_hash',side_effect=AssertionError('must not open output')),self.assertRaisesRegex(ValueError,'E CLOSED'):
            authorize(args)

    def test_CLI_closed_for_every_E_operation(self):
        import subprocess,sys
        for phase in ('extract','token_audit','accept','verify','evaluate','audit','transfer','verify-transfer'):
            p=subprocess.run([sys.executable,'-m','src.final_e_runner',phase,'--output','/nonexistent/E-must-not-open'],cwd=ROOT,capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertIn('E CLOSED',p.stderr)

    def test_draft_cannot_execute(self):
        with self.assertRaisesRegex(ValueError,'E CLOSED'):
            validate_gate(draft(),{},lock_sha256='0'*64,authorization_sha256='1'*64,phase='extract',producer='2'*40,output='/tmp/e',model='qwen')

    def final_fixture(self):
        d=draft();d.update(status='FINAL',scientific_lock_approved=True,claims_finalized=True,producer_sha='2'*40,
            approval=dict(reviewer='synthetic reviewer',instruction='synthetic approval',reviewed_delivery_sha='3'*40),
            resolutions={k:'synthetic reviewed resolution' for k in d['pending']},pending={},limits=dict(model_seconds=7200,total_seconds=18000,setup_phase_seconds=1800))
        a=dict(schema='final-e-phase-authorization-v1',approved=True,phase='extract',model='qwen',producer_sha='2'*40,lock_sha256='0'*64,
            output=str(Path('/tmp/e').resolve()),instruction='synthetic only',reviewer='synthetic',campaign_id='synthetic')
        return d,a

    def test_hash_scope_producer_and_unresolved_decisions(self):
        d,a=self.final_fixture();kw=dict(lock_sha256='0'*64,authorization_sha256='1'*64,phase='extract',producer='2'*40,output='/tmp/e',model='qwen')
        validate_gate(d,a,**kw)
        for k,v in [('phase','evaluate'),('producer','9'*40),('model','llama'),('lock_sha256','9'*64),('output','/tmp/other')]:
            with self.subTest(field=k),self.assertRaises(ValueError):validate_gate(d,a,**dict(kw,**{k:v}))
        d['pending']={'claim':None}
        with self.assertRaises(ValueError):validate_gate(d,a,**kw)

    def test_unfinished_attempt_is_charged_not_reset(self):
        p=dict(total_seconds=30.,model_seconds={'qwen':20.,'llama':0.},running=True,model='qwen',as_of_unix=100.)
        r=recover_budget(p,110.)
        self.assertEqual((r['total_seconds'],r['model_seconds']['qwen']),(40.,30.))
        self.assertEqual(cap_reason(r,dict(model_seconds=30,total_seconds=100,setup_phase_seconds=1800)),'full_model_cap_qwen')


class RawInterfaceTests(unittest.TestCase):
    def test_synthetic_publication_loader_and_missing_coverage(self):
        from src import final_e_raw as raw
        from src.checkpoint_r2_fresh_store import write_shard
        from src.checkpoint_r2_full_raw import verify_completed
        rows=[dict(id='synthetic_text',statement='fixture',statement_sha256=__import__('hashlib').sha256(b'fixture').hexdigest(),bindings=[dict(logical_id='a'),dict(logical_id='b')])]
        meta=SimpleNamespace(receipt={'synthetic':True},bindings=rows[0]['bindings'])
        access=dict(producer_sha='2'*40,lock_sha256='3'*64,campaign_id='synthetic')
        spec=dict(layers=2,width=3)
        tok=dict(id=rows[0]['id'],statement_sha256=rows[0]['statement_sha256'],token_ids=[1,2],unpadded_token_ids=[1,2],raw_token_ids=[2],
            unpadded_special_tokens_mask=[1,0],attention_mask=[1,1],position_ids=[0,1],readout_index=1,semantic_readout_position=1,
            pad_token_id=0,add_special_tokens=True,truncation=False,chat_template=False)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);mr=root/'qwen';mr.mkdir();write_json(mr/'runtime-origin.json',{'synthetic':True})
            identity=raw.campaign_identity(access,meta);write_json(root/'campaign.json',identity)
            prov=raw.provenance(identity,'qwen',mr/'runtime-origin.json');p=mr/'shards/00000'
            write_shard(p,rows,np.zeros((1,2,3),np.float32),spec,prov,[tok]);write_json(mr/'token-plan.json',[tok])
            rec=verify_completed(p,rows,spec,prov);write_json(mr/'index.json',dict(identity=identity,model='qwen',shards=[rec],unique_texts=1,logical_bindings=2))
            with patch.object(raw,'Metadata',return_value=meta),patch.object(raw,'input_rows',return_value=rows),patch.object(raw,'raw_contract',return_value={'models':{'qwen':spec}}),patch.object(raw,'verify_origin',return_value={}):
                result=raw.verify(root,'qwen',access)
                self.assertEqual((result['unique_texts'],result['logical_bindings']),(1,2))
                with self.assertRaises(FileNotFoundError):raw.verify(root,'qwen',access,require_acceptance=True)
                (mr/'shards/extra').mkdir()
                with self.assertRaises(ValueError):raw.verify(root,'qwen',access)

    def test_recovery_preserves_unpublished_and_rejects_corrupt_receipt(self):
        from src.checkpoint_r2_full_raw import recover_shard
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'shards/00000';p.mkdir(parents=True);(p/'partial').write_bytes(b'preserve')
            self.assertIsNone(recover_shard(p,[],{}, {},root/'quarantine'))
            self.assertEqual(next((root/'quarantine').glob('*/partial')).read_bytes(),b'preserve')
            p.mkdir();write_json(p/'receipt.json',{'schema':'corrupt'})
            with self.assertRaises(ValueError):recover_shard(p,[],{},{},root/'quarantine')
            self.assertTrue((p/'receipt.json').exists())

    def test_nonfinite_features_not_publishable(self):
        from src.checkpoint_r2_fresh_store import write_shard
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError):write_shard(Path(td)/'shard',[{'id':'synthetic'}],np.full((1,2,3),np.nan,np.float32),{'layers':2,'width':3},{},[])

    def test_raw_backend_stays_separate_from_chat(self):
        import inspect
        from src import final_e_raw
        source=inspect.getsource(final_e_raw.worker)
        self.assertIn('FreshBackend',source)
        self.assertNotIn('ChatBackend',source)
        self.assertNotIn('generate(',source)

    def test_carry_forward_exact_receipts(self):
        c=read(ROOT/'config/final_e/carryforward_v1.json')
        for p,h in c['source_files'].items():self.assertEqual(pin(ROOT/p)['sha256'],h)
        a=read(ROOT/'results/final_e_preparation_20261003/accounting-evidence/adoption.json')
        e=read(ROOT/'results/final_e_preparation_20261003/accounting-evidence/llama-end.json')
        self.assertEqual(c['charged_seconds']['qwen'],a['original']['carry_seconds']['qwen'])
        self.assertEqual(c['charged_seconds']['llama'],a['original']['carry_seconds']['llama']+e['active_seconds'])
        for m in ('qwen','llama'):self.assertEqual(c['remaining_seconds'][m],7200-c['charged_seconds'][m])



if __name__=='__main__':unittest.main()
