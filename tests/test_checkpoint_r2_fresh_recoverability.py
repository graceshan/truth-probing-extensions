"""Focused synthetic tests; real fitting remains a separate committed CPU run."""
import copy
import itertools
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np

from src.checkpoint_r2_fresh_adapter import FreshAdapter, bind_rows, check_packet, clean_block, require_fit_group
from src.checkpoint_r2_fresh_inputs import hash_value, text_hash
from src.checkpoint_r2_fresh_recoverability import choose_clean, evaluate, point_metrics
from src.checkpoint_r2_selection_v1 import AtomicCandidate, candidate_id, select_original_atomic_r0
from src.clean_transfer_statistics import interval, weighted_auc
from src.selection_repair_canonical_sensitivity import atomic_weights, compound_weights
from src import selection_repair_objectives as o


class Bindings(unittest.TestCase):
    def setUp(self):
        self.sources=[dict(source_row_id='s'+str(i),statement='same statement',label=str(i%2),split='train') for i in range(2)]
        tid='text_'+text_hash('same statement')
        bindings=[dict(group='P15',logical_id=r['source_row_id'],statement=r['statement'],label=r['label'],split='train',
                       text_id=tid,inventory_offset=i,source_metadata_sha256=hash_value(r)) for i,r in enumerate(self.sources)]
        self.manifest=dict(bindings=bindings,rows=[dict(id=tid,bindings=bindings)])

    def test_distinct_observations_share_text_without_dedup(self):
        records=bind_rows('P15',self.sources,self.manifest)
        self.assertEqual([r['text_row'] for r in records],[0,0])
        self.assertEqual([r['logical_id'] for r in records],['s0','s1'])

    def test_duplicated_logical_observations_rejected(self):
        with self.assertRaisesRegex(ValueError,'duplicate logical observation'):
            bind_rows('P15',[self.sources[0],self.sources[0]],self.manifest)

    def test_duplicate_manifest_bindings_rejected(self):
        bad=copy.deepcopy(self.manifest);bad['bindings'].append(bad['bindings'][0])
        with self.assertRaisesRegex(ValueError,'duplicate logical binding'):bind_rows('P15',self.sources,bad)

    def test_source_hash_and_wrong_text_row_rejected(self):
        bad=copy.deepcopy(self.manifest);bad['bindings'][0]['source_metadata_sha256']='bad'
        with self.assertRaisesRegex(ValueError,'source metadata binding'):bind_rows('P15',self.sources,bad)
        bad=copy.deepcopy(self.manifest);bad['rows'][0]['bindings']=[]
        with self.assertRaisesRegex(ValueError,'wrong text row'):bind_rows('P15',self.sources,bad)

    def test_packet_order_and_hash_rejected(self):
        records=bind_rows('P15',self.sources,self.manifest)
        packet=dict(row_binding_sha256=hash_value(records),ordered_keys=[['P15','s0'],['P15','s1']],scores=np.array([.2,.3]))
        np.testing.assert_array_equal(check_packet(records,packet),packet['scores'])
        packet['ordered_keys'].reverse()
        with self.assertRaisesRegex(ValueError,'order'):check_packet(records,packet)
        packet['ordered_keys'].reverse();packet['row_binding_sha256']='bad'
        with self.assertRaisesRegex(ValueError,'hash'):check_packet(records,packet)

    def test_training_evaluation_separation(self):
        for group in ('atomic_D','D_bare','or_at_least_one_v1'):
            with self.assertRaisesRegex(ValueError,'forbidden'):require_fit_group(group)
        bad=copy.deepcopy(self.manifest);bad['bindings'][0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'split'):bind_rows('P15',self.sources,bad)

    def test_streamed_layer_preserves_repeated_observations(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path=root/'qwen/shards/00000/activations.npy';path.parent.mkdir(parents=True)
            values=np.arange(8,dtype=np.float16).reshape(2,2,2);np.save(path,values)
            adapter=FreshAdapter.__new__(FreshAdapter);adapter.output=root
            adapter.cfg={'models':{'qwen':dict(layers=2,width=2)}};adapter.manifest=dict(rows=[{},{}])
            adapter.indices={'qwen':dict(shards=[dict(path='00000',rows=2)])}
            adapter.bindings={'P15':[dict(text_row=0),dict(text_row=0),dict(text_row=1)]}
            st=path.stat();adapter.file_stats={path:(st.st_size,st.st_mtime_ns,st.st_ctime_ns)}
            result=adapter.layer('qwen',1)['P15']
            np.testing.assert_array_equal(result,values[[0,0,1],1,:]);self.assertEqual(result.dtype,np.float64)


class ObjectiveAdapter(unittest.TestCase):
    def setUp(self):
        self.rows=[]
        for pair in ('p0','p1'):
            for op,a,b,order in itertools.product(('AND','OR'),(0,1),(0,1),('AB','BA')):
                self.rows.append(dict(group='TC_control_and_final_refit',split='train',pair_id=pair,operator=op,
                                      truth_a=a,truth_b=b,ordering=order,label=int((a and b) if op=='AND' else (a or b))))
        rng=np.random.default_rng(1)
        self.p=np.column_stack((rng.normal(size=(10,2)),np.full(10,.1)))
        self.x=np.column_stack((rng.normal(size=(32,2)),np.full(32,999.)))
        self.pre=o.fit_p_preprocessing(self.p)

    def test_C_clean_exact_mean_BCE_and_gradient(self):
        block=clean_block(self.x,self.rows,self.pre);theta=np.array([.2,-.3,.7,.4])
        loss,grad=o.objective_and_gradient(theta,'r0',block)
        z=block.features@theta[:-1]+theta[-1]
        expected=np.logaddexp(0,np.where(block.targets==1,-z,z)).mean()+.001*np.dot(theta[:-1],theta[:-1])
        self.assertAlmostEqual(loss,expected,places=13)
        for i in range(len(theta)):
            d=np.zeros(len(theta));d[i]=1e-6
            finite=(o.objective_and_gradient(theta+d,'r0',block)[0]-o.objective_and_gradient(theta-d,'r0',block)[0])/2e-6
            self.assertAlmostEqual(grad[i],finite,places=7)

    def test_C_clean_duplication_invariance_and_P_only_statistics(self):
        before=self.pre.mean.copy();block=clean_block(self.x,self.rows,self.pre)
        duplicate=clean_block(np.repeat(self.x,2,axis=0),[r for row in self.rows for r in (row,row)],self.pre)
        theta=np.array([.2,-.3,.7,.4])
        a=o.objective_and_gradient(theta,'r0',block);b=o.objective_and_gradient(theta,'r0',duplicate)
        self.assertAlmostEqual(a[0],b[0],places=13);np.testing.assert_allclose(a[1],b[1],atol=1e-13)
        np.testing.assert_array_equal(self.pre.mean,before)
        self.assertTrue(self.pre.constant_mask[2]);self.assertTrue((block.features[:,2]==0).all())

    def test_C_clean_excludes_D_and_requires_complete_pairs(self):
        bad=copy.deepcopy(self.rows);bad[0]['group']='D_bare'
        with self.assertRaisesRegex(ValueError,'training block'):clean_block(self.x,bad,self.pre)
        with self.assertRaisesRegex(ValueError,'both operators'):clean_block(self.x[:-1],self.rows[:-1],self.pre)

    def test_training_order_rejected(self):
        sources=[dict(source_row_id='s'+str(i),statement='same statement',label='1',split='train') for i in range(2)]
        tid='text_'+text_hash('same statement')
        bindings=[dict(group='P15',logical_id=r['source_row_id'],statement=r['statement'],label='1',split='train',text_id=tid,
                       inventory_offset=i,source_metadata_sha256=hash_value(r)) for i,r in enumerate(sources)]
        manifest=dict(bindings=bindings,rows=[dict(id=tid,bindings=bindings)])
        with self.assertRaisesRegex(ValueError,'source order'):bind_rows('P15',sources[::-1],manifest)

    def test_reviewed_solver_with_clean_adapter(self):
        head=o.fit_readout('r0',clean_block(self.x,self.rows,self.pre),self.pre)
        self.assertTrue(head.converged);self.assertIs(head.preprocessing,self.pre)
        self.assertLessEqual(head.attempts[-1].gradient_infinity_norm,1e-4)


class SelectionAndUncertainty(unittest.TestCase):
    def test_full_paired_evaluation_integration_and_undefined_macro(self):
        topics=('animal_class','cities','element_symb','inventors','sp_en_trans')
        atoms=[];comp=[]
        for topic in topics:
            for person,label in itertools.product((topic+'a',topic+'b'),(0,1)):
                atoms.append(dict(topic=topic,person_key=person,label=label))
            for op,a,b,order in itertools.product(('AND','OR'),(0,1),(0,1),('AB','BA')):
                comp.append(dict(topic=topic,pair_id=topic,person_a=topic+'a',person_b=topic+'b',operator=op,
                                 truth_a=a,truth_b=b,ordering=order,label=int(a and b) if op=='AND' else int(a or b)))
        rows=dict(P15=atoms,atomic_D=atoms,D_bare=comp)
        bindings={g:[dict(group=g,logical_id=str(i)) for i in range(len(r))] for g,r in rows.items()}
        adapter=SimpleNamespace(rows=rows,bindings=bindings)
        cfg=dict(models={m:dict(layers=[0],canonical_layer=0) for m in ('qwen','llama')},
                 bootstrap=dict(seed=1729,replicates=2000,minimum_valid=1800),C_clean_selector=dict(tie_tolerance=1e-12,tie_seed=20261008))
        identity=dict(producer_sha='synthetic-test',code_hashes={'synthetic':'fixture'})
        scores={g:np.array([r['label'] for r in rlist],float) for g,rlist in rows.items()}
        fits=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for model,family in itertools.product(('qwen','llama'),('R0','C_clean','reduced_LR')):
                name=f'{model}_{family}_L00';folder=root/'fits'/name;folder.mkdir(parents=True)
                np.savez(folder/'scores.npz',**scores)
                fits.append(dict(id=name,model=model,family=family,saved_layer=0,valid_for_scoring=True,separation=1.,
                    metrics=point_metrics(adapter,scores),score_bindings={g:dict(row_binding_sha256=hash_value(bindings[g]),
                    ordered_keys=[[r['group'],r['logical_id']] for r in bindings[g]]) for g in rows}))
            with patch('src.checkpoint_r2_fresh_recoverability.plot_curves'):
                result=evaluate(adapter,cfg,root,fits,identity)
            self.assertEqual(result['metric_records'],6*4*7)
            import json
            metrics=json.loads((root/'metrics.json').read_text())['records']
            macro=next(r for r in metrics if r['scope']=='topic_macro' and r['metric']=='AND_auroc')
            self.assertEqual(macro['ci_status'],'insufficient_valid_replicates')
            self.assertGreater(macro['invalid_replicates'],0)
            contrasts=json.loads((root/'contrasts.json').read_text())['records']
            self.assertTrue(any(r.get('comparison')=='same_layer' for r in contrasts))
            self.assertTrue(any(r.get('optimistic_development') for r in contrasts))

    def test_R0_atomic_only_exact_tie_lower_layer(self):
        cs=[AtomicCandidate(candidate_id('qwen',l,'r0'),'qwen',l,'r0',None,True,.9,{}) for l in (8,3)]
        self.assertEqual(select_original_atomic_r0(cs,model='qwen')['selected_id'],candidate_id('qwen',3,'r0'))
        cs[0]=AtomicCandidate(candidate_id('qwen',8,'r0'),'qwen',8,'r0',None,True,.9000000000001,{})
        self.assertEqual(select_original_atomic_r0(cs,model='qwen')['selected_id'],candidate_id('qwen',8,'r0'))

    def test_clean_boundary_then_atomic_then_separation(self):
        records=[dict(id='a',valid_for_scoring=True,OR_mixed_vs_FF_auroc=.8,atomic_auroc=.95,separation=.2),
                 dict(id='b',valid_for_scoring=True,OR_mixed_vs_FF_auroc=.8,atomic_auroc=.95,separation=.3),
                 dict(id='c',valid_for_scoring=True,OR_mixed_vs_FF_auroc=.79,atomic_auroc=1.,separation=9.)]
        cfg=dict(tie_tolerance=1e-12,tie_seed=20261008)
        self.assertEqual(choose_clean(records,cfg)['selected_id'],'b')
        records[0]['valid_for_scoring']=False
        with self.assertRaisesRegex(ValueError,'failed'):choose_clean(records,cfg)

    def test_undefined_and_valid_draw_rules(self):
        self.assertTrue(np.isnan(weighted_auc([1,1],[1.,2.])))
        result=interval(np.r_[np.zeros(1799),np.full(201,np.nan)],dict(minimum_valid=1800))
        self.assertEqual(result['valid_replicates'],1799);self.assertIsNone(result['ci_low'])
        result=interval(np.r_[np.zeros(1800),np.full(200,np.nan)],dict(minimum_valid=1800))
        self.assertEqual(result['ci_status'],'ok')

    def test_person_and_endpoint_product_draws_shared(self):
        import pandas as pd
        atoms=[];pairs=[]
        for topic in ('a','b','c','d','e'):
            atoms += [dict(topic=topic,person_key=topic+'0'),dict(topic=topic,person_key=topic+'1')]
            pairs.append(dict(topic=topic,pair_id=topic,entity_a_id=topic+'0',entity_b_id=topic+'1'))
        opts=dict(seed=1729,replicates=2000)
        aw=atomic_weights(atoms,np.ones(len(atoms),bool),opts)
        cw,_=compound_weights(pd.DataFrame(pairs),np.ones(len(pairs),bool),opts)
        for i in range(5):np.testing.assert_array_equal(cw[:,i],aw[:,2*i]*aw[:,2*i+1])


if __name__=='__main__':unittest.main()
