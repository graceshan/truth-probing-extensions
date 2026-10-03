"""Focused metadata and synthetic checks only; no research fits, model weights or E."""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import copy
import importlib.util
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src.checkpoint_next_chat import diagnostic_json, preserve_fit
from src.checkpoint_next_full_bank_inputs import memberships, FullTrainingAdapter, EXTRACTION_PRODUCER, supplement_acceptance, ledger_position
from src.checkpoint_next_full_bank_selection import bank_specs, selectors, average_contrast
from src.checkpoint_next_full_bank_audit import independent_auc, endpoint
from src.checkpoint_next_inventory import require
from src.checkpoint_r2_selection_v1 import TOPICS, candidate_id
from src.selection_repair_objectives import ReadoutFit, OptimizerSettings


class SerializationTests(unittest.TestCase):
    def test_nonfinite_multidimensional_array(self):
        value=np.array([[1,np.nan],[np.inf,-np.inf]])
        got=diagnostic_json(value)
        self.assertEqual(got,[[1,{'nonfinite':'NaN'}],[{'nonfinite':'+Inf'},{'nonfinite':'-Inf'}]])
        json.dumps(got,allow_nan=False)

    def test_numpy_scalars_and_zero_dimensional_arrays(self):
        for value,tag in [(np.float64(np.nan),'NaN'),(np.float32(np.inf),'+Inf'),(np.array(-np.inf),'-Inf')]:
            self.assertEqual(diagnostic_json(value),{'nonfinite':tag})
        self.assertEqual(diagnostic_json({'tuple':(np.int64(3),np.float32(2),np.bool_(True))}),{'tuple':[3,2,True]})

    def test_failed_fit_preserves_nonfinite_attempts_and_parameters(self):
        @dataclass
        class Attempt:
            initial_parameters:object
            final_parameters:object
            objective:object
        attempts=(Attempt(np.array([np.nan,np.inf]),np.array([-np.inf,np.nan]),np.float64(np.inf)),)
        fit=SimpleNamespace(weights=np.array([np.nan,np.inf]),intercept=np.float64(-np.inf),status='failed',converged=False,attempts=attempts)
        pre=SimpleNamespace(mean=np.zeros(2),population_std=np.ones(2),denominator=np.ones(2),constant_mask=np.zeros(2,bool))
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'failed'
            with self.assertRaisesRegex(ValueError,'optimizer failure preserved'):preserve_fit(p,fit,pre,{'synthetic':True})
            receipt=json.loads((p/'fit-status.json').read_text());self.assertFalse(receipt['prediction_usable'])
            self.assertEqual(receipt['attempts'][0]['final_parameters'],[{'nonfinite':'-Inf'},{'nonfinite':'NaN'}])
            with np.load(p/'parameters.npz',allow_pickle=False) as params:
                self.assertTrue(np.isnan(params['weights'][0]));self.assertTrue(np.isposinf(params['weights'][1]))
                self.assertTrue(np.isneginf(params['intercept']))

    def test_failed_predictions_remain_blocked(self):
        # Use the original objective class without fitting; guard runs first.
        fit=ReadoutFit(mode='r0',weights=np.zeros(2),intercept=0.,preprocessing=None,settings=OptimizerSettings(),attempts=(),status='failed')
        with self.assertRaises(RuntimeError):fit.decision_function(np.ones((1,2)))

    def test_full_bank_failure_keeps_parameters_and_all600_accounting(self):
        from src.checkpoint_next_full_bank_worker import candidate, accounting, folder
        from src.selection_repair_objectives import fit_p_preprocessing
        @dataclass
        class Attempt:
            initial_parameters:object
            final_parameters:object
            objective:object
        X=np.zeros((3040,2));pre=fit_p_preprocessing(X)
        fit=ReadoutFit(mode='r0',weights=np.array([np.nan,np.inf]),intercept=np.nan,preprocessing=pre,
            settings=OptimizerSettings(),attempts=(Attempt(np.array([0,0,0]),np.array([np.nan,np.inf,-np.inf]),np.float32(np.nan)),),status='failed')
        adapter=SimpleNamespace(train=[{'label':i%2} for i in range(3040)])
        spec=next(s for s in bank_specs() if s['model']=='qwen' and s['saved_layer']==0 and s['method']=='r0')
        identity={'cpu_producer':'synthetic'}
        with tempfile.TemporaryDirectory() as d:
            with patch('src.checkpoint_next_full_bank_worker.objective.fit_readout',return_value=fit),patch('src.checkpoint_next_full_bank_worker.score',side_effect=AssertionError('failed predictions must never run')):
                with self.assertRaisesRegex(ValueError,'affected candidate stopped'):candidate(d,spec,adapter,{'outer_train':X},identity,pre)
            receipt=json.loads((folder(d,spec['fit_id'])/'fit.json').read_text())
            self.assertFalse(receipt['valid_for_scoring']);self.assertNotIn('scores',receipt)
            self.assertEqual(receipt['optimizer']['attempts'][0]['objective'],{'nonfinite':'NaN'})
            counts=accounting(d,identity)['counts'];self.assertEqual(counts,{'valid':0,'failed':1,'incomplete':0,'unrun':599})


class MetadataTests(unittest.TestCase):
    def test_membership_sampler_and_pending_coverage(self):
        train,p15,chosen,ledger,queue=memberships()
        self.assertEqual((len(train),len(p15),len(chosen)),(3040,2778,1000))
        self.assertEqual(queue['unique_texts_per_model'],131)
        for model in ('qwen','llama'):
            self.assertEqual(sum(r['status']=='exact_compatible_fresh_reuse' for r in ledger[model]),2909)
            self.assertEqual(sum(r['status']=='missing_unique_text' for r in ledger[model]),131)

    def test_reviewed_csv_integer_positions(self):
        self.assertEqual(ledger_position('0.0'),0)
        self.assertEqual(ledger_position('35842.0'),35842)
        for bad in ('0.5','nan','inf','-1.0'):
            with self.assertRaisesRegex(ValueError,'noninteger ledger position'):ledger_position(bad)

    def test_same600_grid_with_full_exposure(self):
        specs=bank_specs();self.assertEqual(len(specs),600)
        self.assertEqual(len({s['fit_id'] for s in specs}),600)
        self.assertEqual(sum(s['model']=='qwen' for s in specs),280)
        self.assertEqual(sum(s['model']=='llama' for s in specs),320)
        self.assertEqual(sum(s['fit_rows']==1000 for s in specs),120)
        self.assertTrue(all(s['fit_rows']==(1000 if s['method'] in ('burger_t_g','ttpd') else 3040) for s in specs))
        self.assertTrue(all('/full_train/' in s['fit_id'] and '/P15/' in s['canonical_id'] for s in specs))

    def test_missing_actual_acceptance_blocks_feature_loading(self):
        adapter=FullTrainingAdapter.__new__(FullTrainingAdapter);adapter.supplement=None
        with patch('numpy.load',side_effect=AssertionError('no arrays')):
            with self.assertRaisesRegex(ValueError,'pending actual transferred acceptance'):adapter.layer('qwen',17)
        self.assertEqual(EXTRACTION_PRODUCER,'9b0ac1d84b35d7faf2148d43664569aa867cba52')

    def test_cpu_producer_cannot_relabel_supplement(self):
        from src.checkpoint_next_inventory import sha
        with tempfile.TemporaryDirectory() as d:
            campaign=Path(d)/'transferred';q=campaign/'missing_raw/qwen';q.mkdir(parents=True)
            p=q/'identity.json';p.write_text(json.dumps({'producer_commit':'CPU producer is not extraction producer'}))
            transfer=Path(d)/'transfer.json';transfer.write_text(json.dumps({'missing_raw/qwen/identity.json':sha(p)}))
            with patch('src.checkpoint_next_full_bank_inputs.verify_supplement',side_effect=AssertionError('must reject before features')):
                with self.assertRaisesRegex(ValueError,'extraction identity'):supplement_acceptance(campaign,transfer)


class SelectorTests(unittest.TestCase):
    def records(self):
        result=[]
        for s in bank_specs():
            if s['model']!='qwen':continue
            values=dict(primary=.8,atomic=.99,separation=.5,balanced=.85)
            result.append(dict(spec=s,valid_for_scoring=True,atomic=dict(pooled=.99,topics={t:.99 for t in TOPICS}),
                allocated={str(seed):dict(values) for seed in (11,23,37)},oracle=dict(values)))
        return result

    def test_original_reference_and_atomic_once(self):
        records=self.records();chosen=selectors(records,'qwen',.995,'frozen:reduced_LR')
        self.assertEqual(chosen['reference_atomic'],.995)
        self.assertFalse(chosen['wording_used'])
        self.assertEqual(len({r['S_atom_full'] for r in chosen['procedures'].values()}),1)
        self.assertTrue(all('/full_train/' in r['S_full'] for r in chosen['procedures'].values()))
        # A better full canonical LR cannot shift the common gate.
        canonical=candidate_id('qwen',17,'l2_logistic',1)
        r=next(r for r in records if r['spec']['canonical_id']==canonical)
        r['atomic']=dict(pooled=1.,topics={t:1. for t in TOPICS})
        r['allocated']={str(seed):dict(primary=.8,atomic=1.,separation=.5,balanced=.85) for seed in (11,23,37)}
        next_choice=selectors(records,'qwen',.995,'frozen:reduced_LR')
        self.assertEqual(next_choice['reference_atomic'],.995)
        self.assertEqual(len([r for r in next_choice['eligibility'] if r['eligible']]),280)

    def test_existing_randomized_tie_rules(self):
        result=selectors(self.records(),'qwen',.995,'frozen:reduced_LR')
        ids=sorted(s['canonical_id'] for s in bank_specs() if s['model']=='qwen')
        expected=str(np.random.Generator(np.random.PCG64(20261008)).choice(ids))
        atom=np.random.Generator(np.random.PCG64(20261005)).permutation(ids)[0]
        self.assertEqual(result['compound']['11']['selected_id'],expected)
        self.assertEqual(result['atomic_only']['selected_id'],atom)

    def test_failed_or_missing_candidate_never_disappears(self):
        records=self.records();records[0]['valid_for_scoring']=False
        with self.assertRaisesRegex(ValueError,'failed candidate'):selectors(records,'qwen',.995,'frozen:reduced_LR')
        with self.assertRaisesRegex(ValueError,'complete per-model'):selectors(records[1:],'qwen',.995,'frozen:reduced_LR')

    def test_mean_metrics_inside_draws_and_undefined_propagation(self):
        left={11:np.array([.9,np.nan]),23:np.array([.6,.8]),37:np.array([.3,.7])}
        right={s:np.array([.2,.2]) for s in left}
        result=average_contrast(left,right)
        self.assertAlmostEqual(result[0],.4);self.assertTrue(np.isnan(result[1]))


class AuditTests(unittest.TestCase):
    def test_ties_reweighted_and_zero_mass(self):
        y=[1,1,0,0];s=[1,0,0,-1];weights=np.array([[1,1,1,1],[0,1,2,1],[1,0,0,0]])
        got=independent_auc(y,s,weights)
        self.assertAlmostEqual(got[0],.875);self.assertAlmostEqual(got[1],2/3);self.assertTrue(np.isnan(got[2]))

    def test_fixed_five_macro_missing_topic_is_undefined(self):
        rows=[dict(label=y,topic=t,operator='OR',truth_a=y,truth_b=0) for t in TOPICS[:4] for y in (0,1)]
        self.assertTrue(np.isnan(endpoint(rows,np.array([r['label'] for r in rows]),'OR_mixed_vs_FF_auroc','topic_macro','all')[0]))

    def test_independent_pair_calculation_matches_enumeration(self):
        rng=np.random.default_rng(7);y=np.array([0,1,0,1,1,0]);s=rng.integers(-1,2,size=6);w=rng.integers(0,3,size=(7,6))
        for i,got in enumerate(independent_auc(y,s,w)):
            num=sum(w[i,p]*w[i,n]*(float(s[p]>s[n])+.5*float(s[p]==s[n])) for p in np.flatnonzero(y) for n in np.flatnonzero(1-y))
            den=w[i,y==1].sum()*w[i,y==0].sum()
            self.assertTrue(np.isnan(got)) if not den else self.assertAlmostEqual(got,num/den)


if __name__=='__main__':unittest.main()
