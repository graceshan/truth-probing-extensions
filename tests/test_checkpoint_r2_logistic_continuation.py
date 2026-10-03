import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from src import checkpoint_r2_logistic_continuation as numerical
from src.checkpoint_r2_b25_store import FitStore, read
from src.checkpoint_r2_b25_successor_store import SuccessorStore, InvalidFit, ORIGINAL_SHA
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_fresh_inputs import hash_value


class LogisticTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(71)
        self.X = rng.normal(size=(80, 4))
        self.y = np.tile([0, 1], 40)
        self.theta = np.array([.2, -.1, .3, .05, .7])

    def first(self, theta=None, iterations=31, success=True):
        return dict(phase='original_saved_attempt', n_iter=iterations, library_success=success,
                    **numerical.diagnostics(self.theta if theta is None else theta, self.X, self.y, 1))

    def test_analytic_gradient_intercept_and_C_n_scaling(self):
        for C in (.001, .1, 1, 10, np.inf):
            loss, gradient = numerical.loss_gradient(self.theta, self.X, self.y, C)
            eps = 1e-6
            finite = np.array([(numerical.loss_gradient(self.theta+eps*np.eye(5)[j], self.X, self.y, C)[0] -
                                numerical.loss_gradient(self.theta-eps*np.eye(5)[j], self.X, self.y, C)[0])/(2*eps) for j in range(5)])
            np.testing.assert_allclose(gradient, finite, atol=1e-8, rtol=1e-7)
            self.assertAlmostEqual(gradient[-1], (expit(self.X@self.theta[:-1]+self.theta[-1])-self.y).mean())
            unpenalized, gu = numerical.loss_gradient(self.theta, self.X, self.y, np.inf)
            self.assertAlmostEqual(loss-unpenalized, self.theta[:-1]@self.theta[:-1]/(2*C*80))
            np.testing.assert_allclose(gradient-gu, np.r_[self.theta[:-1]/(C*80), 0], atol=1e-15)

    def test_unregularized_special_case(self):
        loss, grad = numerical.loss_gradient(self.theta, self.X, self.y, np.inf)
        z = self.X@self.theta[:-1]+self.theta[-1]
        self.assertEqual(loss, np.logaddexp(0, np.where(self.y==1, -z, z)).mean())
        np.testing.assert_array_equal(grad, np.r_[self.X.T@(expit(z)-self.y)/80, (expit(z)-self.y).mean()])

    def test_first_attempt_parameters_and_zero_initialization_unchanged(self):
        from sklearn.linear_model import LogisticRegression
        for C in (1, np.inf):
            theta, receipt = numerical.first_attempt(self.X, self.y, C)
            old = LogisticRegression(C=1 if np.isinf(C) else C, penalty=None if np.isinf(C) else 'l2',
                                     solver='lbfgs', fit_intercept=True, tol=1e-4, max_iter=100 if np.isinf(C) else 2000)
            old.fit(self.X, self.y)
            np.testing.assert_array_equal(theta, np.r_[old.coef_[0], old.intercept_[0]])
            self.assertEqual(receipt['n_iter'], old.n_iter_[0])
            self.assertIsInstance(receipt['termination_message'], str)

    def test_library_success_gradient_failure_triggers_warm_budget(self):
        theta = self.theta.copy(); captured = {}
        def minimize(fun, x, **kw):
            captured.update(x=x.copy(), options=kw['options'], jac=kw['jac'], dtype=x.dtype)
            return SimpleNamespace(x=x, success=True, status=0, message='library success, gradient still bad', nit=13)
        with patch.object(numerical.optimize, 'minimize', side_effect=minimize):
            params, receipt = numerical.complete(self.X, self.y, 1, theta, [self.first(iterations=1234)])
        np.testing.assert_array_equal(captured['x'], theta)
        self.assertEqual(captured['dtype'], np.float64)
        self.assertEqual(captured['options'], dict(numerical.SETTINGS, maxiter=8766))
        self.assertTrue(captured['jac']);self.assertEqual(receipt['total_iterations'], 1247)
        self.assertFalse(receipt['valid_for_scoring']);self.assertEqual(len(receipt['attempts']), 2)

    def test_already_valid_parameters_unchanged(self):
        params, receipt = numerical.logistic(self.X, self.y, 1)
        self.assertTrue(receipt['valid_for_scoring'])
        theta = np.r_[params['coef'], params['intercept']]
        with patch.object(numerical.optimize, 'minimize', side_effect=AssertionError('unexpected refit')):
            again, details = numerical.complete(self.X, self.y, 1, theta, receipt['attempts'])
        np.testing.assert_array_equal(params['coef'], again['coef'])
        self.assertEqual(params['intercept'], again['intercept'])
        self.assertEqual(len(details['attempts']), len(receipt['attempts']))

    def test_real_synthetic_warm_continuation_passes(self):
        _, receipt = numerical.complete(self.X, self.y, 1, self.theta, [self.first()])
        self.assertTrue(receipt['valid_for_scoring'])
        self.assertEqual(len(receipt['attempts']), 2)
        self.assertLessEqual(receipt['attempts'][-1]['gradient_infinity_norm'], 1e-4)

    def test_spent_prior_attempts_not_reset_and_no_second_continuation(self):
        attempts = [self.first(iterations=100), dict(self.first(iterations=1200), phase='prior_cold_attempt')]
        with patch.object(numerical.optimize, 'minimize', return_value=SimpleNamespace(x=self.theta, success=False, status=1, message='failed', nit=8700)) as call:
            params, receipt = numerical.complete(self.X, self.y, 1, self.theta, attempts)
        self.assertEqual(call.call_args.kwargs['options']['maxiter'], 8700)
        self.assertEqual(receipt['total_iterations'], 10000)
        with patch.object(numerical.optimize, 'minimize', side_effect=AssertionError('second continuation')):
            _, again = numerical.complete(self.X, self.y, 1, self.theta, receipt['attempts'])
        self.assertFalse(again['valid_for_scoring']);self.assertEqual(len(again['attempts']), 3)

    def test_exhausted_or_excessive_budget(self):
        with patch.object(numerical.optimize, 'minimize', side_effect=AssertionError('budget exceeded')):
            _, receipt = numerical.complete(self.X, self.y, 1, self.theta, [self.first(iterations=10000)])
        self.assertFalse(receipt['valid_for_scoring']);self.assertEqual(receipt['total_iterations'], 10000)
        with self.assertRaisesRegex(ValueError, 'cumulative budget'):
            numerical.complete(self.X, self.y, 1, self.theta, [self.first(iterations=10001)])

    def test_success_is_required_even_with_small_gradient(self):
        params, good = numerical.logistic(self.X, self.y, 1)
        theta = np.r_[params['coef'],params['intercept']]
        with patch.object(numerical.optimize, 'minimize', return_value=SimpleNamespace(x=theta, success=False, status=2, message='abnormal line search', nit=1)):
            _, receipt = numerical.complete(self.X, self.y, 1, theta, [dict(good['attempts'][-1], phase='original_saved_attempt', library_success=False)])
        self.assertFalse(receipt['valid_for_scoring'])

    def test_hard_function_bound(self):
        def consume(fun,x,**kw):
            for _ in range(50001):fun(x)
            self.fail('evaluation guard failed')
        with patch.object(numerical.optimize, 'minimize', side_effect=consume):
            _, receipt = numerical.complete(self.X, self.y, 1, self.theta, [self.first()])
        self.assertFalse(receipt['valid_for_scoring'])
        self.assertEqual(receipt['attempts'][-1]['function_evaluations'], 50000)
        self.assertEqual(receipt['total_iterations'], 10000)

    def test_TTPD_composition_and_features_unchanged(self):
        rows = [dict(form='affirmative' if i%2 else 'negated', dataset=('topic' if i%2 else 'neg_topic')+str(i%5)) for i in range(80)]
        from src.selection_repair_capacity_methods_v1 import fit as original_fit, score
        old, detail = original_fit('ttpd', self.X, self.y, rows, None, 0)
        fresh, receipt = numerical.fit('ttpd', self.X, self.y, rows, None, 0)
        self.assertTrue(detail['valid_for_scoring']);self.assertTrue(receipt['valid_for_scoring'])
        for key in old:np.testing.assert_array_equal(old[key],fresh[key])
        np.testing.assert_array_equal(score('ttpd',old,self.X),score('ttpd',fresh,self.X))

    def test_failed_TTPD_polarity_cannot_feed_truth_subfit(self):
        rows = [dict(form='affirmative' if i%2 else 'negated', dataset=('topic' if i%2 else 'neg_topic')+str(i%5)) for i in range(80)]
        with patch.object(numerical, 'logistic', return_value=({'coef':np.ones(4),'intercept':np.array(0.)}, {'valid_for_scoring':False})) as call:
            _, receipt = numerical.fit('ttpd', self.X, self.y, rows, None, 0)
        self.assertEqual(call.call_count, 1);self.assertFalse(receipt['valid_for_scoring'])
        self.assertIsNone(receipt['head_optimizer'])


class CheckpointTests(unittest.TestCase):
    def fixture(self, root):
        rows = [dict(label=i%2, source_row_id=str(i)) for i in range(4)]
        adapter = SimpleNamespace(rows={'P15':rows}, bindings={'P15':[]})
        store = object.__new__(SuccessorStore)
        FitStore.__init__(store, root, dict(producer_sha='successor',configuration_sha256='configuration'),adapter)
        store.continued = {}
        reusable = SimpleNamespace(execution={'config':{'models':{'qwen':{'canonical_layer':17,'C':1}}}})
        spec = dict(candidate_id='qwen/P15/L01/l2_logistic/C=1',model='qwen',saved_layer=1,method='l2_logistic',C=1)
        return store,reusable,spec

    def test_persistent_bank_failure_blocks_prediction_and_stage(self):
        with tempfile.TemporaryDirectory() as root:
            store,reusable,spec = self.fixture(root)
            with patch.object(numerical,'fit',return_value=({'coef':np.zeros(2),'intercept':np.array(0.)},{'valid_for_scoring':False})):
                with self.assertRaisesRegex(InvalidFit,'persistent bank failure'):
                    store.bank(spec,{'P15':np.ones((4,2))},None,reusable)
            self.assertFalse((store.folder(spec['candidate_id'])/'scores.npz').exists())
            with self.assertRaisesRegex(ValueError,'invalid completed fit'):store.scores(spec['candidate_id'])
            with self.assertRaisesRegex(InvalidFit,'persistent bank failure'):
                store.bank(spec,{'P15':np.ones((4,2))},None,reusable)

    def test_stale_identity_and_parameter_corruption_fatal(self):
        with tempfile.TemporaryDirectory() as root:
            store,_,spec = self.fixture(root);folder=store.folder(spec['candidate_id']);folder.mkdir(parents=True)
            np.savez(folder/'parameters.npz',coef=np.ones(2))
            record=dict(store.identity,id=spec['candidate_id'],valid_for_scoring=False,parameters=pin(folder/'parameters.npz'))
            (folder/'fit.json').write_text(json.dumps(record))
            store.load(spec['candidate_id'])
            (folder/'parameters.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'parameter corruption'):store.load(spec['candidate_id'])
            record['producer_sha']='wrong';(folder/'fit.json').write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError,'producer/config/input'):store.load(spec['candidate_id'])

    def test_import_preserves_parameter_producer_and_source_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root);old=root/'original';gate=root/'gate';new=root/'new';old.mkdir();gate.mkdir();new.mkdir()
            identity=dict(producer_sha='new',configuration_sha256='newconfig',inventory_sha256='inventory',representation_manifest_sha256='representation',recoverability_producer_sha='recover',recoverability_completion_sha256='recoverhash')
            bindings={'P15':[dict(group='P15',logical_id='x')]};adapter=SimpleNamespace(bindings=bindings)
            for i in range(68):
                folder=old/'fits'/str(i);folder.mkdir(parents=True)
                np.savez(folder/'parameters.npz',coef=np.array([i]),intercept=np.array(0.))
                np.savez(folder/'scores.npz',P15=np.array([float(i)]))
                record=dict(identity,id=str(i),producer_sha=ORIGINAL_SHA,configuration_sha256='oldconfig',valid_for_scoring=True,parameters=pin(folder/'parameters.npz'),scores=pin(folder/'scores.npz'),score_bindings={'P15':dict(row_binding_sha256=hash_value(bindings['P15']),ordered_keys=[['P15','x']])},seconds=1.)
                if i==0:record['parameter_producer_sha']='recover'
                (folder/'fit.json').write_text(json.dumps(record))
            (gate/'completion.json').write_text(json.dumps({'records':[{'candidate_id':str(i),'valid_for_scoring':True} for i in range(4)]}))
            store=SuccessorStore(new,identity,adapter,old,gate);imports=store.import_valid()
            self.assertEqual(len(imports),68);self.assertEqual(store.load('0')['parameter_producer_sha'],'recover')
            self.assertEqual(store.load('1')['parameter_producer_sha'],ORIGINAL_SHA)
            self.assertEqual(store.load('1')['imported_source_fit'],pin(old/'fits/1/fit.json'))
            self.assertEqual(store.load('1')['seconds'],0.)
            self.assertEqual(store.import_valid(),imports)
            (old/'fits/2/scores.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'corrupt original'):store.import_valid()


if __name__ == '__main__':
    unittest.main()
