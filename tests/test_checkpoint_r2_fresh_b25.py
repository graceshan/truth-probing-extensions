import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import selection_repair_objectives as objective
from src.checkpoint_r2_b25_adapter import B25Adapter, validate_fold, compound_block
from src.checkpoint_r2_b25_selection import rank, mean_fold_statistics, final_refit
from src.checkpoint_r2_b25_store import FitStore
from src.checkpoint_r2_b25_statistics import summaries, allocated_statistics, evaluate_locked
from src.checkpoint_r2_fresh_recoverability import save
from src.checkpoint_r2_fresh_inputs import ROOT, hash_value
from src.checkpoint_r2_selection_v1 import AtomicCandidate, candidate_id, atomic_eligibility, select_atom_all, average_sample_metrics, TOPICS
from src.checkpoint_r2_fresh_b25 import bank_specs, atomic_id, compact_record
from src.selection_repair_canonical_sensitivity import atomic_weights, compound_weights


class FrozenAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = B25Adapter(json.loads((ROOT / 'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json').read_text()))

    def test_full_declared_bank(self):
        bank = bank_specs()
        self.assertEqual(len({r['candidate_id'] for r in bank}), 600)
        self.assertEqual(sum(r['model'] == 'qwen' for r in bank), 280)
        self.assertEqual(sum(r['planned_rows'] == 700 for r in bank), 120)
        self.assertFalse(any('C_clean' in r['candidate_id'] for r in bank))

    def test_exact_sampler(self):
        a = self.adapter
        members = json.loads((ROOT / 'data/checkpoint_r2_v1/memberships.json').read_text())
        self.assertEqual([a.rows['P15'][i] for i in a.balanced], members['balanced'])
        self.assertEqual(len(a.balanced_rows), 700)
        self.assertTrue(all(r['dataset'] == r['source_row_id'].rsplit(':', 1)[0] for r in a.balanced_rows))

    def test_all_folds_and_exact_facts(self):
        a = self.adapter
        for seed, allocation in a.allocations.items():
            self.assertEqual(len(allocation['isolated']), 100)
            self.assertEqual(len(allocation['people']), 50)
            for fold in allocation['folds']:
                self.assertEqual(len(fold['train']), 320)
                self.assertEqual(len(fold['heldout']), 80)
                validate_fold(a.rows[f'B25_{seed}'], fold['train'], fold['heldout'], a.facts)

    def test_leaked_entity_rejected(self):
        a = self.adapter; fold = a.allocations[11]['folds'][0]
        rows = [dict(r) for r in a.rows['B25_11']]
        rows[fold['train'][0]]['person_a'] = rows[fold['heldout'][0]]['person_a']
        with self.assertRaisesRegex(ValueError, 'entity leakage'):
            validate_fold(rows, fold['train'], fold['heldout'], a.facts)

    def test_leaked_fact_rejected(self):
        a = self.adapter; fold = a.allocations[11]['folds'][0]
        rows = [dict(r) for r in a.rows['B25_11']]
        rows[fold['train'][0]]['fact_a_id'] = rows[fold['heldout'][0]]['fact_a_id']
        with self.assertRaisesRegex(ValueError, 'fact leakage'):
            validate_fold(rows, fold['train'], fold['heldout'], a.facts)

    def test_incomplete_fold_partition_rejected(self):
        a = self.adapter; fold = a.allocations[11]['folds'][0]
        with self.assertRaisesRegex(ValueError, 'partition'):
            validate_fold(a.rows['B25_11'], fold['train'][:-1], fold['heldout'], a.facts)

    def test_wording_bound_to_bare_rows(self):
        a = self.adapter; bare = {r['example_id']: r for r in a.rows['D_bare']}
        for group in a.receipt['heldout_wording_rows']:
            self.assertEqual(len(a.rows[group]), 3864)
            for row in a.rows[group]:
                source = bare[row['example_id']]
                self.assertEqual((row['pair_id'], row['fact_a_id'], row['fact_b_id'], row['label']),
                                 (source['pair_id'], source['fact_a_id'], source['fact_b_id'], source['label']))

    def test_compound_block_duplication_and_half_weight(self):
        rows = self.adapter.rows['B25_11']
        rng = np.random.default_rng(1); X = rng.normal(size=(400, 3)); P = rng.normal(size=(12, 3))
        pre = objective.fit_p_preprocessing(P); p = objective.prepare_block(P, np.arange(12) % 2, pre)
        block = compound_block(X, rows, pre)
        duplicate = compound_block(np.tile(X, (2, 1)), rows + rows, pre)
        self.assertTrue(np.allclose(block.weights, .0025, atol=1e-15))
        beta = np.array([.2, -.1, .4, .3])
        loss, gradient = objective.objective_and_gradient(beta, 'compound', p, block)
        doubled_loss, doubled_gradient = objective.objective_and_gradient(beta, 'compound', p, duplicate)
        self.assertAlmostEqual(loss, doubled_loss, places=14)
        np.testing.assert_allclose(gradient, doubled_gradient, atol=1e-14, rtol=0)
        def bce(b):
            z = b.features @ beta[:-1] + beta[-1]
            return np.dot(b.weights, np.logaddexp(0, np.where(b.targets == 1, -z, z)))
        self.assertAlmostEqual(loss, .5 * bce(p) + .5 * bce(block) + .001 * np.dot(beta[:-1], beta[:-1]), places=14)

    def test_wording_cannot_fit(self):
        a = self.adapter; group = 'or_at_least_one_v1'; pre = objective.fit_p_preprocessing(np.ones((4, 2)))
        with self.assertRaisesRegex(ValueError, 'allocated B25'):
            compound_block(np.ones((3864, 2)), a.rows[group], pre)

    def test_isolated_equal_exposure_and_dedup_key(self):
        a = self.adapter; facts = a.allocations[11]['isolated']
        w = objective.isolated_pair_weights([r['pair_id'] for r in facts], [r['fact_id'] for r in facts])
        np.testing.assert_array_equal(w, np.full(100, .01))
        old = atomic_id(a, 'qwen', 17, 11)
        reverse = SimpleNamespace(allocations={11: dict(isolated=list(reversed(facts)))})
        self.assertEqual(old, atomic_id(reverse, 'qwen', 17, 11))
        self.assertNotEqual(old, atomic_id(a, 'qwen', 18, 11))


class SelectionTests(unittest.TestCase):
    def record(self, name, primary=.8, atomic=.998, separation=2, balanced=.8, valid=True):
        return dict(id=name, primary=primary, atomic=atomic, separation=separation, balanced=balanced, valid_for_scoring=valid)

    def select(self, records, **kwargs):
        return rank(records, reference_atomic=1., fallback_id='reference', arm='test', **kwargs)

    def test_no_AND_gate(self):
        r = self.record('a'); r['AND_auroc'] = .01
        self.assertEqual(self.select([r])['selected_id'], 'a')
        self.assertFalse(self.select([r])['AND_gate'])

    def test_atomic_eligibility_and_invalid_candidates(self):
        records = [self.record('weak', primary=1., atomic=.9949), self.record('invalid', valid=False), self.record('ok')]
        trace = self.select(records)
        self.assertEqual(trace['selected_id'], 'ok'); self.assertEqual(len(trace['eligibility']), 3)

    def test_shared_bank_restriction(self):
        trace = self.select([self.record('a', primary=.9), self.record('b')], eligible_ids=['b'])
        self.assertEqual(trace['selected_id'], 'b')
        self.assertEqual(trace['shared_atomic_eligible_ids'], ['b'])

    def test_shared_atomic_selector_mismatch_rejected(self):
        c = AtomicCandidate(candidate_id('qwen', 17, 'r0'), 'qwen', 17, 'r0', None, True, 1., dict.fromkeys(TOPICS, 1.))
        with self.assertRaisesRegex(ValueError, 'eligibility identity mismatch'):
            select_atom_all([c], model='qwen', reduced_lr_atomic_auc=1., expected_s_all_eligible_ids=[])
        self.assertTrue(atomic_eligibility([c], 'qwen', 1.)[0]['eligible'])

    def test_tie_order_and_tolerance(self):
        records = [self.record('c'), self.record('a'), self.record('b')]
        expected = str(np.random.Generator(np.random.PCG64(20261008)).choice(['a', 'b', 'c']))
        self.assertEqual(self.select(records)['selected_id'], expected)
        self.assertEqual(self.select(list(reversed(records)))['selected_id'], expected)
        self.assertEqual(self.select([self.record('a', primary=.8 + 2e-12), self.record('b', atomic=1.)])['selected_id'], 'a')

    def test_atomic_then_separation_tie_break(self):
        self.assertEqual(self.select([self.record('a'), self.record('b', atomic=1.)])['selected_id'], 'b')
        self.assertEqual(self.select([self.record('a'), self.record('b', separation=3.)])['selected_id'], 'b')

    def test_explicit_empty_and_final_refit_fallback(self):
        trace = self.select([self.record('a', atomic=.9)])
        self.assertEqual(trace['status'], 'fallback_no_eligible_candidate')
        trace = self.select([self.record('a')])
        out = final_refit(trace, fit_id='new', valid=True, atomic=.994, reference_atomic=1.)
        self.assertEqual(out['selected_fit_id'], 'reference')
        self.assertEqual(out['chosen_id'], 'a')
        self.assertEqual(final_refit(trace, fit_id='new', valid=True, atomic=.995, reference_atomic=1.)['selected_fit_id'], 'new')

    def test_degenerate_scores_visible(self):
        trace = self.select([self.record('a', separation=None)])
        self.assertIn('degenerate_standardized_separation', trace['eligibility'][0]['reasons'])

    def test_folds_average_statistics_not_cross_head_scores(self):
        folds = [dict(self.record(str(i), primary=.8 + i * .01), fold=i) for i in range(5)]
        self.assertAlmostEqual(mean_fold_statistics(folds)['primary'], .82)
        shifted = [np.array([0., 1.]) + 100 * i for i in range(5)]
        self.assertEqual(np.mean([roc_auc_score([0, 1], s) for s in shifted]), 1.)
        self.assertLess(roc_auc_score(np.tile([0, 1], 5), np.concatenate(shifted)), 1.)
        folds[0]['primary'] = None
        self.assertIsNone(mean_fold_statistics(folds)['primary'])
        folds[0]['valid_for_scoring'] = False
        with self.assertRaisesRegex(ValueError, 'invalid fold'):
            mean_fold_statistics(folds)

    def test_metric_average_propagates_undefined(self):
        values = {s: np.array([1., np.nan]) for s in (11, 23, 37)}
        actual = average_sample_metrics(values)
        self.assertEqual(actual[0], 1.); self.assertTrue(np.isnan(actual[1]))
        with self.assertRaises(ValueError):
            average_sample_metrics({11: np.ones(2)})


def tiny_adapter():
    rows = {}; features = {}
    atom = [dict(topic=t, label=y, person_key=t + str(p), source_row_id=f'{t}/{p}/{y}') for t in TOPICS for p in range(2) for y in (0, 1)]
    compound = []
    for t in TOPICS:
        for op in ('AND', 'OR'):
            for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)):
                for order in ('AB', 'BA'):
                    compound.append(dict(topic=t, label=int(a and b) if op == 'AND' else int(a or b), truth_a=a, truth_b=b, operator=op, ordering=order,
                        person_a=t + '0', person_b=t + '1', pair_id=t, example_id=f'{t}/{op}/{a}/{b}/{order}'))
    rows.update(P15=atom, atomic_D=atom, D_bare=compound)
    for seed in (11, 23, 37): rows[f'B25_{seed}'] = compound
    for g in ('and_both_following_v1', 'or_explicit_or_both_v1', 'or_at_least_one_v1'):
        rows[g] = [r for r in compound if r['operator'] == ('AND' if g.startswith('and_') else 'OR')]
    bindings = {}
    for g, records in rows.items():
        bindings[g] = [dict(group=g, logical_id=str(i), text_row=i) for i in range(len(records))]
        features[g] = np.array([[int(r['label']), .1 * i] for i, r in enumerate(records)], dtype=float)
    layer = lambda model, saved_layer, wording=False: features
    return SimpleNamespace(rows=rows, bindings=bindings, receipt=dict(heldout_wording_rows={g: len(rows[g]) for g in rows if g.endswith('_v1')}), layer=layer), features


class StoreAndInferenceTests(unittest.TestCase):
    def test_compact_inventory_retains_order_hash_and_fit_status(self):
        record = dict(id='fit', valid_for_scoring=False, failure='explicit', score_bindings={'g': dict(row_binding_sha256='bound', ordered_keys=[['g', 'b'], ['g', 'a']])})
        compact = compact_record(record)
        self.assertEqual(compact['failure'], 'explicit')
        self.assertEqual(compact['score_group_bindings']['g']['ordered_keys_sha256'], hash_value([['g', 'b'], ['g', 'a']]))
        self.assertEqual(compact['score_group_bindings']['g']['rows'], 2)
        self.assertNotIn('score_bindings', compact)

    def test_checkpoint_reuse_alignment_and_corruption(self):
        adapter, X = tiny_adapter()
        with tempfile.TemporaryDirectory() as tmp:
            store = FitStore(tmp, dict(producer_sha='synthetic_only'), adapter)
            name = 'synthetic-head'
            store.publish(name, dict(model='qwen', layer=0, valid_for_scoring=True, scoring_method='difference_of_means'), dict(coef=np.array([1., 0.]), intercept=np.array(0.)), X)
            np.testing.assert_array_equal(store.scores(name)['atomic_D'], X['atomic_D'][:, 0])
            adapter.bindings['atomic_D'] = list(reversed(adapter.bindings['atomic_D']))
            with self.assertRaisesRegex(ValueError, 'row binding hash'):
                store.scores(name)
            (store.folder(name) / 'parameters.npz').write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'parameter corruption'):
                store.load(name)

    def test_complete_synthetic_locked_evaluation_json_and_metrics(self):
        adapter, X = tiny_adapter(); options = dict(seed=1729, replicates=2000, minimum_valid=1800)
        arms = ['S_all', 'R_all', 'S_atom_all', 'original_atomic_selected_R0', 'reduced_LR', 'S_fixed', 'R_fixed', 'LR_only_selection',
                'repair_at_S_all', 'R0_fixed', 'R0_at_R_all', 'atomic_only_fixed', 'atomic_only_at_R_all', 'bank_oracle', 'C_clean_fixed',
                'C_clean_D_selected', 'C_clean_at_S_all', 'C_clean_at_R_all', 'C_clean_at_S_atom_all', 'S_all_equal_boundary_sensitivity', 'R_all_equal_boundary_sensitivity']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); store = FitStore(root, dict(producer_sha='synthetic_only'), adapter)
            procedures = {}
            for model in ('qwen', 'llama'):
                name = model + '/tiny'
                store.publish(name, dict(model=model, layer=0, valid_for_scoring=True, scoring_method='difference_of_means'), dict(coef=np.array([1., 0.]), intercept=np.array(0.)), X)
                procedures[model] = {str(seed): dict.fromkeys(arms, name) for seed in (11, 23, 37)}
            locked = dict(procedures=procedures); save(root / 'locked.json', locked)
            old = root / 'old'; old.mkdir()
            weights = dict(atomic_D=atomic_weights(adapter.rows['atomic_D'], np.ones(20, bool), options))
            weights['D_bare'], _ = compound_weights(pd.DataFrame(adapter.rows['D_bare']).rename(columns={'person_a': 'entity_a_id', 'person_b': 'entity_b_id'}), np.ones(80, bool), options)
            np.savez(old / 'bootstrap_weights.npz', **weights)
            support = {k: {t: dict(all_degree_four=False, entities=2) for t in TOPICS} for k in ('D_graph', 'E_graph')}
            result = evaluate_locked(adapter, store, root, dict(producer_sha='synthetic_only'), locked, SimpleNamespace(root=old), options, support)
            self.assertEqual(result['unique_evaluated_heads'], 2)
            self.assertEqual(len(json.loads((root / 'metrics.json').read_text())['records']), 2 * 21 * 4 * 77)
            contrasts = json.loads((root / 'contrasts.json').read_text())['records']
            self.assertTrue(all(r['point'] == 0. for r in contrasts))
            self.assertFalse(json.loads((root / 'sufficiency.json').read_text())['records'][0]['joint_control_sufficiency_claim'])


if __name__ == '__main__':
    unittest.main()
