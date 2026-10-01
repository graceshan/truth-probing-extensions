"""Fact-only regressions for frozen traversal, evidence and policy propagation."""
import copy
import csv
import io
import json
import unittest

from src import tc_topup_audit as a


def decode(value):
    return list(csv.DictReader(io.StringIO(value.decode())))


class TCTopupTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs, cls.summary = a.build()
        cls.reviews = a.load(a.ROOT, a.PACKAGE / 'reviews.json')
        cls.historical = a.load(a.ROOT, a.old.PACKAGE / 'reviews.json')
        cls.completed = a.load(a.ROOT, a.prior.PACKAGE / 'TC_completed_pairs.json')
        cls.queue = a.read_csv(a.ROOT, a.PACKAGE / 'traversal_queue.csv')
        cls.rows = a.read_csv(a.ROOT, a.prior.OVERLAY / 'row_manifest.csv')
        cls.current = decode(cls.outputs[a.OVERLAY / 'row_manifest.csv'])
        cls.actions, cls.banned, _ = a.restrictions(cls.rows, cls.historical + cls.reviews)

    def test_deterministic_regeneration(self):
        for p, value in self.outputs.items():
            self.assertEqual((a.ROOT / p).read_bytes(), value, str(p))

    def test_capacity_and_all_attempts_preserved(self):
        self.assertEqual((self.summary['new_attempts'], self.summary['new_completed_pairs'],
                          self.summary['new_unresolved'], self.summary['new_rejected']), (27, 24, 3, 0))
        self.assertEqual(self.summary['cumulative_attempts'], 111)
        for row in self.summary['capacity']:
            self.assertEqual(row['usable_entities'], 20)
            self.assertEqual(row['possible_unordered_pairs'], 190)
            self.assertEqual(row['remaining_successful_topups'], 0)
        self.assertEqual(sum(r['cumulative_unresolved'] for r in self.summary['capacity']), 11)
        combined = json.loads(self.outputs[a.PACKAGE / 'combined_TC_completed_pairs.json'])
        self.assertEqual(combined[:76], self.completed)
        ev = json.loads(self.outputs[a.PACKAGE / 'combined_evidence.json'])
        original = a.load(a.ROOT, a.old.PACKAGE / 'evidence.json')
        self.assertTrue(all(ev[k] == v for k, v in original.items()))

    def test_traversal_rejects_reopen_skip_and_overrun(self):
        a.validate_traversal(self.reviews, self.queue, self.completed, self.historical)
        for kind in ('reopen', 'skip', 'overrun'):
            reviews = copy.deepcopy(self.reviews)
            if kind == 'reopen':
                reviews.append(next(r for r in self.historical if r['status'] == 'unresolved'))
            elif kind == 'skip':
                reviews = [r for r in reviews if not (r['topic'] == 'inventors' and r['rank'] == 24)]
            else:
                r = copy.deepcopy(reviews[0]); r['rank'] = 21; reviews.append(r)
            reviews.sort(key=lambda r: (r['topic'], r['rank']))
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                a.validate_traversal(reviews, self.queue, self.completed, self.historical)

    def test_identity_evidence_hash_and_queue_mutations_fail(self):
        order = a.load(a.ROOT, a.old.PACKAGE / 'candidate_order.json')
        proposals = a.load(a.ROOT, a.old.PACKAGE / 'negative_proposals.json')
        for kind in ('identity', 'hash', 'evidence', 'replacement'):
            reviews = copy.deepcopy(self.reviews)
            evidence = a.load(a.ROOT, a.PACKAGE / 'evidence.json')
            if kind == 'identity':
                reviews[0]['person_key'] = 'different_person'
            elif kind == 'hash':
                reviews[0]['facts'][0]['statement_sha256'] = '0' * 64
            elif kind == 'evidence':
                evidence[reviews[0]['facts'][0]['evidence_ids'][0]]['fact_bindings'] = []
            else:
                f = reviews[0]['facts'][0]; f['statement'] = 'Different affirmative.'
                f['statement_sha256'] = a.digest(f['statement'].encode())
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                a.old.validate_pairs(reviews, order, self.rows, evidence, proposals, a.ROOT)

    def test_exact_pair_actions_preserve_labels_and_other_claims(self):
        self.assertEqual(set(self.actions), {'inventors:155', 'neg_inventors:155',
                                            'inventors:380', 'neg_inventors:380'})
        for before, after in zip(self.rows, self.current):
            self.assertTrue(all(after[k] == v for k, v in before.items()))
            self.assertEqual(after[a.STATUS], 'quarantined' if before['source_row_id'] in self.actions
                             else before[a.prior.STATUS])
        # Giffard's supported France pair survives; no whole-person exclusion.
        for sid in ('inventors:386', 'neg_inventors:386'):
            self.assertEqual(next(r[a.STATUS] for r in self.current if r['source_row_id'] == sid), 'admitted')
        self.assertEqual(self.summary['confirmed_new_label_errors'], 0)
        self.assertTrue(self.summary['correction_base_must_advance_before_fitting'])

    def test_new_unresolved_positive_and_negative_both_propagate(self):
        for label in (0, 1):
            review = copy.deepcopy(self.reviews[0])
            fact = next(f for f in review['facts'] if f['label'] == label)
            fact['judgment'] = 'unresolved'; review['status'] = 'unresolved'
            actions, banned, _ = a.restrictions(self.rows, [review])
            source = next(r for r in self.rows if r['source_row_id'] == fact['source_row_id'])
            for sid in (source['source_row_id'], source['paired_source_row_id']):
                self.assertIn(sid, actions)
                survivor = next(r for r in self.current if r['source_row_id'] == sid)
                for name in self.summary['current_membership_totals']:
                    with self.subTest(label=label, row=sid, membership=name), self.assertRaisesRegex(ValueError, 'survives'):
                        a.guard_memberships({name: [survivor]}, self.current, actions, banned)

    def test_generated_duplicate_cannot_bypass_source_pair_or_registry(self):
        review = copy.deepcopy(next(r for r in self.reviews if r['topic'] == 'inventors' and r['rank'] == 24))
        fact = review['facts'][1]
        fact.pop('source_row_id'); fact.update(origin='recovered_negative_pipeline', fact_id='alternate_registry_ref')
        actions, banned, _ = a.restrictions(self.rows, [review])
        self.assertIn('inventors:155', actions)
        self.assertIn('neg_inventors:155', actions)
        inventory = decode(self.outputs[a.OVERLAY / 'fact_inventory.csv'])
        matches = [r for r in inventory if a.signature(r) ==
                   (review['topic'], review['person_key'], fact['statement_sha256'])]
        self.assertTrue(any(r['origin'] != 'source' for r in matches))
        for r in matches:
            self.assertEqual(r[a.ELIGIBLE], 'False')
            with self.assertRaisesRegex(ValueError, 'eligibility'):
                a.guard_eligibility([{**r, a.ELIGIBLE: 'True'}], self.current, banned)

    def test_confirmed_conflict_uses_paired_exclusion_without_relabel(self):
        review = copy.deepcopy(self.reviews[0]); review['facts'][1]['judgment'] = 'supported_true'
        actions, _, _ = a.restrictions(self.rows, [review])
        self.assertEqual(len(actions), 2)
        self.assertTrue(all(action['status'] == 'excluded' for action in actions.values()))

    def test_current_memberships_counts_and_balanced_changes(self):
        self.assertEqual(self.summary['current_membership_totals'], {
            'corrected_outer_training': 3040, 'corrected_outer_training_balanced': 1000,
            'P15_admitted_membership': 2778, 'P15_balanced_exposure': 700,
            'P10_admitted_membership': 2864, 'P10_balanced_exposure': 800})
        for name in self.summary['current_membership_totals']:
            rows = decode(self.outputs[a.PROJECTION / (name + '.csv')])
            a.guard_memberships({name: rows}, self.current, self.actions, self.banned)
        for n in (15, 10):
            self.assertEqual(self.outputs[a.PROJECTION / f'A{n}_proposal.json'],
                             (a.ROOT / a.prior.PACKAGE / f'A{n}_proposal.json').read_bytes())
            prior = a.read_csv(a.ROOT, a.prior.PACKAGE / f'P{n}_balanced_exposure.csv')
            now = decode(self.outputs[a.PROJECTION / f'P{n}_balanced_exposure.csv'])
            self.assertNotEqual([r['source_row_id'] for r in prior], [r['source_row_id'] for r in now])
            self.assertEqual(len(now), len(prior))
            self.assertEqual(sum(r['label'] == '1' for r in now), len(now) // 2)
        self.assertFalse(self.summary['negative_queue_regenerated'])

    def test_eligibility_and_original_queue_columns(self):
        for path in (a.OVERLAY / 'source_fact_eligibility.csv', a.OVERLAY / 'fact_inventory.csv',
                     a.PROJECTION / 'audited_fact_eligibility.csv'):
            a.guard_eligibility(decode(self.outputs[path]), self.current, self.banned)
        status = decode(self.outputs[a.PROJECTION / 'TC_queue_status.csv'])
        for before, after in zip(self.queue, status):
            self.assertTrue(all(after[k] == v for k, v in before.items()))
        for row in status:
            if row['tc_review_status'] != 'usable':
                self.assertEqual(row['tc_pair_eligible'], 'False')


if __name__ == '__main__':
    unittest.main()
