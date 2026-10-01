"""Regression coverage for claim-level quarantine propagation, never probe fitting."""
import copy
import csv
import io
import json
import unittest

from src import capacity_pilot_projection as p


def decode(data):
    return list(csv.DictReader(io.StringIO(data.decode())))


class ProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs, cls.summary = p.build()
        cls.rows = p.read_csv(p.ROOT, p.old.OVERLAY)
        cls.reviews = p.load(p.ROOT, p.old.PACKAGE / 'reviews.json')
        cls.proposals = p.load(p.ROOT, p.old.PACKAGE / 'negative_proposals.json')
        cls.audit, cls.affected, cls.unresolved = p.inspect_outcomes(cls.rows, cls.reviews, cls.proposals)
        cls.current = decode(cls.outputs[p.OVERLAY / 'row_manifest.csv'])

    def test_deterministic_checked_in_outputs(self):
        for path, content in self.outputs.items():
            self.assertEqual((p.ROOT / path).read_bytes(), content, str(path))

    def test_all_outcomes_and_exact_affected_pairs(self):
        self.assertEqual(len(self.audit), 83)
        self.assertEqual(sum(len(r['facts']) for r in self.audit), 166)
        self.assertEqual(self.affected, {'inventors:23', 'neg_inventors:23', 'inventors:157', 'neg_inventors:157'})
        self.assertEqual(len(self.summary['unresolved_generated_candidates']), 6)
        self.assertEqual(len(self.summary['unresolved_source_facts']), 2)

    def test_preserve_every_prior_column_and_other_people_rows(self):
        for before, after in zip(self.rows, self.current):
            self.assertTrue(all(after[k] == v for k, v in before.items()))
            expected = 'quarantined' if before['source_row_id'] in self.affected else before['successor_status']
            self.assertEqual(after[p.STATUS], expected)
        people = {r['person_key'] for r in self.rows if r['source_row_id'] in self.affected}
        remaining = [r for r in self.current if r['person_key'] in people and r[p.STATUS] == 'admitted']
        self.assertEqual(len(remaining), 4)
        self.assertEqual({r['source_row_id'] for r in remaining},
                         {'inventors:58', 'neg_inventors:58', 'inventors:283', 'neg_inventors:283'})

    def test_bindings_reject_identity_and_hash_mutations(self):
        for mutation in ('identity', 'hash'):
            reviews = copy.deepcopy(self.reviews)
            r = next(r for r in reviews if r['topic'] == 'inventors' and r['rank'] == 12)
            if mutation == 'identity':
                r['person_key'] = 'different_person'
            else:
                r['facts'][1]['statement_sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                p.inspect_outcomes(self.rows, reviews, self.proposals)

    def test_new_unresolved_source_fact_and_negation_cannot_survive(self):
        # Discover a NEW restriction dynamically, rather than hard-coding the two reviewed IDs.
        reviews = copy.deepcopy(self.reviews)
        review = next(r for r in reviews if r['topic'] == 'cities')
        fact = next(f for f in review['facts'] if f['label'] == 0)
        fact['judgment'], review['status'] = 'unresolved', 'unresolved'
        _, affected, unresolved = p.inspect_outcomes(self.rows, reviews, self.proposals)
        row = next(r for r in self.rows if r['source_row_id'] == fact['source_row_id'])
        for sid in (row['source_row_id'], row['paired_source_row_id']):
            self.assertIn(sid, affected)
            survivor = next(r for r in self.current if r['source_row_id'] == sid)
            for name in self.summary['current_membership_totals']:
                with self.subTest(source_row=sid, membership=name), self.assertRaisesRegex(ValueError, 'survives'):
                    p.assert_no_unresolved({name: [survivor]}, affected, unresolved)

    def test_all_current_memberships_and_atomic_eligibility(self):
        for name in self.summary['current_membership_totals']:
            rows = decode(self.outputs[p.PACKAGE / (name + '.csv')])
            p.assert_no_unresolved({name: rows}, self.affected, self.unresolved)
            self.assertTrue(all(r['split'] == 'train' for r in rows))
        elig = decode(self.outputs[p.OVERLAY / 'source_fact_eligibility.csv'])
        self.assertTrue(all(r['current_atomic_eligible'] == 'False' for r in elig if r['source_row_id'] in self.affected))

    def test_source_and_registry_fact_eligibility_regression(self):
        inventory = decode(self.outputs[p.OVERLAY / 'fact_inventory.csv'])
        changed = [r for r in inventory if p.signature(r) in self.unresolved and r['eligible'] == 'True']
        self.assertEqual(len(changed), 4)  # two affirmatives + two historical registry aliases
        self.assertEqual(sum(r['origin'] == 'source' for r in changed), 2)
        for row in changed:
            self.assertEqual(row['current_eligible'], 'False')
            bad = dict(row, current_eligible='True')
            with self.assertRaisesRegex(ValueError, 'fact eligibility'):
                p.assert_fact_eligibility([bad], self.affected, self.unresolved, self.current)
        reviewed = decode(self.outputs[p.PACKAGE / 'audited_fact_eligibility.csv'])
        self.assertEqual(sum(r['current_eligible'] == 'False' for r in reviewed), 8)

    def test_frozen_proposals_and_queue_not_redrawn(self):
        for name in ('A15_proposal.json', 'A10_proposal.json', 'TC_completed_pairs.json'):
            self.assertEqual(self.outputs[p.PACKAGE / name], (p.ROOT / p.old.PACKAGE / name).read_bytes())
        prior = p.read_csv(p.ROOT, p.old.PACKAGE / 'TC_topup_review_queue.csv')
        current = decode(self.outputs[p.PACKAGE / 'TC_topup_review_queue.csv'])
        self.assertEqual(len(prior), len(current))
        for a, b in zip(prior, current):
            self.assertTrue(all(b[k] == v for k, v in a.items()))
        self.assertEqual(sum(r['current_negative_status'] == 'quarantined_source_claim_requires_resolution'
                             for r in current), 2)

    def test_verified_counts_and_sampler_changes(self):
        self.assertEqual(self.summary['admitted_by_split'], {'train': 3044, 'validation': 1012, 'test': 1002})
        expected = {'corrected_outer_training': 3044, 'corrected_outer_training_balanced': 1000,
                    'P15_admitted_membership': 2782, 'P10_admitted_membership': 2868,
                    'P15_balanced_exposure': 700, 'P10_balanced_exposure': 800}
        self.assertEqual(self.summary['current_membership_totals'], expected)
        for n in (15, 10):
            new = decode(self.outputs[p.PACKAGE / f'P{n}_balanced_exposure.csv'])
            old = p.read_csv(p.ROOT, p.old.PACKAGE / f'P{n}_balanced_exposure.csv')
            self.assertNotEqual([r['source_row_id'] for r in old], [r['source_row_id'] for r in new])
            self.assertEqual(len(new), len(old))
            self.assertEqual(sum(r['label'] == '0' for r in new), len(new) // 2)
        self.assertTrue(self.summary['correction_base_must_advance_before_fitting'])
        self.assertEqual(self.summary['confirmed_new_label_errors'], 0)


if __name__ == '__main__':
    unittest.main()
