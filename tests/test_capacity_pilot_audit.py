"""Audit-contract tests: identities, tampering, ordering and actual row exposure."""
import copy
import csv
import json
import unittest

from src import capacity_pilot_audit as a


class CapacityPilotAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = list(csv.DictReader((a.ROOT / a.OVERLAY).open()))
        cls.order = a.read_json(a.ROOT, 'candidate_order.json')
        cls.reviews = a.read_json(a.ROOT, 'reviews.json')
        cls.evidence = a.read_json(a.ROOT, 'evidence.json')
        cls.proposals = a.read_json(a.ROOT, 'negative_proposals.json')
        cls.outputs, cls.summary = a.build()

    def validate(self, reviews=None, evidence=None):
        a.validate_pairs(reviews or self.reviews, self.order, self.rows,
                         evidence or self.evidence, self.proposals, a.ROOT)

    def test_checked_in_outputs_match_rebuild(self):
        for name, value in self.outputs.items():
            self.assertEqual((a.ROOT / a.PACKAGE / name).read_bytes(), value, name)

    def test_eligibility_aliases_and_cross_partition(self):
        keys = {r['person_key'] for r in self.order}
        for k in keys:
            self.assertEqual({r['split'] for r in self.rows if r['person_key'] == k}, {'train'})
        aliases = [r for r in self.order if 'Philo Farnsworth' in r['entities']]
        self.assertEqual(len(aliases), 1)
        self.assertEqual(len(aliases[0]['entity_ids']), 2)
        self.assertFalse(any('Simjian' in n for o in self.order for n in o['entities']))

    def test_order_independent_of_source_order(self):
        # Ordering of keys must survive row permutation; statement presentation may differ.
        shuffled = list(reversed(self.rows))
        key_order = lambda rr: [(r['topic'], r['rank'], r['person_key']) for r in a.candidate_order(rr)]
        self.assertEqual(key_order(shuffled), key_order(self.rows))

    def test_foreign_word_positive_fails(self):
        rr = copy.deepcopy(self.reviews)
        spanish = [r for r in rr if r['topic'] == 'sp_en_trans']
        spanish[0]['facts'][0] = copy.deepcopy(spanish[1]['facts'][0])
        with self.assertRaisesRegex(ValueError, 'binding mismatch'):
            self.validate(rr)

    def test_unreviewed_pipeline_candidate_cannot_pass(self):
        rr = copy.deepcopy(self.reviews)
        r = next(r for r in rr if r['topic'] == 'sp_en_trans')
        r['facts'][1]['candidate_rank'] += 1
        with self.assertRaisesRegex(ValueError, 'pipeline fact/entity'):
            self.validate(rr)

    def test_wrong_evidence_and_missing_coverage_fail(self):
        rr = copy.deepcopy(self.reviews)
        rr[0]['facts'][0]['evidence_ids'] = ['cities_1']
        with self.assertRaisesRegex(ValueError, 'evidence/fact'):
            self.validate(rr)
        rr[0]['facts'][0]['evidence_ids'] = []
        with self.assertRaisesRegex(ValueError, 'coverage'):
            self.validate(rr)

    def test_unresolved_cannot_be_reserved(self):
        rr = copy.deepcopy(self.reviews)
        next(r for r in rr if r['status'] == 'unresolved')['status'] = 'usable'
        with self.assertRaisesRegex(ValueError, 'completed judgments'):
            self.validate(rr)

    def test_no_redraw_or_skipping(self):
        rr = copy.deepcopy(self.reviews)
        rr.remove(next(r for r in rr if r['status'] == 'unresolved'))
        with self.assertRaisesRegex(ValueError, 'gap/redraw'):
            a.validate_attempt_sequence(rr, self.order)

    def test_negation_does_not_fill_second_fact(self):
        rr = copy.deepcopy(self.reviews)
        r = rr[0]
        f = r['facts'][1]
        source = next(s for s in self.rows if s['source_row_id'] == r['facts'][0]['source_row_id'])
        negated = next(s for s in self.rows if s['source_row_id'] == source['paired_source_row_id'])
        f.update(source_row_id=negated['source_row_id'], statement=negated['statement'],
                 statement_sha256=negated['statement_sha256'])
        with self.assertRaises(ValueError):
            self.validate(rr)

    def test_paired_status_mutation_fails(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['successor_status'] = 'quarantined'
        with self.assertRaisesRegex(ValueError, 'paired source'):
            a.validate_source_pairs(rows)

    def test_membership_nested_complete_and_disjoint(self):
        proposals = {}
        for n in (10, 15):
            selected = json.loads(self.outputs[f'A{n}_proposal.json'])['completed_pairs']
            self.assertEqual(len(selected), 5*n)
            keys = proposals[n] = {r['person_key'] for r in selected}
            p = list(csv.DictReader(self.outputs[f'P{n}_admitted_membership.csv'].decode().splitlines()))
            removed = list(csv.DictReader(self.outputs[f'P{n}_excluded_source_variants.csv'].decode().splitlines()))
            self.assertFalse(keys & {r['person_key'] for r in p})
            self.assertEqual({r['source_row_id'] for r in removed},
                             {r['source_row_id'] for r in self.rows if r['split'] == 'train' and r['person_key'] in keys})
            a.validate_source_pairs(p)
        self.assertLess(proposals[10], proposals[15])

    def test_sampler_exposure_and_topup(self):
        for n, size in ((10, 160), (15, 140)):
            exposure = [r for r in self.summary['exposure_counts'] if r['scenario'] == f'P{n}_balanced_exposure']
            self.assertEqual(len(exposure), 5)
            self.assertTrue(all(r['rows'] == size and r['label_0'] == r['label_1'] == size//2 for r in exposure))
        self.assertEqual(self.summary['topics']['inventors']['TC_completed'], 16)
        self.assertEqual(sum(v['TC_additional_usable_pairs_needed'] for v in self.summary['topics'].values()), 24)


if __name__ == '__main__':
    unittest.main()
