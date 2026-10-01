"""Fact-only regression tests; no scoring, activations or compound construction."""
import copy
import csv
import io
import json
import unittest

from src import final_partition_fact_audit as a
from src.validated_negatives import development_only


def decode(data):
    return list(csv.DictReader(io.StringIO(data.decode())))


class FinalFactAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs, cls.summary = a.build()
        cls.rows = a.csv_read(a.ROOT, a.PREVIOUS/'row_manifest.csv')
        cls.entities = a.load(a.ROOT, a.PACKAGE/'entity_audit_order.json')
        cls.queue = a.load(a.ROOT, a.PACKAGE/'negative_candidate_queue.json')
        cls.reviews = a.load(a.ROOT, a.PACKAGE/'reviews.json')
        cls.evidence = a.load(a.ROOT, a.PACKAGE/'evidence.json')
        cls.current, cls.actions, cls.banned = a.propagate(cls.rows, cls.reviews)

    def test_regeneration_exact_bytes(self):
        for p, data in self.outputs.items():
            self.assertEqual((a.ROOT/p).read_bytes(), data, str(p))

    def test_derived_capacity_and_complete_inventory(self):
        expected = dict(animal_class=16, cities=146, element_symb=18, inventors=11, sp_en_trans=34)
        for r in self.summary['capacity']:
            self.assertEqual(r['usable_entities'], expected[r['topic']])
            self.assertEqual(r['attempted_entities']+r['inventory_excluded'], r['original_E_entities'])
            self.assertEqual(r['usable_entities']+r['unresolved_entities'], r['attempted_entities'])
            self.assertEqual(r['recipe_pairs'], 2*r['usable_entities'])
            self.assertEqual(r['expected_binary_rows'], 32*r['usable_entities'])
        self.assertEqual(self.summary['expected_recipe_pairs'], 450)
        self.assertEqual(self.summary['expected_binary_rows'], 7200)
        self.assertEqual(self.summary['attempted_negative_candidates'], 261)
        self.assertEqual(self.summary['admitted_by_split'], dict(train=3040, validation=1012, test=974))
        self.assertEqual(len(self.actions), 28)
        self.assertEqual(self.summary['new_confirmed_label_errors'], 0)

    def test_recovered_guards_stay_closed(self):
        with self.assertRaisesRegex(ValueError, 'test candidates are locked'):
            development_only('test')
        cfg = a.load(a.ROOT, 'config/clean_protocol/generalization_protocols.json')
        self.assertIs(cfg['final_evaluation_enabled'], False)
        self.assertFalse(self.summary['final_evaluation_enabled'])
        self.assertFalse(self.summary['production_compounds_generated'])

    def test_no_identity_label_or_partition_edits(self):
        for old, new in zip(self.rows, self.current):
            self.assertTrue(all(new[k] == v for k, v in old.items()))
            if old['split'] != 'test':
                self.assertEqual(new[a.STATUS], old['tc_successor_status'])
        self.assertFalse(self.summary['non_E_inputs_changed'])
        self.assertTrue(self.summary['v4_train_D_fits_reusable'])
        for n in (10,15):
            self.assertEqual(self.outputs[a.PROJECTION/f'A{n}_proposal.json'],
                             (a.ROOT/a.PILOT/f'A{n}_proposal.json').read_bytes())
        self.assertEqual(self.outputs[a.PROJECTION/'TC_completed_pairs.json'],
            (a.ROOT/a.BASE/'tc_topup_audit_v1/combined_TC_completed_pairs.json').read_bytes())

    def test_pair_quarantine_does_not_ban_whole_person(self):
        byid = {r['source_row_id']:r for r in self.current}
        # France is unresolved but Jeffreys's UK positive remains admitted.
        self.assertEqual(byid['inventors:183'][a.STATUS], 'quarantined')
        self.assertEqual(byid['neg_inventors:183'][a.STATUS], 'quarantined')
        self.assertEqual(byid['inventors:253'][a.STATUS], 'admitted')
        self.assertEqual(byid['neg_inventors:253'][a.STATUS], 'admitted')

    def test_new_unresolved_source_both_labels_and_negations(self):
        for label in ('0','1'):
            source = next(r for r in self.rows if r['topic']=='cities' and r['split']=='test'
                          and r['form']=='affirmative' and r['label']==label)
            review = dict(topic=source['topic'], person_key=source['person_key'], entity_id=source['entity_id'],
                facts=[dict(statement=source['statement'], statement_sha256=source['statement_sha256'], judgment='unresolved')])
            after, actions, banned = a.propagate(self.rows, [review])
            for sid in (source['source_row_id'], source['paired_source_row_id']):
                self.assertEqual(actions[sid]['status'],'quarantined')
                stale = next(r for r in self.current if r['source_row_id']==sid)
                for name in ('train','D','E','P15','P10','balanced'):
                    with self.subTest(label=label, sid=sid, membership=name), self.assertRaisesRegex(ValueError,'restricted paired'):
                        a.guard_memberships({name:[stale]}, after)
            self.assertFalse(a.eligible_claim(source['topic'],source['person_key'],source['statement_sha256'],
                                              int(label),after,banned,{}))

    def test_generated_duplicate_and_confirmed_conflict_propagation(self):
        source = next(r for r in self.rows if r['source_row_id']=='inventors:183')
        for judgment, status in [('unresolved','quarantined'),('supported_true','excluded')]:
            # No source reference: exact signature still triggers both source forms.
            review = dict(topic=source['topic'], person_key=source['person_key'], entity_id=source['entity_id'],
                facts=[dict(origin='recovered_negative_pipeline', fact_id='alternate-reference',
                            statement=source['statement'],statement_sha256=source['statement_sha256'],judgment=judgment)])
            after, actions, banned = a.propagate(self.rows,[review])
            self.assertEqual(actions[source['source_row_id']]['status'],status)
            self.assertEqual(actions[source['paired_source_row_id']]['status'],status)
            self.assertFalse(a.eligible_claim(source['topic'],source['person_key'],source['statement_sha256'],
                                              0,after,banned,{}))

    def test_evidence_identity_queue_and_spanish_mutations_fail(self):
        for kind in ('entity','hash','binding','replacement','spanish','skip','reopen'):
            reviews, ev = copy.deepcopy(self.reviews), copy.deepcopy(self.evidence)
            if kind=='entity': reviews[0]['person_key']='other'
            elif kind=='hash': reviews[0]['facts'][0]['statement_sha256']='0'*64
            elif kind=='binding': ev[reviews[0]['facts'][0]['evidence_ids'][0]]['fact_bindings']=[]
            elif kind=='replacement': reviews[0]['facts'][1]['candidate_rank']+=1
            elif kind=='spanish': next(r for r in reviews if r['topic']=='sp_en_trans')['spanish_checks']['direction']='English→Spanish'
            elif kind=='skip': reviews.pop(0)
            else:
                r=next(r for r in reviews if len(r['facts'])==3)
                r['facts'][1]['judgment']='unresolved'
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                a.validate_reviews(self.rows,self.entities,self.queue,reviews,ev)

    def test_rejected_true_candidate_advances_without_redraw(self):
        r=next(r for r in self.reviews if r['entity']=='John Ericsson')
        self.assertEqual([f.get('candidate_rank') for f in r['facts'][1:]],[2,3])
        self.assertEqual([f['judgment'] for f in r['facts'][1:]],['supported_true','supported_false'])
        q=[q for q in self.queue if q['entity_id']==r['entity_id']]
        self.assertFalse(q[0]['base_eligible'])
        self.assertIn('inventors:240',q[0]['source_matches'])

    def test_all_alternate_fact_records_are_restricted(self):
        for name in ('source_fact_eligibility.csv','fact_inventory.csv'):
            for r in decode(self.outputs[a.OVERLAY/name]):
                if a.signature(r) in self.banned or r.get('source_row_id',r.get('fact_ref')) in self.actions:
                    self.assertEqual(r[a.ELIGIBLE],'False')
        for q in json.loads(self.outputs[a.PROJECTION/'E_negative_candidate_eligibility.json']):
            if a.signature(q) in self.banned or not q['base_eligible']:
                self.assertIs(q[a.ELIGIBLE],False)

    def test_hash_ranking_independent_of_input_row_order(self):
        manifest=a.csv_read(a.ROOT,'data/clean_protocol/entity_partitions/manifest.csv')
        _, entities, queue=a.reconstruct_queue(list(reversed(self.rows)),list(reversed(manifest)))
        self.assertEqual(entities,self.entities)
        self.assertEqual([(q['fact_id'],q['candidate_rank'],q['ranking_hash']) for q in queue],
                         [(q['fact_id'],q['candidate_rank'],q['ranking_hash']) for q in self.queue])

    def test_pairing_recipe_small_capacity_and_order(self):
        self.assertFalse(a.capacity_only(['a','b','c','d'],'cities')['degree_four_possible'])
        for n in (5,6,11,34,146):
            ids=[str(i) for i in range(n)]
            self.assertEqual(a.capacity_only(ids,'cities'),a.capacity_only(ids[::-1],'cities'))
            self.assertEqual(a.capacity_only(ids,'cities')['recipe_pairs'],2*n)
        with self.assertRaises(ValueError): a.capacity_only(['a','a'],'cities')
