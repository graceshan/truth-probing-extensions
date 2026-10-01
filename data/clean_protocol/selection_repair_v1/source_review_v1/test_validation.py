"""Mutation tests for review identities, scope, evidence and pairing."""
import copy
import csv
import importlib.util
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

P=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('t2b_validate',P/'validate.py')
v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)


class ReviewValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.package=Path(self.tmp.name)/'package'
        shutil.copytree(P,self.package)

    def change(self,name,fn):
        path=self.package/name
        data=json.loads(path.read_text());fn(data)
        path.write_text(json.dumps(data))

    def fails(self,pattern):
        with self.assertRaisesRegex(ValueError,pattern):
            v.check(v.ROOT,self.package,check_hashes=False)

    def test_frozen_package_and_inputs(self):
        report=v.check()
        self.assertEqual(report['sample_size'],30)
        self.assertEqual(report['counts'],{'supported_false':4,'supported_true':2,'unresolved':24})
        self.assertEqual(report['recommended_exclusion_rows'],52)

    def test_missing_row(self):
        self.change('reviews.json',lambda x:x.pop())
        self.fails('Exactly 30')

    def test_duplicate(self):
        self.change('reviews.json',lambda x:x.__setitem__(1,copy.deepcopy(x[0])))
        self.fails('Duplicate')

    def test_order(self):
        self.change('reviews.json',lambda x:x.reverse())
        self.fails('identity/order')

    def test_identity(self):
        for field in v.IDENTITIES:
            with self.subTest(field=field):
                path=self.package/'reviews.json';original=path.read_bytes()
                self.change('reviews.json',lambda x:x[0].__setitem__(field,'changed'))
                self.fails('identity/order');path.write_bytes(original)

    def test_invalid_status(self):
        self.change('reviews.json',lambda x:x[0].__setitem__('judgment','probably_false'))
        self.fails('Invalid judgment')

    def test_missing_reasoning(self):
        self.change('reviews.json',lambda x:x[0].__setitem__('reasoning',''))
        self.fails('Missing reasoning')

    def test_human_claim(self):
        self.change('reviews.json',lambda x:x[0].__setitem__('reviewer_method','human verified'))
        self.fails('misrepresented')

    def test_evidence_reference(self):
        self.change('reviews.json',lambda x:x[0].__setitem__('evidence_ids',['not_inspected']))
        self.fails('Unknown evidence')

    def test_missing_excerpt(self):
        self.change('evidence.json',lambda x:x['edison']['observations'][0].__setitem__('excerpt',''))
        self.fails('excerpt/locator')

    def test_no_weak_source_resolved(self):
        self.change('reviews.json',lambda x:x[0].__setitem__('evidence_ids',['kirlian']))
        self.fails('authoritative evidence')

    def test_franklin_not_in_denominator(self):
        self.change('supplementary_reviews.json',lambda x:x.__setitem__('sample_denominator_contribution',1))
        self.fails('entered sample')

    def test_historical_status_preserved(self):
        self.change('supplementary_reviews.json',lambda x:x['franklin_france']['historical_record_unchanged'].__setitem__('validation_status','supported_true'))
        self.fails('Historical judgment changed')

    def test_paired_exclusion(self):
        p=self.package/'recommended_exclusions.csv'
        lines=p.read_text().splitlines();p.write_text('\n'.join(lines[:-1])+'\n')
        self.fails('Exclusion coverage')

    def test_rule_not_applied(self):
        self.change('proposed_rule_changes.json',lambda x:x.__setitem__('status','applied'))
        self.fails('prematurely')

    def test_hash_integrity(self):
        p=self.package/'review_policy.md';p.write_text(p.read_text()+'\nmutation\n')
        with self.assertRaisesRegex(ValueError,'Package bytes changed'):
            v.check(v.ROOT,self.package)


if __name__=='__main__':unittest.main()
