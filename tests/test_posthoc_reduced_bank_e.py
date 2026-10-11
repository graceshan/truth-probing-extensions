"""Focused failure-mode tests, without loading any historical data or E artifacts."""
import unittest
from copy import deepcopy
import numpy as np
from src.posthoc_reduced_bank_e import boundary_auc, independent_auc, maxima, validate_bank, validate_order


class PosthocTests(unittest.TestCase):
    def test_score_ties_half_credit(self):
        for y,s,expected in [([0,1],[7,7],.5), ([0,0,1,1],[0,1,1,2],.875),
                             ([0,1,0,1],[1,1,1,1],.5), ([0,1],[2,1],0.)]:
            self.assertEqual(boundary_auc(y,s),expected)
            self.assertEqual(independent_auc(y,s),expected)

    def test_all_exact_winners_preserved_and_tolerance_separate(self):
        rows=[dict(candidate_id=n,E_auroc=v,eligible=True,status='scored')
              for n,v in [('a',.8),('b',.8),('c',.8-5e-13)]]
        found=maxima(rows,True)
        self.assertEqual(found['exact_winner_ids'],['a','b'])
        self.assertEqual(found['tolerance_tie_ids'],['a','b','c'])

    def test_eligibility_changes_only_restricted_maximum(self):
        rows=[dict(candidate_id='a',E_auroc=.7,eligible=True,status='scored'),
              dict(candidate_id='b',E_auroc=.9,eligible=False,status='scored')]
        self.assertEqual(maxima(rows,True)['exact_winner_ids'],['a'])
        self.assertEqual(maxima(rows,False)['exact_winner_ids'],['b'])
        with self.assertRaises(ValueError):maxima([rows[1]],True)

    def test_missing_duplicate_substituted_or_invalid_candidate_blocks(self):
        specs=[dict(candidate_id='qwen/P15/L00/a'),dict(candidate_id='qwen/P15/L00/b')]
        rows=[dict(id=s['candidate_id'],spec=s,valid_for_scoring=True) for s in specs]
        validate_bank(rows,specs)
        for bad in [rows[:1],[rows[0],rows[0]]]:
            with self.assertRaises(ValueError):validate_bank(bad,specs)
        for field,value in [('id','qwen/full_train/L00/b'),('valid_for_scoring',False)]:
            bad=deepcopy(rows);bad[1][field]=value
            with self.assertRaises(ValueError):validate_bank(bad,specs)

    def test_row_reordering_or_label_change_blocks(self):
        expected=[dict(example_id='a',label=0),dict(example_id='b',label=1)]
        validate_order(expected,expected)
        with self.assertRaises(ValueError):validate_order(expected[::-1],expected)
        bad=deepcopy(expected);bad[0]['label']=1
        with self.assertRaises(ValueError):validate_order(bad,expected)

    def test_nonfinite_or_unscored_candidate_blocks(self):
        with self.assertRaises(ValueError):boundary_auc([0,1],[0,np.nan])
        with self.assertRaises(ValueError):boundary_auc([0,1],[0])
        with self.assertRaises(ValueError):maxima([dict(candidate_id='a',E_auroc=.8,eligible=True,status='missing')],True)


if __name__ == '__main__':
    unittest.main()
