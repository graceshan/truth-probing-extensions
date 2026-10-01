"""Exact Section 9 wording and reviewed cohort tests; no model/scoring execution."""
import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src import heldout_and_preparation as p
from src.heldout_and_wording import CONDITION, render, validate_spec


class HeldoutAndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.outputs,cls.summaries=p.build()
        cls.e_bare=p.normalized(p.table(cls.outputs[p.E_OUTPUT/'bare_compounds.csv']))
        cls.e_new=p.normalized(p.table(cls.outputs[p.E_OUTPUT/(CONDITION+'.csv')]))
        cls.facts={f['fact_id']:dict(f,truth=bool(f['label']),eligible=True) for f in json.loads(cls.outputs[p.E_OUTPUT/'isolated_facts.json'])}

    def test_exact_pure_renderer_spec_and_operator(self):
        spec=p.old.load(p.ROOT,p.SPEC);validate_spec(spec)
        a="The Spanish word 'árbol' means 'tree'.";b='Oxygen has the symbol O.'
        self.assertEqual(render(a,b),'Both of the following are true: '+a+' '+b)
        self.assertEqual(render(b,a),'Both of the following are true: '+b+' '+a)
        with self.assertRaises(ValueError):render(a,b,'OR')
        with self.assertRaises(ValueError):render(a+' ',b)
        wrong=copy.deepcopy(spec);wrong['prefix']+=' '
        with self.assertRaises(ValueError):validate_spec(wrong)
        subprocess.run([sys.executable,'-c',"import sys; from src.heldout_and_wording import render; assert not any(k in sys.modules for k in ['torch','transformers','numpy','pandas','sklearn'])"],check=True)

    def test_exact_D_E_counts_and_every_truth_surface_cell(self):
        for split,out,n,rows in [('D',p.D_OUTPUT,483,3864),('E',p.E_OUTPUT,450,3600)]:
            with self.subTest(split=split):
                self.assertEqual(self.summaries[split]['pairs'],n)
                self.assertEqual(self.summaries[split]['and_wording_rows'],rows)
                new=p.normalized(p.table(self.outputs[out/(CONDITION+'.csv')]))
                bare=p.normalized(p.table(self.outputs[out/'bare_compounds.csv']))
                original={r['example_id']:r for r in bare}
                self.assertEqual(len({r['example_id'] for r in new}),rows)
                for r in new:
                    b=original[r['base_example_id']]
                    first,second=(b['fact_a_statement'],b['fact_b_statement']) if b['ordering']=='AB' else (b['fact_b_statement'],b['fact_a_statement'])
                    self.assertEqual(r['statement'],'Both of the following are true: '+first+' '+second)
                    self.assertEqual(r['compound_label'],b['canonical_truth_a'] and b['canonical_truth_b'])
                    for k,v in b.items():
                        if k not in ('example_id','statement','template_id'):self.assertEqual(r[k],v)
                index=p.table(self.outputs[out/'condition_index.csv'])
                for r in index:
                    if r['condition_id']==CONDITION:
                        self.assertEqual(r['reused_bare'],str(r['operator']=='OR'))
                        if r['operator']=='OR':self.assertEqual(r['selected_example_id'],r['base_example_id'])

    def test_reject_substitution_ineligible_wrong_operator_label_and_surface(self):
        base=next(r for r in self.e_bare if r['operator']=='AND')
        for kind in ('substituted','ineligible','operator','label'):
            row=copy.deepcopy(base);facts=copy.deepcopy(self.facts)
            if kind=='substituted':row['fact_a_statement']='Substituted.'
            elif kind=='ineligible':facts[row['fact_a_id']]['eligible']=False
            elif kind=='operator':row['operator']='OR'
            else:row['compound_label']=not row['compound_label']
            with self.subTest(kind=kind),self.assertRaises(ValueError):p.wording(row,facts)
        for field,value in [('statement','Both of the following are true: Changed.'),('pair_id','wrong'),('surface_first_entity_id','wrong'),('compound_label',not self.e_new[0]['compound_label'])]:
            rows=copy.deepcopy(self.e_new);rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):p.validate_wording(rows,self.e_bare,self.facts)

    def test_D_exact_reviewed_eligibility_and_no_redraw(self):
        source=p.old.read_csv(p.ROOT,p.old.BASE/'candidate_overlay_v4/row_manifest.csv')
        ledger=p.old.read_csv(p.ROOT,p.old.BASE/'candidate_overlay_v4/fact_inventory.csv')
        facts=p.old.read_csv(p.ROOT,p.INPUTS/'D_isolated_facts.csv')
        actual=p.d_fact_eligibility(facts,source,ledger)
        expected=p.old.read_csv(p.ROOT,p.INPUTS/'development_fact_eligibility.csv')
        self.assertEqual(actual,expected)
        allowed={r['pair_id'] for r in p.old.read_csv(p.ROOT,p.INPUTS/'development_pair_eligibility.csv') if r['tc_current_eligible']=='True'}
        self.assertEqual({r['pair_id'] for r in p.table(self.outputs[p.D_OUTPUT/'bare_compounds.csv'])},allowed)
        f=next(f for f in facts if f['truth']=='True' and next(r for r in actual if r['fact_id']==f['fact_id'])['tc_current_eligible']=='True')
        for r in source:
            if r['entity_id']==f['entity_id'] and r['statement']==f['statement']:r['tc_successor_status']='quarantined'
        changed=p.d_fact_eligibility(facts,source,ledger)
        self.assertEqual(next(r for r in changed if r['fact_id']==f['fact_id'])['tc_current_eligible'],'False')

    def test_historical_examples_and_bindings_preserved(self):
        for name in ('bare_compounds.csv','wording_compounds.csv','pairs.csv','isolated_facts.json','constituent_evidence.json','admitted_atomic_E.csv','entity_degrees.csv','pair_hash_order.json'):
            self.assertEqual(self.outputs[p.E_OUTPUT/name],(p.ROOT/p.old.OUTPUT/name).read_bytes())
        old=p.old.read_csv(p.ROOT,p.old.OUTPUT/'extraction_bindings.csv')
        new=p.table(self.outputs[p.E_OUTPUT/'extraction_bindings.csv'])
        self.assertEqual(new[:len(old)],old)
        preservation=json.loads(self.outputs[p.E_OUTPUT/'non_E_preservation.json'])
        self.assertTrue(all(v['unchanged'] for v in preservation.values()))

    def test_extraction_identity_reversibility_and_dedup_derived(self):
        for out in (p.E_OUTPUT,p.D_OUTPUT):
            files={path.name:data for path,data in self.outputs.items() if path.parent==out}
            bindings=p.table(files['extraction_bindings.csv']);texts=p.table(files['extraction_texts.csv'])
            p.validate_reversible(bindings,texts,files)
            self.assertEqual(len(texts),len({r['text_id'] for r in bindings}))
            bindings[-1]['record_index']='0'
            with self.assertRaises(ValueError):p.validate_reversible(bindings,texts,files)
        self.assertEqual(self.summaries['E']['extraction_bindings'],15824+3600)
        self.assertEqual(self.summaries['D']['extraction_bindings'],3864)

    def test_determinism_and_immutable_preflight(self):
        again,_=p.build();self.assertEqual(self.outputs,again)
        with tempfile.TemporaryDirectory() as root:
            p.materialize(root,self.outputs,True);p.materialize(root,again)
            path=Path(root)/p.E_OUTPUT/(CONDITION+'.csv');path.write_bytes(b'changed')
            with self.assertRaises(ValueError):p.materialize(root,again,True)

    def test_no_prediction_activation_or_model_access(self):
        code='''
import sys
from src import heldout_and_preparation as p
allowed={'input_inventory.json','preparation_receipt.json'}
def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=str(args[0])
        assert not path.endswith(('.npz','.npy','.safetensors')),path
        assert '/acts/' not in path and '/scores/' not in path,path
        if '/results/' in path:assert path.rsplit('/',1)[-1] in allowed,path
sys.addaudithook(guard)
p.build()
assert not any(k in sys.modules for k in ['torch','transformers','sklearn'])
'''
        subprocess.run([sys.executable,'-c',code],cwd=p.ROOT,check=True,capture_output=True)
