"""Production data checks, including adversarial fact/label mutations; no extraction."""
import ast
import copy
import csv
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src import final_e_preparation as e
from src.clean_compounds import render_binary
from src.validated_negatives import development_only
from src.validation_compound_benchmark import degree_four_pairs


def csv_rows(payload):
    return list(csv.DictReader(io.StringIO(payload.decode())))


class FinalPreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.files,cls.meta=e.build()
        cls.manifest=e.EntityManifest(e.ROOT/'data/clean_protocol/entity_partitions/manifest.csv',
                                     e.ROOT/'data/clean_protocol/entity_partitions/metadata.json')
        cls.completed=e.load(e.ROOT,e.AUDIT/'completed_E_fact_pairs.json')
        cls.reviews=e.load(e.ROOT,e.AUDIT/'reviews.json')
        cls.evidence=e.load(e.ROOT,e.AUDIT/'evidence.json')
        cls.source=e.read_csv(e.ROOT,e.OVERLAY/'row_manifest.csv')
        cls.eligibility=e.read_csv(e.ROOT,e.PROJECTION/'audited_fact_eligibility.csv')
        cls.queue=e.load(e.ROOT,e.PROJECTION/'E_negative_candidate_eligibility.json')
        cls.facts=json.loads(cls.files['isolated_facts.json'])
        cls.rows=csv_rows(cls.files['bare_compounds.csv'])
        for r in cls.rows:
            for k in e.BOOL_FIELDS: r[k]=r[k]=='True'
            r['generation_seed']=int(r['generation_seed'])
        cls.requested={r['topic']:r['pairs'] for r in cls.meta['counts']}

    def test_deterministic_regeneration_and_saved_hashes(self):
        again,_=e.build()
        self.assertEqual(self.files,again)
        with tempfile.TemporaryDirectory() as root:
            e.materialize(root,self.files,write=True)
            e.materialize(root,again)
            for name,payload in self.files.items():
                self.assertEqual((Path(root)/e.OUTPUT/name).read_bytes(),payload,name)
        for name,digest in self.meta['output_sha256'].items():
            self.assertEqual(e.sha256(self.files[name]),digest)

    def test_capacity_degree_and_truth_cells(self):
        self.assertEqual((self.meta['entities'],self.meta['unordered_pairs'],self.meta['bare_rows']),(225,450,7200))
        self.assertEqual({r['topic']:r['entities'] for r in self.meta['counts']},
                         dict(animal_class=16,cities=146,element_symb=18,inventors=11,sp_en_trans=34))
        self.assertEqual({r['degree'] for r in csv_rows(self.files['entity_degrees.csv'])},{'4'})
        e.validate_rows(self.rows,self.manifest,self.facts,self.requested)
        for n in (5,6,11,34,146):
            ids=[str(i) for i in range(n)]
            self.assertEqual(e.degree_four(ids,'cities'),e.degree_four(ids[::-1],'cities'))
        with self.assertRaises(ValueError): e.degree_four(['a']*5,'cities')

    def test_incorrect_label_order_statement_and_split_rejected(self):
        for field,value in [('compound_label',not self.rows[0]['compound_label']),('split','validation'),
                            ('statement','Substituted fact.'),('surface_first_entity_id','wrong')]:
            rows=copy.deepcopy(self.rows);rows[0][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                e.validate_rows(rows,self.manifest,self.facts,self.requested)

    def validate(self,completed=None,reviews=None,evidence=None,source=None):
        return e.validate_facts(completed or self.completed,reviews or self.reviews,evidence or self.evidence,
            source or self.source,self.eligibility,self.queue,self.manifest,
            {r['person_key'] for r in self.source if r['split']!='test'})

    def test_unresolved_constituent_and_wrong_person_split_rejected(self):
        for kind in ('unresolved','person','split','evidence','text'):
            completed,reviews,ev=copy.deepcopy(self.completed),copy.deepcopy(self.reviews),copy.deepcopy(self.evidence)
            target=next(r for r in reviews if r['entity_id']==completed[0]['entity_id'])
            for r in (completed[0],target):
                if kind=='unresolved': r['facts'][0]['judgment']='unresolved'
                elif kind=='person': r['person_key']='wrong-reviewed-person'
                elif kind=='split': r['split']='validation'
                elif kind=='text': r['facts'][0]['statement']='The raccoon is a bird.'
            if kind=='evidence': ev[completed[0]['facts'][0]['evidence_ids'][0]]['fact_bindings']=[]
            with self.subTest(kind=kind),self.assertRaises(ValueError): self.validate(completed,reviews,ev)

    def test_restricted_generated_duplicate_and_paired_quarantine_rejected(self):
        source=copy.deepcopy(self.source)
        fact=self.completed[0]['facts'][0]
        row=next(r for r in source if r['source_row_id']==fact['source_row_id'])
        row[e.STATUS]='quarantined'
        with self.assertRaisesRegex(ValueError,'paired source'):
            e.validate_source_and_preservation(e.ROOT,source)
        partner=next(r for r in source if r['source_row_id']==row['paired_source_row_id'])
        partner[e.STATUS]='quarantined'
        with self.assertRaisesRegex(ValueError,'restricted source duplicate'): self.validate(source=source)
        # Even a generated negative with no source ID must obey an exact restricted duplicate.
        source=copy.deepcopy(self.source)
        r=self.completed[0];f=r['facts'][1]
        source.append(dict(source[0],source_row_id='alternate',entity_id=r['entity_id'],person_key=r['person_key'],
            topic=r['topic'],split='test',statement=f['statement'],statement_sha256=f['statement_sha256'],
            label='0',e_successor_status='quarantined'))
        with self.assertRaisesRegex(ValueError,'restricted source duplicate'): self.validate(source=source)

    def test_wording_matches_established_renderer_ast_and_exact_bindings(self):
        # Evaluate only the recovered pure text-expression AST as an independent oracle.
        module=ast.parse((e.ROOT/'src/priority2_input_controls.py').read_text())
        fn=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='variants')
        assignment=next(n for n in ast.walk(fn) if isinstance(n,ast.Assign) and
                        any(isinstance(t,ast.Name) and t.id=='text' for t in n.targets))
        expr=compile(ast.Expression(assignment.value),'<recovered text renderer>','eval')
        raw={r['example_id']:r for r in self.rows}
        wording=csv_rows(self.files['wording_compounds.csv'])
        index=csv_rows(self.files['condition_index.csv'])
        self.assertEqual(len(index),3*len(self.rows))
        for r in index:
            if r['operator']=='AND' or r['condition_id']=='bare':
                self.assertEqual(r['base_example_id'],r['selected_example_id'])
                self.assertEqual(r['reused_bare'],'True')
        for r in wording:
            b=raw[r['base_example_id']]
            first,second=(b['fact_a_statement'],b['fact_b_statement']) if b['ordering']=='AB' else (b['fact_b_statement'],b['fact_a_statement'])
            expected=eval(expr,{'__builtins__':{},'render_binary':render_binary},
                dict(first=first,second=second,BOTH=e.BOTH,LEAST=e.LEAST,JUX='juxtaposition_v1'))
            self.assertEqual(r['statement'],expected[r['condition_id']])
            for k in e.FIELDS:
                if k not in ('example_id','statement','template_id'): self.assertEqual(r[k],str(b[k]))

    def test_reversible_text_dedup_and_atomic_admission(self):
        texts={r['text_id']:r for r in csv_rows(self.files['extraction_texts.csv'])}
        bindings=csv_rows(self.files['extraction_bindings.csv'])
        atomic=csv_rows(self.files['admitted_atomic_E.csv'])
        self.assertEqual(len(atomic),974)
        self.assertTrue(all(r[e.STATUS]=='admitted' and r['split']=='test' for r in atomic))
        self.assertEqual(len(bindings),7200+7200+450+974)
        tables={'bare':self.rows,'wording':csv_rows(self.files['wording_compounds.csv']),
                'isolated':self.facts,'atomic':atomic}
        for b in bindings:
            row=tables[b['kind']][int(b['record_index'])]
            self.assertEqual(texts[b['text_id']]['statement'],row['statement'])
            self.assertEqual(b['statement_sha256'],e.sha256(row['statement'].encode()))
        self.assertLess(len(texts),len(bindings))
        self.assertEqual(len({r['source_row_id'] for r in atomic}),len(atomic))

    def test_final_scoring_and_recovered_test_guards_remain_closed(self):
        self.assertIs(self.meta['final_evaluation_enabled'],False)
        with self.assertRaises(ValueError): development_only('test')
        with self.assertRaises(ValueError): degree_four_pairs(self.manifest,'cities','test',292,0)
        self.assertFalse(self.meta['non_E_inputs_changed'])
        self.assertTrue(all(r['unchanged'] for r in json.loads(self.files['non_E_preservation.json']).values()))

    def test_immutable_preflight_no_partial_write(self):
        with tempfile.TemporaryDirectory() as root:
            dest=Path(root)/e.OUTPUT;dest.mkdir(parents=True)
            (dest/'last.json').write_bytes(b'old')
            with self.assertRaises(ValueError):
                e.materialize(root,{'first.json':b'new','last.json':b'changed'},write=True)
            self.assertFalse((dest/'first.json').exists())
            self.assertEqual((dest/'last.json').read_bytes(),b'old')

    def test_entrypoint_does_not_import_models_or_read_caches_predictions(self):
        code='''
import sys
from pathlib import Path
from src import final_e_preparation as e

def guard(event,args):
    if event=='open' and isinstance(args[0],(str,bytes)):
        path=str(args[0])
        assert not any('/'+part+'/' in path for part in ('acts','results')),path
        assert not path.endswith(('.npz','.npy','.safetensors')),path
sys.addaudithook(guard)
e.build()
assert not any(name.split('.')[0] in {'torch','transformers','sklearn'} for name in sys.modules)
'''
        subprocess.run([sys.executable,'-c',code],cwd=e.ROOT,check=True,capture_output=True)
