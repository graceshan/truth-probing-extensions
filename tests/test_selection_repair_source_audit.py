"""Synthetic audit invariants plus immutable source-provenance checks; no models."""
import copy
import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.selection_repair_source_audit import (
    CONFIG, FIELDS, alias_map, apply_overlay, countries, csv_payload, digest,
    load_sources, negation, normalized_name, review_sample, run,
)

ROOT=Path(__file__).resolve().parents[1]


def pair(name, split, country, label, index):
    base=dict(entity=name,entity_id='entity_'+name,split=split,topic='inventors',
              raw_country=country,normalized_countries=[country], row_index=index)
    a=dict(base,source_row_id=f'inventors:{index}',dataset='inventors',form='affirmative',label=label,
           paired_source_row_id=f'neg_inventors:{index}')
    b=dict(base,source_row_id=f'neg_inventors:{index}',dataset='neg_inventors',form='negated',label=1-label,
           paired_source_row_id=f'inventors:{index}')
    return [a,b]


def config():
    return dict(alias_groups=[],positive_country_evidence=[],country_component_mapping={'US':'US'})


def evidence():
    return {'bio':dict(status='confirmed_identity',url='https://example.org/biography',excerpt='A Full Name is also A Name.')}


class AuditTests(unittest.TestCase):
    def test_country_mapping_explicit(self):
        mapping=json.loads((ROOT/CONFIG).read_text())['country_component_mapping']
        self.assertEqual(countries(' Austria /the U.S/the U.S.',mapping),['Austria','United States'])
        self.assertNotEqual(countries('England',mapping),countries('the U.K',mapping))
        with self.assertRaises(ValueError): countries('Atlantis',mapping)

    def test_paired_exclusion_preserves_labels_and_inputs(self):
        rows=pair('A','train','US',1,0)+pair('A','train','US',0,1)+pair('A','train','Canada',0,2)
        original=copy.deepcopy(rows)
        out,_=apply_overlay(rows,config(),{})
        self.assertEqual(rows,original)
        self.assertEqual([r['status'] for r in out],['admitted','admitted','excluded','excluded','admitted','admitted'])
        self.assertEqual([r['label'] for r in out],[r['label'] for r in original])
        self.assertEqual(out[3]['positive_support_row_ids'],['inventors:0'])

    def test_cross_partition_quarantines_every_form_including_e(self):
        rows=pair('A Name','train','US',1,0)+pair('A Full Name','validation','US',1,1)+pair('A Name','test','France',0,2)
        cfg=config();cfg['alias_groups']=[dict(names=['A Name','A Full Name'],status='confirmed',evidence_id='bio')]
        out,decisions=apply_overlay(rows,cfg,evidence())
        self.assertTrue(all(r['status']=='excluded' for r in out))
        self.assertEqual([r['split'] for r in out],[r['split'] for r in rows])
        self.assertEqual(len({r['person_key'] for r in out}),1)
        self.assertEqual(decisions[0]['action'],'quarantine_all_partitions')

    def test_same_partition_alias_grouping_and_support(self):
        rows=pair('A Name','train','US',1,0)+pair('A Full Name','train','US',0,1)
        cfg=config();cfg['alias_groups']=[dict(names=['A Name','A Full Name'],status='confirmed',evidence_id='bio')]
        out,_=apply_overlay(rows,cfg,evidence())
        self.assertEqual(out[0]['person_key'],out[2]['person_key'])
        self.assertEqual([r['status'] for r in out],['admitted','admitted','excluded','excluded'])
        with self.assertRaises(ValueError):apply_overlay(rows,cfg,{})

    def test_normalized_candidates_do_not_prove_identity(self):
        rows=pair('A Name','train','US',1,0)+pair('A Full Name','test','US',1,1)
        out,decisions=apply_overlay(rows,config(),{})
        self.assertTrue(all(r['status']=='admitted' for r in out))
        self.assertNotEqual(out[0]['person_key'],out[2]['person_key'])
        self.assertEqual(decisions[0]['status'],'unresolved')
        self.assertEqual(normalized_name('A  NAME'),('a','name'))

    def test_rejected_review_not_positive(self):
        cfg=config();cfg['positive_country_evidence']=[dict(status='rejected_true_or_ambiguous',entity='A',country='US',evidence_id='bio')]
        with self.assertRaises(ValueError):apply_overlay(pair('A','train','US',0,0),cfg,evidence())
        out,_=apply_overlay(pair('A','train','US',0,0),config(),{})
        self.assertEqual(out[0]['status'],'admitted')

    def test_sampling_deterministic_simple_random_not_quota(self):
        cfg=json.loads((ROOT/CONFIG).read_text())
        rows=sum((pair(str(i),('train','validation','test')[i%3],'US',0,i) for i in range(100)),[])
        out,_=apply_overlay(rows,config(),{})
        frame,q=review_sample(out,cfg['sampling'])
        frame2,q2=review_sample(list(reversed(out)),cfg['sampling'])
        self.assertEqual(frame,frame2);self.assertEqual(q,q2)
        self.assertEqual(len(frame),100);self.assertEqual(len(q),30)
        self.assertEqual(len({r['source_row_id'] for r in q}),30)
        self.assertTrue(all(r['review_status']=='pending' for r in q))
        self.assertEqual([r['sampling_frame_index'] for r in q][:5],[99, 59, 16, 58, 9])
        with self.assertRaises(ValueError): review_sample(out[:20],cfg['sampling'])

    def test_original_sources_and_exact_pair_identity(self):
        cfg=json.loads((ROOT/CONFIG).read_text())
        paths=list((ROOT/cfg['source_dir']).glob('*.csv'))
        before={p:digest(p.read_bytes()) for p in paths}
        rows,_=load_sources(ROOT,cfg)
        lookup={r['source_row_id']:r for r in rows}
        self.assertEqual(set(r['split'] for r in rows),{'train','validation','test'})
        for r in rows:
            self.assertEqual(r['statement_sha256'],digest(r['statement'].encode()))
            self.assertEqual(json.loads(r['original_row_json'])['statement'],r['statement'])
            self.assertEqual(r['source_row'],r['row_index']+1)
            p=lookup[r['paired_source_row_id']]
            self.assertEqual(p['label'],1-r['label'])
            if r['form']=='affirmative':self.assertEqual(negation(r['statement'],r['topic']),p['statement'])
        self.assertEqual(before,{p:digest(p.read_bytes()) for p in paths})

    def test_csv_roundtrip_exact_statement(self):
        value='Émile lived in the U.S.\n'
        payload=csv_payload([dict(statement=value,normalized_countries=['United States'])],FIELDS)
        import io
        row=next(csv.DictReader(io.StringIO(payload.decode())))
        self.assertEqual(row['statement'],value)

    def test_known_alias_cases_and_manifest_derived_counts(self):
        from src.selection_repair_source_audit import counts
        cfg=json.loads((ROOT/CONFIG).read_text())
        rows,_=load_sources(ROOT,cfg)
        ev=json.loads((ROOT/Path(CONFIG).parent/'identity_evidence.json').read_text())
        out,decisions=apply_overlay(rows,cfg,ev)
        sim=[r for r in out if 'Simjian' in r['entity']]
        far=[r for r in out if 'Farnsworth' in r['entity']]
        self.assertTrue(all(r['status']=='excluded' for r in sim))
        self.assertEqual(len({r['person_key'] for r in far}),1)
        self.assertTrue(all(r['split']=='train' for r in far))
        table=counts(out)
        total={r['stage']:r['rows'] for r in table if r['topic']=='ALL' and r['split']=='ALL' and r['label']=='ALL'}
        self.assertEqual(total['before'],total['admitted']+total['excluded'])
        self.assertEqual(total['before'],len(rows))

    def test_immutable_outputs_and_tamper_detection(self):
        import shutil
        cfg=json.loads((ROOT/CONFIG).read_text())
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for directory in (cfg['source_dir'],cfg['partition_dir'],str(Path(CONFIG).parent)):
                shutil.copytree(ROOT/directory,root/directory)
            for filename in ('src/selection_repair_source_audit.py','scripts/46_audit_selection_repair_sources.py','src/data.py','src/entity_partitions.py'):
                (root/filename).parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(ROOT/filename,root/filename)
            before={p.relative_to(root):digest(p.read_bytes()) for p in root.rglob('*') if p.is_file()}
            first=run(root)
            self.assertEqual(first,run(root))
            self.assertEqual(before,{p:digest((root/p).read_bytes()) for p in before})
            out=root/cfg['output_dir']
            for name,receipt in first['outputs'].items():
                self.assertEqual(digest((out/name).read_bytes()),receipt['sha256'])
            (out/'counts.csv').write_text('tampered')
            with self.assertRaisesRegex(ValueError,'no overwrite'):run(root)
            source=root/cfg['source_dir']/'inventors.csv'
            source.write_bytes(source.read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'Source hash mismatch'):run(root)

    def test_no_historical_loader_activation_or_prediction_interface(self):
        import ast
        tree=ast.parse((ROOT/'src/selection_repair_source_audit.py').read_text())
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        self.assertFalse(any(x and any(t in x for t in ['probe','compound','extract','validated_negatives']) for x in imports))


if __name__=='__main__':unittest.main()
