"""Synthetic atomic facts only; no compound generation or real test facts."""

import csv
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from compound_fixtures import synthetic_fixture, write_manifest
from src.clean_compounds import ATOMIC_TEMPLATES, CompoundGenerator, TEMPLATE_VERSION
from src.entity_partitions import csv_bytes, entity_id, save_outputs, sha256
from src.inventor_country_audit import audit_migration, inventor_source_records, migrate_registry
from src.inventor_country_semantics import (
    SEMANTICS_VERSION, classify_inventor_candidate, country_components, eligible_false_country,
)
from src.negative_audit import build_audit
from src.negative_review_batches import build_review_batch
from src.validated_negatives import AUDIT_FILES, apply_reviews, build_registry, load_registry

ROOT = Path(__file__).resolve().parents[1]


class InventorCountryTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        _, records, sources = synthetic_fixture(self.root)
        names = ['Henry Ford', 'Hans von Ohain', 'George Washington Carver', 'Luther Simjian', 'Daniel Fahrenheit', 'Only Germany']
        values = ['the U.S', 'the U.S', 'the U.S', 'Turkey/the U.S', 'Poland/Germany', 'Germany']
        atomic = []
        for row in records:
            if row['topic'] != 'inventors':
                continue
            index = int(row['entity'].split('-')[-1])
            if index < 6:
                row.update(entity=names[index], entity_id=entity_id('inventors', names[index]), split='validation')
                value = values[index]
            else:
                # Invalid components on test rows must never reach the semantic parser.
                value = '/DO-NOT-INTERPRET/' if row['split'] == 'test' else 'Object-8'
            atomic.append({'statement': ATOMIC_TEMPLATES['inventors'].format(entity=row['entity'], object=value),
                           'label': int(row['compound_usable'])})
        atomic += [{'statement': 'Luther Simjian lived in the U.S.', 'label': 0},
                   {'statement': 'Daniel Fahrenheit lived in Germany.', 'label': 0}]
        (self.root / 'sources/inventors.csv').write_bytes(csv_bytes(atomic, ('statement', 'label')))
        for source in sources:
            source['sha256'] = sha256((self.root / 'sources' / source['file']).read_bytes())
        write_manifest(self.root, records, sources)
        files, _ = build_audit(self.root / 'sources', self.root / 'manifest.csv', self.root / 'manifest_metadata.json')
        save_outputs(self.root / 'audit', files)
        self.config = json.loads((ROOT / 'config/clean_protocol/validated_negatives.json').read_text())
        self.config.update(entity_manifest='manifest.csv', entity_manifest_metadata='manifest_metadata.json', source_audit_dir='audit',
                           source_audit_files={name: sha256(files[name]) for name in AUDIT_FILES})
        original = build_registry(self.config, self.root, 'validation')
        noninventor = next(r for r in original.rows if r['topic'] == 'cities')
        self.old = apply_reviews(original, [{'fact_id': noninventor['fact_id'], 'validation_status': 'externally_validated_false',
                                            'evidence_source': 'synthetic', 'evidence_note': 'Synthetic test only.',
                                            'reviewer_or_method': 'synthetic', 'validation_version': 'synthetic-v1'}], 'synthetic-v1')
        self.semantic = {**self.config, 'registry_version': 'validated-negative-registry-v2',
                         'inventor_country_semantics': SEMANTICS_VERSION, 'inventor_source_file': 'sources/inventors.csv'}

    def test_parser_only_normalizes_whitespace(self):
        self.assertEqual(country_components(' Poland /  Germany '), ('Poland', 'Germany'))
        self.assertEqual(country_components('Turkey/ the  U.S'), ('Turkey', 'the U.S'))
        self.assertNotEqual(country_components('US'), country_components('the U.S'))
        self.assertNotEqual(country_components('Turkey'), country_components('Türkiye'))
        self.assertNotEqual(country_components('germany'), country_components('Germany'))
        for value in ('', '/Germany', 'Poland/', 'Poland//Germany'):
            with self.assertRaises(ValueError):
                country_components(value)

    def test_any_known_true_component_wins_over_false_labels(self):
        self.assertEqual(classify_inventor_candidate('Turkey/the U.S', {'the U.S'}, {'Turkey/the U.S'}), 'invalid_known_true')
        self.assertEqual(classify_inventor_candidate('the U.S', {'Turkey/the U.S'}, {'the U.S'}), 'invalid_known_true')
        self.assertFalse(eligible_false_country('Poland/Germany', {'the U.S'}))
        self.assertEqual(classify_inventor_candidate('Poland/Germany', {'the U.S'}, set()), 'unverified_negative')
        self.assertEqual(classify_inventor_candidate('Poland', {'the U.S'}, {'Poland/Germany'}), 'unverified_negative')

    def test_migration_preserves_noninventors_and_uses_single_country_pool(self):
        new = migrate_registry(self.old, self.semantic, self.root)
        for topic in ('cities', 'sp_en_trans', 'element_symb', 'animal_class'):
            self.assertEqual([r for r in self.old.rows if r['topic'] == topic], [r for r in new.rows if r['topic'] == topic])
        inventors = [r for r in new.rows if r['topic'] == 'inventors']
        self.assertTrue(all('/' not in r['candidate_object'] for r in inventors))
        self.assertIn('Turkey', {r['candidate_object'] for r in inventors})
        self.assertIn('Poland', {r['candidate_object'] for r in inventors})
        for entity, forbidden in [('Henry Ford', 'Turkey/the U.S'), ('Hans von Ohain', 'Turkey/the U.S'),
                                   ('George Washington Carver', 'Poland/Germany'), ('Luther Simjian', 'the U.S'),
                                   ('Daniel Fahrenheit', 'Germany')]:
            self.assertFalse(any(r['entity'] == entity and r['candidate_object'] == forbidden for r in inventors))
        self.assertFalse(any(r['validation_status'] == 'externally_validated_false' for r in inventors))
        old = {r['fact_id']: r for r in self.old.rows}
        for r in inventors:
            if r['fact_id'] in old:
                self.assertEqual(r['ranking_hash'], old[r['fact_id']]['ranking_hash'])
                self.assertEqual(r['validation_status'], old[r['fact_id']]['validation_status'])

    def test_deterministic_migration_reload_and_shuffle_invariance(self):
        new = migrate_registry(self.old, self.semantic, self.root)
        self.assertEqual(new.files(), migrate_registry(self.old, self.semantic, self.root).files())
        save_outputs(self.root / 'registry', new.files(parent_sha256=self.old.digest))
        self.assertEqual(new.digest, load_registry(self.semantic, self.root, 'validation', self.root / 'registry').digest)
        for name in ('entity_knowledge.csv', 'candidate_audit.csv'):
            path = self.root / 'audit' / name
            reader = csv.DictReader(io.StringIO(path.read_text()))
            rows = list(reader)
            path.write_bytes(csv_bytes(rows[::-1], reader.fieldnames))
            self.semantic['source_audit_files'][name] = sha256(path.read_bytes())
        shuffled = migrate_registry(self.old, self.semantic, self.root)
        self.assertEqual(new.rows, shuffled.rows)

    def test_audit_reports_raw_values_overlap_and_removed_source_support(self):
        new = migrate_registry(self.old, self.semantic, self.root)
        files, summary = audit_migration(self.old, new, self.root / 'sources/inventors.csv')
        self.assertEqual(len(summary['specific_exclusions']), 3)
        self.assertTrue(summary['removed_previously_accepted_candidates'])
        self.assertTrue(any(r['entity'] == 'Luther Simjian' for r in summary['newly_unavailable_entities']))
        records = list(csv.DictReader(io.StringIO(files['source_objects.csv'].decode())))
        self.assertTrue(any(r['raw_object'] == 'Turkey/the U.S' and r['source_label'] == '1' for r in records))
        self.assertFalse(any(r['split'] == 'test' for r in records))

    def test_true_atomic_facts_preserved_and_legacy_unsafe_registry_fails(self):
        before = (self.root / 'sources/inventors.csv').read_bytes()
        gen = CompoundGenerator(self.old.manifest, self.root / 'sources', 'validation', 0, TEMPLATE_VERSION)
        facts = gen._load_facts('inventors')  # Atomic constituents only; no compounds generated.
        for (identity, truth), fact in facts.items():
            if truth:
                if self.old.manifest.entities[identity].entity == 'Luther Simjian':
                    self.assertEqual(fact.statement, 'Luther Simjian lived in Turkey/the U.S.')
            else:
                self.assertNotIn('/', fact.statement)
        legacy = CompoundGenerator(self.old.manifest, self.root / 'sources', 'validation', 0, TEMPLATE_VERSION, negative_registry=self.old)
        with self.assertRaisesRegex(ValueError, 'unsafe inventor negative'):
            legacy._load_facts('inventors')
        self.assertEqual(before, (self.root / 'sources/inventors.csv').read_bytes())

    def test_inventor_only_batch_scope_and_test_guards(self):
        new = migrate_registry(self.old, self.semantic, self.root)
        ids = {r['entity_id'] for r in new.rows if r['entity'] in ('Henry Ford', 'Hans von Ohain')}
        files = build_review_batch(new, self.root / 'audit', topics=('inventors',), entity_ids=ids)
        rows = list(csv.DictReader(io.StringIO(files['review_batch.csv'].decode())))
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r['topic'] == 'inventors' and '/' not in r['candidate_object'] for r in rows))
        with patch.object(self.old, 'split', 'test'), patch.object(Path, 'read_bytes', side_effect=AssertionError('test facts read')):
            with self.assertRaisesRegex(ValueError, 'validation-only'):
                migrate_registry(self.old, self.semantic, self.root)
        result = subprocess.run(['python3', '-B', str(ROOT / 'scripts/25_rebuild_inventor_negatives.py'), '--split', 'test'],
                                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('validation-only', result.stderr)


if __name__ == '__main__':
    unittest.main()
