"""Read-only E v2 metadata adapter. No generation, judgments or representation I/O."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from src.checkpoint_r2_fresh_inputs import ROOT, file_hash, hash_value, require, text_hash

BASE = '754ff1d7d362cf76ddb2bca681b32b2aeb7118f5'
E = 'data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v2'
V5 = 'data/clean_protocol/selection_repair_v1/candidate_overlay_v5'
CONDITIONS = ('bare', 'or_explicit_or_both_v1', 'or_at_least_one_v1', 'and_both_following_v1')


def read(path):
    return json.loads(Path(path).read_text())


def csv_rows(path):
    with Path(path).open(newline='') as f:
        return list(csv.DictReader(f))


def truth(value):
    require(value in ('True', 'False', '0', '1', 0, 1), 'invalid frozen Boolean')
    return int(value in ('True', '1', 1))


def adapt_compound(row, pair):
    a, b = truth(row['canonical_truth_a']), truth(row['canonical_truth_b'])
    require(row['ordering'] in ('AB', 'BA') and row['operator'] in ('AND', 'OR'), 'order/operator')
    first, second = (a, b) if row['ordering'] == 'AB' else (b, a)
    entities = (row['entity_a_id'], row['entity_b_id'])
    surface = entities if row['ordering'] == 'AB' else entities[::-1]
    require((row['surface_first_entity_id'], row['surface_second_entity_id']) == surface, 'surface identities')
    require((truth(row['surface_first_truth']), truth(row['surface_second_truth'])) == (first, second), 'surface targets')
    label = a & b if row['operator'] == 'AND' else a | b
    require(truth(row['compound_label']) == label, 'truth-cell label')
    require((pair['entity_a_id'], pair['entity_b_id']) == entities and pair['topic'] == row['topic'], 'pair endpoints')
    return dict(row, label=label, truth_a=a, truth_b=b, canonical_truth_a=a, canonical_truth_b=b, surface_first_truth=first, surface_second_truth=second, person_a=pair['person_a_key'], person_b=pair['person_b_key'])


class Metadata:
    def __init__(self, root=ROOT):
        self.root = Path(root)
        cfg = read(self.root/'config/final_e/preparation_v1.json')
        for name, expected in cfg['scientific_sha256'].items():
            require(file_hash(self.root/name) == expected, 'scientific input changed: '+name)
        self.manifest = read(self.root/E/'manifest.json')
        for name, expected in self.manifest['output_sha256'].items():
            require(file_hash(self.root/E/name) == expected, 'E v2 output changed: '+name)
        self.sources = {name: csv_rows(self.root/E/name) for name in (
            'bare_compounds.csv', 'wording_compounds.csv', 'and_both_following_v1.csv', 'admitted_atomic_E.csv')}
        self.sources['isolated_facts.json'] = read(self.root/E/'isolated_facts.json')
        # Historical manifest fields are not scientific-lock approval.
        self.pairs = csv_rows(self.root/E/'pairs.csv')
        pairs = {r['pair_id']: r for r in self.pairs}
        require(len(pairs) == len(self.pairs) == 450, 'E pair coverage')
        self.bindings = csv_rows(self.root/E/'extraction_bindings.csv')
        texts = csv_rows(self.root/E/'extraction_texts.csv')
        self.texts = [dict(id=r['text_id'], statement=r['statement'], statement_sha256=r['statement_sha256']) for r in texts]
        self.text_index = {r['id']: i for i, r in enumerate(self.texts)}
        require(len(self.text_index) == len(texts) == 19190 and len(self.bindings) == 19424, 'E inventory counts')
        require(len({b['binding_id'] for b in self.bindings}) == len(self.bindings), 'duplicate binding identity')
        self.lookup = {}
        counts = Counter()
        for b in self.bindings:
            key = (b['artifact'], b['logical_id'])
            require(key not in self.lookup, 'duplicate logical/source identity')
            source = self.sources[b['artifact']][int(b['record_index'])]
            logical = source['fact_id'] if b['kind'] == 'isolated' else source['source_row_id'] if b['kind'] == 'atomic' else source['example_id']
            require(logical == b['logical_id'], 'source record/order identity')
            t = self.texts[self.text_index[b['text_id']]]
            require(text_hash(source['statement']) == b['statement_sha256'] == t['statement_sha256'] and
                    source['statement'] == t['statement'] and t['id'] == 'text_'+t['statement_sha256'], 'text binding')
            for field in ('fact_id', 'pair_id', 'fact_a_id', 'fact_b_id', 'source_row_id', 'entity_id', 'person_key'):
                if b[field]:
                    require(b[field] == source[field], 'logical metadata '+field)
            self.lookup[key] = b
            counts[b['text_id']] += 1
        require(all(counts[r['text_id']] == int(r['binding_count']) for r in texts), 'dedup counts')
        eligible = {r['source_row_id']: r for r in csv_rows(self.root/V5/'source_fact_eligibility.csv')}
        statuses = {r['source_row_id']: r for r in csv_rows(self.root/V5/'row_manifest.csv')}
        atoms = self.sources['admitted_atomic_E.csv']
        expected = {k for k, r in eligible.items() if r['split'] == 'test' and truth(r['e_current_eligible'])}
        require({r['source_row_id'] for r in atoms} == expected and len(atoms) == 974, 'current v5 atomic membership')
        for r in atoms:
            s = statuses[r['source_row_id']]
            require(s['e_successor_status'] == 'admitted' and all(s[k] == r[k] for k in
                    ('person_key', 'entity_id', 'label', 'statement_sha256', 'statement')), 'v5 atomic eligibility/binding')
        bare = self.sources['bare_compounds.csv']
        require(len(bare) == 7200 and len(self.sources['isolated_facts.json']) == 450, 'compound/fact counts')
        cells = defaultdict(set)
        for r in bare:
            cells[r['pair_id']].add((r['operator'], r['ordering'], truth(r['canonical_truth_a']), truth(r['canonical_truth_b'])))
        require(all(len(c) == 16 for c in cells.values()) and set(cells) == set(pairs), 'complete truth cells/orders')
        self.rows = {'atomic_E': [dict(r, label=truth(r['label'])) for r in atoms]}
        self.group_bindings = {'atomic_E': [self.lookup['admitted_atomic_E.csv', r['source_row_id']] for r in atoms]}
        index = csv_rows(self.root/E/'condition_index.csv')
        require(len(index) == 28800, 'condition index coverage')
        source_lookup = {(name, r['example_id']): r for name, rs in self.sources.items()
                         if name.endswith('.csv') and name != 'admitted_atomic_E.csv' for r in rs}
        for condition in CONDITIONS:
            entries = [r for r in index if r['condition_id'] == condition]
            require([r['base_example_id'] for r in entries] == [r['example_id'] for r in bare], 'shared wording support/order')
            rows, bindings = [], []
            for entry, original in zip(entries, bare):
                r = source_lookup[entry['artifact'], entry['selected_example_id']]
                require(entry['pair_id'] == original['pair_id'] == r['pair_id'] and entry['operator'] == r['operator'], 'wording identity')
                for k in ('canonical_truth_a', 'canonical_truth_b', 'ordering', 'fact_a_id', 'fact_b_id', 'compound_label'):
                    require(original[k] == r[k], 'wording changed target '+k)
                rows.append(dict(adapt_compound(r, pairs[r['pair_id']]), base_example_id=original['example_id']))
                bindings.append(self.lookup[entry['artifact'], r['example_id']])
            self.rows[condition], self.group_bindings[condition] = rows, bindings
        compound_people = {r[k] for r in self.pairs for k in ('person_a_key', 'person_b_key')}
        atomic_people = {r['person_key'] for r in atoms}
        entities = {r[k] for r in self.pairs for k in ('entity_a_id', 'entity_b_id')}
        require(len(entities) == 225, 'compound entity count')
        self.receipt = dict(schema='final-e-metadata-v1', base=BASE, manifest_sha256=file_hash(self.root/E/'manifest.json'),
            logical_bindings=len(self.bindings), unique_texts=len(texts), deduplication_savings=len(self.bindings)-len(texts),
            group_rows={g: len(v) for g, v in self.rows.items()}, compound_entities=len(entities), compound_pairs=len(pairs),
            isolated_facts=450, added_wording_rows={c: 3600 for c in CONDITIONS[1:]}, topics=self.manifest['counts'],
            atomic_people=len(atomic_people), compound_people=len(compound_people), shared_people=len(atomic_people & compound_people),
            atomic_only_people=len(atomic_people-compound_people), compound_only_people=len(compound_people-atomic_people),
            binding_sha256=hash_value(self.bindings), text_sha256=hash_value(self.texts),
            ordered_groups={g: hash_value(b) for g, b in self.group_bindings.items()},
            eligibility='current v5; historical P_A_frozen field superseded by reviewed checkpoint', E_outputs_opened=False)

    def indices(self, group):
        return [self.text_index[b['text_id']] for b in self.group_bindings[group]]
