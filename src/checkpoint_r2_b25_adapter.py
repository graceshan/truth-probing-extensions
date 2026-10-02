"""Frozen fresh B25, fact and wording bindings; no historical parameter reuse."""
import csv
import json
from pathlib import Path

import numpy as np

from src.checkpoint_r2_fresh_adapter import FreshAdapter
from src.checkpoint_r2_fresh_inputs import ROOT, hash_value, require
from src.checkpoint_r2_common_v1 import FrozenBindings
from src.checkpoint_r2_full_inputs import WORDING
from src import selection_repair_objectives as objective

SEEDS = (11, 23, 37)
EVAL = ('atomic_D', 'D_bare')


def resolve(group, sources, manifest, logical_key):
    """Preserve source order and observations; resolve only exact hashed identities."""
    lookup = {(b['group'], b['logical_id']): b for b in manifest['bindings']}
    texts = {r['id']: i for i, r in enumerate(manifest['rows'])}
    require(len(lookup) == len(manifest['bindings']), 'duplicate logical binding')
    result = []
    for source in sources:
        logical = source[logical_key]
        require((group, logical) in lookup, 'missing fresh binding')
        b = lookup[group, logical]
        require(b['source_metadata_sha256'] == hash_value(source), 'source metadata hash')
        require(b['statement'] == source['statement'], 'source statement binding')
        result.append(dict(group=group, logical_id=logical, text_id=b['text_id'],
                           text_row=texts[b['text_id']], inventory_offset=b['inventory_offset'],
                           source_metadata_sha256=b['source_metadata_sha256']))
    require(len({(r['group'], r['logical_id']) for r in result}) == len(result), 'duplicate logical observation')
    require([r['inventory_offset'] for r in result] == sorted(r['inventory_offset'] for r in result), 'source order changed')
    return result


def validate_fold(rows, train, heldout, facts):
    require(set(train).isdisjoint(heldout) and sorted(list(train) + list(heldout)) == list(range(len(rows))), 'fold row partition')
    people = lambda indices: {rows[i][k] for i in indices for k in ('person_a', 'person_b')}
    fact_ids = lambda indices: {rows[i][k] for i in indices for k in ('fact_a_id', 'fact_b_id')}
    require(not people(train) & people(heldout), 'held-out entity leakage')
    require(not fact_ids(train) & fact_ids(heldout), 'held-out fact leakage')
    require(all(facts[f]['person_key'] not in people(heldout) for f in fact_ids(train)), 'held-out fact person leakage')


def compound_block(X, rows, pre):
    require(all(r['split'] == 'train' and r['group'].startswith('B25_') for r in rows), 'only allocated B25 training rows')
    weights = objective.compound_pair_weights([r['pair_id'] for r in rows], [r['operator'] for r in rows],
        [(int(r['truth_a']), int(r['truth_b'])) for r in rows], [0 if r['ordering'] == 'AB' else 1 for r in rows])
    return objective.prepare_block(X, [int(r['label']) for r in rows], pre, weights)


class B25Adapter(FreshAdapter):
    def __init__(self, acceptance):
        super().__init__(acceptance['output'], acceptance)
        base = ROOT / 'data/checkpoint_r2_v1'
        self.frozen = FrozenBindings(ROOT)
        self.facts = self.frozen.facts
        self.pairs = self.frozen.pairs
        groups = json.loads((base / 'pair_groups.json').read_text())
        compounds = list(csv.DictReader((base / 'compound_bindings.csv').open()))
        for seed in SEEDS:
            group = f'B25_{seed}'
            self.sources[group] = [r for r in compounds if r['group'] == group]
            self.rows[group] = [self.frozen.adapt_row(r) for r in self.sources[group]]
            self.bindings[group] = resolve(group, self.sources[group], self.manifest, 'example_id')
        self.sources['TC_isolated'] = [self.facts[b['logical_id']] for b in self.manifest['bindings'] if b['group'] == 'TC_isolated']
        self.rows['TC_isolated'] = [dict(r, label=int(r['truth']), group='TC_isolated', logical_id=r['fact_id']) for r in self.sources['TC_isolated']]
        self.bindings['TC_isolated'] = resolve('TC_isolated', self.sources['TC_isolated'], self.manifest, 'fact_id')
        # Recreate wording metadata exactly as the producer did, before accessing arrays.
        from src.checkpoint_r2_inputs import uid
        from src.final_e_preparation import render_wording
        from src.heldout_and_wording import render as render_and
        bare = [r for r in compounds if r['group'] == 'D_bare']
        for group in sorted(WORDING):
            sources = []
            for r in bare:
                if group.startswith('and_') != (r['operator'] == 'AND'):
                    continue
                first, second = (r['fact_a_id'], r['fact_b_id']) if r['ordering'] == 'AB' else (r['fact_b_id'], r['fact_a_id'])
                a, b = self.facts[first]['statement'], self.facts[second]['statement']
                statement = render_and(a, b, 'AND') if r['operator'] == 'AND' else render_wording(group, a, b)
                sources.append(dict(r, statement=statement, group=group))
            self.sources[group] = sources
            # Producer source metadata retains bare example_id; logical wording ID is distinct.
            resolved = [dict(r, logical_id=uid('r2wording', r['example_id'], group)) for r in sources]
            lookup = {(b['group'], b['logical_id']): b for b in self.manifest['bindings']}
            texts = {r['id']: i for i, r in enumerate(self.manifest['rows'])}
            self.bindings[group] = []
            self.rows[group] = []
            for r, source in zip(resolved, sources):
                b = lookup[group, r['logical_id']]
                require(b['source_metadata_sha256'] == hash_value(source) and b['statement'] == r['statement'], 'wording source binding')
                self.bindings[group].append(dict(group=group, logical_id=r['logical_id'], text_id=b['text_id'], text_row=texts[b['text_id']], inventory_offset=b['inventory_offset'], source_metadata_sha256=b['source_metadata_sha256']))
                self.rows[group].append(dict(r, label=int(r['label']), truth_a=int(r['truth_a']), truth_b=int(r['truth_b'])))
            require(len(self.rows[group]) == 3864, 'wording coverage')
        members = json.loads((base / 'memberships.json').read_text())
        p_lookup = {r['source_row_id']: i for i, r in enumerate(self.rows['P15'])}
        self.balanced = [p_lookup[r['source_row_id']] for r in members['balanced']]
        require([self.rows['P15'][i] for i in self.balanced] == members['balanced'] and len(self.balanced) == 700, 'exact balanced sampler/order')
        self.balanced_rows = [dict(r, dataset=r['source_row_id'].rsplit(':', 1)[0]) for r in members['balanced']]
        isolated = {r['fact_id']: i for i, r in enumerate(self.rows['TC_isolated'])}
        self.allocations = {}
        a_people = {r['person_key'] for r in json.loads((base / 'entities.json').read_text()) if r['A15']}
        for seed in SEEDS:
            group = f'B25_{seed}'; rows = self.rows[group]; pairs = groups[group]
            require(len(rows) == 400 and len(pairs) == 25 and len({r['pair_id'] for r in rows}) == 25, 'B25 pair/row counts')
            people = {r[k] for r in rows for k in ('person_a', 'person_b')}
            require(len(people) == 50 and people <= a_people and not people & {r['person_key'] for r in self.rows['P15']}, 'A/P entity exclusion')
            fold_of = {r['pair_id']: r['fold'] for r in pairs}
            folds = []
            for fold in range(5):
                held = [i for i, r in enumerate(rows) if fold_of[r['pair_id']] == fold]
                train = [i for i, r in enumerate(rows) if fold_of[r['pair_id']] != fold]
                require(len(train) == 320 and len(held) == 80 and len({rows[i]['topic'] for i in held}) == 5, 'five balanced folds')
                validate_fold(rows, train, held, self.facts)
                folds.append(dict(fold=fold, train=train, heldout=held, training_bindings_sha256=hash_value([self.bindings[group][i] for i in train]), heldout_bindings_sha256=hash_value([self.bindings[group][i] for i in held])))
            # Canonical pair/fact ordering for deduplication; all four exact affirmative facts/pair.
            fact_rows = []
            for entry in sorted(pairs, key=lambda r: r['pair_id']):
                pair = self.pairs[entry['pair_id']]
                for fact_id in sorted([pair[k][truth] for k in ('facts_a', 'facts_b') for truth in ('0', '1')]):
                    require(fact_id in isolated, 'allocated fact missing from fresh isolated bank')
                    fact_rows.append(dict(pair_id=entry['pair_id'], fact_id=fact_id, isolated_row=isolated[fact_id], label=int(self.facts[fact_id]['truth']), person_key=self.facts[fact_id]['person_key']))
            require(len(fact_rows) == len({r['fact_id'] for r in fact_rows}) == 100, 'exact B25 isolated exposure')
            self.allocations[seed] = dict(group=group, pairs=pairs, folds=folds, isolated=fact_rows, people=sorted(people),
                compound_binding_sha256=hash_value(self.bindings[group]), isolated_binding_sha256=hash_value(fact_rows))
        self.receipt.update(B25={str(s): self.allocations[s] for s in SEEDS}, balanced_binding_sha256=hash_value(members['balanced']),
            all_group_bindings={g: hash_value(b) for g, b in self.bindings.items()}, heldout_wording_rows={g: len(self.rows[g]) for g in sorted(WORDING)})

    def layer(self, model, layer, wording=False):
        # Omit unused full TC matrices during B25 fits. No wording array enters fitting/selection.
        groups = list(EVAL) + ['P15'] + (sorted(WORDING) if wording else ['TC_isolated'] + [f'B25_{s}' for s in SEEDS])
        original = self.bindings
        try:
            self.bindings = {g: original[g] for g in groups}
            return super().layer(model, layer)
        finally:
            self.bindings = original

    def isolated_block(self, X, seed, pre):
        rows = self.allocations[seed]['isolated']
        weights = objective.isolated_pair_weights([r['pair_id'] for r in rows], [r['fact_id'] for r in rows])
        return objective.prepare_block(X[[r['isolated_row'] for r in rows]], [r['label'] for r in rows], pre, weights)
