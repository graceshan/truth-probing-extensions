"""Frozen R2 metadata adapter and integration checks; no fitting or score-file IO.

The only selector entry points in this successor are synthetic validation APIs.
They are not authorization to execute the production protocol.
"""
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from src.checkpoint_r2_inputs import build, check, encoded, sha, uid, validate_row
from src.checkpoint_r2_selection_v1 import atomic_eligibility, select_atom_all, select_constituent_layer

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path('data/checkpoint_r2_v1')
CONTRACT = 'config/checkpoint_r2/common_integration_v1.json'
CONTRACT_SHA256 = '895ced573ad47af73e3bceaf5dd80a0b39693ca82367d0f5264b85de71755a57'
ROLES = {
    'constituent_source_validation': 'T_C_source_validation',
    'constituent_source_fit': 'T_C_source_fit',
    'TC_control_and_final_refit': 'full_T_C_refit',
    'D_bare': 'D_evaluation',
    'B25_11': 'B25_11', 'B25_23': 'B25_23', 'B25_37': 'B25_37',
}
BITS = ('truth_a', 'truth_b', 'surface_first_truth', 'surface_second_truth', 'label')


def binary(value):
    """Do not treat nonempty strings, floats, missing values as truth labels."""
    if type(value) is str and value in ('0', '1'):
        return int(value)
    if type(value) in (bool, int) and value in (0, 1):
        return int(value)
    raise ValueError('expected literal 0/1 or native binary bool/int')


def verify_hashes(root, expected):
    for name, digest in expected.items():
        check(sha((root / name).read_bytes()) == digest, 'hash mismatch: ' + name)


def contract(root=ROOT):
    data = (root / CONTRACT).read_bytes()
    check(sha(data) == CONTRACT_SHA256, 'integration contract hash mismatch')
    cfg = json.loads(data)
    verify_hashes(root, {p: v['sha256'] for p, v in cfg['bindings'].items()})
    manifest = json.loads((root / PACKAGE / 'manifest.json').read_bytes())
    verify_hashes(root, manifest['input_code_config_sha256'])
    verify_hashes(root, {str(PACKAGE / p): v for p, v in manifest['output_sha256'].items()})
    return cfg


def membership_receipt(root=ROOT, *, rebuild=False):
    cfg = contract(root)
    manifest = json.loads((root / PACKAGE / 'manifest.json').read_bytes())
    check(manifest['ordered_v4_v5_equivalence_hashes'] == cfg['ordered_v4_v5_equivalence_hashes'],
          'ordered identity receipt mismatch')
    members = json.loads((root / PACKAGE / 'memberships.json').read_bytes())
    entities = json.loads((root / PACKAGE / 'entities.json').read_bytes())
    for name, rows in members.items():
        check(sha(encoded(rows)) == cfg['ordered_v4_v5_equivalence_hashes'][name],
              'ordered membership mismatch: ' + name)
        check(len({r['source_row_id'] for r in rows}) == len(rows), 'duplicate source identity')
    a15 = {r['person_key'] for r in entities if r['A15'] is True}
    check(members['P15'] == [r for r in members['outer_train'] if r['person_key'] not in a15],
          'P15 all-variant exclusion mismatch')
    check({r['source_row_id'] for r in members['balanced']} <= {r['source_row_id'] for r in members['P15']},
          'balanced membership leakage')
    with (root / PACKAGE / 'compound_bindings.csv').open() as f:
        d_pairs = {r['pair_id'] for r in csv.DictReader(f) if r['group'] == 'D_bare'}
    counts = {k: len(v) for k, v in members.items()}
    counts.update(A15=len(a15), TC=len(entities), compound_D_pairs=len(d_pairs))
    check(counts == cfg['adopted_memberships'], 'membership counts')
    if rebuild:
        outputs, rebuilt = build(root)
        for name, data in outputs.items():
            check((root / PACKAGE / name).read_bytes() == data, 'frozen rebuild mismatch: ' + name)
        check(rebuilt['ordered_v4_v5_equivalence_hashes'] == cfg['ordered_v4_v5_equivalence_hashes'],
              'v4/v5 rebuild mismatch')
    return dict(counts=counts, ordered_v4_v5_equivalence_hashes=cfg['ordered_v4_v5_equivalence_hashes'],
                upstream_metadata_rebuilt=rebuild, historical_fit_provenance='v4',
                conditional_P15_reuse_matches=36, historical_fits_newly_admitted=0)


class FrozenBindings:
    """Adapt exact frozen CSV occurrences without collapsing shared examples."""
    def __init__(self, root=ROOT):
        self.root = root
        self.cfg = contract(root)
        def read(name):
            return json.loads((root / PACKAGE / name).read_bytes())
        self.facts, self.pairs = read('facts.json'), read('pairs.json')
        self.groups = {g: {r['pair_id'] for r in rows} for g, rows in read('pair_groups.json').items()}
        with (root / PACKAGE / 'constituent_split.csv').open() as f:
            self.person_roles = {(r['topic'], r['person_key']): r['role'] for r in csv.DictReader(f)}
        with (root / PACKAGE / 'compound_bindings.csv').open() as f:
            frozen = list(csv.DictReader(f))
        # D IDs are authoritative frozen identities, not regenerated from text.
        self.d_rows = {r['example_id']: r for r in frozen if r['group'] == 'D_bare'}
        self._rows = [self.adapt_row(r) for r in frozen]
        keys = [(r['group'], r['example_id']) for r in self._rows]
        check(len(keys) == len(set(keys)), 'duplicate group/example binding')
        cells = defaultdict(list)
        for r in self._rows:
            cells[r['group'], r['pair_id']].append((r['truth_a'], r['truth_b'], r['operator'], r['ordering']))
        expected = set(product((0, 1), (0, 1), ('AND', 'OR'), ('AB', 'BA')))
        check(all(len(v) == 16 and set(v) == expected for v in cells.values()), 'incomplete pair group')

    def adapt_row(self, original):
        row = dict(original)
        for key in BITS:
            row[key] = binary(row[key])
        check(row['group'] in ROLES, 'unknown group')
        for key in ('fact_a_id', 'fact_b_id'):
            check(row[key] in self.facts, 'unknown fact')
            fact = self.facts[row[key]]
            check(type(fact['eligible']) is bool and fact['eligible'], 'ineligible fact')
            binary(fact['truth'])
            check(sha(fact['statement'].encode()) == fact['statement_sha256'], 'fact statement hash')
        validate_row(row, self.facts)
        if row['group'] == 'D_bare':
            check(row['example_id'] in self.d_rows, 'unknown D example')
            expected = dict(self.d_rows[row['example_id']])
            for key in BITS:
                expected[key] = binary(expected[key])
            check(row == expected, 'authoritative D identity mismatch')
        else:
            check(row['pair_id'] in self.groups[row['group']], 'pair outside group')
            pair = self.pairs[row['pair_id']]
            check(all(row[k] == pair[k] for k in ('topic', 'person_a', 'person_b', 'split')), 'pair identity mismatch')
            check(row['fact_a_id'] == pair['facts_a'][str(row['truth_a'])] and
                  row['fact_b_id'] == pair['facts_b'][str(row['truth_b'])], 'pair fact mismatch')
            check(row['example_id'] == uid('r2example', row['pair_id'], row['fact_a_id'], row['fact_b_id'],
                                           row['operator'], row['ordering']), 'example identity mismatch')
            if row['group'].startswith('constituent_source_'):
                role = row['group'].removeprefix('constituent_')
                check(all(self.person_roles.get((row['topic'], row[p])) == role for p in ('person_a', 'person_b')),
                      'source person role mismatch')
        row.update(canonical_truth_a=row['truth_a'], canonical_truth_b=row['truth_b'],
                   binding_id=uid('r2binding', row['group'], row['example_id']), role=ROLES[row['group']])
        first, second = ('a', 'b') if row['ordering'] == 'AB' else ('b', 'a')
        for position, side in (('first', first), ('second', second)):
            row['surface_' + position + '_person_key'] = row['person_' + side]
            row['surface_' + position + '_fact_id'] = row['fact_' + side + '_id']
        return row

    def block(self, group, operator):
        check(group in ROLES and operator in ('AND', 'OR'), 'unknown block')
        rows = [dict(r) for r in self._rows if r['group'] == group and r['operator'] == operator]
        descriptor = dict(contract_sha256=CONTRACT_SHA256, group=group, operator=operator,
                          role=ROLES[group], rows=rows)
        return dict(group=group, operator=operator, role=ROLES[group],
                    metadata_sha256=sha(encoded(descriptor)),
                    ordered_keys=[[r['group'], r['example_id']] for r in rows], rows=rows)

    def align(self, block, packet):
        """Strict comparison, never silently reorder a score vector."""
        expected = self.block(block['group'], block['operator'])
        check(block == expected, 'stale or altered metadata block')
        check(packet['metadata_sha256'] == block['metadata_sha256'], 'score metadata binding mismatch')
        check(packet['ordered_keys'] == block['ordered_keys'], 'ordered score alignment mismatch')
        check(packet['score_kind'] == 'synthetic_validation', 'only synthetic validation authorized')
        scores = np.asarray(packet['scores'], dtype=np.float64)
        check(scores.shape == (len(block['rows']), 2) and np.isfinite(scores).all(), 'invalid score array')
        return scores

    def select_synthetic(self, block, packets_by_layer, *, source_operator):
        check(block['group'] == 'constituent_source_validation' and block['role'] == 'T_C_source_validation',
              'source-validation-only selection')
        check(block['operator'] == source_operator, 'target operator in source selection')
        scores = {layer: self.align(block, packet) for layer, packet in packets_by_layer.items()}
        return select_constituent_layer(source_operator=source_operator, role=block['role'],
                                        validation_binding=block['metadata_sha256'],
                                        rows=pd.DataFrame(block['rows']), scores_by_layer=scores)

    def receipt(self):
        return dict(rows=len(self._rows), unique_group_example_bindings=len(self._rows),
                    repeated_example_occurrences=len(self._rows) - len({r['example_id'] for r in self._rows}),
                    groups=dict(Counter(r['group'] for r in self._rows)),
                    blocks=[{k: v for k, v in self.block(g, op).items() if k not in ('rows', 'ordered_keys')}
                            for g in ROLES for op in ('AND', 'OR')])


def shared_atomic_bank(candidates, *, model, reduced_lr_atomic_auc):
    records = atomic_eligibility(candidates, model, reduced_lr_atomic_auc)
    payload = dict(model=model, reduced_lr_atomic_auc=reduced_lr_atomic_auc, eligibility=records,
                   eligible_ids=[r['candidate_id'] for r in records if r['eligible']])
    return dict(payload, binding_sha256=sha(encoded(payload)))


def select_atomic_synthetic(candidates, *, bank, s_all_eligible_ids):
    """One shared retention bank; S_all ranking and its separate tie rule untouched."""
    expected = shared_atomic_bank(candidates, model=bank['model'], reduced_lr_atomic_auc=bank['reduced_lr_atomic_auc'])
    check(bank == expected, 'stale shared atomic eligibility bank')
    check(sorted(s_all_eligible_ids) == bank['eligible_ids'], 'S_all shared eligibility mismatch')
    return select_atom_all(candidates, model=bank['model'], reduced_lr_atomic_auc=bank['reduced_lr_atomic_auc'],
                           expected_s_all_eligible_ids=s_all_eligible_ids)


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args]).decode().strip()


def preservation_receipt(root=ROOT):
    """Check worktree bytes against BOTH source trees, including historical files."""
    cfg = contract(root)
    expected, counts = {}, {}
    for commit, tree in cfg['source_trees'].items():
        check(git(root, 'rev-parse', commit + '^{tree}') == tree, 'source tree mismatch')
        check(subprocess.run(['git', '-C', str(root), 'merge-base', '--is-ancestor', commit, 'HEAD']).returncode == 0,
              'missing exact source ancestor')
        listing = git(root, 'ls-tree', '-r', commit).splitlines()
        counts[commit] = len(listing)
        for line in listing:
            meta, path = line.split('\t', 1)
            mode, kind, blob = meta.split()
            check(kind == 'blob' and mode in ('100644', '100755'), 'unsupported preservation object')
            check(path not in expected or expected[path] == blob, 'conflicting original blob')
            expected[path] = blob
    for path, blob in expected.items():
        data = (root / path).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()
        check(actual == blob, 'historical file changed: ' + path)
    parents = git(root, 'show', '-s', '--format=%P', cfg['normal_merge_commit']).split()
    check(parents == list(cfg['source_commits'].values()), 'normal merge parent mismatch')
    return dict(source_commits=cfg['source_commits'], source_trees=cfg['source_trees'],
                normal_merge_commit=cfg['normal_merge_commit'], merge_parents=parents,
                source_file_counts=counts, preserved_union_files=len(expected),
                ordered_path_blob_sha256=sha(encoded(expected)), all_original_bytes_preserved=True)


def local_preservation(root, expected):
    """Optional explicit local check. Portable suite never requires mac1 files."""
    verify_hashes(root, expected)
    return dict(files_checked=len(expected), all_preserved=True, sha256=expected)


def preparation_inventory(root=ROOT):
    """Read preserved planning metadata, not fit parameters, arrays or scores."""
    from src.checkpoint_r2_selection_v1 import candidate_id
    path = root / 'results/checkpoint_r2_cpu_preflight_v1_20261002'
    with (path / 'bank_candidates.csv').open() as f:
        bank = list(csv.DictReader(f))
    expected = set()
    for model, layers in (('qwen', 28), ('llama', 32)):
        for layer in range(layers):
            expected.update(candidate_id(model, layer, 'l2_logistic', C) for C in (.001, .01, .1, 1., 10.))
            expected.update(candidate_id(model, layer, method) for method in
                            ('difference_of_means', 'mass_mean_covariance', 'burger_t_g', 'ttpd', 'r0'))
    check(len(bank) == 600 and {r['candidate_id'] for r in bank} == expected, 'bank inventory mismatch')
    reuse = json.loads((path / 'conditional_reuse.json').read_bytes())
    check(len(reuse) == len({r['candidate_id'] for r in reuse}) == 36, 'conditional reuse inventory mismatch')
    check(all(r['candidate_id'] in expected and r['historical_correction'] == 'v4' and
              r['production_adopted'] is False and r['preprocessing_cohort'] == 'P15' for r in reuse),
          'historical provenance or adoption changed')
    with (path / 'planned_repair_folds.csv').open() as f:
        folds = list(csv.DictReader(f))
    expected_folds = {(m, l, s, f) for m, n in (('qwen', 28), ('llama', 32))
                      for l in range(n) for s in (11, 23, 37) for f in range(5)}
    check(len(folds) == 900 and {(r['model'], int(r['layer']), int(r['sample_seed']), int(r['fold']))
                               for r in folds} == expected_folds, 'repair fold inventory mismatch')
    return dict(bank_candidates=600, per_model=dict(Counter(r['model'] for r in bank)),
                repair_fold_fits_before_refits_controls=900, conditional_P15_reuse_matches=36,
                reuse_adopted=0, historical_provenance='v4', external_fit_files_rechecked=False,
                status='metadata inventory only; bridge/runtime gates still pending')
