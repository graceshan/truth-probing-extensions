"""Score-blind fresh pilot selection. Only frozen train/development raw inputs."""
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261002
TOPICS = ('animal_class', 'cities', 'element_symb', 'inventors', 'sp_en_trans')
PACKAGE = 'data/checkpoint_r2_v1/'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2,
                       allow_nan=False) + '\n').encode()


def hash_value(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def quantiles(candidates, n, salt):
    """Equal-population UTF-8-length bins; hash/seed chooses within each bin."""
    require(len(candidates) >= n, 'insufficient selection stratum: ' + salt)
    ordered = sorted(candidates, key=lambda r: (len(r['statement'].encode()), r['id']))
    selected = []
    for i in range(n):
        bucket = ordered[i * len(ordered) // n:(i + 1) * len(ordered) // n]
        selected.append(min(bucket, key=lambda r: text_hash(f'{SEED}/{salt}/{r["id"]}')))
    return selected


def build_manifest(root=ROOT):
    def read(name):
        return json.loads((root / (PACKAGE + name)).read_text())
    members, facts = read('memberships.json'), read('facts.json')
    atomic = {r['source_row_id']: r for name in ('P15', 'atomic_D') for r in members[name]}
    compounds = list(csv.DictReader((root / (PACKAGE + 'compound_bindings.csv')).open()))
    compound = {r['example_id']: r for r in compounds}
    wording_sources = defaultdict(list)
    for r in compounds:
        if r['group'] == 'D_bare':
            wording_sources[(r['pair_id'], r['fact_a_id'], r['fact_b_id'])].append(r)
    inventory = list(csv.DictReader((root / (PACKAGE + 'extraction_inventory.csv')).open()))
    texts = {}
    for offset, binding in enumerate(inventory):
        if binding['kind'] != 'raw':
            continue
        logical = binding['logical_id']
        if logical in atomic:
            source = atomic[logical]
            split, topic, order, op = source['split'], source['topic'], '', ''
        elif logical in facts:
            source = facts[logical]
            split, topic, order, op = source['split'], source['topic'], '', ''
        else:
            if logical in compound:
                source = compound[logical]
                order, op = source['ordering'], source['operator']
            else:
                # Wording identities bind exact facts and surface order. Never reverse-engineer text.
                candidates = wording_sources[(binding['pair_id'], binding['fact_a_id'], binding['fact_b_id'])]
                from src.checkpoint_r2_inputs import uid
                matches = [r for r in candidates
                           if uid('r2wording', r['example_id'], binding['group']) == logical]
                require(len(matches) == 1, 'ambiguous wording identity')
                source = matches[0]
                order, op = source['ordering'], source['operator']
            split, topic = source['split'], source['topic']
        require(split in ('train', 'validation') and topic in TOPICS, 'non-train/development raw input')
        require(binding['text_id'] == 'text_' + text_hash(binding['statement']), 'text hash')
        item = texts.setdefault(binding['text_id'], dict(id=binding['text_id'],
                                statement=binding['statement'], statement_sha256=text_hash(binding['statement']),
                                split=split, topic=topic, bindings=[]))
        require((item['statement'], item['split'], item['topic']) ==
                (binding['statement'], split, topic), 'cross-partition/topic text collision')
        item['bindings'].append(dict(inventory_offset=offset, **binding, split=split, topic=topic,
                                     ordering=order, operator=op))
    require(len(texts) == 35843, 'raw inventory count differs')
    chosen, used = {}, set()
    # 16 calibration and 48 verification inputs per topic. Quotas do not use labels.
    quotas = {
        'calibration': [('P15', '', 4), ('TC_control_and_final_refit', 'AND/AB', 3),
                        ('TC_control_and_final_refit', 'AND/BA', 3),
                        ('TC_control_and_final_refit', 'OR/AB', 3),
                        ('TC_control_and_final_refit', 'OR/BA', 3)],
        'verification': [('P15', '', 6), ('atomic_D', '', 6), ('D_bare', 'AND/AB', 3),
                         ('D_bare', 'AND/BA', 3), ('D_bare', 'OR/AB', 3), ('D_bare', 'OR/BA', 3),
                         ('and_both_following_v1', '', 8), ('or_explicit_or_both_v1', '', 8),
                         ('or_at_least_one_v1', '', 8)]}
    for stage, strata in quotas.items():
        chosen[stage] = []
        for topic in TOPICS:
            for group, surface, n in strata:
                eligible = [r for r in texts.values() if r['topic'] == topic and r['id'] not in used
                            and any(b['group'] == group and (not surface or
                                    b['operator'] + '/' + b['ordering'] == surface) for b in r['bindings'])]
                rows = quantiles(eligible, n, f'{stage}/{topic}/{group}/{surface}')
                for r in rows:
                    chosen[stage].append(dict(r, selection_stratum=f'{topic}/{group}/{surface}'))
                    used.add(r['id'])
    require(len(used) == 320 and all(r['split'] == 'train' for r in chosen['calibration']), 'pilot bound')
    return dict(schema='r2-fresh-pilot-inputs-v1', seed=SEED,
                algorithm='topic/group/operator/order quotas; UTF-8-length equal-population bins; '
                          'minimum SHA256(seed/stage/topic/group/surface/text_id) within each bin; '
                          'calibration first, exclude its text IDs from verification; preserve ALL inventory bindings in original order',
                quotas={stage: [list(q) for q in rows] for stage, rows in quotas.items()},
                unique_inputs_per_model=320, raw_unique_inventory_per_model=len(texts),
                models=['qwen', 'llama'], stages=chosen,
                ordered_ids_sha256={s: hash_value([r['id'] for r in rows]) for s, rows in chosen.items()})
