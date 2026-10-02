"""Complete frozen raw inventory, deduplicated only at exact-text inference level."""
import csv
import json
from collections import Counter
from src.checkpoint_r2_fresh_inputs import ROOT, TOPICS, canonical, hash_value, require, text_hash
from src.checkpoint_r2_fresh_pilot import plan as pilot_plan

WORDING = {'and_both_following_v1', 'or_explicit_or_both_v1', 'or_at_least_one_v1'}


def reconstruct(inventory, memberships, facts, compounds, allowed_groups):
    from src.checkpoint_r2_inputs import uid
    from src.final_e_preparation import render_wording
    from src.heldout_and_wording import render as render_and
    atomic = {r['source_row_id']: r for name in ('P15', 'atomic_D') for r in memberships[name]}
    atomic_groups = {name: {r['source_row_id'] for r in memberships[name]} for name in ('P15', 'atomic_D')}
    sources = {(r['group'], r['example_id']): r for r in compounds}
    for r in compounds:
        if r['group'] == 'D_bare':
            for group in WORDING:
                if (group.startswith('and_')) == (r['operator'] == 'AND'):
                    first, second = (r['fact_a_id'], r['fact_b_id']) if r['ordering'] == 'AB' else (r['fact_b_id'], r['fact_a_id'])
                    a, b = facts[first]['statement'], facts[second]['statement']
                    statement = render_and(a, b, 'AND') if r['operator'] == 'AND' else render_wording(group, a, b)
                    sources[(group, uid('r2wording', r['example_id'], group))] = dict(r, statement=statement, group=group)
    texts, bindings, excluded = {}, [], Counter()
    for offset, b in enumerate(inventory):
        if b['kind'] != 'raw':
            require(b['kind'] in ('behavior', 'chat'), 'unknown/E workload forbidden')
            excluded[b['kind']] += 1
            continue
        require(b['group'] in allowed_groups, 'unknown/E raw group forbidden')
        logical = b['logical_id']
        if b['group'] in ('P15', 'atomic_D'):
            source = atomic.get(logical)
            require(source is not None and logical in atomic_groups[b['group']], 'atomic source/group mismatch')
        elif b['group'] in ('TC_isolated', 'D_isolated'):
            source = facts.get(logical)
            require(source is not None and source['eligible'], 'unknown/ineligible fact')
            require(source['split'] == ('train' if b['group'] == 'TC_isolated' else 'validation'), 'fact group/split')
        else:
            source = sources.get((b['group'], logical))
            require(source is not None, 'unknown compound/wording source identity')
            require((b['pair_id'], b['fact_a_id'], b['fact_b_id'], b['label']) ==
                    (source['pair_id'], source['fact_a_id'], source['fact_b_id'], source['label']), 'ordered fact/pair/label bindings')
        split, topic = source['split'], source['topic']
        require(split in ('train', 'validation') and topic in TOPICS, 'E/test partition forbidden')
        require(b['statement'] == source['statement'], 'source text mismatch')
        require(b['text_id'] == 'text_' + text_hash(b['statement']), 'text ID/hash mismatch')
        if b['group'] in ('P15', 'atomic_D'):
            require(not any(b[k] for k in ('pair_id','fact_a_id','fact_b_id')), 'atomic extra binding')
            require(b['label'] == str(source['label']), 'atomic label binding')
        if b['group'] in ('TC_isolated', 'D_isolated'):
            require(b['fact_a_id']==logical and not b['fact_b_id'] and not b['pair_id'], 'isolated fact binding')
            require(b['label'] == str(int(source['truth'])), 'fact label binding')
        binding = dict(b, inventory_offset=offset, split=split, topic=topic,
                       ordering=source.get('ordering', ''), operator=source.get('operator', ''),
                       source_metadata_sha256=hash_value(source))
        bindings.append(binding)
        r = texts.setdefault(b['text_id'], dict(id=b['text_id'], statement=b['statement'],
                statement_sha256=text_hash(b['statement']), split=split, topic=topic, bindings=[]))
        require((r['statement'], r['split'], r['topic']) == (b['statement'], split, topic), 'cross-partition/topic text collision')
        r['bindings'].append(binding)
    require(len({(b['group'], b['logical_id']) for b in bindings}) == len(bindings), 'duplicate logical binding')
    rows = list(texts.values())  # first occurrence; bindings retain original inventory offsets
    require(sorted((b for r in rows for b in r['bindings']), key=lambda b: b['inventory_offset']) == bindings,
            'ordered binding coverage')
    return dict(schema='r2-full-raw-inputs-v1', ordering='unique text first occurrence; all bindings in original CSV order',
                unique_texts=len(rows), logical_bindings=len(bindings), rows=rows, bindings=bindings,
                ordered_ids_sha256=hash_value([r['id'] for r in rows]),
                ordered_bindings_sha256=hash_value(bindings), excluded_workloads=dict(excluded),
                group_counts=dict(Counter(b['group'] for b in bindings)))


def full_manifest(cfg, root=ROOT):
    pilot, _ = pilot_plan(root, rebuild=False)  # preserves frozen input/contract checks
    require(cfg['models']==pilot['models'] and cfg['execution']==pilot['execution'] and cfg['runtime_lock']==pilot['runtime_lock'], 'validated pilot representation changed')
    base = root / 'data/checkpoint_r2_v1'
    manifest = reconstruct(list(csv.DictReader((base/'extraction_inventory.csv').open())),
        json.loads((base/'memberships.json').read_text()), json.loads((base/'facts.json').read_text()),
        list(csv.DictReader((base/'compound_bindings.csv').open())), set(cfg['raw_groups']))
    require(manifest['unique_texts'] == cfg['expected_unique_texts'] == pilot['raw_inventory']['unique_inputs_per_model'], 'unique raw count')
    require(manifest['logical_bindings'] == cfg['expected_logical_bindings'] == pilot['raw_inventory']['logical_bindings_per_model'], 'logical raw count')
    estimates = json.loads((base/'resource_estimates.json').read_text())['raw']
    require((manifest['unique_texts'], manifest['logical_bindings']) == (estimates['unique_inputs_per_model'], estimates['logical_bindings_per_model']), 'inventory/estimate counts')
    payload = {m: manifest['unique_texts'] * s['layers'] * s['width'] * 2 for m,s in cfg['models'].items()}
    require(payload == estimates['fp16_all_layer_bytes_per_model'], 'reconstructed tensor estimate')
    manifest['fp16_payload_bytes'] = payload
    return manifest
