"""Pinned v2 adoption, cache inventory and score-independent development eligibility."""
import csv
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from src import selection_repair_candidate_mapping as oldmap
from src import selection_repair_source_closure as closure
from src import clean_transfer_contracts as c
from src import priority2_input_controls as controls
from src.clean_compounds import stable_id
from src.selection_repair_cache_export_remote import DIRECTORIES

CONFIG = Path('config/clean_protocol/canonical_sensitivity_v1.json')
REPORT = Path('results/t2_canonical_sensitivity_20261001')
ROOT_REMOTE = '/workspace/truth-probing-artifacts/'
GEN = 'priority2_input_controls_v1/data/'
RAW = 'pinned_compound_qwen2_5/qwen2_5_7b_a09a354_bs1_bf16_v1/'

require = oldmap.require
sha = oldmap.sha
identity = oldmap.identity
json_bytes = oldmap.json_bytes
canonical = oldmap.canonical
parse_rows = oldmap.parse_rows


def csv_bytes(rows):
    require(bool(rows), 'empty table needs explicit schema')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return stream.getvalue().encode()


class Inputs:
    def __init__(self, root, payload, inventory_identity):
        self.root, self.payload = Path(root), Path(payload)
        self.reader = oldmap.MetadataReader()
        inventory = json.loads(self.reader.read(self.payload, 'inventory.json', inventory_identity))
        self.inventory = {r['original_path']: r for r in inventory}
        require(len(self.inventory) == len(inventory), 'duplicate inventory path')
        self.used = {}

    def remote(self, relative, expected=None):
        path = ROOT_REMOTE + relative
        require(path in self.inventory, 'missing restored dependency: '+path)
        record = self.inventory[path]
        if expected:
            require(all(record[k] == expected[k] for k in ('bytes', 'sha256')), 'inventory/spec mismatch: '+path)
        local = self.payload / record['archive_path']
        # Historical compact probe archives are permitted; activation arrays are never restored here.
        require(local.suffix in ('.json', '.csv', '.npz'), 'unexpected package input')
        data = local.read_bytes()
        require(identity(data) == {k: record[k] for k in ('bytes', 'sha256')}, 'restored input changed: '+path)
        self.used[path] = {k: record[k] for k in ('archive_path', 'bytes', 'sha256')}
        return data

    def unchanged(self):
        self.reader.unchanged()
        for path, record in self.used.items():
            require(identity((self.payload/record['archive_path']).read_bytes()) ==
                    {k: record[k] for k in ('bytes', 'sha256')}, 'input changed during preparation: '+path)


def verify_adoption(root):
    root = Path(root)
    config_bytes = (root/CONFIG).read_bytes(); config = json.loads(config_bytes)
    decision_bytes = (root/config['adoption_path']).read_bytes(); decision = json.loads(decision_bytes)
    require(sha(decision_bytes) == config['adoption_sha256'], 'stale adoption record')
    require(decision['correction_overlay'] == 'candidate_overlay_v2' and decision['admission_column'] == 'successor_status'
            and decision['correction_adopted_for_scope'] is True and decision['recorded_representation_adopted_for_scope'] is True,
            'bounded v2 adoption required')
    require(not decision['production_bank_adopted'] and not decision['historical_execution_independently_replayed'], 'adoption scope expanded')
    for pin in [config['fit_source_pin'], *decision['bindings'].values()]:
        require(identity((root/pin['path']).read_bytes()) == {k: pin[k] for k in ('bytes', 'sha256')}, 'stale adopted input: '+pin['path'])
    return config, decision


def fact_eligibility(facts, manifest, ledger):
    """Exact constituent provenance/evidence; old compound labels never grant eligibility."""
    source = defaultdict(list)
    for row in manifest:
        if row['split'] == 'validation' and row['form'] == 'affirmative':
            source[(row['topic'], row['entity_id'], row['statement'])].append(row)
    external = defaultdict(list)
    for row in ledger:
        if row['split'] == 'validation' and row['origin'] != 'source':
            external[row['fact_ref']].append(row)
    results = []
    for fact in facts:
        require(fact['split'] == 'validation' and fact['truth'] in ('True', 'False'), 'non-development fact')
        truth = fact['truth'] == 'True'
        if truth:
            require(fact['fact_id'] == stable_id('fact', [fact['topic'], fact['entity_id'], fact['statement'], True]), 'positive fact ID mismatch')
        originals = source.get((fact['topic'], fact['entity_id'], fact['statement']), [])
        conflicts = [r for r in originals if r['successor_status'] != 'admitted' or r['label'] != str(int(truth))]
        accepted_source = [r for r in originals if r['successor_status'] == 'admitted' and r['label'] == str(int(truth))]
        accepted_registry = [r for r in external[fact['fact_id']] if all(r[k] == fact[k] for k in ('topic', 'entity_id', 'statement'))
                             and r['label'] == '0_candidate' and r['eligible'] == 'True'
                             and r['support'] in ('historical_external_negative', 'historical_source_label_negative')]
        # Negative compound IDs must bind the accepted registry entry, even when source-supported.
        allowed = not conflicts and bool(accepted_source if truth else accepted_registry)
        reasons = ['restricted_or_conflicting_exact_source_claim'] if conflicts else []
        if truth and not accepted_source: reasons.append('no_admitted_exact_positive_source')
        if not truth and not accepted_registry: reasons.append('no_eligible_exact_registry_negative')
        people = {r['person_key'] for r in accepted_source + accepted_registry}
        all_people = {r['person_key'] for r in originals}
        require(len(people) <= 1 and len(all_people) <= 1, 'ambiguous constituent person identity')
        results.append(dict(fact_key=fact['fact_key'], fact_id=fact['fact_id'], entity_id=fact['entity_id'],
                            topic=fact['topic'], statement_sha256=sha(fact['statement'].encode()), truth=fact['truth'],
                            person_key=next(iter(people or all_people), fact['entity_id']), eligible=allowed,
                            reasons=json.dumps(reasons), source_row_ids=json.dumps([r['source_row_id'] for r in originals]),
                            source_evidence_tier='retained_source_label_not_human_verification' if truth else '',
                            registry_evidence=json.dumps([dict(origin=r['origin'], support=r['support'], evidence_ref=r['evidence_ref']) for r in accepted_registry])))
    return results


def pair_eligibility(raw, facts, mapping):
    by_fact = {r['fact_key']: r for r in facts}
    by_example = {r['base_example_id']: r for r in mapping}
    require(len(by_fact) == len(facts) and len(by_example) == len(mapping), 'duplicate fact/map identity')
    groups = defaultdict(list)
    for row in raw: groups[row['pair_id']].append(row)
    result = []
    for pair, rows in sorted(groups.items()):
        keys = {by_example[r['example_id']][f'fact_{side}_key'] for r in rows for side in ('a', 'b')}
        require(len(rows) == 16 and len(keys) == 4, 'incomplete compound pair/fact group')
        leaves = {(r['operator'], r['canonical_truth_a'], r['canonical_truth_b'], r['ordering']) for r in rows}
        require(len(leaves) == 16, 'duplicate compound variants')
        rejected = sorted(k for k in keys if not by_fact[k]['eligible'])
        result.append(dict(pair_id=pair, topic=rows[0]['topic'], entity_a_id=rows[0]['entity_a_id'],
                           entity_b_id=rows[0]['entity_b_id'], eligible=not rejected,
                           original_rows=16, retained_rows=0 if rejected else 16,
                           rejected_fact_keys=json.dumps(rejected), all_fact_keys=json.dumps(sorted(keys))))
    return result


def check_extraction(rows, manifest, progress, tensor, fingerprint):
    data = manifest['data']
    require(data['expected_activation_shape'][0] == len(rows), 'extraction row count')
    require(data['ordered_example_id_sha256'] == sha(canonical([r['example_id'] for r in rows]))
            and data['ordered_statement_sha256'] == sha(canonical([r['statement'] for r in rows])), 'extraction row order')
    recorded_fingerprint = manifest.get('representation_fingerprint', manifest.get('pin', {}).get('fingerprint'))
    require(recorded_fingerprint == fingerprint, 'extraction representation fingerprint')
    reduced = dict(manifest); reduced['code'] = {k: manifest['code'][k] for k in ('implementation_version', 'source_sha256')}
    reduced.pop('smoke_test', None); reduced.pop('historical_atomic_compatibility', None)
    require(progress['complete'] is True and progress['next_row'] == len(rows)
            and progress['activation_file_sha256'] == tensor['sha256']
            and progress['identity_sha256'] == sha(canonical(reduced)), 'incomplete extraction/finalization identity')
    cursor = 0
    for chunk in progress['chunks']:
        stop = chunk['stop']
        require(chunk['start'] == cursor and cursor < stop <= len(rows)
                and chunk['ordered_example_id_sha256'] == sha(canonical([r['example_id'] for r in rows[cursor:stop]]))
                and chunk['ordered_statement_sha256'] == sha(canonical([r['statement'] for r in rows[cursor:stop]])), 'chunk row binding mismatch')
        cursor = stop
    require(cursor == len(rows), 'incomplete extraction coverage')


def prepare(root, payload, output):
    root, payload, output = Path(root), Path(payload), Path(output)
    require(not output.exists(), 'preparation output exists; no overwrite')
    config, adoption = verify_adoption(root)
    closure_receipt = closure.run(root, check_only=True)
    old_outputs, old_report = oldmap.build(root, payload)
    require(old_report['metadata_alignment_passed'], 'v1 mapping incomplete')
    v1 = parse_rows((root/closure.prior.PACKAGE/'row_manifest.csv').read_bytes())
    v2 = parse_rows((root/closure.SUCCESSOR/'row_manifest.csv').read_bytes())
    before = closure.admitted_identity_bytes(v1, 'candidate_status')
    after = closure.admitted_identity_bytes(v2, 'successor_status')
    require(before == after == (root/closure.SUCCESSOR/'admitted_identities.csv').read_bytes(), 'ordered admission identity mismatch')
    require(sha(after) == closure_receipt['v1_admitted_identity_sha256'] == closure_receipt['v2_admitted_identity_sha256'], 'closure admission hash mismatch')
    physical = json.loads((root/adoption['bindings']['physical_receipt']['path']).read_text())
    inv = next(r for r in physical['input_identities'] if r['location'] == 'inventory.json')
    inputs = Inputs(root, payload, inv)
    specs = {key: json.loads((root/adoption['bindings'][key]['path']).read_text()) for key in ('qwen_spec', 'qwen_controls_spec', 'llama_spec')}
    outputs, maps, sidecars, requests = {}, {}, {}, {}
    by_id = {r['source_row_id']: r for r in v2}
    for cache, directory in oldmap.CACHES.items():
        model, split = cache.split('/')
        key = model+'_'+split
        rows = parse_rows(old_outputs[key+'.provisional.csv'])
        require(len(rows) == sum(r['successor_status'] == 'admitted' and r['split'] == split for r in v2), 'v2 manifest mapping coverage')
        for row in rows:
            require(by_id[row['source_row_id']]['successor_status'] == 'admitted', 'nonadmitted v2 mapping')
            row['successor_status'] = row.pop('candidate_status')
            row['mapping_status'] = 'adopted_only_for_bounded_canonical_sensitivity'
        maps[key] = rows; outputs[key+'.mapping_v2.csv'] = csv_bytes(rows)
        request = physical['remote_requests'][cache]
        sidecars[key] = parse_rows(inputs.remote(directory.removeprefix(ROOT_REMOTE)+'/metadata.csv', request['companions'][directory+'/metadata.csv']))
        indices = [int(r['tensor_row_index']) for r in rows] if split == 'train' else list(range(len(sidecars[key])))
        requests[key] = {**request, 'layer': config['models'][model]['layer'], 'row_indices': indices,
                         'row_identity_sha256': sha(canonical([sidecars[key][i]['dataset']+':'+sidecars[key][i]['row_index'] for i in indices]))}
    atomic_mask = []
    for pos, row in enumerate(sidecars['qwen_validation']):
        original = by_id[row['dataset']+':'+row['row_index']]
        atomic_mask.append(dict(tensor_row_index=pos, source_row_id=original['source_row_id'], topic=original['topic'],
                                person_key=original['person_key'], successor_status=original['successor_status'],
                                eligible=original['successor_status'] == 'admitted', label=original['label']))
    outputs['atomic_D_eligibility.csv'] = csv_bytes(atomic_mask)
    # All provenance is frozen and verified before any probe score is calculated.
    raw_bytes = inputs.remote(RAW+'metadata.csv', specs['qwen_spec']['inputs']['compound']['files']['metadata.csv'])
    raw_frame = controls.validate_base(pd.read_csv(io.BytesIO(raw_bytes)))
    raw = parse_rows(raw_bytes)
    generation = specs['qwen_controls_spec']['inputs']['generation']['files']
    facts = parse_rows(inputs.remote(GEN+'isolated_facts.csv', generation['isolated_facts.csv']))
    mapping = parse_rows(inputs.remote(GEN+'constituent_map.csv', generation['constituent_map.csv']))
    # Independently rebuild the existing identifier, including exact statement and truth.
    fact_lookup = {r['fact_key']: r for r in facts}
    require(len(fact_lookup) == len(facts), 'duplicate isolated key')
    for fact in facts:
        record = {k: fact[k] for k in ('fact_id','entity_id','topic','statement','truth','split','protocol','evaluation_phase')}
        record['truth'] = record['truth'] == 'True'
        require(controls.identifier('fact', record) == fact['fact_key'], 'isolated fact-key mismatch')
    map_lookup = {r['base_example_id']: r for r in mapping}
    require(set(map_lookup) == {r['example_id'] for r in raw} and len(map_lookup) == len(mapping), 'constituent-map coverage')
    for row in raw:
        for side in ('a','b'):
            fact = fact_lookup[map_lookup[row['example_id']][f'fact_{side}_key']]
            require(all(fact[k] == value for k,value in {'fact_id':row[f'fact_{side}_id'], 'entity_id':row[f'entity_{side}_id'],
                    'topic':row['topic'], 'statement':row[f'fact_{side}_statement'], 'truth':row[f'canonical_truth_{side}']}.items()), 'constituent identity mismatch')
    ledger = parse_rows((root/closure.prior.PACKAGE/'fact_inventory.csv').read_bytes())
    eligible_facts = fact_eligibility(facts, v2, ledger)
    pairs = pair_eligibility(raw, eligible_facts, mapping)
    outputs['development_fact_eligibility.csv'] = csv_bytes(eligible_facts)
    outputs['development_pair_eligibility.csv'] = csv_bytes(pairs)
    both = parse_rows(inputs.remote(GEN+'or_explicit_or_both_v1.csv', generation['or_explicit_or_both_v1.csv']))
    raw_by_id = {r['example_id']:r for r in raw}
    require(len(both) == len(raw)//2 and len({r['base_example_id'] for r in both}) == len(both), 'or-both correspondence')
    for row in both:
        original = raw_by_id[row['base_example_id']]
        require(original['operator']=='OR' and all(row[k] == original[k] for k in controls.IDENTITY), 'or-both constituent identity')
        require(row['statement'] == original['statement'].removesuffix('.')+', or both.', 'or-both rendering mismatch')
    wanted = {r['example_id']:r['statement'] for r in raw + both}
    wanted.update({r['fact_key']:r['statement'] for r in facts})
    for key, files, prefix, model in [
        ('qwen_raw', specs['qwen_spec']['inputs']['compound']['files'], RAW, 'qwen'),
        ('qwen_controls', specs['qwen_controls_spec']['inputs']['extraction']['files'], DIRECTORIES['qwen_controls']+'/', 'qwen'),
        ('llama_transfer', specs['llama_spec']['inputs']['transfer'], DIRECTORIES['llama_transfer']+'/', 'llama')]:
        companion = {name:inputs.remote(prefix+name, files[name]) for name in ('metadata.csv','extraction_manifest.json','progress.json')}
        rows = parse_rows(companion['metadata.csv']); manifest = json.loads(companion['extraction_manifest.json']); progress = json.loads(companion['progress.json'])
        fingerprint = physical['caches'][model+'/train']['representation_fingerprint']
        check_extraction(rows, manifest, progress, files['activations.npy'], fingerprint)
        require(manifest['data']['sidecar_sha256'] == sha(companion['metadata.csv']), 'extraction sidecar hash')
        selected = list(range(len(rows))) if key == 'qwen_raw' else [i for i,r in enumerate(rows) if r['condition_id'] in ('raw_reference','or_explicit_or_both_v1','isolated_constituents_v1')]
        require(all(r['split']=='validation' and r['protocol']=='entity_disjoint' for r in rows), 'development cache scope')
        require(all(rows[i]['example_id'] in wanted and rows[i]['statement']==wanted[rows[i]['example_id']] for i in selected), 'transfer row statement/ID binding')
        require(len({r['example_id'] for r in rows}) == len(rows), 'duplicate transfer sidecar identity')
        requests[key] = dict(path=ROOT_REMOTE+prefix+'activations.npy', **files['activations.npy'],
                             companions={ROOT_REMOTE+prefix+name:files[name] for name in companion},
                             shape=manifest['data']['expected_activation_shape'], layer=config['models'][model]['layer'], row_indices=selected,
                             row_identity_sha256=sha(canonical([rows[i]['example_id'] for i in selected])))
    for model, prefix, expected in [('qwen','atomic_probe_selection/qwen25_7b/',specs['qwen_spec']['inputs']['probe']['files']),
                                   ('llama','llama31_replication_v1/probe/',specs['llama_spec']['inputs']['probe'])]:
        for name in ('selected_probe.npz','selection.json','validation_metrics.csv'):
            inputs.remote(prefix+name,expected[name])
    outputs['endpoint_declaration.json'] = json_bytes(config)
    outputs['export_request.json'] = json_bytes(dict(schema_version=1, adoption_sha256=config['adoption_sha256'], caches=requests))
    inventory = dict(metadata_inputs=inputs.used, config_sha256=sha((root/CONFIG).read_bytes()),
                     adoption_sha256=config['adoption_sha256'], closure_receipt_sha256=adoption['bindings']['closure_receipt']['sha256'],
                     ordered_admitted_identity_sha256=sha(after), admission_column='successor_status',
                     expected_rows={key:len(rows) for key,rows in maps.items()},
                     original_atomic_D_rows=len(atomic_mask), retained_atomic_D_rows=sum(r['eligible'] for r in atomic_mask),
                     original_pairs=len(pairs), retained_pairs=sum(r['eligible'] for r in pairs),
                     original_facts=len(facts), eligible_facts=sum(r['eligible'] for r in eligible_facts),
                     source_tensor_bytes=sum(r['bytes'] for r in requests.values()),
                     selected_export_bytes_estimate=sum(len(r['row_indices'])*r['shape'][2]*2+128 for r in requests.values()),
                     physical_sources_need_current_verification=True, missing_dependencies=[])
    outputs['input_inventory.json'] = json_bytes(inventory)
    receipt = dict(stage='frozen_before_fitting_and_scoring', production_usable=False, corrected_full_training_reference_only=True,
                   adoption_sha256=config['adoption_sha256'], ordered_admission_equal=True,
                   artifacts={name:identity(data) for name,data in outputs.items()},
                   source_sha256={name:sha((root/name).read_bytes()) for name in ('src/selection_repair_sensitivity_inputs.py','src/selection_repair_cache_export_remote.py')})
    outputs['preparation_receipt.json'] = json_bytes(receipt)
    inputs.unchanged(); output.mkdir(parents=True)
    for name,data in outputs.items(): (output/name).write_bytes(data)
    return inventory


def verify_preparation(root, directory):
    config, adoption = verify_adoption(root)
    directory = Path(directory)
    receipt = json.loads((directory/'preparation_receipt.json').read_text())
    require(receipt['adoption_sha256'] == config['adoption_sha256'], 'mixed preparation/adoption')
    for name,expected in receipt['artifacts'].items():
        require(identity((directory/name).read_bytes()) == expected, 'stale frozen preparation: '+name)
    for name,digest in receipt['source_sha256'].items():
        require(sha((Path(root)/name).read_bytes()) == digest, 'preparation code changed: '+name)
    inventory = json.loads((directory/'input_inventory.json').read_text())
    require(inventory['config_sha256'] == sha((Path(root)/CONFIG).read_bytes()), 'endpoint configuration changed')
    return config, inventory
