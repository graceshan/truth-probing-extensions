"""Versioned v3 admission/projection adapter over immutable physical cache exports."""
import json
from pathlib import Path

from src import selection_repair_sensitivity_inputs as prior
from src import capacity_pilot_projection as projection
from src.selection_repair_sensitivity_verification import verify_exports

CONFIG = Path('config/clean_protocol/canonical_sensitivity_v3.json')
CORRECTION_COMMIT = '2ffbe79cdb22eb772c1b842ab9f0e9d8d11a8e8c'
STATUS = 'audit_successor_status'
QUARANTINED = {'inventors:23','inventors:157','neg_inventors:23','neg_inventors:157'}
require, sha, identity = prior.require, prior.sha, prior.identity
json_bytes, canonical, parse_rows, csv_bytes = prior.json_bytes, prior.canonical, prior.parse_rows, prior.csv_bytes
Inputs, ROOT_REMOTE, GEN, RAW = prior.Inputs, prior.ROOT_REMOTE, prior.GEN, prior.RAW


def verify_adoption(root):
    root = Path(root)
    config = json.loads((root/CONFIG).read_text())
    data = (root/config['adoption_path']).read_bytes(); decision = json.loads(data)
    require(sha(data) == config['adoption_sha256'], 'stale v3 adoption')
    require(decision['reviewed_source_commit'] == CORRECTION_COMMIT
            and decision['correction_overlay'] == 'candidate_overlay_v3'
            and decision['admission_column'] == STATUS and decision['fact_eligibility_column'] == 'current_eligible'
            and decision['projection'] == 'capacity_pilot_projection_v2', 'wrong correction/admission version')
    require(decision['correction_adopted_for_scope'] is True and decision['recorded_representation_adopted_for_scope'] is True
            and decision['production_bank_adopted'] is False and decision['P_A_frozen'] is False, 'v3 adoption scope')
    for pin in [config['fit_source_pin'], *decision['bindings'].values()]:
        require(identity((root/pin['path']).read_bytes()) == {k:pin[k] for k in ('bytes','sha256')}, 'stale v3 input: '+pin['path'])
    old = json.loads((root/prior.CONFIG).read_text())
    require(config['prior_endpoint_declaration_sha256'] == sha((root/prior.CONFIG).read_bytes()), 'stale endpoint predecessor')
    for key in ('models','endpoints','scopes','topics','bootstrap','refresh','fit_source_pin'):
        require(config[key] == old[key], 'predeclared procedure changed: '+key)
    return config, decision


def current_fact_eligibility(facts, manifest, ledger):
    """Explicit in-memory schema adapter; no historical field grants eligibility."""
    require(all(STATUS in r for r in manifest), 'missing current admission column')
    require(all(r.get('current_eligible') in ('True','False') for r in ledger), 'missing current fact eligibility')
    current_manifest = [{**r, 'successor_status':r[STATUS]} for r in manifest]
    current_ledger = [{**r, 'eligible':r['current_eligible']} for r in ledger]
    rows = prior.fact_eligibility(facts, current_manifest, current_ledger)
    for row in rows:
        row['current_eligible'] = row.pop('eligible')
    return rows


def bind_mapping(manifest, historical_map, exported_tensor_rows, split):
    by_id = {r['source_row_id']:r for r in manifest}
    require(len(by_id) == len(manifest), 'duplicate manifest source identity')
    admitted = {r['source_row_id'] for r in manifest if r['split']==split and r[STATUS]=='admitted'}
    require(not admitted & QUARANTINED, 'quarantined source survives current admission')
    positions = {row:i for i,row in enumerate(exported_tensor_rows)}
    require(len(positions)==len(exported_tensor_rows), 'duplicate original export indices')
    result = []
    for old in historical_map:
        source = by_id[old['source_row_id']]
        for field in ('dataset','row_index','source_sha256','source_row_sha256','statement_sha256',
                      'entity_id','person_key','split','partition_role','label'):
            require(old[field]==source[field], 'source/hash/person binding changed: '+field)
        if source[STATUS] != 'admitted': continue
        tensor_index = int(old['tensor_row_index'])
        require(source['split']==split and tensor_index in positions, 'missing tensor export binding')
        result.append(dict(old, **{STATUS:source[STATUS]}, export_row_index=positions[tensor_index],
                           mapping_status='v3_bounded_canonical_sensitivity_only'))
    ids = [r['source_row_id'] for r in result]
    require(len(ids)==len(set(ids)) and set(ids)==admitted, 'missing/ambiguous current mapping')
    require(len({r['tensor_row_index'] for r in result})==len(result), 'duplicate tensor binding')
    return result


def sampler_bindings(root):
    """Recompute actual identities, not just exposure counts; no reservation."""
    root = Path(root)
    outputs, summary = projection.build(root)
    for name,data in outputs.items():
        require((root/name).read_bytes()==data, 'successor projection differs: '+str(name))
    require(set(summary['changed_source_ids'])==QUARANTINED, 'unexpected correction set')
    report = dict(projection='capacity_pilot_projection_v2',P_A_frozen=False,
                  admitted_by_split=summary['admitted_by_split'],memberships={},sampler={})
    for name, expected_count in [('corrected_outer_training',3044),('P15_admitted_membership',2782),('P10_admitted_membership',2868)]:
        path=root/projection.PACKAGE/(name+'.csv');rows=parse_rows(path.read_bytes())
        require(len(rows)==expected_count and not ({r['source_row_id'] for r in rows}&QUARANTINED), 'current projection membership')
        require(all(r[STATUS]=='admitted' for r in rows), 'historical admission used')
        report['memberships'][name]=dict(rows=len(rows),file=identity(path.read_bytes()),
                                        ordered_source_ids_sha256=sha(canonical([r['source_row_id'] for r in rows])))
    recorded=json.loads((root/projection.PACKAGE/'balanced_membership_changes.json').read_text())
    for n, count, changed in [(15,700,72),(10,800,68)]:
        membership=parse_rows((root/projection.PACKAGE/f'P{n}_admitted_membership.csv').read_bytes())
        sampled=projection.balanced(membership)
        saved=parse_rows((root/projection.PACKAGE/f'P{n}_balanced_exposure.csv').read_bytes())
        before=parse_rows((root/projection.old.PACKAGE/f'P{n}_balanced_exposure.csv').read_bytes())
        ids=lambda rows:[r['source_row_id'] for r in rows]
        require(ids(sampled)==ids(saved) and len(saved)==count, 'ordered sampler identity mismatch')
        removed=sorted(set(ids(before))-set(ids(sampled)));added=sorted(set(ids(sampled))-set(ids(before)))
        require(removed==recorded[str(n)]['removed'] and added==recorded[str(n)]['added']
                and len(removed)==len(added)==changed, 'sampler identity delta mismatch')
        require(not set(ids(saved)) & QUARANTINED, 'quarantined sampler identity')
        report['sampler'][str(n)]=dict(rows=count,removed=removed,added=added,
            ordered_source_ids_sha256=sha(canonical(ids(saved))),file=identity((root/projection.PACKAGE/f'P{n}_balanced_exposure.csv').read_bytes()))
    return report


def prepare(root, payload, parent_prepared, artifacts, output):
    root,payload,parent_prepared,artifacts,output=map(Path,(root,payload,parent_prepared,artifacts,output))
    require(not output.exists(), 'v3 preparation exists; no overwrite')
    config,decision=verify_adoption(root)
    require(sha((parent_prepared/'preparation_receipt.json').read_bytes()) == config['source_export_preparation_sha256'],
            'unrecognized physical export preparation')
    physical=verify_exports(root,parent_prepared,artifacts)
    _,old_inventory=prior.verify_preparation(root,parent_prepared)
    projection_report=sampler_bindings(root)
    manifest=parse_rows((root/projection.OVERLAY/'row_manifest.csv').read_bytes())
    ledger=parse_rows((root/projection.OVERLAY/'fact_inventory.csv').read_bytes())
    request=json.loads((parent_prepared/'export_request.json').read_text())
    outputs={};maps={}
    for key in old_inventory['expected_rows']:
        split=key.split('_')[1]
        rows=bind_mapping(manifest,parse_rows((parent_prepared/(key+'.mapping_v2.csv')).read_bytes()),
                          request['caches'][key]['row_indices'],split)
        require(len(rows)==sum(r['split']==split and r[STATUS]=='admitted' for r in manifest), 'manifest coverage')
        maps[key]=rows;outputs[key+'.mapping_v3.csv']=csv_bytes(rows)
    require([r['source_row_id'] for r in maps['qwen_train']]==[r['source_row_id'] for r in maps['llama_train']], 'cross-model train membership')
    projected_train=parse_rows((root/projection.PACKAGE/'corrected_outer_training.csv').read_bytes())
    require([r['source_row_id'] for r in maps['qwen_train']]==[r['source_row_id'] for r in projected_train], 'ordered admission/projection mismatch')
    def read(relative):
        record=old_inventory['metadata_inputs'][ROOT_REMOTE+relative]
        data=(payload/record['archive_path']).read_bytes()
        require(identity(data)=={k:record[k] for k in ('bytes','sha256')}, 'metadata changed: '+relative)
        return data
    by_id={r['source_row_id']:r for r in manifest}
    atomic=[]
    for row in parse_rows((parent_prepared/'atomic_D_eligibility.csv').read_bytes()):
        source=by_id[row['source_row_id']]
        atomic.append({**{k:v for k,v in row.items() if k not in ('eligible','successor_status')},
                       STATUS:source[STATUS], 'current_eligible':source[STATUS]=='admitted'})
    require(sum(r['current_eligible'] for r in atomic)==1012, 'D manifest coverage')
    # Confirm the common D mask uses the same original sidecar identity at every position.
    qrows=parse_rows(read('atomic_repair/qwen25_a09a354_bs1_bf16_v1/validation/metadata.csv'))
    lrows=parse_rows(read('llama31_replication_v1/atomic/validation/metadata.csv'))
    require(len(qrows)==len(lrows)==len(atomic), 'D sidecar lengths')
    for i,(q,l,m) in enumerate(zip(qrows,lrows,atomic)):
        require(q['dataset']+':'+q['row_index']==l['dataset']+':'+l['row_index']==m['source_row_id']
                and q['statement']==l['statement'] and sha(q['statement'].encode())==by_id[m['source_row_id']]['statement_sha256'], 'D sidecar/order mismatch')
    raw=parse_rows(read(RAW+'metadata.csv'));facts=parse_rows(read(GEN+'isolated_facts.csv'))
    mapping=parse_rows(read(GEN+'constituent_map.csv'))
    current_facts=current_fact_eligibility(facts,manifest,ledger)
    pairs=prior.pair_eligibility(raw,[{**r,'eligible':r['current_eligible']} for r in current_facts],mapping)
    for row in pairs:row['current_eligible']=row.pop('eligible')
    outputs['atomic_D_eligibility.csv']=csv_bytes(atomic)
    outputs['development_fact_eligibility.csv']=csv_bytes(current_facts)
    outputs['development_pair_eligibility.csv']=csv_bytes(pairs)
    outputs['sampler_bindings.json']=json_bytes(projection_report)
    # This is the original physical export request, not a current admission decision.
    outputs['export_request.json']=(parent_prepared/'export_request.json').read_bytes()
    outputs['endpoint_declaration.json']=json_bytes(config)
    inventory=dict(old_inventory,config_sha256=sha((root/CONFIG).read_bytes()),adoption_sha256=config['adoption_sha256'],
        admission_column=STATUS,projected_fact_eligibility='current_eligible',correction_commit=CORRECTION_COMMIT,
        expected_rows={k:len(v) for k,v in maps.items()},parent_prepared_root=str(parent_prepared.resolve()),
        physical_export_root=str(artifacts.resolve()),parent_preparation_receipt=identity((parent_prepared/'preparation_receipt.json').read_bytes()),
        original_atomic_D_rows=len(atomic),retained_atomic_D_rows=sum(r['current_eligible'] for r in atomic),
        original_pairs=len(pairs),retained_pairs=sum(r['current_eligible'] for r in pairs),original_facts=len(facts),
        eligible_facts=sum(r['current_eligible'] for r in current_facts),physical_sources_need_current_verification=False,
        ordered_admitted_identity_sha256=sha(canonical([r['source_row_id'] for r in projected_train])),
        ordered_admitted_identity_scope='current corrected outer train; full identity hashes bound by mapping artifacts',
        correction_receipt_sha256=decision['bindings']['v3_correction_receipt']['sha256'],
        projection_receipt_sha256=decision['bindings']['v3_projection_receipt']['sha256'])
    outputs['input_inventory.json']=json_bytes(inventory)
    outputs['physical_export_binding.json']=json_bytes(physical)
    receipt=dict(schema_version=3,stage='v3_frozen_before_fitting_and_scoring',production_usable=False,P_A_frozen=False,
        correction_commit=CORRECTION_COMMIT,adoption_sha256=config['adoption_sha256'],
        authoritative_admission=STATUS,projected_fact_eligibility='current_eligible',quarantined_ids_absent=sorted(QUARANTINED),
        artifacts={name:identity(data) for name,data in outputs.items()},
        source_sha256={name:sha((root/name).read_bytes()) for name in ('src/selection_repair_sensitivity_v3.py',
            'src/selection_repair_canonical_sensitivity_v3.py','src/selection_repair_canonical_sensitivity.py')})
    outputs['preparation_receipt.json']=json_bytes(receipt)
    verify_adoption(root);output.mkdir(parents=True)
    for name,data in outputs.items():(output/name).write_bytes(data)
    return inventory


def verify_preparation(root,prepared):
    root,prepared=Path(root),Path(prepared)
    config,decision=verify_adoption(root)
    receipt=json.loads((prepared/'preparation_receipt.json').read_text())
    require(receipt.get('schema_version')==3 and receipt['correction_commit']==CORRECTION_COMMIT
            and receipt['adoption_sha256']==config['adoption_sha256']
            and receipt['authoritative_admission']==STATUS and receipt['projected_fact_eligibility']=='current_eligible', 'mixed/stale v3 preparation')
    required={k+'.mapping_v3.csv' for k in ('qwen_train','llama_train','qwen_validation','llama_validation')}
    required |= {'atomic_D_eligibility.csv','development_fact_eligibility.csv','development_pair_eligibility.csv',
                 'sampler_bindings.json','export_request.json','endpoint_declaration.json','input_inventory.json','physical_export_binding.json'}
    require(set(receipt['artifacts'])==required, 'incomplete v3 preparation')
    for name,pin in receipt['artifacts'].items():
        require(identity((prepared/name).read_bytes())==pin, 'stale v3 preparation: '+name)
    for name,digest in receipt['source_sha256'].items():
        require(sha((root/name).read_bytes())==digest, 'v3 preparation code changed: '+name)
    inventory=json.loads((prepared/'input_inventory.json').read_text())
    require(inventory['config_sha256']==sha((root/CONFIG).read_bytes())
            and inventory['adoption_sha256']==config['adoption_sha256'], 'stale v3 configuration')
    parent=Path(inventory['parent_prepared_root'])
    prior.verify_preparation(root,parent)
    require(identity((parent/'preparation_receipt.json').read_bytes())==inventory['parent_preparation_receipt']
            and sha((parent/'preparation_receipt.json').read_bytes())==config['source_export_preparation_sha256'], 'physical export predecessor changed')
    require((parent/'export_request.json').read_bytes()==(prepared/'export_request.json').read_bytes(), 'mixed physical export request')
    return config,inventory
