"""Source-only T2A overlay. Explicitly permits outer E facts, never predictions.

No activation, probe, compound, or historical development loader is used here.
Original source rows and partition assignments are immutable inputs.
"""
import csv
import hashlib
import io
import json
import re
import shutil
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.data import ENTITY_PATTERNS
from src.entity_partitions import TOPICS, SPLITS, entity_id

CONFIG = 'config/clean_protocol/selection_repair_v1/source_audit.json'
FIELDS = ('source_row_id', 'dataset', 'row_index', 'source_row', 'source_file',
          'source_sha256', 'source_row_sha256', 'original_row_json', 'statement',
          'statement_sha256', 'label', 'topic', 'form', 'entity', 'entity_id',
          'split', 'partition_role', 'paired_source_row_id', 'raw_country',
          'normalized_countries', 'person_key', 'status', 'exclusion_reasons',
          'positive_support_row_ids', 'positive_evidence_ids', 'manifest_status')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def countries(raw, mapping):
    components = [' '.join(x.split()) for x in raw.split('/')]
    require(all(x and x in mapping for x in components), f'Unknown country component: {raw}')
    return sorted({mapping[x] for x in components})


def negation(statement, topic):
    substitutions = {'cities': (' is in ', ' is not in '),
                     'sp_en_trans': (' means ', ' does not mean '),
                     'inventors': (' lived in ', ' did not live in '),
                     'element_symb': (' has the symbol ', ' does not have the symbol '),
                     'animal_class': (' is ', ' is not ')}
    before, after = substitutions[topic]
    require(statement.count(before) == 1, 'Ambiguous negation pairing')
    return statement.replace(before, after, 1)


def load_sources(root, config):
    """Only these ten source CSVs and the original partition manifest are read."""
    partition = root / config['partition_dir']
    metadata = json.loads((partition / 'metadata.json').read_text())
    hashes = {r['file']: r for r in metadata['sources']}
    members = list(csv.DictReader(io.StringIO((partition / 'manifest.csv').read_text())))
    lookup = {(r['topic'], r['entity']): r for r in members}
    require(len(lookup) == len(members), 'Duplicate partition entity')
    require(all(r['entity_id'] == entity_id(r['topic'], r['entity']) and
                r['split'] in SPLITS for r in members), 'Invalid partition identity')
    records, receipts = [], []
    for topic in TOPICS:
        forms = {}
        for form, dataset in [('affirmative', topic), ('negated', 'neg_' + topic)]:
            path = root / config['source_dir'] / (dataset + '.csv')
            payload = path.read_bytes()
            require(digest(payload) == hashes[path.name]['sha256'], f'Source hash mismatch: {path}')
            source = list(csv.DictReader(io.StringIO(payload.decode('utf-8'))))
            require(len(source) == hashes[path.name]['rows'], 'Source row count mismatch')
            receipts.append({'path': str(path.relative_to(root)), 'sha256': digest(payload), 'bytes': len(payload)})
            forms[form] = []
            for i, original in enumerate(source):
                statement, label = original['statement'], original['label']
                match = re.search(ENTITY_PATTERNS[topic], statement)
                require(match is not None and label in ('0', '1'), 'Malformed source row')
                entity = match[1]
                require((topic, entity) in lookup, 'Source entity missing partition')
                member = lookup[topic, entity]
                raw = ''
                if topic == 'inventors':
                    pattern = r'.+? (?:lived|did not live) in (.+)\.'
                    obj = re.fullmatch(pattern, statement)
                    require(obj is not None, 'Invalid inventor country syntax')
                    raw = obj[1]
                row = dict(source_row_id=f'{dataset}:{i}', dataset=dataset, row_index=i,
                           source_row=i+1, source_file=str(path.relative_to(root)),
                           source_sha256=digest(payload), source_row_sha256=digest(encoded(original).encode()),
                           original_row_json=encoded(original), statement=statement,
                           statement_sha256=digest(statement.encode('utf-8')), label=int(label),
                           topic=topic, form=form, entity=entity, entity_id=member['entity_id'],
                           split=member['split'], partition_role={'train':'outer_train','validation':'D','test':'E'}[member['split']],
                           raw_country=raw, normalized_countries=countries(raw, config['country_component_mapping']) if raw else [])
                forms[form].append(row)
        affirmative, negative = forms['affirmative'], forms['negated']
        require(len(affirmative) == len(negative), 'Unpaired source forms')
        # Source negations preserve row order. Verify every transformed statement,
        # entity, label and partition; retain duplicate text as distinct source rows.
        for a, n in zip(affirmative, negative):
            require(n['statement'] == negation(a['statement'], topic) and
                    n['label'] == 1-a['label'] and n['entity_id'] == a['entity_id'] and
                    n['split'] == a['split'], 'Negation pair mismatch')
            a['paired_source_row_id'], n['paired_source_row_id'] = n['source_row_id'], a['source_row_id']
        records.extend(affirmative + negative)
    require({(r['topic'], r['entity']) for r in records} == set(lookup), 'Partition/source entity coverage differs')
    return records, receipts


def normalized_name(name):
    return tuple(re.findall(r'[a-z]+', unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()))


def alias_map(rows, groups, evidence):
    names = sorted({r['entity'] for r in rows if r['topic'] == 'inventors'})
    mapping = {name: entity_id('inventors', name) for name in names}
    decisions, grouped = [], set()
    for group in groups:
        ns = group['names']
        require(len(ns) >= 2 and len(set(ns)) == len(ns) and set(ns) <= set(names), 'Invalid alias members')
        require(not grouped.intersection(ns), 'Overlapping alias groups need explicit adjudication')
        if group['status'] == 'confirmed':
            ev = evidence.get(group['evidence_id'], {})
            require(ev.get('status') == 'confirmed_identity' and ev.get('url') and ev.get('excerpt'), 'Alias lacks identity evidence')
            key = 'person_' + digest(encoded(sorted(ns)).encode())
            mapping.update({name:key for name in ns})
            grouped.update(ns)
        else:
            require(group['status'] == 'unresolved', 'Unknown alias status')
        decisions.append(dict(group))
    known = {frozenset(g['names']) for g in groups}
    for i, a in enumerate(names):
        for b in names[i+1:]:
            x, y = normalized_name(a), normalized_name(b)
            if (x == y or (len(x)>1 and len(y)>1 and (x[0],x[-1]) == (y[0],y[-1]))) and frozenset((a,b)) not in known:
                decisions.append({'names':[a,b], 'status':'unresolved', 'reason':'Normalized-name candidate only; identity unproven'})
    return mapping, decisions


def apply_overlay(rows, config, evidence):
    rows = [dict(r) for r in rows]
    mapping, decisions = alias_map(rows, config['alias_groups'], evidence)
    partitions = defaultdict(set)
    for r in rows:
        r['person_key'] = mapping[r['entity']] if r['topic'] == 'inventors' else r['entity_id']
        partitions[r['person_key']].add(r['split'])
        r.update(exclusion_reasons=[], positive_support_row_ids=[], positive_evidence_ids=[], manifest_status='provisional')
    quarantine = {k for k, parts in partitions.items() if len(parts) > 1}
    supports = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r['topic'] == 'inventors' and r['form'] == 'affirmative' and r['label'] == 1:
            for c in r['normalized_countries']:
                supports[r['person_key']][c].append(r['source_row_id'])
    external = defaultdict(lambda: defaultdict(list))
    for item in config['positive_country_evidence']:
        # Rejected/ambiguous review entries cannot silently become positive evidence.
        require(item['status'] == 'supported_true' and item.get('evidence_id') in evidence,
                'Country support must be explicitly adjudicated supported_true')
        ev = evidence[item['evidence_id']]
        require(ev.get('status') == 'supported_true' and ev.get('url') and ev.get('excerpt'), 'Unverified country evidence')
        require(item['entity'] in mapping, 'Unknown country evidence subject')
        for c in countries(item['country'], config['country_component_mapping']):
            external[mapping[item['entity']]][c].append(item['evidence_id'])
    by_id = {r['source_row_id']:r for r in rows}
    require(len(by_id) == len(rows), 'Duplicate source row IDs')
    for r in rows:
        if r['person_key'] in quarantine:
            r['exclusion_reasons'].append('confirmed_person_spans_outer_partitions')
        if r['topic'] != 'inventors' or r['form'] != 'affirmative' or r['label'] != 0:
            continue
        ids = sorted({i for c in r['normalized_countries'] for i in supports[r['person_key']][c]})
        eids = sorted({i for c in r['normalized_countries'] for i in external[r['person_key']][c]})
        if ids or eids:
            for target in (r, by_id[r['paired_source_row_id']]):
                target['exclusion_reasons'].append('false_affirmative_country_overlaps_supported_true')
                target['positive_support_row_ids'] = ids
                target['positive_evidence_ids'] = eids
    for r in rows:
        r['exclusion_reasons'] = sorted(set(r['exclusion_reasons']))
        r['status'] = 'excluded' if r['exclusion_reasons'] else 'admitted'
    for d in decisions:
        d['partitions'] = sorted({r['split'] for r in rows if r['topic']=='inventors' and r['entity'] in d['names']})
        d['action'] = ('quarantine_all_partitions' if len(d['partitions'])>1 else 'common_person_key') if d['status']=='confirmed' else 'pending_identity_review'
    return rows, decisions


def review_sample(rows, sampling):
    require(sampling['seed'] == 20261001 and sampling['size'] == 30 and
            sampling['rng'] == 'numpy.random.Generator(PCG64)' and
            sampling['stable_order'] == ['dataset', 'row_index'], 'Unexpected sampling policy')
    frame = sorted([dict(r) for r in rows if r['status']=='admitted' and r['topic']=='inventors'
                    and r['form']=='affirmative' and r['label']==0], key=lambda r:(r['dataset'], r['row_index']))
    require(len(frame) >= 30, 'Fewer than 30 retained candidates; do not shrink or resample')
    indices = np.random.Generator(np.random.PCG64(20261001)).choice(len(frame), size=30, replace=False)
    queue = [dict(frame[int(i)], sampling_frame_index=int(i), draw_index=j,
                  review_status='pending', review_evidence='', review_decision='') for j,i in enumerate(indices)]
    return frame, queue


def counts(rows):
    result=[]
    for stage in ('before','admitted','excluded'):
        selected = [r for r in rows if stage=='before' or r['status']==stage]
        for topic in ('ALL',)+TOPICS:
            for split in ('ALL',)+SPLITS:
                for label in ('ALL',0,1):
                    subset=[r for r in selected if (topic=='ALL' or r['topic']==topic) and
                            (split=='ALL' or r['split']==split) and (label=='ALL' or r['label']==label)]
                    result.append(dict(stage=stage,topic=topic,split=split,label=label,rows=len(subset),
                                       unique_entities=len({r['entity_id'] for r in subset}),
                                       unique_person_or_entity_keys=len({r['person_key'] for r in subset})))
    return result


def csv_payload(rows, fields):
    stream=io.StringIO(newline='')
    writer=csv.DictWriter(stream,fieldnames=fields,lineterminator='\n',extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        writer.writerow({k:encoded(v) if isinstance(v,(list,dict)) else v for k,v in row.items() if k in fields})
    return stream.getvalue().encode('utf-8')


def run(root):
    root=Path(root).resolve()
    config=json.loads((root/CONFIG).read_text())
    require(config['protocol']=='selection_repair_v1' and config['audit_version']=='source_audit_v1', 'Unknown audit version')
    evidence_path=Path(CONFIG).parent/'identity_evidence.json'
    evidence=json.loads((root/evidence_path).read_text())
    rows, source_receipts=load_sources(root,config)
    rows, aliases=apply_overlay(rows,config,evidence)
    frame, queue=review_sample(rows,config['sampling'])
    table=counts(rows)
    inputs=[root/CONFIG, root/evidence_path, root/config['partition_dir']/'manifest.csv',
            root/config['partition_dir']/'metadata.json', root/'src/selection_repair_source_audit.py',
            root/'scripts/46_audit_selection_repair_sources.py', root/'src/data.py', root/'src/entity_partitions.py']
    # Opaque hashing preserves review status distinctions; no registry entry is promoted to truth.
    for directory in ('data/clean_protocol/validated_negatives', 'data/clean_protocol/audits/inventor_single_country_v1',
                      'data/clean_protocol/audits/inventor_source_consistency_v1'):
        inputs.extend(sorted(p for p in (root/directory).rglob('*') if p.is_file()))
    receipts=source_receipts+[{'path':str(p.relative_to(root)), 'sha256':digest(p.read_bytes()), 'bytes':p.stat().st_size} for p in inputs]
    outputs={
        'row_manifest.csv':csv_payload(rows,FIELDS),
        'admitted_rows.csv':csv_payload([r for r in rows if r['status']=='admitted'],FIELDS),
        'excluded_rows.csv':csv_payload([r for r in rows if r['status']=='excluded'],FIELDS),
        'review_sampling_frame.csv':csv_payload(frame,FIELDS),
        'review_queue_30.csv':csv_payload(queue,FIELDS+('sampling_frame_index','draw_index','review_status','review_evidence','review_decision')),
        'counts.csv':csv_payload(table, tuple(table[0])),
    }
    summary={'schema_version':1, 'protocol':'selection_repair_v1', 'manifest_status':'provisional',
             'audit_scope':'outer train, validation D, test E SOURCE facts and identities only',
             'source_test_facts_audited':True,'predictions_accessed':False,'activations_accessed':False,
             'fits_performed':0,'compounds_generated':0,'activation_compatibility':'NOT VERIFIED: future reuse requires model/revision, token position, layer and exact sidecar/index checks',
             'identity_convention':'dataset:zero-based row_index; source_row is one-based CSV data row; statement hash is exact UTF-8; source_row_sha256 hashes canonical JSON original columns',
             'alias_decisions':aliases,'unresolved_facts':config['unresolved_facts'],
             'retained_false_facts_status':'unverified: no country-set completeness claim; 30 sampled factual reviews pending',
             'sampling':dict(config['sampling'],frame_rows=len(frame),partition_counts={s:sum(r['split']==s for r in queue) for s in SPLITS}),
             'counts':[r for r in table if r['topic']=='ALL' and r['label']=='ALL'],
             'runtime':{'numpy_version':np.__version__},'inputs':receipts,
             'outputs':{name:{'sha256':digest(payload),'bytes':len(payload)} for name,payload in outputs.items()}}
    outputs['audit_manifest.json']=(json.dumps(summary,indent=2,ensure_ascii=False)+'\n').encode()
    out=root/config['output_dir']
    if out.exists():
        require({p.name for p in out.iterdir()}==set(outputs), 'Existing output set differs; no overwrite')
        require(all((out/name).read_bytes()==payload for name,payload in outputs.items()), 'Existing audit differs; no overwrite or resampling')
    else:
        out.parent.mkdir(parents=True,exist_ok=True)
        staging=Path(tempfile.mkdtemp(prefix='.source-audit-',dir=out.parent))
        try:
            for name,payload in outputs.items(): (staging/name).write_bytes(payload)
            require(all(digest((root/r['path']).read_bytes())==r['sha256'] for r in receipts), 'Input changed during audit')
            staging.rename(out)
        finally:
            if staging.exists(): shutil.rmtree(staging)
    return summary
