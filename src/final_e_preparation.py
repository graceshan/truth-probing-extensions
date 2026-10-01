"""Immutable final E data preparation. No scoring, cache or model entry point.

Uses reviewed facts verbatim. The existing development generator's guard is
never relaxed; only its pure renderer, identities and Boolean validator are reused.
"""
import argparse
import csv
import io
import json
from collections import Counter
from itertools import product
from pathlib import Path

from src.clean_compounds import (FIELDS, EntityManifest, boolean_truth, check,
    render_binary, stable_id, unordered_pair_id, validate_r1)
from src.entity_partitions import canonical_json, csv_bytes, sha256

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path('config/clean_protocol/final_e_preparation_v1.json')
OUTPUT = Path('data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v1')
BASE = Path('data/clean_protocol/selection_repair_v1')
AUDIT = BASE/'final_partition_fact_audit_v1'
OVERLAY = BASE/'candidate_overlay_v5'
PROJECTION = BASE/'final_partition_projection_v1'
PILOT = BASE/'capacity_pilot_projection_v3'
STATUS, ELIGIBLE = 'e_successor_status', 'e_current_eligible'
BOTH, LEAST = 'or_explicit_or_both_v1', 'or_at_least_one_v1'
TEMPLATES = {
    BOTH: 'render_binary(first, second, OR).removesuffix(".") + ", or both."',
    LEAST: '"At least one of the following is true: " + first + " " + second',
}
SEMANTIC = ('source_row_id','paired_source_row_id','topic','entity','entity_id',
            'person_key','split','form','label','statement','statement_sha256')
BOOL_FIELDS = ('canonical_truth_a','canonical_truth_b','surface_first_truth',
               'surface_second_truth','compound_label')
CODE = ('src/final_e_preparation.py','scripts/54_prepare_final_e_benchmark.py',
        'src/clean_compounds.py','src/entity_partitions.py','src/data.py',
        'src/inventor_country_semantics.py','src/priority2_input_controls.py',
        'src/validation_compound_benchmark.py','tests/test_final_e_preparation.py')


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode()


def load(root, path):
    return json.loads((root/path).read_bytes())


def read_csv(root, path):
    return list(csv.DictReader(io.StringIO((root/path).read_text())))


def sig(row):
    return row['topic'], row['person_key'], row['statement_sha256']


def semantic(rows, status):
    return [{**{k:r[k] for k in SEMANTIC}, 'status':r[status]} for r in rows]


def validate_facts(completed, reviews, evidence, source, eligibility, queue, manifest, other_keys):
    """Reject unresolved or substituted facts, including alternate restricted refs."""
    expected = [dict(r, facts=[r['facts'][0],r['facts'][-1]]) for r in reviews if r['status']=='usable']
    check(completed == expected, 'completed pairs differ from exact reviewed judgments')
    by_source = {r['source_row_id']:r for r in source}
    allowed = {(sig(r),r['label']):r for r in eligibility}
    candidates = {r['fact_id']:r for r in queue}
    banned = {sig(dict(f,topic=r['topic'],person_key=r['person_key']))
              for r in reviews for f in r['facts'] if f['judgment']=='unresolved'}
    check(len({r['entity_id'] for r in completed}) == len(completed) and
          len({r['person_key'] for r in completed}) == len(completed), 'duplicate entity/person')
    facts = []
    for r in completed:
        e = manifest.entities[r['entity_id']]
        check(r['status']=='usable' and r['split']==e.split=='test' and e.compound_usable and
              (r['topic'],r['entity'])==(e.topic,e.entity), 'wrong split/entity binding')
        check(r['person_key'] not in other_keys, 'person overlaps train/D/A/T_C')
        rows = [s for s in source if s['entity_id']==r['entity_id']]
        check(rows and {s['person_key'] for s in rows}=={r['person_key']} and
              {s['split'] for s in rows}=={'test'}, 'wrong person binding')
        check(len(r['facts'])==2 and [f['label'] for f in r['facts']]==[1,0], 'missing true/false pair')
        for f in r['facts']:
            label = bool(f['label'])
            check(f['judgment']==('supported_true' if label else 'supported_false'), 'unresolved constituent')
            check(sha256(f['statement'].encode())==f['statement_sha256'], 'fact text hash mismatch')
            claim = r['topic'],r['person_key'],f['statement_sha256']
            check(claim not in banned, 'unresolved exact duplicate')
            matches = [s for s in source if sig(s)==claim]
            check(all(s[STATUS]=='admitted' and s['statement']==f['statement'] and
                      s['label']==str(f['label']) for s in matches), 'restricted source duplicate')
            a = allowed[(claim,str(f['label']))]
            check(a[ELIGIBLE]=='True' and a['completed_pair_eligible']=='True' and
                  a['entity_id']==r['entity_id'] and a['split']=='test' and
                  a['statement']==f['statement'] and a['judgment']==f['judgment'], 'ineligible audited fact')
            fid = stable_id('fact',[r['topic'],r['entity_id'],f['statement'],label])
            if label:
                s = by_source[f['source_row_id']]
                check(s in matches and s['entity_id']==r['entity_id'] and s['form']=='affirmative',
                      'positive source reference mismatch')
            else:
                check(f['fact_id']==fid, 'negative fact ID mismatch')
                q = candidates[fid]
                check(q[ELIGIBLE] is True and q['judgment']=='supported_false' and
                      q['entity_id']==r['entity_id'] and q['person_key']==r['person_key'] and
                      q['statement']==f['statement'], 'unaccepted negative candidate')
            check(f['evidence_ids'], 'missing evidence')
            for evid in f['evidence_ids']:
                ev = evidence[evid]
                binding = dict(topic=r['topic'],entity_id=r['entity_id'],person_key=r['person_key'],
                               statement_sha256=f['statement_sha256'],judgment=f['judgment'])
                check(binding in ev['fact_bindings'] and ev['http_status']==200 and
                      ev['response_sha256'] and ev['observations'], 'evidence binding mismatch')
            facts.append(dict(f, fact_id=fid, topic=r['topic'], entity=r['entity'],
                entity_id=r['entity_id'],person_key=r['person_key'],split='test',
                source_matches=[s['source_row_id'] for s in matches]))
    check(len({f['fact_id'] for f in facts})==len(facts), 'duplicate fact ID')
    return sorted(facts,key=lambda f:f['fact_id'])


def degree_four(ids, topic):
    check(len(ids)==len(set(ids)) and len(ids)>=5, 'degree four needs >=5 unique entities')
    ordered = sorted(ids,key=lambda eid:(sha256(canonical_json(
        ['validation-degree4-ring-v1',0,topic,'test',eid])),eid))
    pairs = sorted({tuple(sorted((ordered[i],ordered[(i+d)%len(ids)])))
                    for i in range(len(ids)) for d in (1,2)})
    degrees = Counter(eid for pair in pairs for eid in pair)
    check(len(pairs)==2*len(ids) and set(degrees)==set(ids) and
          set(degrees.values())=={4}, 'invalid degree-four graph')
    return pairs, ordered, degrees


def render_wording(condition, first, second):
    # Exact TEMPLATES / variants() definitions in priority2_input_controls.py.
    # Reimplemented purely to avoid importing its cache/evaluation dependencies.
    if condition==BOTH:
        return render_binary(first,second,'OR').removesuffix('.')+', or both.'
    check(condition==LEAST, 'undefined wording template')
    return 'At least one of the following is true: '+first+' '+second


def example(manifest, facts, a, b, ta, tb, operator, ordering):
    manifest.authorize_pair(a,b,'test')
    fa,fb = facts[(a,ta)],facts[(b,tb)]
    first,second = (fa,fb) if ordering=='AB' else (fb,fa)
    pid = unordered_pair_id(manifest,a,b,'test')
    template = f"clean-binary-v1/{fa['topic']}/{operator.lower()}"
    statement = render_binary(first['statement'],second['statement'],operator)
    eid = stable_id('example',[pid,'test',fa['fact_id'],fb['fact_id'],operator,ordering,template,statement])
    return dict(zip(FIELDS,(eid,statement,fa['topic'],'test',a,b,fa['fact_id'],fb['fact_id'],pid,
        ta,tb,first['entity_id'],second['entity_id'],bool(first['label']),bool(second['label']),
        operator,ordering,template,boolean_truth(operator,(ta,tb)),0,fa['statement'],fb['statement'])))


def validate_rows(rows, manifest, facts, requested):
    validate_r1(rows,manifest,'test',requested)
    lookup = {(f['entity_id'],bool(f['label'])):f for f in facts}
    for row in rows:
        expected = example(manifest,lookup,row['entity_a_id'],row['entity_b_id'],
            row['canonical_truth_a'],row['canonical_truth_b'],row['operator'],row['ordering'])
        check(row==expected, 'compound exact binding/rendering/identity mismatch')


def validate_source_and_preservation(root, source):
    byid = {r['source_row_id']:r for r in source}
    check(len(byid)==len(source), 'duplicate source identity')
    for r in source:
        p = byid[r['paired_source_row_id']]
        check(p['paired_source_row_id']==r['source_row_id'] and
              all(r[k]==p[k] for k in (STATUS,'person_key','entity_id','topic','split')) and
              {r['label'],p['label']}=={'0','1'} and {r['form'],p['form']}=={'affirmative','negated'},
              'paired source quarantine propagation failure')
    eligibility = read_csv(root,OVERLAY/'source_fact_eligibility.csv')
    for r in eligibility:
        s=byid[r['source_row_id']]
        check(r[ELIGIBLE]==str(s[STATUS]=='admitted') and r['current_status']==s[STATUS],
              'stale source eligibility')
    restrictions = {sig(r) for r in source if r[STATUS]!='admitted'}
    for r in read_csv(root,OVERLAY/'fact_inventory.csv'):
        check(sig(r) not in restrictions or r[ELIGIBLE]=='False', 'restricted registry duplicate')
    comparison = {}
    old = read_csv(root,BASE/'candidate_overlay_v4/row_manifest.csv')
    for split in ('train','validation'):
        left = semantic([r for r in old if r['split']==split],'tc_successor_status')
        right = semantic([r for r in source if r['split']==split],STATUS)
        check(left==right, 'non-E semantic change')
        comparison[split]=dict(unchanged=True,semantic_sha256=sha256(encoded(right)))
    for name in ('train_admitted_membership','validation_admitted_membership','test_admitted_membership'):
        split = name.split('_')[0]
        member = read_csv(root,PROJECTION/(name+'.csv'))
        check(member==[r for r in source if r['split']==split and r[STATUS]=='admitted'], 'stale membership')
    for name in ('P15_admitted_membership','P10_admitted_membership','P15_balanced_exposure',
                 'P10_balanced_exposure','train_balanced_exposure'):
        oldname = 'corrected_outer_training_balanced' if name=='train_balanced_exposure' else name
        left = semantic(read_csv(root,PILOT/(oldname+'.csv')),'tc_successor_status')
        right = semantic(read_csv(root,PROJECTION/(name+'.csv')),STATUS)
        check(left==right, 'non-E sampler/membership change')
        comparison[name]=dict(unchanged=True,semantic_sha256=sha256(encoded(right)))
    for name, oldpath in [('A15_proposal.json',PILOT/'A15_proposal.json'),
                         ('A10_proposal.json',PILOT/'A10_proposal.json'),
                         ('TC_completed_pairs.json',BASE/'tc_topup_audit_v1/combined_TC_completed_pairs.json'),
                         ('outer_training_reference.csv',PILOT/'outer_training_reference.csv')]:
        left,right = (root/oldpath).read_bytes(),(root/PROJECTION/name).read_bytes()
        check(left==right, 'A/T_C/reference changed')
        comparison[name]=dict(unchanged=True,sha256=sha256(right))
    return comparison


def build(root=ROOT):
    root = Path(root)
    config = load(root,CONFIG)
    check(config['final_evaluation_enabled'] is False and config['P_A_frozen'] is False and
          config['split']=='test' and type(config['seed']) is int and config['seed']==0 and
          config['adapter_version']=='final-e-preparation-v1' and config['schema_version']==1 and
          config['pair_recipe']=='final-e-degree4-ring-v1' and config['hash_namespace']=='validation-degree4-ring-v1' and
          config['template_version']=='clean-binary-v1' and config['status_column']==STATUS and
          config['eligibility_column']==ELIGIBLE and config['correction_base']==OVERLAY.name and
          config['projection']==PROJECTION.name and config['wording_conditions']==[BOTH,LEAST], 'invalid preparation scope')
    for path,digest in config['input_sha256'].items():
        check(sha256((root/path).read_bytes())==digest,f'input hash changed: {path}')
    check(load(root,Path('config/clean_protocol/generalization_protocols.json'))['final_evaluation_enabled'] is False,
          'final evaluation must stay closed')
    manifest = EntityManifest(root/'data/clean_protocol/entity_partitions/manifest.csv',
                              root/'data/clean_protocol/entity_partitions/metadata.json')
    source = read_csv(root,OVERLAY/'row_manifest.csv')
    comparison = validate_source_and_preservation(root,source)
    other = {r['person_key'] for r in source if r['split']!='test'}
    for n in (10,15):
        other.update(r['person_key'] for r in load(root,PROJECTION/f'A{n}_proposal.json')['completed_pairs'])
    other.update(r['person_key'] for r in load(root,PROJECTION/'TC_completed_pairs.json'))
    completed = load(root,AUDIT/'completed_E_fact_pairs.json')
    evidence = load(root,AUDIT/'evidence.json')
    facts = validate_facts(completed,load(root,AUDIT/'reviews.json'),evidence,source,
        read_csv(root,PROJECTION/'audited_fact_eligibility.csv'),
        load(root,PROJECTION/'E_negative_candidate_eligibility.json'),manifest,other)
    lookup = {(f['entity_id'],bool(f['label'])):f for f in facts}
    rows,pairs,degrees,counts,orders = [],[],[],[],{}
    audit_counts = {r['topic']:r for r in read_csv(root,AUDIT/'capacity_counts.csv')}
    for topic in sorted(audit_counts):
        ids=[r['entity_id'] for r in completed if r['topic']==topic]
        topic_pairs,order,degree = degree_four(ids,topic)
        orders[topic]=order
        for a,b in topic_pairs:
            pairs.append(dict(pair_id=unordered_pair_id(manifest,a,b,'test'),topic=topic,split='test',
                              entity_a_id=a,entity_b_id=b,person_a_key=lookup[(a,True)]['person_key'],
                              person_b_key=lookup[(b,True)]['person_key']))
            rows.extend(example(manifest,lookup,a,b,ta,tb,op,order)
                for ta,tb in product((True,False),repeat=2) for op in ('AND','OR') for order in ('AB','BA'))
        degrees.extend(dict(topic=topic,entity_id=eid,person_key=lookup[(eid,True)]['person_key'],degree=d)
                       for eid,d in sorted(degree.items()))
        count=dict(topic=topic,entities=len(ids),pairs=len(topic_pairs),bare_rows=16*len(topic_pairs),
            or_both_rows=8*len(topic_pairs),at_least_one_rows=8*len(topic_pairs),isolated_facts=2*len(ids),
            admitted_atomic_rows=sum(r['topic']==topic and r['split']=='test' and r[STATUS]=='admitted' for r in source))
        for key,ak in [('entities','usable_entities'),('pairs','recipe_pairs'),('bare_rows','expected_binary_rows')]:
            check(count[key]==int(audit_counts[topic][ak]), 'audit capacity differs from generation')
        counts.append(count)
    requested = {r['topic']:r['pairs'] for r in counts}
    rows.sort(key=lambda r:r['example_id'])
    validate_rows(rows,manifest,facts,requested)
    check((len(completed),len(pairs),len(rows))==(225,450,7200), 'reviewed E capacity changed')
    bare_bytes = csv_bytes(rows,FIELDS)
    decoded=list(csv.DictReader(io.StringIO(bare_bytes.decode())))
    for r in decoded:
        for k in BOOL_FIELDS:
            check(r[k] in ('True','False'), 'invalid Boolean serialization')
            r[k]=r[k]=='True'
        r['generation_seed']=int(r['generation_seed'])
    check(decoded==rows,'bare CSV round trip')
    validate_rows(decoded,manifest,facts,requested)
    wording=[]
    for r in rows:
        if r['operator']!='OR': continue
        first,second=(r['fact_a_statement'],r['fact_b_statement']) if r['ordering']=='AB' else (r['fact_b_statement'],r['fact_a_statement'])
        for condition in config['wording_conditions']:
            text=render_wording(condition,first,second)
            wording.append(dict(r, example_id=stable_id('e_wording',[condition,r['example_id'],text]),
                statement=text,template_id=condition,base_example_id=r['example_id'],condition_id=condition))
    wording.sort(key=lambda r:r['example_id'])
    check(len({r['example_id'] for r in rows+wording})==len(rows+wording),'duplicate example identity')
    check(Counter(r['condition_id'] for r in wording)=={BOTH:len(rows)//2,LEAST:len(rows)//2}, 'wording coverage gap')
    # Each OR wording family retains the exact bare AND rows, as in Priority 2.
    wording_by_base={(r['condition_id'],r['base_example_id']):r for r in wording}
    condition_index=[]
    for condition in ('bare',BOTH,LEAST):
        for r in rows:
            selected=r if condition=='bare' or r['operator']=='AND' else wording_by_base[(condition,r['example_id'])]
            condition_index.append(dict(condition_id=condition,base_example_id=r['example_id'],
                selected_example_id=selected['example_id'],pair_id=r['pair_id'],operator=r['operator'],
                reused_bare=str(selected is r),artifact='bare_compounds.csv' if selected is r else 'wording_compounds.csv'))
    check(Counter(r['condition_id'] for r in condition_index)==dict.fromkeys(('bare',BOTH,LEAST),len(rows)),
          'wording family coverage gap')
    atomic=[r for r in source if r['split']=='test' and r[STATUS]=='admitted']
    files={'bare_compounds.csv':bare_bytes,'wording_compounds.csv':csv_bytes(wording,tuple(wording[0])),
        'pairs.csv':csv_bytes(pairs,tuple(pairs[0])),'entity_degrees.csv':csv_bytes(degrees,tuple(degrees[0])),
        'counts.csv':csv_bytes(counts,tuple(counts[0])),'isolated_facts.json':encoded(facts),
        'admitted_atomic_E.csv':csv_bytes(atomic,tuple(atomic[0])),
        'pair_hash_order.json':encoded(orders),'non_E_preservation.json':encoded(comparison),
        'condition_index.csv':csv_bytes(condition_index,tuple(condition_index[0]))}
    selected_evidence={key:evidence[key] for f in facts for key in f['evidence_ids']}
    files['constituent_evidence.json']=encoded(selected_evidence)
    # Reversible logical-item -> exact text binding. Text IDs never replace source IDs.
    bindings,texts=[],{}
    def bind(kind,identity,statement,path,record):
        text_id='text_'+sha256(statement.encode())
        check(text_id not in texts or texts[text_id]==statement,'text digest collision')
        texts[text_id]=statement
        bindings.append(dict(binding_id=stable_id('extraction_binding',[kind,identity]),kind=kind,
            logical_id=identity,text_id=text_id,statement_sha256=sha256(statement.encode()),
            artifact=path,record_index=record,source_row_id='',fact_id='',pair_id='',fact_a_id='',fact_b_id='',
            entity_id='',person_key='',condition_id=''))
    for kind,rs,path in [('bare',rows,'bare_compounds.csv'),('wording',wording,'wording_compounds.csv')]:
        for i,r in enumerate(rs):
            bind(kind,r['example_id'],r['statement'],path,i)
            bindings[-1].update(pair_id=r['pair_id'],fact_a_id=r['fact_a_id'],fact_b_id=r['fact_b_id'],
                               condition_id=r.get('condition_id','bare'))
    for i,f in enumerate(facts):
        bind('isolated',f['fact_id'],f['statement'],'isolated_facts.json',i)
        bindings[-1].update(fact_id=f['fact_id'],entity_id=f['entity_id'],person_key=f['person_key'],
                           source_row_id=f.get('source_row_id',''))
    for i,r in enumerate(atomic):
        bind('atomic',r['source_row_id'],r['statement'],'admitted_atomic_E.csv',i)
        bindings[-1].update(source_row_id=r['source_row_id'],entity_id=r['entity_id'],person_key=r['person_key'])
    check(len({b['binding_id'] for b in bindings})==len(bindings),'duplicate extraction binding')
    for b in bindings:
        table={'bare':rows,'wording':wording,'isolated':facts,'atomic':atomic}[b['kind']]
        r=table[b['record_index']]
        check(texts[b['text_id']]==r['statement'] and sha256(r['statement'].encode())==b['statement_sha256'],
              'irreversible text binding')
    text_counts=Counter(b['text_id'] for b in bindings)
    text_rows=[dict(text_id=k,statement=v,statement_sha256=sha256(v.encode()),
                   binding_count=text_counts[k]) for k,v in sorted(texts.items())]
    files['extraction_bindings.csv']=csv_bytes(bindings,tuple(bindings[0]))
    files['extraction_texts.csv']=csv_bytes(text_rows,tuple(text_rows[0]))
    # All CSV fields must serialize without loss (typed bare validation above).
    for name,payload in list(files.items()):
        if name.endswith('.csv'):
            decoded=list(csv.DictReader(io.StringIO(payload.decode())))
            check(csv_bytes(decoded,tuple(decoded[0]))==payload,f'CSV serialization mismatch: {name}')
    files['configuration.json']=(root/CONFIG).read_bytes()
    metadata=dict(adapter_version=config['adapter_version'],scope_decision=config['scope_decision'],
        final_evaluation_enabled=False,P_A_frozen=False,extraction_launched=False,predictions_accessed=False,
        counts=counts,entities=len(completed),unordered_pairs=len(pairs),bare_rows=len(rows),
        wording_rows=dict(Counter(r['condition_id'] for r in wording)),
        condition_rows=dict(Counter(r['condition_id'] for r in condition_index)),isolated_facts=len(facts),
        admitted_atomic_rows=len(atomic),extraction_bindings=len(bindings),unique_texts=len(texts),
        repeated_text_bindings=len(bindings)-len(texts),non_E_inputs_changed=False,
        statistical_note='Data counts are not independent statistical sample sizes: rows share pairs, entities and facts; wording variants share bare identities.',
        pair_recipe='final-e-degree4-ring-v1 adapts validation-degree4-ring-v1 without changing its hash namespace; seed 0, split test, offsets 1/2, canonical endpoints.',
        templates=TEMPLATES,template_source='src/priority2_input_controls.py:TEMPLATES and variants',
        pending_templates=['No distinct held-out AND wording or additional October 2–3 template definition is present in the reviewed repository; no such wording was invented.'],
        extraction_policy='Inventory only. Deduplicate exact UTF-8 raw text within the same future model/tokenizer/representation/prompt contract; every logical/source identity remains in reversible bindings. No activation cache selected or reused.',
        preserved_audit='All 36 initial exclusions, 35 unresolved entities, 261 negative attempts and the rejected candidate remain in the hash-bound original audit.',
        inputs_sha256=config['input_sha256'],config_sha256=sha256((root/CONFIG).read_bytes()),
        code_sha256={p:sha256((root/p).read_bytes()) for p in CODE},
        output_sha256={p:sha256(v) for p,v in files.items()},
        checks=['exact audited facts/evidence/person bindings','v5 paired quarantine and duplicate restrictions',
                'train/D/A/T_C and sampler unchanged','degree four and audit capacity','all sixteen truth/operator/order variants',
                'central Boolean labels and exact renderer','unique identities','lossless serialization',
                'reversible extraction identity/text map','final scoring disabled'])
    files['manifest.json']=encoded(metadata)
    return files,metadata


def materialize(root,files,write=False):
    # Preflight all files before any writes; immutable version refuses any overwrite.
    for name,payload in files.items():
        path=Path(root)/OUTPUT/name
        check(not path.exists() or path.read_bytes()==payload,f'differing immutable output: {name}')
        check(write or path.exists(),f'missing output: {name}')
    if write:
        (Path(root)/OUTPUT).mkdir(parents=True,exist_ok=True)
        for name,payload in files.items():
            path=Path(root)/OUTPUT/name
            if not path.exists(): path.write_bytes(payload)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write',action='store_true',help='create missing immutable data only')
    args=parser.parse_args()
    files,meta=build()
    materialize(ROOT,files,args.write)
    print(json.dumps({k:meta[k] for k in ('entities','unordered_pairs','bare_rows','wording_rows',
        'isolated_facts','admitted_atomic_rows','extraction_bindings','unique_texts','final_evaluation_enabled')},indent=2))


if __name__=='__main__':
    main()
