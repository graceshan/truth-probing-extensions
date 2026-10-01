"""Data-only immutable D/E Section 9 wording; never load predictions or activations."""
import argparse
import csv
import io
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from src import final_e_preparation as old
from src.heldout_and_wording import CONDITION, render, validate_spec
from src.clean_compounds import FIELDS, boolean_truth, check, render_binary, stable_id, validate_r1

ROOT=old.ROOT
CONFIG=Path('config/clean_protocol/heldout_and_preparation_v1.json')
SPEC=Path('config/clean_protocol/heldout_and_wording_v1.json')
INPUTS=Path('data/clean_protocol/heldout_and_inputs_v1')
E_OUTPUT=old.OUTPUT.with_name('final_e_preparation_v2')
D_OUTPUT=old.OUTPUT.with_name('development_heldout_and_v1')
CODE=('src/heldout_and_wording.py','src/heldout_and_preparation.py',
      'scripts/58_prepare_heldout_and_wording.py','tests/test_heldout_and_wording.py')


def table(data):
    return list(csv.DictReader(io.StringIO(data.decode())))


def csv_bytes(rows):
    return old.csv_bytes(rows,tuple(rows[0]))


def normalized(rows):
    rows=[dict(r) for r in rows]
    for r in rows:
        for key in old.BOOL_FIELDS:
            check(r[key] in ('True','False'), 'invalid Boolean serialization')
            r[key]=r[key]=='True'
        r['generation_seed']=int(r['generation_seed'])
    return rows


def d_fact_eligibility(facts,source,ledger):
    """Pure metadata implementation of the reviewed v4 current_fact_eligibility contract.

    Reproduced output must equal the hash-pinned authoritative v4 table in full.
    Original source labels alone do not accept generated negatives.
    """
    results=[]
    for f in facts:
        check(f['split']=='validation' and f['truth'] in ('True','False'), 'wrong D split/truth')
        truth=f['truth']=='True'
        record={k:v for k,v in f.items() if k!='fact_key'};record['truth']=truth
        digest=old.sha256(json.dumps(record,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode())
        check(f['fact_key']=='fact_'+digest,'D fact-key binding')
        if truth:
            check(f['fact_id']==stable_id('fact',[f['topic'],f['entity_id'],f['statement'],True]),'D positive identity')
        originals=[r for r in source if r['split']=='validation' and r['form']=='affirmative' and
                   all(r[k]==f[k] for k in ('topic','entity_id','statement'))]
        conflicts=[r for r in originals if r['tc_successor_status']!='admitted' or r['label']!=str(int(truth))]
        accepted=[r for r in originals if r['tc_successor_status']=='admitted' and r['label']==str(int(truth))]
        registry=[r for r in ledger if r['split']=='validation' and r['origin']!='source' and r['fact_ref']==f['fact_id'] and
                  all(r[k]==f[k] for k in ('topic','entity_id','statement')) and r['label']=='0_candidate' and
                  r['tc_current_eligible']=='True' and r['support'] in ('historical_external_negative','historical_source_label_negative')]
        allowed=not conflicts and bool(accepted if truth else registry)
        reasons=['restricted_or_conflicting_exact_source_claim'] if conflicts else []
        if truth and not accepted: reasons.append('no_admitted_exact_positive_source')
        if not truth and not registry: reasons.append('no_eligible_exact_registry_negative')
        people={r['person_key'] for r in accepted+registry};original_people={r['person_key'] for r in originals}
        check(len(people)<=1 and len(original_people)<=1,'ambiguous D person')
        results.append(dict(fact_key=f['fact_key'],fact_id=f['fact_id'],entity_id=f['entity_id'],topic=f['topic'],
            statement_sha256=old.sha256(f['statement'].encode()),truth=f['truth'],
            person_key=next(iter(people or original_people),f['entity_id']),reasons=json.dumps(reasons),
            source_row_ids=json.dumps([r['source_row_id'] for r in originals]),
            source_evidence_tier='retained_source_label_not_human_verification' if truth else '',
            registry_evidence=json.dumps([dict(origin=r['origin'],support=r['support'],evidence_ref=r['evidence_ref']) for r in registry]),
            tc_current_eligible=str(allowed)))
    return results


def d_pair_eligibility(raw,facts,mapping):
    byfact={r['fact_key']:r for r in facts};bymap={r['base_example_id']:r for r in mapping}
    check(len(byfact)==len(facts) and len(bymap)==len(mapping),'duplicate D fact/map')
    check(set(bymap)=={r['example_id'] for r in raw},'D map coverage')
    groups=defaultdict(list)
    for row in raw:groups[row['pair_id']].append(row)
    pairs=[]
    for pair,rows in sorted(groups.items()):
        keys={bymap[r['example_id']][f'fact_{side}_key'] for r in rows for side in ('a','b')}
        check(len(rows)==16 and len(keys)==4,'incomplete D pair')
        rejected=sorted(k for k in keys if byfact[k]['tc_current_eligible']!='True')
        pairs.append(dict(pair_id=pair,topic=rows[0]['topic'],entity_a_id=rows[0]['entity_a_id'],entity_b_id=rows[0]['entity_b_id'],
            original_rows='16',retained_rows='0' if rejected else '16',rejected_fact_keys=json.dumps(rejected),
            all_fact_keys=json.dumps(sorted(keys)),tc_current_eligible=str(not rejected)))
    return pairs


def validate_base(rows,facts,manifest,split):
    requested=Counter({topic:len({r['pair_id'] for r in rows if r['topic']==topic}) for topic in {r['topic'] for r in rows}})
    validate_r1(rows,manifest,split,requested)
    for r in rows:
        for side in ('a','b'):
            f=facts[r[f'fact_{side}_id']]
            check(f['eligible'] is True,'ineligible constituent')
            check(f['entity_id']==r[f'entity_{side}_id'] and f['topic']==r['topic'] and f['split']==split and
                  f['statement']==r[f'fact_{side}_statement'] and f['truth']==r[f'canonical_truth_{side}'],
                  'substituted text or constituent identity')
        first,second=(r['fact_a_statement'],r['fact_b_statement']) if r['ordering']=='AB' else (r['fact_b_statement'],r['fact_a_statement'])
        check(r['statement']==render_binary(first,second,r['operator']),'bare renderer mismatch')
        check(r['example_id']==stable_id('example',[r['pair_id'],split,r['fact_a_id'],r['fact_b_id'],
              r['operator'],r['ordering'],r['template_id'],r['statement']]),'bare example identity')


def wording(row,facts):
    check(row['operator']=='AND','wrong operator for AND wording')
    for side in ('a','b'):
        f=facts[row[f'fact_{side}_id']]
        check(f['eligible'] is True,'ineligible constituent')
        check(f['statement']==row[f'fact_{side}_statement'] and f['entity_id']==row[f'entity_{side}_id'] and
              f['truth']==row[f'canonical_truth_{side}'] and f['split']==row['split'],'substituted text/fact binding')
    check(row['compound_label']==boolean_truth('AND',(row['canonical_truth_a'],row['canonical_truth_b'])),'wrong AND label')
    check(row['ordering'] in ('AB','BA'),'wrong surface order')
    first,second=(row['fact_a_statement'],row['fact_b_statement']) if row['ordering']=='AB' else (row['fact_b_statement'],row['fact_a_statement'])
    text=render(first,second,row['operator'])
    return dict(row,example_id=stable_id('and_wording',[CONDITION,row['split'],row['example_id'],text]),
        statement=text,template_id=CONDITION,base_example_id=row['example_id'],condition_id=CONDITION)


def validate_wording(rows,bare,facts):
    expected={r['example_id']:r for r in bare if r['operator']=='AND'}
    check(len(rows)==len(expected) and len({r['example_id'] for r in rows})==len(rows) and
          {r['base_example_id'] for r in rows}==set(expected),'wording identity/coverage')
    groups=defaultdict(list)
    for r in rows:
        check(r==wording(expected[r['base_example_id']],facts),'changed wording, label or fact/pair binding')
        groups[r['pair_id']].append(r)
    for group in groups.values():
        cells={(r['canonical_truth_a'],r['canonical_truth_b'],r['ordering']) for r in group}
        check(len(group)==8 and cells==set(product((True,False),(True,False),('AB','BA'))),'incomplete AND truth/order cells')


def extend_inventory(bindings,texts,rows,filename):
    """Append reversible references; preserve every preexisting logical binding."""
    bindings=[dict(r) for r in bindings]
    bytext={r['text_id']:r['statement'] for r in texts}
    for i,r in enumerate(rows):
        tid='text_'+old.sha256(r['statement'].encode())
        check(tid not in bytext or bytext[tid]==r['statement'],'text identity collision')
        bytext[tid]=r['statement']
        bindings.append(dict(binding_id=stable_id('extraction_binding',['wording',r['example_id']]),kind='wording',
            logical_id=r['example_id'],text_id=tid,statement_sha256=old.sha256(r['statement'].encode()),artifact=filename,
            record_index=str(i),source_row_id='',fact_id='',pair_id=r['pair_id'],fact_a_id=r['fact_a_id'],fact_b_id=r['fact_b_id'],
            entity_id='',person_key='',condition_id=CONDITION))
    check(len({r['binding_id'] for r in bindings})==len(bindings),'duplicate extraction ID')
    counts=Counter(r['text_id'] for r in bindings)
    for r in bindings:
        check(r['statement_sha256']==old.sha256(bytext[r['text_id']].encode()),'text binding mismatch')
    texts=[dict(text_id=k,statement=v,statement_sha256=old.sha256(v.encode()),binding_count=counts[k]) for k,v in sorted(bytext.items())]
    return bindings,texts


def condition_index(bare,new):
    chosen={r['base_example_id']:r for r in new}
    return [dict(condition_id=CONDITION,base_example_id=r['example_id'],selected_example_id=chosen.get(r['example_id'],r)['example_id'],
        pair_id=r['pair_id'],operator=r['operator'],reused_bare=str(r['operator']=='OR'),
        artifact=CONDITION+'.csv' if r['operator']=='AND' else 'bare_compounds.csv') for r in bare]


def validate_reversible(bindings,texts,files):
    tables={name:table(payload) for name,payload in files.items() if name.endswith('.csv') and name not in ('extraction_bindings.csv','extraction_texts.csv')}
    if 'isolated_facts.json' in files:tables['isolated_facts.json']=json.loads(files['isolated_facts.json'])
    text={r['text_id']:r['statement'] for r in texts}
    for b in bindings:
        row=tables[b['artifact']][int(b['record_index'])]
        check(row['statement']==text[b['text_id']],'irreversible extraction binding')
        key={'bare':'example_id','wording':'example_id','isolated':'fact_id','atomic':'source_row_id'}[b['kind']]
        check(row[key]==b['logical_id'],'wrong extraction logical identity')
        for key in ('pair_id','fact_a_id','fact_b_id','source_row_id'):
            if b.get(key):check(row.get(key)==b[key],'wrong extraction provenance binding')


def build(root=ROOT):
    root=Path(root);config=old.load(root,CONFIG);spec=old.load(root,SPEC);validate_spec(spec)
    check(config['schema_version']==1 and config['version']=='heldout-and-preparation-v1' and
          config['condition_id']==CONDITION and config['specification']==str(SPEC) and
          config['E_base']==old.OUTPUT.name and config['E_successor']==E_OUTPUT.name and
          config['final_evaluation_enabled'] is False and config['P_A_frozen'] is False and
          config['held_out_from_compound_fitting'] is True and config['held_out_from_selection'] is True,
          'preparation scope changed')
    for path,digest in config['input_sha256'].items():
        check(old.sha256((root/path).read_bytes())==digest,'changed pinned input: '+path)
    old_files,_=old.build(root)
    for name,payload in old_files.items():check((root/old.OUTPUT/name).read_bytes()==payload,'historical E v1 changed')
    manifest=old.EntityManifest(root/'data/clean_protocol/entity_partitions/manifest.csv',root/'data/clean_protocol/entity_partitions/metadata.json')
    preservation=old.validate_source_and_preservation(root,old.read_csv(root,old.OVERLAY/'row_manifest.csv'))
    # E facts remain exactly the completed audited facts verified by the v1 builder.
    e_facts={f['fact_id']:dict(f,truth=bool(f['label']),eligible=True) for f in json.loads(old_files['isolated_facts.json'])}
    e_bare=normalized(table(old_files['bare_compounds.csv']))
    validate_base(e_bare,e_facts,manifest,'test')
    # D: reproduce the exact reviewed v4 admission tables from recovered metadata.
    d_raw=old.read_csv(root,INPUTS/'D_original_metadata.csv');d_original=old.read_csv(root,INPUTS/'D_isolated_facts.csv')
    mapping=old.read_csv(root,INPUTS/'D_constituent_map.csv')
    d_current=d_fact_eligibility(d_original,old.read_csv(root,old.BASE/'candidate_overlay_v4/row_manifest.csv'),
                               old.read_csv(root,old.BASE/'candidate_overlay_v4/fact_inventory.csv'))
    check(d_current==old.read_csv(root,INPUTS/'development_fact_eligibility.csv'),'D v4 fact eligibility differs')
    d_pairs=d_pair_eligibility(d_raw,d_current,mapping)
    check(d_pairs==old.read_csv(root,INPUTS/'development_pair_eligibility.csv'),'D v4 pair cohort differs')
    bykey={f['fact_key']:f for f in d_original};bymap={r['base_example_id']:r for r in mapping}
    current={f['fact_key']:f for f in d_current}
    for row in d_raw:
        for side in ('a','b'):
            f=bykey[bymap[row['example_id']][f'fact_{side}_key']]
            check(all(f[k]==v for k,v in dict(fact_id=row[f'fact_{side}_id'],entity_id=row[f'entity_{side}_id'],
                topic=row['topic'],statement=row[f'fact_{side}_statement'],truth=row[f'canonical_truth_{side}']).items()),'D map identity mismatch')
    admitted={r['pair_id'] for r in d_pairs if r['tc_current_eligible']=='True'}
    d_bare=normalized([r for r in d_raw if r['pair_id'] in admitted])
    d_facts={f['fact_id']:dict(f,truth=f['truth']=='True',eligible=current[f['fact_key']]['tc_current_eligible']=='True',
                person_key=current[f['fact_key']]['person_key']) for f in d_original}
    validate_base(d_bare,d_facts,manifest,'validation')
    e_people={f['person_key'] for f in e_facts.values()}
    d_people={d_facts[r[f'fact_{side}_id']]['person_key'] for r in d_bare for side in ('a','b')}
    check(not e_people & d_people,'D/E person overlap')
    outputs={};summaries={}
    for split,bare,facts,out in [('E',e_bare,e_facts,E_OUTPUT),('D',d_bare,d_facts,D_OUTPUT)]:
        new=sorted((wording(r,facts) for r in bare if r['operator']=='AND'),key=lambda r:r['example_id'])
        validate_wording(new,bare,facts)
        old_ids={r['example_id'] for r in bare}
        if split=='E':old_ids.update(r['example_id'] for r in table(old_files['wording_compounds.csv']))
        check(not old_ids & {r['example_id'] for r in new},'wording ID collides with historical example')
        if split=='E':
            # Keep every prior benchmark and factual file byte-identical; extend only indexes/counts.
            files={k:v for k,v in old_files.items() if k not in ('manifest.json','configuration.json','condition_index.csv','counts.csv','extraction_bindings.csv','extraction_texts.csv')}
            files['predecessor_manifest.json']=old_files['manifest.json']
            index=table(old_files['condition_index.csv'])+condition_index(bare,new)
            bindings,texts=extend_inventory(table(old_files['extraction_bindings.csv']),table(old_files['extraction_texts.csv']),new,CONDITION+'.csv')
            check(bindings[:len(table(old_files['extraction_bindings.csv']))]==table(old_files['extraction_bindings.csv']),'historical extraction bindings changed')
        else:
            used={r[f'fact_{side}_id'] for r in bare for side in ('a','b')}
            files={'bare_compounds.csv':csv_bytes(bare),'constituent_facts.json':old.encoded([facts[k] for k in sorted(used)]),
                   'development_fact_eligibility.csv':(root/INPUTS/'development_fact_eligibility.csv').read_bytes(),
                   'development_pair_eligibility.csv':(root/INPUTS/'development_pair_eligibility.csv').read_bytes()}
            index=condition_index(bare,new)
            bindings,texts=extend_inventory([],[],new,CONDITION+'.csv')
        files[CONDITION+'.csv']=csv_bytes(new)
        files['condition_index.csv']=csv_bytes(index)
        files['extraction_bindings.csv']=csv_bytes(bindings);files['extraction_texts.csv']=csv_bytes(texts)
        validate_reversible(bindings,texts,files)
        counts=[]
        for topic in sorted({r['topic'] for r in bare}):
            selected=[r for r in bare if r['topic']==topic]
            counts.append(dict(topic=topic,entities=len({r[f'entity_{s}_id'] for r in selected for s in ('a','b')}),
                pairs=len({r['pair_id'] for r in selected}),bare_rows=len(selected),and_wording_rows=sum(r['operator']=='AND' for r in selected)))
        if split=='E':
            previous_counts={r['topic']:r for r in table(old_files['counts.csv'])}
            for r in counts:
                r.update({k:int(v) for k,v in previous_counts[r['topic']].items() if k not in r})
        files['counts.csv']=csv_bytes(counts);files['wording_specification.json']=(root/SPEC).read_bytes()
        files['configuration.json']=(root/CONFIG).read_bytes();files['non_E_preservation.json']=old.encoded(preservation)
        for name,payload in files.items():
            if name.endswith('.csv'):check(csv_bytes(table(payload))==payload,'lossy CSV serialization: '+name)
        # Validate the actual serialized wording with full typed identity equality.
        validate_wording(normalized(table(files[CONDITION+'.csv'])),bare,facts)
        summary=dict(split=split,counts=counts,pairs=sum(r['pairs'] for r in counts),and_wording_rows=len(new),
            extraction_bindings=len(bindings),unique_texts=len(texts),deduplication_savings=len(bindings)-len(texts),
            extraction_inventory_scope='complete E bare/OR/AND/isolated/atomic inventory' if split=='E' else 'new D AND wording only; bare OR rows reuse their original identities via condition_index',
            condition_changes={'bare':'none',old.BOTH:'OR only; reuse bare AND',old.LEAST:'OR only; reuse bare AND',CONDITION:'AND only; reuse bare OR'} if split=='E' else {CONDITION:'AND only; reuse bare OR'},
            section_9_gap_corrected=True,held_out_from_compound_fitting=True,held_out_from_selection=True,
            final_evaluation_enabled=False,P_A_frozen=False,extraction_launched=False,non_E_membership_changed=False,
            statistical_note='Rows share entities, pairs and truth cells; these are data counts, not independent statistical sample sizes.',
            code_sha256={p:old.sha256((root/p).read_bytes()) for p in CODE},
            config_sha256=old.sha256((root/CONFIG).read_bytes()),input_sha256=config['input_sha256'],
            output_sha256={k:old.sha256(v) for k,v in files.items()})
        files['manifest.json']=old.encoded(summary)
        outputs.update({out/name:data for name,data in files.items()});summaries[split]=summary
    check(summaries['E']['pairs']==450 and summaries['E']['and_wording_rows']==3600,'unexpected E coverage')
    check(summaries['D']['pairs']==483 and summaries['D']['and_wording_rows']==3864,'unexpected D coverage')
    return outputs,summaries


def materialize(root,outputs,write=False):
    for path,data in outputs.items():
        target=Path(root)/path
        check(not target.exists() or target.read_bytes()==data,'differing immutable output: '+str(path))
        check(write or target.exists(),'missing output: '+str(path))
    if write:
        for path,data in outputs.items():
            target=Path(root)/path;target.parent.mkdir(parents=True,exist_ok=True)
            if not target.exists():target.write_bytes(data)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');args=p.parse_args()
    outputs,summaries=build();materialize(ROOT,outputs,args.write)
    print(json.dumps({s:{k:v for k,v in m.items() if k in ('pairs','and_wording_rows','extraction_bindings','unique_texts','deduplication_savings','final_evaluation_enabled')} for s,m in summaries.items()},indent=2))


if __name__=='__main__':main()
