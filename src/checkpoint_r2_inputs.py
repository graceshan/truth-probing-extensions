"""Revision 2 deterministic data preparation. No model, activation or score IO."""
import argparse
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
import numpy as np
from src.clean_compounds import render_binary
from src.heldout_and_wording import render as render_and
from src.final_e_preparation import render_wording

ROOT = Path(__file__).resolve().parents[1]
BASE = 'data/clean_protocol/selection_repair_v1/'
PROJ = BASE + 'final_partition_projection_v1/'
D = 'data/clean_protocol/compounds/entity_disjoint/development_heldout_and_v1/'
CONFIG = 'config/checkpoint_r2/protocol_v1.json'
OUTPUT = Path('data/checkpoint_r2_v1')
TOPICS = ('animal_class', 'cities', 'element_symb', 'inventors', 'sp_en_trans')
SEMANTIC = ('source_row_id','paired_source_row_id','topic','entity','entity_id','person_key','split','form','label','statement','statement_sha256','source_sha256','source_row_sha256')


def check(ok, message):
    if not ok:
        raise ValueError(message)


def encoded(x):
    return (json.dumps(x, sort_keys=True, ensure_ascii=False, indent=2)+'\n').encode()


def sha(x):
    return hashlib.sha256(x).hexdigest()


def uid(kind, *parts):
    return kind+'_'+sha(json.dumps(parts,ensure_ascii=False,separators=(',',':')).encode())


def csv_bytes(rows):
    f=io.StringIO(newline=''); w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    return f.getvalue().encode()


def rng(seed):
    return np.random.Generator(np.random.PCG64(seed))


def ordered_semantics(rows,status):
    return [dict({k:r[k] for k in SEMANTIC}, current_status=r[status]) for r in rows]


def validate_fact(f, entity, source, ledger, evidence):
    check(f['label'] in (0,1) and f['judgment']==('supported_true' if f['label'] else 'supported_false'), 'unresolved or conflicting fact')
    check(sha(f['statement'].encode())==f['statement_sha256'], 'statement hash')
    exact=[r for r in source if r['topic']==entity['topic'] and r['person_key']==entity['person_key'] and r['statement']==f['statement']]
    check(all(r['split']=='train' and r['e_successor_status']=='admitted' and int(r['label'])==f['label'] for r in exact), 'restricted exact source duplicate or wrong split')
    registry=[r for r in ledger if r['topic']==entity['topic'] and r['person_key']==entity['person_key'] and r['statement']==f['statement']]
    check(all(r['e_current_eligible']=='True' and r['split']=='train' for r in registry), 'restricted registry duplicate')
    if f['origin']=='source':
        check(any(r['source_row_id']==f['source_row_id'] for r in exact), 'source identity binding')
    check(bool(f['evidence_ids']), 'missing evidence')
    for eid in f['evidence_ids']:
        check(eid in evidence,'missing evidence record')
        check(any(b['person_key']==entity['person_key'] and b['topic']==entity['topic'] and b['statement_sha256']==f['statement_sha256'] and b['judgment']==f['judgment'] for b in evidence[eid]['fact_bindings']), 'wrong evidence/person binding')


def choose_pairs(keys, seed, n=100):
    possible=list(combinations(sorted(keys),2))
    return [possible[int(i)] for i in sorted(rng(seed).choice(len(possible),size=n,replace=False))]


def validate_row(row, facts):
    a,b=facts[row['fact_a_id']],facts[row['fact_b_id']]
    check(a['eligible'] and b['eligible'],'ineligible constituent')
    check(a['split']==b['split']==row['split'] and a['topic']==b['topic']==row['topic'],'wrong split/topic')
    check(a['person_key']==row['person_a'] and b['person_key']==row['person_b'] and row['person_a']!=row['person_b'],'wrong person binding')
    ta,tb=int(a['truth']),int(b['truth'])
    check((ta,tb)==(row['truth_a'],row['truth_b']),'substituted truth')
    check(row['operator'] in ('AND','OR'),'wrong operator')
    check(row['label']==int((ta and tb) if row['operator']=='AND' else (ta or tb)),'incorrect compound label')
    check(row['ordering'] in ('AB','BA'),'wrong surface order')
    first,second=(a,b) if row['ordering']=='AB' else (b,a)
    check((row['surface_first_truth'],row['surface_second_truth'])==(int(first['truth']),int(second['truth'])),'surface label binding')
    check(row['statement']==render_binary(first['statement'],second['statement'],row['operator']),'substituted constituent text')


def build(root=ROOT):
    inputs={}
    def read(path,kind='json'):
        data=(root/path).read_bytes();inputs[path]=sha(data)
        return json.loads(data) if kind=='json' else list(csv.DictReader(io.StringIO(data.decode())))
    cfg=read(CONFIG)
    for spec in cfg['sources'].values():
        check(sha((root/spec['text_path']).read_bytes())==spec['text_sha256'],'plan text hash')
        inputs[spec['text_path']]=spec['text_sha256']
    check(not cfg['production_execution_enabled'] and not cfg['final_evaluation_enabled'],'execution gate')
    v4=read(BASE+'candidate_overlay_v4/row_manifest.csv','csv');v5=read(BASE+'candidate_overlay_v5/row_manifest.csv','csv')
    ledger=read(BASE+'candidate_overlay_v5/fact_inventory.csv','csv')
    oldledger=read(BASE+'candidate_overlay_v4/fact_inventory.csv','csv')
    check(len(v4)==len(v5),'overlay row count')
    check([{k:r[k] for k in SEMANTIC} for r in v4]==[{k:r[k] for k in SEMANTIC} for r in v5],'overlay ordered identity mutation')
    oldfacts=[r for r in oldledger if r['split'] in ('train','validation')]
    newfacts=[r for r in ledger if r['split'] in ('train','validation')]
    check(oldfacts==[{k:r[k] for k in oldfacts[0]} for r in newfacts],'non-E registry mutation')
    memberships={};equivalence={}
    for split,name in [('train','outer_train'),('validation','atomic_D')]:
        old=[r for r in v4 if r['split']==split];new=[r for r in v5 if r['split']==split]
        check(ordered_semantics(old,'tc_successor_status')==ordered_semantics(new,'e_successor_status'),'non-E admission mutation')
        membership=read(PROJ+split+'_admitted_membership.csv','csv')
        check(membership==[r for r in new if r['e_successor_status']=='admitted'],'projection mismatch')
        memberships[name]=membership;equivalence[name]=sha(encoded(ordered_semantics(membership,'e_successor_status')))
    for name,file in [('P15','P15_admitted_membership.csv'),('balanced','P15_balanced_exposure.csv')]:
        new=read(PROJ+file,'csv');old=read(BASE+'capacity_pilot_projection_v3/'+file,'csv')
        check(ordered_semantics(old,'tc_successor_status')==ordered_semantics(new,'e_successor_status'),'P/sampler ordered mismatch')
        memberships[name]=new;equivalence[name]=sha(encoded(ordered_semantics(new,'e_successor_status')))
    a=read(PROJ+'A15_proposal.json')['completed_pairs']; tc=read(PROJ+'TC_completed_pairs.json')
    check(a==read(BASE+'capacity_pilot_projection_v3/A15_proposal.json')['completed_pairs'],'A15 mutation')
    check(tc==read(BASE+'tc_topup_audit_v1/combined_TC_completed_pairs.json'),'TC mutation')
    evidence=read(BASE+'tc_topup_audit_v1/combined_evidence.json')
    for name,records in [('A15',a),('TC',tc)]:
        check(len({r['person_key'] for r in records})==len(records),'duplicate person')
        check(Counter(r['topic'] for r in records)==Counter({t:15 if name=='A15' else 20 for t in TOPICS}),'topic allocation')
        equivalence[name]=sha(encoded(records))
    tc_by_person={r['person_key']:r for r in tc}
    check(all(r==tc_by_person.get(r['person_key']) for r in a),'A15 exact facts differ from TC')
    ap={r['person_key'] for r in a};tp={r['person_key'] for r in tc}; pp={r['person_key'] for r in memberships['P15']}
    check(ap<=tp and not ap&pp,'P/A leakage')
    check(memberships['P15']==[r for r in memberships['outer_train'] if r['person_key'] not in ap],'all-alias P exclusion')
    check({r['source_row_id'] for r in memberships['balanced']}<={r['source_row_id'] for r in memberships['P15']},'sampler leakage')
    forbidden={r['person_key'] for r in v5 if r['split'] in ('validation','test')}
    check(not (tp|ap|pp)&forbidden,'D/E fitting leakage')
    bysource={r['source_row_id']:r for r in v5}
    for r in v5:
        check((r['e_successor_status']=='admitted')==(bysource[r['paired_source_row_id']]['e_successor_status']=='admitted'),'paired quarantine')
        check(sha(r['statement'].encode())==r['statement_sha256'],'source text hash')
    facts={}; byperson={}; entity_records=[]
    for e in tc:
        check(e['status']=='usable' and {f['label'] for f in e['facts']}=={0,1},'incomplete TC pair')
        check({r['person_key'] for r in v5 if r['entity_id'] in e['entity_ids']}=={e['person_key']},'alias/person map')
        byperson[e['person_key']]={}
        for f in e['facts']:
            validate_fact(f,e,v5,ledger,evidence)
            fid=uid('r2fact',e['topic'],e['person_key'],f['statement_sha256'],f['label'])
            facts[fid]=dict(f, fact_id=fid,topic=e['topic'],person_key=e['person_key'],entity_ids=e['entity_ids'],split='train',truth=bool(f['label']),eligible=True)
            byperson[e['person_key']][f['label']]=fid
        entity_records.append(dict(topic=e['topic'],person_key=e['person_key'],entity_ids=e['entity_ids'],aliases=e['entities'],A15=e['person_key'] in ap,fact_ids=byperson[e['person_key']]))
    d=read(D+'bare_compounds.csv','csv'); df=read(D+'constituent_facts.json')
    d_elig=read(D+'development_fact_eligibility.csv','csv'); dp=read(D+'development_pair_eligibility.csv','csv')
    check(len({r['pair_id'] for r in d})==483 and len(d)==7728,'D count')
    check({r['pair_id'] for r in d}=={r['pair_id'] for r in dp if r['tc_current_eligible']=='True'},'D pair admission')
    for f in df:
        check(f['eligible'] and f['split']=='validation' and f['person_key'] not in (tp|pp),'D identity eligibility')
        check(any(r['fact_id']==f['fact_id'] and r['tc_current_eligible']=='True' and r['person_key']==f['person_key'] and r['statement_sha256']==sha(f['statement'].encode()) for r in d_elig),'D eligibility binding')
        exact=[r for r in ledger if r['topic']==f['topic'] and r['person_key']==f['person_key'] and r['statement']==f['statement']]
        check(exact and all(r['e_current_eligible']=='True' and r['split']=='validation' for r in exact),'v5 D duplicate restriction')
        facts[f['fact_id']]=dict(f,statement_sha256=sha(f['statement'].encode()),evidence_path=D+'development_fact_eligibility.csv')
    pairs={};groups=defaultdict(list);rows=[]
    def add_pair(group,topic,a,b,seed,fold=None):
        a,b=sorted((a,b));pid=uid('r2pair',topic,a,b)
        pairs[pid]=dict(pair_id=pid,topic=topic,person_a=a,person_b=b,facts_a=byperson[a],facts_b=byperson[b],split='train')
        groups[group].append(dict(pair_id=pid,fold=fold))
        for ta,tb,op,order in product((0,1),(0,1),('AND','OR'),('AB','BA')):
            fa,fb=byperson[a][ta],byperson[b][tb];first,second=(fa,fb) if order=='AB' else (fb,fa)
            row=dict(example_id=uid('r2example',pid,fa,fb,op,order),pair_id=pid,topic=topic,split='train',person_a=a,person_b=b,fact_a_id=fa,fact_b_id=fb,truth_a=ta,truth_b=tb,operator=op,ordering=order,surface_first_truth=int(facts[first]['truth']),surface_second_truth=int(facts[second]['truth']),label=int((ta and tb) if op=='AND' else (ta or tb)),statement=render_binary(facts[first]['statement'],facts[second]['statement'],op))
            validate_row(row,facts);rows.append(dict(row,group=group))
    splits=[];unused={}
    for topic in TOPICS:
        aks=sorted(e['person_key'] for e in a if e['topic']==topic);tks=sorted(e['person_key'] for e in tc if e['topic']==topic)
        for seed in cfg['sampling']['B25_seeds']:
            order=list(rng(seed).permutation(aks));folds=list(map(int,rng(seed).permutation(5)));unused[f'{seed}/{topic}']=order[10:]
            for i in range(5):add_pair(f'B25_{seed}',topic,*order[2*i:2*i+2],seed,folds[i])
        validation=set(rng(cfg['sampling']['constituent_split_seed']).permutation(aks)[:4]);fitting=sorted(set(tks)-validation)
        check(not validation&pp and len(fitting)==16,'source validation leakage')
        splits.extend(dict(topic=topic,person_key=k,role='source_validation' if k in validation else 'source_fit') for k in tks)
        for x,y in choose_pairs(tks,cfg['sampling']['TC_pair_seed']):add_pair('TC_control_and_final_refit',topic,x,y,cfg['sampling']['TC_pair_seed'])
        for x,y in choose_pairs(fitting,cfg['sampling']['source_fit_pair_seed']):add_pair('constituent_source_fit',topic,x,y,cfg['sampling']['source_fit_pair_seed'])
        for x,y in combinations(sorted(validation),2):add_pair('constituent_source_validation',topic,x,y,cfg['sampling']['constituent_split_seed'])
    # Keep authoritative D example IDs, pair IDs and exact facts, never regenerate pairs.
    drows=[]
    for r in d:
        row=dict(example_id=r['example_id'],pair_id=r['pair_id'],topic=r['topic'],split=r['split'],person_a=facts[r['fact_a_id']]['person_key'],person_b=facts[r['fact_b_id']]['person_key'],fact_a_id=r['fact_a_id'],fact_b_id=r['fact_b_id'],truth_a=int(r['canonical_truth_a']=='True'),truth_b=int(r['canonical_truth_b']=='True'),operator=r['operator'],ordering=r['ordering'],surface_first_truth=int(r['surface_first_truth']=='True'),surface_second_truth=int(r['surface_second_truth']=='True'),label=int(r['compound_label']=='True'),statement=r['statement'])
        check(facts[row['fact_a_id']]['entity_id']==r['entity_a_id'] and facts[row['fact_b_id']]['entity_id']==r['entity_b_id'],'D entity binding')
        validate_row(row,facts);drows.append(row)
    behavior=[]
    for t in TOPICS:
        available=sorted({r['pair_id'] for r in drows if r['topic']==t})
        selected=sorted(rng(cfg['sampling']['behavior_seed']).choice(available,10,replace=False))
        behavior.extend(selected)
    bset=set(behavior);brows=[r for r in drows if r['pair_id'] in bset]
    check(len(brows)==800,'behavior count')
    bbindings=[dict(pair_id=p,fact_id=f) for p in behavior for f in sorted({r[k] for r in brows if r['pair_id']==p for k in ('fact_a_id','fact_b_id')})]
    check(len(bbindings)==200,'isolated behavior coverage')
    # Reversible inventories; raw content identity deliberately separate from row/fact identity.
    inventory=[]
    def add(kind,logical,text,group='',pair='',fa='',fb='',label=''):
        inventory.append(dict(kind=kind,group=group,logical_id=logical,pair_id=pair,fact_a_id=fa,fact_b_id=fb,label=label,text_id='text_'+sha(text.encode()),statement=text))
    for name in ('P15','atomic_D'):
        for r in memberships[name]:add('raw',r['source_row_id'],r['statement'],name,label=r['label'])
    for r in rows+drows:
        add('raw',r['example_id'],r['statement'],r.get('group','D_bare'),r['pair_id'],r['fact_a_id'],r['fact_b_id'],r['label'])
    for fid,f in facts.items():add('raw',fid,f['statement'],'TC_isolated' if f['split']=='train' else 'D_isolated',fa=fid,label=int(f['truth']))
    for r in drows:
        first,second=(r['fact_a_id'],r['fact_b_id']) if r['ordering']=='AB' else (r['fact_b_id'],r['fact_a_id'])
        for cond in (('and_both_following_v1',) if r['operator']=='AND' else ('or_explicit_or_both_v1','or_at_least_one_v1')):
            text=render_and(facts[first]['statement'],facts[second]['statement'],'AND') if r['operator']=='AND' else render_wording(cond,facts[first]['statement'],facts[second]['statement'])
            add('raw',uid('r2wording',r['example_id'],cond),text,cond,r['pair_id'],r['fact_a_id'],r['fact_b_id'],r['label'])
    # Model-specific chat tokenization is deferred; these are exact user-message inputs.
    for prompt_idx,prompt in enumerate(cfg['behavior']['prompts'],1):
        for r in brows:add('behavior',r['example_id'],prompt.format(statement=r['statement']),f'prompt{prompt_idx}',r['pair_id'],r['fact_a_id'],r['fact_b_id'],r['label'])
        for b in bbindings:
            f=facts[b['fact_id']];add('behavior',b['fact_id'],prompt.format(statement=f['statement']),f'prompt{prompt_idx}',b['pair_id'],b['fact_id'],label=int(f['truth']))
    for r in memberships['P15']:add('chat',r['source_row_id'],cfg['behavior']['prompts'][0].format(statement=r['statement']),'P15',label=r['label'])
    for r in [r for r in rows if r['group']=='TC_control_and_final_refit']+brows:
        add('chat',r['example_id'],cfg['behavior']['prompts'][0].format(statement=r['statement']),r.get('group','behavior_D'),r['pair_id'],r['fact_a_id'],r['fact_b_id'],r['label'])
    counts={k:len(v) for k,v in memberships.items()};counts.update(A15=len(a),TC=len(tc),compound_D_pairs=len({r['pair_id'] for r in drows}))
    check(counts==cfg['expected'],'expected membership counts')
    for seed in cfg['sampling']['B25_seeds']:
        selected=[pairs[p['pair_id']] for p in groups[f'B25_{seed}']]
        check(len(selected)==25 and len({p[s] for p in selected for s in ('person_a','person_b')})==50,'B25 disjointness')
        check(Counter(p['fold'] for p in groups[f'B25_{seed}'])==Counter({i:5 for i in range(5)}),'fold balance')
    # Freeze a small short/long stratified bridge, train-only calibration disjoint by source ID.
    bridge=[];cal=[]
    for topic in TOPICS:
        for name in ('P15','atomic_D'):
            available=sorted([r for r in memberships[name] if r['topic']==topic],key=lambda r:(len(r['statement']),r['source_row_id']))
            for r in (available[0],available[-1]):bridge.append(dict(kind='atomic',id=r['source_row_id'],statement=r['statement'],split=r['split'],topic=topic,person_key=r['person_key']))
            if name=='P15':
                for r in (available[1],available[-2]):cal.append(dict(kind='atomic',id=r['source_row_id'],statement=r['statement'],split='train',topic=topic,person_key=r['person_key']))
        candidates=sorted({r['pair_id'] for r in drows if r['topic']==topic},key=lambda p:(sum(len(r['statement']) for r in drows if r['pair_id']==p),p))
        for p in (candidates[0],candidates[-1]):
            bridge.extend(dict(kind='compound',id=r['example_id'],statement=r['statement'],split='validation',topic=topic,pair_id=p) for r in drows if r['pair_id']==p)
    check(len(bridge)==180 and len(cal)==10 and not {r['id'] for r in cal}&{r['id'] for r in bridge},'bridge overlap')
    estimates={}
    for kind in ('raw','behavior','chat'):
        selected=[r for r in inventory if r['kind']==kind];texts={r['text_id']:r['statement'] for r in selected}
        utf8=sum(len(t.encode()) for t in texts.values())
        estimates[kind]=dict(logical_bindings_per_model=len(selected),unique_inputs_per_model=len(texts),utf8_bytes_unique=utf8,
            exact_tokens=None,token_status='pending pinned tokenizer; no characters-as-measured-tokens claim',
            planning_content_token_range=[(utf8+5)//6,utf8],token_range_assumption='heuristic UTF-8 bytes/6 to bytes; not guaranteed; excludes native chat framing and answer/generation tokens',
            seconds_at_5_20_50_inputs_per_second={str(rate):len(texts)/rate for rate in (5,20,50)},runtime_status='planning sensitivity only, no measured GPU throughput')
        if kind=='raw':estimates[kind]['fp16_all_layer_bytes_per_model']={m:len(texts)*v['layers']*v['width']*2 for m,v in cfg['bank']['models'].items()}
        if kind=='chat':estimates[kind]['fp16_one_to_two_layer_bytes_per_model']={m:[len(texts)*v['width']*2,len(texts)*v['width']*4] for m,v in cfg['bank']['models'].items()}
    estimates['behavior']['two_models_two_prompts_logical_combinations']=2*estimates['behavior']['logical_bindings_per_model']
    estimates['behavior']['additional_work']='per input two candidate continuation scores plus greedy <=8 new tokens; native EOS, tokenization and actual prefix sharing pending; no behavioral outputs produced'
    estimates['behavior']['max_new_tokens_across_two_models']=2*estimates['behavior']['unique_inputs_per_model']*8
    rawtexts={r['text_id'] for r in inventory if r['kind']=='raw'}
    estimates['overlap_behavior_chat_unique_user_messages']=len({r['text_id'] for r in inventory if r['kind']=='behavior'}&{r['text_id'] for r in inventory if r['kind']=='chat'})
    for path in ('src/checkpoint_r2_inputs.py','src/checkpoint_r2_bridge.py','src/clean_compounds.py','src/final_e_preparation.py','src/heldout_and_wording.py','src/selection_repair_objectives.py','docs/selection_repair_objectives.md'):
        inputs[path]=sha((root/path).read_bytes())
    outputs={'memberships.json':encoded({k:ordered_semantics(v,'e_successor_status') for k,v in memberships.items()}),'entities.json':encoded(entity_records),'facts.json':encoded(facts),'pairs.json':encoded(pairs),'pair_groups.json':encoded(dict(groups)),'constituent_split.csv':csv_bytes(splits),'B25_unused_entities.json':encoded(unused),'behavior_pairs.json':encoded(behavior),'behavior_isolated_bindings.json':encoded(bbindings),'compound_bindings.csv':csv_bytes(rows+[dict(r,group='D_bare') for r in drows]),'extraction_inventory.csv':csv_bytes(inventory),'bridge_identities.json':encoded(dict(independent=bridge,calibration=cal)),'resource_estimates.json':encoded(estimates)}
    receipt=dict(status='passed_local_data_validation',counts=counts,ordered_v4_v5_equivalence_hashes=equivalence,input_code_config_sha256=inputs,output_sha256={k:sha(v) for k,v in outputs.items()},numpy_version=np.__version__,groups={k:len(v) for k,v in groups.items()},behavior_unique_people=len({r[s] for r in brows for s in ('person_a','person_b')}),behavior_unique_facts=len({r['fact_id'] for r in bbindings}),final_evaluation_enabled=False,production_execution_enabled=False,representation_reuse_verified=False)
    outputs['manifest.json']=encoded(receipt)
    return outputs,receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--write',action='store_true');args=p.parse_args()
    outputs,receipt=build()
    for name,data in outputs.items():
        path=ROOT/OUTPUT/name
        check(not path.exists() or path.read_bytes()==data,'immutable output differs: '+name)
        check(args.write or path.exists(),'missing output: '+name)
    if args.write:
        (ROOT/OUTPUT).mkdir(parents=True,exist_ok=True)
        for name,data in outputs.items():
            path=ROOT/OUTPUT/name
            if not path.exists():path.write_bytes(data)
    print(json.dumps({k:receipt[k] for k in ('status','counts','groups','behavior_unique_people','behavior_unique_facts')},indent=2))

if __name__=='__main__':main()
