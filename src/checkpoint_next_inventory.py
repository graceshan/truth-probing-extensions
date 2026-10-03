"""Exact frozen bindings; stdlib metadata only, no weights or research fitting."""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'config/checkpoint_next/behavior_prep_v1.json'
TOPICS = ('animal_class', 'cities', 'element_symb', 'inventors', 'sp_en_trans')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def config():
    c = json.loads(CONFIG.read_text())
    for p, h in c['source_sha256'].items():
        require(sha(ROOT / p) == h, 'frozen source changed: ' + p)
    require(c['E'] is False and c['layers'] == {'qwen': [17, 18], 'llama': [15, 12]}, 'scope/layers changed')
    return c


def read(name):
    return json.loads((ROOT / 'data/checkpoint_r2_v1' / name).read_text())


def inventory():
    c = config()
    members, facts, pairs, isolated = (read(p) for p in
        ('memberships.json', 'facts.json', 'behavior_pairs.json', 'behavior_isolated_bindings.json'))
    with (ROOT / 'data/checkpoint_r2_v1/compound_bindings.csv').open() as handle:
        compounds = list(csv.DictReader(handle))
    selected = [r for r in compounds if r['group'] == 'D_bare' and r['pair_id'] in pairs]
    tc = [r for r in compounds if r['group'] == 'TC_control_and_final_refit']
    require(len(pairs) == len(set(pairs)) == 50 and len(selected) == 800 and len(isolated) == 200, 'frozen behavior counts')
    require(len(tc) == 8000 and len(members['P15']) == 2778 and len(members['outer_train']) == 3040, 'fitting counts')
    by_pair = defaultdict(list)
    for r in selected:
        by_pair[r['pair_id']].append(r)
    leaves = {(op, a, b, order) for op in ('AND', 'OR') for a in ('0', '1')
              for b in ('0', '1') for order in ('AB', 'BA')}
    require(all(len(rs) == 16 and {(r['operator'], r['truth_a'], r['truth_b'], r['ordering']) for r in rs} == leaves
                for rs in by_pair.values()), 'incomplete compound cells/orders')
    require(Counter(rs[0]['topic'] for rs in by_pair.values()) == {t: 10 for t in TOPICS}, 'topic quotas')
    iso_pairs = defaultdict(set)
    for b in isolated:
        f = facts[b['fact_id']]
        require(b['pair_id'] in pairs and f['eligible'] and f['split'] == 'validation', 'isolated binding')
        iso_pairs[b['pair_id']].add(b['fact_id'])
    require(set(iso_pairs) == set(pairs) and all(len(v) == 4 for v in iso_pairs.values()), 'four facts per pair')
    bindings, jobs = [], {}

    def add(source, group, prompt, role, pair_id='', fact_id=''):
        logical = source.get('example_id', source.get('source_row_id', fact_id))
        text = c['prompts'][prompt - 1].format(statement=source['statement'])
        job_id = digest({'prompt': prompt, 'user_content': text})
        label = int(source.get('label', source.get('truth', False)))
        b = dict(binding_id=digest([role, group, prompt, logical, pair_id, fact_id]),
            job_id=job_id, role=role, group=group, prompt=prompt, logical_id=logical,
            pair_id=pair_id, fact_id=fact_id, label=label, topic=source['topic'],
            split=source['split'], operator=source.get('operator', ''), ordering=source.get('ordering', ''),
            truth_a=source.get('truth_a', ''), truth_b=source.get('truth_b', ''),
            person_a=source.get('person_a', ''), person_b=source.get('person_b', ''),
            source_metadata_sha256=digest(source))
        require(b['split'] in ('train', 'validation'), 'E forbidden')
        bindings.append(b)
        j = jobs.setdefault(job_id, dict(id=job_id, prompt=prompt, user_content=text,
            statement=source['statement'], statement_sha256=hashlib.sha256(source['statement'].encode()).hexdigest(),
            topic=source['topic'], split=source['split'], behavior=False, chat=False, bindings=[]))
        require((j['topic'], j['split'], j['user_content']) == (b['topic'], b['split'], text), 'cross-split text collision')
        j[role] = True
        j['bindings'].append(b['binding_id'])

    for prompt in (1, 2):
        for r in selected:
            add(r, 'behavior_compound', prompt, 'behavior', r['pair_id'])
        for b in isolated:
            f = dict(facts[b['fact_id']])
            add(f, 'behavior_isolated', prompt, 'behavior', b['pair_id'], b['fact_id'])
    for r in members['P15']:
        add(r, 'P15', 1, 'chat')
    for r in tc:
        add(r, 'TC_control_and_final_refit', 1, 'chat', r['pair_id'])
    for r in selected:
        add(r, 'behavior_D', 1, 'chat', r['pair_id'])
    require(len(set(b['binding_id'] for b in bindings)) == len(bindings), 'duplicate logical bindings')
    # Independently compare exact prompted text/label/pair/fact bindings to inherited frozen inventory.
    with (ROOT / 'data/checkpoint_r2_v1/extraction_inventory.csv').open() as handle:
        inherited = list(csv.DictReader(handle))
    expected = Counter((r['kind'], r['group'], r['logical_id'], r['pair_id'], int(r['label']), r['statement'])
                       for r in inherited if r['kind'] in ('behavior', 'chat'))
    actual = Counter((b['role'], 'prompt'+str(b['prompt']) if b['role']=='behavior' else b['group'],
                      b['logical_id'], b['pair_id'], b['label'], jobs[b['job_id']]['user_content']) for b in bindings)
    require(actual == expected, 'inherited exact behavior/chat inventory mismatch')
    selections = json.loads((ROOT / 'results/checkpoint_r2_fresh_constituents_20261003/locked-selection.json').read_text())
    require({m: s['AND']['selected_layer'] for m,s in selections['selections'].items()} == {'qwen':18,'llama':12}, 'raw source-selected layers')
    heads = [dict(id=f'{m}/{family}/L{layer}', model=m, family=family, layer=layer,
             preprocess_group='P15', preprocess_bindings_sha256=digest([b for b in bindings if b['group']=='P15']),
             fitting_group='P15' if family=='R0' else 'TC_control_and_final_refit',
             fitting_identities_sha256=digest([b['logical_id'] for b in bindings if b['group']==('P15' if family=='R0' else 'TC_control_and_final_refit')]),
             D_support_sha256=digest([r['example_id'] for r in selected]),
             decision_rule='zero_logit_ties_true', objective=c['objectives'][family])
             for m in ('qwen','llama') for layer in c['layers'][m] for family in ('R0','C_clean')]
    require(len(heads)==8, 'eight heads required')
    preflight = []
    for group, sources in [('P15', members['P15']), ('TC_control_and_final_refit', tc)]:
        for t in TOPICS:
            rows = sorted((r for r in sources if r['topic']==t), key=lambda r:(len(r['statement'].encode()),r.get('example_id',r.get('source_row_id'))))
            for r in (rows[0], rows[-1]):
                for prompt in (1,2):
                    preflight.append(dict(id=digest([group,t,r['statement'],prompt]),group=group,topic=t,split='train',prompt=prompt,
                        statement=r['statement'], user_content=c['prompts'][prompt-1].format(statement=r['statement']),
                        behavior=True,chat=prompt==1))
    behaviors = {j['id'] for j in jobs.values() if j['behavior']}
    chats = {j['id'] for j in jobs.values() if j['chat']}
    accounting = dict(behavior_pairs=50,compounds=800,isolated_bindings=200,
        isolated_unique_facts=len({b['fact_id'] for b in isolated}),
        logical_behavior_per_model=2000, logical_behavior_all_models=4000,
        logical_chat_per_model=11578, logical_chat_all_models=23156,
        behavior_unique_inputs_per_model=len(behaviors),chat_unique_inputs_per_model=len(chats),
        exact_prompted_text_overlap_per_model=len(behaviors & chats),
        union_unique_inputs_per_model=len(jobs), verified_token_hidden_overlap='pending real training preflight; no numerical reuse presumed',
        group_counts=dict(Counter(b['group'] for b in bindings)), fp16_chat_payload_bytes={m:len(chats)*2*s['width']*2 for m,s in c['models'].items()},
        preflight_inputs_per_model=len(preflight), runtime_storage_quota_forecast='unmeasured',
        deduplication='exact prompt index/user UTF-8 content; preserve every logical binding; token/representation equality separately required')
    return dict(schema='checkpoint-next-inventory-v1', config_sha256=sha(CONFIG), pairs=pairs,
        bindings=bindings, jobs=list(jobs.values()), heads=heads, preflight=preflight, accounting=accounting)


def missing_raw_interface(required, available, representation_sha256):
    """mac 2 submits exact admitted bindings; available rows must be loader-verified fresh rows.

    No historical tensors; output groups by unique text, never assumes row-count=text-count.
    """
    outer = {r['source_row_id']: r for r in read('memberships.json')['outer_train']}
    require(len(required)==len({r['source_row_id'] for r in required}), 'duplicate required binding')
    verified = {}
    for r in available:
        require(r['representation_sha256']==representation_sha256 and r['loader_verified'] is True, 'unverified representation')
        require(r['statement_sha256']==hashlib.sha256(r['statement'].encode()).hexdigest() and r['token_ids'], 'exact raw binding missing')
        key=r['statement']
        if key in verified:
            require(verified[key]['token_ids']==r['token_ids'], 'ambiguous raw token binding')
        verified[key]=r
    reused, missing = [], {}
    for r in required:
        require(r['source_row_id'] in outer and r == outer[r['source_row_id']], 'not exact admitted outer train')
        match=verified.get(r['statement'])
        if match:
            reused.append(dict(binding=r, fresh_match=match))
        else:
            k=r['statement_sha256'];j=missing.setdefault(k,dict(statement=r['statement'],statement_sha256=k,bindings=[]))
            require(j['statement']==r['statement'], 'text hash collision')
            j['bindings'].append(r)
    return dict(schema='checkpoint-next-missing-raw-v1',representation_sha256=representation_sha256,
        required_bindings=len(required),reused_bindings=len(reused),missing_bindings=sum(len(r['bindings']) for r in missing.values()),
        missing_unique_texts=len(missing), reused=reused, rows=list(missing.values()),
        tokens_for_missing='must bind under accepted offline tokenizer before extraction',
        status='inventory only; no extraction authorized; shared 7200-second/model budget')
