"""Score-blind E factual review and capacity projection. Never generates examples.

A separate fact-only path leaves all recovered development/final-scoring guards
intact. Original partitions, source text, labels and historical packages persist.
"""
import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.clean_compounds import ATOMIC_TEMPLATES, stable_id
from src.entity_partitions import TRUE_OBJECT_PATTERNS, canonical_json
from src.inventor_country_semantics import true_country_components
from src.validated_negatives import object_rank
from src import capacity_pilot_projection as sampler

ROOT = Path(__file__).resolve().parents[1]
BASE = Path('data/clean_protocol/selection_repair_v1')
PACKAGE = BASE / 'final_partition_fact_audit_v1'
PREVIOUS = BASE / 'candidate_overlay_v4'
OVERLAY = BASE / 'candidate_overlay_v5'
PROJECTION = BASE / 'final_partition_projection_v1'
PILOT = BASE / 'capacity_pilot_projection_v3'
STATUS, ELIGIBLE = 'e_successor_status', 'e_current_eligible'
TOPICS = tuple(sorted(ATOMIC_TEMPLATES))
INPUT_NAMES = ('pre_judgment_lock.json', 'original_E_inventory.json', 'entity_audit_order.json',
               'negative_candidate_queue.json', 'reviews.json', 'evidence.json', 'retrieval_log.json')
SEMANTIC = ('source_row_id', 'paired_source_row_id', 'topic', 'entity', 'entity_id',
            'person_key', 'split', 'form', 'label', 'statement', 'statement_sha256')


def check(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()


def load(root, path):
    return json.loads((Path(root) / path).read_text())


def csv_read(root, path):
    with (Path(root) / path).open(newline='') as f:
        return list(csv.DictReader(f))


def csv_bytes(rows, fields):
    stream = io.StringIO(newline='')
    w = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    w.writeheader()
    w.writerows(rows)
    return stream.getvalue().encode()


def signature(row):
    return row['topic'], row['person_key'], row['statement_sha256']


def reconstruct_queue(rows, manifest):
    """Recovered object/rank semantics, v4 admission, no model information or RNG."""
    parts, by_entity = defaultdict(set), defaultdict(list)
    for r in rows:
        parts[r['person_key']].add(r['split'])
        by_entity[r['entity_id']].append(r)
    inventory, entities = [], []
    for e in sorted((e for e in manifest if e['split'] == 'test'),
                    key=lambda e: (e['topic'], e['entity_id'])):
        source = by_entity[e['entity_id']]
        keys = {r['person_key'] for r in source}
        check(len(keys) == 1, 'ambiguous reviewed person binding')
        key = next(iter(keys))
        aff = [r for r in source if r['form'] == 'affirmative']
        pos = [r for r in aff if r['label'] == '1' and r['tc_successor_status'] == 'admitted']
        eligible = e['compound_usable'] == 'True' and bool(pos) and parts[key] == {'test'}
        inventory.append(dict(topic=e['topic'], entity=e['entity'], entity_id=e['entity_id'],
            person_key=key, split='test', original_compound_usable=e['compound_usable'],
            eligible_for_audit=eligible, source_rows=[{k: r[k] for k in
                ('source_row_id', 'statement', 'statement_sha256', 'label', 'form',
                 'tc_successor_status', 'paired_source_row_id')} for r in source],
            exclusion_reason='' if eligible else 'original compound unusable, no admitted positive, or cross-partition person'))
        if eligible:
            true = {re.fullmatch(TRUE_OBJECT_PATTERNS[e['topic']], r['statement'])[2]
                    for r in aff if r['label'] == '1'}
            chosen = min(pos, key=lambda r: int(r['row_index']))
            entities.append(dict(topic=e['topic'], entity=e['entity'], entity_id=e['entity_id'],
                person_key=key, split='test', positive_source_id=chosen['source_row_id'],
                positive_statement=chosen['statement'], positive_statement_sha256=chosen['statement_sha256'],
                known_true_objects=sorted(true)))
    check(len({e['person_key'] for e in entities}) == len(entities), 'duplicate E person')
    queue = []
    for e in entities:
        topic = e['topic']
        pool = set().union(*(set(x['known_true_objects']) for x in entities if x['topic'] == topic))
        known = set(e['known_true_objects'])
        if topic == 'inventors':
            pool, known = true_country_components(pool), true_country_components(known)
        ranked = sorted(pool - known, key=lambda v: (object_rank(0, topic, e['entity_id'], v), v))
        for rank, value in enumerate(ranked, 1):
            statement = ATOMIC_TEMPLATES[topic].format(entity=e['entity'], object=value)
            matches = [r for r in rows if r['form'] == 'affirmative' and
                       r['person_key'] == e['person_key'] and r['statement'] == statement]
            queue.append(dict(topic=topic, entity_id=e['entity_id'], person_key=e['person_key'],
                entity=e['entity'], candidate_rank=rank, candidate_object=value,
                ranking_hash=object_rank(0, topic, e['entity_id'], value), statement=statement,
                statement_sha256=digest(statement.encode()),
                fact_id=stable_id('fact', [topic, e['entity_id'], statement, False]),
                source_matches=[r['source_row_id'] for r in matches],
                base_eligible=all(r['tc_successor_status'] == 'admitted' for r in matches)))
    return inventory, entities, queue


def validate_reviews(rows, entities, queue, reviews, evidence):
    check([(r['topic'], r['entity_id']) for r in reviews] ==
          [(e['topic'], e['entity_id']) for e in entities], 'entity traversal gap, redraw or extra review')
    source = {r['source_row_id']: r for r in rows}
    for e, r in zip(entities, reviews):
        check(all(r[k] == e[k] for k in ('topic', 'entity', 'entity_id', 'person_key', 'split')),
              'review entity/person/partition mismatch')
        fs = r['facts']
        check(len(fs) >= 2, 'independent affirmative pair required')
        a = fs[0]
        check(a['origin'] == 'source' and a['label'] == 1 and
              a['source_row_id'] == e['positive_source_id'] and
              a['statement'] == e['positive_statement'], 'positive source binding mismatch')
        eligible = [q for q in queue if q['entity_id'] == e['entity_id'] and q['base_eligible']]
        check(len(fs) - 1 <= len(eligible), 'queue overrun')
        for n, (f, q) in enumerate(zip(fs[1:], eligible)):
            check(f['origin'] == 'recovered_negative_pipeline' and f['label'] == 0 and
                  all(f[k] == q[k] for k in ('candidate_rank', 'ranking_hash', 'candidate_object',
                      'statement', 'statement_sha256', 'fact_id')), 'negative ordering/fact binding mismatch')
            if n < len(fs) - 2:
                check(a['judgment'] == 'supported_true' and f['judgment'] == 'supported_true',
                      'reopened unresolved candidate or review beyond stopping rule')
        check(fs[-1]['judgment'] != 'supported_true' or len(fs)-1 == len(eligible),
              'stopped before replacement queue exhausted')
        expected = 'usable' if a['judgment'] == 'supported_true' and fs[-1]['judgment'] == 'supported_false' else 'unresolved'
        check(r['status'] == expected, 'pair status inconsistent with judgments')
        for f in fs:
            check(f['judgment'] in ('supported_true', 'supported_false', 'unresolved'), 'unknown judgment')
            check(digest(f['statement'].encode()) == f['statement_sha256'], 'statement hash mismatch')
            parsed = re.fullmatch(TRUE_OBJECT_PATTERNS[r['topic']], f['statement'])
            check(parsed and parsed[1] == r['entity'], 'fact is not exact same-entity affirmative')
            check(f['reasoning'].strip() and f['evidence_ids'], 'missing reasoning/evidence')
            if f['origin'] == 'source':
                s = source[f['source_row_id']]
                check(s['statement'] == f['statement'] and s['person_key'] == r['person_key'] and
                      s['entity_id'] == r['entity_id'] and s['split'] == 'test' and
                      int(s['label']) == f['label'] and s['form'] == 'affirmative', 'source identity mismatch')
            bind = dict(topic=r['topic'], entity_id=r['entity_id'], person_key=r['person_key'],
                        statement_sha256=f['statement_sha256'], judgment=f['judgment'])
            for key in f['evidence_ids']:
                ev = evidence[key]
                check(bind in ev['fact_bindings'], 'evidence binding mismatch')
                check(ev['http_status'] == 200 and ev['response_sha256'] and ev['url'] and
                      ev['observations'] and all(o['locator'] and o['excerpt'] for o in ev['observations']),
                      'missing locatable substantive evidence')
        if r['topic'] == 'sp_en_trans':
            s = r['spanish_checks']
            check(s['headword'] == r['entity'] and s['direction'] == 'Spanish→English' and
                  all(s[k] for k in ('spelling_diacritics', 'polysemy_and_dialect', 'scope')),
                  'Spanish sense/dialect/spelling/direction check absent')


def propagate(rows, reviews):
    actions, banned = {}, set()
    for review in reviews:
        for fact in review['facts']:
            sig = review['topic'], review['person_key'], fact['statement_sha256']
            matches = [r for r in rows if signature(r) == sig]
            check(all(r['statement'] == fact['statement'] and r['form'] == 'affirmative' and
                      r['entity_id'] == review['entity_id'] for r in matches), 'exact duplicate identity mismatch')
            unresolved = fact['judgment'] == 'unresolved'
            conflicts = [r for r in matches if fact['judgment'] != 'unresolved' and
                         int(r['label']) != int(fact['judgment'] == 'supported_true')]
            if unresolved:
                banned.add(sig)
            for r in matches if unresolved else conflicts:
                if r['tc_successor_status'] != 'admitted':
                    continue
                for sid in (r['source_row_id'], r['paired_source_row_id']):
                    action = dict(status='quarantined' if unresolved else 'excluded',
                        affirmative_source_id=r['source_row_id'], judgment=fact['judgment'],
                        statement_sha256=fact['statement_sha256'], person_key=review['person_key'],
                        reason='unresolved_exact_source_pair' if unresolved else 'confirmed_source_label_conflict_pair')
                    check(sid not in actions or actions[sid] == action, 'inconsistent paired action')
                    actions[sid] = action
    successor = [dict(r, **{STATUS: actions[r['source_row_id']]['status'] if r['source_row_id'] in actions
                           else r['tc_successor_status']},
                      e_correction_reason=actions.get(r['source_row_id'], {}).get('reason', ''),
                      e_correction_ref=str(PACKAGE / 'reviews.json') if r['source_row_id'] in actions else '') for r in rows]
    validate_pairs(successor)
    return successor, actions, banned


def validate_pairs(rows):
    by_id = {r['source_row_id']: r for r in rows}
    check(len(by_id) == len(rows), 'duplicate source row')
    for r in rows:
        partner = by_id[r['paired_source_row_id']]
        check(partner['paired_source_row_id'] == r['source_row_id'] and
              all(partner[k] == r[k] for k in (STATUS, 'person_key', 'entity_id', 'split', 'topic')) and
              {r['label'], partner['label']} == {'0', '1'} and
              {r['form'], partner['form']} == {'affirmative', 'negated'}, 'paired handling mismatch')


def eligible_claim(topic, key, statement_sha, label, rows, banned, judgments):
    sig = topic, key, statement_sha
    matches = [r for r in rows if signature(r) == sig]
    if sig in banned or any(r[STATUS] != 'admitted' for r in matches):
        return False
    j = judgments.get(sig)
    return j is None or j == ('supported_true' if int(label) else 'supported_false')


def guard_memberships(memberships, source):
    current = {r['source_row_id']: r for r in source}
    for name, membership in memberships.items():
        for r in membership:
            check(current[r['source_row_id']][STATUS] == 'admitted' and r[STATUS] == 'admitted',
                  f'restricted paired source survives membership {name}')
        validate_pairs(membership)


def semantic_rows(rows, status):
    return [{**{k: r[k] for k in SEMANTIC}, 'current_status': r[status]} for r in rows]


def capacity_only(ids, topic):
    """Arithmetic and in-memory graph verification, not a production pair manifest."""
    check(len(set(ids)) == len(ids), 'duplicate compound entity')
    n = len(ids)
    if n < 5:
        return dict(usable_entities=n, true_false_fact_pairs=n, unordered_capacity=n*(n-1)//2,
                    recipe_pairs=0, expected_binary_rows=0, degree_four_possible=False)
    ordered = sorted(ids, key=lambda eid: (digest(canonical_json(
        ['validation-degree4-ring-v1', 0, topic, 'test', eid])), eid))
    pairs = {tuple(sorted((ordered[i], ordered[(i+offset) % n])))
             for i in range(n) for offset in (1, 2)}
    degrees = Counter(eid for pair in pairs for eid in pair)
    check(len(pairs) == 2*n and set(degrees.values()) == {4}, 'recovered pairing recipe capacity failed')
    check(all(a != b for a, b in pairs), 'self pair')
    return dict(usable_entities=n, true_false_fact_pairs=n, unordered_capacity=n*(n-1)//2,
                recipe_pairs=len(pairs), expected_binary_rows=16*len(pairs), degree_four_possible=True)


def build(root=ROOT):
    root = Path(root)
    lock = load(root, PACKAGE / 'pre_judgment_lock.json')
    check(lock['final_evaluation_enabled'] is False and lock['negative_candidate_seed'] == 0 and
          lock['correction_base'] == 'candidate_overlay_v4', 'audit lock changed')
    for p, sha in lock['historical_sha256'].items():
        check(digest((root / p).read_bytes()) == sha, f'historical bytes changed: {p}')
    for p, sha in lock['queue_sha256'].items():
        check(digest((root / PACKAGE / p).read_bytes()) == sha, 'pre-judgment queue changed')
    config = load(root, 'config/clean_protocol/generalization_protocols.json')
    check(config['final_evaluation_enabled'] is False, 'final scoring must stay disabled')
    rows = csv_read(root, PREVIOUS / 'row_manifest.csv')
    manifest = csv_read(root, 'data/clean_protocol/entity_partitions/manifest.csv')
    inventory, entities, queue = reconstruct_queue(rows, manifest)
    for name, value in zip(('original_E_inventory.json', 'entity_audit_order.json', 'negative_candidate_queue.json'),
                           (inventory, entities, queue)):
        check(encoded(value) == (root / PACKAGE / name).read_bytes(), 'nonreproducible frozen ordering/inventory')
    reviews = load(root, PACKAGE / 'reviews.json')
    evidence = load(root, PACKAGE / 'evidence.json')
    retrievals = load(root, PACKAGE / 'retrieval_log.json')
    validate_reviews(rows, entities, queue, reviews, evidence)
    for ev in evidence.values():
        if 'reused_from' in ev:
            check(digest((root / ev['reused_from']).read_bytes()) == ev['package_sha256'], 'reused evidence bytes changed')
            old = load(root, ev['reused_from'])[ev['original_evidence_id']]
            check(old == ev['original_evidence'] and ev['url'] == old['url'] and
                  ev['response_sha256'] == old['response_sha256'], 'reused evidence provenance changed')
            check(ev.get('reused_response_row_selection') or ev['observations'] == old['observations'],
                  'recovered evidence silently rewritten')
        else:
            check(any(r['url'] == ev['url'] and r.get('response_sha256') == ev['response_sha256']
                      and r['http_status'] == 200 for r in retrievals), 'unbound evidence retrieval')
    successor, actions, banned = propagate(rows, reviews)
    check(all(all(a[k] == v for k, v in b.items()) for b, a in zip(rows, successor)), 'historical row fields changed')
    judgments = {signature(dict(f, topic=r['topic'], person_key=r['person_key'])): f['judgment']
                 for r in reviews for f in r['facts']}
    by_id = {r['source_row_id']: r for r in successor}
    complete = [r for r in reviews if r['status'] == 'usable']
    for r in complete:
        for f in (r['facts'][0], r['facts'][-1]):
            check(eligible_claim(r['topic'], r['person_key'], f['statement_sha256'], f['label'],
                                 successor, banned, judgments), 'restricted fact survives completed E')
    fitting_keys = {r['person_key'] for r in successor if r['split'] != 'test'}
    check(not fitting_keys & {r['person_key'] for r in reviews}, 'E person overlaps train/D')
    outputs = {}
    def jout(p, v): outputs[p] = encoded(v)
    def cout(p, rs, fields=None): outputs[p] = csv_bytes(rs, fields or list(rs[0]))
    cout(OVERLAY / 'row_manifest.csv', successor)
    for status in ('admitted', 'excluded', 'quarantined'):
        cout(OVERLAY / (status+'_rows.csv'), [r for r in successor if r[STATUS] == status])
    cout(OVERLAY / 'status_changes.csv', [r for r in successor if r['source_row_id'] in actions], list(successor[0]))
    source_elig = [dict(source_row_id=r['source_row_id'], paired_source_row_id=r['paired_source_row_id'],
        topic=r['topic'], person_key=r['person_key'], entity_id=r['entity_id'], split=r['split'],
        statement_sha256=r['statement_sha256'], label=r['label'], form=r['form'],
        current_status=r[STATUS], **{ELIGIBLE: str(r[STATUS] == 'admitted')}) for r in successor]
    cout(OVERLAY / 'source_fact_eligibility.csv', source_elig)
    previous_facts = csv_read(root, PREVIOUS / 'fact_inventory.csv')
    facts = []
    for f in previous_facts:
        allowed = eligible_claim(f['topic'], f['person_key'], f['statement_sha256'], f['label'], successor, banned, judgments)
        if f['origin'] == 'source':
            allowed &= by_id[f['fact_ref']][STATUS] == 'admitted'
        facts.append(dict(f, **{ELIGIBLE: str(f['tc_current_eligible'] == 'True' and allowed)},
                          e_reason='historical_eligibility_retained' if allowed else 'restricted_exact_claim_or_label'))
    cout(OVERLAY / 'fact_inventory.csv', facts)
    audited = []
    for r in reviews:
        for f in r['facts']:
            allowed = f['judgment'] == ('supported_true' if f['label'] else 'supported_false') and eligible_claim(
                r['topic'], r['person_key'], f['statement_sha256'], f['label'], successor, banned, judgments)
            audited.append(dict(topic=r['topic'], entity_id=r['entity_id'], person_key=r['person_key'], split='test',
                fact_ref=f.get('source_row_id', f.get('fact_id')), statement=f['statement'],
                statement_sha256=f['statement_sha256'], label=f['label'], judgment=f['judgment'],
                **{ELIGIBLE: str(allowed)}, completed_pair_eligible=str(allowed and r['status'] == 'usable')))
    cout(PROJECTION / 'audited_fact_eligibility.csv', audited)
    # Propagate source restrictions to every alternate E queue reference, even unattempted candidates.
    audit_by_fact = {f['fact_id']: f for r in reviews for f in r['facts'] if 'fact_id' in f}
    current_queue = []
    for q in queue:
        j = audit_by_fact.get(q['fact_id'], {}).get('judgment', 'not_adjudicated')
        allowed = q['base_eligible'] and j == 'supported_false' and eligible_claim(
            q['topic'], q['person_key'], q['statement_sha256'], 0, successor, banned, judgments)
        current_queue.append(dict(q, judgment=j, **{ELIGIBLE: allowed},
            attempted=q['fact_id'] in audit_by_fact,
            reason='base_restriction_preserved' if not q['base_eligible'] else
                   'completed_negative' if allowed else 'unreviewed_rejected_or_unresolved'))
    jout(PROJECTION / 'E_negative_candidate_eligibility.json', current_queue)
    for r in source_elig + facts + audited:
        if signature(r) in banned or r.get('source_row_id', r.get('fact_ref')) in actions:
            check(r[ELIGIBLE] == 'False', 'restricted source or duplicate registry fact survives')
    memberships = {s+'_admitted_membership': [r for r in successor if r['split'] == s and r[STATUS] == 'admitted']
                   for s in ('train', 'validation', 'test')}
    akeys = {}
    for size in (15, 10):
        name = f'A{size}_proposal.json'
        outputs[PROJECTION / name] = (root / PILOT / name).read_bytes()
        akeys[size] = {r['person_key'] for r in load(root, PILOT / name)['completed_pairs']}
        membership = [r for r in memberships['train_admitted_membership'] if r['person_key'] not in akeys[size]]
        memberships[f'P{size}_admitted_membership'] = membership
        memberships[f'P{size}_balanced_exposure'] = sampler.balanced(membership)
        check(not akeys[size] & {r['person_key'] for r in membership}, 'P/A overlap')
    check(akeys[10] <= akeys[15] and not akeys[15] & {r['person_key'] for r in reviews}, 'A nesting/E leak')
    memberships['train_balanced_exposure'] = sampler.balanced(memberships['train_admitted_membership'])
    guard_memberships(memberships, successor)
    for name, member in memberships.items(): cout(PROJECTION / (name+'.csv'), member)
    outputs[PROJECTION / 'outer_training_reference.csv'] = (root / PILOT / 'outer_training_reference.csv').read_bytes()
    tcpath = BASE / 'tc_topup_audit_v1/combined_TC_completed_pairs.json'
    tc = load(root, tcpath)
    outputs[PROJECTION / 'TC_completed_pairs.json'] = (root / tcpath).read_bytes()
    check(not {r['person_key'] for r in tc} & {r['person_key'] for r in reviews}, 'T_C/E overlap')
    for r in tc:
        for f in r['facts']:
            check(eligible_claim(r['topic'], r['person_key'], f['statement_sha256'], f['label'],
                                 successor, banned, judgments), 'E correction affects T_C')
    comparisons = {}
    for split in ('train', 'validation', 'test'):
        old = semantic_rows([r for r in rows if r['split'] == split], 'tc_successor_status')
        new = semantic_rows([r for r in successor if r['split'] == split], STATUS)
        changed = [b['source_row_id'] for a, b in zip(old, new) if a != b]
        comparisons[split] = dict(before_sha256=digest(encoded(old)), after_sha256=digest(encoded(new)),
            changed_source_ids=changed, semantic_identity_label_statement_person_key_changed=False,
            admission_changes=len(changed))
        check(all({k: a[k] for k in SEMANTIC} == {k: b[k] for k in SEMANTIC}
                  for a, b in zip(old, new)), 'semantic identity/label/statement/person changed')
    for size in (15, 10):
        name = f'A{size}_proposal.json'
        comparisons[f'A{size}'] = dict(unchanged=True, before_sha256=digest((root / PILOT / name).read_bytes()),
                                     after_sha256=digest(outputs[PROJECTION / name]))
    comparisons['T_C'] = dict(unchanged=True, before_sha256=digest((root / tcpath).read_bytes()),
                             after_sha256=digest(outputs[PROJECTION / 'TC_completed_pairs.json']))
    for size in (15, 10):
        for suffix in ('admitted_membership', 'balanced_exposure'):
            name = f'P{size}_{suffix}'
            old = csv_read(root, PILOT / (name+'.csv'))
            check(semantic_rows(old, 'tc_successor_status') == semantic_rows(memberships[name], STATUS),
                  'non-E fitting/sampler input changed: invalidate affected work')
            comparisons[name] = dict(unchanged=True, semantic_sha256=digest(encoded(semantic_rows(old, 'tc_successor_status'))))
    non_e_changed = any(comparisons[s]['changed_source_ids'] for s in ('train', 'validation'))
    jout(PROJECTION / 'semantic_comparison_to_v4.json', comparisons)
    capacities = []
    for topic in TOPICS:
        rs = [r for r in reviews if r['topic'] == topic]
        admitted = [r for r in complete if r['topic'] == topic]
        capacities.append(dict(topic=topic, original_E_entities=sum(e['topic']==topic for e in inventory),
            inventory_excluded=sum(e['topic']==topic and not e['eligible_for_audit'] for e in inventory),
            attempted_entities=len(rs), attempted_negative_candidates=sum(len(r['facts'])-1 for r in rs),
            unresolved_entities=sum(r['status']=='unresolved' for r in rs),
            rejected_true_negative_candidates=sum(f['judgment']=='supported_true' for r in rs for f in r['facts'][1:]),
            **capacity_only([r['entity_id'] for r in admitted], topic)))
    cout(PACKAGE / 'capacity_counts.csv', capacities)
    jout(PACKAGE / 'completed_E_fact_pairs.json', [dict(r, facts=[r['facts'][0], r['facts'][-1]]) for r in complete])
    jout(PACKAGE / 'unresolved_cases.json', [r for r in reviews if r['status']=='unresolved'])
    jout(PACKAGE / 'inventory_exclusions.json', [e for e in inventory if not e['eligible_for_audit']])
    reused = {k:dict(path=e['reused_from'], original_evidence_id=e['original_evidence_id'],
                    package_sha256=e['package_sha256'], bound_facts=e['fact_bindings'])
              for k, e in evidence.items() if 'reused_from' in e}
    jout(PACKAGE / 'evidence_reuse_receipt.json', reused)
    summary = dict(status='bounded_score_blind_E_fact_audit_complete_with_unresolved_exclusions',
        correction_base=str(OVERLAY), base=str(PREVIOUS), status_column=STATUS, eligibility_column=ELIGIBLE,
        final_evaluation_enabled=False, P_A_frozen=False, fitting_performed=False,
        E_activations_accessed=False, predictions_accessed=False, production_compounds_generated=False,
        original_entities=len(inventory), attempted_entities=len(reviews),
        attempted_negative_candidates=sum(len(r['facts'])-1 for r in reviews),
        completed_entities=len(complete), unresolved_entities=sum(r['status']=='unresolved' for r in reviews),
        unresolved_fact_judgments=sum(f['judgment']=='unresolved' for r in reviews for f in r['facts']),
        changed_source_ids=sorted(actions), new_confirmed_label_errors=sum(a['status']=='excluded' for a in actions.values())//2,
        source_claims_quarantined=sum(a['status']=='quarantined' for a in actions.values())//2,
        admitted_by_split=dict(Counter(r['split'] for r in successor if r[STATUS]=='admitted')),
        memberships={k:len(v) for k,v in memberships.items()}, capacity=capacities,
        reused_evidence_entries=len(reused), expected_recipe_pairs=sum(c['recipe_pairs'] for c in capacities),
        expected_binary_rows=sum(c['expected_binary_rows'] for c in capacities),
        non_E_inputs_changed=non_e_changed, v4_train_D_fits_reusable=not non_e_changed,
        outstanding_decisions=['Review unresolved exclusions and E-only overlay v5 before future final benchmark authorization.',
          'Authorize a production test-split adaptation of the recovered degree-four recipe separately; counts here are a capacity projection, no final generation/scoring config.',
          'Imported canonical sensitivity used v3; earlier v4 train restrictions already require mac 2 adoption/cache-mapping review. E-only v5 introduces no additional train/D invalidation.',
          'Final P/A reservation, fitting, final scoring and any E activation extraction remain unperformed and unauthorized by this package.'],
        recommendation='Advance E admission to v5 before any later authorized use. No confirmed source label error was established; unresolved exact source facts require paired quarantine. All source labels and partitions persist. Train/D and v4-bound A/T_C facts are semantically unchanged. The historical sensitivity package remains historical and is not silently rebound to v4/v5.')
    jout(PACKAGE / 'recommendation.json', summary)
    jout(OVERLAY / 'correction_receipt.json', dict(base=str(PREVIOUS), successor=str(OVERLAY),
        status_column=STATUS, eligibility_column=ELIGIBLE, actions=actions,
        semantic_comparison=comparisons, non_E_inputs_changed=non_e_changed,
        exact_duplicate_policy='Match topic/person/statement hash and exact text; restrict both source forms and every alternate fact reference.',
        original_labels_preserved=True, admitted_by_split=summary['admitted_by_split']))
    receipt = dict(inputs={str(PACKAGE/n):digest((root/PACKAGE/n).read_bytes()) for n in INPUT_NAMES},
        producer_sha256=digest((root/'src/final_partition_fact_audit.py').read_bytes()),
        outputs={str(k):digest(v) for k,v in outputs.items()},
        checks=['pre-judgment hashes and reconstructed fixed queue', 'all exact fact/entity/person/evidence bindings',
                'no unresolved replacement; first-supported-false stopping', 'paired quarantine and duplicate registry propagation',
                'train/D/A/T_C semantic and membership preservation', 'actual balanced sampler membership',
                'complete-pair-only degree-four capacity', 'final scoring remains disabled'])
    jout(PACKAGE / 'validation_receipt.json', receipt)
    jout(PROJECTION / 'projection_receipt.json', receipt)
    return outputs, summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--write', action='store_true')
    args = p.parse_args()
    outputs, summary = build()
    # Preflight every path before any mutation; historical bytes are never overwritten.
    for path, payload in outputs.items():
        target = ROOT/path
        check(not target.exists() or target.read_bytes() == payload, f'differing immutable output: {path}')
        check(args.write or target.exists(), f'missing output: {path}')
    if args.write:
        for path, payload in outputs.items():
            target = ROOT/path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): target.write_bytes(payload)
    print(json.dumps({k: summary[k] for k in ('completed_entities','unresolved_entities','admitted_by_split',
          'expected_recipe_pairs','expected_binary_rows','non_E_inputs_changed')}, indent=2))


if __name__ == '__main__':
    main()
