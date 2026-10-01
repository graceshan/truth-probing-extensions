"""Project a frozen fact-only T_C top-up through conservative source corrections.

Only source/evidence records and the existing row sampler are read. No compound
construction, model inputs, activations, predictions or fitting are involved.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from src import capacity_pilot_audit as old
from src import capacity_pilot_projection as prior

ROOT, BASE = prior.ROOT, prior.BASE
PACKAGE = BASE / 'tc_topup_audit_v1'
OVERLAY = BASE / 'candidate_overlay_v4'
PROJECTION = BASE / 'capacity_pilot_projection_v3'
STATUS = 'tc_successor_status'
ELIGIBLE = 'tc_current_eligible'
require, digest, encoded, csv_data = old.require, old.digest, old.encoded, old.csv_data
load, read_csv, signature = prior.load, prior.read_csv, prior.signature
INPUTS = ('pre_judgment_lock.json', 'traversal_queue.csv', 'reviews.json', 'evidence.json', 'retrieval_log.json')


def validate_traversal(reviews, queue, completed, historical):
    """No old unresolved reopening, skipped rank, redraw or review after capacity."""
    require([(r['topic'], r['rank']) for r in reviews] ==
            sorted((r['topic'], r['rank']) for r in reviews), 'noncanonical traversal')
    visited = {(r['topic'], r['rank']) for r in historical + completed}
    require(not visited & {(r['topic'], r['rank']) for r in reviews}, 'previous attempt reopened')
    ledger = []
    for topic in old.TOPICS:
        candidates = sorted((q for q in queue if q['topic'] == topic), key=lambda q: int(q['rank']))
        candidates = [q for q in candidates if (topic, int(q['rank'])) not in visited]
        attempts = [r for r in reviews if r['topic'] == topic]
        require([r['rank'] for r in attempts] == [int(q['rank']) for q in candidates[:len(attempts)]],
                'queue gap or replacement')
        count = sum(r['topic'] == topic for r in completed)
        for r, q in zip(attempts, candidates):
            require(count < 20, 'review after stopping rule')
            require(q['state'] == 'not_adjudicated', 'previous unresolved reopened')
            require(r['person_key'] == q['person_key'], 'queued person mismatch')
            require(r['facts'][0]['statement'] == q['positive_statement']
                    and r['facts'][1]['statement'] == q['negative_statement']
                    and r['facts'][1].get('source_row_id', r['facts'][1].get('fact_id')) == q['negative_provenance'],
                    'frozen fact replacement')
            count += r['status'] == 'usable'
            ledger.append(dict(topic=topic, rank=r['rank'], person_key=r['person_key'],
                               status=r['status'], cumulative_usable=count))
        require(count == 20 or len(attempts) == len(candidates), 'unexhausted shortfall')
        require(count <= 20, 'T_C exceeds target')
    return ledger


def restrictions(rows, reviews):
    """Match by exact claim, including source duplicates of generated candidates."""
    actions, banned, audit = {}, set(), []
    for review in reviews:
        for fact in review['facts']:
            sig = review['topic'], review['person_key'], fact['statement_sha256']
            require(digest(fact['statement'].encode()) == sig[2], 'statement hash mismatch')
            matches = [r for r in rows if signature(r) == sig]
            require(all(r['statement'] == fact['statement'] and r['entity_id'] in review['entity_ids']
                        and r['form'] == 'affirmative' for r in matches), 'exact source duplicate mismatch')
            if fact['origin'] == 'source':
                require(any(r['source_row_id'] == fact['source_row_id'] and int(r['label']) == fact['label']
                            for r in matches), 'source identity mismatch')
            unresolved = fact['judgment'] == 'unresolved'
            # Compare each source duplicate's own label, not only the proposed fact label.
            conflicts = [r for r in matches if fact['judgment'] in ('supported_true', 'supported_false')
                         and int(r['label']) != int(fact['judgment'] == 'supported_true')]
            if unresolved or conflicts:
                banned.add(sig)
                for row in matches if unresolved else conflicts:
                    if row[prior.STATUS] != 'admitted':
                        continue
                    status = 'quarantined' if unresolved else 'excluded'
                    for sid in (row['source_row_id'], row['paired_source_row_id']):
                        require(sid not in actions or actions[sid]['status'] == status, 'conflicting source actions')
                        actions[sid] = dict(status=status, topic=review['topic'], rank=review['rank'],
                                            person_key=review['person_key'], statement_sha256=sig[2],
                                            judgment=fact['judgment'], reason='unresolved_exact_source_pair' if unresolved
                                            else 'confirmed_source_label_conflict_pair')
            audit.append(dict(topic=review['topic'], rank=review['rank'], person_key=review['person_key'],
                              fact_ref=fact.get('source_row_id', fact.get('fact_id')), statement_sha256=sig[2],
                              judgment=fact['judgment'], origin=fact['origin'],
                              matched_source_ids=[r['source_row_id'] for r in matches],
                              conflicting_source_ids=[r['source_row_id'] for r in conflicts]))
    return actions, banned, audit


def validate_source_pairs(rows):
    old.validate_source_pairs([{**r, 'successor_status': r[STATUS]} for r in rows])


def guard_memberships(memberships, rows, actions, banned):
    restricted_ids = {r['source_row_id'] for r in rows if r[STATUS] != 'admitted'} | set(actions)
    for name, membership in memberships.items():
        for row in membership:
            require(row['source_row_id'] not in restricted_ids and signature(row) not in banned
                    and row[STATUS] == 'admitted', f'restricted source pair survives: {name}/{row["source_row_id"]}')
            require(row['split'] == 'train', 'outer partition leak')
        validate_source_pairs(membership)


def claim_eligible(topic, key, sha, rows, banned):
    sig = topic, key, sha
    matches = [r for r in rows if signature(r) == sig]
    # Any restricted exact duplicate bars an alternate reference from bypassing it.
    return sig not in banned and all(r[STATUS] == 'admitted' for r in matches)


def guard_eligibility(records, rows, banned):
    by_id = {r['source_row_id']: r for r in rows}
    for r in records:
        allowed = claim_eligible(r['topic'], r['person_key'], r['statement_sha256'], rows, banned)
        source = by_id.get(r.get('fact_ref', r.get('source_row_id')))
        allowed &= source is None or source[STATUS] == 'admitted'
        require(allowed or r[ELIGIBLE] == 'False', 'restricted source/registry fact eligibility survives')


def build(root=ROOT):
    root = Path(root)
    lock = load(root, PACKAGE / 'pre_judgment_lock.json')
    for path, sha in lock['historical_sha256'].items():
        require(digest((root / path).read_bytes()) == sha, f'historical bytes changed: {path}')
    queue_bytes = (root / PACKAGE / 'traversal_queue.csv').read_bytes()
    require(digest(queue_bytes) == lock['queue_sha256']
            and queue_bytes == (root / prior.PACKAGE / 'TC_topup_review_queue.csv').read_bytes(), 'frozen queue changed')
    require(lock['target_per_topic'] == 20 and lock['base'] == 'candidate_overlay_v3', 'traversal base changed')
    rows = read_csv(root, prior.OVERLAY / 'row_manifest.csv')
    prior.validate_current(rows)
    reviews = load(root, PACKAGE / 'reviews.json')
    evidence = load(root, PACKAGE / 'evidence.json')
    historical = load(root, old.PACKAGE / 'reviews.json')
    recovered = load(root, old.PACKAGE / 'tc_recovered_pairs.json')
    completed = load(root, prior.PACKAGE / 'TC_completed_pairs.json')
    require(completed == [r for r in historical if r['status'] == 'usable'] + recovered,
            'historical completed pairs changed')
    all_reviews = historical + recovered + reviews
    combined_evidence = load(root, old.PACKAGE / 'evidence.json')
    require(not combined_evidence.keys() & evidence.keys(), 'historical evidence overwritten')
    combined_evidence.update(evidence)
    # Historical source selection is validated on its frozen v2 selection fields.
    # Current eligibility is checked separately below, never by redrawing the queue.
    order = load(root, old.PACKAGE / 'candidate_order.json')
    proposals = load(root, old.PACKAGE / 'negative_proposals.json')
    old.validate_pairs(all_reviews, order, rows, combined_evidence, proposals, root)
    for e in evidence.values():
        if 'reused_response_from' in e:
            path = root / e['reused_response_from']
            require(digest(path.read_bytes()) == e['package_sha256'], 'reused response package changed')
            original = json.loads(path.read_text())[e['original_evidence_id']]
            require(all(e[k] == original[k] for k in ('url', 'response_sha256')), 'reused response changed')
        if 'reused_preliminary_lead_from' in e:
            path = root / e['reused_preliminary_lead_from']
            require(digest(path.read_bytes()) == e['package_sha256'],
                    'recovered preliminary lead changed')
            require(any(r.get('url') == e['url'] and r.get('response_sha256') == e['response_sha256']
                        for r in json.loads(path.read_text())), 'recovered lead response mismatch')
    retrievals = load(root, PACKAGE / 'retrieval_log.json')
    for eid, e in evidence.items():
        if not any(k in e for k in ('reused_from', 'reused_response_from', 'reused_preliminary_lead_from')):
            require(any(r['url'] == e['url'] and r['response_sha256'] == e['response_sha256']
                        and eid in r['evidence_ids'] and r['http_status'] == 200 for r in retrievals),
                    'evidence has no successful bound retrieval')
    for r in reviews:
        if r['topic'] == 'sp_en_trans':
            require(r['spanish_checks']['direction'] == 'Spanish→English'
                    and r['spanish_checks']['headword'] == r['entities'][0]
                    and all(r['spanish_checks'][k] for k in ('spelling_diacritics', 'polysemy_and_dialect', 'scope')),
                    'Spanish headword/sense checks missing')
    queue = read_csv(root, PACKAGE / 'traversal_queue.csv')
    ledger = validate_traversal(reviews, queue, completed, historical)
    actions, banned, audit = restrictions(rows, all_reviews)
    successor = []
    for row in rows:
        action = actions.get(row['source_row_id'])
        successor.append(dict(row, **{STATUS: action['status'] if action else row[prior.STATUS]},
                              tc_correction_ref=(f'tc_topup_audit_v1/reviews.json/{action["topic"]}/'
                                                 f'{action["rank"]}/{action["statement_sha256"]}') if action else '',
                              tc_correction_reason=action['reason'] if action else ''))
    validate_source_pairs(successor)
    for before, after in zip(rows, successor):
        require(all(after[k] == v for k, v in before.items()), 'prior overlay fields changed')
        require((after[STATUS] != before[prior.STATUS]) == (before['source_row_id'] in actions),
                'unexpected row/person restriction')
    combined = completed + [r for r in reviews if r['status'] == 'usable']
    require(len({r['person_key'] for r in combined}) == len(combined), 'duplicate completed person')
    heldout_keys = {r['person_key'] for r in successor if r['split'] != 'train'}
    require(not heldout_keys & {r['person_key'] for r in combined}, 'T_C person overlaps D/E')
    for r in combined:
        require(any(s['person_key'] == r['person_key'] and s['form'] == 'affirmative' and s['label'] == '1'
                    and s['split'] == 'train' and s[STATUS] == 'admitted' for s in successor),
                'completed person lacks admitted positive-bearing train source')
        for fact in r['facts']:
            require(claim_eligible(r['topic'], r['person_key'], fact['statement_sha256'], successor, banned),
                    'restricted fact survives completed T_C')
    outputs = {}

    def json_out(path, data):
        outputs[path] = encoded(data)

    def csv_out(path, data, fields=None):
        outputs[path] = csv_data(data, fields or list(data[0]))

    json_out(PACKAGE / 'traversal_receipt.json', dict(ledger=ledger, target=20,
             skipped_prior_attempts=[dict(topic=q['topic'], rank=int(q['rank']), person_key=q['person_key'], state=q['state'])
                                     for q in queue if (q['topic'], int(q['rank'])) in
                                     {(r['topic'], r['rank']) for r in historical + completed}],
             rule=lock['traversal'], queue_sha256=lock['queue_sha256']))
    json_out(PACKAGE / 'combined_TC_completed_pairs.json', combined)
    json_out(PACKAGE / 'combined_evidence.json', combined_evidence)
    json_out(PACKAGE / 'all_review_outcome_audit.json', audit)
    json_out(PACKAGE / 'correction_proposals.json', dict(base=str(prior.OVERLAY), successor=str(OVERLAY),
             actions=actions, confirmed_label_errors=sum(a['status'] == 'excluded' for a in actions.values()) // 2,
             policy='Unresolved source facts quarantine exact source pairs, including exact registry duplicates. '
                    'Confirmed conflicts exclude exact pairs. Original labels are not rewritten. No person-wide ban.'))
    csv_out(OVERLAY / 'row_manifest.csv', successor)
    for status in ('admitted', 'excluded', 'quarantined'):
        csv_out(OVERLAY / (status + '_rows.csv'), [r for r in successor if r[STATUS] == status])
    csv_out(OVERLAY / 'status_changes.csv', [r for r in successor if r['source_row_id'] in actions])
    row_eligibility = [dict(source_row_id=r['source_row_id'], paired_source_row_id=r['paired_source_row_id'],
                       topic=r['topic'], person_key=r['person_key'], entity_id=r['entity_id'], split=r['split'],
                       statement_sha256=r['statement_sha256'], label=r['label'], form=r['form'],
                       current_status=r[STATUS], **{ELIGIBLE: str(r[STATUS] == 'admitted')},
                       correction_ref=r['tc_correction_ref']) for r in successor]
    guard_eligibility(row_eligibility, successor, banned)
    csv_out(OVERLAY / 'source_fact_eligibility.csv', row_eligibility)
    inventory = []
    for f in read_csv(root, prior.OVERLAY / 'fact_inventory.csv'):
        allowed = claim_eligible(f['topic'], f['person_key'], f['statement_sha256'], successor, banned)
        inventory.append(dict(f, **{ELIGIBLE: str(f['current_eligible'] == 'True' and allowed)},
                              tc_reason='historical_eligibility_retained' if allowed else 'restricted_exact_claim'))
    guard_eligibility(inventory, successor, banned)
    csv_out(OVERLAY / 'fact_inventory.csv', inventory)
    audited = []
    for r in all_reviews:
        for f in r['facts']:
            eligible = f['judgment'] == ('supported_true' if f['label'] else 'supported_false')
            eligible &= claim_eligible(r['topic'], r['person_key'], f['statement_sha256'], successor, banned)
            audited.append(dict(topic=r['topic'], rank=r['rank'], person_key=r['person_key'],
                                fact_ref=f.get('source_row_id', f.get('fact_id')), statement_sha256=f['statement_sha256'],
                                origin=f['origin'], judgment=f['judgment'], **{ELIGIBLE: str(eligible)},
                                completed_pair_eligible=str(r['status'] == 'usable' and eligible)))
    guard_eligibility(audited, successor, banned)
    csv_out(PROJECTION / 'audited_fact_eligibility.csv', audited)
    train = [r for r in successor if r['split'] == 'train' and r[STATUS] == 'admitted']
    memberships = dict(corrected_outer_training=train, corrected_outer_training_balanced=prior.balanced(train))
    # Preserve original source reference bytes; this is not a redefined reference.
    outputs[PROJECTION / 'outer_training_reference.csv'] = (root / old.PACKAGE / 'outer_training_reference.csv').read_bytes()
    keys_by_size, deltas = {}, {}
    for n in (15, 10):
        name = f'A{n}_proposal.json'
        outputs[PROJECTION / name] = (root / prior.PACKAGE / name).read_bytes()
        a = load(root, prior.PACKAGE / name)['completed_pairs']
        require(all(r in completed for r in a), 'A facts changed')
        keys = keys_by_size[n] = {r['person_key'] for r in a}
        require(not keys & heldout_keys, 'A outer partition leak')
        p = memberships[f'P{n}_admitted_membership'] = [r for r in train if r['person_key'] not in keys]
        sample = memberships[f'P{n}_balanced_exposure'] = prior.balanced(p)
        require(not keys & {r['person_key'] for r in p}, 'P/A person/alias overlap')
        csv_out(PROJECTION / f'P{n}_excluded_source_variants.csv',
                [r for r in successor if r['split'] == 'train' and r['person_key'] in keys])
        before = read_csv(root, prior.PACKAGE / f'P{n}_balanced_exposure.csv')
        before_ids = {r['source_row_id'] for r in before}
        after_ids = {r['source_row_id'] for r in sample}
        deltas[str(n)] = dict(removed=sorted(before_ids - after_ids), added=sorted(after_ids - before_ids),
                             previous_sha256=digest((root / prior.PACKAGE / f'P{n}_balanced_exposure.csv').read_bytes()),
                             current_sha256=digest(csv_data(sample, list(successor[0]))))
    require(keys_by_size[10] <= keys_by_size[15], 'A10 is not nested')
    guard_memberships(memberships, successor, actions, banned)
    counts = []
    for name, membership in memberships.items():
        csv_out(PROJECTION / (name + '.csv'), membership)
        counts += old.count_rows(membership, name)
    csv_out(PROJECTION / 'exposure_counts.csv', counts)
    json_out(PROJECTION / 'balanced_membership_changes.json', deltas)
    current_queue = []
    by_rank = {(r['topic'], r['rank']): r for r in all_reviews}
    for q in queue:
        r = by_rank.get((q['topic'], int(q['rank'])))
        fact_elig = {f['label']: f['judgment'] == ('supported_true' if f['label'] else 'supported_false') and
                    claim_eligible(r['topic'], r['person_key'], f['statement_sha256'], successor, banned)
                    for f in r['facts']} if r else {}
        current_queue.append(dict(q, tc_review_status=r['status'] if r else 'not_adjudicated',
                                  tc_positive_eligible=str(fact_elig.get(1, False)),
                                  tc_negative_eligible=str(fact_elig.get(0, False)),
                                  tc_pair_eligible=str(bool(r and r['status'] == 'usable' and all(fact_elig.values())))))
    csv_out(PROJECTION / 'TC_queue_status.csv', current_queue)
    capacity = []
    for t in old.TOPICS:
        accepted = [r for r in combined if r['topic'] == t]
        attempted = [r for r in reviews if r['topic'] == t]
        previous = [r for r in historical + recovered if r['topic'] == t]
        capacity.append(dict(topic=t, previous_completed=sum(r['topic'] == t for r in completed),
                         new_attempted=len(attempted), new_completed=sum(r['status'] == 'usable' for r in attempted),
                         new_unresolved=sum(r['status'] == 'unresolved' for r in attempted),
                         new_rejected=sum(r['status'] == 'rejected' for r in attempted),
                         cumulative_attempted=len(previous) + len(attempted),
                         cumulative_unresolved=sum(r['status'] == 'unresolved' for r in previous + attempted),
                         cumulative_rejected=sum(r['status'] == 'rejected' for r in previous + attempted),
                         usable_entities=len(accepted), possible_unordered_pairs=len(accepted) * (len(accepted) - 1) // 2,
                         planned_control_cap=100, remaining_successful_topups=max(0, 20 - len(accepted)),
                         overlap_A15=sum(r['person_key'] in keys_by_size[15] for r in accepted),
                         overlap_prospective_P15=sum(r['person_key'] in {s['person_key'] for s in memberships['P15_admitted_membership']} for r in accepted)))
    csv_out(PACKAGE / 'capacity_counts.csv', capacity)
    summary = dict(status='prospective_TC_capacity_complete_PA_unfrozen', correction_base=str(OVERLAY),
                   previous_base=str(prior.OVERLAY), status_column=STATUS, eligibility_column=ELIGIBLE,
                   changed_source_ids=sorted(actions), changed_source_claims=[r for r in successor
                       if r['source_row_id'] in actions and r['form'] == 'affirmative'],
                   confirmed_new_label_errors=sum(a['status'] == 'excluded' for a in actions.values()) // 2,
                   correction_base_must_advance_before_fitting=bool(actions),
                   admitted_by_split=dict(Counter(r['split'] for r in successor if r[STATUS] == 'admitted')),
                   current_membership_totals={k: len(v) for k, v in memberships.items()},
                   A15_keys=len(keys_by_size[15]), A10_keys=len(keys_by_size[10]), A_membership_unchanged=True,
                   preserved_completed_pairs=len(completed), new_attempts=len(reviews),
                   new_completed_pairs=sum(r['status'] == 'usable' for r in reviews),
                   new_unresolved=sum(r['status'] == 'unresolved' for r in reviews),
                   new_rejected=sum(r['status'] == 'rejected' for r in reviews),
                   cumulative_attempts=len(all_reviews), combined_completed_pairs=len(combined), capacity=capacity,
                   exposure_counts=counts, sampler=dict(function='balanced_burger_indices', seed=0,
                   numpy=prior.np.__version__, pandas=prior.pd.__version__), negative_queue_regenerated=False,
                   production_compounds_generated=False, fitting_performed=False, B50_committed=False,
                   recommendation='Use candidate_overlay_v4 (tc_successor_status) and capacity_pilot_projection_v3 '
                   'for any later authorized fitting. New unresolved source judgments require a correction-base '
                   'advance; no new confirmed label error was established. Preserve the fixed A15/A10 proposals '
                   'and all 76 earlier T_C pairs. T_C completion does not enlarge A, freeze P/A, authorize fitting '
                   'or commit to B=50. mac 2 owns adoption/cache mapping and sensitivity records. '
                   'Historical eligibility/status columns are references, not the v4 authority.')
    json_out(PACKAGE / 'recommendation.json', summary)
    json_out(PROJECTION / 'recommendation.json', summary)
    json_out(OVERLAY / 'correction_receipt.json', dict(base=str(prior.OVERLAY), status_column=STATUS,
             eligibility_column=ELIGIBLE, judgment_source=str(PACKAGE / 'reviews.json'), actions=actions,
             admitted_by_split=summary['admitted_by_split'], confirmed_label_errors=summary['confirmed_new_label_errors'],
             outputs={str(p): digest(v) for p, v in outputs.items() if p.parent == OVERLAY}))
    receipt = dict(inputs={str(PACKAGE / n): digest((root / PACKAGE / n).read_bytes()) for n in INPUTS},
                   code_sha256=digest((root / 'src/tc_topup_audit.py').read_bytes()),
                   outputs={str(p): digest(v) for p, v in outputs.items()},
                   checks=['historical bytes', 'frozen traversal/stopping rule; old unresolved skipped',
                           'all fact hashes/entities/evidence bindings', 'source/registry exact duplicate policy propagation',
                           'paired corrections and current fitting exclusion', 'original labels/partitions/identities',
                           '76 completed pairs and evidence preserved', 'unchanged nested A proposals',
                           'T_C/P/A person identities and no D/E overlap', 'deterministic sampler and derived counts',
                           'no negative queue regeneration; no fitting or compounds'])
    json_out(PACKAGE / 'validation_receipt.json', receipt)
    json_out(PROJECTION / 'projection_receipt.json', receipt)
    return outputs, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    outputs, summary = build()
    for path, value in outputs.items():
        target = ROOT / path
        require(not target.exists() or target.read_bytes() == value, f'differing output: {path}')
        require(args.write or target.exists(), f'missing output: {path}')
    if args.write:
        for path, value in outputs.items():
            target = ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(value)
    print(json.dumps({k: summary[k] for k in ('changed_source_ids', 'new_attempts', 'new_completed_pairs',
          'new_unresolved', 'combined_completed_pairs', 'admitted_by_split', 'current_membership_totals')}, indent=2))


if __name__ == '__main__':
    main()
