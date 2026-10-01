"""Bounded seven-claim closure and non-default successor to the T2C overlay.

Source facts only. Preserve all v1 columns; successor_status is the new decision.
No source relabeling, entity reservation, or experimental interfaces.
"""
from collections import Counter
from pathlib import Path

from src import selection_repair_candidate_overlay as prior

BASE = prior.BASE
CLOSURE = BASE / 'source_closure_v1'
SUCCESSOR = BASE / 'candidate_overlay_v2'
EXPECTED_IDS = ['inventors:' + str(n) for n in (2, 68, 139, 233, 234, 240, 377)]
METHOD = 'Codex-assisted external review, not human verification'
IDENTITIES = ('source_row_id', 'source_sha256', 'source_row_sha256', 'statement_sha256',
              'label', 'split', 'entity_id', 'person_key', 'paired_source_row_id')


def validate_decisions(manifest, queue, decisions, evidence):
    require = prior.require
    require([r['source_row_id'] for r in queue] == EXPECTED_IDS, 'Frozen seven-case queue mismatch')
    require([r['source_row_id'] for r in decisions] == EXPECTED_IDS, 'Decision coverage/order mismatch')
    by_id = {r['source_row_id']: r for r in manifest}
    for q, d in zip(queue, decisions):
        r = by_id[d['source_row_id']]
        require(r['candidate_status'] == 'excluded', 'Unexpected v1 status')
        require(d['original_label'] == r['label'] == '0', 'Original label changed')
        for key in ('source_row_id', 'paired_source_row_id', 'statement_sha256', 'split', 'person_key'):
            require(d[key] == q[key] == r[key], 'Decision identity changed: ' + key)
        require(d['previous_v1_judgment'] == q['judgment'] == 'supported_true', 'Historical judgment changed')
        require(d['judgment'] in {'supported_true', 'unresolved'}, 'Invalid closure judgment')
        expected = ('excluded', 'corroborated_label_conflict') if d['judgment'] == 'supported_true' else (
            'quarantined', 'unresolved_not_confirmed_error')
        require((d['successor_status'], d['classification']) == expected, 'Judgment/action mismatch or readmission')
        require(d['sample_denominator_contribution'] == 0 and d['reviewer_method'] == METHOD, 'Sample/reviewer changed')
        require(d['reasoning'] and d['limitations'] and d['review_date'], 'Missing reasoning or limitations')
        require(d['v1_evidence_ids'] == q['evidence_ids'], 'Historical evidence reference changed')
        require(d['evidence_ids'] and all(k in evidence for k in d['evidence_ids']), 'Unknown evidence')
        if d['judgment'] == 'supported_true':
            require(any(evidence[k]['corroborates_residence'] for k in d['evidence_ids']), 'No residential corroboration')
    for key, e in evidence.items():
        require(e['id'] == key and e['url'].startswith('https://'), 'Invalid evidence identity')
        require(all(e[k] for k in ('title', 'authority', 'retrieval_date', 'inspection', 'actual_support_and_limitations')), 'Evidence context missing')
        require(e['fetch_result'].startswith('200') and len(e['response_sha256']) == 64, 'Evidence retrieval missing')
        require(e['observations'] and all(o['locator'] and o['excerpt'] for o in e['observations']), 'Missing source text/locator')
        require(type(e['corroborates_residence']) is bool, 'Evidence scope missing')
        require(e['reviewer_method'] == METHOD, 'Evidence reviewer changed')


def apply_successor(manifest, decisions):
    """Retain v1 decision fields and append the explicitly versioned successor."""
    decision_by_id = {}
    for d in decisions:
        for row_id in (d['source_row_id'], d['paired_source_row_id']):
            prior.require(row_id not in decision_by_id, 'Duplicate paired decision')
            decision_by_id[row_id] = d
    result = []
    for r in manifest:
        d = decision_by_id.get(r['source_row_id'])
        result.append({**r, 'successor_status': d['successor_status'] if d else r['candidate_status'],
                       'closure_decision_ref': 'source_closure_v1/decisions.json#' + d['source_row_id'] if d else '',
                       'successor_manifest_status': 'candidate_pending_review_not_production'})
    by_id = {r['source_row_id']: r for r in result}
    for old, new in zip(manifest, result):
        prior.require(all(new[k] == v for k, v in old.items()), 'Historical row modified')
        prior.require((old['candidate_status'] == 'admitted') == (new['successor_status'] == 'admitted'), 'Admission membership changed')
        prior.require(new['successor_status'] == by_id[new['paired_source_row_id']]['successor_status'], 'Pair handling changed')
    return result


def admitted_identity_bytes(records, status_field):
    return prior.csv_bytes([{k: r[k] for k in IDENTITIES} for r in records if r[status_field] == 'admitted'])


def build(root):
    directory = root / CLOSURE
    locks = prior.load(directory / 'historical_lock.json')
    for name, digest in locks.items():
        prior.require(prior.sha((root / name).read_bytes()) == digest, 'Historical file changed: ' + name)
    frozen = prior.load(directory / 'pre_review_receipt.json')
    for name, key in [('policy.json', 'policy_sha256'), ('queue.json', 'queue_sha256')]:
        prior.require(prior.sha((directory / name).read_bytes()) == frozen[key], 'Frozen closure policy/queue changed')
    # Validate every derived v1 output using the unchanged historical implementation.
    prior.run(root, check_only=True)
    manifest = prior.rows(root / prior.PACKAGE / 'row_manifest.csv')
    queue = prior.load(directory / 'queue.json')
    decisions = prior.load(directory / 'decisions.json')
    evidence = prior.load(directory / 'evidence.json')
    validate_decisions(manifest, queue, decisions, evidence)
    for e in evidence.values():
        if 'v1_evidence_id' in e:
            old_path = root / prior.PACKAGE / 'evidence.json'
            prior.require(prior.sha(old_path.read_bytes()) == e['v1_evidence_sha256'], 'Saved evidence changed')
            old = prior.load(old_path)[e['v1_evidence_id']]
            prior.require(old['response_sha256'] == e['response_sha256'], 'Saved response identity changed')
    records = apply_successor(manifest, decisions)
    recommendation = prior.load(directory / 'recommendation.json')
    prior.require(recommendation['status'] == 'awaiting_review_not_production_default', 'Premature recommendation adoption')
    prior.require(recommendation['recommended_overlay'] == str(SUCCESSOR), 'Wrong recommended overlay')
    for status, key in [('excluded', 'corroborated_excluded_pairs'), ('quarantined', 'unresolved_quarantined_pairs')]:
        prior.require(recommendation[key] == [d['source_row_id'] for d in decisions if d['successor_status'] == status], 'Recommendation decisions mismatch')
    prior.require(recommendation['admission_membership_changed'] is False, 'Recommendation membership changed')
    prior.require(recommendation['admitted_counts'] == dict(Counter(r['split'] for r in records if r['successor_status'] == 'admitted')), 'Recommendation counts mismatch')
    roles = recommendation['role_policy']
    prior.require(roles == dict(T_C_may_overlap=['P', 'A'], T_C_excludes=['D', 'E'],
        T_C_twenty_per_topic_scope='outer_train', extra_T_C_facts_may_come_from_P_without_enlarging_A=True,
        reservation_pilot_per_topic=15, fallback_per_topic=10,
        twenty_per_topic_requires='deliberate B=50 commitment', entities_selected_or_reserved=False), 'Role interpretation changed')
    attempts = Counter((r['case'], r['role']) for r in prior.load(directory / 'retrieval_log.json'))
    for (case, role), count in attempts.items():
        prior.require(case in {'walton', 'kapany', 'gobel', 'bengio', 'ericsson'}, 'Unbounded follow-up subject')
        prior.require(role in {'discovery_only_not_evidence', 'source_attempt'}, 'Unknown retrieval role')
        prior.require(count <= (3 if role == 'discovery_only_not_evidence' else 6), 'Bounded retrieval limit exceeded')
    before = admitted_identity_bytes(manifest, 'candidate_status')
    after = admitted_identity_bytes(records, 'successor_status')
    prior.require(before == after, 'Ordered admitted identities changed')
    effective = [{**r, 'candidate_status': r['successor_status']} for r in records]
    counts = prior.count_rows(effective)
    # Capacity/evidence availability is unchanged. Only restriction categories move.
    capacity = prior.rows(root / prior.PACKAGE / 'capacity.csv')
    for row in capacity:
        for stage in ('original', 'admitted', 'excluded', 'quarantined'):
            count = next(c for c in counts if (c['topic'], c['split'], c['stage'], c['label'], c['form']) == (
                row['topic'], row['split'], stage, 'all', 'all'))
            for suffix, key in [('rows', 'rows'), ('entities', 'original_entities'), ('person_keys', 'person_keys')]:
                row[stage + '_' + suffix] = str(count[key])
    transitions = [{k: r[k] for k in (*IDENTITIES, 'candidate_status', 'successor_status', 'closure_decision_ref')}
                   for r in records if r['candidate_status'] != r['successor_status']]
    outputs = {SUCCESSOR / 'row_manifest.csv': prior.csv_bytes(records),
               SUCCESSOR / 'admitted_identities.csv': after,
               SUCCESSOR / 'counts.csv': prior.csv_bytes(counts),
               SUCCESSOR / 'capacity.csv': prior.csv_bytes(capacity),
               SUCCESSOR / 'status_changes.csv': prior.csv_bytes(transitions)}
    for status in ('admitted', 'excluded', 'quarantined'):
        outputs[SUCCESSOR / (status + '_rows.csv')] = prior.csv_bytes([r for r in records if r['successor_status'] == status])
    inputs = dict(locks)
    for name in ('policy.json', 'queue.json', 'pre_review_receipt.json', 'historical_lock.json',
                 'decisions.json', 'evidence.json', 'retrieval_log.json', 'recommendation.json'):
        inputs[str(CLOSURE / name)] = prior.sha((directory / name).read_bytes())
    for name in ('src/selection_repair_source_closure.py', 'scripts/48_close_selection_repair_sources.py',
                 'tests/test_selection_repair_source_closure.py', 'docs/selection_repair_source_closure_v1.md'):
        inputs[name] = prior.sha((root / name).read_bytes())
    receipt = dict(status='recommendation_pending_review', production_default=False,
                   source_only=True, experiments_run=False, activation_compatibility_verified=False,
                   classification_counts=dict(Counter(r['classification'] for r in decisions)),
                   admission_membership_changed=False, admitted_rows=sum(r['successor_status'] == 'admitted' for r in records),
                   admitted_by_partition=dict(Counter(r['split'] for r in records if r['successor_status'] == 'admitted')),
                   v1_admitted_identity_sha256=prior.sha(before), v2_admitted_identity_sha256=prior.sha(after),
                   row_status_counts=dict(Counter(r['successor_status'] for r in records)),
                   changed_rows=len(transitions), frozen_sample_and_42_case_queue_unchanged=True,
                   historical_inputs=inputs,
                   outputs={str(p): {'sha256': prior.sha(data), 'bytes': len(data)} for p, data in outputs.items()})
    outputs[CLOSURE / 'closure_receipt.json'] = prior.json_bytes(receipt)
    return outputs, receipt


def run(root, check_only=False):
    outputs, receipt = build(root)
    for path, payload in outputs.items():
        target = root / path
        prior.require(not target.is_symlink(), 'Output symlink forbidden')
        if target.exists():
            prior.require(target.read_bytes() == payload, 'Existing closure output differs: ' + str(path))
        else:
            prior.require(not check_only, 'Missing closure output: ' + str(path))
    if not check_only:
        for path, payload in outputs.items():
            target = root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                with target.open('xb') as stream:
                    stream.write(payload)
    return receipt
