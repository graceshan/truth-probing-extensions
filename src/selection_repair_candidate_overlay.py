"""T2C source-only candidate overlay. No experiment/data-partition loader.

All historical columns survive unchanged. The new candidate status never replaces
T2A status or source labels. Capacity counts affirmative propositions, not their
negations, and never treats a missing country as false evidence.
"""
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path


BASE = Path('data/clean_protocol/selection_repair_v1')
PACKAGE = BASE / 'candidate_overlay_v1'
AUDIT = BASE / 'source_audit_v1'
REVIEW = BASE / 'source_review_v1'
REGISTRIES = (
    Path('data/clean_protocol/validated_negatives/v1/train/registry.csv'),
    Path('data/clean_protocol/validated_negatives/v1/validation_reviewed_batch_001_inventor_manual_external_review_v5/registry.csv'),
)
FRANKLIN = 'fact_49c7c4369a56220424ffb4d35bab8ec46e341f69329209a10d008b109a4383c4'
STATUS = 'candidate_pending_review_not_production'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    return json.loads(path.read_text())


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def json_bytes(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + '\n').encode()


def csv_bytes(records):
    require(bool(records), 'Empty table requires an explicit schema')
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(records)
    return stream.getvalue().encode()


def target_triggers(row, policy):
    triggers = []
    if '/' in row['raw_country']:
        triggers.append('slash_country_unspecified_operator')
    if ' and ' in row['entity']:
        triggers.append('joint_person_subject')
    if row['raw_country'] in ('Ancient Greece', 'the Soviet Union'):
        triggers.append('historical_polity')
    if row['entity'] in policy['geography_watchlist'] and row['raw_country'] == 'the U.S':
        triggers.append('pre_US_polity_or_current_territory_ambiguity')
    if row['source_row_id'] in policy['posting_leads']:
        triggers.append('residential_posting_lead')
    if row['label'] == '1' and row['entity'] in policy['reused_positive_support_entities']:
        triggers.append('reuse_existing_institutional_positive_chronology')
    return triggers


def validate_inputs(root):
    package = root / PACKAGE
    frozen = load(package / 'pre_adjudication_freeze.json')
    for filename, key in [('policy.json', 'policy_sha256'), ('targeted_queue.json', 'queue_sha256'),
                          ('screening_inventory.json', 'screening_sha256')]:
        require(sha((package / filename).read_bytes()) == frozen[key], 'Pre-adjudication bytes changed: ' + filename)
    for path, digest in frozen['input_hashes'].items():
        require(sha((root / path).read_bytes()) == digest, 'Historical input changed: ' + path)
    lock = load(package / 'input_lock.json')
    for path, digest in lock.items():
        require(sha((root / path).read_bytes()) == digest, 'Locked input changed: ' + path)
    originals = rows(root / AUDIT / 'row_manifest.csv')
    by_id = {r['source_row_id']: r for r in originals}
    require(len(by_id) == len(originals), 'Duplicate source IDs')
    for r in originals:
        require(sha(r['statement'].encode()) == r['statement_sha256'], 'Statement hash changed')
        pair = by_id[r['paired_source_row_id']]
        require(pair['paired_source_row_id'] == r['source_row_id'], 'Asymmetric pair')
        require(int(pair['label']) == 1-int(r['label']), 'Pair labels inconsistent')
        require(all(pair[k] == r[k] for k in ('person_key', 'entity_id', 'split', 'topic')), 'Pair identity changed')
    queue = load(package / 'targeted_queue.json')
    reviews = load(package / 'targeted_reviews.json')
    evidence = load(package / 'evidence.json')
    policy = load(package / 'policy.json')
    carry_ids = {r['source_row_id'] for r in rows(root / REVIEW / 'recommended_exclusions.csv')}
    retained = [r for r in originals if r['topic'] == 'inventors' and r['form'] == 'affirmative'
                and r['status'] == 'admitted' and r['source_row_id'] not in carry_ids]
    expected_targets = sorted([r for r in retained if target_triggers(r, policy)], key=lambda r: int(r['row_index']))
    require([r['source_row_id'] for r in queue] == [r['source_row_id'] for r in expected_targets], 'Target policy coverage changed')
    screen = load(package / 'screening_inventory.json')
    require(len(screen) == frozen['false_screen_count'], 'Screen count changed')
    require([r['source_row_id'] for r in screen] == [r['source_row_id'] for r in retained if r['label'] == '0'], 'False-screen coverage changed')
    for r in screen:
        triggers = target_triggers(by_id[r['source_row_id']], policy)
        require(r['triggers'] == triggers, 'Screen trigger mismatch')
        require(r['screen_status'] == ('targeted_review_required' if triggers else 'screened_no_trigger_in_bounded_material'), 'Screen promoted to judgment')
    require(len(queue) == len(reviews) == frozen['queue_count'], 'Target coverage incomplete')
    require(len({r['source_row_id'] for r in queue}) == len(queue), 'Duplicate target')
    for q, review in zip(queue, reviews):
        original = by_id[q['source_row_id']]
        require(all(q[k] == v for k, v in original.items()), 'Queue source identity changed')
        require(q['target_triggers'] == target_triggers(original, policy), 'Target triggers changed')
        for k in ('source_row_id', 'paired_source_row_id', 'statement_sha256', 'split', 'person_key'):
            require(q[k] == review[k], 'Target order/identity mismatch')
        require(q['label'] == review['original_label'], 'Target original label changed')
        require(review['judgment'] in {'supported_true', 'supported_false', 'unresolved'}, 'Invalid judgment')
        require(review['sample_denominator_contribution'] == 0, 'Follow-up added to sample')
        require(review['reasoning'] and review['limitations'] and review['review_date'], 'Missing review context')
        require(review['reviewer_method'] == 'Codex-assisted external review, not human verification', 'Reviewer misrepresented')
        require(review['evidence_ids'] and all(e in evidence for e in review['evidence_ids']), 'Unknown evidence')
    for key, e in evidence.items():
        if 'existing_package' in e:
            require(e['existing_package'] == '../source_review_v1/evidence.json', 'Unexpected evidence path')
            path = root / REVIEW / 'evidence.json'
            require(sha(path.read_bytes()) == e['file_sha256'], 'Reused evidence changed')
            require(e['existing_id'] in load(path), 'Missing reused evidence')
        else:
            require(e['id'] == key and e['url'].startswith('https://') and e['title'], 'Evidence identity missing')
            require(e['retrieval_date'] and len(e['response_sha256']) == 64, 'Evidence provenance missing')
            require(e['fetch_result'].startswith('200') and e['observations'], 'Uninspected evidence')
            require(all(o['locator'] and o['excerpt'] for o in e['observations']), 'Missing inspected locator')
    # T2B judgments, queue order and denominator are inputs, never rewritten.
    sample = rows(root / AUDIT / 'review_queue_30.csv')
    old_reviews = load(root / REVIEW / 'reviews.json')
    require(len(sample) == len(old_reviews) == 30, 'Frozen sample size changed')
    for q, r in zip(sample, old_reviews):
        require(all(q[k] == r[k] for k in ('draw_index', 'source_row_id', 'statement_sha256', 'person_key')), 'Sample changed')
    return originals, reviews, old_reviews


def apply_overlay(originals, recommendations, reviews):
    """Pure row transformation; collect all reasons and resolve overlaps once."""
    by_id = {r['source_row_id']: r for r in originals}
    restrictions = defaultdict(list)
    for r in originals:
        if r['status'] == 'excluded':
            simjian = 'confirmed_person_spans_outer_partitions' in json.loads(r['exclusion_reasons'])
            restrictions[r['source_row_id']].append((
                'quarantined' if simjian else 'excluded', 't2a:' + r['exclusion_reasons'],
                'source_audit_v1/row_manifest.csv#' + r['source_row_id']))
    for rec in recommendations:
        r = by_id[rec['source_row_id']]
        require(all(rec[k] == r[k] for k in ('paired_source_row_id', 'source_sha256', 'statement_sha256', 'split', 'label', 'person_key')), 'Carry identity changed')
        status = 'excluded' if rec['sample_judgment'] == 'supported_true' else 'quarantined'
        require(rec['sample_judgment'] in {'supported_true', 'unresolved'}, 'Invalid carry')
        restrictions[r['source_row_id']].append((status, 't2b:' + rec['reason'],
                                                'source_review_v1/reviews.json#draw=' + rec['sample_draw_index']))
    for review in reviews:
        original = by_id[review['source_row_id']]
        conflict = review['judgment'] == ('supported_true' if original['label'] == '0' else 'supported_false')
        if conflict or review['judgment'] == 'unresolved':
            for row_id in (review['source_row_id'], review['paired_source_row_id']):
                restrictions[row_id].append((
                    'excluded' if conflict else 'quarantined',
                    't2c:confirmed_label_conflict' if conflict else 't2c:unresolved_targeted_claim',
                    'candidate_overlay_v1/targeted_reviews.json#' + review['source_row_id']))
    result = []
    for original in originals:
        reasons = sorted(set(restrictions[original['source_row_id']]))
        statuses = {r[0] for r in reasons}
        status = 'excluded' if 'excluded' in statuses else 'quarantined' if statuses else 'admitted'
        result.append({**original, 'candidate_status': status,
                       'candidate_reasons': json.dumps(sorted({r[1] for r in reasons})),
                       'candidate_evidence_refs': json.dumps(sorted({r[2] for r in reasons})),
                       'candidate_manifest_status': STATUS})
    by_id = {r['source_row_id']: r for r in result}
    for r in result:
        require(r['candidate_status'] == by_id[r['paired_source_row_id']]['candidate_status'], 'Pair restriction incomplete')
    return result


def count_rows(manifest):
    result = []
    for topic, split in sorted({(r['topic'], r['split']) for r in manifest}):
        for stage in ('original', 'admitted', 'excluded', 'quarantined'):
            for label in ('all', '0', '1'):
                for form in ('all', 'affirmative', 'negated'):
                    subset = [r for r in manifest if r['topic'] == topic and r['split'] == split
                              and (stage == 'original' or r['candidate_status'] == stage)
                              and (label == 'all' or r['label'] == label)
                              and (form == 'all' or r['form'] == form)]
                    result.append(dict(topic=topic, split=split, stage=stage, label=label, form=form,
                                       rows=len(subset), original_entities=len({r['entity_id'] for r in subset}),
                                       person_keys=len({r['person_key'] for r in subset})))
    return result


def inventory(manifest, reviews, old_reviews, registries):
    """Conservative fact ledger: rejected/unverified history never supplies capacity."""
    judgments = {r['source_row_id']: ('t2b', r) for r in old_reviews}
    judgments.update({r['source_row_id']: ('t2c', r) for r in reviews})
    facts = []
    entities = defaultdict(list)
    blocked_people = {r['person_key'] for r in manifest if r['topic'] == 'inventors' and r['candidate_status'] != 'admitted'}
    for r in manifest:
        entities[(r['entity_id'], r['split'])].append(r)
        if r['form'] != 'affirmative':
            continue
        audit = judgments.get(r['source_row_id'])
        agrees = audit and audit[1]['judgment'] == ('supported_true' if r['label'] == '1' else 'supported_false')
        facts.append(dict(fact_ref=r['source_row_id'], origin='source', topic=r['topic'], split=r['split'],
                          entity_id=r['entity_id'], person_key=r['person_key'], statement=r['statement'],
                          statement_sha256=r['statement_sha256'], label=r['label'],
                          status=r['candidate_status'], support='external_source_review' if agrees else 'source_label_only',
                          evidence_ref=(audit[0] + ':' + r['source_row_id']) if audit else '',
                          eligible=r['candidate_status'] == 'admitted', reason=r['candidate_reasons']))
    registry_counts = []
    for path, records in registries:
        for (topic, split, status), count in sorted(Counter((r['topic'], r['split'], r['validation_status']) for r in records).items()):
            registry_counts.append(dict(registry=str(path), topic=topic, split=split, historical_status=status, facts=count))
        for r in records:
            if r['validation_status'] == 'unverified':
                continue  # counted explicitly above, never offered as available negatives
            source = entities.get((r['entity_id'], r['split']), [])
            person = source[0]['person_key'] if source else ''
            accepted = r['validation_status'] in {'externally_validated_false', 'source_supported_false'}
            exact = [s for s in source if s['form'] == 'affirmative' and s['statement'] == r['statement']]
            reason = 'available_historical_fact'
            if not accepted or r['fact_id'] == FRANKLIN:
                reason = 'historical_rejected_or_unresolved_not_negative_evidence'
            elif not source:
                reason = 'no_exact_original_entity_partition'
            elif any(s['candidate_status'] != 'admitted' or s['label'] != '0' for s in exact):
                reason = 'exact_source_fact_restricted_or_conflicting'
            elif person in blocked_people:
                reason = 'inventor_subject_requires_revalidation_after_source_restriction'
            elif not any(s['candidate_status'] == 'admitted' for s in source):
                reason = 'no_admitted_source_rows'
            support = 'historical_external_negative' if r['validation_status'] == 'externally_validated_false' else 'historical_source_label_negative'
            if not accepted:
                support = 'historical_rejected_or_unresolved'
            facts.append(dict(fact_ref=r['fact_id'], origin=str(path), topic=r['topic'], split=r['split'],
                              entity_id=r['entity_id'], person_key=person, statement=r['statement'],
                              statement_sha256=sha(r['statement'].encode()), label='0_candidate',
                              status=r['validation_status'], support=support,
                              evidence_ref=r['evidence_source'], eligible=reason == 'available_historical_fact', reason=reason))
    return facts, registry_counts


def capacity(manifest, facts):
    entity_table, totals = [], []
    for topic, split in sorted({(r['topic'], r['split']) for r in manifest}):
        subset = [r for r in manifest if (r['topic'], r['split']) == (topic, split)]
        for person in sorted({r['person_key'] for r in subset}):
            original = [r for r in subset if r['person_key'] == person]
            fs = [f for f in facts if (f['topic'], f['split'], f['person_key']) == (topic, split, person) and f['eligible']]
            def count(label, supports):
                return len({f['statement_sha256'] for f in fs if f['label'] in label and f['support'] in supports})
            pos = count({'1'}, {'source_label_only', 'external_source_review'})
            neg = count({'0'}, {'source_label_only', 'external_source_review'})
            audited_pos = count({'1'}, {'external_source_review'})
            audited_neg = count({'0'}, {'external_source_review'})
            hist_neg = count({'0_candidate'}, {'historical_external_negative'})
            hist_source = count({'0_candidate'}, {'historical_source_label_negative'})
            entity_table.append(dict(topic=topic, split=split, person_key=person,
                original_entity_ids=json.dumps(sorted({r['entity_id'] for r in original})),
                original_rows=len(original), admitted_rows=sum(r['candidate_status'] == 'admitted' for r in original),
                excluded_rows=sum(r['candidate_status'] == 'excluded' for r in original),
                quarantined_rows=sum(r['candidate_status'] == 'quarantined' for r in original),
                positive_source_facts=pos, negative_source_facts=neg,
                audited_positive_facts=audited_pos, audited_negative_facts=audited_neg,
                historical_external_negative_facts=hist_neg, historical_source_negative_facts=hist_source,
                provisional_source_pair=bool(pos and neg),
                positive_plus_historical_reviewed_negative=bool(pos and hist_neg),
                audited_source_pair=bool(audited_pos and audited_neg),
                audited_pair_including_historical_negative=bool(audited_pos and (audited_neg or hist_neg))))
        ps = [r for r in entity_table if (r['topic'], r['split']) == (topic, split)]
        summary = dict(topic=topic, split=split)
        for stage in ('original', 'admitted', 'excluded', 'quarantined'):
            rs = subset if stage == 'original' else [r for r in subset if r['candidate_status'] == stage]
            summary[stage + '_rows'] = len(rs)
            summary[stage + '_entities'] = len({r['entity_id'] for r in rs})
            summary[stage + '_person_keys'] = len({r['person_key'] for r in rs})
        summary['positive_entity_upper_bound'] = sum(bool(p['positive_source_facts']) for p in ps)
        for key in ('positive_source_facts', 'negative_source_facts', 'audited_positive_facts', 'audited_negative_facts',
                    'historical_external_negative_facts', 'historical_source_negative_facts', 'provisional_source_pair',
                    'positive_plus_historical_reviewed_negative', 'audited_source_pair', 'audited_pair_including_historical_negative'):
            summary[key] = sum(p[key] for p in ps)
        for n in (10, 15, 20):
            summary[f'entities_{n}_capacity'] = (
                'audited_bound_only_pending_review' if summary['audited_pair_including_historical_negative'] >= n else
                'provisional_upper_bound_only_requires_fact_review' if summary['positive_entity_upper_bound'] >= n else
                'insufficient_retained_positive_entity_bound')
        totals.append(summary)
    return entity_table, totals


def build(root):
    originals, reviews, old_reviews = validate_inputs(root)
    manifest = apply_overlay(originals, rows(root / REVIEW / 'recommended_exclusions.csv'), reviews)
    facts, registry_counts = inventory(manifest, reviews, old_reviews, [(p, rows(root / p)) for p in REGISTRIES])
    person_table, capacities = capacity(manifest, facts)
    outputs = {'row_manifest.csv': csv_bytes(manifest), 'counts.csv': csv_bytes(count_rows(manifest)),
               'fact_inventory.csv': csv_bytes(facts), 'historical_registry_counts.csv': csv_bytes(registry_counts),
               'capacity_by_person.csv': csv_bytes(person_table), 'capacity.csv': csv_bytes(capacities)}
    for stage in ('admitted', 'excluded', 'quarantined'):
        outputs[stage + '_rows.csv'] = csv_bytes([r for r in manifest if r['candidate_status'] == stage])
    inputs = dict(load(root / PACKAGE / 'input_lock.json'))
    for name in ('policy.json', 'targeted_queue.json', 'screening_inventory.json', 'pre_adjudication_freeze.json',
                 'targeted_reviews.json', 'evidence.json', 'retrieval_log.json', 'input_lock.json', 'preservation_baseline.json'):
        inputs[str(PACKAGE / name)] = sha((root / PACKAGE / name).read_bytes())
    for path in ('src/selection_repair_candidate_overlay.py', 'scripts/47_build_selection_repair_candidate.py'):
        inputs[path] = sha((root / path).read_bytes())
    receipt = dict(status=STATUS, production_default=False, activation_compatibility_verified=False,
        source_only_E_audit=True, experiments_run=False, sample_unchanged=True,
        sample_counts=dict(Counter(r['judgment'] for r in old_reviews)),
        targeted_counts=dict(Counter(r['judgment'] for r in reviews)),
        row_counts=dict(Counter(r['candidate_status'] for r in manifest)),
        original_rows=len(manifest), inputs=inputs,
        outputs={name: {'sha256': sha(payload), 'bytes': len(payload)} for name, payload in outputs.items()})
    outputs['candidate_receipt.json'] = json_bytes(receipt)
    return outputs, receipt


def run(root, check_only=False):
    outputs, receipt = build(root)
    directory = root / PACKAGE
    # Check every collision before writing any output. No overwrite or partial repair.
    for name, payload in outputs.items():
        path = directory / name
        require(not path.is_symlink(), 'Output symlink forbidden')
        if path.exists():
            require(path.read_bytes() == payload, 'Existing candidate differs: ' + name)
        else:
            require(not check_only, 'Missing candidate output: ' + name)
    if not check_only:
        for name, payload in outputs.items():
            path = directory / name
            if not path.exists():
                with path.open('xb') as stream:
                    stream.write(payload)
    return receipt
