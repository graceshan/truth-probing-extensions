"""Propagate existing factual judgments into a successor overlay and pilot projection.

No factual adjudication, candidate redraw, queue initialization or model fitting.
Historical v1 remains independently reproducible against candidate_overlay_v2.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from src import capacity_pilot_audit as old
from src.atomic_probe_methods import balanced_burger_indices

ROOT = old.ROOT
BASE = old.PACKAGE.parent
OVERLAY = BASE / 'candidate_overlay_v3'
PACKAGE = BASE / 'capacity_pilot_projection_v2'
STATUS = 'audit_successor_status'
require, digest, encoded, csv_data = old.require, old.digest, old.encoded, old.csv_data


def load(root, path):
    return json.loads((root / path).read_text())


def read_csv(root, path):
    with (root / path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def signature(row):
    return row['topic'], row['person_key'], row['statement_sha256']


def inspect_outcomes(rows, reviews, proposals):
    """Bind every existing judgment before discovering claim-level restrictions."""
    by_id = {r['source_row_id']: r for r in rows}
    queue = {(p['topic'], p['rank']): p['candidate'] for p in proposals}
    audit, affected, unresolved = [], set(), set()
    for review in reviews:
        facts = []
        for fact in review['facts']:
            require(digest(fact['statement'].encode()) == fact['statement_sha256'], 'judgment statement hash mismatch')
            matches = [r for r in rows if signature(r) ==
                       (review['topic'], review['person_key'], fact['statement_sha256'])]
            if fact['origin'] == 'source':
                row = by_id[fact['source_row_id']]
                require(row in matches and row['statement'] == fact['statement']
                        and row['entity_id'] in review['entity_ids'] and row['entity'] in review['entities']
                        and row['form'] == 'affirmative' and row['split'] == 'train'
                        and int(row['label']) == fact['label'], 'source judgment identity mismatch')
            else:
                candidate = queue[review['topic'], review['rank']]
                require(fact['origin'] == 'recovered_negative_pipeline'
                        and all(fact[k] == candidate[k] for k in
                                ('fact_id', 'statement', 'entity_id', 'ranking_hash', 'candidate_rank'))
                        and candidate['entity_id'] in review['entity_ids'], 'frozen generated candidate mismatch')
            admitted_matches = [r for r in matches if r['successor_status'] == 'admitted']
            if fact['judgment'] == 'unresolved':
                unresolved.add((review['topic'], review['person_key'], fact['statement_sha256']))
                for row in admitted_matches:
                    require(row['form'] == 'affirmative', 'unresolved affirmative bound to negation')
                    affected.update((row['source_row_id'], row['paired_source_row_id']))
            facts.append(dict(fact, entity_ids=review['entity_ids'],
                              matched_source_row_ids=[r['source_row_id'] for r in matches],
                              admitted_source_row_ids=[r['source_row_id'] for r in admitted_matches],
                              action='quarantine_exact_source_pair' if fact['judgment'] == 'unresolved' and admitted_matches
                              else 'generated_unresolved_not_usable' if fact['judgment'] == 'unresolved'
                              else 'preserve_existing_judgment'))
        audit.append(dict(topic=review['topic'], rank=review['rank'], person_key=review['person_key'],
                          entities=review['entities'], status=review['status'], facts=facts))
    return audit, affected, unresolved


def validate_current(rows):
    old.validate_source_pairs([{**r, 'successor_status': r[STATUS]} for r in rows])


def assert_no_unresolved(memberships, affected, unresolved):
    """Enforce the policy at every current source-row fitting boundary, including negations."""
    for name, rows in memberships.items():
        for row in rows:
            require(row['source_row_id'] not in affected and signature(row) not in unresolved,
                    f'unresolved source pair survives in current fitting membership: {name}/{row["source_row_id"]}')
            require(row[STATUS] == 'admitted', f'nonadmitted current fitting row: {name}')
        validate_current(rows)


def assert_fact_eligibility(records, affected, unresolved, source):
    for row in records:
        blocked = row['fact_ref'] in affected or signature(row) in unresolved
        matched = [s for s in source if signature(s) == signature(row)]
        blocked |= bool(matched) and not any(s[STATUS] == 'admitted' for s in matched)
        require(not blocked or row['current_eligible'] == 'False',
                'unresolved/restricted fact survives current fact eligibility')


def balanced(rows):
    frame = pd.DataFrame(rows)
    for column in ('label', 'row_index'):
        frame[column] = frame[column].astype(int)
    return [rows[int(i)] for i in balanced_burger_indices(frame, seed=0)]


def build(root=ROOT):
    root = Path(root)
    lock = load(root, PACKAGE / 'input_lock.json')
    for path, expected in lock['historical_sha256'].items():
        require(digest((root / path).read_bytes()) == expected, f'historical artifact changed: {path}')
    rows = read_csv(root, old.OVERLAY)
    reviews = load(root, old.PACKAGE / 'reviews.json')
    proposals = load(root, old.PACKAGE / 'negative_proposals.json')
    order = load(root, old.PACKAGE / 'candidate_order.json')
    recovered = load(root, old.PACKAGE / 'tc_recovered_pairs.json')
    old.validate_source_pairs(rows)
    # Validate against the historical base only; never regenerate its negative queue on v3.
    old.validate_pairs(reviews + recovered, order, rows, load(root, old.PACKAGE / 'evidence.json'), proposals, root)
    old.validate_attempt_sequence(reviews, order)
    audit, affected, unresolved = inspect_outcomes(rows, reviews, proposals)
    refs = {sid: f'reviews.json/topic={r["topic"]}/rank={r["rank"]}/statement_sha256={f["statement_sha256"]}'
            for r in audit for f in r['facts'] if f['action'] == 'quarantine_exact_source_pair'
            for sid in f['admitted_source_row_ids']}
    successor = []
    for row in rows:
        ref = refs.get(row['source_row_id'], refs.get(row['paired_source_row_id'], ''))
        successor.append(dict(row, **{STATUS: 'quarantined' if row['source_row_id'] in affected else row['successor_status']},
                              audit_correction_ref=ref,
                              audit_correction_reason='unresolved_exact_affirmative_and_paired_negation' if ref else ''))
    validate_current(successor)
    for before, after in zip(rows, successor):
        require(all(after[k] == v for k, v in before.items()), 'historical overlay column changed')
        require((after[STATUS] != before['successor_status']) == (before['source_row_id'] in affected),
                'unexpected claim/person restriction')
    outputs = {}

    def put_csv(path, data, fields=None):
        outputs[path] = csv_data(data, fields or list(data[0]))

    def put_json(path, data):
        outputs[path] = encoded(data)

    put_json(PACKAGE / 'review_outcome_audit.json', audit)
    put_csv(OVERLAY / 'row_manifest.csv', successor)
    for status in ('admitted', 'excluded', 'quarantined'):
        put_csv(OVERLAY / (status + '_rows.csv'), [r for r in successor if r[STATUS] == status])
    changed = [r for r in successor if r['source_row_id'] in affected]
    put_csv(OVERLAY / 'status_changes.csv', changed)
    row_eligibility = [dict(source_row_id=r['source_row_id'], paired_source_row_id=r['paired_source_row_id'],
                            topic=r['topic'], person_key=r['person_key'], entity_id=r['entity_id'],
                            split=r['split'], statement_sha256=r['statement_sha256'], label=r['label'], form=r['form'],
                            current_status=r[STATUS], current_atomic_eligible=str(r[STATUS] == 'admitted'),
                            correction_ref=r['audit_correction_ref']) for r in successor]
    put_csv(OVERLAY / 'source_fact_eligibility.csv', row_eligibility)
    # Historical inventory includes both source affirmatives and exact registry duplicates.
    inventory = read_csv(root, BASE / 'candidate_overlay_v1/fact_inventory.csv')
    projected = []
    for fact in inventory:
        matches = [r for r in successor if signature(r) == signature(fact)]
        is_unresolved = signature(fact) in unresolved
        restricted = bool(matches) and not any(r[STATUS] == 'admitted' for r in matches)
        eligible = fact['eligible'] == 'True' and not is_unresolved and not restricted
        projected.append(dict(fact, current_eligible=str(eligible),
                              current_source_status=matches[0][STATUS] if matches else 'not_a_source_row',
                              current_judgment='unresolved' if is_unresolved else 'historical_evidence_tier_retained',
                              current_reason='unresolved_exact_claim' if is_unresolved else
                              'restricted_exact_source_claim' if restricted else 'historical_eligibility_retained'))
    assert_fact_eligibility(projected, affected, unresolved, successor)
    put_csv(OVERLAY / 'fact_inventory.csv', projected)
    # Every reviewed fact also has explicit eligibility for the audited-pair use case.
    audited_eligibility = []
    for review in reviews + recovered:
        for fact in review['facts']:
            matched = [r for r in successor if signature(r) ==
                       (review['topic'], review['person_key'], fact['statement_sha256'])]
            eligible = fact['judgment'] == ('supported_true' if fact['label'] else 'supported_false')
            eligible &= not matched or any(r[STATUS] == 'admitted' for r in matched)
            audited_eligibility.append(dict(fact_ref=fact.get('source_row_id', fact.get('fact_id')),
                                            topic=review['topic'], person_key=review['person_key'],
                                            statement_sha256=fact['statement_sha256'], judgment=fact['judgment'],
                                            origin=fact['origin'], current_eligible=str(eligible)))
    assert_fact_eligibility(audited_eligibility, affected, unresolved, successor)
    put_csv(PACKAGE / 'audited_fact_eligibility.csv', audited_eligibility)
    # Preserve the reference, frozen order, judgments, negative queue, A and completed T_C bytes.
    preserved = ('candidate_order.json', 'reviews.json', 'negative_proposals.json', 'negative_pipeline_provenance.json',
                 'A15_proposal.json', 'A10_proposal.json', 'TC_completed_pairs.json', 'outer_training_reference.csv')
    preservation = {n: digest((root / old.PACKAGE / n).read_bytes()) for n in preserved}
    for name in ('A15_proposal.json', 'A10_proposal.json', 'TC_completed_pairs.json'):
        outputs[PACKAGE / name] = (root / old.PACKAGE / name).read_bytes()
        pairs = load(root, old.PACKAGE / name)
        if isinstance(pairs, dict):
            pairs = pairs['completed_pairs']
        for review in pairs:
            for fact in review['facts']:
                record = next(r for r in audited_eligibility if r['person_key'] == review['person_key']
                              and r['statement_sha256'] == fact['statement_sha256'])
                require(record['current_eligible'] == 'True', 'frozen A/T_C fact became ineligible; no replacement permitted')
    memberships, counts, deltas = {}, [], {}
    train = [r for r in successor if r['split'] == 'train' and r[STATUS] == 'admitted']
    memberships['corrected_outer_training'] = train
    memberships['corrected_outer_training_balanced'] = balanced(train)
    keys_by_size = {}
    for n in (15, 10):
        pairs = load(root, old.PACKAGE / f'A{n}_proposal.json')['completed_pairs']
        keys = keys_by_size[n] = {r['person_key'] for r in pairs}
        require(not keys & {r['person_key'] for r in successor if r['split'] != 'train'}, 'outer partition leak')
        p = [r for r in train if r['person_key'] not in keys]
        memberships[f'P{n}_admitted_membership'] = p
        sampled = memberships[f'P{n}_balanced_exposure'] = balanced(p)
        put_csv(PACKAGE / f'P{n}_excluded_source_variants.csv',
                [r for r in successor if r['split'] == 'train' and r['person_key'] in keys])
        before = read_csv(root, old.PACKAGE / f'P{n}_balanced_exposure.csv')
        before_ids = {r['source_row_id'] for r in before}
        after_ids = {r['source_row_id'] for r in sampled}
        deltas[str(n)] = dict(removed=sorted(before_ids - after_ids), added=sorted(after_ids - before_ids),
                             previous_sha256=digest((root / old.PACKAGE / f'P{n}_balanced_exposure.csv').read_bytes()),
                             current_sha256=digest(csv_data(sampled, list(successor[0]))))
    require(keys_by_size[10] <= keys_by_size[15], 'fallback no longer nested')
    assert_no_unresolved(memberships, affected, unresolved)
    for name, selected in memberships.items():
        put_csv(PACKAGE / (name + '.csv'), selected)
        counts += old.count_rows(selected, name)
    put_csv(PACKAGE / 'exposure_counts.csv', counts)
    put_json(PACKAGE / 'balanced_membership_changes.json', deltas)
    queue = read_csv(root, old.PACKAGE / 'TC_topup_review_queue.csv')
    for q in queue:
        # Preserve proposed text/provenance and rank; annotate, do not replace the candidate.
        q['current_negative_eligible'] = 'False'
        q['current_negative_status'] = ('quarantined_source_claim_requires_resolution' if q['negative_provenance'] in affected
                                        else 'unresolved' if q['state'] == 'unresolved' else 'not_adjudicated')
    put_csv(PACKAGE / 'TC_topup_review_queue.csv', queue)
    facts = [f for r in audit for f in r['facts']]
    summary = dict(status='successor_capacity_projection_not_final_PA', correction_base=str(OVERLAY),
                   reviewed_outcomes=len(audit), reviewed_facts=len(facts),
                   unresolved_source_facts=[f['source_row_id'] for f in facts if f['origin'] == 'source' and f['judgment'] == 'unresolved'],
                   unresolved_generated_candidates=[f['fact_id'] for f in facts if f['origin'] != 'source' and f['judgment'] == 'unresolved'],
                   changed_source_ids=sorted(affected),
                   admitted_by_split=dict(Counter(r['split'] for r in successor if r[STATUS] == 'admitted')),
                   current_membership_totals={n: len(v) for n, v in memberships.items()},
                   A_membership_unchanged=True, A15_keys=len(keys_by_size[15]), A10_keys=len(keys_by_size[10]),
                   preserved_historical_hashes=preservation, exposure_counts=counts,
                   sampler=dict(function='balanced_burger_indices', seed=0, numpy=np.__version__, pandas=pd.__version__),
                   confirmed_new_label_errors=0, correction_base_must_advance_before_fitting=True,
                   recommendation='Use candidate_overlay_v3 and this successor projection before any future authorized fitting. '
                   'Unresolved source claims and paired negations are quarantined, not relabeled or confirmed wrong. '
                   'Historical v1 projections remain references only. P/A stays unfrozen; no fitting authorization.',
                   status_column=STATUS, negative_queue_regenerated=False)
    put_json(PACKAGE / 'recommendation.json', summary)
    put_json(OVERLAY / 'correction_receipt.json', dict(base=str(old.OVERLAY), status_column=STATUS,
             judgment_source=str(old.PACKAGE / 'reviews.json'), affected_source_ids=sorted(affected),
             confirmed_label_errors=0, correction_base_must_advance_before_fitting=True,
             admitted_by_split=summary['admitted_by_split'],
             outputs={str(p): digest(v) for p, v in outputs.items() if p.parent == OVERLAY}))
    put_json(PACKAGE / 'projection_receipt.json', dict(
        input_lock_sha256=digest((root / PACKAGE / 'input_lock.json').read_bytes()),
        code_sha256=digest((root / 'src/capacity_pilot_projection.py').read_bytes()),
        outputs={str(p): digest(v) for p, v in outputs.items()},
        checks=['all historical bytes', 'all review identity/hash bindings', 'exact-claim paired quarantine',
                'current fitting membership exclusion', 'registry/source/audited fact eligibility propagation',
                'unchanged A15/A10/T_C facts', 'nested proposals', 'deterministic actual sampler',
                'no candidate/negative queue regeneration']))
    return outputs, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true')
    args = parser.parse_args()
    outputs, summary = build()
    for path, value in outputs.items():
        target = ROOT / path
        require(not target.exists() or target.read_bytes() == value, f'differing successor output: {path}')
        require(args.write or target.exists(), f'missing successor output: {path}')
    if args.write:
        for path, value in outputs.items():
            target = ROOT / path
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.write_bytes(value)
    print(json.dumps({k: summary[k] for k in ('changed_source_ids', 'admitted_by_split', 'current_membership_totals')}, indent=2))


if __name__ == '__main__':
    main()
