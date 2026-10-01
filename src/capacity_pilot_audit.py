"""Fact-only, deterministic prospective A/P capacity accounting. Never fits a model."""
import argparse
import csv
import hashlib
import io
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.atomic_probe_methods import balanced_burger_indices
from src.validated_negatives import initialize

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = Path('data/clean_protocol/selection_repair_v1/capacity_pilot_audit_v1')
OVERLAY = PACKAGE.parent / 'candidate_overlay_v2/row_manifest.csv'
TOPICS = ('animal_class', 'cities', 'element_symb', 'inventors', 'sp_en_trans')
INPUTS = ('pre_review_lock.json', 'pre_judgment_receipt.json', 'candidate_order.json',
          'negative_proposals.json', 'negative_pipeline_provenance.json', 'policy.json',
          'reviews.json', 'evidence.json', 'tc_recovered_pairs.json',
          'correction_proposals.json', 'retrieval_log.json')


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + '\n').encode()


def read_json(root, name):
    return json.loads((root / PACKAGE / name).read_text())


def csv_data(rows, fields):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def candidate_order(rows):
    parts, eligible = defaultdict(set), defaultdict(set)
    for row in rows:
        parts[row['person_key']].add(row['split'])
    for row in rows:
        if (row['split'] == 'train' and row['successor_status'] == 'admitted'
                and row['form'] == 'affirmative' and row['label'] == '1'
                and parts[row['person_key']] == {'train'}):
            eligible[row['topic']].add(row['person_key'])
    result = []
    for topic in sorted(eligible):
        keys = sorted(eligible[topic])
        random.Random(20261002).shuffle(keys)
        for rank, key in enumerate(keys, 1):
            selected = [r for r in rows if r['topic'] == topic and r['person_key'] == key
                        and r['split'] == 'train']
            result.append(dict(topic=topic, rank=rank, person_key=key,
                               entities=sorted({r['entity'] for r in selected}),
                               entity_ids=sorted({r['entity_id'] for r in selected}),
                               admitted_affirmatives=[dict(source_row_id=r['source_row_id'],
                                                           statement=r['statement'], label=int(r['label']))
                                                       for r in selected if r['successor_status'] == 'admitted'
                                                       and r['form'] == 'affirmative']))
    return result


def validate_pairs(reviews, order, rows, evidence, proposals, root):
    by_rank = {(o['topic'], o['rank']): o for o in order}
    source = {r['source_row_id']: r for r in rows}
    proposed = {(p['topic'], p['rank']): p['candidate'] for p in proposals}
    seen = set()
    for review in reviews:
        rk = review['topic'], review['rank']
        require(rk not in seen, 'duplicate review')
        seen.add(rk)
        original = by_rank[rk]
        require(all(review[k] == original[k] for k in ('person_key', 'entity_ids', 'entities')),
                'review identity mismatch')
        require(review['status'] in ('usable', 'rejected', 'unresolved'), 'unknown review status')
        require(len(review['facts']) == 2 and {f['label'] for f in review['facts']} == {0, 1},
                'two opposite affirmative facts required')
        for fact in review['facts']:
            require(fact['statement_sha256'] == digest(fact['statement'].encode()), 'statement hash mismatch')
            require(fact['reasoning'].strip() and fact['evidence_ids'], 'evidence coverage missing')
            for eid in fact['evidence_ids']:
                e = evidence[eid]
                require(e['url'] and e['title'] and len(e['response_sha256']) == 64
                        and e['observations'] and all(o['excerpt'] and o['locator'] for o in e['observations']),
                        'incomplete locatable evidence')
                binding = dict(topic=review['topic'], person_key=review['person_key'],
                               statement_sha256=fact['statement_sha256'], judgment=fact['judgment'])
                require(binding in e['fact_bindings'], 'evidence/fact binding mismatch')
                if 'reused_from' in e:
                    path = root / e['reused_from']
                    require(digest(path.read_bytes()) == e['package_sha256'], 'reused evidence changed')
                    old = json.loads(path.read_text())[e['original_evidence_id']]
                    require(all(e[k] == old[k] for k in ('url', 'response_sha256', 'observations')),
                            'reused evidence mismatch')
            if fact['origin'] == 'source':
                s = source[fact['source_row_id']]
                require(s['successor_status'] == 'admitted' and s['split'] == 'train'
                        and s['form'] == 'affirmative' and s['person_key'] == review['person_key']
                        and s['topic'] == review['topic'] and s['statement'] == fact['statement']
                        and int(s['label']) == fact['label'], 'source fact/entity binding mismatch')
                # Fixed source-fact rule: numerical source row index, not CSV lexical IDs.
                expected = min((a for a in original['admitted_affirmatives'] if a['label'] == fact['label']),
                               key=lambda a: int(a['source_row_id'].split(':')[1]))
                require(expected['source_row_id'] == fact['source_row_id'], 'source fact selection changed')
            else:
                p = proposed[rk]
                require(fact['origin'] == 'recovered_negative_pipeline' and fact['label'] == 0
                        and not any(a['label'] == 0 for a in original['admitted_affirmatives']),
                        'generated negative cannot replace an admitted source negative')
                require(all(fact[k] == p[k] for k in ('fact_id', 'statement', 'entity_id',
                                                     'candidate_rank', 'ranking_hash'))
                        and p['entity_id'] in review['entity_ids'] and p['split'] == 'train',
                        'pipeline fact/entity binding mismatch')
            require(fact['judgment'] in ('supported_true', 'supported_false', 'unresolved'), 'bad judgment')
        complete = all(f['judgment'] == ('supported_true' if f['label'] else 'supported_false')
                       for f in review['facts'])
        require((review['status'] == 'usable') == complete, 'usable pair lacks completed judgments')
        for ref in review.get('reused_judgments', []):
            path = root / ref['path']
            require(digest(path.read_bytes()) == ref['sha256'], 'reused judgment changed')
            old = next(r for r in json.loads(path.read_text()) if r['source_row_id'] == ref['source_row_id'])
            f = next(f for f in review['facts'] if f.get('source_row_id') == ref['source_row_id'])
            require(old['judgment'] == f['judgment'] and old['statement_sha256'] == f['statement_sha256']
                    and old['person_key'] == review['person_key'], 'reused judgment binding mismatch')


def validate_attempt_sequence(reviews, order):
    for topic in TOPICS:
        attempts = [r for r in reviews if r['topic'] == topic]
        require([r['rank'] for r in attempts] == list(range(1, len(attempts) + 1)), 'review order gap/redraw')
        usable = [r for r in attempts if r['status'] == 'usable']
        require(len(usable) <= 15, 'A exceeds pilot size')
        if len(usable) == 15:
            require(attempts[-1] == usable[-1], 'review continued after pilot completion')
        else:
            require(len(attempts) == sum(o['topic'] == topic for o in order), 'unreported unexhausted shortfall')


def validate_source_pairs(rows):
    lookup = {r['source_row_id']: r for r in rows}
    require(len(lookup) == len(rows), 'duplicate source identity')
    for r in rows:
        p = lookup[r['paired_source_row_id']]
        require(p['paired_source_row_id'] == r['source_row_id'] and p['form'] != r['form']
                and int(p['label']) == 1 - int(r['label'])
                and all(p[k] == r[k] for k in ('person_key', 'entity_id', 'topic', 'split', 'successor_status')),
                'paired source handling mismatch')


def count_rows(rows, scenario):
    result = []
    for topic in TOPICS:
        selected = [r for r in rows if r['topic'] == topic]
        result.append(dict(scenario=scenario, topic=topic, rows=len(selected),
                           label_0=sum(int(r['label']) == 0 for r in selected),
                           label_1=sum(int(r['label']) == 1 for r in selected),
                           affirmative=sum(r['form'] == 'affirmative' for r in selected),
                           negated=sum(r['form'] == 'negated' for r in selected),
                           entities=len({r['entity_id'] for r in selected}),
                           person_keys=len({r['person_key'] for r in selected})))
    return result


def build(root=ROOT):
    root = Path(root)
    lock = read_json(root, 'pre_review_lock.json')
    for p, expected in lock['historical_hashes'].items():
        require(digest((root / p).read_bytes()) == expected, f'historical input changed: {p}')
    pre = read_json(root, 'pre_judgment_receipt.json')
    for name, expected in pre['files'].items():
        require(digest((root / PACKAGE / name).read_bytes()) == expected, 'pre-judgment input changed')
    require(lock['candidate_order_sha256'] == digest((root / PACKAGE / 'candidate_order.json').read_bytes()),
            'pre-review order hash mismatch')
    rows = list(csv.DictReader((root / OVERLAY).open()))
    validate_source_pairs(rows)
    order = candidate_order(rows)
    require(order == read_json(root, 'candidate_order.json'), 'deterministic order mismatch')
    # Rebuild recovered queues without any generated compound or activation access.
    cfg = json.loads((root / 'config/clean_protocol/validated_negatives_inventor_single_country.json').read_text())
    _, registry, provenance = initialize(cfg, root, 'train')
    require(provenance == read_json(root, 'negative_pipeline_provenance.json'), 'pipeline provenance changed')
    proposals = read_json(root, 'negative_proposals.json')
    expected = []
    for o in order:
        if o['topic'] not in ('inventors', 'sp_en_trans') or any(a['label'] == 0 for a in o['admitted_affirmatives']):
            continue
        candidates = sorted((r for r in registry if r['entity_id'] in o['entity_ids']),
                            key=lambda r: (r['entity_id'], r['candidate_rank']))
        expected.append(dict(topic=o['topic'], rank=o['rank'], person_key=o['person_key'],
                             candidate=candidates[0] if candidates else None))
    require(proposals == expected, 'negative proposal ranking changed')
    reviews, recovered = read_json(root, 'reviews.json'), read_json(root, 'tc_recovered_pairs.json')
    evidence = read_json(root, 'evidence.json')
    validate_pairs(reviews + recovered, order, rows, evidence, proposals, root)
    validate_attempt_sequence(reviews, order)
    outputs, counts = {}, []
    outer = [r for r in rows if r['split'] == 'train']
    admitted = [r for r in outer if r['successor_status'] == 'admitted']
    fields = list(rows[0])
    outputs['outer_training_reference.csv'] = csv_data(outer, fields)
    outputs['v2_admitted_training_reference.csv'] = csv_data(admitted, fields)
    counts += count_rows(outer, 'original_outer_training') + count_rows(admitted, 'v2_admitted_training')
    memberships = {}
    sampler_seed = json.loads((root / 'config/clean_protocol/atomic_probe_methods.json').read_text())['burger_sampling_seed']
    require(sampler_seed == 0, 'sampler seed changed')
    for name, reference in [('original_outer_training', outer), ('v2_admitted_training', admitted)]:
        frame = pd.DataFrame(reference)
        frame['label'] = frame['label'].astype(int)
        frame['row_index'] = frame['row_index'].astype(int)
        sampled = [reference[int(i)] for i in balanced_burger_indices(frame, seed=sampler_seed)]
        validate_source_pairs(sampled)
        counts += count_rows(sampled, name + '_balanced_exposure')
        outputs[name + '_balanced_exposure.csv'] = csv_data(sampled, fields)
    for n in (15, 10):
        selected = [r for t in TOPICS for r in [v for v in reviews if v['topic'] == t and v['status'] == 'usable'][:n]]
        keys = {r['person_key'] for r in selected}
        memberships[n] = keys
        # All source variants are listed, even ones already excluded/quarantined by v2.
        removed = [r for r in outer if r['person_key'] in keys]
        raw_p = [r for r in outer if r['person_key'] not in keys]
        p = [r for r in admitted if r['person_key'] not in keys]
        require(not keys & {r['person_key'] for r in p}, 'P/A person overlap')
        require(not keys & {r['person_key'] for r in rows if r['split'] != 'train'}, 'outer partition identity overlap')
        validate_source_pairs(p)
        frame = pd.DataFrame(p)
        for column in ('label', 'row_index'):
            frame[column] = frame[column].astype(int)
        indices = balanced_burger_indices(frame, seed=sampler_seed)
        sampled = [p[int(i)] for i in indices]
        validate_source_pairs(sampled)
        counts += count_rows(removed, f'A{n}_all_source_variants')
        counts += count_rows(raw_p, f'P{n}_original_source_remainder')
        counts += count_rows(p, f'P{n}_admitted') + count_rows(sampled, f'P{n}_balanced_exposure')
        outputs[f'A{n}_proposal.json'] = encoded(dict(status='pilot_proposal_not_final_reservation',
                                                     requested_per_topic=n, completed_pairs=selected))
        for name, rr in [('excluded_source_variants', removed), ('original_source_remainder', raw_p),
                         ('admitted_membership', p), ('balanced_exposure', sampled)]:
            outputs[f'P{n}_{name}.csv'] = csv_data(rr, fields)
    require(memberships[10] <= memberships[15], 'fallback not nested')
    # Recovered Maxwell adds T_C capacity without changing sequential A selection.
    tc = [r for r in reviews + recovered if r['status'] == 'usable']
    require(len({r['person_key'] for r in tc}) == len(tc), 'duplicate T_C person')
    outputs['TC_completed_pairs.json'] = encoded(tc)
    tc_keys = {r['person_key'] for r in tc}
    queue = []
    for o in order:
        if o['person_key'] in tc_keys:
            continue
        attempted = next((r for r in reviews if r['person_key'] == o['person_key']), None)
        positives = [a for a in o['admitted_affirmatives'] if a['label'] == 1]
        negatives = [a for a in o['admitted_affirmatives'] if a['label'] == 0]
        positive = min(positives, key=lambda a: int(a['source_row_id'].split(':')[1]))
        negative = (min(negatives, key=lambda a: int(a['source_row_id'].split(':')[1])) if negatives
                    else next(p['candidate'] for p in proposals
                              if p['topic'] == o['topic'] and p['rank'] == o['rank']))
        queue.append(dict(topic=o['topic'], rank=o['rank'], person_key=o['person_key'],
                          entities=' | '.join(o['entities']),
                          positive_statement=positive['statement'],
                          negative_statement=negative['statement'] if negative else '',
                          negative_provenance=negative.get('source_row_id', negative.get('fact_id', '')) if negative else '',
                          state=attempted['status'] if attempted else 'not_adjudicated',
                          workload='negative follow-up; positive already supported' if attempted else 'complete positive and negative fact review',
                          role='T_C only; may overlap P/A; does not enlarge A',
                          preliminary_evidence='retrieval_log.json' if o['topic'] == 'inventors' and o['rank'] in (24, 26) else ''))
    outputs['TC_topup_review_queue.csv'] = csv_data(queue, list(queue[0]))
    outputs['exposure_counts.csv'] = csv_data(counts, list(counts[0]))
    summary = dict(status='pilot_proposal_not_final_reservation', reservation_seed=20261002,
                   balanced_sampler=dict(function='src.atomic_probe_methods.balanced_burger_indices', seed=0,
                                         numpy_version=np.__version__, pandas_version=pd.__version__),
                   topics={}, exposure_counts=counts, corrections=read_json(root, 'correction_proposals.json'))
    for t in TOPICS:
        rr = [r for r in reviews if r['topic'] == t]
        status = Counter(r['status'] for r in rr)
        complete_tc = sum(r['topic'] == t for r in tc)
        summary['topics'][t] = dict(eligible=sum(o['topic'] == t for o in order), attempted=len(rr),
                                    usable=status['usable'], rejected=status['rejected'], unresolved=status['unresolved'],
                                    A15_shortfall=max(0, 15-status['usable']), A10_shortfall=max(0, 10-status['usable']),
                                    TC_completed=complete_tc, TC_additional_usable_pairs_needed=max(0, 20-complete_tc),
                                    pilot_last_rank=rr[-1]['rank'])
    outputs['summary.json'] = encoded(summary)
    receipt = dict(status='fact_audit_validation_passed_not_production_approval',
                   inputs={n: digest((root / PACKAGE / n).read_bytes()) for n in INPUTS},
                   code={p: digest((root / p).read_bytes()) for p in
                         ('src/capacity_pilot_audit.py', 'src/atomic_probe_methods.py', 'src/validated_negatives.py')},
                   outputs={n: digest(v) for n, v in outputs.items()},
                   checks=['historical bytes', 'fixed seed/order/eligibility', 'pipeline ranks and provenance',
                           'exact fact/entity bindings', 'completed evidence coverage', 'paired source variants',
                           'person-level P/A and outer disjointness', 'nested fallback', 'actual balanced sampler',
                           'reused judgment bindings', 'T_C overlap without A expansion'])
    outputs['validation_receipt.json'] = encoded(receipt)
    return outputs, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Create missing outputs; never overwrite differing files')
    args = parser.parse_args()
    outputs, summary = build()
    for name, data in outputs.items():
        path = ROOT / PACKAGE / name
        require(not path.exists() or path.read_bytes() == data, f'differing existing output: {name}')
        require(args.write or path.exists(), f'missing output: {name}')
    if args.write:
        for name, data in outputs.items():
            path = ROOT / PACKAGE / name
            if not path.exists():
                path.write_bytes(data)
    print(json.dumps(summary['topics'], indent=2))


if __name__ == '__main__':
    main()
