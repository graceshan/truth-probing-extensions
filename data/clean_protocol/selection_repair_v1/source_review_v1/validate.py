#!/usr/bin/env python3
"""Validate the separate T2B review package; no models, predictions or writes."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[3]
QUEUE = 'data/clean_protocol/selection_repair_v1/source_audit_v1/review_queue_30.csv'
ROWS = 'data/clean_protocol/selection_repair_v1/source_audit_v1/row_manifest.csv'
STATUSES = {'supported_false','supported_true','unresolved'}
IDENTITIES = ('draw_index','source_row_id','paired_source_row_id','dataset','row_index',
              'source_row','source_file','source_sha256','source_row_sha256',
              'statement','statement_sha256','split','partition_role','label',
              'entity','entity_id','person_key')
METHOD = 'Codex-assisted external review; not human verification'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(payload):
    return hashlib.sha256(payload).hexdigest()


def load(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def check(root=ROOT, package=PACKAGE, check_hashes=True):
    """check_hashes=False is for semantic mutation tests, never the CLI."""
    queue=csv_rows(root/QUEUE)
    rows={r['source_row_id']:r for r in csv_rows(root/ROWS)}
    reviews=load(package/'reviews.json')
    evidence=load(package/'evidence.json')
    supplementary=load(package/'supplementary_reviews.json')
    exclusions=csv_rows(package/'recommended_exclusions.csv')
    rules=load(package/'proposed_rule_changes.json')
    require(len(queue)==len(reviews)==30, 'Exactly 30 frozen reviews required')
    require([r['draw_index'] for r in queue]==[str(i) for i in range(30)], 'Frozen draw order invalid')
    require(len({r['source_row_id'] for r in reviews})==30, 'Duplicate sample review')
    for original,r in zip(queue,reviews):
        require(all(r.get(k)==original[k] for k in IDENTITIES), 'Frozen identity/order mismatch')
        require(r['label']=='0' and r['statement_sha256']==sha(r['statement'].encode()), 'Exact statement/label mismatch')
        require(r['original_queue_review_status']==original['review_status']=='pending', 'Original pending status changed')
        require(r['judgment'] in STATUSES, 'Invalid judgment')
        require(r['reviewer_method']==METHOD and r['review_complete'] is True, 'Reviewer or completion misrepresented')
        require(r.get('reasoning','').strip() and r.get('limitations','').strip() and r.get('review_date'), 'Missing reasoning or limitations')
        require(r.get('evidence_ids') and len(set(r['evidence_ids']))==len(r['evidence_ids']), 'Missing/duplicate evidence references')
        require(all(e in evidence for e in r['evidence_ids']), 'Unknown evidence reference')
        expected='retain_provisionally' if r['judgment']=='supported_false' else 'exclude_affirmative_and_paired_negation_pending_adjudication'
        require(r['recommendation']==expected, 'Recommendation inconsistent with judgment')
        if r['judgment']!='unresolved':
            require(any(evidence[e]['authority'] not in ('secondary encyclopedia','low-confidence secondary lead only') for e in r['evidence_ids']), 'Resolved judgment lacks authoritative evidence')
    for key,e in evidence.items():
        require(e.get('id')==key and e.get('url','').startswith('https://'), 'Invalid evidence URL/ID')
        require(all(e.get(k) for k in ('title','retrieval_date','authority','inspection_method','paraphrase','limitations')), 'Missing evidence context')
        require(len(e.get('response_sha256',''))==64 and all(c in '0123456789abcdef' for c in e['response_sha256']), 'Missing response hash')
        require(e.get('observations'), 'Missing inspected evidence text')
        require(all(o.get('locator') and o.get('excerpt','').strip() for o in e['observations']), 'Missing excerpt/locator')
        require(sum(len(o['excerpt'].split()) for o in e['observations'])<=250,'Excessive source quotation')
    require(supplementary['sample_denominator_contribution']==0, 'Supplementary review entered sample')
    require(supplementary['reviewer_method']==METHOD, 'Supplementary reviewer misrepresented')
    franklin=supplementary['franklin_france']
    registry_path=root/franklin['historical_registry']
    require(sha(registry_path.read_bytes())==franklin['historical_registry_sha256'], 'Historical registry changed')
    historical=franklin['historical_record_unchanged']
    records=[r for r in csv_rows(registry_path) if r['fact_id']==historical['fact_id']]
    require(records==[historical] and historical['validation_status']=='rejected_true_or_ambiguous', 'Historical judgment changed')
    require(franklin['new_judgment']=='supported_true' and franklin['evidence_ids']==['franklin'],'Franklin adjudication missing')
    require(len(supplementary['alias_corroborations'])==2, 'Alias corroborations incomplete')
    for record in [franklin]+supplementary['alias_corroborations']:
        require(record['reasoning'] and record['evidence_ids'] and all(e in evidence for e in record['evidence_ids']), 'Missing supplementary evidence')
    expected_pairs=[]
    for r in reviews:
        if r['judgment']!='supported_false':
            for row_id in (r['source_row_id'],r['paired_source_row_id']):
                expected_pairs.append((row_id,r))
    require(len(exclusions)==len(expected_pairs) and len({r['source_row_id'] for r in exclusions})==len(exclusions), 'Exclusion coverage mismatch')
    for ex,(row_id,review) in zip(exclusions,expected_pairs):
        original=rows[row_id]
        require(ex['source_row_id']==row_id and all(ex[k]==original[k] for k in ('paired_source_row_id','source_sha256','statement_sha256','split','label','entity_id','person_key')), 'Exclusion identity mismatch')
        require(ex['sample_draw_index']==review['draw_index'] and ex['sample_judgment']==review['judgment'],'Exclusion provenance mismatch')
        require(ex['action']=='recommended_exclusion_only_not_applied','Applied exclusion not authorized')
        pair=rows[original['paired_source_row_id']]
        require(pair['source_row_id'] in {r['source_row_id'] for r in exclusions} and int(pair['label'])==1-int(original['label']), 'Paired negation missing')
    require(rules['status']=='proposed_only_requires_review' and rules['score_blind'] is True,'Rule applied prematurely')
    require(len(rules['rules'])==3 and all(r.get('trigger') and r.get('action') for r in rules['rules']),'Rule scope incomplete')
    counts=dict(Counter(r['judgment'] for r in reviews))
    if check_hashes:
        manifest=load(package/'package_manifest.json')
        require(manifest['sample_size']==30 and manifest['counts']==counts, 'Manifest counts mismatch')
        require(manifest['t2a_status']=='provisional_unchanged','T2A promoted prematurely')
        require(manifest['confirmed_errors_are_not_total_error_rate'] is True,'Unresolved treated as correct')
        for item in manifest['inputs']:
            p=root/item['path']
            require(sha(p.read_bytes())==item['sha256'] and p.stat().st_size==item['bytes'],'Frozen input changed: '+item['path'])
        require(set(manifest['files'])=={p.name for p in package.iterdir() if p.is_file()}-{'package_manifest.json'},'Package file coverage changed')
        for name,receipt in manifest['files'].items():
            p=package/name
            require(sha(p.read_bytes())==receipt['sha256'] and p.stat().st_size==receipt['bytes'],'Package bytes changed: '+name)
    return {'sample_size':30,'counts':counts,'recommended_exclusion_rows':len(exclusions),'t2a_status':'provisional_unchanged'}


if __name__=='__main__':
    import sys
    if len(sys.argv)!=1:
        raise SystemExit('No CLI overrides: validate the frozen package and inputs.')
    print(json.dumps(check(),indent=2))
