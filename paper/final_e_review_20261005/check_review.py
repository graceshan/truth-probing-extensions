#!/usr/bin/env python3
"""Read-only paper provenance/content checks; no scientific runtime imports or calculations.

Run from this worktree with Python 3.11 -B. Writes only the explicitly named review
receipt. PDF text must first be extracted with pdf_text.swift; rebuild is separate.
"""
import collections
import csv
import fcntl
import hashlib
import html
import json
import os
import pathlib
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
REVIEW = pathlib.Path(__file__).resolve().parent
REPO = REVIEW.parents[1]
PAPER = REPO / 'paper/final_e_20261005'
DURABLE = pathlib.Path('/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1')
SCIENCE = DURABLE / 'scientific-evaluation-20261005-v1'
CANDIDATE = 'c30846d90e0695f1f55f29f2bb92d5a1a754ad8a'
AUDIT = '3dcd696ad17358ae8a89e458cb79a86a98792291'
INVENTORY = '4646dcc19e0eeb90aee535addff3b0b7a6f952b19aadec059d323a3f0176e9b8'
COMPLETION = '632ed93f295027701e1d798a94c758e332c747e071aee410ebb6972114d408bc'
LOCK = '88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e'
RECON = '03c451b2bb6c013a442f833923b8960d2b64cdef57f0015ec3bfa21c60591fb4'
COMPANION = 'bf7f8abe6d1dbf67209f4f2b296c2f367742d4f00b5cd7d5e5a4d6e2ce87b60e'

def read(p):
    return json.loads(p.read_text())

def pin(p):
    b = p.read_bytes()
    return {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}

def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args])

def main():
    assert pin(PAPER / 'inventory.json')['sha256'] == INVENTORY
    inventory = read(PAPER / 'inventory.json')['files']
    observed = {str(p.relative_to(PAPER)): pin(p) for p in PAPER.rglob('*')
                if p.is_file() and p.name != 'inventory.json'}
    assert inventory == observed
    tracked = git('ls-tree', '-r', '--name-only', CANDIDATE, '--', 'paper/final_e_20261005').decode().splitlines()
    assert len(tracked) == len(observed) + 1
    for p in tracked:
        assert (REPO / p).read_bytes() == git('show', CANDIDATE + ':' + p)
    assert not git('diff', CANDIDATE, '--', 'paper/final_e_20261005').strip()
    receipts = {}
    for p, expected in [
        (SCIENCE / 'evaluation/completion.json', COMPLETION),
        (SCIENCE / 'audit-reconstruction/completion.json', RECON),
        (SCIENCE / 'operation-receipts/audit.audit.json', COMPANION),
        (DURABLE / 'transport/final_lock_20261004.json', LOCK),
        (REPO / 'config/final_e/final_lock_20261004.json', LOCK),
        (REPO / 'results/final_e_scientific_audit_20261005/audit-companion.json', COMPANION),
        (REPO / 'results/final_e_scientific_audit_20261005/reconstruction-completion.json', RECON)]:
        receipts[str(p)] = pin(p)
        assert receipts[str(p)]['sha256'] == expected
    # Acceptance is inherited from these exact published bytes, not rerun here.
    audit_dir = REPO / 'results/final_e_scientific_audit_20261005'
    for p in [REPO / 'docs/final_e/SCIENTIFIC_AUDIT_20261005.md',
              *[audit_dir / n for n in ['audit-handoff.json', 'audit-companion.json',
                                       'audit-operation.json', 'audit-process.json',
                                       'scope-coverage.json', 'principal-results.csv']]]:
        assert p.read_bytes() == git('show', AUDIT + ':' + str(p.relative_to(REPO)))
    handoff = read(audit_dir / 'audit-handoff.json')
    assert handoff['scientific_acceptance'] == 'accepted_under_the_unchanged_frozen_lock'
    for field in ['source_completion', 'reconstruction_completion', 'audit_operation',
                  'audit_companion', 'authorization', 'lock']:
        r = handoff[field]
        assert pin(pathlib.Path(r['path'])) == {k: r[k] for k in ['bytes', 'sha256']}
    lock = read(REPO / 'config/final_e/final_lock_20261004.json')
    prep = read(REPO / 'config/final_e/preparation_v1.json')
    configs = read(PAPER / 'protocol-source-pins.json')
    for p, expected in configs.items():
        assert pin(REPO / p)['sha256'] == expected
        assert expected == lock['immutable_contract'].get(p, prep['scientific_sha256'].get(p))
    thresholds = read(PAPER / 'evidence/threshold-sources.json')
    for name in ['practical_gain', 'atomic_loss_margin', 'AND_loss_margin', 'constituent_transfer_margin']:
        t = thresholds[name]
        source = read(REPO / t.get('source', thresholds['source']))
        for key in t['json_pointer'].strip('/').split('/'):
            source = source[key]
        assert source == t['value']
    completion = read(SCIENCE / 'evaluation/completion.json')
    verification = read(PAPER / 'evidence/source-verification.json')
    sources, source_pins = {}, {}
    for name, description in verification['sources'].items():
        p = SCIENCE / 'evaluation' / (name + '.json')
        assert p.resolve() == pathlib.Path(description['path']).resolve()
        source_pins[name] = pin(p)
        assert source_pins[name] == completion['files'][p.name]
        assert source_pins[name] == {k: description[k] for k in ['bytes', 'sha256']}
        rows = read(p)['records']
        sources[p.name] = rows
        keys = description['identity_fields']
        assert len({tuple(r[k] for k in keys) for r in rows}) == len(rows)
    extracted = read(PAPER / 'evidence/extracted-rows.json')
    for rid, e in extracted.items():
        row = sources[e['source_file']][e['record_index_zero_based']]
        keys = verification['sources'][e['source_file'][:-5]]['identity_fields']
        assert set(e['identity']) == set(keys)
        assert e['identity'] == {k: row[k] for k in keys}
        assert e['row'] == row
    # Independently match the two principal CSVs by science identity, ignoring role labels.
    principal = []
    arows = list(csv.DictReader((audit_dir / 'principal-results.csv').open()))
    erows = list(csv.DictReader((REPO / 'results/final_e_scientific_evaluation_20261005/principal-results.csv').open()))
    keys = verification['sources']['scalar-contrasts']['identity_fields']
    index = {tuple(str(r[k]) for k in keys): (i, r) for i, r in enumerate(sources['scalar-contrasts.json'])}
    for a in arows:
        # Audit CSV omits topic because its principal rows are all pooled/macro.
        assert a['scope'] in ['pooled', 'topic_macro']
        identity = tuple('all' if k == 'topic' else a[k] for k in keys)
        i, r = index[identity]
        matches = [e for e in erows if tuple(e[k] for k in keys) == identity]
        assert len(matches) == 1
        for k in ['point', 'ci_low', 'ci_high', 'valid_replicates', 'invalid_replicates']:
            assert float(a[k]) == r[k] == float(matches[0][k])
        principal.append({'source_row': f'scalar-contrasts:{i}', 'identity': dict(zip(keys, identity)),
                          'audit_role': a['role'], 'evaluation_role': matches[0]['role']})
    # Catalog partition check includes membership, scope/order/topic/sample sets and adverse counts.
    catalog = read(PAPER / 'evidence/secondary-catalog.json')
    for filename, rows in sources.items():
        groups = [g for g in catalog if g['source_file'] == filename]
        fields = sorted(groups[0]['identity_group'])
        buckets = collections.defaultdict(list)
        for row in rows:
            buckets[tuple(row[k] for k in fields)].append(row)
        assert len(groups) == len(buckets)
        seen = set()
        for g in groups:
            key = tuple(g['identity_group'][k] for k in fields)
            assert key not in seen
            seen.add(key)
            rr = buckets[key]
            assert len(rr) == g['row_count']
            assert g['source_sha256'] == source_pins[filename[:-5]]['sha256']
            for column, field in [('samples', 'sample'), ('scopes', 'scope'), ('topics', 'topic'), ('orders', 'surface_order')]:
                assert sorted({str(r.get(field, 'not_applicable')) for r in rr}) == g[column]
            assert g['negative_point_rows'] == sum(r['point'] is not None and r['point'] < 0 for r in rr)
            assert g['classifications'] == dict(collections.Counter(r.get('classification', r.get('absolute_classification', 'not_applicable')) for r in rr))
    # Compare each serialized CSV row, in order, with actual PDF text, SVG comments and LaTeX.
    pdfs = read(REVIEW / 'evidence/pdf-text.json')
    rendered_rows = 0
    for name in ['main', 'retention', 'secondary', 'wording', 'constituent']:
        rows = list(csv.DictReader((PAPER / f'tables/{name}.csv').open()))
        tex = (PAPER / f'tables/{name}.tex').read_text()
        pdftext = ' '.join(pdfs[f'tables/{name}.pdf']['text'].split())
        svg = (PAPER / f'tables/{name}.svg').read_text()
        comments = [html.unescape(c).strip() for c in re.findall(r'<!--(.*?)-->', svg, re.S)]
        position = 0
        # PDFKit returns the model column after the body for these two PDFs.
        # Check its sequence separately; spatial association is visually reviewed.
        if name in ['wording', 'constituent']:
            assert ' '.join(r['Model'] for r in rows) in pdftext
        for row in rows:
            text = ' '.join(v for k, v in row.items() if name not in ['wording', 'constituent'] or k != 'Model')
            position = pdftext.index(text, position) + len(text)
            for value in row.values():
                assert value in comments, (name, value)
                escaped = value.replace('_', r'\_').replace('<', r'\textless{}').replace('>', r'\textgreater{}')
                assert escaped in tex, (name, escaped)
            rendered_rows += 1
    svg_pins = {}
    for p in PAPER.glob('*/*.svg'):
        root = ET.fromstring(p.read_bytes())
        ids = {e.attrib['id'] for e in root.iter() if 'id' in e.attrib}
        for e in root.iter():
            for k, v in e.attrib.items():
                if k.endswith('href'):
                    assert v.startswith('#') and v[1:] in ids
                for target in re.findall(r'url\(#(.*?)\)', v):
                    assert target in ids
        svg_pins[str(p.relative_to(PAPER))] = pin(p)
    # Claim predicates inspect already computed rows; no estimate or interval is recomputed.
    claims = {c['claim_id']: c for c in read(PAPER / 'claim-evidence.json')}
    cr = lambda cid: [extracted[r]['row'] for r in claims[cid]['supporting_rows']]
    assert all(r['ci_low'] > 0 for r in cr('C1') if 'left' in r)
    assert all(r['ci_low'] > .02 for r in cr('C2'))
    assert all(r['ci_low'] > 0 and r['scope'] == 'topic_macro' for r in cr('C3'))
    assert len(cr('C4')) == 8 and all(r['ci_low'] >= -r['retention_margin'] and r['interval_supported'] for r in cr('C4'))
    assert all(r['point'] < 0 and r['ci_high'] < 0 for r in cr('C5'))
    assert all(r['ci_low'] > (.02 if r['scope'] == 'pooled' else 0) for r in cr('C6'))
    assert all(r['ci_low'] > 0 for cid in ['C7', 'C8'] for r in cr(cid))
    assert all(r['point'] < 0 for r in cr('C10'))
    assert sum(r['ci_high'] < 0 for r in cr('C10')) == 11
    crossing = [r for r in cr('C10') if r['ci_low'] < 0 < r['ci_high']]
    assert len(crossing) == 1 and 'llama/B25_37/' in crossing[0]['head'] and crossing[0]['condition'] == 'or_at_least_one_v1'
    assert collections.Counter(r['classification'] for r in cr('C11')) == {
        'material_deficit_lower_bound_above_05': 7, 'unresolved_crosses_05': 2, 'upper_bound_below_05': 3}
    # External links resolve to all five pinned complete tables on this shared Mac.
    links = []
    for p in PAPER.glob('*.md'):
        for target in re.findall(r'\]\(([^)]+)\)', p.read_text()):
            if '://' in target or target.startswith('#'):
                continue
            assert (p.parent / target.split('#')[0]).exists(), (p.name, target)
            links.append({'file': p.name, 'target': target})
    rebuilt = pathlib.Path('/tmp/final-e-mac2-paper-review-render')
    rebuild_files = {}
    for p in rebuilt.rglob('*'):
        if p.is_file():
            relative = str(p.relative_to(rebuilt))
            assert p.read_bytes() == (PAPER / relative).read_bytes(), relative
            rebuild_files[relative] = pin(p)
    assert len(rebuild_files) == 45
    report = {'status': 'PASS', 'candidate_commit': CANDIDATE, 'candidate_inventory_sha256': INVENTORY,
              'candidate_files_checked_against_git': len(tracked), 'inventory_files': len(inventory),
              'inventory_bytes_excluding_inventory': sum(p['bytes'] for p in inventory.values()),
              'receipt_pins': receipts, 'protocol_pins': configs, 'source_pins': source_pins,
              'source_rows': {n: len(r) for n, r in sources.items()}, 'extracted_rows': len(extracted),
              'catalog_groups': len(catalog), 'catalog_partition_and_fields': 'exact',
              'principal_identity_matches': principal, 'table_rows_compared_with_pdf_svg_tex': rendered_rows,
              'svg_internal_references': 'resolved; no external dependencies', 'svg_pins': svg_pins,
              'claims_checked': list(claims), 'links_checked': links, 'byte_identical_rebuild_files': rebuild_files,
              'scientific_recomputation': False, 'raw_or_array_files_opened': False}
    (REVIEW / 'evidence/independent-check.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: report[k] for k in ['status', 'candidate_files_checked_against_git', 'catalog_groups', 'extracted_rows', 'table_rows_compared_with_pdf_svg_tex']}))

if __name__ == '__main__':
    with open('/tmp/truth-probing-cpu-exclusive.lock', 'a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            sys.exit('CPU slot occupied; no review checks started')
        main()
