#!/usr/bin/env python3
"""Presentation verification only: source pins, copied rows, formatting and file structure."""
import argparse, collections, csv, fcntl, hashlib, json, pathlib, sys, xml.etree.ElementTree as ET
HERE=pathlib.Path(__file__).resolve().parent
EXPECTED='632ed93f295027701e1d798a94c758e332c747e071aee410ebb6972114d408bc'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--package',default=str(HERE));ap.add_argument('--receipt');args=ap.parse_args()
    root=pathlib.Path(args.package);v=json.loads((root/'evidence/source-verification.json').read_text())
    assert v['completion_sha256']==EXPECTED
    sources={}
    for name,pin in v['sources'].items():
        p=pathlib.Path(pin['path']); assert digest(p)==pin['sha256']
        comp=p.parent/'completion.json';assert digest(comp)==EXPECTED
        expected=json.loads(comp.read_text())['files'][p.name];assert digest(p)==expected['sha256'] and p.stat().st_size==expected['bytes']
        rows=json.loads(p.read_text())['records'];sources[p.name]=rows
        keys=[tuple(r[k] for k in pin['identity_fields']) for r in rows]; assert len(keys)==len(set(keys))
    extracted=json.loads((root/'evidence/extracted-rows.json').read_text())
    for rid,e in extracted.items():
        row=sources[e['source_file']][e['record_index_zero_based']]
        assert row==e['row'],rid
        assert all(row[k]==val for k,val in e['identity'].items()),rid
        assert e['source_sha256']==v['sources'][e['source_file'][:-5]]['sha256']
    cells=json.loads((root/'evidence/display-map.json').read_text())
    tables={n:list(csv.DictReader((root/'tables'/f'{n}.csv').open())) for n in ['main','retention','secondary','wording','constituent']}
    def cellrow(c,r):
        rows=tables[c['asset']];model=r.get('model',r.get('head','').split(':')[-1].split('/')[0]).title()
        scope='Pooled' if r['scope']=='pooled' else 'Five-topic macro'
        selected=[t for t in rows if t['Model']==model]
        if c['asset'] in ['main','retention','secondary']: selected=[t for t in selected if t['Scope']==scope]
        if c['asset']=='retention': selected=[t for t in selected if t['Endpoint']==r['metric'].replace('_auroc','')]
        if c['asset']=='secondary': selected=[t for t in selected if t['Comparison']==r['left']+' - '+r['right']]
        if c['asset']=='wording':
            label={'or_explicit_or_both_v1':'Or both','or_at_least_one_v1':'At least one'}[r['condition']]
            sample=r['head'].split('/')[1].split('_')[1]
            selected=[t for t in selected if t['Wording']==label and t['Sample']==sample]
        if c['asset']=='constituent': selected=[t for t in selected if t['Endpoint']==r['metric']]
        assert len(selected)==1,(c,selected)
        return selected[0]
    for c in cells:
        r=extracted[c['source_row']]['row']
        if 'display' in c:
            text=(f"{r['point']:.6f} [{r['ci_low']:.6f}, {r['ci_high']:.6f}]" if c['field']=='interval' else f"{r[c['field']]:.6f}")
            assert c['display']==text
            t=cellrow(c,r);assert t[c['column']]==text
            assert text in (root/'tables'/f"{c['asset']}.tex").read_text()
            assert text in (root/'tables'/f"{c['asset']}.md").read_text()
            if c['asset']=='retention':
                assert t['Loss boundary']==f"{-r['retention_margin']:.6f}" and r['interval_supported'] is True and t['Frozen decision']=='Supported'
            if c['asset']=='constituent':
                assert t['Frozen classification']=={'material_deficit_lower_bound_above_05':'Deficit > .05','unresolved_crosses_05':'Unresolved at .05','upper_bound_below_05':'Upper bound < .05'}[r['classification']]
        else: assert c['values']==[r[k] for k in c['fields']]
    catalog=json.loads((root/'evidence/secondary-catalog.json').read_text())
    for name,rows in sources.items():assert sum(g['row_count'] for g in catalog if g['source_file']==name)==len(rows)
    claims=json.loads((root/'claim-evidence.json').read_text());assert len(claims)==12
    for c in claims:
        assert c['supporting_rows'] and c['required_qualifications']
        assert all(r in extracted for r in c['supporting_rows'])
        assert all((root/p).exists() for p in c['assets'])
    assets=[]
    for folder,names in [('figures',['primary-gain','retention']),('tables',list(tables))]:
        for name in names:
            for ext in ['pdf','svg','png']:
                p=root/folder/(name+'.'+ext);b=p.read_bytes();assert len(b)>1000
                if ext=='pdf':assert b.startswith(b'%PDF') and b'%%EOF' in b[-100:]
                elif ext=='svg':assert ET.fromstring(b).tag.endswith('svg')
                else:assert b.startswith(b'\x89PNG\r\n\x1a\n')
                assets.append(str(p.relative_to(root)))
    result={'status':'passed','scope':'presentation/source-copy verification only; no scientific metrics, contrasts, fits or bootstrap calculations','source_tables':len(sources),'source_rows_catalogued':sum(map(len,sources.values())),'exact_extracted_rows_checked':len(extracted),'display_bindings_checked':len(cells),'table_rows_checked':sum(map(len,tables.values())),'claim_entries':len(claims),'vector_and_preview_files_checked':assets,'unresolved_issues':[]}
    if args.receipt:pathlib.Path(args.receipt).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':
    with open('/tmp/truth-probing-cpu-exclusive.lock','a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:sys.exit('CPU slot occupied; no checks started.')
        main()
