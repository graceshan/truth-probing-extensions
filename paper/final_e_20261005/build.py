#!/usr/bin/env python3
"""Format pinned, already audited tables. No scientific computation or runtime imports."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/truth-probing-paper-mpl')
import argparse, collections, csv, fcntl, hashlib, json, pathlib, sys
HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]
PACKAGE = pathlib.Path('/Users/apple/truth-probing-backups/20261001T164224Z/final-e-launch-package-20261004-b2c6dde-88b114dd-v1')
COMPLETION = '632ed93f295027701e1d798a94c758e332c747e071aee410ebb6972114d408bc'
LOCK = '88b114ddbcb3e1313858fa42201e87704d3f992d9b346aab754ef4e5edd2be4e'
AUDIT = 'bf7f8abe6d1dbf67209f4f2b296c2f367742d4f00b5cd7d5e5a4d6e2ce87b60e'
RECON = '03c451b2bb6c013a442f833923b8960d2b64cdef57f0015ec3bfa21c60591fb4'
MEAN = 'mean_11_23_37'
METRIC = 'OR_mixed_vs_FF_auroc'
IDENTITIES = {
 'scalar-metrics': ['model','panel','arm','condition','metric','sample','scope','topic'],
 'scalar-contrasts': ['model','panel','left','right','condition','metric','sample','scope','topic'],
 'constituent-metrics': ['model','arm','condition','metric','scope','topic','surface_order','saved_layer','source_operator','target_operator','trained_operator','evaluation_operator'],
 'constituent-contrasts': ['model','contrast','condition','metric','scope','topic','surface_order','saved_layer','source_operator','target_operator'],
 'wording-paired': ['head','condition','metric','scope','topic'],
}
def sha(b): return hashlib.sha256(b).hexdigest()
def check(path, expected):
    b=path.read_bytes()
    if sha(b) != expected: raise RuntimeError(f'SOURCE MISMATCH: {path}; expected {expected}; observed {sha(b)}')
    return b
def dump(path, obj): path.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False)+'\n')
def fmt(x): return f'{x:.6f}'
def interval(r): return f"{fmt(r['point'])} [{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]"
def csvout(path, rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); w.writeheader(); w.writerows(rows)
def build(args):
    out=pathlib.Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=True)
    for name in ['figures','tables','evidence']: (out/name).mkdir(exist_ok=True)
    source=pathlib.Path(args.evaluation).resolve()
    completion=json.loads(check(source/'completion.json',COMPLETION))
    lock=json.loads(check(REPO/'config/final_e/final_lock_20261004.json',LOCK))
    check(PACKAGE/'transport/final_lock_20261004.json',LOCK)
    companion=json.loads(check(REPO/'results/final_e_scientific_audit_20261005/audit-companion.json',AUDIT))
    assert companion['status']=='complete_report_reconstruction_passed'
    assert companion['original_completion']['sha256']==COMPLETION
    assert companion['reconstructed_completion']['sha256']==RECON
    check(REPO/'results/final_e_scientific_audit_20261005/reconstruction-completion.json',RECON)
    assert companion['independent_principal']['status']=='independent_principal_calculation_passed'
    assert lock['interpretation_contract']['gain']['practical_margin']==.02
    assert lock['interpretation_contract']['retention']['margins']=={'AND':.02,'atomic':.005}
    config_pins=json.loads((HERE/'protocol-source-pins.json').read_text())
    prep=json.loads((REPO/'config/final_e/preparation_v1.json').read_text())
    for p,h in config_pins.items():
        assert h == lock['immutable_contract'].get(p,prep['scientific_sha256'].get(p)), p
        check(REPO/p,h)
    data={}; indexes={}; provenance={}; extracted={}; cells=[]
    for name,keys in IDENTITIES.items():
        path=source/(name+'.json'); entry=completion['files'][path.name]
        b=check(path,entry['sha256']); assert len(b)==entry['bytes']
        rows=json.loads(b)['records']; data[name]=rows; idx={}
        for i,r in enumerate(rows):
            identity=tuple(r[k] for k in keys)
            assert identity not in idx, (name,identity)
            idx[identity]=i
        indexes[name]=idx
        provenance[name]={'path':str(path),'sha256':entry['sha256'],'bytes':entry['bytes'],'rows':len(rows),'identity_fields':keys}
    def take(name, **identity):
        keys=IDENTITIES[name]; assert set(identity)==set(keys),(name,identity)
        i=indexes[name][tuple(identity[k] for k in keys)]; r=data[name][i]
        rid=f'{name}:{i}'
        extracted[rid]={'source_file':name+'.json','source_sha256':provenance[name]['sha256'],'record_index_zero_based':i,'identity':identity,'row':r}
        return rid,r
    def metric(model,arm,panel,scope):
        return take('scalar-metrics',model=model,arm=arm,panel=panel,condition='bare',metric=METRIC,sample=MEAN,scope=scope,topic='all')
    def contrast(model,left,right,panel,scope,metric=METRIC,condition='bare'):
        return take('scalar-contrasts',model=model,left=left,right=right,panel=panel,condition=condition,metric=metric,sample=MEAN,scope=scope,topic='all')
    def bind(asset,row,column,ref,field='interval'):
        rid,r=ref; val=interval(r) if field=='interval' else fmt(r[field])
        cells.append(dict(asset=asset,row=row,column=column,source_row=rid,field=field,display=val))
        return val
    # Compare the published principal summaries by scientific identity, never by display role.
    principal=[]
    evalcsv=list(csv.DictReader((REPO/'results/final_e_scientific_evaluation_20261005/principal-results.csv').open()))
    auditcsv=list(csv.DictReader((REPO/'results/final_e_scientific_audit_20261005/principal-results.csv').open()))
    for a in auditcsv:
        identity={k:a[k] for k in IDENTITIES['scalar-contrasts'] if k!='topic'}; identity['topic']='all'
        rid,r=take('scalar-contrasts',**identity)
        matches=[e for e in evalcsv if all(e[k]==str(v) for k,v in identity.items())]; assert len(matches)==1
        for k in ['point','ci_low','ci_high','valid_replicates','invalid_replicates']:
            assert float(a[k])==r[k]==float(matches[0][k]),(rid,k)
        principal.append({'source_row':rid,'audit_role':a['role'],'evaluation_role':matches[0]['role'],'audit_status':{k:v for k,v in a.items() if k.startswith('reviewed_')}})
    main=[]; retention=[]; secondary=[]; wording=[]; constituent=[]
    for model in ['qwen','llama']:
        for scope in ['pooled','topic_macro']:
            k=f'{model}/{scope}'; row={'Model':model.title(),'Scope':'Pooled' if scope=='pooled' else 'Five-topic macro'}
            for arm,panel in [('R_all','b25'),('S_all','b25'),('S_full','full_bank')]: row[arm]=bind('main',k,arm,metric(model,arm,panel,scope),'point')
            row['Original R_all - S_all']=bind('main',k,'Original R_all - S_all',contrast(model,'R_all','S_all','b25',scope))
            row['Sensitivity R_all - S_full']=bind('main',k,'Sensitivity R_all - S_full',contrast(model,'R_all','S_full','full_bank',scope))
            main.append(row)
            for endpoint,condition,margin in [('atomic_auroc','atomic_E',.005),('AND_auroc','bare',.02)]:
                ref=contrast(model,'R_all','reduced_LR','b25',scope,endpoint,condition)
                assert ref[1]['retention_margin']==margin and ref[1]['interval_supported']
                retention.append({'Model':model.title(),'Endpoint':endpoint.replace('_auroc',''),'Scope':row['Scope'],'R_all - reduced_LR':bind('retention',k+'/'+endpoint,'R_all - reduced_LR',ref),'Loss boundary':fmt(-margin),'Frozen decision':'Supported'})
            for left,right in [('S_full','S_atom_full'),('R_all','S_atom_full')]:
                secondary.append({'Model':model.title(),'Scope':row['Scope'],'Comparison':left+' - '+right,'Difference [95% interval]':bind('secondary',k+'/'+left,'Difference [95% interval]',contrast(model,left,right,'full_bank',scope))})
    registry=json.loads((REPO/'config/final_e/head_registry_v1.json').read_text())
    aliases={(a['model'],a['sample']):a['head'] for a in registry['scalar_aliases'] if a['arm']=='R_all' and a['panel']=='b25'}
    for model in ['qwen','llama']:
        for condition,label in [('or_explicit_or_both_v1','Or both'),('or_at_least_one_v1','At least one')]:
            for sample in [11,23,37]:
                ref=take('wording-paired',head=aliases[model,sample],condition=condition,metric=METRIC,scope='pooled',topic='all')
                wording.append({'Model':model.title(),'Wording':label,'Sample':str(sample),'Wording - bare [95% interval]':bind('wording',f'{model}/{condition}/{sample}','Wording - bare [95% interval]',ref)})
        for endpoint in ['first_marginal','second_marginal','first_given_second_0','first_given_second_1','second_given_first_0','second_given_first_1']:
            ref=take('constituent-contrasts',model=model,contrast='target_trained_minus_source_trained',condition='or_explicit_or_both_v1',metric=endpoint,scope='pooled',topic='all',surface_order='all',saved_layer=18 if model=='qwen' else 12,source_operator='AND',target_operator='OR')
            decision={'upper_bound_below_05':'Upper bound < .05','unresolved_crosses_05':'Unresolved at .05','material_deficit_lower_bound_above_05':'Deficit > .05'}[ref[1]['classification']]
            constituent.append({'Model':model.title(),'Endpoint':endpoint,'Target - transferred [95% interval]':bind('constituent',model+'/'+endpoint,'Target - transferred [95% interval]',ref),'Frozen classification':decision})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.hashsalt':'final-e-paper-20261005','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    geometry=[]
    def save(fig,folder,name):
        fig.canvas.draw()
        # Layout verification records all in-layout text extent checks against the canvas.
        renderer=fig.canvas.get_renderer(); bounds=fig.bbox
        bad=[]
        from matplotlib.text import Text
        for artist in fig.findobj(Text):
            if artist.get_visible() and artist.get_text() and artist.get_in_layout():
                b=artist.get_window_extent(renderer)
                if b.width and b.height and (b.x0 < -1 or b.y0 < -1 or b.x1>bounds.width+1 or b.y1>bounds.height+1): bad.append(artist.get_text())
        # Tight export includes ticks outside the base canvas, if any; require clean base layout anyway.
        assert not bad,(name,bad)
        geometry.append({'asset':name,'text_outside_canvas':bad})
        for ext in ['pdf','svg','png']:
            metadata={'Creator':'Final-E formatting only'}
            if ext=='pdf': metadata.update(CreationDate=None,ModDate=None)
            if ext=='svg': metadata.update(Date=None)
            fig.savefig(out/folder/(name+'.'+ext),dpi=180,bbox_inches='tight',metadata=metadata)
            if ext=='svg':
                svg=out/folder/(name+'.svg')
                svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
        plt.close(fig)
    captions={
     'main':'Bare OR TF/FT versus FF AUROC (TT excluded), averaged at the metric level across samples 11/23/37.\nPooled R_all - S_all is the original primary; macro is a qualification. R_all - S_full is a separate sensitivity.\nBrackets are paired pointwise conditional 95% intervals. Rounded values do not determine decisions.',
     'retention':'R_all minus frozen reduced-training canonical LR on E; sample-mean paired pointwise conditional 95% intervals.\nAtomic retention requires the lower bound >= -0.005; AND requires >= -0.020, at both scopes.\nQwen atomic differences remain negative. Retention within a margin does not establish zero loss.',
     'secondary':'Bare OR boundary, sample means. These separately frozen comparisons do not replace the original primary.\nS_full uses compound labels for selection; S_atom_full uses atomic labels for selection.\nPointwise conditional 95% intervals; no simultaneous coverage.',
     'wording':'Frozen R_all heads; pooled OR TF/FT versus FF; each sample shown separately. Direction: wording minus bare.\nBoth OR wording variants were excluded from fitting/selection. No averaging of interval endpoints.\nPointwise conditional 95% intervals. See evidence/extracted-rows.json for exact heads and condition IDs.',
     'constituent':'AND-trained transfer to OR versus matched-layer OR-trained heads, explicit "or both" wording, pooled, all orders.\nDirection: target-trained minus source-trained on target; positive values indicate a transfer deficit.\nSaved layers Qwen 18 / Llama 12. Frozen .05 classifications and pointwise conditional 95% intervals.',
    }
    def texescape(s):
        for a,b in [('\\',r'\textbackslash{}'),('_',r'\_'),('%',r'\%'),('&',r'\&'),('>=',r'$\geq$'),('<',r'\textless{}'),('>',r'\textgreater{}')]: s=s.replace(a,b)
        return s
    for name,rows in [('main',main),('retention',retention),('secondary',secondary),('wording',wording),('constituent',constituent)]:
        csvout(out/'tables'/f'{name}.csv',rows)
        headers=list(rows[0]); vals=[[r[k] for k in headers] for r in rows]
        md='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(v)+' |\n' for v in vals)+'\n'+captions[name].replace('\n',' ')+'\n'
        (out/'tables'/f'{name}.md').write_text(md)
        tex='\\begin{table*}[t]\n\\centering\n\\resizebox{\\textwidth}{!}{%\n\\begin{tabular}{'+'l'*len(headers)+'}\n\\hline\n'
        tex+=' & '.join(map(texescape,headers))+r' \\'+'\n\\hline\n'
        tex+=''.join(' & '.join(map(texescape,v))+r' \\'+'\n' for v in vals)
        tex+='\\hline\n\\end{tabular}}\n\\caption{'+texescape(captions[name].replace('\n',' '))+'}\n\\label{tab:final-e-'+name+'}\n\\end{table*}\n'
        (out/'tables'/f'{name}.tex').write_text(tex)
        width=16 if name=='main' else 12
        fig,ax=plt.subplots(figsize=(width,2.7+len(rows)*.34)); ax.axis('off')
        fig.subplots_adjust(left=.025,right=.975,top=.83,bottom=.27)
        table=ax.table(cellText=vals,colLabels=[h.replace('Original ','Original\n').replace('Sensitivity ','Sensitivity\n') for h in headers],loc='center',cellLoc='left',colLoc='left')
        table.auto_set_font_size(False); table.set_fontsize(9.5); table.scale(1,1.55)
        table.auto_set_column_width(list(range(len(headers))))
        for (i,j),cell in table.get_celld().items():
            cell.set_edgecolor('#d8dee6'); cell.set_linewidth(.5)
            if i==0: cell.set_facecolor('#e5edf5'); cell.set_text_props(weight='bold'); cell.set_height(cell.get_height()*1.45)
            elif i%2:cell.set_facecolor('#f5f7fa')
        fig.suptitle({'main':'Held-out-entity OR discrimination','retention':'Atomic and AND retention','secondary':'Separate full-bank comparisons','wording':'Wording sensitivity of the learned readout','constituent':'Constituent transfer: explicit OR wording'}[name],x=.035,y=.95,ha='left',fontsize=15,weight='bold')
        fig.text(.035,.055,captions[name],fontsize=10,linespacing=1.6)
        save(fig,'tables',name)
    colors={'qwen':'#176b9b','llama':'#b64d21'}
    fig,ax=plt.subplots(figsize=(10,5)); fig.subplots_adjust(left=.20,right=.97,top=.83,bottom=.28)
    fig.suptitle('Learning beyond compound-guided selection',x=.05,y=.95,ha='left',fontsize=16,weight='bold')
    ys=[];labels=[]
    for i,(model,scope) in enumerate([(m,s) for m in ['qwen','llama'] for s in ['pooled','topic_macro']]):
        rid,r=contrast(model,'R_all','S_all','b25',scope); y=3-i
        ax.errorbar(r['point'],y,xerr=[[r['point']-r['ci_low']],[r['ci_high']-r['point']]],fmt='o' if scope=='pooled' else 's',color=colors[model],capsize=4,ms=7)
        ys.append(y);labels.append(model.title()+' / '+('pooled' if scope=='pooled' else 'macro'))
        cells.append({'asset':'primary-gain','source_row':rid,'fields':['point','ci_low','ci_high'],'values':[r[k] for k in ['point','ci_low','ci_high']]})
    ax.axvline(0,color='#666',lw=1);ax.axvline(.02,color='#555',ls='--',lw=1)
    ax.set_yticks(ys,labels);ax.set_xlim(-.01,.22);ax.set_xticks([0,.05,.10,.15,.20]);ax.set_ylim(-.6,3.6);ax.grid(axis='x',alpha=.18);ax.set_xlabel('R_all - S_all AUROC')
    fig.text(.05,.055,'Pooled = original primary; fixed-five-topic macro = separate qualification. Dashed line: .02 practical gain.\nBare OR TF/FT versus FF; TT excluded. Points and paired 95% intervals use the frozen sample means.\nIntervals are pointwise and conditional on frozen fits, selectors and support.',fontsize=10,linespacing=1.5)
    save(fig,'figures','primary-gain')
    fig,axs=plt.subplots(1,2,figsize=(12,5.2)); fig.subplots_adjust(left=.15,right=.96,top=.77,bottom=.31,wspace=.42)
    fig.suptitle('Retention against the frozen reduced-training LR',x=.035,y=.96,ha='left',fontsize=16,weight='bold')
    for ax,endpoint,condition,margin in zip(axs,['atomic_auroc','AND_auroc'],['atomic_E','bare'],[.005,.02]):
        for i,(model,scope) in enumerate([(m,s) for m in ['qwen','llama'] for s in ['pooled','topic_macro']]):
            rid,r=contrast(model,'R_all','reduced_LR','b25',scope,endpoint,condition)
            ax.errorbar(r['point'],3-i,xerr=[[r['point']-r['ci_low']],[r['ci_high']-r['point']]],fmt='o' if scope=='pooled' else 's',color=colors[model],capsize=4,ms=6)
            cells.append({'asset':'retention-figure','source_row':rid,'fields':['point','ci_low','ci_high','retention_margin'],'values':[r[k] for k in ['point','ci_low','ci_high','retention_margin']]})
        ax.axvline(-margin,color='#a83232',ls='--',label=f'Loss boundary {-margin:.3f}');ax.axvline(0,color='#777',lw=1)
        ax.set_yticks(ys,labels); ax.set_ylim(-.6,3.6);ax.set_title(('Atomic' if endpoint=='atomic_auroc' else 'AND')+f' / loss margin {margin:.3f}',pad=16)
        ax.set_xlabel('R_all - reduced_LR AUROC'); ax.grid(axis='x',alpha=.18)
    axs[0].set_xlim(-.0058,.0025);axs[0].set_xticks([-.005,0,.002]);axs[1].set_xlim(-.027,.082);axs[1].set_xticks([-.02,0,.02,.04,.06,.08])
    fig.text(.035,.055,'Different x-axis scales. Dashed lines mark the endpoint-specific loss boundaries. All eight frozen checks pass.\nRetention requires each paired interval lower bound >= its loss boundary. Qwen atomic losses remain visible.\nSample means; pointwise conditional 95% intervals. Passing all checks gives no joint 95% coverage.',fontsize=10,linespacing=1.5)
    save(fig,'figures','retention')
    # An exhaustive comparison catalog, not an additional statistical analysis.
    catalog=[]
    for name,rows in data.items():
        groups=collections.defaultdict(list)
        fields=[k for k in IDENTITIES[name] if k not in ['sample','scope','topic','surface_order','head','saved_layer']]
        if name=='wording-paired': fields=['head','condition','metric']
        for r in rows: groups[tuple(r[k] for k in fields)].append(r)
        for key,rr in sorted(groups.items(),key=lambda x:str(x[0])):
            catalog.append({'source_file':name+'.json','source_sha256':provenance[name]['sha256'],'identity_group':dict(zip(fields,key)),'row_count':len(rr),'samples':sorted({str(r.get('sample','not_applicable')) for r in rr}),'scopes':sorted({r['scope'] for r in rr}),'topics':sorted({r['topic'] for r in rr}),'orders':sorted({r.get('surface_order','not_applicable') for r in rr}),'negative_point_rows':sum(r['point'] is not None and r['point']<0 for r in rr),'classifications':dict(collections.Counter(r.get('classification',r.get('absolute_classification','not_applicable')) for r in rr))})
    # Preserve exact all-scope adverse constituent rows; no favorable-result filter in the catalog.
    for name in ['constituent-metrics','constituent-contrasts']:
        for r in data[name]:
            if r.get('absolute_classification') in ['point_below_90','point_at_least_90_bound_uncertain'] or r.get('classification') in ['material_deficit_lower_bound_above_05','unresolved_crosses_05']:
                take(name,**{k:r[k] for k in IDENTITIES[name]})
    # Frozen control and sensitivity rows: every contrast, both models/scopes and every condition/metric.
    for r in data['scalar-contrasts']:
        if r['sample']==MEAN and r['scope'] in ['pooled','topic_macro']:
            take('scalar-contrasts',**{k:r[k] for k in IDENTITIES['scalar-contrasts']})
    descriptive={(a['panel'],a['model'],a['arm']) for a in registry['scalar_aliases'] if a['category']=='descriptive_D_selected_reference'}
    descriptive_refs=[]
    for r in data['scalar-metrics']:
        if (r['panel'],r['model'],r['arm']) in descriptive and r['sample']==MEAN and r['scope'] in ['pooled','topic_macro'] and r['condition']=='bare' and r['metric']==METRIC:
            rid,_=take('scalar-metrics',**{k:r[k] for k in IDENTITIES['scalar-metrics']}); descriptive_refs.append(rid)
    dump(out/'evidence'/'extracted-rows.json',extracted)
    dump(out/'evidence'/'display-map.json',cells)
    dump(out/'evidence'/'threshold-sources.json',{
      'source':'config/final_e/final_lock_20261004.json','sha256':LOCK,
      'practical_gain':{'json_pointer':'/interpretation_contract/gain/practical_margin','value':.02,'assets':['figures/primary-gain.pdf','results.md']},
      'atomic_loss_margin':{'json_pointer':'/interpretation_contract/retention/margins/atomic','value':.005,'assets':['figures/retention.pdf','tables/retention.pdf']},
      'AND_loss_margin':{'json_pointer':'/interpretation_contract/retention/margins/AND','value':.02,'assets':['figures/retention.pdf','tables/retention.pdf']},
      'constituent_transfer_margin':{'source':'config/final_e/preparation_v1.json','sha256':config_pins['config/final_e/preparation_v1.json'],'json_pointer':'/constituent_margins/transfer_loss','value':.05,'also_in_each_displayed_source_row':'planning_margin'},
      'protocol_source_pins':config_pins})
    dump(out/'evidence'/'source-verification.json',{'completion_sha256':COMPLETION,'lock_sha256':LOCK,'audit_companion_sha256':AUDIT,'reconstruction_completion_sha256':RECON,'sources':provenance,'principal_csv_identity_matches':principal,'layout_checks':geometry,'scientific_calculations_performed':False})
    dump(out/'evidence'/'secondary-catalog.json',catalog)
    # Compact source index links complete external tables; the arrays are never opened.
    md='# Secondary-results inventory\n\nAll frozen rows remain authoritative, including unfavorable and inconclusive outcomes. The [machine-readable catalog](evidence/secondary-catalog.json) covers every source row by scientific group, retaining model/head, arms, conditions, endpoints, sample, topic, scope and order coverage. Counts describe dependent report rows, not independent findings. The [exact extracted rows](evidence/extracted-rows.json) include all displayed rows, every sample-mean pooled/macro scalar contrast, and all unfavorable/uncertain constituent classifications. No new metric or interval is computed.\n\n'
    for name,p in provenance.items():
        md+=f"- [{name}: complete source table]({p['path']}) — {p['rows']:,} rows; SHA-256 `{p['sha256']}`.\n"
    md+='\n[Wording table](tables/wording.pdf): all six frozen R_all heads, both OR wordings, pooled OR boundary; wording minus bare. Every point is negative. Eleven of twelve intervals are wholly negative; Llama sample 37, at-least-one wording, crosses zero. This is not wording invariance. The complete source also retains AND wording, all other heads, macro and topic rows, and unchanged-operator comparisons.\n\n[Constituent table](tables/constituent.pdf): all six constituent AUROC endpoints for both models, AND-to-OR transfer at the saved source-selected layer, explicit-or-both wording, pooled, all surface orders. Positive target-minus-source differences mean worse transfer. Some exceed the frozen .05 deficit criterion; this precludes an unqualified transfer-success claim. The complete sources retain both directions, every wording, topic, AB/BA order, marginal/conditional metrics, fixed-threshold decisions, joint correctness, Boolean composition and min/max ranking.\n\n[Separate full-bank table](tables/secondary.pdf): S_full minus S_atom_full and R_all minus S_atom_full retain their own evidence roles. D-selected references remain descriptive; they are not E-selected maxima.\n\nFrozen scalar comparisons (complete conditions, metrics, samples and topic/order scope live in the linked catalog):\n\n'
    contrasts=json.loads((REPO/'config/final_e/contrasts_v1.json').read_text())
    for panel,pairs in contrasts.items():
        md+=f'**{panel}**\n\n'+''.join(f'- `{a} − {b}`\n' for a,b in pairs)+'\n'
    md+='Controls include fixed-layer selection/repair, LR-only selection, repair at the selected atomic layer, R0 and atomic-only adaptation, C_clean and the D-selected bank references. These cannot establish that the atomic-loss term causes retention or that information lies outside the collective bank span. Equal-boundary sensitivity does not replace the original pooled primary. No E-driven comparator search occurred.\n\n'
    for name,key in [('constituent-metrics','absolute_classification'),('constituent-contrasts','classification')]:
        counts=collections.Counter(r[key] for r in data[name] if key in r)
        md+=f"Existing {name} classifications: "+'; '.join(f'`{k}`: {v:,} rows' for k,v in sorted(counts.items()))+'.\n\n'
    md+='Negative points in the complete sources: '+', '.join(f"{sum(r['point'] is not None and r['point']<0 for r in data[n]):,} {n} rows" for n in ['scalar-contrasts','constituent-contrasts','wording-paired'])+'. These signs alone are not significance or failure classifications. Classification fields are absent where the frozen report did not define them; absence is not failure. AB/BA constituent records can be point-only; no interval is invented.\n'
    (out/'secondary-results.md').write_text(md)
    # Results text uses exactly the same pinned rows as the display map.
    q=main[0];l=main[2]
    results=f'''# Final-E results draft

Under the fixed B25 allocation, learning the compound-supervised readout improved held-out-entity OR-boundary discrimination beyond compound-guided selection of an atomic-trained readout in both models (claim C1). On pooled bare OR TF/FT versus FF, excluding TT, Qwen's R_all, S_all and S_full AUROCs were {q['R_all']}, {q['S_all']} and {q['S_full']}; Llama's were {l['R_all']}, {l['S_all']} and {l['S_full']}. The original paired R_all − S_all gains were {q['Original R_all - S_all']} for Qwen and {l['Original R_all - S_all']} for Llama. Both lower bounds exceeded the separately declared .02 practical-gain threshold (C2). Values are metric-level means over samples 11/23/37; brackets denote pointwise conditional 95% intervals.

The fixed-five-topic macro gains were {main[1]['Original R_all - S_all']} for Qwen and {main[3]['Original R_all - S_all']} for Llama (C3). These support a separate equal-topic qualification, not a co-primary or a claim that every topic benefits. See [main table](tables/main.pdf) and [primary-gain figure](figures/primary-gain.pdf).

Both models met all four required retention checks against the frozen reduced-training canonical LR: atomic and AND AUROC, each pooled and macro (C4). The loss margins were .005 for atomic and .02 for AND. Qwen's atomic differences were {retention[0]['R_all - reduced_LR']} pooled and {retention[2]['R_all - reduced_LR']} macro (C5). These small negative differences remain losses even though they are within the declared margin. Retention does not establish zero loss, calibration or unchanged behavior. The [retention table](tables/retention.pdf) and [figure](figures/retention.pdf) show all eight checks.

The separate restored-full-training sensitivity, R_all − S_full, was {q['Sensitivity R_all - S_full']} pooled for Qwen and {l['Sensitivity R_all - S_full']} for Llama (C6); both lower bounds exceeded .02, and their separately reported macro intervals were also positive. This supports robustness to the tested full-training comparator without replacing the original primary. Exposure, preprocessing and balanced sampling change together, S_full still uses compound labels, and full-training repair was not tested.

S_full − S_atom_full had positive pooled and macro intervals in both models, supporting its separate comparison of compound-guided versus atomic-guided selection in the full bank (C7). R_all − S_atom_full was also positive at both scopes in both models (C8). These comparisons appear in the [secondary table](tables/secondary.pdf) and have different roles from the primary. D-selected bank and clean-compound references remain descriptive (C9).

The secondary results constrain broader interpretations. Across the six frozen R_all heads, both OR wording variants had negative pooled OR-boundary differences relative to bare wording; eleven of the twelve paired intervals were wholly negative, while Llama sample 37 for at-least-one wording crossed zero (C10). The [wording table](tables/wording.pdf) retains every sample rather than averaging interval endpoints. Constituent transfer also had unfavorable findings: under explicit-or-both wording, matched-layer OR-trained heads exceeded AND-trained transfer on several constituent endpoints by more than the frozen .05 margin (C11). The [constituent table](tables/constituent.pdf) shows all six pooled endpoints for both models, including uncertain comparisons. The [complete secondary inventory](secondary-results.md) retains control, sensitivity, topic, order and wording coverage, including point-below-.90 and unresolved classifications. Counts of such dependent rows are not independent findings.

Taken together, the original positive gain and separate retention checks support the locked benchmark improvement-with-retention claim in both models (C12). Practical magnitude, equal-topic weighting and restored-full-training robustness each rely on their own checks. This conjunction has no simultaneous or joint 95% coverage and concerns held-out entities within the benchmark, not unseen topics, calibration, behavioral truthfulness or causal use.
'''
    (out/'results.md').write_text(results)
    def refs(asset, predicate=lambda r:True):
        return sorted({c['source_row'] for c in cells if c['asset']==asset and predicate(extracted[c['source_row']]['row'])})
    scope_pooled=lambda r:r['scope']=='pooled'
    primary=lambda r:r.get('left')=='R_all' and r.get('right')=='S_all'
    specs=[
      ('C1','The compound-supervised readout improves pooled held-out-entity bare OR-boundary AUROC beyond compound-guided atomic-readout selection in both models.','original primary',refs('primary-gain',scope_pooled)+refs('main',lambda r:'arm' in r and r['scope']=='pooled'),['tables/main.pdf','figures/primary-gain.pdf'],'Separate models; TT excluded; fixed B25 benchmark; comparison is R_all - S_all.'),
      ('C2','Both original-primary lower bounds exceed the declared .02 practical-gain threshold.','practical qualification',refs('primary-gain',scope_pooled),['figures/primary-gain.pdf'],'Practical gain is a separate criterion applied to unrounded bounds.'),
      ('C3','Both models have positive fixed-five-topic macro R_all - S_all intervals.','equal-topic qualification',refs('primary-gain',lambda r:r['scope']=='topic_macro'),['tables/main.pdf','figures/primary-gain.pdf'],'Macro is not co-primary; does not imply each topic or sample improves; .02 is not required for macro support.'),
      ('C4','Atomic and AND retention against the frozen reduced-training canonical LR is supported at pooled and macro scopes in both models.','retention criteria',refs('retention'),['tables/retention.pdf','figures/retention.pdf'],'Atomic margin .005; AND .02; lower-bound rule inclusive; AUROC only; no joint coverage.'),
      ('C5','Qwen has small negative atomic differences and wholly negative intervals while meeting the .005 retention margin.','unfavorable qualification',refs('retention',lambda r:r['model']=='qwen' and r['metric']=='atomic_auroc'),['tables/retention.pdf'],'Retention within a margin does not establish zero loss.'),
      ('C6','R_all exceeds S_full in pooled OR-boundary AUROC with lower bounds above .02; macro intervals are also positive.','full-training sensitivity',refs('main',lambda r:r.get('right')=='S_full'),['tables/main.pdf'],'Does not replace primary; atomic exposure, preprocessing and balanced sampling change together; S_full uses compound labels; full-training repair untested.'),
      ('C7','S_full - S_atom_full has positive pooled and macro intervals in both models.','separate selection-label-benefit comparison',refs('secondary',lambda r:r['left']=='S_full'),['tables/secondary.pdf'],'Compound-guided versus atomic-guided selection within the tested full bank; not learned repair or pure exposure effect.'),
      ('C8','R_all - S_atom_full has positive pooled and macro intervals in both models.','additional preserved comparison',refs('secondary',lambda r:r['left']=='R_all'),['tables/secondary.pdf'],'Different comparator from original primary; not a replacement for it.'),
      ('C9','D-selected bank and C_clean references remain descriptive.','descriptive references',descriptive_refs,['secondary-results.md'],'Selection used D; no E search; fixed optimistic references; cannot claim information outside collective bank span.'),
      ('C10','Both OR wording variants lower all six R_all pooled boundary points; eleven of twelve intervals are wholly negative and Llama sample 37 at-least-one crosses zero.','unfavorable wording results',refs('wording'),['tables/wording.pdf'],'Per-sample same-head wording minus bare; no interval-endpoint averaging; pointwise conditional; not wording invariance.'),
      ('C11','Constituent transfer has material deficits under explicit-or-both wording, alongside uncertain comparisons.','unfavorable constituent results',refs('constituent'),['tables/constituent.pdf','secondary-results.md'],'Pooled all-order AND-to-OR at source-selected layers; target-trained minus source-trained; six AUROC endpoints; .05 frozen classifications; other topics/orders and .90 classifications retained in full sources.'),
      ('C12','Both models meet the locked benchmark improvement-with-retention criteria.','conjunction of separate checks',refs('primary-gain',scope_pooled)+refs('retention'),['results.md','tables/main.pdf','tables/retention.pdf'],'Not simultaneous or joint 95% coverage; held-out entities within benchmark; no unseen-topic, calibration, behavior, causal-use or atomic-loss-mechanism claim.'),
    ]
    claims=[]
    for cid,wording_text,role,rowids,assets,qualification in specs:
        claims.append({'claim_id':cid,'wording':wording_text,'evidential_role':role,'supporting_rows':rowids,'models':sorted({extracted[x]['row']['model'] if 'model' in extracted[x]['row'] else extracted[x]['row']['head'].split(':')[1].split('/')[0] for x in rowids}),'endpoints':sorted({extracted[x]['row']['metric'] for x in rowids}),'comparisons':sorted({str((extracted[x]['row'].get('left',extracted[x]['row'].get('arm','wording/constituent')),extracted[x]['row'].get('right',extracted[x]['row'].get('contrast','see identity')))) for x in rowids}),'assets':assets,'required_qualifications':qualification})
    dump(out/'claim-evidence.json',claims)
    claimmd='# Claim-to-evidence map\n\nEach row ID below resolves to the full-precision source row, complete scientific identity, zero-based record index and file SHA-256 in [extracted-rows.json](evidence/extracted-rows.json). Every displayed value additionally has a [display binding](evidence/display-map.json). The [machine-readable map](claim-evidence.json) records models, endpoints and comparisons. All results claims require pointwise conditional inference and the unchanged lock.\n\n'
    for c in claims:
        claimmd+=f"**{c['claim_id']}: {c['wording']}**\n\nRole: {c['evidential_role']}. Required qualifications: {c['required_qualifications']}\n\nAssets: "+', '.join(f'[{x}]({x})' for x in c['assets'])+'\n\nSource rows: '+', '.join('`'+x+'`' for x in c['supporting_rows'])+'\n\n'
    claimmd+='**Methods and audit provenance.** Model IDs/revisions, representation, allocation, fitting and selection are pinned in [protocol-source-pins.json](protocol-source-pins.json) and linked at the corresponding methods paragraphs. E counts come from the pinned v2 manifest; frame sizes and inference from the final lock. The audit counts and acceptance come from the hash-pinned companion receipt and accepted audit report, not this formatting build. Limitations state the scope and exclusions in that lock. No novelty or literature claim is proposed.\n'
    (out/'claim-evidence.md').write_text(claimmd)
    print(json.dumps({'status':'rendered','display_bindings':len(cells),'exact_extracted_rows':len(extracted),'catalog_groups':len(catalog),'figures':2,'rendered_tables':5,'output':str(out)}))

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--evaluation',default=str(PACKAGE/'scientific-evaluation-20261005-v1/evaluation'))
    ap.add_argument('--output',default=str(HERE)); args=ap.parse_args()
    with open('/tmp/truth-probing-cpu-exclusive.lock','a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: sys.exit('CPU slot occupied; no outputs written. Retry after the current holder finishes.')
        build(args)
if __name__=='__main__': main()
