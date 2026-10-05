"""Post-audit preservation and locked interpretation; no new scientific metrics."""
from pathlib import Path
import collections
import csv
import importlib.util
import json
import subprocess
import sys
import time

spec=importlib.util.spec_from_file_location('launcher',Path(__file__).with_name('run_once.py'))
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
P,S,C,E=run.PACKAGE,run.SCIENCE,run.CODE,run.EVIDENCE


def positive(row):
    return 'unresolved' if row['ci_low'] is None else 'supported' if row['ci_low']>0 else 'unfavorable' if row['ci_high']<=0 else 'unresolved'


def practical(row):
    return 'unresolved' if row['ci_low'] is None else 'supported' if row['ci_low']>.02 else 'below_margin' if row['ci_high']<.02 else 'unresolved'


def retained(row,margin):
    return 'unresolved' if row['ci_low'] is None else 'supported' if row['ci_low']>=-margin else 'loss' if row['ci_high']<-margin else 'unresolved'


def main():
    started=time.monotonic();receipt=dict(status='checking',new_fits=0,new_selection=False,RunPod_contacted=False)
    try:
        pre=run.read(E/'preflight.json');process=run.read(E/'audit-process.json')
        op=run.read(S/'operation-receipts/audit.json');audit=run.read(S/'operation-receipts/audit.audit.json')
        assert process['returncode']==0 and process['attempts']==1 and process['retry'] is False
        assert op['status']=='completed' and op['error'] is None and op['cap_seconds']==10800
        assert audit['status']=='complete_report_reconstruction_passed' and audit['head_predictions']==85
        assert audit['new_fits']==0 and audit['new_selection'] is False
        assert audit['independent_principal']['status']=='independent_principal_calculation_passed'
        assert audit['independent_principal']['maximum_absolute_difference']<=2e-12
        assert audit['original_completion']==pre['source_completion']==run.pin(S/'evaluation/completion.json')
        assert audit['reconstructed_completion']==run.pin(S/'audit-reconstruction/completion.json')
        source=run.read(S/'evaluation/completion.json');rebuilt=run.read(S/'audit-reconstruction/completion.json')
        receipt['source_inventory']=run.inventory(S/'evaluation',{**source['files'],'completion.json':run.pin(S/'evaluation/completion.json')})
        receipt['reconstructed_inventory']=run.inventory(S/'audit-reconstruction',{**rebuilt['files'],'completion.json':run.pin(S/'audit-reconstruction/completion.json')})
        assert run.pin(P/'handoff/campaign-checksums.json')==pre['raw_manifest']
        receipt['raw_inventory']=run.inventory(P/'handoff/campaign',run.read(P/'handoff/campaign-checksums.json')['files'])
        for mapping in [pre['controls'],pre['existing_receipts']]:
            for p,x in mapping.items():assert run.pin(p)==x
        blocker=pre['historical_blocker'];assert run.pin(blocker['path'])=={k:blocker[k] for k in ('bytes','sha256')}
        receipt['blocker_preserved']=blocker
        for rel,x in pre['unrelated_pins'].items():assert run.pin(run.REPO/rel)==x
        lock=run.read(P/'transport/final_lock_20261004.json')
        assert run.pin(P/'transport/final_lock_20261004.json')['sha256']==run.LOCK
        for rel,h in lock['immutable_contract'].items():assert run.pin(C/rel)['sha256']==h
        original_preflight=json.loads(run.git(run.REPO,'show',run.DELIVERY+':results/final_e_scientific_evaluation_20261005/preflight.json'))
        for x in original_preflight['artifacts']:assert run.pin(x['path'])=={k:x[k] for k in ('bytes','sha256')}
        receipt['saved_artifacts_preserved']=len(original_preflight['artifacts'])
        review=Path(__file__).resolve().parents[2]
        for w in pre['worktrees']:
            if Path(w['path'])==review:continue
            assert run.git(w['path'],'rev-parse','HEAD')==w['head'] and run.git(w['path'],'status','--porcelain')==w['status'],w['path']
        assert run.git(C,'rev-parse','HEAD')==run.PRODUCER and run.git(C,'status','--porcelain')==''
        receipt['existing_worktrees_preserved']=len(pre['worktrees'])-1
        receipt['unrelated_files_preserved']=len(pre['unrelated_pins'])
        receipt['controls_and_existing_receipts_preserved']=True
        receipt['scientific_lock_and_six_pins_preserved']=True
        receipt['producer_clean_detached_unchanged']=True
        rows=run.read(S/'evaluation/scalar-contrasts.json')['records']
        principals=[];interpretation={}
        def pick(model,panel,left,right,condition,metric,scope):
            selected=[r for r in rows if (r['model'],r['panel'],r['left'],r['right'],r['condition'],r['metric'],r['scope'],str(r['sample']))==
                (model,panel,left,right,condition,metric,scope,'mean_11_23_37')]
            assert len(selected)==1;return selected[0]
        for model in ('qwen','llama'):
            primary=pick(model,'b25','R_all','S_all','bare','OR_mixed_vs_FF_auroc','pooled')
            macro=pick(model,'b25','R_all','S_all','bare','OR_mixed_vs_FF_auroc','topic_macro')
            full=pick(model,'full_bank','R_all','S_full','bare','OR_mixed_vs_FF_auroc','pooled')
            assessment=dict(original_positive=positive(primary),original_practical=practical(primary),
                original_macro_positive=positive(macro),full_training_positive=positive(full),full_training_practical=practical(full),retention={})
            for metric,condition,margin in [('atomic_auroc','atomic_E',.005),('AND_auroc','bare',.02)]:
                for scope in ('pooled','topic_macro'):
                    row=pick(model,'b25','R_all','reduced_LR',condition,metric,scope)
                    status=retained(row,margin);assessment['retention'][metric+':'+scope]=status
                    assert row['retention_margin']==margin and row['interval_supported']==(status=='supported')
                    principals.append(dict(role='retention',**row,reviewed_status=status))
            assessment['retention_all_scopes_supported']=all(x=='supported' for x in assessment['retention'].values())
            assessment['model_supported_claim']=assessment['original_positive']=='supported' and assessment['retention_all_scopes_supported']
            interpretation[model]=assessment
            for role,panel,left,right in [('original_primary','b25','R_all','S_all'),('full_training_sensitivity','full_bank','R_all','S_full'),
                    ('secondary_selection_comparison','full_bank','S_full','S_atom_full'),('secondary_atomic_only_comparison','full_bank','R_all','S_atom_full')]:
                for scope in ('pooled','topic_macro'):
                    row=pick(model,panel,left,right,'bare','OR_mixed_vs_FF_auroc',scope)
                    principals.append(dict(role=role,**row,reviewed_positive=positive(row),reviewed_practical=practical(row)))
        published=run.read(S/'operation-receipts/locked-interpretation.json')
        assert interpretation==published['interpretation'],'published interpretation mismatch'
        receipt['interpretation']=interpretation
        receipt['principal_records']=principals
        receipt['interpretation_agrees_with_mac1']=True
        receipt['replicated_claim_supported']=all(x['model_supported_claim'] for x in interpretation.values())
        receipt['scope']='Locked interval/role interpretation only; no new fits, selection, metrics or analysis'
        census={}
        for name in ['scalar-metrics','scalar-contrasts','constituent-metrics','constituent-contrasts','wording-paired']:
            records=run.read(S/'evaluation'/f'{name}.json')['records']
            census[name]=dict(rows=len(records),negative_points=sum(r.get('point') is not None and r['point']<0 for r in records),
                undefined_points=sum(r.get('point') is None for r in records),undefined_intervals=sum('ci_low' in r and r['ci_low'] is None for r in records))
            if name.startswith('constituent'):
                census[name]['existing_classifications']=dict(collections.Counter(r.get('classification',r.get('absolute_classification','not_classified')) for r in records))
        receipt['complete_report_census']=census
        receipt['audit_operation']=dict(path=str(S/'operation-receipts/audit.json'),**run.pin(S/'operation-receipts/audit.json'))
        receipt['companion_audit_receipt']=dict(path=str(S/'operation-receipts/audit.audit.json'),**run.pin(S/'operation-receipts/audit.audit.json'))
        receipt['status']='passed'
    except BaseException as e:
        receipt.update(status='failed_preserved',error=dict(type=type(e).__name__,message=str(e)))
        raise
    finally:
        receipt['seconds']=time.monotonic()-started
        run.write(E/'post-audit-verification.json',receipt)
    print(json.dumps({k:receipt[k] for k in ['status','seconds','interpretation_agrees_with_mac1','replicated_claim_supported','source_inventory','raw_inventory']}))


if __name__=='__main__':main()
