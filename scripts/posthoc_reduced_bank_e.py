#!/usr/bin/env python3
"""Execute the separately authorized post-hoc reduced-bank E analysis, CPU only."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
import collections
import csv
import datetime
import fcntl
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits
from src.posthoc_reduced_bank_e import require, validate_bank, validate_order, boundary_auc, maxima
from src.checkpoint_r2_fresh_inputs import hash_value
from src.checkpoint_r2_fresh_store import pin, load_features
from src.checkpoint_r2_fresh_b25 import bank_specs, atomic_point
from src.checkpoint_r2_selection_v1 import AtomicCandidate, atomic_eligibility, TOPICS
from src.final_e_inventory import Metadata, read
from src.final_e_registry import Source, apply_head
from src import final_e_raw as raw

BASE='a8b4cd326daa243338dc7941c19d98ce75f5b77f'
B25_DELIVERY='1187140c7f79b0b64439eff382d420f51d974a25'
B25_PRODUCER='5e1bf47a457558cbc0735bd8f9f13ca029cbf3a4'
SMALL=ROOT/'results/posthoc_reduced_bank_e_20261011'


def write_json(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def write_csv(path,rows):
    require(bool(rows),'empty CSV')
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def git(*args):
    return subprocess.check_output(['git','-C',str(ROOT),*args],text=True).strip()


def run(args):
    started_clock=time.monotonic();created=datetime.datetime.now(datetime.timezone.utc).isoformat()
    require(platform.system()=='Darwin','this delivery is authorized for local macOS')
    subprocess.run(['git','-C',str(ROOT),'merge-base','--is-ancestor',BASE,'HEAD'],check=True)
    # Existing scientific modules/config/data/lock must stay exactly at reviewed base.
    tracked=git('ls-tree','-r','--name-only',BASE,'--','src','config','data','paper','results').splitlines()
    subprocess.run(['git','-C',str(ROOT),'diff','--exit-code',BASE,'--',*tracked],check=True,stdout=subprocess.DEVNULL)
    external=Path(args.external).resolve();small=Path(args.output).resolve()
    require(not any((p/'.git').exists() for p in [external,*external.parents]),'large scores must be outside Git')
    require(not external.exists(),'fresh external output required')
    require((small/'analysis_spec.json').is_file(),'post-hoc specification required before E scoring')
    require(not (small/'candidate_results.csv').exists(),'fresh small result files required')
    spec=read(small/'analysis_spec.json')
    require(spec['base_delivery']==BASE and spec['b25_producer']==B25_PRODUCER,'analysis specification identity')
    external.mkdir(parents=True);(external/'scores').mkdir()
    ledger={}
    def record(path,expected=None):
        p=Path(path).resolve();actual=pin(p)
        if expected:require(actual=={k:expected[k] for k in ('sha256','bytes')},'input hash changed: '+str(p))
        ledger[str(p)]=actual
        return actual
    specification_pin=record(small/'analysis_spec.json')
    write_json(external/'analysis_spec.json',spec)
    started=dict(started_utc=created,specification=specification_pin,analysis_code_commit=git('rev-parse','HEAD'),
                 base_delivery=BASE,command=[sys.executable,*sys.argv],external=str(external))
    write_json(external/'started.json',started)
    # Follow the reviewed handoffs and accepted manifests, never a directory glob.
    eval_handoff_path=ROOT/'results/final_e_scientific_evaluation_20261005/evaluation-handoff.json'
    audit_handoff_path=ROOT/'results/final_e_scientific_audit_20261005/audit-handoff.json'
    for p in [eval_handoff_path,audit_handoff_path,ROOT/'paper/final_e_20261005/README.md',ROOT/'paper/final_e_review_20261005/review.json']:
        record(p)
    handoff=read(eval_handoff_path);audit=read(audit_handoff_path)
    require(audit['status']=='passed' and audit['scientific_acceptance']=='accepted_under_the_unchanged_frozen_lock','unaccepted E lineage')
    for name in ('source_completion','reconstruction_completion','audit_companion','lock','raw_manifest'):
        record(audit[name]['path'],audit[name])
    companion=read(audit['audit_companion']['path'])
    require(companion['status']=='complete_report_reconstruction_passed','missing successful original audit')
    evaluation=Path(handoff['report']);completion=read(evaluation/'completion.json')
    record(handoff['roots']['path'],handoff['roots']);roots=read(handoff['roots']['path'])
    delivery_path=ROOT/'results/checkpoint_r2_b25_convergence_fix_20261002/delivery.json'
    record(delivery_path)
    require(subprocess.check_output(['git','-C',str(ROOT),'show',B25_DELIVERY+':'+str(delivery_path.relative_to(ROOT))])==delivery_path.read_bytes(),'B25 delivery lineage changed')
    delivery=read(delivery_path)
    require(delivery['producer_sha']==B25_PRODUCER and delivery['status']=='completed','B25 convergence successor incomplete')
    require(Path(roots['b25'])==Path(delivery['external_results']),'reduced bank roots differ')
    record(roots['b25_manifest'],delivery['external_manifest'])
    bank_source=Source('b25',roots['b25'],roots['b25_manifest'],delivery['external_manifest'],Path(roots['b25']).name+'/')
    def source_json(relative):
        ref=bank_source.ref(relative);ledger[str((bank_source.root/relative).resolve())]={k:ref[k] for k in ('bytes','sha256')}
        return read(bank_source.root/relative)
    bank_completion=source_json('completion.json');bank=source_json('bank.json');locked=source_json('locked.json')
    require(bank_completion['status']=='completed' and not bank_completion['fit_failures'] and not bank_completion['blockers'],'incomplete reduced bank')
    require(bank['producer_sha']==bank_completion['producer_sha']==B25_PRODUCER,'bank producer')
    validate_bank(bank['records'],bank_specs())
    registry_path=ROOT/'config/final_e/head_registry_v1.json';record(registry_path);registry=read(registry_path)
    candidates=[];jobs=collections.defaultdict(list);bindings=[]
    for model in ('qwen','llama'):
        records=[r for r in bank['records'] if r['model']==model]
        require(len(records)==spec['candidate_counts'][model],'model coverage')
        original=locked['selection']['shared_atomic_banks'][model]
        atomic=[]
        for r in records:
            s=r['spec'];topics={t:next(v['point'] for v in r['metrics'] if v['metric']=='atomic_auroc' and v['scope']=='topic' and v['topic']==t) for t in TOPICS}
            atomic.append(AtomicCandidate(r['id'],model,s['saved_layer'],s['method'],s['C'],True,atomic_point(r),topics))
        eligible=atomic_eligibility(atomic,model,original['reference_atomic'])
        require(eligible==original['eligibility'],'original D eligibility changed')
        emap={r['candidate_id']:r for r in eligible}
        for r in records:
            s=r['spec'];folder='fits/'+hash_value(r['id'])[:24]
            fit=source_json(folder+'/fit.json')
            require(fit['spec']==s and fit['metrics']==r['metrics'] and fit['parameters']==r['parameters'] and fit['valid_for_scoring'] is True,'fit differs from complete bank inventory')
            fitref=bank_source.ref(folder+'/fit.json');parref=bank_source.ref(folder+'/parameters.npz')
            require({k:parref[k] for k in ('bytes','sha256')}==fit['parameters'],'parameter pin')
            ledger[str((bank_source.root/parref['path']).resolve())]=fit['parameters']
            producer=fit.get('parameter_producer_sha',fit['producer_sha'])
            head=dict(id='b25:'+r['id'],frozen_id=r['id'],family='scalar',model=model,saved_layer=s['saved_layer'],
                      method=fit['scoring_method'],parameter_producer=producer,fit=fitref,parameters=parref,
                      preprocessing=None,preprocessing_metadata=None)
            if head['id'] in registry['heads']:
                require(all(registry['heads'][head['id']][k]==v for k,v in head.items()),'overlapping head identity differs')
            settings=dict(C=s['C'],training=fit['training'],optimizer=fit['optimizer'],
                          preprocessing='saved parameter archive, exact original method-specific inference')
            # Preserve all fit settings in the small provenance receipt; compact CSV.
            bindings.append(dict(head=head,spec=s,settings=settings,training_representation_manifest_sha256=fit['representation_manifest_sha256']))
            row=dict(candidate_id=r['id'],model=model,bank='reduced_P15',method=s['method'],saved_layer=s['saved_layer'],
                hf_hidden_state_index=s['saved_layer']+1,C=s['C'],settings=json.dumps(dict(C=s['C'],training_rows=fit['training']['rows'],training_group=fit['training']['group'],preprocessing='original saved parameters'),sort_keys=True),
                atomic_D_auroc=atomic_point(r),eligible=emap[r['id']]['eligible'],
                eligibility_reference_atomic_D=original['reference_atomic'],parameter_producer=producer,
                parameters_sha256=parref['sha256'],fit_sha256=fitref['sha256'],status='pending')
            candidates.append(row);jobs[model,s['saved_layer']].append((row,head))
    write_json(small/'head_provenance.json',bindings)
    print('Verified complete original reduced bank, all parameters, preprocessing and D eligibility.',flush=True)

    metadata=Metadata();bare=metadata.rows['bare']
    selected=np.array([i for i,r in enumerate(bare) if r['operator']=='OR' and r['truth_a']+r['truth_b']<2])
    expected_rows=[dict(example_id=bare[i]['example_id'],pair_id=bare[i]['pair_id'],topic=bare[i]['topic'],
        ordering=bare[i]['ordering'],truth_a=bare[i]['truth_a'],truth_b=bare[i]['truth_b'],label=bare[i]['label'],
        text_id=metadata.group_bindings['bare'][i]['text_id'],bare_row_index=int(i)) for i in selected]
    require(len(bare)==7200 and len(selected)==2700 and len({r['pair_id'] for r in expected_rows})==450,'E support mismatch')
    require(collections.Counter((r['truth_a'],r['truth_b'],r['ordering']) for r in expected_rows)==
            {(a,b,o):450 for a,b in [(0,0),(0,1),(1,0)] for o in ('AB','BA')},'truth cell/order coverage')
    labels=np.array([r['label'] for r in expected_rows],dtype=int)
    require(int(labels.sum())==1800,'label orientation')
    support=dict(rows=expected_rows,boundary_row_binding_sha256=hash_value(expected_rows),
                 bare_binding_sha256=metadata.receipt['ordered_groups']['bare'],bare_indices=selected.tolist(),metadata=metadata.receipt)
    write_json(external/'support.json',support)
    write_json(small/'support_receipt.json',dict(external_file=str(external/'support.json'),artifact=pin(external/'support.json'),
        boundary_row_binding_sha256=support['boundary_row_binding_sha256'],bare_binding_sha256=support['bare_binding_sha256'],
        pairs=450,boundary_rows=2700,positive_rows=1800,negative_rows=900,orders=['AB','BA']))
    for rel in ['config/final_e/preparation_v1.json','data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v2/manifest.json',
                'data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v2/bare_compounds.csv',
                'data/clean_protocol/compounds/entity_disjoint/final_e_preparation_v2/extraction_bindings.csv']:
        record(ROOT/rel)
    def accepted_file(relative):
        path=evaluation/relative;record(path,completion['files'][relative]);return path
    source_metrics=read(accepted_file('scalar-metrics.json'))['records']
    source_metrics_audit=Path(audit['reconstruction_completion']['path']).parent/'scalar-metrics.json'
    audited_completion=read(audit['reconstruction_completion']['path'])
    record(source_metrics_audit,audited_completion['files']['scalar-metrics.json'])
    require(read(source_metrics_audit)['records']==source_metrics,'original audited E metrics differ')
    accepted_scores={}
    def saved_scores(name):
        if name not in accepted_scores:
            folder='scores/'+hash_value(name)[:24]
            b=read(accepted_file(folder+'/binding.json'))
            require(b['head']==name and b['head_sha256']==hash_value(registry['heads'][name]) and b['group_bindings']==metadata.receipt['ordered_groups'],'accepted score identity/order')
            path=accepted_file(folder+'/scores.npz');require(pin(path)==b['artifact'],'accepted prediction artifact')
            with np.load(path,allow_pickle=False) as z:values=z['bare'].copy()
            require(values.shape==(7200,) and np.isfinite(values).all(),'accepted score coverage')
            accepted_scores[name]=values
        return accepted_scores[name]
    fixed=[];fixed_means={}
    for model in ('qwen','llama'):
        for arm in ('S_all','R_all'):
            per_sample=[]
            for sample in (11,23,37):
                alias=next(r for r in registry['scalar_aliases'] if (r['panel'],r['model'],r['arm'],r['sample'])==('b25',model,arm,sample))
                require(locked['procedures'][model][str(sample)][arm]==registry['heads'][alias['head']]['frozen_id'],'original selected procedure changed')
                point=boundary_auc(labels,saved_scores(alias['head'])[selected])
                original=next(r for r in source_metrics if (r['panel'],r['model'],r['arm'],r['sample'],r['condition'],r['metric'],r['scope'])==('b25',model,arm,sample,'bare','OR_mixed_vs_FF_auroc','pooled'))
                require(abs(point-original['point'])<=1e-14,'fixed procedure E metric mismatch')
                fixed.append(dict(model=model,arm=arm,sample=sample,head=alias['head'],E_auroc=point,accepted_E_auroc=original['point']))
                per_sample.append(point)
            mean=float(np.mean(per_sample))
            original=next(r for r in source_metrics if (r['panel'],r['model'],r['arm'],r['sample'],r['condition'],r['metric'],r['scope'])==('b25',model,arm,'mean_11_23_37','bare','OR_mixed_vs_FF_auroc','pooled'))
            require(abs(mean-original['point'])<=1e-14,'fixed metric aggregation mismatch')
            fixed.append(dict(model=model,arm=arm,sample='mean_11_23_37',head='metric mean only',E_auroc=mean,accepted_E_auroc=original['point']))
            fixed_means[model,arm]=mean
    write_csv(small/'fixed_procedures.csv',fixed)

    rawroot=Path(handoff['raw_campaign']);raw_manifest=read(handoff['raw_manifest']['path'])
    overlap=[];score_files={};raw_verification={};overlap_metric_checks=0
    for model in ('qwen','llama'):
        print('Verify accepted E representation:',model,flush=True)
        raw_verification[model]=raw.verify(rawroot,model,completion['access'],require_acceptance=True)
        for name in ('index.json','acceptance.json','token-audit.json','runtime-origin.json','token-plan.json'):
            p=rawroot/model/name;record(p,raw_manifest['files'][str(p.relative_to(rawroot))])
        # Reuse the accepted loader once per shard, retaining its verified mmap.
        config=raw.raw_contract();input_rows=raw.input_rows(metadata)
        size=read(ROOT/'config/final_e/preparation_v1.json')['shard_rows']
        identity=raw.campaign_identity(completion['access'],metadata)
        arrays=[];ordered=[]
        for index,start in enumerate(range(0,len(input_rows),size)):
            folder=rawroot/model/'shards'/f'{index:05d}'
            array,rs,receipt=load_features(folder,expected_rows=input_rows[start:start+size],expected_spec=config['models'][model],
                expected_provenance=raw.provenance(identity,model,rawroot/model/'runtime-origin.json'))
            for name,expected in receipt['files'].items():
                path=folder/name
                require(raw_manifest['files'][str(path.relative_to(rawroot))]==expected,'raw accepted manifest mismatch')
                ledger[str(path.resolve())]=expected  # Loader just checked each SHA-256.
            record(folder/'receipt.json',raw_manifest['files'][str((folder/'receipt.json').relative_to(rawroot))])
            arrays.append(array);ordered.extend(r['id'] for r in rs)
        require(ordered==[r['id'] for r in metadata.texts],'raw text row order mismatch')
        indices=np.array(metadata.indices('bare'));starts=np.cumsum([0]+[len(x) for x in arrays])
        actual_rows=[dict(r,text_id=ordered[indices[r['bare_row_index']]]) for r in expected_rows]
        validate_order(actual_rows,expected_rows)
        for layer in range(28 if model=='qwen' else 32):
            X=np.empty((7200,3584 if model=='qwen' else 4096),dtype=np.float64)
            for array,start,end in zip(arrays,starts[:-1],starts[1:]):
                take=np.flatnonzero((indices>=start)&(indices<end))
                X[take]=array[indices[take]-start,layer,:]
            require(np.isfinite(X).all(),'nonfinite features')
            for row,head in jobs[model,layer]:
                values=apply_head(head,X,roots)
                require(values.shape==(7200,) and np.isfinite(values).all(),'candidate prediction coverage')
                scores=values[selected];point=boundary_auc(labels,scores)
                if head['id'] in registry['heads']:
                    old=saved_scores(head['id']);diff=float(np.max(np.abs(values-old)))
                    require(np.allclose(values,old,rtol=1e-9,atol=1e-8),'accepted frozen score mismatch')
                    require(abs(point-boundary_auc(labels,old[selected]))<=1e-14,'accepted metric mismatch')
                    metric_rows=[r for r in source_metrics if r.get('head')==head['id'] and r['condition']=='bare' and r['metric']=='OR_mixed_vs_FF_auroc' and r['scope']=='pooled']
                    require(bool(metric_rows),'overlap lacks accepted metric')
                    require(all(abs(point-r['point'])<=1e-14 for r in metric_rows),'overlap report metric mismatch')
                    overlap_metric_checks+=len(metric_rows)
                    overlap.append(dict(head=head['id'],candidate_id=row['candidate_id'],bare_rows=7200,boundary_rows=2700,
                        max_absolute_score_difference=diff,exact_equal=bool(np.array_equal(values,old)),accepted_metric_rows=len(metric_rows),E_auroc=point))
                relative='scores/'+hash_value(head['id'])[:24]+'.npy'
                with (external/relative).open('xb') as stream:np.save(stream,scores,allow_pickle=False)
                score_files[relative]=pin(external/relative)
                row.update(E_auroc=point,status='scored',score_file=relative,score_sha256=score_files[relative]['sha256'],
                           row_binding_sha256=support['boundary_row_binding_sha256'])
            print(model,'saved layer',layer,'10 reduced candidates scored',flush=True)
        del X,arrays
    require(len(candidates)==600 and all(r['status']=='scored' for r in candidates),'incomplete bank')
    expected_overlap={b['head']['id'] for b in bindings if b['head']['id'] in registry['heads']}
    require({r['head'] for r in overlap}==expected_overlap,'overlap coverage')
    summary=[];winners=[]
    for model in ('qwen','llama'):
        rr=[r for r in candidates if r['model']==model]
        for label,restriction in [('eligible',True),('unrestricted',False)]:
            maximum=maxima(rr,restriction)
            summary.append(dict(model=model,scope=label,total_candidates=len(rr),eligible_candidates=sum(r['eligible'] for r in rr),
                S_all=fixed_means[model,'S_all'],maximum=maximum['maximum'],R_all=fixed_means[model,'R_all'],
                maximum_minus_S_all=maximum['maximum']-fixed_means[model,'S_all'],R_all_minus_maximum=fixed_means[model,'R_all']-maximum['maximum'],
                exact_winner_ids=json.dumps(maximum['exact_winner_ids']),tolerance_tie_ids=json.dumps(maximum['tolerance_tie_ids'])))
            for name in maximum['tolerance_tie_ids']:
                winner=next(r for r in rr if r['candidate_id']==name)
                winners.append(dict(winner,maximum_scope=label,exact_maximum=name in maximum['exact_winner_ids']))
    candidates.sort(key=lambda r:(r['model'],r['candidate_id']))
    write_csv(small/'candidate_results.csv',candidates);write_csv(small/'summary.csv',summary)
    write_csv(small/'winning_candidates.csv',winners);write_csv(small/'overlap_checks.csv',overlap)
    write_json(external/'score_manifest.json',dict(files=score_files,rows=2700,candidates=600,
        boundary_row_binding_sha256=support['boundary_row_binding_sha256'],support=pin(external/'support.json'),specification=specification_pin))
    for module in list(sys.modules.values()):
        path=getattr(module,'__file__',None)
        if path and Path(path).resolve().is_relative_to(ROOT):record(path)
    record(Path(__file__))
    for path in ROOT.glob('scripts/*posthoc_reduced_bank_e.py'):record(path)
    write_json(small/'input_hashes.json',ledger)
    runtime=dict(wall_seconds=time.monotonic()-started_clock,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 peak_rss_units='bytes on macOS; whole analysis process',platform=platform.platform(),python=sys.version,numpy=np.__version__,
                 process_count=1,blas_thread_limit=1,threadpools=threadpool_info())
    require(all(r['num_threads']==1 for r in runtime['threadpools']),'BLAS thread limit not enforced')
    receipt=dict(status='completed_pending_independent_audit',started=started,runtime=runtime,source_base=BASE,
        b25_delivery=B25_DELIVERY,b25_completion=pin(bank_source.root/'completion.json'),b25_producer=B25_PRODUCER,
        parameter_producers=dict(collections.Counter(r['parameter_producer'] for r in candidates)),
        training_representation_manifest_sha256=bank['representation_manifest_sha256'],
        original_E_extraction_producer=completion['access']['producer_sha'],scientific_lock_sha256=completion['access']['lock_sha256'],
        historical_access_record_used_only_for_identity=True,new_authority='User request and separately saved post-hoc analysis_spec.json',
        raw_verification=raw_verification,coverage=dict(qwen=280,llama=320),overlap_heads=len(overlap),overlap_metric_rows=overlap_metric_checks,
        overlap_all_exact=all(r['exact_equal'] for r in overlap),external=str(external),score_manifest=pin(external/'score_manifest.json'),
        new_fits=0,new_extractions=0,full_train_candidates=0,new_bootstrap=0,procedure_or_lock_changes=False)
    write_json(small/'execution.json',receipt)
    write_json(external/'completion.json',dict(status='completed',small_output=str(small),execution=pin(small/'execution.json'),score_manifest=pin(external/'score_manifest.json')))
    print(json.dumps(summary,indent=2),flush=True);print(json.dumps(runtime,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--external',type=Path,required=True)
    p.add_argument('--output',type=Path,default=SMALL)
    args=p.parse_args()
    with open('/tmp/truth-probing-cpu-exclusive.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with threadpool_limits(limits=1):run(args)


if __name__=='__main__':main()
