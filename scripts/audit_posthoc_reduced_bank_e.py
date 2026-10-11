#!/usr/bin/env python3
"""Independent saved-score audit. No refitting, re-extraction or new bootstrap."""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
import csv
import fcntl
import json
from pathlib import Path
import resource
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits,threadpool_info
from src.posthoc_reduced_bank_e import independent_auc, require
from src.checkpoint_r2_fresh_inputs import hash_value
from src.checkpoint_r2_fresh_store import pin
from src.final_e_inventory import Metadata,read


def rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def audit(args):
    started=time.monotonic();small=args.results.resolve()
    execution=read(small/'execution.json');external=Path(execution['external'])
    require(not (small/'audit.json').exists(),'preserve prior audit; choose fresh output after a failed audit')
    manifest=read(external/'score_manifest.json');support=read(external/'support.json')
    require(pin(external/'score_manifest.json')==execution['score_manifest'],'score manifest changed')
    require(pin(external/'support.json')==manifest['support'],'row binding artifact changed')
    require(pin(small/'analysis_spec.json')==manifest['specification'],'specification changed')
    input_hashes=read(small/'input_hashes.json')
    def verified(path):
        p=Path(path).resolve();require(pin(p)==input_hashes[str(p)],'source changed: '+str(p));return p
    metadata=Metadata()
    # Reconstruct labels/row order from accepted metadata, not from scoring CSVs.
    expected=[];selected=[]
    for i,r in enumerate(metadata.rows['bare']):
        if r['operator']=='OR' and (r['truth_a'],r['truth_b']) in ((0,0),(0,1),(1,0)):
            selected.append(i)
            expected.append(dict(example_id=r['example_id'],pair_id=r['pair_id'],topic=r['topic'],ordering=r['ordering'],
                truth_a=r['truth_a'],truth_b=r['truth_b'],label=int(bool(r['truth_a'] or r['truth_b'])),
                text_id=metadata.group_bindings['bare'][i]['text_id'],bare_row_index=i))
    require(expected==support['rows'] and selected==support['bare_indices'],'independent row/order/label reconstruction')
    require(len(expected)==2700 and len({r['pair_id'] for r in expected})==450,'audit E support')
    require(hash_value(expected)==support['boundary_row_binding_sha256']==manifest['boundary_row_binding_sha256'],'boundary identity digest')
    labels=np.array([r['label'] for r in expected]);require(labels.sum()==1800,'audit positive count')
    data=rows(small/'candidate_results.csv')
    # Reconstruct the complete grid independently, not by calling scoring inventory helper.
    expected_ids=set()
    for model,layers in [('qwen',28),('llama',32)]:
        for layer in range(layers):
            prefix=f'{model}/P15/L{layer:02d}/'
            expected_ids.update(prefix+'l2_logistic/C='+c for c in ['0.001','0.01','0.1','1','10'])
            expected_ids.update(prefix+m for m in ['difference_of_means','mass_mean_covariance','burger_t_g','ttpd','r0'])
    require(len(data)==len(expected_ids)==600 and {r['candidate_id'] for r in data}==expected_ids,'independent complete grid coverage')
    bankpath=next(Path(p) for p in input_hashes if p.endswith('run-5e1bf47/bank.json'))
    bank=read(verified(bankpath));bankrows={r['id']:r for r in bank['records']}
    lockpath=bankpath.parent/'locked.json';locked=read(verified(lockpath))
    expected_score_files=set();points={};eligible={};max_error=0.
    for r in data:
        cid=r['candidate_id'];model=r['model'];original=bankrows[cid]
        require(r['status']=='scored' and r['bank']=='reduced_P15','invalid row status')
        require(r['row_binding_sha256']==hash_value(expected),'candidate support/order changed')
        require(int(r['saved_layer'])==original['spec']['saved_layer'] and int(r['hf_hidden_state_index'])==int(r['saved_layer'])+1,'saved layer indexing')
        require(r['method']==original['spec']['method'],'method changed')
        old_C=original['spec']['C'];require((r['C']=='' if old_C is None else float(r['C'])==old_C),'regularization setting changed')
        atom=next(v['point'] for v in original['metrics'] if v['metric']=='atomic_auroc' and v['scope']=='pooled')
        topics=[v['point'] for v in original['metrics'] if v['metric']=='atomic_auroc' and v['scope']=='topic']
        reference=locked['selection']['shared_atomic_banks'][model]['reference_atomic']
        allowed=bool(original['valid_for_scoring'] and len(topics)==5 and all(np.isfinite([atom,*topics])) and atom>=reference-.005)
        olddecision=next(x['eligible'] for x in locked['selection']['shared_atomic_banks'][model]['eligibility'] if x['candidate_id']==cid)
        require(allowed==olddecision==(r['eligible']=='True') and float(r['atomic_D_auroc'])==atom,'independent original D gate mismatch')
        expected_score_files.add(r['score_file']);path=external/r['score_file']
        require(pin(path)==manifest['files'][r['score_file']] and pin(path)['sha256']==r['score_sha256'],'saved score hash changed')
        score=np.load(path,allow_pickle=False);require(score.shape==(2700,) and np.isfinite(score).all(),'score support/finite')
        auc=independent_auc(labels,score);error=abs(auc-float(r['E_auroc']));max_error=max(error,max_error)
        require(error<=1e-14,'independent average-rank AUROC mismatch')
        points[cid]=auc;eligible[cid]=allowed
    require(expected_score_files==set(manifest['files']),'extra or missing score array')
    # Recompute fixed allocation AUROCs from original accepted scores, then average.
    registry=read(verified(ROOT/'config/final_e/head_registry_v1.json'))
    handoff=read(verified(ROOT/'results/final_e_scientific_evaluation_20261005/evaluation-handoff.json'))
    evaluation=Path(handoff['report']);accepted=read(verified(evaluation/'scalar-metrics.json'))['records']
    fixed=rows(small/'fixed_procedures.csv');means={};fixed_count=0
    for model in ('qwen','llama'):
        for arm in ('S_all','R_all'):
            allocation_values=[]
            for sample in (11,23,37):
                alias=next(r for r in registry['scalar_aliases'] if (r['panel'],r['model'],r['arm'],r['sample'])==('b25',model,arm,sample))
                folder=evaluation/'scores'/hash_value(alias['head'])[:24]
                binding=read(verified(folder/'binding.json'))
                require(binding['group_bindings']==metadata.receipt['ordered_groups'],'fixed score row binding')
                with np.load(verified(folder/'scores.npz'),allow_pickle=False) as z:score=z['bare'][selected]
                auc=independent_auc(labels,score);allocation_values.append(auc)
                rr=next(r for r in fixed if (r['model'],r['arm'],r['sample'])==(model,arm,str(sample)))
                require(rr['head']==alias['head'] and abs(float(rr['E_auroc'])-auc)<=1e-14,'fixed allocation mismatch')
                old=next(r for r in accepted if (r['panel'],r['model'],r['arm'],r['sample'],r['condition'],r['metric'],r['scope'])==('b25',model,arm,sample,'bare','OR_mixed_vs_FF_auroc','pooled'))
                require(abs(old['point']-auc)<=1e-14,'accepted fixed metric mismatch');fixed_count+=1
            mean=sum(allocation_values)/3
            rr=next(r for r in fixed if (r['model'],r['arm'],r['sample'])==(model,arm,'mean_11_23_37'))
            old=next(r for r in accepted if (r['panel'],r['model'],r['arm'],r['sample'],r['condition'],r['metric'],r['scope'])==('b25',model,arm,'mean_11_23_37','bare','OR_mixed_vs_FF_auroc','pooled'))
            require(abs(mean-float(rr['E_auroc']))<=1e-14 and abs(mean-old['point'])<=1e-14,'fixed mean mismatch')
            means[model,arm]=mean
    summary=rows(small/'summary.csv');winners=rows(small/'winning_candidates.csv')
    require(len(summary)==4,'two scopes per model')
    reconstructed=[]
    for model in ('qwen','llama'):
        all_ids=[cid for cid in expected_ids if cid.startswith(model+'/')]
        require(len(all_ids)==(280 if model=='qwen' else 320),'model candidate count')
        for scope in ('eligible','unrestricted'):
            ids=[cid for cid in all_ids if scope=='unrestricted' or eligible[cid]]
            value=max(points[cid] for cid in ids)
            exact=sorted(cid for cid in ids if points[cid]==value)
            near=sorted(cid for cid in ids if value-points[cid]<=1e-12)
            r=next(r for r in summary if (r['model'],r['scope'])==(model,scope))
            require(json.loads(r['exact_winner_ids'])==exact and json.loads(r['tolerance_tie_ids'])==near,'independent maximum/tie identities mismatch')
            matching=[r for r in winners if r['model']==model and r['maximum_scope']==scope]
            require(sorted(r['candidate_id'] for r in matching)==near,'winner table coverage')
            require(sorted(r['candidate_id'] for r in matching if r['exact_maximum']=='True')==exact,'exact winner flags')
            expected_values=dict(S_all=means[model,'S_all'],maximum=value,R_all=means[model,'R_all'],
                maximum_minus_S_all=value-means[model,'S_all'],R_all_minus_maximum=means[model,'R_all']-value)
            require(all(abs(float(r[k])-v)<=1e-14 for k,v in expected_values.items()),'independent descriptive differences mismatch')
            reconstructed.append(dict(model=model,scope=scope,**expected_values,exact_winner_ids=exact,tolerance_tie_ids=near))
    result=dict(status='PASS',candidate_AUROCs_independently_recomputed=600,candidate_coverage=dict(qwen=280,llama=320),
        independent_method='Mann–Whitney average-rank sum with half-credit exact score ties; no production AUC function or maximum helper called',
        maximum_AUROC_error=max_error,original_D_eligibility_independently_verified=True,all_row_label_order_bindings_verified=True,
        fixed_allocation_AUROCs_independently_verified=fixed_count,means_of_three_AUROCs_verified=4,
        maxima_and_all_differences=reconstructed,new_bootstrap=0,new_fits=0,
        run_execution=pin(small/'execution.json'),score_manifest=pin(external/'score_manifest.json'),
        wall_seconds=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        process_count=1,blas_thread_limit=1,threadpools=threadpool_info(),audit_script=pin(Path(__file__)))
    require(all(r['num_threads']==1 for r in result['threadpools']),'audit thread count')
    with (small/'audit.json').open('x') as f:json.dump(result,f,indent=2,sort_keys=True);f.write('\n')
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=ROOT/'results/posthoc_reduced_bank_e_20261011')
    args=p.parse_args()
    with open('/tmp/truth-probing-cpu-exclusive.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        with threadpool_limits(limits=1):audit(args)


if __name__=='__main__':main()
