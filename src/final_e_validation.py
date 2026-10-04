"""Development-only reproduction of frozen reports, no fitting or E output access."""
import argparse
import fcntl
import json
import os
import time
from pathlib import Path
import numpy as np
from src.final_e_inventory import ROOT, Metadata, read
from src.final_e_registry import resolve, apply_head
from src.final_e_reporting import scalar_result, scalar_tables, constituent_result, constituent_tables, schedules
from src.checkpoint_r2_fresh_inputs import hash_value, require
from src.checkpoint_r2_fresh_store import pin, write_json
from src.checkpoint_r2_b25_adapter import B25Adapter
from src.checkpoint_r2_fresh_adapter import check_packet


def close(a,b,label):
    if a is None or b is None:
        require(a is b,'undefined mismatch: '+label);return 0.
    if isinstance(a,(float,int)) and isinstance(b,(float,int)):
        require(np.isclose(a,b,rtol=0,atol=2e-12),'published D numeric mismatch: '+label)
        return abs(a-b)
    require(a==b,'published D metadata mismatch: '+label);return 0.


def compare_records(got, expected, keys, fields):
    key=lambda r:tuple(str(r[k]) for k in keys)
    lookup={key(r):r for r in expected};require(len(lookup)==len(expected),'duplicate expected endpoint')
    require(len({key(r) for r in got})==len(got),'duplicate computed endpoint')
    require({key(r) for r in got}==set(lookup),'published D endpoint coverage')
    maximum=0.
    for r in got:
        e=lookup[key(r)]
        for f in fields:
            if f in e:maximum=max(maximum,close(r.get(f),e[f],str(key(r))+':'+f))
    return dict(rows=len(got),max_abs_difference=maximum)


def source_files(roots):
    b=read(roots['b25_manifest']);c=read(roots['constituent_manifest']);f=read(ROOT/'results/checkpoint_next_full_bank_results_20261003/external-output-manifest.json')
    return dict(b25={k.removeprefix(Path(roots['b25']).name+'/'):v for k,v in b['artifacts'].items()},
        constituent={k.removeprefix(Path(roots['constituent']).name+'/'):v for k,v in c['files'].items()},
        full_bank={k.removeprefix('bank/'):v for k,v in f['files'].items()})


def head_application(adapter, registry, roots, checked_path):
    """Bounded saved-D features: apply every selected head, no fit/inference."""
    from src.checkpoint_r2_fresh_store import load_features
    groups=('atomic_D','D_bare');selected={g:list(range(min(16,len(adapter.rows[g])))) for g in groups}
    wanted={adapter.bindings[g][i]['text_row'] for g in groups for i in selected[g]}
    matrices={}
    for model in ('qwen','llama'):
        found={};offset=0
        for shard in adapter.indices[model]['shards']:
            n=shard['rows'];hit=sorted(wanted & set(range(offset,offset+n)))
            if hit:
                folder=adapter.output/model/'shards'/shard['path']
                require(pin(folder/'receipt.json')==shard['receipt'],'accepted D raw receipt')
                values,_,_=load_features(folder,expected_rows=adapter.manifest['rows'][offset:offset+n],expected_spec=adapter.cfg['models'][model])
                for j in hit:found[j]=np.array(values[j-offset],dtype=np.float64)
                del values
            offset+=n
        require(set(found)==wanted,'D bounded application coverage')
        matrices[model]={g:np.array([found[adapter.bindings[g][i]['text_row']] for i in selected[g]]) for g in groups}
    checks=[]
    for name,h in registry['heads'].items():
        if h['family']=='scalar':
            folder='evaluations/'+hash_value(h['frozen_id'])[:24]
            with np.load(checked_path(h['parameters']['source'],folder+'/scores.npz'),allow_pickle=False) as z:
                for g in groups:
                    expected=z[g][selected[g]];actual=apply_head(h,matrices[h['model']][g][:,h['saved_layer'],:],roots)
                    require(np.allclose(actual,expected,rtol=1e-9,atol=1e-8),'saved D head/preprocessing application '+name)
                    checks.append(dict(head=name,group=g,rows=len(actual),max_abs_difference=float(np.max(np.abs(actual-expected)))))
        else:
            # Every final head occurs in at least one locked bare score packet.
            case=next(c for c in registry['constituent_cases'] if name in c['heads'])
            spec=dict(heads=[registry['heads'][n]['frozen_id'] for n in case['heads']],group='D_bare',operator=case['evaluation_operator'])
            folder='scores/'+hash_value(hash_value(spec))[:24]
            binding=read(checked_path('constituent',folder+'/binding.json'))
            expected=np.load(checked_path('constituent',folder+'/scores.npy'),allow_pickle=False)
            lookup={tuple(k):i for i,k in enumerate(binding['packet']['ordered_keys'])}
            chosen=[j for j in selected['D_bare'] if adapter.rows['D_bare'][j]['operator']==case['evaluation_operator']]
            col=case['heads'].index(name)
            target=np.array([expected[lookup[('D_bare',adapter.bindings['D_bare'][j]['logical_id'])],col] for j in chosen])
            actual=apply_head(h,matrices[h['model']]['D_bare'][chosen,h['saved_layer'],:],roots)
            require(np.allclose(actual,target,rtol=1e-9,atol=1e-8),'constituent saved standardizer/head '+name)
            checks.append(dict(head=name,group='D_bare',rows=len(actual),max_abs_difference=float(np.max(np.abs(actual-target)))))
    return checks


def run(roots, destination, recompute_draws=False):
    started=time.monotonic();destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    registry=read(ROOT/'config/final_e/head_registry_v1.json');sources=source_files(roots)
    def path(source,rel):
        p=Path(roots[source])/rel
        require(pin(p)==sources[source][rel],'published artifact '+source+'/'+rel)
        return p
    adapter=B25Adapter(read(ROOT/'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json'))
    groups=['atomic_D','D_bare']+sorted(adapter.receipt['heldout_wording_rows'])
    rows={g:adapter.rows[g] for g in groups}
    options=read(ROOT/'config/final_e/preparation_v1.json')['bootstrap']
    with np.load(path('b25','bootstrap_weights.npz'),allow_pickle=False) as z:weights={g:z[g] for g in groups}
    # Reconstruct the complete inherited D schedules, not just load and trust them.
    from src.selection_repair_canonical_sensitivity import atomic_weights,compound_weights
    import pandas as pd
    aw=atomic_weights(rows['atomic_D'],np.ones(len(rows['atomic_D']),bool),options)
    cw,_=compound_weights(pd.DataFrame(rows['D_bare']).rename(columns={'person_a':'entity_a_id','person_b':'entity_b_id'}),np.ones(len(rows['D_bare']),bool),options)
    require(np.array_equal(aw,weights['atomic_D']) and np.array_equal(cw,weights['D_bare']),'D schedule reconstruction')
    applications=head_application(adapter,registry,roots,path)
    scalar={};point_checks=[];draw_checks=[]
    for n,h in registry['heads'].items():
        if h['family']!='scalar':continue
        source=h['parameters']['source'];folder='evaluations/'+hash_value(h['frozen_id'])[:24]
        binding=read(path(source,folder+('/score_binding.json' if source=='b25' else '/score-binding.json')))
        require(binding['fit_id']==h['frozen_id'],'frozen score head')
        scores_path=path(source,folder+'/scores.npz');require(pin(scores_path)==binding['artifact'],'saved score pin')
        with np.load(scores_path,allow_pickle=False) as z:
            vals={g:check_packet(adapter.bindings[g],dict(binding['groups'][g],scores=z[g])) for g in groups}
        got=scalar_result(rows,vals,weights if recompute_draws else None,options)
        saved=read(path(source,folder+'/metrics.json'))['records']
        point_checks.append(compare_records(got['records'],saved,('condition','metric','scope','topic'),('point',)))
        drawname='bootstrap_metrics.npy' if (Path(roots[source])/folder/'bootstrap_metrics.npy').exists() else 'bootstrap-metrics.npy'
        d=np.load(path(source,folder+'/'+drawname),allow_pickle=False)
        if recompute_draws:
            require(np.allclose(got['draws'],d,rtol=0,atol=2e-12,equal_nan=True),'recomputed D bootstrap draws')
            draw_checks.append(dict(head=n,draws=int(d.size),maximum=float(np.nanmax(np.abs(got['draws']-d)))))
        else:got['draws']=d
        # Interval aggregation still runs through the successor reporting engine.
        from src.clean_transfer_statistics import interval
        for i,r in enumerate(got['records']):r.update(interval(got['draws'][i],options))
        scalar[n]=got
        print(json.dumps(dict(head=n,point_rows=len(saved),draws_recomputed=recompute_draws,seconds=time.monotonic()-started)),flush=True)
    sm,sc=scalar_tables(registry,scalar,read(ROOT/'config/final_e/contrasts_v1.json'),options)
    fields=('point','ci_low','ci_high','valid_replicates','invalid_replicates','total_replicates','valid_fraction','ci_status','retention_margin','point_retained','interval_supported')
    reports={}
    for panel in ('b25','full_bank'):
        for kind,got in [('metrics',sm),('contrasts',sc)]:
            expected=read(path(panel,kind+'.json'))['records']
            actual=[dict(r,seed=r['sample']) for r in got if r['panel']==panel]
            # Full-bank original arms are explicit aliases in the new report.
            keys=('model','seed','arm','condition','metric','scope','topic') if kind=='metrics' else ('model','seed','left','right','condition','metric','scope','topic')
            reports[panel+'_'+kind]=compare_records(actual,expected,keys,fields)
    # Exact final constituent panel only; no source-validation curves or layer search.
    expected=read(path('constituent','metrics.json'))['records'];cases=registry['constituent_cases'];cresults={};done={}
    for case in cases:
        match=[r for r in expected if all(r[k]==case[k] for k in ('model','source_operator','target_operator','arm','trained_operator','evaluation_operator','saved_layer'))]
        cresults[hash_value(case)]={}
        for group in sorted({r['condition'] for r in match}):
            folder=next(r['score_folder'] for r in match if r['condition']==group)
            if folder not in done:
                binding=read(path('constituent',folder+'/binding.json'))
                require(binding['head_ids']==[registry['heads'][k]['frozen_id'] for k in case['heads']],'constituent head ordering')
                vals=np.load(path('constituent',folder+'/scores.npy'),allow_pickle=False)
                indices=[i for i,r in enumerate(rows[group]) if r['operator']==case['evaluation_operator']]
                rs=[rows[group][i] for i in indices]
                bindings=[adapter.bindings[group][i] for i in indices];packet=binding['packet']
                require(packet['row_binding_sha256']==hash_value(bindings) and packet['ordered_keys']==[[b['group'],b['logical_id']] for b in bindings]
                    and packet['model']==case['model'] and packet['saved_layer']==case['saved_layer']
                    and packet['operator']==case['evaluation_operator'] and packet['representation_index_sha256']==pin(adapter.output/case['model']/'index.json')['sha256'], 'constituent representation/binding/order')
                require(pin(path('constituent',folder+'/scores.npy'))==binding['artifact'], 'constituent score pin')
                got=constituent_result(rs,vals,case['evaluation_operator'],weights[group][:,indices] if recompute_draws else None,options)
                saved=read(path('constituent',folder+'/metrics.json'))['records']
                point_checks.append(compare_records(got['records'],saved,('metric','scope','topic','surface_order'),('point','observations','status')))
                d=np.load(path('constituent',folder+'/bootstrap-metrics.npy'),allow_pickle=False)
                if recompute_draws:require(np.allclose(got['draws'],d,rtol=0,atol=2e-12,equal_nan=True),'constituent D draws')
                else:
                    got['draws']=d;j=0
                    from src.checkpoint_r2_constituent_metrics import absolute_status
                    for r in got['records']:
                        if r['surface_order']=='all':
                            r.update(interval(d[j],options));j+=1
                            if 'absolute_classification' in r:r['absolute_classification']=absolute_status(r)
                done[folder]=got
            cresults[hash_value(case)][group]=done[folder]
    cm,cc=constituent_tables(cases,cresults,options)
    reports['constituent_metrics']=compare_records(cm,expected,('model','source_operator','arm','condition','metric','scope','topic','surface_order'),fields+('observations','status','absolute_classification'))
    reports['constituent_contrasts']=compare_records(cc,read(path('constituent','paired-contrasts.json'))['records'],('model','source_operator','condition','metric','scope','topic','surface_order'),fields+('classification','planning_margin'))
    adapter.unchanged()
    result=dict(status='passed',seconds=time.monotonic()-started,registry_sha256=pin(ROOT/'config/final_e/head_registry_v1.json')['sha256'],
        saved_head_application=applications,scalar_heads=len(scalar),constituent_score_panels=len(done),all_point_checks=point_checks,reports=reports,
        bootstrap_recomputed=recompute_draws,draw_checks=draw_checks,D_schedules_reconstructed=True,E_outputs_opened=False,fits=0,
        one_process=True,threads=1)
    write_json(destination/'D-reproduction.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',required=True);p.add_argument('--output',required=True);p.add_argument('--recompute-draws',action='store_true')
    args=p.parse_args()
    require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')),'one thread')
    with open('/tmp/truth-probing-cpu-exclusive.lock','a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(run(read(args.roots),args.output,args.recompute_draws)['reports'],indent=2))


if __name__=='__main__':main()
