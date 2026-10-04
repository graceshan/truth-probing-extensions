"""One serialized saved-D reference calculation; no E I/O or fitting."""
import argparse,fcntl,json,os,time,resource
from pathlib import Path
import numpy as np
from src.final_e_inventory import ROOT,read
from src.final_e_registry import resolve
from src.final_e_validation import source_files
from src.final_e_reference import Checks,constituent_endpoint,principal
from src.checkpoint_next_full_bank_audit import endpoint
from src.checkpoint_r2_b25_adapter import B25Adapter
from src.checkpoint_r2_fresh_store import pin,write_json
from src.checkpoint_r2_fresh_adapter import check_packet
from src.checkpoint_r2_fresh_inputs import hash_value,require


def run(roots,destination):
    start=time.monotonic();out=Path(destination);out.mkdir(parents=True,exist_ok=False)
    registry=read(ROOT/'config/final_e/head_registry_v1.json');sources=source_files(roots);checks=Checks()
    # Reconstruct all identities from the reviewed delivery manifests, not guesses.
    def path(src,rel):
        p=Path(roots[src])/rel;require(pin(p)==sources[src][rel],'reviewed saved D artifact');return p
    adapter=B25Adapter(read(ROOT/'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json'))
    opts=read(ROOT/'config/final_e/preparation_v1.json')['bootstrap']
    with np.load(path('b25','bootstrap_weights.npz'),allow_pickle=False) as z:weights={g:z[g].copy() for g in z.files}
    cached={};heads=0;scalar_checks=0
    for name,h in registry['heads'].items():
        if h['family']!='scalar':continue
        for ref in (h['fit'],h['parameters']):resolve(ref,roots)
        src=h['parameters']['source'];folder='evaluations/'+hash_value(h['frozen_id'])[:24]
        binding=read(path(src,folder+('/score_binding.json' if src=='b25' else '/score-binding.json')))
        scorespath=path(src,folder+'/scores.npz');require(pin(scorespath)==binding['artifact'],'saved D score pin')
        records=read(path(src,folder+'/metrics.json'))['records']
        drawname='bootstrap_metrics.npy' if src=='b25' else 'bootstrap-metrics.npy'
        draws=np.load(path(src,folder+'/'+drawname),allow_pickle=False)
        with np.load(scorespath,allow_pickle=False) as z:
            for i,r in enumerate(records):
                if not principal(r):continue
                g=r['condition'];s=check_packet(adapter.bindings[g],dict(binding['groups'][g],scores=z[g]))
                point=endpoint(adapter.rows[g],s,r['metric'],r['scope'],r['topic'])[0]
                d=endpoint(adapter.rows[g],s,r['metric'],r['scope'],r['topic'],weights[g])
                checks.compare([point],[np.nan if r['point'] is None else r['point']]);checks.compare(d,draws[i]);checks.interval(r,d,opts)
                cached[name,g,r['metric'],r['scope'],r['topic']]=(point,d);scalar_checks+=1
        heads+=1
    alias={(a['panel'],a['model'],a['arm'],str(a['sample'])):a['head'] for a in registry['scalar_aliases']}
    for m in ('qwen','llama'):
        for arm in ('R_all','S_all','reduced_LR','bank_oracle'):
            for seed in (11,23,37):alias['full_bank',m,arm,str(seed)]=alias['b25',m,arm,str(seed)]
    def value(panel,model,arm,sample,row):
        seeds=(11,23,37) if sample=='mean_11_23_37' else (int(sample),)
        vals=[cached[alias[panel,model,arm,str(seed)],row['condition'],row['metric'],row['scope'],row['topic']] for seed in seeds]
        return np.mean([v[0] for v in vals]),np.mean([v[1] for v in vals],axis=0)
    paired=0;means=0;retention=0
    for panel in ('b25','full_bank'):
        for r in read(path(panel,'metrics.json'))['records']:
            if not principal(r) or r['seed']!='mean_11_23_37':continue
            p,d=value(panel,r['model'],r['arm'],str(r['seed']),r)
            checks.compare([p],[np.nan if r['point'] is None else r['point']]);checks.interval(r,d,opts);means+=1
        for r in read(path(panel,'contrasts.json'))['records']:
            if not principal(r):continue
            if r['model']=='qwen_minus_llama':
                qp,qd=value(panel,'qwen','R_all',str(r['seed']),r);sp,sd=value(panel,'qwen','S_all',str(r['seed']),r)
                lp,ld=value(panel,'llama','R_all',str(r['seed']),r);rp,rd=value(panel,'llama','S_all',str(r['seed']),r)
                lp,ld,rp,rd=qp-sp,qd-sd,lp-rp,ld-rd
            else:
                lp,ld=value(panel,r['model'],r['left'],str(r['seed']),r);rp,rd=value(panel,r['model'],r['right'],str(r['seed']),r)
            checks.compare([lp-rp],[np.nan if r['point'] is None else r['point']]);checks.interval(r,ld-rd,opts);checks.retention(r)
            paired+=1;retention+=int('retention_margin' in r)
    # Every final saved constituent panel, with explicit AB/BA and Boolean rules.
    expected=read(path('constituent','metrics.json'))['records'];folders=sorted({r['score_folder'] for r in expected});constituent_records=0
    for folder in folders:
        binding=read(path('constituent',folder+'/binding.json'));scores=np.load(path('constituent',folder+'/scores.npy'),allow_pickle=False)
        require(pin(path('constituent',folder+'/scores.npy'))==binding['artifact'],'saved constituent scores')
        packet=binding['packet'];group=packet['group'];operator=packet['operator']
        indices=[i for i,r in enumerate(adapter.rows[group]) if r['operator']==operator];rows=[adapter.rows[group][i] for i in indices]
        bindings=[adapter.bindings[group][i] for i in indices]
        require(packet['row_binding_sha256']==hash_value(bindings) and packet['ordered_keys']==[[b['group'],b['logical_id']] for b in bindings],'constituent support/order')
        records=read(path('constituent',folder+'/metrics.json'))['records'];draws=np.load(path('constituent',folder+'/bootstrap-metrics.npy'),allow_pickle=False)
        j=0
        for r in records:
            if r['scope'] in ('pooled','topic_macro'):
                p=constituent_endpoint(rows,scores,r['metric'],r['scope'],r['topic'],r['surface_order'])[0]
                checks.compare([p],[np.nan if r['point'] is None else r['point']]);constituent_records+=1
                if r['surface_order']=='all':
                    d=constituent_endpoint(rows,scores,r['metric'],r['scope'],r['topic'],weights=weights[group][:,indices])
                    checks.compare(d,draws[j]);checks.interval(r,d,opts)
            if r['surface_order']=='all':j+=1
    adapter.unchanged()
    receipt=dict(checks.receipt(),scalar_heads=heads,scalar_endpoints=scalar_checks,mean_endpoints=means,paired_contrasts=paired,
        retention_decisions=retention,constituent_panels=len(folders),constituent_records=constituent_records,replicates=2000,
        seconds=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        E_outputs_opened=False,new_fits=0,implementation='existing full-bank sorted-pair AUROC reference plus explicit constituent confusion/Boolean/min-max arithmetic')
    write_json(out/'independent-D-reference.json',receipt);return receipt


def main():
    p=argparse.ArgumentParser();p.add_argument('--roots',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    require(all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')),'one thread')
    with open('/tmp/truth-probing-cpu-exclusive.lock','a') as slot:
        fcntl.flock(slot,fcntl.LOCK_EX|fcntl.LOCK_NB)
        print(json.dumps(run(read(args.roots),args.output),indent=2))


if __name__=='__main__':main()
