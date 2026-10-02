#!/usr/bin/env python3
"""Rebuild/validate preparation schemas, inventory, timings and cost assumptions.

Never runs a fitting worker. Timing files must come from the frozen diagnostic
runner; production readiness remains false until a separate reviewed handoff.
"""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src import checkpoint_r2_cpu_preflight_v1 as p
from src import checkpoint_r2_result_schema_v1 as schema


def write(path, value):
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def check(output, timings):
    cfg=p.config(ROOT)
    inventory_path=output/'reusable_components_and_inputs.json'
    if inventory_path.exists():
        pinned=json.loads(inventory_path.read_text())
        for component in pinned['components']:
            for category in ('code','inputs'):
                for name,pin in component[category].items():
                    p.bound(ROOT/name,pin)
        for record in pinned['additional_local_export_headers']:
            p.bound(record['path'],record['binding'])
            a=np.load(record['path'],mmap_mode='r',allow_pickle=False)
            p.require(list(a.shape)==record['shape'] and a.dtype.str==record['dtype'],'export header mismatch')
    bank=pd.read_csv(output/'bank_candidates.csv')
    expected=p.grid(cfg)
    p.require(bank.candidate_id.tolist()==[r['candidate_id'] for r in expected],'bank identity mismatch')
    reuse=p.saved_fit_inventory(ROOT)
    p.require(reuse==json.loads((output/'conditional_reuse.json').read_text()),'reuse inventory changed')
    fold=pd.read_csv(output/'planned_repair_folds.csv')
    expected_folds={(m,l,seed,f) for m,s in cfg['models'].items() for l in s['layers'] for seed in [11,23,37] for f in range(5)}
    p.require(set(fold[['model','layer','sample_seed','fold']].itertuples(index=False,name=None))==expected_folds and len(fold)==900,'fold coverage mismatch')
    write(output/'result_schema.json',schema.SCHEMA)
    pd.DataFrame(schema.templates()).to_csv(output/'unrun_result_templates.csv',index=False)
    summary=json.loads((timings/'summary.json').read_text())
    p.require(summary['config']==p.identity(ROOT/p.CONFIG),'timing config mismatch')
    p.require(summary['loader']==p.identity(ROOT/'src/checkpoint_r2_cpu_preflight_v1.py') and summary['runner']==p.identity(ROOT/'scripts/66_checkpoint_r2_cpu_preflight.py'),'timing implementation changed')
    p.require(len(summary['cases'])==18 and len({r['case'] for r in summary['cases']})==18,'timing case coverage')
    for r in summary['cases']:
        path=timings/(r['case'].replace(':','_')+'.json')
        p.require(json.loads(path.read_text())==r,'timing checkpoint mismatch')
        if r['status']=='converged':
            p.require(r['details']['valid_for_scoring'] and not r['D_metrics_computed'] and not r['params_saved'],'invalid timing scope')
            p.require(all(x['num_threads']==1 for x in r['threads']),'thread count changed')
    write(output/'timing_summary.json',summary)
    flat=[]
    for r in summary['cases']:
        if r['status']!='converged':
            flat.append(dict(case=r['case'],status=r['status'],error=r.get('error')));continue
        d=r['details'];attempts=d.get('attempts',[])
        flat.append({**{k:r[k] for k in ['case','kind','model','method','saved_layer','fitting_rows','width','dtype','adaptation_rows','status','wall_fit_including_preprocessing_seconds','peak_process_rss_bytes','preprocessing']},
                     'iterations':sum(a['iterations'] for a in attempts) if attempts else d.get('final_n_iter'),
                     'gradient_infinity_norm':attempts[-1]['gradient_infinity_norm'] if attempts else d.get('gradient_infinity_norm'),
                     'convergence_kind':'iterative' if attempts or 'final_n_iter' in d else 'closed_form'})
    pd.DataFrame(flat).to_csv(output/'timings.csv',index=False)
    old=json.loads((ROOT/'results/t2_capacity_pilot_v1_20261001/fits.json').read_text())
    bycase={r['case']:r for r in summary['cases']}
    costs=[]
    def seconds(kind,model,method):
        r=bycase[f'{kind}:{model}:{method}']
        p.require(r['status']=='converged','cost projection unsupported: missing representative timing')
        return r['wall_fit_including_preprocessing_seconds']
    for model,spec in cfg['models'].items():
        layers=len(spec['layers'])
        # Raw core conditional on all 36 exact reuse gates passing. No settings
        # chosen from outcome metrics; broad timing sensitivity factors are not
        # claimed measured confidence or guaranteed runtime bounds.
        bank_low=bank_high=0
        for method in ['l2_logistic',*cfg['other_methods']]:
            count=5*layers-3 if method=='l2_logistic' else layers-3
            if method in ('l2_logistic','r0','mass_mean_covariance'):
                low=seconds('historical',model,method)
                high=max(low,seconds('synthetic',model,method))
            else:
                prior=[r['elapsed_seconds'] for r in old if r['configuration']['model']==model and r['configuration']['cohort']=='P15' and r['configuration']['method']==method]
                low,high=min(prior),max(prior)
            bank_low+=count*low;bank_high+=count*high*3
        costs.append(dict(model=model,family='additional_bank',fits=10*layers-18,low_seconds=bank_low,high_seconds=bank_high,
                          assumption='36 historical matches accepted total; unmeasured C/layer spread: high is 3x max current synthetic/historical timing; DoM/t_G/TTPD supplemented by prior pilot costs'))
        repair=seconds('synthetic',model,'synthetic_compound_objective')
        mean=max(seconds('synthetic',model,'r0'),seconds('historical',model,'r0'))
        families=[('repair_folds',layers*15,repair,1),('repair_refits_upper',9,repair,3178/3098),
                  ('isolated_controls_upper',6,repair,2878/3098),('C_clean',layers,mean,8000/2778),
                  ('constituent_source',layers*4,mean,4000/2778),('constituent_full_refits_upper',8,mean,4000/2778)]
        for family,n,rate,scale in families:
            costs.append(dict(model=model,family=family,fits=n,low_seconds=n*rate*scale,high_seconds=n*rate*scale*10,
                              assumption='Dense float64 row-cost scaling from fresh diagnostics; 1x–10x conditioning/iteration sensitivity, not actual TC/B25 timing or guaranteed bound'))
    pd.DataFrame(costs).to_csv(output/'cpu_cost_scenarios.csv',index=False)
    low=sum(r['low_seconds'] for r in costs);high=sum(r['high_seconds'] for r in costs)
    chat_low=sum(2*max(seconds('synthetic',m,'r0'),seconds('historical',m,'r0'))*(1+8000/2778) for m in cfg['models'])
    write(output/'cpu_cost_assumptions.json',dict(status='conditional_planning_scenario_not_production_runtime_measurement',
          serial_single_BLAS_thread=True,estimated_new_raw_core_fits_upper=sum(r['fits'] for r in costs),
          fitting_low_seconds=low,fitting_high_seconds=high,
          unmeasured_io_scoring_bootstrap_reserve_seconds=[1800,7200],
          raw_core_scenario_hours=[(low+1800)/3600,(high+7200)/3600],
          optional_Prompt1_chat_CPU_fits=8,chat_CPU_scenario_seconds=[chat_low,10*chat_low],
          total_CPU_with_chat_scenario_hours=[(low+chat_low+1800)/3600,(high+10*chat_low+7200)/3600],
          bounds_depend_on='All reuse/bridge gates pass; comparable hardware/software; no severe optimization tail; serial fits, layer streaming; no GPU/extraction costs; chat CPU fits are an additional shape-scaled scenario',
          convergence_tail='10000-iteration caps can exceed these planning scenarios by orders of magnitude; stop for cost review if representative production fits exceed scenario rather than relaxing convergence or silently dropping fits',
          unmeasured=['all five Cs across all layers','actual mixed TC/constituent condition numbers','actual repair A blocks and fold/refit convergence','production score output IO','all B25/transfer bootstrap tables','chat fits'],
          memory=dict(measured_max_RSS_bytes=max(r.get('peak_process_rss_bytes',0) for r in summary['cases']),
                      local_RAM_bytes=34359738368,logical_CPUs=10,planned_processes=1,
                      provisional_per_worker_budget_bytes=8*2**30,note='8 GiB is a planning ceiling for streamed larger blocks, not a measured production peak; avoid all-layer/all-score materialization'),
          no_production_fit_started=True))
    write(output/'preparation_validation.json',dict(status='passed',base=p.BASE,common_checkpoint_sha=None,bank_ids_checked=600,
          per_model={'qwen':280,'llama':320},conditional_fit_files_and_parameters_verified=36,fold_ids_checked=900,
          schema_rows_checked=len(schema.templates()),benchmark_cases=18,
          benchmark_valid=sum(r['status']=='converged' for r in summary['cases']),
          current_code_dependencies_match_frozen_config=True,P_A_frozen=False,production_executable=False,
          timing_external={str(x):p.identity(x) for x in sorted(timings.glob('*.json'))}))
    print(json.dumps(dict(status='passed',fits=sum(r['fits'] for r in costs),fitting_minutes=[low/60,high/60],scenario_hours=[(low+1800)/3600,(high+7200)/3600]),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timings',type=Path,required=True)
    args=parser.parse_args();check(args.output,args.timings)
