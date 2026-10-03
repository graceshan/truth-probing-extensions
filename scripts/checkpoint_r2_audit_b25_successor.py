"""Read-only independent audit: fresh B25 successor checkpoint and principal results.
Usage: python audit-successor.py CHECKOUT RUN OUTPUT
"""
import json,sys,hashlib,time,math,platform
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
wt,run,out=map(Path,sys.argv[1:]);sys.path.insert(0,str(wt))
from src.checkpoint_r2_b25_adapter import B25Adapter
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_fresh_inputs import hash_value,file_hash
read=lambda p:json.loads(p.read_text())
recover_root=Path(read(wt/'results/checkpoint_r2_fresh_recoverability_20261002/delivery.json')['external_results'])
started=time.monotonic();execution=read(run/'execution.json');identity={k:execution[k] for k in ['producer_sha','configuration_sha256','inventory_sha256','representation_manifest_sha256','recoverability_producer_sha','recoverability_completion_sha256']}
assert execution['configuration_sha256']==file_hash(wt/'config/checkpoint_r2/b25_convergence_fix_v1.json')
for name,sha in execution['code_hashes'].items():assert file_hash(wt/name)==sha,name
completion=read(run/'completion.json')
for name,v in completion['artifacts'].items():assert pin(run/name)==v,name
adapter=B25Adapter(read(wt/'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json'))
paths={read(p)['id']:p for p in (run/'fits').glob('*/fit.json')};fits={name:{k:v for k,v in read(p).items() if k!='score_bindings'} for name,p in paths.items()}
counters=Counter();diagnostics=[]

def equal(a,b):
 assert a is not None and b is not None and math.isclose(float(a),float(b),rel_tol=1e-10,abs_tol=1e-12),(a,b)

def digest(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()

def subgradient(X,y,w,b,C):
 z=X@w+b;r=expit(z)-y;n=len(y)
 g=np.r_[X.T@r/n+w/(C*n),r.mean()]
 v=np.logaddexp(0,np.where(y==1,-z,z)).mean()+w@w/(2*C*n)
 return float(v),g

def metric(rows,scores,metric,scope,topic,weights=None):
 y=np.array([int(r['label']) for r in rows]);topics=np.array([r['topic'] for r in rows])
 if metric=='atomic_auroc':mask=np.ones(len(y),bool)
 else:
  op=np.array([r['operator'] for r in rows]);cell=np.array([int(r['truth_a'])+int(r['truth_b']) for r in rows])
  mask=op==('AND' if metric.startswith('AND') else 'OR')
  if metric=='OR_mixed_vs_FF_auroc':mask&=cell<2
  if metric=='AND_TT_vs_mixed_auroc':mask&=cell>0
 def auc(m):
  weight=None if weights is None else weights[m]
  if weight is not None and (weight[y[m]==0].sum()==0 or weight[y[m]==1].sum()==0):return np.nan
  return roc_auc_score(y[m],scores[m],sample_weight=weight)
 if scope=='topic_macro':return np.mean([auc(mask&(topics==t)) for t in sorted(set(topics))])
 if scope=='topic':mask&=topics==topic
 return auc(mask)

with threadpool_limits(limits=1):
 for model,layer in sorted({(r['model'],r['layer']) for r in fits.values()}):
  features=adapter.layer(model,layer)
  P=features['P15'];py=np.array([int(r['label']) for r in adapter.rows['P15']]);mean=P.mean(0);std=P.std(0,ddof=0);constant=np.all(P==P[0],axis=0)|(std==0);std[constant]=0;den=np.maximum(std,1e-6);z={}
  for group,X in features.items():
   zz=np.zeros_like(X);zz[:,~constant]=(X[:,~constant]-mean[~constant])/den[~constant];z[group]=zz
  for name,small in fits.items():
   if (small['model'],small['layer'])!=(model,layer):continue
   r=read(paths[name])
   assert all(r[k]==v for k,v in identity.items());folder=paths[name].parent
   assert pin(folder/'parameters.npz')==r['parameters'];params=dict(np.load(folder/'parameters.npz',allow_pickle=False))
   if r.get('imported_valid_without_refit'):
    original=Path(r['imported_source_root'])/'fits'/folder.name
    assert pin(original/'fit.json')==r['imported_source_fit'];source=read(original/'fit.json')
    assert r['parameter_producer_sha']==source.get('parameter_producer_sha',source['producer_sha'])
    assert r['parameters']==source['parameters'] and r['scores']==source['scores'];counters['imports_verified']+=1
   if r.get('reused') and not r.get('imported_valid_without_refit'):
    source_folder=recover_root/'fits'/r['reuse_source_id'];assert pin(source_folder/'fit.json')==r['reuse_source_fit'];assert pin(source_folder/'head.npz')==r['reuse_source_parameters'];source_params=dict(np.load(source_folder/'head.npz',allow_pickle=False))
    for k in source_params:np.testing.assert_array_equal(params[k],source_params[k])
    assert r['parameter_producer_sha']=='3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f';counters['recoverability_reuses_verified']+=1
   if r['family']=='bank':
    inds=adapter.balanced if r['training']['group']=='balanced' else list(range(len(P)))
    train=P[inds];rows=adapter.balanced_rows if r['training']['group']=='balanced' else adapter.rows['P15'];ty=py[inds]
    expected_rows=[adapter.rows['P15'][i] for i in adapter.balanced] if r['training']['group']=='balanced' else adapter.rows['P15']
    assert r['training']['source_rows_sha256']==hash_value(expected_rows)
   elif r['family'] in ('repair_fold','repair_refit'):
    group=f"B25_{r['seed']}";indices=adapter.allocations[r['seed']]['folds'][r['fold']]['train'] if r['family']=='repair_fold' else list(range(len(adapter.rows[group])))
    train=features[group][indices];rows=[adapter.rows[group][i] for i in indices];ty=np.array([int(row['label']) for row in rows]);assert r['training']['ordered_bindings_sha256']==hash_value([adapter.bindings[group][i] for i in indices])
   elif r['family']=='atomic_only':
    facts=sorted(adapter.allocations[r['seed']]['isolated'],key=lambda f:f['fact_id']);indices=[f['isolated_row'] for f in facts];group='TC_isolated';train=features[group][indices];ty=np.array([f['label'] for f in facts]);assert r['training']['fact_ids']==[f['fact_id'] for f in facts]
   else:
    assert r['family']=='C_clean';train=None
   if train is not None:assert digest(train)==r['training']['feature_float64_sha256'];counters['source_feature_hashes_verified']+=1
   if r['scoring_method']=='l2_logistic':
    value,gradient=subgradient(train,ty,params['coef'],params['intercept'],r['spec']['C']);norm=float(abs(gradient).max());details=r['optimizer'];last=details['attempts'][-1] if 'attempts' in details else details
    equal(value,last['objective']);equal(norm,last['gradient_infinity_norm'])
    success=last.get('library_success',last.get('final_converged',False));assert bool(np.isfinite(gradient).all() and success and norm<=1e-4)==r['valid_for_scoring']
    if 'total_iterations' in details:assert details['total_iterations']==sum(a['n_iter'] for a in details['attempts'])<=10000
    diagnostics.append(dict(id=name,objective=value,gradient_infinity_norm=norm,valid=r['valid_for_scoring']))
   elif r['scoring_method']=='ttpd':
    polarity_y=np.array([row['form']=='affirmative' for row in rows],dtype=int)
    for sub,X,y,w,b in [('polarity_optimizer',train,polarity_y,params['polarity_coef'],params['polarity_intercept'].item())]:
     value,g=subgradient(X,y,w,b,np.inf);norm=float(abs(g).max());detail=r['optimizer'][sub];last=detail['attempts'][-1]
     equal(value,last['objective']);equal(norm,last['gradient_infinity_norm']);assert detail['valid_for_scoring']==bool(last.get('library_success',last.get('library_converged',False)) and np.isfinite(g).all() and norm<=1e-4)
     if 'total_iterations' in detail:assert detail['total_iterations']<=10000
    if r['optimizer'].get('head_optimizer') is not None:
     X=np.column_stack((train@params['t_g'],train@params['polarity_coef']));value,g=subgradient(X,ty,params['head_coef'],params['head_intercept'],np.inf);detail=r['optimizer']['head_optimizer'];last=detail['attempts'][-1];equal(value,last['objective']);equal(abs(g).max(),last['gradient_infinity_norm'])
     if 'total_iterations' in detail:assert detail['total_iterations']<=10000
   elif r['scoring_method']=='r0' and r['family']!='C_clean':
    w=params['standardized_coef'];b=params['standardized_intercept'];value,g=subgradient(z['P15'],py,w,b,np.inf)
    if r['family']!='bank':
     other,gg=subgradient(z[group][indices],ty,w,b,np.inf);value=.5*value+.5*other;g=.5*g+.5*gg
    value+=.001*w@w;g[:-1]+=.002*w;last=r['optimizer']['attempts'][-1];equal(value,last['objective']);equal(abs(g).max(),last['gradient_infinity_norm']);assert bool(np.isfinite(g).all() and last.get('library_success',False) and abs(g).max()<=1e-4)==r['valid_for_scoring']
   if not r['valid_for_scoring']:
    assert not (folder/'scores.npz').exists();counters['invalid_unscored']+=1;continue
   assert pin(folder/'scores.npz')==r['scores'];scores=dict(np.load(folder/'scores.npz',allow_pickle=False))
   for group,values in scores.items():
    packet=r['score_bindings'][group];assert packet['row_binding_sha256']==hash_value(adapter.bindings[group]);assert packet['ordered_keys']==[[a['group'],a['logical_id']] for a in adapter.bindings[group]]
    if r['scoring_method']=='r0':
     for k,actual in [('mean',mean),('population_std',std),('constant_mask',constant),('denominator',den)]:np.testing.assert_array_equal(params[k],actual)
     prediction=z[group]@params['standardized_coef']+params['standardized_intercept']
    elif r['scoring_method']=='ttpd':prediction=np.column_stack((features[group]@params['t_g'],features[group]@params['polarity_coef']))@params['head_coef']+params['head_intercept']
    else:prediction=features[group]@params['coef']+params['intercept']
    np.testing.assert_array_equal(values,prediction);counters['score_vectors_exactly_recomputed']+=1
   for m in r['metrics']:
    group='atomic_D' if m['metric']=='atomic_auroc' else 'D_bare';equal(metric(adapter.rows[group],scores[group],m['metric'],m['scope'],m['topic']),m['point']);counters['sklearn_point_checks']+=1
   if r['family']=='repair_fold':
    fold=adapter.allocations[r['seed']]['folds'][r['fold']];group=f"B25_{r['seed']}";held=fold['heldout'];heldrows=[adapter.rows[group][i] for i in held];actual=metric(heldrows,scores[group][held],'OR_mixed_vs_FF_auroc','pooled','all')
    if (run/'folds.json').exists():saved=next(f for f in read(run/'folds.json')['records'] if f['id']==name);equal(actual,saved['primary'])
    counters['heldout_fold_checks']+=1
  print('independent checkpoint audit',model,layer,flush=True)
  del features,z;__import__('gc').collect()

# When completed, independently reconstruct evaluated wordings and principal paired draws.
principal=[]
if completion['status']=='completed':
 lock=read(run/'locked.json');assert not lock['wording_used_before_lock'];weights=dict(np.load(run/'bootstrap_weights.npz',allow_pickle=False))
 recover=Path(read(wt/'results/checkpoint_r2_fresh_recoverability_20261002/delivery.json')['external_results']);old=dict(np.load(recover/'bootstrap_weights.npz',allow_pickle=False))
 for group in ('atomic_D','D_bare'):np.testing.assert_array_equal(weights[group],old[group])
 evals={read(p)['fit_id']:p.parent for p in (run/'evaluations').glob('*/score_binding.json')}
 principal_ids={arms[arm] for seeds in lock['procedures'].values() for arms in seeds.values() for arm in ['R_all','S_all']}
 with threadpool_limits(limits=1):
  for model,layer in sorted({(fits[name]['model'],fits[name]['layer']) for name in evals}):
   features=adapter.layer(model,layer,wording=True)
   for name,folder in evals.items():
    r=fits[name]
    if (r['model'],r['layer'])!=(model,layer):continue
    p=dict(np.load(paths[name].parent/'parameters.npz',allow_pickle=False));binding=read(folder/'score_binding.json');assert binding['lock_sha256']==pin(run/'locked.json')['sha256'];assert pin(folder/'scores.npz')==binding['artifact'];scores=dict(np.load(folder/'scores.npz',allow_pickle=False));metric_receipt=read(folder/'metrics.json');draws=np.load(folder/'bootstrap_metrics.npy',allow_pickle=False);assert pin(folder/'bootstrap_metrics.npy')==metric_receipt['draws']
    for group,values in scores.items():
     X=features[group]
     if r['scoring_method']=='r0':
      active=~p['constant_mask'];zz=np.zeros_like(X);zz[:,active]=(X[:,active]-p['mean'][active])/p['denominator'][active];pred=zz@p['standardized_coef']+p['standardized_intercept']
     elif r['scoring_method']=='ttpd':pred=np.column_stack((X@p['t_g'],X@p['polarity_coef']))@p['head_coef']+p['head_intercept']
     else:pred=X@p['coef']+p['intercept']
     np.testing.assert_array_equal(values,pred);counters['locked_score_vectors_recomputed']+=1
    for i,m in enumerate(metric_receipt['records']):
     group=m['condition'];equal(metric(adapter.rows[group],scores[group],m['metric'],m['scope'],m['topic']),m['point']);counters['locked_sklearn_point_checks']+=1
     # Principal OR-boundary endpoint: all 2000 independent weighted sklearn draws.
     if name in principal_ids and (group,m['metric'],m['scope'])==('D_bare','OR_mixed_vs_FF_auroc','pooled'):
      selected=np.array([r['operator']=='OR' and int(r['truth_a'])+int(r['truth_b'])<2 for r in adapter.rows[group]])
      labels=np.array([int(r['label']) for r in adapter.rows[group]])[selected];values=scores[group][selected]
      own=np.array([roc_auc_score(labels,values,sample_weight=w[selected]) if w[selected][labels==0].sum() and w[selected][labels==1].sum() else np.nan for w in weights[group]])
      np.testing.assert_allclose(own,draws[i],rtol=1e-12,atol=1e-12,equal_nan=True);counters['principal_bootstrap_head_draws']+=2000
    print('independent locked audit',name,flush=True)
  # Principal contrasts: average per-seed metrics within each shared draw.
  contrasts=read(run/'contrasts.json')['records'];metrics=read(run/'metrics.json')['records'];templates=read(next(iter(evals.values()))/'metrics.json')['records'];i=next(i for i,m in enumerate(templates) if (m['condition'],m['metric'],m['scope'])==('D_bare','OR_mixed_vs_FF_auroc','pooled'))
  for model in ['qwen','llama']:
   byseed={}
   for seed in ['11','23','37']:
    heads=lock['procedures'][model][seed];armdraw={arm:np.load(evals[heads[arm]]/'bootstrap_metrics.npy',allow_pickle=False)[i] for arm in ['R_all','S_all']};points={arm:metric(adapter.rows['D_bare'],dict(np.load(evals[heads[arm]]/'scores.npz',allow_pickle=False))['D_bare'],'OR_mixed_vs_FF_auroc','pooled','all') for arm in ['R_all','S_all']};byseed[seed]=(points['R_all']-points['S_all'],armdraw['R_all']-armdraw['S_all'])
   byseed['mean_11_23_37']=(np.mean([v[0] for v in byseed.values()]),np.mean([v[1] for v in byseed.values()],axis=0))
   for seed,(point,draw) in byseed.items():
    record=next(r for r in contrasts if (r['model'],r['seed'],r['left'],r['right'],r['condition'],r['metric'],r['scope'])==(model,seed,'R_all','S_all','D_bare','OR_mixed_vs_FF_auroc','pooled'));equal(point,record['point']);valid=draw[np.isfinite(draw)];assert len(valid)==record['valid_replicates'];lo,hi=np.percentile(valid,[2.5,97.5]) if len(valid)>=1800 else (None,None);assert len(valid)>=1800 or (record['ci_status']!='ok' and record['ci_low'] is None and record['ci_high'] is None);
    if len(valid)>=1800:equal(lo,record['ci_low']);equal(hi,record['ci_high'])
    principal.append(dict(model=model,seed=seed,point=point,ci_low=lo,ci_high=hi,valid_draws=len(valid)))
adapter.unchanged()
receipt=dict(status='passed_independent_successor_audit',producer_sha=identity['producer_sha'],completion_status=completion['status'],completion_identity=pin(run/'completion.json'),hostname=platform.node(),interpreter=sys.executable,fit_receipts_audited=len(fits),counts=dict(counters),failures=diagnostics if completion['status']!='completed' else [r for r in diagnostics if not r['valid']],principal_contrasts=principal,scope='independent NumPy training gradients and score reconstruction; sklearn point AUROCs; completed-only paired principal bootstrap metrics/contrasts; no refitting',seconds=time.monotonic()-started,audit_script_sha256=file_hash(Path(__file__)))
out.write_text(json.dumps(receipt,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps({k:receipt[k] for k in ['status','counts','seconds']},indent=2),flush=True)
