"""Independent read-only result audit: saved heads, row scores and paired intervals."""
import json,pathlib,sys,time,hashlib,platform,math
import numpy as np
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
wt,run,out=map(pathlib.Path,sys.argv[1:]);sys.path.insert(0,str(wt))
from src.checkpoint_r2_fresh_adapter import FreshAdapter
from src.checkpoint_r2_fresh_inputs import file_hash,hash_value
from src.checkpoint_r2_fresh_store import pin
started=time.monotonic()
read=lambda p:json.loads(p.read_text())
completion=read(run/'completion.json');exe=read(run/'execution.json');cfg=exe['config'];ident=exe['producer_sha'];assert ident=='3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f'
assert completion['producer_sha']==ident and completion['configuration_sha256']==file_hash(wt/'config/checkpoint_r2/fresh_recoverability_v1.json')
for n,p in completion['artifacts'].items():assert pin(run/n)==p,n
assert completion['planned_fits']==122
accept=read(wt/'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json');adapter=FreshAdapter(accept['output'],accept)
fits=read(run/'fit_inventory.json')['records'];assert len(fits)==len({r['id'] for r in fits})==122
from collections import Counter
assert Counter(r['family'] for r in fits)=={'R0':60,'C_clean':60,'reduced_LR':2}
metrics=read(run/'metrics.json')['records'];contrast=read(run/'contrasts.json')['records'];sel=read(run/'selected.json')['models']
assert len(metrics)==122*28
for artifact in run.rglob('*.json'):
 d=read(artifact)
 if artifact.name not in ['metrics.json'] or artifact.parent==run:
  assert d.get('producer_sha')==ident,artifact
# Result artifact files have a hash-bound JSON companion with producer identity.
actual_scores={};prechecked=0;scorechecked=0
with threadpool_limits(limits=1):
 for model in ('qwen','llama'):
  for layer in cfg['models'][model]['layers']:
   X=adapter.layer(model,layer)
   with np.load(run/'preprocessing'/model/f'L{layer:02d}.npz',allow_pickle=False) as pre:
    mean=X['P15'].mean(0);std=X['P15'].std(0,ddof=0);constant=np.all(X['P15']==X['P15'][0],axis=0)|(std==0);std[constant]=0
    np.testing.assert_array_equal(mean,pre['mean']);np.testing.assert_array_equal(std,pre['population_std']);np.testing.assert_array_equal(constant,pre['constant_mask']);np.testing.assert_array_equal(np.maximum(std,1e-6),pre['denominator'])
    prechecked+=1
    for fit in [r for r in fits if r['model']==model and r['saved_layer']==layer]:
     group=fit['training_group'];assert hashlib.sha256(X[group].tobytes()).hexdigest()==fit['feature_float64_sha256']
     assert fit['training_bindings_sha256']==hash_value(adapter.bindings[group]) and fit['fit_rows']==len(adapter.rows[group])
     if not fit['valid_for_scoring']:continue
     folder=run/'fits'/fit['id'];assert pin(folder/'head.npz')==fit['parameters'] and pin(folder/'scores.npz')==fit['scores']
     with np.load(folder/'head.npz',allow_pickle=False) as head, np.load(folder/'scores.npz',allow_pickle=False) as scores:
      for g in ('P15','atomic_D','D_bare'):
       assert fit['score_bindings'][g]['row_binding_sha256']==hash_value(adapter.bindings[g])
       assert fit['score_bindings'][g]['ordered_keys']==[[r['group'],r['logical_id']] for r in adapter.bindings[g]]
       if fit['family']=='reduced_LR':pred=X[g]@head['coef']+head['intercept']
       else:
        z=np.zeros_like(X[g]);active=~constant;z[:,active]=(X[g][:,active]-mean[active])/np.maximum(std[active],1e-6)
        pred=z@head['standardized_coef']+head['standardized_intercept']
       np.testing.assert_array_equal(pred,scores[g]);scorechecked+=1
      actual_scores[fit['id']]={g:scores[g].copy() for g in ('P15','atomic_D','D_bare')}
   del X
   print('audited saved preprocessing and scores',model,layer,flush=True)
# Reconstruct the frozen integer draw schedules without calling production helpers.
with np.load(run/'bootstrap_weights.npz',allow_pickle=False) as saved:
 weights={g:saved[g].copy() for g in ('atomic_D','D_bare')}
for group in ('atomic_D','D_bare'):
 rng=np.random.Generator(np.random.PCG64(1729));rows=adapter.rows[group];expected=np.zeros_like(weights[group])
 for topic in sorted({r['topic'] for r in rows}):
  indices=[i for i,r in enumerate(rows) if r['topic']==topic]
  people=sorted({rows[i]['person_key'] for i in indices}) if group=='atomic_D' else sorted({rows[i][k] for i in indices for k in ('person_a','person_b')})
  lookup={p:i for i,p in enumerate(people)};counts=rng.multinomial(len(people),np.full(len(people),1/len(people)),size=2000)
  for i in indices:
   r=rows[i];expected[:,i]=counts[:,lookup[r['person_key']]] if group=='atomic_D' else counts[:,lookup[r['person_a']]]*counts[:,lookup[r['person_b']]]
 np.testing.assert_array_equal(expected,weights[group])
TOPICS=sorted({r['topic'] for r in adapter.rows['atomic_D']})
bootstrap_draw_ids=[0,1,17,1729,1999]
def auc(y,s,w):
 if np.dot(y,w)<=0 or np.dot(1-y,w)<=0:return np.nan
 return roc_auc_score(y,s,sample_weight=w)
def equal(a,b):
 if a is None:assert not np.isfinite(b),(a,b)
 else:assert np.isfinite(b) and math.isclose(a,b,abs_tol=1e-12,rel_tol=1e-12),(a,b)
def check_ci(r,draws):
 good=draws[np.isfinite(draws)];assert r['valid_replicates']==len(good) and r['invalid_replicates']==2000-len(good)
 if len(good)>=1800:
  low,high=np.quantile(good,[.025,.975],method='linear');equal(r['ci_low'],low);equal(r['ci_high'],high);assert r['ci_status']=='ok'
 else:assert r['ci_low'] is None and r['ci_high'] is None and r['ci_status']=='insufficient_valid_replicates'
boot={};points={};metricchecks=0;drawchecks=0
for fit in fits:
 if not fit['valid_for_scoring']:continue
 scores=actual_scores[fit['id']];saved=np.load(run/'fits'/fit['id']/'bootstrap_metrics.npy',allow_pickle=False);assert saved.shape==(28,2000)
 records=[r for r in metrics if r['id']==fit['id']];actual_points=[]
 atom=adapter.rows['atomic_D'];comp=adapter.rows['D_bare'];op=np.array([r['operator'] for r in comp]);cell=np.array([int(r['truth_a'])+int(r['truth_b']) for r in comp])
 masks=[('atomic_auroc','atomic_D',np.ones(len(atom),bool)),('AND_auroc','D_bare',op=='AND'),('OR_auroc','D_bare',op=='OR'),('OR_mixed_vs_FF_auroc','D_bare',(op=='OR')&(cell<2))]
 for j,(metric,group,mask) in enumerate(masks):
  rows=adapter.rows[group];y=np.array([int(r['label']) for r in rows]);topics=np.array([r['topic'] for r in rows]);topic_points=[];topic_draws=[]
  for k,(scope,topic,selection) in enumerate([('pooled','all',mask)]+[('topic',t,mask&(topics==t)) for t in TOPICS]):
   i=j*7+k;r=records[i];assert (r['metric'],r['scope'],r['topic'])==(metric,scope,topic)
   point=auc(y[selection],scores[group][selection],np.ones(sum(selection)));equal(r['point'],point);actual_points.append(point);metricchecks+=1
   for draw in bootstrap_draw_ids:equal(None if not np.isfinite(saved[i,draw]) else float(saved[i,draw]),auc(y[selection],scores[group][selection],weights[group][draw,selection]));drawchecks+=1
   check_ci(r,saved[i])
   if scope=='topic':topic_points.append(point);topic_draws.append(saved[i])
  i=j*7+6;macro=np.mean(topic_points);equal(records[i]['point'],macro);actual_points.append(macro)
  np.testing.assert_array_equal(np.mean(topic_draws,axis=0),saved[i]);check_ci(records[i],saved[i]);metricchecks+=1
 boot[fit['id']]=saved;points[fit['id']]=np.array(actual_points)
contrastchecks=0
for r in contrast:
 if r.get('status')=='fit_failed':continue
 template=[m for m in metrics if m['id']==r['left']]
 i=next(i for i,m in enumerate(template) if (m['metric'],m['scope'],m['topic'])==(r['metric'],r['scope'],r['topic']))
 equal(r['point'],points[r['left']][i]-points[r['right']][i]);check_ci(r,boot[r['left']][i]-boot[r['right']][i]);contrastchecks+=1
 if 'retention_margin' in r:
  assert r['retention_margin']==(.005 if r['metric']=='atomic_auroc' else .02)
  assert r['point_retained']==(r['point'] is not None and r['point']>=-r['retention_margin'])
  assert r['interval_supported']==(r['ci_low'] is not None and r['ci_low']>=-r['retention_margin'])
for model in ('qwen','llama'):
 r0=[r for r in fits if r['model']==model and r['family']=='R0' and r['valid_for_scoring']]
 if r0:
  best=max(next(x['point'] for x in r['metrics'] if x['metric']=='atomic_auroc' and x['scope']=='pooled') for r in r0)
  chosen=min((r for r in r0 if next(x['point'] for x in r['metrics'] if x['metric']=='atomic_auroc' and x['scope']=='pooled')==best),key=lambda r:r['saved_layer'])
  assert sel[model]['R0_atomic_selected_id']==chosen['id']
 chosen=sel[model]['C_clean_D_selector']['selected_id']
 if chosen:
  cc=[r for r in fits if r['model']==model and r['family']=='C_clean']
  for metric in ['OR_mixed_vs_FF_auroc','atomic_auroc']:
   vals={r['id']:next(x['point'] for x in r['metrics'] if x['metric']==metric and x['scope']=='pooled') for r in cc};best=max(vals.values());cc=[r for r in cc if best-vals[r['id']]<=1e-12]
  best=max(r['separation'] for r in cc);cc=[r for r in cc if best-r['separation']<=1e-12];tied=sorted(r['id'] for r in cc)
  assert chosen==str(np.random.Generator(np.random.PCG64(20261008)).choice(tied))
adapter.unchanged()
receipt=dict(status='independent_result_audit_passed',producer_sha=ident,configuration_sha256=completion['configuration_sha256'],hostname=platform.node(),scope='read-only CPU result re-execution; no new fits',saved_P15_preprocessing_layers_verified=prechecked,score_vectors_exactly_recomputed=scorechecked,all_saved_head_training_feature_hashes_verified=True,point_metric_checks=metricchecks,independent_sklearn_bootstrap_draw_checks=drawchecks,draw_ids=bootstrap_draw_ids,all_2000_draw_integer_schedules_verified=True,all_metric_percentile_intervals_verified=True,paired_contrast_checks=contrastchecks,retention_margins_verified=True,selectors_verified=True,artifacts_verified=len(completion['artifacts']),seconds=time.monotonic()-started,audit_script_sha256=file_hash(pathlib.Path(__file__)),fit_failures=completion['failures'])
out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(receipt,flush=True)
