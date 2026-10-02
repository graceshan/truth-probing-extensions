"""Frozen-ID-only compatibility runner. No production, fitting, AUROC or E route.

Heavy model libraries are imported only by the separately acknowledged run path.
The historical comparator and every common-base file remain byte-preserved.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid
import numpy as np
from src.checkpoint_r2_bridge import compare_bridge

ROOT = Path(__file__).resolve().parents[1]
CONFIG = 'config/checkpoint_r2/bridge_runner_v1.json'
COMMON = '895ced573ad47af73e3bceaf5dd80a0b39693ca82367d0f5264b85de71755a57'
CONFIG_SHA = '3312a3eed3349dcfb6920bdb5822b7ffab74332a00f31634ef3231130d500231'
BRIDGE = 'data/checkpoint_r2_v1/bridge_identities.json'
HISTORY = 'results/checkpoint_r2_preflight_v1/historical_bridge_bindings.json'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def text_hash(text):return hashlib.sha256(text.encode()).hexdigest()


def strict_json(value):
    """Keep missing/undefined diagnostics JSON-null, with their paths explicit."""
    undefined=[]
    def clean(x,path):
        if isinstance(x,np.ndarray):return clean(x.tolist(),path)
        if isinstance(x,np.generic):return clean(x.item(),path)
        if isinstance(x,float) and not math.isfinite(x):
            undefined.append(path);return None
        if isinstance(x,dict):return {k:clean(v,path+'/'+str(k)) for k,v in x.items()}
        if isinstance(x,(list,tuple)):return [clean(v,path+'/'+str(i)) for i,v in enumerate(x)]
        return x
    result=clean(value,'')
    if undefined:result={'diagnostics':result,'undefined_nonfinite_paths':undefined}
    return (json.dumps(result,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode()


def stable_hash(value):return hashlib.sha256(strict_json(value)).hexdigest()


def beneath(root,relative):
    p=Path(root)/relative
    require(not Path(relative).is_absolute() and '..' not in Path(relative).parts,'unsafe relative path')
    require(p.resolve().is_relative_to(Path(root).resolve()),'path escapes artifact root')
    return p


def write_new(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    payload=strict_json(value)
    # Exclusive creation: a receipt can never overwrite another run or step.
    tmp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.partial')
    try:
        with tmp.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
        os.link(tmp,path)  # atomic publication, fails if destination already exists
    finally:
        if tmp.exists():tmp.unlink()


class Store:
    def __init__(self,path):self.path=Path(path);self.counter=0
    def event(self,value):
        write_new(self.path/'events'/f'{self.counter:05d}.json',value);self.counter+=1
    def watch(self,budget):
        # Only this run's volatile watchdog pointer is replaced; events are immutable.
        tmp=self.path/'watchdog.tmp'
        with tmp.open('wb') as f:f.write(strict_json(budget.snapshot()));f.flush();os.fsync(f.fileno())
        os.replace(tmp,self.path/'watchdog.json')
    def array(self,name,array):
        p=self.path/(name+'.npy');p.parent.mkdir(parents=True,exist_ok=True)
        tmp=p.with_suffix('.partial')
        with tmp.open('xb') as f:np.save(f,array,allow_pickle=False);f.flush();os.fsync(f.fileno())
        os.link(tmp,p);tmp.unlink()
        return {'path':str(p.relative_to(self.path)),'sha256':digest(p),'shape':list(array.shape),'dtype':str(array.dtype)}


def limits_exceeded(s,now):
    dt=max(0.,now-s['as_of_monotonic']);active=s['active_seconds']+(dt if s['phase']=='active' else 0)
    if active>=7200:return 'two_hour_active_cap'
    if s['phase']=='setup' and s['phase_seconds']+dt>=1800:return 'setup_timeout_separate_from_active_cap'
    for model,seconds in s['gpu_seconds'].items():
        if seconds+(dt if s['gpu_inflight']==model else 0)>=3600:return 'one_GPU_hour_cap_'+model
    return None


class Budget:
    def __init__(self,clock=time.monotonic):
        self.clock=clock;self.phase='setup';self.start=clock();self.totals={'active':0.,'setup':0.};self.gpu={'qwen':0.,'llama':0.};self.inflight=None;self.gpu_start=None
    def transition(self,phase):
        require(phase in self.totals,'invalid phase');now=self.clock();self.totals[self.phase]+=now-self.start;self.phase=phase;self.start=now
    def snapshot(self):
        now=self.clock();dt=now-self.start;gpu=dict(self.gpu)
        if self.inflight:gpu[self.inflight]+=now-self.gpu_start
        return dict(phase=self.phase,phase_seconds=dt,active_seconds=self.totals['active']+(dt if self.phase=='active' else 0),setup_seconds=self.totals['setup']+(dt if self.phase=='setup' else 0),gpu_seconds=gpu,gpu_inflight=self.inflight,as_of_monotonic=now,download_seconds=0,queue_seconds=0,gpu_accounting='conservative wall time for entire backend lifetime, including setup and host work; not profiler-measured kernels')
    def check(self):
        reason=limits_exceeded(self.snapshot(),self.clock())
        if reason:raise TimeoutError(reason)
    def begin_gpu(self,model):
        self.check();require(self.inflight is None,'GPU phase misuse');self.inflight=model;self.gpu_start=self.clock()
    def end_gpu(self):
        require(self.inflight is not None,'no GPU interval');self.gpu[self.inflight]+=self.clock()-self.gpu_start;self.inflight=None;self.gpu_start=None


class Plan:
    def __init__(self,root=ROOT):
        self.root=Path(root)
        require(digest(self.root/CONFIG)==CONFIG_SHA,'runner configuration changed')
        self.config=json.loads((self.root/CONFIG).read_text())
        require(self.config['frozen_sha256']['config/checkpoint_r2/common_integration_v1.json']==COMMON,'wrong common anchor')
        for p,h in self.config['frozen_sha256'].items():require(digest(self.root/p)==h,'frozen input/code changed: '+p)
        self.identities=json.loads((self.root/BRIDGE).read_text());self.history=json.loads((self.root/HISTORY).read_text())
        require(set(self.identities)=={'calibration','independent'},'unexpected bridge stage')
        require({k:len(v) for k,v in self.identities.items()}=={'calibration':10,'independent':180},'incomplete or extra IDs')
        allrows=sum(self.identities.values(),[])
        require(len({r['id'] for r in allrows})==190,'ambiguous/cross-stage ID')
        require(all(r['split']=='train' for r in self.identities['calibration']),'calibration not train-only')
        require(all(r['split'] in ('train','validation') for r in allrows),'E forbidden')
        require(all(r['kind'] in ('atomic','compound') for r in allrows),'unknown identity kind')
        require(len(self.history['bindings'])==380,'extra historical bindings')
        for model in ('qwen','llama'):
            for stage,rows in self.identities.items():self.validate_ordered_bindings(rows,self.bindings(model,stage),model,stage)
        require(self.config['caps']['active_seconds']==7200 and self.config['caps']['gpu_seconds_per_model']==3600,'changed caps')
        require(all(not self.config[k] for k in ('production_execution_enabled','final_evaluation_enabled','historical_fit_adoption_enabled')),'closed gates changed')
    def bindings(self,model,stage):return [b for b in self.history['bindings'] if b['model']==model and b['stage']==stage]
    @staticmethod
    def validate_ordered_bindings(rows,bindings,model,stage):
        require(len(rows)==len(bindings),'historical cardinality')
        seen=set()
        for r,b in zip(rows,bindings):
            require(r['id']==b['id'] and r['split']==b['split'] and r['split'] in ('train','validation'),'ordered historical identity/split')
            require(b['model']==model and b['stage']==stage and (stage!='calibration' or r['split']=='train'),'stage/model binding')
            require(text_hash(r['statement'])==b['statement_sha256'],'changed statement')
            key=(b['metadata_path'],b['historical_index']);require(key not in seen,'ambiguous mapping');seen.add(key)
            require(type(b['historical_index']) is int and b['historical_index']>=0,'invalid tensor row')
    def historical_metadata(self,artifact_root):
        tables={}
        for relative,sha in self.config['historical_file_sha256'].items():
            p=beneath(artifact_root,relative);require(digest(p)==sha,'historical metadata/head hash: '+relative)
            if p.name=='metadata.csv':tables[relative]=list(csv.DictReader(p.open()))
        for b in self.history['bindings']:
            rows=tables[b['metadata_path']];i=b['historical_index'];require(i<len(rows),'tensor index beyond metadata')
            r=rows[i];identity=r.get('example_id') or r['dataset']+':'+r['row_index']
            require(identity==b['id'] and r['split']==b['split'] and text_hash(r['statement'])==b['statement_sha256'],'historical indexed row mismatch')
            require(r.get('condition_id','raw_reference')=='raw_reference','wrong historical surface condition')
            rel=b['remote_tensor'].removeprefix('/workspace/truth-probing-artifacts/')
            require(rel in self.config['tensors'] and self.config['tensors'][rel]['metadata_path']==b['metadata_path'],'tensor binding')
        return dict(ordered_bindings=380,metadata_files=len(tables),historical_hashes_verified=len(self.config['historical_file_sha256']),tensor_values_read=False)
    def dry_run(self,artifact_root):
        validation=self.historical_metadata(artifact_root)
        return dict(mode='metadata_only',common_commit=self.config['common_commit'],runner_config_sha256=CONFIG_SHA,frozen_sha256=self.config['frozen_sha256'],counts={s:len(r) for s,r in self.identities.items()},validation=validation,models=self.config['models'],caps=self.config['caps'],storage_payload_bytes={m:(410*2+220*4)*c['layers']*c['width'] for m,c in self.config['models'].items()},storage_note='190 historical saved rows + 220 fresh saved rows + 220 fresh compute-as-float32 rows per model; metadata/header/event overhead extra; reserve >=1GiB durable output plus weights/cache.',historical_tokens='missing, never copied from fresh',historical_execution_verified=False,verified_reuse=False,real_inference_executed=False,production_execution_enabled=False,historical_fit_adoption_enabled=False,final_evaluation_enabled=False)


def select_readout(states,mask,layers,width):
    """Testable all-layer HF readout; states[0] is embedding, final entry post-norm."""
    mask=np.asarray(mask);require(mask.ndim==2 and set(np.unique(mask))<={0,1} and np.all(mask.sum(1)>0),'invalid mask')
    require(len(states)==layers+1,'incomplete layer coverage')
    positions=np.where(mask,np.arange(mask.shape[1]),-1).max(1)
    chosen=[]
    for state in states[1:]:
        require(state.shape==(len(mask),mask.shape[1],width),'hidden-state shape')
        chosen.append(np.asarray(state)[np.arange(len(mask)),positions])
    return np.stack(chosen,axis=1),positions.tolist()


def validate_packet(rows,packet,layers,width):
    require(packet['ids']==[r['id'] for r in rows],'backend identity/order')
    require(packet['statement_sha256']==[text_hash(r['statement']) for r in rows],'backend changed text')
    state=packet['compute'];require(state.shape==(len(rows),layers,width) and state.dtype==np.float32,'incomplete layers/compute save representation')
    require(np.isfinite(state).all(),'nonfinite hidden state')
    logical={'tokens':[],'positions':[],'masks':[]}
    require(len(packet['physical'])==len(rows),'physical row cardinality')
    for p in packet['physical']:
        ids,mask,pos=p['token_ids'],p['attention_mask'],p['position_ids']
        require(len(ids)==len(mask)==len(pos)>0 and set(mask)<={0,1} and sum(mask)>0,'token/mask shape')
        real=[i for i,v in enumerate(mask) if v];require(real==list(range(real[0],real[-1]+1)),'noncontiguous padding')
        require(pos==[sum(mask[:i+1])-1 if mask[i] else 0 for i in range(len(mask))],'semantic positions')
        require(p['readout_index']==real[-1] and p['semantic_readout_position']==sum(mask)-1,'last-real-token readout')
        require(all(ids[i]==p['pad_token_id'] for i,v in enumerate(mask) if not v),'padding token binding')
        logical['tokens'].append([ids[i] for i in real]);logical['positions'].append(list(range(len(real))));logical['masks'].append([1]*len(real))
    return logical


class FrozenHead:
    def __init__(self,coef,intercept,layer,binding):
        self.coef=np.asarray(coef,dtype=np.float64);self.intercept=np.asarray(intercept,dtype=np.float64);self.layer=layer;self.binding=binding
        require(self.coef.ndim==2 and self.coef.shape[0]==1 and self.intercept.shape==(1,),'head shape')
        require(np.isfinite(self.coef).all() and np.isfinite(self.intercept).all(),'nonfinite head')
    def score(self,hidden):
        require(0<=self.layer<hidden.shape[1] and hidden.shape[2]==self.coef.shape[1],'head/layer mismatch')
        return hidden[:,self.layer,:].astype(np.float64)@self.coef.T+self.intercept
    @classmethod
    def load(cls,root,spec):
        p=beneath(root,spec['probe']['path']);require(digest(p)==spec['probe']['sha256'],'frozen head hash')
        with np.load(p,allow_pickle=False) as a:
            require(set(a.files)=={'coef','intercept','classes','layer','C','n_iter','max_iter'},'head schema')
            require(int(a['layer'])==spec['fixed_layer'] and float(a['C'])==spec['fixed_C'] and np.array_equal(a['classes'],[0,1]),'head configuration')
            require(a['coef'].shape==(1,spec['width']),'head width')
            return cls(a['coef'].copy(),a['intercept'].copy(),int(a['layer']),dict(probe_sha256=spec['probe']['sha256'],preprocessing=spec['preprocessing'],preprocessing_sha256=stable_hash(spec['preprocessing'])))


class Historical:
    def __init__(self,plan,root,model,budget,store):
        self.plan=plan;self.model=model;self.arrays={}
        for rel,spec in plan.config['tensors'].items():
            if spec['model']!=model:continue
            budget.check();p=beneath(root,rel);before=p.stat()
            require(digest(p)==spec['sha256'],'historical tensor hash')
            require(before.st_size==p.stat().st_size and before.st_mtime_ns==p.stat().st_mtime_ns,'tensor changed during hash')
            a=np.load(p,mmap_mode='r',allow_pickle=False);require(a.shape==tuple(spec['shape']) and a.dtype==np.float16,'historical tensor shape/dtype')
            self.arrays[rel]=a
            store.event(dict(kind='tensor_verified',path=rel,sha256=spec['sha256'],shape=list(a.shape)))
    def stage(self,stage):
        bindings=self.plan.bindings(self.model,stage)
        return np.stack([self.arrays[b['remote_tensor'].removeprefix('/workspace/truth-probing-artifacts/')][b['historical_index']].copy() for b in bindings])


def packet_record(rows,packets,head,model):
    hidden=np.concatenate([p['compute'].astype(np.float16) for p in packets])
    logical={k:sum((p['logical'][k] for p in packets),[]) for k in ('tokens','positions','masks')}
    return dict(ids=[r['id'] for r in rows],split='train_and_validation',**logical,hidden=hidden,scores=head.score(hidden),contract={'model':model,'semantic_readout':'all HF[1:] at last real token; FP16 stored','logical_token_view':'padding removed only after physical token/mask/position validation'},probe_bindings=head.binding)


def run_stage(plan,model,stage,rows,backend,head,budget,store):
    spec=plan.config['models'][model];size=2 if stage.endswith('_padding') else 1
    padding='left' if stage.endswith('left_padding') else 'right' if stage.endswith('right_padding') else None
    packets=[]
    for offset in range(0,len(rows),size):
        selected=rows[offset:offset+size];budget.check();require(budget.inflight==model,'unmetered backend');store.watch(budget)
        packet=backend.forward(selected,padding)
        store.watch(budget)
        packet['logical']=validate_packet(selected,packet,spec['layers'],spec['width'])
        prefix=f'{model}/{stage}/{offset:03d}'
        saved=packet['compute'].astype(np.float16);require(np.isfinite(saved).all(),'nonfinite saved state')
        store.event(dict(kind='forward',model=model,stage=stage,offset=offset,ids=packet['ids'],statement_sha256=packet['statement_sha256'],physical=packet['physical'],execution=packet['execution'],compute=store.array(prefix+'_compute_f32',packet['compute']),saved=store.array(prefix+'_saved_f16',saved),frozen_scores=head.score(saved).tolist(),probe_bindings=head.binding,budget=budget.snapshot()))
        packets.append(packet);budget.check()
    return packet_record(rows,packets,head,model),np.concatenate([p['compute'] for p in packets])


def compare_historical(plan,model,stage,rows,fresh,hidden,head):
    require(hidden.shape==fresh['hidden'].shape,'historical layer coverage')
    # Unknowns are explicit sentinels, NEVER inferred from a successful fresh run.
    historical=dict(ids=[r['id'] for r in rows],split='train_and_validation',tokens=[None]*len(rows),positions=[None]*len(rows),masks=[None]*len(rows),hidden=hidden,scores=head.score(hidden),contract={'model':model,'recorded_historical_contract':plan.history['contracts'][model]['recorded_manifests'],'historical_execution_verified':False},probe_bindings=head.binding)
    raw=compare_bridge(fresh['ids'],historical,fresh,historical_contract_verified=False)
    numerical=all(raw['checks'][k] for k in ('hidden_shape','hidden_finite','hidden_exact','scores_shape','scores_finite','scores_exact','probe_bindings'))
    return dict(stage=stage,verdict='unverifiable' if numerical else 'failed_numerical_with_unverifiable_provenance',verified_reuse=False,strict_comparator=raw,frozen_score_diagnostics={'ids':fresh['ids'],'historical_scores':historical['scores'].tolist(),'fresh_scores':fresh['scores'].tolist(),'fresh_minus_historical':(fresh['scores']-historical['scores']).tolist(),'scope':'compatibility only; no predictions or research AUROC'},missing_evidence=['historical token IDs','historical per-row positions/masks','verified historical execution'],sentinel_note='Comparator token/position/mask/contract false checks denote missing historical evidence, not reconstructed metadata or evidence of a known token mismatch.')


def execute_models(plan,artifact_root,store,budget,backend_factory,history_factory=Historical):
    results={}
    try:
        for model in ('qwen','llama'):
            budget.transition('setup');store.watch(budget)
            plan.historical_metadata(artifact_root)
            head=FrozenHead.load(artifact_root,plan.config['models'][model])
            history=history_factory(plan,artifact_root,model,budget,store)
            backend=None;budget.begin_gpu(model);store.watch(budget)
            try:
                backend=backend_factory(model,plan.config)
                store.event(dict(kind='setup_complete',model=model,actual_runtime=backend.runtime,head_binding=head.binding,budget=budget.snapshot()))
                budget.check();budget.transition('active');store.watch(budget)
                rows=plan.identities['calibration'];baseline,compute=run_stage(plan,model,'calibration_baseline',rows,backend,head,budget,store)
                h=history.stage('calibration');store.array(model+'/historical_calibration_f16',h)
                store.event(dict(kind='calibration_historical',model=model,result=compare_historical(plan,model,'calibration',rows,baseline,h,head)))
                for stage in ('calibration_repeat','calibration_right_padding','calibration_left_padding'):
                    observed,states=run_stage(plan,model,stage,rows,backend,head,budget,store)
                    strict=compare_bridge(baseline['ids'],baseline,observed,False)
                    exact=all(strict['checks'].values()) and np.array_equal(compute,states)
                    store.event(dict(kind='calibration_check',model=model,stage=stage,exact_compute_equal=np.array_equal(compute,states),all_exact_checks_passed=exact,strict_comparator=strict,claim='within-configuration repeatability or declared batch/padding invariance ONLY; not historical compatibility'))
                    require(exact,'calibration mismatch; stop without tuning: '+stage)
                rows=plan.identities['independent'];fresh,_=run_stage(plan,model,'independent',rows,backend,head,budget,store)
                h=history.stage('independent');store.array(model+'/historical_independent_f16',h)
                results[model]=compare_historical(plan,model,'independent',rows,fresh,h,head)
                store.event(dict(kind='independent_comparison',model=model,result=results[model]))
            finally:
                if backend is not None:backend.close()
                budget.end_gpu();store.watch(budget)
        budget.check()
        status='completed_diagnostics_unverifiable_history'
        if any(v['verdict'].startswith('failed') for v in results.values()):status='completed_with_numerical_failures'
    except BaseException as exc:
        store.event(dict(kind='failure',error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc(),budget=budget.snapshot()))
        write_new(store.path/'failure.json',dict(status='failed_or_capped',error_type=type(exc).__name__,error=str(exc),partial_results=results,budget=budget.snapshot(),verified_reuse=False,production_execution_enabled=False,final_evaluation_enabled=False))
        raise
    write_new(store.path/'completion.json',dict(status=status,results=results,budget=budget.snapshot(),verified_reuse=False,historical_fit_adoption_enabled=False,production_execution_enabled=False,final_evaluation_enabled=False))


def supervise(command,output):
    process=subprocess.Popen(command,start_new_session=True)
    try:
        while process.poll() is None:
            p=Path(output)/'watchdog.json'
            if p.exists():
                s=json.loads(p.read_text());reason=limits_exceeded(s,time.monotonic())
                if reason:
                    os.killpg(process.pid,signal.SIGKILL);process.wait()
                    write_new(Path(output)/'watchdog_termination.json',dict(reason=reason,last_state=s,returncode=process.returncode,verified_reuse=False))
                    return 124
            time.sleep(.1)
    except BaseException as exc:
        if process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
        write_new(Path(output)/'supervisor_interruption.json',dict(error_type=type(exc).__name__,error=str(exc),returncode=process.returncode,verified_reuse=False))
        raise
    if process.returncode and not (Path(output)/'failure.json').exists():
        write_new(Path(output)/'worker_exit.json',dict(returncode=process.returncode,status='interrupted_or_failed_before_receipt',verified_reuse=False))
    return process.returncode


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('dry-run','run'),default='dry-run');p.add_argument('--historical-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--acknowledge-bounded-inference',action='store_true')
    p.add_argument('--internal-worker',action='store_true',help=argparse.SUPPRESS);a=p.parse_args()
    plan=Plan();output=a.output.absolute()
    if a.mode=='dry-run':output=output.resolve()  # macOS /tmp is a symlink; metadata-only receipts are safe here.
    require(output.resolve()==output,'output must be canonical, not symlinked')
    require(not output.is_relative_to(a.historical_root.resolve()) and not a.historical_root.resolve().is_relative_to(output),'output overlaps historical artifacts')
    if a.mode=='dry-run':
        require(not a.internal_worker and not a.acknowledge_bounded_inference,'dry-run cannot execute')
        write_new(output,plan.dry_run(a.historical_root));print('metadata dry run passed; no inference or tensor values');return
    require(a.acknowledge_bounded_inference,'real inference requires explicit later authorization and acknowledgment')
    require(not output.is_relative_to(ROOT) and not str(output).startswith(('/tmp/','/private/tmp/')),'real output must be external durable storage')
    if a.internal_worker:
        launch=json.loads((output/'launch.json').read_text());require(launch['parent_pid']==os.getppid(),'worker must be launched by supervisor')
        from src.checkpoint_r2_bridge_backend import HFBackend
        budget=Budget();store=Store(output);store.watch(budget)
        execute_models(plan,a.historical_root,store,budget,HFBackend);return
    output.mkdir(parents=False,exist_ok=False)
    write_new(output/'launch.json',dict(parent_pid=os.getpid(),common_sha=plan.config['common_commit'],config_sha256=CONFIG_SHA,runner_sha256=digest(Path(__file__)),backend_sha256=digest(ROOT/'src/checkpoint_r2_bridge_backend.py'),argv=sys.argv,production_execution_enabled=False,final_evaluation_enabled=False))
    # Parent owns a watchdog before any child setup; bounds even a hung import/load.
    initial=Budget();Store(output).watch(initial)
    code=supervise([sys.executable,'-B','-m','src.checkpoint_r2_bridge_runner',*sys.argv[1:],'--internal-worker'],output)
    if code:raise SystemExit(code)

if __name__=='__main__':main()
