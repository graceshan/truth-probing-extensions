"""Reviewer synthetic chain; no real E lock, data, inference or fitting."""
import copy
import contextlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, Mock
import numpy as np
from src import final_e_raw as raw, final_e_runner as runner
from src.final_e_inventory import ROOT, read
from src.checkpoint_r2_fresh_inputs import hash_value, text_hash
from src.checkpoint_r2_fresh_store import write_json, pin
from src.checkpoint_r2_fresh_pilot import atomic_pointer
from src.checkpoint_r2_selection_v1 import TOPICS


class Fixture:
    def __init__(self, root):
        self.root=Path(root);self.output=self.root/'synthetic-campaign';self.output.mkdir()
        (self.output/'campaign.lock').touch()
        self.access=dict(producer_sha='2'*40,lock_sha256='3'*64,campaign_id='synthetic-only',limits=dict(model_seconds=7200,total_seconds=18000,setup_phase_seconds=1800))
        self.cfg=read(ROOT/'config/checkpoint_r2/full_raw_v1.json')
        self.cfg['models']={m:dict(layers=1,width=w) for m,w in [('qwen',3584),('llama',4096)]}
        self.atoms=[dict(label=y,topic=t,person_key=t+str(y),example_id=f'a{i}-{y}') for i,t in enumerate(TOPICS) for y in (0,1)]
        self.bare=[dict(label=a&b if op=='AND' else a|b,truth_a=a,truth_b=b,operator=op,ordering=o,topic=t,
                       person_a=t+'0',person_b=t+'1',pair_id=t,example_id=f'c{i}-{op}-{o}-{a}{b}')
                   for i,t in enumerate(TOPICS) for op in ('AND','OR') for o in ('AB','BA') for a in (0,1) for b in (0,1)]
        self.groups={'atomic_E':self.atoms,'bare':self.bare}
        for c in ('or_explicit_or_both_v1','or_at_least_one_v1','and_both_following_v1'):
            self.groups[c]=[dict(r,base_example_id=r['example_id']) for r in self.bare]
        self.rows=[];self.indices={}
        for g,rs in self.groups.items():
            self.indices[g]=[]
            for i,r in enumerate(rs):
                statement='synthetic '+g+' '+str(i);identity=g+str(i)
                self.indices[g].append(len(self.rows));self.rows.append(dict(id=identity,statement=statement,statement_sha256=text_hash(statement),bindings=[dict(binding_id=identity)]))
        self.meta=SimpleNamespace(rows=self.groups,bindings=[b for r in self.rows for b in r['bindings']],texts=self.rows,
            receipt=dict(synthetic_only=True,atomic_people=10,compound_people=10,shared_people=10,ordered_groups={g:hash_value(v) for g,v in self.groups.items()}),
            indices=lambda g:self.indices[g])
        self.opts=dict(replicates=20,seed=1729,minimum_valid=18)
        self.prep=dict(shard_rows=128,bootstrap=self.opts)
        self.carry=read(ROOT/'config/final_e/carryforward_v1.json')
        self.seed=dict(total_seconds=sum(self.carry['charged_seconds'].values()),model_seconds=copy.deepcopy(self.carry['charged_seconds']),running=False,model=None,as_of_unix=1.)
        write_json(self.output/'budget-adoption.json',dict(campaign=raw.campaign_identity(self.access,self.meta),carryforward=pin(ROOT/'config/final_e/carryforward_v1.json'),initial=self.seed))
        atomic_pointer(self.output/'full-budget.json',self.seed)
        self.registry=dict(heads={},scalar_aliases=[],constituent_cases=[],selection_evidence={})
        self.roots={}
        for model in ('qwen','llama'):
            hr=self.root/model;hr.mkdir();self.roots[model]=str(hr)
            for kind in ('scalar','first','second'):
                width=self.cfg['models'][model]['width'];coef=np.zeros(width);coef[0 if kind!='second' else 1]=1
                p=hr/(kind+'.npz');np.savez(p,**({'coef':coef,'intercept':0.} if kind=='scalar' else {'weights':coef,'intercept':0.}))
                fit=dict(id=kind,producer_sha='4'*40,model=model,layer=0,scoring_method='l2_logistic',valid_for_scoring=True,parameters=pin(p))
                ref=lambda n:dict(source=model,path=n,**pin(hr/n))
                h=dict(frozen_id=kind,parameter_producer='4'*40,model=model,saved_layer=0,method='l2_logistic',family='scalar',fit=None,parameters=ref(kind+'.npz'))
                if kind!='scalar':
                    np.savez(hr/(kind+'-pre.npz'),mean=np.zeros(width),denominator=np.ones(width),constant_mask=np.zeros(width,bool))
                    write_json(hr/(kind+'-pre.json'),dict(fit_group='synthetic'))
                    fit['preprocessing_artifact']=pin(hr/(kind+'-pre.npz'))
                    h.update(family='constituent',method='constituent',preprocessing=ref(kind+'-pre.npz'),preprocessing_metadata=ref(kind+'-pre.json'))
                write_json(hr/(kind+'.json'),fit);h['fit']=ref(kind+'.json');self.registry['heads'][model+':'+kind]=h
            for sample in (11,23,37):
                for arm in ('R_all','S_all','S_atom_all','reduced_LR','bank_oracle'):
                    self.registry['scalar_aliases'].append(dict(panel='b25',model=model,sample=sample,arm=arm,head=model+':scalar',category='synthetic'))
            for source in ('AND','OR'):
                target='OR' if source=='AND' else 'AND'
                for arm,op in [('source_on_source',source),('source_on_target',target),('target_on_target',target)]:
                    self.registry['constituent_cases'].append(dict(model=model,source_operator=source,target_operator=target,arm=arm,evaluation_operator=op,saved_layer=0,heads=[model+':first',model+':second']))
        write_json(self.root/'roots.json',self.roots)
        self.stack=contextlib.ExitStack()
    def __enter__(self):
        orig=raw.read
        def rd(p):
            p=Path(p)
            if p==ROOT/'config/final_e/preparation_v1.json':return self.prep
            if p==ROOT/'config/final_e/head_registry_v1.json':return self.registry
            if p==ROOT/'config/final_e/contrasts_v1.json':return dict(b25=[['R_all','S_all'],['S_all','reduced_LR']])
            if p==ROOT/'results/final_e_preparation_20261003/bootstrap.json':
                from src.final_e_reporting import schedules
                return schedules(self.meta,self.opts)[1]
            return orig(p)
        for mod in (raw,runner):
            self.stack.enter_context(patch.object(mod,'Metadata',return_value=self.meta));self.stack.enter_context(patch.object(mod,'read',side_effect=rd))
        self.stack.enter_context(patch.object(raw,'input_rows',return_value=self.rows))
        self.stack.enter_context(patch.object(raw,'raw_contract',return_value=self.cfg))
        self.stack.enter_context(patch.object(raw,'verify_origin',return_value={}))
        return self
    def __exit__(self,*args):self.stack.close()
    def packet(self,r):
        return dict(id=r['id'],statement_sha256=r['statement_sha256'],token_ids=[1,2],unpadded_token_ids=[1,2],raw_token_ids=[2],unpadded_special_tokens_mask=[1,0],attention_mask=[1,1],position_ids=[0,1],readout_index=1,semantic_readout_position=1,pad_token_id=0,add_special_tokens=True,truncation=False,chat_template=False)
    def extract(self,model):
        fx=self
        class FakeBackend:
            details={}
            def __init__(self,*_):pass
            def forward(self,rs,batch):
                a=np.zeros((len(rs),1,fx.cfg['models'][model]['width']),np.float32)
                for j,r in enumerate(rs):
                    i=fx.rows.index(r);a[j,0,:2]=[(-1 if i%2 else 1),(-1 if i%3 else 1)]
                return a,[fx.packet(r) for r in rs]
            def peak(self):return dict(synthetic=True)
            def close(self):pass
        attempt=self.output/'attempts'/('01-qwen' if model=='qwen' else '02-llama');attempt.mkdir(parents=True)
        access=dict(self.access,model=model,phase='extract')
        write_json(attempt/'access.json',access)
        args=SimpleNamespace(output=str(self.output),attempt=str(attempt),model=model,volume=str(self.root),cache=str(self.root/'cache'),parent_pid=__import__('os').getppid())
        with patch.object(raw.platform,'system',return_value='Linux'),patch.object(raw.ctypes,'CDLL',return_value=SimpleNamespace(prctl=lambda *_:0)),patch.object(raw.os.path,'ismount',return_value=True),patch.object(raw,'runtime_matches'),patch.object(raw,'evidence',return_value={'backend':{}}),patch.object(raw,'filesystem_prerequisites',return_value={}),patch.object(raw,'storage_probe',return_value={}),patch.object(raw,'offline_tokenizer',return_value=({'snapshot':'synthetic'},None)),patch.object(raw,'single_packet',side_effect=lambda _,r,__ :self.packet(r)),patch('src.checkpoint_r2_fresh_backend.runtime_receipt',return_value={}),patch('src.checkpoint_r2_fresh_backend.FreshBackend',FakeBackend):
            raw.worker(args,access)
        budget=read(self.output/'full-budget.json');progress=read(attempt/'progress.json')
        for i in progress['intervals']:budget['model_seconds'][i['model']]+=i['seconds']
        budget['total_seconds']+=read(attempt/'resource.json')['wall_seconds']+1
        atomic_pointer(self.output/'full-budget.json',budget)
        write_json(attempt/'supervisor-exit.json',dict(returncode=0,interrupted=False,cap_reason=None,budget=budget,limits=self.access['limits'],budget_reset_allowed=False))
        return args
    def accepted(self):
        for m in ('qwen','llama'):
            args=self.extract(m)
            with patch.object(raw,'offline_tokenizer',return_value=({},None)),patch.object(raw,'single_packet',side_effect=lambda _,r,__ :self.packet(r)):
                raw.audit(args,self.access)
            raw.accept(args,self.access)
        return self


class HandoffTests(unittest.TestCase):
    def test_complete_fake_backend_chain(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-final-e-') as td,Fixture(td) as f:
            f.accepted()
            manifest=f.root/'transfer.json';args=SimpleNamespace(command='transfer',output=str(f.output),manifest=str(manifest))
            runner.transfer(args,f.access)
            import shutil
            local=f.root/'local-campaign';shutil.copytree(f.output,local)
            args.command='verify-transfer';args.output=str(local);runner.transfer(args,f.access)
            for m in ('qwen','llama'):raw.verify(local,m,f.access,require_acceptance=True)
            args=SimpleNamespace(output=str(local),roots=str(f.root/'roots.json'),report=str(f.root/'report'),source_report=None,operation_receipt=str(f.root/'audit-operation.json'))
            runner.evaluate(args,f.access)
            args.source_report=args.report;args.report=str(f.root/'audit-reconstruction')
            runner.audit_report(args,f.access)
            self.assertEqual(read(f.root/'audit-operation.audit.json')['status'],'complete_report_reconstruction_passed')

    def test_completed_shards_index_and_corruption(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-final-e-') as td,Fixture(td) as f:
            f.extract('qwen');raw.verify(f.output,'qwen',f.access)
            p=f.output/'qwen/shards/00000/activations.npy'
            with p.open('ab') as s:s.write(b'corrupt')
            with self.assertRaises(ValueError):raw.verify(f.output,'qwen',f.access)

    def test_unsettled_reduced_and_changed_accounting_rejected(self):
        for mutation in ('running','reduced','snapshot','cap','missing'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(prefix='synthetic-final-e-') as td,Fixture(td) as f:
                f.accepted();p=f.output/'full-budget.json';b=read(p)
                if mutation=='running':b['running']=True
                elif mutation=='reduced':b['model_seconds']['qwen']=0
                elif mutation=='snapshot':b['total_seconds']+=1
                elif mutation=='cap':b['model_seconds']['qwen']=7200;b['total_seconds']+=7200
                else:p.unlink()
                if mutation!='missing':atomic_pointer(p,b)
                with self.assertRaises((ValueError,FileNotFoundError)):raw.verify(f.output,'qwen',f.access,require_acceptance=True)

    def test_wrong_campaign_token_audit_and_receipts(self):
        for mutation in ('producer','lock','audit','receipt','marker','missing_shard','transfer'):
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory(prefix='synthetic-final-e-') as td,Fixture(td) as f:
                f.accepted();access=dict(f.access)
                if mutation=='producer':access['producer_sha']='9'*40
                elif mutation=='lock':access['lock_sha256']='9'*64
                elif mutation=='audit':
                    p=f.output/'qwen/token-audit.json';v=read(p);v['model_loaded']=True;atomic_pointer(p,v)
                elif mutation=='receipt':
                    p=f.output/'qwen/acceptance.json';v=read(p);v['unexpected']='altered';atomic_pointer(p,v)
                elif mutation=='marker':(f.output/'qwen/shards/00000/completed.json').unlink()
                elif mutation=='missing_shard':
                    import shutil;shutil.rmtree(f.output/'qwen/shards/00000')
                else:
                    args=SimpleNamespace(output=str(f.output),manifest=str(f.root/'transfer.json'),command='transfer')
                    runner.transfer(args,access);(f.output/'unrelated-extra').write_text('altered')
                    args.command='verify-transfer'
                    with self.assertRaises(ValueError):runner.transfer(args,access)
                    continue
                with self.assertRaises((ValueError,FileNotFoundError)):raw.verify(f.output,'qwen',access,require_acceptance=True)

    def test_operation_cap_stops_and_preserves_failure_receipt(self):
        import signal,time
        with tempfile.TemporaryDirectory(prefix='synthetic-final-e-') as td,Fixture(td) as f:
            args=SimpleNamespace(command='accept',output=str(f.output),operation_receipt=str(f.root/'synthetic-operation.json'))
            original=signal.setitimer
            def timer(kind,seconds):return original(kind,.02 if seconds else 0)
            with patch.object(runner.signal,'setitimer',side_effect=timer),patch.object(runner,'dispatch',side_effect=lambda *_:time.sleep(.1)):
                with self.assertRaises(TimeoutError):runner.run_authorized(args,f.access)
            r=read(args.operation_receipt)
            self.assertEqual(r['status'],'failed');self.assertEqual(r['error']['type'],'TimeoutError');self.assertEqual(r['cap_seconds'],1800)
            with self.assertRaises(ValueError):runner.run_authorized(args,f.access)

    def test_shared_formula_error_is_found_by_reference(self):
        from src.final_e_reference import Checks, constituent_endpoint
        from src.final_e_reporting import constituent_result, scalar_result
        from src.checkpoint_next_full_bank_audit import independent_auc
        opts=dict(replicates=20,minimum_valid=18,seed=1729)
        # Deliberate ties and vanished classes: half credit, no redraw.
        y=np.array([0,1,0,1]);scores=np.array([0.,0.,1.,2.]);w=np.array([[1,2,3,4],[0,1,0,2]])
        self.assertAlmostEqual(independent_auc(y,scores,w)[0],17/24)
        self.assertTrue(np.isnan(independent_auc(y,scores,w)[1]))
        with tempfile.TemporaryDirectory(prefix='synthetic-final-e-') as td,Fixture(td) as f:
            rows=[r for r in f.bare if r['operator']=='OR'];s=np.array([[0.,(-1.)**i] for i in range(len(rows))])
            weights=np.ones((20,len(rows)));weights[0,:]=0
            result=constituent_result(rows,s,'OR',weights,opts);c=Checks()
            draw_index=0
            for r in result['records']:
                p=constituent_endpoint(rows,s,r['metric'],r['scope'],r['topic'],r['surface_order'])[0]
                c.compare([p],[np.nan if r['point'] is None else r['point']])
                if r['surface_order']=='all':
                    d=constituent_endpoint(rows,s,r['metric'],r['scope'],r['topic'],weights=weights)
                    c.compare(d,result['draws'][draw_index]);draw_index+=1
            with self.assertRaises(ValueError):c.compare([.5],[.6])

    def test_report_corruption_is_not_reconstructed_as_success(self):
        with tempfile.TemporaryDirectory(prefix='synthetic-final-e-') as td,Fixture(td) as f:
            f.accepted()
            args=SimpleNamespace(output=str(f.output),roots=str(f.root/'roots.json'),report=str(f.root/'report'),source_report=None,operation_receipt=str(f.root/'audit.json'))
            runner.evaluate(args,f.access)
            p=Path(args.report)/'scalar-metrics.json';p.write_text(p.read_text()+' ')
            args.source_report=args.report;args.report=str(f.root/'audit-rebuild')
            with self.assertRaisesRegex(ValueError,'report artifact hash'):runner.audit_report(args,f.access)
            self.assertFalse(Path(args.report).exists())

    def test_exhausted_worker_cap_blocks_subprocess_and_preserves_ledger(self):
        from src.checkpoint_r2_full_raw import supervise
        with tempfile.TemporaryDirectory(prefix='synthetic-final-e-cap-') as td:
            root=Path(td);attempt=root/'attempt';attempt.mkdir()
            ledger=dict(total_seconds=7200.,model_seconds={'qwen':7200.,'llama':0.},running=False,model=None,as_of_unix=1.)
            atomic_pointer(root/'full-budget.json',ledger)
            with patch('src.checkpoint_r2_full_raw.subprocess.Popen',side_effect=AssertionError('must not spawn')):
                with self.assertRaisesRegex(ValueError,'exhausted'):supervise(['synthetic-only'],root,attempt,dict(model_seconds=7200,total_seconds=18000,setup_phase_seconds=1800))
            self.assertEqual(read(root/'full-budget.json'),ledger)

    def test_reference_fixed_five_macro_and_positive_affine_ranking(self):
        from src.checkpoint_next_full_bank_audit import endpoint,independent_auc
        rows=[dict(label=y,topic=TOPICS[0]) for y in (0,1)]
        self.assertTrue(np.isnan(endpoint(rows,np.array([0.,1.]),'atomic_auroc','topic_macro','all')[0]))
        scores=np.array([-1.,0.,0.,2.]);labels=np.array([0,1,0,1])
        self.assertEqual(independent_auc(labels,scores)[0],independent_auc(labels,7*scores+13)[0])

    def test_consistent_evaluator_error_passes_replay_but_fails_reference(self):
        from src import final_e_reporting as reporting
        original=reporting.scalar_result
        def faulty(*args,**kwargs):
            result=original(*args,**kwargs)
            result['points']=result['points']+.01;result['draws']=result['draws']+.01
            for r in result['records']:
                if r['point'] is not None:r['point']+=.01
                for k in ('ci_low','ci_high'):
                    if r.get(k) is not None:r[k]+=.01
            return result
        with tempfile.TemporaryDirectory(prefix='synthetic-final-e-') as td,Fixture(td) as f:
            f.accepted()
            args=SimpleNamespace(output=str(f.output),roots=str(f.root/'roots.json'),report=str(f.root/'report'),source_report=None,operation_receipt=str(f.root/'audit.json'))
            with patch.object(reporting,'scalar_result',side_effect=faulty):
                runner.evaluate(args,f.access)
                args.source_report=args.report;args.report=str(f.root/'audit-rebuild')
                with self.assertRaisesRegex(ValueError,'independent numerical mismatch'):runner.audit_report(args,f.access)
            self.assertEqual(read(Path(args.report)/'scalar-metrics.json')['records'],read(Path(args.source_report)/'scalar-metrics.json')['records'])
            self.assertFalse((f.root/'audit.audit.json').exists())
