"""Single-thread synthetic integration regressions; no research models or fitting."""
import copy
import json
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from src.checkpoint_next_measurement import ChatBackend, continuation_logprob
from src.checkpoint_next_inventory import inventory, digest, sha
from src.checkpoint_next_store import publish_json


class Tensor(np.ndarray):
    def float(self):return self.astype(np.float32).view(Tensor)
    def double(self):return self.astype(np.float64).view(Tensor)
    def cpu(self):return self
    def unsqueeze(self,axis):return np.expand_dims(self,axis).view(Tensor)
    def gather(self,axis,indices):return np.take_along_axis(self,indices,axis).view(Tensor)
    @property
    def device(self):return 'synthetic'


def tensor(v,**kw):return np.asarray(v).view(Tensor)


class Torch:
    long=np.int64
    cuda=SimpleNamespace(synchronize=lambda:None)
    inference_mode=staticmethod(nullcontext)
    tensor=staticmethod(tensor)
    arange=staticmethod(lambda n,**kw:tensor(np.arange(n)))
    ones=staticmethod(lambda shape,**kw:tensor(np.ones(shape,dtype=int)))
    isfinite=staticmethod(np.isfinite)
    @staticmethod
    def log_softmax(v,dim):
        a=v-np.max(v,axis=dim,keepdims=True)
        return tensor(a-np.log(np.exp(a).sum(axis=dim,keepdims=True)))


class Config:
    def __init__(self,**kwargs):self.__dict__.update(kwargs)


class SyntheticModel:
    """Pinned GenerationMixin policy: grow attention mask, retain explicit positions.

    Every prepared input is checked; distinct step-dependent logits exercise actual
    decoding, including prompt preservation and native EOS. No repeat-only oracle.
    """
    def __init__(self,eos_at=None):self.eos_at=eos_at;self.trace=[]
    def prepare_inputs_for_generation(self,input_ids,attention_mask,position_ids=None,**kw):
        if position_ids is None:position_ids=attention_mask.cumsum(-1)-1
        return dict(input_ids=input_ids,attention_mask=attention_mask,position_ids=position_ids)
    def __call__(self,input_ids,attention_mask,position_ids,**kw):
        n=input_ids.shape[-1]
        assert attention_mask.shape[-1]==position_ids.shape[-1]==n
        assert position_ids[0].tolist()==list(range(n))
        logits=np.zeros((1,n,128),np.float32)
        for i in range(n):
            step=i-1
            token=127 if self.eos_at is not None and step==self.eos_at else 65+(i%20)
            logits[0,i,token]=4+i/10
        return SimpleNamespace(logits=tensor(logits))
    def generate(self,input_ids,attention_mask,generation_config,**kwargs):
        cfg=generation_config
        assert not cfg.do_sample and not cfg.use_cache and cfg.num_beams==1 and cfg.max_new_tokens==8
        prefix=input_ids.copy();mask=attention_mask.copy()
        for _ in range(cfg.max_new_tokens):
            prepared=self.prepare_inputs_for_generation(prefix,mask,**kwargs)
            self.trace.append({k:v.tolist() for k,v in prepared.items()})
            out=self(**prepared);token=int(out.logits[0,-1].argmax())
            prefix=tensor(np.concatenate([prefix,[[token]]],axis=1))
            mask=tensor(np.concatenate([mask,[[1]]],axis=1))
            if token in cfg.eos_token_id:break
        return prefix


def backend(eos_at=None):
    b=ChatBackend.__new__(ChatBackend);b.torch=Torch();b.model=SyntheticModel(eos_at)
    b.tokenizer=SimpleNamespace(bos_token_id=0,all_special_ids=[127],
        decode=lambda ids,**kw:''.join('<EOS>' if i==127 else chr(i) for i in ids))
    b.raw=SimpleNamespace(pad_id=127);b.termination_ids=[127]
    return b


class GenerationTests(unittest.TestCase):
    def test_growing_sequence_eight_steps_and_early_native_end(self):
        with patch.dict('sys.modules',{'transformers':SimpleNamespace(GenerationConfig=Config)}):
            for model_name in ('qwen','llama'):
                for eos_at,steps in ((None,8),(2,3)):
                    b=backend(eos_at);b.model_name=model_name
                    result=b.generate({'prompt_ids':[1,2]})
                    self.assertEqual(len(result['token_ids']),steps)
                    self.assertEqual(result['token_ids'],[66+i for i in range(steps)] if eos_at is None else [66,67,127])
                    for step,p in enumerate(b.model.trace):
                        self.assertEqual(len(p['input_ids'][0]),2+step)
                        self.assertEqual(p['attention_mask'],[[1]*(2+step)])
                        self.assertEqual(p['position_ids'],[list(range(2+step))])
                    self.assertIn('position_ids',b.arguments([1,2]))
    def test_old_policy_fails_second_step(self):
        b=backend()
        with self.assertRaises(AssertionError):
            b.model.generate(**b.arguments([1,2]),generation_config=Config(do_sample=False,use_cache=False,num_beams=1,max_new_tokens=8,eos_token_id=[127]))
    def test_actual_score_and_independent_prospective_oracles(self):
        with patch.dict('sys.modules',{'transformers':SimpleNamespace(GenerationConfig=Config)}):
            b=backend();packet=dict(prompt_ids=[1,2],candidates={'True':[66,67],'False':[3,5,7]})
            result=b.score(packet)
            for name,tokens in packet['candidates'].items():
                logits=b.model(**b.arguments(packet['prompt_ids']+tokens)).logits[0].tolist()
                expected=continuation_logprob(logits,packet['prompt_ids'],tokens)
                self.assertAlmostEqual(result[name.lower()]['logprob'],expected['logprob'],places=5)
            generated=b.generate(packet)
            check=b.preflight_checks(packet,result,generated,0.00001)
            self.assertEqual(len(check['prepared_steps']),8)
            bad=copy.deepcopy(result);bad['true']['token_logprobs'][0]+=1
            with self.assertRaisesRegex(ValueError,'causal'):b.preflight_checks(packet,bad,generated,0.00001)
            wrong=copy.deepcopy(generated);wrong['token_ids'][0]+=1
            with self.assertRaisesRegex(ValueError,'greedy'):b.preflight_checks(packet,result,wrong,0.00001)


class QueueTests(unittest.TestCase):
    def test_deterministic_queue_metadata(self):
        from src.checkpoint_next_missing import metadata,validate_queue
        one=metadata();self.assertEqual(one,metadata())
        self.assertEqual(one['unique_texts_per_model'],131)
        self.assertEqual(one['additional_reused_bindings'],131)
        self.assertEqual(len(validate_queue()['rows']),131)
        self.assertIsNone(one['runtime_seconds'])
    def test_queue_evidence_corruption_blocks(self):
        from src.checkpoint_next_missing import settings
        with patch('src.checkpoint_next_missing.sha',return_value='wrong'):
            with self.assertRaisesRegex(ValueError,'evidence'):settings()


class FailureTests(unittest.TestCase):
    def test_flagged_fit_all_attempts_saved_and_prediction_refused(self):
        from src.checkpoint_next_chat import preserve_fit
        from src.selection_repair_objectives import AttemptRecord,ReadoutFit,OptimizerSettings,fit_p_preprocessing
        pre=fit_p_preprocessing(np.array([[0.,1.],[1.,1.]]))
        attempt=AttemptRecord(2000,np.zeros(3),np.ones(3),1.,.9,2000,2100,1,'budget',False,.1,.2,True,True,True,'flagged')
        fit=ReadoutFit('r0',np.ones(2),1.,pre,OptimizerSettings(),(attempt,attempt),'flagged')
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'fit'
            with self.assertRaisesRegex(ValueError,'preserved'):preserve_fit(p,fit,pre,dict(head='synthetic'))
            r=json.loads((p/'fit-status.json').read_text())
            self.assertFalse(r['prediction_usable']);self.assertEqual(len(r['attempts']),2)
            self.assertEqual(r['attempts'][0]['initial_parameters'],[0.,0.,0.])
            self.assertEqual(r['attempts'][1]['final_parameters'],[1.,1.,1.])
            self.assertTrue((p/'parameters.npz').exists());self.assertFalse((p/'scores.npz').exists())
            with self.assertRaises(RuntimeError):fit.decision_function(np.ones((1,2)))
    def test_shared_ledger_all_phases_and_interrupted_charge(self):
        from src.checkpoint_next_runner import attempts_elapsed
        with tempfile.TemporaryDirectory() as d:
            for phase,seconds in [('missing_raw',10),('preflight',20),('launch',30)]:
                p=Path(d)/'attempts/qwen'/phase;p.mkdir(parents=True)
                publish_json(p/'start.json',dict(phase=phase));publish_json(p/'end.json',dict(active_seconds=seconds))
            self.assertEqual(attempts_elapsed(d,'qwen'),60)


class ReportingAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.x=inventory()
    def test_isolated_logical_weighting(self):
        from src.checkpoint_next_reporting import isolated_rates
        rows=[dict(r,likelihood_correct=True,generation_correct=False,generation_decision=None,
            likelihood_decision=1,format_failure=True) for r in self.x['bindings'] if r['group']=='behavior_isolated' and r['prompt']==1]
        report=isolated_rates(rows)
        self.assertEqual((report[0]['rows'],report[0]['unique_facts']),(200,150))
        self.assertEqual(report[0]['format_failures'],200);self.assertEqual(report[0]['valid_factual_errors'],0)
        self.assertIsNone(report[0]['disagreement_rate'])
    def fixtures(self,root):
        from src.checkpoint_r2_fresh_store import publish
        x=self.x;raw=[];artifacts=[]
        groups={g:[r['logical_id'] for r in x['bindings'] if r['role']=='chat' and r['group']==g]
                for g in ('P15','TC_control_and_final_refit','behavior_D')}
        for head in x['heads']:
            folder=root/head['id'];folder.mkdir(parents=True)
            meta=dict(head,preprocessing_membership='original_P15',P15_identities_sha256=digest(groups['P15']),regime='prompt1_chat')
            raw.append(dict(meta,regime='raw',frozen_rule_source_sha256='a'*64))
            publish(folder/'parameters.npz',lambda f:np.savez(f,weights=[1.]))
            publish(folder/'scores.npz',lambda f:np.savez(f,**{g:np.zeros(len(v)) for g,v in groups.items()}))
            status=dict(status='converged',prediction_usable=True)
            publish_json(folder/'fit-status.json',status)
            receipt=dict(status,head=meta,producer_commit='b'*40,preprocessing_source='original chat-P15 only',
                objective=head['objective'],parameters_sha256=sha(folder/'parameters.npz'),scores_sha256=sha(folder/'scores.npz'),
                fit_status_sha256=sha(folder/'fit-status.json'),score_identities=groups,
                verification=dict(model=head['model'],status='verified',identity=dict(producer_commit='b'*40,inventory_sha256=digest(x))))
            publish_json(folder/'receipt.json',receipt);artifacts.append(dict(id=head['id'],receipt_sha256=sha(folder/'receipt.json')))
        publish_json(root/'completion.json',dict(status='completed',producer_commit='b'*40,heads=artifacts))
        return raw
    def test_eight_heads_adapter_and_identity_corruption_guards(self):
        from src.checkpoint_next_reporting import verified_chat_heads
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);raw=self.fixtures(root)
            heads=verified_chat_heads(self.x,root,'b'*40,raw)
            self.assertEqual(len(heads),8);self.assertTrue(all(len(h['scores'])==800 for h in heads))
            bad=copy.deepcopy(raw);bad[0]['layer']=99
            with self.assertRaisesRegex(ValueError,'mismatch'):verified_chat_heads(self.x,root,'b'*40,bad)
            bad=copy.deepcopy(raw);bad[0]['fitting_identities_sha256']='full-training'
            with self.assertRaisesRegex(ValueError,'mismatch'):verified_chat_heads(self.x,root,'b'*40,bad)
            (root/heads[0]['id']/'scores.npz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'hash'):verified_chat_heads(self.x,root,'b'*40,raw)


    def test_chat_800_comparison_keeps_full_raw_separate(self):
        import csv
        from src.checkpoint_next_inventory import ROOT
        from src.checkpoint_next_reporting import chat_behavior_comparison
        rows=[dict(r,likelihood_correct=True,generation_correct=True,format_failure=False)
              for r in self.x['bindings'] if r['group']=='behavior_compound' and r['prompt']==1]
        with (ROOT/'data/checkpoint_r2_v1/compound_bindings.csv').open() as handle:
            full=[r['example_id'] for r in csv.DictReader(handle) if r['group']=='D_bare']
        with tempfile.TemporaryDirectory() as d:
            raw=self.fixtures(Path(d))[0]
            chat=dict(raw,regime='prompt1_chat',head_id=raw['id'],
                      logical_ids=[r['logical_id'] for r in rows],scores=[1.]*800)
            raw.update(logical_ids=full,scores=[1.]*7728)
            result=chat_behavior_comparison(rows,chat,raw)
            self.assertEqual((result['rows'],result['pairs']),(800,50))
            self.assertEqual(sum(r['rows'] for r in result['chat_joint']['records']),1600)
            raw.update(logical_ids=chat['logical_ids'],scores=chat['scores'])
            with self.assertRaisesRegex(ValueError,'full-D'):chat_behavior_comparison(rows,chat,raw)


class ExecutionEvidenceTests(unittest.TestCase):
    def test_peak_saved_before_teardown_and_all_duration_categories(self):
        import argparse,os
        from contextlib import ExitStack
        from src.checkpoint_next_runner import worker,verify_execution_evidence
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);attempt=out/'attempts/qwen/one';attempt.mkdir(parents=True)
            root=out/'missing_raw/qwen';root.mkdir(parents=True)
            args=argparse.Namespace(output=d,attempt=str(attempt),model='qwen',parent_pid=os.getppid(),cache=d)
            expected={k:'same' for k in ('tokenizer_backend_sha256','model_implementation_sha256','tokenizer_implementation_sha256','actual_flags')}
            fake=SimpleNamespace(raw=SimpleNamespace(details=expected,peak=lambda:dict(allocated=11,reserved=22)))
            def close():self.assertTrue((attempt/'peak-before-teardown.json').exists())
            fake.close=close
            cfg=dict(models=dict(qwen={}),layers=dict(qwen=[17,18]),runtime_lock={},execution={})
            patches={'src.checkpoint_next_runner.config':cfg,'src.checkpoint_next_runner.frozen':{},
                'src.checkpoint_next_runner.identity':{'producer_commit':'a'*40},
                'src.checkpoint_next_runner.platform.system':'Linux',
                'src.checkpoint_r2_fresh_backend.runtime_receipt':{},
                'src.checkpoint_r2_full_support.evidence':{'backend':expected},
                'src.checkpoint_r2_full_support.verify_snapshot':{},
                'src.checkpoint_r2_full_raw.runtime_matches':None,
                'src.checkpoint_next_measurement.ChatBackend':fake,
                'src.checkpoint_next_missing.settings':{},
                'src.checkpoint_next_missing.extract':dict(status='completed',jobs=131)}
            with ExitStack() as stack:
                for name,value in patches.items():stack.enter_context(patch(name,return_value=value))
                stack.enter_context(patch('ctypes.CDLL',return_value=SimpleNamespace(prctl=lambda *a:0)))
                worker(args,attempt,'missing_raw',{})
            completion=json.loads((root/'completion.json').read_text())
            evidence=verify_execution_evidence(out,'qwen',completion)
            self.assertEqual(evidence['peak_cuda_bytes'],dict(allocated=11,reserved=22))
            self.assertEqual(set(evidence['measured_seconds']),{'loading','tokenization','likelihood','generation',
                'chat_extraction','raw_extraction','preflight_checks','writing_verification','teardown'})


class MissingRecoveryTests(unittest.TestCase):
    def test_real_writer_loader_exact_131_resume_quarantine_and_corruption(self):
        from contextlib import ExitStack
        from src.checkpoint_next_missing import extract,verify_supplement
        from src.checkpoint_r2_full_support import single_packet
        from src.checkpoint_r2_fresh_inputs import text_hash
        class Tokenizer:
            pad_token_id=0
            def encode(self,text,add_special_tokens,**kw):return [1,2] if add_special_tokens else [2]
            def __call__(self,text,**kw):return dict(input_ids=[[1,2]],special_tokens_mask=[[1,0]])
        tokenizer=Tokenizer();spec=dict(layers=2,width=3)
        rows=[dict(id=str(i),statement=str(i),statement_sha256=text_hash(str(i))) for i in range(131)]
        calls=[]
        def forward(rs,batch):
            self.assertEqual(batch,1);calls.append(rs[0]['id'])
            return np.zeros((1,2,3),np.float32),[single_packet(tokenizer,rs[0],1024)]
        fake=SimpleNamespace(raw=SimpleNamespace(spec=spec,tokenizer=tokenizer,settings={'max_tokens':1024},forward=forward))
        with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
            for name,value in [('raw_rows',rows),('config',dict(models=dict(qwen=spec)))]:
                stack.enter_context(patch('src.checkpoint_next_missing.'+name,return_value=value))
            stack.enter_context(patch('src.checkpoint_r2_fresh_store.storage_probe',return_value={'synthetic_reservation':True}))
            out=Path(d);attempt=out/'attempt1';attempt.mkdir()
            args=SimpleNamespace(output=d,attempt=str(attempt),model='qwen')
            costs=dict(tokenization=0.,writing_verification=0.,raw_extraction=0.)
            partial=out/'missing_raw/qwen/shards/00000';partial.mkdir(parents=True)
            (partial/'interrupted').write_text('keep')
            summary=extract(args,fake,{'producer_commit':'a'*40},costs)
            self.assertEqual(len(calls),131);self.assertEqual(summary['verification']['unique_texts'],131)
            self.assertEqual(next((partial.parent.parent/'quarantine').iterdir()).joinpath('interrupted').read_text(),'keep')
            args.attempt=str(out/'attempt2');Path(args.attempt).mkdir()
            extract(args,fake,{'producer_commit':'a'*40},costs)
            self.assertEqual(len(calls),131)
            (partial/'tokens.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'hash'):verify_supplement(d,'qwen',{'producer_commit':'a'*40})

if __name__=='__main__':unittest.main()
