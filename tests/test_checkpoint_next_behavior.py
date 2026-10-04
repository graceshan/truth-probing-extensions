"""Focused synthetic tests. No model dependencies, inference, or research heads."""
import copy
import gzip
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from src.checkpoint_next_inventory import config, digest, inventory, missing_raw_interface, read
from src.checkpoint_next_measurement import render_packet, continuation_logprob, likelihood_result, parse_generation, native_terminations
from src.checkpoint_next_reporting import expand, isolated_mask, joint_errors, match_raw_chat, rates, aucs, report
from src.checkpoint_next_store import publish_json, recover, save_shard, verify_shard
from src.checkpoint_next_runner import attempts_elapsed, authorized


class Tokenizer:
    chat_template='synthetic native template'
    all_special_ids=[999]
    eos_token_id=999
    def __init__(self,model='qwen',bad_boundary=False):self.model=model;self.bad_boundary=bad_boundary
    def encode(self,text,**kwargs):
        ids=[ord(c) for c in text]
        if self.bad_boundary and text.endswith(('True','False')):ids[0]+=1
        return ids
    def decode(self,ids,**kwargs):return ''.join('<END>' if i==999 else chr(i) for i in ids)
    def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=False):
        assert messages==[{'role':'user','content':messages[0]['content']}]
        if self.model=='qwen':
            s='<|im_start|>system\nimplicit default<|im_end|>\n<|im_start|>user\n'+messages[0]['content']+'<|im_end|>\n<|im_start|>assistant\n'
        else:
            s='<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n'+messages[0]['content']+'<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
        return self.encode(s) if tokenize else s


def synthetic(job):
    packet=render_packet(Tokenizer(),job['user_content'],'qwen')
    terms=lambda ids:dict(logprob=math.fsum([-1.]*len(ids)),token_logprobs=[-1.]*len(ids))
    r=dict(id=job['id'],job_sha256=digest(job),packet=packet,seconds={})
    if job['behavior']:
        r['likelihood']=likelihood_result(terms(packet['candidates']['True']),terms(packet['candidates']['False']))
        r['generation']=parse_generation(Tokenizer(),[ord(c) for c in 'True']+[999],[999])
    if job['chat']:r.update(chat_index=0,saved_layers=[17,18])
    return r


class MeasurementTests(unittest.TestCase):
    def test_native_templates_no_system_and_boundary(self):
        for model in ('qwen','llama'):
            p=render_packet(Tokenizer(model),'exact input',model)
            self.assertEqual(p['system_messages'],0)
            self.assertEqual(p['candidates']['True'],[ord(c) for c in 'True'])
            self.assertEqual(p['readout_index'],len(p['prompt_ids'])-1)
            self.assertNotIn('implicit default',p['rendered'])
        with self.assertRaisesRegex(ValueError,'boundary'):
            render_packet(Tokenizer(bad_boundary=True),'x','qwen')

    def test_full_sequence_causal_index_and_no_normalization(self):
        p=[0,0];c=[1,2];logits=[[20,0,0],[0,3,0],[0,0,4],[99,0,0]]
        score=continuation_logprob(logits,p,c)
        expected=(3-math.log(2+math.exp(3)))+(4-math.log(2+math.exp(4)))
        self.assertAlmostEqual(score['logprob'],expected)
        self.assertEqual(len(score['token_logprobs']),2)
        uniform=continuation_logprob([[0]*3]*4,p,c)
        self.assertAlmostEqual(uniform['logprob'],-2*math.log(3))
        with self.assertRaises(ValueError):continuation_logprob([[float('nan')]*3]*4,p,c)
        self.assertEqual(likelihood_result(score,score)['decision'],1)

    def test_native_end_of_turn_sets_without_generation_download(self):
        class Native:
            def __init__(self,names):
                self.names=names;self.all_special_ids=list(range(len(names)));self.eos_token_id=0
            def convert_tokens_to_ids(self,name):return self.names.index(name)
            def decode(self,ids,**kwargs):return self.names[ids[0]]
        q=Native(['<|im_end|>','<|endoftext|>'])
        l=Native(['<|end_of_text|>','<|eom_id|>','<|eot_id|>'])
        self.assertEqual(native_terminations(q,0,'qwen'),[0,1])
        self.assertEqual(native_terminations(l,0,'llama'),[0,1,2])
        with self.assertRaises(ValueError):native_terminations(q,5,'qwen')

    def test_strict_generation(self):
        t=Tokenizer()
        for text,label in [(' True \n',1),('False',0)]:
            r=parse_generation(t,[ord(c) for c in text]+[999],[999]);self.assertEqual(r['decision'],label)
        for text in ('true','FALSE','True.','True or','Yes','', 'True\nX'):
            self.assertTrue(parse_generation(t,[ord(c) for c in text],[999])['format_failure'])
        self.assertTrue(parse_generation(t,[ord(c) for c in 'True']+[999,ord('X')],[999])['format_failure'])
        with self.assertRaises(ValueError):parse_generation(t,[1]*9,[999])


class InventoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.x=inventory()
    def test_counts_and_exact_frozen_exposure(self):
        a=self.x['accounting']
        self.assertEqual((a['compounds'],a['isolated_bindings'],a['isolated_unique_facts']),(800,200,150))
        self.assertEqual((a['behavior_unique_inputs_per_model'],a['chat_unique_inputs_per_model'],a['union_unique_inputs_per_model']),(1900,11578,12678))
        self.assertEqual(a['exact_prompted_text_overlap_per_model'],800)
        self.assertEqual(len(self.x['heads']),8)
        self.assertEqual({(h['model'],h['layer']) for h in self.x['heads']},{('qwen',17),('qwen',18),('llama',15),('llama',12)})
        self.assertTrue(all(j['split']=='train' for j in self.x['preflight']))
        self.assertEqual(len(self.x['preflight']),40)
        self.assertEqual(config()['runtime_lock']['packages']['transformers'],'4.56.2')
        self.assertIsNone(config()['forecast']['runtime_seconds'])
    def test_missing_raw_is_text_deduplicated_and_exact(self):
        required=read('memberships.json')['outer_train'][:10]
        available=[dict(statement=required[0]['statement'],statement_sha256=required[0]['statement_sha256'],
            representation_sha256='accepted',loader_verified=True,token_ids=[1,2])]
        out=missing_raw_interface(required,available,'accepted')
        self.assertEqual(out['required_bindings'],10)
        self.assertEqual(out['reused_bindings'],sum(r['statement']==required[0]['statement'] for r in required))
        changed=copy.deepcopy(required);changed[0]['label']='2'
        with self.assertRaises(ValueError):missing_raw_interface(changed,available,'accepted')
        available[0]['loader_verified']=False
        with self.assertRaises(ValueError):missing_raw_interface(required,available,'accepted')


class ChatPreparationTests(unittest.TestCase):
    def test_existing_objective_and_P15_only_preprocessing(self):
        from src.checkpoint_next_chat import prepare_blocks
        from src.selection_repair_objectives import objective_and_gradient
        x=inventory();p=[r for r in x['bindings'] if r['group']=='P15'];tc=[r for r in x['bindings'] if r['group']=='TC_control_and_final_refit']
        pf=np.column_stack((np.arange(2778)/2778,np.ones(2778)))
        cf=np.column_stack((np.arange(8000)/8000+5,np.full(8000,9.)))
        for family in ('R0','C_clean'):
            head=next(h for h in x['heads'] if h['family']==family)
            pre,block=prepare_blocks(head,p,pf,tc,cf)
            self.assertEqual(pre.p_rows,2778)
            self.assertAlmostEqual(pre.mean[0],pf[:,0].mean())
            self.assertTrue(pre.constant_mask[1])
            self.assertTrue(np.all(block.features[:,1]==0))
            loss,gradient=objective_and_gradient(np.zeros(3),'r0',block)
            self.assertAlmostEqual(loss,math.log(2))
            self.assertTrue(np.isfinite(gradient).all())
        head=next(h for h in x['heads'] if h['family']=='R0')
        with self.assertRaises(ValueError):prepare_blocks(head,p[:-1],pf[:-1],tc,cf)


class RecoveryTests(unittest.TestCase):
    def test_publish_reopen_resume_corruption_stop_and_preserve_partial(self):
        job=dict(id='x',user_content='fixture',behavior=True,chat=True)
        rec=synthetic(job);identity={'producer_commit':'a'*40};state=np.zeros((2,3),np.float16)
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'shard';q=Path(d)/'quarantine'
            save_shard(p,[job],[rec],[state],identity,[17,18],3)
            self.assertTrue(recover(p,[job],identity,[17,18],3,q))
            with self.assertRaises(FileExistsError):publish_json(p/'commit.json',{})
            with self.assertRaises(ValueError):verify_shard(p,[job],{'producer_commit':'b'*40},[17,18],3)
            (p/'records.json').write_text('{}')
            with self.assertRaises(ValueError):recover(p,[job],identity,[17,18],3,q)
            self.assertTrue(p.exists());self.assertFalse(q.exists())
            partial=Path(d)/'partial';partial.mkdir();(partial/'saved').write_text('keep')
            self.assertFalse(recover(partial,[job],identity,[17,18],3,q))
            self.assertEqual(next(q.iterdir()).joinpath('saved').read_text(),'keep')
    def test_supervisor_stops_setup_at_cap_without_model_loading(self):
        import argparse,signal
        from src.checkpoint_next_runner import supervise
        class Process:
            pid=123456789
            returncode=None
            def poll(self):return self.returncode
            def wait(self,timeout=None):self.returncode=-9;return -9
        proc=Process()
        with tempfile.TemporaryDirectory() as d:
            args=argparse.Namespace(model='llama',output=d,authorization='unused',cache=d,original_campaign=d,original_manifest=d)
            publish_json(Path(d)/'adoption.json',{'synthetic':True})
            cfg={'caps':{'active_gpu_seconds_per_model':7200,'setup_seconds_per_phase':0}}
            approval={'missing_raw_active_seconds':0,'missing_raw_accounting_sha256':'a'*64}
            with patch('src.checkpoint_next_adoption.receipt',return_value={'original':{'carry_seconds':{'llama':41.1570940479869}}}),patch('src.checkpoint_next_runner.config',return_value=cfg),patch('src.checkpoint_next_runner.frozen',return_value={}),patch('src.checkpoint_next_runner.authorized',return_value=approval),patch('src.checkpoint_next_runner.identity',return_value={'producer_commit':'a'*40}),patch('src.checkpoint_next_runner.subprocess.Popen',return_value=proc),patch('src.checkpoint_next_runner.os.killpg') as kill:
                with self.assertRaisesRegex(ValueError,'stopped'):supervise(args,'missing_raw')
                kill.assert_called_with(proc.pid,signal.SIGKILL)
            end=next(Path(d).glob('attempts/llama/*/end.json'))
            start=json.loads(end.with_name('start.json').read_text())
            self.assertEqual(start['remaining_gpu_seconds'],7158.842905952013)
            self.assertEqual(start['campaign_budget_before_attempt']['original_seconds'],41.1570940479869)
            self.assertEqual(json.loads(end.read_text())['status'],'setup cap')

    def test_crash_charge_does_not_reset_budget(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'attempts/qwen/one';p.mkdir(parents=True)
            publish_json(p/'start.json',{'utc':1});publish_json(p/'active.json',{'utc':10})
            with patch('src.checkpoint_next_runner.time.time',return_value=110):
                self.assertEqual(attempts_elapsed(d,'qwen'),100)
            with patch('src.checkpoint_next_runner.time.time',return_value=210):
                self.assertEqual(attempts_elapsed(d,'qwen'),100)


class ReportingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=inventory()
        cls.records=[synthetic(j) for j in cls.x['jobs'] if j['behavior'] and j['prompt']==1]
    def test_mask_only_isolated_four_correct(self):
        rows=expand(self.x,self.records,1)
        for r in rows:r['likelihood_correct']=True
        self.assertEqual(len(isolated_mask(rows)),50)
        compounds=[r for r in rows if r['group']=='behavior_compound']
        for r in compounds:r['likelihood_correct']=False
        self.assertEqual(len(isolated_mask(rows)),50)
        isolated=next(r for r in rows if r['group']=='behavior_isolated');isolated['likelihood_correct']=False
        self.assertEqual(len(isolated_mask(rows)),49)
    def test_denominators_format_vs_factual(self):
        rows=expand(self.x,self.records,1)[:3]
        rows[0].update(format_failure=True,generation_correct=False,generation_decision=None)
        rows[1].update(format_failure=False,generation_correct=False,generation_decision=0,likelihood_decision=1)
        rows[2].update(format_failure=False,generation_correct=True,generation_decision=1,likelihood_decision=1)
        r=rates(rows)
        self.assertEqual((r['rows'],r['valid_generated'],r['format_failures'],r['valid_factual_errors'],r['disagreement_denominator']),(3,2,1,1,2))
        self.assertAlmostEqual(r['generation_accuracy'],1/3)
    def test_joint_no_threshold_for_auroc_only(self):
        rows=[r for r in expand(self.x,self.records,1) if r['group']=='behavior_compound']
        probe=dict(logical_ids=[r['logical_id'] for r in rows],scores=[1.]*800,decision_rule=None)
        self.assertIsNone(joint_errors(rows,probe)['joint_errors'])
        probe.update(decision_rule='zero_logit_ties_true',frozen_rule_source_sha256='a'*64)
        out=joint_errors(rows,probe)
        self.assertEqual(sum(r['rows'] for r in out['records']),1600)
        probe['decision_rule']='D_fit_threshold'
        with self.assertRaises(ValueError):joint_errors(rows,probe)
        probe['decision_rule']=None;probe['logical_ids']=probe['logical_ids'][:-1]
        with self.assertRaises(ValueError):joint_errors(rows,probe)
    def test_raw_chat_match_refuses_full_training_and_layer_search(self):
        raw=dict(model='qwen',family='R0',layer=17,fitting_identities_sha256='fit',D_support_sha256='D',
            preprocessing_membership='original_P15',P15_identities_sha256='P',regime='raw')
        chat=dict(raw,regime='prompt1_chat')
        self.assertEqual(match_raw_chat(raw,chat)['status'],'matched')
        chat['preprocessing_membership']='full_training'
        with self.assertRaises(ValueError):match_raw_chat(raw,chat)
    def test_empty_mask_undefined_macro_and_small_shared_bootstrap(self):
        out=report(self.x,self.records,1,dict(replicates=20,seed=1729,minimum_valid=1))
        self.assertEqual(out['populations']['full_primary']['rows'],800)
        secondary=out['populations']['isolated_correct_secondary']
        self.assertEqual(secondary['rows'],0)
        self.assertTrue(all(r['point'] is None for r in secondary['ranking']))
        self.assertEqual(out['bootstrap']['unit'],'topic-stratified entity; endpoint multiplicity products')


if __name__=='__main__':unittest.main()
