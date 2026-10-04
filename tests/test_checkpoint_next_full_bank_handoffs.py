"""Synthetic 131-shard handoffs; real writer/loader, no model inference or fits."""
import copy
from contextlib import ExitStack
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from src import checkpoint_next_full_bank_handoffs as h
from src.checkpoint_next_inventory import sha
from src.checkpoint_next_adoption import policy
from src.checkpoint_next_missing import raw_rows, provenance, verify_supplement
from src.checkpoint_next_runner import _missing_acceptance
from src.checkpoint_r2_fresh_store import write_shard, pin
from src.checkpoint_r2_full_support import single_packet
from src.checkpoint_next_store import publish_json


class MixedHandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template=tempfile.TemporaryDirectory();base=Path(cls.template.name)
        cls.spec=dict(layers=2,width=3,revision='synthetic-only')
        cls.backend=dict(tokenizer_backend_sha256='synthetic-backend',tokenizer_implementation_sha256='synthetic-implementation')
        cls.p=copy.deepcopy(policy());cls.rows=raw_rows()
        old=base/'original';new=base/'successor'
        class Tokenizer:
            pad_token_id=0
            def encode(self,text,add_special_tokens,**kw):return [1,2] if add_special_tokens else [2]
            def __call__(self,text,**kw):return dict(input_ids=[[1,2]],special_tokens_mask=[[1,0]])
        with cls.patches():
            for model,entries in cls.p['original_attempts'].items():
                for e in entries:
                    path=old/'attempts'/model/e['attempt'];path.mkdir(parents=True)
                    publish_json(path/'start.json',dict(utc=1.,phase=e['phase']))
                    publish_json(path/'end.json',dict(active_seconds=e['active_seconds'],exit_code=0 if model=='qwen' and e['phase']=='missing_raw' else 1))
                    e['start_sha256']=sha(path/'start.json');e['end_sha256']=sha(path/'end.json')
            for root,model,attempt in [(old,'qwen',cls.p['original_attempts']['qwen'][0]['attempt']),(new,'llama','completed')]:
                folder=root/'missing_raw'/model;idn=h.expected_identity(model);folder.mkdir(parents=True)
                publish_json(folder/'identity.json',idn)
                rows=cls.rows;plan=[single_packet(Tokenizer(),r,1024) for r in rows]
                publish_json(folder/'token-plan.json',plan)
                for i,row in enumerate(rows):write_shard(folder/'shards'/f'{i:05d}',[row],np.zeros((1,2,3),np.float32),cls.spec,provenance(model,idn),[plan[i]])
                receipt=verify_supplement(root,model,idn)
                publish_json(folder/'token-audit.json',dict(identity=idn,queue_sha256=receipt['queue_sha256'],token_plan_sha256=receipt['token_plan_sha256'],rows=131,status='independently_reconstructed',model_loaded=False,**cls.backend))
                execution=root/'attempts'/model/attempt;execution.mkdir(parents=True,exist_ok=True)
                publish_json(execution/'execution-evidence.json',dict(status='completed',measured_seconds={'extraction':1.},peak_cuda_bytes={'allocated':0,'reserved':0}))
                publish_json(execution/'peak-before-teardown.json',dict(peak_cuda_bytes={'allocated':0,'reserved':0}))
                publish_json(folder/'completion.json',dict(identity=idn,phase='missing_raw',jobs=131,status='completed',verification=receipt,attempt=attempt,execution_evidence_sha256=sha(execution/'execution-evidence.json')))
            publish_json(old/'missing_raw/qwen/acceptance.json',_missing_acceptance(old,'qwen',h.expected_identity('qwen')))
            cls.p['original_acceptance_sha256']=sha(old/'missing_raw/qwen/acceptance.json')
            cls.write_manifest(old,base/'original-transfer.json');cls.p['original_manifest_sha256']=sha(base/'original-transfer.json')
            carry={m:sum(e['active_seconds'] for e in es) for m,es in cls.p['original_attempts'].items()}
            publish_json(new/'adoption.json',dict(schema='checkpoint-next-adoption-v1',successor_producer=h.PRODUCERS['llama'],policy_sha256=sha(h.ROOT/'config/checkpoint_next/raw_only_fix_v1.json'),execution_output=str(new),full_preflight_acceptance=False,original=dict(original_producer=h.PRODUCERS['qwen'],original_manifest_sha256=cls.p['original_manifest_sha256'],original_identities=cls.p['original_identities'],original_attempts=cls.p['original_attempts'],carry_seconds=carry,qwen_acceptance_sha256=cls.p['original_acceptance_sha256'])))
            total=carry['llama']
            for i,(name,seconds) in enumerate([('failed',2.),('completed',3.)]):
                path=new/'attempts/llama'/name;path.mkdir(parents=True,exist_ok=True)
                publish_json(path/'start.json',dict(utc=float(i+2),phase='missing_raw',adoption_sha256=sha(new/'adoption.json'),remaining_gpu_seconds=7200-total,campaign_budget_before_attempt=dict(original_seconds=carry['llama'],successor_seconds=total-carry['llama'],charged_seconds=total,remaining_seconds=7200-total)))
                publish_json(path/'end.json',dict(active_seconds=seconds,exit_code=1 if name=='failed' else 0));total+=seconds
            result=cls.verify_saved('llama',dict(campaign=str(new)),old,base/'original-transfer.json')
            publish_json(new/'missing_raw/llama/acceptance.json',result['accepted'])
            cls.write_manifest(new,base/'successor-transfer.json')

    @classmethod
    def patches(cls):
        stack=ExitStack()
        stack.enter_context(patch.object(h,'policy',return_value=cls.p))
        stack.enter_context(patch('src.checkpoint_next_missing.raw_rows',return_value=cls.rows))
        stack.enter_context(patch('src.checkpoint_next_missing.config',return_value={'models':dict.fromkeys(h.PRODUCERS,cls.spec)}))
        stack.enter_context(patch('src.checkpoint_next_runner.config',return_value={'models':dict.fromkeys(h.PRODUCERS,cls.spec)}))
        stack.enter_context(patch('src.checkpoint_r2_full_support.evidence',return_value={'backend':cls.backend}))
        return stack

    @classmethod
    def verify_saved(cls,model,entry,original,manifest):
        # Reconstruct with the real raw loader and frozen explicit identity; only
        # subprocess transport/runtime spec is synthetic, never acceptance flags.
        root=Path(entry['campaign']);result=_missing_acceptance(root,model,h.expected_identity(model))
        if model=='llama':
            a=h.read(root/'adoption.json');carry=a['original']['carry_seconds'][model]
            own=sum(r['end']['active_seconds'] for r in h.settled(root,model,True))
            result.update(adoption_sha256=sha(root/'adoption.json'),original_attempts=a['original']['original_attempts'][model],campaign_budget=dict(original_seconds=carry,successor_seconds=own,charged_seconds=carry+own,remaining_seconds=7200-carry-own))
        return dict(accepted=result,identity=h.expected_identity(model),measured_execution={'synthetic':True},model_spec=cls.spec,backend=cls.backend)

    @classmethod
    def write_manifest(cls,root,path):
        path.write_text(json.dumps({str(p.relative_to(root)):sha(p) for p in sorted(root.rglob('*')) if p.is_file()},sort_keys=True))

    @classmethod
    def tearDownClass(cls):cls.template.cleanup()

    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.base=Path(self.temp.name)
        shutil.copytree(self.template.name,self.base,dirs_exist_ok=True)
        self.original=self.base/'original';self.new=self.base/'successor'
        self.b={'schema':'checkpoint-next-full-bank-handoffs-v2','qwen':dict(campaign=str(self.original),transfer_manifest=str(self.base/'original-transfer.json'),verifier_checkout='qwen-exact-checkout'),'llama':dict(campaign=str(self.new),transfer_manifest=str(self.base/'successor-transfer.json'),verifier_checkout='llama-exact-checkout')}
        self.stack=self.patches();self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(h,'verify_in_checkout',side_effect=self.verify_saved))

    def refresh(self,model='llama'):
        e=self.b[model];self.write_manifest(Path(e['campaign']),Path(e['transfer_manifest']))

    def change_json(self,path,change):
        value=h.read(path);change(value);path.write_text(json.dumps(value))

    def test_valid_mixed_producers_preserve_originals_and_charge_failures(self):
        before={str(p):sha(p) for root in (self.original,self.new) for p in root.rglob('*') if p.is_file()}
        result=h.acceptance(self.b)
        self.assertEqual(result['status'],'transferred_independently_reopened')
        self.assertEqual(result['adoption']['total_llama_seconds'],41.1570940479869+5)
        for model in h.PRODUCERS:
            r=result['models'][model];self.assertEqual(r['identity']['producer_commit'],h.PRODUCERS[model]);self.assertEqual(len(r['verification']['receipts']),131)
            self.assertEqual(len(r['ordered_token_bindings']),131);self.assertEqual(len(r['shard_files']),524)
        self.assertEqual(before,{str(p):sha(p) for root in (self.original,self.new) for p in root.rglob('*') if p.is_file()})

    def test_llama_pending_blocks_full_bank(self):
        self.b['llama']=None
        result=h.acceptance(self.b,False);self.assertNotIn('llama',result['models']);self.assertIsNone(result['adoption'])
        with self.assertRaisesRegex(ValueError,'Llama pending'):h.acceptance(self.b)

    def test_wrong_and_swapped_producers(self):
        path=self.new/'missing_raw/llama/identity.json';original=path.read_text()
        for producer in ('CPU-HEAD',h.PRODUCERS['qwen']):
            path.write_text(original);self.change_json(path,lambda v:v.update(producer_commit=producer));self.refresh()
            with self.assertRaisesRegex(ValueError,'identity'):h.acceptance(self.b)

    def test_altered_complete_identity(self):
        self.change_json(self.new/'missing_raw/llama/identity.json',lambda v:v.update(inventory_sha256='altered'))
        self.refresh()
        with self.assertRaisesRegex(ValueError,'identity'):h.acceptance(self.b)

    def test_missing_shard_even_with_new_manifest(self):
        shutil.rmtree(self.new/'missing_raw/llama/shards/00001');self.refresh()
        with self.assertRaisesRegex(ValueError,'coverage'):h.acceptance(self.b)

    def test_corrupt_shard_even_with_new_manifest(self):
        (self.new/'missing_raw/llama/shards/00000/activations.npy').write_bytes(b'corrupt');self.refresh()
        with self.assertRaisesRegex(ValueError,'shard hash'):h.acceptance(self.b)

    def test_altered_row_binding_even_with_new_shard_hash(self):
        shard=self.new/'missing_raw/llama/shards/00000';p=shard/'rows.json'
        self.change_json(p,lambda v:v[0].update(statement='similar is not identical'))
        self.change_json(shard/'receipt.json',lambda v:v['files'].update({'rows.json':pin(p)}));self.refresh()
        with self.assertRaisesRegex(ValueError,'bindings|ordered rows'):h.acceptance(self.b)

    def test_token_plan_and_audit_mismatch(self):
        p=self.new/'missing_raw/llama/token-plan.json';self.change_json(p,lambda v:v[0]['token_ids'].append(3));self.refresh()
        with self.assertRaisesRegex(ValueError,'token plan'):h.acceptance(self.b)

    def test_independent_audit_backend_mismatch(self):
        self.change_json(self.new/'missing_raw/llama/token-audit.json',lambda v:v.update(tokenizer_backend_sha256='wrong'));self.refresh()
        with self.assertRaisesRegex(ValueError,'audit'):h.acceptance(self.b)

    def test_changed_acceptance_receipt(self):
        self.change_json(self.new/'missing_raw/llama/acceptance.json',lambda v:v.update(active_seconds=0));self.refresh()
        with self.assertRaisesRegex(ValueError,'actual acceptance'):h.acceptance(self.b)

    def test_incomplete_ledger_never_recovered_or_written(self):
        end=self.new/'attempts/llama/failed/end.json';end.unlink();self.refresh()
        with self.assertRaises((ValueError,FileNotFoundError)):h.acceptance(self.b)
        self.assertFalse(end.exists())

    def test_inconsistent_ledger_or_adoption(self):
        self.change_json(self.new/'attempts/llama/completed/start.json',lambda v:v['campaign_budget_before_attempt'].update(charged_seconds=0));self.refresh()
        with self.assertRaisesRegex(ValueError,'accounting'):h.acceptance(self.b)

    def test_adoption_does_not_replace_extraction_identity(self):
        self.change_json(self.new/'adoption.json',lambda v:v['original']['carry_seconds'].update(llama=0));self.refresh()
        with self.assertRaisesRegex(ValueError,'accounting'):h.acceptance(self.b)

    def test_nonfinite_charge_and_unexpected_phase(self):
        self.change_json(self.new/'attempts/llama/failed/end.json',lambda v:v.update(active_seconds=float('nan')));self.refresh()
        with self.assertRaisesRegex(ValueError,'nonfinite'):h.acceptance(self.b)

    def test_changed_pinned_qwen_acceptance(self):
        self.change_json(self.original/'missing_raw/qwen/acceptance.json',lambda v:v.update(active_seconds=0))
        with self.assertRaisesRegex(ValueError,'transfer'):h.acceptance(self.b)

    def test_swapped_artifact_roots(self):
        self.b['qwen']['campaign']=self.b['llama']['campaign']
        with self.assertRaisesRegex(ValueError,'transfer'):h.acceptance(self.b)

    def test_completion_identity_mismatch(self):
        self.change_json(self.new/'missing_raw/llama/completion.json',lambda v:v['identity'].update(producer_commit=h.PRODUCERS['qwen']))
        self.refresh()
        with self.assertRaisesRegex(ValueError,'completion identity'):h.acceptance(self.b)

    def test_unexpected_attempt_phase(self):
        self.change_json(self.new/'attempts/llama/failed/start.json',lambda v:v.update(phase='preflight'));self.refresh()
        with self.assertRaisesRegex(ValueError,'unexpected attempt phase'):h.acceptance(self.b)

    def test_transfer_hash_and_symlink_rejection(self):
        (self.new/'unmanifested').write_text('changed')
        with self.assertRaisesRegex(ValueError,'transfer'):h.acceptance(self.b)
        (self.new/'unmanifested').unlink();(self.new/'link').symlink_to(self.new/'adoption.json')
        with self.assertRaisesRegex(ValueError,'symlinks'):h.acceptance(self.b)


class VerifierDispatchTests(unittest.TestCase):
    def test_swapped_verifier_is_blocked_before_subprocess(self):
        with tempfile.TemporaryDirectory() as d:
            entry=dict(verifier_checkout=d,campaign=d)
            with patch.object(h.subprocess,'check_output',side_effect=[str(Path(d).resolve()),h.PRODUCERS['qwen']]),patch.object(h.subprocess,'run',side_effect=AssertionError('must not dispatch')):
                with self.assertRaisesRegex(ValueError,'wrong/swapped'):h.verify_in_checkout('llama',entry,d,d)

    def test_exact_producer_dispatch_has_no_accept_audit_or_recovery(self):
        with tempfile.TemporaryDirectory() as d:
            entry=dict(verifier_checkout=d,campaign=d)
            with patch.object(h.subprocess,'check_output',side_effect=[str(Path(d).resolve()),h.PRODUCERS['llama'],'',h.PRODUCERS['llama'],'']),patch.object(h.subprocess,'run',return_value=type('Result',(),dict(returncode=0,stdout='{}',stderr=''))()) as run:
                self.assertEqual(h.verify_in_checkout('llama',entry,d,d),{})
                self.assertEqual(run.call_args.kwargs['cwd'],Path(d).resolve());self.assertIn("verify(root,'missing_raw',model)",run.call_args.args[0][2])
                self.assertNotIn('accept-missing',run.call_args.args[0]);self.assertEqual(run.call_args.kwargs['env']['OPENBLAS_NUM_THREADS'],'1')


if __name__=='__main__':unittest.main()
