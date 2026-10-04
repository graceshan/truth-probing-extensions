"""Raw-only successor regressions. One CPU thread; no model loading or E data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from src.checkpoint_next_adoption import policy, phase_allowed, verify_original, create, verify, receipt, budget, POLICY
from src.checkpoint_next_inventory import sha
from src.checkpoint_next_store import publish_json


class RawSeparationTests(unittest.TestCase):
    def test_raw_backend_never_initializes_chat_or_termination(self):
        from src.checkpoint_next_runner import backend_for_phase
        sentinel=object()
        with patch('src.checkpoint_r2_fresh_backend.FreshBackend',return_value=sentinel) as raw, \
             patch('src.checkpoint_next_measurement.ChatBackend',side_effect=AssertionError('chat initialized')), \
             patch('src.checkpoint_next_measurement.native_terminations',side_effect=AssertionError('termination inspected')):
            self.assertIs(backend_for_phase('missing_raw',{'revision':'pinned'},'snapshot',{'batch_size':1},'llama',[15,12]),sentinel)
            raw.assert_called_once_with({'revision':'pinned'},'snapshot',{'batch_size':1})

    def test_only_llama_missing_raw_allowed(self):
        phase_allowed('llama','missing_raw')
        for model,phase in [('qwen','missing_raw'),('qwen','preflight'),('llama','preflight'),('llama','launch')]:
            with self.assertRaisesRegex(ValueError,'blocked'):phase_allowed(model,phase)

    def test_original_charge_and_scientific_pins(self):
        p=policy()
        self.assertEqual(sum(e['active_seconds'] for e in p['original_attempts']['qwen']),160.83801023702836)
        self.assertEqual(sum(e['active_seconds'] for e in p['original_attempts']['llama']),41.1570940479869)
        self.assertEqual(len(p['qwen_partial_ids']),16)
        self.assertFalse(p['full_preflight_acceptance'])

    def test_block_before_backend_or_authorization_loading(self):
        from src.checkpoint_next_runner import authorized
        for model,phase in [('qwen','missing_raw'),('llama','preflight'),('qwen','launch')]:
            with self.assertRaisesRegex(ValueError,'blocked'):
                authorized(SimpleNamespace(model=model),phase,{})

    def test_raw_authorization_has_no_behavior_or_chat_prerequisite(self):
        from src.checkpoint_next_runner import authorized, identity
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);publish_json(out/'adoption.json',{'synthetic':True})
            a=dict(identity('llama',{}),phase='missing_raw',reviewed=True,E=False,
                   adoption_sha256=sha(out/'adoption.json'))
            publish_json(out/'authorization.json',a)
            args=SimpleNamespace(model='llama',output=d,original_campaign=d,original_manifest=d,
                                 authorization=out/'authorization.json')
            with patch('src.checkpoint_next_adoption.verify'), \
                 patch('src.checkpoint_next_runner.producer',return_value=a['producer_commit']), \
                 patch('src.checkpoint_next_runner.subprocess.check_output',return_value=''):
                self.assertEqual(authorized(args,'missing_raw',{}),a)
            a['adoption_sha256']='wrong';(out/'authorization.json').write_text(json.dumps(a))
            with patch('src.checkpoint_next_adoption.verify'):
                with self.assertRaisesRegex(ValueError,'adoption authorization'):authorized(args,'missing_raw',{})


class AdoptionTests(unittest.TestCase):
    def fixture(self,root):
        old=root/'original';old.mkdir();manifest=root/'manifest.json';out=root/'successor'
        p=copy.deepcopy(policy());x={'preflight':[{'id':i} for i in p['qwen_partial_ids']]}
        for model,entries in p['original_attempts'].items():
            for e in entries:
                folder=old/'attempts'/model/e['attempt'];folder.mkdir(parents=True)
                publish_json(folder/'start.json',dict(phase=e['phase']))
                publish_json(folder/'end.json',dict(active_seconds=e['active_seconds'],exit_code=0 if e['phase']=='missing_raw' and model=='qwen' else 1))
                e['start_sha256']=sha(folder/'start.json');e['end_sha256']=sha(folder/'end.json')
        acceptance={'status':'accepted','identity':p['original_identities']['qwen']}
        publish_json(old/'missing_raw/qwen/acceptance.json',acceptance)
        (old/'preflight/qwen/shards/00000').mkdir(parents=True)
        publish_json(old/'preflight/qwen/shards/00000/commit.json',{'partial':True})
        p['original_acceptance_sha256']=sha(old/'missing_raw/qwen/acceptance.json')
        publish_json(manifest,{str(f.relative_to(old)):sha(f) for f in old.rglob('*') if f.is_file()})
        p['original_manifest_sha256']=sha(manifest)
        return old,manifest,out,p,x,acceptance

    def test_preservation_adoption_cumulative_charges_and_substitution_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            old,manifest,out,p,x,a=self.fixture(Path(d))
            before={str(f):f.read_bytes() for f in old.rglob('*') if f.is_file()}
            with patch('src.checkpoint_next_adoption.policy',return_value=p), \
                 patch('src.checkpoint_next_runner._missing_acceptance',return_value=a), \
                 patch('src.checkpoint_next_adoption.verify_shard') as loader, \
                 patch('src.checkpoint_next_runner.producer',return_value='b'*40):
                made=create(out,old,manifest,x)
                self.assertEqual(verify(out,old,manifest,x,execution=True),made)
                self.assertEqual(loader.call_args.args[2],p['original_identities']['qwen'])
                self.assertFalse(made['original']['partial_is_full_acceptance'])
                self.assertIsNone(made['original']['llama_acceptance'])
                self.assertEqual(made['original']['qwen_accepted_raw_shards'],131)
                with self.assertRaisesRegex(ValueError,'resume'):create(out,old,manifest,x)
                for name,seconds in [('failed',3.),('resumed',7.)]:
                    folder=out/'attempts/llama'/name;folder.mkdir(parents=True)
                    publish_json(folder/'start.json',{'phase':'missing_raw'})
                    publish_json(folder/'end.json',{'active_seconds':seconds,'exit_code':1 if name=='failed' else 0})
                costs=budget(out,'llama',made)
                self.assertEqual(costs['charged_seconds'],41.1570940479869+10.)
                self.assertEqual(costs['remaining_seconds'],7200-costs['charged_seconds'])
                self.assertEqual(budget(out,'qwen',made)['remaining_seconds'],7039.161989762972)
                self.assertEqual(before,{str(f):f.read_bytes() for f in old.rglob('*') if f.is_file()})
                path=out/'adoption.json'
                for field in ('producer','ledger','destination'):
                    bad=copy.deepcopy(made)
                    if field=='producer':bad['successor_producer']='c'*40
                    elif field=='ledger':bad['original']['carry_seconds']['llama']=0.
                    else:bad['execution_output']=str(out/'reset')
                    path.write_text(json.dumps(bad))
                    with self.assertRaisesRegex(ValueError,'substitution|relocated'):
                        verify(out,old,manifest,x,execution=True)
                path.write_text(json.dumps(made))
                # Corruption stops reuse; no copy, rewrite or new acceptance.
                target=old/'missing_raw/qwen/acceptance.json';target.write_text('{}')
                with self.assertRaisesRegex(ValueError,'artifact changed'):verify(out,old,manifest,x)

    def test_interrupted_successor_attempt_never_replenishes(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'attempts/llama/interrupted';f.mkdir(parents=True)
            publish_json(f/'start.json',{'phase':'missing_raw'})
            publish_json(f/'active.json',{'utc':100.})
            a={'original':{'carry_seconds':{'llama':41.1570940479869}}}
            with patch('src.checkpoint_next_runner.time.time',return_value=112.):
                first=budget(d,'llama',a)
            with patch('src.checkpoint_next_runner.time.time',return_value=500.):
                self.assertEqual(first,budget(d,'llama',a))
            self.assertEqual(first['successor_seconds'],12.)
            self.assertEqual(json.loads((f/'end.json').read_text())['status'],'interruption_conservative_charge')

    def test_acceptance_binds_cumulative_ledger_and_saved_receipt(self):
        from src.checkpoint_next_runner import missing_acceptance, verify as verify_output
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);publish_json(out/'adoption.json',{'synthetic':True})
            adoption={'original':{'carry_seconds':{'llama':41.1570940479869},
                                  'original_attempts':policy()['original_attempts']}}
            new_identity={'producer_commit':'b'*40}
            with patch('src.checkpoint_next_adoption.receipt',return_value=adoption), \
                 patch('src.checkpoint_next_runner.identity',return_value=new_identity), \
                 patch('src.checkpoint_next_runner.frozen',return_value={}), \
                 patch('src.checkpoint_next_runner._missing_acceptance',side_effect=lambda *a: {'verification':{'identity':a[2]}}):
                result=missing_acceptance(out,'llama',{})
                self.assertEqual(result['verification']['identity'],new_identity)
                self.assertEqual(result['campaign_budget']['charged_seconds'],41.1570940479869)
                path=out/'missing_raw/llama/acceptance.json';publish_json(path,result)
                self.assertEqual(verify_output(out,'missing_raw','llama'),result)
                path.write_text('{}')
                with self.assertRaisesRegex(ValueError,'saved supplement acceptance'):verify_output(out,'missing_raw','llama')
                folder=out/'attempts/llama/over-cap';folder.mkdir(parents=True)
                publish_json(folder/'start.json',{'phase':'missing_raw'})
                publish_json(folder/'end.json',{'active_seconds':7200})
                with self.assertRaisesRegex(ValueError,'cap exceeded'):missing_acceptance(out,'llama',{})

    def test_unknown_original_producer_and_manifest_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            old,manifest,out,p,x,a=self.fixture(Path(d))
            with patch('src.checkpoint_next_adoption.policy',return_value=p), \
                 patch('src.checkpoint_next_runner._missing_acceptance',return_value=a), \
                 patch('src.checkpoint_next_adoption.verify_shard'), \
                 patch('src.checkpoint_next_runner.producer',return_value='b'*40):
                made=create(out,old,manifest,x)
                made=copy.deepcopy(made)
                made['original']['original_identities']['qwen']['producer_commit']='c'*40
                (out/'adoption.json').write_text(json.dumps(made))
                with self.assertRaisesRegex(ValueError,'substitution'):receipt(out)
                manifest.write_text('{}')
                with self.assertRaisesRegex(ValueError,'manifest'):verify_original(old,manifest,x)


if __name__=='__main__':unittest.main()
