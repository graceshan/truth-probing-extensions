"""Full admitted atomic bindings. Preparation reads metadata; layer() is future CPU-only.

Extraction identity is always the original extraction producer, never the CPU producer.
"""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.checkpoint_next_inventory import ROOT, digest, sha, require, inventory
from src.checkpoint_r2_fresh_inputs import hash_value, text_hash
from src.checkpoint_r2_b25_adapter import B25Adapter, SEEDS
from src.checkpoint_next_missing import validate_queue, verify_supplement, provenance
from src.atomic_probe_methods import balanced_burger_indices

EXTRACTION_PRODUCER = '9b0ac1d84b35d7faf2148d43664569aa867cba52'
BASE = 'fc083594dc17d1fc38f7b52dd8030b0a7f2cb67d'
CONFIG = ROOT/'config/checkpoint_next/full_bank_prep_v1.json'
EVIDENCE = ROOT/'results/checkpoint_next_score_diagnosis_20261003'


def read(path):
    return json.loads(Path(path).read_text())


def configuration():
    cfg=read(CONFIG)
    for path,h in cfg['frozen_sha256'].items():
        require(sha(ROOT/path)==h, 'frozen full-bank dependency changed: '+path)
    require(cfg['candidates']==600 and cfg['repair_fits']==0 and cfg['processes']==cfg['blas_threads']==1
            and cfg['worker_cap_seconds']==10800 and cfg['E'] is False,'full-bank scope')
    require(cfg['supplement_producer']==EXTRACTION_PRODUCER,'extraction producer changed')
    return cfg


def ledger_position(value):
    # Reviewed pandas CSV positions are rendered as 0.0 where pending rows made
    # the column nullable. Accept only finite, exact, nonnegative integers.
    number=float(value)
    require(np.isfinite(number) and 0<=number<2**53 and number.is_integer(),'noninteger ledger position')
    return int(number)


def memberships():
    q=validate_queue();m=read(ROOT/'data/checkpoint_r2_v1/memberships.json')
    train,p15=m['outer_train'],m['P15'];pids={r['source_row_id'] for r in p15}
    require(len(train)==len({r['source_row_id'] for r in train})==3040 and len(p15)==2778,'admitted membership')
    require(all(r['split']=='train' and r['current_status']=='admitted' and
                r['statement_sha256']==text_hash(r['statement']) for r in train),'admitted text/labels')
    require([r for r in train if r['source_row_id'] in pids]==p15,'P15 subset/order')
    frame=pd.DataFrame(train);frame['label']=frame['label'].astype(int)
    frame['row_index']=frame['source_row_id'].str.rsplit(':',n=1).str[-1].astype(int)
    chosen=balanced_burger_indices(frame,seed=0).tolist()
    with (EVIDENCE/'full-training-balanced-1000.csv').open() as f: sampler=list(csv.DictReader(f))
    require(len(chosen)==1000 and chosen==[int(r['outer_training_position']) for r in sampler] and
            [train[i]['source_row_id'] for i in chosen]==[r['source_row_id'] for r in sampler],'exact full sampler')
    with (EVIDENCE/'full-training-coverage.csv').open() as f: coverage=list(csv.DictReader(f))
    require(len(coverage)==6080,'reviewed full coverage ledger')
    ledger={}
    for model in ('qwen','llama'):
        # Original ledger is interleaved by row then model; retain exact per-model source order.
        rows=[r for r in coverage if r['model']==model]
        require([r['source_row_id'] for r in rows]==[r['source_row_id'] for r in train],'ledger ordered identities')
        require(sum(r['status']=='exact_compatible_fresh_reuse' for r in rows)==2909 and
                sum(r['status']=='missing_unique_text' for r in rows)==131,'2909+131 full exposure')
        require(all(r['text_id']=='text_'+b['statement_sha256'] and r['source_metadata_sha256']==hash_value(b)
                    for r,b in zip(rows,train)),'ledger source identity')
        ledger[model]=rows
    return train,p15,chosen,ledger,q


def supplement_acceptance(campaign, transfer_manifest):
    """Independent downstream reopen of a transferred original-producer supplement.

    Caller must provide the external transfer manifest. No metadata-only verified flag,
    changed producer label, or current-HEAD identity is accepted.
    """
    from src.checkpoint_next_runner import verify_execution_evidence
    from src.checkpoint_r2_full_support import evidence
    from src.checkpoint_next_inventory import config as extraction_config
    campaign=Path(campaign).resolve();transfer=read(transfer_manifest)
    actual={str(p.relative_to(campaign)):sha(p) for p in sorted(campaign.rglob('*')) if p.is_file()}
    require(all(not p.is_symlink() for p in campaign.rglob('*')),'transfer symlinks forbidden')
    require(actual==transfer,'complete transfer file/hash mismatch')
    x=inventory();cfg=extraction_config();receipts={}
    for model in ('qwen','llama'):
        root=campaign/'missing_raw'/model;idn=read(root/'identity.json')
        expected=dict(producer_commit=EXTRACTION_PRODUCER,config_sha256=sha(ROOT/'config/checkpoint_next/behavior_prep_v1.json'),
            inventory_sha256=digest(x),model=model,integration_config_sha256=sha(ROOT/'config/checkpoint_next/integration_fix_v1.json'),
            paths='full-sequence-likelihood/greedy/last-input-chat-v2')
        require(idn==expected,'supplement extraction identity: CPU SHA must not relabel extraction')
        receipt=verify_supplement(campaign,model,idn)
        completion=read(root/'completion.json');require(completion['status']=='completed' and completion['verification']==receipt,'supplement completion')
        execution=verify_execution_evidence(campaign,model,completion)
        audit=read(root/'token-audit.json');backend=evidence(cfg,model+'/runtime-before-inference.json')['backend']
        require(audit['identity']==idn and audit['token_plan_sha256']==receipt['token_plan_sha256'] and
                audit['queue_sha256']==receipt['queue_sha256'] and audit['rows']==131 and
                audit['status']=='independently_reconstructed' and audit['model_loaded'] is False and
                all(audit[k]==backend[k] for k in ('tokenizer_backend_sha256','tokenizer_implementation_sha256')),'supplement token audit')
        receipt['token_audit_sha256']=sha(root/'token-audit.json')
        ledger=[]
        for p in sorted((campaign/'attempts'/model).glob('*/start.json')):
            if read(p)['phase']!='missing_raw':continue
            end=p.with_name('end.json');v=read(end)
            require(v['active_seconds']>=0,'extraction charge')
            ledger.append(dict(attempt=p.parent.name,start_sha256=sha(p),end_sha256=sha(end),**v))
        require(ledger,'real extraction attempt ledger')
        accepted=dict(verification=receipt,completion_sha256=sha(root/'completion.json'),ledger=ledger,
            active_seconds=sum(r['active_seconds'] for r in ledger),status='accepted')
        require(read(root/'acceptance.json')==accepted,'actual transferred acceptance required')
        receipts[model]=dict(acceptance_sha256=sha(root/'acceptance.json'),identity=idn,verification=receipt,
                             measured_execution=execution)
    return dict(status='transferred_independently_reopened',campaign=str(campaign),
                transfer_manifest_sha256=sha(transfer_manifest),extraction_producer=EXTRACTION_PRODUCER,models=receipts)


class FullTrainingAdapter(B25Adapter):
    def __init__(self, campaign=None, transfer_manifest=None):
        configuration()
        acceptance=read(ROOT/'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json')
        super().__init__(acceptance)
        self.manifest_hash=hash_value(self.manifest)
        train,p15,chosen,ledger,q=memberships()
        require(self.rows['P15']==p15,'original P15 unchanged')
        self.train=train;self.full_balanced=chosen;self.ledger=ledger;self.queue=q
        self.supplement=None
        if campaign is not None:
            require(transfer_manifest is not None,'transfer manifest required')
            self.supplement=supplement_acceptance(campaign,transfer_manifest)
        self.supplement_status='pending_actual_transferred_acceptance' if self.supplement is None else self.supplement['status']
        texts={r['id']:i for i,r in enumerate(self.manifest['rows'])}
        self.bindings['outer_train']=[dict(group='outer_train',logical_id=r['source_row_id'],text_id='text_'+r['statement_sha256'],
            text_row=texts.get('text_'+r['statement_sha256']),source_metadata_sha256=hash_value(r)) for r in train]
        self.rows['outer_train']=train
        self.receipt['group_rows']['outer_train']=3040
        self.receipt['original_P15_all_A_variants_excluded']=self.receipt.pop('all_A_variants_excluded')
        self.receipt['original_P15_balanced_binding_sha256']=self.receipt.pop('balanced_binding_sha256')
        d={r['person_key'] for r in self.rows['atomic_D']}|{r[k] for r in self.rows['D_bare'] for k in ('person_a','person_b')}
        require(not d & {r['person_key'] for r in train},'D/training entity overlap')
        require(all(set(self.allocations[s]['people']) <= {r['person_key'] for r in train} for s in SEEDS),'A restored exposure')
        self.receipt.update(full_train_rows=3040,existing_fresh_representations_per_model=2909,
            supplement_texts_per_model=131,supplement_status=self.supplement_status,
            outer_train_sha256=hash_value(train),P15_sha256=hash_value(p15),sampler_rows=1000,
            sampler_sha256=hash_value([train[i] for i in chosen]),
            full_binding_sha256=hash_value(self.bindings['outer_train']),
            A_identities_in_atomic_training=True,comparison='advantaged full-training atomic comparator versus frozen reduced-P repair; no full-training repair')

    def verify_existing_metadata(self):
        """Verify ledger locations against accepted fresh index/row/token receipts; no tensors."""
        for model,index in self.indices.items():
            lookup={r['text_id']:r for r in self.ledger[model] if r['status']=='exact_compatible_fresh_reuse'}
            offset=0;seen=set()
            for shard in index['shards']:
                folder=self.output/model/'shards'/shard['path']
                receipt=read(folder/'receipt.json')
                require(sha(folder/'receipt.json')==shard['receipt']['sha256'],'accepted shard receipt')
                require(receipt['model']==self.cfg['models'][model] and receipt['provenance']['commit']=='7656a0eb68a5fd7ff7dac1c73efeab982d540cc2','fresh extraction provenance')
                for name in ('rows.json','tokens.json'):
                    require(sha(folder/name)==receipt['files'][name]['sha256'],'source metadata artifact pin')
                rows=read(folder/'rows.json');tokens=read(folder/'tokens.json')
                require(rows==self.manifest['rows'][offset:offset+len(rows)] and len(tokens)==len(rows),'fresh row order')
                for local,(r,t) in enumerate(zip(rows,tokens)):
                    if r['id'] not in lookup:continue
                    c=lookup[r['id']]
                    require(c['shard']==shard['path'] and ledger_position(c['local_row'])==local and ledger_position(c['global_row'])==offset+local and
                        c['representation_receipt_sha256']==sha(folder/'receipt.json') and
                        c['activation_file_sha256']==receipt['files']['activations.npy']['sha256'] and
                        c['token_binding_sha256']==hash_value(t) and c['token_ids_sha256']==hash_value(t['token_ids']) and
                        t['id']==r['id'] and t['statement_sha256']==r['statement_sha256'],'ledger exact representation/text/token location')
                    require(t['token_ids']==t['unpadded_token_ids'] and t['position_ids']==list(range(len(t['token_ids']))) and
                        t['attention_mask']==[1]*len(t['token_ids']) and t['readout_index']==len(t['token_ids'])-1 and
                        t['add_special_tokens'] and not t['chat_template'] and not t['truncation'],'fresh raw token contract')
                    seen.add(r['id'])
                offset+=len(rows)
            require(seen==set(lookup),'all2909 fresh metadata accounted')
        return dict(status='existing_fresh_metadata_verified',tensor_values_loaded=False,large_tensor_bytes_rehashed=False,
                    original_full_acceptance_sha256=sha(ROOT/'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json'),
                    supplement_status=self.supplement_status)

    def layer(self,model,layer,wording=False):
        require(self.supplement is not None,'131 texts/model pending actual transferred acceptance; fitting blocked')
        groups=['P15','atomic_D','D_bare']+[f'B25_{s}' for s in SEEDS]
        if wording:groups+=sorted(self.receipt['heldout_wording_rows'])
        original=self.bindings
        try:
            self.bindings={g:original[g] for g in groups}
            # Direct parent implementation avoids including unused TC matrices.
            from src.checkpoint_r2_fresh_adapter import FreshAdapter
            result=FreshAdapter.layer(self,model,layer)
            existing=[dict(b) for b in original['outer_train'] if b['text_row'] is not None]
            self.bindings={'outer_train_existing':existing}
            full=np.empty((3040,self.cfg['models'][model]['width']),dtype=np.float64)
            X=FreshAdapter.layer(self,model,layer)['outer_train_existing']
            dst=[i for i,b in enumerate(original['outer_train']) if b['text_row'] is not None];full[dst]=X
        finally:self.bindings=original
        from src.checkpoint_r2_fresh_store import load_features
        from src.checkpoint_next_missing import raw_rows
        queue_rows=raw_rows();indices={b['text_id']:i for i,b in enumerate(original['outer_train'])}
        campaign=Path(self.supplement['campaign']);idn=self.supplement['models'][model]['identity']
        for i,row in enumerate(queue_rows):
            folder=campaign/'missing_raw'/model/'shards'/f'{i:05d}'
            values,_,_=load_features(folder,expected_rows=[row],expected_spec=self.cfg['models'][model],expected_provenance=provenance(model,idn))
            full[indices[row['id']]]=values[0,layer,:]
        require(np.isfinite(full).all(),'full training coverage/finite features')
        result['outer_train']=full
        return result
