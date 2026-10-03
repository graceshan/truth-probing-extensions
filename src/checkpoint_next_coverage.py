"""Metadata-only full-training coverage: no tensor loading, preprocessing or fit."""
from collections import Counter
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from src.atomic_probe_methods import balanced_burger_indices, BURGER_TOPICS
from src.checkpoint_r2_fresh_inputs import ROOT, hash_value, text_hash, require
from src.checkpoint_next_inputs import read


def coverage(inputs, out):
    members = read(ROOT / 'data/checkpoint_r2_v1/memberships.json')
    train, p15 = members['outer_train'], members['P15']
    pids = {r['source_row_id'] for r in p15}; additional = [r for r in train if r['source_row_id'] not in pids]
    require((len(train), len(p15), len(additional)) == (3040, 2778, 262), 'full-training membership counts')
    require(len({r['source_row_id'] for r in train}) == 3040 and all(r['split'] == 'train' and r['current_status'] == 'admitted' for r in train), 'exact admitted training identities')
    require([r for r in train if r['source_row_id'] in pids] == p15, 'P15 exact subset/order')
    facts = inputs.b.facts
    texts = {r['id']: r for r in inputs.manifest['rows']}
    locations = {}
    shards_checked = 0
    # Rehash accepted fresh representation files as opaque bytes. No np.load of activations.
    # Every row/token/representation receipt is checked against the accepted index.
    for model, index in inputs.b.indices.items():
        base = inputs.b.output / model
        inputs.verify(base / 'index.json', dict(bytes=(base / 'index.json').stat().st_size, sha256=inputs.acceptance['models'][model]['index_sha256']))
        offset = 0
        for shard in index['shards']:
            folder = base / 'shards' / shard['path']
            inputs.verify(folder / 'receipt.json', shard['receipt'])
            receipt = read(folder / 'receipt.json')
            require(receipt['model'] == inputs.fresh_cfg['models'][model], 'fresh model snapshot/representation identity')
            prov = receipt['provenance']
            require(prov['commit'] == inputs.acceptance['reviewed_producer_commit'] and prov['contract_sha256'] == inputs.cfg['input_pins']['config/checkpoint_r2/full_raw_v1.json']['sha256'] and prov['manifest_sha256'] == inputs.acceptance['manifest_sha256'] and prov['model'] == model and prov['runtime_receipt_sha256'] == inputs.acceptance['models'][model]['runtime_origin_sha256'] and prov['batch_size'] == 1 and prov['padding'] == 'none', 'accepted fresh numerical provenance')
            for name, expected in receipt['files'].items(): inputs.verify(folder / name, expected)
            rows, tokens = read(folder / 'rows.json'), read(folder / 'tokens.json')
            require(rows == inputs.manifest['rows'][offset:offset + len(rows)] and hash_value(rows) == receipt['ordered_rows_sha256'], 'physical row/text/identity binding')
            require([r['id'] for r in rows] == receipt['ordered_ids'] == shard['ordered_ids'], 'shard row order')
            require(receipt['shape'] == [len(rows), receipt['model']['layers'], receipt['model']['width']] and receipt['dtype'] == 'float16' and receipt['saved_layers'] == list(range(receipt['model']['layers'])) and receipt['hf_hidden_state_indices'] == list(range(1, receipt['model']['layers'] + 1)) and receipt['final_layer'] == 'post-final-RMSNorm' and not receipt['embedding_saved'], 'saved layer axes')
            require(len(tokens) == len(rows) and hash_value(tokens) == shard['tokens_sha256'], 'token receipt binding')
            for local, (r, t) in enumerate(zip(rows, tokens)):
                require(t['id'] == r['id'] and t['statement_sha256'] == r['statement_sha256'] == text_hash(r['statement']), 'token/text identity')
                require(t['token_ids'] == t['unpadded_token_ids'] and all(t['attention_mask']) and t['position_ids'] == list(range(len(t['token_ids']))) and t['readout_index'] == len(t['token_ids']) - 1 and t['semantic_readout_position'] == t['readout_index'] and t['add_special_tokens'] and not t['chat_template'] and not t['truncation'], 'unpadded last-input token convention')
                locations[model, r['id']] = dict(shard=shard['path'], local_row=local, global_row=offset + local,
                    token_binding_sha256=hash_value(t), token_ids_sha256=hash_value(t['token_ids']),
                    activation_file_sha256=receipt['files']['activations.npy']['sha256'],
                    representation_receipt_sha256=inputs.used[str((folder / 'receipt.json').resolve())]['sha256'])
            offset += len(rows); shards_checked += 1
            print('verified fresh metadata and opaque hashes', model, shard['path'], flush=True)
        require(offset == inputs.manifest['unique_texts'], 'accepted inventory coverage')
    rows_out, missing = [], {}
    for row in train:
        tid = 'text_' + row['statement_sha256']
        require(row['statement_sha256'] == text_hash(row['statement']), 'training statement hash')
        item = texts.get(tid); matches = []
        if item:
            require(item['statement'] == row['statement'] and item['topic'] == row['topic'] and item['split'] == row['split'], 'compatible physical text contract')
            for b in item['bindings']:
                if b['group'] == 'P15':
                    source = next(r for r in p15 if r['source_row_id'] == b['logical_id'])
                    compatible = all(source[k] == row[k] for k in ('person_key', 'entity_id', 'statement_sha256', 'label', 'topic', 'split'))
                elif b['group'] == 'TC_isolated':
                    source = facts[b['logical_id']]
                    compatible = source['eligible'] and source['person_key'] == row['person_key'] and source['statement_sha256'] == row['statement_sha256'] and row['entity_id'] in source['entity_ids'] and int(source['truth']) == int(row['label']) and source['topic'] == row['topic'] and source['split'] == row['split']
                else:
                    continue
                require(hash_value(source) == b['source_metadata_sha256'], 'reuse reviewed source metadata')
                if compatible: matches.append((b['group'], b['logical_id'], b['source_metadata_sha256']))
            require(matches, 'existing text lacks compatible reviewed identity binding')
        else:
            missing.setdefault(tid, dict(text_id=tid, statement=row['statement'], statement_sha256=row['statement_sha256'], topic=row['topic'], split='train', bindings=[]))['bindings'].append(row)
        for model in inputs.cfg['models']:
            loc = locations.get((model, tid))
            require(bool(loc) == bool(matches), 'per-model representation coverage mismatch')
            rows_out.append(dict(model=model, source_row_id=row['source_row_id'], source_metadata_sha256=hash_value(row),
                person_key=row['person_key'], topic=row['topic'], form=row['form'], label=int(row['label']), text_id=tid,
                additional=row['source_row_id'] not in pids, status='exact_compatible_fresh_reuse' if matches else 'missing_unique_text',
                compatible_bindings=json_string(matches), **(loc or {})))
    pd.DataFrame(rows_out).to_csv(out / 'full-training-coverage.csv', index=False)
    pd.DataFrame(additional).to_csv(out / 'additional-262-bindings.csv', index=False)
    frame = pd.DataFrame(train)
    frame['label'] = frame['label'].astype(int)
    frame['row_index'] = frame['source_row_id'].str.rsplit(':', n=1).str[-1].astype(int)
    chosen = balanced_burger_indices(frame, seed=0)
    balanced = [train[int(i)] for i in chosen]
    require(len(balanced) == 1000, 'expected full t_G/TTPD exposure')
    # Compare to the already reviewed full-training membership, without using its tensors.
    projection = ROOT / 'data/clean_protocol/selection_repair_v1/final_partition_projection_v1/train_balanced_exposure.csv'
    inputs.verify(projection, inputs.cfg['input_pins'][str(projection.relative_to(ROOT))])
    with projection.open() as f: expected = list(csv.DictReader(f))
    require([r['source_row_id'] for r in balanced] == [r['source_row_id'] for r in expected], 'full sampler exact ordered identities')
    pd.DataFrame([dict(sampler_position=i, outer_training_position=int(chosen[i]), **r) for i, r in enumerate(balanced)]).to_csv(out / 'full-training-balanced-1000.csv', index=False)
    queue = dict(schema='checkpoint-next-missing-raw-inputs-v1', source_document_sha256=inputs.cfg['document_pin']['sha256'],
        outer_train_membership_sha256=hash_value(train), accepted_manifest_sha256=inputs.acceptance['manifest_sha256'],
        extraction_contract_sha256=inputs.cfg['input_pins']['config/checkpoint_r2/full_raw_v1.json']['sha256'],
        model_contracts=inputs.fresh_cfg['models'], numerical_contract=inputs.fresh_cfg['execution'],
        rows=list(missing.values()), unique_texts_per_model=len(missing), model_input_combinations=2 * len(missing),
        missing_token_ids='not fabricated; mac 1 must bind native pinned tokenizer IDs before any separately authorized inference',
        extraction_authorized_by_this_task=False,
        completion_requirements='Exact text/person/source bindings, per-model token sequences, accepted raw BF16/FP16 contract, ordered all-layer hashes, quota/runtime/transfer receipts; no historical tensor substitution.')
    (out / 'missing-input-queue.json').write_text(json_string(queue, indent=2) + '\n')
    summary = dict(schema='checkpoint-next-full-training-coverage-v1', metadata_only=True, feature_tensors_loaded=0,
        opaque_fresh_feature_files_rehashed=shards_checked, train_bindings=3040, P15_bindings=2778, additional_bindings=262,
        additional_unique_texts=len({r['statement_sha256'] for r in additional}),
        missing_unique_texts_per_model=len(missing), missing_model_input_combinations=2 * len(missing),
        missing_bindings=sum(len(r['bindings']) for r in missing.values()),
        additional_reused_bindings=262 - sum(len(r['bindings']) for r in missing.values()),
        full_unique_texts=len({r['statement_sha256'] for r in train}),
        outer_training_ordered_membership_sha256=hash_value(train), P15_ordered_membership_sha256=hash_value(p15),
        additional_ordered_membership_sha256=hash_value(additional), sampler_ordered_membership_sha256=hash_value(balanced),
        sampler_ordered_source_ids_sha256=hash_value([r['source_row_id'] for r in balanced]), sampler_rows=1000,
        sampler='RandomState(0), original BURGER_TOPICS order; same sampled row indices for affirmative/negated pairs; equal minimum affirmative dataset count',
        sampler_topic_order=list(BURGER_TOPICS), sampler_topic_counts=dict(Counter(r['topic'] for r in balanced)),
        sampler_dataset_counts=dict(Counter(r['source_row_id'].rsplit(':', 1)[0] for r in balanced)),
        preprocessing='Later standardized procedures use full admitted 3040-row population mean/std, constant mask and denominator=1 for constant features. Raw LR keeps existing C scaling; DoM/covariance-MM/t_G/TTPD keep their existing raw objectives. t_G/TTPD use this exact 1000-row balanced exposure; no preprocessing or fitting executed here.',
        A_exposure='Full training restores all admitted A15 person variants. Selection A identities overlap atomic fitting; D remains entity-disjoint.',
        later_required_contrasts=inputs.cfg['later_required_contrasts'],
        later_required_reports=inputs.cfg['later_required_reports'],
        full_bank_fit_authorized=False, historical_tensors_used=0)
    from src.checkpoint_r2_fresh_store import pin
    summary['missing_queue_pin'] = pin(out / 'missing-input-queue.json')
    (out / 'coverage-summary.json').write_text(json_string(summary, indent=2) + '\n')
    return summary


def json_string(value, indent=None):
    import json
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=indent)
