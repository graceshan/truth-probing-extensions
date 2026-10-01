"""Metadata/physical cache preflight, independent of pending correction decisions."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import shlex
import subprocess

from src import clean_atomic_extraction as atomic
from src import clean_transfer_contracts as c
from src import llama_replication_contracts as llama
from src.data import ENTITY_PATTERNS
from src.entity_partitions import entity_id
from src.t1_score_package import INVENTORY_SHA256, read_json, safe_file

BASE = 'dae31152eb35f91a28130acaef1c9630a681ef3f'
REMOTE_ROOT = '/workspace/truth-probing-artifacts/'
QWEN = 'atomic_repair/qwen25_a09a354_bs1_bf16_v1'
LLAMA = 'llama31_replication_v1/atomic'
FIELDS = list(atomic.FIELDS)
SCOPE = 'Preflight only; original source-row correspondence, independent of pending factual correction decisions'


def record(path):
    return c.record(path)


def csv_rows(path):
    with path.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames, list(reader)


def typed_rows(rows, split=None, llama_ids=False):
    result=[]; seen=set()
    for raw in rows:
        c.require(set(raw) == set(FIELDS+(['example_id'] if llama_ids else [])), 'sidecar schema mismatch')
        c.require(raw['row_index'].isdigit() and raw['label'] in ['0','1'], 'invalid original row index/label')
        row = {k:raw[k] for k in FIELDS}
        row['row_index'],row['label'] = int(row['row_index']),int(row['label'])
        c.require(row['split'] in atomic.COUNTS and (split is None or row['split'] == split), 'wrong/forbidden partition')
        c.require(row['topic'] in atomic.TOPICS and row['form'] in ['affirmative','negated'], 'topic/form mismatch')
        c.require(row['dataset'] == ('neg_' if row['form'] == 'negated' else '')+row['topic'], 'dataset identity mismatch')
        identity = row['dataset']+':'+str(row['row_index'])
        c.require(identity not in seen and row['statement'].strip() and row['entity_id'], 'duplicate/empty source identity')
        if llama_ids: c.require(raw['example_id'] == identity, 'Llama example/source ID mismatch')
        seen.add(identity); result.append(row)
    return result


def verify_correspondence(rows, originals, partition_rows):
    """Do not join on statement text or apply alias/admission decisions."""
    correspondences=[]
    for cache_row,row in enumerate(rows):
        key = (row['dataset'],row['row_index'])
        c.require(key in originals, 'source row missing')
        expected = originals[key]
        c.require(row == expected['row'], 'source statement/label/entity/partition differs at '+str(key))
        c.require(hashlib.sha256(row['statement'].encode('utf-8')).hexdigest() == expected['statement_sha256'], 'statement hash differs')
        correspondences.append(dict(cache_row=cache_row,source_row_id=f'{key[0]}:{key[1]}',
            source_sha256=expected['source_sha256'],statement_sha256=expected['statement_sha256'],original_split=row['split']))
    c.require({(r['dataset'],r['row_index']) for r in rows} == partition_rows, 'original partition coverage mismatch')
    duplicates=Counter(row['statement'] for row in rows)
    return dict(rows=len(rows),unique_source_row_ids=len(correspondences),
        duplicate_statement_groups=sum(n>1 for n in duplicates.values()),
        duplicate_statement_rows=sum(n for n in duplicates.values() if n>1),
        ordered_source_row_ids_sha256=c.ordered_hash(x['source_row_id'] for x in correspondences),
        ordered_statement_sha256=c.ordered_hash(r['statement'] for r in rows),
        preliminary_correspondence_sha256=c.digest(c.canonical(correspondences)),
        correspondence_fields=list(correspondences[0]) if correspondences else [],
        final_selection_map_created=False,admitted_rows_or_person_keys_used=False)


class Inputs:
    def __init__(self, repo, payload):
        self.repo,self.payload=repo,payload
        self.receipts=[];self.sources={};self.paths={}
        path=safe_file(payload,'inventory.json')
        c.require(c.file_hash(path)==INVENTORY_SHA256,'restored inventory identity mismatch')
        self.receipts.append(dict(location='inventory.json',**record(path)))
        for item in read_json(path):
            c.require(item['original_path'] not in self.sources,'duplicate inventory original path')
            self.sources[item['original_path']]=item

    def package(self, relative):
        source=REMOTE_ROOT+relative
        c.require(source in self.sources,'missing packaged evidence: '+relative)
        item=self.sources[source];path=safe_file(self.payload,item['archive_path'])
        actual=record(path)
        c.require(actual=={k:item[k] for k in ['bytes','sha256']},'packaged evidence changed: '+relative)
        self.receipts.append(dict(location=item['archive_path'],original_path=source,**actual));self.paths[str(path)]=actual
        return path

    def checkout(self, relative):
        path=safe_file(self.repo,relative);actual=record(path)
        self.receipts.append(dict(location=relative,origin='checkout',**actual));self.paths[str(path)]=actual
        return path

    def json(self, relative): return read_json(self.package(relative))

    def packaged_report(self, relative):
        matches=[x for x in self.sources.values() if x['archive_path']==relative]
        c.require(len(matches)==1,'packaged report missing/ambiguous')
        item=matches[0];path=safe_file(self.payload,relative);actual=record(path)
        c.require(actual=={k:item[k] for k in ['bytes','sha256']},'packaged report identity mismatch')
        self.receipts.append(dict(location=relative,original_path=item['original_path'],**actual));self.paths[str(path)]=actual
        return read_json(path)

    def unchanged(self):
        c.require(c.file_hash(self.payload/'inventory.json')==INVENTORY_SHA256,'inventory changed')
        for path,identity in self.paths.items(): c.require(record(Path(path))==identity,'input changed during preflight')


def original_sources(inputs):
    # These paths and partition assignments are historical; no correction config is loaded.
    meta=read_json(inputs.checkout('data/clean_protocol/entity_partitions/metadata.json'))
    path=inputs.checkout('data/clean_protocol/entity_partitions/manifest.csv')
    c.require(c.file_hash(path)==meta['manifest_sha256'],'original partition manifest hash mismatch')
    _,members=csv_rows(path); lookup={(r['topic'],r['entity']):r for r in members}
    c.require(len(lookup)==len(members),'duplicate original partition identity')
    for row in members:
        c.require(row['entity_id']==entity_id(row['topic'],row['entity']) and row['split'] in ['train','validation','test'],'invalid original entity assignment')
    hashes={r['file']:r for r in meta['sources']};originals={};by_split={s:set() for s in atomic.COUNTS}
    for topic in atomic.TOPICS:
        for form,dataset in [('affirmative',topic),('negated','neg_'+topic)]:
            path=inputs.checkout('data/tiu_datasets/'+dataset+'.csv'); _,rows=csv_rows(path)
            sha=c.file_hash(path)
            c.require(sha==hashes[path.name]['sha256'] and len(rows)==hashes[path.name]['rows'],'original source identity mismatch')
            for index,source in enumerate(rows):
                match=re.search(ENTITY_PATTERNS[topic],source['statement'])
                c.require(match is not None,'unrecognized original entity')
                member=lookup[topic,match[1]]
                if member['split'] not in atomic.COUNTS: continue
                c.require(source['label'] in ['0','1'],'invalid original label')
                row=dict(dataset=dataset,row_index=index,statement=source['statement'],entity_id=member['entity_id'],
                         topic=topic,form=form,split=member['split'],label=int(source['label']))
                key=dataset,index;by_split[row['split']].add(key)
                originals[key]=dict(row=row,source_sha256=sha,statement_sha256=c.digest(source['statement'].encode('utf-8')))
    return originals,by_split,meta['manifest_sha256']


def local_preflight(inputs):
    originals,partition_rows,partition_hash=original_sources(inputs)
    allowed_path=inputs.checkout(str(atomic.INPUT)); columns,raw=csv_rows(allowed_path)
    c.require(columns==FIELDS,'historical export schema mismatch')
    allowed=typed_rows(raw)
    qmanifest=inputs.json(QWEN+'/extraction_manifest.json');completion=inputs.json(QWEN+'/completion.json')
    binding=qmanifest['input']
    c.require(c.file_hash(allowed_path)==binding['file_sha256'] and c.digest(atomic.canonical(allowed))==binding['allowed_rows_sha256']==atomic.INPUT_DIGEST,'allowed export byte/order identity mismatch')
    c.require(binding['columns']==FIELDS and binding['counts']==atomic.COUNTS,'historical input contract differs')
    c.require(completion['complete'] is True and completion['counts']==atomic.COUNTS and
              completion['manifest_sha256']==c.file_hash(inputs.package(QWEN+'/extraction_manifest.json')),'Qwen completion binding mismatch')
    c.require(qmanifest['contract']==atomic.contract(),'Qwen extraction representation differs')
    resolved=qmanifest['resolved_model'];qc=resolved['model_config']
    c.require(resolved['model_resolved_revision']==resolved['tokenizer_resolved_revision']==atomic.REVISION,'Qwen revision mismatch')
    c.require((qc['model_type'],qc['num_hidden_layers'],qc['hidden_size'])==('qwen2',28,3584),'Qwen architecture mismatch')
    c.require(c.digest(atomic.canonical(qc))==resolved['model_config_sha256'],'Qwen embedded configuration hash mismatch')
    c.require(qmanifest['runtime']['torch'].split('+')[0]=='2.11.0' and qmanifest['runtime']['transformers']=='5.12.1','Qwen recorded runtime mismatch')
    c.require(qmanifest['smoke']['passed'] is True and qmanifest['smoke']['batch_size']==1 and qmanifest['smoke']['padding'] is False,'Qwen recorded smoke missing')
    qspec=read_json(inputs.checkout('config/clean_protocol/pinned_qwen25_lr_transfer_v1.json'))
    lspec=read_json(inputs.checkout('config/clean_protocol/llama31_transfer_v1.json'))
    pin=llama.validate_pin(read_json(inputs.checkout('config/clean_protocol/llama31_representation_v1.json')))
    c.require(lspec['pin']==pin,'Llama spec representation differs')
    qrepresentation=qspec['representation']['descriptor']
    c.require(c.digest(c.canonical(qrepresentation))==qspec['representation']['fingerprint'],'Qwen representation fingerprint mismatch')
    for key in ['model','model_revision','tokenizer_revision','compute_dtype','saved_dtype','readout','batch_size','padding','input','chat_template','add_special_tokens','truncation','use_cache','attention','explicit_semantic_position_ids']:
        c.require(qrepresentation[key]==qmanifest['contract'][key],'Qwen spec/extraction representation mismatch: '+key)
    c.require(qrepresentation['num_hidden_layers']==28 and qrepresentation['hidden_size']==3584 and
              qrepresentation['layer_convention']['saved_layer_to_hf_index']==list(range(1,29)),'Qwen saved layer mismatch')
    caches={};requests={};all_rows={};warnings=[]
    for model,prefix in [('qwen',QWEN),('llama',LLAMA)]:
        for split in atomic.COUNTS:
            cache=model+'/'+split;directory=prefix+'/'+split
            path=inputs.package(directory+'/metadata.csv'); columns,raw=csv_rows(path)
            c.require(columns==(['example_id'] if model=='llama' else [])+FIELDS,'cache metadata column order')
            rows=typed_rows(raw,split,llama_ids=model=='llama');all_rows[cache]=rows
            expected=[r for r in allowed if r['split']==split]
            c.require(rows==expected,'cache row order differs from bound original export: '+cache)
            correspondence=verify_correspondence(rows,originals,partition_rows[split])
            c.require(c.digest(atomic.canonical(rows))==binding['partition_rows_sha256'][split],'partition ordered record hash mismatch')
            companions=[directory+'/metadata.csv']
            if model=='qwen':
                tensor=completion['outputs'][split+'/activations.npy']
                c.require(record(path)==completion['outputs'][split+'/metadata.csv'],'Qwen completion/sidecar mismatch')
                for name in ['metadata.csv','activations.npy']:
                    c.require(completion['outputs'][split+'/'+name]==qspec['inputs']['atomic']['files'][split+'/'+name],'Qwen recorded file identities conflict')
                shape=qmanifest['activation_shapes'][split]
                c.require(shape==[len(rows),28,3584],'Qwen manifest shape mismatch')
                companions += [prefix+'/completion.json',prefix+'/extraction_manifest.json']
                representation=qmanifest['contract'];fingerprint=qspec['representation']['fingerprint']
                tokenizer={k:resolved[k] for k in ['tokenizer_resolved_revision','tokenizer_class','tokenizer_backend_sha256','tokenizer_config_file_sha256']}
            else:
                manifest=inputs.json(directory+'/extraction_manifest.json');progress=inputs.json(directory+'/progress.json')
                c.require(manifest['pin']==pin and manifest['input_binding']==binding and manifest['stage']==split,'Llama source/representation binding mismatch')
                shape=manifest['data']['expected_activation_shape']
                c.require(shape==[len(rows),32,4096] and manifest['data']['metadata_columns']==columns,'Llama shape/schema mismatch')
                c.require(manifest['data']['sidecar_sha256']==manifest['data']['benchmark_sha256']==c.file_hash(path),'Llama sidecar digest mismatch')
                ids=[r['dataset']+':'+str(r['row_index']) for r in rows];statements=[r['statement'] for r in rows]
                c.require(manifest['data']['ordered_example_id_sha256']==c.ordered_hash(ids) and
                          manifest['data']['ordered_statement_sha256']==c.ordered_hash(statements),'Llama ordered identity digest mismatch')
                tensor=lspec['inputs']['atomic']['cache_files'][split]['activations.npy']
                identity=dict(manifest);identity.pop('smoke_test');identity['code']={k:manifest['code'][k] for k in ['implementation_version','source_sha256']}
                c.require(progress['complete'] is True and progress['next_row']==len(rows) and
                          progress['activation_file_sha256']==tensor['sha256'] and progress['identity_sha256']==c.digest(c.canonical(identity)),'Llama finalized tensor/manifest binding mismatch')
                cursor=0
                for chunk in progress['chunks']:
                    stop=chunk['stop']
                    c.require(chunk['start']==cursor and cursor<stop<=len(rows) and
                              chunk['ordered_example_id_sha256']==c.ordered_hash(ids[cursor:stop]) and
                              chunk['ordered_statement_sha256']==c.ordered_hash(statements[cursor:stop]),'Llama chunk row coverage mismatch')
                    cursor=stop
                c.require(cursor==len(rows),'incomplete Llama row coverage')
                smoke=manifest['smoke_test']
                c.require(smoke['passed'] is True and smoke['policy']=='llama-unpadded-repeat-native-position-exact-v1' and
                          smoke['batch_size']==1 and smoke['padding'] is False and smoke['statements_sha256']==c.ordered_hash([statements[0],statements[-1]]),'Llama recorded smoke binding mismatch')
                c.require(manifest['runtime']['torch'].split('+')[0]=='2.11.0' and manifest['runtime']['transformers']=='5.12.1','Llama runtime record mismatch')
                for name in ['metadata.csv','extraction_manifest.json','progress.json']:
                    c.require(record(inputs.package(directory+'/'+name))==lspec['inputs']['atomic']['cache_files'][split][name],'Llama frozen cache file mismatch')
                companions += [directory+'/extraction_manifest.json',directory+'/progress.json']
                representation=pin['representation'];fingerprint=pin['fingerprint']
                tokenizer={k:pin[k] for k in ['tokenizer_class','tokenizer_backend_sha256','special_tokens','files']}
            remote_path=REMOTE_ROOT+directory+'/activations.npy'
            requests[cache]=dict(path=remote_path,**tensor,shape=shape,
                companions={REMOTE_ROOT+name:record(inputs.package(name)) for name in companions})
            caches[cache]=dict(local_binding_status='verified',**correspondence,original_partition=split,
                representation=representation,representation_fingerprint=fingerprint,tokenizer_recorded_identity=tokenizer,
                expected_tensor=dict(path=remote_path,**tensor,shape=shape,dtype='<f2',fortran_order=False),
                sidecar=record(path),physical_status='pending',corrected_fit_authorized=False)
    c.require(all_rows['qwen/train']==all_rows['llama/train'] and all_rows['qwen/validation']==all_rows['llama/validation'],'cross-model row correspondence differs')
    c.require({r['entity_id'] for r in all_rows['qwen/train']}.isdisjoint(r['entity_id'] for r in all_rows['qwen/validation']),'original train/validation entity overlap')
    # Verify the recorded per-statement smoke hashes without re-running inference.
    lookup={(r['dataset'],r['row_index']):r for r in allowed}
    smoke_rows=[]
    for row in qmanifest['smoke']['comparisons']:
        key=row['dataset'],row['row_index']
        c.require(key in lookup and lookup[key]['split']==row['split'] and c.digest(lookup[key]['statement'].encode())==row['statement_sha256'],'Qwen smoke sample statement binding differs')
        smoke_rows.append(lookup[key])
    c.require(c.digest(atomic.canonical(smoke_rows))==qmanifest['smoke']['sample_rows_sha256'],'Qwen recorded smoke row-order digest differs')
    excluded={x['original_path']:x for x in inputs.packaged_report('reports/excluded_activations.json')}
    for request in requests.values():
        c.require(request['path'] in excluded and request['bytes']==excluded[request['path']]['bytes'],'tensor path/size differs from excluded inventory')
    alternate=[]
    for suite in ['atomic_method_suite_pinned_v1','atomic_method_matched_pinned_v1']:
        path=inputs.package('pinned_atomic_methods_qwen25_v1/'+suite+'/qwen25_7b/allowed_atomic_rows.csv')
        _,rows=csv_rows(path);rows=typed_rows(rows)
        key=lambda row:(row['dataset'],row['row_index'])
        same={key(r):r for r in rows}=={key(r):r for r in allowed}
        c.require(same,'alternate allowed export records disagree')
        alternate.append(dict(suite=suite,**record(path),records_match_by_source_id=same,
            exact_order_matches=rows==allowed,positions_differing=sum(a!=b for a,b in zip(rows,allowed)),
            use_as_tensor_row_order=False))
    producer=inputs.json('pinned_transfer_qwen25_v1/scoring/representation_binding.json')['producer_source_audit']['files']['src/pinned_atomic_probes.py']
    source_differences=[]
    for model,manifest in [('qwen',qmanifest),('llama',inputs.json(LLAMA+'/train/extraction_manifest.json'))]:
        hashes=manifest['source_sha256'] if model=='qwen' else manifest['code']['source_sha256']
        for name,sha in hashes.items():
            current=c.file_hash(inputs.checkout(name))
            if sha!=current:source_differences.append(dict(model=model,path=name,recorded_sha256=sha,checkout_sha256=current))
    return dict(caches=caches,remote_requests=requests,original_partition_manifest_sha256=partition_hash,
        bound_original_export=record(allowed_path),alternate_exports=alternate,
        historical_producer_conflict=dict(path='src/pinned_atomic_probes.py',**producer,status='unresolved',
            limitation='Tensor byte equality does not identify the historical implementation that executed.'),
        extraction_source_hash_differences=source_differences)


def remote_preflight(requests, script_path, identity_file, timeout=900):
    # The remote program reads only allowlisted cache files; no remote installation or output files.
    program=script_path.read_text()
    command=['ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
             '-o','ConnectTimeout=15','-p','22114','-i',str(identity_file.expanduser()),'root@69.30.85.22',
             'python3 -B -c '+shlex.quote(program)]
    try:
        process=subprocess.run(command,input=json.dumps(requests),text=True,capture_output=True,timeout=timeout)
    except (OSError,subprocess.TimeoutExpired) as exc:
        return dict(status='pending',reason=type(exc).__name__,attempted=True,cache_results={})
    if process.returncode:
        return dict(status='pending',reason=process.stderr.strip()[:1500],exit_code=process.returncode,attempted=True,cache_results={})
    try:
        results=json.loads(process.stdout)
        c.require(set(results)==set(requests),'incomplete remote response')
        for cache,result in results.items():
            if result['status']=='verified':
                actual=result['tensor'];expected=requests[cache]
                c.require(all(actual[k]==expected[k] for k in ['path','bytes','sha256']) and
                          actual['header']['shape']==expected['shape'] and actual['header']['dtype']=='<f2' and
                          actual['header']['fortran_order'] is False and actual['stable_during_read'] is True,'remote measurement conflict')
    except (ValueError,KeyError,TypeError):
        return dict(status='pending',reason='invalid remote measurement response',attempted=True,cache_results={})
    return dict(status='verified' if all(x['status']=='verified' for x in results.values()) else 'incomplete_or_conflict',
                attempted=True,exit_code=0,cache_results=results)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--payload-root',type=Path,required=True)
    parser.add_argument('--output-root',type=Path,required=True,help='New external report directory')
    parser.add_argument('--ssh',action='store_true',help='Attempt read-only physical verification on the specified Runpod host')
    parser.add_argument('--identity-file',type=Path,default=Path('~/.ssh/id_ed25519'))
    args=parser.parse_args();repo=Path(__file__).resolve().parents[1];payload=args.payload_root.resolve();output=args.output_root.resolve()
    c.require(not output.exists() and not output.is_relative_to(repo) and not output.is_relative_to(payload.parent),'report output must be new and outside checkout/backup')
    output.mkdir(parents=True,exist_ok=False)
    inputs=Inputs(repo,payload)
    try:
        local=local_preflight(inputs)
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps(dict(status='local_checks_failed',error=str(exc)),indent=2)+'\n')
        raise
    remote=(remote_preflight(local['remote_requests'],repo/'src/t2_cache_remote.py',args.identity_file) if args.ssh else
            dict(status='pending',attempted=False,reason='SSH not requested',cache_results={}))
    for cache,result in local['caches'].items():
        result['physical_status']=remote['cache_results'].get(cache,{}).get('status','pending')
        result['compatibility']='conditional_reuse' if result['physical_status']=='verified' else 'not_yet_physically_verified'
    inputs.unchanged()
    unique_receipts={x['location']:x for x in inputs.receipts}
    report=dict(scope=SCOPE,base_commit=BASE,checkout_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
        host_platform=dict(system=platform.system(),macos_version=platform.mac_ver()[0]),
        completed_utc=datetime.now(timezone.utc).isoformat(),**local,remote=remote,input_identities=list(unique_receipts.values()),
        missing_evidence=['Actual remote tensor measurements are pending.' if remote['status']!='verified' else 'No missing physical tensor identities.',
            'Standalone historical tokenizer/model config bytes are not in the restored package; embedded identity records agree but live tokenizer replay was not performed.',
            'The final approved factual audit, admitted rows, person keys, corrected split/grouping policy and corrected-fit target representation are not yet supplied.'],
        reuse_conditions=['Physical bytes, NPY header and live sidecar/manifest hashes must match before using a tensor.',
            'Use original dataset + zero-based source row index + source-file hash + exact statement hash; never text alone or alternate export order.',
            'Final admitted rows and person keys must come from the subsequently approved audit version; this preflight makes no corrected selection or A/P reservation.',
            'The intended fit must explicitly adopt the verified historical representation, tokenizer, token position, layers and dtype.',
            'Label or person-key corrections alone do not change activations of unchanged exact text; apply approved targets and leakage exclusions later without altering historical labels.'],
        fresh_extraction_required_when=['An approved required source row/exact statement is absent, changed or not uniquely bound to a cache row.',
            'Model/revision, tokenizer/preprocessing, readout token, saved layer convention, numerical extraction format or required dtype differ from the accepted target contract.',
            'Tensor/sidecar integrity conflicts or missing bytes cannot be resolved by retrieving a byte-verified original cache.',
            'The approved future policy requires representation evidence beyond what these historical records establish.'],
        invariants=dict(historical_artifacts_modified=False,final_selection_map_created=False,fitting=False,predictions=False,
            final_test_activations_accessed=False,ap_reservation=False,benchmark_generation=False,factual_approval=False),
        code_sha256={name:c.file_hash(repo/name) for name in ['src/t2_cache_preflight.py','src/t2_cache_remote.py','scripts/47_preflight_atomic_caches.py']})
    (output/'compatibility.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    lines=['# T2 atomic cache compatibility preflight','',SCOPE,'',
        '| Cache | Source-row/representation binding | Physical tensor | Rows × layers × width | Bytes |',
        '|---|---|---|---|---|']
    for cache,item in local['caches'].items():
        tensor=item['expected_tensor'];lines.append(f"| {cache} | {item['local_binding_status']} | {item['physical_status']} | {' × '.join(map(str,tensor['shape']))} | {tensor['bytes']} |")
    lines+=['','Sizes/shapes in the table are recorded expectations until physical verification succeeds. All expect little-endian C-order float16.',
        '',f"Remote status: {remote['status']}. {remote.get('reason','')}",'',
        'Local checks bind every cache row to exact original text, label, source-file hash, zero-based source index, and original partition. Qwen and Llama row orders match.',
        'Duplicate text remains keyed by distinct source-row IDs. No correction decisions or person keys were loaded.',
        '', '## Missing evidence and conflicts','']
    lines += ['- '+x for x in report['missing_evidence']]
    lines += ['- `pinned_atomic_probes.py`: working-file `'+report['historical_producer_conflict']['current_sha256']+'` versus recorded producer/Git `'+report['historical_producer_conflict']['recorded_sha256']+'`; unresolved. Matching tensors would not resolve execution provenance.',
        '- The two packaged method-suite allowed-row exports contain the same rows but differ in ordering from the extraction-bound original export; never use their row positions as tensor indices.',
        '- Other recorded extraction source-hash differences are preserved in the JSON report, not silently accepted as execution proof.',
        '', '## Reuse conditions','']
    lines += ['- '+x for x in report['reuse_conditions']]
    lines += ['','## When fresh extraction is required','']+['- '+x for x in report['fresh_extraction_required_when']]
    lines += ['','No physical-verification success or final corrected-fit authorization is inferred from matching counts or filenames.','']
    (output/'compatibility.md').write_text('\n'.join(lines))
    print(json.dumps(dict(output_root=str(output),local_status='verified',remote_status=remote['status'],
                         caches={k:dict(rows=v['rows'],duplicate_statement_rows=v['duplicate_statement_rows'],physical_status=v['physical_status']) for k,v in local['caches'].items()}),indent=2))
