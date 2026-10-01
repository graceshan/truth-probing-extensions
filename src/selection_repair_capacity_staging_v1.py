"""Versioned v4 pilot request/staging; one-shot host-verified SSH, metadata-bound rows."""
import copy
import json
import platform
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path
from datetime import datetime, timezone
from src import selection_repair_sensitivity_v4 as s
from src.selection_repair_canonical_sensitivity_v4 import BoundInputs
from src.selection_repair_sensitivity_verification import audit_cache
from src.selection_repair_cache_export_remote import DIRECTORIES

LAYERS={'qwen':[17,18,22],'llama':[10,15,20]}

def make_request(root,payload,prepared,artifacts):
    bundle=BoundInputs(root,payload,prepared,artifacts)
    atom=s.parse_rows((Path(prepared)/'atomic_D_eligibility.csv').read_bytes())
    pairs=s.parse_rows((Path(prepared)/'development_pair_eligibility.csv').read_bytes())
    allowed={r['pair_id'] for r in pairs if r['tc_current_eligible']=='True'}
    raw=s.parse_rows(bundle.read(s.RAW+'metadata.csv'))
    eligible_raw={r['example_id']:r for r in raw if r['pair_id'] in allowed}
    requests={};bindings={}
    for key in ('qwen_train','qwen_validation','qwen_raw','llama_train','llama_validation','llama_transfer'):
        source=s.parse_rows(bundle.read(DIRECTORIES[key]+'/metadata.csv'))
        if key.endswith('train'):
            mapped=s.parse_rows((Path(prepared)/(key+'.mapping_v4.csv')).read_bytes())
            rows=[int(r['tensor_row_index']) for r in mapped]
            identities=mapped
        elif key.endswith('validation'):
            rows=[i for i,r in enumerate(atom) if r['tc_current_eligible']=='True']
            identities=[atom[i] for i in rows]
        else:
            rows=[i for i,r in enumerate(source) if r['example_id'] in eligible_raw]
            s.require(len(rows)==len(eligible_raw) and len({source[i]['example_id'] for i in rows})==len(rows),'bare compound coverage')
            for i in rows:
                a,b=source[i],eligible_raw[source[i]['example_id']]
                s.require(a.get('condition_id','raw_reference')=='raw_reference', 'non-bare condition')
                for field in ('statement','split','protocol','evaluation_phase'):
                    s.require(a[field]==b[field],'bare metadata binding: '+field)
            identities=[dict(tensor_row_index=i, sidecar=source[i], compound=eligible_raw[source[i]['example_id']]) for i in rows]
        s.require(rows==sorted(set(rows)), 'original sidecar order required')
        bindings[key]=dict(original_row_indices=rows,identities=identities,sidecar=s.identity(bundle.read(DIRECTORIES[key]+'/metadata.csv')))
        for layer in LAYERS[key.split('_')[0]]:
            if layer==bundle.config['models'][key.split('_')[0]]['layer']:continue
            spec=copy.deepcopy(bundle.request[key]);spec.update(layer=layer,row_indices=rows,row_identity_sha256=s.sha(s.canonical(identities)))
            requests[key+'_L'+str(layer)]=spec
    bundle.unchanged()
    return dict(schema_version=1,correction_commit=s.CORRECTION_COMMIT,adoption_sha256=bundle.config['adoption_sha256'],
                preparation_receipt=s.identity((Path(prepared)/'preparation_receipt.json').read_bytes()),caches=requests),bindings


def verify(root,prepared,output):
    root,prepared,output=map(Path,(root,prepared,output))
    config,_=s.verify_preparation(root,prepared)
    request=(output/'export_request.json').read_bytes();req=json.loads(request)
    remote_bytes=(output/'export_receipt.json').read_bytes();remote=json.loads(remote_bytes)
    stage=json.loads((output/'staging_receipt.json').read_text())
    s.require(req['adoption_sha256']==config['adoption_sha256'] and req['correction_commit']==s.CORRECTION_COMMIT
              and req['preparation_receipt']==s.identity((prepared/'preparation_receipt.json').read_bytes()),'stale pilot request')
    s.require(stage['remote_program_sha256']==s.sha((root/'src/selection_repair_capacity_export_v2.py').read_bytes()),'pilot exporter changed')
    s.require(stage['remote_receipt']==s.identity(remote_bytes) and stage['export_request_sha256']==remote['request_sha256']==s.sha(request),'pilot receipt changed')
    s.require(stage['row_bindings']==s.identity((output/'row_bindings.json').read_bytes()),'pilot row bindings changed')
    s.require(stage['local_platform']=='Darwin' and remote['remote_hostname']=='ef7f7534c328','pilot host mismatch')
    s.require(all(remote[k] is False for k in ('model_extraction','final_test_access','remote_files_written')),'pilot scope mismatch')
    expected={k+'_L'+str(l) for k in DIRECTORIES if k!='qwen_controls' for l in ([18,22] if k.startswith('qwen') else [10,20])}
    s.require(set(req['caches'])==set(remote['caches'])==expected,'pilot export set')
    result={key:audit_cache(output/(key+'.npy'),spec,remote['caches'][key]) for key,spec in req['caches'].items()}
    return result


def stage(root,payload,prepared,artifacts,output):
    root,output=Path(root),Path(output)
    s.require(platform.system()=='Darwin' and not output.exists() and not output.resolve().is_relative_to(root.resolve()),'new external macOS destination required')
    request,bindings=make_request(root,payload,prepared,artifacts)
    estimate=sum(len(v['row_indices'])*v['shape'][2]*2+128 for v in request['caches'].values())
    free=shutil.disk_usage(output.parent).free
    s.require(free>estimate*3+1024**3,'insufficient free storage')
    output.mkdir();data=s.json_bytes(request)
    (output/'export_request.json').write_bytes(data);(output/'row_bindings.json').write_bytes(s.json_bytes(bindings))
    program=(root/'src/selection_repair_capacity_export_v2.py').read_text()
    command=['ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15','-p','22114','-i','/Users/apple/.ssh/id_ed25519','root@69.30.85.22','python3 -B -c '+shlex.quote(program)]
    with (output/'selected_layers.tar').open('xb') as stream:
        process=subprocess.run(command,input=data,stdout=stream,stderr=subprocess.PIPE,timeout=1800)
    if process.returncode:
        (output/'transfer_failure.json').write_bytes(s.json_bytes(dict(exit_code=process.returncode,blocker=process.stderr.decode(errors='replace'))))
        raise RuntimeError('SSH export failed; no automatic retries; see transfer_failure.json')
    allowed={'export_receipt.json'}|{k+'.npy' for k in request['caches']};seen=set()
    with tarfile.open(output/'selected_layers.tar','r:') as archive:
        for member in archive:
            s.require(member.name in allowed and member.name not in seen and member.isfile() and member.size<512*1024**2,'unexpected export member')
            seen.add(member.name)
            with (output/member.name).open('xb') as stream:shutil.copyfileobj(archive.extractfile(member),stream)
    receipt=dict(local_platform=platform.system(),local_hostname=platform.node(),completed_utc=datetime.now(timezone.utc).isoformat(),
                 free_bytes_before=free,expected_export_bytes=estimate,export_request_sha256=s.sha(data),
                 remote_program_sha256=s.sha(program.encode()),remote_receipt=s.identity((output/'export_receipt.json').read_bytes()),
                 row_bindings=s.identity((output/'row_bindings.json').read_bytes()),interactive_shell=False,model_extraction=False,final_test_access=False)
    (output/'staging_receipt.json').write_bytes(s.json_bytes(receipt))
    validated=verify(root,prepared,output)
    (output/'verification.json').write_bytes(s.json_bytes(validated))
    print(json.dumps({'verified_exports':len(validated),'bytes':estimate,'local_platform':platform.system()},indent=2))
