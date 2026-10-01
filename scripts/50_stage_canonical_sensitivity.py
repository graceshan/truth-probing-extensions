#!/usr/bin/env python3
"""Verify/export fixed cached layers over one-shot SSH; keep tensors outside checkout."""
import argparse
import json
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.selection_repair_sensitivity_inputs import verify_preparation, sha, identity, require, json_bytes
from src.selection_repair_cache_export_remote import header


def stage(prepared, output):
    prepared, output = Path(prepared).resolve(), Path(output).resolve()
    require(platform.system() == 'Darwin', 'staging must run on macOS')
    config, inventory = verify_preparation(ROOT, prepared)
    require(not output.is_relative_to(ROOT) and not output.exists(), 'new external artifact directory required')
    free = shutil.disk_usage(output.parent).free
    require(free > inventory['selected_export_bytes_estimate']*3 + 1024**3, 'insufficient free local storage')
    output.mkdir()
    request = (prepared/'export_request.json').read_bytes()
    program = (ROOT/'src/selection_repair_cache_export_remote.py').read_text()
    command = ['ssh','-T','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','StrictHostKeyChecking=yes',
               '-o','ConnectTimeout=15','-p','22114','-i','/Users/apple/.ssh/id_ed25519','root@69.30.85.22',
               'python3 -B -c '+shlex.quote(program)]
    with (output/'selected_layers.tar').open('xb') as stream:
        process = subprocess.run(command, input=request, stdout=stream, stderr=subprocess.PIPE, timeout=1800)
    if process.returncode:
        failure = dict(status='not_computed', ssh_exit_code=process.returncode,
                       blocker=process.stderr.decode(errors='replace'), local_platform=platform.system())
        (output/'transfer_failure.json').write_bytes(json_bytes(failure))
        raise RuntimeError('One-shot SSH transfer failed; see '+str(output/'transfer_failure.json'))
    requests = json.loads(request)['caches']
    allowed = {'export_receipt.json'} | {key+'.npy' for key in requests}
    seen = set()
    with tarfile.open(output/'selected_layers.tar', 'r:') as archive:
        for member in archive:
            require(member.name in allowed and member.name not in seen and member.isfile(), 'unexpected tar member')
            require(member.size < 512*1024*1024, 'export member too large')
            seen.add(member.name)
            with (output/member.name).open('xb') as stream:
                shutil.copyfileobj(archive.extractfile(member), stream)
    remote = json.loads((output/'export_receipt.json').read_text())
    require(remote['request_sha256'] == sha(request) and remote['remote_hostname'] == 'ef7f7534c328', 'remote receipt/request mismatch')
    require(set(remote['caches']) == set(requests), 'incomplete export response')
    for key, record in remote['caches'].items():
        if record['status'] != 'verified': continue
        expected = requests[key]; path = output/(key+'.npy')
        require(record['original_row_indices'] == expected['row_indices'] and record['saved_layer_index'] == expected['layer']
                and record['row_identity_sha256'] == expected['row_identity_sha256'], 'row/layer export identity mismatch')
        require(all(record['source'][k] == expected[k] for k in ('path','bytes','sha256')), 'source tensor receipt mismatch')
        require(identity(path.read_bytes()) == {k:record['export'][k] for k in ('bytes','sha256')}, 'local export hash mismatch')
        with path.open('rb') as stream: meta = header(stream)
        require(meta['shape'] == record['export']['shape'] == [len(expected['row_indices']),expected['shape'][2]]
                and meta['computed_file_bytes'] == path.stat().st_size, 'local export header mismatch')
    verify_preparation(ROOT, prepared)
    receipt = dict(status='complete' if all(r['status']=='verified' for r in remote['caches'].values()) else 'partial',
                   local_platform=platform.system(), completed_utc=datetime.now(timezone.utc).isoformat(),
                   free_bytes_before=free, expected_export_bytes=inventory['selected_export_bytes_estimate'],
                   preparation_receipt_sha256=sha((prepared/'preparation_receipt.json').read_bytes()),
                   export_request_sha256=sha(request), remote_program_sha256=sha(program.encode()),
                   remote_receipt=identity((output/'export_receipt.json').read_bytes()),
                   remote_hostname=remote['remote_hostname'], cache_status={k:r['status'] for k,r in remote['caches'].items()},
                   final_test_access=False, model_extraction=False, interactive_shell=False)
    (output/'staging_receipt.json').write_bytes(json_bytes(receipt))
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--artifact-root', type=Path, required=True)
    args = parser.parse_args()
    stage(args.prepared_root, args.artifact_root)
