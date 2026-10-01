"""Pilot v4 additional layers only. Separate from the immutable canonical exporter."""
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import socket
import stat
import struct
import sys
import tarfile

ROOT = Path('/workspace/truth-probing-artifacts')
DIRECTORIES = {
    'qwen_train': 'atomic_repair/qwen25_a09a354_bs1_bf16_v1/train',
    'qwen_validation': 'atomic_repair/qwen25_a09a354_bs1_bf16_v1/validation',
    'llama_train': 'llama31_replication_v1/atomic/train',
    'llama_validation': 'llama31_replication_v1/atomic/validation',
    'qwen_raw': 'pinned_compound_qwen2_5/qwen2_5_7b_a09a354_bs1_bf16_v1',
        'llama_transfer': 'llama31_replication_v1/transfer',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token(st):
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def header(handle):
    require(handle.read(6) == b'\x93NUMPY', 'not NPY')
    version = tuple(handle.read(2))
    require(version in ((1, 0), (2, 0)), 'unsupported NPY version')
    width = 2 if version == (1, 0) else 4
    length = struct.unpack('<H' if width == 2 else '<I', handle.read(width))[0]
    require(0 < length <= 65536, 'bad NPY header length')
    fields = ast.literal_eval(handle.read(length).decode('latin1').strip())
    require(set(fields) == {'descr', 'shape', 'fortran_order'}, 'bad NPY fields')
    require(fields['descr'] == '<f2' and fields['fortran_order'] is False, 'expected C-order little-endian float16')
    require(isinstance(fields['shape'], tuple) and len(fields['shape']) in (2, 3)
            and all(type(n) is int and n > 0 for n in fields['shape']), 'bad NPY shape')
    return dict(shape=list(fields['shape']), dtype=fields['descr'], fortran_order=False,
                header_bytes=handle.tell(), computed_file_bytes=handle.tell()+2*math.prod(fields['shape']), version=list(version))


def open_regular(path):
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(ROOT) and path.resolve() == path, 'unsafe/linked path')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'not a regular file')
    handle = os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), 'rb')
    require(token(before) == token(os.fstat(handle.fileno())), 'file changed before open')
    return handle, before


def measure(path, tensor=False):
    handle, before = open_regular(path)
    with handle:
        meta = header(handle) if tensor else None
        handle.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
        require(token(before) == token(os.fstat(handle.fileno())) == token(Path(path).stat()), 'file changed while hashing')
    if meta:
        require(meta['computed_file_bytes'] == before.st_size, 'NPY payload length mismatch')
    return dict(path=str(path), bytes=before.st_size, sha256=digest.hexdigest(), header=meta,
                stable_during_read=True), token(before)


def export_layer(request, measured, source_token):
    rows, layer = request['row_indices'], request['layer']
    shape = measured['header']['shape']
    require(len(shape) == 3 and shape == request['shape'], 'source shape mismatch')
    require(rows and rows == sorted(set(rows)) and all(type(i) is int and 0 <= i < shape[0] for i in rows), 'bad original row indices')
    require(type(layer) is int and 0 <= layer < shape[1], 'bad saved layer index')
    fields = repr({'descr': '<f2', 'fortran_order': False, 'shape': (len(rows), shape[2])})
    payload = fields.encode('latin1')
    payload += b' ' * ((64 - (10 + len(payload) + 1) % 64) % 64) + b'\n'
    output = io.BytesIO(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(payload)) + payload)
    output.seek(0, 2)
    handle, before = open_regular(request['path'])
    with handle:
        require(token(before) == source_token, 'source changed after verification')
        for row in rows:
            handle.seek(measured['header']['header_bytes'] + (row*shape[1]+layer)*shape[2]*2)
            block = handle.read(shape[2]*2)
            require(len(block) == shape[2]*2, 'truncated layer row')
            output.write(block)
        require(token(before) == token(os.fstat(handle.fileno())) == token(Path(request['path']).stat()), 'source changed during export')
    return output.getvalue()


def verify_and_export(key, request):
    require(key in DIRECTORIES, 'cache not allowlisted')
    directory = ROOT / DIRECTORIES[key]
    require(request['path'] == str(directory/'activations.npy'), 'tensor path mismatch')
    require(request['layer'] in ([18,22] if key.startswith('qwen') else [10,20]), 'layer not predeclared')
    companions = {str(directory/'metadata.csv')}
    if key in ('qwen_train', 'qwen_validation'):
        companions |= {str(directory.parent/name) for name in ('completion.json', 'extraction_manifest.json')}
    else:
        companions |= {str(directory/name) for name in ('extraction_manifest.json', 'progress.json')}
    require(set(request['companions']) == companions, 'companion allowlist mismatch')
    checked = {}
    for path, expected in request['companions'].items():
        actual, _ = measure(path)
        require(all(actual[k] == expected[k] for k in ('bytes', 'sha256')), 'companion identity mismatch: '+path)
        checked[path] = actual
    actual, source_token = measure(request['path'], tensor=True)
    require(all(actual[k] == request[k] for k in ('bytes', 'sha256')), 'tensor identity mismatch')
    data = export_layer(request, actual, source_token)
    require(checked == {path: measure(path)[0] for path in checked}, 'companions changed during verification/export')
    return data, dict(status='verified', source=actual, companions=checked, original_row_indices=request['row_indices'],
                      saved_layer_index=request['layer'], row_identity_sha256=request['row_identity_sha256'],
                      source_stable_through_export=True, export=dict(bytes=len(data), sha256=sha(data),
                      shape=[len(request['row_indices']), request['shape'][2]], dtype='<f2', fortran_order=False))


def main():
    require(socket.gethostname() == 'ef7f7534c328', 'unexpected remote hostname')
    raw = sys.stdin.buffer.read()
    request = json.loads(raw)
    require(set(request['caches']) == {k+'_L'+str(l) for k in DIRECTORIES for l in ([18,22] if k.startswith('qwen') else [10,20])}, 'exact twelve pilot exports required')
    receipt = dict(remote_hostname=socket.gethostname(), request_sha256=sha(raw), caches={}, model_extraction=False,
                   final_test_access=False, remote_files_written=False)
    with tarfile.open(fileobj=sys.stdout.buffer, mode='w|') as archive:
        for key, spec in request['caches'].items():
            try:
                data, result = verify_and_export(key.rsplit('_L',1)[0], spec)
                info = tarfile.TarInfo(key+'.npy'); info.size=len(data)
                archive.addfile(info, io.BytesIO(data))
                receipt['caches'][key] = result
            except (OSError, ValueError, SyntaxError, struct.error) as exc:
                receipt['caches'][key] = dict(status='not_computed', dependency_error=str(exc))
        data = (json.dumps(receipt, sort_keys=True, indent=2)+'\n').encode()
        info = tarfile.TarInfo('export_receipt.json'); info.size=len(data)
        archive.addfile(info, io.BytesIO(data))


if __name__ == '__main__':
    main()
