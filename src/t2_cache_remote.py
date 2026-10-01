"""Standalone stdlib-only read-only verifier sent to SSH as code, never installed."""
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys

ROOT = Path('/workspace/truth-probing-artifacts')
CACHES = {
    'qwen/train': ROOT/'atomic_repair/qwen25_a09a354_bs1_bf16_v1/train',
    'qwen/validation': ROOT/'atomic_repair/qwen25_a09a354_bs1_bf16_v1/validation',
    'llama/train': ROOT/'llama31_replication_v1/atomic/train',
    'llama/validation': ROOT/'llama31_replication_v1/atomic/validation',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def npy_header(handle):
    """Read a literal NPY header only, never array values or pickle objects."""
    require(handle.read(6) == b'\x93NUMPY', 'not a NumPy array')
    version = tuple(handle.read(2))
    require(version in [(1, 0), (2, 0)], 'unsupported NPY version')
    width = 2 if version == (1, 0) else 4
    size = struct.unpack('<H' if width == 2 else '<I', handle.read(width))[0]
    require(0 < size <= 65536, 'invalid NPY header length')
    payload = handle.read(size)
    require(len(payload) == size, 'truncated NPY header')
    fields = ast.literal_eval(payload.decode('latin1').strip())
    require(isinstance(fields, dict) and set(fields) == {'descr', 'fortran_order', 'shape'}, 'invalid NPY schema')
    require(isinstance(fields['shape'], tuple) and len(fields['shape']) == 3 and
            all(type(x) is int and x > 0 for x in fields['shape']), 'invalid activation shape')
    require(fields['descr'] == '<f2' and fields['fortran_order'] is False, 'expected little-endian C-order float16')
    return dict(version=list(version), shape=list(fields['shape']), dtype=fields['descr'],
                fortran_order=fields['fortran_order'], header_bytes=handle.tell(),
                computed_file_bytes=handle.tell()+math.prod(fields['shape'])*2)


def measure(path, tensor=False):
    """Hash one fixed file; reject links and files changing during the read."""
    path = Path(path)
    require(path.is_absolute() and path.is_relative_to(ROOT) and path.resolve() == path, 'unsafe/linked remote path')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'not a regular file')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as handle:
        opened = os.fstat(handle.fileno())
        require((before.st_dev,before.st_ino) == (opened.st_dev,opened.st_ino), 'file replaced before read')
        header = npy_header(handle) if tensor else None
        handle.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
        after = os.fstat(handle.fileno())
    stable = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
    require(stable(before) == stable(after) == stable(path.stat()), 'file changed during read')
    if header: require(header['computed_file_bytes'] == after.st_size, 'NPY payload size mismatch')
    return dict(path=str(path), bytes=after.st_size, sha256=digest.hexdigest(), header=header,
                stable_during_read=True)


def verify(requests):
    require(set(requests) == set(CACHES), 'only the four atomic train/validation caches are allowed')
    results = {}
    for cache, directory in CACHES.items():
        expected = requests[cache]
        try:
            allowed = {str(directory/'metadata.csv')}
            if cache.startswith('qwen/'):
                allowed.update(str(directory.parent/name) for name in ['completion.json','extraction_manifest.json'])
            else:
                allowed.update(str(directory/name) for name in ['extraction_manifest.json','progress.json'])
            require(set(expected['companions']) == allowed and expected['path'] == str(directory/'activations.npy'), 'remote file allowlist differs')
            companions = {}
            for path, record in expected['companions'].items():
                actual = measure(path)
                require(all(actual[k] == record[k] for k in ['bytes','sha256']), 'live companion differs: '+path)
                companions[path] = actual
            tensor = measure(expected['path'], tensor=True)
            # Recheck companions after tensor hashing, so a sidecar change cannot pass unnoticed.
            require(companions == {path:measure(path) for path in companions}, 'live companion changed during tensor read')
            match = (tensor['bytes'] == expected['bytes'] and tensor['sha256'] == expected['sha256'] and
                     tensor['header']['shape'] == expected['shape'])
            results[cache] = dict(status='verified' if match else 'conflict', tensor=tensor, companions=companions,
                                  error=None if match else 'physical tensor identity differs from recorded contract')
        except FileNotFoundError as exc:
            results[cache] = dict(status='missing', error=str(exc))
        except (OSError, ValueError, SyntaxError, struct.error) as exc:
            results[cache] = dict(status='conflict', error=str(exc))
    return results


if __name__ == '__main__':
    print(json.dumps(verify(json.load(sys.stdin)), sort_keys=True))
