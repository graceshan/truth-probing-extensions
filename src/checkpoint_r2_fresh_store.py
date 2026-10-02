"""Immutable NPY shards and downstream fresh feature loader; no historical fits."""
import json
import os
import uuid
from pathlib import Path
import numpy as np
from src.checkpoint_r2_fresh_inputs import canonical, file_hash, hash_value, require


def fsync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish(path, write):
    """Crash leaves a .partial; exclusive link publishes complete data; fsync directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.' + uuid.uuid4().hex + '.partial')
    with partial.open('xb') as f:
        write(f)
        f.flush()
        os.fsync(f.fileno())
    os.link(partial, path)
    fsync_dir(path.parent)
    partial.unlink()
    fsync_dir(path.parent)


def write_json(path, value):
    publish(path, lambda f: f.write(canonical(value)))


def pin(path):
    return dict(sha256=file_hash(path), bytes=Path(path).stat().st_size)


def write_shard(path, rows, compute, spec, provenance, tokens):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    require(compute.shape == (len(rows), spec['layers'], spec['width']), 'feature shape')
    require(compute.dtype == np.float32 and np.isfinite(compute).all(), 'compute dtype/nonfinite')
    with np.errstate(over='ignore'):
        features = compute.astype(np.float16)
    require(np.isfinite(features).all(), 'FP16 overflow')
    publish(path / 'activations.npy', lambda f: np.save(f, features, allow_pickle=False))
    write_json(path / 'rows.json', rows)
    write_json(path / 'tokens.json', tokens)
    receipt = dict(schema='r2-fresh-shard-v1', model=spec, provenance=provenance,
                   shape=list(features.shape), dtype='float16', axes=['input', 'saved_layer', 'feature'],
                   saved_layers=list(range(spec['layers'])), hf_hidden_state_indices=list(range(1, spec['layers'] + 1)),
                   final_layer='post-final-RMSNorm', embedding_saved=False,
                   ordered_ids=[r['id'] for r in rows], ordered_rows_sha256=hash_value(rows),
                   payload_sha256=__import__('hashlib').sha256(features.tobytes()).hexdigest(),
                   files={name: pin(path / name) for name in ('activations.npy', 'rows.json', 'tokens.json')})
    write_json(path / 'receipt.json', receipt)
    restored, restored_rows, _ = load_features(path, expected_rows=rows, expected_spec=spec,
                                            expected_provenance=provenance)
    require(np.array_equal(restored, features) and restored_rows == rows, 'write/reopen values')
    return dict(path=str(path), receipt=pin(path / 'receipt.json'), output_bytes=sum(
                f.stat().st_size for f in path.iterdir()), payload_bytes=features.nbytes)


def load_features(path, *, expected_rows=None, expected_spec=None, expected_provenance=None):
    """Downstream loader: hash, ordered composite bindings, dtype and layer axes are mandatory."""
    path = Path(path)
    receipt = json.loads((path / 'receipt.json').read_text())
    require(receipt['schema'] == 'r2-fresh-shard-v1', 'wrong feature schema')
    require(set(receipt['files']) == {'activations.npy', 'rows.json', 'tokens.json'}, 'shard file set')
    for name, identity in receipt['files'].items():
        require(pin(path / name) == identity, 'shard hash: ' + name)
    rows = json.loads((path / 'rows.json').read_text())
    require(receipt['ordered_ids'] == [r['id'] for r in rows]
            and len(set(receipt['ordered_ids'])) == len(rows)
            and receipt['ordered_rows_sha256'] == hash_value(rows), 'shard IDs/order/bindings')
    spec = receipt['model']
    if expected_rows is not None:
        require(rows == expected_rows, 'downstream ordered rows')
    if expected_spec is not None:
        require(spec == expected_spec, 'downstream model identity')
    if expected_provenance is not None:
        require(receipt['provenance'] == expected_provenance, 'downstream representation identity')
    require(receipt['axes'] == ['input', 'saved_layer', 'feature']
            and receipt['saved_layers'] == list(range(spec['layers']))
            and receipt['hf_hidden_state_indices'] == list(range(1, spec['layers'] + 1))
            and receipt['final_layer'] == 'post-final-RMSNorm' and not receipt['embedding_saved'], 'layer convention')
    values = np.load(path / 'activations.npy', mmap_mode='r', allow_pickle=False)
    require(values.shape == (len(rows), spec['layers'], spec['width']) == tuple(receipt['shape'])
            and values.dtype == np.float16 and receipt['dtype'] == 'float16'
            and np.isfinite(values).all(), 'shard axes/dtype/nonfinite')
    require(__import__('hashlib').sha256(values.tobytes()).hexdigest() == receipt['payload_sha256'], 'payload values')
    return values, rows, receipt


def storage_probe(parent, reserve_bytes):
    """Real allocation/publication/recovery test. Reported statvfs is informational only."""
    parent = Path(parent)
    require(parent.is_dir(), 'storage parent must exist')
    probe = parent / ('.fresh-storage-' + uuid.uuid4().hex)
    probe.mkdir()
    try:
        expected_hash = __import__('hashlib').sha256()
        def allocation(f):
            left = reserve_bytes
            while left:
                chunk = os.urandom(min(1024 * 1024, left))
                expected_hash.update(chunk)
                f.write(chunk)
                left -= len(chunk)
        publish(probe / 'reserved', allocation)
        identity = pin(probe / 'reserved')
        require(identity['bytes'] == reserve_bytes and identity['sha256'] == expected_hash.hexdigest(), 'storage allocation/readback')
        # Actual writer uses exclusive link, fsync, orphan recovery and immutable names.
        try:
            publish(probe / 'reserved', lambda f: f.write(b'overwrite'))
        except FileExistsError:
            pass
        else:
            raise ValueError('storage allowed immutable overwrite')
        require(pin(probe / 'reserved') == identity, 'immutable publication changed existing data')
        orphans = list(probe.glob('*.partial'))
        require(len(orphans) == 1, 'expected recoverable unpublished partial')
        orphans[0].unlink()
        os.replace(probe / 'reserved', probe / 'recovered')
        fsync_dir(probe)
        require(pin(probe / 'recovered') == identity, 'storage recovery changed data')
        fs = os.statvfs(parent)
        return dict(parent=str(parent), actual_write_and_readback_bytes=reserve_bytes,
                    operations=['fsync_file', 'exclusive_atomic_link', 'fsync_directory', 'refuse_overwrite',
                                'remove_partial', 'atomic_replace', 'reopen_hash'],
                    reported_free_bytes=fs.f_bavail * fs.f_frsize,
                    quota_verified='bounded reservation only; full production quota NOT established',
                    power_loss_durability='not established by process-level checks')
    finally:
        for p in probe.iterdir():
            p.unlink()
        probe.rmdir()
        fsync_dir(parent)
