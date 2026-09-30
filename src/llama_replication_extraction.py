"""Llama-only pinned extraction adapters; no floating revision or TEST interface."""
import inspect
import platform
from pathlib import Path

import numpy as np
import pandas as pd

from src import clean_transfer_contracts as c
from src import clean_atomic_extraction as atomic
from src import llama_replication_contracts as l
from src import priority2_input_controls as p
from src.repaired_atomic_cache import array_header


def pin_revision(revision, root=c.ROOT):
    """Metadata/tokenizer only: never downloads weights. Revision must be explicit."""
    from transformers import AutoTokenizer
    from transformers.utils.hub import cached_file, extract_commit_hash
    c.require(not c.safe_path(root, l.PIN).exists(), 'refusing existing model pin')
    descriptor = l.representation(revision)
    paths = {name: Path(cached_file(l.MODEL, name, revision=revision))
             for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json']}
    c.require(all(extract_commit_hash(str(path), None) == revision for path in paths.values()), 'HF resolution mismatch')
    config = c.read_json(paths['config.json'])
    l.discover(config)
    tokenizer = AutoTokenizer.from_pretrained(l.MODEL, revision=revision, trust_remote_code=False)
    pin = l.validate_pin(dict(schema_version=1, representation=descriptor,
        fingerprint=c.digest(c.canonical(descriptor)), config=config,
        files={name: c.record(path) for name, path in paths.items()},
        tokenizer_backend_sha256=c.digest(tokenizer.backend_tokenizer.to_str().encode()),
        tokenizer_class=type(tokenizer).__name__, special_tokens={key: getattr(tokenizer, key, None)
        for key in ['bos_token_id', 'eos_token_id', 'add_bos_token', 'add_eos_token']}))
    c.publish_json(c.safe_path(root, l.PIN), pin)
    return dict(pin=l.PIN, revision=revision, weights_downloaded=False, scores_computed=0)


def load_model(pin):
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.hub import cached_file, extract_commit_hash
    atomic.require_runtime()
    revision = pin['representation']['model_revision']
    for name, identity in pin['files'].items():
        path = cached_file(l.MODEL, name, revision=revision)
        c.require(extract_commit_hash(path, None) == revision and c.record(Path(path)) == identity, 'pinned HF file changed')
    tokenizer = AutoTokenizer.from_pretrained(l.MODEL, revision=revision, trust_remote_code=False)
    c.require(c.digest(tokenizer.backend_tokenizer.to_str().encode()) == pin['tokenizer_backend_sha256'] and
              type(tokenizer).__name__ == pin['tokenizer_class'] and
              {key: getattr(tokenizer, key, None) for key in pin['special_tokens']} == pin['special_tokens'], 'tokenizer changed')
    model = AutoModelForCausalLM.from_pretrained(l.MODEL, revision=revision, trust_remote_code=False,
        dtype=torch.bfloat16, attn_implementation='sdpa').to('cuda').eval()
    l.discover(model.config.to_dict())
    c.require(model.config._commit_hash == revision and model.dtype == torch.bfloat16 and
              model.config._attn_implementation == 'sdpa', 'live model contract mismatch')
    runtime = dict(torch=torch.__version__, transformers=transformers.__version__, numpy=np.__version__,
        python=platform.python_version(), cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        model_implementation_sha256=c.file_hash(Path(inspect.getfile(type(model)))))
    return model, tokenizer, runtime


def readout(model, tokenizer, statement, *, native=False):
    """Native-position/full-logit reference is a smoke-only path."""
    import torch
    from src.extract import saved_hidden_states, last_real_token_indices
    c.require(not model.training and isinstance(statement, str) and statement.strip(), 'eval/raw statement required')
    encoded = tokenizer([statement], return_tensors='pt', padding=False, truncation=False,
                        add_special_tokens=True, return_attention_mask=True).to(next(model.parameters()).device)
    mask = encoded['attention_mask']
    c.require(mask.ndim == 2 and mask.shape[0] == 1 and mask.shape[1] > 0 and bool((mask == 1).all()) and
              mask.shape == encoded['input_ids'].shape, 'batch-one/no-padding required')
    c.require(mask.shape[1] <= model.config.max_position_embeddings, 'no truncation allowed')
    if not native:
        encoded['position_ids'] = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    with torch.inference_mode():
        output = model(**encoded, output_hidden_states=True, use_cache=False, logits_to_keep=0 if native else 1)
        states = saved_hidden_states(output.hidden_states, model.config.num_hidden_layers)
        position = -1 if native else last_real_token_indices(mask).item()
        values = torch.stack([x[0, position] for x in states]).float().cpu().numpy()
    c.require(values.shape == (model.config.num_hidden_layers, model.config.hidden_size) and
              np.isfinite(values).all(), 'invalid readout')
    return values


def smoke(model, tokenizer, statements):
    for statement in statements:
        a = readout(model, tokenizer, statement)
        b = readout(model, tokenizer, statement)
        native = readout(model, tokenizer, statement, native=True)
        c.require(np.array_equal(a, b) and np.array_equal(a, native), 'repeat/native-position smoke failed')
    return dict(passed=True, policy='llama-unpadded-repeat-native-position-exact-v1',
                statements_sha256=c.ordered_hash(statements), samples=len(statements), batch_size=1, padding=False)


class Rows:
    """Projected statement adapter for the existing durable ExtractionWriter."""
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True).copy()
        c.require(self.frame.example_id.is_unique and self.frame.statement.str.strip().ne('').all(), 'row identity')
        self.payload = p.csv_bytes(self.frame)
        self.hashes = dict(benchmark_sha256=c.digest(self.payload), sidecar_sha256=c.digest(self.payload),
            ordered_example_id_sha256=c.ordered_hash(self.frame.example_id),
            ordered_statement_sha256=c.ordered_hash(self.frame.statement))

    def verify_source(self):
        c.require(p.csv_bytes(self.frame) == self.payload, 'rows changed')

    def verify_rows(self, start, rows):
        c.require(rows.reset_index(drop=True).equals(self.frame.iloc[start:start+len(rows)].reset_index(drop=True)), 'row order')

    def verify_sidecar(self, path):
        c.require(path.read_bytes() == self.payload, 'sidecar changed')


def atomic_rows(root):
    partitions, identity = atomic.load_allowed_rows(root)
    frames = {}
    for split, rows in partitions.items():
        frame = pd.DataFrame(rows)
        frame.insert(0, 'example_id', [r['dataset']+':'+str(r['row_index']) for r in rows])
        frames[split] = Rows(frame)
    return frames, identity


def transfer_rows(root):
    identities = l.data_files(root)  # complete metadata hashed as opaque bytes
    raw = pd.read_csv(c.safe_path(root, p.RAW), usecols=['example_id', 'statement', 'split', 'protocol', 'evaluation_phase'],
                      dtype=str, keep_default_na=False).assign(condition_id=p.RAW_ID)
    variants = pd.read_csv(c.safe_path(root, p.DATA+'/statements.csv'), usecols=p.STATEMENT_FIELDS,
                           dtype=str, keep_default_na=False)
    variants = variants[variants.condition_id.isin([p.BOTH, p.JUX, p.ISO])]
    frame = pd.concat([raw[p.STATEMENT_FIELDS], variants[p.STATEMENT_FIELDS]], ignore_index=True)
    c.require(frame.groupby('condition_id').size().to_dict() == l.COUNTS and
              set(frame.split) <= {'development', 'validation'} and set(frame.protocol) == {'entity_disjoint'} and
              set(frame.evaluation_phase) <= {'development', 'development_validation'}, 'transfer scope/counts')
    return Rows(frame), identities


def smoke_statements(stage, rows):
    if stage == 'transfer':
        return [text for _, group in rows.frame.groupby('condition_id')
                for text in [group.statement.iloc[0], group.statement.iloc[-1]]]
    return [rows.frame.statement.iloc[i] for i in [0, len(rows.frame)-1]]


def cache_path(root, stage):
    c.require(stage in ['train', 'validation', 'transfer'], 'TEST/unknown partition forbidden')
    return c.safe_path(root, l.TRANSFER if stage == 'transfer' else l.ATOMIC+'/'+stage)


def verify_cache(root, stage, rows, binding):
    path = cache_path(root, stage)
    files = {name: c.record(c.safe_path(root, str(path.relative_to(root)/name))) for name in l.CACHE_FILES}
    c.require(not (path/'activations.partial.npy').exists(), 'partial array')
    manifest, progress = [c.read_json(path/name) for name in l.CACHE_FILES[:2]]
    pin = l.read_pin(root)
    c.require(manifest['pin'] == pin and manifest['input_binding'] == binding and manifest['stage'] == stage and
              manifest['data'] == dict(**rows.hashes, metadata_columns=list(rows.frame),
                  expected_activation_shape=[len(rows.frame), l.LAYERS, l.WIDTH]), 'cache binding changed')
    gate = manifest['smoke_test']
    c.require(gate['passed'] is True and gate['policy'] == 'llama-unpadded-repeat-native-position-exact-v1' and
              gate['batch_size'] == 1 and gate['padding'] is False and
              gate['samples'] == len(smoke_statements(stage, rows)) and
              gate['statements_sha256'] == c.ordered_hash(smoke_statements(stage, rows)), 'smoke missing')
    runtime = manifest['runtime']
    c.require(runtime['torch'].split('+')[0] == '2.11.0' and runtime['transformers'] == '5.12.1' and
              bool(runtime['gpu']), 'runtime identity')
    if stage == 'transfer':
        c.require(manifest['atomic_replay']['passed'] is True and manifest['atomic_replay']['float16_exact'] is True,
                  'atomic replay missing')
    rows.verify_sidecar(path/'metadata.csv')
    array_header(path/'activations.npy', (len(rows.frame), l.LAYERS, l.WIDTH))
    # Same identity as ExtractionWriter, computed without loading activation values.
    identity = dict(manifest)
    identity.pop('smoke_test')
    identity['code'] = {k: manifest['code'][k] for k in ['implementation_version', 'source_sha256']}
    c.require(progress['complete'] is True and progress['next_row'] == len(rows.frame) and
              progress['activation_file_sha256'] == files['activations.npy']['sha256'] and
              progress['identity_sha256'] == c.digest(c.canonical(identity)), 'incomplete/mismatched cache')
    cursor = 0
    for chunk in progress['chunks']:
        stop = chunk['stop']
        c.require(chunk['start'] == cursor and cursor < stop <= len(rows.frame) and
                  chunk['ordered_example_id_sha256'] == c.ordered_hash(rows.frame.example_id.iloc[cursor:stop]) and
                  chunk['ordered_statement_sha256'] == c.ordered_hash(rows.frame.statement.iloc[cursor:stop]), 'coverage gap')
        cursor = stop
    c.require(cursor == len(rows.frame), 'incomplete coverage')
    return files


def run(stage, mode, root=c.ROOT):
    c.require(stage in ['atomic', 'transfer'] and mode in ['plan', 'smoke', 'extract'], 'stage/mode')
    pin = l.read_pin(root)
    if stage == 'atomic':
        groups, binding = atomic_rows(root)
    else:
        from src.llama_replication_probes import verify_probe
        selection, _, _, atomic_binding = verify_probe(root)
        rows, data = transfer_rows(root)
        groups = {'transfer': rows}
        binding = dict(data=data, atomic=atomic_binding, probe={name: c.record(c.safe_path(root, l.PROBE+'/'+name))
            for name in ['selection.json', 'selected_probe.npz', 'validation_metrics.csv']})
    plan = dict(shapes={key: [len(rows.frame), l.LAYERS, l.WIDTH] for key, rows in groups.items()},
                representation=pin, scores_computed=0, model_loaded=False)
    if mode == 'plan': return plan
    l.committed(root, l.PIN)
    if mode == 'extract':
        for key in groups: c.require(not cache_path(root, key).exists(), 'no overwrite/resume')
    model, tokenizer, runtime = load_model(pin)
    replay = None
    if stage == 'transfer':
        atomic_groups, _ = atomic_rows(root)
        for split, rows in atomic_groups.items():
            array = np.load(cache_path(root, split)/'activations.npy', mmap_mode='r', allow_pickle=False)
            for index in [0, len(rows.frame)-1]:
                fresh = readout(model, tokenizer, rows.frame.statement.iloc[index]).astype(np.float16)
                c.require(np.array_equal(fresh, array[index]), 'atomic float16 replay failed')
        replay = dict(passed=True, float16_exact=True, samples=4)
    for key, rows in groups.items():
        statements = smoke_statements(key, rows)
        gate = smoke(model, tokenizer, statements)
        manifest = dict(pin=pin, stage=key, input_binding=binding, runtime=runtime,
            data=dict(**rows.hashes, metadata_columns=list(rows.frame),
                      expected_activation_shape=[len(rows.frame), l.LAYERS, l.WIDTH]),
            code=dict(implementation_version='llama31-replication-v1', source_sha256=l.sources(root)),
            smoke_test=gate, atomic_replay=replay)
        if mode == 'extract':
            from src.clean_extraction import ExtractionWriter
            with ExtractionWriter(cache_path(root, key), rows, manifest, resume=False) as writer:
                for i, statement in enumerate(rows.frame.statement):
                    writer.append(i, rows.frame.iloc[i:i+1], readout(model, tokenizer, statement).astype(np.float16)[None])
                writer.finalize()
        else:
            c.publish_json(c.safe_path(root, l.OUTPUT+'/smoke_'+key+'.json'), manifest)
    return {**plan, 'model_loaded': True, 'extracted': mode == 'extract'}
