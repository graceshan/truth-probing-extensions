"""Fixed cross-family replication policy. No production identities are guessed."""
from pathlib import Path
import re
import subprocess

from src import clean_transfer_contracts as c
from src import priority2_input_controls as p

MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
LAYERS, WIDTH = 32, 4096
PIN = 'config/clean_protocol/llama31_representation_v1.json'
SPEC = 'config/clean_protocol/llama31_transfer_v1.json'
ATOMIC = 'acts/clean_protocol/atomic/llama31_8b_instruct_pinned_v1'
TRANSFER = 'acts/clean_protocol/llama31_transfer_v1'
PROBE = 'results/clean_protocol/atomic_probes_pinned_v1/llama31_8b_instruct'
OUTPUT = 'results/clean_protocol/llama31_transfer_v1'
COUNTS = {p.RAW_ID: 8384, p.BOTH: 4192, p.JUX: 4192, p.ISO: 524}
SCORE_COLUMNS = ['example_id', 'condition_id', 'frozen_probe_score']
CACHE_FILES = ['extraction_manifest.json', 'progress.json', 'metadata.csv', 'activations.npy']
P2_SPEC_SHA = 'd72b0e3a8853ee5adf5bb205e12a40cb647fae175d965010b29407c0209e9c42'
SOURCES = ['src/llama_replication_contracts.py', 'src/llama_replication_extraction.py',
           'src/llama_replication_probes.py', 'src/llama_replication_transfer.py',
           'src/llama_replication_evaluation.py', 'src/extract.py', 'src/clean_extraction.py',
           'src/clean_atomic_extraction.py', 'src/clean_atomic_probes.py',
           'src/clean_transfer_contracts.py', 'src/clean_transfer_statistics.py',
           'src/clean_transfer_evaluation.py', 'src/priority2_evaluation.py',
           'src/priority2_input_controls.py', 'src/clean_compounds.py',
           'src/repaired_atomic_cache.py', 'src/entity_partitions.py',
           'scripts/45_llama_replication.py']


def representation(revision):
    c.require(isinstance(revision, str) and re.fullmatch('[0-9a-f]{40}', revision), 'immutable HF revision required')
    return dict(model=MODEL, model_revision=revision, tokenizer_revision=revision,
        architecture='LlamaForCausalLM', model_type='llama', num_hidden_layers=LAYERS, hidden_size=WIDTH,
        input='exact raw statement', add_special_tokens=True, chat_template=False,
        truncation=False, padding=False, batch_size=1, compute_dtype='bfloat16', saved_dtype='float16',
        attention='sdpa', position_ids='explicit arange(0, sequence_length); native default equivalence gated',
        use_cache=False, hidden_states='HF hidden_states[1:]; embedding excluded',
        readout='last real token', final_saved_layer='post-final-RMSNorm',
        saved_layer_to_hf_index=list(range(1, LAYERS+1)), torch='2.11.0', transformers='5.12.1')


def validate_pin(pin):
    c.require(set(pin) == {'schema_version', 'representation', 'fingerprint', 'files', 'config',
                          'tokenizer_backend_sha256', 'tokenizer_class', 'special_tokens'}, 'pin schema')
    r = pin['representation']
    c.require(pin['schema_version'] == 1 and r == representation(r['model_revision']) and
              pin['fingerprint'] == c.digest(c.canonical(r)), 'representation pin mismatch')
    discover(pin['config'])
    c.require(set(pin['files']) == {'config.json', 'tokenizer_config.json', 'tokenizer.json'}, 'pin file allowlist')
    for record in pin['files'].values(): validate_record(record)
    c.require(re.fullmatch('[0-9a-f]{64}', pin['tokenizer_backend_sha256']) and
              isinstance(pin['tokenizer_class'], str) and pin['tokenizer_class'] and
              set(pin['special_tokens']) == {'bos_token_id', 'eos_token_id', 'add_bos_token', 'add_eos_token'}, 'tokenizer pin')
    return pin


def discover(config):
    c.require(config.get('model_type') == 'llama' and config.get('architectures') == ['LlamaForCausalLM'], 'Llama architecture required')
    dims = config.get('num_hidden_layers'), config.get('hidden_size')
    c.require(dims == (LAYERS, WIDTH), 'unexpected Llama layer count/hidden width')
    return dims


def validate_record(record):
    c.require(set(record) == {'sha256', 'bytes'} and isinstance(record['sha256'], str) and
              re.fullmatch('[0-9a-f]{64}', record['sha256']) and type(record['bytes']) is int and
              record['bytes'] > 0, 'invalid measured file identity')


def read_pin(root):
    return validate_pin(c.read_json(c.safe_path(root, PIN)))


def committed(root, name):
    path = c.safe_path(root, name)
    result = subprocess.run(['git', 'show', 'HEAD:'+name], cwd=root, capture_output=True)
    c.require(result.returncode == 0 and result.stdout == path.read_bytes(), 'freeze and commit '+name+' first')


def sources(root):
    return {name: c.file_hash(c.safe_path(root, name)) for name in SOURCES}


def references(root):
    """Only frozen policy/identities, never Qwen scores or metric outputs."""
    c.require(c.file_hash(c.safe_path(root, p.SPEC)) == P2_SPEC_SHA, 'frozen Qwen policy identity')
    spec = c.read_json(c.safe_path(root, p.SPEC))
    c.require(spec['schema_version'] == 1 and spec['condition_counts'] == p.condition_counts() and
              spec['isolated_scoring_source'] == 'fresh_priority2_extraction', 'canonical Priority-2 policy required')
    return spec


def data_files(root):
    ref = references(root)
    files = {p.RAW: ref['lr_spec']['inputs']['compound']['files']['metadata.csv']}
    for name in ['statements.csv', 'scoring_index.csv', 'constituent_map.csv', 'isolated_facts.csv',
                 p.BOTH+'.csv', p.JUX+'.csv']:
        files[p.DATA+'/'+name] = ref['inputs']['generation']['files'][name]
    for name, identity in files.items():
        c.require(c.record(c.safe_path(root, name)) == identity, 'canonical same-fact bytes changed: '+name)
    return files


EVALUATION_FILES = [name+extension for name in ['primary_metrics', 'boundary_metrics', 'boolean_metrics',
    'juxtaposition_geometry', 'paired_contrasts'] for extension in ['.csv', '.json']] + [
    'bootstrap_summary.csv', 'bootstrap_draws.npz', 'bootstrap_pair_weights.npz', 'evaluation_manifest.json']


def policy(root):
    ref = references(root)
    return dict(benchmark=ref['benchmark'], bootstrap=ref['bootstrap'], statistics=ref['statistics'],
        formal_metrics=ref['formal_metrics'], boolean_metrics=ref['boolean_metrics'],
        juxtaposition_metrics=ref['juxtaposition_metrics'], composition=ref['composition'],
        paired_contrasts=[x for x in ref['paired_contrasts'] if x['left'] in [p.BOTH, p.EXTERNAL]],
        schedule_sha256=ref['inputs']['lr_reference']['schedule_sha256'],
        conditions=COUNTS, scoring=dict(batch_size=256, dtype='float64', blas_threads=1,
        operation='affine', preprocessing='none', metadata_projection=p.STATEMENT_FIELDS),
        prohibitions=ref['prohibitions'], outputs=dict(root=OUTPUT, score_columns=SCORE_COLUMNS,
            score_files=['scores/scores.csv', 'scores/scoring_manifest.json'],
            evaluation_directory='evaluation', evaluation_files=EVALUATION_FILES))
