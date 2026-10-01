"""Read-only, inventory-mapped access to the historical T1 score package."""
from pathlib import Path
import json
import re

import numpy as np
import pandas as pd

from src import clean_transfer_contracts as c
from src import priority2_input_controls as p
from src.clean_transfer_evaluation import validate_metadata

# Established by the macOS restore check against the independently verified archive.
INVENTORY_SHA256 = '836aeb833ba32ea65a3068823971a66280b71bdd81e448492e5baaa410e5c744'
ARTIFACT_SOURCE = '/workspace/truth-probing-artifacts/'
BENCHMARK = p.RAW
SPECS = dict(qwen_lr='pinned_qwen25_lr_transfer_v1', methods='pinned_qwen25_method_transfer_v1',
             controls='priority2_input_controls_v1', llama='llama31_transfer_v1')
SCORE_DIRS = dict(qwen_lr='pinned_transfer_qwen25_v1/scoring',
                  methods='pinned_method_transfer_qwen25_v1',
                  controls='priority2_input_controls_v1/scoring',
                  llama='llama31_replication_v1/analysis/scores')
EVAL_DIRS = dict(qwen_lr='pinned_transfer_qwen25_v1/evaluation',
                 methods='pinned_method_transfer_qwen25_v1/evaluation',
                 controls='priority2_input_controls_v1/evaluation',
                 llama='llama31_replication_v1/analysis/evaluation')
COMPOUND_METADATA = 'pinned_compound_qwen2_5/qwen2_5_7b_a09a354_bs1_bf16_v1/metadata.csv'
DATA = 'priority2_input_controls_v1/data/'


def read_json(path):
    return json.loads(Path(path).read_text(), object_pairs_hook=c.no_duplicates)


def safe_file(root, name):
    c.require(isinstance(name, str) and name and not name.startswith('/') and
              '\\' not in name and all(x not in ('', '.', '..') for x in name.split('/')),
              'unsafe inventory path')
    path = root / name
    c.require(path.resolve() == path.absolute() and path.is_file(), 'missing/linked input: '+name)
    return path


def exact_join(metadata, scores, key='example_id'):
    """Reject extras and omissions before a one-to-one join, including NaN IDs."""
    for table in (metadata, scores):
        c.require(key in table and table[key].notna().all() and
                  table[key].astype(str).str.strip().ne('').all() and table[key].is_unique,
                  'invalid/duplicate join IDs')
    c.require(set(metadata[key]) == set(scores[key]), 'join ID coverage mismatch')
    c.require('frozen_probe_score' in scores and
              np.isfinite(scores.frozen_probe_score.to_numpy(float)).all(), 'nonfinite score')
    result = metadata.merge(scores, on=key, how='left', validate='one_to_one')
    c.require(len(result) == len(metadata), 'join cardinality changed')
    return result


class Package:
    def __init__(self, payload_root, repo_root):
        self.root, self.repo = Path(payload_root).resolve(), Path(repo_root).resolve()
        self.inputs, self.checks, self.gaps = [], [], []
        inventory = safe_file(self.root, 'inventory.json')
        c.require(c.file_hash(inventory) == INVENTORY_SHA256, 'untrusted/changed package inventory')
        self.inventory = read_json(inventory)
        self.inputs.append(dict(kind='inventory', local_path=str(inventory), **c.record(inventory)))
        self.by_source, self.by_hash, self.initial = {}, {}, {}
        for item in self.inventory:
            path = safe_file(self.root, item['archive_path'])
            identity = c.record(path)
            c.require(identity == {k: item[k] for k in ('bytes', 'sha256')}, 'package hash mismatch: '+item['archive_path'])
            c.require(item['archive_path'] not in self.initial, 'duplicate inventory path')
            c.require(item['original_path'] not in self.by_source, 'duplicate inventory source')
            self.initial[item['archive_path']] = identity
            self.by_source[item['original_path']] = path
            self.by_hash.setdefault(identity['sha256'], []).append(item['archive_path'])
            self.inputs.append(dict(kind='payload', local_path=str(path), **item))
        actual = set()
        for path in self.root.rglob('*'):
            c.require(not path.is_symlink(), 'linked payload entry')
            if path.is_file(): actual.add(path.relative_to(self.root).as_posix())
            else: c.require(path.is_dir(), 'special payload entry')
        c.require(actual == set(self.initial) | {'inventory.json'}, 'unexpected/missing payload files')
        self.checks.append(dict(check='package_inventory', payload_files=len(self.initial), status='pass'))

    def artifact(self, relative):
        # Source strings are dictionary keys only; never paths opened on the host.
        source = ARTIFACT_SOURCE + relative
        c.require(source in self.by_source, 'artifact not in inventory: '+relative)
        return self.by_source[source]

    def json(self, relative):
        return read_json(self.artifact(relative))

    def csv(self, relative, strings=False):
        return pd.read_csv(self.artifact(relative), keep_default_na=False,
                           **({'dtype': str} if strings else {'float_precision': 'round_trip'}))

    def checkout(self, relative, expected=None):
        path = safe_file(self.repo, relative)
        identity = c.record(path)
        if expected is not None:
            c.require(identity == expected if isinstance(expected, dict) else identity['sha256'] == expected,
                      'checkout identity mismatch: '+relative)
        self.inputs.append(dict(kind='checkout', local_path=str(path), logical_path=relative, **identity))
        return path

    def verify(self, relative, expected):
        actual = c.record(self.artifact(relative))
        c.require(actual == expected if isinstance(expected, dict) else actual['sha256'] == expected,
                  'recorded identity mismatch: '+relative)

    def reference_records(self, value, location):
        """Audit available upstream byte identities without opening excluded arrays."""
        if isinstance(value, dict):
            if set(value) == {'bytes', 'sha256'}:
                matches = [name for name in self.by_hash.get(value['sha256'], [])
                           if self.initial[name]['bytes'] == value['bytes']]
                entry = dict(reference=location, expected=value, matching_archive_paths=matches)
                if not matches:
                    self.gaps.append(dict(**entry, status='upstream_bytes_unavailable',
                        reason='excluded activation tensor' if 'activations.npy' in location else
                               'not present in supplied score package; not required to evaluate finalized scores'))
                else:
                    self.checks.append(dict(**entry, status='pass'))
            else:
                for key, child in value.items(): self.reference_records(child, location+'/'+key)
        elif isinstance(value, list):
            for i, child in enumerate(value): self.reference_records(child, location+'/'+str(i))

    def unchanged(self):
        for name, expected in self.initial.items():
            c.require(c.record(safe_file(self.root, name)) == expected, 'payload changed during run: '+name)
        c.require(c.file_hash(self.root/'inventory.json') == INVENTORY_SHA256, 'inventory changed during run')
        for entry in self.inputs:
            if entry['kind'] == 'checkout':
                c.require(c.record(Path(entry['local_path'])) == {k:entry[k] for k in ('bytes','sha256')},
                          'checkout input changed during run')


def ordered_ids(frame, manifest, field='example_id', hash_field='ordered_example_id_sha256'):
    c.require(frame[field].is_unique and c.ordered_hash(frame[field]) == manifest[hash_field],
              'ordered score identity mismatch: '+field)


def bind_constituents(frame, facts, mapping):
    c.require(mapping.base_example_id.is_unique and set(mapping.base_example_id) == set(frame.example_id),
              'constituent map coverage')
    c.require(facts.fact_key.is_unique, 'duplicate isolated fact')
    joined = frame.merge(mapping, left_on='example_id', right_on='base_example_id', validate='one_to_one')
    lookup = facts.set_index('fact_key')
    for side in ('a', 'b'):
        keys = joined[f'fact_{side}_key']
        c.require(set(keys) <= set(lookup.index), 'unknown constituent key')
        for target, source in [('fact_id', f'fact_{side}_id'), ('entity_id', f'entity_{side}_id'),
                               ('statement', f'fact_{side}_statement'), ('topic', 'topic')]:
            c.require(keys.map(lookup[target]).eq(joined[source]).all(), 'constituent identity mismatch: '+target)
        truth = keys.map(lookup.truth).astype(str).isin(['True','1'])
        c.require(truth.eq(joined[f'canonical_truth_{side}']).all(), 'constituent truth mismatch')


def load_inputs(package):
    """Verify score-package bindings; upstream fitting/extraction is not repeated."""
    specs, manifests, evaluations = {}, {}, {}
    for family, name in SPECS.items():
        spec_path = package.checkout('config/clean_protocol/'+name+'.json')
        spec = specs[family] = read_json(spec_path)
        manifest = manifests[family] = package.json(SCORE_DIRS[family]+'/scoring_manifest.json')
        evaluation = evaluations[family] = package.json(EVAL_DIRS[family]+'/evaluation_manifest.json')
        c.require(manifest['complete'] is True and evaluation['complete'] is True and
                  manifest['analysis_spec_sha256'] == evaluation['analysis_spec_sha256'] == c.file_hash(spec_path),
                  'analysis spec/manifest mismatch: '+family)
        c.require(manifest['fit_operations'] == 0 and manifest['compound_truth_columns_materialized'] is False,
                  'scoring scope mismatch')
        for field in ['test_accessed','test_artifacts_accessed','compound_labels_used']:
            if field in manifest: c.require(manifest[field] is False, 'forbidden scoring scope')
        for name, record in evaluation['outputs'].items(): package.verify(EVAL_DIRS[family]+'/'+name, record)
        policy = spec.get('policy', spec)
        c.require(policy['bootstrap'] == specs['qwen_lr']['bootstrap'] and evaluation['bootstrap'] == policy['bootstrap'],
                  'bootstrap specification mismatch')
        c.require(policy['benchmark'] == specs['qwen_lr']['benchmark'], 'benchmark policy mismatch')
        package.reference_records(spec.get('inputs', {}), family+'/inputs')
        package.checks.append(dict(check='frozen_spec_and_evaluation_files', family=family, status='pass'))
    lr, methods, controls, llama = [specs[k] for k in SPECS]
    c.require(controls['lr_spec'] == methods['lr_spec'] == lr, 'embedded LR spec mismatch')
    c.require(controls['lr_spec_sha256'] == methods['lr_spec_sha256'] == manifests['qwen_lr']['analysis_spec_sha256'],
              'embedded LR hash mismatch')
    for family in ('qwen_lr','controls'):
        spec, manifest = specs[family], manifests[family]
        c.require(manifest['inputs'] == spec['inputs'] and manifest['representation_fingerprint'] == c.FINGERPRINT and
                  manifest['selected_layer'] == spec['expected_selection']['layer'] and manifest['C'] == spec['expected_selection']['C'] and
                  manifest['selected_probe_sha256'] == lr['inputs']['probe']['files']['selected_probe.npz']['sha256'],
                  'probe/input binding mismatch')
    for group, prefix in [('atomic','atomic_repair/qwen25_a09a354_bs1_bf16_v1'),
                          ('probe','atomic_probe_selection/qwen25_7b'),
                          ('compound', COMPOUND_METADATA.rsplit('/',1)[0])]:
        for name, record in lr['inputs'][group]['files'].items():
            if name.endswith('activations.npy'): continue
            package.verify(prefix+'/'+name, record)
    metadata_record = lr['inputs']['compound']['files']['metadata.csv']
    path = package.checkout(BENCHMARK, metadata_record)
    package.verify(COMPOUND_METADATA, metadata_record)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    frame = validate_metadata(raw, lr['benchmark'])
    benchmark_meta = read_json(package.checkout(str(Path(BENCHMARK).parent/'metadata.json')))
    for name, sha in benchmark_meta['code_sha256'].items(): package.checkout(name, sha)
    c.require(benchmark_meta['final_test'] is False and benchmark_meta['test_candidate_queues_accessed'] == 0 and
              benchmark_meta['test_entities_in_output'] == benchmark_meta['test_examples_generated'] == 0 and
              benchmark_meta['evaluation_phase'] == 'development' and benchmark_meta['requested_split'] == 'validation',
              'benchmark is not development-only')
    package.checks.append(dict(check='recovered_benchmark', rows=len(frame), pairs=frame.pair_id.nunique(),
        sha256=metadata_record['sha256'], producer_hashes=len(benchmark_meta['code_sha256']), status='pass'))

    lm = manifests['qwen_lr']
    for name, key in [('row_scores.csv','row_scores'),('representation_binding.json','representation_binding')]:
        package.verify(SCORE_DIRS['qwen_lr']+'/'+name, lm[key])
    qwen = package.csv(SCORE_DIRS['qwen_lr']+'/row_scores.csv')
    c.require(list(qwen) == lm['score_columns'] == ['example_id','frozen_probe_score'] and len(qwen) == lm['rows'], 'LR schema/count')
    ordered_ids(qwen,lm)
    c.require(qwen.example_id.tolist() == raw.example_id.tolist(), 'LR benchmark order')
    qwen_frame = exact_join(frame,qwen).sort_values('example_id').reset_index(drop=True)
    mm = manifests['methods']
    for name,key in [('method_scores.csv','method_scores'),('method_binding.json','method_binding')]:
        package.verify(SCORE_DIRS['methods']+'/'+name,mm[key])
    binding = package.json(SCORE_DIRS['methods']+'/method_binding.json')
    c.require(mm['compound_inputs'] == lr['inputs']['compound'] and mm['scoring'] == methods['scoring'] and
              mm['representation_fingerprint'] == binding['representation_fingerprint'] == c.FINGERPRINT and
              binding['conditions'] == mm['conditions'] == methods['conditions'] and
              binding['suites'] == mm['suites'] == methods['suites'], 'method manifest/binding mismatch')
    package.reference_records(methods['suites'],'methods/suites')
    method_scores = package.csv(SCORE_DIRS['methods']+'/method_scores.csv')
    c.require(list(method_scores) == mm['score_columns'] and len(method_scores) == mm['rows'] and mm['examples'] == len(frame), 'method schema/count')
    c.require(not method_scores.duplicated(['analysis_group','method','example_id']).any(), 'duplicate method score')
    c.require(set(zip(method_scores.analysis_group,method_scores.method)) ==
              {(x['analysis_group'],x['method']) for x in methods['conditions']}, 'method condition coverage')
    for condition in methods['conditions']:
        rows = method_scores[(method_scores.analysis_group == condition['analysis_group']) & (method_scores.method == condition['method'])]
        ordered_ids(rows,mm)
        c.require(rows.example_id.tolist() == raw.example_id.tolist() and rows.atomic_layer.eq(condition['atomic_layer']).all(), 'method ID/layer mismatch')
        exact_join(frame,rows[['example_id','frozen_probe_score']])
    common = method_scores[(method_scores.analysis_group == 'faithful_common_layer') & (method_scores.method == 'l2_logistic')]
    c.require(np.allclose(common.frozen_probe_score,qwen.frozen_probe_score,rtol=0,atol=methods['reproduction']['score_atol']), 'LR method score discrepancy')

    cm = manifests['controls']
    for name,record in cm['outputs'].items(): package.verify(SCORE_DIRS['controls']+'/'+name,record)
    generation = package.json(DATA+'generation_manifest.json')
    c.require(generation['complete'] is True and generation['test_accessed'] is False and generation['resampled'] is False and
              generation['raw_benchmark'] == metadata_record and generation['templates'] == controls['templates'] and
              generation['condition_counts'] == controls['condition_counts'] == p.condition_counts(), 'control generation binding')
    c.require(cm['isolated_scoring_source'] == generation['isolated_scoring_source'] == controls['isolated_scoring_source'] == 'fresh_priority2_extraction', 'isolated provenance')
    for name,record in controls['inputs']['generation']['files'].items(): package.verify(DATA+name,record)
    for name,record in generation['outputs'].items(): package.verify(DATA+name,record)
    # Regenerate exact same-fact identities, labels, wording and mappings in memory.
    tables, facts, mapping = p.variants(raw)
    for name,table in {**{k+'.csv':v for k,v in tables.items()},'isolated_facts.csv':facts,'constituent_map.csv':mapping}.items():
        payload = p.csv_bytes(table)
        c.require(dict(bytes=len(payload),sha256=c.digest(payload)) == generation['outputs'][name], 'regenerated control differs: '+name)
    bind_constituents(frame,facts,mapping)
    scores,isolated = [package.csv(SCORE_DIRS['controls']+'/'+n) for n in ('condition_scores.csv','isolated_scores.csv')]
    c.require(list(scores) == cm['score_columns'] and list(isolated) == cm['isolated_columns'] and
              len(scores) == cm['condition_rows'] and len(isolated) == cm['isolated_rows'], 'control score schema/count')
    index = package.csv(DATA+'scoring_index.csv',True)
    c.require(scores.iloc[:,:3].equals(index), 'condition index mismatch')
    ordered_ids(scores,cm); ordered_ids(isolated,cm,'fact_key','ordered_fact_key_sha256')
    for key,table in tables.items():
        part = scores[scores.condition_id == key]
        exact_join(table,part[['example_id','frozen_probe_score']])
        c.require(table.set_index('example_id').base_example_id.to_dict() == part.set_index('example_id').base_example_id.to_dict(), 'control base join mismatch')
    exact_join(facts,isolated,'fact_key')
    sources = package.csv(DATA+'isolated_sources.csv',True)
    c.require(sources.fact_key.tolist() == isolated.fact_key.tolist() and sources.example_id.equals(sources.fact_key) and
              sources.source_kind.eq('fresh_extraction').all() and sources.cache_split.eq('').all() and sources.cache_row_index.eq('').all(), 'isolated source mapping')

    llm = manifests['llama']
    package.verify(SCORE_DIRS['llama']+'/scores.csv',llm['score_file'])
    llama_scores = package.csv(SCORE_DIRS['llama']+'/scores.csv')
    c.require(list(llama_scores) == llm['columns'] and len(llama_scores) == llm['rows'] and
              llm['selected'] == llama['selected'] and llm['representation_fingerprint'] == llama['pin']['fingerprint'], 'Llama manifest binding')
    ordered_ids(llama_scores,llm)
    c.require(llm['ordered_example_id_sha256'] == llama['ordered_example_id_sha256'], 'Llama frozen identities')
    statements = package.csv(DATA+'statements.csv',True)
    expected = pd.concat([raw[['example_id']].assign(condition_id=p.RAW_ID),
        statements[statements.condition_id.isin([p.BOTH,p.JUX,p.ISO])][['example_id','condition_id']]],ignore_index=True)
    c.require(llama_scores[['example_id','condition_id']].equals(expected) and
              llama_scores.groupby('condition_id').size().to_dict() == llama['policy']['conditions'], 'Llama exact condition coverage')
    for condition,metadata in [(p.RAW_ID,frame),(p.BOTH,tables[p.BOTH]),(p.JUX,tables[p.JUX]),(p.ISO,facts.rename(columns={'fact_key':'example_id'}))]:
        exact_join(metadata,llama_scores[llama_scores.condition_id == condition][['example_id','frozen_probe_score']])
    llama_frame = exact_join(frame,llama_scores[llama_scores.condition_id == p.RAW_ID][['example_id','frozen_probe_score']]).sort_values('example_id').reset_index(drop=True)
    # Verify all available frozen LR references at their relocated locations.
    for refs in (methods['lr_reference']['files'],controls['inputs']['lr_reference']['files']):
        for name,record in refs.items():
            relative = 'pinned_transfer_qwen25_v1/'+(name if name.startswith('evaluation/') else 'scoring/'+name)
            package.verify(relative,record)
    for family in SPECS:
        package.checks.append(dict(check='score_hash_manifest_ids_labels_joins',family=family,status='pass'))
    return dict(specs=specs,manifests=manifests,evaluations=evaluations,frame=frame.sort_values('example_id').reset_index(drop=True),qwen=qwen_frame,
                methods=method_scores,tables=tables,facts=facts,mapping=mapping,controls=scores,
                isolated=isolated,llama_scores=llama_scores,llama=llama_frame)
