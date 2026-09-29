"""Deterministic same-fact input controls and exact isolated-fact coverage audit."""
import pandas as pd

from src import clean_transfer_contracts as c
from src.clean_transfer_evaluation import validate_metadata
from src.clean_compounds import render_binary
from src.repaired_atomic_cache import RepairedAtomicCache

VERSION = 'priority2-input-controls-v1'
RAW = 'data/clean_protocol/compounds/entity_disjoint/development_validation_v1/development_validation_compounds.csv'
DATA = 'data/clean_protocol/priority2_input_controls_v1'
ACTS = 'acts/clean_protocol/priority2_input_controls_v1/qwen25_a09a354_bs1_bf16_v1'
OUTPUT = 'results/clean_protocol/priority2_input_controls_v1'
SPEC = 'config/clean_protocol/priority2_input_controls_v1.json'
PREFLIGHT = 'results/clean_protocol/priority2_input_controls_preflight_v1.json'
RAW_ID = 'raw_reference'
BOTH = 'or_explicit_or_both_v1'
LEAST = 'or_at_least_one_v1'
JUX = 'juxtaposition_v1'
ISO = 'isolated_constituents_v1'
EXTERNAL = 'isolated_external_minmax_v1'
BOOLEAN = 'isolated_external_boolean_v1'
TEXT_CONDITIONS = [BOTH, LEAST, JUX]
TEMPLATES = {BOTH: 'render_binary(first, second, OR).removesuffix(".") + ", or both."',
             LEAST: '"At least one of the following is true: " + first + " " + second',
             JUX: 'first + " " + second', ISO: 'exact isolated fact statement, unchanged'}
IDENTITY = ['pair_id', 'topic', 'split', 'protocol', 'evaluation_phase', 'canonical_truth_a', 'canonical_truth_b',
            'ordering', 'entity_a_id', 'entity_b_id', 'fact_a_id', 'fact_b_id', 'fact_a_statement', 'fact_b_statement']
STATEMENT_FIELDS = ['example_id', 'condition_id', 'statement', 'split', 'protocol', 'evaluation_phase']
SOURCE_FIELDS = ['fact_key', 'source_kind', 'cache_split', 'cache_row_index', 'example_id']
FILES = [BOTH+'.csv', LEAST+'.csv', JUX+'.csv', 'isolated_facts.csv', 'isolated_sources.csv',
         'constituent_map.csv', 'statements.csv', 'scoring_index.csv', 'coverage.json', 'generation_manifest.json']


def identifier(kind, value):
    return kind + '_' + c.digest(c.canonical(value))


def csv_bytes(frame):
    return frame.to_csv(index=False, lineterminator='\n').encode('utf-8')


def validate_base(raw):
    frame = validate_metadata(raw, c.BENCHMARK)
    c.require(set(IDENTITY) <= set(frame), 'missing exact constituent identity fields')
    for side in ['a', 'b']:
        values = frame[f'fact_{side}_statement']
        c.require(values.map(lambda x: isinstance(x, str) and x.endswith('.') and x.strip() == x).all(),
                  'complete unmodified fact sentences required')
    inventory, mapping = [], []
    for row in frame.to_dict('records'):
        keys = []
        for side in ['a', 'b']:
            fact = dict(fact_id=row[f'fact_{side}_id'], entity_id=row[f'entity_{side}_id'], topic=row['topic'],
                        statement=row[f'fact_{side}_statement'], truth=bool(row[f'canonical_truth_{side}']),
                        split=row['split'], protocol=row['protocol'], evaluation_phase=row['evaluation_phase'])
            key = identifier('fact', fact)
            inventory.append(dict(fact_key=key, **fact))
            keys.append(key)
        mapping.append(dict(base_example_id=row['example_id'], fact_a_key=keys[0], fact_b_key=keys[1]))
        first, second = ((row['fact_a_statement'], row['fact_b_statement']) if row['ordering'] == 'AB' else
                         (row['fact_b_statement'], row['fact_a_statement']))
        c.require(row['statement'] == render_binary(first, second, row['operator']), 'raw wording/fact identity mismatch')
    facts = pd.DataFrame(inventory).drop_duplicates().sort_values('fact_key').reset_index(drop=True)
    c.require(facts.fact_id.is_unique and facts.fact_key.is_unique and len(facts) == 2*c.BENCHMARK['entities'],
              'fact identity is not one-to-one with entity/truth')
    grouped = frame.groupby(['pair_id', 'canonical_truth_a', 'canonical_truth_b', 'ordering'])
    for _, part in grouped:
        c.require(len(part) == 2 and set(part.operator) == {'AND', 'OR'} and
                  all(part[k].nunique() == 1 for k in IDENTITY), 'AND/OR constituent identity mismatch')
    return frame, facts, pd.DataFrame(mapping)


def variants(raw):
    frame, facts, mapping = validate_base(raw)
    tables = {k: [] for k in TEXT_CONDITIONS}
    for row in frame[frame.operator == 'OR'].sort_values('example_id').to_dict('records'):
        paired = frame[(frame.pair_id == row['pair_id']) & (frame.ordering == row['ordering']) &
                       (frame.canonical_truth_a == row['canonical_truth_a']) & (frame.canonical_truth_b == row['canonical_truth_b'])]
        raw_and = paired.loc[paired.operator == 'AND', 'example_id'].item()
        base = {k: row[k] for k in IDENTITY}
        tuple_id = identifier('pair_truth_order', base)
        first, second = (row['fact_a_statement'], row['fact_b_statement']) if row['ordering'] == 'AB' else (row['fact_b_statement'], row['fact_a_statement'])
        text = {BOTH: render_binary(first, second, 'OR').removesuffix('.') + ', or both.',
                LEAST: 'At least one of the following is true: ' + first + ' ' + second,
                JUX: first + ' ' + second}
        for condition in TEXT_CONDITIONS:
            record = dict(example_id=identifier('input', [condition, tuple_id, text[condition]]), condition_id=condition,
                          template_id=condition, base_example_id=tuple_id if condition == JUX else row['example_id'],
                          base_tuple_id=tuple_id, statement=text[condition], **base)
            if condition == JUX:
                record.update(raw_and_example_id=raw_and, raw_or_example_id=row['example_id'])
            else:
                record.update(operator='OR', compound_label=bool(row['compound_label']))
            tables[condition].append(record)
    tables = {k: pd.DataFrame(rows).sort_values('example_id').reset_index(drop=True) for k, rows in tables.items()}
    for table in tables.values():
        c.require(len(table) == c.ROWS//2 and table.base_example_id.is_unique and table.example_id.is_unique, 'condition coverage mismatch')
    return tables, facts, mapping


def coverage(facts, cache):
    lookup = {}
    for split in ['train', 'validation']:
        for index, row in enumerate(cache.rows[split]):
            key = (row['statement'], row['entity_id'], row['topic'], bool(row['label']))
            lookup.setdefault(key, []).append((split, index, row))
    sources, details = [], []
    for row in facts.to_dict('records'):
        key = (row['statement'], row['entity_id'], row['topic'], row['truth'])
        matches = lookup.get(key, [])
        c.require(len(matches) <= 1, 'ambiguous exact isolated-fact match; no arbitrary cache row selection')
        if matches:
            split, index, atom = matches[0]
            source = dict(fact_key=row['fact_key'], source_kind='repaired', cache_split=split, cache_row_index=index, example_id='')
            detail = dict(source, fact_id=row['fact_id'], entity_id=row['entity_id'], statement=row['statement'],
                          dataset=atom['dataset'], row_index=atom['row_index'])
        else:
            source = dict(fact_key=row['fact_key'], source_kind='extract', cache_split='', cache_row_index='', example_id=row['fact_key'])
            detail = dict(source, fact_id=row['fact_id'], entity_id=row['entity_id'], statement=row['statement'])
        sources.append(source)
        details.append(detail)
    sources = pd.DataFrame(sources, columns=SOURCE_FIELDS)
    missing = facts[facts.fact_key.isin(sources.loc[sources.source_kind == 'extract', 'fact_key'])]
    report = dict(schema_version=1, exact_match_fields=['statement', 'entity_id', 'topic', 'truth'],
        provenance_bridge='canonical fact_id -> exact tuple -> repaired dataset/row_index and cache split/index',
        required=len(facts), covered=len(facts)-len(missing), missing=len(missing),
        extraction_required=bool(len(missing)), identities=details, atomic_test_accessed=False)
    return sources, missing, report


def build(raw, cache):
    tables, facts, mapping = variants(raw)
    sources, missing, audit = coverage(facts, cache)
    statements = [tables[k][STATEMENT_FIELDS] for k in TEXT_CONDITIONS]
    extra = missing.rename(columns={'fact_key': 'example_id'}).assign(condition_id=ISO)
    statements.append(extra[STATEMENT_FIELDS])
    extraction = pd.concat(statements, ignore_index=True).sort_values(['condition_id', 'example_id']).reset_index(drop=True)
    index = pd.concat([tables[k][['example_id', 'condition_id', 'base_example_id']] for k in TEXT_CONDITIONS], ignore_index=True)
    return {**{k+'.csv': csv_bytes(v) for k, v in tables.items()}, 'isolated_facts.csv': csv_bytes(facts),
        'isolated_sources.csv': csv_bytes(sources), 'constituent_map.csv': csv_bytes(mapping),
        'statements.csv': csv_bytes(extraction), 'scoring_index.csv': csv_bytes(index), 'coverage.json': c.canonical(audit)+b'\n'}, audit


def generate(root=c.ROOT, *, audit_only=False):
    path = c.safe_path(root, RAW)
    c.require(c.file_hash(path) == c.METADATA_SHA, 'canonical raw benchmark hash mismatch')
    cache = RepairedAtomicCache(root)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    payloads, audit = build(raw, cache)
    if audit_only:
        return audit
    output = c.safe_path(root, DATA)
    c.require(not output.exists(), 'refusing existing generation output')
    output.mkdir(parents=True)
    for name, payload in payloads.items():
        with (output / name).open('xb') as handle: handle.write(payload)
    c.publish_json(output / 'generation_manifest.json', dict(complete=True, version=VERSION,
        raw_benchmark=c.record(path), repaired_cache_files=cache.files, templates=TEMPLATES,
        condition_counts={k: c.ROWS//2 for k in TEXT_CONDITIONS}, isolated_facts=audit['required'], missing_facts=audit['missing'],
        test_accessed=False, resampled=False,
        source_sha256={name:c.file_hash(c.ROOT/name) for name in ['src/priority2_input_controls.py','src/clean_compounds.py']},
        outputs={n: c.record(output / n) for n in payloads}))
    return dict(output=DATA, coverage=audit, generated=True)


def verify_generation(root, cache=None):
    paths = {n: c.safe_path(root, DATA+'/'+n) for n in FILES}
    manifest = c.read_json(paths['generation_manifest.json'])
    c.require(manifest['complete'] is True and manifest['version'] == VERSION and manifest['templates'] == TEMPLATES and
              manifest['raw_benchmark']['sha256'] == c.METADATA_SHA and manifest['test_accessed'] is False and
              manifest['resampled'] is False and manifest['condition_counts'] == {k: c.ROWS//2 for k in TEXT_CONDITIONS}, 'generation identity mismatch')
    c.require(set(manifest['outputs']) == set(FILES)-{'generation_manifest.json'}, 'generation file allowlist')
    files = {n: c.record(p) for n, p in paths.items()}
    c.require(all(files[n] == r for n, r in manifest['outputs'].items()), 'generation hash mismatch')
    if cache is not None: c.require(manifest['repaired_cache_files'] == cache.files, 'generation/cache identity mismatch')
    return manifest, files
