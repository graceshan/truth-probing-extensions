"""Frozen Priority-2 specification and truth-blind frozen-LR scoring."""
import subprocess
import csv
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import clean_transfer_contracts as c
from src import method_transfer_contracts as reference
from src import priority2_input_controls as p
from src import priority2_extraction as extraction
from src.repaired_atomic_cache import RepairedAtomicCache
from src.pinned_compound_scoring import verify_probe, affine
from src.pinned_method_binding import lr_reference

SCORE_COLUMNS = ['example_id', 'condition_id', 'base_example_id', 'frozen_probe_score']
ISOLATED_COLUMNS = ['fact_key', 'frozen_probe_score']
SOURCES = ['src/priority2_input_controls.py', 'src/priority2_extraction.py', 'src/priority2_transfer.py',
           'src/priority2_evaluation.py', 'src/pinned_compound_scoring.py', 'src/repaired_atomic_cache.py',
           'src/clean_transfer_statistics.py', 'src/clean_transfer_evaluation.py', 'src/clean_compounds.py']


def contrasts():
    or_metrics = ['or_auroc', 'and_minus_or_auroc', 'or_mixed_vs_ff_auroc', 'or_tf_vs_ff_auroc', 'or_ft_vs_ff_auroc', 'or_tt_vs_ff_auroc']
    primary = [d['id'] for d in c.metric_definitions() if d['category'] == 'primary']
    return [dict(left=k, right=p.RAW_ID, metrics=or_metrics) for k in [p.BOTH, p.LEAST]] + [dict(left=p.EXTERNAL, right=p.RAW_ID, metrics=primary)]


def template(lr_spec, generation, cache, lr_ref, counts):
    return dict(schema_version=1, analysis_id=p.VERSION, lr_spec=lr_spec, lr_spec_sha256=reference.LR_SPEC_SHA,
        inputs=dict(generation=dict(directory=p.DATA, files=generation), extraction=dict(directory=p.ACTS, files=cache), lr_reference=lr_ref),
        representation_fingerprint=c.FINGERPRINT, templates=p.TEMPLATES, condition_counts=counts,
        isolated_scoring_source=p.ISOLATED_SCORING_SOURCE,
        scoring=lr_spec['scoring'], expected_selection=lr_spec['expected_selection'], bootstrap=lr_spec['bootstrap'],
        statistics=lr_spec['statistics'], benchmark=lr_spec['benchmark'],
        formal_metrics=[d for d in lr_spec['metrics'] if d['category'] in ['primary', 'boundary']],
        boolean_metrics=[d for d in lr_spec['metrics'] if d['category'] == 'threshold'],
        juxtaposition_metrics=[dict(id='juxtaposition_'+name+'_geometry_auroc', positive=pos, negative=neg, category='geometry')
            for name,pos,neg in [('tt_vs_mixed',['TT'],['TF','FT']), ('mixed_vs_ff',['TF','FT'],['FF']),
                                 ('tt_vs_ff',['TT'],['FF']), ('tf_vs_ft',['TF'],['FT'])]],
        paired_contrasts=contrasts(), composition=dict(continuous_and='min(s_A,s_B)', continuous_or='max(s_A,s_B)',
            boolean_and='(s_A>=0) and (s_B>=0)', boolean_or='(s_A>=0) or (s_B>=0)',
            interpretation='two isolated model evaluations plus a known parse/external composition rule', boolean_auroc_reported=False),
        prohibitions={**lr_spec['prohibitions'], 'template_selection': True, 'new_entity_split': True,
                      'juxtaposition_formal_truth_label': True, 'compound_normalization': True},
        outputs=dict(root=p.OUTPUT, score_columns=SCORE_COLUMNS, isolated_columns=ISOLATED_COLUMNS,
            score_files=['condition_scores.csv','isolated_scores.csv','scoring_manifest.json'],
            evaluation_files=['formal_metrics.csv','formal_metrics.json','paired_contrasts.csv','boolean_metrics.csv',
                'juxtaposition_geometry.csv','juxtaposition_cell_statistics.csv','bootstrap_summary.csv',
                'bootstrap_draws.npz','bootstrap_pair_weights.npz','evaluation_manifest.json']))


def validate(spec):
    lr = c.validate_spec(spec['lr_spec'])
    entries = spec['inputs']
    c.require(set(entries) == {'generation','extraction','lr_reference'}, 'input schema')
    for key,directory,names in [('generation',p.DATA,p.FILES),('extraction',p.ACTS,extraction.EXTRACTION_FILES)]:
        c.require(set(entries[key]) == {'directory','files'} and entries[key]['directory'] == directory, 'input path schema')
        reference.records(entries[key]['files'], names)
    ref = entries['lr_reference']
    reference.records(ref['files'], reference.LR_FILES)
    c.require(set(ref) == {'directory','files','schedule_sha256'} and ref['directory'] == c.OUTPUT and
              ref['files']['evaluation/evaluation_manifest.json']['sha256'] == reference.LR_EVALUATION_SHA and
              ref['files']['evaluation/primary_metrics.csv']['sha256'] == reference.LR_PRIMARY_SHA, 'LR reference identity')
    counts = spec['condition_counts']
    c.require(counts == p.condition_counts() and all(type(v) is int for v in counts.values()), 'condition counts')
    c.require(c.canonical(spec) == c.canonical(template(lr, entries['generation']['files'], entries['extraction']['files'], ref, counts)),
              'unknown keys or altered frozen policy')
    return spec


def inspect(root):
    c.require(c.file_hash(c.safe_path(root,c.SPEC)) == reference.LR_SPEC_SHA, 'canonical LR spec changed')
    lr = c.validate_spec(c.read_json(c.safe_path(root,c.SPEC)))
    cache = RepairedAtomicCache(root)
    probe = verify_probe(root,cache)
    c.require(cache.files == lr['inputs']['atomic']['files'] and probe['files'] == lr['inputs']['probe']['files'], 'canonical atomic/probe bytes differ')
    generated, files = p.verify_generation(root,cache)
    acts, statements = extraction.verify(root,cache,files)
    ref, _ = lr_reference(root,lr)
    counts = generated['condition_counts']
    spec = validate(template(lr,files,acts,ref,counts))
    # These generated tables contain identities/pointers only, never truth columns.
    def identity_table(name, columns):
        path = c.safe_path(root,p.DATA+'/'+name)
        with path.open(encoding='utf-8',newline='') as handle:
            c.require(next(csv.reader(handle)) == columns, 'truth-free identity table header required')
        return pd.read_csv(path,usecols=columns,dtype=str,keep_default_na=False)
    index = identity_table('scoring_index.csv',['example_id','condition_id','base_example_id'])
    sources = identity_table('isolated_sources.csv',p.SOURCE_FIELDS)
    facts = pd.read_csv(c.safe_path(root,p.DATA+'/isolated_facts.csv'),
        usecols=['fact_key','fact_id','entity_id','topic','statement'],dtype=str,keep_default_na=False).set_index('fact_key')
    c.require(facts.index.is_unique and facts.statement.is_unique and set(facts.index) == set(sources.fact_key), 'isolated fact identity coverage')
    c.require(list(index) == ['example_id','condition_id','base_example_id'] and index.example_id.is_unique and
              index.groupby('condition_id').size().to_dict() == {k:c.ROWS//2 for k in p.TEXT_CONDITIONS}, 'scoring index invalid')
    validate_isolated_sources(sources)
    for row in sources.itertuples():
        selected = statements[(statements.condition_id == p.ISO) & (statements.example_id == row.example_id)]
        c.require(len(selected) == 1 and selected.statement.iloc[0] == facts.loc[row.fact_key,'statement'], 'isolated extraction statement mismatch')
    c.require(set(index.example_id) == set(statements.loc[statements.condition_id != p.ISO,'example_id']) and
              set(sources.example_id) == set(statements.loc[statements.condition_id == p.ISO,'example_id']), 'extraction identity coverage')
    joined = index.merge(statements[['example_id','condition_id']],on='example_id',validate='one_to_one',suffixes=('_index','_extraction'))
    c.require(joined.condition_id_index.eq(joined.condition_id_extraction).all(), 'condition ID mapping mismatch')
    return spec,probe,statements,index,sources


def recheck(root,spec):
    for entry in [*spec['lr_spec']['inputs'].values(), *spec['inputs'].values()]:
        for name,expected in entry['files'].items():
            c.require(c.record(c.safe_path(root,entry['directory']+'/'+name)) == expected, 'input changed')
    c.require(c.file_hash(c.safe_path(root,c.SPEC)) == reference.LR_SPEC_SHA, 'LR spec changed')


def preflight(root=c.ROOT):
    spec,*_ = inspect(root)
    recheck(root,spec)
    c.publish_json(c.safe_path(root,p.PREFLIGHT),dict(candidate_spec=spec,scores_computed=0,fit_operations=0,compound_truth_columns_materialized=False))
    return dict(preflight=p.PREFLIGHT,scores_computed=0)


def freeze_spec(root=c.ROOT):
    prior = c.read_json(c.safe_path(root,p.PREFLIGHT))
    c.require(prior['scores_computed'] == prior['fit_operations'] == 0 and prior['compound_truth_columns_materialized'] is False, 'invalid preflight')
    spec,*_ = inspect(root)
    c.require(spec == validate(prior['candidate_spec']), 'preflight identities changed')
    recheck(root,spec)
    c.publish_json(c.safe_path(root,p.SPEC),spec)
    return dict(spec=p.SPEC,sha256=c.file_hash(c.safe_path(root,p.SPEC)),scores_computed=0)


def committed_spec(root):
    r = subprocess.run(['git','show','HEAD:'+p.SPEC],cwd=root,capture_output=True)
    c.require(r.returncode == 0 and r.stdout == c.safe_path(root,p.SPEC).read_bytes(), 'commit exact frozen spec before scoring')


def code():
    return dict(version=p.VERSION,source_sha256={name:c.file_hash(c.ROOT/name) for name in SOURCES})


def validate_isolated_sources(sources):
    c.require(list(sources) == p.SOURCE_FIELDS and sources.fact_key.is_unique and
              len(sources) == p.condition_counts()[p.ISO] and sources.fact_key.str.strip().ne('').all() and
              sources.source_kind.eq('fresh_extraction').all() and sources.cache_split.eq('').all() and
              sources.cache_row_index.eq('').all() and sources.example_id.eq(sources.fact_key).all(),
              'fresh isolated source contract')


def isolated_readouts(sources, extracted):
    """Select only fresh isolated scores; no cache, archive, or probe read API."""
    validate_isolated_sources(sources)
    c.require(set(extracted) == set(sources.example_id), 'fresh isolated extraction coverage incomplete')
    isolated = {r.fact_key: extracted[r.example_id] for r in sources.itertuples()}
    c.require(len(isolated) == len(sources) and np.isfinite(list(isolated.values())).all(), 'isolated source coverage incomplete')
    return isolated


def score(root=c.ROOT):
    destination = c.safe_path(root,p.OUTPUT+'/scores')
    c.require(not destination.exists(), 'refusing existing score output')
    spec = validate(c.read_json(c.safe_path(root,p.SPEC)))
    spec_sha = c.file_hash(c.safe_path(root,p.SPEC))
    committed_spec(root)
    current,probe,statements,index,sources = inspect(root)
    c.require(current == spec, 'frozen inputs differ')
    scores = {}
    with threadpool_limits(limits=1):
        array = np.load(c.safe_path(root,p.ACTS+'/activations.npy'),mmap_mode='r',allow_pickle=False)
        for start in range(0,len(statements),spec['scoring']['batch_size']):
            stop = min(len(statements),start+spec['scoring']['batch_size'])
            result = affine(array[start:stop,probe['layer'],:],probe['coef'],probe['intercept'])
            scores.update(zip(statements.example_id.iloc[start:stop],map(float,result)))
        del array
        isolated_ids = statements.loc[statements.condition_id == p.ISO, 'example_id']
        isolated = isolated_readouts(sources, {key: scores[key] for key in isolated_ids})
    table = index.copy()
    table['frozen_probe_score'] = table.example_id.map(scores)
    isolated_table = pd.DataFrame(dict(fact_key=sources.fact_key,frozen_probe_score=sources.fact_key.map(isolated)))
    c.require(list(table) == SCORE_COLUMNS and list(isolated_table) == ISOLATED_COLUMNS and
              np.isfinite(table.frozen_probe_score).all() and np.isfinite(isolated_table.frozen_probe_score).all(), 'missing/nonfinite scores')
    recheck(root,spec)
    c.require(c.file_hash(c.safe_path(root,p.SPEC)) == spec_sha, 'spec changed')
    destination.mkdir(parents=True)
    for name,frame in [('condition_scores.csv',table),('isolated_scores.csv',isolated_table)]:
        with (destination/name).open('x',encoding='utf-8',newline='') as handle: frame.to_csv(handle,index=False,float_format='%.17g')
    c.publish_json(destination/'scoring_manifest.json',dict(complete=True,version=p.VERSION,analysis_spec_sha256=spec_sha,
        inputs=spec['inputs'],representation_fingerprint=c.FINGERPRINT,selected_layer=probe['layer'],C=probe['C'],
        selected_probe_sha256=probe['files']['selected_probe.npz']['sha256'],
        condition_rows=len(table),isolated_rows=len(isolated_table),
        isolated_scoring_source=p.ISOLATED_SCORING_SOURCE,
        ordered_example_id_sha256=c.ordered_hash(table.example_id),ordered_fact_key_sha256=c.ordered_hash(isolated_table.fact_key),
        score_columns=SCORE_COLUMNS,isolated_columns=ISOLATED_COLUMNS,fit_operations=0,compound_truth_columns_materialized=False,
        compound_labels_used=False,test_accessed=False,provenance=code(),
        outputs={name:c.record(destination/name) for name in ['condition_scores.csv','isolated_scores.csv']}))
    return dict(output=str(destination),complete=True,condition_rows=len(table),isolated_rows=len(isolated_table))
