"""Fixed v1 policy and strict measured-artifact schema; no scientific data reads."""
import re
import subprocess

from src import clean_transfer_contracts as c
from src.atomic_probe_methods import METHODS

VERSION = 'pinned-qwen25-method-transfer-v1'
SPEC = 'config/clean_protocol/pinned_qwen25_method_transfer_v1.json'
PREFLIGHT = 'results/clean_protocol/pinned_qwen25_method_transfer_preflight_v1.json'
OUTPUT = 'results/clean_protocol/entity_disjoint_method_transfer_qwen25_pinned_v1'
SUITES = {k: f'results/clean_protocol/atomic_method_{v}_pinned_v1/qwen25_7b'
          for k, v in [('faithful', 'suite'), ('matched', 'matched')]}
GROUPS = ['faithful_common_layer', 'faithful_method_selected_layer', 'matched_1000_common_layer']
COLUMNS = ['example_id', 'analysis_group', 'method', 'atomic_layer', 'frozen_probe_score']
LR_SPEC_SHA = '2a20d4170be37e704b80ace8c2d7f65ae21c668ebeedeeb9fda07d53cfa316b5'
LR_EVALUATION_SHA = '4d941c5754849e86005487b47a39363df2efd5227adc77566e6599fc4d93933b'
LR_PRIMARY_SHA = 'b14bcf53e14049e3694b246de7efb5b5af4445e15f6fba6eddd66fe141c779c6'
LR_FILES = ['row_scores.csv', 'scoring_manifest.json', 'evaluation/evaluation_manifest.json', 'evaluation/primary_metrics.csv']
THRESHOLDS = {m: dict(applicable=m in ('l2_logistic', 'ttpd'),
    threshold=0. if m in ('l2_logistic', 'ttpd') else None,
    reason='frozen binary logistic decision function' if m in ('l2_logistic', 'ttpd') else 'raw direction has no calibrated zero classification threshold') for m in METHODS}
SOURCES = ['src/method_transfer_contracts.py', 'src/pinned_method_binding.py',
           'src/pinned_method_scoring.py', 'src/pinned_method_evaluation.py',
           'scripts/39_score_pinned_method_transfer.py', 'scripts/40_evaluate_pinned_method_transfer.py',
           'src/pinned_atomic_method_suite.py', 'src/atomic_probe_methods.py',
           'src/atomic_method_suite.py', 'src/clean_atomic_probes.py', 'src/clean_atomic_extraction.py',
           'src/clean_transfer_contracts.py',
           'src/pinned_compound_scoring.py', 'src/clean_transfer_statistics.py',
           'src/clean_transfer_evaluation.py', 'src/clean_compounds.py', 'src/repaired_atomic_cache.py']


def suite_names(mode):
    layers = range(c.LAYERS) if mode == 'faithful' else [17]
    return ['provenance.json', 'allowed_atomic_rows.csv', 'burger_training_rows.csv',
            'validation_metrics.csv', 'summary.json', 'report.md', 'verification.json'] + [
            f'layers/layer_{l:02d}.{ext}' for l in layers for ext in ['npz', 'json']]


def contrasts():
    primary = [r['id'] for r in c.metric_definitions() if r['category'] == 'primary']
    return [dict(id=m + '_minus_lr', left=[GROUPS[0], m], right=[GROUPS[0], 'l2_logistic'], metrics=primary,
                 role='primary_method_contrast') for m in METHODS if m != 'l2_logistic'] + [
            dict(id=m + '_matched_minus_faithful', left=[GROUPS[2], m], right=[GROUPS[0], m], metrics=primary,
                 role='supplementary_training_subset_contrast') for m in METHODS]


def template(lr_spec, suites, reference, conditions):
    return dict(schema_version=1, analysis_id=VERSION, lr_spec=lr_spec, lr_spec_sha256=LR_SPEC_SHA,
        suites=suites, lr_reference=reference, conditions=conditions, representation=lr_spec['representation'],
        analysis_groups=[dict(id=g, role=r) for g, r in zip(GROUPS, ['primary', 'secondary', 'supplementary'])],
        methods=list(METHODS), benchmark=lr_spec['benchmark'], scoring=lr_spec['scoring'],
        metrics=lr_spec['metrics'], statistics=lr_spec['statistics'], bootstrap=lr_spec['bootstrap'],
        reporting=lr_spec['reporting'], threshold_applicability=THRESHOLDS, paired_contrasts=contrasts(),
        reproduction=dict(score_rtol=0., score_atol=1e-12, primary_atol=1e-14, schedule='exact_sha256',
                          ttpd_rtol=1e-9, ttpd_atol=1e-8),
        prohibitions={**lr_spec['prohibitions'], 'compound_method_selection': True, 'normalization': True,
                      'compound_training_subset_selection': True, 'method_winner_ranking': True},
        outputs=dict(root=OUTPUT, score_columns=COLUMNS, score_files=['method_scores.csv', 'method_binding.json', 'scoring_manifest.json'],
            evaluation_files=['primary_metrics.csv', 'primary_metrics.json', 'boundary_metrics.csv', 'geometry_metrics.csv',
                'topic_metrics.csv', 'paired_method_contrasts.csv', 'threshold_metrics.csv', 'bootstrap_summary.csv',
                'bootstrap_summary.json', 'bootstrap_draws.npz', 'bootstrap_pair_weights.npz', 'evaluation_manifest.json']))


def records(files, names):
    c.require(set(files) == set(names), 'file allowlist mismatch')
    for r in files.values():
        c.require(set(r) == {'sha256', 'bytes'} and isinstance(r['sha256'], str) and
                  re.fullmatch('[0-9a-f]{64}', r['sha256']) and type(r['bytes']) is int and r['bytes'] > 0,
                  'invalid measured file identity')


def validate(spec):
    lr = c.validate_spec(spec['lr_spec'])
    c.require(set(spec['suites']) == set(SUITES), 'suite set differs')
    for mode, directory in SUITES.items():
        entry = spec['suites'][mode]
        c.require(set(entry) == {'directory', 'files'} and entry['directory'] == directory, 'suite path mismatch')
        records(entry['files'], suite_names(mode))
    ref = spec['lr_reference']
    c.require(set(ref) == {'directory', 'files', 'schedule_sha256'} and ref['directory'] == c.OUTPUT and
              re.fullmatch('[0-9a-f]{64}', ref['schedule_sha256']), 'LR reference mismatch')
    records(ref['files'], LR_FILES)
    c.require(ref['files']['evaluation/evaluation_manifest.json']['sha256'] == LR_EVALUATION_SHA and
              ref['files']['evaluation/primary_metrics.csv']['sha256'] == LR_PRIMARY_SHA, 'canonical LR reference hash mismatch')
    conditions = spec['conditions']
    c.require(len(conditions) == 15 and [(r['analysis_group'], r['method']) for r in conditions] ==
              [(g, m) for g in GROUPS for m in METHODS], 'condition set/order mismatch')
    for r in conditions:
        c.require(set(r) == {'analysis_group', 'method', 'suite', 'atomic_layer', 'archive', 'archive_sha256', 'fit_rows'}, 'condition schema')
        mode = 'matched' if r['analysis_group'] == GROUPS[2] else 'faithful'
        layer = r['atomic_layer']
        c.require(r['suite'] == mode and type(layer) is int and 0 <= layer < c.LAYERS and
                  (layer == 17 or (r['analysis_group'] == GROUPS[1] and r['method'] != 'l2_logistic')), 'condition layer/suite')
        c.require(r['archive'] == f'layers/layer_{layer:02d}.npz' and r['archive_sha256'] ==
                  spec['suites'][mode]['files'][r['archive']]['sha256'], 'condition archive identity')
        c.require(r['fit_rows'] == (1000 if mode == 'matched' or r['method'] in ['burger_t_g', 'ttpd'] else 3144), 'fit exposure')
    c.require(c.canonical(spec) == c.canonical(template(lr, spec['suites'], ref, conditions)),
              'unknown keys or altered frozen policy')
    return spec


def load(root):
    return validate(c.read_json(c.safe_path(root, SPEC)))


def committed_spec(root):
    result = subprocess.run(['git', 'show', 'HEAD:' + SPEC], cwd=root, capture_output=True)
    c.require(result.returncode == 0 and result.stdout == c.safe_path(root, SPEC).read_bytes(),
              'finalized specification must be committed unchanged before scoring')


def code():
    head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=c.ROOT, capture_output=True, text=True)
    return dict(version=VERSION, git_revision=head.stdout.strip(), source_sha256={p: c.file_hash(c.ROOT / p) for p in SOURCES})
