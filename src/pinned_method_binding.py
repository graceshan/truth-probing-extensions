"""Read-only atomic-suite verification. Never fits or reads compound truth columns."""
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src import clean_transfer_contracts as c
from src import method_transfer_contracts as m
from src import pinned_atomic_method_suite as atomic
from src.atomic_method_suite import choose_layers
from src.atomic_probe_methods import METHODS
from src.clean_transfer_evaluation import verify_score_inputs
from src.pinned_compound_scoring import verify_compound


def parameters(path, methods):
    with np.load(path, allow_pickle=False) as archive:
        all_params = {k: archive[k].copy() for k in archive.files}
    c.require({k.split('__')[0] for k in all_params} == set(methods), 'archive method set')
    required = {
        'l2_logistic': {'coef', 'intercept', 'classes', 'C'},
        'difference_of_means': {'coef', 'intercept', 'mean_false', 'mean_true', 'raw_direction'},
        'mass_mean_covariance': {'coef', 'intercept', 'mean_false', 'mean_true', 'raw_direction', 'covariance_eigenvalues', 'pinv_atol'},
        'burger_t_g': {'coef', 'intercept', 't_g', 't_p', 'dataset_names', 'dataset_means', 'ols_gram'},
    }
    required['ttpd'] = required['burger_t_g'] | {'polarity_coef', 'polarity_intercept', 'polarity_classes', 'head_coef', 'head_intercept', 'classes'}
    resolved = {}
    for method in methods:
        p = {k[len(method)+2:]: v for k, v in all_params.items() if k.startswith(method + '__')}
        c.require(set(p) == required[method], 'parameter key schema mismatch')
        for key, value in p.items():
            c.require(value.dtype.kind in 'biufU' and (key == 'dataset_names' and value.dtype.kind == 'U' or
                      value.dtype.kind != 'U' and np.isfinite(value).all()), 'unsafe/nonfinite parameter')
        c.require(p['coef'].shape == (c.WIDTH,) and p['intercept'].shape == (), 'affine parameter shape')
        for key in ['mean_false', 'mean_true', 'raw_direction', 't_g', 't_p', 'polarity_coef', 'covariance_eigenvalues']:
            if key in p: c.require(p[key].shape == (c.WIDTH,), 'direction shape')
        if method in ['l2_logistic', 'ttpd']:
            c.require(np.array_equal(p['classes'], [0, 1]), 'class orientation')
        if method == 'l2_logistic': c.require(p['C'].shape == () and p['C'].item() == 1., 'LR C differs')
        if method == 'mass_mean_covariance': c.require(p['pinv_atol'].item() == .001, 'MM cutoff differs')
        if method in ['difference_of_means', 'mass_mean_covariance', 'burger_t_g']:
            c.require(p['intercept'].item() == 0., 'raw direction intercept changed')
        if method in ['burger_t_g', 'ttpd']:
            c.require(p['dataset_names'].shape == (10,) and len(set(p['dataset_names'])) == 10 and
                      p['dataset_means'].shape == (10, c.WIDTH) and p['ols_gram'].shape == (2, 2), 'truth direction dataset schema')
        if method == 'burger_t_g':
            c.require(np.array_equal(p['coef'], p['t_g']), 'raw t_G coefficient changed')
        if method == 'ttpd':
            c.require(p['head_coef'].shape == (2,) and p['head_intercept'].shape == () and
                      p['polarity_intercept'].shape == (1,) and np.array_equal(p['polarity_classes'], [0, 1]), 'TTPD head schema')
            c.require(np.allclose(p['coef'], p['head_coef'][0]*p['t_g'] + p['head_coef'][1]*p['polarity_coef'], rtol=1e-9, atol=1e-8)
                      and p['intercept'] == p['head_intercept'], 'TTPD fused parameters differ')
        resolved[method] = p
    return resolved


def verify_suite(root, mode, data, probe, identity):
    directory = m.SUITES[mode]
    paths = {name: c.safe_path(root, directory + '/' + name) for name in m.suite_names(mode)}
    receipt = c.read_json(paths['verification.json'])
    c.require(receipt['complete'] is True and receipt['archive_metrics_reproduced'] is True and
              receipt['ttpd_fused_explicit_agree'] is True and receipt['canonical_inputs_unchanged'] is True,
              'incomplete atomic suite verification')
    for flag in ['atomic_test_accessed', 'compound_data_accessed', 'validation_sign_flip']:
        c.require(receipt[flag] is False, 'forbidden suite provenance: ' + flag)
    c.require(receipt['canonical_lr_reused_without_refit'] is (mode == 'faithful') and
              receipt['method_fits_performed'] == (112 if mode == 'faithful' else 5), 'suite LR fit policy')
    m.records(receipt['outputs'], set(paths) - {'verification.json'})
    files = {name: c.record(path) for name, path in paths.items()}
    c.require(all(files[n] == r for n, r in receipt['outputs'].items()), 'suite output hash mismatch')
    provenance, summary = [c.read_json(paths[n]) for n in ['provenance.json', 'summary.json']]
    for document in [provenance, summary]:
        c.require(document['schema_version'] == 1 and document['comparison'] == mode and
                  document['representation_fingerprint'] == c.FINGERPRINT and
                  document['atomic_test_accessed'] is False and document['compound_data_accessed'] is False and
                  document['lr_refitted'] is (mode == 'matched'), 'suite comparison/provenance mismatch')
    c.require(provenance['implementation']['version'] == atomic.VERSION and provenance['implementation']['policy'] == atomic.POLICY,
              'atomic producer policy mismatch')
    for key in ['representation', 'repaired_cache', 'allowed_atomic_rows_sha256', 'ordered_partition_identities',
                'burger_training_rows_sha256', 'burger_ordered_identities_sha256', 'partition_counts', 'burger_dataset_counts']:
        c.require(provenance[key] == identity[key], 'suite repaired/sample identity mismatch: ' + key)
    for key in ['directory', 'files', 'layer', 'C']:
        c.require(provenance['frozen_lr'][key] == identity['frozen_lr'][key], 'suite canonical LR binding mismatch')
    c.require(paths['allowed_atomic_rows.csv'].read_bytes() == data.rows.to_csv(index=False).encode() and
              paths['burger_training_rows.csv'].read_bytes() == data.burger_rows.to_csv(index=False).encode(), 'Burger/allowed row table mismatch')
    c.require(summary['primary_layer'] == 17 and summary['canonical_lr_C'] == 1. and summary['validation_sign_flip'] is False and
              summary['fit_split'] == 'train' and summary['selection_split'] == 'validation' and summary['selection_rule'] ==
              (atomic.POLICY['secondary_rule'] if mode == 'faithful' else 'fixed layer; no layer/C selection'), 'suite selection semantics')
    rows, archives = [], {}
    with threadpool_limits(limits=1):
        for layer in (range(c.LAYERS) if mode == 'faithful' else [17]):
            name = f'layers/layer_{layer:02d}'
            record = c.read_json(paths[name + '.json'])
            methods = list(METHODS) if layer == 17 else list(METHODS[1:])
            c.require(record['layer'] == layer and record['archive_verified'] is True and
                      record['representation_fingerprint'] == c.FINGERPRINT and
                      record['parameter_archive_sha256'] == files[name + '.npz']['sha256'], 'layer archive hash/identity mismatch')
            c.require(len(record['metrics']) == len(methods) and {r['method'] for r in record['metrics']} == set(methods), 'layer method coverage')
            p = parameters(paths[name + '.npz'], methods)
            if layer == 17:
                if mode == 'faithful':
                    c.require(np.array_equal(p['l2_logistic']['coef'], probe['coef']) and
                              p['l2_logistic']['intercept'] == probe['intercept'] and
                              record['method_details']['l2_logistic']['reused_without_refitting'] is True,
                              'canonical LR not reused unchanged')
                else:
                    c.require(record['method_details']['l2_logistic']['configuration'] == probe['selection']['probe_configuration'],
                              'matched LR solver policy')
                    optimization = record['method_details']['l2_logistic']['optimization']
                    c.require(optimization['final_converged'] is True and optimization['convergence_warning'] is False and
                              optimization['final_convergence_status'] == 'converged' and optimization['initial_max_iter'] == 2000 and
                              optimization['final_max_iter'] == (10000 if optimization['retry_needed'] else 2000),
                              'matched LR convergence policy')
            for key in ['polarity_optimizer', 'head_optimizer']:
                attempts = record['method_details']['ttpd'][key]
                c.require([a['max_iter'] for a in attempts] in [[100], [100, 10000]] and
                          attempts[-1]['convergence_warnings'] == [] and
                          (len(attempts) == 1 or bool(attempts[0]['convergence_warnings'])), 'TTPD convergence policy')
            for row in record['metrics']:
                subset = mode == 'matched' or row['method'] in ['burger_t_g', 'ttpd']
                c.require(row['comparison'] == mode and row['layer'] == layer and row['fit_rows'] == (1000 if subset else 3144)
                          and row['validation_rows'] == 1040 and row['parameters'] == name + '.npz' and
                          row['parameter_archive_sha256'] == files[name + '.npz']['sha256'] and
                          row['ordered_fit_row_ids_sha256'] == atomic.identity_hash(data.burger_rows if subset else data.partitions['train']) and
                          row['ordered_validation_row_ids_sha256'] == atomic.identity_hash(data.partitions['validation']) and
                          row['validation_activation_slice_sha256'] == record['validation_activation_sha256'], 'method metric identity mismatch')
                c.require(record['method_details'][row['method']]['fit_rows'] == row['fit_rows'], 'fit detail row mismatch')
            # Atomic VALIDATION only, no fits. This checks actual saved diagnostics
            # and explicit/fused TTPD on every archived layer, before production use.
            atomic.verify_archive(paths[name + '.npz'].read_bytes(), record, data)
            rows.extend(record['metrics'])
            archives[layer] = p
    rows.sort(key=lambda r: (r['method'], r['layer']))
    table = pd.read_csv(paths['validation_metrics.csv'], keep_default_na=False, float_precision='round_trip')
    c.require(len(table) == (113 if mode == 'faithful' else 5), 'validation metric count')
    pd.testing.assert_frame_equal(table.sort_index(axis=1), pd.DataFrame(rows).sort_index(axis=1), check_dtype=False, check_exact=True)
    chosen = choose_layers(rows, 17)
    if mode == 'matched': chosen.pop('secondary_method_selected_layer')
    c.require(summary['comparisons'] == chosen, 'secondary winners/summary differ from validation-only rule')
    return dict(files=files, directory=directory), chosen, archives


def lr_reference(root, lr_spec):
    scores, projected, _, scoring = verify_score_inputs(root, lr_spec)  # truth-blind identity projection only
    paths = {n: c.safe_path(root, c.OUTPUT + '/' + n) for n in m.LR_FILES}
    files = {n: c.record(p) for n, p in paths.items()}
    manifest = c.read_json(paths['evaluation/evaluation_manifest.json'])
    c.require(files['evaluation/evaluation_manifest.json']['sha256'] == m.LR_EVALUATION_SHA and
              files['evaluation/primary_metrics.csv']['sha256'] == m.LR_PRIMARY_SHA, 'canonical LR result identity mismatch')
    c.require(manifest['complete'] is True and manifest['implementation_version'] == 'clean-pinned-transfer-evaluation-v1' and
              manifest['analysis_spec_sha256'] == m.LR_SPEC_SHA and manifest['row_scores'] == files['row_scores.csv'] and
              manifest['scoring_manifest_sha256'] == files['scoring_manifest.json']['sha256'] and
              manifest['outputs']['primary_metrics.csv'] == files['evaluation/primary_metrics.csv'] and
              manifest['rows'] == c.ROWS and manifest['matched_rows'] == c.ROWS // 2 and
              manifest['metadata'] == lr_spec['inputs']['compound']['files']['metadata.csv'] and
              manifest['representation_fingerprint'] == c.FINGERPRINT and manifest['bootstrap'] == lr_spec['bootstrap'], 'LR evaluation chain mismatch')
    c.require(all(manifest[k] is False for k in ['atomic_or_compound_test_accessed', 'activation_arrays_opened', 'probe_archives_opened']), 'LR prohibited access')
    return dict(directory=c.OUTPUT, files=files, schedule_sha256=manifest['schedule_sha256']), scores


def inspect(root):
    lr_path = c.safe_path(root, c.SPEC)
    c.require(c.file_hash(lr_path) == m.LR_SPEC_SHA, 'frozen LR spec changed')
    lr_spec = c.validate_spec(c.read_json(lr_path))
    data, probe, _, identity = atomic.prepare(root)
    compound = verify_compound(root, data.cache)
    for key, files in [('atomic', data.cache.files), ('probe', probe['files']), ('compound', compound['files'])]:
        c.require(files == lr_spec['inputs'][key]['files'], 'canonical inputs differ from frozen LR spec')
    suites, selections, archives = {}, {}, {}
    for mode in m.SUITES:
        suites[mode], selections[mode], archives[mode] = verify_suite(root, mode, data, probe, identity)
    c.require(suites['faithful']['files']['burger_training_rows.csv'] == suites['matched']['files']['burger_training_rows.csv'], 'Burger tables differ')
    conditions, affine = [], {}
    for group in m.GROUPS:
        mode = 'matched' if group == m.GROUPS[2] else 'faithful'
        policy = 'secondary_method_selected_layer' if group == m.GROUPS[1] else 'primary_fixed_model_layer'
        for method in METHODS:
            row = selections[mode][policy][method]
            layer = row['layer']
            name = f'layers/layer_{layer:02d}.npz'
            conditions.append(dict(analysis_group=group, method=method, suite=mode, atomic_layer=layer,
                archive=name, archive_sha256=suites[mode]['files'][name]['sha256'], fit_rows=row['fit_rows']))
            params = archives[mode][layer][method]
            affine[group, method] = (params['coef'], float(params['intercept']))
    reference, lr_scores = lr_reference(root, lr_spec)
    spec = m.validate(m.template(lr_spec, suites, reference, conditions))
    binding = dict(schema_version=1, representation_fingerprint=c.FINGERPRINT, suites=suites, conditions=conditions,
        atomic_validation_archives_reproduced=True, ttpd_explicit_fused_verified=True,
        atomic_activation_rows_read=data.activation_rows_read, fit_operations=0,
        atomic_test_accessed=False, compound_labels_used=False)
    return spec, binding, affine, compound['identities'], lr_scores
