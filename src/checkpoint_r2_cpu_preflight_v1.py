"""Pinned CPU preparation inventories and historical P-only benchmark loading."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from src.checkpoint_r2_selection_v1 import candidate_id, require

BASE = 'f1f0872efe5e8ece5911e9df1be377f32a514bb1'
CONFIG = 'config/clean_protocol/checkpoint_r2_cpu_preflight_v1.json'
CONFIG_SHA = 'd049f6b0cb84be8215463535fc399f486e72eebee4e81a7408135c0022cf2efa'
PACKAGE = 'results/t2_capacity_pilot_v1_20261001'


def identity(path):
    path = Path(path);before = path.stat();h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'unstable file')
    return dict(bytes=after.st_size, sha256=h.hexdigest())


def bound(path, pin):
    require(identity(path) == {k: pin[k] for k in ('bytes', 'sha256')}, 'stale binding: ' + str(path))
    return Path(path)


def config(root):
    path = Path(root) / CONFIG
    require(identity(path)['sha256'] == CONFIG_SHA, 'predeclared configuration changed')
    cfg = json.loads(path.read_text())
    for p, pin in cfg['pinned_components'].items():
        bound(Path(root) / p, pin)
    return cfg


def grid(cfg):
    result = []
    for model, settings in cfg['models'].items():
        for layer in settings['layers']:
            specs = [('l2_logistic', C) for C in cfg['raw_lr_C']] + [(m, None) for m in cfg['other_methods']]
            for method, C in specs:
                result.append(dict(candidate_id=candidate_id(model, layer, method, C), model=model,
                                   saved_layer=layer, method=method, C=C, cohort='P15',
                                   planned_rows=700 if method in ('burger_t_g', 'ttpd') else 2778,
                                   production_status='unrun_pending_common_base_and_bridge'))
    require(len(result) == len({r['candidate_id'] for r in result}) == 600, 'bank count/identity mismatch')
    return result


def saved_fit_inventory(root):
    root = Path(root)
    completion = json.loads((root / PACKAGE / 'completion.json').read_text())
    location = json.loads((root / PACKAGE / 'external_artifacts.json').read_text())
    directory = Path(location['pilot_root'])
    bound(directory / 'completion.json', identity(root / PACKAGE / 'completion.json'))
    records = []
    for fit in json.loads(bound(directory / 'fits.json', completion['artifacts']['fits.json']).read_text()):
        name = fit['id'];meta_path = f'fits/{name}/fit.json';param_path = f'fits/{name}/parameters.npz'
        saved = json.loads(bound(directory / meta_path, completion['artifacts'][meta_path]).read_text())
        if saved['configuration']['cohort'] != 'P15':
            continue
        pin = completion['artifacts'][param_path]
        bound(directory / param_path, pin)
        require(saved['parameters'] == pin and saved['valid_for_scoring'], 'invalid saved fit')
        c = saved['configuration'];C = c['C'] if c['method'] == 'l2_logistic' else None
        records.append(dict(candidate_id=candidate_id(c['model'], c['layer'], c['method'], C),
                            historical_id=name, historical_correction='v4', production_adopted=False,
                            reuse_status='conditional_match_pending_exact_v5_membership_and_representation',
                            fit_path=str(directory / meta_path), fit_binding=completion['artifacts'][meta_path],
                            parameter_path=str(directory / param_path), parameter_binding=pin,
                            fitting_identity=saved['fitting_identity'], fit_rows=saved['fit_rows'],
                            preprocessing_cohort=saved['preprocessing_fitting_cohort']))
    require(len(records) == len({r['candidate_id'] for r in records}) == 36, 'duplicate/missing reusable fit')
    return records


def historical_P15(root, model):
    """Only canonical training export loaded; no D/E arrays or scores accessed."""
    root = Path(root);cfg = config(root);settings = cfg['models'][model]
    external = json.loads((root / PACKAGE / 'external_artifacts.json').read_text())
    prepared = Path(external['prepared_root'])
    prep_pin = json.loads((root / 'results/t2_canonical_sensitivity_v4_20261001/preparation_receipt.json').read_text())
    map_path = bound(prepared / (model + '_train.mapping_v4.csv'), prep_pin['artifacts'][model + '_train.mapping_v4.csv'])
    mapping = pd.read_csv(map_path)
    inventory_path = bound(prepared / 'input_inventory.json', prep_pin['artifacts']['input_inventory.json'])
    inventory = json.loads(inventory_path.read_text());exports = Path(inventory['physical_export_root'])
    receipt_path = exports / 'export_receipt.json'
    sensitivity = json.loads((root / 'results/t2_canonical_sensitivity_v4_20261001/sensitivity_receipt.json').read_text())
    bound(receipt_path, sensitivity['export_receipt'])
    receipt = json.loads(receipt_path.read_text())['caches'][model + '_train']
    require(receipt['saved_layer_index'] == settings['canonical_layer'] and receipt['source_stable_through_export'], 'layer/source mismatch')
    path = bound(exports / (model + '_train.npy'), receipt['export'])
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    require(list(array.shape) == receipt['export']['shape'] and array.dtype.str == receipt['export']['dtype']
            and bool(array.flags.f_contiguous) == receipt['export']['fortran_order'], 'NPY header mismatch')
    from src.selection_repair_capacity_pilot_v1 import cohorts
    members, _ = cohorts(root);rows = members['P15']
    require(mapping.source_row_id.is_unique, 'duplicate training map')
    lookup = mapping.set_index('source_row_id')
    selected = lookup.loc[[r['source_row_id'] for r in rows]]
    indices = selected.export_row_index.to_numpy(int)
    require(np.array_equal(np.asarray(receipt['original_row_indices'])[indices], selected.tensor_row_index.to_numpy()), 'original tensor row mismatch')
    X = np.asarray(array[indices], dtype=np.float64);y = np.array([int(r['label']) for r in rows])
    require(X.shape == (2778, settings['width']) and np.isfinite(X).all(), 'P15 feature shape')
    name = f"{model}_L{settings['canonical_layer']}_l2_logistic_P15"
    completion = json.loads((root / PACKAGE / 'completion.json').read_text())
    pilot = Path(external['pilot_root']);fit_path = bound(pilot / f'fits/{name}/fit.json', completion['artifacts'][f'fits/{name}/fit.json'])
    fit = json.loads(fit_path.read_text())
    from src.selection_repair_sensitivity_v4 import canonical, sha
    require(sha(canonical(rows)) == fit['fitting_identity']['membership_sha256'], 'P membership identity mismatch')
    require(sha(X.tobytes()) == fit['fitting_identity']['feature_float64_sha256'] and
            sha(y.astype('<i8').tobytes()) == fit['fitting_identity']['labels_sha256'], 'P15 materialized fitting identity mismatch')
    return X, y, rows, dict(scope='historical_v4_P15_diagnostic_only', export=identity(path),
                            export_receipt=identity(receipt_path), mapping=identity(map_path),
                            fitting_identity=fit['fitting_identity'], D_or_E_arrays_loaded=False)


def write_inventory(root, output):
    root, output = Path(root), Path(output)
    cfg = config(root);require(not output.exists(), 'refuse inventory overwrite')
    bank = grid(cfg);reuse = saved_fit_inventory(root);lookup = {r['candidate_id']: r for r in reuse}
    for r in bank:
        r['historical_match'] = r['candidate_id'] in lookup
        r['historical_id'] = lookup[r['candidate_id']]['historical_id'] if r['historical_match'] else None
    output.mkdir(parents=True)
    pd.DataFrame(bank).to_csv(output / 'bank_candidates.csv', index=False)
    (output / 'conditional_reuse.json').write_text(json.dumps(reuse, indent=2, sort_keys=True) + '\n')
    folds = [dict(model=m, layer=l, sample_seed=seed, fold=f, status='unrun', reason='common base, frozen A/P, allocated pairs/folds and compatible states pending')
             for m, spec in cfg['models'].items() for l in spec['layers'] for seed in cfg['B25_seeds'] for f in range(5)]
    require(len(folds) == 900, 'repair fold count')
    pd.DataFrame(folds).to_csv(output / 'planned_repair_folds.csv', index=False)
    inventory = dict(base_sha=BASE, common_checkpoint_sha=None, bridge_receipt=None, stage='preparation_only',
                     bank_expected=600, per_model={'qwen':280, 'llama':320}, conditional_reusable=36,
                     executable_production_candidates=0, additional_bank_fits_if_all_reuse_gates_pass=564,
                     historical_excluded={'full_A_exposed':36, 'P10_wrong_cohort':36},
                     repair_fold_fits=900, repair_all_layer_full_A_refits_not_required=True,
                     post_fold_repair_refits=dict(R_all=6, fixed=6, at_S_all=6, maximum_distinct=18,
                                                 dedup_key='model, selected layer, sample exact allocation, objective, preprocessing, representation'),
                     matched_atomic_only_refits=dict(fixed=6, at_R_all=6, maximum_distinct=12,
                                                    dedup_key='model, layer, exact sample fact identities, objective, representation; same-layer arms share head'),
                     C_clean_all_layers=60,C_clean_procedure_layer_extra_fits=0,
                     constituent_source_fits=240,constituent_full_TC_refits_max=16,
                     chat_fits_max=8,chat_status='future Prompt1 only, no execution authorized',
                     all_declared_core_fits_upper=1846,core_fits_after_36_conditional_reuses_upper=1810,
                     core_count_formula='600 bank + 60 clean + 900 folds + 18 repair refits + 12 isolated controls + 240 constituent source + 16 constituent full refits',
                     checkpoint_plus_chat_fits_upper=1854,
                     unimplemented_production=['v5 membership/representation gate','all-layer bank orchestration and score binding','original S_all/R_all eligibility/fallback orchestration','C_clean P-preprocessing adapter','B25 identity and fold assembler','full transfer/decision/composition evaluator','paired B25 inference and cross-model effects','behavior/chat'],
                     executable_now=['synthetic selector checks','pinned objective tests','metadata inventory','bounded synthetic timings','verified canonical v4 P15-only timing diagnostics'],
                     gates=['exact reviewed mac1 common checkpoint SHA','v5 train/D/A15/TC identity equality receipts','A15 method-damage scope amendment and explicit freeze','independent representation bridge with predeclared tolerances and cap verdict','all-layer exports with hashes, headers and row/layer bindings','seeded B25 exact pairs/facts/folds and TC source-only split/pair identities'],
                     P_A_frozen=False,production_started=False)
    (output / 'execution_inventory.json').write_text(json.dumps(inventory, indent=2, sort_keys=True) + '\n')
    return inventory
