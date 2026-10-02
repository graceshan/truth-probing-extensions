"""Authorized frozen fresh B25 bank, 900 folds, locked refits and wording transfer."""
import argparse
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from src.checkpoint_r2_fresh_inputs import ROOT, file_hash, hash_value, require
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_fresh_recoverability import save, verify_current_inputs
from src.checkpoint_r2_b25_adapter import B25Adapter, SEEDS
from src.checkpoint_r2_b25_store import FitStore, Reusable, read, RECOVER, RECOVER_SHA
from src.checkpoint_r2_b25_selection import rank, mean_fold_statistics, final_refit
from src.checkpoint_r2_b25_statistics import allocated_statistics, summaries, evaluate_locked
from src.checkpoint_r2_selection_v1 import AtomicCandidate, candidate_id, atomic_eligibility, select_atom_all, select_original_atomic_r0, TOPICS
from src.checkpoint_r2_cpu_preflight_v1 import grid

CONFIG = 'config/checkpoint_r2/fresh_b25_v1.json'


def checked_save(path, value):
    if Path(path).exists():
        require(read(path) == value, 'completed receipt differs: ' + str(path))
    else:
        save(path, value)


def compact_record(record):
    """Keep complete fit identities/status; row-key lists stay hash-bound per fit."""
    result = {k: v for k, v in record.items() if k != 'score_bindings'}
    if 'score_bindings' in record:
        result['score_group_bindings'] = {g: dict(row_binding_sha256=p['row_binding_sha256'], rows=len(p['ordered_keys']),
            ordered_keys_sha256=hash_value(p['ordered_keys'])) for g, p in record['score_bindings'].items()}
    return result


def configuration():
    cfg = read(ROOT / CONFIG)
    require(cfg['bank_candidates'] == 600 and cfg['repair_fold_fits'] == 900 and cfg['processes'] == cfg['blas_threads'] == 1, 'frozen stage bounds')
    for name, digest in cfg['frozen_hashes'].items():
        require(file_hash(ROOT / name) == digest, 'frozen dependency changed: ' + name)
    require({p: importlib.metadata.version(p) for p in cfg['local_packages']} == cfg['local_packages'], 'numerical environment changed')
    require(cfg['historical_reuse'] is False and cfg['wording_selection'] is False and cfg['E_evaluation'] is False, 'forbidden stage configuration')
    return cfg


def bank_specs():
    specs = grid(read(ROOT / 'config/clean_protocol/checkpoint_r2_cpu_preflight_v1.json'))
    return [{k: r[k] for k in ('candidate_id', 'model', 'saved_layer', 'method', 'C', 'cohort', 'planned_rows')} for r in specs]


def fold_id(model, layer, seed, fold):
    return f'{model}/B25_{seed}/L{layer:02d}/repair/fold={fold}'


def refit_id(model, layer, seed):
    return f'{model}/B25_{seed}/L{layer:02d}/repair/full'


def atomic_id(adapter, model, layer, seed):
    facts = sorted(r['fact_id'] for r in adapter.allocations[seed]['isolated'])
    return f'{model}/B25_isolated/L{layer:02d}/facts={hash_value(facts)}/atomic_only'


def atomic_point(record):
    return next(r['point'] for r in record['metrics'] if r['metric'] == 'atomic_auroc' and r['scope'] == 'pooled')


def validate_reuse(adapter, reusable, cfg):
    """Before real fitting, reconstruct every potential bank reuse on fresh rows."""
    from src.checkpoint_r2_b25_store import array_hash, npz
    from src.selection_repair_capacity_methods_v1 import score
    records = []
    for model in ('qwen', 'llama'):
        spec = cfg['models'][model]
        for layer in range(spec['layers']):
            features = adapter.layer(model, layer); pre = reusable.pre(model, layer, features['P15'])
            for family in ['R0'] + (['reduced_LR'] if layer == spec['fixed_layer'] else []):
                params, source = reusable.head(model, layer, family, pre)
                require(source['training_bindings_sha256'] == hash_value(adapter.bindings['P15']) and source['feature_float64_sha256'] == array_hash(features['P15']) and source['fit_rows'] == 2778, 'pre-fit exact bank reuse exposure')
                method = 'r0' if family == 'R0' else 'l2_logistic'
                if family == 'reduced_LR':
                    require(source['C'] == spec['fixed_C'] and source['preprocessing'] == 'none', 'pre-fit raw LR settings')
                old = npz(reusable.root / 'fits' / source['id'] / 'scores.npz')
                for group in ('P15', 'atomic_D', 'D_bare'):
                    require(source['score_bindings'][group]['row_binding_sha256'] == hash_value(adapter.bindings[group]), 'pre-fit reused score binding')
                    require(np.array_equal(score(method, params, features[group]), old[group]), 'pre-fit reuse score differs')
                records.append(dict(candidate_id=candidate_id(model, layer, method, spec['fixed_C'] if family == 'reduced_LR' else None),
                    source_id=source['id'], source_parameter_producer_sha=RECOVER_SHA, parameters=source['parameters'], training_bindings_sha256=source['training_bindings_sha256'],
                    feature_float64_sha256=source['feature_float64_sha256'], exact_preprocessing=True, three_score_vectors_exact=True))
            print('pre-fit reuse verified', model, layer, flush=True)
            del features, pre; gc.collect()
    require(len(records) == len({r['candidate_id'] for r in records}) == 62, 'exact pre-fit bank reuse inventory')
    adapter.unchanged(); reusable.unchanged()
    return records


def stage_one(adapter, reusable, store, cfg):
    specs = bank_specs(); bank, folds = [], []
    for model in ('qwen', 'llama'):
        for layer in range(cfg['models'][model]['layers']):
            start = time.monotonic(); features = adapter.layer(model, layer)
            pre = reusable.pre(model, layer, features['P15'])
            for spec in [r for r in specs if r['model'] == model and r['saved_layer'] == layer]:
                record = store.bank(spec, features, pre, reusable)
                entry = dict(compact_record(record), allocated={})
                if record['valid_for_scoring']:
                    scores = store.scores(record['id'])
                    for seed in SEEDS:
                        group = f'B25_{seed}'
                        entry['allocated'][str(seed)] = allocated_statistics(adapter.rows[group], scores[group], scores['P15'], atomic_point(record))
                    entry['oracle'] = allocated_statistics(adapter.rows['D_bare'], scores['D_bare'], scores['P15'], atomic_point(record))
                    entry['all_bare_points'], _ = summaries(adapter.rows['D_bare'], scores['D_bare'])
                bank.append(entry)
            for seed in SEEDS:
                for fold in adapter.allocations[seed]['folds']:
                    name = fold_id(model, layer, seed, fold['fold'])
                    record = store.adaptation(name, model, layer, seed, features, pre, 'compound', fold['train'], fold['fold'])
                    entry = dict(id=name, model=model, layer=layer, seed=seed, fold=fold['fold'], valid_for_scoring=record['valid_for_scoring'],
                                 training_bindings_sha256=fold['training_bindings_sha256'], heldout_bindings_sha256=fold['heldout_bindings_sha256'], failure=record.get('failure'))
                    if record['valid_for_scoring']:
                        scores = store.scores(name); group = f'B25_{seed}'
                        entry.update(allocated_statistics(adapter.rows[group], scores[group], scores['P15'], atomic_point(record), fold['heldout']))
                    folds.append(entry)
            print('completed bank/folds layer', model, layer, 'seconds', time.monotonic() - start, flush=True)
            del features, pre; gc.collect()
    require(len(bank) == len({r['id'] for r in bank}) == 600 and len(folds) == len({r['id'] for r in folds}) == 900, 'complete bank/fold inventory')
    require(sum(r['reused'] for r in bank) == 62, '62 exact fresh bank reuses required')
    # Original preparation rows bind the exact 900 planned tuple identities.
    import csv
    planned = list(csv.DictReader((ROOT / 'results/checkpoint_r2_cpu_preflight_v1_20261002/planned_repair_folds.csv').open()))
    require({(r['model'], int(r['layer']), int(r['sample_seed']), int(r['fold'])) for r in planned} == {(r['model'], r['layer'], r['seed'], r['fold']) for r in folds}, 'declared fold identities changed')
    return bank, folds


def select_procedures(bank, folds, cfg, reusable):
    traces = {}; requests = {}; shared = {}; anchors = {}
    for model in ('qwen', 'llama'):
        records = [r for r in bank if r['model'] == model]; spec = cfg['models'][model]
        reference = candidate_id(model, spec['fixed_layer'], 'l2_logistic', spec['fixed_C'])
        ref = next(r for r in records if r['id'] == reference); reference_atomic = atomic_point(ref)
        candidates = []
        for r in records:
            c = r['spec']; valid = r['valid_for_scoring']
            topics = {t: next(x['point'] for x in r['metrics'] if x['metric'] == 'atomic_auroc' and x['scope'] == 'topic' and x['topic'] == t) for t in TOPICS} if valid else {}
            candidates.append(AtomicCandidate(r['id'], model, c['saved_layer'], c['method'], c['C'], bool(valid), atomic_point(r) if valid else None, topics))
        eligibility = atomic_eligibility(candidates, model, reference_atomic)
        eligible = [r['candidate_id'] for r in eligibility if r['eligible']]
        atom = select_atom_all(candidates, model=model, reduced_lr_atomic_auc=reference_atomic, expected_s_all_eligible_ids=eligible)
        require(atom['selected_id'] is not None, 'S_atom_all unresolved')
        r0 = select_original_atomic_r0(candidates, model=model)
        inherited = read(reusable.root / 'selected.json')['models'][model]['R0_atomic_selected_id']
        require(r0['selected_id'] == candidate_id(model, int(inherited[-2:]), 'r0'), 'original R0 selection differs from accepted run')
        oracle_records = [dict(id=r['id'], valid_for_scoring=r['valid_for_scoring'], **r.get('oracle', dict(primary=None, atomic=None, balanced=None, separation=None))) for r in records]
        oracle = rank(oracle_records, reference_atomic=reference_atomic, fallback_id=reference, arm='optimistic_D_bank_oracle', eligible_ids=eligible)
        shared[model] = dict(eligibility=eligibility, eligible_ids=eligible, reference=reference, reference_atomic=reference_atomic, S_atom_all=atom, original_R0=r0, bank_oracle=oracle,
                             bank_candidates=len(records), bank_C_clean_candidates=0)
        anchors[model] = dict(reference=reference, reference_atomic=reference_atomic, S_atom_all=atom['selected_id'], original_R0=r0['selected_id'], bank_oracle=oracle['selected_id'])
        traces[model] = {}; requests[model] = {}
        for seed in SEEDS:
            scores = [dict(id=r['id'], valid_for_scoring=r['valid_for_scoring'], **r['allocated'].get(str(seed), dict(primary=None, atomic=None, balanced=None, separation=None))) for r in records]
            selected = {}
            for arm, subset, sensitivity in [('S_all', scores, False), ('S_fixed', [r for r in scores if next(b['layer'] for b in records if b['id'] == r['id']) == spec['fixed_layer']], False),
                ('LR_only_selection', [r for r in scores if '/l2_logistic/' in r['id']], False), ('S_all_equal_boundary_sensitivity', scores, True)]:
                selected[arm] = rank(subset, reference_atomic=reference_atomic, fallback_id=reference, arm=arm, eligible_ids=eligible, balanced=sensitivity)
            repair_candidates = []
            for layer in range(spec['layers']):
                layer_folds = [f for f in folds if (f['model'], f['seed'], f['layer']) == (model, seed, layer)]
                statistics = mean_fold_statistics(layer_folds)
                repair_candidates.append(dict(id=f'{model}/B25_{seed}/L{layer:02d}/repair_cv', layer=layer, valid_for_scoring=True, **statistics))
            for arm, sensitivity in [('R_all', False), ('R_all_equal_boundary_sensitivity', True)]:
                selected[arm] = rank(repair_candidates, reference_atomic=reference_atomic, fallback_id=reference, arm=arm, balanced=sensitivity)
                trace = selected[arm]
                trace['selected_layer'] = spec['fixed_layer'] if trace['chosen_id'] is None else next(r['layer'] for r in repair_candidates if r['id'] == trace['chosen_id'])
            layer_for = lambda ident: next(r['layer'] for r in records if r['id'] == ident)
            s_layer = layer_for(selected['S_all']['selected_id'])
            layers = {spec['fixed_layer'], s_layer}
            layers |= {selected[arm]['selected_layer'] for arm in ('R_all', 'R_all_equal_boundary_sensitivity') if selected[arm]['chosen_id'] is not None}
            requests[model][str(seed)] = dict(repair_layers=sorted(layers), S_all_layer=s_layer)
            traces[model][str(seed)] = dict(selected=selected, repair_layer_means=repair_candidates)
    return dict(shared_atomic_banks=shared, anchors=anchors, traces=traces, requests=requests)


def refits_and_lock(adapter, store, reusable, selection, cfg):
    # All 900 fold fits finish before any full-allocation refit/control.
    procedures = {}; aliases = []
    for model in ('qwen', 'llama'):
        layer_requests = {}
        for seed in SEEDS:
            for layer in selection['requests'][model][str(seed)]['repair_layers']:
                layer_requests.setdefault(layer, []).append(seed)
        for layer, seeds in sorted(layer_requests.items()):
            features = adapter.layer(model, layer); pre = reusable.pre(model, layer, features['P15'])
            for seed in seeds:
                store.adaptation(refit_id(model, layer, seed), model, layer, seed, features, pre, 'compound')
            del features, pre; gc.collect()
        procedures[model] = {}
        a = selection['anchors'][model]; spec = cfg['models'][model]
        for seed in SEEDS:
            trace = selection['traces'][model][str(seed)]['selected']; request = selection['requests'][model][str(seed)]
            p = {arm: trace[arm]['selected_id'] for arm in ('S_all', 'S_fixed', 'LR_only_selection', 'S_all_equal_boundary_sensitivity')}
            p.update(reduced_LR=a['reference'], S_atom_all=a['S_atom_all'], original_atomic_selected_R0=a['original_R0'], bank_oracle=a['bank_oracle'],
                R0_fixed=candidate_id(model, spec['fixed_layer'], 'r0'), R_fixed=refit_id(model, spec['fixed_layer'], seed), repair_at_S_all=refit_id(model, request['S_all_layer'], seed))
            for arm in ('R_all', 'R_all_equal_boundary_sensitivity'):
                chosen = trace[arm]
                if chosen['chosen_id'] is None:
                    p[arm] = a['reference']; chosen['selected_fit_id'] = a['reference']
                else:
                    fit = store.load(refit_id(model, chosen['selected_layer'], seed))
                    result = final_refit(chosen, fit_id=fit['id'], valid=fit['valid_for_scoring'], atomic=atomic_point(fit) if fit['valid_for_scoring'] else None, reference_atomic=a['reference_atomic'])
                    trace[arm] = result; p[arm] = result['selected_fit_id']
                trace[arm]['returned_layer'] = store.load(p[arm])['layer']
            r_layer = store.load(p['R_all'])['layer']
            p['R0_at_R_all'] = candidate_id(model, r_layer, 'r0')
            # Procedure-layer controls use the actually returned head's layer; pre-fallback choice stays in trace.
            for arm, layer in [('atomic_only_fixed', spec['fixed_layer']), ('atomic_only_at_R_all', r_layer)]:
                p[arm] = atomic_id(adapter, model, layer, seed)
            cc_selected = read(reusable.root / 'selected.json')['models'][model]['C_clean_D_selector']['selected_id']
            for arm, layer in [('C_clean_fixed', spec['fixed_layer']), ('C_clean_D_selected', int(cc_selected[-2:])), ('C_clean_at_S_all', request['S_all_layer']),
                               ('C_clean_at_R_all', r_layer), ('C_clean_at_S_atom_all', store.load(a['S_atom_all'])['layer'])]:
                p[arm] = f'{model}/TC/L{layer:02d}/C_clean'
            procedures[model][str(seed)] = p
    # Canonical exposure identities deduplicate coincident requested controls, never observations.
    pending = {}
    for model, seeds in procedures.items():
        for seed, arms in seeds.items():
            for arm, name in arms.items():
                aliases.append(dict(model=model, seed=int(seed), arm=arm, fit_id=name, atomic_fact_binding=adapter.allocations[int(seed)]['isolated_binding_sha256'] if arm.startswith('atomic_only') else None))
                if arm.startswith('atomic_only'):
                    pending.setdefault((model, store.load(arms['R_all'])['layer'] if arm.endswith('R_all') else cfg['models'][model]['fixed_layer']), {})[name] = (int(seed), 'isolated')
                elif arm.startswith('C_clean'):
                    pending.setdefault((model, int(name.split('/L')[1][:2])), {})[name] = (int(seed), 'clean')
    for (model, layer), names in sorted(pending.items()):
        features = adapter.layer(model, layer); pre = reusable.pre(model, layer, features['P15'])
        for name, (seed, mode) in sorted(names.items()):
            if mode == 'isolated':
                store.adaptation(name, model, layer, seed, features, pre, mode)
            else:
                store.clean(model, layer, features, pre, reusable)
        del features, pre; gc.collect()
    failures = [store.load(name)['id'] for name in {a['fit_id'] for a in aliases} if not store.load(name)['valid_for_scoring']]
    require(not failures, 'invalid required final control fits: ' + str(failures))
    # C_clean never enters bank membership or selection; all wording arrays are still unread.
    return dict(procedures=procedures, aliases=aliases, selection=selection, wording_used_before_lock=False,
                procedure_layer_control_policy='returned head layer; chosen/refit-before-fallback layer preserved in selection trace',
                fixed_and_S_layer_control_policy='raw full-allocation repair controls with explicit retention; only CV-selected R_all procedures apply inherited refit fallback')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['validate', 'run'])
    parser.add_argument('--expected-commit'); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = configuration()
    require(platform.system() == 'Darwin', 'approved CPU host is macOS; RSS units bytes')
    acceptance = read(ROOT / RECOVER / 'acceptance.json')
    verify_current_inputs(acceptance, read(ROOT / 'config/checkpoint_r2/fresh_recoverability_v1.json'))
    reusable = Reusable(); adapter = B25Adapter(acceptance)
    require(reusable.execution['config']['local_packages'] == cfg['local_packages'], 'reuse numerical runtime differs')
    require(reusable.completion['inventory_sha256'] == cfg['inventory_sha256'] and reusable.completion['input_manifest_sha256'] == hash_value(adapter.manifest), 'reuse representation differs')
    if args.mode == 'validate':
        reuses = validate_reuse(adapter, reusable, cfg)
        save(args.output, dict(status='focused_input_adapter_reuse_validation_passed_no_real_fits', host=platform.node(), interpreter=sys.executable, python=sys.version,
             configuration_sha256=file_hash(ROOT / CONFIG), recoverability_producer=RECOVER_SHA, recoverability_artifacts=len(reusable.completion['artifacts']), bank_candidates=len(bank_specs()), exact_bank_reuses=reuses, adapter=adapter.receipt))
        return
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(sha == args.expected_commit and len(sha) == 40, 'exact committed B25 producer required')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).strip(), 'clean committed configuration/code required')
    destination = args.output.resolve()
    require(not destination.is_relative_to(ROOT) and not destination.is_relative_to(Path(acceptance['handoff'])) and not destination.is_relative_to(reusable.root), 'fresh separate derived directory required')
    require(not destination.exists() or args.resume, 'refuse output overwrite')
    destination.mkdir(parents=True, exist_ok=args.resume)
    identity = dict(producer_sha=sha, configuration_sha256=file_hash(ROOT / CONFIG), inventory_sha256=cfg['inventory_sha256'], representation_manifest_sha256=hash_value(adapter.manifest),
                    recoverability_producer_sha=RECOVER_SHA, recoverability_completion_sha256=pin(reusable.root / 'completion.json')['sha256'])
    checked_save(destination / 'execution.json', dict(identity, config=cfg, host=platform.node(), platform=platform.platform(), interpreter=sys.executable, python=sys.version,
        code_hashes={p: file_hash(ROOT / p) for p in cfg['code_paths']}, adapter=adapter.receipt))
    started = time.monotonic(); store = FitStore(destination, identity, adapter)
    with threadpool_limits(limits=1):
        require(all(p['num_threads'] == 1 for p in threadpool_info()), 'one BLAS/OpenMP thread required')
        if not (destination / 'threads.json').exists():
            save(destination / 'threads.json', dict(identity, pid=os.getpid(), processes=1, pools=threadpool_info()))
        bank, folds = stage_one(adapter, reusable, store, cfg)
        checked_save(destination / 'bank.json', dict(identity, declared=600, records=bank))
        checked_save(destination / 'folds.json', dict(identity, planned=900, records=folds))
        invalid = [r['id'] for r in bank + folds if not r['valid_for_scoring']]
        if invalid:
            checked_save(destination / 'blocked.json', dict(identity, status='validity_blocked', blockers=invalid, primary_comparison='blocked; complete bank/fold candidate inventory retained; no silent restrictions'))
            print('BLOCKED', invalid, flush=True); return
        selection = select_procedures(bank, folds, cfg, reusable)
        locked = dict(identity, **refits_and_lock(adapter, store, reusable, selection, cfg))
        checked_save(destination / 'locked.json', locked)
        evaluation = evaluate_locked(adapter, store, destination, identity, locked, reusable, cfg['bootstrap'], read(ROOT / cfg['precision_support']))
    adapter.unchanged(); reusable.unchanged()
    records = [compact_record(read(path)) for path in sorted((destination / 'fits').glob('*/fit.json'))]
    checked_save(destination / 'fit_inventory.json', dict(identity, records=records, planned_bank=600, planned_folds=900, bank_reused=62,
        family_counts={f: sum(r['family'] == f for r in records) for f in sorted({r['family'] for r in records})}))
    artifacts = {str(p.relative_to(destination)): pin(p) for p in sorted(destination.rglob('*')) if p.is_file() and p.name != 'completion.json'}
    save(destination / 'completion.json', dict(identity, status='completed', runtime_seconds=time.monotonic() - started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, fit_records=len(records), fit_failures=[r['id'] for r in records if not r['valid_for_scoring']],
        backup_unchanged=True, recoverability_unchanged=True, historical_reuse=False, E_evaluated=False, behavior_constituents_interventions_evaluated=False,
        evaluation=evaluation, artifacts=artifacts))
    print('COMPLETED', destination, flush=True)


if __name__ == '__main__':
    main()
