"""Committed bounded training-only gate, then unchanged frozen B25 procedures."""
import argparse
import gc
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
from threadpoolctl import threadpool_info, threadpool_limits

from src import checkpoint_r2_fresh_b25 as frozen
from src import checkpoint_r2_logistic_continuation as numerical
from src.checkpoint_r2_b25_adapter import B25Adapter
from src.checkpoint_r2_fresh_adapter import FreshAdapter
from src.checkpoint_r2_b25_store import Reusable, read, npz, array_hash
from src.checkpoint_r2_b25_successor_store import SuccessorStore, InvalidFit, ORIGINAL_SHA
from src.checkpoint_r2_fresh_inputs import ROOT, file_hash, hash_value, require
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_fresh_recoverability import save, verify_current_inputs

CONFIG = 'config/checkpoint_r2/b25_convergence_fix_v1.json'


def configuration():
    cfg = read(ROOT/CONFIG)
    frozen.configuration()
    require(cfg['scientific_configuration_sha256'] == file_hash(ROOT/frozen.CONFIG), 'scientific configuration changed')
    require(cfg['continuation_settings'] == numerical.SETTINGS and cfg['total_iteration_limit'] == numerical.TOTAL_ITERATIONS, 'declared numerical settings mismatch')
    for name, sha in cfg['new_source_hashes'].items():
        require(file_hash(ROOT/name) == sha, 'successor source changed: '+name)
    diagnosis = read(ROOT/cfg['diagnosis'])
    for name, identity in diagnosis['pinned_library_sources'].items():
        require(pin(Path(name)) == identity, 'pinned installed library changed: '+name)
    return cfg


def lineage():
    delivery = read(ROOT/'results/checkpoint_r2_fresh_b25_20261002/delivery.json')
    path = Path(delivery['external_review'])/'manifest.json'
    require(pin(path) == delivery['external_manifest'], 'original halt manifest corruption')
    manifest = read(path); root = Path(manifest['root'])
    require(manifest['producer_sha'] == ORIGINAL_SHA, 'original manifest producer')
    for name, identity in manifest['artifacts'].items():
        require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe manifest path')
        require(pin(root/name) == identity, 'original artifact changed: '+name)
    original = Path(delivery['external_results'])
    execution = read(original/'execution.json')
    require(execution['producer_sha'] == ORIGINAL_SHA and execution['configuration_sha256'] == file_hash(ROOT/frozen.CONFIG), 'original producer/configuration')
    for name, sha in execution['code_hashes'].items():
        require(file_hash(ROOT/name) == sha, 'original scientific source changed: '+name)
    return original, dict(delivery_sha='60e651370e5c08635f151822073a8b9c01e092ca', original_producer_sha=ORIGINAL_SHA,
                         original_manifest=pin(path), original_artifacts_verified=len(manifest['artifacts']))


def training_layer(adapter, model, layer):
    original = adapter.bindings
    try:
        adapter.bindings = {'P15': original['P15']}
        return FreshAdapter.layer(adapter, model, layer)['P15']
    finally:
        adapter.bindings = original


def gate(adapter, original, destination, identity):
    diagnosis = read(ROOT/'results/checkpoint_r2_b25_convergence_fix_20261002/training-diagnosis.json')
    source_paths = {read(p)['id']: p for p in (original/'fits').glob('*/fit.json')}
    records = []; y = np.array([int(r['label']) for r in adapter.rows['P15']], dtype=np.int64)
    for layer in (1, 2):
        X = training_layer(adapter, 'qwen', layer)
        for expected in diagnosis['failures']:
            name = expected['candidate_id']; path = source_paths[name]; source = read(path)
            if source['layer'] != layer:
                continue
            require(pin(path) == expected['source_fit'] and source['parameters'] == expected['source_parameters'], 'failed-source identity changed')
            require(array_hash(X) == expected['feature_float64_sha256'] and array_hash(y) == expected['label_int64_sha256'], 'training feature/label binding')
            require(hash_value(adapter.rows['P15']) == expected['ordered_source_rows_sha256'] and hash_value(adapter.bindings['P15']) == expected['training_bindings_sha256'], 'training order/exposure changed')
            folder = destination/'continued'/hash_value(name)[:24]
            receipt_path = folder/'continuation.json'
            if receipt_path.exists():
                receipt = read(receipt_path)
                require(all(receipt[k] == v for k, v in identity.items()) and receipt['candidate_id'] == name, 'stale continuation checkpoint')
                require(receipt['source_fit'] == pin(path) and pin(folder/'parameters.npz') == receipt['parameters'], 'continuation checkpoint corruption')
            else:
                require(not folder.exists(), 'partial continuation files preserved; investigate before resume')
                folder.mkdir(parents=True)
                source_params = npz(path.parent/'parameters.npz')
                require(pin(path.parent/'parameters.npz') == source['parameters'], 'failed parameter corruption')
                theta = np.r_[source_params['coef'], source_params['intercept']]
                first = dict(phase='original_saved_attempt', n_iter=source['optimizer']['final_n_iter'],
                             library_success=source['optimizer']['final_converged'], termination_code=None,
                             termination_message='not retained in original receipt', **numerical.diagnostics(theta, X, y, source['spec']['C']))
                require(first['objective'] == expected['objective'] and first['gradient_infinity_norm'] == expected['gradient_infinity_norm'], 'original diagnostic mismatch')
                started = time.monotonic()
                params, optimizer = numerical.complete(X, y, source['spec']['C'], theta, [first])
                np.savez(folder/'parameters.npz', **params)
                receipt = dict(identity, candidate_id=name, C=source['spec']['C'], original_optimizer=source['optimizer'],
                    source_fit=pin(path), source_parameters=source['parameters'], source_parameter_producer_sha=ORIGINAL_SHA,
                    training_feature_sha256=array_hash(X), training_label_sha256=array_hash(y),
                    training_source_rows_sha256=hash_value(adapter.rows['P15']), training_bindings_sha256=hash_value(adapter.bindings['P15']),
                    optimizer=optimizer, valid_for_scoring=optimizer['valid_for_scoring'], parameters=pin(folder/'parameters.npz'),
                    seconds=time.monotonic()-started, scope='training-only continuation; no D/A/wording arrays or metrics')
                save(receipt_path, receipt)
            records.append(dict(identity, candidate_id=name, C=receipt['C'], folder=str(folder.relative_to(destination)),
                training_feature_sha256=receipt['training_feature_sha256'], training_source_rows_sha256=receipt['training_source_rows_sha256'],
                parameters=receipt['parameters'], receipt=pin(receipt_path), valid_for_scoring=receipt['valid_for_scoring'],
                before=receipt['optimizer']['attempts'][0], after=receipt['optimizer']['attempts'][-1],
                total_iterations=receipt['optimizer']['total_iterations'], seconds=receipt['seconds']))
            print('CONTINUATION', name, receipt['valid_for_scoring'], records[-1]['after']['gradient_infinity_norm'], flush=True)
        del X; gc.collect()
    require(len(records) == 4, 'four declared continuation gates')
    adapter.unchanged()
    result = dict(identity, status='passed' if all(r['valid_for_scoring'] for r in records) else 'persistent_failure_blocked', records=records,
                  peak_rss_bytes=read(destination/'completion.json')['peak_rss_bytes'] if (destination/'completion.json').exists() else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  scope='training-only; no validation/wording/selection')
    frozen.checked_save(destination/'completion.json', result)
    return result


def production(adapter, reusable, destination, gate_path, original, identity, cfg):
    scientific = frozen.configuration()
    store = SuccessorStore(destination, identity, adapter, original, gate_path)
    imports = store.import_valid()
    frozen.checked_save(destination/'imports.json', dict(identity, valid_without_numerical_refit=imports))
    started = time.monotonic()
    try:
        bank, folds = frozen.stage_one(adapter, reusable, store, scientific)
        frozen.checked_save(destination/'bank.json', dict(identity, declared=600, records=bank))
        frozen.checked_save(destination/'folds.json', dict(identity, planned=900, records=folds))
        require(all(r['valid_for_scoring'] for r in bank+folds), 'persistent fit failure blocks complete stage')
        selection = frozen.select_procedures(bank, folds, scientific, reusable)
        locked = dict(identity, **frozen.refits_and_lock(adapter, store, reusable, selection, scientific))
        frozen.checked_save(destination/'locked.json', locked)
        evaluation = frozen.evaluate_locked(adapter, store, destination, identity, locked, reusable, scientific['bootstrap'], read(ROOT/scientific['precision_support']))
        status, blockers = 'completed', []
    except (InvalidFit, ValueError) as exc:
        status, blockers, evaluation = 'validity_blocked', [str(exc)], None
        save(destination/'blocked.json', dict(identity, status=status, blockers=blockers, persistent_failure_gate_unchanged=True))
    adapter.unchanged(); reusable.unchanged()
    records = [frozen.compact_record(read(p)) for p in sorted((destination/'fits').glob('*/fit.json'))]
    save(destination/'fit_inventory.json', dict(identity, records=records, planned_bank=600, planned_folds=900,
        accounting=dict(imported_valid=sum(r.get('imported_valid_without_refit', False) for r in records),
                        continued_original_failures=sum(r.get('numerical_lineage') == 'continued_original_invalid' for r in records),
                        interrupted_fold_reruns=sum(r['id'] == 'qwen/B25_37/L02/repair/fold=2' for r in records),
                        total_published=len(records))))
    artifacts = {str(p.relative_to(destination)): pin(p) for p in sorted(destination.rglob('*')) if p.is_file() and p.name != 'completion.json'}
    save(destination/'completion.json', dict(identity, status=status, runtime_seconds=time.monotonic()-started,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, fit_records=len(records),
        fit_failures=[r['id'] for r in records if not r['valid_for_scoring']], blockers=blockers,
        original_published_fit_seconds=sum(r.get('original_seconds', 0) or 0 for r in records if r.get('imported_valid_without_refit')),
        continued_gate_seconds=sum(r['seconds'] for r in read(gate_path/'completion.json')['records']),
        successor_published_fit_seconds=sum(r.get('seconds', 0) for r in records if not r.get('imported_valid_without_refit')),
        cumulative_iteration_policy='per-logistic subfit <=10000 including saved original attempts; all attempts retained',
        backup_unchanged=True, recoverability_unchanged=True, historical_reuse=False, E_evaluated=False,
        evaluation=evaluation, artifacts=artifacts))
    print(status.upper(), destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['gate', 'run'])
    parser.add_argument('--expected-commit', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gate', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); cfg = configuration(); original, lineage_receipt = lineage()
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    require(sha == args.expected_commit and len(sha) == 40, 'exact committed successor required')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=ROOT, text=True).strip(), 'clean committed successor required')
    require(platform.system() == 'Darwin', 'approved macOS CPU host required')
    acceptance = read(ROOT/frozen.RECOVER/'acceptance.json')
    verify_current_inputs(acceptance, read(ROOT/'config/checkpoint_r2/fresh_recoverability_v1.json'))
    adapter = B25Adapter(acceptance)
    destination = args.output.resolve()
    require(not destination.is_relative_to(ROOT) and not destination.is_relative_to(Path(acceptance['handoff'])) and
            not destination.is_relative_to(original) and not destination.is_relative_to(Path(acceptance['output'])), 'fresh separate output required')
    require(not destination.exists() or args.resume, 'refuse output overwrite')
    destination.mkdir(parents=True, exist_ok=args.resume)
    identity = dict(producer_sha=sha, configuration_sha256=file_hash(ROOT/CONFIG), inventory_sha256=acceptance['inventory_sha256'],
        representation_manifest_sha256=hash_value(adapter.manifest), recoverability_producer_sha=frozen.RECOVER_SHA,
        recoverability_completion_sha256=pin(Path(read(ROOT/frozen.RECOVER/'delivery.json')['external_results'])/'completion.json')['sha256'])
    frozen.checked_save(destination/'execution.json', dict(identity, config=cfg, scientific_config=frozen.configuration(),
        lineage=lineage_receipt, host=platform.node(), platform=platform.platform(), interpreter=sys.executable, python=sys.version,
        code_hashes={p:file_hash(ROOT/p) for p in cfg['code_paths']}, mode=args.mode))
    with threadpool_limits(limits=1):
        require(all(p['num_threads'] == 1 for p in threadpool_info()), 'one BLAS/OpenMP thread required')
        if not (destination/'threads.json').exists():
            save(destination/'threads.json', dict(identity, pid=os.getpid(), processes=1, pools=threadpool_info()))
        if args.mode == 'gate':
            gate(adapter, original, destination, identity)
        else:
            require(args.gate is not None, 'verified training-only gate required')
            completion = read(args.gate/'completion.json')
            require(completion['status'] == 'passed' and all(completion[k] == v for k,v in identity.items()), 'continuation gate identity/status mismatch')
            production(adapter, Reusable(), destination, args.gate, original, identity, cfg)
    # Rehash the original package after every successor stage; original lineage is immutable.
    lineage()


if __name__ == '__main__':
    main()
