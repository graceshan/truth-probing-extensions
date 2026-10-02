"""Hash-bound reusable fresh heads and restartable B25 per-fit checkpoints."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np

from src import selection_repair_objectives as objective
from src import selection_repair_capacity_methods_v1 as methods
from src.checkpoint_r2_fresh_inputs import ROOT, hash_value, require
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_fresh_adapter import check_packet
from src.checkpoint_r2_fresh_recoverability import save, finite, point_metrics, separation

RECOVER = 'results/checkpoint_r2_fresh_recoverability_20261002'
RECOVER_SHA = '3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f'


def read(path):
    return json.loads(Path(path).read_text())


def array_hash(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k].copy() for k in data.files}


def normalized_parameters(w, intercept, pre):
    active = ~pre.constant_mask
    coef = np.zeros(len(w)); coef[active] = w[active] / pre.denominator[active]
    return dict(coef=coef, intercept=np.array(intercept - pre.mean @ coef),
                standardized_coef=w, standardized_intercept=np.array(intercept),
                mean=pre.mean, population_std=pre.population_std, denominator=pre.denominator, constant_mask=pre.constant_mask)


class Reusable:
    def __init__(self):
        package = ROOT / RECOVER
        delivery = read(package / 'delivery.json')
        self.root = Path(delivery['external_results'])
        require(pin(self.root / 'completion.json')['sha256'] == delivery['external_completion_sha256'], 'recoverability completion changed')
        require((package / 'completion.json').read_bytes() == (self.root / 'completion.json').read_bytes(), 'committed/external completion mismatch')
        self.completion = read(self.root / 'completion.json')
        require(self.completion['producer_sha'] == RECOVER_SHA and self.completion['valid_fits'] == 122 and not self.completion['failures'], 'recoverability producer/status')
        for name, identity in self.completion['artifacts'].items():
            require(not Path(name).is_absolute() and '..' not in Path(name).parts, 'unsafe completion path')
            require(pin(self.root / name) == identity, 'reusable artifact hash: ' + name)
        self.execution = read(self.root / 'execution.json')
        self.records = {r['id']: r for r in read(self.root / 'fit_inventory.json')['records']}
        for r in self.records.values():
            require(r['producer_sha'] == RECOVER_SHA and r['valid_for_scoring'] and not r['historical_parameters_reused'], 'ineligible head producer')
        self.stats = {p: (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) for p in self.root.rglob('*') if p.is_file()}

    def pre(self, model, layer, X):
        path = self.root / 'preprocessing' / model / f'L{layer:02d}.npz'
        data = npz(path); actual = objective.fit_p_preprocessing(X)
        for name, value in data.items():
            require(np.array_equal(value, getattr(actual, name)), 'P-only preprocessing differs: ' + name)
        return actual

    def source(self, model, layer, family):
        return self.records.get(f'{model}_{family}_L{layer:02d}')

    def head(self, model, layer, family, pre):
        source = self.source(model, layer, family)
        require(source is not None, 'reusable fresh head missing')
        require(source['inventory_sha256'] == self.completion['inventory_sha256'] and source['input_manifest_sha256'] == self.completion['input_manifest_sha256'], 'representation identity')
        data = npz(self.root / 'fits' / source['id'] / 'head.npz')
        if family == 'reduced_LR':
            params = data
        else:
            require(source['preprocessing_group'] == 'P15' and source['optimizer']['settings'] == asdict(objective.OptimizerSettings()), 'reused numerical settings')
            require(source['objective'] == ('mean P15 BCE + .001 ||w||^2' if family == 'R0' else 'mean TC BCE + .001 ||w||^2'), 'reused objective')
            params = normalized_parameters(data['standardized_coef'], data['standardized_intercept'], pre)
        return params, source

    def unchanged(self):
        require(all((p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns) == stat for p, stat in self.stats.items()), 'recoverability run was modified')


class FitStore:
    def __init__(self, root, identity, adapter):
        self.root, self.identity, self.adapter = Path(root), identity, adapter

    def folder(self, name):
        return self.root / 'fits' / hash_value(name)[:24]

    def load(self, name):
        folder = self.folder(name); path = folder / 'fit.json'
        if not path.exists():
            return None
        record = read(path)
        require(record['id'] == name and all(record[k] == self.identity[k] for k in self.identity), 'checkpoint producer/config/input changed')
        if 'parameters' in record:
            require(pin(folder / 'parameters.npz') == record['parameters'], 'checkpoint parameter corruption')
        if 'scores' in record:
            require(pin(folder / 'scores.npz') == record['scores'], 'checkpoint score corruption')
        return record

    def parameters(self, name):
        record = self.load(name)
        require(record is not None and record['valid_for_scoring'], 'invalid completed fit')
        return npz(self.folder(name) / 'parameters.npz')

    def scores(self, name):
        record = self.load(name)
        require(record is not None and record['valid_for_scoring'], 'invalid completed fit')
        data = npz(self.folder(name) / 'scores.npz')
        return {g: check_packet(self.adapter.bindings[g], dict(record['score_bindings'][g], scores=s)) for g, s in data.items()}

    def publish(self, name, metadata, params, features, source=None):
        folder = self.folder(name)
        if folder.exists():
            # Incomplete task-owned output is preserved rather than overwritten.
            quarantine = self.root / 'quarantine'; quarantine.mkdir(exist_ok=True)
            folder.rename(quarantine / (folder.name + '-' + str(time.time_ns())))
        folder.mkdir(parents=True)
        record = dict(self.identity, id=name, **metadata)
        if params is not None:
            np.savez(folder / 'parameters.npz', **params); record['parameters'] = pin(folder / 'parameters.npz')
        if record['valid_for_scoring']:
            groups = ['P15', 'atomic_D', 'D_bare'] + [f'B25_{s}' for s in (11, 23, 37)]
            scores = {g: methods.score(record['scoring_method'], params, features[g]) for g in groups}
            packets = {}
            for g, values in scores.items():
                packet = dict(row_binding_sha256=hash_value(self.adapter.bindings[g]), ordered_keys=[[r['group'], r['logical_id']] for r in self.adapter.bindings[g]], scores=values)
                check_packet(self.adapter.bindings[g], packet)
                packets[g] = {k: v for k, v in packet.items() if k != 'scores'}
            if source:
                old = npz(Path(source['external_folder']) / 'scores.npz')
                for g in ('P15', 'atomic_D', 'D_bare'):
                    require(np.array_equal(old[g], scores[g]), 'reused score reconstruction differs')
            np.savez(folder / 'scores.npz', **scores)
            record.update(scores=pin(folder / 'scores.npz'), score_bindings=packets, metrics=point_metrics(self.adapter, scores))
        save(folder / 'fit.json', record)
        print(json.dumps(dict(id=name, valid=record['valid_for_scoring'], reused=record.get('reused', False), seconds=record.get('seconds'), failure=record.get('failure'))), flush=True)
        return record

    def bank(self, spec, features, pre, reusable):
        name = spec['candidate_id']; existing = self.load(name)
        group = 'balanced' if spec['method'] in ('burger_t_g', 'ttpd') else 'P15'
        X = features['P15'][self.adapter.balanced] if group == 'balanced' else features['P15']
        rows = self.adapter.balanced_rows if group == 'balanced' else self.adapter.rows['P15']
        exposure = dict(group=group, rows=len(rows), source_rows_sha256=hash_value([self.adapter.rows['P15'][i] for i in self.adapter.balanced]) if group == 'balanced' else hash_value(rows), feature_float64_sha256=array_hash(X))
        if existing:
            require(existing['training'] == exposure and existing['spec'] == spec, 'resumed bank exposure/settings mismatch')
            return existing
        start = time.monotonic(); method = spec['method']; source = None; params = None
        metadata = dict(spec=spec, model=spec['model'], layer=spec['saved_layer'], family='bank', scoring_method=method, training=exposure, reused=False, historical_reuse=False)
        try:
            family = 'R0' if method == 'r0' else ('reduced_LR' if method == 'l2_logistic' and spec['saved_layer'] == reusable.execution['config']['models'][spec['model']]['canonical_layer'] and spec['C'] == reusable.execution['config']['models'][spec['model']]['C'] else None)
            if family:
                params, old = reusable.head(spec['model'], spec['saved_layer'], family, pre)
                require(old['fit_rows'] == 2778 and old['training_bindings_sha256'] == hash_value(self.adapter.bindings['P15']) and old['feature_float64_sha256'] == exposure['feature_float64_sha256'], 'exact reused P15 exposure')
                if family == 'reduced_LR':
                    require(old['C'] == spec['C'] and old['preprocessing'] == 'none', 'exact reused raw LR C/transform')
                source = dict(old, external_folder=str(reusable.root / 'fits' / old['id']))
                detail = old['optimizer']; valid = True
                metadata.update(reused=True, parameter_producer_sha=RECOVER_SHA, reuse_source_id=old['id'], reuse_source_parameters=old['parameters'], reuse_source_fit=pin(reusable.root / 'fits' / old['id'] / 'fit.json'))
            else:
                params, detail = methods.fit(method, X, [int(r['label']) for r in rows], rows, spec['C'], spec['saved_layer'])
                valid = detail['valid_for_scoring'] and all(np.isfinite(v).all() for v in params.values() if np.asarray(v).dtype.kind not in 'US')
            metadata.update(valid_for_scoring=bool(valid), optimizer=detail)
            if not valid:
                metadata['failure'] = 'reviewed fit validity policy failed; no predictions admitted'
        except Exception as exc:
            metadata.update(valid_for_scoring=False, failure=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
        metadata['seconds'] = time.monotonic() - start
        return self.publish(name, metadata, params, features, source)

    def adaptation(self, name, model, layer, seed, features, pre, mode, indices=None, fold=None):
        group = f'B25_{seed}'; rows = self.adapter.rows[group]
        if mode == 'compound':
            indices = list(range(len(rows))) if indices is None else list(indices)
            training = dict(group=group, ordered_bindings_sha256=hash_value([self.adapter.bindings[group][i] for i in indices]), feature_float64_sha256=array_hash(features[group][indices]), rows=len(indices))
        else:
            facts = sorted(self.adapter.allocations[seed]['isolated'], key=lambda r: r['fact_id'])
            indices = [r['isolated_row'] for r in facts]
            training = dict(group='TC_isolated_allocated', fact_ids=[r['fact_id'] for r in facts], feature_float64_sha256=array_hash(features['TC_isolated'][indices]), rows=100)
        existing = self.load(name)
        if existing:
            require(existing['training'] == training, 'resumed adaptation exposure differs')
            return existing
        p = objective.prepare_block(features['P15'], [int(r['label']) for r in self.adapter.rows['P15']], pre)
        start = time.monotonic(); params = None
        metadata = dict(model=model, layer=layer, family='repair_fold' if fold is not None else ('repair_refit' if mode == 'compound' else 'atomic_only'),
                        seed=seed, fold=fold, mode=mode, scoring_method='r0', training=training, reused=False,
                        P15_bindings_sha256=hash_value(self.adapter.bindings['P15']), P15_feature_sha256=array_hash(features['P15']), preprocessing='P15-only',
                        objective='0.5 mean P15 BCE + 0.5 mean allocated ' + ('compound' if mode == 'compound' else 'isolated-fact') + ' BCE + .001 ||w||^2', historical_reuse=False)
        try:
            if mode == 'compound':
                from src.checkpoint_r2_b25_adapter import compound_block
                block = compound_block(features[group][indices], [rows[i] for i in indices], pre)
            else:
                weights = objective.isolated_pair_weights([r['pair_id'] for r in facts], [r['fact_id'] for r in facts])
                block = objective.prepare_block(features['TC_isolated'][indices], [r['label'] for r in facts], pre, weights)
            head = objective.fit_readout(mode, p, pre, block)
            params = normalized_parameters(head.weights, head.intercept, pre)
            attempts = []
            for attempt in head.attempts:
                entry = asdict(attempt); entry.pop('initial_parameters'); entry.pop('final_parameters'); attempts.append(entry)
            metadata.update(valid_for_scoring=bool(head.converged), optimizer=dict(status=head.status, attempts=attempts, settings=asdict(head.settings)))
            if not head.converged:
                metadata['failure'] = 'reviewed adaptation convergence/gradient/finite policy failed'
        except Exception as exc:
            metadata.update(valid_for_scoring=False, failure=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
        metadata['seconds'] = time.monotonic() - start
        return self.publish(name, metadata, params, features)

    def clean(self, model, layer, features, pre, reusable):
        name = f'{model}/TC/L{layer:02d}/C_clean'; existing = self.load(name)
        if existing:
            return existing
        params, old = reusable.head(model, layer, 'C_clean', pre)
        require(old['fit_rows'] == 8000 and old['training_bindings_sha256'] == hash_value(self.adapter.bindings['TC_control_and_final_refit']), 'exact reused C_clean exposure')
        source = dict(old, external_folder=str(reusable.root / 'fits' / old['id']))
        metadata = dict(model=model, layer=layer, family='C_clean', scoring_method='r0', valid_for_scoring=True,
                        reused=True, parameter_producer_sha=RECOVER_SHA, reuse_source_id=old['id'], reuse_source_parameters=old['parameters'],
                        reuse_source_fit=pin(reusable.root / 'fits' / old['id'] / 'fit.json'), training_group='TC_control_and_final_refit',
                        training_bindings_sha256=old['training_bindings_sha256'], optimizer=old['optimizer'], historical_reuse=False, seconds=0.)
        return self.publish(name, metadata, params, features, source)
