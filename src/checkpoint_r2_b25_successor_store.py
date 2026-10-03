"""Hash-verified successor checkpoints with explicit original-parameter lineage."""
import shutil
import time
import traceback
from pathlib import Path
import numpy as np
from src import checkpoint_r2_logistic_continuation as successor
from src.checkpoint_r2_b25_store import FitStore, RECOVER_SHA, read, array_hash, npz
from src.checkpoint_r2_fresh_inputs import hash_value, require
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_fresh_recoverability import save

ORIGINAL_SHA = 'b25581ca9c22d93c8c988c93c0c5491219d6cfec'


class InvalidFit(ValueError):
    pass


class SuccessorStore(FitStore):
    def __init__(self, root, identity, adapter, original, gate):
        super().__init__(root, identity, adapter)
        self.original, self.gate = Path(original), Path(gate)
        self.original_records = {read(p)['id']: (p, read(p)) for p in (self.original/'fits').glob('*/fit.json')}
        self.continued = {r['candidate_id']: r for r in read(self.gate/'completion.json')['records']}
        require(len(self.continued) == 4 and all(r['valid_for_scoring'] for r in self.continued.values()), 'all four continuation gates required')

    def import_valid(self):
        imported = []
        for name, (path, source) in sorted(self.original_records.items()):
            if not source['valid_for_scoring']:
                continue
            require(source['producer_sha'] == ORIGINAL_SHA, 'original producer mismatch')
            for key in ('inventory_sha256', 'representation_manifest_sha256', 'recoverability_producer_sha', 'recoverability_completion_sha256'):
                require(source[key] == self.identity[key], 'incompatible input: '+key)
            for key, filename in [('parameters', 'parameters.npz'), ('scores', 'scores.npz')]:
                require(pin(path.parent/filename) == source[key], 'corrupt original '+filename)
            record = dict(source, **self.identity,
                parameter_producer_sha=source.get('parameter_producer_sha', ORIGINAL_SHA),
                imported_valid_without_refit=True, imported_source_root=str(self.original),
                imported_source_fit=pin(path), imported_source_producer_sha=ORIGINAL_SHA,
                imported_source_configuration_sha256=source['configuration_sha256'],
                original_seconds=source.get('seconds'), seconds=0.0)
            destination = self.folder(name)
            if self.load(name) is not None:
                require(self.load(name) == record, 'import checkpoint changed')
            else:
                require(not destination.exists(), 'partial import directory must be preserved and investigated')
                destination.mkdir(parents=True)
                for filename in ('parameters.npz', 'scores.npz'):
                    shutil.copyfile(path.parent/filename, destination/filename)
                save(destination/'fit.json', record)
            self.scores(name)  # Strict identity, hashes and ordered packet checks.
            imported.append(dict(id=name, source_fit=pin(path), source_parameters=source['parameters'],
                                 parameter_producer_sha=record['parameter_producer_sha']))
        require(len(imported) == 68, 'all 68 valid original checkpoints required')
        return imported

    def fresh_or_continued(self, spec, X, rows):
        name = spec['candidate_id']
        if name not in self.continued:
            params, detail = successor.fit(spec['method'], X, [int(r['label']) for r in rows], rows, spec['C'], spec['saved_layer'])
            return params, detail, dict(parameter_producer_sha=self.identity['producer_sha'], numerical_lineage='new_successor_fit')
        entry = self.continued[name]
        require(entry['producer_sha'] == self.identity['producer_sha'] and entry['configuration_sha256'] == self.identity['configuration_sha256'], 'gate producer/config mismatch')
        require(entry['training_feature_sha256'] == array_hash(X) and entry['training_source_rows_sha256'] == hash_value(rows), 'continued training binding mismatch')
        require(entry['C'] == spec['C'], 'continued C mismatch')
        path = self.gate/entry['folder']
        require(pin(path/'parameters.npz') == entry['parameters'] and pin(path/'continuation.json') == entry['receipt'], 'gate corruption')
        receipt = read(path/'continuation.json')
        return npz(path/'parameters.npz'), receipt['optimizer'], dict(
            parameter_producer_sha=self.identity['producer_sha'], numerical_lineage='continued_original_invalid',
            original_parameter_producer_sha=ORIGINAL_SHA, original_source_fit=receipt['source_fit'],
            original_source_parameters=receipt['source_parameters'], original_optimizer=receipt['original_optimizer'],
            continuation_receipt=entry['receipt'], continuation_seconds=receipt['seconds'])

    def bank(self, spec, features, pre, reusable):
        name = spec['candidate_id']; existing = self.load(name)
        group = 'balanced' if spec['method'] in ('burger_t_g', 'ttpd') else 'P15'
        X = features['P15'][self.adapter.balanced] if group == 'balanced' else features['P15']
        rows = self.adapter.balanced_rows if group == 'balanced' else self.adapter.rows['P15']
        exposure = dict(group=group, rows=len(rows), source_rows_sha256=hash_value([self.adapter.rows['P15'][i] for i in self.adapter.balanced]) if group == 'balanced' else hash_value(rows), feature_float64_sha256=array_hash(X))
        if existing:
            require(existing['training'] == exposure and existing['spec'] == spec, 'resumed bank exposure/settings mismatch')
            if not existing['valid_for_scoring']:
                raise InvalidFit('persistent bank failure: '+name)
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
                params, detail, lineage = self.fresh_or_continued(spec, X, rows)
                metadata.update(lineage)
                valid = detail['valid_for_scoring'] and all(np.isfinite(v).all() for v in params.values() if np.asarray(v).dtype.kind not in 'US')
            metadata.update(valid_for_scoring=bool(valid), optimizer=detail)
            if not valid:
                metadata['failure'] = 'reviewed fit validity policy failed; no predictions admitted'
        except Exception as exc:
            metadata.update(valid_for_scoring=False, failure=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
        metadata['seconds'] = time.monotonic() - start
        record = self.publish(name, metadata, params, features, source)
        if not record['valid_for_scoring']:
            raise InvalidFit('persistent bank failure: '+name)
        return record

    def adaptation(self, *args, **kwargs):
        record = super().adaptation(*args, **kwargs)
        if record['family'] == 'repair_fold' and not record['valid_for_scoring']:
            raise InvalidFit('persistent repair-fold failure: '+record['id'])
        return record
