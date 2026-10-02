"""Read-only independent audit of the bounded B25 validity halt; no refitting.

Usage: python scripts/checkpoint_r2_audit_b25_halt.py CHECKOUT RUN NEW_RECEIPT
"""
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
from collections import Counter

import numpy as np
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

wt, run, out = map(Path, sys.argv[1:])
sys.path.insert(0, str(wt))
from src.checkpoint_r2_b25_adapter import B25Adapter
from src.checkpoint_r2_fresh_inputs import hash_value, file_hash
from src.checkpoint_r2_fresh_store import pin

read = lambda p: json.loads(p.read_text())
started = time.monotonic()
execution = read(run / 'execution.json')
producer = execution['producer_sha']
assert producer == 'b25581ca9c22d93c8c988c93c0c5491219d6cfec'
assert execution['configuration_sha256'] == file_hash(wt / 'config/checkpoint_r2/fresh_b25_v1.json')
for name, sha in execution['code_hashes'].items():
    assert file_hash(wt / name) == sha, name
adapter = B25Adapter(read(wt / 'results/checkpoint_r2_fresh_recoverability_20261002/acceptance.json'))
paths = {read(p)['id']: p for p in (run / 'fits').glob('*/fit.json')}
fits = {name: read(path) for name, path in paths.items()}
assert len(fits) == 72
assert Counter(r['family'] for r in fits.values()) == {'bank': 30, 'repair_fold': 42}
assert sum(r.get('reused', False) for r in fits.values()) == 3
for r in fits.values():
    assert r['producer_sha'] == producer
    folder = paths[r['id']].parent
    assert pin(folder / 'parameters.npz') == r['parameters']
    if r['valid_for_scoring']:
        assert pin(folder / 'scores.npz') == r['scores']
        if r['scoring_method'] == 'r0':
            assert r['optimizer']['attempts'][-1]['gradient_infinity_norm'] <= 1e-4
        elif r['scoring_method'] == 'l2_logistic':
            assert r['optimizer']['raw_gradient_target_met']
        elif r['scoring_method'] == 'ttpd':
            assert r['optimizer']['polarity_optimizer']['valid_for_scoring']
            assert r['optimizer']['head_optimizer']['valid_for_scoring']
    else:
        assert not (folder / 'scores.npz').exists(), 'invalid head was scored'
        assert r['scoring_method'] == 'l2_logistic'
        assert not r['optimizer']['retry_needed'] and not r['optimizer']['initial_convergence_warning']
        assert r['optimizer']['finite'] and r['optimizer']['final_converged']
        assert r['optimizer']['gradient_infinity_norm'] > 1e-4
failures = []
score_checks = 0
metric_checks = 0
fold_checks = 0
feature_checks = 0

def equal(a, b):
    assert math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12), (a, b)

with threadpool_limits(limits=1):
    for model, layer in sorted({(r['model'], r['layer']) for r in fits.values()}):
        X = adapter.layer(model, layer)
        mean = X['P15'].mean(0); std = X['P15'].std(0, ddof=0)
        constant = np.all(X['P15'] == X['P15'][0], axis=0) | (std == 0); std[constant] = 0
        z = {}
        for group in X:
            zz = np.zeros_like(X[group]); active = ~constant
            zz[:, active] = (X[group][:, active] - mean[active]) / np.maximum(std[active], 1e-6)
            z[group] = zz
        for name, r in fits.items():
            if (r['model'], r['layer']) != (model, layer): continue
            folder = paths[name].parent
            with np.load(folder / 'parameters.npz', allow_pickle=False) as saved:
                params = {k: saved[k].copy() for k in saved.files}
            if r['family'] == 'bank':
                train = X['P15'][adapter.balanced] if r['training']['group'] == 'balanced' else X['P15']
            else:
                fold = adapter.allocations[r['seed']]['folds'][r['fold']]
                group = f"B25_{r['seed']}"
                train = X[group][fold['train']]
                assert hash_value([adapter.bindings[group][i] for i in fold['train']]) == r['training']['ordered_bindings_sha256']
                assert r['P15_bindings_sha256'] == hash_value(adapter.bindings['P15'])
                assert hashlib.sha256(X['P15'].tobytes()).hexdigest() == r['P15_feature_sha256']
            assert hashlib.sha256(train.tobytes()).hexdigest() == r['training']['feature_float64_sha256']; feature_checks += 1
            if not r['valid_for_scoring']:
                C = r['spec']['C']; y = np.array([int(row['label']) for row in adapter.rows['P15']]); n = len(y)
                logits = X['P15'] @ params['coef'] + params['intercept']
                residual = expit(logits) - y
                gradient = np.r_[X['P15'].T @ residual / n + params['coef'] / (C * n), residual.mean()]
                value = np.logaddexp(0, np.where(y == 1, -logits, logits)).mean() + params['coef'] @ params['coef'] / (2 * C * n)
                norm = float(np.max(np.abs(gradient)))
                equal(norm, r['optimizer']['gradient_infinity_norm']); equal(value, r['optimizer']['objective'])
                assert norm > 1e-4
                failures.append(dict(candidate_id=name, gradient_infinity_norm=norm, required_maximum=1e-4, ratio_to_target=norm / 1e-4,
                    objective=float(value), library_converged=True, convergence_warning=False, retry_permitted=False,
                    source_fit_sha256=pin(paths[name])['sha256'], parameter_sha256=r['parameters']['sha256'], failed_scores_absent=True))
                continue
            with np.load(folder / 'scores.npz', allow_pickle=False) as data:
                scores = {g: data[g].copy() for g in data.files}
            for group, values in scores.items():
                packet = r['score_bindings'][group]
                assert packet['row_binding_sha256'] == hash_value(adapter.bindings[group])
                assert packet['ordered_keys'] == [[b['group'], b['logical_id']] for b in adapter.bindings[group]]
                method = r['scoring_method']
                if method == 'r0':
                    np.testing.assert_array_equal(params['mean'], mean); np.testing.assert_array_equal(params['population_std'], std)
                    np.testing.assert_array_equal(params['constant_mask'], constant); np.testing.assert_array_equal(params['denominator'], np.maximum(std, 1e-6))
                    prediction = z[group] @ params['standardized_coef'] + params['standardized_intercept']
                elif method == 'ttpd':
                    prediction = np.column_stack((X[group] @ params['t_g'], X[group] @ params['polarity_coef'])) @ params['head_coef'] + params['head_intercept']
                else:
                    prediction = X[group] @ params['coef'] + params['intercept']
                np.testing.assert_array_equal(prediction, values); score_checks += 1
            for metric in r['metrics']:
                group = 'atomic_D' if metric['metric'] == 'atomic_auroc' else 'D_bare'
                rows = adapter.rows[group]; y = np.array([int(row['label']) for row in rows]); topics = np.array([row['topic'] for row in rows])
                if group == 'atomic_D': mask = np.ones(len(rows), bool)
                else:
                    op = np.array([row['operator'] for row in rows]); cell = np.array([int(row['truth_a']) + int(row['truth_b']) for row in rows])
                    mask = (op == 'AND') if metric['metric'] == 'AND_auroc' else (op == 'OR')
                    if metric['metric'] == 'OR_mixed_vs_FF_auroc': mask &= cell < 2
                if metric['scope'] == 'topic': mask &= topics == metric['topic']
                if metric['scope'] == 'topic_macro':
                    actual = np.mean([roc_auc_score(y[mask & (topics == t)], scores[group][mask & (topics == t)]) for t in sorted(set(topics))])
                else: actual = roc_auc_score(y[mask], scores[group][mask])
                equal(actual, metric['point']); metric_checks += 1
            if r['family'] == 'repair_fold':
                held = fold['heldout']; rows = adapter.rows[group := f"B25_{r['seed']}"]
                op = np.array([rows[i]['operator'] for i in held]); cell = np.array([int(rows[i]['truth_a']) + int(rows[i]['truth_b']) for i in held])
                y = np.array([int(rows[i]['label']) for i in held]); values = scores[group][held]
                mask = (op == 'OR') & (cell < 2)
                actual = roc_auc_score(y[mask], values[mask])
                fold_checks += 1
                assert 0 <= actual <= 1
        print('independently audited halted-run layer', model, layer, flush=True)
assert len(failures) == 4
adapter.unchanged()
assert not (run / 'locked.json').exists() and not (run / 'evaluations').exists()
receipt = dict(status='independent_audit_confirms_frozen_validity_blocker', producer_sha=producer,
    configuration_sha256=execution['configuration_sha256'], hostname=platform.node(), interpreter=sys.executable, python=sys.version,
    scope='read-only reconstruction of all published valid checkpoint scores and all failed raw LR training-gradient diagnostics; no new fitting or failed-head D scoring',
    fit_receipts_audited=len(fits), valid_fits=68, invalid_fits=4, source_feature_hashes_verified=feature_checks,
    score_vectors_exactly_recomputed=score_checks, independent_sklearn_point_checks=metric_checks, heldout_fold_boundary_checks=fold_checks,
    failures=sorted(failures, key=lambda r: r['candidate_id']), primary_B25_metrics_and_contrasts='not computed; incomplete bank and failed frozen convergence checks',
    wording_arrays_unread_and_unscored=True, no_selector_or_refit_lock=True, audit_seconds=time.monotonic() - started,
    audit_script_sha256=file_hash(Path(__file__)))
assert not out.exists();out.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');print(json.dumps(receipt,indent=2),flush=True)
