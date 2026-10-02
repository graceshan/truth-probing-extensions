"""Independent ROC-integration and saved-draw verification of precision results.

Reads saved development scores only. Reuses input/graph validation, but computes
AUROC by trapezoidal ROC integration rather than the analysis pair-count code.
"""
import io
import json
import subprocess
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from src import selection_repair_precision_proxy_v1 as p


def roc_integral(labels, scores, weights):
    """Independent tied-threshold trapezoidal ROC, with undefined classes as NaN."""
    order = np.argsort(-np.asarray(scores), kind='stable')
    y = np.asarray(labels)[order]
    w = np.asarray(weights, float)[:, order]
    ends = np.r_[np.flatnonzero(np.diff(np.asarray(scores)[order])), len(order) - 1]
    tp = np.cumsum(w * y, axis=1)[:, ends]
    fp = np.cumsum(w * (1 - y), axis=1)[:, ends]
    with np.errstate(divide='ignore', invalid='ignore'):
        tpr = np.column_stack([np.zeros(len(w)), tp / tp[:, -1:]])
        fpr = np.column_stack([np.zeros(len(w)), fp / fp[:, -1:]])
    return np.sum(np.diff(fpr, axis=1) * (tpr[:, 1:] + tpr[:, :-1]) / 2, axis=1)


def independent_evaluate(labels, scores, idx, weights, base):
    point = roc_auc_score(labels[idx], scores[idx], sample_weight=base[idx])
    draws = np.concatenate([roc_integral(labels[idx], scores[idx], weights[i:i + 64, idx] * base[idx])
                            for i in range(0, len(weights), 64)])
    return float(point), draws


def linear_quantile(values, probability):
    x = np.sort(values)
    index = (len(x) - 1) * probability
    lo, hi = int(np.floor(index)), int(np.ceil(index))
    return float(x[lo] + (index - lo) * (x[hi] - x[lo]))


def validate(root, output):
    config = p.config_at(root)
    completion = json.loads((output / 'completion.json').read_text())
    for name, pin in completion['artifacts'].items():
        p.read_bound(output / name, pin)
    for name, pin in completion['external'].items():
        raw = p.read_bound(name, pin)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        actual = archive['delta']
    records = json.loads((output / 'results.json').read_text())
    frames, scores, diagnostics, _ = p.load_inputs(root, config)
    weights, schedule = p.schedules(frames['atomic'], frames['bare'], config['bootstrap'])
    p.require(schedule == json.loads((output / 'bootstrap_binding.json').read_text()), 'schedule mismatch')
    p.require(diagnostics == json.loads((output / 'support.json').read_text()), 'support metadata mismatch')
    prior = {}
    prior_pins = {}
    for cohort, spec in config['inputs'].items():
        directory = Path(spec['root'])
        manifest = json.loads((root / spec['completion']['path']).read_text())
        metric_bytes = p.read_bound(directory / 'metrics.json', manifest['artifacts']['metrics.json'])
        draw_bytes = p.read_bound(directory / 'scores/paired_draws.npz', manifest['artifacts']['scores/paired_draws.npz'])
        metrics = json.loads(metric_bytes)
        with np.load(io.BytesIO(draw_bytes), allow_pickle=False) as archive:
            values = archive[cohort]
        index = {}
        for i, r in enumerate(metrics):
            if cohort == 'P10' and r['baseline'] != 'full':
                continue
            key = (r['model'], r['layer'], r['method'], r['metric'], r['scope'], r['topic'])
            p.require(key not in index, 'duplicate prior metric identity')
            index[key] = (r[cohort + '_auroc'], values[:, i])
        prior[cohort] = index
        prior_pins[cohort] = {name: manifest['artifacts'][name] for name in ['metrics.json', 'scores/paired_draws.npz']}
    cache = {}
    max_auc_error = 0.0
    max_prior_error = 0.0
    max_interval_error = 0.0
    checked_values = 0
    native_count = 0
    projected_count = 0
    for r in records:
        kind = 'atomic' if r['endpoint'] == 'atomic_auroc' else 'bare'
        scope, topic = r['scope'], r['topic']
        if r['status'] == 'unsupported':
            p.require(r['draw_column'] is None and r['total_replicates'] == 0 and
                      not p.projection_supported(kind, scope, topic, diagnostics)[0], 'unsupported projection reported as data')
            continue
        cohort, model, layer, method = r['cohort'], r['model'], r['saved_layer'], r['method']
        # Existing full/P15/P10 archived metric draws provide exact independent
        # checkpoint parity for every native contrast and every replicate.
        keys = [(model, layer, m, r['endpoint'], scope, topic) for m in [method, r['reference']]]
        (point, method_draws), (ref_point, ref_draws) = [prior[cohort][k] for k in keys]
        if r['precision_mode'] == 'native_D':
            expected = method_draws - ref_draws
            native_count += 1
        elif kind == 'bare' or scope != 'pooled':
            expected = method_draws - ref_draws
            count_key = 'entities' if kind == 'bare' else 'persons'
            table = 'graph' if kind == 'bare' else 'atomic'
            scale = np.sqrt(diagnostics['D_' + table][topic][count_key] / diagnostics['E_' + table][topic][count_key]) if scope == 'topic' else 1
            expected = (point - ref_point) + scale * (expected - (point - ref_point))
            projected_count += 1
        else:
            expected = None
            projected_count += 1
        if expected is not None:
            got = actual[:, r['draw_column']]
            p.require(np.array_equal(np.isnan(got), np.isnan(expected)), 'prior draw undefined-pattern mismatch')
            err = float(np.nanmax(np.abs(got - expected)))
            max_prior_error = max(max_prior_error, err)
            p.require(err < 2e-14, 'saved checkpoint paired draws do not align')
        # Independently recompute individual metrics from row scores using
        # sklearn for points and tied-threshold trapezoid integration for ALL draws.
        values = []
        for method_name in [method, r['reference']]:
            key = (cohort, model, method_name, r['endpoint'], scope, topic, r['precision_mode'])
            if key not in cache:
                definition = next(x for x in config['endpoints'] if x['id'] == r['endpoint'])
                frame = frames[kind]
                if kind == 'atomic':
                    mask = np.ones(len(frame), bool)
                    labels = frame.label.to_numpy(int)
                else:
                    mask = ((frame.operator.to_numpy() == definition['operator']) &
                            frame.cell.isin(definition['positive'] + definition['negative']).to_numpy())
                    labels = frame.cell.isin(definition['positive']).to_numpy(int)
                score = scores[(cohort, model, method_name)][kind]
                base = np.ones(len(frame))
                if r['precision_mode'] != 'native_D' and kind == 'atomic' and scope == 'pooled':
                    base = np.array([diagnostics['E_atomic'][t]['rows'] / diagnostics['D_atomic'][t]['rows'] for t in frame.topic])
                topics = config['topics'] if scope == 'topic_macro' else [topic]
                components = []
                for t in topics:
                    selected = mask if t == 'all' else mask & (frame.topic.to_numpy() == t)
                    idx = np.flatnonzero(selected)
                    components.append(independent_evaluate(labels, score, idx, weights[kind], base))
                cache[key] = (float(np.mean([v[0] for v in components])),
                              np.mean([v[1] for v in components], axis=0))
            values.append(cache[key])
        (a, ad), (b, bd) = values
        center = a - b
        expected = center + r['transport_scale'] * ((ad - bd) - center)
        got = actual[:, r['draw_column']]
        p.require(np.array_equal(np.isnan(got), np.isnan(expected)), 'independent undefined-pattern mismatch')
        err = float(np.nanmax(np.abs(got - expected)))
        max_auc_error = max(max_auc_error, err, abs(center - r['delta_D_proxy']))
        p.require(err < 2e-13 and abs(center - r['delta_D_proxy']) < 2e-13, 'independent ROC mismatch')
        checked_values += len(got)
        valid = got[np.isfinite(got)]
        p.require(len(valid) == r['valid_replicates'] and len(got) - len(valid) == r['invalid_replicates'], 'invalid count mismatch')
        if len(valid) >= config['bootstrap']['minimum_valid']:
            lo, hi = [linear_quantile(valid, q) for q in (.025, .975)]
            err = max(abs(lo - r['ci_low']), abs(hi - r['ci_high']), abs((hi - lo) / 2 - r['half_width']))
            max_interval_error = max(max_interval_error, err)
            p.require(err < 2e-14, 'independent percentile mismatch')
        else:
            p.require(r['ci_low'] is None and r['ci_status'] == 'insufficient_valid_replicates', 'undefined CI silently accepted')
    p.require(native_count == 168 and projected_count == 120 and len(records) == 336, 'configuration coverage mismatch')
    changed = subprocess.check_output(['git', 'diff', config['reviewed_start'], '--name-status'], cwd=root, text=True)
    p.require(all(line.startswith('A\t') for line in changed.splitlines()), 'tracked historical file changed')
    report = dict(status='passed', native_records=168, approximate_records=120, unsupported_projection_records=48,
                  independent_draw_values_checked=checked_values, max_independent_roc_error=max_auc_error,
                  max_archived_draw_error=max_prior_error, max_independent_percentile_error=max_interval_error,
                  all_input_and_output_hashes_verified=True, all_current_tracked_changes_are_new_paths=True,
                  no_fit_or_E_score_access=True, P_A_frozen=False, prior_draw_bindings=prior_pins,
                  validator_code=p.binding(Path(__file__).read_bytes()))
    p.dump(output / 'validation.json', report)
    return report
