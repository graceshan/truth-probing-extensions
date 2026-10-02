"""Bound B25 endpoint metrics and shared-draw metric averaging; no selection."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.checkpoint_r2_fresh_recoverability import finite, save, separation
from src.checkpoint_r2_fresh_inputs import hash_value, require
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_selection_v1 import average_sample_metrics, TOPICS
from src.clean_transfer_statistics import weighted_auc, interval
from src.selection_repair_canonical_sensitivity import auc_draws, atomic_weights, compound_weights
from src.selection_repair_precision_proxy_v1 import precision_interval, projection_supported

METRICS = ('atomic_auroc', 'AND_auroc', 'OR_auroc', 'OR_mixed_vs_FF_auroc', 'AND_TT_vs_mixed_auroc')


def masks(rows, atomic=False):
    if atomic:
        return [('atomic_auroc', np.ones(len(rows), bool))]
    op = np.array([r['operator'] for r in rows]); cell = np.array([int(r['truth_a']) + int(r['truth_b']) for r in rows])
    definitions = [('AND_auroc', op == 'AND'), ('OR_auroc', op == 'OR'),
                   ('OR_mixed_vs_FF_auroc', (op == 'OR') & (cell < 2)), ('AND_TT_vs_mixed_auroc', (op == 'AND') & (cell > 0))]
    return [(metric, mask) for metric, mask in definitions if mask.any()]


def summaries(rows, scores, *, atomic=False, weights=None, options=None):
    y = np.array([int(r['label']) for r in rows]); topic_of = np.array([r['topic'] for r in rows])
    records, samples = [], []
    for metric, mask in masks(rows, atomic):
        topic_points, topic_samples = [], []
        for scope, topic, selected in [('pooled', 'all', mask)] + [('topic', t, mask & (topic_of == t)) for t in TOPICS]:
            point = weighted_auc(y[selected], scores[selected])
            record = dict(metric=metric, scope=scope, topic=topic, point=finite(point))
            if weights is not None:
                point, draws = auc_draws(scores, y, np.flatnonzero(selected), weights)
                record.update(point=finite(point), **interval(draws, options)); samples.append(draws)
                if scope == 'topic':
                    topic_samples.append(draws)
            records.append(record)
            if scope == 'topic':
                topic_points.append(point)
        record = dict(metric=metric, scope='topic_macro', topic='all', point=finite(np.mean(topic_points)))
        if weights is not None:
            draws = np.mean(topic_samples, axis=0); record.update(interval(draws, options)); samples.append(draws)
        records.append(record)
    return records, None if weights is None else np.asarray(samples)


def allocated_statistics(rows, scores, p_scores, atomic, indices=None):
    if indices is not None:
        rows = [rows[i] for i in indices]; scores = scores[indices]
    records, _ = summaries(rows, scores)
    value = lambda metric: next(r['point'] for r in records if r['metric'] == metric and r['scope'] == 'pooled')
    primary, and_boundary = value('OR_mixed_vs_FF_auroc'), value('AND_TT_vs_mixed_auroc')
    return dict(primary=primary, balanced=finite((primary + and_boundary) / 2), atomic=atomic,
                separation=separation(p_scores, scores, rows), AND_boundary=and_boundary)


def evaluate_locked(adapter, store, destination, identity, locked, reusable, options, precision_support):
    """Score wordings only after the immutable lock, using unchanged fitted heads."""
    destination = Path(destination)
    require((destination / 'locked.json').exists() and json.loads((destination / 'locked.json').read_text()) == locked, 'immutable procedure lock required')
    atom = adapter.rows['atomic_D']; comp = adapter.rows['D_bare']
    weights = dict(atomic_D=atomic_weights(atom, np.ones(len(atom), bool), options))
    weights['D_bare'], schedule = compound_weights(pd.DataFrame(comp).rename(columns={'person_a': 'entity_a_id', 'person_b': 'entity_b_id'}), np.ones(len(comp), bool), options)
    old_weights = np.load(reusable.root / 'bootstrap_weights.npz', allow_pickle=False)
    for g in ('atomic_D', 'D_bare'):
        require(np.array_equal(weights[g], old_weights[g]), 'recoverability D bootstrap schedule mismatch')
    old_weights.close()
    # Wording rows share exact bare identities/endpoints/order, not independent new draws.
    for group in sorted(adapter.receipt['heldout_wording_rows']):
        bare_indices = {r['example_id']: i for i, r in enumerate(comp)}
        indices = [bare_indices[r['example_id']] for r in adapter.rows[group]]
        require(all(adapter.rows[group][i]['pair_id'] == comp[j]['pair_id'] for i, j in enumerate(indices)), 'wording D pair schedule identity')
        weights[group] = weights['D_bare'][:, indices]
    np.savez(destination / 'bootstrap_weights.npz', **weights)
    weight_binding = hash_value({g: __import__('hashlib').sha256(w.tobytes()).hexdigest() for g, w in weights.items()})
    save(destination / 'bootstrap_binding.json', dict(identity, weights=pin(destination / 'bootstrap_weights.npz'), binding_sha256=weight_binding,
        compound_schedule_sha256=schedule, reuse_D_schedule_exact=True, shared_across_heads_models_seeds=True, seed=1729, replicates=2000,
        wording_endpoint_multiplicities_bound_to_bare=True, undefined_policy='no redraw/imputation/topic-dropping; ordinary all-five mean'))
    ids = sorted({value for model in locked['procedures'].values() for seed in model.values() for value in seed.values()})
    groups = {}
    for name in ids:
        record = store.load(name); require(record['valid_for_scoring'], 'locked invalid head')
        groups.setdefault((record['model'], record['layer']), []).append(name)
    scored = {}
    for (model, layer), names in sorted(groups.items()):
        features = adapter.layer(model, layer, wording=True)
        for name in names:
            record = store.load(name); params = store.parameters(name)
            from src.selection_repair_capacity_methods_v1 import score
            values = {g: score(record['scoring_method'], params, X) for g, X in features.items() if g != 'P15'}
            previous = store.scores(name)
            for g in ('atomic_D', 'D_bare'):
                require(np.array_equal(values[g], previous[g]), 'locked bare/atomic score reconstruction')
            folder = destination / 'evaluations' / store.folder(name).name; folder.mkdir(parents=True, exist_ok=True)
            if not (folder / 'scores.npz').exists():
                np.savez(folder / 'scores.npz', **values)
                save(folder / 'score_binding.json', dict(identity, fit_id=name, lock_sha256=pin(destination / 'locked.json')['sha256'],
                     artifact=pin(folder / 'scores.npz'), groups={g: dict(row_binding_sha256=hash_value(adapter.bindings[g]), ordered_keys=[[r['group'], r['logical_id']] for r in adapter.bindings[g]]) for g in values}))
            else:
                binding = json.loads((folder / 'score_binding.json').read_text())
                require(pin(folder / 'scores.npz') == binding['artifact'] and binding['lock_sha256'] == pin(destination / 'locked.json')['sha256'], 'resumed locked evaluation changed')
                with np.load(folder / 'scores.npz', allow_pickle=False) as saved:
                    require(all(np.array_equal(saved[g], values[g]) for g in values), 'resumed scores differ')
            templates, point, draws = [], [], []
            for g in ['atomic_D', 'D_bare'] + sorted(adapter.receipt['heldout_wording_rows']):
                records, samples = summaries(adapter.rows[g], values[g], atomic=g == 'atomic_D', weights=weights[g], options=options)
                templates.extend(dict(condition=g, **r) for r in records)
                point.extend(r['point'] for r in records); draws.extend(samples)
            scored[name] = dict(records=templates, points=np.array([np.nan if p is None else p for p in point]), draws=np.array(draws))
            if not (folder / 'bootstrap_metrics.npy').exists():
                np.save(folder / 'bootstrap_metrics.npy', scored[name]['draws'], allow_pickle=False)
                save(folder / 'metrics.json', dict(identity, fit_id=name, records=templates, draws=pin(folder / 'bootstrap_metrics.npy')))
            print('evaluated locked', name, flush=True)
        del features
    template = scored[ids[0]]['records']
    key = lambda r: (r['condition'], r['metric'], r['scope'], r['topic'])
    require(all([key(r) for r in scored[name]['records']] == [key(r) for r in template] for name in ids), 'metric/draw alignment')
    procedures = {}; metric_rows = []
    for model, seeds in locked['procedures'].items():
        arms = sorted(seeds['11'])
        require(all(set(seeds[str(s)]) == set(arms) for s in (11, 23, 37)), 'missing seed controls')
        for arm in arms:
            for seed in (11, 23, 37):
                name = seeds[str(seed)][arm]; procedures[model, arm, str(seed)] = scored[name]
                metric_rows.extend(dict(model=model, arm=arm, seed=seed, fit_id=name, optimistic_development=arm in ('bank_oracle', 'C_clean_D_selected'), **r) for r in scored[name]['records'])
            points = average_sample_metrics({s: procedures[model, arm, str(s)]['points'] for s in (11, 23, 37)})
            draws = average_sample_metrics({s: procedures[model, arm, str(s)]['draws'] for s in (11, 23, 37)})
            procedures[model, arm, 'mean_11_23_37'] = dict(points=points, draws=draws)
            metric_rows.extend(dict(model=model, arm=arm, seed='mean_11_23_37', fit_id='metric_mean_not_head', optimistic_development=arm in ('bank_oracle', 'C_clean_D_selected'),
                condition=r['condition'], metric=r['metric'], scope=r['scope'], topic=r['topic'], point=finite(points[i]), **interval(draws[i], options)) for i, r in enumerate(template))
    pairs = [('R_all', 'S_all'), ('S_all', 'S_atom_all'), ('R_all', 'atomic_only_at_R_all'), ('R_all', 'R0_at_R_all'),
             ('R_fixed', 'S_fixed'), ('R_fixed', 'atomic_only_fixed'), ('R_fixed', 'R0_fixed'), ('repair_at_S_all', 'S_all'),
             ('S_all', 'reduced_LR'), ('R_all', 'reduced_LR'), ('S_fixed', 'reduced_LR'), ('R_fixed', 'reduced_LR'),
             ('LR_only_selection', 'S_all'), ('S_all', 'original_atomic_selected_R0'), ('R_all', 'bank_oracle'),
             ('C_clean_fixed', 'S_all'), ('C_clean_D_selected', 'S_all'), ('C_clean_at_S_all', 'S_all'), ('C_clean_at_R_all', 'R_all'),
             ('C_clean_at_S_atom_all', 'S_atom_all'), ('C_clean_D_selected', 'reduced_LR'), ('C_clean_fixed', 'reduced_LR'),
             ('C_clean_at_S_all', 'reduced_LR'), ('C_clean_at_R_all', 'reduced_LR'), ('R0_at_R_all', 'reduced_LR'),
             ('S_atom_all', 'reduced_LR'), ('original_atomic_selected_R0', 'reduced_LR'), ('atomic_only_fixed', 'reduced_LR'), ('atomic_only_at_R_all', 'reduced_LR'),
             ('R_all_equal_boundary_sensitivity', 'S_all_equal_boundary_sensitivity')]
    contrasts = []; precision = []
    for model in locked['procedures']:
        for seed in ('11', '23', '37', 'mean_11_23_37'):
            for left, right in pairs:
                lhs, rhs = procedures[model, left, seed], procedures[model, right, seed]
                differences = lhs['draws'] - rhs['draws']; points = lhs['points'] - rhs['points']
                for i, r in enumerate(template):
                    record = dict(model=model, seed=seed, left=left, right=right, condition=r['condition'], metric=r['metric'], scope=r['scope'], topic=r['topic'],
                        point=finite(points[i]), **interval(differences[i], options), optimistic_development='bank_oracle' in (left, right) or 'C_clean_D_selected' in (left, right))
                    if right == 'reduced_LR' and r['metric'] in ('atomic_auroc', 'AND_auroc'):
                        margin = .005 if r['metric'] == 'atomic_auroc' else .02
                        record.update(retention_margin=margin, point_retained=bool(points[i] >= -margin),
                            interval_supported=record['ci_low'] is not None and record['ci_low'] >= -margin)
                    contrasts.append(record)
                    if (left, right) in [('R_all', 'S_all'), ('C_clean_D_selected', 'S_all')]:
                        detail = precision_interval(differences[i], points[i], options)
                        precision.append(dict(record, **detail, precision_mode='actual_B25_native_D', practical_margin=.02,
                            scope_note='conditional paired precision of actual frozen B25 results; no E predictions or power claim'))
                        if r['condition'] == 'D_bare' and r['metric'] in ('OR_mixed_vs_FF_auroc', 'AND_auroc'):
                            allowed, reason = projection_supported('bare', r['scope'], r['topic'], precision_support)
                            proxy = dict(record, precision_mode='approximate_E_entity_count_proxy', status='unsupported', reason=reason)
                            if allowed:
                                topic = r['topic']; factor = np.sqrt(precision_support['D_graph'][topic]['entities'] / precision_support['E_graph'][topic]['entities'])
                                proxy.update(status='approximate_count_transport', transport_scale=float(factor), **precision_interval(points[i] + factor * (differences[i] - points[i]), points[i], options))
                            else:
                                proxy.update(ci_low=None, ci_high=None, valid_replicates=0, invalid_replicates=0, ci_status='unsupported')
                            precision.append(proxy)
    # Direct paired cross-model interaction, not comparison of significance labels.
    for seed in ('11', '23', '37', 'mean_11_23_37'):
        q = procedures['qwen', 'R_all', seed]; qs = procedures['qwen', 'S_all', seed]
        l = procedures['llama', 'R_all', seed]; ls = procedures['llama', 'S_all', seed]
        points = (q['points'] - qs['points']) - (l['points'] - ls['points'])
        draws = (q['draws'] - qs['draws']) - (l['draws'] - ls['draws'])
        contrasts.extend(dict(model='qwen_minus_llama', seed=seed, left='R_all_minus_S_all_qwen', right='R_all_minus_S_all_llama',
            condition=r['condition'], metric=r['metric'], scope=r['scope'], topic=r['topic'], point=finite(points[i]), **interval(draws[i], options), optimistic_development=False) for i, r in enumerate(template))
    sufficiency = []
    for model in locked['procedures']:
        def delta(left, right, metric, condition):
            return next(r for r in contrasts if (r['model'], r['seed'], r['left'], r['right'], r['metric'], r['condition'], r['scope']) == (model, 'mean_11_23_37', left, right, metric, condition, 'pooled'))
        rs = delta('R_all', 'S_all', 'OR_mixed_vs_FF_auroc', 'D_bare'); cs = delta('C_clean_D_selected', 'S_all', 'OR_mixed_vs_FF_auroc', 'D_bare')
        control = next(r for r in metric_rows if r['model'] == model and r['arm'] == 'C_clean_D_selected' and r['seed'] == 'mean_11_23_37' and r['condition'] == 'D_bare' and r['metric'] == 'OR_mixed_vs_FF_auroc' and r['scope'] == 'pooled')
        atomic = delta('S_all', 'reduced_LR', 'atomic_auroc', 'atomic_D'); AND = delta('S_all', 'reduced_LR', 'AND_auroc', 'D_bare')
        criteria = dict(repair_gap_upper_below_02=rs['ci_high'] is not None and rs['ci_high'] < .02, control_gap_upper_below_02=cs['ci_high'] is not None and cs['ci_high'] < .02,
            control_lower_at_least_90=control['ci_low'] is not None and control['ci_low'] >= .90, selection_atomic_retention=atomic['interval_supported'], selection_AND_retention=AND['interval_supported'])
        sufficiency.append(dict(model=model, scope='exploratory optimistic development operational criteria only; no E/sufficiency lock', criteria=criteria,
            all_components_met_on_D=all(criteria.values()), C_clean_atomic_retention=delta('C_clean_D_selected', 'reduced_LR', 'atomic_auroc', 'atomic_D')['interval_supported'], joint_control_sufficiency_claim=False))
    save(destination / 'metrics.json', dict(identity, weight_binding=weight_binding, records=metric_rows))
    save(destination / 'contrasts.json', dict(identity, weight_binding=weight_binding, records=contrasts))
    save(destination / 'precision.json', dict(identity, records=precision, limitation='actual-D conditional precision; count transport unsupported for pooled/macro/inventors compound endpoints; no E predictions'))
    save(destination / 'sufficiency.json', dict(identity, records=sufficiency))
    for name, rows in [('metrics', metric_rows), ('contrasts', contrasts), ('precision', precision)]:
        pd.DataFrame(rows).to_csv(destination / (name + '.csv'), index=False)
    return dict(unique_evaluated_heads=len(ids), metric_rows=len(metric_rows), contrast_rows=len(contrasts), precision_rows=len(precision), sufficiency=sufficiency)
