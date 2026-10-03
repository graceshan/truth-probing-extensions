"""Sequential independent sklearn/rank-group reconstruction, no fits or activations."""
import argparse
import json
import resource
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, confusion_matrix, balanced_accuracy_score
from threadpoolctl import threadpool_limits
from src.checkpoint_next_inputs import Inputs, OR_GROUPS, read
from src.checkpoint_r2_fresh_inputs import ROOT, hash_value, require
from src.checkpoint_r2_fresh_store import pin


def ranked_auc(y, score, weights):
    """Independent stable tied-score groups, including draws with zero total class mass."""
    order = np.argsort(score, kind='stable'); v = score[order]; y = y[order]
    if not len(v): return np.full(len(weights), np.nan)
    boundaries = np.r_[0, np.flatnonzero(v[1:] != v[:-1]) + 1]
    result = []
    for start in range(0, len(weights), 64):
        w = weights[start:start + 64, order].astype(float)
        p = np.add.reduceat(w * y, boundaries, axis=1); n = np.add.reduceat(w * (1 - y), boundaries, axis=1)
        num = (p * (np.cumsum(n, axis=1) - n / 2)).sum(axis=1)
        den = p.sum(axis=1) * n.sum(axis=1)
        result.append(np.divide(num, den, out=np.full(len(w), np.nan), where=den > 0))
    return np.concatenate(result)


def audit(out):
    started = time.monotonic(); completion = read(out / 'completion.json')
    for rel, expected in completion['output_files'].items(): require(pin(out / rel) == expected, 'analysis output corruption: ' + rel)
    inputs = Inputs(); topics = inputs.cfg['topics']; checked = dict(cell_points=0, pooled_draw_values=0, macro_draw_values=0, mass_values=0, published_sample_mean_points=0, confusion_strata=0, min_max_points=0)
    maximum_error = 0.; global_values = {}
    cells = pd.read_csv(out / 'boundary-cells.csv'); met = pd.read_csv(out / 'boundary-decomposition.csv')
    published = pd.read_csv(ROOT / 'results/checkpoint_r2_b25_convergence_fix_20261002/metrics.csv')
    for model in ('qwen', 'llama'):
        for arm in ('R_all', 'S_all'):
            for seed in (11, 23, 37):
                panel = inputs.b25(model, seed, arm)
                for group in OR_GROUPS:
                    rr = inputs.b.rows[group]; keep = np.array([r['operator'] == 'OR' and int(r['truth_a']) + int(r['truth_b']) < 2 for r in rr]); rows = [r for r, k in zip(rr, keep) if k]
                    score = panel['scores'][group][keep]; y = np.asarray([int(r['label']) for r in rows]); topic = np.asarray([r['topic'] for r in rows]); w = np.vstack((np.ones(len(y)), inputs.weights[group][:, keep]))
                    saved = dict(np.load(out / 'arrays' / f'{model}-{arm}-{seed}-{group}.npz', allow_pickle=False))
                    point_matrix = np.empty((5, 5)); mass = np.empty_like(saved['mass']); diagdraw = []
                    den = w[:, y == 1].sum(axis=1) * w[:, y == 0].sum(axis=1)
                    for i, ti in enumerate(topics):
                        pm = w[:, (y == 1) & (topic == ti)].sum(axis=1)
                        within = topic == ti
                        diagdraw.append(ranked_auc(y[within], score[within], w[:, within]))
                        for j, tj in enumerate(topics):
                            mask = ((y == 1) & (topic == ti)) | ((y == 0) & (topic == tj))
                            point_matrix[i, j] = roc_auc_score(y[mask], score[mask])
                            nm = w[:, (y == 0) & (topic == tj)].sum(axis=1)
                            mass[:, i, j] = np.divide(pm * nm, den, out=np.full(len(w), np.nan), where=den > 0)
                            checked['cell_points'] += 1
                    np.testing.assert_allclose(point_matrix, saved['matrix'][0], atol=1e-12, rtol=0)
                    np.testing.assert_allclose(mass, saved['mass'], atol=1e-12, rtol=0, equal_nan=True); checked['mass_values'] += mass.size
                    direct = ranked_auc(y, score, w)
                    macro = np.mean(diagdraw, axis=0)
                    np.testing.assert_allclose(direct, saved['values'][:, 0], atol=1e-12, rtol=0, equal_nan=True)
                    np.testing.assert_allclose(macro, saved['values'][:, 1], atol=1e-12, rtol=0, equal_nan=True)
                    np.testing.assert_allclose(roc_auc_score(y, score), (point_matrix * mass[0]).sum(), atol=1e-12, rtol=0)
                    maximum_error = max(maximum_error, float(np.nanmax(abs(direct - saved['values'][:, 0]))))
                    checked['pooled_draw_values'] += 2000; checked['macro_draw_values'] += 2000
                    # Reconstruct every diagonal/off-diagonal contribution from saved independent-checked cells and masses.
                    contrib = np.where(mass == 0, 0., saved['matrix'] * mass); contrib[den == 0] = np.nan
                    np.testing.assert_allclose(contrib, saved['contribution'], rtol=0, atol=1e-12, equal_nan=True)
                    global_values[model, arm, str(seed), group] = saved['values']
            for group in OR_GROUPS:
                avg = np.mean([global_values[model, arm, str(s), group] for s in (11, 23, 37)], axis=0)
                frame = met[(met.model == model) & (met.arm == arm) & (met.seed.astype(str) == 'mean_11_23_37') & (met.condition == group)]
                for metric, col, scope in [('pooled', 0, 'pooled'), ('topic_macro', 1, 'topic_macro')]:
                    rec = frame[frame.metric == metric].iloc[0]
                    pub = published[(published.model == model) & (published.arm == arm) & (published.seed.astype(str) == 'mean_11_23_37') & (published.condition == group) & (published.metric == 'OR_mixed_vs_FF_auroc') & (published.scope == scope)].iloc[0]
                    np.testing.assert_allclose([rec.point, pub.point], avg[0, col], atol=1e-12, rtol=0)
                    good = np.isfinite(avg[1:, col]); require(rec.undefined_replicates == 2000 - int(good.sum()), 'mean undefined count')
                    np.testing.assert_allclose([rec.ci_low, rec.ci_high], np.quantile(avg[1:, col][good], [.025, .975]), atol=1e-12, rtol=0)
                    checked['published_sample_mean_points'] += 1
            print('independent pooled/cell audit', model, arm, flush=True)
    decisions = pd.read_csv(out / 'frozen-confusion-joint-boolean.csv')
    minmax = pd.read_csv(out / 'same-support-min-max.csv')
    for (model, source, op, group), frame in decisions.groupby(['model', 'source_operator', 'evaluation_operator', 'condition']):
        p = inputs.constituent(model, source, group, op); rr, score = p['rows'], p['scores']
        first = np.asarray([int(r['truth_a']) if r['ordering'] == 'AB' else int(r['truth_b']) for r in rr]); second = np.asarray([int(r['truth_b']) if r['ordering'] == 'AB' else int(r['truth_a']) for r in rr])
        true = np.asarray([int(r['label']) for r in rr]); pred = score >= 0
        boolean = np.logical_and(pred[:, 0], pred[:, 1]) if op == 'AND' else np.logical_or(pred[:, 0], pred[:, 1])
        topic = np.asarray([r['topic'] for r in rr]); order = np.asarray([r['ordering'] for r in rr]); cell = np.asarray([('T' if int(r['truth_a']) else 'F') + ('T' if int(r['truth_b']) else 'F') for r in rr])
        for rec in frame.to_dict('records'):
            mask = (np.ones(len(rr), bool) if rec['topic'] == 'all' else topic == rec['topic']) & (np.ones(len(rr), bool) if rec['surface_order'] == 'all' else order == rec['surface_order']) & (np.ones(len(rr), bool) if rec['canonical_truth_cell'] == 'all' else cell == rec['canonical_truth_cell'])
            require(int(mask.sum()) == rec['observations'], 'confusion denominator')
            for name, y, yp in [('first', first, pred[:, 0]), ('second', second, pred[:, 1]), ('boolean', true, boolean)]:
                tn, fp, fn, tp = confusion_matrix(y[mask], yp[mask], labels=[0, 1]).ravel()
                require([rec[name + '_' + k] for k in ('TN', 'FP', 'FN', 'TP')] == [tn, fp, fn, tp], 'independent confusion counts')
                if len(np.unique(y[mask])) == 2: np.testing.assert_allclose(rec[name + '_balanced_accuracy'], balanced_accuracy_score(y[mask], yp[mask]), rtol=0, atol=1e-12)
                else: require(pd.isna(rec[name + '_balanced_accuracy']), 'undefined balanced accuracy required')
            correct1, correct2 = pred[mask, 0] == first[mask], pred[mask, 1] == second[mask]
            require([rec[k] for k in ('both_correct', 'first_correct_second_wrong', 'first_wrong_second_correct', 'both_wrong')] == [int((correct1 & correct2).sum()), int((correct1 & ~correct2).sum()), int((~correct1 & correct2).sum()), int((~correct1 & ~correct2).sum())], 'joint correctness counts')
            checked['confusion_strata'] += 1
        f = minmax[(minmax.model == model) & (minmax.source_operator == source) & (minmax.evaluation_operator == op) & (minmax.condition == group)]
        for rec in f.to_dict('records'):
            mask = (np.ones(len(rr), bool) if rec['topic'] == 'all' else topic == rec['topic']) & (np.ones(len(rr), bool) if rec['surface_order'] == 'all' else order == rec['surface_order'])
            values = np.min(score, axis=1) if rec['composition'] == 'min' else np.max(score, axis=1)
            np.testing.assert_allclose(rec['point'], roc_auc_score(true[mask], values[mask]), rtol=0, atol=1e-12); checked['min_max_points'] += 1
    # Independent metadata-only missing queue reconstruction and deterministic sampler.
    members = read(ROOT / 'data/checkpoint_r2_v1/memberships.json'); full = members['outer_train']; pids = {r['source_row_id'] for r in members['P15']}
    add = [r for r in full if r['source_row_id'] not in pids]; require(len(add) == 262, '262 additional rows')
    accepted = {r['statement_sha256']: r for r in inputs.manifest['rows']}
    missing = {r['statement_sha256'] for r in full if r['statement_sha256'] not in accepted}
    queue = read(out / 'missing-input-queue.json'); cov = read(out / 'coverage-summary.json')
    require(missing == {r['statement_sha256'] for r in queue['rows']} and len(missing) == cov['missing_unique_texts_per_model'], 'independent missing queue set')
    exposure = pd.read_csv(out / 'full-training-coverage.csv')
    for row in full:
        records = exposure[exposure.source_row_id == row['source_row_id']]
        require(len(records) == 2 and set(records.model) == {'qwen', 'llama'}, 'full per-model binding coverage')
        if row['statement_sha256'] in missing: require(set(records.status) == {'missing_unique_text'}, 'missing coverage status')
        else: require(set(records.status) == {'exact_compatible_fresh_reuse'}, 'fresh reuse coverage status')
    # Original topic order, RandomState seed, pair row alignment; no production sampler call.
    rng = np.random.RandomState(0); chosen = []
    topic_order = ('cities', 'sp_en_trans', 'inventors', 'animal_class', 'element_symb')
    size = min(sum(r['topic'] == t and r['form'] == 'affirmative' for r in full) for t in topic_order)
    for t in topic_order:
        a = [i for i, r in enumerate(full) if r['topic'] == t and r['form'] == 'affirmative']; n = [i for i, r in enumerate(full) if r['topic'] == t and r['form'] == 'negated']
        for i, j in zip(a, n): require(full[i]['entity_id'] == full[j]['entity_id'] and full[i]['source_row_id'].rsplit(':', 1)[1] == full[j]['source_row_id'].rsplit(':', 1)[1] and int(full[i]['label']) == 1 - int(full[j]['label']), 'independent paired sampler identities')
        selected = rng.choice(len(a), size, replace=False); chosen += [a[i] for i in selected] + [n[i] for i in selected]
    sampled = [full[i] for i in chosen]; require(len(sampled) == 1000 and hash_value(sampled) == cov['sampler_ordered_membership_sha256'], 'independent full sampler')
    require(pd.read_csv(out / 'full-training-balanced-1000.csv').source_row_id.tolist() == [r['source_row_id'] for r in sampled], 'sampler file order')
    inputs.unchanged()
    receipt = dict(status='passed_independent_saved_score_and_metadata_audit', producer_sha=completion['producer_sha'],
        checks=checked, maximum_pooled_reconstruction_difference=maximum_error,
        missing_unique_texts_per_model=len(missing), sampler_membership_sha256=hash_value(sampled),
        seconds=time.monotonic() - started, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        no_fits=True, no_activations_loaded=True, no_behavior_chat_E=True,
        scope='Separate sklearn pair-cell/decision/min-max point calculations; stable rank-group pooled and fixed-five-topic bootstrap reconstruction; sample-mean published agreement; metadata queue and independent RandomState sampler. Not a second-machine replication.')
    (out / 'independent-audit.json').write_text(json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False) + '\n')
    print(json.dumps(receipt, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--run', type=Path, required=True); args = parser.parse_args()
    with threadpool_limits(limits=1): audit(args.run)


if __name__ == '__main__': main()
