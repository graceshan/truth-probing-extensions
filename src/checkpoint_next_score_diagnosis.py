"""Frozen saved-score description and metadata-only input inventory; no fitting route."""
import argparse
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr, kendalltau
from threadpoolctl import threadpool_limits, threadpool_info
from src.checkpoint_next_inputs import Inputs, CONFIG, OR_GROUPS, matched, read
from src.checkpoint_next_coverage import coverage
from src.checkpoint_r2_fresh_inputs import ROOT, hash_value, require, file_hash
from src.checkpoint_r2_fresh_store import pin
from src.checkpoint_r2_constituent_metrics import summaries
from src.checkpoint_r2_selection_v1 import TOPICS


def finite(v):
    return float(v) if np.isfinite(v) else None


def save(p, value):
    Path(p).write_text(json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def ci(draw):
    good = np.isfinite(draw); n = int(good.sum())
    bounds = np.quantile(draw[good], [.025, .975], method='linear') if n >= 1800 else [np.nan, np.nan]
    return dict(ci_low=finite(bounds[0]), ci_high=finite(bounds[1]), valid_replicates=n,
                undefined_replicates=int(len(draw) - n), total_replicates=len(draw),
                interval='95% pointwise percentile, conditional on frozen fitted heads and support; no redraw')


def cross_auc(score, positive, negative, weights):
    """Exact pair wins and half ties; returns AUROC and both class masses per draw."""
    p, n = np.asarray(positive), np.asarray(negative)
    order = n[np.argsort(score[n], kind='stable')]
    left = np.searchsorted(score[order], score[p], side='left')
    right = np.searchsorted(score[order], score[p], side='right')
    out, pmass, nmass = [], [], []
    for start in range(0, len(weights), 64):
        w = weights[start:start + 64]
        wp, wn = w[:, p].astype(float), w[:, order].astype(float)
        cumulative = np.column_stack((np.zeros(len(w)), np.cumsum(wn, axis=1)))
        wins = .5 * (cumulative[:, left] + cumulative[:, right])
        pm, nm = wp.sum(axis=1), wn.sum(axis=1); den = pm * nm
        out.append(np.divide((wp * wins).sum(axis=1), den, out=np.full(len(w), np.nan), where=den > 0))
        pmass.append(pm); nmass.append(nm)
    return np.concatenate(out), np.concatenate(pmass), np.concatenate(nmass)


DECOMP = ('pooled', 'topic_macro', 'diagonal_contribution', 'off_diagonal_contribution',
          'diagonal_mass', 'off_diagonal_mass', 'diagonal_mass_normalized_auc', 'off_diagonal_mass_normalized_auc')


def decompose(rows, scores, weights):
    y = np.asarray([int(r['label']) for r in rows]); topic = np.asarray([r['topic'] for r in rows])
    require(set(r['operator'] for r in rows) == {'OR'} and all(int(r['truth_a']) + int(r['truth_b']) < 2 for r in rows), 'OR mixed/FF boundary support')
    w = np.vstack((np.ones(len(rows)), weights))
    matrix = np.empty((len(w), 5, 5)); mass = np.empty_like(matrix)
    totalden = w[:, y == 1].sum(axis=1) * w[:, y == 0].sum(axis=1)
    for i, ti in enumerate(TOPICS):
        for j, tj in enumerate(TOPICS):
            a, p, n = cross_auc(scores, np.flatnonzero((y == 1) & (topic == ti)), np.flatnonzero((y == 0) & (topic == tj)), w)
            matrix[:, i, j] = a
            mass[:, i, j] = np.divide(p * n, totalden, out=np.full(len(w), np.nan), where=totalden > 0)
    # Undefined zero-mass cells contribute exactly zero, except zero total denominator.
    contribution = np.where(mass == 0, 0., matrix * mass)
    contribution[totalden == 0] = np.nan
    diag = np.eye(5, dtype=bool)
    dc = contribution[:, diag].sum(axis=1); oc = contribution[:, ~diag].sum(axis=1)
    dm = mass[:, diag].sum(axis=1); om = mass[:, ~diag].sum(axis=1)
    values = np.column_stack((dc + oc, matrix[:, diag].mean(axis=1), dc, oc, dm, om,
        np.divide(dc, dm, out=np.full(len(w), np.nan), where=dm > 0),
        np.divide(oc, om, out=np.full(len(w), np.nan), where=om > 0)))
    direct = cross_auc(scores, np.flatnonzero(y == 1), np.flatnonzero(y == 0), w)[0]
    np.testing.assert_allclose(values[:, 0], direct, rtol=0, atol=1e-12, equal_nan=True)
    return dict(matrix=matrix, mass=mass, contribution=contribution, values=values,
                maximum_reconstruction_error=float(np.nanmax(abs(values[:, 0] - direct))))


def emit_decomp(meta, result, cells, metrics):
    for i, ti in enumerate(TOPICS):
        for j, tj in enumerate(TOPICS):
            cells.append(dict(meta, positive_topic=ti, negative_topic=tj, diagonal=i == j,
                point=finite(result['matrix'][0, i, j]), mass=finite(result['mass'][0, i, j]),
                contribution=finite(result['contribution'][0, i, j]), **ci(result['matrix'][1:, i, j])))
    for k, name in enumerate(DECOMP):
        metrics.append(dict(meta, metric=name, point=finite(result['values'][0, k]), **ci(result['values'][1:, k])))


def strata(rows):
    topic = np.asarray([r['topic'] for r in rows]); order = np.asarray([r['ordering'] for r in rows])
    cell = np.asarray([('T' if int(r['truth_a']) else 'F') + ('T' if int(r['truth_b']) else 'F') for r in rows])
    for t in ('all', *TOPICS):
        for c in ('all', 'TT', 'TF', 'FT', 'FF'):
            for o in ('all', 'AB', 'BA'):
                mask = (np.ones(len(rows), bool) if t == 'all' else topic == t) & (np.ones(len(rows), bool) if c == 'all' else cell == c) & (np.ones(len(rows), bool) if o == 'all' else order == o)
                if mask.any(): yield dict(topic=t, canonical_truth_cell=c, surface_order=o), mask


def delta_records(meta, rows, bare, wording, label, other, dest, ranks):
    delta = wording - bare
    for s, keep in strata(rows):
        d = delta[keep]; b = bare[keep]; v = wording[keep]
        rb, rv = rankdata(b), rankdata(v); change = rv - rb
        ranks.append(dict(meta, **s, observations=len(d), spearman=finite(spearmanr(b, v).statistic) if len(d) > 1 and np.ptp(b) and np.ptp(v) else None,
            kendall_tau_b=finite(kendalltau(b, v).statistic) if len(d) > 1 else None,
            mean_absolute_rank_change=float(abs(change).mean()), max_absolute_rank_change=float(abs(change).max()),
            fraction_rank_changed=float(np.mean(change != 0)), sign_changes=int(np.sum((b >= 0) != (v >= 0))),
            rank_scope='average tied ranks recomputed within this exact stratum; no head concatenation'))
        dest.append(dict(meta, **s, grouping='truth_cell_order', label='all', other_truth='all', observations=len(d),
            mean=float(d.mean()), median=float(np.median(d)), std_population=float(d.std()),
            q25=float(np.quantile(d, .25)), q75=float(np.quantile(d, .75)), minimum=float(d.min()), maximum=float(d.max())))
        if s['canonical_truth_cell'] != 'all': continue
        for lab in (0, 1):
            for oth in (('all', 0, 1) if other is not None else ('all',)):
                idx = keep & (label == lab) & (np.ones(len(rows), bool) if oth == 'all' else other == oth)
                d = delta[idx]
                if not len(d): continue
                dest.append(dict(meta, **s, grouping='label_other_truth', label=lab, other_truth=oth, observations=len(d),
                    mean=float(d.mean()), median=float(np.median(d)), std_population=float(d.std()),
                    q25=float(np.quantile(d, .25)), q75=float(np.quantile(d, .75)), minimum=float(d.min()), maximum=float(d.max())))


def confusion(y, pred):
    tp, tn, fp, fn = (int(np.sum(m)) for m in ((y == 1) & pred, (y == 0) & ~pred, (y == 0) & pred, (y == 1) & ~pred))
    ba = .5 * (tp / (tp + fn) + tn / (tn + fp)) if (tp + fn) and (tn + fp) else None
    return dict(TP=tp, TN=tn, FP=fp, FN=fn, accuracy=(tp + tn) / len(y), balanced_accuracy=ba)


def decisions(meta, rows, score, records):
    a = np.array([int(r['truth_a']) for r in rows]); b = np.array([int(r['truth_b']) for r in rows]); ab = np.array([r['ordering'] == 'AB' for r in rows])
    first, second = np.where(ab, a, b), np.where(ab, b, a)
    pred = score >= 0; correct = pred == np.column_stack((first, second))
    truth = np.array([int(r['label']) for r in rows]); boolean = pred.all(axis=1) if meta['evaluation_operator'] == 'AND' else pred.any(axis=1)
    for s, idx in strata(rows):
        item = dict(meta, **s, observations=int(idx.sum()), frozen_rule='logit>=0; exact ties True')
        for i, (name, y) in enumerate((('first', first), ('second', second))):
            item.update({name + '_' + k: v for k, v in confusion(y[idx], pred[idx, i]).items()})
            item[name + '_zero_ties'] = int(np.sum(score[idx, i] == 0))
        item.update({'boolean_' + k: v for k, v in confusion(truth[idx], boolean[idx]).items()})
        item.update(both_correct=int(np.sum(correct[idx].all(axis=1))), first_correct_second_wrong=int(np.sum(correct[idx, 0] & ~correct[idx, 1])),
            first_wrong_second_correct=int(np.sum(~correct[idx, 0] & correct[idx, 1])), both_wrong=int(np.sum(~correct[idx].any(axis=1))))
        item['joint_correctness'] = item['both_correct'] / item['observations']
        records.append(item)


def plots(out, decomp, plot_panels):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    short = ['animal', 'cities', 'element', 'inventors', 'translation']
    for model in ('qwen', 'llama'):
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
        base = decomp[model, 'R_all', 'mean_11_23_37', 'D_bare']['matrix'][0]
        for col, g in enumerate(OR_GROUPS):
            matrix = decomp[model, 'R_all', 'mean_11_23_37', g]['matrix'][0]
            for row, values in [(0, matrix), (1, matrix - base)]:
                ax = axes[row, col]; im = ax.imshow(values, vmin=0 if row == 0 else -.35, vmax=1 if row == 0 else .35, cmap='viridis' if row == 0 else 'RdBu')
                ax.set_xticks(range(5), short, rotation=35, ha='right'); ax.set_yticks(range(5), short)
                for i in range(5):
                    for j in range(5): ax.text(j, i, f'{values[i,j]:.2f}', ha='center', va='center', fontsize=8)
                ax.set_title(g.replace('or_', '').replace('_v1', '') + (' − bare' if row else ''))
                fig.colorbar(im, ax=ax, shrink=.7)
        fig.suptitle(f'{model}: frozen R_all OR-boundary cells, mean of three separate heads\nRows: mixed-positive topic; columns: FF-negative topic')
        fig.savefig(out / f'{model}-R_all-boundary-matrices.png', dpi=150); plt.close(fig)
    for category in ('R_all', 'constituent'):
        selected = [p for p in plot_panels if p['category'] == category]
        fig, axes = plt.subplots(2, len(selected) // 2, figsize=(12, 7), squeeze=False, constrained_layout=True)
        for ax, p in zip(axes.flat, selected):
            b, v = p['bare'], p['wording']; topics = np.asarray([r['topic'] for r in p['rows']])
            for i, t in enumerate(TOPICS):
                mask = topics == t
                ax.scatter(b[mask], v[mask], s=3, alpha=.25, label=t, color=plt.get_cmap('tab10')(i))
            lo = min(b.min(), v.min()); hi = max(b.max(), v.max())
            ax.plot([lo, hi], [lo, hi], color='black', linewidth=.7)
            ax.axhline(0, color='gray', linewidth=.5); ax.axvline(0, color='gray', linewidth=.5)
            ax.set_xlabel('Matched bare score'); ax.set_ylabel('Wording score'); ax.set_title(p['title'], fontsize=9)
        handles, labels = axes.flat[0].get_legend_handles_labels(); fig.legend(handles, labels, loc='outside lower center', ncol=5, markerscale=3)
        fig.savefig(out / f'{category}-matched-scores.png', dpi=150); plt.close(fig)


def run(out, expected):
    started = time.monotonic()
    require(platform.system() == 'Darwin', 'this execution must be macOS')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT).decode().strip()
    require(head == expected, 'exact producer HEAD required')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=ROOT).strip(), 'tracked worktree must be clean')
    out.mkdir(parents=True, exist_ok=False); (out / 'arrays').mkdir()
    save(out / 'execution.json', dict(producer_sha=expected, hostname=platform.node(), platform=platform.platform(), interpreter=sys.executable,
        python=sys.version, pid=os.getpid(), processes=1, config=pin(CONFIG), environment={k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}, threadpools=threadpool_info()))
    inputs = Inputs(); verify_start = time.monotonic(); cov = coverage(inputs, out)
    coverage_seconds = time.monotonic() - verify_start
    cells, metrics, deltas, ranks, checks, constituent_metrics, decision_records, composition_extra = [], [], [], [], [], [], [], []
    results = {}; panels = []; plot_panels = []; matched_rows = []
    for model in inputs.cfg['models']:
        for arm in ('R_all', 'S_all'):
            for seed in (11, 23, 37):
                p = inputs.b25(model, seed, arm)
                panels.append({k: p[k] for k in ('model', 'arm', 'seed', 'head_id', 'saved_layer', 'score_folder')})
                for group in OR_GROUPS:
                    rows = inputs.b.rows[group]; score = p['scores'][group]
                    keep = np.array([r['operator'] == 'OR' and int(r['truth_a']) + int(r['truth_b']) < 2 for r in rows])
                    rr = [r for r, k in zip(rows, keep) if k]; result = decompose(rr, score[keep], inputs.weights[group][:, keep])
                    key = model, arm, str(seed), group; results[key] = result
                    meta = dict(model=model, arm=arm, seed=str(seed), condition=group, head_id=p['head_id'], saved_layer=p['saved_layer'])
                    emit_decomp(meta, result, cells, metrics)
                    np.savez(out / 'arrays' / ('-'.join(key) + '.npz'), **{k: result[k] for k in ('matrix', 'mass', 'contribution', 'values')})
                    for scope, k in [('pooled', 0), ('topic_macro', 1)]:
                        idx = next(i for i, r in enumerate(p['metrics']['records']) if (r['condition'], r['metric'], r['scope']) == (group, 'OR_mixed_vs_FF_auroc', scope))
                        published = p['metrics']['records'][idx]
                        np.testing.assert_allclose(result['values'][0, k], published['point'], rtol=0, atol=1e-12)
                        np.testing.assert_allclose(result['values'][1:, k], p['bootstrap'][idx], rtol=0, atol=1e-12, equal_nan=True)
                        checks.append(dict(meta, scope=scope, point=result['values'][0, k], published=published['point'], bootstrap_values_verified=2000, maximum_reconstruction_error=result['maximum_reconstruction_error']))
                for group in OR_GROUPS[1:]:
                    wr = inputs.b.rows[group]; idx = matched(inputs.b.rows['D_bare'], wr); b = p['scores']['D_bare'][idx]; v = p['scores'][group]
                    meta = dict(model=model, arm=arm, seed=str(seed), condition=group, position='global', saved_layer=p['saved_layer'], head_id=p['head_id'])
                    label = np.array([int(r['label']) for r in wr]); delta_records(meta, wr, b, v, label, None, deltas, ranks)
                    if arm == 'R_all' and seed == 11:
                        plot_panels.append(dict(category='R_all', title=f'{model} R_all seed 11, {group}', rows=wr, bare=b, wording=v))
                    matched_rows.append(pd.DataFrame([dict(meta, example_id=r['example_id'], topic=r['topic'], ordering=r['ordering'], truth_a=r['truth_a'], truth_b=r['truth_b'], label=r['label'], bare_score=float(x), wording_score=float(y), delta=float(y-x)) for r, x, y in zip(wr, b, v)]))
                print('completed saved global head', model, arm, seed, flush=True)
            for group in OR_GROUPS:
                parts = [results[model, arm, str(s), group] for s in (11, 23, 37)]
                result = {k: np.mean([p[k] for p in parts], axis=0) for k in ('matrix', 'mass', 'contribution', 'values')}
                results[model, arm, 'mean_11_23_37', group] = result
                meta = dict(model=model, arm=arm, seed='mean_11_23_37', condition=group, head_id='mean_of_metrics_not_scores', saved_layer='separate_heads')
                emit_decomp(meta, result, cells, metrics)
            for seed in ('11', '23', '37', 'mean_11_23_37'):
                for group in OR_GROUPS[1:]:
                    bare, word = results[model, arm, seed, 'D_bare'], results[model, arm, seed, group]
                    result = {k: word[k] - bare[k] for k in ('matrix', 'mass', 'contribution', 'values')}
                    emit_decomp(dict(model=model, arm=arm, seed=seed, condition=group + '_minus_bare', head_id='paired_metric_difference', saved_layer='same_frozen_head'), result, cells, metrics)
        # Original repair gains are reported separately from residual wording decomposition.
        for seed in ('11', '23', '37', 'mean_11_23_37'):
            left, right = results[model, 'R_all', seed, 'D_bare'], results[model, 'S_all', seed, 'D_bare']
            result = {k: left[k] - right[k] for k in ('matrix', 'mass', 'contribution', 'values')}
            emit_decomp(dict(model=model, arm='R_all_minus_S_all', seed=seed, condition='D_bare', head_id='paired_metric_difference', saved_layer='separate_frozen_heads'), result, cells, metrics)
        for source in ('AND', 'OR'):
            for op in ('AND', 'OR'):
                groups = ('D_bare', 'and_both_following_v1') if op == 'AND' else OR_GROUPS
                bare_panel = inputs.constituent(model, source, 'D_bare', op)
                for group in groups:
                    p = bare_panel if group == 'D_bare' else inputs.constituent(model, source, group, op)
                    rr, score = p['rows'], p['scores']; idx = matched(inputs.b.rows['D_bare'], rr)
                    weights = inputs.weights['D_bare'][:, idx]
                    meta = {k: p[k] for k in ('model', 'source_operator', 'evaluation_operator', 'condition', 'saved_layer', 'score_folder')}
                    panels.append(dict(meta, head_ids=p['head_ids']))
                    records, draws = summaries(rr, score, op, weights=weights, options=inputs.cfg['bootstrap'])
                    old = p['metrics']['records']; require(len(old) == len(records), 'constituent metric support')
                    for r, saved in zip(records, old):
                        require(all(r[k] == saved[k] for k in ('metric', 'scope', 'topic', 'surface_order')), 'constituent endpoint alignment')
                        require((r['point'] is None and saved['point'] is None) or abs(r['point'] - saved['point']) <= 1e-12, 'constituent published point reconstruction')
                        constituent_metrics.append(dict(meta, **r))
                    np.testing.assert_allclose(draws, p['bootstrap'], rtol=0, atol=1e-12, equal_nan=True)
                    checks.append(dict(meta, endpoint_points_verified=len(records), endpoint_bootstrap_values_verified=int(draws.size)))
                    decisions(meta, rr, score, decision_records)
                    y = np.array([int(r['label']) for r in rr]); tp = np.array([r['topic'] for r in rr]); order = np.array([r['ordering'] for r in rr])
                    for method, scalar in [('min', score.min(axis=1)), ('max', score.max(axis=1))]:
                        for t in ('all', *TOPICS):
                            for o in ('all', 'AB', 'BA'):
                                mask = (np.ones(len(rr), bool) if t == 'all' else tp == t) & (np.ones(len(rr), bool) if o == 'all' else order == o)
                                w = np.vstack((np.ones(mask.sum()), weights[:, mask])); vals = scalar[mask]; labels = y[mask]
                                draw = cross_auc(vals, np.flatnonzero(labels == 1), np.flatnonzero(labels == 0), w)[0]
                                composition_extra.append(dict(meta, composition=method, topic=t, surface_order=o, point=finite(draw[0]), **ci(draw[1:])))
                    if group != 'D_bare':
                        midx = matched(bare_panel['rows'], rr); bscore = bare_panel['scores'][midx]
                        for i, position in enumerate(('FIRST', 'SECOND')):
                            first = np.array([int(r['surface_first_truth']) for r in rr]); second = np.array([int(r['surface_second_truth']) for r in rr])
                            label, other = (first, second) if i == 0 else (second, first)
                            dm = dict(meta, arm='constituent', seed='not_B25_sample', position=position, head_id=p['head_ids'][i])
                            delta_records(dm, rr, bscore[:, i], score[:, i], label, other, deltas, ranks)
                            matched_rows.append(pd.DataFrame([dict(dm, example_id=r['example_id'], topic=r['topic'], ordering=r['ordering'], truth_a=r['truth_a'], truth_b=r['truth_b'], label=int(lab), other_truth=int(oth), bare_score=float(x), wording_score=float(y), delta=float(y-x)) for r, lab, oth, x, y in zip(rr, label, other, bscore[:, i], score[:, i])]))
                            if group == 'or_explicit_or_both_v1' and op == 'OR':
                                plot_panels.append(dict(category='constituent', title=f'{model} {source}→OR {position}, or both', rows=rr, bare=bscore[:, i], wording=score[:, i]))
                    print('completed selected constituent saved panel', model, source, op, group, flush=True)
    # Add same-support paired wording differences of marginal/conditional/decision AUROCs.
    cdf = pd.DataFrame(constituent_metrics); word_differences = []
    for (model, source, op), frame in cdf.groupby(['model', 'source_operator', 'evaluation_operator']):
        base = frame[frame.condition == 'D_bare'].set_index(['metric', 'scope', 'topic', 'surface_order'])
        for group in frame.condition.unique():
            if group == 'D_bare': continue
            word = frame[frame.condition == group].set_index(['metric', 'scope', 'topic', 'surface_order'])
            bp = inputs.constituent(model, source, 'D_bare', op); wp = inputs.constituent(model, source, group, op)
            bs = [r for r in bp['metrics']['records'] if r['surface_order'] == 'all']; ws = [r for r in wp['metrics']['records'] if r['surface_order'] == 'all']
            for i, (br, wr) in enumerate(zip(bs, ws)):
                require(all(br[k] == wr[k] for k in ('metric', 'scope', 'topic')), 'paired wording metrics support')
                word_differences.append(dict(model=model, source_operator=source, evaluation_operator=op, condition=group + '_minus_bare',
                    metric=br['metric'], scope=br['scope'], topic=br['topic'], surface_order='all',
                    point=None if br['point'] is None or wr['point'] is None else wr['point'] - br['point'], **ci(wp['bootstrap'][i] - bp['bootstrap'][i])))
    for name, records in [('boundary-cells', cells), ('boundary-decomposition', metrics), ('matched-deltas', deltas), ('matched-rank-changes', ranks),
                          ('constituent-endpoints', constituent_metrics), ('constituent-wording-differences', word_differences),
                          ('frozen-confusion-joint-boolean', decision_records), ('same-support-min-max', composition_extra), ('published-reconstruction', checks)]:
        pd.DataFrame(records).to_csv(out / (name + '.csv'), index=False)
    pd.concat(matched_rows, ignore_index=True).to_csv(out / 'arrays/matched-row-scores.csv', index=False)
    save(out / 'resolved-panels.json', dict(panels=panels, scores_used='saved D only', no_head_score_concatenation=True))
    plots(out, results, plot_panels)
    inputs.unchanged()
    save(out / 'input-verification.json', dict(status='passed', producer_sha=expected, source_files=inputs.used,
        paired_bootstrap_schedule_regenerated=True, fresh_feature_use='opaque-byte hashes only; no activations loaded',
        historical_tensor_use=False, behavior_chat_E_accessed=False, source_stats_unchanged=True))
    completion = dict(status='completed', producer_sha=expected, seconds=time.monotonic() - started, coverage_and_hash_verification_seconds=coverage_seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, hostname=platform.node(), interpreter=sys.executable,
        global_head_sample_panels=12, global_condition_panels=36, selected_constituent_panels=20,
        failures=[], coverage_gaps=cov['missing_unique_texts_per_model'], processes=1, threads=threadpool_info(),
        source_preservation='All accessed source hashes bound and pre/post size/mtime/ctime unchanged; original worktrees not edited.',
        output_files={str(p.relative_to(out)): pin(p) for p in sorted(out.rglob('*')) if p.is_file()})
    save(out / 'completion.json', completion)
    print(json.dumps({k: completion[k] for k in ('status', 'producer_sha', 'seconds', 'peak_rss_bytes', 'coverage_gaps')}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--expected-producer', required=True)
    args = parser.parse_args()
    def cap_hit(*_): raise TimeoutError('frozen analysis safety cap reached; preserve partial output')
    signal.signal(signal.SIGALRM, cap_hit)
    signal.alarm(read(CONFIG)['analysis_safety_cap_seconds'])
    try:
        with threadpool_limits(limits=1): run(args.output, args.expected_producer)
    except BaseException as exc:
        if args.output.is_dir(): save(args.output / 'failure.json', dict(type=type(exc).__name__, message=str(exc), producer_sha=args.expected_producer))
        raise


if __name__ == '__main__': main()
