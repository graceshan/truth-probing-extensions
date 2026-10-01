"""Prespecified saved-D surrogate precision; no fitting or activation/E score access.

The count-only transport is a conditional approximation, not E power. The
configuration was committed before any contrast calculation. Historical code
and artifacts remain immutable inputs.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.clean_transfer_statistics import AUC, EntityBootstrap, interval

CONFIG = Path('config/clean_protocol/precision_proxy_v1.json')
CONFIG_SHA = 'a5dfe794d68c2490bfa6fa35d383f98559c4d8cc8e4122e5c0ab7f7b90d21dfb'
FREEZE_COMMIT = 'cd112e9'
CORRECTION = 'fd2d7be3b910c3fdba7ff6b2c2891eb0ab1b7992'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def binding(raw):
    return dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))


def check_bytes(raw, expected, name):
    require(binding(raw) == {k: expected[k] for k in ('sha256', 'bytes')},
            f'stale or mixed-version input: {name}')
    return raw


def read_bound(path, expected):
    return check_bytes(Path(path).read_bytes(), expected, str(path))


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + '\n')


def config_at(root):
    raw = (root / CONFIG).read_bytes()
    require(binding(raw)['sha256'] == CONFIG_SHA, 'predeclared configuration changed')
    committed = subprocess.check_output(['git', 'show', f'{FREEZE_COMMIT}:{CONFIG}'], cwd=root)
    require(raw == committed, 'configuration must match pre-result checkpoint')
    return json.loads(raw)


def validate_rows(atomic, bare, topics):
    require(atomic.source_row_id.is_unique and bare.example_id.is_unique,
            'duplicate row identities')
    require(not atomic[['source_row_id', 'person_key', 'topic', 'label']].isna().any().any(),
            'missing atomic row identity')
    require(set(atomic.topic) == set(bare.topic) == set(topics), 'missing topic')
    require(set(atomic.label) == {0, 1}, 'nonbinary labels')
    cols = ['pair_id', 'topic', 'entity_a_id', 'entity_b_id']
    pairs = bare[cols].drop_duplicates()
    require(pairs.pair_id.is_unique, 'inconsistent pair endpoints')
    require((pairs.entity_a_id != pairs.entity_b_id).all(), 'self pair')
    require(not pairs.isna().any().any(), 'missing pair identity')
    require(bare.groupby('pair_id').size().eq(16).all(), 'incomplete variant group')
    require(bare.groupby(['pair_id', 'operator', 'cell']).size().eq(2).all(),
            'truth/surface groups not complete')
    require(set(bare.operator) == {'AND', 'OR'} and set(bare.cell) == {'TT', 'TF', 'FT', 'FF'},
            'unexpected endpoint definitions')
    require(not bare.duplicated(['pair_id', 'operator', 'cell', 'ordering']).any(),
            'duplicate surface order')
    labels = np.where(bare.operator == 'AND', bare.cell == 'TT', bare.cell != 'FF').astype(int)
    require(np.array_equal(labels, bare.compound_label), 'compound label mismatch')


def graph_summary(pairs):
    """Metadata-only graph support, including topology beyond degree counts."""
    output = {}
    for topic, group in pairs.groupby('topic', sort=True):
        edges = [tuple(sorted((a, b))) for a, b in zip(group.entity_a_id, group.entity_b_id)]
        require(len(edges) == len(set(edges)), 'duplicate unordered pair')
        adjacency = {}
        for a, b in edges:
            require(a != b, 'self pair')
            adjacency.setdefault(a, set()).add(b)
            adjacency.setdefault(b, set()).add(a)
        remaining = set(adjacency)
        sizes = []
        while remaining:
            stack = [min(remaining)]
            seen = set()
            while stack:
                v = stack.pop()
                if v not in seen:
                    seen.add(v)
                    stack.extend(adjacency[v] - seen)
            sizes.append(len(seen))
            remaining -= seen
        degrees = np.array([len(v) for v in adjacency.values()])
        triangles = sum(len(adjacency[a] & adjacency[b]) for a, b in edges) // 3
        output[topic] = dict(entities=len(adjacency), pairs=len(edges),
                             degree_min=int(degrees.min()), degree_max=int(degrees.max()),
                             degree_mean=float(degrees.mean()),
                             degree_counts={str(k): int((degrees == k).sum()) for k in sorted(set(degrees))},
                             component_sizes=sorted(sizes, reverse=True), triangles=triangles,
                             all_degree_four=bool((degrees == 4).all()))
    return output


def atomic_summary(frame):
    result = {}
    for topic, group in frame.groupby('topic', sort=True):
        sizes = group.groupby('person_key').size()
        label_profiles = group.groupby('person_key').label.agg(['count', 'sum'])
        result[topic] = dict(rows=len(group), persons=group.person_key.nunique(),
                             labels={str(k): int(v) for k, v in group.label.value_counts().sort_index().items()},
                             cluster_size_counts={str(k): int(v) for k, v in sizes.value_counts().sort_index().items()},
                             persons_with_both_labels=int(((label_profiles['sum'] > 0) &
                                                           (label_profiles['sum'] < label_profiles['count'])).sum()))
    return result


def schedules(atomic, bare, options):
    rng = np.random.Generator(np.random.PCG64(options['seed']))
    atomic_w = np.zeros((options['replicates'], len(atomic)), dtype=np.int16)
    for topic in sorted(atomic.topic.unique()):
        idx = np.flatnonzero(atomic.topic.to_numpy() == topic)
        persons = sorted(set(atomic.iloc[idx].person_key))
        lookup = {p: i for i, p in enumerate(persons)}
        counts = rng.multinomial(len(persons), np.full(len(persons), 1 / len(persons)),
                                 size=options['replicates'])
        atomic_w[:, idx] = counts[:, [lookup[p] for p in atomic.iloc[idx].person_key]]
    compound = EntityBootstrap(bare, options)
    bare_w = compound.row_weights(0, options['replicates']).astype(np.int16)
    support = []
    pairs = bare[['pair_id', 'topic']].drop_duplicates().set_index('pair_id')
    for topic in sorted(bare.topic.unique()):
        idx = [i for i, p in enumerate(compound.pair_ids) if pairs.loc[p, 'topic'] == topic]
        counts = (compound.weights[:, idx] > 0).sum(axis=1)
        support.append(dict(topic=topic, retained_pairs=len(idx),
                            positive_weight_pairs_min=int(counts.min()),
                            positive_weight_pairs_median=float(np.median(counts)),
                            positive_weight_pairs_max=int(counts.max()),
                            zero_edge_replicates=int((counts == 0).sum())))
    receipt = dict(atomic_weight_sha256=hashlib.sha256(atomic_w.tobytes()).hexdigest(),
                   compound_schedule_sha256=compound.sha256,
                   compound_weight_sha256=hashlib.sha256(bare_w.tobytes()).hexdigest(),
                   options=options, graph_bootstrap_support=support,
                   shared_across_all_methods_models_cohorts=True)
    return dict(atomic=atomic_w, bare=bare_w), receipt


def endpoint(frame, definition):
    if definition['kind'] == 'atomic':
        return np.ones(len(frame), bool), frame.label.to_numpy(int)
    cells = frame.cell.to_numpy()
    mask = ((frame.operator.to_numpy() == definition['operator']) &
            np.isin(cells, definition['positive'] + definition['negative']))
    return mask, np.isin(cells, definition['positive']).astype(int)


def evaluate(scores, labels, idx, weights, base_weights=None):
    base = np.ones(len(scores)) if base_weights is None else np.asarray(base_weights, float)
    require(base.shape == scores.shape and (base >= 0).all() and np.isfinite(base).all(),
            'invalid example weights')
    fn = AUC(idx, scores[idx], labels[idx])
    point = float(fn(base[None])[0])
    draws = np.concatenate([fn(weights[i:i + 64] * base[None])
                            for i in range(0, len(weights), 64)])
    return point, draws


def macro(values):
    """Undefined topics propagate: never silently change the macro estimand."""
    return np.mean(np.asarray(values), axis=0)


def precision_interval(draws, center, options):
    result = interval(draws, options)
    lo, hi = result['ci_low'], result['ci_high']
    result['half_width'] = None if lo is None else (hi - lo) / 2
    result['max_center_distance'] = None if lo is None else max(abs(center - lo), abs(hi - center))
    return result


def projection_supported(kind, scope, topic, diagnostics):
    if kind == 'bare':
        if scope != 'topic':
            return False, 'No pooled graph transport defined; macro requires all topics, including unsupported inventors.'
        d, e = diagnostics['D_graph'][topic], diagnostics['E_graph'][topic]
        if not (d['all_degree_four'] and e['all_degree_four']):
            return False, 'Retained D degree profile differs from E degree-four recipe; count scaling unsupported.'
        return True, 'Approximate count-only transport; different graphs and score populations remain unverified.'
    if scope != 'topic' and any(diagnostics['D_atomic'][t]['persons'] != diagnostics['E_atomic'][t]['persons']
                                for t in diagnostics['D_atomic']):
        return False, 'Aggregate atomic transport requires equal per-topic person counts.'
    return True, 'Approximate person-count transport; E cluster row/label profiles and score distributions unmodeled.'


def load_inputs(root, config):
    frames, scores, receipts = {}, {}, {}
    for cohort, spec in config['inputs'].items():
        for key in ['external_manifest', 'completion']:
            read_bound(root / spec[key]['path'], spec[key])
        completion = json.loads((root / spec['completion']['path']).read_bytes())
        require(completion['correction_commit'] == CORRECTION and not completion['P_A_frozen'],
                'fit correction provenance changed')
        external = Path(spec['root'])
        require(json.loads((root / spec['external_manifest']['path']).read_bytes())['pilot_root'] == str(external),
                'external root mismatch')
        require((external / 'completion.json').read_bytes() == (root / spec['completion']['path']).read_bytes(),
                'external completion mismatch')
        raw = {}
        for name, expected in spec['artifacts'].items():
            require(expected == completion['artifacts'][name], 'artifact not bound by completion')
            raw[name] = read_bound(external / name, expected)
        current_frames = {kind: pd.read_csv(io.BytesIO(raw[name])) for kind, name in
                          [('atomic', 'atomic_D_rows.csv'), ('bare', 'bare_D_rows.csv')]}
        validate_rows(current_frames['atomic'], current_frames['bare'], config['topics'])
        if frames:
            for kind in frames:
                require(frames[kind].equals(current_frames[kind]), 'ordered D row alignment mismatch')
        frames = current_frames
        receipts[cohort] = spec
        for model, layer in config['models'].items():
            for method in [config['reference']] + config['contrasts']:
                name = f'{model}_L{layer}_{method}_{cohort}'
                fit = json.loads(raw[f'fits/{name}/fit.json'])
                require(fit['valid_for_scoring'], 'flagged saved fit')
                require(all(fit['configuration'][k] == v for k, v in
                            dict(model=model, layer=layer, method=method, cohort=cohort).items()),
                        'saved fit configuration mismatch')
                with np.load(io.BytesIO(raw[f'scores/{name}.npz']), allow_pickle=False) as archive:
                    require(set(archive.files) == {'atomic', 'bare'}, 'unexpected score archive')
                    scores[(cohort, model, method)] = {}
                    for kind in frames:
                        a = archive[kind]
                        require(a.shape == (len(frames[kind]),) and np.isfinite(a).all(),
                                'nonfinite or misaligned scores')
                        scores[(cohort, model, method)][kind] = a
    require(len(frames['atomic']) == 1012 and len(frames['bare']) == 7728, 'unexpected retained D size')
    e = {}
    for name, pin in config['E_metadata'].items():
        git_path = f"{config['E_metadata_commit']}:{pin['path']}"
        blob = subprocess.check_output(['git', 'rev-parse', git_path], cwd=root, text=True).strip()
        require(blob == pin['blob'], 'E Git blob mismatch')
        raw = subprocess.check_output(['git', 'show', git_path], cwd=root)
        check_bytes(raw, pin, name)
        e[name] = pd.read_csv(io.BytesIO(raw)) if name.endswith('.csv') else json.loads(raw)
    for name, pin in config['E_metadata'].items():
        if name != 'manifest.json':
            require(e['manifest.json']['output_sha256'][name] == pin['sha256'], 'E manifest mismatch')
    ea, ep, ed = e['admitted_atomic_E.csv'], e['pairs.csv'], e['entity_degrees.csv']
    require(len(ea) == config['expected_atomic_E_rows'] and ea.source_row_id.is_unique,
            'E admitted atomic identity/count mismatch')
    require(ep.pair_id.is_unique and not ed.duplicated(['topic', 'entity_id']).any(), 'E graph duplicate')
    require(set(ea.e_successor_status) == {'admitted'}, 'E current status not admitted')
    diagnostics = dict(D_graph=graph_summary(frames['bare'][['pair_id', 'topic', 'entity_a_id', 'entity_b_id']].drop_duplicates()),
                       E_graph=graph_summary(ep), D_atomic=atomic_summary(frames['atomic']), E_atomic=atomic_summary(ea))
    for topic in config['topics']:
        g = ep[ep.topic == topic]
        degrees = pd.concat([g.entity_a_id, g.entity_b_id]).value_counts().sort_index()
        recorded = ed[ed.topic == topic].set_index('entity_id').degree.sort_index()
        require(degrees.to_dict() == recorded.to_dict(), 'E degree sidecar mismatch')
        require(len(degrees) == config['expected_compound_E_entities'][topic], 'E entity count mismatch')
        c = e['counts.csv'].set_index('topic').loc[topic]
        require(int(c.pairs) == len(g) and int(c.admitted_atomic_rows) == diagnostics['E_atomic'][topic]['rows'],
                'E count receipt mismatch')
    return frames, scores, diagnostics, receipts


def compute(config, frames, scores, weights, diagnostics):
    records, draw_columns = [], []
    for cohort in config['cohorts']:
        for model, layer in config['models'].items():
            for method in config['contrasts']:
                for definition in config['endpoints']:
                    kind = definition['kind']
                    frame = frames[kind]
                    mask, labels = endpoint(frame, definition)
                    current = scores[(cohort, model, method)][kind]
                    baseline = scores[(cohort, model, config['reference'])][kind]
                    groups = [('pooled', 'all', np.ones(len(frame), bool))] + [
                        ('topic', t, frame.topic.to_numpy() == t) for t in config['topics']]
                    native, projected = {}, {}
                    for scope, topic, group in groups:
                        idx = np.flatnonzero(mask & group)
                        native[(scope, topic)] = (evaluate(current, labels, idx, weights[kind]),
                                                  evaluate(baseline, labels, idx, weights[kind]))
                        allowed, _ = projection_supported(kind, scope, topic, diagnostics)
                        if allowed:
                            base = np.ones(len(frame))
                            if kind == 'atomic' and scope == 'pooled':
                                base = np.array([diagnostics['E_atomic'][t]['rows'] /
                                                 diagnostics['D_atomic'][t]['rows'] for t in frame.topic])
                            projected[(scope, topic)] = (evaluate(current, labels, idx, weights[kind], base),
                                                         evaluate(baseline, labels, idx, weights[kind], base))
                    native[('topic_macro', 'all')] = tuple(
                        (float(macro([native[('topic', t)][j][0] for t in config['topics']])),
                         macro([native[('topic', t)][j][1] for t in config['topics']])) for j in (0, 1))
                    if kind == 'atomic':
                        projected[('topic_macro', 'all')] = tuple(
                            (float(macro([projected[('topic', t)][j][0] for t in config['topics']])),
                             macro([projected[('topic', t)][j][1] for t in config['topics']])) for j in (0, 1))
                    for scope, topic in native:
                        for mode in ['native_D', 'approximate_E_count_proxy']:
                            key = dict(cohort=cohort, model=model, saved_layer=layer, method=method,
                                       reference=config['reference'], endpoint=definition['id'], role=definition['role'],
                                       scope=scope, topic=topic, precision_mode=mode)
                            allowed, reason = projection_supported(kind, scope, topic, diagnostics)
                            if mode != 'native_D' and not allowed:
                                records.append(dict(**key, status='unsupported', reason=reason,
                                                    total_replicates=0, valid_replicates=0, invalid_replicates=0,
                                                    ci_status='not_attempted', draw_column=None))
                                continue
                            source = native if mode == 'native_D' else projected
                            (point, draws), (ref_point, ref_draws) = source[(scope, topic)]
                            center = point - ref_point
                            delta = draws - ref_draws
                            factor = 1.0
                            if mode != 'native_D' and scope == 'topic':
                                table, count = ('graph', 'entities') if kind == 'bare' else ('atomic', 'persons')
                                factor = np.sqrt(diagnostics['D_' + table][topic][count] /
                                                 diagnostics['E_' + table][topic][count])
                                delta = center + factor * (delta - center)
                            interval_result = precision_interval(delta, center, config['bootstrap'])
                            records.append(dict(**key, status='computed', reason='Native conditional paired bootstrap' if mode == 'native_D' else reason,
                                                method_D_proxy_auroc=point, reference_D_proxy_auroc=ref_point,
                                                delta_D_proxy=center, transport_scale=float(factor),
                                                draw_column=len(draw_columns), **interval_result))
                            draw_columns.append(delta)
    return records, np.stack(draw_columns, axis=1)


def run(root, output, external):
    started = time.monotonic()
    require(platform.system() == 'Darwin', 'run this task in local macOS')
    config = config_at(root)
    require(not output.exists() and not external.exists(), 'refuse to overwrite precision outputs')
    frames, scores, diagnostics, receipts = load_inputs(root, config)
    weights, schedule = schedules(frames['atomic'], frames['bare'], config['bootstrap'])
    records, draws = compute(config, frames, scores, weights, diagnostics)
    require(len(records) == 336 and len({tuple(r[k] for k in ['cohort', 'model', 'method', 'endpoint', 'scope', 'topic', 'precision_mode']) for r in records}) == 336,
            'prespecified coverage failure')
    external.mkdir(parents=True)
    np.savez_compressed(external / 'paired_contrast_draws.npz', delta=draws)
    output.mkdir(parents=True)
    dump(output / 'results.json', records)
    pd.DataFrame(records).to_csv(output / 'results.csv', index=False)
    dump(output / 'support.json', diagnostics)
    dump(output / 'bootstrap_binding.json', schedule)
    dump(output / 'input_bindings.json', dict(configuration=binding((root / CONFIG).read_bytes()),
                                             pre_result_config_commit=FREEZE_COMMIT, inputs=receipts,
                                             E_metadata_commit=config['E_metadata_commit'], E_metadata=config['E_metadata'],
                                             code={p: binding((root / p).read_bytes()) for p in
                                                   ['src/selection_repair_precision_proxy_v1.py', 'src/clean_transfer_statistics.py',
                                                    'scripts/64_precision_proxy_v1.py']}))
    dump(output / 'completion.json', dict(status='completed', hostname=platform.node(), platform=platform.platform(),
                                          elapsed_seconds=time.monotonic() - started,
                                          contrasts=8, native_summaries=168, projected_computed=sum(r['status'] == 'computed' and r['precision_mode'] != 'native_D' for r in records),
                                          new_fits=0, E_scores_accessed=False, P_A_frozen=False,
                                          correction_provenance='v4', runtime=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__),
                                          artifacts={p.name: binding(p.read_bytes()) for p in output.iterdir()},
                                          external={str(external / 'paired_contrast_draws.npz'): binding((external / 'paired_contrast_draws.npz').read_bytes())}))
    return records
