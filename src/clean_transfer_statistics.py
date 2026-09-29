"""Fixed weighted statistics and endpoint bootstrap; no data loading or fitting."""
import numpy as np

from src import clean_transfer_contracts as c


def divide(a, b):
    return np.divide(a, b, out=np.full(np.broadcast_shapes(np.shape(a), np.shape(b)), np.nan), where=np.asarray(b) != 0)


class AUC:
    """Sort once; weighted positive-negative comparisons with half-credit ties."""
    def __init__(self, indices, scores, labels):
        indices, scores, labels = np.asarray(indices, int), np.asarray(scores, float), np.asarray(labels)
        c.require(indices.shape == scores.shape == labels.shape and np.isfinite(scores).all() and
                  np.isin(labels, [0, 1]).all(), 'invalid AUROC inputs')
        order = np.argsort(scores, kind='stable')
        self.indices, self.labels = indices[order], labels[order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(scores[order])) + 1] if len(scores) else np.array([], int)

    def __call__(self, weights):
        if not len(self.indices):
            return np.full(len(weights), np.nan)
        w = weights[:, self.indices]
        pos = np.add.reduceat(w * self.labels, self.starts, axis=1)
        neg = np.add.reduceat(w * (1 - self.labels), self.starts, axis=1)
        numerator = (pos * (np.cumsum(neg, axis=1) - neg + .5 * neg)).sum(axis=1)
        return divide(numerator, pos.sum(axis=1) * neg.sum(axis=1))


def weighted_auc(labels, scores, weights=None):
    scores = np.asarray(scores, float)
    weights = np.ones(len(scores)) if weights is None else np.asarray(weights, float)
    c.require(weights.shape == scores.shape and scores.ndim == 1 and np.isfinite(weights).all() and
              (weights >= 0).all(), 'invalid AUROC weights')
    return float(AUC(np.arange(len(scores)), scores, labels)(weights[None])[0])


class Mean:
    def __init__(self, indices, values):
        self.indices, self.values = np.asarray(indices, int), np.asarray(values, float)

    def __call__(self, weights):
        w = weights[:, self.indices]
        return divide((w * self.values).sum(axis=1), w.sum(axis=1))


class EntityBootstrap:
    def __init__(self, frame, options):
        pairs = frame[['pair_id', 'topic', 'entity_a_id', 'entity_b_id']].drop_duplicates().sort_values('pair_id').reset_index(drop=True)
        c.require(pairs.pair_id.is_unique, 'inconsistent pair endpoints')
        self.pair_ids = pairs.pair_id.tolist()
        lookup = {p: i for i, p in enumerate(self.pair_ids)}
        self.row_indices = np.asarray([lookup[p] for p in frame.pair_id])
        self.weights = np.zeros((options['replicates'], len(pairs)), dtype=np.int64)
        self.multiplicities, self.entities = {}, {}
        rng = np.random.Generator(np.random.PCG64(options['seed']))
        for topic in sorted(pairs.topic.unique()):
            group = pairs[pairs.topic == topic]
            entities = sorted(set(group.entity_a_id) | set(group.entity_b_id))
            index = {e: i for i, e in enumerate(entities)}
            counts = rng.multinomial(len(entities), np.full(len(entities), 1 / len(entities)), size=options['replicates'])
            self.weights[:, group.index] = counts[:, [index[e] for e in group.entity_a_id]] * counts[:, [index[e] for e in group.entity_b_id]]
            self.multiplicities[topic], self.entities[topic] = counts, entities
        self.sha256 = c.digest(c.canonical(dict(pairs=self.pair_ids, entities=self.entities, options=options)) +
                               self.weights.astype('<i8').tobytes())

    def row_weights(self, start, stop):
        return self.weights[start:stop, self.row_indices]


def interval(samples, options):
    samples = np.asarray(samples, float)
    valid = samples[np.isfinite(samples)]
    enough = len(valid) >= options['minimum_valid']
    low, high = np.quantile(valid, [.025, .975], method='linear') if enough else (None, None)
    return dict(total_replicates=len(samples), valid_replicates=len(valid), invalid_replicates=len(samples)-len(valid),
                valid_fraction=len(valid)/len(samples), ci_low=None if low is None else float(low),
                ci_high=None if high is None else float(high),
                ci_status='ok' if enough else 'insufficient_valid_replicates')


def describe(values):
    values = np.asarray(values, float)
    c.require(len(values) > 0 and np.isfinite(values).all(), 'empty/nonfinite descriptive group')
    qs = np.quantile(values, [.1, .25, .5, .75, .9], method='linear')
    return dict(n=len(values), mean=float(values.mean()), std=float(values.std(ddof=0)),
                **dict(zip(['q10', 'q25', 'median', 'q75', 'q90'], map(float, qs))))


class MetricPlan:
    """Versioned metric definitions evaluated on one shared row-weight schedule."""
    def __init__(self, frame, matched, spec):
        self.order = np.argsort(frame.example_id.to_numpy(), kind='stable')
        frame = frame.iloc[self.order].reset_index(drop=True)
        self.frame, self.spec = frame, spec
        self.records, self.functions = [], []
        scores = frame.frozen_probe_score.to_numpy(float)
        labels = frame.compound_label.to_numpy(int)
        cells, operators = frame.cell.to_numpy(), frame.operator.to_numpy()
        predicted = scores >= 0
        lookup = {eid: i for i, eid in enumerate(frame.example_id)}
        and_indices = np.array([lookup[e] for e in matched.example_id_and])
        deltas = matched.delta_or_minus_and.to_numpy(float)
        scopes = [('pooled', 'all')] + [('topic', t) for t in spec['benchmark']['topics']]
        for scope, topic in scopes:
            mask = np.ones(len(frame), bool) if scope == 'pooled' else (frame.topic == topic).to_numpy()
            current = {}
            for definition in spec['metrics']:
                kind, name = definition['kind'], definition['id']
                selected = mask & (operators == definition['operator']) if 'operator' in definition else mask.copy()
                if kind == 'auc':
                    pos, neg = np.isin(cells, definition['positive']), np.isin(cells, definition['negative'])
                    idx = np.flatnonzero(selected & (pos | neg))
                    function = AUC(idx, scores[idx], pos[idx])
                elif kind == 'compound_auc':
                    idx = np.flatnonzero(mask)
                    function = AUC(idx, scores[idx], labels[idx])
                elif kind == 'difference':
                    function = ('difference', [current[x] for x in definition['operands']])
                elif kind in ('accuracy', 'true_response'):
                    if kind == 'true_response':
                        selected &= cells == definition['cell']
                    idx = np.flatnonzero(selected)
                    function = Mean(idx, (predicted if kind == 'true_response' else predicted == labels)[idx])
                elif kind == 'balanced_accuracy':
                    means = [Mean(np.flatnonzero(selected & (labels == y)),
                                  (predicted == labels)[selected & (labels == y)]) for y in (0, 1)]
                    function = lambda w, means=means: .5 * (means[0](w) + means[1](w))
                elif kind == 'matched_mean':
                    keep = np.ones(len(matched), bool) if scope == 'pooled' else (matched.topic == topic).to_numpy()
                    if definition['cell'] != 'all':
                        keep = keep & (matched.cell.to_numpy() == definition['cell'])
                    function = Mean(and_indices[keep], deltas[keep])
                else:
                    raise ValueError('unknown metric kind')
                current[name] = len(self.records)
                self.records.append(dict(metric_id=f'{scope}/{topic}/{name}', metric=name, scope=scope, topic=topic,
                                         category=definition['category'], primary=definition['category'] == 'primary',
                                         operator=definition.get('operator', ''), cell=definition.get('cell', 'all')))
                self.functions.append(function)
        for definition in spec['metrics']:
            indices = [i for i, r in enumerate(self.records) if r['scope'] == 'topic' and r['metric'] == definition['id']]
            c.require(len(indices) == 5, 'macro requires exactly five topics')
            self.records.append(dict(metric_id='topic_macro/all/' + definition['id'], metric=definition['id'],
                                     scope='topic_macro', topic='all', category=definition['category'],
                                     primary=definition['category'] == 'primary', operator=definition.get('operator', ''),
                                     cell=definition.get('cell', 'all')))
            self.functions.append(('mean', indices))

    def evaluate(self, weights):
        weights = np.asarray(weights, float)
        c.require(weights.ndim == 2 and weights.shape[1] == len(self.frame) and np.isfinite(weights).all() and
                  (weights >= 0).all(), 'invalid metric weights')
        weights = weights[:, self.order]
        result = np.empty((len(weights), len(self.records)))
        for i, fn in enumerate(self.functions):
            if isinstance(fn, tuple):
                op, indices = fn
                result[:, i] = result[:, indices[0]] - result[:, indices[1]] if op == 'difference' else result[:, indices].mean(axis=1)
            else:
                result[:, i] = fn(weights)
        return result
