"""Revision 2 pure selection helpers. No fitting, file access, or production bank.

Callers must bind audited membership, score ordering, and representation before
production use. The source role guard is not a substitute for provenance checks.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.clean_transfer_statistics import weighted_auc

TOPICS = ('animal_class', 'cities', 'element_symb', 'inventors', 'sp_en_trans')
ENDPOINTS = ('first_marginal', 'second_marginal', 'first_given_second_0',
             'first_given_second_1', 'second_given_first_0', 'second_given_first_1')
TOLERANCE = 1e-12
SEED = 20261005


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def candidate_id(model, layer, method, C=None):
    require(model in ('qwen', 'llama') and type(layer) is int and 0 <= layer < (28 if model == 'qwen' else 32),
            'invalid model or saved layer')
    if method == 'l2_logistic':
        require(C in (.001, .01, .1, 1., 10.), 'undeclared C')
        suffix = f'/C={C:g}'
    else:
        require(method in ('difference_of_means', 'mass_mean_covariance', 'burger_t_g', 'ttpd', 'r0') and C is None,
                'invalid method or irrelevant C')
        suffix = ''
    return f'{model}/P15/L{layer:02d}/{method}{suffix}'


@dataclass(frozen=True)
class AtomicCandidate:
    """Only atomic summaries are accepted: no compound scores/labels/gates."""
    candidate_id: str
    model: str
    layer: int
    method: str
    C: float | None
    fit_valid: bool
    pooled_atomic_auc: float | None
    topic_atomic_auc: dict


def atomic_eligibility(candidates, model, reduced_lr_atomic_auc):
    """Shared S_all/S_atom_all retention bank; failures remain in the receipt."""
    require(model in ('qwen', 'llama'), 'invalid model')
    require(reduced_lr_atomic_auc is not None and np.isfinite(reduced_lr_atomic_auc)
            and 0 <= reduced_lr_atomic_auc <= 1, 'undefined reduced LR reference')
    require(len({c.candidate_id for c in candidates}) == len(candidates), 'duplicate candidate identity')
    records = []
    for c in sorted(candidates, key=lambda c: c.candidate_id):
        require(c.model == model, 'select independently per model; mixed model bank')
        require(type(c.fit_valid) is bool, 'fit_valid must be an explicit boolean')
        require(c.candidate_id == candidate_id(c.model, c.layer, c.method, c.C), 'noncanonical candidate ID')
        values = [c.pooled_atomic_auc] + [c.topic_atomic_auc.get(t) for t in TOPICS]
        reasons = []
        if not c.fit_valid:
            reasons.append('invalid_fit')
        if set(c.topic_atomic_auc) != set(TOPICS) or any(v is None or not np.isfinite(v) or not 0 <= v <= 1 for v in values):
            reasons.append('missing_or_undefined_atomic_endpoint')
        elif c.pooled_atomic_auc < reduced_lr_atomic_auc - .005:
            reasons.append('atomic_retention_failed')
        records.append(dict(candidate_id=c.candidate_id, eligible=not reasons, reasons=reasons,
                            pooled_atomic_auc=c.pooled_atomic_auc if c.pooled_atomic_auc is not None and np.isfinite(c.pooled_atomic_auc) else None,
                            macro_atomic_auc=None if any(v is None or not np.isfinite(v) for v in values[1:])
                            else float(np.mean(values[1:]))))
    return records


def select_atom_all(candidates, *, model, reduced_lr_atomic_auc, expected_s_all_eligible_ids):
    """Select once per model; caller supplies S_all's exact retention-eligible IDs.

    Ties are relative to the maximum, not chained approximate comparisons.
    A fresh PCG64 generator per model permutes sorted final IDs. Empty/invalid
    banks return a visible no-valid-candidate receipt, never an arbitrary fallback.
    """
    records = atomic_eligibility(candidates, model, reduced_lr_atomic_auc)
    eligible = [r for r in records if r['eligible']]
    ids = sorted(r['candidate_id'] for r in eligible)
    require(ids == sorted(expected_s_all_eligible_ids) and len(ids) == len(set(expected_s_all_eligible_ids)),
            'S_all eligibility identity mismatch')
    receipt = dict(selector='S_atom_all', model=model, eligibility=records, eligible_ids=ids,
                   seed=SEED, tolerance=TOLERANCE, pooled_tie_ids=[], macro_tie_ids=[], randomized_tie_order=[],
                   selected_id=None, status='no_valid_candidate', compound_inputs_used=False)
    if not eligible:
        return receipt
    best = max(r['pooled_atomic_auc'] for r in eligible)
    pooled = [r for r in eligible if best - r['pooled_atomic_auc'] <= TOLERANCE]
    best_macro = max(r['macro_atomic_auc'] for r in pooled)
    final = sorted(r['candidate_id'] for r in pooled if best_macro - r['macro_atomic_auc'] <= TOLERANCE)
    order = np.random.Generator(np.random.PCG64(SEED)).permutation(final).tolist()
    receipt.update(status='selected', selected_id=order[0], best_pooled=best, best_macro_among_pooled_ties=best_macro,
                   pooled_tie_ids=sorted(r['candidate_id'] for r in pooled), macro_tie_ids=final, randomized_tie_order=order)
    return receipt


def select_original_atomic_r0(candidates, *, model):
    """Separate original reference: pooled atomic AUROC, EXACT ties lower layer."""
    valid = [c for c in candidates if c.model == model and c.method == 'r0' and c.fit_valid and
             c.pooled_atomic_auc is not None and np.isfinite(c.pooled_atomic_auc) and 0 <= c.pooled_atomic_auc <= 1]
    if not valid:
        return dict(selector='original_atomic_R0', status='no_valid_candidate', selected_id=None)
    best = max(c.pooled_atomic_auc for c in valid)
    tied = sorted((c for c in valid if c.pooled_atomic_auc == best), key=lambda c: c.layer)
    return dict(selector='original_atomic_R0', status='selected', selected_id=tied[0].candidate_id,
                exact_tie_ids=[c.candidate_id for c in tied], rule='exact pooled tie then lower layer')


def binary(values):
    require(all(v in (False, True, 0, 1) for v in values), 'truth labels must be binary values, not strings')
    return np.asarray(values, dtype=int)


def surface_labels(rows):
    """Canonical A/B truth must be swapped for BA; never derive it from text."""
    a, b = binary(rows.canonical_truth_a), binary(rows.canonical_truth_b)
    order = rows.ordering.to_numpy()
    require(np.isin(order, ['AB', 'BA']).all(), 'unknown surface order')
    first, second = np.where(order == 'AB', a, b), np.where(order == 'AB', b, a)
    for key, expected in [('surface_first_truth', first), ('surface_second_truth', second)]:
        if key in rows:
            require(np.array_equal(binary(rows[key]), expected), 'surface truth mismatch')
    return first, second


def constituent_endpoints(rows, scores):
    """Evaluation primitive usable on separately bound source or target data.

    Equal row weights on complete per-pair truth/order groups. Every missing
    topic/stratum is explicit and invalidates the six-topic macro selector.
    """
    first, second = surface_labels(rows)
    scores = np.asarray(scores, dtype=np.float64)
    require(scores.shape == (len(rows), 2) and np.isfinite(scores).all(), 'invalid constituent scores')
    masks = [np.ones(len(rows), bool), np.ones(len(rows), bool), second == 0, second == 1, first == 0, first == 1]
    labels = [first, second, first, first, second, second]
    heads = [0, 1, 0, 0, 1, 1]
    values = {}
    for topic in (*TOPICS, 'pooled'):
        group = np.ones(len(rows), bool) if topic == 'pooled' else rows.topic.to_numpy() == topic
        values[topic] = {}
        for name, mask, label, head in zip(ENDPOINTS, masks, labels, heads):
            idx = mask & group
            values[topic][name] = None if len(np.unique(label[idx])) < 2 else weighted_auc(label[idx], scores[idx, head])
    macros = {e: None if any(values[t][e] is None for t in TOPICS)
              else float(np.mean([values[t][e] for t in TOPICS])) for e in ENDPOINTS}
    valid = all(v is not None for v in macros.values())
    return dict(per_topic={t: values[t] for t in TOPICS}, pooled=values['pooled'], macro=macros,
                valid=valid, minimum_macro=min(macros.values()) if valid else None,
                mean_macro=float(np.mean(list(macros.values()))) if valid else None)


def select_constituent_layer(*, source_operator, role, validation_binding, rows, scores_by_layer):
    """Only internal source validation enters selection; no target/D argument."""
    require(source_operator in ('AND', 'OR') and role == 'T_C_source_validation', 'source-only selection role required')
    require(isinstance(validation_binding, str) and bool(validation_binding), 'bound source-validation identity required')
    require(set(rows.operator) == {source_operator}, 'target operator leaked into source selection')
    require(set(rows.topic) <= set(TOPICS), 'unknown topic')
    require(rows.example_id.is_unique, 'duplicate validation example')
    # Membership/group completeness is an upstream production gate. Undefined
    # strata here stay visible, rather than deleting an incomplete topic.
    curves = []
    for layer, scores in sorted(scores_by_layer.items()):
        require(type(layer) is int and layer >= 0, 'invalid saved layer')
        try:
            endpoints = constituent_endpoints(rows, scores)
            curves.append(dict(layer=layer, status='valid' if endpoints['valid'] else 'undefined_endpoint', **endpoints))
        except ValueError as exc:
            curves.append(dict(layer=layer, status='invalid_scores', valid=False, reason=str(exc)))
    valid = [r for r in curves if r['valid']]
    receipt = dict(selector='source_only_six_endpoint', source_operator=source_operator,
                   role=role, validation_binding=validation_binding, curves=curves,
                   selected_layer=None, status='no_valid_layer', minimum_tie_layers=[], mean_tie_layers=[],
                   tolerance=TOLERANCE, target_or_D_used=False)
    if not valid:
        return receipt
    best = max(r['minimum_macro'] for r in valid)
    first = [r for r in valid if best - r['minimum_macro'] <= TOLERANCE]
    mean = max(r['mean_macro'] for r in first)
    final = [r for r in first if mean - r['mean_macro'] <= TOLERANCE]
    receipt.update(status='selected', selected_layer=min(r['layer'] for r in final),
                   minimum_tie_layers=[r['layer'] for r in first], mean_tie_layers=[r['layer'] for r in final],
                   best_minimum=best, best_mean_among_minimum_ties=mean)
    return receipt


def average_sample_metrics(sample_metrics, *, seeds=(11, 23, 37)):
    """Average already-computed metrics/draws; undefined samples propagate."""
    require(set(sample_metrics) == set(seeds), 'missing or unexpected adaptation sample')
    values = [np.asarray(sample_metrics[s], dtype=float) for s in seeds]
    require(len({v.shape for v in values}) == 1, 'sample draw alignment mismatch')
    return np.mean(values, axis=0)
