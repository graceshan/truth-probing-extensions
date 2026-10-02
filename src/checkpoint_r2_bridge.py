"""Bounded compatibility diagnostics only; no model/cache loading or research metrics.

The caller supplies already-bound bridge arrays, token records and frozen-probe
scores. Fresh GPU producer integration remains pending runtime availability.
"""
import numpy as np


def compare_bridge(expected_ids, historical, fresh, historical_contract_verified=False):
    """Conservative exact contract. Unknown historical execution never earns reuse.

    Both records: ids, tokens, positions, masks, hidden [N,L,H], scores [N,K],
    contract. Compare *all* intended reuse layers and frozen diagnostic probes.
    Scores must be produced by the same frozen heads/preprocessing, never refit.
    This function computes neither predictions nor AUROC; caller binds head hashes.
    """
    if not expected_ids or len(expected_ids)>180 or len(set(expected_ids))!=len(expected_ids):
        raise ValueError('invalid bounded bridge identities')
    for r in (historical,fresh):
        if r['ids']!=expected_ids or r.get('split') not in ('train','validation','train_and_validation'):
            raise ValueError('wrong bridge identity/split; E is forbidden')
        if any(len(r[k])!=len(expected_ids) for k in ('tokens','positions','masks','hidden','scores')):
            raise ValueError('bridge row count')
    model=fresh['contract'].get('model')
    if model not in ('qwen','llama'):
        raise ValueError('unknown bridge model')
    layers,width={'qwen':(28,3584),'llama':(32,4096)}[model]
    if np.asarray(fresh['hidden']).shape!=(len(expected_ids),layers,width):
        raise ValueError('missing reuse layers or wrong hidden width')
    if np.asarray(fresh['scores']).ndim!=2 or np.asarray(fresh['scores']).shape[1]==0 or not fresh['probe_bindings']:
        raise ValueError('missing frozen-probe diagnostics')
    checks={k:historical[k]==fresh[k] for k in ('tokens','positions','masks','contract','probe_bindings')}
    errors={}
    for key in ('hidden','scores'):
        a,b=np.asarray(historical[key]),np.asarray(fresh[key])
        shape_ok=a.shape==b.shape and (a.ndim==3 if key=='hidden' else a.ndim==2)
        checks[key+'_shape']=shape_ok
        checks[key+'_finite']=bool(np.isfinite(a).all() and np.isfinite(b).all())
        checks[key+'_exact']=bool(shape_ok and a.dtype==b.dtype and a.tobytes()==b.tobytes())
        if shape_ok and checks[key+'_finite']:
            delta=np.abs(a.astype(np.float64)-b.astype(np.float64))
            scale=np.maximum(np.abs(a.astype(np.float64)),np.finfo(np.float64).tiny)
            with np.errstate(over='ignore'):
                rel=delta/scale
            errors[key]={'max_absolute':float(delta.max()),'mean_absolute':float(delta.mean()),'max_relative':float(rel.max()),'per_layer_max_absolute':delta.max(axis=(0,2)).tolist() if key=='hidden' else None}
    all_pass=all(checks.values())
    verdict='failed' if not all_pass else ('passed' if historical_contract_verified else 'unverifiable')
    return dict(verdict=verdict,checks=checks,errors=errors,reuse_permitted=all_pass and historical_contract_verified,
                diagnostic_only=True,criterion='exact bytes/tokens/positions/masks/contract/frozen heads; no aggregate AUROC',
                historical_execution_verified=historical_contract_verified)
