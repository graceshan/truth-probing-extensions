"""Prespecified capacity methods; fixed cohort-local transforms and label-blind inference."""
from dataclasses import asdict
import warnings
import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from src import atomic_probe_methods as old
from src import selection_repair_objectives as objective
from src.selection_repair_canonical_sensitivity import fit_reference, gradient_diagnostics

METHODS=('l2_logistic','difference_of_means','mass_mean_covariance','burger_t_g','ttpd','r0')


def unregularized(X,y):
    """Original TTPD 100/10000 cold retry policy, with independently checked gradients."""
    attempts=[]
    for budget in (100,10000):
        model=LogisticRegression(penalty=None,fit_intercept=True,solver='lbfgs',tol=1e-4,max_iter=budget)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(X,y)
        converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        diagnostics=gradient_diagnostics(X,y,model.coef_[0],float(model.intercept_[0]),float('inf'))
        attempts.append(dict(max_iter=budget,n_iter=int(model.n_iter_[0]),library_converged=converged,
                             warnings=[str(w.message) for w in caught],**diagnostics))
        if converged:break
    valid=converged and diagnostics['finite'] and diagnostics['gradient_infinity_norm']<=1e-4
    return model,dict(attempts=attempts,valid_for_scoring=valid,
                      policy='unregularized L-BFGS tol=1e-4; cold 10000 retry only on initial 100-iteration convergence warning')


def fit(method,X,y,rows,C,layer):
    X=np.asarray(X,dtype=np.float64);y=np.asarray(y,dtype=int)
    if method=='l2_logistic':
        head,details=fit_reference(X,y,C,layer)
        return dict(coef=head.coef_[0],intercept=head.intercept_[0]),details
    if method=='r0':
        pre=objective.fit_p_preprocessing(X)
        block=objective.prepare_block(X,y,pre)
        head=objective.fit_readout('r0',block,pre)
        active=~pre.constant_mask
        coef=np.zeros(X.shape[1]);coef[active]=head.weights[active]/pre.denominator[active]
        intercept=head.intercept-np.dot(pre.mean,coef)
        params=dict(coef=coef,intercept=np.array(intercept),standardized_coef=head.weights,
                    standardized_intercept=np.array(head.intercept),mean=pre.mean,population_std=pre.population_std,
                    denominator=pre.denominator,constant_mask=pre.constant_mask)
        attempts=[]
        for a in head.attempts:
            record=asdict(a)
            record.pop('initial_parameters');record.pop('final_parameters');attempts.append(record)
        if head.converged:old.check(np.allclose(head.decision_function(X),X@coef+intercept,rtol=1e-9,atol=1e-8),'R0 fused inference differs')
        return params,dict(valid_for_scoring=head.converged,status=head.status,attempts=attempts,
            preprocessing=pre.policy,preprocessing_fit_rows=pre.p_rows,settings=asdict(head.settings),
            objective='mean BCE on fitting cohort + 0.001*sum(standardized_w**2); free intercept',
            full_cohort_policy='full-cohort R0 is A-exposed diagnostic; preprocessing fitted on this cohort only')
    if method=='difference_of_means':head=old.difference_of_means(X,y)
    elif method=='mass_mean_covariance':head=old.covariance_mass_mean(X,y,atol=1e-3)
    elif method in ('burger_t_g','ttpd'):
        p=np.array([1 if r['form']=='affirmative' else -1 for r in rows])
        datasets=np.array([r['dataset'] for r in rows])
        head=old.learn_truth_directions(X,y,p,datasets)
        if method=='ttpd':
            polarity,polarity_record=unregularized(X,(p==1).astype(int))
            features=np.column_stack((X@head.parameters['t_g'],X@polarity.coef_[0]))
            truth,truth_record=unregularized(features,y)
            params=dict(head.parameters,coef=truth.coef_[0,0]*head.parameters['t_g']+truth.coef_[0,1]*polarity.coef_[0],
                        intercept=truth.intercept_[0],polarity_coef=polarity.coef_[0],polarity_intercept=polarity.intercept_,
                        polarity_classes=polarity.classes_,head_coef=truth.coef_[0],head_intercept=truth.intercept_[0],classes=truth.classes_)
            details=dict(valid_for_scoring=polarity_record['valid_for_scoring'] and truth_record['valid_for_scoring'],
                         polarity_optimizer=polarity_record,head_optimizer=truth_record,truth_fit=head.details,
                         inference='raw X@t_G and X@polarity coefficient; no form inputs, intercept projection or eval centering')
            return params,details
    else:raise ValueError('unknown pilot method')
    return head.parameters,dict(head.details,valid_for_scoring=True,convergence='closed_form_not_iterative',
                                finite=all(np.isfinite(v).all() for v in head.parameters.values() if np.asarray(v).dtype.kind not in 'US'))


def score(method,params,X):
    """No inference labels, operators, form indicators or evaluation statistics."""
    X=np.asarray(X,dtype=np.float64)
    if method=='ttpd':
        result=np.column_stack((X@params['t_g'],X@params['polarity_coef']))@params['head_coef']+params['head_intercept'].item()
    elif method=='r0':
        z=np.zeros_like(X);active=~params['constant_mask']
        z[:,active]=(X[:,active]-params['mean'][active])/params['denominator'][active]
        result=z@params['standardized_coef']+params['standardized_intercept'].item()
    else:result=X@params['coef']+params['intercept'].item()
    old.check(np.isfinite(result).all(),'nonfinite pilot score')
    old.check(np.allclose(result,X@params['coef']+params['intercept'].item(),rtol=1e-9,atol=1e-8),'explicit/fused readout mismatch')
    return result
