"""Independent validation boundaries and fail-closed convergence tests."""
import numpy as np
import pytest
from src.selection_repair_capacity_validation_v1 import check
from src import selection_repair_capacity_methods_v1 as m


def test_independent_validator_rejects_nonfinite_and_wrong_values():
    check(.7,.7)
    for value in [float('nan'),float('inf'),.701]:
        with pytest.raises(ValueError,match='independent'):check(value,.7)


def test_ttpd_library_success_cannot_hide_failed_gradient(monkeypatch):
    class WrongHead:
        classes_=np.array([0,1]);coef_=np.array([[100.,100.]])
        intercept_=np.array([0.]);n_iter_=np.array([1])
        def __init__(self,**kwargs):pass
        def fit(self,X,y):return self
    monkeypatch.setattr(m,'LogisticRegression',WrongHead)
    X=np.array([[1.,0.],[0.,1.],[-1.,0.],[0.,-1.]])
    y=np.array([0,0,1,1])
    _,result=m.unregularized(X,y)
    assert result['attempts'][0]['library_converged'] is True
    assert result['attempts'][0]['gradient_infinity_norm']>1e-4
    assert result['valid_for_scoring'] is False
