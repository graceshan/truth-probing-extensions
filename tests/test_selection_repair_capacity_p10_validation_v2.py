"""Independent validation and complete original-trigger accounting checks."""
import json
from pathlib import Path
import numpy as np
import pytest
from src import selection_repair_capacity_p10_v2 as p
from src.selection_repair_capacity_p10_validation_v2 import close,verify_convergence
ROOT=Path(__file__).resolve().parents[1]


def test_invalid_values_rejected():
    close([.5,.6],[.5,.6])
    for values in ([.5,.7],[.5,float('nan')],[.5,float('inf')]):
        with pytest.raises(ValueError,match='independent'):close(values,[.5,.6])


@pytest.mark.parametrize('method',['l2_logistic','r0','ttpd'])
def test_success_flag_does_not_override_bad_gradient(method):
    r=dict(configuration={'method':method},valid_for_scoring=True,status='converged',finite_parameters=True,
           final_converged=True,gradient_infinity_norm=.1,attempts=[dict(status='converged',gradient_infinity_norm=.1)])
    for key in ('polarity_optimizer','head_optimizer'):
        r[key]=dict(valid_for_scoring=True,attempts=[dict(library_converged=True,finite=True,gradient_infinity_norm=.1)])
    with pytest.raises(ValueError,match='gradient'):verify_convergence(r)


def test_all_14_original_triggers_are_retained_even_if_cleared():
    old=json.loads((ROOT/p.PACKAGE/'metrics.json').read_text());records=[]
    for r in old:
        for baseline in ('full','P15'):
            records.append(dict(**{k:r[k] for k in ('model','layer','method','metric','scope','topic')},
                comparison='P10_minus_'+baseline,delta=0. if baseline=='full' else -r['withholding_delta'],
                paired_delta_ci=dict(ci_low=-.01,ci_high=.01),triggered=False))
    result=p.trigger_followup(old,records)
    assert len(result['original_14'])==14
    assert all(r['trajectory']=='improved' and r['threshold_status']=='cleared' for r in result['original_14'])
    assert not result['new_P10_vs_full_triggers'] and not result['P10_vs_P15_triggers']
