"""Synthetic tampering regressions for independent result checking."""
import pytest
from src.selection_repair_sensitivity_v3_validation import check_identities,check_values
from src.selection_repair_sensitivity_v3 import identity


def test_changed_and_missing_result_artifact_rejected(tmp_path):
    path=tmp_path/'metrics.json';path.write_text('original')
    pin={'metrics.json':identity(path.read_bytes())}
    check_identities(tmp_path,pin)
    path.write_text('modified')
    with pytest.raises(ValueError,match='artifact changed'):check_identities(tmp_path,pin)
    path.unlink()
    with pytest.raises(ValueError,match='artifact changed'):check_identities(tmp_path,pin)


def test_metric_mismatch_is_not_hidden_by_default_relative_tolerance():
    check_values([.999999],[.999999+1e-14])
    with pytest.raises(ValueError,match='metric/interval'):check_values([.999999],[.999999+1e-7])
    with pytest.raises(ValueError,match='metric/interval'):check_values([float('nan')],[float('nan')])
