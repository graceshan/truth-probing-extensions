"""Synthetic historical-cache tests; no probes or real test data are loaded."""
import copy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.atomic_activation_compatibility import (
    PRIMARY_TOPICS, absolute_differences, compatibility_decision, historical_compatibility_smoke,
    load_compatibility_pin, require_both_gates, save_compatibility_pin, select_historical_sample,
)
from src.clean_extraction import MODELS, extract_clean, resume_identity


def padding_report(maximum=0., mean=0., passed=True):
    return {"passed": passed, "comparisons": {side: {"saved_float16": {"per_layer": [
        {"max_absolute_difference": maximum, "mean_absolute_difference": mean} for _ in range(2)]}}
        for side in ("left", "right")}}


def gate_manifest():
    return {"model": {"identifier": "synthetic", "resolved_commit_sha": "a" * 40},
            "tokenizer": {"resolved_commit_sha": "b" * 40},
            "code": {"implementation_version": "test", "source_sha256": {"test": "hash"}},
            "historical_sample_sha256": "sample",
            "smoke_test": {"passed": True}, "historical_atomic_compatibility": {
                "passed": True, "sample_sha256": "sample", "current_model_identifier": "synthetic",
                "current_model_commit_sha": "a" * 40, "current_tokenizer_commit_sha": "b" * 40}}


@pytest.mark.parametrize("padding,historical", [(False, True), (True, False), (False, False)])
def test_both_gates_required_before_any_array(padding, historical, tmp_path):
    manifest = gate_manifest()
    manifest["smoke_test"]["passed"] = padding
    manifest["historical_atomic_compatibility"]["passed"] = historical
    with pytest.raises(ValueError, match="smoke required"):
        extract_clean(None, None, None, manifest, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_historical_gate_fails():
    manifest = gate_manifest()
    del manifest["historical_atomic_compatibility"]
    with pytest.raises(ValueError, match="historical atomic"):
        require_both_gates(manifest)


@pytest.mark.parametrize("field", ["current_model_commit_sha", "current_tokenizer_commit_sha",
                                 "current_model_identifier", "sample_sha256"])
def test_compatibility_evidence_must_match_current_snapshot(field):
    manifest = gate_manifest()
    manifest["historical_atomic_compatibility"][field] = "changed"
    with pytest.raises(ValueError, match="mismatch"):
        require_both_gates(manifest)


def test_byte_identical_passes_and_precast_quantization_is_reported():
    fresh = np.full((10, 2, 3), 1.0001, dtype=np.float32)
    old = fresh.astype(np.float16)
    report = compatibility_decision(old, fresh, padding_report())
    assert report["passed"] and report["float16_byte_identical"]
    assert report["fresh_before_float16_conversion"]["max_absolute_difference"] > 0
    assert report["after_float16_conversion"]["max_absolute_difference"] == 0


def test_nonidentical_must_fit_observed_envelope_every_layer():
    old = np.zeros((10, 2, 3), dtype=np.float16)
    fresh = old.astype(np.float32)
    fresh[0, 1, 0] = 0.125
    assert compatibility_decision(old, fresh, padding_report(.125, .125))["passed"]
    failed = compatibility_decision(old, fresh, padding_report(.0625, .125))
    assert not failed["passed"] and not failed["float16_byte_identical"]
    assert failed["after_float16_conversion"]["per_layer"][0]["max_absolute_difference"] == 0
    assert failed["after_float16_conversion"]["per_layer"][1]["max_absolute_difference"] == .125
    assert not compatibility_decision(old, fresh, padding_report(.125, 0.))["passed"]


def test_identical_history_cannot_bypass_failed_padding():
    values = np.zeros((10, 2, 3), dtype=np.float16)
    assert not compatibility_decision(values, values, padding_report(passed=False))["passed"]


def test_nonfinite_comparison_fails():
    with pytest.raises(ValueError, match="nonfinite"):
        absolute_differences(np.zeros((1, 2, 3)), np.full((1, 2, 3), np.nan))


def test_pin_roundtrip_and_no_overwrite(tmp_path):
    manifest = gate_manifest()
    report = tmp_path / "report.json"
    report.write_text(json.dumps(manifest))
    pin = tmp_path / "pin.json"
    save_compatibility_pin(pin, report, manifest)
    old_bytes = pin.read_bytes()
    receipt, loaded = load_compatibility_pin(pin, "synthetic")
    assert loaded == manifest and receipt["model_commit_sha"] == "a" * 40
    assert receipt["tokenizer_commit_sha"] == "b" * 40
    save_compatibility_pin(pin, report, manifest)
    assert old_bytes == pin.read_bytes()
    changed = copy.deepcopy(manifest)
    changed["model"]["resolved_commit_sha"] = "c" * 40
    changed["historical_atomic_compatibility"]["current_model_commit_sha"] = "c" * 40
    other = tmp_path / "other.json"
    other.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="no overwrite"):
        save_compatibility_pin(pin, other, changed)
    assert pin.read_bytes() == old_bytes


def test_failed_gate_never_creates_pin(tmp_path):
    manifest = gate_manifest()
    manifest["historical_atomic_compatibility"]["passed"] = False
    with pytest.raises(ValueError, match="historical atomic"):
        save_compatibility_pin(tmp_path / "pin.json", tmp_path / "report.json", manifest)
    assert not (tmp_path / "pin.json").exists()


def test_pin_evidence_tamper_fails(tmp_path):
    manifest = gate_manifest()
    report = tmp_path / "report.json"
    report.write_text(json.dumps(manifest))
    pin = tmp_path / "pin.json"
    save_compatibility_pin(pin, report, manifest)
    report.write_text(report.read_text() + "\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_compatibility_pin(pin, "synthetic")


def test_mutable_revision_cannot_pass_gate():
    manifest = gate_manifest()
    manifest["model"]["resolved_commit_sha"] = "main"
    manifest["historical_atomic_compatibility"]["current_model_commit_sha"] = "main"
    with pytest.raises(ValueError, match="immutable model revision"):
        require_both_gates(manifest)


@pytest.mark.parametrize("key", ["model_commit_sha", "tokenizer_commit_sha"])
def test_receipt_revision_tamper_fails(tmp_path, key):
    manifest = gate_manifest()
    report = tmp_path / "report.json"
    report.write_text(json.dumps(manifest))
    pin = tmp_path / "pin.json"
    save_compatibility_pin(pin, report, manifest)
    receipt = json.loads(pin.read_text())
    receipt[key] = "c" * 40
    pin.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="revision mismatch"):
        load_compatibility_pin(pin, "synthetic")


def test_observed_metrics_not_resume_identity_but_sample_is():
    manifest = gate_manifest()
    other = copy.deepcopy(manifest)
    other["historical_atomic_compatibility"]["max_absolute_difference"] = 0.01
    assert resume_identity(manifest) == resume_identity(other)
    other["historical_sample_sha256"] = "changed-cache"
    assert resume_identity(manifest) != resume_identity(other)


@pytest.fixture(params=["qwen2_5_7b", "qwen3_8b"])
def historical_fixture(tmp_path, monkeypatch, request):
    key = request.param
    spec = dict(MODELS[key], layers=2, hidden_size=3)
    monkeypatch.setitem(MODELS, key, spec)
    cfg_dir = tmp_path / "config/clean_protocol"
    cfg_dir.mkdir(parents=True)
    source_dir, acts_dir = tmp_path / "source", tmp_path / "historical"
    source_dir.mkdir()
    acts_dir.mkdir()
    model_config = {"model": spec["identifier"], "n_layers": 2, "hidden_size": 3, "dtype": "float16",
                    "activation_pattern": "historical/{dataset}.npy", "metadata_pattern": "historical/{dataset}.csv"}
    config = {"entity_manifest": "entities.csv", "source_dir": "source",
              "models": {"qwen25_7b" if key == "qwen2_5_7b" else key: model_config}}
    (cfg_dir / "atomic_probes.json").write_text(json.dumps(config))
    templates = {"cities": "The city of {e} is {neg}in X.", "sp_en_trans": "Spanish word '{e}' {neg}means 'x'.",
                 "inventors": "{e} {verb} in X.", "element_symb": "{e} {verb} the symbol X.",
                 "animal_class": "The {e} is {neg}an animal."}
    entities = []
    for topic in PRIMARY_TOPICS:
        for i, split in enumerate(["train", "train", "validation", "test"]):
            entities.append(dict(topic=topic, entity=f"Entity{i}", entity_id=f"{topic}_{i}", split=split))
        for negative in (False, True):
            dataset = ("neg_" if negative else "") + topic
            verb = ("did not live" if negative else "lived") if topic == "inventors" else ("does not have" if negative else "has")
            statements = [templates[topic].format(e=f"Entity{i}", neg="not " if negative else "", verb=verb) for i in range(4)]
            frame = pd.DataFrame({"statement": statements, "label": ["DO_NOT_LOAD"] * 4})
            frame.to_csv(source_dir / f"{dataset}.csv", index=False)
            frame.to_csv(acts_dir / f"{dataset}.csv", index=False)
            array = np.ones((4, 2, 3), dtype=np.float16)
            array[3] = np.nan  # forbidden row; it must never be inspected
            np.save(acts_dir / f"{dataset}.npy", array)
    pd.DataFrame(entities).to_csv(tmp_path / "entities.csv", index=False)
    return tmp_path, key


def test_deterministic_primary_only_label_free_selection(historical_fixture):
    root, key = historical_fixture
    real_read, real_load = pd.read_csv, np.load
    accessed = []
    def read_csv(*args, **kwargs):
        assert "usecols" in kwargs and "label" not in kwargs["usecols"]
        return real_read(*args, **kwargs)
    def load(path, **kwargs):
        assert kwargs == {"mmap_mode": "r", "allow_pickle": False}
        arr = real_load(path, **kwargs)
        class RestrictedArray:
            shape, dtype = arr.shape, arr.dtype
            def __getitem__(self, index):
                assert isinstance(index, int) and index in (0, 1, 2)
                accessed.append(index)
                return arr[index]
        return RestrictedArray()
    with patch("pandas.read_csv", side_effect=read_csv), patch("numpy.load", side_effect=load):
        provenance, values = select_historical_sample(key, root)
        other, same = select_historical_sample(key, root)
    assert provenance == other
    np.testing.assert_array_equal(values, same)
    assert values.shape == (10, 2, 3) and len(accessed) == 20
    assert {r["topic"] for r in provenance["rows"]} == set(PRIMARY_TOPICS)
    assert {r["split"] for r in provenance["rows"]} == {"train", "validation"}
    assert {r["form"] for r in provenance["rows"]} == {"affirmative", "negated"}
    assert not provenance["labels_loaded"] and provenance["test_activation_rows_read"] == 0


def test_historical_smoke_uses_exact_sidecar_single_rows(historical_fixture):
    import torch
    from types import SimpleNamespace
    root, key = historical_fixture
    calls = []
    def extract(model, tokenizer, statements, **kwargs):
        assert kwargs == {"padding": False} and len(statements) == 1
        calls.extend(statements)
        return torch.ones((1, 2, 3), dtype=torch.float32)
    manifest = gate_manifest()
    manifest["numerics"] = {"batch_size": 4}
    with patch("src.atomic_activation_compatibility.extract_batch", side_effect=extract), \
         patch("src.atomic_activation_compatibility.smoke_compare", return_value=padding_report()):
        report = historical_compatibility_smoke(SimpleNamespace(dtype=torch.float32), None, key, manifest, root=root)
    assert report["passed"] and report["float16_byte_identical"]
    assert calls == [r["statement"] for r in report["sample"]["rows"]]
    assert report["current_model_commit_sha"] == "a" * 40
    assert report["current_tokenizer_commit_sha"] == "b" * 40
