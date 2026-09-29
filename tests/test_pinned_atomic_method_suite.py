"""Synthetic atomic-only cache/probe fixtures. No scientific datasets are read."""
import ast
import builtins
from contextlib import contextmanager
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import atomic_method_suite as shared
from src import atomic_methods_matched as matched
from src import atomic_probe_methods as formulas
from src import clean_atomic_probes as lr
from src import clean_transfer_contracts as c
from src import pinned_atomic_method_suite as suite
from src import repaired_atomic_cache as cache
from src.pinned_compound_scoring import PROBE_CONFIG


@pytest.fixture
def atomic(tmp_path, monkeypatch):
    """Keep real row counts/layers; only width and scientific identities are synthetic."""
    monkeypatch.setattr(cache, "WIDTH", 6)
    monkeypatch.setattr(c, "WIDTH", 6)
    directory = tmp_path / c.ATOMIC
    records = []
    for topic, nt, nv in zip(cache.TOPICS, [600, 300, 300, 272, 100], [200, 100, 100, 80, 40]):
        for form in ("affirmative", "negated"):
            for split, n, offset in (("train", nt, 0), ("validation", nv, nt)):
                for i in range(n):
                    records.append(dict(zip(cache.FIELDS, (
                        ("neg_" if form == "negated" else "") + topic, offset + i,
                        f"Synthetic {topic} {form} {split} {i}.", f"{topic}_{split}_{i}",
                        topic, form, split, i % 2 if form == "affirmative" else 1 - i % 2))))
    parts = {s: [r for r in records if r["split"] == s] for s in suite.COUNTS}
    monkeypatch.setattr(cache, "INPUT_DIGEST", cache.digest(cache.canonical(records)))
    rng = np.random.default_rng(71)
    values = {}
    for split, rows in parts.items():
        (directory / split).mkdir(parents=True)
        pd.DataFrame(rows).to_csv(directory / split / "metadata.csv", index=False)
        X = rng.normal(size=(len(rows), 28, 6))
        X[:, :, 0] += np.asarray([r["label"] * 2 - 1 for r in rows])[:, None] * .9
        X[:, :, 1] += np.asarray([1 if r["form"] == "affirmative" else -1 for r in rows])[:, None] * .7
        values[split] = X.astype(np.float16)
        np.save(directory / split / "activations.npy", values[split])
    manifest = dict(schema_version=1, implementation="qwen25-pinned-atomic-repair-v1",
        contract=cache.contract(), activation_shapes={s: [n, 28, 6] for s, n in suite.COUNTS.items()},
        resolved_model=dict(model_resolved_revision=c.REVISION, tokenizer_resolved_revision=c.REVISION),
        runtime=dict(torch="2.11.0+cu128", transformers="5.12.1"),
        input=dict(allowed_rows_sha256=cache.INPUT_DIGEST, counts=suite.COUNTS, columns=list(cache.FIELDS),
                   partition_rows_sha256={s: cache.digest(cache.canonical(r)) for s, r in parts.items()},
                   path="atomic_test/never_follow.csv"),
        smoke=dict(passed=True, batch_size=1, padding=False, policy="unpadded-single-repeat-and-independent-forward-exact-v1"),
        smoke_report=dict(path="compound_data/never_follow.json"))
    (directory / "extraction_manifest.json").write_bytes(cache.canonical(manifest))
    completion = dict(complete=True, counts=suite.COUNTS,
        manifest_sha256=c.file_hash(directory / "extraction_manifest.json"),
        outputs={name: c.record(directory / name) for name in cache.OUTPUT_FILES})
    (directory / "completion.json").write_text(json.dumps(completion))
    validated = cache.RepairedAtomicCache(tmp_path)
    coef = np.array([[1., 0., 0., 0., 0., 0.]])
    intercept = np.array([.12])
    val = pd.DataFrame(parts["validation"])
    metrics = lr.diagnostics(val.label.to_numpy(), values["validation"][:, 17, :].astype(np.float64) @ coef[0] + intercept[0],
                             val.topic.to_numpy(), val.form.to_numpy())
    monkeypatch.setattr(c, "EXPECTED_SELECTION", {**c.EXPECTED_SELECTION, "validation_auroc": metrics["overall_auroc"]})
    monkeypatch.setattr(c, "FINGERPRINT", c.digest(c.canonical(suite.representation(manifest["contract"]))))
    probe = tmp_path / c.PROBE
    probe.mkdir(parents=True)
    np.savez_compressed(probe / "selected_probe.npz", coef=coef, intercept=intercept, classes=np.array([0, 1]),
                        layer=np.array(17), C=np.array(1.), n_iter=np.array([12]), max_iter=np.array(2000))
    grid = []
    for layer in range(28):
        for C in c.C_VALUES:
            row = dict(layer=layer, C=C, final_converged=True, final_convergence_status="converged",
                       convergence_warning=False, final_max_iter=2000, final_n_iter=12,
                       **{"validation_" + k: v for k, v in metrics.items()})
            if (layer, C) != (17, 1.):
                row["validation_overall_auroc"] = .5
            grid.append(row)
    pd.DataFrame(grid).to_csv(probe / "validation_metrics.csv", index=False, float_format="%.17g")
    config = dict(schema_version=1, loader="pinned-repaired-cache-v1", cache_dir=c.ATOMIC,
        C_values=c.C_VALUES, models={"qwen25_7b": dict(model=c.MODEL, n_layers=28, hidden_size=6,
        dtype="float16", results_dir=c.PROBE)})
    structure = dict(validated.structural, adapter_code_sha256={"src/pinned_atomic_probes.py": "a" * 64})
    selection = dict(model=c.MODEL, selected_layer=17, selected_C=1., selection_rule=c.SELECTION_RULE,
        C_values=c.C_VALUES, probe_configuration=PROBE_CONFIG, preprocessing="none",
        label_convention="0=false, 1=true; score >= 0 predicts true",
        layer_convention="zero-based transformer-block output; embedding excluded", structural=structure,
        provenance=dict(config=config, config_sha256=c.digest(json.dumps(config, ensure_ascii=False, separators=(",", ":")).encode()),
                        model_key="qwen25_7b", code_sha256={}, git_revision=None),
        selected_probe_sha256=c.file_hash(probe / "selected_probe.npz"),
        validation_metrics_sha256=c.file_hash(probe / "validation_metrics.csv"),
        test_evaluated=False, test_decision_scores_computed=0,
        activation_rows_read=dict(train=3144 * 28, validation=1040 * 28, test=0),
        all_final_fits_converged=True, selected_probe_max_iter=2000,
        selected_validation_metrics=next(r for r in grid if (r["layer"], r["C"]) == (17, 1.)))
    (probe / "selection.json").write_text(json.dumps(selection))
    return tmp_path, values, selection


@contextmanager
def atomic_io_guard(monkeypatch, root):
    """Fail on any data read outside the six atomic files, three LR files or new outputs."""
    allowed = {root / c.ATOMIC / name for name in ("completion.json", "extraction_manifest.json", *cache.OUTPUT_FILES)}
    allowed |= {root / c.PROBE / name for name in c.PROBE_FILES}
    opened = []
    with monkeypatch.context() as patch:
        def check(path):
            if isinstance(path, int):
                return
            path = Path(os.fsdecode(path)).absolute()
            if path.is_relative_to(root):
                assert path in allowed or any(path.is_relative_to(root / v) for v in suite.OUTPUTS.values()), path
                opened.append(path)
            elif path.is_relative_to(suite.ROOT):
                assert path.suffix in {".py", ".pyc"} or str(path.relative_to(suite.ROOT)) in suite.SOURCE_FILES, path
        for module in (builtins, io, os):
            original = module.open
            def guarded(path, *args, _original=original, **kwargs):
                check(path)
                return _original(path, *args, **kwargs)
            patch.setattr(module, "open", guarded)
        def forbidden(*args, **kwargs):
            raise AssertionError("historical data loader called")
        patch.setattr(shared, "MethodData", forbidden)
        patch.setattr(matched, "MethodData", forbidden)
        yield opened


def test_check_only_zero_fits_zero_matrices_exact_identities(atomic, monkeypatch):
    root, _, _ = atomic
    original = np.load
    def guarded(path, *args, **kwargs):
        assert Path(path).name == "selected_probe.npz"
        assert kwargs["allow_pickle"] is False
        return original(path, *args, **kwargs)
    monkeypatch.setattr(np, "load", guarded)
    monkeypatch.setattr(suite, "fit_layer", lambda *a, **k: pytest.fail("fitting in preflight"))
    monkeypatch.setattr(suite, "fit_matched_methods", lambda *a, **k: pytest.fail("fitting in preflight"))
    with atomic_io_guard(monkeypatch, root) as opened:
        result = suite.run(root, suite="both", check_only=True)
    assert result["fits_performed"] == result["scores_computed"] == result["activation_matrices_materialized"] == 0
    assert result["activation_rows_read"] == {"train": 0, "validation": 0}
    assert result["provenance"]["partition_counts"] == {"train": 3144, "validation": 1040}
    assert set(result["provenance"]["burger_dataset_counts"].values()) == {100}
    assert len(set(opened)) == 9
    assert not (root / suite.OUTPUTS["faithful"]).exists()
    audit = result["provenance"]["frozen_lr"]["producer_source_audit"]["files"]
    assert audit["src/pinned_atomic_probes.py"]["recorded_sha256"] == "a" * 64


def test_loader_and_deterministic_paired_sample(atomic, monkeypatch):
    root, values, _ = atomic
    with atomic_io_guard(monkeypatch, root):
        data = suite.RepairedMethodData(root)
        assert set(data.partitions) == {"train", "validation"}
        for forbidden in ("test", "TEST", "compound", "../test", "/tmp/test"):
            with pytest.raises(PermissionError):
                data.matrix(forbidden, 17)
        for layer in (-1, 28, True):
            with pytest.raises(ValueError):
                data.matrix("train", layer)
        assert np.array_equal(data.burger_indices, formulas.balanced_burger_indices(data.partitions["train"], seed=0))
        assert np.array_equal(data.burger_indices, suite.RepairedMethodData(root).burger_indices)
        for topic in cache.TOPICS:
            a = data.burger_rows.query("topic == @topic and form == 'affirmative'")
            n = data.burger_rows.query("topic == @topic and form == 'negated'")
            assert len(a) == len(n) == 100
            assert list(a.entity_id) == list(n.entity_id)
            assert np.array_equal(a.label, 1 - n.label.to_numpy())
        for split in suite.COUNTS:
            X, digest = data.matrix(split, 17)
            assert X.dtype == np.float64
            np.testing.assert_array_equal(X, values[split][:, 17, :])
            assert digest == c.digest(values[split][:, 17, :].tobytes())
        assert data.activation_rows_read == suite.COUNTS


@pytest.mark.parametrize("mutation,match", [
    ("probe", "probe hash"), ("metrics", "validation metrics hash"),
    ("identity", "repaired-cache identity"), ("layer", "layer/C"),
    ("C", "layer/C"), ("test", "test access"), ("convergence", "unconverged"),
    ("fingerprint", "fingerprint"), ("cache", "output hash"),
])
def test_hard_failures(atomic, monkeypatch, mutation, match):
    root, _, selection = atomic
    probe = root / c.PROBE
    if mutation in ("probe", "metrics"):
        path = probe / ("selected_probe.npz" if mutation == "probe" else "validation_metrics.csv")
        with path.open("ab") as f:
            f.write(b"changed")
    elif mutation == "cache":
        path = root / c.ATOMIC / "train/activations.npy"
        with path.open("r+b") as f:
            f.seek(-2, 2)
            f.write(b"xx")
    elif mutation == "fingerprint":
        monkeypatch.setattr(c, "FINGERPRINT", "0" * 64)
    else:
        if mutation == "identity":
            selection["structural"]["approved_input_digest"] = "0" * 64
        if mutation == "layer":
            selection["selected_layer"] = 22
        if mutation == "C":
            selection["selected_C"] = .01
        if mutation == "test":
            selection["test_evaluated"] = True
        if mutation == "convergence":
            selection["all_final_fits_converged"] = False
        (probe / "selection.json").write_text(json.dumps(selection))
    with pytest.raises(ValueError, match=match):
        suite.run(root, check_only=True)


def test_count_and_path_protections(atomic):
    root, _, _ = atomic
    path = root / c.ATOMIC / "completion.json"
    completion = json.loads(path.read_text())
    completion["counts"]["train"] = 3143
    path.write_text(json.dumps(completion))
    with pytest.raises(ValueError, match="counts"):
        suite.run(root, check_only=True)
    path.unlink()
    path.symlink_to(root / "atomic_test/forbidden.json")
    with pytest.raises(ValueError, match="symlink"):
        suite.run(root, check_only=True)


def test_secondary_ties_and_validation_only_selection():
    rows = [dict(method=m, layer=l, validation_overall_auroc=.8 if l in (2, 3) else .7,
                 train_auc=1. if l == 17 else .2) for m in formulas.METHODS for l in (2, 3, 17)]
    chosen = shared.choose_layers(rows, 17)
    assert all(r["layer"] == 17 for r in chosen["primary_fixed_model_layer"].values())
    assert all(r["layer"] == 2 for r in chosen["secondary_method_selected_layer"].values())


def test_faithful_and_matched_end_to_end(atomic, monkeypatch):
    root, values, _ = atomic
    immutable = {str(p.relative_to(root)): p.read_bytes() for base in (root / c.ATOMIC, root / c.PROBE) for p in base.rglob("*") if p.is_file()}
    sentinels = [root / "results/clean_protocol" / tree / "qwen25_7b/sentinel" for tree in ("atomic_method_suite_v1", "atomic_method_matched_v1")]
    for path in sentinels:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"historical; must stay unchanged")
    formulas_before = (suite.ROOT / "src/atomic_probe_methods.py").read_bytes()
    solver = matched.fit_converged_probe
    fit_calls = []
    def fit_spy(X, y, C, *, layer):
        fit_calls.append((len(X), C, layer))
        return solver(X, y, C, layer=layer)
    monkeypatch.setattr(matched, "fit_converged_probe", fit_spy)
    finalized = []
    publish = c.publish_json
    def publish_spy(path, value):
        assert path.name == "verification.json" and not path.exists()
        assert all((path.parent / name).is_file() for name in ("summary.json", "report.md", "validation_metrics.csv"))
        assert all(c.record(path.parent / name) == record for name, record in value["outputs"].items())
        finalized.append(path)
        publish(path, value)
    monkeypatch.setattr(c, "publish_json", publish_spy)
    with atomic_io_guard(monkeypatch, root):
        faithful = suite.run(root, suite="faithful")["faithful"]
        assert fit_calls == []  # No canonical LR refit anywhere in 28-layer sweep.
        control = suite.run(root, suite="matched")["matched"]
        assert fit_calls == [(1000, 1., 17)]
    assert len(finalized) == 2
    assert all((root / name).read_bytes() == payload for name, payload in immutable.items())
    assert all(path.read_bytes() == b"historical; must stay unchanged" for path in sentinels)
    assert (suite.ROOT / "src/atomic_probe_methods.py").read_bytes() == formulas_before
    assert not faithful["lr_refitted"] and control["lr_refitted"]
    assert "secondary_method_selected_layer" not in control["comparisons"]
    canonical = np.load(root / c.PROBE / "selected_probe.npz", allow_pickle=False)
    saved = []
    for mode, count in (("faithful", 113), ("matched", 5)):
        output = root / suite.OUTPUTS[mode]
        table = pd.read_csv(output / "validation_metrics.csv")
        assert len(table) == count
        assert set(table.validation_rows) == {1040}
        assert set(table.method) == set(formulas.METHODS)
        expected_full = table.method.isin(["l2_logistic", "difference_of_means", "mass_mean_covariance"]) if mode == "faithful" else np.zeros(count, dtype=bool)
        np.testing.assert_array_equal(table.fit_rows, np.where(expected_full, 3144, 1000))
        saved.append((output / "burger_training_rows.csv").read_bytes())
        assert len(pd.read_csv(output / "allowed_atomic_rows.csv")) == 4184
        for path in sorted((output / "layers").glob("*.npz")):
            layer = int(path.stem.split("_")[1])
            with np.load(path, allow_pickle=False) as archive:
                params = {k: archive[k] for k in archive.files}
                assert all(v.dtype.kind != "O" for v in params.values())
                t = {k.split("__", 1)[1]: v for k, v in params.items() if k.startswith("ttpd__")}
                assert {"t_g", "t_p", "dataset_means", "polarity_coef", "polarity_intercept", "polarity_classes", "head_coef", "head_intercept", "classes", "coef", "intercept"} <= set(t)
                V = values["validation"][:, layer, :].astype(np.float64)
                explicit = formulas.FittedMethod("ttpd", t, {}).decision_function(V)
                np.testing.assert_allclose(explicit, V @ t["coef"] + t["intercept"], rtol=1e-9, atol=1e-8)
                if mode == "faithful" and layer == 17:
                    np.testing.assert_array_equal(params["l2_logistic__coef"], canonical["coef"][0])
                    np.testing.assert_array_equal(params["l2_logistic__intercept"], canonical["intercept"][0])
        with pytest.raises(ValueError, match="refusing existing"):
            suite.run(root, suite=mode)
    assert saved[0] == saved[1]
    # The same exact subset gives identical t_G/TTPD parameters in both comparisons.
    with np.load(root / suite.OUTPUTS["faithful"] / "layers/layer_17.npz", allow_pickle=False) as a, np.load(root / suite.OUTPUTS["matched"] / "layers/layer_17.npz", allow_pickle=False) as b:
        for key in a.files:
            if key.startswith(("burger_t_g__", "ttpd__")):
                np.testing.assert_array_equal(a[key], b[key])
    canonical.close()


def test_no_validation_based_sign_flip(atomic):
    root, _, _ = atomic
    data, probe, baseline, _ = suite.prepare(root)
    # Synthetic validation labels are deliberately reversed AFTER strict loading.
    # This isolated fitting-helper test checks no sign is chosen using validation.
    data.partitions["validation"]["label"] = 1 - data.partitions["validation"].label
    record, _ = suite.fit_layer(data, 17, baseline, 17, data.burger_indices, suite.POLICY)
    assert all(r["validation_overall_auroc"] < .5 for r in record["metrics"])


def test_representation_descriptor_matches_producer_without_loading_hf():
    # Extract the pure metadata function through AST, avoiding heavyweight imports.
    tree = ast.parse((suite.ROOT / "src/clean_extraction.py").read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "layer_convention")
    namespace = {"require": c.require}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "producer_metadata", "exec"), namespace)
    descriptor = suite.representation(cache.contract())
    assert descriptor["layer_convention"] == namespace["layer_convention"](28)
    assert c.digest(c.canonical(descriptor)) == "59df237b800f49892889e72ee6f852314b8430eb14f9c64dc0d7fff83e832821"
