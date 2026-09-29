"""Synthetic cache/benchmark and tiny random BF16 Qwen2; no real GPU/data work."""
import ast
import builtins
import copy
import csv
import io
import json
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from src import clean_atomic_extraction as atomic
from src import clean_extraction as legacy
from src import atomic_activation_compatibility as historical
from src import repaired_atomic_cache as validation
from src import pinned_compound_extraction as compound
from test_clean_atomic_extraction import tiny  # shared actual 28-layer tiny Qwen2 fixture


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_receipt(root, manifest):
    cache = root / validation.CACHE
    (cache / "extraction_manifest.json").write_bytes(atomic.canonical(manifest))
    receipt = {"complete": True, "counts": dict(validation.COUNTS),
               "manifest_sha256": atomic.file_hash(cache / "extraction_manifest.json"),
               "outputs": {name: {"sha256": atomic.file_hash(cache / name), "bytes": (cache / name).stat().st_size}
                           for name in validation.OUTPUT_FILES}}
    (cache / "completion.json").write_bytes(atomic.canonical(receipt))
    return receipt


@pytest.fixture
def fixture(tmp_path, monkeypatch, tiny):
    root = tmp_path.resolve()
    model, tokenizer = tiny
    model.config._commit_hash = atomic.REVISION
    monkeypatch.setattr(compound, "WIDTH", 16)
    monkeypatch.setattr(compound, "ROW_COUNT", 6)
    monkeypatch.setattr(validation, "WIDTH", 16)
    monkeypatch.setattr(validation, "COUNTS", {"train": 20, "validation": 20})
    records = []
    for i, topic in enumerate(atomic.TOPICS):
        for form in ("affirmative", "negated"):
            for split, offset in (("train", 0), ("validation", 2)):
                for label in (0, 1):
                    records.append(dict(zip(atomic.FIELDS, (
                        ("neg_" if form == "negated" else "") + topic, offset + label,
                        "A " + "small " * (i + 1) + "box.", f"entity_{topic}_{split}",
                        topic, form, split, label))))
    monkeypatch.setattr(validation, "INPUT_DIGEST", atomic.digest(atomic.canonical(records)))
    parts = {s: [r for r in records if r["split"] == s] for s in validation.COUNTS}
    values = {}
    for split, rows in parts.items():
        directory = root / validation.CACHE / split
        write_csv(directory / "metadata.csv", rows, atomic.FIELDS)
        for row in rows:
            if row["statement"] not in values:
                values[row["statement"]] = atomic.readout(model, tokenizer, row["statement"]).numpy().astype(np.float16)
        np.save(directory / "activations.npy", np.stack([values[r["statement"]] for r in rows]))
    info = {"model_resolved_revision": atomic.REVISION, "tokenizer_resolved_revision": atomic.REVISION,
            "config_file_sha256": "a" * 64, "tokenizer_config_file_sha256": "b" * 64,
            "tokenizer_backend_sha256": atomic.digest(tokenizer.backend_tokenizer.to_str().encode())}
    manifest = {"schema_version": 1, "implementation": "qwen25-pinned-atomic-repair-v1",
                "contract": atomic.contract(), "activation_shapes": {s: [20, 28, 16] for s in parts},
                "resolved_model": info, "runtime": {"torch": "2.11.0+cu128", "transformers": "5.12.1"},
                "input": {"allowed_rows_sha256": validation.INPUT_DIGEST, "counts": validation.COUNTS,
                          "columns": list(atomic.FIELDS), "partition_rows_sha256": {
                              s: atomic.digest(atomic.canonical(rows)) for s, rows in parts.items()}},
                "smoke": {"passed": True, "batch_size": 1, "padding": False,
                          "policy": "unpadded-single-repeat-and-independent-forward-exact-v1"}}
    save_receipt(root, manifest)
    benchmark_rows = [{"example_id": f"example_{i}", "statement": f"A small box and {i} other boxes.",
                       "pair_id": f"pair_{i}", "topic": "cities", "split": "validation", "operator": "AND",
                       "ordering": "AB", "canonical_truth_a": "True", "canonical_truth_b": "False",
                       "compound_label": "False", "protocol": "entity_disjoint", "evaluation_phase": "development",
                       "extra_preserved": "Unicode é, full metadata\nwith newline"} for i in range(6)]
    write_csv(root / compound.BENCHMARK, benchmark_rows, tuple(benchmark_rows[0]))
    monkeypatch.setattr(compound, "BENCHMARK_SHA256", atomic.file_hash(root / compound.BENCHMARK))
    for name in compound.SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic provenance\n")
    monkeypatch.setattr(atomic, "load_pinned_model", lambda: (model, tokenizer, copy.deepcopy(info)))
    monkeypatch.setattr(compound, "runtime_provenance", lambda: {"torch": "2.11.0", "gpu": "synthetic CPU"})
    return root, model, tokenizer, manifest, benchmark_rows


def guard(monkeypatch, root):
    permitted = {root / validation.CACHE / name for name in
                 ("completion.json", "extraction_manifest.json", *validation.OUTPUT_FILES)}
    permitted |= {root / compound.BENCHMARK, *(root / name for name in compound.SOURCES)}
    opened = []
    def check(path):
        if isinstance(path, int):
            return
        path = Path(os.fsdecode(path)).absolute()
        if path.is_relative_to(root):
            assert path in permitted or path.is_relative_to(root / compound.OUTPUT) or path.is_relative_to(root / compound.REPORTS), path
            opened.append(path)
        elif path.is_relative_to(compound.ROOT):
            assert path.suffix in {".py", ".pyc"}, f"real data access forbidden: {path}"
    for module in (builtins, io, os):
        original = module.open
        def wrapped(path, *args, _original=original, **kwargs):
            check(path)
            return _original(path, *args, **kwargs)
        monkeypatch.setattr(module, "open", wrapped)
    for module, names in ((historical, ("historical_compatibility_smoke", "require_both_gates", "load_compatibility_pin")),
                          (legacy, ("extract_clean", "smoke_compare", "build_manifest")),
                          (atomic, ("load_allowed_rows", "smoke"))):
        for name in names:
            monkeypatch.setattr(module, name, Mock(side_effect=AssertionError(f"forbidden call: {name}")))
    return opened


@pytest.mark.parametrize("mode", ["plan", "smoke", "extract"])
def test_isolated_lifecycle_no_historical_test_or_method_dependencies(fixture, monkeypatch, mode):
    root, model, tokenizer, _, _ = fixture
    opened = guard(monkeypatch, root)
    if mode == "plan":
        monkeypatch.setattr(atomic, "load_pinned_model", Mock(side_effect=AssertionError("model loaded in plan")))
    result = compound.run(mode, root=root)
    assert result["shape"] == [6, 28, 16]
    assert result["activation_arrays_created"] == (mode == "extract")
    assert result["model_loaded"] == (mode != "plan")
    assert not (root / "results/clean_protocol/atomic_probes_pinned_v1").exists()
    assert not (root / "data/tiu_datasets").exists()
    if mode != "extract":
        assert not (root / compound.OUTPUT).exists()
    else:
        output = root / compound.OUTPUT
        assert (output / "metadata.csv").read_bytes() == (root / compound.BENCHMARK).read_bytes()
        values = np.load(output / "activations.npy", allow_pickle=False)
        assert values.shape == (6, 28, 16) and values.dtype == np.float16
        benchmark = compound.load_benchmark(root)
        expected = np.stack([atomic.readout(model, tokenizer, s).numpy().astype(np.float16) for s in benchmark.frame.statement])
        np.testing.assert_array_equal(values, expected)
        progress = json.loads((output / "progress.json").read_text())
        assert progress["complete"] and progress["next_row"] == 6
    assert root / validation.CACHE / "completion.json" in opened


def test_full_production_row_count_shape_and_no_model_in_plan(fixture, monkeypatch):
    root, _, _, manifest, rows = fixture
    assert compound.LAYERS == 28
    monkeypatch.setattr(compound, "ROW_COUNT", 8384)
    monkeypatch.setattr(compound, "WIDTH", 3584)
    monkeypatch.setattr(validation, "WIDTH", 3584)
    for split in validation.COUNTS:
        np.save(root / validation.CACHE / split / "activations.npy", np.zeros((20, 28, 3584), dtype=np.float16))
    manifest["activation_shapes"] = {s: [20, 28, 3584] for s in validation.COUNTS}
    save_receipt(root, manifest)
    all_rows = [{**rows[0], "example_id": f"example_{i}"} for i in range(8384)]
    write_csv(root / compound.BENCHMARK, all_rows, tuple(rows[0]))
    monkeypatch.setattr(compound, "BENCHMARK_SHA256", atomic.file_hash(root / compound.BENCHMARK))
    monkeypatch.setattr(atomic, "load_pinned_model", Mock(side_effect=AssertionError("no model")))
    assert compound.run("plan", root=root)["shape"] == [8384, 28, 3584]
    write_csv(root / compound.BENCHMARK, all_rows[:-1], tuple(rows[0]))
    with pytest.raises(ValueError, match="8,384"):
        compound.load_benchmark(root)


@pytest.mark.parametrize("failure", ["completion", "file_hash", "manifest_hash", "manifest_convention", "fingerprint"])
def test_repaired_binding_corruption_fails_before_model(fixture, monkeypatch, failure):
    root, _, _, manifest, _ = fixture
    cache = root / validation.CACHE
    if failure == "completion":
        (cache / "completion.json").unlink()
    elif failure == "file_hash":
        with (cache / "train/activations.npy").open("ab") as handle:
            handle.write(b"bad")
    elif failure == "manifest_hash":
        (cache / "extraction_manifest.json").write_text("{}")
    elif failure == "manifest_convention":
        manifest["contract"]["padding"] = True
        save_receipt(root, manifest)
    else:
        manifest["representation_fingerprint"] = "0" * 64
        save_receipt(root, manifest)
    monkeypatch.setattr(atomic, "load_pinned_model", Mock(side_effect=AssertionError("no model")))
    with pytest.raises((ValueError, FileNotFoundError)):
        compound.run("extract", root=root)
    assert not (root / compound.OUTPUT).exists()


@pytest.mark.parametrize("field", ["model_resolved_revision", "tokenizer_resolved_revision", "tokenizer_backend_sha256"])
def test_fresh_revision_or_tokenizer_mismatch_blocks(fixture, monkeypatch, field):
    root, model, tokenizer, manifest, _ = fixture
    info = copy.deepcopy(manifest["resolved_model"])
    info[field] = "0" * 64
    monkeypatch.setattr(atomic, "load_pinned_model", lambda: (model, tokenizer, info))
    with pytest.raises(ValueError, match="mismatch"):
        compound.run("extract", root=root)
    assert not (root / compound.OUTPUT).exists()


def test_stable_fingerprint_excludes_runtime_and_probe_choices():
    settings = atomic.contract()
    fields = compound.representation_fields(settings)
    baseline = compound.fingerprint(fields)
    settings.update(timestamp="later", gpu="different", output_path="elsewhere", git_commit="changed",
                    torch="runtime-only", transformers="runtime-only", selected_layer=17, selected_C=1.0,
                    selection_json="must never be read")
    assert compound.fingerprint(compound.representation_fields(settings)) == baseline
    for name, value in (("model_revision", "b" * 40), ("batch_size", 2), ("compute_dtype", "float32")):
        changed = {**settings, name: value}
        assert compound.fingerprint(compound.representation_fields(changed)) != baseline
    assert not ({"selected_layer", "selected_C", "torch", "gpu", "timestamp"} & set(fields))


def test_orchestrator_and_cache_validator_have_no_probe_fitting_or_scoring_dependency():
    for module in (compound, validation):
        tree = ast.parse(Path(module.__file__).read_text())
        imports = [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        assert not any(any(word in name for word in ("probe", "method", "sklearn", "transfer")) for name in imports)
        calls = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        assert not {"fit", "fit_and_save", "predict", "decision_function", "score"} & set(calls)


@pytest.mark.parametrize("failure", ["atomic_test", "compound_test", "benchmark_hash"])
def test_disallowed_data_rejected(fixture, failure):
    root, _, _, manifest, rows = fixture
    if failure == "atomic_test":
        path = root / validation.CACHE / "train/metadata.csv"
        with path.open(newline="") as handle:
            records = list(csv.DictReader(handle))
        records[0]["split"] = "test"
        write_csv(path, records, atomic.FIELDS)
        save_receipt(root, manifest)
    else:
        rows[0]["split" if failure == "compound_test" else "statement"] = "test" if failure == "compound_test" else "altered"
        write_csv(root / compound.BENCHMARK, rows, tuple(rows[0]))
    with pytest.raises(ValueError):
        compound.run("plan", root=root)


def test_exact_replay_samples_and_detailed_differences(fixture):
    root, model, tokenizer, _, _ = fixture
    cache, _ = compound.bind_repaired_cache(root)
    samples = compound.atomic_sample(cache)
    assert len(samples) == 10
    for sample in samples:
        row = cache.rows[sample["split"]][sample["cache_row_index"]]
        assert row["statement"] == sample["statement"] and row["row_index"] == sample["row_index"]
        assert (sample["split"], sample["form"]) in {("train", "affirmative"), ("validation", "negated")}
    report = compound.replay_atomic(model, tokenizer, cache, samples)
    assert report["passed"] and report["saved_float16_byte_equal"]
    assert report["max_absolute_difference"] == report["mean_absolute_difference"] == 0
    assert len(report["per_layer"]) == 28
    assert report["sample_sha256"] == compound.fingerprint(samples)
    # A genuine one-ULP float16 discrepancy in a completed, rehashed repaired cache
    # is still forbidden; no numerical tolerance can make it pass.
    sample = samples[0]
    path = root / validation.CACHE / sample["split"] / "activations.npy"
    values = np.load(path, allow_pickle=False)
    i = sample["cache_row_index"]
    values[i, 0, 0] = np.nextafter(values[i, 0, 0], np.float16(np.inf))
    np.save(path, values)
    save_receipt(root, cache.manifest)
    with pytest.raises(ValueError, match="exact atomic replay failed"):
        compound.run("extract", root=root)
    assert not (root / compound.OUTPUT).exists()
    report_path = next((root / compound.REPORTS).glob("*.json"))
    failed = json.loads(report_path.read_text())["smoke_test"]["atomic_replay"]
    assert not failed["saved_float16_byte_equal"] and failed["max_absolute_difference"] > 0


def test_smoke_forwards_are_single_unpadded_and_independent(fixture):
    root, model, tokenizer, _, _ = fixture
    calls = []
    def intercept(_model, _args, kwargs):
        assert kwargs["input_ids"].shape[0] == 1 and (kwargs["attention_mask"] == 1).all()
        assert kwargs["use_cache"] is False and kwargs["output_hidden_states"] is True
        torch.testing.assert_close(kwargs["position_ids"], torch.arange(kwargs["input_ids"].shape[1]).unsqueeze(0))
        calls.append(kwargs["logits_to_keep"])
    handle = model.register_forward_pre_hook(intercept, with_kwargs=True)
    try:
        result = compound.run("smoke", root=root)
    finally:
        handle.remove()
    assert calls == [1] * 10 + [1, 1, 0] * 4
    manifest = json.loads(Path(result["diagnostics"]).read_text())
    assert manifest["smoke_test"]["compound_smoke"]["passed"]
    assert manifest["representation"]["layer_convention"]["saved_layer_to_hf_index"] == list(range(1, 29))
    assert manifest["representation"]["layer_convention"]["final_saved_state_post_final_norm"] is True


def test_independent_forward_mismatch_blocks_extraction(fixture, monkeypatch):
    root, _, _, _, _ = fixture
    original = atomic.reference_readout
    monkeypatch.setattr(atomic, "reference_readout", lambda *args: original(*args) + 1)
    with pytest.raises(ValueError, match="compound smoke"):
        compound.run("extract", root=root)
    assert not (root / compound.OUTPUT).exists()


def test_overwrite_and_old_output_names_refused(fixture):
    root = fixture[0]
    with pytest.raises(ValueError, match="versioned"):
        compound.run("plan", root=root, output=compound.OUTPUT_PARENT / "qwen2_5_7b")
    path = root / compound.OUTPUT
    path.mkdir(parents=True)
    with pytest.raises(ValueError, match="overwrite"):
        compound.run("extract", root=root)


def test_resume_replays_gates_and_only_writes_uncommitted_rows(fixture, monkeypatch):
    root, model, tokenizer, _, _ = fixture
    report = compound.run("smoke", root=root)
    manifest = json.loads(Path(report["diagnostics"]).read_text())
    benchmark = compound.load_benchmark(root)
    output = root / compound.OUTPUT
    with legacy.ExtractionWriter(output, benchmark, manifest) as writer:
        for i in range(2):
            value = compound.pinned_readout(model, tokenizer, benchmark.frame.statement.iloc[i]).numpy().astype(np.float16)[None]
            writer.append(i, benchmark.frame.iloc[i:i + 1], value)
    before = json.loads((output / "progress.json").read_text())["chunks"]
    spy = Mock(wraps=compound.pinned_readout)
    monkeypatch.setattr(compound, "pinned_readout", spy)
    compound.run("extract", root=root, resume=True)
    assert spy.call_count == 10 + 8 + 4  # replay + repeat smoke + only four remaining rows
    progress = json.loads((output / "progress.json").read_text())
    assert progress["complete"] and progress["chunks"][:2] == before


@pytest.mark.parametrize("change", ["code", "runtime", "fingerprint", "cache_identity"])
def test_resume_identity_and_representation_protection(fixture, change):
    root, model, tokenizer, _, _ = fixture
    report = compound.run("smoke", root=root)
    manifest = json.loads(Path(report["diagnostics"]).read_text())
    benchmark = compound.load_benchmark(root)
    output = root / compound.OUTPUT
    with legacy.ExtractionWriter(output, benchmark, manifest):
        pass
    changed = copy.deepcopy(manifest)
    if change == "code":
        changed["code"]["source_sha256"]["src/extract.py"] = "changed"
    elif change == "runtime":
        changed["numerics"]["runtime"]["gpu"] = "changed"
    elif change == "fingerprint":
        changed["representation_fingerprint"] = "0" * 64
    else:
        changed["repaired_atomic_binding"]["repaired_cache_identity_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="mismatch"):
        compound.extract_pinned(model, tokenizer, benchmark, changed, output, resume=True)
    assert json.loads((output / "progress.json").read_text())["next_row"] == 0
