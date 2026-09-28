"""Synthetic repair tests. Never open real atomic sources, caches, or test data."""
import builtins
import csv
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import BatchEncoding, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

from src import clean_atomic_extraction as repair


@pytest.fixture
def allowed(tmp_path, monkeypatch):
    rows = []
    for i, topic in enumerate(repair.TOPICS):
        for split in ("train", "validation"):
            for form in ("affirmative", "negated"):
                rows.append(dict(zip(repair.FIELDS, (
                    ("neg_" if form == "negated" else "") + topic, 0 if split == "train" else 1,
                    f"A {'small ' * (i + 1)}box, é.\nSecond line.",
                    f"entity_{topic}_{split}", topic, form, split, i % 2))))
    path = tmp_path / repair.INPUT
    path.parent.mkdir(parents=True)
    write_rows(path, rows)
    monkeypatch.setattr(repair, "COUNTS", {"train": 10, "validation": 10})
    monkeypatch.setattr(repair, "INPUT_DIGEST", repair.digest(repair.canonical(rows)))
    return tmp_path, path, rows


def write_rows(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=repair.FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_load_exact_allowed_records_and_digest(allowed):
    root, _, rows = allowed
    parts, info = repair.load_allowed_rows(root)
    assert parts == {s: [r for r in rows if r["split"] == s] for s in repair.COUNTS}
    assert info["allowed_rows_sha256"] == repair.INPUT_DIGEST
    assert set(parts) == {"train", "validation"}


@pytest.mark.parametrize("mutation,error", [
    ("test", "only train/validation"), ("unknown_split", "only train/validation"),
    ("duplicate", "duplicate"), ("statement", "digest"), ("label", "label"),
    ("entity_overlap", "entity overlap"), ("count", "counts"),
    ("dataset", "dataset/topic/form"),
])
def test_invalid_input_fails_before_model_load(allowed, monkeypatch, mutation, error):
    root, path, rows = allowed
    if mutation in {"test", "unknown_split"}:
        rows[0]["split"] = mutation
    elif mutation == "duplicate":
        rows.append(rows[0].copy())
    elif mutation == "statement":
        rows[0]["statement"] += " altered"
    elif mutation == "label":
        rows[0]["label"] = 2
    elif mutation == "entity_overlap":
        rows[2]["entity_id"] = rows[0]["entity_id"]
    elif mutation == "count":
        rows.pop()
    else:
        rows[0]["dataset"] = "facts"
    write_rows(path, rows)
    loader = Mock(side_effect=AssertionError("must not load model"))
    monkeypatch.setattr(repair, "load_pinned_model", loader)
    with pytest.raises(ValueError, match=error):
        repair.run("extract", root=root)
    loader.assert_not_called()
    assert not (root / repair.OUTPUT_PARENT).exists()


def test_symlink_input_rejected_before_read(allowed):
    root, path, _ = allowed
    path.unlink()
    path.symlink_to(root / "forbidden-mixed-source.csv")
    with pytest.raises(ValueError, match="symlink"):
        repair.load_allowed_rows(root)


@pytest.mark.parametrize("output", ["acts/cities.npy", "acts/clean_protocol/atomic/../historical",
                                    "acts/clean_protocol/atomic/new", "/tmp/out"])
def test_historical_or_unversioned_output_rejected(allowed, output):
    with pytest.raises(ValueError, match="versioned"):
        repair.run("plan", root=allowed[0], output=output)


@pytest.mark.parametrize("mode", ["plan", "smoke", "extract"])
def test_existing_output_refused_even_if_empty(allowed, monkeypatch, mode):
    root = allowed[0]
    path = root / repair.OUTPUT
    path.mkdir(parents=True)
    loader = Mock(side_effect=AssertionError("must not load model"))
    monkeypatch.setattr(repair, "load_pinned_model", loader)
    with pytest.raises(ValueError, match="refusing overwrite"):
        repair.run(mode, root=root)
    assert list(path.iterdir()) == []
    loader.assert_not_called()


def test_symlink_output_ancestor_rejected(allowed):
    root = allowed[0]
    (root / "acts").symlink_to(root / "forbidden-historical-dir", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        repair.run("plan", root=root)


@pytest.fixture
def tiny(monkeypatch):
    # Actual pinned Qwen2 architecture; all 28 saved layers, reduced width.
    monkeypatch.setattr(repair, "WIDTH", 16)
    torch.manual_seed(7)
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "A": 2, "small": 3,
                                  "box": 4, ".": 5}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    config = Qwen2Config(vocab_size=6, hidden_size=16, intermediate_size=24, num_hidden_layers=28,
                        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=128)
    config._attn_implementation = "sdpa"
    model = Qwen2ForCausalLM(config).to(torch.bfloat16).eval()
    return model, tokenizer


def test_every_saved_layer_matches_hooks_including_final_norm(tiny):
    model, tokenizer = tiny
    captured = []
    hooks = [block.register_forward_hook(lambda _m, _i, out: captured.append(
        (out[0] if isinstance(out, tuple) else out).detach().float().cpu())) for block in model.model.layers]
    normalized = []
    hooks.append(model.model.norm.register_forward_hook(
        lambda _m, _i, out: normalized.append(out.detach().float().cpu())))
    try:
        values = repair.readout(model, tokenizer, "A small box.")
    finally:
        for hook in hooks:
            hook.remove()
    assert values.shape == (28, 16)
    for i in range(27):
        torch.testing.assert_close(values[i], captured[i][0, -1], rtol=0, atol=0)
    torch.testing.assert_close(values[-1], normalized[0][0, -1], rtol=0, atol=0)
    assert not torch.equal(values[-1], captured[-1][0, -1])
    assert repair.contract()["saved_layer_to_hf_index"] == list(range(1, 29))
    assert repair.contract()["selected_layer"] is None and repair.contract()["selected_C"] is None


def test_smoke_uses_only_unpadded_single_forwards(allowed, tiny):
    model, tokenizer = tiny
    parts, _ = repair.load_allowed_rows(allowed[0])
    calls, token_calls = [], []

    class RecordingTokenizer:
        def __call__(self, statements, **kwargs):
            token_calls.append((statements, kwargs))
            return tokenizer(statements, **kwargs)

    def intercept(_model, args, kwargs):
        assert not args
        assert kwargs["input_ids"].shape[0] == 1
        assert (kwargs["attention_mask"] == 1).all()
        assert kwargs["use_cache"] is False and kwargs["output_hidden_states"] is True
        torch.testing.assert_close(kwargs["position_ids"], torch.arange(kwargs["input_ids"].shape[1]).unsqueeze(0))
        calls.append(kwargs["logits_to_keep"])

    handle = model.register_forward_pre_hook(intercept, with_kwargs=True)
    try:
        report = repair.smoke(model, RecordingTokenizer(), parts)
    finally:
        handle.remove()
    assert report["passed"] and report["forward_calls"] == 30
    assert calls == [1, 1, 0] * 10
    allowed_statements = {r["statement"] for rows in parts.values() for r in rows}
    for statements, kwargs in token_calls:
        assert len(statements) == 1 and statements[0] in allowed_statements
        assert kwargs == {"return_tensors": "pt", "padding": False, "truncation": False,
                          "add_special_tokens": True, "return_attention_mask": True}


def test_unexpected_padding_rejected(tiny):
    model, _ = tiny
    tokenizer = lambda *_a, **_k: BatchEncoding({"input_ids": torch.tensor([[2, 1]]),
                                               "attention_mask": torch.tensor([[1, 0]])})
    with pytest.raises(ValueError, match="unpadded"):
        repair.readout(model, tokenizer, "A")


def test_over_context_is_rejected_without_truncation(tiny):
    model, tokenizer = tiny
    model.config.max_position_embeddings = 2
    with pytest.raises(ValueError, match="no truncation"):
        repair.readout(model, tokenizer, "A small box.")


@pytest.mark.parametrize("bad_component", ["model_snapshot", "tokenizer_snapshot", "config", "loaded_model"])
def test_revision_mismatch_fails_before_gpu_use(monkeypatch, tmp_path, bad_component):
    import transformers
    from transformers.utils import hub
    monkeypatch.setattr(repair, "require_runtime", lambda: None)
    monkeypatch.setattr(hub, "cached_file", lambda _model, name, **kw: name)
    def commit(path, _):
        bad = (bad_component == "model_snapshot" and path == "config.json") or (
            bad_component == "tokenizer_snapshot" and path == "tokenizer_config.json")
        return "b" * 40 if bad else repair.REVISION
    monkeypatch.setattr(hub, "extract_commit_hash", commit)
    cfg = SimpleNamespace(_commit_hash="b" * 40 if bad_component == "config" else repair.REVISION,
                          model_type="qwen2", num_hidden_layers=28, hidden_size=3584)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", Mock(return_value=cfg))
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", Mock())
    model = Mock(config=SimpleNamespace(_commit_hash="b" * 40))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", Mock(return_value=model))
    with pytest.raises(ValueError, match="revision mismatch"):
        repair.load_pinned_model()
    model.to.assert_not_called()


@pytest.mark.parametrize("torch_version,transformers_version,message", [
    ("2.8.0+cu128", "5.12.1", "torch"), ("2.11.0+cu128", "5.11.0", "transformers")])
def test_runtime_versions_enforced(monkeypatch, torch_version, transformers_version, message):
    import transformers
    monkeypatch.setattr(torch, "__version__", torch_version)
    monkeypatch.setattr(transformers, "__version__", transformers_version)
    with pytest.raises(ValueError, match=message):
        repair.require_runtime()


def test_runtime_accepts_cuda_build_suffix(monkeypatch):
    import transformers
    monkeypatch.setattr(torch, "__version__", "2.11.0+cu128")
    monkeypatch.setattr(transformers, "__version__", "5.12.1")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    repair.require_runtime()


def test_loader_pins_both_snapshots_and_compute_settings(monkeypatch, tmp_path):
    import transformers
    from transformers.utils import hub
    monkeypatch.setattr(repair, "require_runtime", lambda: None)
    snapshot = tmp_path / "snapshots" / repair.REVISION
    snapshot.mkdir(parents=True)
    for name in ("config.json", "tokenizer_config.json"):
        (snapshot / name).write_text("{}")
    def cached(model, name, *, revision):
        assert model == repair.MODEL and revision == repair.REVISION
        return str(snapshot / name)
    monkeypatch.setattr(hub, "cached_file", cached)
    cfg = SimpleNamespace(_commit_hash=repair.REVISION, model_type="qwen2", num_hidden_layers=28,
                          hidden_size=3584, _attn_implementation="sdpa", to_dict=lambda: {"revision": repair.REVISION})
    config_loader = Mock(return_value=cfg)
    tokenizer = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: "{}"))
    tokenizer_loader = Mock(return_value=tokenizer)
    model = Mock(config=cfg, dtype=torch.bfloat16)
    model.to.return_value = model
    model.eval.return_value = model
    model_loader = Mock(return_value=model)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", config_loader)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer_loader)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_loader)
    actual_model, actual_tokenizer, info = repair.load_pinned_model()
    assert actual_model is model and actual_tokenizer is tokenizer
    config_loader.assert_called_once_with(repair.MODEL, revision=repair.REVISION)
    tokenizer_loader.assert_called_once_with(repair.MODEL, revision=repair.REVISION)
    model_loader.assert_called_once_with(repair.MODEL, revision=repair.REVISION, config=cfg,
                                         dtype=torch.bfloat16, attn_implementation="sdpa")
    model.to.assert_called_once_with("cuda")
    model.eval.assert_called_once_with()
    assert info["model_resolved_revision"] == info["tokenizer_resolved_revision"] == repair.REVISION


def install_io_guard(monkeypatch, root, allowed_input):
    """Reject EVERY repo data access except the single input and new outputs.

    This denies mixed CSVs, entity manifests, old caches, and any test artifact
    even if absent; no forbidden sentinel file has to be created or opened.
    """
    reads = []
    def check(path, mode):
        if isinstance(path, int):
            return
        path = Path(os.fsdecode(path)).absolute()
        if not path.is_relative_to(root):
            # Python/library imports are allowed; real repository data is not.
            if path.is_relative_to(repair.ROOT):
                assert path.suffix in {".py", ".pyc"}, f"forbidden real repo access: {path}"
            return
        relative = path.relative_to(root)
        safe = path == allowed_input or relative in map(Path, repair.SOURCES) or (
            relative.is_relative_to(repair.OUTPUT_PARENT) or relative.is_relative_to(repair.REPORTS))
        assert safe, f"forbidden path accessed: {path}"
        reads.append((relative, mode))
        if path == allowed_input:
            assert mode == "r" or mode == "rb" or mode == os.O_RDONLY
    for module in (builtins, io, os):
        original = module.open
        def guarded(path, mode="r", *args, _original=original, **kwargs):
            check(path, mode)
            return _original(path, mode, *args, **kwargs)
        monkeypatch.setattr(module, "open", guarded)
    return reads


@pytest.mark.parametrize("mode", ["plan", "smoke", "extract"])
def test_lifecycle_only_opens_allowed_input_and_new_outputs(allowed, monkeypatch, tiny, mode):
    root, input_path, rows = allowed
    model, tokenizer = tiny
    for name in repair.SOURCES:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic implementation provenance\n")
    monkeypatch.setattr(repair, "load_pinned_model", lambda: (model, tokenizer, {"revision": repair.REVISION}))
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "synthetic CPU test")
    # Real provenance code runs; git receives no data paths.
    reads = install_io_guard(monkeypatch, root, input_path)
    result = repair.run(mode, root=root)
    assert sum(p == repair.INPUT for p, _ in reads) == 1
    assert result["activation_arrays_created"] == (mode == "extract")
    output = root / repair.OUTPUT
    if mode != "extract":
        assert not output.exists()
    else:
        assert {p.name for p in output.iterdir()} == {"train", "validation", "extraction_manifest.json", "completion.json"}
        manifest = json.loads((output / "extraction_manifest.json").read_text())
        completion = json.loads((output / "completion.json").read_text())
        assert manifest["input"]["allowed_rows_sha256"] == repair.INPUT_DIGEST
        assert manifest["contract"]["selected_layer"] is None
        assert set(manifest["source_sha256"]) == set(repair.SOURCES)
        assert completion["complete"]
        assert completion["manifest_sha256"] == repair.file_hash(output / "extraction_manifest.json")
        for split in ("train", "validation"):
            assert {p.name for p in (output / split).iterdir()} == {"metadata.csv", "activations.npy"}
            with (output / split / "metadata.csv").open(newline="") as handle:
                saved = list(csv.DictReader(handle))
            original = [r for r in rows if r["split"] == split]
            assert [r["statement"] for r in saved] == [r["statement"] for r in original]
            assert [r["row_index"] for r in saved] == [str(r["row_index"]) for r in original]
            values = np.load(output / split / "activations.npy", allow_pickle=False)
            assert values.shape == (10, 28, 16) and values.dtype == np.float16
            np.testing.assert_array_equal(values[0], repair.readout(model, tokenizer, original[0]["statement"]).numpy().astype(np.float16))
        for relative, record in completion["outputs"].items():
            assert repair.file_hash(output / relative) == record["sha256"]


@pytest.mark.parametrize("failure", ["comparison", "exception"])
def test_failed_smoke_saves_diagnostic_but_never_arrays(allowed, monkeypatch, failure):
    root = allowed[0]
    monkeypatch.setattr(repair, "load_pinned_model", lambda: (None, None, {}))
    monkeypatch.setattr(repair, "provenance", lambda *_: {})
    def failed(*_):
        if failure == "exception":
            raise ValueError("synthetic forward failure")
        return {"passed": False}
    monkeypatch.setattr(repair, "smoke", failed)
    with pytest.raises(ValueError):
        repair.run("extract", root=root)
    assert not (root / repair.OUTPUT).exists()
    reports = list((root / repair.REPORTS).glob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["smoke"]["passed"] is False


def test_writer_does_not_replace_concurrently_created_directory(allowed):
    root = allowed[0]
    path = repair.output_path(root)
    path.mkdir(parents=True)
    sentinel = path / "keep"
    sentinel.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError):
        repair.write_arrays(path, {}, None, None, {"smoke": {"passed": True}})
    assert sentinel.read_bytes() == b"unchanged"


def test_repeat_variation_fails_exact_smoke(allowed, monkeypatch):
    parts, _ = repair.load_allowed_rows(allowed[0])
    values = iter([torch.zeros(28, 3584), torch.ones(28, 3584)] * 10)
    monkeypatch.setattr(repair, "readout", lambda *_: next(values))
    monkeypatch.setattr(repair, "reference_readout", lambda *_: torch.zeros(28, 3584))
    assert not repair.smoke(None, None, parts)["passed"]


def test_interrupted_writer_never_marks_complete_or_allows_overwrite(allowed, monkeypatch):
    root = allowed[0]
    parts, _ = repair.load_allowed_rows(root)
    path = repair.output_path(root)
    monkeypatch.setattr(repair, "readout", Mock(side_effect=RuntimeError("synthetic interruption")))
    with pytest.raises(RuntimeError, match="interruption"):
        repair.write_arrays(path, parts, None, None, {"smoke": {"passed": True}})
    assert (path / "train/activations.partial.npy").exists()
    assert not (path / "train/activations.npy").exists()
    assert not (path / "completion.json").exists()
    with pytest.raises(ValueError, match="refusing overwrite"):
        repair.output_path(root)


def test_writer_refuses_extra_partition(allowed):
    root = allowed[0]
    parts, _ = repair.load_allowed_rows(root)
    parts["test"] = []  # invalid synthetic request; not a test-data file
    path = repair.output_path(root)
    with pytest.raises(ValueError, match="only complete train/validation"):
        repair.write_arrays(path, parts, None, None, {"smoke": {"passed": True}})
    assert list(path.iterdir()) == []
