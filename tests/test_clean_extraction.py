"""Synthetic extraction-only tests. No trained weights, probes or dataset splits read."""
import copy
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from src.clean_extraction import (
    Benchmark, ExtractionWriter, MODELS, SMOKE_STATEMENTS, build_manifest, canonical_json,
    difference_report, extract_clean, layer_convention, ordered_hash, resume_identity,
    smoke_compare, validate_model,
)
from src.extract import extract_acts, extract_batch, last_real_token_indices, saved_hidden_states, semantic_position_ids


@pytest.mark.parametrize("mask,expected", [
    ([[1, 1, 1]], [2]), ([[1, 1, 1, 0, 0]], [2]), ([[0, 0, 1, 1, 1]], [4]),
    ([[0, 1, 0, 1, 0], [1, 0, 0, 0, 0]], [3, 0]),
])
def test_last_real_token_indices(mask, expected):
    assert last_real_token_indices(torch.tensor(mask)).tolist() == expected


@pytest.mark.parametrize("mask", [[[0, 0]], [[1, 0], [0, 0]], [[1, 2]], [[1, -1]],
                                 [[1, float("nan")]], [], [[]], [1, 1]])
def test_malformed_mask_fails(mask):
    with pytest.raises(ValueError):
        last_real_token_indices(torch.tensor(mask))


def test_semantic_positions_both_sides():
    mask = torch.tensor([[1, 1, 1, 0, 0], [0, 0, 1, 1, 1]])
    assert semantic_position_ids(mask).tolist() == [[0, 1, 2, 0, 0], [0, 0, 0, 1, 2]]


def test_layer_index_mapping_assertion():
    assert saved_hidden_states(tuple(range(5)), 4) == (1, 2, 3, 4)
    with pytest.raises(ValueError, match="count"):
        saved_hidden_states(tuple(range(4)), 4)


@pytest.mark.parametrize("key,config_class", [("qwen2_5_7b", Qwen2Config), ("qwen3_8b", Qwen3Config)])
def test_selected_atomic_layers(key, config_class):
    spec = MODELS[key]
    cfg = config_class(num_hidden_layers=spec["layers"], hidden_size=spec["hidden_size"])
    validate_model(cfg, key)
    mapping = layer_convention(cfg.num_hidden_layers)["saved_layer_to_hf_index"]
    assert mapping[spec["selected_layer"]] == {"qwen2_5_7b": 18, "qwen3_8b": 29}[key]
    cfg.hidden_size += 1
    with pytest.raises(ValueError, match="architecture"):
        validate_model(cfg, key)


@pytest.fixture
def fixture(tmp_path):
    frame = pd.DataFrame([{
        "example_id": f"example_{i}", "statement": s, "pair_id": f"pair_{i}", "topic": "synthetic",
        "split": "validation", "operator": "AND", "ordering": "AB", "canonical_truth_a": "True",
        "canonical_truth_b": "False", "compound_label": "False", "protocol": "entity_disjoint",
        "evaluation_phase": "development", "extra_column": f"Unicode é, row\n{i}",
    } for i, s in enumerate(SMOKE_STATEMENTS)])
    path = tmp_path / "benchmark.csv"
    frame.to_csv(path, index=False)
    benchmark = Benchmark(path)
    manifest = {"model": {"identifier": "synthetic", "config_sha256": "original", "revision": "v1"},
                "data": {**benchmark.hashes, "metadata_columns": list(frame.columns),
                         "expected_activation_shape": [4, 2, 3]},
                "numerics": {"batch_size": 2}, "smoke_test": {"passed": True},
                "code": {"implementation_version": "test-v1", "source_sha256": {"test": "hash"},
                         "timestamp_utc": "t0"}}
    manifest["model"]["resolved_commit_sha"] = "a" * 40
    manifest["tokenizer"] = {"resolved_commit_sha": "a" * 40}
    manifest["historical_sample_sha256"] = "fixture-sample"
    manifest["historical_atomic_compatibility"] = {
        "passed": True, "current_model_identifier": "synthetic", "current_model_commit_sha": "a" * 40,
        "current_tokenizer_commit_sha": "a" * 40, "sample_sha256": "fixture-sample"}
    return benchmark, manifest, tmp_path / "clean"


def values(start, stop):
    return np.arange(start * 6, stop * 6, dtype=np.float16).reshape(stop - start, 2, 3)


def start_checkpoint(fixture):
    b, m, p = fixture
    with ExtractionWriter(p, b, m) as writer:
        writer.append(0, b.frame.iloc[:2], values(0, 2))


def test_full_sidecar_preservation_and_resume(fixture):
    b, m, p = fixture
    start_checkpoint(fixture)
    assert (p / "metadata.csv").read_bytes() == b.payload
    with ExtractionWriter(p, b, m, resume=True) as writer:
        assert writer.next_row == 2
        writer.append(2, b.frame.iloc[2:], values(2, 4))
        writer.finalize()
    assert not (p / "activations.partial.npy").exists()
    np.testing.assert_array_equal(np.load(p / "activations.npy"), values(0, 4))
    with ExtractionWriter(p, b, m, resume=True) as writer:
        writer.finalize()  # fully verified, no rewriting
        assert writer.progress["complete"]
    assert json.loads((p / "progress.json").read_text())["next_row"] == 4


def test_metadata_row_order_mismatch_fails(fixture):
    b, m, p = fixture
    with ExtractionWriter(p, b, m) as writer:
        with pytest.raises(ValueError, match="row-order"):
            writer.append(0, b.frame.iloc[:2].iloc[::-1], values(0, 2))
        assert writer.next_row == 0


@pytest.mark.parametrize("change", ["reorder", "statement", "extra_column"])
def test_sidecar_hash_mismatch_prevents_resume(fixture, change):
    b, m, p = fixture
    start_checkpoint(fixture)
    frame = b.frame.copy()
    if change == "reorder":
        frame = frame.iloc[::-1]
    else:
        frame.loc[0, change] = "altered"
    frame.to_csv(p / "metadata.csv", index=False)
    with pytest.raises(ValueError, match="metadata.*mismatch"):
        ExtractionWriter(p, b, m, resume=True)


def test_altered_benchmark_prevents_resume(fixture):
    b, m, p = fixture
    start_checkpoint(fixture)
    b.path.write_bytes(b.payload + b"\n")
    with pytest.raises(ValueError, match="benchmark content hash"):
        ExtractionWriter(p, b, m, resume=True)
    new_b = Benchmark(b.path)
    new_m = copy.deepcopy(m)
    new_m["data"].update(new_b.hashes)
    with pytest.raises(ValueError, match="resume.*mismatch"):
        ExtractionWriter(p, new_b, new_m, resume=True)


@pytest.mark.parametrize("section,key,value", [
    ("model", "identifier", "another-model"), ("model", "config_sha256", "changed"),
    ("model", "revision", "v2"), ("numerics", "batch_size", 1),
    ("code", "source_sha256", {"test": "changed"}),
    ("data", "ordered_example_id_sha256", "reordered"),
    ("data", "ordered_statement_sha256", "changed"),
])
def test_changed_provenance_prevents_resume(fixture, section, key, value):
    b, m, p = fixture
    start_checkpoint(fixture)
    changed = copy.deepcopy(m)
    changed[section][key] = value
    with pytest.raises(ValueError, match="mismatch"):
        ExtractionWriter(p, b, changed, resume=True)


def test_timestamp_may_change(fixture):
    _, m, _ = fixture
    other = copy.deepcopy(m)
    other["code"]["timestamp_utc"] = "t1"
    assert resume_identity(m) == resume_identity(other)


def test_refuse_existing_output_and_concurrent_writer(fixture):
    b, m, p = fixture
    with ExtractionWriter(p, b, m):
        with pytest.raises(BlockingIOError):
            ExtractionWriter(p, b, m, resume=True)
    with pytest.raises(ValueError, match="refusing overwrite"):
        ExtractionWriter(p, b, m)


def test_finalization_requires_complete_once_only_coverage(fixture):
    b, m, p = fixture
    with ExtractionWriter(p, b, m) as writer:
        writer.append(0, b.frame.iloc[:2], values(0, 2))
        with pytest.raises(ValueError, match="once"):
            writer.append(0, b.frame.iloc[:2], values(0, 2))
        with pytest.raises(ValueError, match="incomplete"):
            writer.finalize()


def test_committed_chunk_corruption_fails(fixture):
    b, m, p = fixture
    start_checkpoint(fixture)
    arr = np.load(p / "activations.partial.npy", mmap_mode="r+")
    arr[0, 0, 0] = 100
    arr.flush()
    del arr
    with pytest.raises(ValueError, match="corrupted"):
        ExtractionWriter(p, b, m, resume=True)


def test_journal_overlap_fails(fixture):
    b, m, p = fixture
    start_checkpoint(fixture)
    journal = json.loads((p / "progress.json").read_text())
    journal["chunks"].append(journal["chunks"][0])
    (p / "progress.json").write_text(json.dumps(journal))
    with pytest.raises(ValueError, match="gap/overlap"):
        ExtractionWriter(p, b, m, resume=True)


def test_crash_after_flush_before_journal_recomputes_only_uncommitted_rows(fixture):
    b, m, p = fixture
    start_checkpoint(fixture)
    with ExtractionWriter(p, b, m, resume=True) as writer:
        with patch("src.clean_extraction.atomic_json", side_effect=OSError("simulated crash")):
            with pytest.raises(OSError):
                writer.append(2, b.frame.iloc[2:], values(2, 4))
    with ExtractionWriter(p, b, m, resume=True) as writer:
        assert writer.next_row == 2
        writer.append(2, b.frame.iloc[2:], values(2, 4))
        writer.finalize()
        assert len(writer.progress["chunks"]) == 2


def test_crash_after_final_rename_recovers(fixture):
    b, m, p = fixture
    with ExtractionWriter(p, b, m) as writer:
        writer.append(0, b.frame, values(0, 4))
        with patch("src.clean_extraction.atomic_json", side_effect=OSError("simulated crash")):
            with pytest.raises(OSError):
                writer.finalize()
    with ExtractionWriter(p, b, m, resume=True) as writer:
        assert writer.next_row == 4
        writer.finalize()
        assert writer.progress["complete"]


def test_ordered_hash_unambiguous():
    assert ordered_hash(["a", "b"]) != ordered_hash(["b", "a"])
    assert ordered_hash(["a\nb", "c"]) != ordered_hash(["a", "b\nc"])


def tiny_model_and_tokenizer(kind, dtype=torch.float32, attention="sdpa"):
    torch.manual_seed(123)
    torch.set_num_threads(1)
    words = sorted(set(" ".join(SMOKE_STATEMENTS).replace(".", " .").replace(",", " ,").split()))
    backend = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, **{w: i + 2 for i, w in enumerate(words)}},
                                 unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    cfg_cls, model_cls, layers = {"qwen2": (Qwen2Config, Qwen2ForCausalLM, 28),
                                  "qwen3": (Qwen3Config, Qwen3ForCausalLM, 36)}[kind]
    cfg = cfg_cls(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
                  num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
                  head_dim=8, max_position_embeddings=128, pad_token_id=0)
    cfg._attn_implementation = attention
    return model_cls(cfg).to(dtype=dtype).eval(), tokenizer


@pytest.fixture(scope="module", params=[(kind, dtype, attention)
    for kind in ("qwen2", "qwen3") for dtype in (torch.float32, torch.bfloat16)
    for attention in ("sdpa", "eager")])
def architecture_smoke(request):
    kind, dtype, attention = request.param
    model, tokenizer = tiny_model_and_tokenizer(kind, dtype, attention)
    report = smoke_compare(model, tokenizer)
    report.update(architecture=kind, attention=attention, weights="random tiny-width; NOT pretrained",
                  num_layers=model.config.num_hidden_layers, hidden_size=model.config.hidden_size,
                  torch_version=torch.__version__, transformers_version=__import__("transformers").__version__)
    if os.environ.get("CLEAN_EXTRACTION_SMOKE_REPORT_DIR"):
        directory = Path(os.environ["CLEAN_EXTRACTION_SMOKE_REPORT_DIR"])
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"tiny_{kind}_{attention}_{str(dtype).split('.')[-1]}.json").write_bytes(canonical_json(report) + b"\n")
    return model, tokenizer, report


def test_single_vs_right_padded_smoke(architecture_smoke):
    _, _, report = architecture_smoke
    assert report["comparisons"]["right"]["compute_readout"]["passed"], report
    assert report["comparisons"]["right"]["saved_float16"]["passed"], report


def test_single_vs_left_padded_smoke(architecture_smoke):
    _, _, report = architecture_smoke
    assert report["comparisons"]["left"]["compute_readout"]["passed"], report
    assert report["comparisons"]["left"]["saved_float16"]["passed"], report


def test_unpadded_reference_unchanged(architecture_smoke):
    model, tokenizer, report = architecture_smoke
    assert report["comparisons"]["single_explicit_positions"]["compute_readout"]["max_absolute_difference"] == 0
    # Independent old code path: default positions/cache, full logits, physical
    # sum(mask)-1, HF[1:]. Confirms the optimized clean path preserves readouts.
    enc = tokenizer([SMOKE_STATEMENTS[0]], return_tensors="pt", padding=False)
    with torch.inference_mode():
        out = model(**enc, output_hidden_states=True)
    old = torch.stack([h[0, int(enc.attention_mask.sum()) - 1] for h in out.hidden_states[1:]])
    new = extract_acts(model, tokenizer, [SMOKE_STATEMENTS[0]])[0]
    np.testing.assert_array_equal(new, old.float().numpy().astype(np.float16))


@pytest.mark.parametrize("kind", ["qwen2", "qwen3"])
def test_hf_embedding_block_and_final_norm_mapping(kind):
    model, tokenizer = tiny_model_and_tokenizer(kind)
    captures = {}
    handles = [model.model.embed_tokens.register_forward_hook(lambda m, a, o: captures.update(embedding=o)),
               model.model.norm.register_forward_hook(lambda m, a, o: captures.update(norm=o, pre_norm=a[0]))]
    for i, block in enumerate(model.model.layers):
        handles.append(block.register_forward_hook(lambda m, a, o, i=i: captures.update({i: o})))
    try:
        with torch.inference_mode():
            out = model(**tokenizer([SMOKE_STATEMENTS[0]], return_tensors="pt"), output_hidden_states=True)
        assert len(out.hidden_states) == model.config.num_hidden_layers + 1
        torch.testing.assert_close(out.hidden_states[0], captures["embedding"], rtol=0, atol=0)
        for s in range(model.config.num_hidden_layers - 1):
            torch.testing.assert_close(out.hidden_states[s + 1], captures[s], rtol=0, atol=0)
        torch.testing.assert_close(out.hidden_states[-1], captures["norm"], rtol=0, atol=0)
        assert not torch.equal(out.hidden_states[-1], captures["pre_norm"])
    finally:
        for h in handles:
            h.remove()


def test_wrong_token_readout_fails_smoke_tolerance():
    ref = np.ones((4, 3, 32), dtype=np.float32)
    wrong = ref.copy()
    wrong[0, 1] = 0
    assert not difference_report(ref, wrong, torch.bfloat16)["passed"]


def test_end_to_end_extract_resume_uses_first_uncompleted_row(fixture):
    b, m, p = fixture
    start_checkpoint(fixture)
    calls = []
    def forward(model, tokenizer, statements, **kw):
        calls.extend(statements)
        return torch.from_numpy(values(2, 4).astype(np.float32))
    with patch("src.clean_extraction.extract_batch", side_effect=forward):
        extract_clean(None, None, b, m, p, resume=True)
    assert calls == b.frame.statement.iloc[2:].tolist()
    np.testing.assert_array_equal(np.load(p / "activations.npy"), values(0, 4))


@pytest.mark.parametrize("kind", ["qwen2", "qwen3"])
def test_actual_position_ids_default_vs_clean(kind):
    model, tokenizer = tiny_model_and_tokenizer(kind)
    tokenizer.padding_side = "left"
    captured = []
    handle = model.model.rotary_emb.register_forward_pre_hook(lambda m, args: captured.append(args[1].clone()))
    try:
        extract_batch(model, tokenizer, SMOKE_STATEMENTS[:2], padding=True, explicit_positions=False)
        extract_batch(model, tokenizer, SMOKE_STATEMENTS[:2], padding=True)
    finally:
        handle.remove()
    enc = tokenizer(list(SMOKE_STATEMENTS[:2]), return_tensors="pt", padding=True)
    assert captured[0].tolist() == [list(range(enc.input_ids.shape[1]))]
    torch.testing.assert_close(captured[1], semantic_position_ids(enc.attention_mask), rtol=0, atol=0)
    assert captured[1][0, -1] < captured[0][0, -1]  # shorter statement's semantic last position


def test_manifest_schema_and_no_credentials(fixture):
    b, _, _ = fixture
    model, tokenizer = tiny_model_and_tokenizer("qwen2")
    spec = MODELS["qwen2_5_7b"]
    # Manifest-only fixture: no forward after replacing the tiny config dimensions.
    model.config.hidden_size = spec["hidden_size"]
    model.config._name_or_path = spec["identifier"]
    model.config._commit_hash = spec["revision"]
    tokenizer.init_kwargs["token"] = "synthetic-secret-must-not-be-serialized"
    manifest = build_manifest(b, model, tokenizer, "qwen2_5_7b", spec["revision"], 4)
    assert manifest["data"]["expected_activation_shape"] == [4, 28, 3584]
    assert manifest["data"]["sidecar_sha256"] == b.hashes["benchmark_sha256"]
    assert manifest["activation_convention"]["saved_layer_to_hf_index"][17] == 18
    assert manifest["activation_convention"]["final_saved_state_post_final_norm"]
    assert manifest["activation_convention"]["explicit_position_ids"]
    assert manifest["prompt_regime"]["chat_template_used"] is False
    assert manifest["prompt_regime"]["truncation_enabled"] is False
    assert manifest["numerics"]["saved_activation_dtype"] == "float16"
    assert len(manifest["model"]["identity_sha256"]) == 64
    for name in ("bos", "eos", "pad"):
        assert f"{name}_token" in manifest["tokenizer"] and f"{name}_token_id" in manifest["tokenizer"]
    assert "synthetic-secret" not in canonical_json(manifest).decode()


def test_resumed_array_header_mismatch_fails(fixture):
    b, m, p = fixture
    start_checkpoint(fixture)
    np.save(p / "activations.partial.npy", np.zeros((4, 2, 4), dtype=np.float16))
    with pytest.raises(ValueError, match="header mismatch"):
        ExtractionWriter(p, b, m, resume=True)


def test_source_change_before_finalization_fails(fixture):
    b, m, p = fixture
    with ExtractionWriter(p, b, m) as writer:
        writer.append(0, b.frame, values(0, 4))
        b.path.write_bytes(b.payload + b"\n")
        with pytest.raises(ValueError, match="benchmark content hash"):
            writer.finalize()


@pytest.mark.parametrize("smoke", [{}, {"passed": False}])
def test_missing_or_failed_smoke_never_creates_array(fixture, smoke):
    b, m, p = fixture
    m["smoke_test"] = smoke
    with pytest.raises(ValueError, match="padding smoke required"):
        extract_clean(None, None, b, m, p)
    assert not p.exists()
