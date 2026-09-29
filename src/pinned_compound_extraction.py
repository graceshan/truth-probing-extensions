"""Canonical Qwen2.5 compound extraction, bound to repaired atomic representations.

No probe artifacts, historical cache readers, or historical compatibility gates.
"""
from datetime import datetime, timezone
from pathlib import Path
import platform
import re
import subprocess
import uuid

import numpy as np
import torch
import transformers

from src import clean_atomic_extraction as atomic
from src.clean_extraction import Benchmark, ExtractionWriter, canonical_json, digest, layer_convention
from src.repaired_atomic_cache import RepairedAtomicCache, CACHE

ROOT = atomic.ROOT
BENCHMARK = Path("data/clean_protocol/compounds/entity_disjoint/development_validation_v1/"
                 "development_validation_compounds.csv")
BENCHMARK_SHA256 = "96e96515e780086065694155e3061bdfe6f4bb80af062a63c2deecd6bfad6f94"
ROW_COUNT, LAYERS, WIDTH = 8384, 28, 3584
OUTPUT_PARENT = Path("acts/clean_protocol/entity_disjoint/development_validation_v1")
OUTPUT = OUTPUT_PARENT / "qwen2_5_7b_a09a354_bs1_bf16_v1"
REPORTS = Path("results/clean_protocol/pinned_compound_extraction")
VERSION = "qwen25-pinned-compound-extraction-v1"
SOURCES = ("src/pinned_compound_extraction.py", "src/repaired_atomic_cache.py",
           "src/clean_atomic_extraction.py", "src/clean_extraction.py", "src/extract.py",
           "scripts/35_extract_pinned_compound_activations.py")
require, file_hash, plain_path = atomic.require, atomic.file_hash, atomic.plain_path


def representation_fields(settings):
    """Explicit allowlist: no runtime, path, timestamp, or probe selection fields."""
    keys = ("model", "model_revision", "tokenizer_revision", "compute_dtype", "attention",
            "batch_size", "padding", "input", "add_special_tokens", "chat_template", "truncation",
            "explicit_semantic_position_ids", "use_cache", "output_hidden_states", "saved_dtype", "readout")
    return {"schema_version": 1, **{k: settings[k] for k in keys},
            "num_hidden_layers": LAYERS, "hidden_size": WIDTH,
            "layer_convention": layer_convention(LAYERS)}


def fingerprint(fields):
    return digest(canonical_json(fields))


def bind_repaired_cache(root):
    cache = RepairedAtomicCache(root)
    require((cache.n_layers, cache.hidden_size) == (LAYERS, WIDTH), "repaired architecture mismatch")
    fields = representation_fields(cache.manifest["contract"])
    expected = fingerprint(representation_fields(atomic.contract()))
    actual = fingerprint(fields)
    require(actual == expected, "repaired representation fingerprint mismatch")
    require(cache.manifest.get("representation_fingerprint", actual) == actual,
            "recorded repaired representation fingerprint mismatch")
    binding = {"passed": True, "representation": fields, "representation_fingerprint": actual,
               "cache_path": str(CACHE), "repaired_cache_files": cache.files,
               "repaired_cache_identity_sha256": fingerprint(cache.files)}
    return cache, binding


def validate_live_binding(cache, model_info):
    for name in ("model_resolved_revision", "tokenizer_resolved_revision"):
        require(model_info.get(name) == atomic.REVISION, f"fresh {name} mismatch")
    # These content identities supplement the stable representation descriptor.
    # Never follow paths recorded in either atomic manifest or probe artifacts.
    for name in ("config_file_sha256", "tokenizer_config_file_sha256", "tokenizer_backend_sha256"):
        expected = cache.manifest["resolved_model"].get(name)
        require(isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None and
                model_info.get(name) == expected, f"fresh/repaired {name} mismatch")


def load_benchmark(root):
    benchmark = Benchmark(plain_path(root, BENCHMARK))
    require(len(benchmark.frame) == ROW_COUNT, "expected exactly 8,384 development/validation rows")
    require(benchmark.hashes["benchmark_sha256"] == BENCHMARK_SHA256, "frozen benchmark hash mismatch")
    return benchmark


def output_path(root, output, resume=False):
    output = Path(output)
    require(output.parent == OUTPUT_PARENT and
            re.fullmatch(r"qwen2_5_7b_a09a354_bs1_bf16_v[1-9][0-9]*", output.name) is not None,
            "output must be a versioned pinned-representation compound directory")
    path = plain_path(root, output)
    if resume:
        require(path.is_dir(), "resume requires an existing pinned output directory")
        for name in (".writer.lock", "extraction_manifest.json", "progress.json", "metadata.csv",
                     "activations.partial.npy", "activations.npy", "progress.json.tmp", "extraction_manifest.json.tmp"):
            plain_path(root, output / name)
    else:
        require(not path.exists(), "refusing overwrite; use --resume for this exact run")
    return path


def pinned_readout(model, tokenizer, statement):
    # Shared with the repaired atomic producer; used by replay, smoke AND extraction.
    return atomic.readout(model, tokenizer, statement)


def atomic_sample(cache):
    samples = []
    for topic in atomic.TOPICS:
        for split, form in (("train", "affirmative"), ("validation", "negated")):
            candidates = [(i, r) for i, r in enumerate(cache.rows[split])
                          if r["topic"] == topic and r["form"] == form]
            require(bool(candidates), "missing repaired replay stratum")
            index, row = min(candidates, key=lambda item: digest(canonical_json(
                [item[1]["dataset"], item[1]["row_index"], item[1]["statement"]])))
            samples.append({"dataset": row["dataset"], "row_index": row["row_index"],
                            "cache_row_index": index, "split": split, "topic": topic, "form": form,
                            "entity_id": row["entity_id"], "statement": row["statement"]})
    return samples


def differences(reference, current):
    require(reference.shape == current.shape, "comparison shape mismatch")
    require(np.isfinite(reference).all() and np.isfinite(current).all(), "nonfinite comparison values")
    # Inputs: [examples, layers, hidden]. Report differences in saved float16 units.
    delta = np.abs(reference.astype(np.float64) - current.astype(np.float64))
    return {"max_absolute_difference": float(delta.max()), "mean_absolute_difference": float(delta.mean()),
            "per_layer": [{"saved_layer": i, "max_absolute_difference": float(delta[:, i].max()),
                           "mean_absolute_difference": float(delta[:, i].mean())} for i in range(LAYERS)],
            "saved_float16_byte_equal": reference.tobytes(order="C") == current.tobytes(order="C")}


def replay_atomic(model, tokenizer, cache, samples):
    references, current, rows = [], [], []
    for sample in samples:
        path = plain_path(cache.root, CACHE / sample["split"] / "activations.npy")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        reference = np.array(array[sample["cache_row_index"]], copy=True)
        del array  # Only the selected permitted row is numerically materialized.
        fresh = pinned_readout(model, tokenizer, sample["statement"]).numpy().astype(np.float16)
        comparison = differences(reference[None], fresh[None])
        rows.append({**sample, **comparison})
        references.append(reference)
        current.append(fresh)
    result = differences(np.stack(references), np.stack(current))
    return {**result, "passed": result["saved_float16_byte_equal"],
            "policy": "exact-repaired-atomic-float16-replay-v1", "batch_size": 1, "padding": False,
            "sample_sha256": fingerprint(samples), "rows": rows,
            "selection_rule": "per topic: train affirmative + validation negated; minimum SHA256 compact JSON [dataset,row_index,statement]"}


def compound_smoke(model, tokenizer, benchmark):
    records = benchmark.frame[["example_id", "statement", "split"]].to_dict("records")
    samples = sorted(records, key=lambda r: fingerprint([r["example_id"], r["statement"]]))[:4]
    rows = []
    for row in samples:
        first = pinned_readout(model, tokenizer, row["statement"]).numpy()
        repeat = pinned_readout(model, tokenizer, row["statement"]).numpy()
        reference = atomic.reference_readout(model, tokenizer, row["statement"]).numpy()
        checks = {}
        for name, candidate in (("repeat", repeat), ("independent_full_logits", reference)):
            check = differences(first.astype(np.float16)[None], candidate.astype(np.float16)[None])
            check["compute_equal"] = bool(np.array_equal(first, candidate))
            checks[name] = check
        rows.append({**row, "checks": checks})
    passed = all(c["compute_equal"] and c["saved_float16_byte_equal"] for row in rows for c in row["checks"].values())
    return {"passed": passed, "policy": "unpadded-single-repeat-direct-exact-v1", "batch_size": 1,
            "padding": False, "sample_sha256": fingerprint(samples), "forward_calls": 3 * len(samples), "rows": rows,
            "selection_rule": "four smallest SHA256 compact JSON [example_id,statement]"}


def runtime_provenance():
    return {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "gpu": torch.cuda.get_device_name(),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
            "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled()}


def make_manifest(root, output, benchmark, binding, samples, model_info):
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    return {"schema_version": 1, "representation": binding["representation"],
            "representation_fingerprint": binding["representation_fingerprint"],
            "repaired_atomic_binding": binding, "atomic_replay_sample_sha256": fingerprint(samples),
            "resolved_model": model_info, "execution_contract": atomic.contract(),
            "numerics": {"batch_size": 1, "runtime": runtime_provenance()},
            "output_path": str(output),
            "data": {**benchmark.hashes, "benchmark_path": str(BENCHMARK),
                     "metadata_columns": list(benchmark.frame.columns), "number_of_examples": ROW_COUNT,
                     "expected_activation_shape": [ROW_COUNT, LAYERS, WIDTH]},
            "code": {"implementation_version": VERSION,
                     "source_sha256": {p: file_hash(plain_path(root, p)) for p in SOURCES},
                     "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                     "git_head_context_only": head.stdout.strip() if head.returncode == 0 else None},
            "smoke_test": {}}


def require_canonical_gates(manifest):
    expected = fingerprint(representation_fields(atomic.contract()))
    binding = manifest["repaired_atomic_binding"]
    require(fingerprint(manifest["representation"]) == manifest["representation_fingerprint"] == expected ==
            binding["representation_fingerprint"], "compound representation fingerprint mismatch")
    require(binding["passed"] is True and fingerprint(binding["repaired_cache_files"]) ==
            binding["repaired_cache_identity_sha256"], "repaired cache binding mismatch")
    gates = manifest["smoke_test"]
    require(gates.get("live_binding_passed") is True, "fresh model/tokenizer binding required")
    replay = gates.get("atomic_replay", {})
    require(replay.get("passed") is True and replay.get("saved_float16_byte_equal") is True and
            replay.get("sample_sha256") == manifest["atomic_replay_sample_sha256"], "exact repaired atomic replay required")
    require(gates.get("compound_smoke", {}).get("passed") is True, "passing unpadded compound smoke required")


def extract_pinned(model, tokenizer, benchmark, manifest, output, *, resume=False):
    """Thin gate-specific loop around the original durable writer, unchanged."""
    require_canonical_gates(manifest)
    require(manifest["execution_contract"] == atomic.contract(), "execution convention mismatch")
    require(manifest["data"]["expected_activation_shape"] == [ROW_COUNT, LAYERS, WIDTH] and
            len(benchmark.frame) == ROW_COUNT, "compound extraction shape mismatch")
    with ExtractionWriter(output, benchmark, manifest, resume=resume) as writer:
        for start in range(writer.next_row, ROW_COUNT):
            rows = benchmark.frame.iloc[start:start + 1].copy()
            benchmark.verify_rows(start, rows)
            values = pinned_readout(model, tokenizer, rows.statement.iloc[0]).numpy().astype(np.float16)[None]
            writer.append(start, rows, values)
            if writer.next_row % 100 == 0:
                print(f"Committed {writer.next_row}/{ROW_COUNT} compound rows", flush=True)
        writer.finalize()


def run(mode="plan", *, root=ROOT, output=OUTPUT, resume=False):
    require(mode in {"plan", "smoke", "extract"}, "unknown mode")
    require(not resume or mode == "extract", "--resume requires --mode extract")
    path = output_path(root, output, resume)
    benchmark = load_benchmark(root)
    cache, binding = bind_repaired_cache(root)
    samples = atomic_sample(cache)
    plan = {"mode": mode, "representation_fingerprint": binding["representation_fingerprint"],
            "representation": binding["representation"], "repaired_cache_identity": binding["repaired_cache_identity_sha256"],
            "benchmark_sha256": benchmark.hashes["benchmark_sha256"], "shape": [ROW_COUNT, LAYERS, WIDTH],
            "output": str(output), "atomic_replay_sample": samples, "atomic_replay_sample_sha256": fingerprint(samples),
            "model_loaded": False, "activation_arrays_created": False}
    if mode == "plan":
        return plan
    model, tokenizer, model_info = atomic.load_pinned_model()
    manifest = make_manifest(root, output, benchmark, binding, samples, model_info)
    reports = plain_path(root, REPORTS)
    reports.mkdir(parents=True, exist_ok=True)
    report = reports / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex + ".json")
    # Diagnostic locations/timestamps are excluded by the writer's existing code/smoke identity rules.
    manifest["code"]["diagnostics_path"] = str(report.relative_to(Path(root).resolve()))
    try:
        validate_live_binding(cache, model_info)
        manifest["smoke_test"]["live_binding_passed"] = True
        manifest["smoke_test"]["atomic_replay"] = replay_atomic(model, tokenizer, cache, samples)
        require(manifest["smoke_test"]["atomic_replay"]["passed"], "exact atomic replay failed; extraction blocked")
        manifest["smoke_test"]["compound_smoke"] = compound_smoke(model, tokenizer, benchmark)
        require_canonical_gates(manifest)
    except Exception as exc:
        manifest["smoke_test"]["error"] = {"type": type(exc).__name__, "message": str(exc)}
        atomic.write_new_json(report, manifest)
        print(f"Failed canonical diagnostics: {report}", flush=True)
        raise
    atomic.write_new_json(report, manifest)
    print(f"Canonical diagnostics: {report}", flush=True)
    if mode == "extract":
        output_path(root, output, resume)  # recheck after the full-weight gates
        extract_pinned(model, tokenizer, benchmark, manifest, path, resume=resume)
    return {**plan, "model_loaded": True, "activation_arrays_created": mode == "extract", "diagnostics": str(report)}
