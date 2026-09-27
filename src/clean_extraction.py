"""Development/validation activation extraction only: no probe or test-data APIs.

The journal commits contiguous chunks AFTER durable data writes. A crash may
recompute the uncommitted tail, but each row belongs to exactly one committed
chunk. Completed chunks are hash-verified and never rewritten on resume.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import torch
import transformers

from src.extract import extract_batch

VERSION = "clean-extraction-v2-atomic-compatibility"
ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = Path("data/clean_protocol/compounds/entity_disjoint/development_validation_v1/"
                 "development_validation_compounds.csv")
OUTPUT_ROOT = Path("acts/clean_protocol/entity_disjoint/development_validation_v1")
MODELS = {
    "qwen2_5_7b": {"identifier": "Qwen/Qwen2.5-7B-Instruct", "layers": 28, "hidden_size": 3584,
                    "model_type": "qwen2", "selected_layer": 17,
                    "revision": "a09a35458c702b33eeacc393d103063234e8bc28"},
    "qwen3_8b": {"identifier": "Qwen/Qwen3-8B", "layers": 36, "hidden_size": 4096,
                  "model_type": "qwen3", "selected_layer": 28,
                  "revision": "b968826d9c46dd6066d109eabc6255188de91218"},
}
REQUIRED_COLUMNS = ("example_id", "statement", "pair_id", "topic", "split", "operator",
                    "ordering", "canonical_truth_a", "canonical_truth_b", "compound_label")
SMOKE_STATEMENTS = (
    "Rain falls.",
    "A small green box is beside the window.",
    "The copper key is on the wooden table and the blue book is inside the drawer.",
    "The lantern is lit or the garden gate is closed, but the quiet room has two chairs.",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical_json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def ordered_hash(values):
    # JSON array encoding is unambiguous for delimiters, newlines and Unicode.
    return digest(canonical_json(list(values)))


class Benchmark:
    def __init__(self, path):
        self.path = Path(path)
        self.payload = self.path.read_bytes()
        header = next(csv.reader(io.StringIO(self.payload.decode("utf-8"))))
        require(len(header) == len(set(header)), "duplicate metadata columns")
        self.frame = pd.read_csv(io.BytesIO(self.payload), dtype=str, keep_default_na=False)
        require(set(REQUIRED_COLUMNS) <= set(self.frame.columns), "missing required metadata columns")
        require(len(self.frame) > 0, "empty benchmark")
        require(not self.frame.example_id.duplicated().any(), "duplicate example_id")
        for col in REQUIRED_COLUMNS:
            require(self.frame[col].str.strip().ne("").all(), f"empty {col}")
        require(set(self.frame.split) <= {"validation", "development"}, "only development/validation allowed")
        if "protocol" in self.frame:
            require(set(self.frame.protocol) == {"entity_disjoint"}, "wrong protocol")
        if "evaluation_phase" in self.frame:
            require(set(self.frame.evaluation_phase) <= {"development", "development_validation"},
                    "wrong evaluation phase")
        self.hashes = {
            "benchmark_sha256": digest(self.payload),
            "ordered_example_id_sha256": ordered_hash(self.frame.example_id),
            "ordered_statement_sha256": ordered_hash(self.frame.statement),
            "sidecar_sha256": digest(self.payload),
        }

    def verify_source(self):
        require(file_hash(self.path) == self.hashes["benchmark_sha256"], "benchmark content hash mismatch")

    def verify_sidecar(self, path):
        other = Benchmark(path)
        require(other.hashes == self.hashes and other.frame.equals(self.frame),
                "metadata row-order/content/hash mismatch")

    def verify_rows(self, start, rows):
        expected = self.frame.iloc[start:start + len(rows)].reset_index(drop=True)
        require(rows.reset_index(drop=True).equals(expected), "metadata row-order/content mismatch")


def layer_convention(n_layers):
    require(n_layers > 0, "invalid layer count")
    return {
        "output_hidden_states": True, "embedding_state_included_by_hf": True,
        "embedding_state_saved": False, "saved_layer_zero_hf_index": 1,
        "saved_layer_to_hf_index": list(range(1, n_layers + 1)),
        "mapping": "saved[s] = HF hidden_states[s + 1]; s = 0..num_hidden_layers-1",
        "intermediate_states": "block residual output before the next block",
        "final_saved_state_post_final_norm": True,
        "final_normalization": "model final RMSNorm (not final block pre-norm residual)",
        "token_readout": "last real token",
        "last_real_token_algorithm": "max(where(attention_mask == 1, arange(sequence_length), -1)); reject all-padding rows",
        "explicit_position_ids": True,
        "position_ids_algorithm": "(attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)",
    }


def validate_model(config, model_key):
    spec = MODELS[model_key]
    require((config.model_type, config.num_hidden_layers, config.hidden_size) ==
            (spec["model_type"], spec["layers"], spec["hidden_size"]), "model/config architecture mismatch")
    require(0 <= spec["selected_layer"] < config.num_hidden_layers, "selected layer out of range")
    require(layer_convention(config.num_hidden_layers)["saved_layer_to_hf_index"][spec["selected_layer"]]
            == spec["selected_layer"] + 1, "atomic layer mapping changed")


def code_provenance():
    def git(*args):
        r = subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True)
        return r.stdout.strip() if r.returncode == 0 else None
    head, status = git("rev-parse", "HEAD"), git("status", "--porcelain")
    return {
        "implementation_version": VERSION, "git_commit_sha": head if status == "" else None,
        "git_head_sha": head, "git_dirty": status != "",
        "source_sha256": {p: file_hash(ROOT / p) for p in
                          ("src/extract.py", "src/clean_extraction.py", "src/atomic_activation_compatibility.py",
                           "src/data.py", "scripts/28_extract_clean_activations.py")},
    }


def build_manifest(benchmark, model, tokenizer, model_key, revision, batch_size):
    validate_model(model.config, model_key)
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "use an immutable model commit SHA")
    require(model.config._commit_hash == revision, "loaded model revision mismatch")
    require(tokenizer.padding_side in {"left", "right"}, "invalid padding side")
    require(batch_size > 0, "invalid batch size")
    spec = MODELS[model_key]
    model_config = model.config.to_dict()
    # Config fields are model parameters, not download/authentication kwargs.
    config_hash = digest(canonical_json(model_config))
    model_identity_hash = digest(canonical_json({"identifier": spec["identifier"], "revision": revision,
                                                "config_sha256": config_hash}))
    tok = {"identifier": spec["identifier"], "requested_revision": revision,
           "resolved_commit_sha": getattr(tokenizer, "_clean_resolved_commit_hash", None) or tokenizer.init_kwargs.get("_commit_hash"),
           "revision_resolution": "HF cached tokenizer_config.json snapshot path",
           "tokenizer_config_sha256": getattr(tokenizer, "_clean_tokenizer_config_sha256", None),
           "class": tokenizer.__class__.__name__, "vocab_size": tokenizer.vocab_size,
           "vocab_size_with_added_tokens": len(tokenizer),
           "padding_side": tokenizer.padding_side, "truncation_side": tokenizer.truncation_side,
           "add_special_tokens": True,
           "backend_sha256": digest(tokenizer.backend_tokenizer.to_str().encode()),
           "special_tokens_map": {k: str(v) for k, v in tokenizer.special_tokens_map.items()}}
    for name in ("bos", "eos", "pad"):
        tok[f"{name}_token"] = getattr(tokenizer, f"{name}_token")
        tok[f"{name}_token_id"] = getattr(tokenizer, f"{name}_token_id")
    import transformers.utils.output_capturing as capturing
    device = next(model.parameters()).device
    benchmark_path = benchmark.path.resolve()
    if benchmark_path.is_relative_to(ROOT):
        benchmark_path = benchmark_path.relative_to(ROOT)
    return {
        "schema_version": 1,
        "model": {"identifier": spec["identifier"], "requested_revision": revision,
                  "resolved_commit_sha": model.config._commit_hash,
                  "config_identifier": model.config.name_or_path,
                  "config_sha256": config_hash, "config": model_config,
                  "identity_sha256": model_identity_hash,
                  "num_transformer_layers": spec["layers"], "hidden_size": spec["hidden_size"],
                  "selected_atomic_probe_saved_layer": spec["selected_layer"],
                  "attention_implementation": model.config._attn_implementation,
                  "implementation_source_sha256": file_hash(inspect.getfile(type(model))),
                  "hf_output_capture_source_sha256": file_hash(inspect.getfile(capturing))},
        "tokenizer": tok,
        "prompt_regime": {"input": "raw statement text from metadata.statement without modification",
                          "chat_template_used": False,
                          "qwen3_thinking_mode": "not applicable because no chat template is used",
                          "tokenizer_call_options": {"return_tensors": "pt", "padding": batch_size > 1,
                                                     "truncation": False, "add_special_tokens": True,
                                                     "return_attention_mask": True},
                          "truncation_enabled": False, "max_length": None,
                          "over_context_policy": "raise; never truncate",
                          "model_call_options": {"output_hidden_states": True, "use_cache": False,
                                                 "logits_to_keep": 1}},
        "activation_convention": layer_convention(spec["layers"]),
        "numerics": {"model_compute_dtype": str(model.dtype), "saved_activation_dtype": "float16",
                     "device": str(device), "batch_size": batch_size,
                     "torch_version": torch.__version__, "transformers_version": transformers.__version__,
                     "numpy_version": np.__version__, "cuda_version": torch.version.cuda,
                     "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                     "float32_matmul_precision": torch.get_float32_matmul_precision(),
                     "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                     "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32},
        "data": {"protocol": "entity_disjoint", "evaluation_phase": "development_validation",
                 "benchmark_path": str(benchmark_path), **benchmark.hashes,
                 "number_of_examples": len(benchmark.frame),
                 "metadata_columns": list(benchmark.frame.columns),
                 "expected_activation_shape": [len(benchmark.frame), spec["layers"], spec["hidden_size"]],
                 "ordered_hash_encoding": "UTF-8 compact JSON array, ensure_ascii=false"},
        "code": {**code_provenance(), "timestamp_utc": datetime.now(timezone.utc).isoformat()},
    }


def resume_identity(manifest):
    # Timestamp and whole-repository dirty status may change independently of
    # extraction. The actual implementation hashes and every data/numeric field
    # must match. Smoke numeric observations are diagnostic, not configuration.
    value = json.loads(json.dumps(manifest))
    code = value["code"]
    value["code"] = {k: code[k] for k in ("implementation_version", "source_sha256")}
    value.pop("smoke_test", None)
    value.pop("historical_atomic_compatibility", None)
    return digest(canonical_json(value))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("wb") as f:
        f.write(canonical_json(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)
    fsync_directory(path.parent)


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


class ExtractionWriter:
    """Locked float16 NPY memmap plus durable, hash-checked contiguous journal."""
    def __init__(self, output, benchmark, manifest, *, resume=False):
        self.output, self.benchmark = Path(output), benchmark
        self.array = None
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = (self.output / ".writer.lock").open("a+b")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._open(manifest, resume)
        except BaseException:
            self.close()
            raise

    def _open(self, manifest, resume):
        self.benchmark.verify_source()
        self.manifest_path = self.output / "extraction_manifest.json"
        self.progress_path = self.output / "progress.json"
        self.sidecar = self.output / "metadata.csv"
        self.partial = self.output / "activations.partial.npy"
        self.final = self.output / "activations.npy"
        self.identity = resume_identity(manifest)
        self.shape = tuple(manifest["data"]["expected_activation_shape"])
        require(self.shape[0] == len(self.benchmark.frame), "activation/metadata row count mismatch")
        require(all(manifest["data"].get(k) == v for k, v in self.benchmark.hashes.items()),
                "manifest benchmark hashes mismatch")
        require(manifest["data"]["metadata_columns"] == list(self.benchmark.frame.columns),
                "manifest metadata columns mismatch")
        if resume:
            old = json.loads(self.manifest_path.read_text())
            require(resume_identity(old) == self.identity, "resume benchmark/model/config/provenance mismatch")
            self.benchmark.verify_sidecar(self.sidecar)
            self.progress = json.loads(self.progress_path.read_text())
            require(self.progress["identity_sha256"] == self.identity, "progress identity mismatch")
            require(not (self.partial.exists() and self.final.exists()), "ambiguous activation files")
            if self.final.exists():
                require(self.progress["next_row"] == self.shape[0], "premature final activation file")
                path = self.final
            else:
                require(not self.progress["complete"], "completed activation file missing")
                path = self.partial
            # Validate data, coverage and hashes before opening anything writable.
            self.array = np.load(path, mmap_mode="r", allow_pickle=False)
            self.verify_chunks()
            self.array = None
            self.array = np.load(path, mmap_mode="r" if path == self.final else "r+", allow_pickle=False)
        else:
            require(set(p.name for p in self.output.iterdir()) <= {".writer.lock"},
                    "output directory is not empty; refusing overwrite (use --resume for this exact run)")
            with self.sidecar.open("xb") as f:
                f.write(self.benchmark.payload)  # byte-exact FULL source sidecar
                f.flush()
                os.fsync(f.fileno())
            self.benchmark.verify_sidecar(self.sidecar)
            atomic_json(self.manifest_path, manifest)
            self.array = np.lib.format.open_memmap(self.partial, mode="w+", dtype=np.float16, shape=self.shape)
            self.progress = {"identity_sha256": self.identity, "next_row": 0, "complete": False, "chunks": []}
            self._flush()
            atomic_json(self.progress_path, self.progress)

    @property
    def next_row(self):
        return self.progress["next_row"]

    def verify_chunks(self):
        require(self.array.shape == self.shape and self.array.dtype == np.float16, "activation header mismatch")
        cursor = 0
        for c in self.progress["chunks"]:
            require(c["start"] == cursor and cursor < c["stop"] <= self.shape[0],
                    "chunk coverage gap/overlap; rows must be committed exactly once")
            rows = self.benchmark.frame.iloc[cursor:c["stop"]]
            require(c["ordered_example_id_sha256"] == ordered_hash(rows.example_id) and
                    c["ordered_statement_sha256"] == ordered_hash(rows.statement), "chunk row alignment mismatch")
            values = self.array[cursor:c["stop"]]
            require(np.isfinite(values).all(), "nonfinite saved activations")
            require(digest(values.tobytes(order="C")) == c["activation_sha256"], "committed activation chunk corrupted")
            cursor = c["stop"]
        require(cursor == self.next_row, "progress does not match chunk coverage")
        if self.progress["complete"]:
            require(cursor == self.shape[0], "incomplete row coverage")
            require(file_hash(self.final) == self.progress["activation_file_sha256"], "final activation hash mismatch")

    def _flush(self):
        self.array.flush()
        with self.partial.open("rb") as f:
            os.fsync(f.fileno())

    def append(self, start, rows, values):
        require(not self.progress["complete"] and start == self.next_row,
                "rows must be appended once in fixed order")
        require(len(rows) > 0, "empty chunk")
        self.benchmark.verify_rows(start, rows)
        values = np.asarray(values)
        require(values.shape == (len(rows), *self.shape[1:]), "activation chunk shape mismatch")
        require(values.dtype == np.float16 and np.isfinite(values).all(), "invalid float16 activation chunk")
        stop = start + len(rows)
        self.array[start:stop] = values
        self._flush()
        self.progress["chunks"].append({"start": start, "stop": stop,
                                        "activation_sha256": digest(values.tobytes(order="C")),
                                        "ordered_example_id_sha256": ordered_hash(rows.example_id),
                                        "ordered_statement_sha256": ordered_hash(rows.statement)})
        self.progress["next_row"] = stop
        atomic_json(self.progress_path, self.progress)

    def finalize(self):
        self.benchmark.verify_source()
        self.benchmark.verify_sidecar(self.sidecar)
        require(self.next_row == self.shape[0], "cannot finalize incomplete extraction")
        self.verify_chunks()
        if self.progress["complete"]:
            return
        if self.partial.exists():
            self._flush()
            self.array = None
            require(not self.final.exists(), "refusing to overwrite final activation file")
            os.rename(self.partial, self.final)
            fsync_directory(self.output)
        self.progress.update(complete=True, activation_file_sha256=file_hash(self.final),
                             finalized_utc=datetime.now(timezone.utc).isoformat())
        atomic_json(self.progress_path, self.progress)

    def close(self):
        self.array = None
        if getattr(self, "lock", None) is not None:
            self.lock.close()
            self.lock = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def difference_report(reference, candidate, dtype):
    """Bound max AND mean error against each reference layer's RMS scale.

    BF16 unit roundoff ~0.0039; allow accumulated kernel-order error up to
    2% RMS at the worst coordinate and 0.3% RMS on average. FP16/FP32 use
    tighter bounds. No cancellation across layers or float16 casts can hide
    a failure: compare both compute readouts and actual saved values.
    """
    tolerances = {torch.float32: (1e-4, 1e-5), torch.float16: (5e-3, 1e-3),
                  torch.bfloat16: (2e-2, 3e-3)}
    max_tol, mean_tol = tolerances[dtype]
    ref, got = np.asarray(reference, dtype=np.float32), np.asarray(candidate, dtype=np.float32)
    require(ref.shape == got.shape and np.isfinite(ref).all() and np.isfinite(got).all(),
            "invalid smoke comparison")
    diff = np.abs(ref - got)
    scale = np.maximum(np.sqrt(np.mean(ref ** 2, axis=(0, 2))), 1e-6)
    maxima, means = diff.max(axis=(0, 2)), diff.mean(axis=(0, 2))
    max_limits, mean_limits = 1e-6 + max_tol * scale, 1e-7 + mean_tol * scale
    return {"max_absolute_difference": float(diff.max()), "mean_absolute_difference": float(diff.mean()),
            "max_rms_fraction_tolerance": max_tol, "mean_rms_fraction_tolerance": mean_tol,
            "absolute_floors": {"max": 1e-6, "mean": 1e-7},
            "passed": bool(np.all(maxima <= max_limits) and np.all(means <= mean_limits)),
            "per_layer": [{"saved_layer": i, "reference_rms": float(scale[i]),
                           "max_absolute_difference": float(maxima[i]), "mean_absolute_difference": float(means[i]),
                           "max_limit": float(max_limits[i]), "mean_limit": float(mean_limits[i])}
                          for i in range(ref.shape[1])],
            "per_statement": [{"statement_index": i, "max_absolute_difference": float(d.max()),
                               "mean_absolute_difference": float(d.mean())} for i, d in enumerate(diff)]}


def smoke_compare(model, tokenizer, statements=SMOKE_STATEMENTS, *, batch_size=None):
    statements = list(statements)
    batch_size = len(statements) if batch_size is None else batch_size
    require(batch_size >= 2, "padding smoke requires batch size >= 2")
    lengths = [len(tokenizer(s, add_special_tokens=True)["input_ids"]) for s in statements]
    require(len(set(lengths)) > 1, "smoke statements must exercise actual padding")
    old_side = tokenizer.padding_side
    reference = torch.cat([extract_batch(model, tokenizer, [s], padding=False, explicit_positions=False,
                                         use_cache=model.config.use_cache)
                           for s in statements]).numpy()
    report = {"statements": statements, "token_lengths": lengths,
              "padding_batch_size": batch_size,
            "reference": "legacy unpadded batch_size=1 without explicit position_ids; model default use_cache",
              "compute_dtype": str(model.dtype), "device": str(next(model.parameters()).device),
              "comparisons": {}}
    try:
        clean_single = torch.cat([extract_batch(model, tokenizer, [s], padding=False) for s in statements]).numpy()
        candidates = {"single_explicit_positions": clean_single}
        for side in ("right", "left"):
            tokenizer.padding_side = side
            candidates[side] = np.concatenate([
                extract_batch(model, tokenizer, statements[start:start + batch_size], padding=True).numpy()
                for start in range(0, len(statements), batch_size)])
        for name, values in candidates.items():
            report["comparisons"][name] = {
                "compute_readout": difference_report(reference, values, model.dtype),
                "saved_float16": difference_report(reference.astype(np.float16), values.astype(np.float16),
                                                   torch.bfloat16 if model.dtype == torch.bfloat16 else torch.float16)}
    finally:
        tokenizer.padding_side = old_side
    report["passed"] = all(r[stage]["passed"] for r in report["comparisons"].values()
                           for stage in ("compute_readout", "saved_float16"))
    return report


def extract_clean(model, tokenizer, benchmark, manifest, output, *, resume=False):
    """One model load, sequential extraction, checkpoint after every batch."""
    from src.atomic_activation_compatibility import require_both_gates
    require_both_gates(manifest)
    batch_size = manifest["numerics"]["batch_size"]
    with ExtractionWriter(output, benchmark, manifest, resume=resume) as writer:
        for start in range(writer.next_row, len(benchmark.frame), batch_size):
            rows = benchmark.frame.iloc[start:start + batch_size].copy()
            benchmark.verify_rows(start, rows)
            values = extract_batch(model, tokenizer, rows.statement.tolist(), padding=batch_size > 1)
            writer.append(start, rows, values.numpy().astype(np.float16))
            print(f"Committed {writer.next_row}/{len(benchmark.frame)} rows", flush=True)
        writer.finalize()
