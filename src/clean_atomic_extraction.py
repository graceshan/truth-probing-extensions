"""Pinned Qwen2.5 repair using ONLY the previously exported allowed atomic rows.

No mixed-source, entity-manifest, historical-activation or probe loader belongs
in this module. There is no test split, historical compatibility gate or resume.
"""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import inspect
import io
import json
from pathlib import Path
import platform
import re
import subprocess
import uuid


ROOT = Path(__file__).resolve().parents[1]
INPUT = Path("results/clean_protocol/atomic_method_suite_v1/qwen25_7b/allowed_atomic_rows.csv")
INPUT_DIGEST = "1faf186cb28b48311e85fd41921316fad9d06b3d28c9756aa5e8b381a950f1d6"
COUNTS = {"train": 3144, "validation": 1040}
FIELDS = ("dataset", "row_index", "statement", "entity_id", "topic", "form", "split", "label")
TOPICS = ("cities", "sp_en_trans", "inventors", "element_symb", "animal_class")
MODEL = "Qwen/Qwen2.5-7B-Instruct"
REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
LAYERS, WIDTH = 28, 3584
OUTPUT_PARENT = Path("acts/clean_protocol/atomic")
OUTPUT = OUTPUT_PARENT / "qwen25_a09a354_bs1_bf16_v1"
REPORTS = Path("results/clean_protocol/atomic_repair_smoke")
SOURCES = ("src/clean_atomic_extraction.py", "src/extract.py",
           "scripts/33_extract_clean_atomic_activations.py")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    # Match the original allowed_rows_sha256 encoding, including insertion order.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plain_path(root, relative):
    """Reject symlinks, including ancestor symlinks, before any file is opened."""
    root = Path(root).resolve()
    path = root / relative
    require(path.is_absolute() and path.is_relative_to(root), "path outside repository")
    require(path.resolve() == path.absolute(), "symlink or noncanonical path forbidden")
    return path


def load_allowed_rows(root=ROOT):
    """One data-file read; no lookup or provenance reconstruction from old sources."""
    path = plain_path(root, INPUT)
    payload = path.read_bytes()
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""))
    require(tuple(reader.fieldnames or ()) == FIELDS, "unexpected allowed-row schema")
    rows, identities = [], set()
    entities = {split: set() for split in COUNTS}
    for row in reader:
        require(None not in row and all(row.get(k) is not None for k in FIELDS), "malformed row")
        # Reject before any row enters an allowed partition or digest.
        require(row["split"] in COUNTS, "only train/validation splits are permitted")
        require(row["topic"] in TOPICS and row["form"] in {"affirmative", "negated"}, "invalid topic/form")
        expected_dataset = ("neg_" if row["form"] == "negated" else "") + row["topic"]
        require(row["dataset"] == expected_dataset, "dataset/topic/form mismatch")
        require(row["row_index"].isdigit() and row["label"] in {"0", "1"}, "invalid row index/label")
        require(bool(row["statement"].strip()) and bool(row["entity_id"]), "empty statement/entity")
        row["row_index"], row["label"] = int(row["row_index"]), int(row["label"])
        identity = (row["dataset"], row["row_index"])
        require(identity not in identities, "duplicate dataset/row_index")
        identities.add(identity)
        entities[row["split"]].add(row["entity_id"])
        rows.append(row)
    partitions = {split: [r for r in rows if r["split"] == split] for split in COUNTS}
    require({s: len(r) for s, r in partitions.items()} == COUNTS, "allowed-row counts mismatch")
    require(entities["train"].isdisjoint(entities["validation"]), "train/validation entity overlap")
    record_digest = digest(canonical(rows))
    require(record_digest == INPUT_DIGEST, "approved allowed-row digest mismatch")
    return partitions, {"path": str(INPUT), "file_sha256": digest(payload),
                        "allowed_rows_sha256": record_digest, "counts": dict(COUNTS),
                        "columns": list(FIELDS),
                        "digest_encoding": "ordered records; integer row_index/label; UTF-8 compact JSON; no key sorting",
                        "partition_rows_sha256": {s: digest(canonical(r)) for s, r in partitions.items()}}


def output_path(root=ROOT, output=OUTPUT):
    output = Path(output)
    require(output.parent == OUTPUT_PARENT and
            re.fullmatch(r"qwen25_a09a354_bs1_bf16_v[1-9][0-9]*", output.name) is not None,
            "output must be a versioned Qwen2.5 directory under acts/clean_protocol/atomic")
    path = plain_path(root, output)
    require(not path.exists(), "refusing overwrite: output already exists (including partial runs)")
    return path


def contract():
    return {"model": MODEL, "model_revision": REVISION, "tokenizer_revision": REVISION,
            "torch": "2.11.0", "transformers": "5.12.1", "device": "cuda",
            "compute_dtype": "bfloat16", "attention": "sdpa", "batch_size": 1,
            "padding": False, "input": "exact raw statement", "add_special_tokens": True,
            "chat_template": False, "truncation": False, "explicit_semantic_position_ids": True,
            "use_cache": False, "logits_to_keep": 1, "output_hidden_states": True,
            "saved_dtype": "float16", "readout": "last real token",
            "hidden_states": "HF hidden_states[1:]; embedding excluded",
            "saved_layer_to_hf_index": list(range(1, LAYERS + 1)),
            "final_saved_layer": "post-final-RMSNorm", "selected_layer": None, "selected_C": None}


def require_runtime():
    import torch
    import transformers
    require(torch.__version__.split("+")[0] == "2.11.0", "requires torch==2.11.0")
    require(transformers.__version__ == "5.12.1", "requires transformers==5.12.1")
    require(torch.cuda.is_available(), "CUDA required")
    require(torch.cuda.is_bf16_supported(), "CUDA BF16 support required")


def load_pinned_model():
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.utils.hub import cached_file, extract_commit_hash

    require_runtime()
    config_path = cached_file(MODEL, "config.json", revision=REVISION)
    tokenizer_path = cached_file(MODEL, "tokenizer_config.json", revision=REVISION)
    require(extract_commit_hash(config_path, None) == REVISION, "resolved model revision mismatch")
    require(extract_commit_hash(tokenizer_path, None) == REVISION, "resolved tokenizer revision mismatch")
    config = AutoConfig.from_pretrained(MODEL, revision=REVISION)
    require(config._commit_hash == REVISION, "config revision mismatch")
    require((config.model_type, config.num_hidden_layers, config.hidden_size) == ("qwen2", LAYERS, WIDTH),
            "model architecture mismatch")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, config=config, dtype=torch.bfloat16, attn_implementation="sdpa")
    require(model.config._commit_hash == REVISION, "loaded model revision mismatch")
    model = model.to("cuda").eval()
    require(model.dtype == torch.bfloat16 and model.config._attn_implementation == "sdpa",
            "loaded compute convention mismatch")
    provenance = {"model_resolved_revision": model.config._commit_hash,
                  "tokenizer_resolved_revision": REVISION,
                  "config_file_sha256": file_hash(config_path),
                  "tokenizer_config_file_sha256": file_hash(tokenizer_path),
                  "model_config": model.config.to_dict(),
                  "model_config_sha256": digest(canonical(model.config.to_dict())),
                  "tokenizer_backend_sha256": digest(tokenizer.backend_tokenizer.to_str().encode()),
                  "tokenizer_class": type(tokenizer).__name__,
                  "model_implementation_sha256": file_hash(inspect.getfile(type(model)))}
    return model, tokenizer, provenance


def readout(model, tokenizer, statement):
    """Exactly one real sequence; reject even unexpected tokenizer-added padding."""
    import torch
    from src.extract import last_real_token_indices, saved_hidden_states, semantic_position_ids

    require(not model.training, "model must be in eval mode")
    encoded = tokenizer([statement], return_tensors="pt", padding=False, truncation=False,
                        add_special_tokens=True, return_attention_mask=True).to(next(model.parameters()).device)
    mask = encoded["attention_mask"]
    require(mask.ndim == 2 and mask.shape[0] == 1 and mask.shape[1] > 0 and bool((mask == 1).all()),
            "only one unpadded sequence is permitted")
    require(encoded["input_ids"].shape == mask.shape, "token/mask shape mismatch")
    require(mask.shape[1] <= model.config.max_position_embeddings, "input exceeds context; no truncation")
    encoded["position_ids"] = semantic_position_ids(mask)
    with torch.inference_mode():
        out = model(**encoded, output_hidden_states=True, use_cache=False, logits_to_keep=1)
        states = saved_hidden_states(out.hidden_states, LAYERS)
        last = last_real_token_indices(mask).item()
        values = torch.stack([state[0, last] for state in states]).float().cpu()
    require(tuple(values.shape) == (LAYERS, WIDTH) and bool(torch.isfinite(values).all()),
            "invalid activation shape/values")
    return values


def reference_readout(model, tokenizer, statement):
    """Independent full-logit direct forward, same unpadded numerical convention."""
    import torch
    encoded = tokenizer([statement], return_tensors="pt", padding=False, truncation=False,
                        add_special_tokens=True, return_attention_mask=True).to(next(model.parameters()).device)
    require(encoded["input_ids"].shape[0] == 1 and bool((encoded["attention_mask"] == 1).all()),
            "reference must be unpadded")
    encoded["position_ids"] = torch.arange(encoded["input_ids"].shape[1],
                                           device=encoded["input_ids"].device).unsqueeze(0)
    with torch.inference_mode():
        out = model(**encoded, output_hidden_states=True, use_cache=False, logits_to_keep=0)
        require(len(out.hidden_states) == LAYERS + 1, "reference hidden-state count changed")
        values = torch.stack([state[0, -1] for state in out.hidden_states[1:]]).float().cpu()
    require(tuple(values.shape) == (LAYERS, WIDTH) and bool(torch.isfinite(values).all()),
            "invalid reference activation shape/values")
    return values


def smoke(model, tokenizer, partitions):
    """Ten deterministic permitted rows, three unpadded forwards per row."""
    import numpy as np
    import torch
    samples = []
    for topic in TOPICS:
        for split, form in (("train", "affirmative"), ("validation", "negated")):
            candidates = [r for r in partitions[split] if r["topic"] == topic and r["form"] == form]
            require(bool(candidates), "missing smoke stratum")
            samples.append(min(candidates, key=lambda r: digest(canonical(
                [r["dataset"], r["row_index"], r["statement"]]))))
    comparisons = []
    for row in samples:
        first = readout(model, tokenizer, row["statement"])
        repeat = readout(model, tokenizer, row["statement"])
        reference = reference_readout(model, tokenizer, row["statement"])
        checks = {}
        for name, candidate in (("repeat", repeat), ("direct_full_logits", reference)):
            delta = (first - candidate).abs()
            first_saved, candidate_saved = first.numpy().astype(np.float16), candidate.numpy().astype(np.float16)
            finite = bool(np.isfinite(first_saved).all() and np.isfinite(candidate_saved).all())
            checks[name] = {"compute_equal": torch.equal(first, candidate),
                            "max_absolute_difference": float(delta.max()),
                            "mean_absolute_difference": float(delta.mean()),
                            "saved_float16_byte_equal": first_saved.tobytes() == candidate_saved.tobytes(),
                            "saved_finite": finite}
        buffer = io.BytesIO()
        np.save(buffer, first.numpy().astype(np.float16), allow_pickle=False)
        restored = np.load(io.BytesIO(buffer.getvalue()), allow_pickle=False)
        comparisons.append({"dataset": row["dataset"], "row_index": row["row_index"],
                            "split": row["split"], "statement_sha256": digest(row["statement"].encode()),
                            "checks": checks, "storage_roundtrip_equal":
                            restored.tobytes() == first.numpy().astype(np.float16).tobytes()})
    passed = all(r["storage_roundtrip_equal"] and all(
        c["compute_equal"] and c["saved_float16_byte_equal"] and c["saved_finite"]
        for c in r["checks"].values()) for r in comparisons)
    return {"policy": "unpadded-single-repeat-and-independent-forward-exact-v1", "passed": passed,
            "batch_size": 1, "padding": False, "forward_calls": 3 * len(samples),
            "sample_rows_sha256": digest(canonical(samples)), "comparisons": comparisons,
            "scope": "same loaded model/runtime; no historical equivalence or cross-runtime determinism claim"}


def provenance(root, input_info, model_info):
    import numpy as np
    import torch
    import transformers
    import transformers.utils.output_capturing as capturing

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, text=True, capture_output=True)
    return {"schema_version": 1, "implementation": "qwen25-pinned-atomic-repair-v1",
            "created_utc": datetime.now(timezone.utc).isoformat(), "input": input_info,
            "activation_shapes": {s: [n, LAYERS, WIDTH] for s, n in COUNTS.items()},
            "contract": contract(), "resolved_model": model_info,
            "source_sha256": {p: file_hash(plain_path(root, p)) for p in SOURCES},
            "git_head_context_only": head.stdout.strip() if head.returncode == 0 else None,
            "hf_output_capture_sha256": file_hash(inspect.getfile(capturing)),
            "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                        "torch": torch.__version__, "transformers": transformers.__version__,
                        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                        "gpu": torch.cuda.get_device_name(),
                        "float32_matmul_precision": torch.get_float32_matmul_precision(),
                        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                        "sdpa_flash_enabled": torch.backends.cuda.flash_sdp_enabled(),
                        "sdpa_mem_efficient_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
                        "sdpa_math_enabled": torch.backends.cuda.math_sdp_enabled()},
            "data_access": {"only_data_input": str(INPUT), "historical_activations_opened": False,
                            "mixed_sources_opened": False, "entity_manifest_opened": False,
                            "test_artifacts_opened": False},
            "probe_selection": "not performed; no historical layer or C reused"}


def write_new_json(path, value):
    with Path(path).open("xb") as handle:
        handle.write(canonical(value) + b"\n")


def write_arrays(path, partitions, model, tokenizer, manifest):
    """Exclusive directory reservation; incomplete runs are never reusable outputs."""
    import numpy as np
    import os
    require(manifest["smoke"]["passed"], "passing unpadded smoke required")
    # Recheck canonical path immediately before reserving the directory.
    require(path.resolve() == path.absolute(), "symlink output forbidden")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    require(set(partitions) == set(COUNTS) and
            all(len(partitions[s]) == n and all(r["split"] == s for r in partitions[s])
                for s, n in COUNTS.items()), "writer accepts only complete train/validation partitions")
    write_new_json(path / "extraction_manifest.json", manifest)
    outputs = {}
    for split in COUNTS:
        rows = partitions[split]
        directory = path / split
        directory.mkdir()
        metadata = directory / "metadata.csv"
        with metadata.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        partial = directory / "activations.partial.npy"
        # File-object creation is exclusive; np.save cannot overwrite another run.
        shape = (len(rows), LAYERS, WIDTH)
        with partial.open("xb") as handle:
            np.lib.format.write_array_header_2_0(handle, {
                "descr": np.dtype(np.float16).str, "fortran_order": False, "shape": shape})
            for index, row in enumerate(rows):
                values = readout(model, tokenizer, row["statement"]).numpy().astype(np.float16)
                require(bool(np.isfinite(values).all()), "nonfinite float16 storage values")
                handle.write(values.tobytes(order="C"))
                if (index + 1) % 100 == 0:
                    print(f"{split}: {index + 1}/{len(rows)}", flush=True)
            handle.flush()
            os.fsync(handle.fileno())
        array = np.load(partial, mmap_mode="r", allow_pickle=False)
        require(array.shape == shape and array.dtype == np.float16, "written array header mismatch")
        del array
        target = directory / "activations.npy"
        # Hard-link publication fails if target exists; never replacement-rename.
        os.link(partial, target)
        partial.unlink()
        for file in (metadata, target):
            outputs[str(file.relative_to(path))] = {"sha256": file_hash(file), "bytes": file.stat().st_size}
    write_new_json(path / "completion.json", {"complete": True, "counts": dict(COUNTS),
                   "manifest_sha256": file_hash(path / "extraction_manifest.json"), "outputs": outputs})


def run(mode="plan", *, root=ROOT, output=OUTPUT):
    require(mode in {"plan", "smoke", "extract"}, "unknown mode")
    partitions, input_info = load_allowed_rows(root)
    path = output_path(root, output)
    plan = {"mode": mode, "input": input_info, "contract": contract(), "output": str(output),
            "shapes": {s: [len(r), LAYERS, WIDTH] for s, r in partitions.items()},
            "model_loaded": False, "activation_arrays_created": False}
    if mode == "plan":
        return plan
    model, tokenizer, model_info = load_pinned_model()
    manifest = provenance(root, input_info, model_info)
    reports = plain_path(root, REPORTS)
    reports.mkdir(parents=True, exist_ok=True)
    report = reports / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "_" + uuid.uuid4().hex + ".json")
    try:
        manifest["smoke"] = smoke(model, tokenizer, partitions)
    except Exception as exc:
        manifest["smoke"] = {"passed": False, "error_type": type(exc).__name__, "error": str(exc)}
        write_new_json(report, manifest)
        print(f"Failed smoke report: {report}", flush=True)
        raise
    write_new_json(report, manifest)
    print(f"Smoke report: {report}", flush=True)
    require(manifest["smoke"]["passed"], "unpadded smoke failed; no activation arrays created")
    if mode == "extract":
        manifest["smoke_report"] = {"path": str(report.relative_to(Path(root).resolve())),
                                    "sha256": file_hash(report)}
        write_arrays(path, partitions, model, tokenizer, manifest)
    return {**plan, "model_loaded": True, "activation_arrays_created": mode == "extract",
            "smoke_report": str(report), "smoke_passed": True}
