#!/usr/bin/env python3
"""Plan (default), tiny full-weight smoke, or explicitly requested extraction.

Only the fixed development/validation benchmark is reachable through this CLI.
The extract mode runs smoke checks before creating any activation array.
"""
import argparse
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.hub import cached_file, extract_commit_hash

from src.clean_extraction import (
    BENCHMARK, MODELS, OUTPUT_ROOT, ROOT, Benchmark, build_manifest,
    canonical_json, extract_clean, file_hash, layer_convention, require, resume_identity, smoke_compare,
)
from src.atomic_activation_compatibility import (
    historical_compatibility_smoke, load_compatibility_pin, require_both_gates, save_compatibility_pin,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--mode", choices=("plan", "smoke", "extract"), default="plan")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--padding-side", choices=("right", "left"), default="right")
    parser.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--revision", help="Immutable HF commit SHA for smoke; extraction must match the passing pin")
    parser.add_argument("--compatibility-pin", type=Path,
                        help="Passing revision receipt; default: results/.../compatibility_pins/MODEL.json")
    args = parser.parse_args()
    require(args.batch_size > 0, "batch size must be positive")
    require(not args.resume or args.mode == "extract", "--resume requires --mode extract")
    spec = MODELS[args.model]
    report_dir = ROOT / "results/clean_protocol/extraction_diagnostics"
    pin_path = args.compatibility_pin or report_dir / "compatibility_pins" / f"{args.model}.json"
    receipt = None
    if args.mode == "extract":
        # Load evidence before downloading a model or touching activation outputs.
        receipt, _ = load_compatibility_pin(pin_path, spec["identifier"])
        revision, tokenizer_revision = receipt["model_commit_sha"], receipt["tokenizer_commit_sha"]
        require(args.revision is None or args.revision == revision, "requested revision differs from passing compatibility pin")
    else:
        revision = tokenizer_revision = args.revision or spec["revision"]
    import re
    require(re.fullmatch(r"[0-9a-f]{40}", revision) is not None, "--revision must be an immutable 40-character HF commit SHA")
    benchmark = Benchmark(ROOT / BENCHMARK)
    require(len(benchmark.frame) == 8384, "expected the frozen 8,384-row benchmark")
    output = ROOT / OUTPUT_ROOT / args.model
    # Resolve symlinks too: never redirect clean output to exploratory paths.
    require(output.resolve() == output.absolute(), "clean output path must not contain symlinks")
    shape = [len(benchmark.frame), spec["layers"], spec["hidden_size"]]
    if args.mode == "plan":
        payload_bytes = shape[0] * shape[1] * shape[2] * 2
        print(json.dumps({"model": {**spec, "revision": revision}, "benchmark": str(BENCHMARK), **benchmark.hashes,
                          "output": str(OUTPUT_ROOT / args.model), "shape": shape,
                          "float16_payload_bytes": payload_bytes,
                          "float16_payload_gib": payload_bytes / 1024 ** 3,
                          "convention": layer_convention(spec["layers"]),
                          "model_loaded": False, "activation_arrays_created": False}, indent=2))
        return
    require(transformers.__version__ == "5.12.1", "use audited transformers==5.12.1")
    require(torch.__version__.split("+")[0] == "2.11.0", "use audited torch==2.11.0")
    if args.device.startswith("cuda"):
        require(torch.cuda.is_available(), "CUDA unavailable; use a GPU host for full-weight smoke/extraction")
    tokenizer_config = cached_file(spec["identifier"], "tokenizer_config.json", revision=tokenizer_revision)
    tokenizer_commit = extract_commit_hash(tokenizer_config, None)
    require(tokenizer_commit == tokenizer_revision, "resolved tokenizer snapshot revision mismatch")
    tokenizer = AutoTokenizer.from_pretrained(spec["identifier"], revision=tokenizer_commit)
    tokenizer._clean_resolved_commit_hash = tokenizer_commit
    tokenizer._clean_tokenizer_config_sha256 = file_hash(tokenizer_config)
    require(tokenizer.pad_token_id is not None, "expected the original tokenizer's pad token")
    tokenizer.padding_side = args.padding_side
    model = AutoModelForCausalLM.from_pretrained(
        spec["identifier"], revision=revision, dtype=getattr(torch, args.dtype),
        attn_implementation=args.attention).to(args.device).eval()
    manifest = build_manifest(benchmark, model, tokenizer, args.model, revision, args.batch_size)
    manifest["tokenizer"]["requested_revision"] = tokenizer_commit
    smoke = smoke_compare(model, tokenizer)
    manifest["smoke_test"] = smoke
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    report_path = report_dir / f"{args.model}_{stamp}.json"
    try:
        require(smoke["passed"], "full-weight padding smoke failed; no historical gate or extraction attempted")
        historical = historical_compatibility_smoke(model, tokenizer, args.model, manifest)
        manifest["historical_atomic_compatibility"] = historical
        manifest["historical_sample_sha256"] = historical["sample_sha256"]
        manifest["historical_compatibility_policy"] = historical["policy"]
    except Exception as exc:
        # Preserve failed-gate diagnostics too; do not mint a passing pin.
        manifest["compatibility_error"] = {"type": type(exc).__name__, "message": str(exc)}
        with report_path.open("xb") as f:
            f.write(canonical_json(manifest) + b"\n")
        print(f"Failed smoke report: {report_path}", flush=True)
        raise
    with report_path.open("xb") as f:
        f.write(canonical_json(manifest) + b"\n")
    print(f"Smoke report: {report_path.relative_to(ROOT)}; padding={smoke['passed']}; "
          f"historical={historical['passed']}; float16_byte_identical={historical['float16_byte_identical']}", flush=True)
    print(json.dumps({"before_float16": {k: historical["fresh_before_float16_conversion"][k]
                                         for k in ("max_absolute_difference", "mean_absolute_difference")},
                      "after_float16": {k: historical["after_float16_conversion"][k]
                                        for k in ("max_absolute_difference", "mean_absolute_difference")}}), flush=True)
    require_both_gates(manifest)
    if args.mode == "extract":
        require(resume_identity(manifest) == receipt["extraction_identity_sha256"],
                "extraction inputs/configuration differ from passing compatibility pin")
        extract_clean(model, tokenizer, benchmark, manifest, output, resume=args.resume)
    else:
        save_compatibility_pin(pin_path, report_path, manifest)
        print(f"Passing revision pin: {pin_path}", flush=True)


if __name__ == "__main__":
    main()
