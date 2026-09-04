#!/usr/bin/env python3
"""Decompose the compound-probe weight into atomic-parallel and orthogonal parts."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ATOMIC_PATH = ROOT / "results" / "qwen_union_probe_gate" / "selected_probe.npz"
COMPOUND_PATH = ROOT / "results" / "r1" / "compound_probe_control_probe.npz"
TASK1_PATH = ROOT / "results" / "r1" / "mechanism" / "probe_direction_similarity.json"
OUTPUT_NPZ = ROOT / "results" / "r1" / "mechanism" / "probe_direction_decomposition.npz"
OUTPUT_JSON = ROOT / "results" / "r1" / "mechanism" / "probe_direction_decomposition.json"
EXPECTED_DIMENSION = 3584
ABS_TOLERANCE = 1e-12


def main() -> None:
    atomic_archive = np.load(ATOMIC_PATH, allow_pickle=False)
    compound_archive = np.load(COMPOUND_PATH, allow_pickle=False)
    with TASK1_PATH.open() as handle:
        task1 = json.load(handle)

    w_atomic = np.asarray(atomic_archive["coef"], dtype=float).reshape(-1)
    w_compound = np.asarray(compound_archive["coef"], dtype=float).reshape(-1)
    if w_atomic.shape != w_compound.shape or w_atomic.size != EXPECTED_DIMENSION:
        raise ValueError("Probe coefficient dimensions do not match the expected 3,584")
    if not np.isfinite(w_atomic).all() or not np.isfinite(w_compound).all():
        raise ValueError("Probe coefficient arrays contain NaN or infinity")
    if bool(task1["sign_flip_required"]) or not bool(task1["sanity_checks_passed"]):
        raise ValueError("Task 1 orientation/sanity checks did not pass")

    atomic_norm = float(np.linalg.norm(w_atomic))
    compound_norm = float(np.linalg.norm(w_compound))
    if atomic_norm == 0 or compound_norm == 0:
        raise ValueError("Cannot decompose a zero-norm probe coefficient vector")

    wa_hat = w_atomic / atomic_norm
    alpha = float(w_compound @ wa_hat)
    w_parallel = alpha * wa_hat
    w_perp = w_compound - w_parallel

    parallel_norm = float(np.linalg.norm(w_parallel))
    perp_norm = float(np.linalg.norm(w_perp))
    compound_squared_norm = compound_norm**2
    fraction_squared_norm_parallel = parallel_norm**2 / compound_squared_norm
    fraction_squared_norm_perp = perp_norm**2 / compound_squared_norm
    reconstruction_max_abs_error = float(
        np.max(np.abs((w_parallel + w_perp) - w_compound))
    )
    dot_perp_atomic = float(w_perp @ w_atomic)
    dot_perp_atomic_unit = float(w_perp @ wa_hat)
    norm_decomposition_error = float(
        compound_squared_norm - (parallel_norm**2 + perp_norm**2)
    )
    task1_cosine = float(task1["cosine_similarity"])
    cosine_squared = task1_cosine**2
    cosine_squared_crosscheck_error = float(
        fraction_squared_norm_parallel - cosine_squared
    )

    checks = {
        "reconstruction": reconstruction_max_abs_error <= ABS_TOLERANCE,
        "perp_orthogonal_to_atomic": abs(dot_perp_atomic) <= ABS_TOLERANCE,
        "perp_orthogonal_to_atomic_unit": abs(dot_perp_atomic_unit) <= ABS_TOLERANCE,
        "pythagorean_norm_decomposition": abs(norm_decomposition_error) <= ABS_TOLERANCE,
        "parallel_fraction_matches_cosine_squared": abs(cosine_squared_crosscheck_error)
        <= ABS_TOLERANCE,
        "squared_norm_fractions_sum_to_one": abs(
            fraction_squared_norm_parallel + fraction_squared_norm_perp - 1.0
        )
        <= ABS_TOLERANCE,
    }
    passed = all(checks.values())
    diagnostics = {
        "status": "PASS" if passed else "FAIL",
        "layer": int(atomic_archive["layer"]),
        "dimension": int(w_atomic.size),
        "alpha": alpha,
        "atomic_norm": atomic_norm,
        "compound_norm": compound_norm,
        "parallel_norm": parallel_norm,
        "perp_norm": perp_norm,
        "fraction_squared_norm_parallel": fraction_squared_norm_parallel,
        "fraction_squared_norm_perp": fraction_squared_norm_perp,
        "task1_cosine_similarity": task1_cosine,
        "cosine_similarity_squared": cosine_squared,
        "cosine_squared_crosscheck_error": cosine_squared_crosscheck_error,
        "reconstruction_max_abs_error": reconstruction_max_abs_error,
        "dot_perp_atomic": dot_perp_atomic,
        "dot_perp_atomic_unit": dot_perp_atomic_unit,
        "norm_decomposition_error": norm_decomposition_error,
        "absolute_tolerance": ABS_TOLERANCE,
        "checks": checks,
        "interpretation_scope": "weight-vector geometry only",
        "sources": {
            "atomic_probe": str(ATOMIC_PATH.relative_to(ROOT)),
            "compound_probe": str(COMPOUND_PATH.relative_to(ROOT)),
            "task1_similarity": str(TASK1_PATH.relative_to(ROOT)),
        },
    }

    OUTPUT_NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_NPZ,
        w_atomic=w_atomic,
        w_compound=w_compound,
        wa_hat=wa_hat,
        w_parallel=w_parallel,
        w_perp=w_perp,
        alpha=np.asarray(alpha),
    )
    with OUTPUT_JSON.open("w") as handle:
        json.dump(diagnostics, handle, indent=2)
        handle.write("\n")
    print(json.dumps(diagnostics, indent=2))
    if not passed:
        raise RuntimeError("Probe direction decomposition validation failed")


if __name__ == "__main__":
    main()
