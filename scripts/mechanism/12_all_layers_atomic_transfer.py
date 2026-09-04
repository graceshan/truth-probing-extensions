#!/usr/bin/env python3
"""Evaluate fixed-procedure atomic-probe transfer to compounds across all layers."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.probes import train_layer_probe


ROOT = REPO_ROOT
MECHANISM_DIR = ROOT / "results" / "r1" / "mechanism"
GATE_DIR = ROOT / "results" / "qwen_union_probe_gate"
GATE_SCRIPT = ROOT / "scripts" / "02_qwen_union_probe_gate.py"
TASK2_SCRIPT = Path(__file__).with_name("02_atomic_pairwise_auc.py")
TASK5_SCRIPT = Path(__file__).with_name("05_boundary_dprime.py")
ATOMIC_METRICS_PATH = GATE_DIR / "per_layer_metrics.csv"
ATOMIC_SPLIT_PATH = GATE_DIR / "split_metadata.csv"
SELECTED_PROBE_PATH = GATE_DIR / "selected_probe.npz"
COMPOUND_ACTS_PATH = ROOT / "acts" / "r1_quadruples.npy"
COMPOUND_SIDECAR_PATH = ROOT / "acts" / "r1_quadruples.csv"
R1_SCORE_TABLE_PATH = ROOT / "results" / "r1" / "r1_score_table.csv"
PAIR_SPLIT_PATH = ROOT / "results" / "r1" / "g_c_cg" / "pair_split.csv"
PASS1_GEOMETRY_PATH = MECHANISM_DIR / "boundary_geometry.csv"
PASS1_PAIRWISE_PATH = MECHANISM_DIR / "pairwise_auc_atomic.csv"
PASS1_DECOMPOSITION_PATH = MECHANISM_DIR / "atomic_auroc_decomposition.json"
OUTPUT_CSV = MECHANISM_DIR / "all_layers_atomic_transfer.csv"
OUTPUT_JSON = MECHANISM_DIR / "all_layers_atomic_transfer_summary.json"
N_LAYERS = 28
DIMENSION = 3584
PRIMARY_LAYER = 22
TOLERANCE = 1e-12


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_connective_metrics(
    row: dict[str, float | int], connective: str, frame: pd.DataFrame,
    grouped_pairwise_auc, dprime,
) -> None:
    prefix = connective.lower()
    cells = {
        cell: frame.loc[frame["cell"].eq(cell), "score"].to_numpy(dtype=float)
        for cell in ("TT", "TF", "FT", "FF")
    }
    mixed = np.concatenate([cells["TF"], cells["FT"]])
    if any(len(cells[cell]) != 200 for cell in cells) or len(mixed) != 400:
        raise ValueError(f"Unexpected {connective} cell sizes")

    comparisons = {
        "TT_vs_mixed": (("TT",), ("TF", "FT")),
        "mixed_vs_FF": (("TF", "FT"), ("FF",)),
        "TT_vs_FF": (("TT",), ("FF",)),
        "TF_vs_FT": (("TF",), ("FT",)),
    }
    for name, (high_cells, low_cells) in comparisons.items():
        _, _, auc = grouped_pairwise_auc(frame, "score", high_cells, low_cells)
        row[f"{prefix}_{name}_auroc"] = auc

    mu_tt = float(np.mean(cells["TT"]))
    mu_mixed = float(np.mean(mixed))
    mu_ff = float(np.mean(cells["FF"]))
    sd_tt = float(np.std(cells["TT"], ddof=1))
    sd_mixed = float(np.std(mixed, ddof=1))
    sd_ff = float(np.std(cells["FF"], ddof=1))
    upper_gap = mu_tt - mu_mixed
    lower_gap = mu_mixed - mu_ff
    total_range = mu_tt - mu_ff
    if total_range == 0:
        raise ValueError(f"Zero total score range for layer {row['layer']} {connective}")
    pooled_sd = float(np.sqrt(
        ((len(cells["TT"]) - 1) * sd_tt**2
         + (len(mixed) - 1) * sd_mixed**2
         + (len(cells["FF"]) - 1) * sd_ff**2)
        / (len(cells["TT"]) + len(mixed) + len(cells["FF"]) - 3)
    ))
    row.update({
        f"{prefix}_dprime_TT_mixed": dprime(cells["TT"], mixed),
        f"{prefix}_dprime_mixed_FF": dprime(mixed, cells["FF"]),
        f"{prefix}_mu_TT": mu_tt,
        f"{prefix}_mu_mixed": mu_mixed,
        f"{prefix}_mu_FF": mu_ff,
        f"{prefix}_upper_gap": upper_gap,
        f"{prefix}_lower_gap": lower_gap,
        f"{prefix}_total_range": total_range,
        f"{prefix}_upper_gap_fraction": upper_gap / total_range,
        f"{prefix}_lower_gap_fraction": lower_gap / total_range,
        f"{prefix}_Delta_norm": (upper_gap - lower_gap) / total_range,
        f"{prefix}_pooled_sd": pooled_sd,
        f"{prefix}_range_over_sd": total_range / pooled_sd,
    })


def main() -> None:
    gate = load_module(GATE_SCRIPT, "qwen_union_probe_gate")
    grouped_pairwise_auc = load_module(TASK2_SCRIPT, "task2_pairwise").grouped_pairwise_auc
    dprime = load_module(TASK5_SCRIPT, "task5_dprime").dprime

    blocks, labels, topics, forms, atomic_train_idx, atomic_test_idx, split_df = gate.build_data(
        str(ROOT / "acts"), str(ROOT / "data" / "tiu_datasets")
    )
    saved_atomic_split = pd.read_csv(ATOMIC_SPLIT_PATH)
    if not split_df.equals(saved_atomic_split):
        raise ValueError("Reconstructed atomic entity split differs from the saved gate split")
    if any(block["acts"].shape[1:] != (N_LAYERS, DIMENSION) for block in blocks):
        raise ValueError("Atomic activation shapes are not uniformly [rows, 28, 3584]")
    atomic_train_entities = set(
        split_df.loc[split_df["partition"].eq("train"), ["topic", "entity"]]
        .itertuples(index=False, name=None)
    )
    atomic_test_entities = set(
        split_df.loc[split_df["partition"].eq("test"), ["topic", "entity"]]
        .itertuples(index=False, name=None)
    )
    if atomic_train_entities & atomic_test_entities:
        raise ValueError("Atomic entity leakage detected")

    r1 = pd.read_csv(R1_SCORE_TABLE_PATH)
    sidecar = pd.read_csv(COMPOUND_SIDECAR_PATH)
    pair_split = pd.read_csv(PAIR_SPLIT_PATH)
    compound_acts = np.load(COMPOUND_ACTS_PATH, mmap_mode="r")
    metadata_columns = [
        "topic", "conjunctA", "conjunctB", "labelA", "labelB", "cell", "connective", "ordering"
    ]
    if not r1[metadata_columns].equals(sidecar[metadata_columns]):
        raise ValueError("R1 metadata is not row-aligned with cached compound activations")
    if compound_acts.shape != (8000, N_LAYERS, DIMENSION):
        raise ValueError(f"Unexpected compound activation shape {compound_acts.shape}")
    train_pairs = set(pair_split.loc[pair_split["partition"].eq("train"), "entity_pair_id"])
    test_pairs = set(pair_split.loc[pair_split["partition"].eq("test"), "entity_pair_id"])
    if len(train_pairs) != 400 or len(test_pairs) != 100 or train_pairs & test_pairs:
        raise ValueError("Compound pair split is not disjoint 400/100")
    compound_test_mask = r1["entity_pair_id"].isin(test_pairs).to_numpy()
    compound_test_indices = np.flatnonzero(compound_test_mask)
    heldout_meta = r1.loc[compound_test_mask].copy().reset_index(drop=True)
    heldout_meta["connective"] = heldout_meta["connective"].str.upper()
    if len(heldout_meta) != 1600 or heldout_meta["entity_pair_id"].nunique() != 100:
        raise ValueError("Compound held-out set is not 1,600 rows / 100 pairs")
    counts = heldout_meta.groupby(["connective", "cell"]).size()
    if len(counts) != 8 or not counts.eq(200).all():
        raise ValueError("Compound held-out connective/cell groups are not all size 200")

    saved_gate_metrics = pd.read_csv(ATOMIC_METRICS_PATH).set_index("layer")
    selected = np.load(SELECTED_PROBE_PATH, allow_pickle=False)
    if int(selected["layer"]) != PRIMARY_LAYER or not np.array_equal(selected["classes"], [0, 1]):
        raise ValueError("Saved operational atomic probe is not the expected layer-22 orientation")

    rows: list[dict[str, float | int]] = []
    for layer in range(N_LAYERS):
        atomic_X = gate.layer_matrix(blocks, layer)
        if not np.isfinite(atomic_X).all():
            raise ValueError(f"Layer {layer} atomic activations contain NaN or infinity")
        if layer == PRIMARY_LAYER:
            coef = np.asarray(selected["coef"], dtype=float).reshape(-1)
            intercept = float(np.asarray(selected["intercept"]).reshape(-1)[0])
        else:
            probe, _ = train_layer_probe(atomic_X, labels, atomic_train_idx, atomic_test_idx)
            coef = probe.coef_[0]
            intercept = float(probe.intercept_[0])
        atomic_scores = atomic_X[atomic_test_idx] @ coef + intercept
        atomic_auc = float(roc_auc_score(labels[atomic_test_idx], atomic_scores))
        saved_atomic_auc = float(saved_gate_metrics.loc[layer, "overall_auroc"])
        if abs(atomic_auc - saved_atomic_auc) > TOLERANCE:
            raise ValueError(
                f"Layer {layer} atomic AUROC {atomic_auc} differs from saved {saved_atomic_auc}"
            )

        compound_X = np.asarray(compound_acts[compound_test_indices, layer, :])
        if compound_X.shape != (1600, DIMENSION) or not np.isfinite(compound_X).all():
            raise ValueError(f"Layer {layer} held-out compound activations are invalid")
        compound_scores = compound_X @ coef + intercept
        if not np.isfinite(compound_scores).all():
            raise ValueError(f"Layer {layer} compound scores contain NaN or infinity")
        scored = heldout_meta[["connective", "cell", "global_truth"]].copy()
        scored["score"] = compound_scores
        row: dict[str, float | int] = {
            "layer": layer,
            "atomic_heldout_auroc": atomic_auc,
        }
        for connective in ("AND", "OR"):
            subset = scored.loc[scored["connective"].eq(connective)]
            row[f"{connective.lower()}_truth_auroc"] = float(
                roc_auc_score(subset["global_truth"], subset["score"])
            )
            add_connective_metrics(row, connective, subset, grouped_pairwise_auc, dprime)
        row["and_minus_or_auroc_gap"] = row["and_truth_auroc"] - row["or_truth_auroc"]
        row["OR_to_AND_range_ratio"] = row["or_total_range"] / row["and_total_range"]
        row["OR_range_compression"] = 1.0 - row["OR_to_AND_range_ratio"]
        row["OR_to_AND_pooled_sd_ratio"] = row["or_pooled_sd"] / row["and_pooled_sd"]
        row["OR_to_AND_range_over_sd_ratio"] = (
            row["or_range_over_sd"] / row["and_range_over_sd"]
        )
        rows.append(row)
        print(
            f"layer {layer:2d}: atomic={atomic_auc:.6f} "
            f"AND={row['and_truth_auroc']:.6f} OR={row['or_truth_auroc']:.6f}",
            flush=True,
        )

    output = pd.DataFrame(rows)
    numeric = output.select_dtypes(include=[np.number])
    if output.shape[0] != N_LAYERS or not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("All-layer output is incomplete or non-finite")

    pass1_geometry = pd.read_csv(PASS1_GEOMETRY_PATH)
    pass1_atomic = pass1_geometry.loc[pass1_geometry["probe"].eq("Atomic")].set_index("connective")
    pass1_pairwise = pd.read_csv(PASS1_PAIRWISE_PATH)
    pass1_pairwise["connective"] = pass1_pairwise["connective"].str.upper()
    pair_lookup = pass1_pairwise.set_index(["connective", "cell_high", "cell_low"])["auroc"]
    with PASS1_DECOMPOSITION_PATH.open() as handle:
        pass1_auc = json.load(handle)["results"]
    layer22 = output.loc[output["layer"].eq(PRIMARY_LAYER)].iloc[0]
    expected22 = {
        "atomic_heldout_auroc": float(saved_gate_metrics.loc[PRIMARY_LAYER, "overall_auroc"]),
        "and_truth_auroc": float(pass1_auc["AND"]["direct_auroc"]),
        "or_truth_auroc": float(pass1_auc["OR"]["direct_auroc"]),
        "and_TT_vs_mixed_auroc": float(pair_lookup.loc[("AND", "TT", "mixed")]),
        "and_mixed_vs_FF_auroc": float(pair_lookup.loc[("AND", "mixed", "FF")]),
        "or_TT_vs_mixed_auroc": float(pair_lookup.loc[("OR", "TT", "mixed")]),
        "or_mixed_vs_FF_auroc": float(pair_lookup.loc[("OR", "mixed", "FF")]),
        "and_range_over_sd": float(pass1_atomic.loc["AND", "range_over_sd"]),
        "or_range_over_sd": float(pass1_atomic.loc["OR", "range_over_sd"]),
    }
    layer22_checks = {
        metric: {
            "sweep_value": float(layer22[metric]),
            "pass1_value": expected,
            "absolute_difference": abs(float(layer22[metric]) - expected),
            "pass": abs(float(layer22[metric]) - expected) <= TOLERANCE,
        }
        for metric, expected in expected22.items()
    }
    layer22_checks["all_pass"] = all(check["pass"] for check in layer22_checks.values())
    if not layer22_checks["all_pass"]:
        raise ValueError(f"Layer-22 reproduction failed: {layer22_checks}")

    headline_metrics = [
        "atomic_heldout_auroc", "and_truth_auroc", "or_truth_auroc",
        "and_minus_or_auroc_gap", "and_TT_vs_mixed_auroc", "and_mixed_vs_FF_auroc",
        "or_TT_vs_mixed_auroc", "or_mixed_vs_FF_auroc", "and_range_over_sd",
        "or_range_over_sd", "OR_range_compression",
    ]
    ranges = {
        metric: {
            "min": float(output[metric].min()),
            "max": float(output[metric].max()),
            "median": float(output[metric].median()),
        }
        for metric in headline_metrics
    }
    high = output.loc[output["atomic_heldout_auroc"].ge(0.99)]
    high_layers = high["layer"].astype(int).tolist()
    summary = {
        "status": "PASS",
        "analysis_scope": "all-layer robustness sweep; operational layer remains frozen at 22",
        "n_layers": N_LAYERS,
        "primary_layer": PRIMARY_LAYER,
        "layer_22_consistency_checks": layer22_checks,
        "headline_metric_ranges_across_layers": ranges,
        "counts": {
            "layers_AND_auroc_gt_OR_auroc": int((output["and_truth_auroc"] > output["or_truth_auroc"]).sum()),
            "layers_OR_TT_mixed_gt_mixed_FF": int((output["or_TT_vs_mixed_auroc"] > output["or_mixed_vs_FF_auroc"]).sum()),
            "layers_OR_range_lt_AND_range": int((output["or_total_range"] < output["and_total_range"]).sum()),
        },
        "high_atomic_performance": {
            "threshold": 0.99,
            "count": int(len(high)),
            "layers": high_layers,
            "median_AND_auroc": float(high["and_truth_auroc"].median()),
            "median_OR_auroc": float(high["or_truth_auroc"].median()),
            "median_OR_mixed_vs_FF_auroc": float(high["or_mixed_vs_FF_auroc"].median()),
            "median_OR_range_compression": float(high["OR_range_compression"].median()),
        },
        "sanity_checks": {
            "compound_heldout_rows_each_layer": 1600,
            "compound_heldout_pairs": 100,
            "each_connective_cell_rows": 200,
            "zero_atomic_entity_overlap": True,
            "zero_compound_pair_overlap": True,
            "all_scores_finite": True,
            "saved_atomic_split_exactly_reproduced": True,
            "selected_layer_not_changed": True,
        },
        "sources": {
            "atomic_split": str(ATOMIC_SPLIT_PATH.relative_to(ROOT)),
            "atomic_saved_metrics": str(ATOMIC_METRICS_PATH.relative_to(ROOT)),
            "selected_layer22_probe": str(SELECTED_PROBE_PATH.relative_to(ROOT)),
            "compound_activations": str(COMPOUND_ACTS_PATH.relative_to(ROOT)),
            "compound_pair_split": str(PAIR_SPLIT_PATH.relative_to(ROOT)),
        },
    }

    output.to_csv(OUTPUT_CSV, index=False)
    with OUTPUT_JSON.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
