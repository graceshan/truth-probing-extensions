#!/usr/bin/env python3
"""Pair-level percentile bootstrap for primary layer-22 atomic-transfer metrics."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[2]
MECHANISM_DIR = ROOT / "results" / "r1" / "mechanism"
HELDOUT_PATH = ROOT / "results" / "r1" / "probe_transfer_comparison_predictions.csv"
SPLIT_PATH = ROOT / "results" / "r1" / "g_c_cg" / "pair_split.csv"
PAIRWISE_PATH = MECHANISM_DIR / "pairwise_auc_atomic.csv"
GEOMETRY_PATH = MECHANISM_DIR / "boundary_geometry.csv"
DECOMPOSITION_PATH = MECHANISM_DIR / "atomic_auroc_decomposition.json"
TASK2_SCRIPT = Path(__file__).with_name("02_atomic_pairwise_auc.py")
TASK5_SCRIPT = Path(__file__).with_name("05_boundary_dprime.py")
OUTPUT_DRAWS = MECHANISM_DIR / "bootstrap_pair_level_draws.csv"
OUTPUT_SUMMARY_CSV = MECHANISM_DIR / "bootstrap_pair_level_summary.csv"
OUTPUT_SUMMARY_JSON = MECHANISM_DIR / "bootstrap_pair_level_summary.json"
SCORE_COLUMN = "atomic_union_to_compounds_score"
N_BOOTSTRAP = 2000
SEED = 0
N_PAIRS = 100
ROWS_PER_PAIR = 16
TOLERANCE = 1e-12


def load_function(path: Path, module_name: str, function_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def calculate_metrics(frame: pd.DataFrame, grouped_pairwise_auc, dprime) -> dict[str, float]:
    result: dict[str, float] = {}
    geometry: dict[str, dict[str, float]] = {}
    for connective in ("AND", "OR"):
        prefix = connective.lower()
        subset = frame.loc[frame["connective"].eq(connective)]
        labels = subset["global_truth"].to_numpy(dtype=int)
        scores = subset[SCORE_COLUMN].to_numpy(dtype=float)
        if not np.array_equal(np.unique(labels), np.array([0, 1])):
            raise ValueError(f"Undefined {connective} AUROC: missing a truth class")
        result[f"{prefix}_truth_auroc"] = float(roc_auc_score(labels, scores))

        for name, high_cells, low_cells in (
            ("TT_vs_mixed", ("TT",), ("TF", "FT")),
            ("mixed_vs_FF", ("TF", "FT"), ("FF",)),
        ):
            _, _, auc = grouped_pairwise_auc(subset, SCORE_COLUMN, high_cells, low_cells)
            result[f"{prefix}_{name}_auroc"] = auc

        tt = subset.loc[subset["cell"].eq("TT"), SCORE_COLUMN].to_numpy(dtype=float)
        mixed = subset.loc[subset["cell"].isin(["TF", "FT"]), SCORE_COLUMN].to_numpy(dtype=float)
        ff = subset.loc[subset["cell"].eq("FF"), SCORE_COLUMN].to_numpy(dtype=float)
        mu_tt, mu_ff = float(np.mean(tt)), float(np.mean(ff))
        total_range = mu_tt - mu_ff
        sd_tt = float(np.std(tt, ddof=1))
        sd_mixed = float(np.std(mixed, ddof=1))
        sd_ff = float(np.std(ff, ddof=1))
        pooled_sd = float(np.sqrt(
            ((len(tt) - 1) * sd_tt**2 + (len(mixed) - 1) * sd_mixed**2 + (len(ff) - 1) * sd_ff**2)
            / (len(tt) + len(mixed) + len(ff) - 3)
        ))
        if pooled_sd == 0:
            raise ValueError(f"Undefined {connective} range/SD: zero pooled SD")
        geometry[connective] = {
            "total_range": total_range,
            "pooled_sd": pooled_sd,
            "range_over_sd": total_range / pooled_sd,
        }
        result[f"{prefix}_total_range"] = total_range
        result[f"{prefix}_pooled_sd"] = pooled_sd
        result[f"{prefix}_range_over_sd"] = total_range / pooled_sd
        result[f"dprime_{connective}_TT_mixed"] = dprime(tt, mixed)
        result[f"dprime_{connective}_mixed_FF"] = dprime(mixed, ff)

    result["and_minus_or_auroc"] = result["and_truth_auroc"] - result["or_truth_auroc"]
    if geometry["AND"]["total_range"] == 0 or geometry["AND"]["range_over_sd"] == 0:
        raise ValueError("Undefined OR/AND ratio due to zero AND denominator")
    result["OR_to_AND_range_ratio"] = geometry["OR"]["total_range"] / geometry["AND"]["total_range"]
    result["OR_range_compression"] = 1.0 - result["OR_to_AND_range_ratio"]
    result["OR_to_AND_pooled_sd_ratio"] = geometry["OR"]["pooled_sd"] / geometry["AND"]["pooled_sd"]
    result["OR_to_AND_range_over_sd_ratio"] = (
        geometry["OR"]["range_over_sd"] / geometry["AND"]["range_over_sd"]
    )
    if not np.isfinite(list(result.values())).all():
        raise ValueError("A bootstrap metric is NaN or infinite")
    return result


def main() -> None:
    grouped_pairwise_auc = load_function(
        TASK2_SCRIPT, "task2_pairwise", "grouped_pairwise_auc"
    )
    dprime = load_function(TASK5_SCRIPT, "task5_dprime", "dprime")
    heldout = pd.read_csv(HELDOUT_PATH)
    split = pd.read_csv(SPLIT_PATH)
    heldout["connective"] = heldout["connective"].str.upper()

    test_pairs_from_split = set(
        split.loc[split["partition"].eq("test"), "entity_pair_id"]
    )
    train_pairs = set(split.loc[split["partition"].eq("train"), "entity_pair_id"])
    heldout_pairs = heldout["entity_pair_id"].drop_duplicates().to_numpy(dtype=object)
    if len(heldout) != 1600 or len(heldout_pairs) != N_PAIRS:
        raise ValueError("Held-out sample is not 1,600 rows / 100 canonical pairs")
    if set(heldout_pairs) != test_pairs_from_split or set(heldout_pairs) & train_pairs:
        raise ValueError("Held-out pairs differ from the saved split or overlap training pairs")
    pair_sizes = heldout.groupby("entity_pair_id").size()
    if not pair_sizes.eq(ROWS_PER_PAIR).all():
        raise ValueError("Not every held-out canonical pair has exactly 16 rows")
    original_counts = heldout.groupby(["connective", "cell"]).size()
    if len(original_counts) != 8 or not original_counts.eq(200).all():
        raise ValueError("Original held-out connective/cell groups are not all size 200")
    if not np.isfinite(heldout[SCORE_COLUMN]).all():
        raise ValueError("Held-out atomic scores contain NaN or infinity")

    pair_indices = {
        pair_id: np.flatnonzero(heldout["entity_pair_id"].to_numpy() == pair_id)
        for pair_id in heldout_pairs
    }
    if not all(len(indices) == ROWS_PER_PAIR for indices in pair_indices.values()):
        raise ValueError("Pair index blocks are not uniformly 16 rows")

    observed = calculate_metrics(heldout, grouped_pairwise_auc, dprime)
    with DECOMPOSITION_PATH.open() as handle:
        direct_saved = json.load(handle)["results"]
    pairwise_saved = pd.read_csv(PAIRWISE_PATH).set_index(
        ["connective", "cell_high", "cell_low"]
    )["auroc"]
    geometry_saved = pd.read_csv(GEOMETRY_PATH)
    geometry_saved = geometry_saved.loc[geometry_saved["probe"].eq("Atomic")].set_index("connective")
    expected = {
        "and_truth_auroc": float(direct_saved["AND"]["direct_auroc"]),
        "or_truth_auroc": float(direct_saved["OR"]["direct_auroc"]),
        "and_TT_vs_mixed_auroc": float(pairwise_saved.loc[("AND", "TT", "mixed")]),
        "and_mixed_vs_FF_auroc": float(pairwise_saved.loc[("AND", "mixed", "FF")]),
        "or_TT_vs_mixed_auroc": float(pairwise_saved.loc[("OR", "TT", "mixed")]),
        "or_mixed_vs_FF_auroc": float(pairwise_saved.loc[("OR", "mixed", "FF")]),
        "and_range_over_sd": float(geometry_saved.loc["AND", "range_over_sd"]),
        "or_range_over_sd": float(geometry_saved.loc["OR", "range_over_sd"]),
        "OR_range_compression": 1.0
        - float(geometry_saved.loc["OR", "total_range"])
        / float(geometry_saved.loc["AND", "total_range"]),
    }
    observed_checks = {
        metric: {
            "observed": observed[metric],
            "expected": value,
            "absolute_difference": abs(observed[metric] - value),
            "pass": abs(observed[metric] - value) <= TOLERANCE,
        }
        for metric, value in expected.items()
    }
    if not all(check["pass"] for check in observed_checks.values()):
        raise ValueError(f"Observed Pass 1 reproduction failed: {observed_checks}")

    rng = np.random.default_rng(SEED)
    draw_rows: list[dict[str, float | int]] = []
    multiplicity_preserved = True
    undefined_metrics = 0
    for replicate in range(N_BOOTSTRAP):
        sampled_pairs = rng.choice(heldout_pairs, size=N_PAIRS, replace=True)
        sampled_indices = np.concatenate([pair_indices[pair_id] for pair_id in sampled_pairs])
        if len(sampled_pairs) != N_PAIRS or len(sampled_indices) != N_PAIRS * ROWS_PER_PAIR:
            raise ValueError(f"Replicate {replicate}: wrong pair-instance or row count")
        replicate_frame = heldout.iloc[sampled_indices]
        sampled_counts = pd.Series(sampled_pairs).value_counts()
        actual_counts = replicate_frame["entity_pair_id"].value_counts()
        expected_row_counts = sampled_counts * ROWS_PER_PAIR
        if not actual_counts.sort_index().equals(expected_row_counts.sort_index()):
            multiplicity_preserved = False
            raise ValueError(f"Replicate {replicate}: sampled-pair multiplicity was not preserved")
        if not set(replicate_frame["entity_pair_id"]).issubset(test_pairs_from_split):
            raise ValueError(f"Replicate {replicate}: non-held-out pair appeared")
        replicate_counts = replicate_frame.groupby(["connective", "cell"]).size()
        if len(replicate_counts) != 8 or not replicate_counts.eq(200).all():
            raise ValueError(f"Replicate {replicate}: connective/cell balance changed")
        try:
            metrics = calculate_metrics(replicate_frame, grouped_pairwise_auc, dprime)
        except ValueError as exc:
            undefined_metrics += 1
            raise ValueError(f"Replicate {replicate} produced an undefined metric: {exc}") from exc
        draw_rows.append({"replicate": replicate, **metrics})
        if (replicate + 1) % 250 == 0:
            print(f"completed {replicate + 1}/{N_BOOTSTRAP}", flush=True)

    draws = pd.DataFrame(draw_rows)
    metric_columns = [column for column in draws.columns if column != "replicate"]
    summary_rows = []
    for metric in metric_columns:
        values = draws[metric].to_numpy(dtype=float)
        summary_rows.append({
            "metric": metric,
            "observed": observed[metric],
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_median": float(np.median(values)),
            "bootstrap_sd": float(np.std(values, ddof=1)),
            "ci_2_5": float(np.quantile(values, 0.025)),
            "ci_97_5": float(np.quantile(values, 0.975)),
        })
    summary_csv = pd.DataFrame(summary_rows)
    sanity = {
        "status": "PASS",
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "bootstrap_unit": "canonical_pair",
        "sampled_pair_instances_per_replicate": N_PAIRS,
        "rows_per_pair": ROWS_PER_PAIR,
        "rows_per_replicate": N_PAIRS * ROWS_PER_PAIR,
        "undefined_metrics": undefined_metrics,
        "multiplicity_preserved": multiplicity_preserved,
        "only_heldout_pairs_sampled": True,
        "all_replicates_balanced_by_connective_cell": True,
        "observed_reproduction_checks": observed_checks,
        "sources": {
            "heldout_scores": str(HELDOUT_PATH.relative_to(ROOT)),
            "pair_split": str(SPLIT_PATH.relative_to(ROOT)),
            "pairwise_implementation": str(TASK2_SCRIPT.relative_to(ROOT)),
            "dprime_implementation": str(TASK5_SCRIPT.relative_to(ROOT)),
        },
    }

    draws.to_csv(OUTPUT_DRAWS, index=False)
    summary_csv.to_csv(OUTPUT_SUMMARY_CSV, index=False)
    with OUTPUT_SUMMARY_JSON.open("w") as handle:
        json.dump(sanity, handle, indent=2)
        handle.write("\n")
    print(summary_csv.to_string(index=False))
    print(json.dumps(sanity, indent=2))


if __name__ == "__main__":
    main()
