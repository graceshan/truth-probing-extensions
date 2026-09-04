#!/usr/bin/env python3
"""Describe held-out boundary geometry for the perpendicular probe component."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MECHANISM_DIR = ROOT / "results" / "r1" / "mechanism"
SCORES_PATH = MECHANISM_DIR / "perp_test_scores.csv"
PAIRWISE_PATH = MECHANISM_DIR / "pairwise_auc_perp.csv"
PASS1_GEOMETRY_PATH = MECHANISM_DIR / "boundary_geometry.csv"
TASK5_SCRIPT = Path(__file__).with_name("05_boundary_dprime.py")
OUTPUT_GEOMETRY = MECHANISM_DIR / "perp_boundary_geometry.csv"
OUTPUT_DPRIME = MECHANISM_DIR / "perp_boundary_dprime.csv"
OUTPUT_SUMMARY = MECHANISM_DIR / "perp_geometry_summary.json"
TOLERANCE = 1e-12


def load_task5_dprime_function():
    spec = importlib.util.spec_from_file_location("task5_boundary_dprime", TASK5_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Task 5 implementation from {TASK5_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.dprime


def main() -> None:
    dprime = load_task5_dprime_function()
    scores = pd.read_csv(SCORES_PATH)
    pairwise = pd.read_csv(PAIRWISE_PATH)
    pass1 = pd.read_csv(PASS1_GEOMETRY_PATH)
    scores["connective"] = scores["connective"].str.upper()
    pairwise["connective"] = pairwise["connective"].str.upper()
    pass1["connective"] = pass1["connective"].str.upper()

    if len(scores) != 1600 or scores["entity_pair_id"].nunique() != 100:
        raise ValueError("Saved perpendicular scores are not the fixed 1,600-row/100-pair set")
    if not np.isfinite(scores["score_perp"]).all():
        raise ValueError("Saved perpendicular scores contain NaN or infinity")
    expected_counts = pd.Series(
        200,
        index=pd.MultiIndex.from_product(
            [["AND", "OR"], ["TT", "TF", "FT", "FF"]],
            names=["connective", "cell"],
        ),
    )
    observed_counts = scores.groupby(["connective", "cell"]).size().reindex(expected_counts.index)
    if not observed_counts.equals(expected_counts):
        raise ValueError(f"Saved perpendicular score cells are not balanced:\n{observed_counts}")

    group_cells = {
        "TT": ("TT",),
        "TF": ("TF",),
        "FT": ("FT",),
        "FF": ("FF",),
        "mixed": ("TF", "FT"),
    }
    comparisons = [
        ("TT_vs_TF", "TT", "TF"),
        ("TT_vs_FT", "TT", "FT"),
        ("TT_vs_FF", "TT", "FF"),
        ("TF_vs_FF", "TF", "FF"),
        ("FT_vs_FF", "FT", "FF"),
        ("TF_vs_FT", "TF", "FT"),
        ("TT_vs_mixed", "TT", "mixed"),
        ("mixed_vs_FF", "mixed", "FF"),
    ]
    cell_statistics: dict[str, dict[str, dict[str, float | int]]] = {}
    dprime_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []

    for connective in ("AND", "OR"):
        subset = scores.loc[scores["connective"].eq(connective)]
        arrays = {
            group: subset.loc[subset["cell"].isin(cells), "score_perp"].to_numpy(dtype=float)
            for group, cells in group_cells.items()
        }
        stats = {
            group: {
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
            }
            for group, values in arrays.items()
        }
        if stats["mixed"]["n"] != 400 or any(stats[cell]["n"] != 200 for cell in ("TT", "TF", "FT", "FF")):
            raise ValueError(f"Unexpected group sizes for {connective}")
        cell_statistics[connective] = stats

        for comparison, high, low in comparisons:
            dprime_rows.append({
                "direction": "Perpendicular",
                "connective": connective,
                "comparison": comparison,
                "group_high": high,
                "group_low": low,
                "n_high": stats[high]["n"],
                "n_low": stats[low]["n"],
                "mean_high": stats[high]["mean"],
                "mean_low": stats[low]["mean"],
                "sd_high": stats[high]["sd"],
                "sd_low": stats[low]["sd"],
                "dprime": dprime(arrays[high], arrays[low]),
            })

        mu_tt = float(stats["TT"]["mean"])
        mu_mixed = float(stats["mixed"]["mean"])
        mu_ff = float(stats["FF"]["mean"])
        upper_gap = mu_tt - mu_mixed
        lower_gap = mu_mixed - mu_ff
        total_range = mu_tt - mu_ff
        delta = upper_gap - lower_gap
        delta_norm = delta / total_range
        upper_fraction = upper_gap / total_range
        lower_fraction = lower_gap / total_range
        n_tt, n_mixed, n_ff = (stats[group]["n"] for group in ("TT", "mixed", "FF"))
        sd_tt, sd_mixed, sd_ff = (stats[group]["sd"] for group in ("TT", "mixed", "FF"))
        pooled_sd = float(np.sqrt(
            ((n_tt - 1) * sd_tt**2 + (n_mixed - 1) * sd_mixed**2 + (n_ff - 1) * sd_ff**2)
            / (n_tt + n_mixed + n_ff - 3)
        ))
        if not np.isclose(upper_fraction + lower_fraction, 1.0, atol=TOLERANCE, rtol=0):
            raise ValueError(f"Gap fractions do not sum to one for {connective}")
        if not np.isclose(delta_norm, upper_fraction - lower_fraction, atol=TOLERANCE, rtol=0):
            raise ValueError(f"Delta_norm identity failed for {connective}")

        auc_lookup = pairwise.loc[pairwise["connective"].eq(connective)].set_index(
            ["higher_cell", "lower_cell"]
        )["auroc"]
        dprime_lookup = {
            row["comparison"]: row["dprime"]
            for row in dprime_rows
            if row["connective"] == connective
        }
        geometry_rows.append({
            "direction": "Perpendicular",
            "connective": connective,
            "mu_TT": mu_tt,
            "mu_mixed": mu_mixed,
            "mu_FF": mu_ff,
            "upper_gap": upper_gap,
            "lower_gap": lower_gap,
            "total_range": total_range,
            "upper_gap_fraction": upper_fraction,
            "lower_gap_fraction": lower_fraction,
            "Delta": delta,
            "Delta_norm": delta_norm,
            "pooled_sd": pooled_sd,
            "range_over_sd": total_range / pooled_sd,
            "dprime_TT_mixed": float(dprime_lookup["TT_vs_mixed"]),
            "dprime_mixed_FF": float(dprime_lookup["mixed_vs_FF"]),
            "pairwise_auc_TT_mixed": float(auc_lookup.loc[("TT", "mixed")]),
            "pairwise_auc_mixed_FF": float(auc_lookup.loc[("mixed", "FF")]),
        })

    dprime_output = pd.DataFrame(dprime_rows)
    geometry_output = pd.DataFrame(geometry_rows)
    comparison_columns = [
        "direction", "connective", "pairwise_auc_TT_mixed", "pairwise_auc_mixed_FF",
        "dprime_TT_mixed", "dprime_mixed_FF", "upper_gap_fraction",
        "lower_gap_fraction", "Delta_norm", "range_over_sd",
    ]
    prior_comparison = pass1.rename(columns={"probe": "direction"})[comparison_columns]
    full_comparison = pd.concat(
        [prior_comparison, geometry_output[comparison_columns]], ignore_index=True
    )
    if len(full_comparison) != 6 or not np.isfinite(
        full_comparison.select_dtypes(include=[np.number]).to_numpy()
    ).all():
        raise ValueError("Full comparison is incomplete or non-finite")

    compound = pass1.loc[pass1["probe"].eq("Compound")].set_index("connective")
    perp = geometry_output.set_index("connective")
    raw_scale_ratios = {
        "AND_total_range": float(perp.loc["AND", "total_range"] / compound.loc["AND", "total_range"]),
        "OR_total_range": float(perp.loc["OR", "total_range"] / compound.loc["OR", "total_range"]),
        "AND_pooled_sd": float(perp.loc["AND", "pooled_sd"] / compound.loc["AND", "pooled_sd"]),
        "OR_pooled_sd": float(perp.loc["OR", "pooled_sd"] / compound.loc["OR", "pooled_sd"]),
    }
    summary = {
        "status": "PASS",
        "score_source": str(SCORES_PATH.relative_to(ROOT)),
        "pairwise_auc_source": str(PAIRWISE_PATH.relative_to(ROOT)),
        "pass1_geometry_source": str(PASS1_GEOMETRY_PATH.relative_to(ROOT)),
        "dprime_implementation_source": str(TASK5_SCRIPT.relative_to(ROOT)),
        "pooling_convention": "TT, pooled TF-union-FT mixed rows, and FF; sample SD ddof=1",
        "cell_statistics": cell_statistics,
        "raw_scale_ratios_perpendicular_over_compound": raw_scale_ratios,
        "full_comparison": full_comparison.to_dict(orient="records"),
        "sanity_checks": {
            "n_rows_is_1600": len(scores) == 1600,
            "n_pairs_is_100": scores["entity_pair_id"].nunique() == 100,
            "each_connective_cell_is_200": bool(observed_counts.eq(200).all()),
            "scores_finite": bool(np.isfinite(scores["score_perp"]).all()),
            "gap_fraction_identities_pass": True,
        },
    }

    geometry_output.to_csv(OUTPUT_GEOMETRY, index=False)
    dprime_output.to_csv(OUTPUT_DPRIME, index=False)
    with OUTPUT_SUMMARY.open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print("GEOMETRY\n" + geometry_output.to_string(index=False))
    print("\nDPRIME\n" + dprime_output.to_string(index=False))
    print("\nCOMPARISON\n" + full_comparison.to_string(index=False))
    print("\nRATIOS\n" + json.dumps(raw_scale_ratios, indent=2))


if __name__ == "__main__":
    main()
