#!/usr/bin/env python3
"""Combine saved boundary results with descriptive mean-allocation geometry."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MECHANISM_DIR = ROOT / "results" / "r1" / "mechanism"
DPRIME_PATH = MECHANISM_DIR / "boundary_dprime.csv"
ATOMIC_AUC_PATH = MECHANISM_DIR / "pairwise_auc_atomic.csv"
COMPOUND_AUC_PATH = MECHANISM_DIR / "pairwise_auc_compound.csv"
OUTPUT_PATH = MECHANISM_DIR / "boundary_geometry.csv"
TOLERANCE = 1e-12


def load_pooled_aucs() -> pd.DataFrame:
    frames = []
    for probe, path in (("Atomic", ATOMIC_AUC_PATH), ("Compound", COMPOUND_AUC_PATH)):
        frame = pd.read_csv(path)
        frame["probe"] = probe
        frame["connective"] = frame["connective"].str.upper()
        frames.append(frame)
    aucs = pd.concat(frames, ignore_index=True)
    selected = aucs.loc[
        ((aucs["cell_high"] == "TT") & (aucs["cell_low"] == "mixed"))
        | ((aucs["cell_high"] == "mixed") & (aucs["cell_low"] == "FF"))
    ].copy()
    selected["metric"] = np.where(
        selected["cell_high"].eq("TT"),
        "pairwise_auc_TT_mixed",
        "pairwise_auc_mixed_FF",
    )
    wide = selected.pivot(index=["probe", "connective"], columns="metric", values="auroc")
    if wide.shape != (4, 2) or wide.isna().any().any():
        raise ValueError("Saved pooled pairwise AUROCs are incomplete or duplicated")
    return wide.reset_index()


def main() -> None:
    dprime = pd.read_csv(DPRIME_PATH)
    dprime["connective"] = dprime["connective"].str.upper()
    main_rows = dprime.loc[dprime["comparison"].isin(["TT_vs_mixed", "mixed_vs_FF"])]
    if len(main_rows) != 8:
        raise ValueError("Saved d-prime file does not contain exactly two boundary rows per case")

    aucs = load_pooled_aucs()
    rows: list[dict[str, object]] = []
    for probe in ("Atomic", "Compound"):
        for connective in ("AND", "OR"):
            case = main_rows.loc[
                main_rows["probe"].eq(probe) & main_rows["connective"].eq(connective)
            ].set_index("comparison")
            if set(case.index) != {"TT_vs_mixed", "mixed_vs_FF"}:
                raise ValueError(f"Missing saved boundary rows for {probe} {connective}")
            upper = case.loc["TT_vs_mixed"]
            lower = case.loc["mixed_vs_FF"]

            mu_tt = float(upper["mean_high"])
            mu_mixed = float(upper["mean_low"])
            mu_ff = float(lower["mean_low"])
            if not np.isclose(mu_mixed, float(lower["mean_high"]), atol=TOLERANCE, rtol=0):
                raise ValueError(f"Mixed-group means disagree for {probe} {connective}")

            n_tt, n_mixed, n_ff = int(upper["n_high"]), int(upper["n_low"]), int(lower["n_low"])
            sd_tt, sd_mixed, sd_ff = (
                float(upper["sd_high"]),
                float(upper["sd_low"]),
                float(lower["sd_low"]),
            )
            if n_mixed != int(lower["n_high"]) or not np.isclose(
                sd_mixed, float(lower["sd_high"]), atol=TOLERANCE, rtol=0
            ):
                raise ValueError(f"Mixed-group size/SD disagree for {probe} {connective}")

            upper_gap = mu_tt - mu_mixed
            lower_gap = mu_mixed - mu_ff
            total_range = mu_tt - mu_ff
            if total_range == 0:
                raise ValueError(f"Zero total range for {probe} {connective}")
            upper_fraction = upper_gap / total_range
            lower_fraction = lower_gap / total_range
            delta = upper_gap - lower_gap
            delta_norm = delta / total_range
            pooled_sd = float(np.sqrt(
                ((n_tt - 1) * sd_tt**2 + (n_mixed - 1) * sd_mixed**2 + (n_ff - 1) * sd_ff**2)
                / (n_tt + n_mixed + n_ff - 3)
            ))
            if not np.isclose(upper_fraction + lower_fraction, 1.0, atol=TOLERANCE, rtol=0):
                raise ValueError(f"Gap fractions do not sum to one for {probe} {connective}")
            if not np.isclose(
                delta_norm, upper_fraction - lower_fraction, atol=TOLERANCE, rtol=0
            ):
                raise ValueError(f"Delta_norm identity failed for {probe} {connective}")

            auc_case = aucs.loc[
                aucs["probe"].eq(probe) & aucs["connective"].eq(connective)
            ]
            if len(auc_case) != 1:
                raise ValueError(f"Missing pooled AUROCs for {probe} {connective}")
            rows.append({
                "probe": probe,
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
                "dprime_TT_mixed": float(upper["dprime"]),
                "dprime_mixed_FF": float(lower["dprime"]),
                "pairwise_auc_TT_mixed": float(auc_case.iloc[0]["pairwise_auc_TT_mixed"]),
                "pairwise_auc_mixed_FF": float(auc_case.iloc[0]["pairwise_auc_mixed_FF"]),
            })

    output = pd.DataFrame(rows)
    if output.shape != (4, 18) or not np.isfinite(output.select_dtypes(include=[np.number])).all().all():
        raise ValueError("Unexpected or non-finite boundary geometry output")
    output.to_csv(OUTPUT_PATH, index=False)
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
