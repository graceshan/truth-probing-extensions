"""Core held-out R1 replication for the frozen Qwen3 atomic probe at layer 21."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


LAYER = 21
N_LAYERS = 36
HIDDEN_SIZE = 4096
C = 0.1
SEED = 0
TOLERANCE = 1e-12
R1_ACTS = REPO_ROOT / "acts" / "qwen3_8b" / "r1" / "r1_quadruples_qwen3_8b_acts.npy"
R1_SIDECAR = REPO_ROOT / "acts" / "qwen3_8b" / "r1" / "r1_quadruples_qwen3_8b_metadata.csv"
R1_SOURCE = REPO_ROOT / "data" / "r1_quadruples.csv"
R1_MASTER = REPO_ROOT / "results" / "r1" / "r1_score_table.csv"
PAIR_SPLIT = REPO_ROOT / "results" / "r1" / "g_c_cg" / "pair_split.csv"
ATOMIC_PROBE = REPO_ROOT / "results" / "qwen3_8b_union_probe_gate" / "selected_probe.npz"
QWEN3_GATE_SUMMARY = REPO_ROOT / "results" / "qwen3_8b_union_probe_gate" / "summary.json"
QWEN25_GATE_SUMMARY = REPO_ROOT / "results" / "qwen_union_probe_gate" / "summary.json"
QWEN25_OVERALL = REPO_ROOT / "results" / "r1" / "probe_transfer_comparison_overall.csv"
QWEN25_CONNECTIVE = REPO_ROOT / "results" / "r1" / "probe_transfer_comparison_by_connective.csv"
QWEN25_GEOMETRY = REPO_ROOT / "results" / "r1" / "mechanism" / "boundary_geometry.csv"
TASK2_SCRIPT = REPO_ROOT / "scripts" / "mechanism" / "02_atomic_pairwise_auc.py"
TASK5_SCRIPT = REPO_ROOT / "scripts" / "mechanism" / "05_boundary_dprime.py"
OUTPUT_DIR = REPO_ROOT / "results" / "qwen3_8b_r1_core"


def load_function(path: Path, module_name: str, function_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)


def classification(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    return {
        "auroc": float(roc_auc_score(labels, scores)),
        "accuracy_at_0": float(accuracy_score(labels, (scores >= 0).astype(int))),
    }


def pairwise_table(frame: pd.DataFrame, score_column: str, probe_name: str, helper):
    comparisons = [
        ("TT", "TF", ("TT",), ("TF",)),
        ("TT", "FT", ("TT",), ("FT",)),
        ("TT", "FF", ("TT",), ("FF",)),
        ("TF", "FF", ("TF",), ("FF",)),
        ("FT", "FF", ("FT",), ("FF",)),
        ("TF", "FT", ("TF",), ("FT",)),
        ("TT", "mixed", ("TT",), ("TF", "FT")),
        ("mixed", "FF", ("TF", "FT"), ("FF",)),
    ]
    rows = []
    for connective in ("AND", "OR"):
        subset = frame.loc[frame["connective"].eq(connective)]
        for high, low, high_cells, low_cells in comparisons:
            n_high, n_low, value = helper(subset, score_column, high_cells, low_cells)
            rows.append({
                "probe": probe_name,
                "connective": connective,
                "cell_high": high,
                "cell_low": low,
                "n_high": n_high,
                "n_low": n_low,
                "auroc": value,
            })
    return pd.DataFrame(rows)


def score_geometry(frame: pd.DataFrame, score_column: str, dprime):
    rows = []
    for connective in ("AND", "OR"):
        subset = frame.loc[frame["connective"].eq(connective)]
        tt = subset.loc[subset["cell"].eq("TT"), score_column].to_numpy(dtype=float)
        mixed = subset.loc[subset["cell"].isin(["TF", "FT"]), score_column].to_numpy(dtype=float)
        ff = subset.loc[subset["cell"].eq("FF"), score_column].to_numpy(dtype=float)
        mu_tt, mu_mixed, mu_ff = map(float, (tt.mean(), mixed.mean(), ff.mean()))
        upper_gap = mu_tt - mu_mixed
        lower_gap = mu_mixed - mu_ff
        total_range = mu_tt - mu_ff
        sd_tt, sd_mixed, sd_ff = (
            float(tt.std(ddof=1)), float(mixed.std(ddof=1)), float(ff.std(ddof=1))
        )
        pooled_sd = float(np.sqrt(
            ((len(tt) - 1) * sd_tt**2 + (len(mixed) - 1) * sd_mixed**2
             + (len(ff) - 1) * sd_ff**2)
            / (len(tt) + len(mixed) + len(ff) - 3)
        ))
        rows.append({
            "connective": connective,
            "mu_TT": mu_tt,
            "mu_mixed": mu_mixed,
            "mu_FF": mu_ff,
            "total_range": total_range,
            "upper_gap": upper_gap,
            "lower_gap": lower_gap,
            "upper_gap_fraction": upper_gap / total_range,
            "lower_gap_fraction": lower_gap / total_range,
            "pooled_sd": pooled_sd,
            "range_over_sd": total_range / pooled_sd,
            "dprime_TT_mixed": dprime(tt, mixed),
            "dprime_mixed_FF": dprime(mixed, ff),
        })
    geometry = pd.DataFrame(rows).set_index("connective")
    geometry_ratios = {
        "OR_to_AND_range_ratio": float(
            geometry.loc["OR", "total_range"] / geometry.loc["AND", "total_range"]
        ),
        "OR_to_AND_pooled_sd_ratio": float(
            geometry.loc["OR", "pooled_sd"] / geometry.loc["AND", "pooled_sd"]
        ),
        "OR_to_AND_range_over_sd_ratio": float(
            geometry.loc["OR", "range_over_sd"] / geometry.loc["AND", "range_over_sd"]
        ),
    }
    return geometry, geometry_ratios


def main() -> None:
    pairwise_helper = load_function(TASK2_SCRIPT, "pass1_pairwise", "grouped_pairwise_auc")
    dprime = load_function(TASK5_SCRIPT, "pass1_dprime", "dprime")

    source = pd.read_csv(R1_SOURCE)
    sidecar = pd.read_csv(R1_SIDECAR)
    master = pd.read_csv(R1_MASTER)
    split = pd.read_csv(PAIR_SPLIT)
    if not source.equals(sidecar):
        raise ValueError("Qwen3 R1 sidecar does not exactly equal the assembled R1 source")
    metadata_columns = [
        "topic", "conjunctA", "conjunctB", "labelA", "labelB", "cell", "connective", "ordering"
    ]
    if not master[metadata_columns].equals(sidecar[metadata_columns]):
        raise ValueError("Canonical-pair master metadata is not row-aligned with Qwen3 R1")

    acts = np.load(R1_ACTS, mmap_mode="r")
    if acts.shape != (8000, N_LAYERS, HIDDEN_SIZE) or acts.dtype != np.float16:
        raise ValueError(f"Unexpected Qwen3 R1 activation array: {acts.shape}, {acts.dtype}")
    table = sidecar.copy()
    table.insert(0, "r1_row_index", np.arange(len(table)))
    table["entity_pair_id"] = master["entity_pair_id"]
    table["global_truth"] = np.where(
        table["connective"].eq("and"), table["cell"].eq("TT"), ~table["cell"].eq("FF")
    ).astype(int)
    if not table["global_truth"].equals(master["global_truth"]):
        raise ValueError("Recomputed R1 global truth differs from the Qwen2.5 master")

    train_pairs = set(split.loc[split["partition"].eq("train"), "entity_pair_id"])
    test_pairs = set(split.loc[split["partition"].eq("test"), "entity_pair_id"])
    all_pairs = set(table["entity_pair_id"])
    if len(all_pairs) != 500 or len(train_pairs) != 400 or len(test_pairs) != 100:
        raise ValueError("Canonical-pair counts differ from 500/400/100")
    if train_pairs & test_pairs or train_pairs | test_pairs != all_pairs:
        raise ValueError("Canonical-pair leakage or incomplete split coverage")
    train_mask = table["entity_pair_id"].isin(train_pairs).to_numpy()
    test_mask = table["entity_pair_id"].isin(test_pairs).to_numpy()
    if train_mask.sum() != 6400 or test_mask.sum() != 1600:
        raise ValueError("R1 split does not produce 6,400/1,600 rows")
    heldout = table.loc[test_mask].copy().reset_index(drop=True)
    heldout["connective"] = heldout["connective"].str.upper()
    counts = heldout.groupby(["connective", "cell"]).size()
    if len(counts) != 8 or not counts.eq(200).all():
        raise ValueError(f"Held-out connective/cell counts are not all 200:\n{counts}")
    if not heldout.groupby("entity_pair_id").size().eq(16).all():
        raise ValueError("A held-out canonical pair does not contain all 16 variants")

    X = np.asarray(acts[:, LAYER, :])
    if X.shape != (8000, HIDDEN_SIZE) or not np.isfinite(X).all():
        raise ValueError("Qwen3 layer-21 R1 activations are invalid")
    atomic_probe = np.load(ATOMIC_PROBE, allow_pickle=False)
    if int(atomic_probe["layer"]) != LAYER or atomic_probe["coef"].shape != (1, HIDDEN_SIZE):
        raise ValueError("Frozen Qwen3 atomic probe has wrong layer or dimension")
    if not np.array_equal(atomic_probe["classes"], [0, 1]):
        raise ValueError("Frozen Qwen3 atomic probe has unexpected score orientation")
    atomic_scores = X @ atomic_probe["coef"][0] + float(atomic_probe["intercept"][0])
    if not np.isfinite(atomic_scores).all():
        raise ValueError("Frozen atomic transfer scores contain NaN or infinity")
    heldout["frozen_atomic_score"] = atomic_scores[test_mask]

    atomic_overall = classification(
        heldout["global_truth"].to_numpy(), heldout["frozen_atomic_score"].to_numpy()
    )
    atomic_connective = {}
    atomic_cells = []
    for connective in ("AND", "OR"):
        subset = heldout.loc[heldout["connective"].eq(connective)]
        atomic_connective[connective] = classification(
            subset["global_truth"].to_numpy(), subset["frozen_atomic_score"].to_numpy()
        )
        for cell in ("TT", "TF", "FT", "FF"):
            values = subset.loc[subset["cell"].eq(cell), "frozen_atomic_score"]
            atomic_cells.append({
                "connective": connective,
                "cell": cell,
                "N": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
            })
    atomic_pairwise = pairwise_table(
        heldout, "frozen_atomic_score", "frozen_atomic", pairwise_helper
    )
    atomic_lookup = atomic_pairwise.set_index(["connective", "cell_high", "cell_low"])["auroc"]
    reconstructed = {
        "AND": float(np.mean([
            atomic_lookup.loc[("AND", "TT", "TF")],
            atomic_lookup.loc[("AND", "TT", "FT")],
            atomic_lookup.loc[("AND", "TT", "FF")],
        ])),
        "OR": float(np.mean([
            atomic_lookup.loc[("OR", "TT", "FF")],
            atomic_lookup.loc[("OR", "TF", "FF")],
            atomic_lookup.loc[("OR", "FT", "FF")],
        ])),
    }
    reconstruction_differences = {
        connective: abs(reconstructed[connective] - atomic_connective[connective]["auroc"])
        for connective in ("AND", "OR")
    }
    if any(value > TOLERANCE for value in reconstruction_differences.values()):
        raise ValueError("Balanced operator AUROC decomposition failed")

    geometry, geometry_ratios = score_geometry(heldout, "frozen_atomic_score", dprime)
    geometry_output = {
        "probe": "frozen_qwen3_atomic_to_compounds",
        "layer": LAYER,
        "by_connective": geometry.reset_index().to_dict(orient="records"),
        "ratios": geometry_ratios,
        "definitions": {
            "mixed": "pooled TF and FT rows",
            "sample_sd_ddof": 1,
            "pooled_sd": "weighted sample-variance pool across TT, mixed, FF as in Pass 1",
        },
    }

    # Same-layer positive control, fit only after all frozen-transfer quantities exist.
    compound_probe = LogisticRegression(
        penalty="l2", C=C, fit_intercept=True, solver="lbfgs",
        max_iter=2000, random_state=SEED,
    )
    compound_probe.fit(X[train_mask], table.loc[train_mask, "global_truth"])
    if not np.array_equal(compound_probe.classes_, [0, 1]):
        raise ValueError("Compound control has unexpected score orientation")
    compound_scores = compound_probe.decision_function(X)
    direct_compound_scores = X @ compound_probe.coef_[0] + compound_probe.intercept_[0]
    if not np.allclose(compound_scores, direct_compound_scores, rtol=0, atol=TOLERANCE):
        raise ValueError("Compound decision_function differs from raw affine score")
    heldout["compound_trained_score"] = compound_scores[test_mask]
    compound_overall = classification(
        heldout["global_truth"].to_numpy(), heldout["compound_trained_score"].to_numpy()
    )
    compound_connective = {}
    for connective in ("AND", "OR"):
        subset = heldout.loc[heldout["connective"].eq(connective)]
        compound_connective[connective] = classification(
            subset["global_truth"].to_numpy(), subset["compound_trained_score"].to_numpy()
        )
    compound_pairwise = pairwise_table(
        heldout, "compound_trained_score", "compound_trained", pairwise_helper
    )
    all_pairwise = pd.concat([atomic_pairwise, compound_pairwise], ignore_index=True)

    verification = {
        "activation_shape": list(acts.shape),
        "activation_dtype": str(acts.dtype),
        "canonical_pairs_total": len(all_pairs),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "rows_per_test_pair": 16,
        "train_test_pair_overlap": [],
        "heldout_pair_membership_source": str(PAIR_SPLIT.relative_to(REPO_ROOT)),
        "heldout_membership_exactly_same_as_qwen2_5": True,
        "heldout_connective_cell_counts": {
            f"{connective}_{cell}": int(counts.loc[(connective, cell)])
            for connective in ("AND", "OR") for cell in ("TT", "TF", "FT", "FF")
        },
        "frozen_atomic_probe_unchanged": True,
        "frozen_atomic_layer": LAYER,
        "normalization": "none",
        "threshold": 0.0,
        "all_scores_finite": True,
    }
    frozen_summary = {
        "model": "Qwen/Qwen3-8B",
        "probe_source": str(ATOMIC_PROBE.relative_to(REPO_ROOT)),
        "verification": verification,
        "overall": atomic_overall,
        "by_connective": atomic_connective,
        "cell_statistics": atomic_cells,
        "operator_decomposition": {
            connective: {
                "direct_auroc": atomic_connective[connective]["auroc"],
                "reconstructed_auroc": reconstructed[connective],
                "absolute_difference": reconstruction_differences[connective],
            }
            for connective in ("AND", "OR")
        },
    }
    compound_summary = {
        "model": "Qwen/Qwen3-8B",
        "role": "same-layer compound-truth positive control",
        "layer": LAYER,
        "split_source": str(PAIR_SPLIT.relative_to(REPO_ROOT)),
        "train_pairs": 400,
        "train_rows": 6400,
        "test_pairs": 100,
        "test_rows": 1600,
        "probe": {
            "type": "LogisticRegression", "penalty": "l2", "C": C,
            "fit_intercept": True, "solver": "lbfgs", "max_iter": 2000,
            "random_state": SEED, "preprocessing": "none",
            "n_iter": int(compound_probe.n_iter_[0]),
            "intercept": float(compound_probe.intercept_[0]),
            "coefficient_norm": float(np.linalg.norm(compound_probe.coef_[0])),
        },
        "overall": compound_overall,
        "by_connective": compound_connective,
        "pairwise_aurocs": compound_pairwise.to_dict(orient="records"),
    }

    with QWEN25_GATE_SUMMARY.open() as handle:
        q25_gate = json.load(handle)
    with QWEN3_GATE_SUMMARY.open() as handle:
        q3_gate = json.load(handle)
    q25_overall = pd.read_csv(QWEN25_OVERALL).set_index("probe")
    q25_conn = pd.read_csv(QWEN25_CONNECTIVE).set_index(["probe", "connective"])
    q25_geometry = pd.read_csv(QWEN25_GEOMETRY).set_index(["probe", "connective"])
    compound_lookup = compound_pairwise.set_index(["connective", "cell_high", "cell_low"])["auroc"]
    comparison_values = [
        ("Atomic selected layer", q25_gate["selected_layer"], q3_gate["selected_layer"]),
        ("Atomic held-out AUROC", q25_gate["selected_layer_metrics"]["overall_auroc"], q3_gate["selected_layer_metrics"]["overall_auroc"]),
        ("facts+neg_facts AUROC", q25_gate["cross_topic"]["combined_auroc"], q3_gate["cross_topic_secondary_evaluation"]["combined_auroc"]),
        ("Frozen transfer overall", q25_overall.loc["atomic_union_to_compounds", "auroc"], atomic_overall["auroc"]),
        ("Frozen AND AUROC", q25_conn.loc[("atomic_union_to_compounds", "and"), "auroc"], atomic_connective["AND"]["auroc"]),
        ("Frozen OR AUROC", q25_conn.loc[("atomic_union_to_compounds", "or"), "auroc"], atomic_connective["OR"]["auroc"]),
        ("AND TT-vs-mixed", q25_geometry.loc[("Atomic", "AND"), "pairwise_auc_TT_mixed"], atomic_lookup.loc[("AND", "TT", "mixed")]),
        ("AND mixed-vs-FF", q25_geometry.loc[("Atomic", "AND"), "pairwise_auc_mixed_FF"], atomic_lookup.loc[("AND", "mixed", "FF")]),
        ("OR TT-vs-mixed", q25_geometry.loc[("Atomic", "OR"), "pairwise_auc_TT_mixed"], atomic_lookup.loc[("OR", "TT", "mixed")]),
        ("OR mixed-vs-FF", q25_geometry.loc[("Atomic", "OR"), "pairwise_auc_mixed_FF"], atomic_lookup.loc[("OR", "mixed", "FF")]),
        ("AND range/SD", q25_geometry.loc[("Atomic", "AND"), "range_over_sd"], geometry.loc["AND", "range_over_sd"]),
        ("OR range/SD", q25_geometry.loc[("Atomic", "OR"), "range_over_sd"], geometry.loc["OR", "range_over_sd"]),
        ("OR/AND range ratio", q25_geometry.loc[("Atomic", "OR"), "total_range"] / q25_geometry.loc[("Atomic", "AND"), "total_range"], geometry_ratios["OR_to_AND_range_ratio"]),
        ("Compound-trained AND AUROC", q25_conn.loc[("compound_trained_to_compounds", "and"), "auroc"], compound_connective["AND"]["auroc"]),
        ("Compound-trained OR AUROC", q25_conn.loc[("compound_trained_to_compounds", "or"), "auroc"], compound_connective["OR"]["auroc"]),
    ]
    comparison = pd.DataFrame(comparison_values, columns=["metric", "Qwen2.5-7B", "Qwen3-8B"])

    heldout_columns = [
        "r1_row_index", "topic", "entity_pair_id", "statement", "conjunctA", "conjunctB",
        "labelA", "labelB", "cell", "connective", "ordering", "global_truth",
        "frozen_atomic_score", "compound_trained_score",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    heldout[heldout_columns].to_csv(OUTPUT_DIR / "heldout_scores.csv", index=False)
    all_pairwise.to_csv(OUTPUT_DIR / "pairwise_aurocs.csv", index=False)
    with (OUTPUT_DIR / "frozen_transfer_summary.json").open("w") as handle:
        json.dump(frozen_summary, handle, indent=2)
        handle.write("\n")
    with (OUTPUT_DIR / "score_geometry.json").open("w") as handle:
        json.dump(geometry_output, handle, indent=2)
        handle.write("\n")
    with (OUTPUT_DIR / "compound_probe_summary.json").open("w") as handle:
        json.dump(compound_summary, handle, indent=2)
        handle.write("\n")
    comparison.to_csv(OUTPUT_DIR / "qwen25_vs_qwen3_comparison.csv", index=False)

    print("FROZEN", json.dumps(frozen_summary, indent=2))
    print("GEOMETRY", json.dumps(geometry_output, indent=2))
    print("COMPOUND", json.dumps(compound_summary, indent=2))
    print("COMPARISON\n" + comparison.to_string(index=False))


if __name__ == "__main__":
    main()
