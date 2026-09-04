"""Cached-data checks for qualitative examples, surface position, CIs, and layers."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "results" / "paper_round2_checks"
SEED_EXAMPLES = 42
BOOTSTRAP_SEED = 0
N_BOOTSTRAP = 2000
TOPICS = ["cities", "sp_en_trans", "inventors", "element_symb", "animal_class"]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    shown = frame[columns].fillna("").astype(str)
    header = "| " + " | ".join(columns) + " |"
    rule = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = ["| " + " | ".join(row) + " |" for row in shown.to_numpy()]
    return "\n".join([header, rule, *rows])


def build_random_examples() -> None:
    rows = []
    for topic in TOPICS:
        for form, name in (("affirmative", topic), ("negated", f"neg_{topic}")):
            path = ROOT / "data" / "tiu_datasets" / f"{name}.csv"
            sampled = pd.read_csv(path).sample(n=1, random_state=SEED_EXAMPLES).iloc[0]
            rows.append({
                "example_type": "primary_atomic", "category": f"{topic}_{form}",
                "source_path": str(path.relative_to(ROOT)), "topic": topic, "form": form,
                "statement": sampled["statement"], "label": int(sampled["label"]),
            })
    for name in ("facts", "neg_facts"):
        path = ROOT / "data" / "tiu_datasets" / f"{name}.csv"
        sampled = pd.read_csv(path).sample(n=1, random_state=SEED_EXAMPLES).iloc[0]
        rows.append({
            "example_type": "heldout_atomic", "category": name,
            "source_path": str(path.relative_to(ROOT)), "topic": "heldout_facts",
            "form": "affirmative" if name == "facts" else "negated",
            "statement": sampled["statement"], "label": int(sampled["label"]),
        })
    r1_path = ROOT / "data" / "r1_quadruples.csv"
    r1 = pd.read_csv(r1_path)
    for connective in ("and", "or"):
        for cell in ("TT", "TF", "FT", "FF"):
            sampled = r1.loc[
                r1["connective"].eq(connective) & r1["cell"].eq(cell)
            ].sample(n=1, random_state=SEED_EXAMPLES).iloc[0]
            rows.append({
                "example_type": "r1_compound", "category": f"{connective.upper()}-{cell}",
                "source_path": str(r1_path.relative_to(ROOT)), "topic": sampled["topic"],
                "statement": sampled["statement"], "conjunctA": sampled["conjunctA"],
                "conjunctB": sampled["conjunctB"], "labelA": int(sampled["labelA"]),
                "labelB": int(sampled["labelB"]), "cell": sampled["cell"],
                "connective": sampled["connective"], "ordering": sampled["ordering"],
            })
    output = pd.DataFrame(rows)
    output.to_csv(OUT / "random_examples.csv", index=False)

    # Verify exact row-wise affirmative -> negated transformations in the pinned CSVs.
    transformations = {
        "cities": lambda s: s.replace(" is in ", " is not in "),
        "sp_en_trans": lambda s: s.replace(" means ", " does not mean "),
        "inventors": lambda s: s.replace(" lived in ", " did not live in "),
        "element_symb": lambda s: s.replace(" has the symbol ", " does not have the symbol "),
        "animal_class": lambda s: re.sub(r" is (a|an) ", r" is not \1 ", s),
    }
    for topic, transform in transformations.items():
        affirmative = pd.read_csv(ROOT / "data" / "tiu_datasets" / f"{topic}.csv")
        negated = pd.read_csv(ROOT / "data" / "tiu_datasets" / f"neg_{topic}.csv")
        if not affirmative["statement"].map(transform).equals(negated["statement"]):
            raise ValueError(f"{topic}: negated template validation failed")
        if not (1 - affirmative["label"]).equals(negated["label"]):
            raise ValueError(f"{topic}: negated labels are not exact complements")

    atomic = output.loc[output["example_type"].ne("r1_compound")]
    compounds = output.loc[output["example_type"].eq("r1_compound")]
    md = f"""# Random qualitative examples

Sampling used seed 42 independently within every requested stratum. For each primary topic,
one row was sampled from its affirmative CSV and one from its negated CSV. One row was
sampled independently from `facts.csv` and `neg_facts.csv`. One R1 row was sampled
independently from each connective-by-cell stratum. No sampled row was replaced.

## Atomic examples

{markdown_table(atomic, ['category', 'topic', 'form', 'statement', 'label', 'source_path'])}

## R1 compound examples

{markdown_table(compounds, ['category', 'statement', 'topic', 'conjunctA', 'conjunctB', 'labelA', 'labelB', 'cell', 'connective', 'ordering'])}

## Template and generation audit

The five affirmative/negated pairs were validated row-by-row against the pinned files in
`data/tiu_datasets/`; the repository contains no separate generator for those source CSVs.
Their exact transformations are:

- cities: `The city of {{entity}} is in {{object}}.` / `The city of {{entity}} is not in {{object}}.`
- sp_en_trans: `The Spanish word '{{entity}}' means '{{object}}'.` / `The Spanish word '{{entity}}' does not mean '{{object}}'.`
- inventors: `{{entity}} lived in {{object}}.` / `{{entity}} did not live in {{object}}.`
- element_symb: `{{entity}} has the symbol {{object}}.` / `{{entity}} does not have the symbol {{object}}.`
- animal_class: `The {{entity}} is a/an {{object}}.` / `The {{entity}} is not a/an {{object}}.`

The entity-recognition patterns for both forms are in `src/data.py::ENTITY_PATTERNS`.
The R1 affirmative templates are defined in `scripts/01_generate_r1_r2_datasets.py::TOPICS`.
`build_atoms` generates a fixed false constituent by replacing the correct object with a
randomly selected different real object from the same topic; these false constituents are
wrong-object affirmatives, never negations. `join_binary` forms
`{{first without period}} and/or {{second without period}}.` and lowercases only a leading
`The` in the second conjunct. `gen_binary` emits both AB and BA surface orders while `_row`
retains canonical conjunctA/conjunctB and labelA/labelB. Model extraction uses raw statement
text with no chat template, as documented and implemented in `src/extract.py::extract_acts`.
"""
    (OUT / "random_examples.md").write_text(md)


def surface_class(cell: str, ordering: str) -> str:
    mapping = {
        ("TF", "AB"): "FIRST_TRUE_SECOND_FALSE",
        ("TF", "BA"): "FIRST_FALSE_SECOND_TRUE",
        ("FT", "AB"): "FIRST_FALSE_SECOND_TRUE",
        ("FT", "BA"): "FIRST_TRUE_SECOND_FALSE",
    }
    return mapping[(cell, ordering)]


def build_surface_check() -> None:
    sources = [
        ("Qwen2.5-7B-Instruct", ROOT / "results/r1/probe_transfer_comparison_predictions.csv", "atomic_union_to_compounds_score"),
        ("Qwen3-8B", ROOT / "results/qwen3_8b_r1_core/heldout_scores.csv", "frozen_atomic_score"),
    ]
    rows = []
    for model, path, score_col in sources:
        frame = pd.read_csv(path)
        frame["connective"] = frame["connective"].str.upper()
        mixed = frame.loc[frame["cell"].isin(["TF", "FT"])].copy()
        mixed["surface_class"] = [
            surface_class(cell, ordering) for cell, ordering in zip(mixed["cell"], mixed["ordering"])
        ]
        for connective in ("AND", "OR"):
            subset = mixed.loc[mixed["connective"].eq(connective)]
            high = subset.loc[subset["surface_class"].eq("FIRST_TRUE_SECOND_FALSE"), score_col]
            low = subset.loc[subset["surface_class"].eq("FIRST_FALSE_SECOND_TRUE"), score_col]
            labels = np.r_[np.ones(len(high)), np.zeros(len(low))]
            values = np.r_[high.to_numpy(), low.to_numpy()]
            rows.append({
                "model": model, "connective": connective,
                "comparison": "FIRST_TRUE_SECOND_FALSE_vs_FIRST_FALSE_SECOND_TRUE",
                "n_first_true_second_false": len(high),
                "mean_first_true_second_false": float(high.mean()),
                "sd_first_true_second_false": float(high.std(ddof=1)),
                "n_first_false_second_true": len(low),
                "mean_first_false_second_true": float(low.mean()),
                "sd_first_false_second_true": float(low.std(ddof=1)),
                "auroc": float(roc_auc_score(labels, values)),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "surface_position_check.csv", index=False)
    md = f"""# TF/FT surface-position check

In `data/r1_quadruples.csv`, `labelA`, `labelB`, and `cell` describe canonical
`conjunctA` and `conjunctB` before surface ordering. This follows
`scripts/01_generate_r1_r2_datasets.py::gen_binary`: A and B are selected first, `_row`
stores their canonical texts and labels, and only `first, second` are swapped when
`ordering == \"BA\"`. Therefore BA reverses surface position relative to canonical A/B.

The existing canonical TF-vs-FT AUROC is not a valid first-surface-vs-second-surface test:
both canonical classes contain both surface configurations because AB and BA are balanced.

Exact mapping:

| Cell | Ordering | Surface class |
|---|---|---|
| TF | AB | FIRST_TRUE_SECOND_FALSE |
| TF | BA | FIRST_FALSE_SECOND_TRUE |
| FT | AB | FIRST_FALSE_SECOND_TRUE |
| FT | BA | FIRST_TRUE_SECOND_FALSE |

AUROC is oriented as the probability that FIRST_TRUE_SECOND_FALSE receives a higher frozen
atomic-probe score than FIRST_FALSE_SECOND_TRUE. The original 100-pair/1,600-row held-out
files were used without resplitting.

{markdown_table(result, result.columns.tolist())}
"""
    (OUT / "surface_position_check.md").write_text(md)


def build_qwen3_bootstrap() -> None:
    bootstrap_module = load_module(ROOT / "scripts/mechanism/13_pair_level_bootstrap.py", "old_bootstrap")
    pairwise = load_module(ROOT / "scripts/mechanism/02_atomic_pairwise_auc.py", "pairwise").grouped_pairwise_auc
    dprime = load_module(ROOT / "scripts/mechanism/05_boundary_dprime.py", "dprime").dprime
    frame = pd.read_csv(ROOT / "results/qwen3_8b_r1_core/heldout_scores.csv")
    frame["connective"] = frame["connective"].str.upper()
    frame["atomic_union_to_compounds_score"] = frame["frozen_atomic_score"]
    pairs = frame["entity_pair_id"].drop_duplicates().to_numpy(dtype=object)
    if len(frame) != 1600 or len(pairs) != 100 or not frame.groupby("entity_pair_id").size().eq(16).all():
        raise ValueError("Qwen3 held-out pair structure is invalid")
    blocks = {pair_id: np.flatnonzero(frame["entity_pair_id"].to_numpy() == pair_id) for pair_id in pairs}
    observed = bootstrap_module.calculate_metrics(frame, pairwise, dprime)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = []
    for replicate in range(N_BOOTSTRAP):
        sampled = rng.choice(pairs, size=100, replace=True)
        indices = np.concatenate([blocks[pair_id] for pair_id in sampled])
        replicate_frame = frame.iloc[indices]
        if len(indices) != 1600:
            raise ValueError("Bootstrap row count changed")
        actual = replicate_frame["entity_pair_id"].value_counts().sort_index()
        expected = (pd.Series(sampled).value_counts() * 16).sort_index()
        if not actual.equals(expected):
            raise ValueError("Bootstrap multiplicity was not preserved")
        draws.append(bootstrap_module.calculate_metrics(replicate_frame, pairwise, dprime))

    draw_frame = pd.DataFrame(draws)
    requested = [
        "and_truth_auroc", "or_truth_auroc", "and_minus_or_auroc",
        "or_mixed_vs_FF_auroc", "OR_to_AND_range_ratio",
        "OR_to_AND_pooled_sd_ratio", "or_TT_vs_mixed_auroc",
        "and_TT_vs_mixed_auroc", "and_mixed_vs_FF_auroc",
    ]
    q25 = pd.read_csv(ROOT / "results/r1/mechanism/bootstrap_pair_level_summary.csv").set_index("metric")
    rows = []
    for metric in requested:
        values = draw_frame[metric].to_numpy()
        rows.append({
            "metric": metric,
            "qwen3_observed": observed[metric],
            "qwen3_ci_2_5": float(np.quantile(values, 0.025)),
            "qwen3_ci_97_5": float(np.quantile(values, 0.975)),
            "qwen3_bootstrap_mean": float(values.mean()),
            "qwen3_bootstrap_sd": float(values.std(ddof=1)),
            "qwen25_observed": float(q25.loc[metric, "observed"]),
            "qwen25_ci_2_5": float(q25.loc[metric, "ci_2_5"]),
            "qwen25_ci_97_5": float(q25.loc[metric, "ci_97_5"]),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "qwen3_bootstrap_ci.csv", index=False)
    metadata = {
        "n_bootstrap": N_BOOTSTRAP, "seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "canonical_pair", "sampled_pair_instances": 100,
        "rows_per_pair": 16, "rows_per_replicate": 1600,
        "multiplicity_preserved": True, "undefined_metrics": 0,
        "implementation_source": "scripts/mechanism/13_pair_level_bootstrap.py",
        "qwen3_score_source": "results/qwen3_8b_r1_core/heldout_scores.csv",
        "qwen25_ci_source": "results/r1/mechanism/bootstrap_pair_level_summary.csv",
        "results": summary.to_dict(orient="records"), "status": "PASS",
    }
    with (OUT / "qwen3_bootstrap_ci.json").open("w") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


def build_layer_manifest() -> None:
    path = ROOT / "results/r1/mechanism/all_layers_atomic_transfer.csv"
    frame = pd.read_csv(path)
    high = frame.loc[frame["atomic_heldout_auroc"].ge(0.99)]
    high_layers = high["layer"].astype(int).tolist()
    layer22 = frame.loc[frame["layer"].eq(22)].iloc[0]
    all_and_gt_or = int((frame["and_truth_auroc"] > frame["or_truth_auroc"]).sum())
    high_and_gt_or = int((high["and_truth_auroc"] > high["or_truth_auroc"]).sum())
    all_boundary = int((frame["or_TT_vs_mixed_auroc"] > frame["or_mixed_vs_FF_auroc"]).sum())
    high_boundary = int((high["or_TT_vs_mixed_auroc"] > high["or_mixed_vs_FF_auroc"]).sum())
    columns = ", ".join(f"`{column}`" for column in frame.columns)
    md = f"""# Qwen2.5 all-layer sweep manifest

## Sources

- Result table: `results/r1/mechanism/all_layers_atomic_transfer.csv`
- Summary: `results/r1/mechanism/all_layers_atomic_transfer_summary.json`
- Source script: `scripts/mechanism/12_all_layers_atomic_transfer.py`
- Atomic gate metrics: `results/qwen_union_probe_gate/per_layer_metrics.csv`
- Existing atomic-only gate figure copied from: `figures/qwen_union_probe_gate_layer_auroc.png`

The existing PNG shows the atomic gate AUROCs. The cached all-layer compound-transfer table
has no separate pre-existing compound-sweep PNG in the repository, so no extraction or
model computation was rerun.

## Available columns

{columns}

## Verified results

- Atomic-AUROC >= 0.99 plateau: layers `{high_layers}`; this is exactly layers 9–27: **{high_layers == list(range(9, 28))}**.
- AND > OR transfer AUROC over all layers: **{all_and_gt_or}/28**.
- AND > OR over the >=0.99 plateau: **{high_and_gt_or}/{len(high)}**.
- OR TT–mixed > OR mixed–FF over all layers: **{all_boundary}/28**.
- OR TT–mixed > OR mixed–FF over the >=0.99 plateau: **{high_boundary}/{len(high)}**.
- Plateau median AND transfer AUROC: **{high['and_truth_auroc'].median():.12f}**.
- Plateau median OR transfer AUROC: **{high['or_truth_auroc'].median():.12f}**.
- Layer 22 AND transfer AUROC: **{layer22['and_truth_auroc']:.12f}**.
- Layer 22 OR transfer AUROC: **{layer22['or_truth_auroc']:.12f}**.
- Layer 22 OR TT–mixed AUROC: **{layer22['or_TT_vs_mixed_auroc']:.12f}**.
- Layer 22 OR mixed–FF AUROC: **{layer22['or_mixed_vs_FF_auroc']:.12f}**.
"""
    (OUT / "layer_sweep_manifest.md").write_text(md)
    shutil.copyfile(
        ROOT / "figures/qwen_union_probe_gate_layer_auroc.png",
        OUT / "qwen25_layer_sweep.png",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    build_random_examples()
    build_surface_check()
    build_qwen3_bootstrap()
    build_layer_manifest()
    print("Created:")
    for path in sorted(OUT.iterdir()):
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
