"""Five-topic Qwen atomic union-probe gate.

Pools affirmative and negated atomic examples while splitting entities within
each topic. The split is constructed once and reused for all 28 layers.
Outputs are saved under results/qwen_union_probe_gate and figures/.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Make direct execution from the repository root work without requiring an
# editable install or a manually supplied PYTHONPATH.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import group_key, load_statements
from src.probes import split_indices, train_layer_probe


TOPICS = ["cities", "sp_en_trans", "inventors", "element_symb", "animal_class"]
FORMS = ["affirmative", "negated"]
DEFAULT_DATASETS_DIR = "data/tiu_datasets"
DEFAULT_ACTS_DIR = "acts"
DEFAULT_RESULTS_DIR = "results/qwen_union_probe_gate"
DEFAULT_FIGURE = "figures/qwen_union_probe_gate_layer_auroc.png"
TEST_SIZE = 0.2
RANDOM_STATE = 0
C = 0.1


def auc(labels, scores):
    """AUROC with an explicit class-presence check for clearer failures."""
    classes = np.unique(labels)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError(f"AUROC subset must contain labels [0, 1], got {classes.tolist()}")
    return float(roc_auc_score(labels, scores))


def load_atomic(name, acts_dir, datasets_dir):
    """Load a memory-mapped activation checkpoint and its pinned source CSV."""
    acts = np.load(Path(acts_dir) / f"{name}.npy", mmap_mode="r")
    meta = load_statements(name, datasets_dir=datasets_dir)
    labels = meta["label"].to_numpy(dtype=int)
    if acts.ndim != 3:
        raise ValueError(f"{name}: expected [rows, layers, hidden], got {acts.shape}")
    if len(meta) != acts.shape[0]:
        raise ValueError(f"{name}: {len(meta)} metadata rows != {acts.shape[0]} activations")
    if not set(np.unique(labels)).issubset({0, 1}):
        raise ValueError(f"{name}: labels are not binary")
    return acts, meta, labels


def build_data(acts_dir, datasets_dir):
    """Load checkpoints and construct one independently grouped split per topic."""
    blocks = []
    split_rows = []
    offset = 0
    expected_shape = None

    for topic in TOPICS:
        topic_blocks = []
        entity_forms = {}
        for form in FORMS:
            name = topic if form == "affirmative" else f"neg_{topic}"
            acts, meta, labels = load_atomic(name, acts_dir, datasets_dir)
            shape = acts.shape[1:]
            if expected_shape is None:
                expected_shape = shape
            if shape != expected_shape:
                raise ValueError(f"{name}: layer/hidden shape {shape} != {expected_shape}")
            entities = group_key(name, datasets_dir=datasets_dir).astype(str)
            entity_forms[form] = set(entities)
            topic_blocks.append((name, form, acts, meta, labels, entities))

        if entity_forms["affirmative"] != entity_forms["negated"]:
            raise ValueError(f"{topic}: affirmative and negated entity sets differ")

        # Split unique entities once for the topic, then apply it to both forms.
        unique_entities = np.array(sorted(entity_forms["affirmative"]), dtype=object)
        entity_train, entity_test = split_indices(
            len(unique_entities), test_size=TEST_SIZE,
            random_state=RANDOM_STATE, groups=unique_entities,
        )
        train_entities = set(unique_entities[entity_train])
        test_entities = set(unique_entities[entity_test])
        intersection = train_entities & test_entities
        if intersection:
            raise AssertionError(f"{topic}: leaked entities: {sorted(intersection)}")

        for name, form, acts, meta, labels, entities in topic_blocks:
            partitions = np.where(np.isin(entities, list(train_entities)), "train", "test")
            if not np.all(np.isin(entities, list(train_entities | test_entities))):
                raise AssertionError(f"{name}: some entities were not assigned")
            blocks.append({
                "name": name, "topic": topic, "form": form, "acts": acts,
                "labels": labels, "entities": entities, "partitions": partitions,
                "offset": offset,
            })
            for row_idx, (entity, label, partition) in enumerate(
                zip(entities, labels, partitions)
            ):
                split_rows.append({
                    "dataset": name, "topic": topic, "form": form,
                    "row_index": row_idx, "entity": entity, "label": int(label),
                    "partition": partition,
                })
            offset += len(labels)

    labels = np.concatenate([b["labels"] for b in blocks])
    topics = np.concatenate([[b["topic"]] * len(b["labels"]) for b in blocks])
    forms = np.concatenate([[b["form"]] * len(b["labels"]) for b in blocks])
    partitions = np.concatenate([b["partitions"] for b in blocks])
    train_idx = np.flatnonzero(partitions == "train")
    test_idx = np.flatnonzero(partitions == "test")
    return blocks, labels, topics, forms, train_idx, test_idx, pd.DataFrame(split_rows)


def layer_matrix(blocks, layer):
    """Materialize only one pooled layer at a time to bound peak memory."""
    return np.concatenate([np.asarray(b["acts"][:, layer, :]) for b in blocks], axis=0)


def evaluate_scores(labels, scores, topics, forms, test_idx):
    held_labels = labels[test_idx]
    held_scores = scores[test_idx]
    held_topics = topics[test_idx]
    held_forms = forms[test_idx]
    metrics = {
        "overall_auroc": auc(held_labels, held_scores),
        "affirmative_auroc": auc(held_labels[held_forms == "affirmative"], held_scores[held_forms == "affirmative"]),
        "negated_auroc": auc(held_labels[held_forms == "negated"], held_scores[held_forms == "negated"]),
    }
    for topic in TOPICS:
        mask = held_topics == topic
        metrics[f"topic_{topic}_auroc"] = auc(held_labels[mask], held_scores[mask])
    return metrics


def save_plot(metrics, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(metrics["layer"], metrics["overall_auroc"], marker="o", label="Overall")
    ax.plot(metrics["layer"], metrics["affirmative_auroc"], marker="o", label="Affirmative")
    ax.plot(metrics["layer"], metrics["negated_auroc"], marker="o", label="Negated")
    ax.set(xlabel="Qwen layer", ylabel="Held-out AUROC", title="Five-topic atomic union probe")
    ax.set_xticks(metrics["layer"])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--acts-dir", default=DEFAULT_ACTS_DIR)
    parser.add_argument("--datasets-dir", default=DEFAULT_DATASETS_DIR)
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--figure", default=DEFAULT_FIGURE)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    blocks, labels, topics, forms, train_idx, test_idx, split_df = build_data(
        args.acts_dir, args.datasets_dir
    )
    n_layers = blocks[0]["acts"].shape[1]
    rows, probes = [], []
    for layer in range(n_layers):
        X = layer_matrix(blocks, layer)
        probe, _ = train_layer_probe(X, labels, train_idx, test_idx)
        scores = probe.decision_function(X)
        row = {"layer": layer, **evaluate_scores(labels, scores, topics, forms, test_idx)}
        rows.append(row)
        probes.append(probe)
        print(
            f"layer {layer:2d}: overall={row['overall_auroc']:.4f} "
            f"affirmative={row['affirmative_auroc']:.4f} "
            f"negated={row['negated_auroc']:.4f}", flush=True
        )

    metrics = pd.DataFrame(rows)
    # idxmax returns the first layer on an exact tie, making the rule deterministic.
    selected_row_idx = int(metrics["overall_auroc"].idxmax())
    selected_layer = int(metrics.loc[selected_row_idx, "layer"])
    selected_probe = probes[selected_layer]

    heldout_blocks = []
    heldout_labels = []
    heldout_forms = []
    for form, name in [("facts", "facts"), ("neg_facts", "neg_facts")]:
        acts, _, y = load_atomic(name, args.acts_dir, args.datasets_dir)
        if acts.shape[1:] != blocks[0]["acts"].shape[1:]:
            raise ValueError(f"{name}: activation shape incompatible with training data")
        heldout_blocks.append(np.asarray(acts[:, selected_layer, :]))
        heldout_labels.append(y)
        heldout_forms.extend([form] * len(y))
    heldout_X = np.concatenate(heldout_blocks)
    heldout_y = np.concatenate(heldout_labels)
    heldout_forms = np.asarray(heldout_forms)
    heldout_scores = selected_probe.decision_function(heldout_X)
    cross_topic = {
        "combined_auroc": auc(heldout_y, heldout_scores),
        "facts_auroc": auc(heldout_y[heldout_forms == "facts"], heldout_scores[heldout_forms == "facts"]),
        "neg_facts_auroc": auc(heldout_y[heldout_forms == "neg_facts"], heldout_scores[heldout_forms == "neg_facts"]),
    }

    metrics.to_csv(results_dir / "per_layer_metrics.csv", index=False)
    split_df.to_csv(results_dir / "split_metadata.csv", index=False)
    split_summary = []
    for topic in TOPICS:
        sub = split_df[split_df["topic"] == topic]
        train_entities = sorted(sub.loc[sub.partition == "train", "entity"].unique())
        test_entities = sorted(sub.loc[sub.partition == "test", "entity"].unique())
        split_summary.append({
            "topic": topic, "n_train_entities": len(train_entities),
            "n_test_entities": len(test_entities), "entity_intersection": [],
            "train_entities": train_entities, "test_entities": test_entities,
        })
    manifest = {
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "probe": {"type": "sklearn.linear_model.LogisticRegression", "penalty": "l2", "C": C, "max_iter": 2000},
        "split": {"method": "GroupShuffleSplit independently within topic", "test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "topics": TOPICS, "forms": FORMS, "n_layers": n_layers,
        "selected_layer": selected_layer,
        "selection_rule": "maximum overall held-out AUROC; first layer on exact tie",
        "selected_layer_metrics": {k: float(v) for k, v in metrics.loc[selected_row_idx].items() if k != "layer"},
        "cross_topic": cross_topic,
        "entity_split_verification": split_summary,
    }
    with open(results_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    np.savez_compressed(
        results_dir / "selected_probe.npz", layer=np.array(selected_layer),
        coef=selected_probe.coef_, intercept=selected_probe.intercept_,
        classes=selected_probe.classes_, C=np.array(selected_probe.C),
        n_iter=selected_probe.n_iter_,
    )
    save_plot(metrics, args.figure)
    print(json.dumps({"selected_layer": selected_layer, "cross_topic": cross_topic}, indent=2))


if __name__ == "__main__":
    main()
