"""Atomic-only union-probe layer selection for the Qwen3-8B replication arm.

This script deliberately has no R1/compound inputs. It reproduces the existing
Qwen2.5 atomic gate on the cached Qwen3 atomic activations and freezes the probe
selected solely by held-out atomic AUROC.
"""

from __future__ import annotations

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
from sklearn.metrics import roc_auc_score

from src.data import group_key
from src.probes import split_indices, train_layer_probe


MODEL = "Qwen/Qwen3-8B"
TOPICS = ["cities", "sp_en_trans", "inventors", "element_symb", "animal_class"]
FORMS = ["affirmative", "negated"]
N_LAYERS = 36
HIDDEN_SIZE = 4096
TEST_SIZE = 0.2
RANDOM_STATE = 0
C = 0.1
DATASETS_DIR = REPO_ROOT / "data" / "tiu_datasets"
ACTS_ROOT = REPO_ROOT / "acts" / "qwen3_8b"
ATOMIC_DIR = ACTS_ROOT / "atomic"
HELDOUT_DIR = ACTS_ROOT / "heldout_atomic"
RESULTS_DIR = REPO_ROOT / "results" / "qwen3_8b_union_probe_gate"
QWEN25_SPLIT = REPO_ROOT / "results" / "qwen_union_probe_gate" / "split_metadata.csv"


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    classes = np.unique(labels)
    if not np.array_equal(classes, [0, 1]):
        raise ValueError(f"AUROC subset must contain labels [0,1], got {classes.tolist()}")
    return float(roc_auc_score(labels, scores))


def load_checkpoint(name: str, directory: Path):
    source = pd.read_csv(DATASETS_DIR / f"{name}.csv")
    metadata_path = directory / f"{name}_qwen3_8b_metadata.csv"
    activation_path = directory / f"{name}_qwen3_8b_acts.npy"
    metadata = pd.read_csv(metadata_path)
    if not metadata.equals(source):
        raise ValueError(f"{name}: Qwen3 metadata does not exactly equal source CSV")
    activations = np.load(activation_path, mmap_mode="r")
    if activations.shape != (len(source), N_LAYERS, HIDDEN_SIZE):
        raise ValueError(f"{name}: unexpected activation shape {activations.shape}")
    if activations.dtype != np.float16:
        raise ValueError(f"{name}: expected float16 activations, got {activations.dtype}")
    labels = source["label"].to_numpy(dtype=int)
    if not np.array_equal(np.unique(labels), [0, 1]):
        raise ValueError(f"{name}: labels are not exactly binary [0,1]")
    return activations, source, labels, activation_path, metadata_path


def build_primary_data():
    blocks = []
    split_rows = []
    source_files = []
    for topic in TOPICS:
        topic_blocks = []
        entity_sets = {}
        for form in FORMS:
            name = topic if form == "affirmative" else f"neg_{topic}"
            acts, metadata, labels, acts_path, metadata_path = load_checkpoint(name, ATOMIC_DIR)
            entities = group_key(name, datasets_dir=str(DATASETS_DIR)).astype(str)
            entity_sets[form] = set(entities)
            topic_blocks.append((name, form, acts, labels, entities))
            source_files.append({
                "name": name,
                "source_csv": str((DATASETS_DIR / f"{name}.csv").relative_to(REPO_ROOT)),
                "activation_file": str(acts_path.relative_to(REPO_ROOT)),
                "metadata_file": str(metadata_path.relative_to(REPO_ROOT)),
                "rows": int(len(metadata)),
            })
        if entity_sets["affirmative"] != entity_sets["negated"]:
            raise ValueError(f"{topic}: affirmative and negated entity sets differ")

        unique_entities = np.array(sorted(entity_sets["affirmative"]), dtype=object)
        train_entity_idx, test_entity_idx = split_indices(
            len(unique_entities),
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            groups=unique_entities,
        )
        train_entities = set(unique_entities[train_entity_idx])
        test_entities = set(unique_entities[test_entity_idx])
        if train_entities & test_entities:
            raise ValueError(f"{topic}: entity leakage detected")

        for name, form, acts, labels, entities in topic_blocks:
            partitions = np.where(np.isin(entities, list(train_entities)), "train", "test")
            if not np.all(np.isin(entities, list(train_entities | test_entities))):
                raise ValueError(f"{name}: some entities were not assigned")
            blocks.append({
                "name": name,
                "topic": topic,
                "form": form,
                "acts": acts,
                "labels": labels,
                "entities": entities,
                "partitions": partitions,
            })
            for row_index, (entity, label, partition) in enumerate(
                zip(entities, labels, partitions)
            ):
                split_rows.append({
                    "dataset": name,
                    "topic": topic,
                    "form": form,
                    "row_index": row_index,
                    "entity": entity,
                    "label": int(label),
                    "partition": partition,
                })

    labels = np.concatenate([block["labels"] for block in blocks])
    topics = np.concatenate([[block["topic"]] * len(block["labels"]) for block in blocks])
    forms = np.concatenate([[block["form"]] * len(block["labels"]) for block in blocks])
    partitions = np.concatenate([block["partitions"] for block in blocks])
    train_idx = np.flatnonzero(partitions == "train")
    test_idx = np.flatnonzero(partitions == "test")
    return (
        blocks, labels, topics, forms, train_idx, test_idx,
        pd.DataFrame(split_rows), source_files,
    )


def layer_matrix(blocks, layer: int) -> np.ndarray:
    matrix = np.concatenate(
        [np.asarray(block["acts"][:, layer, :]) for block in blocks], axis=0
    )
    if matrix.shape[1] != HIDDEN_SIZE or not np.isfinite(matrix).all():
        raise ValueError(f"Layer {layer}: invalid pooled activation matrix")
    return matrix


def evaluate_scores(labels, scores, topics, forms, test_idx):
    held_labels = labels[test_idx]
    held_scores = scores[test_idx]
    held_topics = topics[test_idx]
    held_forms = forms[test_idx]
    metrics = {
        "overall_auroc": auc(held_labels, held_scores),
        "overall_accuracy_at_0": float(np.mean((held_scores >= 0).astype(int) == held_labels)),
        "affirmative_auroc": auc(
            held_labels[held_forms == "affirmative"], held_scores[held_forms == "affirmative"]
        ),
        "negated_auroc": auc(
            held_labels[held_forms == "negated"], held_scores[held_forms == "negated"]
        ),
    }
    for topic in TOPICS:
        mask = held_topics == topic
        metrics[f"topic_{topic}_auroc"] = auc(held_labels[mask], held_scores[mask])
    return metrics


def main() -> None:
    (
        blocks, labels, topics, forms, train_idx, test_idx, split_df, source_files
    ) = build_primary_data()
    saved_qwen25_split = pd.read_csv(QWEN25_SPLIT)
    if not split_df.equals(saved_qwen25_split):
        raise ValueError(
            "Qwen3 split membership does not exactly reproduce the saved Qwen2.5 split"
        )
    if len(labels) != 5212 or len(train_idx) != 4170 or len(test_idx) != 1042:
        raise ValueError(
            f"Unexpected primary row counts: total={len(labels)}, "
            f"train={len(train_idx)}, test={len(test_idx)}"
        )

    train_groups = set(
        split_df.loc[split_df["partition"].eq("train"), ["topic", "entity"]]
        .itertuples(index=False, name=None)
    )
    test_groups = set(
        split_df.loc[split_df["partition"].eq("test"), ["topic", "entity"]]
        .itertuples(index=False, name=None)
    )
    if train_groups & test_groups:
        raise ValueError("Entity leakage detected after pooling topics")

    rows = []
    probes = []
    for layer in range(N_LAYERS):
        X = layer_matrix(blocks, layer)
        probe, helper_accuracy = train_layer_probe(X, labels, train_idx, test_idx)
        expected_options = {
            "penalty": "l2", "C": C, "solver": "lbfgs", "fit_intercept": True,
            "max_iter": 2000, "class_weight": None,
        }
        params = probe.get_params()
        if any(params[key] != value for key, value in expected_options.items()):
            raise ValueError(f"Layer {layer}: logistic options differ: {params}")
        if not np.array_equal(probe.classes_, [0, 1]):
            raise ValueError(f"Layer {layer}: unexpected class orientation")
        scores = probe.decision_function(X)
        direct_scores = X @ probe.coef_[0] + probe.intercept_[0]
        if not np.allclose(scores, direct_scores, rtol=0, atol=1e-12):
            raise ValueError(f"Layer {layer}: decision_function is not raw affine score")
        metrics = evaluate_scores(labels, scores, topics, forms, test_idx)
        if not np.isclose(
            metrics["overall_accuracy_at_0"], helper_accuracy, rtol=0, atol=1e-15
        ):
            raise ValueError(f"Layer {layer}: score-zero accuracy differs from probe.score")
        rows.append({"layer": layer, **metrics, "n_iter": int(probe.n_iter_[0])})
        probes.append(probe)
        print(
            f"layer {layer:2d}: AUROC={metrics['overall_auroc']:.6f} "
            f"accuracy={metrics['overall_accuracy_at_0']:.6f} "
            f"affirmative={metrics['affirmative_auroc']:.6f} "
            f"negated={metrics['negated_auroc']:.6f}",
            flush=True,
        )

    layer_metrics = pd.DataFrame(rows)
    selected_row_index = int(layer_metrics["overall_auroc"].idxmax())
    selected_layer = int(layer_metrics.loc[selected_row_index, "layer"])
    selected_probe = probes[selected_layer]
    selected_metrics = layer_metrics.loc[selected_row_index].to_dict()

    heldout_blocks = []
    heldout_labels = []
    heldout_forms = []
    heldout_sources = []
    for name in ("facts", "neg_facts"):
        acts, metadata, y, acts_path, metadata_path = load_checkpoint(name, HELDOUT_DIR)
        heldout_blocks.append(np.asarray(acts[:, selected_layer, :]))
        heldout_labels.append(y)
        heldout_forms.extend([name] * len(y))
        heldout_sources.append({
            "name": name,
            "source_csv": str((DATASETS_DIR / f"{name}.csv").relative_to(REPO_ROOT)),
            "activation_file": str(acts_path.relative_to(REPO_ROOT)),
            "metadata_file": str(metadata_path.relative_to(REPO_ROOT)),
            "rows": int(len(metadata)),
        })
    heldout_X = np.concatenate(heldout_blocks, axis=0)
    heldout_y = np.concatenate(heldout_labels)
    heldout_forms = np.asarray(heldout_forms)
    if not np.isfinite(heldout_X).all():
        raise ValueError("Secondary held-out activations contain NaN or infinity")
    heldout_scores = selected_probe.decision_function(heldout_X)
    cross_topic = {
        "combined_auroc": auc(heldout_y, heldout_scores),
        "facts_auroc": auc(
            heldout_y[heldout_forms == "facts"], heldout_scores[heldout_forms == "facts"]
        ),
        "neg_facts_auroc": auc(
            heldout_y[heldout_forms == "neg_facts"],
            heldout_scores[heldout_forms == "neg_facts"],
        ),
    }

    entity_verification = []
    for topic in TOPICS:
        subset = split_df.loc[split_df["topic"].eq(topic)]
        topic_train = sorted(subset.loc[subset["partition"].eq("train"), "entity"].unique())
        topic_test = sorted(subset.loc[subset["partition"].eq("test"), "entity"].unique())
        intersection = sorted(set(topic_train) & set(topic_test))
        if intersection:
            raise ValueError(f"{topic}: entity leakage found during final verification")
        entity_verification.append({
            "topic": topic,
            "n_train_entities": len(topic_train),
            "n_test_entities": len(topic_test),
            "entity_intersection": intersection,
        })

    summary = {
        "model": MODEL,
        "replication_arm": "Qwen3-8B atomic-only layer selection",
        "probe": {
            "type": "sklearn.linear_model.LogisticRegression",
            "penalty": "l2",
            "C": C,
            "solver": "lbfgs",
            "fit_intercept": True,
            "max_iter": 2000,
            "class_weight": None,
            "preprocessing": "none",
            "score": "decision_function = X @ coef + intercept",
            "accuracy_convention": "decision score >= 0 predicts label 1",
        },
        "split": {
            "method": "GroupShuffleSplit independently within topic",
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "membership_exactly_matches_saved_qwen2_5_split": True,
        },
        "topics": TOPICS,
        "forms": FORMS,
        "n_layers": N_LAYERS,
        "hidden_size": HIDDEN_SIZE,
        "n_atomic_rows": int(len(labels)),
        "n_train_rows": int(len(train_idx)),
        "n_test_rows": int(len(test_idx)),
        "n_train_unique_topic_entities": len(train_groups),
        "n_test_unique_topic_entities": len(test_groups),
        "zero_entity_leakage": True,
        "selected_layer": selected_layer,
        "selection_rule": "maximum overall held-out atomic AUROC; first layer on exact tie",
        "selection_data": "primary atomic held-out partition only",
        "selected_layer_metrics": {
            key: (int(value) if key in {"layer", "n_iter"} else float(value))
            for key, value in selected_metrics.items()
        },
        "cross_topic_secondary_evaluation": cross_topic,
        "entity_split_verification": entity_verification,
        "primary_sources": source_files,
        "secondary_heldout_sources": heldout_sources,
        "compound_data_loaded": False,
        "probe_frozen_before_compound_evaluation": True,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    layer_metrics.to_csv(RESULTS_DIR / "per_layer_metrics.csv", index=False)
    split_df.to_csv(RESULTS_DIR / "split_metadata.csv", index=False)
    np.savez_compressed(
        RESULTS_DIR / "selected_probe.npz",
        model=np.asarray(MODEL),
        layer=np.asarray(selected_layer),
        coef=selected_probe.coef_,
        intercept=selected_probe.intercept_,
        classes=selected_probe.classes_,
        C=np.asarray(selected_probe.C),
        n_iter=selected_probe.n_iter_,
        solver=np.asarray(selected_probe.solver),
        penalty=np.asarray(selected_probe.penalty),
        fit_intercept=np.asarray(selected_probe.fit_intercept),
        preprocessing=np.asarray("none"),
        random_state=np.asarray(-1),
    )
    with (RESULTS_DIR / "summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps({
        "selected_layer": selected_layer,
        "selected_layer_metrics": summary["selected_layer_metrics"],
        "cross_topic_secondary_evaluation": cross_topic,
    }, indent=2))


if __name__ == "__main__":
    main()
