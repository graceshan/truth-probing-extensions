"""Strict primary atomic TRAIN/VALIDATION loader; test label rows are skipped at CSV read."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

from src.atomic_probe_methods import check
from src.data import entity_key_from_statements
from src.entity_partitions import TOPICS, canonical_json, sha256


def hash_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def selected_labels(path, selected_indices, expected_statements):
    """Primary atomic CSVs have one physical record per line.

    First pass reads only statements. On the label pass, skip every unapproved
    record before column parsing; no test label validation, storage, or hashing.
    """
    allowed = set(map(int, selected_indices))
    frame = pd.read_csv(path, usecols=["statement", "label"], dtype=str, keep_default_na=False,
                        skiprows=lambda line: line > 0 and line - 1 not in allowed)
    check(frame.statement.tolist() == list(expected_statements), "selected CSV row alignment mismatch")
    check(frame.label.isin(["0", "1"]).all(), "nonbinary TRAIN/VALIDATION label")
    return frame.label.astype(int).to_numpy()


class MethodData:
    def __init__(self, root, config, model_key):
        self.root, self.config, self.model_key = Path(root), config, model_key
        self.model = config["models"][model_key]
        self.n_layers, self.hidden_size = self.model["n_layers"], self.model["hidden_size"]
        entity_path = self.root / config["entity_manifest"]
        entities = pd.read_csv(entity_path, usecols=["topic", "entity", "entity_id", "split"], dtype=str)
        check(not entities.duplicated(["topic", "entity"]).any(), "duplicate manifest entity")
        lookup = {(r.topic, r.entity): (r.entity_id, r.split) for r in entities.itertuples()}
        check(set(entities.split) <= {"train", "validation", "test"}, "invalid entity split")
        self.blocks, records, sources, assignments = {}, [], [], []
        for topic in TOPICS:
            for form in ("affirmative", "negated"):
                dataset = topic if form == "affirmative" else "neg_" + topic
                source_path = self.root / config["source_dir"] / f"{dataset}.csv"
                meta_path = self.root / self.model["metadata_pattern"].format(dataset=dataset)
                acts_path = self.root / self.model["activation_pattern"].format(dataset=dataset)
                source = pd.read_csv(source_path, usecols=["statement"], dtype=str, keep_default_na=False)
                meta = pd.read_csv(meta_path, usecols=["statement"], dtype=str, keep_default_na=False)
                check(source.equals(meta), "source/sidecar statement order mismatch")
                check(not source.statement.str.contains(r"[\r\n]", regex=True).any(),
                      "multiline primary atomic statements unsupported by strict row-skipping loader")
                natural_entities = entity_key_from_statements(topic, source.statement)
                check(all((topic, e) in lookup for e in natural_entities), "unassigned atomic entity")
                memberships = [lookup[topic, e] for e in natural_entities]
                check(set(natural_entities) == set(entities.loc[entities.topic == topic, "entity"]),
                      "atomic/manifest entity coverage differs")
                selected = [i for i, (_, split) in enumerate(memberships) if split in {"train", "validation"}]
                statements = source.statement.iloc[selected].tolist()
                labels = selected_labels(source_path, selected, statements)
                sidecar_labels = selected_labels(meta_path, selected, statements)
                check(np.array_equal(labels, sidecar_labels), "TRAIN/VALIDATION sidecar label mismatch")
                array = np.load(acts_path, mmap_mode="r", allow_pickle=False)
                check(array.shape == (len(source), self.n_layers, self.hidden_size) and array.dtype == np.float16,
                      "atomic activation header mismatch")
                del array
                for i, (entity_id, split) in enumerate(memberships):
                    assignments.append([dataset, i, entity_id, split])  # identities, never test labels
                for i, label, statement in zip(selected, labels, statements):
                    entity_id, split = memberships[i]
                    records.append({"dataset": dataset, "row_index": i, "statement": statement,
                                    "entity_id": entity_id, "topic": topic, "form": form,
                                    "split": split, "label": int(label)})
                stat = acts_path.stat()
                self.blocks[dataset] = acts_path
                sources.append({"dataset": dataset, "activation_file": str(acts_path.relative_to(self.root)),
                                "activation_size_bytes": stat.st_size, "activation_mtime_ns": stat.st_mtime_ns,
                                "activation_shape": [len(source), self.n_layers, self.hidden_size],
                                "selected_rows_statement_label_sha256": sha256(canonical_json([
                                    [i, statement, int(label)] for i, statement, label in zip(selected, statements, labels)]))})
        self.rows = pd.DataFrame(records)
        self.partitions = {split: self.rows[self.rows.split == split].reset_index(drop=True)
                           for split in ("train", "validation")}
        check(set(self.partitions["train"].entity_id).isdisjoint(self.partitions["validation"].entity_id),
              "train/validation entity overlap")
        self.provenance = {"model": self.model["model"], "manifest_sha256": hash_file(entity_path),
                           "manifest_metadata_sha256": hash_file(self.root / config["entity_manifest_metadata"]),
                           "assignment_sha256": sha256(canonical_json(assignments)), "sources": sources,
                           "allowed_rows_sha256": sha256(canonical_json(self.rows.to_dict("records"))),
                           "test_labels_loaded": 0, "test_activation_rows_read": 0,
                           "split_counts": self.rows.groupby(["split", "topic", "form"]).size().reset_index(name="rows").to_dict("records")}

    def matrix(self, split, layer):
        check(split in {"train", "validation"}, "atomic TEST is inaccessible")
        check(0 <= layer < self.n_layers, "invalid saved layer")
        rows = self.partitions[split]
        output = np.empty((len(rows), self.hidden_size), dtype=np.float64)
        h = hashlib.sha256()
        for dataset, group in rows.groupby("dataset", sort=False):
            array = np.load(self.blocks[dataset], mmap_mode="r", allow_pickle=False)
            values = np.asarray(array[group.row_index.to_numpy(), layer, :])
            check(np.isfinite(values).all(), "nonfinite allowed activation row")
            h.update(values.tobytes(order="C"))
            output[group.index.to_numpy()] = values
            del array
        return output, h.hexdigest()


def load_baseline(data):
    """Reuse the existing validation-selected L2 model; no fit or test API invoked."""
    from src.atomic_probe_methods import FittedMethod
    directory = data.root / data.model["results_dir"]
    selection_path, archive_path = directory / "selection.json", directory / "selected_probe.npz"
    selection = json.loads(selection_path.read_text())
    check(selection["model"] == data.model["model"], "baseline model mismatch")
    check(not selection["test_evaluated"] and selection["test_decision_scores_computed"] == 0,
          "baseline is not a clean validation selection")
    check(hash_file(archive_path) == selection["selected_probe_sha256"], "baseline parameter hash mismatch")
    grid_path = directory / "validation_metrics.csv"
    check(hash_file(grid_path) == selection["validation_metrics_sha256"], "baseline validation grid mismatch")
    check(sha256(canonical_json(data.config)) == selection["provenance"]["config_sha256"], "baseline configuration changed")
    for field in ("assignment_sha256", "manifest_sha256", "manifest_metadata_sha256"):
        check(selection["structural"][field] == data.provenance[field], f"baseline {field} differs")
    prior_sources = {s["dataset"]: s for s in selection["structural"]["sources"]}
    for source in data.provenance["sources"]:
        for field in ("activation_file", "activation_size_bytes", "activation_mtime_ns", "activation_shape"):
            check(source[field] == prior_sources[source["dataset"]][field], "historical baseline cache provenance changed")
    grid = pd.read_csv(grid_path)
    winner = grid.sort_values(["validation_overall_auroc", "C", "layer"], ascending=[False, True, True]).iloc[0]
    check((int(winner.layer), float(winner.C)) == (selection["selected_layer"], selection["selected_C"]),
          "baseline validation selection does not match its grid")
    with np.load(archive_path, allow_pickle=False) as saved:
        params = {key: saved[key].copy() for key in saved.files}
    check(int(params["layer"]) == selection["selected_layer"] and float(params["C"]) == selection["selected_C"],
          "baseline layer/C differ")
    check(params["coef"].shape == (1, data.hidden_size) and np.array_equal(params["classes"], [0, 1]),
          "baseline dimensions/sign classes incorrect")
    params["coef"] = params["coef"][0]
    params["intercept"] = params["intercept"].reshape(())
    return selection, FittedMethod("l2_logistic", params, {
        "reuse": "existing TRAIN-fit, VALIDATION-selected clean L2 baseline; unchanged coefficients",
        "selection_path": str(selection_path.relative_to(data.root)), "selection_sha256": hash_file(selection_path),
        "parameter_sha256": hash_file(archive_path), "validation_grid_sha256": hash_file(grid_path),
        "fit_configuration": selection["probe_configuration"], "selected_C": selection["selected_C"],
        "sign_orientation": "existing LR classes [0,1]; no sign flip", "fit_rows": len(data.partitions["train"])})
