"""Train/validation atomic probes with an explicitly gated frozen-test path."""

import io
import json
import platform
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits

from src.clean_compounds import EntityManifest, check
from src.data import entity_key_from_statements
from src.entity_partitions import SPLITS, TOPICS, canonical_json, csv_bytes, save_outputs, sha256

C_VALUES = [0.001, 0.01, 0.1, 1.0, 10.0]
PROBE_CONFIG = {"penalty": "l2", "solver": "lbfgs", "fit_intercept": True,
                "max_iter": 2000, "class_weight": None, "tol": 0.0001}
RETRY_MAX_ITER = 10000
OPTIMIZATION_POLICY = {
    "initial_max_iter": PROBE_CONFIG["max_iter"], "retry_max_iter": RETRY_MAX_ITER,
    "retry_trigger": "sklearn.exceptions.ConvergenceWarning only",
    "retry_initialization": "from scratch with identical train data and parameters except max_iter",
    "retry_failure": "raise RuntimeError; do not score or accept the fit",
}
SELECTION_RULE = "maximize pooled validation AUROC; exact ties: smaller C, then lower zero-based layer index"
FORMS = ("affirmative", "negated")


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


@dataclass
class ActivationBlock:
    path: Path
    indices: dict
    labels: dict
    topic: str
    form: str


class AtomicData:
    """Read-only caches. Test labels are discarded unless explicitly authorized.

    Structural loading may parse test identities/labels to verify source alignment
    and binary labels. Those labels never enter development partition objects.
    Only requested split indices are used when reading activation values.
    """

    def __init__(self, root, config, model_key, *, authorize_test=False):
        check(config["schema_version"] == 1 and config["C_values"] == C_VALUES,
              "configuration must use schema 1 and the fixed C grid")
        self.root, self.config, self.model_key = Path(root), config, model_key
        self.model = config["models"][model_key]
        self.n_layers, self.hidden_size = self.model["n_layers"], self.model["hidden_size"]
        self._authorize_test = authorize_test
        self.manifest = EntityManifest(self.root / config["entity_manifest"],
                                       self.root / config["entity_manifest_metadata"])
        lookup = {(e.topic, e.entity): e for e in self.manifest.entities.values()}
        source_hashes = {s["file"]: s["sha256"] for s in self.manifest.metadata["sources"]}
        self.blocks, sources, assignments = [], [], []
        self._entity_sets = {split: set() for split in SPLITS}
        counts = []
        for topic in TOPICS:
            coverage = {}
            for form in FORMS:
                dataset = topic if form == "affirmative" else "neg_" + topic
                source_path = self.root / config["source_dir"] / (dataset + ".csv")
                meta_path = self.root / self.model["metadata_pattern"].format(dataset=dataset)
                acts_path = self.root / self.model["activation_pattern"].format(dataset=dataset)
                source_payload, sidecar_payload = source_path.read_bytes(), meta_path.read_bytes()
                check(sha256(source_payload) == source_hashes[dataset + ".csv"],
                      f"{dataset}: source differs from manifest provenance")
                source = pd.read_csv(io.BytesIO(source_payload))
                sidecar = pd.read_csv(io.BytesIO(sidecar_payload))
                check({"statement", "label"} <= set(sidecar.columns) <= set(source.columns),
                      f"{dataset}: unexpected sidecar columns")
                check(sidecar.equals(source[list(sidecar.columns)]), f"{dataset}: source/sidecar row mismatch")
                check(source["label"].isin([0, 1]).all(), f"{dataset}: nonbinary labels")
                entities = entity_key_from_statements(topic, source["statement"])
                coverage[form] = set(entities)
                check(all((topic, entity) in lookup for entity in entities),
                      f"{dataset}: atomic entity absent from manifest")
                memberships = [lookup[(topic, entity)] for entity in entities]
                splits = np.asarray([e.split for e in memberships])
                indices = {split: np.flatnonzero(splits == split) for split in SPLITS}
                # Test labels are not retained in a default AtomicData instance.
                allowed = SPLITS if authorize_test else ("train", "validation")
                labels = {split: source.iloc[indices[split]]["label"].to_numpy(dtype=np.int64)
                          for split in allowed}
                acts = np.load(acts_path, mmap_mode="r", allow_pickle=False)
                check(acts.shape == (len(source), self.n_layers, self.hidden_size),
                      f"{dataset}: activation dimensions do not match configuration")
                check(str(acts.dtype) == self.model["dtype"], f"{dataset}: unexpected activation dtype")
                shape = list(acts.shape)
                del acts  # Header verification only; no test activation values read.
                self.blocks.append(ActivationBlock(acts_path, indices, labels, topic, form))
                for row_index, entity in enumerate(memberships):
                    self._entity_sets[entity.split].add(entity.entity_id)
                    assignments.append([dataset, row_index, entity.entity_id, entity.split])
                for split in SPLITS:
                    counts.append({"topic": topic, "form": form, "split": split,
                                   "rows": len(indices[split]),
                                   "entities": len(set(entities[indices[split]]))})
                stat = acts_path.stat()
                sources.append({"dataset": dataset, "source_csv": str(source_path.relative_to(self.root)),
                                "source_sha256": sha256(source_payload),
                                "metadata_csv": str(meta_path.relative_to(self.root)),
                                "metadata_sha256": sha256(sidecar_payload), "metadata_columns": list(sidecar.columns),
                                "activation_file": str(acts_path.relative_to(self.root)),
                                "activation_size_bytes": stat.st_size, "activation_mtime_ns": stat.st_mtime_ns,
                                "activation_shape": shape, "activation_dtype": self.model["dtype"]})
            check(coverage["affirmative"] == coverage["negated"],
                  f"{topic}: affirmative/negated entity coverage mismatch")
            check(coverage["affirmative"] == {e.entity for e in self.manifest.entities.values() if e.topic == topic},
                  f"{topic}: atomic coverage differs from manifest entities")
        check(all(self._entity_sets[a].isdisjoint(self._entity_sets[b])
                  for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))),
              "entity split leakage")
        self.counts = []
        for topic in TOPICS:
            for split in SPLITS:
                by_form = [r for r in counts if r["topic"] == topic and r["split"] == split]
                self.counts.append({"topic": topic, "split": split,
                                    "entities": len({e.entity_id for e in self.manifest.entities.values()
                                                     if e.topic == topic and e.split == split}),
                                    "rows": sum(r["rows"] for r in by_form),
                                    "affirmative_rows": by_form[0]["rows"], "negated_rows": by_form[1]["rows"]})
        self.structural = {
            "model": self.model["model"], "manifest_sha256": self.manifest.digest,
            "manifest_version": self.manifest.version,
            "manifest_metadata_sha256": self.manifest.metadata_digest,
            "assignment_sha256": sha256(canonical_json(assignments)),
            "counts": self.counts, "sources": sources,
            "checks": {name: True for name in ("unique_manifest_identity", "complete_atomic_coverage",
                       "compatible_affirmative_negated_coverage", "same_entity_same_split",
                       "pairwise_disjoint_entities", "source_sidecar_alignment", "activation_headers")},
            "activation_values_inspected_by_structural_checks": False,
            "test_metrics_evaluated": False,
        }
        if "extraction_metadata" in self.model:
            path = self.root / self.model["extraction_metadata"]
            self.structural["extraction_metadata"] = {
                "path": self.model["extraction_metadata"], "sha256": sha256(path.read_bytes())}
        self.activation_rows_read = {s: 0 for s in SPLITS}

    def partition(self, split):
        check(split in SPLITS, "unknown partition")
        if split == "test" and not self._authorize_test:
            raise PermissionError("test partition requires explicit final-evaluation authorization")
        return Partition(self, split)


class Partition:
    def __init__(self, data, split):
        if split == "test" and not data._authorize_test:
            raise PermissionError("test partition is locked")
        self._data, self.split = data, split
        self.n_layers, self.hidden_size = data.n_layers, data.hidden_size
        self.entity_ids = frozenset(data._entity_sets[split])
        self.labels = np.concatenate([b.labels[split] for b in data.blocks])
        self.topics = np.concatenate([np.repeat(b.topic, len(b.indices[split])) for b in data.blocks])
        self.forms = np.concatenate([np.repeat(b.form, len(b.indices[split])) for b in data.blocks])

    def matrix(self, layer):
        check(0 <= layer < self.n_layers, "invalid layer index")
        if self.split == "test" and not self._data._authorize_test:
            raise PermissionError("test activation access is locked")
        arrays = []
        for block in self._data.blocks:
            acts = np.load(block.path, mmap_mode="r", allow_pickle=False)
            # Index rows before scoring; never materialize an all-row layer matrix.
            arrays.append(np.asarray(acts[block.indices[self.split], layer, :], dtype=np.float64))
        matrix = np.concatenate(arrays)
        check(np.isfinite(matrix).all(), f"nonfinite {self.split} activations")
        self._data.activation_rows_read[self.split] += len(matrix)
        return matrix


def diagnostics(labels, scores, topics, forms):
    def auc(mask):
        y = labels[mask]
        return float(roc_auc_score(y, scores[mask])) if len(np.unique(y)) == 2 else None
    metrics = {"overall_auroc": auc(np.ones(len(labels), dtype=bool)),
               "affirmative_auroc": auc(forms == "affirmative"),
               "negated_auroc": auc(forms == "negated")}
    per_topic = [auc(topics == topic) for topic in TOPICS]
    metrics.update({f"topic_{topic}_auroc": value for topic, value in zip(TOPICS, per_topic)})
    metrics["topic_macro_auroc"] = float(np.mean(per_topic)) if all(v is not None for v in per_topic) else None
    metrics["accuracy_at_zero"] = float(np.mean((scores >= 0) == labels))
    return metrics


def selection_key(row):
    return (-row["validation_overall_auroc"], row["C"], row["layer"])


def fit_converged_probe(X, labels, C, *, layer):
    """Only a successfully optimized train fit may reach validation scoring."""
    def attempt(max_iter):
        probe = LogisticRegression(C=C, **{**PROBE_CONFIG, "max_iter": max_iter})
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            probe.fit(X, labels)
        convergence_warnings = [str(w.message) for w in caught
                                if issubclass(w.category, ConvergenceWarning)]
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning):
                warnings.warn_explicit(str(warning.message), warning.category,
                                       warning.filename, warning.lineno)
        return probe, convergence_warnings

    probe, initial_warnings = attempt(PROBE_CONFIG["max_iter"])
    initial_n_iter = int(probe.n_iter_[0])
    retry_needed = bool(initial_warnings)
    if retry_needed:
        probe, final_warnings = attempt(RETRY_MAX_ITER)
        if final_warnings:
            raise RuntimeError(
                f"Layer {layer}, C={C:g}: lbfgs did not converge at max_iter={RETRY_MAX_ITER}; "
                "no validation score or selection row accepted. " + " | ".join(final_warnings)
            )
    check(list(probe.classes_) == [0, 1], "incorrect probe class orientation")
    return probe, {
        "initial_max_iter": PROBE_CONFIG["max_iter"], "initial_n_iter": initial_n_iter,
        "initial_convergence_warning": retry_needed, "retry_needed": retry_needed,
        "final_max_iter": RETRY_MAX_ITER if retry_needed else PROBE_CONFIG["max_iter"],
        "final_n_iter": int(probe.n_iter_[0]), "final_converged": True,
        "final_convergence_status": "converged",
        "initial_warning_messages": json.dumps(initial_warnings),
        # Retain these legacy field names, now describing the accepted final fit.
        "n_iter": int(probe.n_iter_[0]), "convergence_warning": False,
    }


def select_probe(train, validation, progress=None):
    check(train.split == "train" and validation.split == "validation", "selection requires train and validation only")
    check(train.entity_ids.isdisjoint(validation.entity_ids), "train/validation entity overlap")
    check(train.n_layers == validation.n_layers and train.hidden_size == validation.hidden_size,
          "partition dimensions differ")
    check(np.array_equal(np.unique(train.labels), [0, 1]), "training labels must contain both classes")
    rows, best, selected_probe = [], None, None
    with threadpool_limits(limits=1):
        for layer in range(train.n_layers):
            train_X, validation_X = train.matrix(layer), validation.matrix(layer)
            for C in C_VALUES:
                probe, optimization = fit_converged_probe(train_X, train.labels, C, layer=layer)
                metrics = diagnostics(validation.labels, probe.decision_function(validation_X),
                                      validation.topics, validation.forms)
                check(metrics["overall_auroc"] is not None, "pooled validation AUROC is undefined")
                row = {"layer": layer, "C": C, **{"validation_" + k: v for k, v in metrics.items()},
                       **optimization}
                rows.append(row)
                if best is None or selection_key(row) < selection_key(best):
                    best, selected_probe = row, probe
            if progress:
                progress(layer, rows[-len(C_VALUES):])
    return rows, best, selected_probe


def provenance(root, config, model_key):
    root = Path(root)
    paths = ("src/clean_atomic_probes.py", "scripts/20_clean_atomic_probe.py", "src/data.py",
             "src/clean_compounds.py", "src/entity_partitions.py")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    return {"git_revision": revision.stdout.strip() if revision.returncode == 0 else None,
            "code_sha256": {p: sha256((root / p).read_bytes()) for p in paths if (root / p).exists()},
            "config": config, "config_sha256": sha256(canonical_json(config)), "model_key": model_key,
            "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__, "blas_threads": 1}


def fit_and_save(data, output_dir, progress=None):
    check(not data._authorize_test, "selection must run with test access locked")
    rows, best, probe = select_probe(data.partition("train"), data.partition("validation"), progress)
    check(data.activation_rows_read["test"] == 0, "test activations accessed during selection")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, coef=probe.coef_, intercept=probe.intercept_, classes=probe.classes_,
                        layer=np.asarray(best["layer"]), C=np.asarray(best["C"]), n_iter=probe.n_iter_,
                        max_iter=np.asarray(probe.max_iter))
    archive = buffer.getvalue()
    metrics_csv = csv_bytes(rows, tuple(rows[0]))
    summary = {"model": data.model["model"], "selected_layer": best["layer"], "selected_C": best["C"],
               "selection_rule": SELECTION_RULE, "selected_validation_metrics": best,
               "C_values": C_VALUES, "probe_configuration": PROBE_CONFIG, "preprocessing": "none",
               "optimization_policy": OPTIMIZATION_POLICY,
               "selected_probe_max_iter": probe.max_iter,
               "retried_configurations": sum(row["retry_needed"] for row in rows),
               "all_final_fits_converged": all(row["final_converged"] for row in rows),
               "label_convention": "0=false, 1=true; score >= 0 predicts true",
               "layer_convention": "zero-based transformer-block output; embedding excluded",
               "structural": data.structural, "provenance": provenance(data.root, data.config, data.model_key),
               "selected_probe_sha256": sha256(archive), "validation_metrics_sha256": sha256(metrics_csv),
               "test_evaluated": False, "test_decision_scores_computed": 0,
               "activation_rows_read": data.activation_rows_read}
    files = {"validation_metrics.csv": metrics_csv, "selected_probe.npz": archive,
             "selection.json": json_bytes(summary), "split_counts.csv": csv_bytes(data.counts, tuple(data.counts[0]))}
    save_outputs(output_dir, files)
    return summary


def load_frozen_probe(data, selection_dir):
    directory = Path(selection_dir)
    selection = json.loads((directory / "selection.json").read_text())
    archive = (directory / "selected_probe.npz").read_bytes()
    check(sha256(archive) == selection["selected_probe_sha256"], "frozen probe hash mismatch")
    check(sha256((directory / "validation_metrics.csv").read_bytes()) == selection["validation_metrics_sha256"],
          "frozen validation metrics hash mismatch")
    check(selection["structural"] == data.structural, "frozen model/manifest/source structure changed")
    check(selection["provenance"]["config_sha256"] == sha256(canonical_json(data.config)), "frozen configuration changed")
    check(selection["selection_rule"] == SELECTION_RULE, "frozen selection rule changed")
    with np.load(io.BytesIO(archive), allow_pickle=False) as saved:
        layer, C = int(saved["layer"]), float(saved["C"])
        check(layer == selection["selected_layer"] and C == selection["selected_C"] and C in C_VALUES,
              "frozen layer/C mismatch")
        check(saved["coef"].shape == (1, data.hidden_size) and saved["intercept"].shape == (1,),
              "frozen coefficient dimensions invalid")
        check(np.isfinite(saved["coef"]).all() and np.isfinite(saved["intercept"]).all(),
              "frozen coefficients must be finite")
        check(np.array_equal(saved["classes"], [0, 1]), "frozen classes invalid")
        probe = LogisticRegression(C=C, **{**PROBE_CONFIG, "max_iter":
            int(saved["max_iter"]) if "max_iter" in saved.files else PROBE_CONFIG["max_iter"]})
        probe.coef_, probe.intercept_, probe.classes_ = saved["coef"].copy(), saved["intercept"].copy(), saved["classes"].copy()
        probe.n_features_in_ = data.hidden_size
    return selection, probe


def evaluate_frozen_test(data, selection_dir):
    """Called only by the explicit final-test CLI path; never fits or selects."""
    if not data._authorize_test:
        raise PermissionError("--evaluate-test is required")
    selection, probe = load_frozen_probe(data, selection_dir)
    test = data.partition("test")
    scores = probe.decision_function(test.matrix(selection["selected_layer"]))
    result = {"model": data.model["model"], "selected_layer": selection["selected_layer"],
              "selected_C": selection["selected_C"], "selected_probe_sha256": selection["selected_probe_sha256"],
              "manifest_sha256": data.manifest.digest, "test_evaluated": True,
              "test_decision_scores_computed": len(scores),
              "metrics": diagnostics(test.labels, scores, test.topics, test.forms),
              "evaluation_provenance": provenance(data.root, data.config, data.model_key)}
    save_outputs(Path(selection_dir) / "final_test", {"test_metrics.json": json_bytes(result)})
    return result
