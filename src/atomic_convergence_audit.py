"""Refit only warned atomic configurations; change only the iteration budget."""

import csv
import io
import json
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from src.clean_atomic_probes import (
    AtomicData, C_VALUES, PROBE_CONFIG, SELECTION_RULE, diagnostics,
    json_bytes, load_frozen_probe, provenance, selection_key,
)
from src.clean_compounds import check
from src.entity_partitions import csv_bytes, save_outputs, sha256

AUDIT_MAX_ITER = 10000


def refit_warned(train, validation, original_rows, parameters, progress=None):
    check(train.split == "train" and validation.split == "validation", "audit requires train/validation only")
    check(train.entity_ids.isdisjoint(validation.entity_ids), "train/validation overlap")
    check(parameters == PROBE_CONFIG, "original scientific configuration differs")
    refit_parameters = {**parameters, "max_iter": AUDIT_MAX_ITER}
    check({k for k in parameters if parameters[k] != refit_parameters[k]} == {"max_iter"},
          "only max_iter may change")
    audits, probes = [], {}
    warned = sorted((r for r in original_rows if r["convergence_warning"]),
                    key=lambda r: (r["layer"], r["C"]))
    last_layer = None
    with threadpool_limits(limits=1):
        for original in warned:
            layer, C = original["layer"], original["C"]
            if layer != last_layer:
                train_X, validation_X = train.matrix(layer), validation.matrix(layer)
                last_layer = layer
            # Same zero initialization/defaults as the original run: no warm start.
            probe = LogisticRegression(C=C, **refit_parameters)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                probe.fit(train_X, train.labels)
            check(list(probe.classes_) == [0, 1], "incorrect fitted class orientation")
            metrics = diagnostics(validation.labels, probe.decision_function(validation_X),
                                  validation.topics, validation.forms)
            warned_again = any(issubclass(w.category, ConvergenceWarning) for w in caught)
            converged = not warned_again
            row = {"layer": layer, "C": C, "original_n_iter": original["n_iter"],
                   "original_validation_auroc": original["validation_overall_auroc"],
                   "original_converged": False, "original_convergence_warning": True,
                   "original_max_iter": parameters["max_iter"], "refit_max_iter": AUDIT_MAX_ITER,
                   "refit_converged": converged, "refit_n_iter": int(probe.n_iter_[0]),
                   "refit_validation_auroc": metrics["overall_auroc"],
                   "absolute_auroc_change": abs(metrics["overall_auroc"] - original["validation_overall_auroc"]),
                   "refit_warning_messages": json.dumps([str(w.message) for w in caught]),
                   **{"refit_validation_" + k: v for k, v in metrics.items() if k != "overall_auroc"}}
            audits.append(row)
            name = f"layer_{layer}_C_{C:g}"
            for key, value in (("coef", probe.coef_), ("intercept", probe.intercept_),
                               ("classes", probe.classes_), ("n_iter", probe.n_iter_)):
                probes[name + "_" + key] = value
            if progress:
                progress(row)
    return audits, probes


def replacement_selection(original_rows, audits):
    replacements = {(r["layer"], r["C"]): r for r in audits}
    revised = []
    for row in original_rows:
        replacement = replacements.get((row["layer"], row["C"]))
        revised_row = dict(row)
        if replacement and replacement["refit_converged"]:
            revised_row["validation_overall_auroc"] = replacement["refit_validation_auroc"]
        revised.append(revised_row)
    before, after = min(original_rows, key=selection_key), min(revised, key=selection_key)
    def choice(row):
        return {key: row[key] for key in ("layer", "C", "validation_overall_auroc")}
    return {"original_selected": choice(before), "selected_after_converged_replacements": choice(after),
            "selected_configuration_changed": (before["layer"], before["C"]) != (after["layer"], after["C"]),
            "all_warned_fits_converged": all(r["refit_converged"] for r in audits),
            "unresolved_configurations": [{"layer": r["layer"], "C": r["C"]} for r in audits
                                           if not r["refit_converged"]],
            "selection_rule": SELECTION_RULE,
            "unresolved_policy": "Only converged refits replace original values; unresolved fits leave conclusion provisional"}


def artifact_fingerprints(directory):
    return {p.name: {"sha256": sha256(p.read_bytes()), "mtime_ns": p.stat().st_mtime_ns}
            for p in sorted(Path(directory).iterdir()) if p.is_file()}


def audit_model(root, config, model_key, progress=None):
    root = Path(root)
    directory = root / config["models"][model_key]["results_dir"]
    before = artifact_fingerprints(directory)
    data = AtomicData(root, config, model_key)  # Test authorization remains false.
    original, _ = load_frozen_probe(data, directory)
    check(original["probe_configuration"] == PROBE_CONFIG, "probe settings changed since original run")
    check(original["preprocessing"] == "none", "unexpected original preprocessing")
    current_provenance = provenance(root, config, model_key)
    for field in ("code_sha256", "config_sha256", "python", "numpy", "pandas", "scikit_learn", "blas_threads"):
        check(current_provenance[field] == original["provenance"][field], f"original runtime/code mismatch: {field}")
    with (directory / "validation_metrics.csv").open(newline="") as handle:
        rows = [{"layer": int(r["layer"]), "C": float(r["C"]), "n_iter": int(r["n_iter"]),
                 "validation_overall_auroc": float(r["validation_overall_auroc"]),
                 "convergence_warning": r["convergence_warning"] == "True"} for r in csv.DictReader(handle)]
    check(len(rows) == data.n_layers * len(C_VALUES) and
          {(r["layer"], r["C"]) for r in rows} == {(l, c) for l in range(data.n_layers) for c in C_VALUES},
          "original grid is incomplete")
    original_winner = min(rows, key=selection_key)
    check((original_winner["layer"], original_winner["C"]) ==
          (original["selected_layer"], original["selected_C"]), "saved winner differs from original grid")
    train, validation = data.partition("train"), data.partition("validation")
    audits, probes = refit_warned(train, validation, rows, original["probe_configuration"], progress)
    for row in audits:
        row["model"] = data.model["model"]
    check(data.activation_rows_read["test"] == 0, "test activation access detected")
    check(before == artifact_fingerprints(directory), "original selection artifacts changed")
    archive = io.BytesIO()
    np.savez_compressed(archive, **probes)
    summary = {"model": data.model["model"], "warned_configurations": len(audits),
               "original_probe_configuration": original["probe_configuration"],
               "refit_probe_configuration": {**original["probe_configuration"], "max_iter": AUDIT_MAX_ITER},
               "only_changed_parameter": "max_iter", "preprocessing": "none",
               "refit_initialization": "from scratch, same estimator defaults as original",
               "selection_comparison": replacement_selection(rows, audits),
               "original_artifacts_before_and_after": before,
               "original_artifacts_unchanged": True, "structural": data.structural,
               "train_labels_sha256": sha256(train.labels.tobytes()),
               "validation_labels_sha256": sha256(validation.labels.tobytes()),
               "provenance": current_provenance,
               "audit_code_sha256": {p: sha256((root / p).read_bytes()) for p in
                                     ("src/atomic_convergence_audit.py", "scripts/21_audit_atomic_convergence.py")},
               "activation_rows_read": data.activation_rows_read,
               "test_evaluated": False, "test_decision_scores_computed": 0}
    output = root / "results/clean_protocol/atomic_lr_convergence_audit" / model_key
    fields = ("model", *(key for key in audits[0] if key != "model")) if audits else ("model", "layer", "C")
    save_outputs(output, {"refits.csv": csv_bytes(audits, fields), "summary.json": json_bytes(summary),
                          "refit_probes.npz": archive.getvalue()})
    return audits, summary
