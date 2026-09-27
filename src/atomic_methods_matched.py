"""Matched-TRAIN control, kept separate from the immutable faithful-method run."""
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.atomic_method_suite import implementation_provenance, json_bytes, save_new
from src.atomic_probe_methods import (METHODS, FittedMethod, check, difference_of_means,
                                     covariance_mass_mean, learn_truth_directions, fit_ttpd)
from src.clean_atomic_method_data import MethodData, hash_file, load_baseline
from src.clean_atomic_probes import PROBE_CONFIG, diagnostics, fit_converged_probe
from src.entity_partitions import canonical_json, sha256


def rows_hash(rows):
    return sha256(canonical_json(rows.to_dict("records")))


def row_ids(rows):
    return [f"{r.dataset}:{r.row_index}" for r in rows.itertuples()]


def resolve_training_subset(train, saved, expected_hash, expected_count):
    """Resolve the saved order, rejecting duplicates, changed labels or non-TRAIN rows."""
    check(len(saved) == expected_count and set(saved.split) == {"train"}, "invalid matched TRAIN rows")
    check(not saved.duplicated(["dataset", "row_index"]).any(), "duplicate matched row")
    check(rows_hash(saved) == expected_hash, "saved training row hash/order mismatch")
    lookup = {(r.dataset, r.row_index): i for i, r in enumerate(train.itertuples())}
    check(len(lookup) == len(train), "duplicate source training row")
    keys = list(zip(saved.dataset, saved.row_index))
    check(all(k in lookup for k in keys), "matched row is not in TRAIN")
    indices = np.array([lookup[k] for k in keys], dtype=int)
    check(rows_hash(train.iloc[indices]) == expected_hash, "matched row contents changed")
    return indices


def fit_matched_methods(X, rows, *, layer, C, covariance_atol):
    """Every fit receives exactly the same X/y in the same saved row order."""
    check(len(X) == len(rows) and set(rows.split) == {"train"}, "fit requires matched TRAIN")
    y = rows.label.to_numpy()
    lr, optimization = fit_converged_probe(X, y, C, layer=layer)
    fitted_lr = FittedMethod("l2_logistic", {
        "coef": lr.coef_[0], "intercept": lr.intercept_[0], "classes": lr.classes_, "C": np.array(C)
    }, {"fit_rows": len(X), "optimization": optimization, "configuration": PROBE_CONFIG,
        "sign_orientation": "LR classes [0,1]; no validation sign flip"})
    mean = difference_of_means(X, y)
    mass = covariance_mass_mean(X, y, covariance_atol)
    p = np.where(rows.form == "affirmative", 1, -1)
    truth = learn_truth_directions(X, y, p, rows.dataset.to_numpy())
    ttpd = fit_ttpd(X, y, p, rows.dataset.to_numpy(), truth=truth)
    return [fitted_lr, mean, mass, truth, ttpd]


def tree_hashes(directory):
    return {str(p.relative_to(directory)): hash_file(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def run_model(root, config, faithful_config, model):
    atomic = json.loads((root / faithful_config["atomic_config"]).read_text())
    data = MethodData(root, atomic, model)
    selection, baseline = load_baseline(data)
    faithful = root / faithful_config["output_dir"] / model
    prior = json.loads((faithful / "provenance.json").read_text())
    check(data.provenance == prior["data"], "faithful data provenance changed")
    check(baseline.details == prior["baseline"], "faithful baseline changed")
    check(selection["selected_C"] == config["lr_C"], "existing LR C differs")
    check(selection["probe_configuration"] == PROBE_CONFIG, "existing LR solver settings differ")
    layer = config["primary_layers"][model]
    check(layer == prior["fixed_primary_layer"] == selection["selected_layer"], "fixed layer changed")
    train, val = data.partitions["train"], data.partitions["validation"]
    saved_path = faithful / "burger_training_rows.csv"
    saved = pd.read_csv(saved_path, keep_default_na=False)
    indices = resolve_training_subset(train, saved, prior["burger_training_rows_sha256"], config["expected_train_rows"])
    allowed = pd.read_csv(faithful / "allowed_atomic_rows.csv", keep_default_na=False)
    check(rows_hash(allowed) == data.provenance["allowed_rows_sha256"], "faithful allowed rows changed")
    check(rows_hash(allowed[allowed.split == "validation"]) == rows_hash(val), "validation rows/order changed")
    check(len(val) == config["expected_validation_rows"], "validation count changed")
    check(set(saved.entity_id).isdisjoint(val.entity_id), "TRAIN/VALIDATION entity overlap")

    full_train, train_hash = data.matrix("train", layer)
    X = full_train[indices]
    del full_train
    V, val_hash = data.matrix("validation", layer)
    faithful_record_path = faithful / "layers" / f"layer_{layer:02d}.json"
    faithful_record = json.loads(faithful_record_path.read_text())
    check(train_hash == faithful_record["train_activation_sha256"], "TRAIN cache slice changed")
    check(val_hash == faithful_record["validation_activation_sha256"], "VALIDATION cache slice changed")
    faithful_archive = faithful_record_path.with_suffix(".npz")
    check(hash_file(faithful_archive) == faithful_record["parameter_archive_sha256"], "faithful archive hash differs")
    fitted = fit_matched_methods(X, saved, layer=layer, C=config["lr_C"],
                                covariance_atol=faithful_config["covariance_pinv_atol"])
    train_ids_hash = sha256(canonical_json(row_ids(saved)))
    validation_ids_hash = sha256(canonical_json(row_ids(val)))
    metrics, params, details = [], {}, {}
    prior_metrics = {r["method"]: r for r in faithful_record["metrics"]}
    for fit in fitted:
        check(fit.details["fit_rows"] == len(saved), "method used different fit rows")
        scores = fit.decision_function(V)
        fit_scores = fit.decision_function(X)
        gap = float(fit_scores[saved.label == 1].mean() - fit_scores[saved.label == 0].mean())
        if fit.method != "burger_t_g":
            check(gap >= -1e-8, "TRAIN sign orientation failed")
        measured = diagnostics(val.label.to_numpy(), scores, val.topic.to_numpy(), val.form.to_numpy())
        if fit.method in {"burger_t_g", "ttpd"}:
            with np.load(faithful_archive, allow_pickle=False) as archive:
                for key, value in fit.parameters.items():
                    check(np.array_equal(value, archive[fit.method + "__" + key]),
                          f"{fit.method} no longer reproduces faithful parameters: {key}")
            check(all(value == prior_metrics[fit.method]["validation_" + key] for key, value in measured.items()),
                  "Bürger validation results differ from faithful run")
        metrics.append({"comparison": "matched_training_data", "method": fit.method, "layer": layer,
                        "fit_rows": len(saved), "validation_rows": len(val),
                        "ordered_training_row_ids_sha256": train_ids_hash,
                        "ordered_validation_row_ids_sha256": validation_ids_hash,
                        "train_true_minus_false_score_mean": gap,
                        **{"validation_" + key: value for key, value in measured.items()},
                        "faithful_validation_overall_auroc": prior_metrics[fit.method]["validation_overall_auroc"],
                        "matched_minus_faithful_overall_auroc": measured["overall_auroc"] - prior_metrics[fit.method]["validation_overall_auroc"]})
        details[fit.method] = fit.details
        params.update({fit.method + "__" + key: value for key, value in fit.parameters.items()})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **params)
    output = root / config["output_dir"] / model
    save_new(output / "parameters.npz", buffer.getvalue())
    # Read back every array safely and reproduce every validation diagnostic.
    with np.load(output / "parameters.npz", allow_pickle=False) as archive:
        for fit, row in zip(fitted, metrics):
            restored = {k: archive[fit.method + "__" + k] for k in fit.parameters}
            for k in restored:
                check(np.array_equal(restored[k], fit.parameters[k]), "parameter roundtrip differs")
            scores = FittedMethod(fit.method, restored, {}).decision_function(V)
            measured = diagnostics(val.label.to_numpy(), scores, val.topic.to_numpy(), val.form.to_numpy())
            check(all(row["validation_" + k] == v for k, v in measured.items()), "saved diagnostics differ")
    save_new(output / "training_rows.csv", saved_path.read_bytes())
    save_new(output / "validation_rows.csv", val.to_csv(index=False).encode())
    save_new(output / "row_ids.json", json_bytes({"id_definition": "dataset + ':' + original zero-based row_index",
              "train": row_ids(saved), "validation": row_ids(val)}))
    save_new(output / "validation_metrics.csv", pd.DataFrame(metrics).to_csv(index=False).encode())
    implementation = implementation_provenance(root, config)
    implementation["implementation"] = "clean-atomic-matched-control-v1"
    for name in ("src/atomic_methods_matched.py", "scripts/31_matched_atomic_methods.py",
                 "config/clean_protocol/atomic_methods_matched.json"):
        implementation["code_sha256"][name] = hash_file(root / name)
    provenance = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "implementation": implementation,
                  "original_method_definitions": faithful_config, "data": data.provenance,
                  "faithful_provenance_sha256": hash_file(faithful / "provenance.json"),
                  "faithful_parameter_archive_sha256": hash_file(faithful_archive),
                  "training_rows_sha256": rows_hash(saved), "ordered_training_row_ids_sha256": train_ids_hash,
                  "validation_rows_sha256": rows_hash(val), "ordered_validation_row_ids_sha256": validation_ids_hash,
                  "matched_training_activations_sha256": sha256(X.astype(np.float16).tobytes()),
                  "matched_activation_hash_encoding": "float16 C-order bytes in training_rows.csv order at fixed layer",
                  "validation_activation_sha256": val_hash, "method_details": details,
                  "files_sha256": {p.name: hash_file(p) for p in output.iterdir() if p.is_file()},
                  "fit_rows": len(saved), "validation_rows": len(val), "layer": layer,
                  "test_labels_loaded": 0, "test_activation_rows_read": 0, "compound_activations_read": 0,
                  "all_saved_parameters_and_validation_metrics_verified": True,
                  "burger_parameters_identical_to_faithful": True, "validation_sign_flips": False}
    save_new(output / "provenance.json", json_bytes(provenance))
    summary = {"model": data.model["model"], "layer": layer, "metrics": metrics,
               "training_rows_sha256": rows_hash(saved), "validation_rows_sha256": rows_hash(val),
               "provenance_sha256": hash_file(output / "provenance.json")}
    save_new(output / "summary.json", json_bytes(summary))
    print(model + ": " + ", ".join(f"{r['method']}={r['validation_overall_auroc']:.6f}" for r in metrics), flush=True)
    return summary


def run(root):
    root = Path(root)
    config = json.loads((root / "config/clean_protocol/atomic_methods_matched.json").read_text())
    faithful_config = json.loads((root / config["faithful_config"]).read_text())
    faithful_path = (root / faithful_config["output_dir"]).resolve()
    output = (root / config["output_dir"]).resolve()
    check(output != faithful_path and faithful_path not in output.parents and output not in faithful_path.parents,
          "matched output must be separate from faithful results")
    check(not output.exists(), "matched output exists; refusing overwrite")
    before = tree_hashes(faithful_path)
    check(bool(before), "missing faithful results")
    save_new(output / "faithful_files_before.json", json_bytes(before))
    with threadpool_limits(limits=1):
        summaries = [run_model(root, config, faithful_config, model) for model in ("qwen25_7b", "qwen3_8b")]
    check(summaries[0]["training_rows_sha256"] == summaries[1]["training_rows_sha256"], "models used different training rows")
    check(summaries[0]["validation_rows_sha256"] == summaries[1]["validation_rows_sha256"], "models used different validation rows")
    check(tree_hashes(faithful_path) == before, "faithful outputs changed")
    save_new(output / "verification.json", json_bytes({"faithful_files_byte_unchanged": True,
             "faithful_files_verified": len(before), "faithful_file_hashes_sha256": hash_file(output / "faithful_files_before.json"),
             "identical_train_and_validation_rows_across_models": True, "all_five_methods_share_exact_ordered_training_rows": True,
             "test_labels_loaded": 0, "test_activation_rows_read": 0, "compound_activations_read": 0}))
    lines = ["# Matched-training-data atomic control", "",
             "All five methods fit the same ordered 1,000 TRAIN rows and evaluate the same 1,040 VALIDATION rows. "
             "The training rows are copied from the faithful Bürger sample, without resampling. "
             "No atomic test labels/activation values or compound data were accessed.", "",
             "LR is newly fit with the existing C=0.01 and solver settings; C is not retuned. "
             "Layers are fixed at Qwen2.5 saved index 17 and Qwen3 index 28. No secondary layer sweep was performed. "
             "Signs use TRAIN/formulas only. There are no validation sign flips or train+validation refits.", ""]
    keys = ["overall", "affirmative", "negated", "topic_cities", "topic_sp_en_trans", "topic_inventors",
            "topic_element_symb", "topic_animal_class", "topic_macro"]
    for summary in summaries:
        lines += [f"## {summary['model']} — saved layer {summary['layer']}", "",
                  "| Method | Overall | Affirmative | Negated | Cities | Spanish | Inventors | Elements | Animals | Topic macro | Δ overall vs faithful |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in summary["metrics"]:
            lines.append("| " + row["method"] + " | " + " | ".join(f"{row['validation_'+k+'_auroc']:.6f}" for k in keys)
                         + f" | {row['matched_minus_faithful_overall_auroc']:+.6f} |")
        lines += [""]
    lines += ["The t_G and TTPD parameters and diagnostics reproduce the faithful fixed-layer results exactly. "
              "Their training data already matched this construction. The other methods were refit on this subset.", "",
              "Shared ordered training-record SHA-256: `" + summaries[0]["training_rows_sha256"] + "`.", "",
              "Each model directory includes training_rows.csv (exact source copy), validation_rows.csv, row_ids.json, "
              "parameters.npz, validation_metrics.csv, summary.json and provenance.json. The provenance records both "
              "ordered row-ID and full-record hashes, cache-slice hashes, method settings and source/version hashes. "
              "verification.json confirms every faithful result file remained byte-for-byte unchanged.", "",
              "AUROCs are validation results. The diagnostic CSV also retains zero-threshold accuracy from the shared "
              "helper; that threshold is not calibrated for raw projection methods.", ""]
    save_new(output / "report.md", "\n".join(lines).encode())
