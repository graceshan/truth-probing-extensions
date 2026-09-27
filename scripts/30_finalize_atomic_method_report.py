#!/usr/bin/env python3
"""Verify method archives/selected validation scores and write the comparison report.

The optional one-time repair changes dataset-name metadata encoding only. It
never enables pickle, refits a classifier, or reads test/compound activations.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.atomic_probe_methods import METHODS, FittedMethod, check
from src.clean_atomic_method_data import MethodData, hash_file, load_baseline
from src.clean_atomic_probes import diagnostics


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def repair_metadata(directory):
    """Rewrite only known object-encoded names, reconstructing them without pickle."""
    with (directory / "burger_training_rows.csv").open() as f:
        names = np.asarray(sorted({r["dataset"] for r in csv.DictReader(f)}))
    repaired = []
    layer_hashes = {}
    for record_path in sorted((directory / "layers").glob("layer_*.json")):
        record = json.loads(record_path.read_text())
        archive_path = record_path.with_suffix(".npz")
        before_hash = hash_file(archive_path)
        check(before_hash == record["parameter_archive_sha256"], "archive mismatch before repair")
        object_names = []
        with zipfile.ZipFile(archive_path) as archive:
            for name in archive.namelist():
                with archive.open(name) as member:
                    version = np.lib.format.read_magic(member)
                    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                    shape, _, dtype = reader(member)
                    if dtype.hasobject:
                        check(name in {"burger_t_g__dataset_names.npy", "ttpd__dataset_names.npy"} and shape == names.shape,
                              "unexpected object parameter; refusing repair")
                        object_names.append(name.removesuffix(".npy"))
        if object_names:
            with np.load(archive_path, allow_pickle=False) as archive:
                parameters = {k: names.copy() if k in object_names else archive[k].copy() for k in archive.files}
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **parameters)
            with np.load(io.BytesIO(buffer.getvalue()), allow_pickle=False) as checked:
                for name, expected in parameters.items():
                    check(checked[name].dtype == expected.dtype and np.array_equal(checked[name], expected),
                          "numeric parameter changed during metadata serialization")
            archive_path.write_bytes(buffer.getvalue())
            record["parameter_archive_sha256"] = hash_file(archive_path)
            record["parameter_string_encoding"] = "Unicode; numerical fitted arrays unchanged"
            write_json(record_path, record)
            repaired.append({"layer": record["layer"], "old_archive_sha256": before_hash,
                             "new_archive_sha256": record["parameter_archive_sha256"],
                             "changed_metadata_fields": object_names, "numeric_arrays_exactly_unchanged": True})
        layer_hashes[record["layer"]] = record["parameter_archive_sha256"]
    metrics_path = directory / "validation_metrics.csv"
    with metrics_path.open() as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    for row in rows:
        row["parameter_archive_sha256"] = layer_hashes[int(row["layer"])]
    # Keep metric text verbatim: only parameter-hash cells change.
    with metrics_path.open("w") as f:
        writer = csv.DictWriter(f, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    for comparison in summary["comparisons"].values():
        for row in comparison.values():
            row["parameter_archive_sha256"] = layer_hashes[row["layer"]]
    summary["validation_metrics_sha256"] = hash_file(metrics_path)
    write_json(path, summary)
    return repaired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repair-dataset-name-metadata", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/clean_protocol/atomic_probe_methods.json").read_text())
    atomic_config = json.loads((ROOT / config["atomic_config"]).read_text())
    output = ROOT / config["output_dir"]
    prior_verification_path = output / "verification.json"
    prior_verification = json.loads(prior_verification_path.read_text()) if prior_verification_path.exists() else {}
    verification = {"test_labels_loaded": 0, "test_activation_rows_read": 0, "compound_activations_read": 0,
                    "models": {}, "finalizer_sha256": hash_file(Path(__file__))}
    lines = ["# Atomic validation method comparison", "",
             "TRAIN-only fits; VALIDATION-only selection. No test or compound activations/labels were used.", "",
             "All methods use the same entity split. LR/mean difference/covariance-MM use 3,144 training rows; "
             "Bürger t_G/TTPD follow the original paired, equal-dataset sampling with 1,000 training rows (seed 0). "
             "All methods evaluate the same 1,040 validation rows per model.", "",
             "MM below means the original Marks–Tegmark covariance-adjusted iid variant, not Bürger's unadjusted "
             "mean-difference-plus-logistic-bias baseline. TTPD includes both unregularized LR stages.", ""]
    labels = {"l2_logistic": "L2 LR", "difference_of_means": "Difference of means", "mass_mean_covariance": "Covariance MM",
              "burger_t_g": "Bürger t_G", "ttpd": "Full TTPD"}
    keys = ["overall", "affirmative", "negated", "topic_cities", "topic_sp_en_trans", "topic_inventors",
            "topic_element_symb", "topic_animal_class", "topic_macro"]
    with threadpool_limits(limits=1):
        for model in ("qwen25_7b", "qwen3_8b"):
            directory = output / model
            repairs = repair_metadata(directory) if args.repair_dataset_name_metadata else []
            summary = json.loads((directory / "summary.json").read_text())
            provenance = json.loads((directory / "provenance.json").read_text())
            data = MethodData(ROOT, atomic_config, model)
            check(data.provenance == provenance["data"], "data provenance changed")
            _, baseline = load_baseline(data)
            check(baseline.details == provenance["baseline"], "frozen LR provenance changed")
            check(hash_file(directory / "validation_metrics.csv") == summary["validation_metrics_sha256"], "metrics hash mismatch")
            archived_layers = list((directory / "layers").glob("layer_*.json"))
            check(len(archived_layers) == data.n_layers, "missing layer results")
            for path in archived_layers:
                record = json.loads(path.read_text())
                check(hash_file(path.with_suffix(".npz")) == record["parameter_archive_sha256"], "parameter hash mismatch")
                with np.load(path.with_suffix(".npz"), allow_pickle=False) as arrays:
                    for name in arrays.files:
                        value = arrays[name]
                        check(not value.dtype.hasobject, "object parameters remain")
                        if value.dtype.kind in "fiu":
                            check(np.isfinite(value).all(), "nonfinite saved parameter")
            val = data.partitions["validation"]
            matrix_cache = {}
            for comparison in summary["comparisons"].values():
                for method, row in comparison.items():
                    layer = row["layer"]
                    if layer not in matrix_cache:
                        matrix_cache[layer] = data.matrix("validation", layer)[0]
                    with np.load(directory / row["parameters"], allow_pickle=False) as archive:
                        parameters = {k.removeprefix(method + "__"): archive[k].copy() for k in archive.files if k.startswith(method + "__")}
                    fit = FittedMethod(method, parameters, {})
                    measured = diagnostics(val.label.to_numpy(), fit.decision_function(matrix_cache[layer]),
                                           val.topic.to_numpy(), val.form.to_numpy())
                    for metric, value in measured.items():
                        expected = row["validation_" + metric]
                        check(value is None and expected is None or value is not None and expected is not None and abs(value-expected) <= 1e-12,
                              f"saved {model}/{method} does not reproduce validation metrics")
            del matrix_cache
            for source in provenance["data"]["sources"]:
                stat = (ROOT / source["activation_file"]).stat()
                check(stat.st_size == source["activation_size_bytes"] and stat.st_mtime_ns == source["activation_mtime_ns"],
                      "historical activation cache metadata changed")
            verification["models"][model] = {"layers_verified": len(archived_layers), "all_archives_pickle_disabled": True,
                "all_numeric_parameters_finite": True, "primary_secondary_metrics_reproduced": True,
                "historical_activation_size_mtime_unchanged": True, "frozen_lr_provenance_unchanged": True,
                "metadata_encoding_repairs": repairs or prior_verification.get("models", {}).get(model, {}).get("metadata_encoding_repairs", []),
                "original_fit_provenance_sha256": hash_file(directory / "provenance.json")}
            lines += [f"## {summary['model']}", ""]
            for title, key in [(f"Primary: fixed model-selected layer {summary['primary_layer']}", "primary_fixed_model_layer"),
                               ("Secondary: each method's validation-selected layer", "secondary_method_selected_layer")]:
                lines += [title, "", "| Method | Layer | Overall | Affirmative | Negated | Cities | Spanish | Inventors | Elements | Animals | Topic macro |",
                          "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
                for method in METHODS:
                    row = summary["comparisons"][key][method]
                    lines.append(f"| {labels[method]} | {row['layer']} | " + " | ".join(f"{row['validation_'+k+'_auroc']:.6f}" for k in keys) + " |")
                lines += [""]
            print(model, "all saved archives and selected validation metrics verified", flush=True)
    snapshot = output / "fit_implementation_snapshot/atomic_probe_methods.py"
    if snapshot.exists():
        for model in ("qwen25_7b", "qwen3_8b"):
            provenance = json.loads((output / model / "provenance.json").read_text())
            check(hash_file(snapshot) == provenance["implementation"]["code_sha256"]["src/atomic_probe_methods.py"],
                  "fitting source snapshot hash mismatch")
        verification["fitting_source_snapshot"] = {"path": str(snapshot.relative_to(ROOT)), "sha256": hash_file(snapshot)}
        verification["current_serialization_fixed_source_sha256"] = hash_file(ROOT / "src/atomic_probe_methods.py")
        lines += ["The numerical fits used the source snapshot retained under `fit_implementation_snapshot/`. "
                  "A metadata-only finalization converted dataset-name object arrays to Unicode for safe pickle-disabled loading. "
                  "Every numerical parameter array was verified unchanged, and all selected validation diagnostics were reproduced "
                  "from the finalized archives. Original fitting provenance and old/new archive hashes are retained.", ""]
    lines += ["No validation-based sign flips or train+validation refits were performed. Secondary selection uses pooled "
              "validation AUROC, with lower-layer tie breaking. L2 reuses the existing all-layer/C validation winner. "
              "These are validation results, not test or compound-transfer estimates.", ""]
    write_json(output / "verification.json", verification)
    (output / "report.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
