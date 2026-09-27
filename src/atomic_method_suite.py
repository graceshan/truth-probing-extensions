"""Train-only fitting and validation-only reporting/selection of atomic methods."""
from pathlib import Path
from datetime import datetime, timezone
import io
import json
import platform
import subprocess
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits, threadpool_info

from src.atomic_probe_methods import (METHODS, balanced_burger_indices, check, covariance_mass_mean,
                                     difference_of_means, fit_ttpd, learn_truth_directions)
from src.clean_atomic_method_data import MethodData, hash_file, load_baseline
from src.clean_atomic_probes import diagnostics
from src.entity_partitions import canonical_json, sha256


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def save_new(path, payload):
    path = Path(path)
    if path.exists():
        check(path.read_bytes() == payload, f"refusing to overwrite different output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(payload)


def choose_layers(rows, primary_layer):
    primary, secondary = {}, {}
    for method in METHODS:
        candidates = [r for r in rows if r["method"] == method]
        check(bool(candidates), f"missing method: {method}")
        fixed = [r for r in candidates if r["layer"] == primary_layer]
        check(len(fixed) == 1, "primary comparison missing fixed model-level layer")
        primary[method] = fixed[0]
        secondary[method] = min(candidates, key=lambda r: (-r["validation_overall_auroc"], r["layer"]))
    return {"primary_fixed_model_layer": primary, "secondary_method_selected_layer": secondary}


def implementation_provenance(root, suite_config):
    files = ("src/atomic_probe_methods.py", "src/clean_atomic_method_data.py", "src/atomic_method_suite.py",
             "src/clean_atomic_probes.py", "src/diff_means.py", "src/data.py", "src/entity_partitions.py",
             "scripts/29_clean_atomic_method_suite.py")
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True)
    return {"implementation": "clean-atomic-method-suite-v1", "git_head": commit.stdout.strip(),
            "git_dirty": bool(status.stdout.strip()),
            "code_sha256": {p: hash_file(root / p) for p in files},
            "configuration": suite_config, "python": platform.python_version(),
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__, "numeric_dtype": "float64", "blas_threads": 1,
            "blas_libraries": [{k: x.get(k) for k in ("internal_api", "version", "architecture", "threading_layer")}
                               for x in threadpool_info()]}


def fit_layer(data, layer, baseline, primary_layer, burger_indices, suite_config):
    start = time.monotonic()
    train, val = data.partitions["train"], data.partitions["validation"]
    X, train_hash = data.matrix("train", layer)
    V, val_hash = data.matrix("validation", layer)
    y, vy = train.label.to_numpy(), val.label.to_numpy()
    fitted = [difference_of_means(X, y), covariance_mass_mean(X, y, suite_config["covariance_pinv_atol"])]
    burger_rows = train.iloc[burger_indices]
    BX, by = X[burger_indices], y[burger_indices]
    p = np.where(burger_rows.form == "affirmative", 1, -1)
    truth = learn_truth_directions(BX, by, p, burger_rows.dataset.to_numpy())
    fitted += [truth, fit_ttpd(BX, by, p, burger_rows.dataset.to_numpy(), truth=truth)]
    if layer == primary_layer:
        fitted.insert(0, baseline)
    metrics, parameters, details = [], {}, {}
    for method in fitted:
        scores = method.decision_function(V)
        check(scores.shape == vy.shape and np.isfinite(scores).all(), "invalid validation scores")
        if method.method == "ttpd":
            check(np.allclose(scores, V @ method.parameters["coef"] + method.parameters["intercept"],
                              rtol=1e-9, atol=1e-8), "TTPD head and fused coefficient differ")
        fit_X, fit_y = (BX, by) if method.method in {"ttpd", "burger_t_g"} else (X, y)
        fit_scores = method.decision_function(fit_X)
        gap = float(fit_scores[fit_y == 1].mean() - fit_scores[fit_y == 0].mean())
        # t_G's sign is fixed by the partial truth regressor; raw dataset offsets
        # can affect a pooled gap. Report it without a data-dependent flip.
        if method.method != "burger_t_g":
            check(gap >= -1e-8, "classifier sign contradicts TRAIN truth orientation")
        diagnostic = diagnostics(vy, scores, val.topic.to_numpy(), val.form.to_numpy())
        check(diagnostic["overall_auroc"] is not None, "undefined validation AUROC")
        metrics.append({"method": method.method, "layer": layer, "fit_rows": len(fit_X),
                        "validation_rows": len(V), "train_true_minus_false_score_mean": gap,
                        **{"validation_" + k: v for k, v in diagnostic.items()}})
        details[method.method] = method.details
        parameters.update({method.method + "__" + key: value for key, value in method.parameters.items()})
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **parameters)
    return {"layer": layer, "metrics": metrics, "method_details": details,
            "train_activation_sha256": train_hash, "validation_activation_sha256": val_hash,
            "activation_hash_encoding": "concatenated selected float16 row bytes in partition/dataset order at this layer",
            "parameter_archive_sha256": sha256(buffer.getvalue()), "elapsed_seconds": time.monotonic() - start}, buffer.getvalue()


def run_model(root, suite_config, model_key, *, resume=False, check_only=False):
    root = Path(root)
    atomic_config = json.loads((root / suite_config["atomic_config"]).read_text())
    data = MethodData(root, atomic_config, model_key)
    selection, baseline = load_baseline(data)
    primary_layer = selection["selected_layer"]
    check(primary_layer == suite_config["primary_layers"][model_key], "model-level fixed layer changed")
    burger_indices = balanced_burger_indices(data.partitions["train"], suite_config["burger_sampling_seed"])
    burger_rows = data.partitions["train"].iloc[burger_indices]
    fingerprint = {"implementation": implementation_provenance(root, suite_config), "data": data.provenance,
                   "baseline": baseline.details, "burger_training_rows_sha256": sha256(canonical_json(burger_rows.to_dict("records"))),
                   "burger_training_rows": len(burger_rows), "fixed_primary_layer": primary_layer}
    if check_only:
        return fingerprint
    output = root / suite_config["output_dir"] / model_key
    identity = sha256(canonical_json(fingerprint))
    if resume:
        prior = json.loads((output / "provenance.json").read_text())
        check(prior["identity_sha256"] == identity, "resume inputs/configuration/code/provenance differ")
    else:
        check(not output.exists(), "new results directory already exists; use --resume for matching inputs")
        save_new(output / "provenance.json", json_bytes({"identity_sha256": identity, **fingerprint,
                 "timestamp_utc": datetime.now(timezone.utc).isoformat(), "test_labels_loaded": 0,
                 "test_activation_rows_read": 0, "test_scores_computed": 0, "compound_activations_read": 0}))
        save_new(output / "allowed_atomic_rows.csv", data.rows.to_csv(index=False).encode())
        save_new(output / "burger_training_rows.csv", burger_rows.to_csv(index=False).encode())
    all_metrics = []
    # Fixed-layer comparison first; then complete method-specific validation sweeps.
    order = [primary_layer] + [i for i in range(data.n_layers) if i != primary_layer]
    with threadpool_limits(limits=1):
        for layer in order:
            path = output / "layers" / f"layer_{layer:02d}.json"
            archive = path.with_suffix(".npz")
            if path.exists():
                check(resume, "unexpected existing layer result")
                record = json.loads(path.read_text())
                check(hash_file(archive) == record["parameter_archive_sha256"], "saved parameter archive changed")
                for split in ("train", "validation"):
                    _, current_hash = data.matrix(split, layer)
                    check(current_hash == record[f"{split}_activation_sha256"], "selected activation values changed")
            else:
                record, payload = fit_layer(data, layer, baseline, primary_layer, burger_indices, suite_config)
                if layer == primary_layer:
                    metrics = next(r for r in record["metrics"] if r["method"] == "l2_logistic")
                    for key, value in metrics.items():
                        if key.startswith("validation_") and key in selection["selected_validation_metrics"]:
                            expected = selection["selected_validation_metrics"][key]
                            check(value is None and expected is None or value is not None and expected is not None and
                                  abs(value - expected) <= 1e-12, f"frozen baseline validation reproduction failed: {key}")
                # Archive first, then the small commit record. Deterministic NPZ
                # content can be reused after an interruption between the writes.
                save_new(archive, payload)
                save_new(path, json_bytes(record))
            for row in record["metrics"]:
                all_metrics.append({**row, "parameters": str(archive.relative_to(output)),
                                    "parameter_archive_sha256": record["parameter_archive_sha256"]})
            print(f"{model_key} layer {layer}: " + ", ".join(f"{r['method']}={r['validation_overall_auroc']:.6f}" for r in record["metrics"])
                  + f" ({record['elapsed_seconds']:.1f}s)", flush=True)
    all_metrics.sort(key=lambda r: (r["method"], r["layer"]))
    chosen = choose_layers(all_metrics, primary_layer)
    save_new(output / "validation_metrics.csv", pd.DataFrame(all_metrics).to_csv(index=False).encode())
    summary = {"model": data.model["model"], "primary_layer": primary_layer,
               "selection_rule": suite_config["secondary_rule"], "comparisons": chosen,
               "baseline_secondary_selection": "existing clean L2 all-layer/C validation winner, parameters reused unchanged",
               "test_labels_loaded": 0, "test_activation_rows_read": 0, "test_scores_computed": 0,
               "compound_activations_read": 0, "fit_split": "train", "selection_split": "validation",
               "no_refit_on_validation": True, "provenance_identity_sha256": identity,
               "validation_metrics_sha256": hash_file(output / "validation_metrics.csv")}
    save_new(output / "summary.json", json_bytes(summary))
    return summary
