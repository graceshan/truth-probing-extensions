"""Canonical repaired-cache atomic methods. TRAIN fits, VALIDATION selection only.

Historical orchestration is never called. The shared fitting functions are pure
numerical operations; the only data adapter here wraps RepairedAtomicCache.
"""
from datetime import datetime, timezone
import io
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits, threadpool_info

from src import clean_transfer_contracts as c
from src.atomic_method_suite import choose_layers, fit_layer, json_bytes
from src.atomic_methods_matched import fit_matched_methods
from src.atomic_probe_methods import FittedMethod, balanced_burger_indices
from src.clean_atomic_probes import diagnostics
from src.pinned_compound_scoring import verify_probe
from src.repaired_atomic_cache import RepairedAtomicCache, FIELDS

ROOT = Path(__file__).resolve().parents[1]
VERSION = "pinned-atomic-method-suite-v1"
OUTPUTS = {
    "faithful": "results/clean_protocol/atomic_method_suite_pinned_v1/qwen25_7b",
    "matched": "results/clean_protocol/atomic_method_matched_pinned_v1/qwen25_7b",
}
COUNTS = {"train": 3144, "validation": 1040}
POLICY = dict(primary_layer=17, burger_sampling_seed=0, burger_rows=1000,
              covariance_pinv_atol=1e-3, numeric_dtype="float64", blas_threads=1,
              secondary_rule="maximum pooled atomic validation AUROC, then lowest saved layer",
              fit_split="train", selection_split="validation", validation_sign_flip=False)
SOURCE_FILES = (
    "src/pinned_atomic_method_suite.py", "scripts/38_pinned_atomic_method_suite.py",
    "src/atomic_probe_methods.py", "src/atomic_method_suite.py", "src/atomic_methods_matched.py",
    "src/clean_atomic_method_data.py", "src/clean_atomic_probes.py", "src/diff_means.py",
    "src/data.py", "src/clean_compounds.py",
    "src/pinned_compound_scoring.py", "src/clean_transfer_contracts.py",
    "src/repaired_atomic_cache.py", "src/clean_atomic_extraction.py", "src/entity_partitions.py",
    "config/clean_protocol/atomic_probe_methods.json",
)


def representation(contract):
    # Exact producer convention, copied as metadata to avoid importing Torch/HF
    # into CPU fitting. Regression-tested against the producer's function.
    convention = dict(
        output_hidden_states=True, embedding_state_included_by_hf=True,
        embedding_state_saved=False, saved_layer_zero_hf_index=1,
        saved_layer_to_hf_index=list(range(1, c.LAYERS + 1)),
        mapping="saved[s] = HF hidden_states[s + 1]; s = 0..num_hidden_layers-1",
        intermediate_states="block residual output before the next block",
        final_saved_state_post_final_norm=True,
        final_normalization="model final RMSNorm (not final block pre-norm residual)",
        token_readout="last real token",
        last_real_token_algorithm="max(where(attention_mask == 1, arange(sequence_length), -1)); reject all-padding rows",
        explicit_position_ids=True,
        position_ids_algorithm="(attention_mask.long().cumsum(-1) - 1).masked_fill(attention_mask == 0, 0)")
    return c.representation(contract, convention)


def row_hash(rows):
    return c.digest(c.canonical(rows.to_dict("records")))


def identity_hash(rows):
    return c.digest(c.canonical(list(zip(rows.dataset, rows.row_index))))


class RepairedMethodData:
    """Only fixed TRAIN/VALIDATION paths; no test or compound partition API."""
    def __init__(self, root=ROOT):
        self.cache = RepairedAtomicCache(root)
        self.root = self.cache.root
        self.n_layers = self.cache.n_layers
        self.partitions = {s: pd.DataFrame(self.cache.rows[s], columns=FIELDS) for s in COUNTS}
        c.require({s: len(rows) for s, rows in self.partitions.items()} == COUNTS, "atomic counts mismatch")
        self.rows = pd.concat(list(self.partitions.values()), ignore_index=True)
        self.burger_indices = balanced_burger_indices(self.partitions["train"], seed=0)
        self.burger_rows = self.partitions["train"].iloc[self.burger_indices]
        c.require(len(self.burger_indices) == 1000 and len(set(self.burger_indices)) == 1000 and
                  len(self.burger_rows.groupby("dataset")) == 10 and
                  self.burger_rows.groupby("dataset").size().eq(100).all(), "expected balanced 1000-row sample")
        self.activation_rows_read = {s: 0 for s in COUNTS}
        self.slice_hashes = {}

    def matrix(self, split, layer):
        if split not in COUNTS:
            raise PermissionError("only train and validation activation access is permitted")
        c.require(type(layer) is int and 0 <= layer < self.n_layers, "invalid saved layer")
        path = c.safe_path(self.root, f"{c.ATOMIC}/{split}/activations.npy")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        selected = np.ascontiguousarray(array[:, layer, :])
        self.activation_rows_read[split] += len(selected)
        digest = c.digest(selected.tobytes(order="C"))
        self.slice_hashes[split, layer] = digest
        if split == "train":
            self.slice_hashes["burger", layer] = c.digest(selected[self.burger_indices].tobytes(order="C"))
        matrix = selected.astype(np.float64)
        c.require(np.isfinite(matrix).all(), "nonfinite activation matrix")
        return matrix, digest


def implementation():
    upstream = c.read_json(ROOT / "config/clean_protocol/atomic_probe_methods.json")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    return dict(version=VERSION, code_sha256={p: c.file_hash(ROOT / p) for p in SOURCE_FILES},
                git_revision=git.stdout.strip(), policy=POLICY, upstream_sources=upstream["sources"],
                upstream_paper=upstream["paper"], python=platform.python_version(), numpy=np.__version__,
                pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)


def prepare(root):
    data = RepairedMethodData(root)
    probe = verify_probe(data.root, data.cache)  # Atomic-only verifier; no compound orchestration.
    descriptor = representation(data.cache.manifest["contract"])
    c.require(c.digest(c.canonical(descriptor)) == c.FINGERPRINT, "repaired representation fingerprint mismatch")
    c.require(probe["layer"] == POLICY["primary_layer"] and probe["C"] == 1., "canonical LR winner mismatch")
    provenance = dict(schema_version=1, implementation=implementation(),
        representation=descriptor, representation_fingerprint=c.FINGERPRINT,
        repaired_cache=dict(directory=c.ATOMIC, files=data.cache.files,
                            identity_sha256=c.digest(c.canonical(data.cache.files))),
        frozen_lr=dict(directory=c.PROBE, files=probe["files"], layer=probe["layer"], C=probe["C"],
                       producer_source_audit=probe["source_audit"]),
        allowed_atomic_rows_sha256=row_hash(data.rows),
        ordered_partition_identities={s: identity_hash(rows) for s, rows in data.partitions.items()},
        burger_training_rows_sha256=row_hash(data.burger_rows),
        burger_ordered_identities_sha256=identity_hash(data.burger_rows),
        partition_counts=COUNTS, burger_dataset_counts={k: int(v) for k, v in data.burger_rows.groupby("dataset").size().items()},
        historical_representation=dict(status="superseded", used_as_gate=False, artifacts_opened=False),
        atomic_test_accessed=False, compound_data_accessed=False)
    baseline = FittedMethod("l2_logistic", dict(coef=probe["coef"], intercept=np.array(probe["intercept"]),
        classes=np.array([0, 1]), C=np.array(probe["C"])),
        dict(fit_rows=COUNTS["train"], reused_without_refitting=True, selected_probe_sha256=probe["files"]["selected_probe.npz"]["sha256"],
             sign_orientation="frozen TRAIN-fitted classes [0,1]; no sign flip"))
    return data, probe, baseline, provenance


def matched_layer(data, probe):
    layer = probe["layer"]
    X, train_hash = data.matrix("train", layer)
    X = X[data.burger_indices]
    V, val_hash = data.matrix("validation", layer)
    train, val = data.burger_rows, data.partitions["validation"]
    fitted = fit_matched_methods(X, train, layer=layer, C=probe["C"], covariance_atol=1e-3)
    params, metrics, details = {}, [], {}
    for fit in fitted:
        c.require(fit.details["fit_rows"] == 1000, "matched fit rows differ")
        scores = fit.decision_function(V)
        train_scores = fit.decision_function(X)
        gap = float(train_scores[train.label == 1].mean() - train_scores[train.label == 0].mean())
        c.require(fit.method == "burger_t_g" or gap >= -1e-8, "TRAIN orientation failed")
        metrics.append(dict(method=fit.method, layer=layer, fit_rows=len(X), validation_rows=len(V),
            train_true_minus_false_score_mean=gap, **{"validation_" + k: v for k, v in
            diagnostics(val.label.to_numpy(), scores, val.topic.to_numpy(), val.form.to_numpy()).items()}))
        params.update({fit.method + "__" + key: value for key, value in fit.parameters.items()})
        details[fit.method] = fit.details
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **params)
    payload = buffer.getvalue()
    return dict(layer=layer, metrics=metrics, method_details=details,
                train_activation_sha256=train_hash, validation_activation_sha256=val_hash,
                parameter_archive_sha256=c.digest(payload)), payload


def verify_archive(payload, record, data):
    """Safely reload all parameters and reproduce metrics, including fused TTPD."""
    c.require(c.digest(payload) == record["parameter_archive_sha256"], "archive hash mismatch")
    V, digest = data.matrix("validation", record["layer"])
    c.require(digest == record["validation_activation_sha256"], "validation slice changed")
    val = data.partitions["validation"]
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        params = {k: archive[k] for k in archive.files}
    for value in params.values():
        c.require(value.dtype.kind in "biufU" and (value.dtype.kind == "U" or np.isfinite(value).all()),
                  "invalid parameter archive array")
    for row in record["metrics"]:
        prefix = row["method"] + "__"
        fit = FittedMethod(row["method"], {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}, {})
        scores = fit.decision_function(V)
        c.require(np.isfinite(scores).all(), "nonfinite restored scores")
        if row["method"] == "ttpd":
            c.require(np.allclose(scores, V @ fit.parameters["coef"] + fit.parameters["intercept"],
                                  rtol=1e-9, atol=1e-8), "TTPD fused/explicit mismatch")
        measured = diagnostics(val.label.to_numpy(), scores, val.topic.to_numpy(), val.form.to_numpy())
        c.require(all(row["validation_" + k] == v for k, v in measured.items()), "archive metric reproduction failed")


def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def verify_inputs(data, probe):
    for directory, files in ((c.ATOMIC, data.cache.files), (c.PROBE, probe["files"])):
        for name, expected in files.items():
            c.require(c.record(c.safe_path(data.root, directory + "/" + name)) == expected, "canonical input changed during fit")


def run_one(data, probe, baseline, provenance, mode):
    output = c.safe_path(data.root, OUTPUTS[mode])
    c.require(not output.exists(), "refusing existing output (including incomplete runs)")
    output.mkdir(parents=True, exist_ok=False)
    initial_reads = dict(data.activation_rows_read)
    write_new(output / "provenance.json", json_bytes({**provenance, "comparison": mode,
        "created_utc": datetime.now(timezone.utc).isoformat(), "lr_refitted": mode == "matched"}))
    write_new(output / "allowed_atomic_rows.csv", data.rows.to_csv(index=False).encode())
    write_new(output / "burger_training_rows.csv", data.burger_rows.to_csv(index=False).encode())
    layers = [probe["layer"]] + ([l for l in range(data.n_layers) if l != probe["layer"]] if mode == "faithful" else [])
    rows = []
    with threadpool_limits(limits=1):
        blas = threadpool_info()
        for layer in layers:
            record, payload = (fit_layer(data, layer, baseline, probe["layer"], data.burger_indices, POLICY)
                               if mode == "faithful" else matched_layer(data, probe))
            if mode == "faithful" and layer == probe["layer"]:
                actual = next(r for r in record["metrics"] if r["method"] == "l2_logistic")
                expected = probe["selection"]["selected_validation_metrics"]
                for key, value in actual.items():
                    if key.startswith("validation_") and key in expected:
                        c.require(value is None and expected[key] is None or value is not None and expected[key] is not None
                                  and abs(value - expected[key]) <= 1e-12, f"canonical LR validation changed: {key}")
            archive = output / "layers" / f"layer_{layer:02d}.npz"
            for row in record["metrics"]:
                subset = mode == "matched" or row["method"] in {"burger_t_g", "ttpd"}
                row.update(comparison=mode, parameters=str(archive.relative_to(output)),
                    parameter_archive_sha256=record["parameter_archive_sha256"],
                    train_activation_slice_sha256=data.slice_hashes["burger" if subset else "train", layer],
                    validation_activation_slice_sha256=record["validation_activation_sha256"],
                    ordered_fit_row_ids_sha256=identity_hash(data.burger_rows if subset else data.partitions["train"]),
                    ordered_validation_row_ids_sha256=identity_hash(data.partitions["validation"]))
                c.require(row["fit_rows"] == (1000 if subset else 3144), "method fit count mismatch")
            write_new(archive, payload)
            verify_archive(archive.read_bytes(), record, data)
            record.update(representation_fingerprint=c.FINGERPRINT, archive_verified=True,
                          activation_hash_encoding="C-order float16 selected-layer bytes in exact exported row order")
            write_new(archive.with_suffix(".json"), json_bytes(record))
            rows.extend(record["metrics"])
            print(f"{mode}: verified saved layer {layer}", flush=True)
    rows.sort(key=lambda r: (r["method"], r["layer"]))
    comparisons = choose_layers(rows, probe["layer"])
    if mode == "matched":
        comparisons.pop("secondary_method_selected_layer")
    summary = dict(schema_version=1, comparison=mode, representation_fingerprint=c.FINGERPRINT,
        primary_layer=probe["layer"], comparisons=comparisons,
        selection_rule=POLICY["secondary_rule"] if mode == "faithful" else "fixed layer; no layer/C selection",
        canonical_lr_C=probe["C"], fit_split="train", selection_split="validation",
        lr_refitted=mode == "matched", validation_sign_flip=False, atomic_test_accessed=False, compound_data_accessed=False)
    write_new(output / "validation_metrics.csv", pd.DataFrame(rows).to_csv(index=False, float_format="%.17g").encode())
    write_new(output / "summary.json", json_bytes(summary))
    report = [f"# Pinned Qwen2.5 atomic methods: {mode}", "", "TRAIN fitting; VALIDATION reporting only. No test or compound access.", ""]
    for title, selected in comparisons.items():
        report += [f"## {title}", "", "| Method | Layer | Fit rows | Validation AUROC |", "|---|---:|---:|---:|"]
        report += [f"| {m} | {r['layer']} | {r['fit_rows']} | {r['validation_overall_auroc']:.12g} |" for m, r in selected.items()]
        report.append("")
    write_new(output / "report.md", "\n".join(report).encode())
    verify_inputs(data, probe)
    verification = dict(complete=True, schema_version=1, archive_metrics_reproduced=True,
        ttpd_fused_explicit_agree=True, canonical_inputs_unchanged=True,
        activation_rows_read={s: data.activation_rows_read[s] - initial_reads[s] for s in COUNTS},
        atomic_test_accessed=False, compound_data_accessed=False, validation_sign_flip=False,
        method_fits_performed=4 * len(layers) if mode == "faithful" else 5,
        canonical_lr_reused_without_refit=mode == "faithful", blas=blas,
        outputs={str(p.relative_to(output)): c.record(p) for p in sorted(output.rglob("*")) if p.is_file()})
    # Completion receipt is published last. An interrupted run must use a fresh directory.
    c.publish_json(output / "verification.json", verification)
    return summary


def run(root=ROOT, *, suite="faithful", check_only=False):
    c.require(suite in {"faithful", "matched", "both"}, "unknown suite")
    modes = ["faithful", "matched"] if suite == "both" else [suite]
    if not check_only:
        for mode in modes:
            c.require(not c.safe_path(root, OUTPUTS[mode]).exists(), "refusing existing output (including incomplete runs)")
    data, probe, baseline, provenance = prepare(root)
    if check_only:
        return dict(check_only=True, suites=modes, fits_performed=0, scores_computed=0,
                    activation_rows_read=data.activation_rows_read, activation_matrices_materialized=0,
                    provenance=provenance)
    return {mode: run_one(data, probe, baseline, provenance, mode) for mode in modes}
