"""Label-free representation compatibility with PRIMARY historical atomic caches.

Never import the probe loader: it parses labels. Read only statement columns,
split identities and ten explicitly selected train/validation activation rows.
"""
from pathlib import Path
import json
import re

import numpy as np

from src.clean_extraction import (MODELS, ROOT, canonical_json, digest, file_hash,
                                  ordered_hash, require, resume_identity, smoke_compare)
from src.data import entity_key_from_statements
from src.extract import extract_batch

PRIMARY_TOPICS = ("cities", "sp_en_trans", "inventors", "element_symb", "animal_class")
POLICY = "historical-float16-within-observed-same-sample-padding-envelope-v1"


def select_historical_sample(model_key, root=ROOT):
    """One train affirmative and one validation negated row per topic (10 rows).

    Selection is the minimum SHA256 of [dataset, row_index, exact_statement]
    within the designated split; no labels, scores or activation values affect it.
    np.load reads the header, then ONLY the selected row is copied from the mmap.
    Whole historical activation files and label-containing CSVs are never hashed.
    """
    import pandas as pd
    root = Path(root)
    config = json.loads((root / "config/clean_protocol/atomic_probes.json").read_text())
    model_config = config["models"]["qwen25_7b" if model_key == "qwen2_5_7b" else model_key]
    spec = MODELS[model_key]
    require(model_config["model"] == spec["identifier"], "historical model identifier mismatch")
    require((model_config["n_layers"], model_config["hidden_size"], model_config["dtype"]) ==
            (spec["layers"], spec["hidden_size"], "float16"), "historical configuration mismatch")
    manifest_path = root / config["entity_manifest"]
    entities = pd.read_csv(manifest_path, usecols=["topic", "entity", "entity_id", "split"],
                           dtype=str, keep_default_na=False)
    require(not entities.duplicated(["topic", "entity"]).any(), "duplicate entity split identity")
    lookup = {(r.topic, r.entity): (r.entity_id, r.split) for r in entities.itertuples()}
    records, values = [], []
    for topic in PRIMARY_TOPICS:
        for form, split in (("affirmative", "train"), ("negated", "validation")):
            dataset = topic if form == "affirmative" else "neg_" + topic
            source_path = root / config["source_dir"] / f"{dataset}.csv"
            sidecar_path = root / model_config["metadata_pattern"].format(dataset=dataset)
            activation_path = root / model_config["activation_pattern"].format(dataset=dataset)
            # No label column is requested or materialized, including for test rows.
            source = pd.read_csv(source_path, usecols=["statement"], dtype=str, keep_default_na=False)
            sidecar = pd.read_csv(sidecar_path, usecols=["statement"], dtype=str, keep_default_na=False)
            require(source.equals(sidecar), f"{dataset}: statement source/sidecar alignment mismatch")
            names = entity_key_from_statements(topic, sidecar.statement)
            require(all((topic, name) in lookup for name in names), "unknown atomic entity")
            eligible = [i for i, name in enumerate(names) if lookup[topic, name][1] == split]
            require(bool(eligible), f"{dataset}: no eligible {split} rows")
            index = min(eligible, key=lambda i: digest(canonical_json([dataset, i, sidecar.statement.iloc[i]])))
            entity_id, actual_split = lookup[topic, names[index]]
            require(actual_split in {"train", "validation"} and actual_split == split,
                    "forbidden atomic split selected")
            array = np.load(activation_path, mmap_mode="r", allow_pickle=False)
            require(array.shape == (len(sidecar), spec["layers"], spec["hidden_size"]) and
                    array.dtype == np.float16, "historical activation header mismatch")
            value = np.array(array[index], copy=True)  # selected row ONLY, all saved layers
            del array
            require(np.isfinite(value).all(), "nonfinite selected historical activation")
            records.append({"dataset": dataset, "topic": topic, "form": form, "split": actual_split,
                            "row_index": index, "entity_id": entity_id,
                            "statement": sidecar.statement.iloc[index],
                            "source_path": str(source_path.relative_to(root)),
                            "sidecar_path": str(sidecar_path.relative_to(root)),
                            "activation_path": str(activation_path.relative_to(root)),
                            "ordered_source_statement_sha256": ordered_hash(source.statement),
                            "ordered_sidecar_statement_sha256": ordered_hash(sidecar.statement),
                            "historical_row_float16_sha256": digest(value.tobytes(order="C"))})
            values.append(value)
    provenance = {"selection": "minimum SHA256([dataset,row_index,statement]); one train affirmative and one validation negated per primary topic",
                  "entity_split_manifest_sha256": file_hash(manifest_path), "rows": records,
                  "labels_loaded": False, "test_activation_rows_read": 0}
    return provenance, np.stack(values)


def absolute_differences(reference, current):
    require(reference.shape == current.shape, "historical/current shape mismatch")
    require(np.isfinite(reference).all() and np.isfinite(current).all(), "nonfinite compatibility input")
    delta = np.abs(reference.astype(np.float64) - current.astype(np.float64))
    return {"max_absolute_difference": float(delta.max()), "mean_absolute_difference": float(delta.mean()),
            "per_layer": [{"saved_layer": i, "max_absolute_difference": float(delta[:, i].max()),
                           "mean_absolute_difference": float(delta[:, i].mean())} for i in range(delta.shape[1])],
            "per_statement": [{"sample_index": i, "max_absolute_difference": float(d.max()),
                               "mean_absolute_difference": float(d.mean())} for i, d in enumerate(delta)]}


def compatibility_decision(historical, fresh, same_sample_padding):
    """No fitted/tuned tolerance: empirical per-layer max AND mean, no multiplier.

    The pre-cast comparison necessarily includes historical float16 quantization.
    Gate on like-for-like float16 values. Exact bytes pass; otherwise both errors
    must be no larger than measured current single-vs-padding noise in every layer.
    This conservative rule may reject harmless kernel variation; never auto-relax it.
    """
    saved = fresh.astype(np.float16)
    before = absolute_differences(historical, fresh)
    after = absolute_differences(historical, saved)
    identical = historical.tobytes(order="C") == saved.tobytes(order="C")
    comparisons = same_sample_padding["comparisons"]
    bounds = []
    for i, layer in enumerate(after["per_layer"]):
        maximum = max(comparisons[side]["saved_float16"]["per_layer"][i]["max_absolute_difference"]
                      for side in ("right", "left"))
        mean = max(comparisons[side]["saved_float16"]["per_layer"][i]["mean_absolute_difference"]
                   for side in ("right", "left"))
        bounds.append({"saved_layer": i, "observed_padding_max": maximum, "observed_padding_mean": mean,
                       "historical_max_minus_padding_max": layer["max_absolute_difference"] - maximum,
                       "historical_mean_minus_padding_mean": layer["mean_absolute_difference"] - mean,
                       "within_observed_envelope": layer["max_absolute_difference"] <= maximum and
                                                   layer["mean_absolute_difference"] <= mean})
    passed = same_sample_padding["passed"] and (identical or all(b["within_observed_envelope"] for b in bounds))
    return {"policy": POLICY, "passed": bool(passed), "historical_saved_dtype": "float16",
            "fresh_before_float16_conversion": before, "after_float16_conversion": after,
            "float16_byte_identical": identical, "observed_padding_comparison_per_layer": bounds,
            "decision": "compatible on sampled rows" if passed else
                        "STOP: historical representation mismatch or same-sample padding failure; no tolerance widening or atomic re-extraction",
            "scope": "representation compatibility on ten sampled rows; no probe evaluation"}


def historical_compatibility_smoke(model, tokenizer, model_key, manifest, *, root=ROOT):
    provenance, historical = select_historical_sample(model_key, root)
    statements = [r["statement"] for r in provenance["rows"]]
    # Exactly the clean single-example regime: raw statements, no padding,
    # add_special_tokens=True, HF[1:], explicit IDs equal 0..length-1.
    fresh = np.concatenate([extract_batch(model, tokenizer, [s], padding=False).numpy() for s in statements])
    # Observe batch noise on these same statements, independent of history.
    padding = smoke_compare(model, tokenizer, statements, batch_size=max(2, manifest["numerics"]["batch_size"]))
    report = compatibility_decision(historical, fresh, padding)
    report.update(sample=provenance, sample_sha256=digest(canonical_json(provenance)),
                  current_model_identifier=manifest["model"]["identifier"],
                  current_model_commit_sha=manifest["model"]["resolved_commit_sha"],
                  current_tokenizer_commit_sha=manifest["tokenizer"]["resolved_commit_sha"],
                  current_compute_dtype=str(model.dtype), historical_reference_batch_size=1,
                  historical_reference_padding=False, same_sample_padding_smoke=padding,
                  general_padding_smoke=manifest["smoke_test"])
    return report


def require_both_gates(manifest):
    require(manifest.get("smoke_test", {}).get("passed") is True,
            "passing full-weight padding smoke required before extraction")
    historical = manifest.get("historical_atomic_compatibility", {})
    require(historical.get("passed") is True, "passing historical atomic compatibility smoke required before extraction")
    for key in ("model", "tokenizer"):
        revision = manifest[key].get("resolved_commit_sha", "")
        require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision) is not None,
                f"resolved immutable {key} revision required")
        require(historical.get(f"current_{key}_commit_sha") == revision, "compatibility revision mismatch")
    require(historical.get("current_model_identifier") == manifest["model"]["identifier"],
            "compatibility model identifier mismatch")
    require(historical.get("sample_sha256") == manifest.get("historical_sample_sha256"),
            "compatibility sample identity mismatch")


def save_compatibility_pin(path, report_path, manifest):
    """Immutable evidence-backed revision receipt. Never write one after failure."""
    require_both_gates(manifest)
    path, report_path = Path(path), Path(report_path)
    receipt = {"schema_version": 1, "model_identifier": manifest["model"]["identifier"],
               "model_commit_sha": manifest["model"]["resolved_commit_sha"],
               "tokenizer_commit_sha": manifest["tokenizer"]["resolved_commit_sha"],
               "extraction_identity_sha256": resume_identity(manifest),
               "report_path": str(report_path.resolve().relative_to(ROOT)) if report_path.resolve().is_relative_to(ROOT)
                              else str(report_path.resolve()),
               "report_sha256": file_hash(report_path)}
    require(json.loads(report_path.read_text()) == manifest, "pin evidence differs from current diagnostics")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old, _ = load_compatibility_pin(path, manifest["model"]["identifier"])
        require(old["extraction_identity_sha256"] == receipt["extraction_identity_sha256"],
                "existing compatibility pin differs; stop for review (no overwrite)")
        return
    with path.open("x") as f:
        json.dump(receipt, f, indent=2)
        f.write("\n")


def load_compatibility_pin(path, identifier):
    receipt = json.loads(Path(path).read_text())
    require(receipt["model_identifier"] == identifier, "pin model mismatch")
    report_path = ROOT / receipt["report_path"]
    require(file_hash(report_path) == receipt["report_sha256"], "pin evidence hash mismatch")
    manifest = json.loads(report_path.read_text())
    require_both_gates(manifest)
    require(receipt["model_identifier"] == manifest["model"]["identifier"], "pin/evidence model mismatch")
    for key in ("model", "tokenizer"):
        require(receipt[f"{key}_commit_sha"] == manifest[key]["resolved_commit_sha"], "pin/evidence revision mismatch")
    require(receipt["extraction_identity_sha256"] == resume_identity(manifest), "pin evidence identity mismatch")
    return receipt, manifest
