"""Strict repaired-cache adapter for the unchanged clean LR selection code."""
import csv
import json
from pathlib import Path

import numpy as np

from src import clean_atomic_probes as selection
from src.clean_atomic_extraction import (
    ROOT, OUTPUT as CACHE, COUNTS, FIELDS, TOPICS, MODEL, REVISION, INPUT_DIGEST,
    LAYERS, WIDTH, canonical, contract, digest, file_hash, plain_path, require,
)

RESULTS = Path("results/clean_protocol/atomic_probes_pinned_v1/qwen25_7b")
MODEL_KEY = "qwen25_7b"
OUTPUT_FILES = tuple(f"{split}/{name}" for split in ("train", "validation")
                     for name in ("activations.npy", "metadata.csv"))
ADAPTER_SOURCES = ("src/pinned_atomic_probes.py", "scripts/34_select_pinned_atomic_probe.py",
                   "src/clean_atomic_extraction.py")


def array_header(path, expected_shape):
    """Read only the NPY header; byte hashes are checked separately."""
    with path.open("rb") as handle:
        version = np.lib.format.read_magic(handle)
        require(version in {(1, 0), (2, 0)}, "unsupported activation NPY version")
        reader = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                  else np.lib.format.read_array_header_2_0)
        shape, fortran, dtype = reader(handle)
        require(shape == expected_shape, "activation shape mismatch")
        require(dtype == np.dtype("float16") and not fortran, "activation dtype/order mismatch")
        require(path.stat().st_size == handle.tell() + int(np.prod(shape)) * dtype.itemsize,
                "activation payload size mismatch")


class RepairedAtomicData:
    """No original AtomicData, source export, entity manifest, or test partition."""
    _authorize_test = False

    def __init__(self, root=ROOT):
        self.root, self.model_key = Path(root).resolve(), MODEL_KEY
        self.n_layers, self.hidden_size = LAYERS, WIDTH
        # Resolve only fixed paths. Never follow filenames supplied by a receipt.
        paths = {name: plain_path(self.root, CACHE / name) for name in
                 ("completion.json", "extraction_manifest.json", *OUTPUT_FILES)}
        completion_payload = paths["completion.json"].read_bytes()
        completion = json.loads(completion_payload)
        require(completion.get("complete") is True, "completed repaired cache required")
        require(completion.get("counts") == COUNTS, "completion counts mismatch")
        outputs = completion.get("outputs", {})
        require(set(outputs) == set(OUTPUT_FILES), "completion output allowlist mismatch")
        manifest_payload = paths["extraction_manifest.json"].read_bytes()
        require(digest(manifest_payload) == completion.get("manifest_sha256"), "extraction manifest hash mismatch")
        manifest = json.loads(manifest_payload)
        require(manifest.get("schema_version") == 1 and
                manifest.get("implementation") == "qwen25-pinned-atomic-repair-v1", "wrong extraction implementation")
        require(manifest.get("contract") == contract(), "extraction convention/revision mismatch")
        resolved = manifest.get("resolved_model", {})
        require(resolved.get("model_resolved_revision") == REVISION and
                resolved.get("tokenizer_resolved_revision") == REVISION, "resolved revision mismatch")
        runtime = manifest.get("runtime", {})
        require(runtime.get("torch", "").split("+")[0] == "2.11.0" and
                runtime.get("transformers") == "5.12.1", "extraction runtime mismatch")
        source = manifest.get("input", {})
        require(source.get("allowed_rows_sha256") == INPUT_DIGEST, "repaired input digest mismatch")
        require(source.get("counts") == COUNTS and source.get("columns") == list(FIELDS), "input counts/schema mismatch")
        require(manifest.get("activation_shapes") == {s: [n, LAYERS, WIDTH] for s, n in COUNTS.items()},
                "manifest activation shapes mismatch")
        smoke = manifest.get("smoke", {})
        require(smoke.get("passed") is True and smoke.get("batch_size") == 1 and smoke.get("padding") is False and
                smoke.get("policy") == "unpadded-single-repeat-and-independent-forward-exact-v1",
                "passing unpadded extraction smoke required")
        files = {"completion.json": {"sha256": digest(completion_payload), "bytes": len(completion_payload)},
                 "extraction_manifest.json": {"sha256": digest(manifest_payload), "bytes": len(manifest_payload)}}
        for name in OUTPUT_FILES:
            path, expected = paths[name], outputs[name]
            require(path.stat().st_size == expected["bytes"], f"output size mismatch: {name}")
            actual = file_hash(path)
            require(actual == expected["sha256"], f"output hash mismatch: {name}")
            files[name] = {"sha256": actual, "bytes": expected["bytes"]}

        self.rows, self.counts = {}, []
        identities = set()
        for split, count in COUNTS.items():
            with paths[f"{split}/metadata.csv"].open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                require(tuple(reader.fieldnames or ()) == FIELDS, "metadata schema mismatch")
                rows = []
                for row in reader:
                    require(None not in row and all(row.get(k) is not None for k in FIELDS), "malformed metadata row")
                    require(row["split"] == split, "metadata contains forbidden/wrong split")
                    require(row["topic"] in TOPICS and row["form"] in {"affirmative", "negated"}, "metadata topic/form invalid")
                    require(row["dataset"] == ("neg_" if row["form"] == "negated" else "") + row["topic"],
                            "metadata dataset mismatch")
                    require(row["row_index"].isdigit() and row["label"] in {"0", "1"}, "metadata row index/label invalid")
                    require(bool(row["statement"].strip()) and bool(row["entity_id"]), "empty metadata statement/entity")
                    row["row_index"], row["label"] = int(row["row_index"]), int(row["label"])
                    identity = (row["dataset"], row["row_index"])
                    require(identity not in identities, "duplicate dataset/row_index")
                    identities.add(identity)
                    rows.append(row)
            require(len(rows) == count, "metadata row count mismatch")
            require({r["label"] for r in rows} == {0, 1}, "both label classes required in each split")
            require(digest(canonical(rows)) == source.get("partition_rows_sha256", {}).get(split),
                    "metadata partition digest mismatch")
            array_header(paths[f"{split}/activations.npy"], (count, LAYERS, WIDTH))
            self.rows[split] = rows
            for topic in TOPICS:
                group = [r for r in rows if r["topic"] == topic]
                self.counts.append({"topic": topic, "split": split, "rows": len(group),
                                    "entities": len({r["entity_id"] for r in group}),
                                    "affirmative_rows": sum(r["form"] == "affirmative" for r in group),
                                    "negated_rows": sum(r["form"] == "negated" for r in group)})
        require({r["entity_id"] for r in self.rows["train"]}.isdisjoint(
                r["entity_id"] for r in self.rows["validation"]), "train/validation entity overlap")
        # Original export order: topic, affirmative/negated dataset, source index.
        # Reconstruct the approved digest without reopening the original export.
        combined = sorted(self.rows["train"] + self.rows["validation"], key=lambda r: (
            TOPICS.index(r["topic"]), r["form"] == "negated", r["row_index"]))
        require(digest(canonical(combined)) == INPUT_DIGEST, "metadata does not reconstruct approved input digest")
        self.model = {"model": MODEL, "n_layers": LAYERS, "hidden_size": WIDTH,
                      "dtype": "float16", "results_dir": str(RESULTS)}
        self.config = {"schema_version": 1, "loader": "pinned-repaired-cache-v1", "cache_dir": str(CACHE),
                       "C_values": selection.C_VALUES, "models": {MODEL_KEY: self.model}}
        self.structural = {"model": MODEL, "cache_dir": str(CACHE), "counts": self.counts,
                           "repaired_cache_files": files, "approved_input_digest": INPUT_DIGEST,
                           "extraction_contract": manifest["contract"],
                           "adapter_code_sha256": {p: file_hash(Path(__file__).resolve().parents[1] / p)
                                                   for p in ADAPTER_SOURCES},
                           "activation_bytes_hashed": True, "activation_values_materialized_for_checks": False,
                           "test_metrics_evaluated": False}
        # Shared fit_and_save checks this zero counter; it grants no test access.
        self.activation_rows_read = {"train": 0, "validation": 0, "test": 0}

    def partition(self, split):
        return RepairedPartition(self, split)


class RepairedPartition:
    def __init__(self, data, split):
        if split not in ("train", "validation"):
            raise PermissionError("pinned loader exposes only train and validation")
        self._data, self.split = data, split
        self.n_layers, self.hidden_size = data.n_layers, data.hidden_size
        rows = data.rows[split]
        self.entity_ids = frozenset(r["entity_id"] for r in rows)
        self.labels = np.asarray([r["label"] for r in rows], dtype=np.int64)
        self.topics = np.asarray([r["topic"] for r in rows])
        self.forms = np.asarray([r["form"] for r in rows])

    def matrix(self, layer):
        if self.split not in ("train", "validation"):
            raise PermissionError("pinned loader exposes only train and validation")
        require(0 <= layer < self.n_layers, "invalid layer index")
        path = plain_path(self._data.root, CACHE / self.split / "activations.npy")
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        matrix = np.asarray(array[:, layer, :], dtype=np.float64)
        require(np.isfinite(matrix).all(), f"nonfinite {self.split} activations")
        self._data.activation_rows_read[self.split] += len(matrix)
        return matrix


def run(*, check_only=False, root=ROOT):
    directory = plain_path(root, RESULTS)
    if not check_only:
        require(not directory.exists(), "refusing existing pinned selection directory")
    data = RepairedAtomicData(root)
    if check_only:
        return {"model": MODEL, "structural": data.structural, "activation_rows_read": data.activation_rows_read,
                "fits_performed": 0, "test_evaluated": False}
    # Reserve once, before fitting, so concurrent runs cannot replace results.
    directory.mkdir(parents=True, exist_ok=False)
    def progress(layer, rows):
        print(f"{MODEL_KEY}: layer {layer}, {len(rows)} C values converged", flush=True)
    return selection.fit_and_save(data, directory, progress)
