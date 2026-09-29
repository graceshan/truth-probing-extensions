"""Strict repaired-cache adapter for the unchanged clean LR selection code."""
from pathlib import Path

import numpy as np

from src import clean_atomic_probes as selection
from src.repaired_atomic_cache import (
    RepairedAtomicCache, OUTPUT_FILES, array_header,
    ROOT, CACHE, COUNTS, FIELDS, TOPICS, MODEL, REVISION, INPUT_DIGEST,
    LAYERS, WIDTH, canonical, contract, digest, file_hash, plain_path, require,
)

RESULTS = Path("results/clean_protocol/atomic_probes_pinned_v1/qwen25_7b")
MODEL_KEY = "qwen25_7b"
ADAPTER_SOURCES = ("src/pinned_atomic_probes.py", "scripts/34_select_pinned_atomic_probe.py",
                   "src/clean_atomic_extraction.py", "src/repaired_atomic_cache.py")


class RepairedAtomicData(RepairedAtomicCache):
    """No original AtomicData, source export, entity manifest, or test partition."""
    _authorize_test = False

    def __init__(self, root=ROOT):
        super().__init__(root)
        self.model_key = MODEL_KEY
        self.model = {"model": MODEL, "n_layers": LAYERS, "hidden_size": WIDTH,
                      "dtype": "float16", "results_dir": str(RESULTS)}
        self.config = {"schema_version": 1, "loader": "pinned-repaired-cache-v1", "cache_dir": str(CACHE),
                       "C_values": selection.C_VALUES, "models": {MODEL_KEY: self.model}}
        self.structural["adapter_code_sha256"] = {
            p: file_hash(Path(__file__).resolve().parents[1] / p) for p in ADAPTER_SOURCES}
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
