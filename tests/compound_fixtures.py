"""Invented atomic statements and manually assigned manifests for tests only."""

import json
from pathlib import Path

from src.clean_compounds import ATOMIC_TEMPLATES, TEMPLATE_VERSION
from src.entity_partitions import FIELDS, TOPICS, csv_bytes, entity_id, sha256


def write_manifest(directory, rows, sources):
    directory = Path(directory)
    payload = csv_bytes(rows, FIELDS)
    (directory / "manifest.csv").write_bytes(payload)
    metadata = {"config": {"schema_version": 1}, "manifest_sha256": sha256(payload),
                "sources": sources, "fixture": "synthetic-only"}
    (directory / "manifest_metadata.json").write_text(json.dumps(metadata))


def synthetic_fixture(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sources").mkdir()
    assignments, sources = [], []
    for topic in TOPICS:
        atomic = []
        # Explicit fixture membership: generator must never recompute these splits.
        for i, split in enumerate(("train",) * 4 + ("validation",) * 2 + ("test",) * 2 + ("train",)):
            entity = f"Synthetic-{i}"
            usable = i != 8
            assignments.append({"topic": topic, "entity": entity, "entity_id": entity_id(topic, entity),
                                "compound_usable": usable, "split": split})
            atomic.append({"statement": ATOMIC_TEMPLATES[topic].format(entity=entity, object=f"Object-{i}"),
                           "label": int(usable)})
        payload = csv_bytes(atomic, ("statement", "label"))
        name = topic + ".csv"
        (directory / "sources" / name).write_bytes(payload)
        sources.append({"file": name, "sha256": sha256(payload), "rows": len(atomic)})
    write_manifest(directory, assignments, sources)
    config = {"schema_version": 1, "entity_manifest": "manifest.csv",
              "entity_manifest_metadata": "manifest_metadata.json", "source_dir": "sources",
              "split": "train", "pair_sampling_seed": 0, "generation_seed": 0,
              "pairs_per_topic": {topic: 1 for topic in TOPICS},
              "template_version": TEMPLATE_VERSION, "output_dir": "clean_protocol/smoke"}
    return config, assignments, sources
