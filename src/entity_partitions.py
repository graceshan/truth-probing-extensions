"""Shared atomic entity assignments; never generates or scores examples."""

import csv
import hashlib
import io
import json
import re
from itertools import combinations
from pathlib import Path

from src.data import ENTITY_PATTERNS


TOPICS = ("cities", "sp_en_trans", "inventors", "element_symb", "animal_class")
SPLITS = ("train", "validation", "test")
FIELDS = ("topic", "entity", "entity_id", "compound_usable", "split")
ALGORITHM = "sha256-topic-entity-usable-stratified-v1"
# The affirmative templates used by scripts/01_generate_r1_r2_datasets.py.
TRUE_OBJECT_PATTERNS = {
    "cities": r"The city of (.+?) is in (.+?)\.",
    "sp_en_trans": r"The Spanish word '(.+?)' means '(.+?)'\.",
    "inventors": r"(.+?) lived in (.+?)\.",
    "element_symb": r"(.+?) has the symbol (.+?)\.",
    "animal_class": r"The (.+?) is an? (.+?)\.",
}


def require(condition, message):
    """Assertions remain active even when Python runs with -O."""
    if not condition:
        raise AssertionError(message)


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha256(payload):
    return hashlib.sha256(payload).hexdigest()


def entity_id(topic, entity):
    """Identity preserves exact spelling, case, and Unicode; no normalization."""
    return "entity_" + sha256(canonical_json([topic, entity]))


def read_entities(source_dir):
    """Union both atomic forms, retaining entities with no true affirmative row.

    Usability is source-label grounded, not an independent factual verification:
    a unique true object must be supplied by the affirmative generator source.
    Negated false statements do not add inferred true-object mappings.
    """
    records, provenance = [], []
    for topic in TOPICS:
        entities, true_objects = set(), {}
        for name in (topic, "neg_" + topic):
            path = Path(source_dir) / (name + ".csv")
            payload = path.read_bytes()
            reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
            require({"statement", "label"} <= set(reader.fieldnames or ()),
                    f"{name}: missing atomic columns")
            row_count = 0
            for row_count, row in enumerate(reader, 1):
                match = re.search(ENTITY_PATTERNS[topic], row["statement"])
                require(match is not None, f"{name}: unparseable entity at row {row_count}")
                entity = match[1]
                entities.add(entity)
                require(row["label"] in ("0", "1"), f"{name}: invalid binary label")
                if name == topic and row["label"] == "1":
                    obj = re.fullmatch(TRUE_OBJECT_PATTERNS[topic], row["statement"])
                    require(obj is not None and obj[1] == entity,
                            f"{name}: true-object parser mismatch at row {row_count}")
                    true_objects.setdefault(entity, set()).add(obj[2])
            provenance.append({"file": path.name, "sha256": sha256(payload), "rows": row_count})
        require(all(len(objects) == 1 for objects in true_objects.values()),
                f"{topic}: ambiguous true-object mapping; source audit required")
        for entity in sorted(entities):
            records.append({"topic": topic, "entity": entity,
                            "entity_id": entity_id(topic, entity),
                            "compound_usable": entity in true_objects})
    return records, provenance


def assign_splits(records, seed):
    """Rank each topic/usability stratum by SHA-256, then allocate 60/20/20.

    Rank key: SHA-256 of canonical JSON [seed, topic, exact entity]. Validation
    and test each get floor(n/5); train gets the remainder. Identity breaks a
    hypothetical rank-hash tie. Output order is topic order then exact entity.
    """
    require(type(seed) is int, "seed must be an explicit integer")
    assigned = []
    for topic in TOPICS:
        for usable in (False, True):
            group = [dict(r) for r in records
                     if r["topic"] == topic and r["compound_usable"] == usable]
            group.sort(key=lambda r: (sha256(canonical_json([seed, topic, r["entity"]])),
                                      r["entity"]))
            heldout = len(group) // 5
            train_end = len(group) - 2 * heldout
            for index, record in enumerate(group):
                record["split"] = ("train" if index < train_end else
                                   "validation" if index < train_end + heldout else "test")
                assigned.append(record)
    return sorted(assigned, key=lambda r: (TOPICS.index(r["topic"]), r["entity"]))


def validate(records, assigned, config):
    keys = [(r["topic"], r["entity"]) for r in assigned]
    require(len(keys) == len(set(keys)), "duplicate topic/entity identity")
    require(len({r["entity_id"] for r in assigned}) == len(assigned), "entity_id collision")
    require(all(r["entity_id"] == entity_id(r["topic"], r["entity"]) for r in assigned),
            "incorrect entity_id")
    require(all(r["split"] in SPLITS for r in assigned), "unknown split")
    source_records = {(r["topic"], r["entity"]): r for r in records}
    require(len(source_records) == len(records), "duplicate source entity record")
    require(set(keys) == set(source_records), "atomic entity coverage changed")
    require(all({k: r[k] for k in FIELDS if k != "split"} ==
                source_records[(r["topic"], r["entity"])] for r in assigned),
            "source identity or compound usability changed")
    split_sets = {s: {(r["topic"], r["entity"]) for r in assigned if r["split"] == s}
                  for s in SPLITS}
    for left, right in combinations(SPLITS, 2):
        require(split_sets[left].isdisjoint(split_sets[right]), "entity split overlap")
    require(assigned == assign_splits(records, config["seed"]), "same-seed reproduction failed")
    require(assigned == assign_splits(list(reversed(records)), config["seed"]),
            "assignments depend on source entity order")
    counts = []
    for topic in TOPICS:
        topic_rows = [r for r in assigned if r["topic"] == topic]
        expected = config["expected_counts"][topic]
        require(len(topic_rows) == expected["entities"], f"{topic}: unexpected entity count")
        require(sum(r["compound_usable"] for r in topic_rows) == expected["compound_usable"],
                f"{topic}: unexpected compound-usable count")
        for usable in (False, True):
            group = [r for r in topic_rows if r["compound_usable"] == usable]
            n = len(group)
            targets = dict(zip(SPLITS, (n - 2 * (n // 5), n // 5, n // 5)))
            for split in SPLITS:
                require(sum(r["split"] == split for r in group) == targets[split],
                        f"{topic}: incorrect stratified allocation")
        for split in SPLITS:
            subset = [r for r in topic_rows if r["split"] == split]
            usable = sum(r["compound_usable"] for r in subset)
            counts.append({"topic": topic, "split": split, "entities": len(subset),
                           "compound_usable": usable,
                           "max_unordered_usable_pairs": usable * (usable - 1) // 2})
    return counts


def csv_bytes(rows, fields):
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def build_manifest(repo_root, config):
    require(config["schema_version"] == 1, "unsupported schema version")
    require(config["split_percentages"] == {"train": 60, "validation": 20, "test": 20},
            "this algorithm implements only 60/20/20")
    require(set(config["expected_counts"]) == set(TOPICS), "unexpected topic configuration")
    records, sources = read_entities(Path(repo_root) / config["source_dir"])
    assigned = assign_splits(records, config["seed"])
    counts = validate(records, assigned, config)
    manifest = csv_bytes(assigned, FIELDS)
    metadata = {
        "config": config, "algorithm": ALGORITHM,
        "entity_id": "entity_ + SHA256(UTF-8 compact JSON [topic, entity], ensure_ascii=False)",
        "rank_key": "SHA256(UTF-8 compact JSON [seed, topic, entity], ensure_ascii=False)",
        "allocation": "Within topic x usability: validation=test=floor(n/5); train=remainder",
        "compound_usable_definition": "Unique true object in affirmative generator source",
        "scope": "Entity assignments only; no final-test examples generated or scored",
        "sources": sources, "manifest_sha256": sha256(manifest),
        "assertions": {name: True for name in (
            "unique_identity", "unique_entity_id", "complete_atomic_entity_coverage",
            "pairwise_disjoint_splits", "same_seed_reproduction", "input_order_independence",
            "expected_topic_counts", "expected_compound_usable_counts", "stratified_counts")},
        "counts": counts,
    }
    files = {
        "manifest.csv": manifest,
        "counts.csv": csv_bytes(counts, tuple(counts[0])),
        "metadata.json": (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    }
    return files, counts


def save_outputs(output_dir, files):
    """Create outputs exclusively; identical reruns verify without overwriting."""
    output_dir = Path(output_dir)
    for name, payload in files.items():
        path = output_dir / name
        if path.exists():
            require(path.read_bytes() == payload, f"refusing to overwrite differing output: {name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, payload in files.items():
        path = output_dir / name
        if not path.exists():
            with path.open("xb") as handle:
                handle.write(payload)
        require(path.read_bytes() == payload, f"saved output verification failed: {name}")
