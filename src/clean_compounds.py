"""Manifest-authorized compound generation, independent of exploratory scripts."""

import csv
import heapq
import io
import json
import re
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from types import MappingProxyType

from src.data import ENTITY_PATTERNS
from src.entity_partitions import (
    SPLITS, TOPICS, TRUE_OBJECT_PATTERNS, canonical_json, csv_bytes, entity_id, sha256,
)

TEMPLATE_VERSION = "clean-binary-v1"
FIELDS = (
    "example_id", "statement", "topic", "split", "entity_a_id", "entity_b_id",
    "fact_a_id", "fact_b_id", "pair_id", "canonical_truth_a", "canonical_truth_b",
    "surface_first_entity_id", "surface_second_entity_id", "surface_first_truth",
    "surface_second_truth", "operator", "ordering", "template_id", "compound_label",
    "generation_seed", "fact_a_statement", "fact_b_statement",
)
ATOMIC_TEMPLATES = {
    "cities": "The city of {entity} is in {object}.",
    "sp_en_trans": "The Spanish word '{entity}' means '{object}'.",
    "inventors": "{entity} lived in {object}.",
    "element_symb": "{entity} has the symbol {object}.",
    "animal_class": "The {entity} is a {object}.",
}


def check(condition, message):
    if not condition:
        raise ValueError(message)


def stable_id(kind, inputs):
    return kind + "_" + sha256(canonical_json([kind + "-v1", *inputs]))


def boolean_truth(operator, values):
    """Nonempty Boolean sequences; n-ary XOR means odd parity, not exactly one."""
    values = tuple(values)
    check(bool(values) and all(type(v) is bool for v in values),
          "truth evaluation requires nonempty Boolean inputs")
    operator = operator.upper()
    if operator == "AND":
        return all(values)
    if operator == "OR":
        return any(values)
    if operator == "XOR":
        return sum(values) % 2 == 1
    raise ValueError(f"unsupported Boolean operator: {operator}")


@dataclass(frozen=True)
class Entity:
    topic: str
    entity: str
    entity_id: str
    compound_usable: bool
    split: str


class EntityManifest:
    """Validate stored membership only; never create or recompute a split."""

    def __init__(self, manifest_path, metadata_path):
        payload = Path(manifest_path).read_bytes()
        metadata_bytes = Path(metadata_path).read_bytes()
        self.metadata = json.loads(metadata_bytes)
        self.digest = sha256(payload)
        self.metadata_digest = sha256(metadata_bytes)
        check(self.metadata["manifest_sha256"] == self.digest, "entity manifest hash mismatch")
        self.version = self.metadata["config"]["schema_version"]
        check(self.version == 1, "unsupported entity manifest version")
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8")))
        check({"topic", "entity", "entity_id", "compound_usable", "split"} <=
              set(reader.fieldnames or ()), "missing entity manifest columns")
        records, identities = {}, set()
        split_sets = {s: set() for s in SPLITS}
        for row in reader:
            check(row["topic"] in TOPICS and row["split"] in SPLITS,
                  "unknown manifest topic or split")
            check(row["compound_usable"] in ("True", "False"), "invalid usability flag")
            key = (row["topic"], row["entity"])
            check(key not in identities, "duplicate entity identity or entity in multiple splits")
            check(row["entity_id"] == entity_id(*key), "invalid manifest entity_id")
            check(row["entity_id"] not in records, "duplicate entity_id")
            identities.add(key)
            split_sets[row["split"]].add(key)
            record = Entity(row["topic"], row["entity"], row["entity_id"],
                            row["compound_usable"] == "True", row["split"])
            records[record.entity_id] = record
        check(bool(records), "empty entity manifest")
        for left, right in combinations(SPLITS, 2):
            check(split_sets[left].isdisjoint(split_sets[right]), "entity split leakage")
        self.entities = MappingProxyType(records)

    def eligible(self, topic, split):
        check(topic in TOPICS and split in SPLITS, "unknown requested topic or split")
        return sorted((e for e in self.entities.values()
                       if e.topic == topic and e.split == split and e.compound_usable),
                      key=lambda e: e.entity_id)

    def authorize_pair(self, left_id, right_id, split):
        check(split in SPLITS, "unknown requested split")
        check(left_id in self.entities and right_id in self.entities, "entity absent from manifest")
        left, right = self.entities[left_id], self.entities[right_id]
        check(left.entity_id != right.entity_id, "self-pairs are forbidden")
        check(left.split == right.split == split, "cross-split or wrong-split pair forbidden")
        check(left.topic == right.topic, "cross-topic pair forbidden")
        check(left.compound_usable and right.compound_usable, "compound-unusable entity forbidden")
        return tuple(sorted((left, right), key=lambda e: e.entity_id))


def unordered_pair_id(manifest, left_id, right_id, split):
    a, b = manifest.authorize_pair(left_id, right_id, split)
    return stable_id("pair", [a.topic, [a.entity_id, b.entity_id]])


def sample_pairs(manifest, topic, split, count, seed):
    """Hash-rank unique unordered candidates; no rejection loop or split assignment."""
    check(type(count) is int and count >= 0, "pair count must be a nonnegative integer")
    check(type(seed) is int, "pair-sampling seed must be an integer")
    entities = manifest.eligible(topic, split)
    capacity = len(entities) * (len(entities) - 1) // 2
    check(count <= capacity,
          f"{topic}/{split}: requested {count} pairs, capacity {capacity} from {len(entities)} usable entities")
    if count == 0:
        return []
    candidates = combinations([e.entity_id for e in entities], 2)
    def rank(pair):
        return (sha256(canonical_json(["pair-sampling-v1", seed, topic, split, list(pair)])), pair)
    return sorted(heapq.nsmallest(count, candidates, key=rank))


@dataclass(frozen=True)
class Fact:
    fact_id: str
    statement: str
    truth: bool


def render_binary(first, second, operator):
    def lower_article(text):
        return "the" + text[3:] if text.startswith("The ") else text
    first, second = first.removesuffix("."), second.removesuffix(".")
    if operator == "XOR":
        return f"Either {lower_article(first)} or {lower_article(second)}, but not both."
    return f"{first} {operator.lower()} {lower_article(second)}."


class CompoundGenerator:
    def __init__(self, manifest, source_dir, split, generation_seed, template_version):
        check(split in SPLITS, "unknown requested split")
        check(type(generation_seed) is int, "generation seed must be an integer")
        check(template_version == TEMPLATE_VERSION, "unsupported template version")
        self.manifest, self.source_dir, self.split = manifest, Path(source_dir), split
        self.seed, self.template_version = generation_seed, template_version
        self._facts, self.source_hashes = {}, {}

    def _load_facts(self, topic):
        """Build facts only for usable entities in the requested partition."""
        if topic in self._facts:
            return self._facts[topic]
        eligible = {e.entity: e for e in self.manifest.eligible(topic, self.split)}
        source_name = topic + ".csv"
        payload = (self.source_dir / source_name).read_bytes()
        expected = {s["file"]: s["sha256"] for s in self.manifest.metadata["sources"]}
        check(source_name in expected and sha256(payload) == expected[source_name],
              "atomic source hash differs from entity-manifest provenance")
        objects = {e: set() for e in eligible}
        true_statements = {}
        for row in csv.DictReader(io.StringIO(payload.decode("utf-8"))):
            match = re.search(ENTITY_PATTERNS[topic], row["statement"])
            check(match is not None, "unparseable source entity")
            if match[1] not in eligible or row["label"] != "1":
                continue
            parsed = re.fullmatch(TRUE_OBJECT_PATTERNS[topic], row["statement"])
            check(parsed is not None and parsed[1] == match[1], "invalid true-object source")
            objects[match[1]].add(parsed[2])
            true_statements[match[1]] = row["statement"]
        check(all(len(v) == 1 for v in objects.values()), "usable entity lacks a unique true object")
        correct = {e: next(iter(v)) for e, v in objects.items()}
        pool = sorted(set(correct.values()))
        facts = {}
        for entity, record in eligible.items():
            candidates = [obj for obj in pool if obj != correct[entity]]
            check(bool(candidates), f"{topic}/{self.split}: no distinct within-split wrong object")
            wrong = min(candidates, key=lambda obj: (
                sha256(canonical_json(["wrong-object-v1", self.seed, topic, record.entity_id, obj])), obj))
            for truth, statement in (
                (True, true_statements[entity]),
                (False, ATOMIC_TEMPLATES[topic].format(entity=entity, object=wrong)),
            ):
                fid = stable_id("fact", [topic, record.entity_id, statement, truth])
                facts[(record.entity_id, truth)] = Fact(fid, statement, truth)
        self._facts[topic] = facts
        self.source_hashes[source_name] = sha256(payload)
        return facts

    def example(self, left_id, right_id, truth_a, truth_b, operator, ordering):
        a, b = self.manifest.authorize_pair(left_id, right_id, self.split)
        check(type(truth_a) is bool and type(truth_b) is bool, "canonical truths must be Boolean")
        check(ordering in ("AB", "BA"), "unsupported surface ordering")
        operator = operator.upper()
        label = boolean_truth(operator, (truth_a, truth_b))
        facts = self._load_facts(a.topic)
        fa, fb = facts[(a.entity_id, truth_a)], facts[(b.entity_id, truth_b)]
        first, second = ((a, fa), (b, fb)) if ordering == "AB" else ((b, fb), (a, fa))
        statement = render_binary(first[1].statement, second[1].statement, operator)
        pid = unordered_pair_id(self.manifest, a.entity_id, b.entity_id, self.split)
        template_id = f"{self.template_version}/{a.topic}/{operator.lower()}"
        eid = stable_id("example", [pid, self.split, fa.fact_id, fb.fact_id, operator,
                                     ordering, template_id, statement])
        return dict(zip(FIELDS, (
            eid, statement, a.topic, self.split, a.entity_id, b.entity_id, fa.fact_id,
            fb.fact_id, pid, truth_a, truth_b, first[0].entity_id, second[0].entity_id,
            first[1].truth, second[1].truth, operator, ordering, template_id, label,
            self.seed, fa.statement, fb.statement,
        )))

    def standard_r1_pair(self, left_id, right_id):
        self.manifest.authorize_pair(left_id, right_id, self.split)
        return [self.example(left_id, right_id, ta, tb, operator, ordering)
                for ta, tb in product((True, False), repeat=2)
                for operator in ("AND", "OR") for ordering in ("AB", "BA")]


def validate_r1(rows, manifest, split, requested):
    check(len({r["example_id"] for r in rows}) == len(rows), "duplicate example_id")
    groups = {}
    for row in rows:
        a, b = manifest.authorize_pair(row["entity_a_id"], row["entity_b_id"], split)
        check(row["split"] == split and row["topic"] == a.topic, "incorrect row topic/split")
        check((row["entity_a_id"], row["entity_b_id"]) == (a.entity_id, b.entity_id),
              "noncanonical entity ordering")
        check(row["pair_id"] == unordered_pair_id(manifest, a.entity_id, b.entity_id, split),
              "incorrect unordered pair_id")
        truths = (row["canonical_truth_a"], row["canonical_truth_b"])
        check(row["compound_label"] == boolean_truth(row["operator"], truths), "incorrect truth label")
        surface = (a.entity_id, b.entity_id, *truths) if row["ordering"] == "AB" else (
            b.entity_id, a.entity_id, truths[1], truths[0])
        check(surface == tuple(row[k] for k in ("surface_first_entity_id", "surface_second_entity_id",
                                               "surface_first_truth", "surface_second_truth")),
              "canonical/surface labels disagree")
        groups.setdefault(row["pair_id"], []).append(row)
    expected_variants = set(product((True, False), (True, False), ("AND", "OR"), ("AB", "BA")))
    for group in groups.values():
        variants = {(r["canonical_truth_a"], r["canonical_truth_b"], r["operator"], r["ordering"])
                    for r in group}
        check(len(group) == 16 and variants == expected_variants, "expected exactly 16 R1 variants")
    check(len(rows) == 16 * sum(requested.values()), "incorrect total R1 row count")
    for topic, count in requested.items():
        check(sum(group[0]["topic"] == topic for group in groups.values()) == count,
              "incorrect per-topic pair count")


def build_generation(config, base_dir):
    """Return deterministic output bytes; no filesystem writes or model imports."""
    check(config["schema_version"] == 1, "unsupported generation schema")
    requested = config["pairs_per_topic"]
    check(bool(requested) and set(requested) <= set(TOPICS), "invalid pairs_per_topic")
    base = Path(base_dir)
    manifest = EntityManifest(base / config["entity_manifest"], base / config["entity_manifest_metadata"])
    generator = CompoundGenerator(manifest, base / config["source_dir"], config["split"],
                                  config["generation_seed"], config["template_version"])
    # Check and sample every requested topic before loading facts or generating rows.
    pairs = {topic: sample_pairs(manifest, topic, config["split"], requested[topic],
                                 config["pair_sampling_seed"])
             for topic in sorted(requested)}
    rows = [row for topic in sorted(pairs) for a, b in pairs[topic]
            for row in generator.standard_r1_pair(a, b)]
    rows.sort(key=lambda row: row["example_id"])
    validate_r1(rows, manifest, config["split"], requested)
    output = csv_bytes(rows, FIELDS)
    metadata = {
        "schema_version": 1, "generator_version": "clean-r1-v1",
        "entity_manifest_sha256": manifest.digest, "entity_manifest_version": manifest.version,
        "entity_manifest_metadata_sha256": manifest.metadata_digest,
        "requested_split": config["split"], "pair_sampling_seed": config["pair_sampling_seed"],
        "generation_seed": config["generation_seed"], "requested_pairs_per_topic": requested,
        "template_version": config["template_version"], "output_row_count": len(rows),
        "output_sha256": sha256(output), "atomic_source_hashes": generator.source_hashes,
        "pair_sampling_algorithm": "Lowest SHA256 ranks of [pair-sampling-v1, seed, topic, split, sorted IDs]",
        "false_fact_algorithm": "Lowest SHA256 rank of [wrong-object-v1, generation_seed, topic, entity_id, object]",
        "false_object_pool": "Correct objects of usable entities in requested topic/split only",
        "assertions_passed": ["manifest_identity_unique", "manifest_splits_disjoint",
                              "requested_split_only", "usable_entities_only", "unique_pairs",
                              "pair_capacity", "canonical_surface_alignment", "boolean_labels",
                              "sixteen_variants_per_pair", "unique_example_ids", "requested_counts"],
    }
    return {"compounds.csv": output,
            "metadata.json": (json.dumps(metadata, ensure_ascii=False, sort_keys=True, indent=2)
                              + "\n").encode("utf-8")}
