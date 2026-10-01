"""Deterministic, source-audited negative queues and score-free review registries."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from types import MappingProxyType

from src.clean_compounds import ATOMIC_TEMPLATES, EntityManifest, check, stable_id
from src.entity_partitions import TOPICS, canonical_json, csv_bytes, save_outputs, sha256
from src.negative_audit import proposition_key

RANKING_VERSION = "negative-candidate-v1"
REGISTRY_VERSION = "validated-negative-registry-v1"
STATUSES = {"source_supported_false", "externally_validated_false", "rejected_true_or_ambiguous", "unverified"}
ACCEPTED = {"source_supported_false", "externally_validated_false"}
REVIEW_FIELDS = ("validation_status", "evidence_source", "evidence_note", "reviewer_or_method", "validation_version")
FIELDS = ("topic", "split", "entity_id", "entity", "candidate_object", "candidate_rank", "ranking_hash",
          "fact_id", "statement", "proposition_key", "source_classification", *REVIEW_FIELDS)
AUDIT_FILES = ("metadata.json", "entity_knowledge.csv", "candidate_audit.csv")


def development_only(split):
    check(split in ("train", "validation"), "test candidates are locked: only train/validation workflows are enabled")


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def object_rank(seed, topic, identity, value):
    check(type(seed) is int, "negative candidate seed must be an integer")
    return sha256(canonical_json([RANKING_VERSION, seed, topic, identity, value]))


def split_rows(path, split):
    """Skip other splits before interpreting any entity knowledge/candidate fields.

    The frozen audit uses mixed-split CSVs. CSV decoding and integrity hashing
    touch their bytes; no test propositions are indexed, classified or returned.
    """
    development_only(split)
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["split"] == split:
                yield row


def initialize(config, root, split):
    development_only(split)  # Must precede all filesystem reads.
    from src.inventor_country_semantics import SEMANTICS_VERSION
    semantic = config.get("inventor_country_semantics")
    check(semantic in (None, SEMANTICS_VERSION), "unsupported inventor semantics")
    version = "validated-negative-registry-v2" if semantic else REGISTRY_VERSION
    check(config["schema_version"] == 1 and config["ranking_version"] == RANKING_VERSION and
          config["registry_version"] == version, "unsupported negative configuration")
    check(type(config["negative_candidate_seed"]) is int, "negative candidate seed must be an integer")
    root = Path(root)
    manifest = EntityManifest(root / config["entity_manifest"], root / config["entity_manifest_metadata"])
    audit = root / config["source_audit_dir"]
    hashes = {name: sha256((audit / name).read_bytes()) for name in AUDIT_FILES}
    check(hashes == config["source_audit_files"], "source audit hash mismatch")
    metadata = json.loads((audit / "metadata.json").read_text())
    check(metadata["entity_manifest_sha256"] == manifest.digest and
          metadata["entity_manifest_version"] == manifest.version, "audit/manifest mismatch")
    check(metadata["audit_complete_without_parse_or_provenance_gaps"], "source audit has unresolved parse/provenance gaps")
    eligible = {e.entity_id: e for e in manifest.entities.values() if e.split == split and e.compound_usable}
    knowledge = {}
    for row in split_rows(audit / "entity_knowledge.csv", split):
        identity = row["entity_id"]
        if identity not in eligible:
            continue
        entity = eligible[identity]
        check((row["topic"], row["entity"]) == (entity.topic, entity.entity), "audit entity identity mismatch")
        check(identity not in knowledge, "duplicate entity knowledge")
        knowledge[identity] = (set(json.loads(row["known_true_objects"])), set(json.loads(row["known_false_objects"])))
    check(set(knowledge) == set(eligible), "incomplete source knowledge for usable entities")
    candidates = {}
    for row in split_rows(audit / "candidate_audit.csv", split):
        identity = row["entity_id"]
        check(identity in eligible, "audit candidate is not an eligible entity")
        entity = eligible[identity]
        check((row["topic"], row["entity"]) == (entity.topic, entity.entity), "audit candidate identity mismatch")
        value = row["candidate_object"]
        true, false = knowledge[identity]
        expected = "invalid_known_true" if value in true else "supported_false" if value in false else "unverified_negative"
        check(row["classification"] == expected, "audit classification contradicts source knowledge")
        check(row["proposition_key"] == proposition_key(entity.topic, entity.entity, value), "audit proposition mismatch")
        key = (identity, value)
        check(key not in candidates, "duplicate audited candidate")
        candidates[key] = row
    pools = {topic: set().union(*(knowledge[e.entity_id][0] for e in eligible.values() if e.topic == topic))
             for topic in TOPICS}
    rows = []
    for identity, entity in sorted(eligible.items()):
        true, _ = knowledge[identity]
        values = pools[entity.topic] - true  # Exclude ALL recorded true objects, including contradictions.
        ranked = sorted(values, key=lambda value: (object_rank(config["negative_candidate_seed"], entity.topic, identity, value), value))
        for rank, value in enumerate(ranked, 1):
            check((identity, value) in candidates, "source audit lacks a candidate relationship")
            source = candidates[(identity, value)]
            supported = source["classification"] == "supported_false"
            refs = json.loads(source["source_proposition_rows"])
            check(not supported or bool(refs), "supported false lacks exact source references")
            statement = ATOMIC_TEMPLATES[entity.topic].format(entity=entity.entity, object=value)
            rows.append(dict(zip(FIELDS, (
                entity.topic, split, identity, entity.entity, value, rank,
                object_rank(config["negative_candidate_seed"], entity.topic, identity, value),
                stable_id("fact", [entity.topic, identity, statement, False]), statement,
                proposition_key(entity.topic, entity.entity, value), source["classification"],
                "source_supported_false" if supported else "unverified",
                json.dumps(sorted(refs)) if supported else "",
                "Exact proposition explicitly labeled false in frozen source audit." if supported else "No established false evidence.",
                "source-negative-audit-v1" if supported else "not_reviewed", config["validation_version"],
            ))))
    if semantic:
        from src.inventor_country_audit import inventor_queue_rows, inventor_source_records
        source_records = inventor_source_records(manifest, root / config["inventor_source_file"])
        rows = [r for r in rows if r["topic"] != "inventors"] + inventor_queue_rows(config, eligible, knowledge, candidates, source_records)
    rows.sort(key=lambda r: (r["topic"], r["entity_id"], r["candidate_rank"]))
    provenance = {
        "registry_version": version, "split": split,
        "negative_candidate_seed": config["negative_candidate_seed"], "ranking_version": RANKING_VERSION,
        "ranking_algorithm": "ascending SHA256(compact UTF-8 JSON [negative-candidate-v1, seed, topic, entity_id, exact object]); exact object tie-break; 1-based rank",
        "candidate_pool": "all recorded true objects of usable entities in same topic/shared split, minus all entity known-true objects",
        "source_audit_files": hashes,
        "source_audit_sha256": sha256(canonical_json([[name, hashes[name]] for name in AUDIT_FILES])),
        "entity_manifest_sha256": manifest.digest, "entity_manifest_version": manifest.version,
        "entity_manifest_metadata_sha256": manifest.metadata_digest,
        "validation_version": config["validation_version"], "compound_usable_entities": len(eligible),
        "test_candidates_processed": 0, "scores_used": False,
    }
    if semantic:
        provenance["inventor_country_semantics"] = semantic
        provenance["inventor_candidate_pool"] = "single country components of same-split usable true objects; exclude all known-true components"
        provenance["inventor_source_sha256"] = sha256((root / config["inventor_source_file"]).read_bytes())
    return manifest, rows, provenance


class ValidatedNegativeRegistry:
    def __init__(self, manifest, rows, expected_rows, provenance):
        development_only(provenance["split"])
        expected = {r["fact_id"]: r for r in expected_rows}
        check(len(expected) == len(expected_rows), "duplicate candidate proposition")
        seen, validated = set(), []
        for raw in rows:
            check(set(raw) == set(FIELDS), "registry accepts exact factual fields only; score/model columns are forbidden")
            row = dict(raw)
            row["candidate_rank"] = int(row["candidate_rank"])
            fid = row["fact_id"]
            check(fid in expected and fid not in seen, "unknown or duplicate registry proposition")
            seen.add(fid)
            baseline = expected[fid]
            check(all(row[k] == baseline[k] for k in FIELDS if k not in REVIEW_FIELDS),
                  "candidate identity, ranking, statement or source classification was modified")
            self._validate_review(row, baseline)
            validated.append(row)
        check(seen == set(expected), "registry must retain the complete deterministic queue")
        validated.sort(key=lambda r: (r["topic"], r["entity_id"], r["candidate_rank"]))
        self.manifest, self.split = manifest, provenance["split"]
        self._expected_rows = tuple(MappingProxyType(dict(row)) for row in expected_rows)
        self.rows = tuple(MappingProxyType(row) for row in validated)
        self.provenance = dict(provenance)
        self.digest = sha256(csv_bytes(self.rows, FIELDS))
        self._by_entity = defaultdict(list)
        for row in self.rows:
            self._by_entity[row["entity_id"]].append(row)

    @staticmethod
    def _validate_review(row, baseline):
        check(all(isinstance(row[k], str) for k in REVIEW_FIELDS), "review fields must be text")
        status = row["validation_status"]
        check(status in STATUSES, "unsupported validation status")
        check(bool(row["validation_version"].strip()), "validation version is required")
        if status == "source_supported_false":
            check(baseline["source_classification"] == "supported_false", "unverified source cannot be promoted to source-supported")
            check(all(row[k] == baseline[k] for k in ("evidence_source", "evidence_note", "reviewer_or_method")),
                  "source-supported evidence must match audit")
        if status in ("externally_validated_false", "rejected_true_or_ambiguous"):
            check(all(row[k].strip() for k in ("evidence_source", "evidence_note", "reviewer_or_method")),
                  "external/rejected decisions require evidence and reviewer/method")

    def selected(self, identity):
        check(identity in self.manifest.entities, "unknown registry entity")
        entity = self.manifest.entities[identity]
        development_only(entity.split)
        check(entity.split == self.split, "registry/entity split mismatch")
        return next((row for row in self._by_entity.get(identity, ()) if row["validation_status"] in ACCEPTED), None)

    def availability(self, topic):
        return {e.entity_id: self.selected(e.entity_id) is not None for e in self.manifest.eligible(topic, self.split)}

    def files(self, *, parent_sha256=None):
        payload = csv_bytes(self.rows, FIELDS)
        metadata = {**self.provenance, "registry_sha256": sha256(payload), "row_count": len(self.rows),
                    "parent_registry_sha256": parent_sha256, "summary": self.summary()}
        return {"registry.csv": payload, "metadata.json": json_bytes(metadata)}

    def summary(self):
        summary = []
        for topic in TOPICS:
            availability = self.availability(topic)
            counts = Counter(row["validation_status"] for row in self.rows if row["topic"] == topic)
            summary.append({"topic": topic, "split": self.split, "entities": len(availability),
                            "candidates": sum(counts.values()), **{s: counts[s] for s in sorted(STATUSES)},
                            "available_entities": sum(availability.values()),
                            "unavailable_entities": sum(not v for v in availability.values())})
        return summary


def build_registry(config, root, split):
    manifest, rows, provenance = initialize(config, root, split)
    return ValidatedNegativeRegistry(manifest, rows, rows, provenance)


def load_registry(config, root, split, directory):
    development_only(split)  # Reject test before opening config/audit/registry facts.
    manifest, expected, provenance = initialize(config, root, split)
    directory = Path(directory)
    metadata = json.loads((directory / "metadata.json").read_text())
    check(all(metadata.get(k) == v for k, v in provenance.items()), "registry provenance mismatch")
    payload = (directory / "registry.csv").read_bytes()
    check(sha256(payload) == metadata["registry_sha256"], "registry content hash mismatch")
    with (directory / "registry.csv").open(newline="", encoding="utf-8") as stream:
        registry = ValidatedNegativeRegistry(manifest, list(csv.DictReader(stream)), expected, provenance)
    check(len(registry.rows) == metadata["row_count"], "registry row count mismatch")
    return registry


def apply_reviews(registry, updates, validation_version):
    """Explicit evidence-only decisions. No scoring, web access or automatic promotion."""
    development_only(registry.split)
    check(isinstance(updates, list), "reviews must be a list")
    rows = {r["fact_id"]: dict(r) for r in registry.rows}
    seen = set()
    fields = {"fact_id", *REVIEW_FIELDS}
    for update in updates:
        check(isinstance(update, dict) and set(update) == fields,
              "review accepts exact evidence fields only; score/model columns are forbidden")
        check(update["validation_version"] == validation_version, "review validation version mismatch")
        fid = update["fact_id"]
        check(fid in rows and fid not in seen, "unknown or duplicate review fact_id")
        seen.add(fid)
        rows[fid].update(update)
    return ValidatedNegativeRegistry(registry.manifest, list(rows.values()), registry._expected_rows, registry.provenance)


def export_candidates(registry, split):
    development_only(split)
    check(split == registry.split, "export split mismatch")
    return csv_bytes(registry.rows, FIELDS)
