"""Entity-disjoint DEVELOPMENT / VALIDATION compounds; no final-test entry point."""

import csv
import io
import json
from collections import Counter
from pathlib import Path

from src.clean_compounds import (
    FIELDS, CompoundGenerator, check, stable_id, unordered_pair_id, validate_r1,
)
from src.entity_partitions import TOPICS, canonical_json, csv_bytes, sha256
from src.generalization_protocols import ProtocolCatalog
from src.validated_negatives import ACCEPTED, json_bytes, load_registry

PAIR_ALGORITHM = "validation-degree4-ring-v1"
LABEL = "DEVELOPMENT / VALIDATION - NOT FINAL TEST"
EXPOSURE = {"protocol": "entity_disjoint", "evaluation_phase": "development",
            "benchmark_label": LABEL}
OUTPUT_FIELDS = (*FIELDS, *EXPOSURE)
BOOL_FIELDS = ("canonical_truth_a", "canonical_truth_b", "surface_first_truth",
               "surface_second_truth", "compound_label")


def degree_four_pairs(manifest, topic, split, count, seed):
    """A seed-permuted ring with offsets 1 and 2 is simple and 4-regular for n>=5.

    This version intentionally supports exactly 2*n pairs, rather than silently
    approximating an incompatible requested degree sequence.
    """
    check(split == "validation", "only development validation pairing is allowed")
    check(type(seed) is int, "pair seed must be an integer")
    check(type(count) is int and count >= 0, "invalid pair count")
    ids = [e.entity_id for e in manifest.eligible(topic, split)]
    n = len(ids)
    capacity = n * (n - 1) // 2
    check(count <= capacity, f"{topic}: requested {count} pairs exceeds capacity {capacity}")
    check(n >= 5 and count == 2 * n, "degree-four algorithm requires n>=5 and exactly 2*n pairs")
    ordered = sorted(ids, key=lambda eid: (
        sha256(canonical_json([PAIR_ALGORITHM, seed, topic, split, eid])), eid))
    pairs = sorted({tuple(sorted((ordered[i], ordered[(i + offset) % n])))
                    for i in range(n) for offset in (1, 2)})
    check(len(pairs) == count, "duplicate pair or wrong pair count")
    degrees = Counter(eid for pair in pairs for eid in pair)
    check(set(degrees) == set(ids) and set(degrees.values()) == {4}, "incomplete or unbalanced coverage")
    for a, b in pairs:
        manifest.authorize_pair(a, b, split)
    return pairs, degrees


def build_validation_benchmark(config, root):
    # These guards precede all manifest, registry, or factual file access.
    check(config["split"] == "validation" and config["evaluation_phase"] == "development"
          and config["protocol"] == "entity_disjoint", "only entity-disjoint DEVELOPMENT / VALIDATION is allowed")
    check(config["schema_version"] == 1 and config["pair_sampling_algorithm"] == PAIR_ALGORITHM,
          "unsupported benchmark configuration")
    check(config["target_average_degree"] == 4, "target degree must remain four")
    requested = config["pairs_per_topic"]
    check(set(requested) == set(TOPICS), "all five validation topics are required")
    root = Path(root)
    catalog = ProtocolCatalog(root / config["protocol_config"], root)
    manifest = catalog.manifest
    scope = catalog.regime("entity_disjoint", "development")["exposure"]["evaluation"]
    check(scope["entity_splits"] == ["validation"] and set(scope["topics"]) == set(TOPICS),
          "protocol evaluation scope mismatch")
    # Capacity and graph construction precede any factual/registry loading.
    sampled = {topic: degree_four_pairs(manifest, topic, "validation", requested[topic],
                                       config["pair_sampling_seed"]) for topic in sorted(TOPICS)}
    negative_config = json.loads((root / config["negative_config"]).read_text())
    registry = load_registry(negative_config, root, "validation", root / config["registry_dir"])
    check(registry.digest == config["registry_sha256"], "reviewed registry hash mismatch")
    check(registry.manifest.digest == manifest.digest, "registry/benchmark manifest mismatch")
    generator = CompoundGenerator(manifest, root / config["source_dir"], "validation",
                                  config["generation_seed"], config["template_version"], negative_registry=registry)
    check(not generator.unavailable_entities, "validation entity without accepted negative; abort rather than drop it")
    selected, pairs, degree_rows, summaries = {}, [], [], []
    for topic, (topic_pairs, degrees) in sampled.items():
        for entity in manifest.eligible(topic, "validation"):
            negative = registry.selected(entity.entity_id)
            accepted = [r for r in registry.rows if r["entity_id"] == entity.entity_id
                        and r["validation_status"] in ACCEPTED]
            check(negative is not None and negative["validation_status"] in ACCEPTED,
                  "missing accepted negative")
            check(negative["candidate_rank"] == min(r["candidate_rank"] for r in accepted),
                  "negative is not lowest-ranked accepted candidate")
            selected[entity.entity_id] = negative
            degree_rows.append({**EXPOSURE, "topic": topic, "split": "validation",
                                "entity_id": entity.entity_id, "degree": degrees[entity.entity_id]})
        for a, b in topic_pairs:
            pid = unordered_pair_id(manifest, a, b, "validation")
            check(pid == unordered_pair_id(manifest, b, a, "validation"), "pair ID depends on surface order")
            pairs.append({**EXPOSURE, "topic": topic, "split": "validation", "entity_a_id": a,
                          "entity_b_id": b, "pair_id": pid})
        summaries.append({"topic": topic, "n_entities": len(degrees), "n_pairs": len(topic_pairs),
                          "n_rows": 16 * len(topic_pairs), "pair_capacity": len(degrees)*(len(degrees)-1)//2,
                          "min_entity_degree": min(degrees.values()), "max_entity_degree": max(degrees.values()),
                          "mean_entity_degree": sum(degrees.values()) / len(degrees),
                          "degree_histogram": dict(sorted(Counter(degrees.values()).items()))})
    check(len({p["pair_id"] for p in pairs}) == len(pairs), "duplicate unordered pair IDs")
    rows = [{**row, **EXPOSURE} for pair in pairs
            for row in generator.standard_r1_pair(pair["entity_a_id"], pair["entity_b_id"])]
    rows.sort(key=lambda r: r["example_id"])
    validate_r1(rows, manifest, "validation", requested)
    seen, false_ids = set(), set()
    false_occurrences = 0
    for row in rows:
        for side in ("a", "b"):
            eid, truth = row[f"entity_{side}_id"], row[f"canonical_truth_{side}"]
            seen.add(eid)
            check(manifest.entities[eid].split == "validation", "train/test entity in output")
            statement = row[f"fact_{side}_statement"]
            check(row[f"fact_{side}_id"] == stable_id("fact", [row["topic"], eid, statement, truth]),
                  "fact identity mismatch")
            if not truth:
                negative = selected[eid]
                check(negative["validation_status"] in ACCEPTED and
                      (row[f"fact_{side}_id"], statement) == (negative["fact_id"], negative["statement"]),
                      "unaccepted or nonselected false constituent")
                false_ids.add(negative["fact_id"])
                false_occurrences += 1
    check(seen == set(selected), "not all usable validation entities appear")
    check(len(false_ids) == len(selected), "not all selected false facts appear")
    output = csv_bytes(rows, OUTPUT_FIELDS)
    # Validate the actual serialized table as well as the in-memory examples.
    decoded = list(csv.DictReader(io.StringIO(output.decode("utf-8"))))
    for row in decoded:
        for key in BOOL_FIELDS:
            check(row[key] in ("True", "False"), "invalid serialized Boolean")
            row[key] = row[key] == "True"
        row["generation_seed"] = int(row["generation_seed"])
    check(decoded == rows, "CSV round-trip mismatch")
    validate_r1(decoded, manifest, "validation", requested)
    selected_rows = [{**EXPOSURE, **{k: r[k] for k in
                     ("topic", "split", "entity_id", "fact_id", "candidate_rank", "validation_status")}}
                     for _, r in sorted(selected.items())]
    files = {"development_validation_compounds.csv": output,
             "development_validation_pairs.csv": csv_bytes(pairs, tuple(pairs[0])),
             "development_validation_entity_degrees.csv": csv_bytes(degree_rows, tuple(degree_rows[0])),
             "development_validation_selected_negative_facts.csv": csv_bytes(selected_rows, tuple(selected_rows[0]))}
    code_paths = ("src/validation_compound_benchmark.py", "src/clean_compounds.py", "src/validated_negatives.py",
                  "src/entity_partitions.py", "src/generalization_protocols.py", "src/inventor_country_audit.py",
                  "src/inventor_country_semantics.py", "scripts/27_generate_validation_compound_benchmark.py")
    repo = Path(__file__).resolve().parents[1]
    metadata = {**EXPOSURE, "schema_version": 1, "requested_split": "validation", "final_test": False,
                "entity_manifest_sha256": manifest.digest, "entity_manifest_version": manifest.version,
                "entity_manifest_metadata_sha256": manifest.metadata_digest,
                "reviewed_negative_registry_sha256": registry.digest,
                "reviewed_negative_registry_version": registry.provenance["registry_version"],
                "reviewed_negative_registry_provenance": registry.provenance,
                "pair_seed": config["pair_sampling_seed"], "pair_sampling_algorithm_version": PAIR_ALGORITHM,
                "pair_sampling_algorithm": "SHA256 order of compact UTF-8 JSON [version, seed, topic, validation, entity_id]; circular offsets 1 and 2; canonical unordered endpoints",
                "pair_id_version": "pair-v1", "pair_ids": [p["pair_id"] for p in pairs],
                "template_version": config["template_version"], "exact_pairs_per_topic": requested,
                "output_row_count": len(rows), "output_pair_count": len(pairs), "entity_count": len(seen),
                "degree_summaries": summaries, "false_constituent_occurrences": false_occurrences,
                "unique_false_fact_count": len(false_ids),
                "accepted_negative_status_counts": dict(sorted(Counter(r["validation_status"] for r in selected.values()).items())),
                "schema": list(OUTPUT_FIELDS), "config": config, "config_sha256": sha256(canonical_json(config)),
                "protocol_sha256": catalog.digest, "protocol_version": catalog.config["protocol_version"],
                "atomic_source_hashes": generator.source_hashes,
                "code_sha256": {p: sha256((repo / p).read_bytes()) for p in code_paths},
                "file_sha256": {name: sha256(payload) for name, payload in files.items()},
                "train_entities_in_output": 0, "test_entities_in_output": 0,
                "test_candidate_queues_accessed": 0, "test_examples_generated": 0,
                "activations_computed": False, "model_or_probe_scores_computed": False,
                "assertions_passed": ["manifest_identity_and_split_disjointness", "development_entity_disjoint_scope",
                    "pair_capacity_before_factual_access", "same_topic_validation_usable_only", "no_self_or_duplicate_pairs",
                    "exact_degree_four_all_entities", "pair_id_surface_invariance", "lowest_ranked_accepted_negatives",
                    "all_emitted_false_facts_accepted", "all_entities_and_selected_false_facts_used", "fact_ids_verified",
                    "canonical_surface_truth_alignment", "centralized_boolean_labels", "all_sixteen_variants_per_pair",
                    "unique_example_ids", "exact_pair_and_row_counts", "serialized_csv_round_trip"]}
    files["metadata.json"] = json_bytes(metadata)
    return files
