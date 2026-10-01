"""Label-free exposure contracts. No fitting, generation, or scoring entry points."""

import json
from pathlib import Path

from src.clean_compounds import EntityManifest, check, unordered_pair_id
from src.entity_partitions import TOPICS, sha256

ROLES = ("atomic_probe_fitting", "layer_C_selection", "compound_fitting", "calibration", "evaluation")
COMPOUND_ROLES = ROLES[2:]
PROTOCOLS = ("new_pairings", "entity_disjoint", "strict_leave_one_topic_out")
ATOMIC_FIELDS = {"topic", "split", "entity_id"}
PAIR_FIELDS = {"topic", "split", "entity_a_id", "entity_b_id", "pair_id"}


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


class ProtocolCatalog:
    """Resolve declarative regimes against the one shared entity manifest.

    Exposure records contain identities only, never statements, labels, scores,
    or activations. Callers must include every entity/pair that actually influenced
    each role, including historical fitting and tuning; a ledger cannot establish
    that an omitted exposure never happened. Success authorizes no evaluation.
    """

    def __init__(self, config_path, root):
        self.config = json.loads(Path(config_path).read_text())
        self.digest = sha256(encoded(self.config))
        self._validate_config()
        root = Path(root)
        self.manifest = EntityManifest(root / self.config["entity_manifest"],
                                       root / self.config["entity_manifest_metadata"])
        check(self.manifest.digest == self.config["entity_manifest_sha256"], "protocol manifest hash mismatch")
        check(self.manifest.version == self.config["entity_manifest_version"], "protocol manifest version mismatch")

    def _validate_config(self):
        c = self.config
        check(c["schema_version"] == 1 and c["protocol_version"] == "clean-generalization-v1",
              "unsupported protocol version")
        check(c["final_evaluation_enabled"] is False, "final evaluation must remain disabled")
        check(c["pair_id_version"] == "pair-v1" and c["topics"] == list(TOPICS), "identity schema changed")
        check(set(c["protocols"]) == set(PROTOCOLS), "expected exactly three protocols")
        expected = {name: {"protocol": name, "heldout_topic": None} for name in PROTOCOLS[:2]}
        expected.update({"holdout_" + t: {"protocol": PROTOCOLS[2], "heldout_topic": t} for t in TOPICS})
        check(c["regimes"] == expected, "expected two base regimes and five strict topic holdouts")
        for name, policy in c["protocols"].items():
            pair, strict = name == "new_pairings", name == "strict_leave_one_topic_out"
            claim = "pair-disjoint" if pair else "strict-leave-one-topic-out" if strict else "entity-disjoint"
            check(policy["claim"] == claim, "incorrect scientific claim")
            check(policy["evaluation_entity_overlap_with_atomic_fitting"] is pair, "incorrect entity overlap policy")
            check(policy["evaluation_pairs_disjoint_from"] == ["compound_fitting", "calibration"],
                  "evaluation pair protection cannot be weakened")
            check(policy["evaluation_entities_disjoint_from"] == ([] if pair else
                  ["atomic_probe_fitting", "compound_fitting", "calibration"]), "entity protection cannot be weakened")
            check(policy["pair_variant_grouping"] == "one_pair_one_compound_role", "variants must stay together")
            check(set(policy["exposure"]) == set(ROLES), "all five exposure roles are required")
            for role, exposure in policy["exposure"].items():
                unit = "compound_pair" if role in COMPOUND_ROLES else "atomic_entity"
                topics = ("heldout" if role == "evaluation" else "non_heldout") if strict else "all"
                expected_exposure = {"unit": unit, "topics": topics}
                if role == "evaluation":
                    expected_exposure["entity_splits_by_phase"] = {
                        "development": ["train"] if pair else ["validation"],
                        "final": ["train"] if pair else ["test"],
                    }
                else:
                    expected_exposure["entity_splits"] = ["validation"] if role == "layer_C_selection" else ["train"]
                check(exposure == expected_exposure, f"{name}/{role}: forbidden exposure policy")

    def regime(self, name, phase="development"):
        """Describe even future final scopes; this never authorizes data access."""
        check(name in self.config["regimes"], "unknown regime")
        check(phase in ("development", "final"), "unknown evaluation phase")
        regime = self.config["regimes"][name]
        policy = self.config["protocols"][regime["protocol"]]
        holdout = regime["heldout_topic"]
        scopes = {}
        for role, exposure in policy["exposure"].items():
            selector = exposure["topics"]
            topics = list(TOPICS) if selector == "all" else (
                [holdout] if selector == "heldout" else [t for t in TOPICS if t != holdout])
            scopes[role] = {"unit": exposure["unit"], "topics": topics,
                            "entity_splits": exposure["entity_splits_by_phase"][phase]
                            if role == "evaluation" else exposure["entity_splits"]}
        return {"regime": name, **regime, "claim": policy["claim"], "phase": phase,
                "scientific_question": policy["scientific_question"], "exposure": scopes,
                "evaluation_pairs_disjoint_from": policy["evaluation_pairs_disjoint_from"],
                "evaluation_entities_disjoint_from": policy["evaluation_entities_disjoint_from"],
                "pair_variant_grouping": policy["pair_variant_grouping"],
                "evaluation_enabled": phase == "development"}

    def validate_exposures(self, name, exposures, *, phase="development"):
        """Check a complete metadata ledger; final ledgers remain locked.

        Atomic records: topic, split, entity_id. Compound records: topic, split,
        entity_a_id, entity_b_id, pair_id, optionally example_id. Multiple variants
        of one pair may occur in the same role, never across compound roles.
        """
        regime = self.regime(name, phase)
        check(regime["evaluation_enabled"], "final evaluation exposure is disabled until the suite is locked")
        check(set(exposures) == set(ROLES), "complete ledger must declare all five roles, including empty roles")
        entities, pairs, owners, examples = {}, {}, {}, {}
        for role in ROLES:
            scope, records = regime["exposure"][role], exposures[role]
            check(isinstance(records, list), "each exposure role must be a list")
            entities[role], pairs[role] = set(), set()
            if role in ("atomic_probe_fitting", "layer_C_selection", "evaluation"):
                check(bool(records), f"{role}: required exposure cannot be empty")
            for row in records:
                fields = PAIR_FIELDS if role in COMPOUND_ROLES else ATOMIC_FIELDS
                optional = {"example_id"} if role in COMPOUND_ROLES else set()
                check(isinstance(row, dict) and fields <= set(row) <= fields | optional,
                      "identity metadata only: missing fields or unexpected labels/statements/scores")
                check(all(isinstance(v, str) and v for v in row.values()), "identity metadata must contain strings")
                check(row["topic"] in scope["topics"], f"{role}: forbidden topic")
                check(row["split"] in scope["entity_splits"], f"{role}: forbidden entity split")
                ids = [row["entity_a_id"], row["entity_b_id"]] if role in COMPOUND_ROLES else [row["entity_id"]]
                for identity in ids:
                    check(identity in self.manifest.entities, "entity absent from shared manifest")
                    entity = self.manifest.entities[identity]
                    check((entity.topic, entity.split) == (row["topic"], row["split"]),
                          "entity topic/split disagrees with shared manifest")
                    entities[role].add(identity)
                if role in COMPOUND_ROLES:
                    pid = unordered_pair_id(self.manifest, *ids, row["split"])
                    check(row["pair_id"] == pid, "pair_id disagrees with shared unordered identity")
                    check(pid not in owners or owners[pid] == role,
                          "pair leakage: all variants of a pair must remain in one compound role")
                    owners[pid] = role
                    pairs[role].add(pid)
                    if "example_id" in row:
                        eid = row["example_id"]
                        check(eid not in examples, "duplicate example_id in exposure ledger")
                        examples[eid] = role
        for role in regime["evaluation_pairs_disjoint_from"]:
            check(pairs["evaluation"].isdisjoint(pairs[role]), f"evaluation pair leakage from {role}")
        for role in regime["evaluation_entities_disjoint_from"]:
            check(entities["evaluation"].isdisjoint(entities[role]), f"evaluation entity leakage from {role}")
        # A row-order-independent digest binds the checked identity ledger.
        normalized = {role: sorted(exposures[role], key=encoded) for role in ROLES}
        return {"regime": name, "phase": phase, "claim": regime["claim"], "checks_passed": True,
                "protocol_sha256": self.digest, "protocol_version": self.config["protocol_version"],
                "manifest_sha256": self.manifest.digest, "manifest_version": self.manifest.version,
                "exposure_sha256": sha256(encoded(normalized)),
                "counts": {role: {"records": len(exposures[role]), "entities": len(entities[role]),
                                  "pairs": len(pairs[role])} for role in ROLES},
                "evaluation_entity_overlap_with_atomic_fitting": len(
                    entities["evaluation"] & entities["atomic_probe_fitting"]),
                "authorizes_generation_or_scoring": False, "test_metrics_computed": False}

    def summary(self):
        return {"protocol_version": self.config["protocol_version"], "protocol_sha256": self.digest,
                "manifest_sha256": self.manifest.digest, "manifest_version": self.manifest.version,
                "final_evaluation_enabled": False,
                "regimes": {name: {phase: self.regime(name, phase) for phase in ("development", "final")}
                            for name in self.config["regimes"]}}
