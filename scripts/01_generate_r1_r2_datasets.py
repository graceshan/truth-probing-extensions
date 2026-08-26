"""Generate the R1 and R2 templating datasets (pure templating, no model/GPU).

Multi-topic version: entity pairs are drawn from all five union-probe training
topics (cities, sp_en_trans, inventors, element_symb, animal_class), balanced so
no topic dominates. Each topic keeps its own natural statement template, taken
from its atomic CSV in data/tiu_datasets/ -- the template functions below are
asserted to reproduce every atomic statement exactly before anything is built,
so synthesized true/false statements match the source style byte-for-byte
(including quirks like animal_class always using "a", e.g. "a amphibian").

Produces four CSVs in data/:
  1. r1_quadruples.csv      -- entity-pair quadruples: 4 cells x {and,or} x {AB,BA}
  2. r1_leakage_control.csv -- same pairs/cells joined with the single-token filler "blah"
  3. r1_dilution.csv        -- n-way conjunctions, n in {2,3,4,6}, dilution ladder
  4. r2_setA_xor.csv        -- "Either A or B, but not both." XOR set (600 TT, 300 else)

False statements are ALWAYS wrong-object (entity paired with a wrong
country/class/translation/symbol/inhabited-country) -- never negations. This is
enforced in every dataset, including the dilution ladder, because the probe
behaves differently on negated statements (a confound). Set B is intentionally
NOT generated (spec'd but not budgeted).

Connective carriers "and"/"or" AND the filler are asserted single-token under
the Qwen2.5-7B tokenizer, so both conjuncts sit at identical token positions.
This loads the TOKENIZER only, never model weights.
"""

import json

import numpy as np
import pandas as pd

SEED = 0
DATA = "data/tiu_datasets"
OUT = "data"
QWEN = "Qwen/Qwen2.5-7B-Instruct"

FILLER = "blah"  # single-token nonsense filler for the leakage control
CELLS = ["TT", "TF", "FT", "FF"]
CELL_LABELS = {"TT": (1, 1), "TF": (1, 0), "FT": (0, 1), "FF": (0, 0)}
TOPIC_ORDER = ["cities", "sp_en_trans", "inventors", "element_symb", "animal_class"]

K_PAIRS_PER_TOPIC = 100
R2_PER_TOPIC = {"TT": 120, "TF": 60, "FT": 60, "FF": 60}  # -> 600 TT, 300 each else
DIL_PER_TOPIC_CELL = 20  # per (topic, n, condition) -> 100 per (n, condition)

# Each topic: (regex with two groups (entity, object), template_fn(entity, object)).
# Object handling matches the source: animal_class always uses "a".
TOPICS = {
    "cities":       (r"The city of (.+?) is in (.+?)\.",
                     lambda e, o: f"The city of {e} is in {o}."),
    "sp_en_trans":  (r"The Spanish word '(.+?)' means '(.+?)'\.",
                     lambda e, o: f"The Spanish word '{e}' means '{o}'."),
    "inventors":    (r"^(.+?) lived in (.+?)\.$",
                     lambda e, o: f"{e} lived in {o}."),
    "element_symb": (r"^(.+?) has the symbol (.+?)\.$",
                     lambda e, o: f"{e} has the symbol {o}."),
    "animal_class": (r"^The (.+?) is an? (.+?)\.$",
                     lambda e, o: f"The {e} is a {o}."),
}


# ---------------------------------------------------------------- join style
# Matches Bürger's compound style: lowercase a leading "The" (cities/animal/
# spanish), keep proper nouns (inventors/elements), strip trailing periods when
# joining, one period closes and/or/filler compounds.

def strip_period(s: str) -> str:
    return s.rstrip().rstrip(".")


def lower_if_article(s: str) -> str:
    """Lowercase a leading 'The ' only; leave proper-noun-initial statements alone."""
    return "the" + s[3:] if s.startswith("The ") else s


def join_binary(first: str, second: str, connective: str) -> str:
    """first is sentence-initial (keeps case); second is mid-sentence (lower a leading 'The')."""
    return f"{strip_period(first)} {connective} {lower_if_article(strip_period(second))}."


def join_xor(first: str, second: str) -> str:
    """Both conjuncts are mid-sentence (after 'Either')."""
    a = lower_if_article(strip_period(first))
    b = lower_if_article(strip_period(second))
    return f"Either {a} or {b}, but not both."


def join_conjunction(stmts: list[str]) -> str:
    out = strip_period(stmts[0])
    for s in stmts[1:]:
        out += f" and {lower_if_article(strip_period(s))}"
    return out + "."


# ---------------------------------------------------------------- sources

def build_topic(topic: str) -> dict:
    """Parse a topic's atomic CSV; assert the template reproduces every statement."""
    pat, tmpl = TOPICS[topic]
    d = pd.read_csv(f"{DATA}/{topic}.csv")
    m = d.statement.str.extract(pat)
    ent, obj = m[0], m[1]
    assert ent.notna().all(), f"{topic}: {ent.isna().sum()} statements failed to parse"
    for e, o, s in zip(ent, obj, d.statement):
        assert tmpl(e, o) == s, f"{topic}: template mismatch\n  got {tmpl(e,o)!r}\n  exp {s!r}"
    correct = {}  # entity -> correct object (from the true rows)
    for e, o in zip(ent[d.label == 1], obj[d.label == 1]):
        correct.setdefault(e, o)
    pool = sorted(set(correct.values()))  # real objects, used as wrong-object candidates
    return {"template": tmpl, "correct": correct, "pool": pool,
            "entities": list(correct.keys())}


def build_atoms(topics: dict, rng) -> dict:
    """Per entity: a true statement and a fixed wrong-object false statement (no negations)."""
    atoms = {}
    for t, td in topics.items():
        tmpl, correct, pool = td["template"], td["correct"], td["pool"]
        pool_arr = np.array(pool, dtype=object)
        for e in td["entities"]:
            c = correct[e]
            choices = pool_arr[pool_arr != c]
            wrong = str(rng.choice(choices))
            atoms[(t, e)] = {"true": tmpl(e, c), "false": tmpl(e, wrong)}
    return atoms


def sample_pairs(entities: list, k: int, rng) -> list[tuple]:
    """k unique unordered pairs of distinct entities (entity reuse across pairs allowed)."""
    seen, pairs = set(), []
    n = len(entities)
    while len(pairs) < k:
        i, j = rng.choice(n, size=2, replace=False)
        key = frozenset((int(i), int(j)))
        if key in seen:
            continue
        seen.add(key)
        pairs.append((entities[int(i)], entities[int(j)]))
    return pairs


def all_pairs(topics: dict, rng) -> list[dict]:
    """K balanced pairs per topic -> [{topic, A, B}], topics interleaved."""
    out = []
    for t in TOPIC_ORDER:
        for a, b in sample_pairs(topics[t]["entities"], K_PAIRS_PER_TOPIC, rng):
            out.append({"topic": t, "A": a, "B": b})
    return out


# ---------------------------------------------------------------- generators

def _row(topic, a_stmt, b_stmt, la, lb, cell, connective, ordering, statement):
    return {"statement": statement, "topic": topic,
            "conjunctA": a_stmt, "conjunctB": b_stmt, "labelA": la, "labelB": lb,
            "cell": cell, "connective": connective, "ordering": ordering}


def gen_binary(pairs, atoms, connectives):
    rows = []
    for p in pairs:
        t, ca, cb = p["topic"], p["A"], p["B"]
        for cell in CELLS:
            la, lb = CELL_LABELS[cell]
            a = atoms[(t, ca)]["true" if la else "false"]
            b = atoms[(t, cb)]["true" if lb else "false"]
            for conn in connectives:
                for ordering in ("AB", "BA"):
                    first, second = (a, b) if ordering == "AB" else (b, a)
                    rows.append(_row(t, a, b, la, lb, cell, conn, ordering,
                                     join_binary(first, second, conn)))
    return pd.DataFrame(rows)


def gen_xor(pairs, atoms):
    by_topic = {t: [p for p in pairs if p["topic"] == t] for t in TOPIC_ORDER}
    rows = []
    for t in TOPIC_ORDER:
        combos = [(p, o) for p in by_topic[t] for o in ("AB", "BA")]
        for cell in CELLS:
            la, lb = CELL_LABELS[cell]
            for p, ordering in combos[: R2_PER_TOPIC[cell]]:
                a = atoms[(t, p["A"])]["true" if la else "false"]
                b = atoms[(t, p["B"])]["true" if lb else "false"]
                first, second = (a, b) if ordering == "AB" else (b, a)
                r = _row(t, a, b, la, lb, cell, "xor", ordering, join_xor(first, second))
                rows.append(r)
    return pd.DataFrame(rows)


def gen_dilution(topics, atoms, rng):
    rows = []
    for t in TOPIC_ORDER:
        ents = topics[t]["entities"]
        for n in (2, 3, 4, 6):
            specs = [("dilute_1false", n - 1, 1), ("all_true", n, 0), ("all_false", 0, n)]
            for cond, n_t, n_f in specs:
                for _ in range(DIL_PER_TOPIC_CELL):
                    idx = rng.choice(len(ents), size=n, replace=False)
                    chosen = [ents[int(i)] for i in idx]
                    labels = [1] * n_t + [0] * n_f
                    stmts = [atoms[(t, e)]["true" if lab else "false"]
                             for e, lab in zip(chosen, labels)]
                    order = rng.permutation(n)  # don't leave the false conjunct last
                    stmts = [stmts[i] for i in order]
                    labels = [labels[i] for i in order]
                    rows.append({
                        "statement": join_conjunction(stmts), "topic": t,
                        "n": n, "condition": cond, "n_true": n_t, "n_false": n_f,
                        "compound_label": int(all(x == 1 for x in labels)),
                        "conjuncts": json.dumps(stmts), "labels": json.dumps(labels),
                    })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tokenizer gate

def assert_single_token_carriers():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(QWEN)
    print("[tokenizer] Qwen2.5-7B single-token check (leading-space form):")
    for w in ("and", "or", FILLER):
        ids = tok.encode(" " + w, add_special_tokens=False)
        assert len(ids) == 1, f"LOAD-BEARING FAIL: ' {w}' -> {len(ids)} tokens {ids}"
        print(f"   {' '+w!r:9} -> 1 token ids={ids}  {'connective' if w in ('and','or') else 'filler'}")


# ---------------------------------------------------------------- report + save

def sample_by_topic(df, per_topic=6):
    parts = [g.sample(min(per_topic, len(g)), random_state=SEED)
             for _, g in df.groupby("topic")]
    return pd.concat(parts).reset_index(drop=True)


def finalize(df, name, has_cell=True):
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(f"{OUT}/{name}.csv", index=False)
    print(f"\n===== {name}  ({len(df)} rows) -> {OUT}/{name}.csv =====")
    print("per-topic counts:", dict(df.topic.value_counts().reindex(TOPIC_ORDER)))
    if has_cell:
        print("cell x topic:")
        print(pd.crosstab(df.cell, df.topic).reindex(index=CELLS, columns=TOPIC_ORDER).to_string())
    samp = sample_by_topic(df)
    samp.to_csv(f"{OUT}/samples/{name}_SAMPLE30.csv", index=False)
    return df, samp


def main():
    import os
    os.makedirs(f"{OUT}/samples", exist_ok=True)
    rng = np.random.default_rng(SEED)

    assert_single_token_carriers()
    topics = {t: build_topic(t) for t in TOPIC_ORDER}
    print("\n[topics] usable entities:", {t: len(topics[t]["entities"]) for t in TOPIC_ORDER})
    atoms = build_atoms(topics, rng)
    pairs = all_pairs(topics, rng)
    print(f"[pairs] {K_PAIRS_PER_TOPIC}/topic -> {len(pairs)} pairs total")

    finalize(gen_binary(pairs, atoms, ("and", "or")), "r1_quadruples")
    finalize(gen_binary(pairs, atoms, (FILLER,)), "r1_leakage_control")
    finalize(gen_xor(pairs, atoms), "r2_setA_xor")

    dil = gen_dilution(topics, atoms, rng).sample(frac=1, random_state=SEED).reset_index(drop=True)
    dil.to_csv(f"{OUT}/r1_dilution.csv", index=False)
    print(f"\n===== r1_dilution  ({len(dil)} rows) -> {OUT}/r1_dilution.csv =====")
    print("per-topic counts:", dict(dil.topic.value_counts().reindex(TOPIC_ORDER)))
    print("n x condition:")
    print(dil.groupby(["n", "condition"]).size().to_string())
    print("topic x condition:")
    print(pd.crosstab(dil.topic, dil.condition).reindex(TOPIC_ORDER).to_string())
    sample_by_topic(dil).to_csv(f"{OUT}/samples/r1_dilution_SAMPLE30.csv", index=False)


if __name__ == "__main__":
    main()
