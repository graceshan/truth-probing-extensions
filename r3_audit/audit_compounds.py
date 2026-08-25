"""Audit Truth_is_Universal compound (conj/disj) datasets — pure CSV parsing.

Truth is inherited from the atomic datasets via a per-subject ORACLE built from
the atomic label==1 rows: subject -> set of true objects. A parsed operand
(subject, object) is True iff its object is in that subject's true set. This
labels every operand (not just those whose exact sentence appears verbatim in
the atomic file), which is what "inherit the true/false label" requires.

`facts` has no subject/object decomposition (each fact is atomic), so it uses an
exact-statement label lookup instead.
"""

import os
import re
import textwrap

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))          # .../truth-probing-extensions/r3_audit
REPO = os.path.dirname(HERE)                               # repo root
# Bürger's released CSVs — cloned as a sibling of this repo; override with $TIU_DATASETS
DATA = os.environ.get("TIU_DATASETS",
                      os.path.normpath(os.path.join(REPO, "..", "Truth_is_Universal", "datasets")))
OUT = HERE                                                 # per-dataset dumps land next to this script
SUMMARY_CSV = os.path.join(REPO, "r3_cell_counts.csv")     # summary at repo root (README's r3_cell_counts.*)
os.makedirs(OUT, exist_ok=True)

BASES = ["cities", "sp_en_trans", "element_symb", "inventors", "animal_class", "facts"]
ANAPHORIC = {"cities", "sp_en_trans", "element_symb", "inventors", "animal_class"}

# regex to pull (subject, object) out of a FULL atomic statement, per dataset
ATOMIC_RE = {
    "cities":       re.compile(r"^The city of (.+?) is in (.+?)\.?$"),
    "sp_en_trans":  re.compile(r"^The Spanish word '(.+?)' means '?(.+?)'?\.?$"),
    "element_symb": re.compile(r"^(.+?) has the symbol (.+?)\.?$"),
    "inventors":    re.compile(r"^(.+?) lived in (.+?)\.?$"),
    "animal_class": re.compile(r"^The (.+?) is a (.+?)\.?$"),
}
# (subject, objA, objB) out of an anaphoric disjunction's inner text
DISJ_RE = {
    "cities":       (re.compile(r"^the city of (.+?) is in (.+)$"),   re.compile(r"^it is in (.+)$")),
    "sp_en_trans":  (re.compile(r"^the Spanish word '(.+?)' means '?(.+?)'?$"), re.compile(r"^it means '?(.+?)'?$")),
    "element_symb": (re.compile(r"^(.+?) has the symbol (.+)$"),      re.compile(r"^it has the symbol (.+)$")),
    "inventors":    (re.compile(r"^(.+?) lived in (.+)$"),            re.compile(r"^he/she lived in (.+)$")),
    "animal_class": (re.compile(r"^the (.+?) is a (.+)$"),            re.compile(r"^it is a (.+)$")),
}

DISJ_PREFIX = "It is the case either that "
CONJ_PREFIX = "It is the case both that "


def norm(s):
    return None if s is None else " ".join(str(s).strip().rstrip(".").lower().split())


def obj_norm(base, s):
    """Object normalizer. sp_en_trans needs extra work: the anaphoric second
    disjunct drops the infinitive 'to ' prefix (means 'to seem' -> means seem')
    and the source CSV has stray single quotes (to 'smile); strip both so a
    meaning matches its atomic form regardless."""
    if s is None:
        return None
    if base == "sp_en_trans":
        s = str(s).replace("'", " ")
        s = " ".join(s.strip().rstrip(".").lower().split())
        if s.startswith("to "):
            s = s[3:]
        return s
    return norm(s)


def build_oracle(base):
    """subject -> set of true objects, from atomic label==1 rows."""
    df = pd.read_csv(f"{DATA}/{base}.csv")
    rx = ATOMIC_RE[base]
    oracle, subjects = {}, set()
    for stmt, lab in zip(df["statement"], df["label"]):
        m = rx.match(str(stmt).strip())
        if not m:
            continue
        subj, obj = norm(m.group(1)), obj_norm(base, m.group(2))
        subjects.add(subj)
        if int(lab) == 1:
            oracle.setdefault(subj, set()).add(obj)
    return oracle, subjects


def label_via_oracle(base, oracle, subjects, subj, obj):
    """1/0 if subject known, else None (can't inherit)."""
    s = norm(subj)
    if s not in subjects:
        return None
    return 1 if obj_norm(base, obj) in oracle.get(s, set()) else 0


def split_operands(stmt, prefix, sep):
    inner = str(stmt).strip()
    if inner.endswith("."):
        inner = inner[:-1]
    if not inner.startswith(prefix):
        return None, None, -1
    inner = inner[len(prefix):]
    parts = inner.split(sep)
    if len(parts) != 2:
        return None, None, len(parts)
    return parts[0].strip(), parts[1].strip(), 2


def parse_disj(base, stmt):
    """-> (opA_desc, opB_desc, subj, objA, objB) using the anaphoric pattern."""
    left, right, n = split_operands(stmt, DISJ_PREFIX, " or that ")
    if n != 2:
        return None, None, None, None, None, n
    rl, rr = DISJ_RE[base]
    ml, mr = rl.match(left), rr.match(right)
    if not (ml and mr):
        return left, right, None, None, None, n
    subj, objA, objB = ml.group(1), ml.group(2), mr.group(1)
    return left, right, subj, objA, objB, n


def audit_anaphoric_disj(base):
    oracle, subjects = build_oracle(base)
    df = pd.read_csv(f"{DATA}/{base}_disj.csv")
    rows, unmatched = [], []
    for i, r in df.iterrows():
        stmt, comp = r["statement"], int(r["label"])
        left, right, subj, objA, objB, n = parse_disj(base, stmt)
        la = label_via_oracle(base, oracle, subjects, subj, objA) if subj else None
        lb = label_via_oracle(base, oracle, subjects, subj, objB) if subj else None
        rows.append({"idx": i, "statement": stmt, "subject": subj,
                     "disjunctA_obj": objA, "disjunctB_obj": objB,
                     "labelA": la, "labelB": lb, "compound_label": comp, "n_parts": n})
        if la is None or lb is None:
            unmatched.append(rows[-1])
    return pd.DataFrame(rows), pd.DataFrame(unmatched)


def audit_facts_disj():
    df = pd.read_csv(f"{DATA}/facts_disj.csv")
    atom = pd.read_csv(f"{DATA}/facts.csv")
    lut = {norm(s): int(l) for s, l in zip(atom["statement"], atom["label"])}
    rows, unmatched = [], []
    for i, r in df.iterrows():
        stmt, comp = r["statement"], int(r["label"])
        left, right, n = split_operands(stmt, DISJ_PREFIX, " or that ")
        la = lut.get(norm(left)) if n == 2 else None
        lb = lut.get(norm(right)) if n == 2 else None
        rows.append({"idx": i, "statement": stmt, "disjunctA": left, "disjunctB": right,
                     "labelA": la, "labelB": lb, "compound_label": comp, "n_parts": n})
        if la is None or lb is None:
            unmatched.append(rows[-1])
    return pd.DataFrame(rows), pd.DataFrame(unmatched)


def audit_conj(base):
    df = pd.read_csv(f"{DATA}/{base}_conj.csv")
    rows, unmatched = [], []
    if base == "facts":
        atom = pd.read_csv(f"{DATA}/facts.csv")
        lut = {norm(s): int(l) for s, l in zip(atom["statement"], atom["label"])}
    else:
        oracle, subjects = build_oracle(base)
        rx = ATOMIC_RE[base]
    for i, r in df.iterrows():
        stmt, comp = r["statement"], int(r["label"])
        left, right, n = split_operands(stmt, CONJ_PREFIX, " and that ")
        if n != 2:
            la = lb = None
            sA = sB = None
        elif base == "facts":
            sA, sB = left, right
            la, lb = lut.get(norm(left)), lut.get(norm(right))
        else:
            # conjuncts are full atomic statements (non-anaphoric)
            mA, mB = rx.match(left[0].upper() + left[1:]), rx.match(right[0].upper() + right[1:])
            sA = f"{mA.group(1)} / {mA.group(2)}" if mA else None
            sB = f"{mB.group(1)} / {mB.group(2)}" if mB else None
            la = label_via_oracle(base, oracle, subjects, mA.group(1), mA.group(2)) if mA else None
            lb = label_via_oracle(base, oracle, subjects, mB.group(1), mB.group(2)) if mB else None
        rows.append({"idx": i, "statement": stmt, "conjunctA": sA, "conjunctB": sB,
                     "labelA": la, "labelB": lb, "compound_label": comp, "n_parts": n})
        if la is None or lb is None:
            unmatched.append(rows[-1])
    return pd.DataFrame(rows), pd.DataFrame(unmatched)


def cell_counts(df):
    ok = df.dropna(subset=["labelA", "labelB"]).copy()
    ok["labelA"] = ok["labelA"].astype(int); ok["labelB"] = ok["labelB"].astype(int)
    return {
        "TT": int(((ok.labelA == 1) & (ok.labelB == 1)).sum()),
        "TF": int(((ok.labelA == 1) & (ok.labelB == 0)).sum()),
        "FT": int(((ok.labelA == 0) & (ok.labelB == 1)).sum()),
        "FF": int(((ok.labelA == 0) & (ok.labelB == 0)).sum()),
    }, len(ok)


def check_or_and(df, op):
    """Consistency: does the compound's own label equal op(labelA,labelB)?"""
    ok = df.dropna(subset=["labelA", "labelB"]).copy()
    ok["labelA"] = ok["labelA"].astype(int); ok["labelB"] = ok["labelB"].astype(int)
    derived = (ok.labelA | ok.labelB) if op == "or" else (ok.labelA & ok.labelB)
    return int((derived != ok["compound_label"]).sum()), len(ok)


summary = []
for kind in ["disj", "conj"]:
    print("\n" + "=" * 78)
    print(f"{kind.upper()}UNCTION DATASETS")
    print("=" * 78)
    for base in BASES:
        if kind == "disj":
            full, un = (audit_facts_disj() if base == "facts" else audit_anaphoric_disj(base))
        else:
            full, un = audit_conj(base)
        cells, n_ok = cell_counts(full)
        n = len(full)
        bad, _ = check_or_and(full, "or" if kind == "disj" else "and")
        full.to_csv(f"{OUT}/{base}_{kind}_parsed.csv", index=False)
        if len(un):
            un.to_csv(f"{OUT}/{base}_{kind}_UNMATCHED.csv", index=False)
        summary.append({"dataset": f"{base}_{kind}", "n": n, "matched": n_ok,
                        "unmatched": n - n_ok, **cells, "logic_mismatch": bad})
        tag = " [anaphoric]" if (kind == "disj" and base in ANAPHORIC) else ""
        print(f"\n{base}_{kind}  (n={n}, matched={n_ok}, unmatched={n-n_ok}){tag}")
        print(f"   TT={cells['TT']:4d}  TF={cells['TF']:4d}  FT={cells['FT']:4d}  FF={cells['FF']:4d}"
              f"   | compound-label vs {'OR' if kind=='disj' else 'AND'} mismatches: {bad}")
        if n - n_ok:
            print(f"   !! {n-n_ok} rows could not inherit a label (see {base}_{kind}_UNMATCHED.csv)")
            for _, u in un.head(3).iterrows():
                print("      -", textwrap.shorten(str(u["statement"]), 96))

print("\n" + "=" * 78 + "\nSUMMARY TABLE (cell counts per dataset)\n" + "=" * 78)
sdf = pd.DataFrame(summary)
print(sdf.to_string(index=False))
sdf.to_csv(SUMMARY_CSV, index=False)

print("\n" + "=" * 78 + "\nTARGETED CONFIRMATIONS\n" + "=" * 78)
disj_tt = {r["dataset"]: r["TT"] for r in summary if r["dataset"].endswith("_disj")}
print("\nDisjunction TT counts (expect 0 for the five anaphoric sets, >0 for facts):")
for base in BASES:
    k = f"{base}_disj"
    flag = "anaphoric" if base in ANAPHORIC else "FULL (non-anaphoric)"
    print(f"   {k:22s} TT={disj_tt[k]:4d}   [{flag}]")
print(f"\nsaved parsed dumps + summary to {OUT}/")
