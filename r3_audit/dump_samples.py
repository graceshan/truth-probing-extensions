"""30 random raw rows per dataset with parsed operands + inherited labels, for
hand-verification, plus a consolidated log of label inconsistencies (rows whose
inherited operand labels contradict the dataset's own compound label)."""

import os

import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))  # reads parsed CSVs / writes samples here
BASES = ["cities", "sp_en_trans", "element_symb", "inventors", "animal_class", "facts"]
SEED = 20260824


def opcols(df):
    for a, b in [("disjunctA_obj", "disjunctB_obj"), ("disjunctA", "disjunctB"),
                 ("conjunctA", "conjunctB")]:
        if a in df.columns:
            return a, b
    raise KeyError(df.columns)


inconsistencies = []
for kind, op in [("disj", "or"), ("conj", "and")]:
    print("\n" + "#" * 100)
    print(f"# {kind.upper()}UNCTION — 30 random rows per dataset")
    print("#" * 100)
    for base in BASES:
        df = pd.read_csv(f"{OUT}/{base}_{kind}_parsed.csv")
        a, b = opcols(df)
        df["cell"] = (df.labelA.astype("Int64").astype(str).replace({"1": "T", "0": "F"})
                      + df.labelB.astype("Int64").astype(str).replace({"1": "T", "0": "F"}))
        derived = (df.labelA.astype("Int64") | df.labelB.astype("Int64")) if op == "or" \
            else (df.labelA.astype("Int64") & df.labelB.astype("Int64"))
        df["consistent"] = (derived == df.compound_label)
        samp = df.sample(n=min(30, len(df)), random_state=SEED).sort_values("idx")
        keep = ["idx", "statement", a, b, "labelA", "labelB", "cell", "compound_label", "consistent"]
        samp[keep].to_csv(f"{OUT}/{base}_{kind}_SAMPLE30.csv", index=False)
        print(f"\n===== {base}_{kind}  (showing 30 of {len(df)}; "
              f"{(~df.consistent).sum()} label-inconsistent rows in full set) =====")
        with pd.option_context("display.max_colwidth", 74, "display.width", 240):
            print(samp[keep].to_string(index=False))
        bad = df[~df.consistent].copy()
        bad.insert(0, "dataset", f"{base}_{kind}")
        inconsistencies.append(bad[["dataset", "idx", "statement", a, b,
                                    "labelA", "labelB", "cell", "compound_label"]]
                               .rename(columns={a: "operandA", b: "operandB"}))

log = pd.concat(inconsistencies, ignore_index=True)
log.to_csv(f"{OUT}/label_inconsistencies.csv", index=False)
print("\n" + "=" * 100)
print(f"LABEL-INCONSISTENCY LOG: {len(log)} rows total where inherited labels "
      f"contradict the dataset's compound label")
print("=" * 100)
print(log.groupby("dataset").size().to_string())
print(f"\nfull log -> {OUT}/label_inconsistencies.csv")
