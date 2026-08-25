# R3 — cell-count audit of Bürger's released compound CSVs

Pure CSV parsing, no model. For each compound dataset in
[`sciai-lab/Truth_is_Universal`](https://github.com/sciai-lab/Truth_is_Universal),
each operand is parsed out and labelled by inheriting truth from the atomic
dataset, then the realized TT/TF/FT/FF cells are tabulated. Reproduce with
`python3 r3_audit/audit_compounds.py && python3 r3_audit/dump_samples.py`
(clone `Truth_is_Universal` as a sibling of this repo, or set `$TIU_DATASETS`).

## Headline

**TT = 0 in all five anaphoric disjunction sets; `facts_disj` is the only
disjunction set that realizes TT.** On a support with no TT cell, OR is
indistinguishable from XOR — the motivation for R3.

This is structural, not a sampling accident: the anaphoric template
("the gazelle is a *mammal* or that **it** is a *mollusk*") shares one subject
across both disjuncts, and each subject is single-valued (one country per city,
one meaning per word, one symbol per element, one class per animal, one
residence per inventor), so two *distinct* objects can never both be true.
`facts_disj` joins two independent full facts, so both can be true.

## Cell counts

Labels inherited from the atomic `label==1` rows via a per-subject truth oracle;
all 12 datasets matched 100 % (0 unmatched). `logic_mismatch` = rows whose
inherited operand labels contradict the dataset's own compound label under
OR / AND (see below).

| dataset | n | TT | TF | FT | FF | logic_mismatch |
|---|--:|--:|--:|--:|--:|--:|
| cities_disj | 500 | **0** | 141 | 114 | 245 | 28 |
| sp_en_trans_disj | 500 | **0** | 121 | 115 | 264 | 0 |
| element_symb_disj | 500 | **0** | 128 | 134 | 238 | 3 |
| inventors_disj | 500 | **0** | 153 | 157 | 190 | 32 |
| animal_class_disj | 500 | **0** | 185 | 159 | 156 | 95 |
| facts_disj | 500 | **39** | 87 | 105 | 269 | 0 |
| cities_conj | 1498 | 750 | 310 | 310 | 128 | 0 |
| sp_en_trans_conj | 500 | 266 | 102 | 89 | 43 | 0 |
| element_symb_conj | 500 | 260 | 93 | 103 | 44 | 0 |
| inventors_conj | 500 | 287 | 76 | 102 | 35 | 0 |
| animal_class_conj | 500 | 242 | 109 | 91 | 58 | 0 |
| facts_conj | 500 | 235 | 119 | 111 | 35 | 1 |

## Method notes

- **Disjunctions are anaphoric** (2nd operand drops the subject); the subject is
  reconstructed from the 1st operand. **Conjunctions are full statements** joined
  by "and that". `facts` (both) uses exact-statement matching since facts have no
  subject/object decomposition.
- **`sp_en_trans` needs a meaning-normalizer**: its anaphoric 2nd disjunct drops
  the infinitive "to " (`means 'to seem'` → `means seem'`) and the source CSV has
  stray quotes (`to 'smile`). Without normalizing these, 27 rows are spuriously
  flagged inconsistent; with it, 0. TT = 0 either way.

## Secondary finding — 159 label inconsistencies in the released CSVs

Rows where the inherited operand labels contradict the dataset's own compound
label (full list: `r3_audit/label_inconsistencies.csv`):

| dataset | count | direction |
|---|--:|---|
| animal_class_disj | 95 | labeled False but contains a true disjunct |
| inventors_disj | 32 | labeled False but contains a true disjunct |
| cities_disj | 28 | both directions |
| element_symb_disj | 3 | labeled False but contains a true disjunct |
| facts_conj | 1 | labeled True but a conjunct is false |

Example: `"the gazelle is a mammal or that it is a mollusk"` is labeled **0**, yet
`"The gazelle is a mammal."` is atomic `label==1`. Consistent with a generator
that built intended-both-false disjunctions but let the false-distractor sampler
draw the subject's *true* object. **None of these are TT**, so the headline is
unaffected — but cite the individual cells with this caveat.

## Files

- `r3_cell_counts.csv` — the table above, machine-readable
- `r3_audit/<dataset>_<kind>_parsed.csv` — every row with parsed operands + inherited labels
- `r3_audit/<dataset>_<kind>_SAMPLE30.csv` — 30 random rows each for hand-verification
- `r3_audit/label_inconsistencies.csv` — all 159 flagged rows
- `r3_audit/audit_compounds.py`, `r3_audit/dump_samples.py` — reproduce everything
