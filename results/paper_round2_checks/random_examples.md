# Random qualitative examples

Sampling used seed 42 independently within every requested stratum. For each primary topic,
one row was sampled from its affirmative CSV and one from its negated CSV. One row was
sampled independently from `facts.csv` and `neg_facts.csv`. One R1 row was sampled
independently from each connective-by-cell stratum. No sampled row was replaced.

## Atomic examples

| category | topic | form | statement | label | source_path |
|---|---|---|---|---|---|
| cities_affirmative | cities | affirmative | The city of Dombivli is in India. | 1.0 | data/tiu_datasets/cities.csv |
| cities_negated | cities | negated | The city of Dombivli is not in India. | 0.0 | data/tiu_datasets/neg_cities.csv |
| sp_en_trans_affirmative | sp_en_trans | affirmative | The Spanish word 'guardar' means 'to save'. | 1.0 | data/tiu_datasets/sp_en_trans.csv |
| sp_en_trans_negated | sp_en_trans | negated | The Spanish word 'guardar' does not mean 'to save'. | 0.0 | data/tiu_datasets/neg_sp_en_trans.csv |
| inventors_affirmative | inventors | affirmative | Carl Sontheimer lived in Switzerland. | 0.0 | data/tiu_datasets/inventors.csv |
| inventors_negated | inventors | negated | Carl Sontheimer did not live in Switzerland. | 1.0 | data/tiu_datasets/neg_inventors.csv |
| element_symb_affirmative | element_symb | affirmative | Nickel has the symbol Ni. | 1.0 | data/tiu_datasets/element_symb.csv |
| element_symb_negated | element_symb | negated | Nickel does not have the symbol Ni. | 0.0 | data/tiu_datasets/neg_element_symb.csv |
| animal_class_affirmative | animal_class | affirmative | The llama is a bird. | 0.0 | data/tiu_datasets/animal_class.csv |
| animal_class_negated | animal_class | negated | The llama is not a bird. | 1.0 | data/tiu_datasets/neg_animal_class.csv |
| facts | heldout_facts | affirmative | The Milky Way is a spiral galaxy. | 1.0 | data/tiu_datasets/facts.csv |
| neg_facts | heldout_facts | negated | The Milky Way isn't a spiral galaxy. | 0.0 | data/tiu_datasets/neg_facts.csv |

## R1 compound examples

| category | statement | topic | conjunctA | conjunctB | labelA | labelB | cell | connective | ordering |
|---|---|---|---|---|---|---|---|---|---|
| AND-TT | The hippopotamus is a mammal and the lobster is a crustacean. | animal_class | The lobster is a crustacean. | The hippopotamus is a mammal. | 1.0 | 1.0 | TT | and | BA |
| AND-TF | The city of Malang is in Indonesia and the city of Asmara is in Nepal. | cities | The city of Malang is in Indonesia. | The city of Asmara is in Nepal. | 1.0 | 0.0 | TF | and | AB |
| AND-FT | Emile Berliner lived in Japan and Thomas Tompion lived in the U.K. | inventors | Emile Berliner lived in Japan. | Thomas Tompion lived in the U.K. | 0.0 | 1.0 | FT | and | AB |
| AND-FF | The Spanish word 'desarrollar' means 'sand' and the Spanish word 'viejo' means 'to save'. | sp_en_trans | The Spanish word 'viejo' means 'to save'. | The Spanish word 'desarrollar' means 'sand'. | 0.0 | 0.0 | FF | and | BA |
| OR-TT | The Spanish word 'clase' means 'class' or the Spanish word 'bosque' means 'forest'. | sp_en_trans | The Spanish word 'clase' means 'class'. | The Spanish word 'bosque' means 'forest'. | 1.0 | 1.0 | TT | or | AB |
| OR-TF | Lithium has the symbol Rh or Fluorine has the symbol F. | element_symb | Fluorine has the symbol F. | Lithium has the symbol Rh. | 1.0 | 0.0 | TF | or | BA |
| OR-FT | The jellyfish is a mammal or the gorilla is a mammal. | animal_class | The jellyfish is a mammal. | The gorilla is a mammal. | 0.0 | 1.0 | FT | or | AB |
| OR-FF | The gorilla is a reptile or the panda is a bird. | animal_class | The gorilla is a reptile. | The panda is a bird. | 0.0 | 0.0 | FF | or | AB |

## Template and generation audit

The five affirmative/negated pairs were validated row-by-row against the pinned files in
`data/tiu_datasets/`; the repository contains no separate generator for those source CSVs.
Their exact transformations are:

- cities: `The city of {entity} is in {object}.` / `The city of {entity} is not in {object}.`
- sp_en_trans: `The Spanish word '{entity}' means '{object}'.` / `The Spanish word '{entity}' does not mean '{object}'.`
- inventors: `{entity} lived in {object}.` / `{entity} did not live in {object}.`
- element_symb: `{entity} has the symbol {object}.` / `{entity} does not have the symbol {object}.`
- animal_class: `The {entity} is a/an {object}.` / `The {entity} is not a/an {object}.`

The entity-recognition patterns for both forms are in `src/data.py::ENTITY_PATTERNS`.
The R1 affirmative templates are defined in `scripts/01_generate_r1_r2_datasets.py::TOPICS`.
`build_atoms` generates a fixed false constituent by replacing the correct object with a
randomly selected different real object from the same topic; these false constituents are
wrong-object affirmatives, never negations. `join_binary` forms
`{first without period} and/or {second without period}.` and lowercases only a leading
`The` in the second conjunct. `gen_binary` emits both AB and BA surface orders while `_row`
retains canonical conjunctA/conjunctB and labelA/labelB. Model extraction uses raw statement
text with no chat template, as documented and implemented in `src/extract.py::extract_acts`.
