# Vendored slice of Truth_is_Universal datasets

These CSVs are copied verbatim from Bürger et al.'s **Truth is Universal**
release so the R3 audit (`r3_audit/`) is reproducible without cloning the full
upstream repo, and so the exact data version used is pinned. The negated
atomics are also vendored so the affirmative+negated union probe can train from
the same pinned slice.

- **Source:** <https://github.com/sciai-lab/Truth_is_Universal> (`datasets/`)
- **Commit:** `605ef00514415deb4806969a172f6e13e0798df7` (2024-11-07)
- **License:** MIT (upstream `LICENSE`)

## What's here (and what isn't)

The six topics used here, each as affirmative atomic + negated atomic +
conjunction + disjunction:

    {cities, sp_en_trans, element_symb, inventors, animal_class, facts}
      × {<topic>.csv, neg_<topic>.csv, <topic>_conj.csv, <topic>_disj.csv}

The R3 audit reads the affirmative atomics + `_conj`/`_disj`; the `neg_*` files
are for training the affirmative+negated union probe.

The rest of the upstream repo (German `_de` variants, `larger_than` /
`smaller_than`, `common_claim`, `counterfact`, model code) is **not** vendored —
clone it as a sibling for anything beyond these six topics, and point the audit
at it with `TIU_DATASETS=/path/to/Truth_is_Universal/datasets`.
