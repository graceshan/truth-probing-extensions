# Claim-to-evidence map

Each row ID below resolves to the full-precision source row, complete scientific identity, zero-based record index and file SHA-256 in [extracted-rows.json](evidence/extracted-rows.json). Every displayed value additionally has a [display binding](evidence/display-map.json). The [machine-readable map](claim-evidence.json) records models, endpoints and comparisons. All results claims require pointwise conditional inference and the unchanged lock.

**C1: The compound-supervised readout improves pooled held-out-entity bare OR-boundary AUROC beyond compound-guided atomic-readout selection in both models.**

Role: original primary. Required qualifications: Separate models; TT excluded; fixed B25 benchmark; comparison is R_all - S_all.

Assets: [tables/main.pdf](tables/main.pdf), [figures/primary-gain.pdf](figures/primary-gain.pdf)

Source rows: `scalar-contrasts:14658`, `scalar-contrasts:378`, `scalar-metrics:22393`, `scalar-metrics:22750`, `scalar-metrics:24892`, `scalar-metrics:25249`, `scalar-metrics:26796`, `scalar-metrics:27867`

**C2: Both original-primary lower bounds exceed the declared .02 practical-gain threshold.**

Role: practical qualification. Required qualifications: Practical gain is a separate criterion applied to unrounded bounds.

Assets: [figures/primary-gain.pdf](figures/primary-gain.pdf)

Source rows: `scalar-contrasts:14658`, `scalar-contrasts:378`

**C3: Both models have positive fixed-five-topic macro R_all - S_all intervals.**

Role: equal-topic qualification. Required qualifications: Macro is not co-primary; does not imply each topic or sample improves; .02 is not required for macro support.

Assets: [tables/main.pdf](tables/main.pdf), [figures/primary-gain.pdf](figures/primary-gain.pdf)

Source rows: `scalar-contrasts:14664`, `scalar-contrasts:384`

**C4: Atomic and AND retention against the frozen reduced-training canonical LR is supported at pooled and macro scopes in both models.**

Role: retention criteria. Required qualifications: Atomic margin .005; AND .02; lower-bound rule inclusive; AUROC only; no joint coverage.

Assets: [tables/retention.pdf](tables/retention.pdf), [figures/retention.pdf](figures/retention.pdf)

Source rows: `scalar-contrasts:18921`, `scalar-contrasts:18927`, `scalar-contrasts:18928`, `scalar-contrasts:18934`, `scalar-contrasts:4641`, `scalar-contrasts:4647`, `scalar-contrasts:4648`, `scalar-contrasts:4654`

**C5: Qwen has small negative atomic differences and wholly negative intervals while meeting the .005 retention margin.**

Role: unfavorable qualification. Required qualifications: Retention within a margin does not establish zero loss.

Assets: [tables/retention.pdf](tables/retention.pdf)

Source rows: `scalar-contrasts:4641`, `scalar-contrasts:4647`

**C6: R_all exceeds S_full in pooled OR-boundary AUROC with lower bounds above .02; macro intervals are also positive.**

Role: full-training sensitivity. Required qualifications: Does not replace primary; atomic exposure, preprocessing and balanced sampling change together; S_full uses compound labels; full-training repair untested.

Assets: [tables/main.pdf](tables/main.pdf)

Source rows: `scalar-contrasts:28938`, `scalar-contrasts:28944`, `scalar-contrasts:33698`, `scalar-contrasts:33704`

**C7: S_full - S_atom_full has positive pooled and macro intervals in both models.**

Role: separate selection-label-benefit comparison. Required qualifications: Compound-guided versus atomic-guided selection within the tested full bank; not learned repair or pure exposure effect.

Assets: [tables/secondary.pdf](tables/secondary.pdf)

Source rows: `scalar-contrasts:29414`, `scalar-contrasts:29420`, `scalar-contrasts:34174`, `scalar-contrasts:34180`

**C8: R_all - S_atom_full has positive pooled and macro intervals in both models.**

Role: additional preserved comparison. Required qualifications: Different comparator from original primary; not a replacement for it.

Assets: [tables/secondary.pdf](tables/secondary.pdf)

Source rows: `scalar-contrasts:29890`, `scalar-contrasts:29896`, `scalar-contrasts:34650`, `scalar-contrasts:34656`

**C9: D-selected bank and C_clean references remain descriptive.**

Role: descriptive references. Required qualifications: Selection used D; no E search; fixed optimistic references; cannot claim information outside collective bank span.

Assets: [secondary-results.md](secondary-results.md)

Source rows: `scalar-metrics:21441`, `scalar-metrics:21447`, `scalar-metrics:23464`, `scalar-metrics:23470`, `scalar-metrics:23940`, `scalar-metrics:23946`, `scalar-metrics:25963`, `scalar-metrics:25969`, `scalar-metrics:27034`, `scalar-metrics:27040`, `scalar-metrics:27153`, `scalar-metrics:27159`, `scalar-metrics:28105`, `scalar-metrics:28111`, `scalar-metrics:28224`, `scalar-metrics:28230`

**C10: Both OR wording variants lower all six R_all pooled boundary points; eleven of twelve intervals are wholly negative and Llama sample 37 at-least-one crosses zero.**

Role: unfavorable wording results. Required qualifications: Per-sample same-head wording minus bare; no interval-endpoint averaging; pointwise conditional; not wording invariance.

Assets: [tables/wording.pdf](tables/wording.pdf)

Source rows: `wording-paired:1022`, `wording-paired:1050`, `wording-paired:1442`, `wording-paired:1470`, `wording-paired:4634`, `wording-paired:4662`, `wording-paired:5474`, `wording-paired:5502`, `wording-paired:5810`, `wording-paired:5838`, `wording-paired:938`, `wording-paired:966`

**C11: Constituent transfer has material deficits under explicit-or-both wording, alongside uncertain comparisons.**

Role: unfavorable constituent results. Required qualifications: Pooled all-order AND-to-OR at source-selected layers; target-trained minus source-trained; six AUROC endpoints; .05 frozen classifications; other topics/orders and .90 classifications retained in full sources.

Assets: [tables/constituent.pdf](tables/constituent.pdf), [secondary-results.md](secondary-results.md)

Source rows: `constituent-contrasts:105`, `constituent-contrasts:112`, `constituent-contrasts:119`, `constituent-contrasts:126`, `constituent-contrasts:133`, `constituent-contrasts:140`, `constituent-contrasts:630`, `constituent-contrasts:637`, `constituent-contrasts:644`, `constituent-contrasts:651`, `constituent-contrasts:658`, `constituent-contrasts:665`

**C12: Both models meet the locked benchmark improvement-with-retention criteria.**

Role: conjunction of separate checks. Required qualifications: Not simultaneous or joint 95% coverage; held-out entities within benchmark; no unseen-topic, calibration, behavior, causal-use or atomic-loss-mechanism claim.

Assets: [results.md](results.md), [tables/main.pdf](tables/main.pdf), [tables/retention.pdf](tables/retention.pdf)

Source rows: `scalar-contrasts:14658`, `scalar-contrasts:378`, `scalar-contrasts:18921`, `scalar-contrasts:18927`, `scalar-contrasts:18928`, `scalar-contrasts:18934`, `scalar-contrasts:4641`, `scalar-contrasts:4647`, `scalar-contrasts:4648`, `scalar-contrasts:4654`

**Methods and audit provenance.** Model IDs/revisions, representation, allocation, fitting and selection are pinned in [protocol-source-pins.json](protocol-source-pins.json) and linked at the corresponding methods paragraphs. E counts come from the pinned v2 manifest; frame sizes and inference from the final lock. The audit counts and acceptance come from the hash-pinned companion receipt and accepted audit report, not this formatting build. Limitations state the scope and exclusions in that lock. No novelty or literature claim is proposed.
