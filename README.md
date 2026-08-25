# Truth probes average their conjuncts

A linear "truth probe" applied to compound statements is not reading the logical
operator — it computes a weighted average of its conjuncts' truth scores plus a
constant, connective-dependent offset:

```
Score(A ⊕ B) ≈ w₁·s(A) + w₂·s(B) + c_⊕
```

This behaves like truth-conditional understanding on the standard evaluations
while being conjunct-averaging. The project measures the decomposition fresh at
7B and shows that training directly on compounds does not fix it — it only buys
a constant offset.

**Model:** Qwen2.5-7B-Instruct (primary), Llama-3.1-8B-Instruct (reserve second
model). Builds on the prior 1.5B work: <https://graceshan.github.io/truth-probing/>

---

## The question

Is a linear truth probe's compound score a readout of the compound's truth
value, or a weighted average of its parts?

---

## The four arms

Written in this order; **executed in reverse priority** — R3 is cheap and
near-certain so it is built last, R1/R2 carry the real uncertainty and get the
clock.

| Arm | Claim | Evidence | GPU |
|---|---|---|---|
| **R3** — motivation, reads first | The existing evaluation *cannot* establish operator-reading. In five of six disjunction sets the anaphoric template forces the second disjunct false, so TT is never realized; on a support without TT, OR ≡ XOR. | Cell-count audit of Bürger's released CSVs | none |
| **Figure 1** | The model distinguishes AND from OR; the probe barely moves. | Per-cell behavioral accuracy overlaid on per-cell probe score, same sentences | — |
| **R1** — spine | The score is a weighted average of the conjuncts plus a constant connective offset, and training directly on compounds does not fix it. | Interaction contrast Δ, gap ratios, union → compound-trained escalation, dilution curve, layer robustness | yes |
| **R2** — support | On a sentence class no additive readout can solve (XOR), the linear probe fails; an MLP on identical activations tells us whether the information is absent or merely not linearly readable. | Linear vs MLP at the critical cell, count-label positive control | yes |

Narrative order R3 → Figure 1 → R1 → R2: lead with *the field's instrument
cannot test this*, then build one that can.

---

## Key methodological commitments

These are the moves that separate "I found a null" from "I found a null and
ruled out the boring reasons it could be spurious." They are not optional.

- **Interaction contrast per connective, never pooled.** `Δ = μ_TT − μ_TF − μ_FT + μ_FF`. Additive → Δ = 0; operator-read AND → large positive; operator-read OR → large negative. Pooling AND and OR manufactures the exact null the additive hypothesis predicts.
- **Offset boxplots are the cancellation control for Δ**, not decoration — a scalar Δ ≈ 0 can be faked by AND-like and OR-like effects averaging across items; the per-cell distributions catch it.
- **Count-label positive control runs before any R2 failure is interpreted.** Relabel identical activations "at least one true"; the linear probe must clear every cell, or the setup can't clear the bar and the truth-label failure is uninterpretable.
- **Rung-zero TT gate.** TT behavioral accuracy ≥ 80% on Set A, or R2 does not run. Reported per cell, never averaged.
- **Union-probe sanity gate.** Held-out AUROC ≥ 0.95 before the direction is allowed to carry the R1 decomposition. Trained on affirmative **and** negated atomics — negations are what make the gate meaningful.
- **Every reported quantity names its probe** (see plan §5). "A probe I train" appears nowhere. Atom scores and compound scores in any single formula come from the *same* probe on the *same* scale.
- **Scale claim is mechanism-level, not number-level.** The 1.5B decomposition was measured on a *cities* probe this project does not run; the 7B numbers are a fresh measurement of the same mechanism, not a like-for-like replication.
- **Prompt-regime caveat honoured.** Behavioral lines use a chat template, probes read raw-statement activations, and those regimes have low cross-regime geometry (Poulis §5). Figure 1's caption states this rather than implying a clean apples-to-apples comparison.

---

## Which probe does what

| Probe | Trained on | Label | Carries |
|---|---|---|---|
| **Union** *(primary)* | cities + neg_cities activations, group-split on city | atomic truth | R1 decomposition (Δ, gap ratios, position slope, dilution); atom scores; Figure 1 probe line |
| **Compound-trained** | compound activations, group-split on city pair | compound truth | R1 escalation: pooled performance, per-cell offset table, 75% ceiling |
| **R2 linear-truth** | Set A (XOR) activations | XOR truth | the failure at TT |
| **R2 count-control** | *identical* Set A activations & splits | "at least one true" | proves the setup can pass an additively-solvable label |
| **R2 MLP** | *identical* Set A activations & splits | XOR truth | absent vs not-linearly-readable |

---

## Prior work being engaged

- **Bürger et al. (2024)** — fit a general truth direction, apply to compounds, report generalization. The critique: every two-conjunct AND/OR is a threshold on conjunct count, so a weighted sum solves them without representing the operator. Their results stand; the inference to truth-conditionality doesn't. Framed as extending an acknowledged limitation, **never as an error.**
- **Bao et al. (2025)** — reuse Bürger's compound data unmodified and evaluate with AUROC, so R3's table covers them too and the conjunct-counting half of the critique applies.
- **Poulis, Crovella & Terzi (2026), arXiv 2604.03754** — nearest prior work; cited in the first paragraph. Their difficulty hierarchy is counting-limited and never isolates boolean composition at fixed retrieval depth 2, which is exactly R2's impossibility set. Their F2 conjunction generalizing ≈ 1.0 is what conjunct-averaging predicts, not counter-evidence — pre-empted explicitly.

---

## Repository layout

> Filled in as the project is built. Activation dumps (`acts/*.npy`) are
> git-ignored; regenerate via extraction.


## Reproducing

1. **Environment.** `pip install torch transformers scikit-learn pandas numpy matplotlib`. Qwen2.5-7B in bf16 is ~15 GB; a 24 GB card suffices, inference-only.
2. **R3 (no GPU).** Parse Bürger's released CSVs (github.com/sciai-lab/Truth_is_Universal), inherit atomic labels, tabulate realized cells, confirm TT = 0 in the five anaphoric disjunction sets. Hand-verify 30 rows per dataset.
3. **Generate datasets.** Atomic (affirmative + negated), R1 quadruples, leakage control, dilution ladder, R2 Set A.
4. **Extract** all layers in one pass; checkpoint to `acts/`. Pick the probe layer empirically; pass the union-probe and rung-zero gates.
5. **Run R1, then R2**, then build the R3 table last.

---

## Pre-clock vs in-clock

The executive summary draws a hard line between the two.

- **Pre-clock (unbilled):** related-work reading, Runpod setup, the HF gating request, the R3 kill-switch / parse de-risk.
- **In-clock (billed):** dataset generation onward — the quadruple structure and single-token connective carrier are the design decision the spine rests on, so they are research, not setup.

Budget: 13 billed, 16 with reserve, 20 hard stop. **Never cut:** union-probe
gate, per-cell rung zero, Figure 1, Δ, compound-trained escalation, count-label
control, MLP control, cell-count table.
