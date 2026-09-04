# When Atomic Truth Directions Fail to Compose

This repository contains code and results for a research project on whether linear truth probes trained on **atomic factual statements** generalize to **compound statements** formed with `and` and `or`.

The main finding is that atomic truth directions transfer very differently across logical operators. Across both **Qwen2.5-7B-Instruct** and **Qwen3-8B**, a frozen atomic truth probe ranks conjunction truth well but performs substantially worse on disjunctions:

| Model | Atomic AUROC | AND transfer | OR transfer |
|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | 0.9996 | 0.9459 | 0.6791 |
| Qwen3-8B | 0.9995 | 0.9750 | 0.7046 |

A balanced `TT / TF / FT / FF` benchmark makes it possible to decompose this gap. The atomic direction separates **TT from mixed-truth compounds** much better than it separates **mixed-truth compounds from FF**. This geometry is favorable for AND, whose positive class is TT, but unfavorable for OR, whose negative class is FF.

For OR, the particularly weak mixed-vs-FF distinction replicates across both models:

- Qwen2.5: **0.580 AUROC**
- Qwen3: **0.611 AUROC**

This does not mean compound truth is absent from the representation. A linear probe trained directly on compound truth at the **same layer** achieves >0.99 AUROC on OR in both models. The result is therefore about what the **atomic truth direction exposes under composition**, rather than an inability of the model or layer to represent compound truth.

## Research question

Linear probes can decode whether simple factual statements are true with extremely high accuracy. But real statements often combine multiple facts, so truth information may be distributed across several parts of a sequence.

I ask:

> **Which constituent-truth distinctions does an atomic truth direction preserve under logical composition, and which does it fail to expose?**

The project focuses on transfer rather than merely asking whether compound truth is linearly decodable: the atomic probe is trained and selected using atomic data only, frozen, and then evaluated on compounds without tuning.

## Experimental design

### Atomic probe

For each model, I train an L2 logistic-regression probe on affirmative and negated atomic statements from five topics:

- cities
- Spanish–English translations
- inventors
- element symbols
- animal classes

Splits are grouped by entity so that entities do not cross the atomic train/test boundary.

Activations are taken from the **last real token** of the raw statement, without a chat template.

The operational layer is selected solely by held-out atomic AUROC:

- **Qwen2.5-7B-Instruct:** layer 22
- **Qwen3-8B:** layer 21

The probe is then frozen before compound results are examined.

### Compound benchmark

The main benchmark contains **8,000 compound statements** organized into a complete balanced truth table:

- AND / OR
- TT / TF / FT / FF
- AB / BA surface ordering

There are 500 canonical fact pairs, each producing 16 compound examples.

Evaluation uses 100 held-out canonical pairs, giving 1,600 held-out rows and exactly 200 examples in each connective × truth-cell combination.

This balanced design is important because it lets overall AND and OR AUROC be decomposed exactly into the pairwise truth-cell distinctions required by each operator.

## Main results

### 1. Atomic truth transfers much better to AND than OR

Frozen atomic probe:

| Model | AND | OR | Gap |
|---|---:|---:|---:|
| Qwen2.5-7B-Instruct | 0.9459 | 0.6791 | 0.2668 |
| Qwen3-8B | 0.9750 | 0.7046 | 0.2704 |

For Qwen3, a 2,000-replicate canonical-pair bootstrap gives:

- AND: 0.9750 `[0.9656, 0.9839]`
- OR: 0.7046 `[0.6871, 0.7257]`
- AND − OR: 0.2704 `[0.2476, 0.2881]`

The nearly identical gap across independently trained probes on two model generations was a central replication check.

### 2. The gap can be localized to constituent-truth boundaries

For a balanced four-cell benchmark:

\[
AUROC_{AND}
=
\frac{
A(TT,TF)+A(TT,FT)+A(TT,FF)
}{3}
\]

while

\[
AUROC_{OR}
=
\frac{
A(TT,FF)+A(TF,FF)+A(FT,FF)
}{3}.
\]

The frozen atomic direction preferentially separates the **upper boundary** (`TT → mixed`) over the **lower boundary** (`mixed → FF`).

| Model | AND TT–mixed | AND mixed–FF | OR TT–mixed | OR mixed–FF |
|---|---:|---:|---:|---:|
| Qwen2.5 | 0.928 | 0.682 | 0.834 | 0.580 |
| Qwen3 | 0.963 | 0.745 | 0.808 | 0.611 |

This matters because AND primarily requires distinguishing TT from the other cells, while OR critically requires distinguishing mixed-truth statements from FF.

### 3. OR score geometry is compressed

Along the frozen atomic direction, the mean TT-to-FF score range is smaller for OR than AND:

| Model | AND range | OR range | OR / AND |
|---|---:|---:|---:|
| Qwen2.5 | 10.84 | 6.17 | 0.569 |
| Qwen3 | 14.45 | 6.58 | 0.456 |

For Qwen3, the 95% bootstrap CI for the range ratio is `[0.398, 0.512]`.

The corresponding OR/AND pooled within-cell SD ratio is 1.025 `[0.922, 1.121]`, so the reduced aggregate signal-to-noise ratio is associated primarily with contraction of the between-cell range rather than an obvious increase in pooled within-cell dispersion.

This is descriptive evidence about frozen score geometry, not a causal explanation of how compound truth is represented.

### 4. Compound truth remains linearly recoverable

A new logistic-regression probe trained directly on compound truth at the same representation layer performs nearly perfectly:

| Model | AND | OR |
|---|---:|---:|
| Qwen2.5 | 0.9988 | 0.9914 |
| Qwen3 | 0.9977 | 0.9950 |

For Qwen3, the OR mixed-vs-FF distinction rises from **0.611** under the frozen atomic direction to **0.993** under compound supervision.

Thus the weak OR transfer does not establish that OR-relevant truth information is absent from the layer. Rather, the atomic direction fails to expose it effectively.

## Robustness and checks

Several checks were run to test the interpretation rather than relying only on the headline layer.

### Layer robustness

For Qwen2.5, standalone atomic AUROC is ≥0.99 at layers 9–27.

Across all **19/19** layers in this high-atomic-performance plateau:

- AND transfer AUROC > OR transfer AUROC
- OR TT-vs-mixed AUROC > OR mixed-vs-FF AUROC

Across all 28 layers, both inequalities hold at 25/28 layers.

### Independent Qwen3 replication

The Qwen3 experiment was run as a separate replication arm:

- a new 4096-dimensional probe was trained from scratch;
- the operational layer was independently selected from atomic data;
- the saved entity split was reused;
- the probe was frozen before compound evaluation;
- no Qwen2.5 probe weights were transferred to Qwen3.

The AND−OR gap replicated at 0.270, compared with 0.267 in Qwen2.5.

### Surface-position check

An initial TF-vs-FT diagnostic was later found not to test surface position: `TF` and `FT` were defined over canonical `A/B` identities before the `AB/BA` ordering transformation.

I therefore recomputed the comparison using actual surface position.

With AUROC oriented as “first constituent true” scoring above “second constituent true”:

| Model | AND | OR |
|---|---:|---:|
| Qwen2.5 | 0.348 | 0.434 |
| Qwen3 | 0.294 | 0.424 |

This reveals a real positional effect, especially for conjunctions: mixed compounds tend to score higher when the **second** constituent is true. The original canonical TF-vs-FT ≈ 0.5 result should therefore not be interpreted as evidence of positional symmetry.

## Repository structure

```text
data/
    Atomic and compound datasets.

acts/
    Extraction metadata and provenance.
    Large activation arrays are intentionally excluded from git.

scripts/
    Activation extraction and analysis scripts.

scripts/mechanism/
    Qwen2.5 mechanism and robustness analyses.

results/
    r1/mechanism/
        Qwen2.5 decomposition, geometry, bootstrap,
        and layer-sweep outputs.

    qwen3_8b_union_probe_gate/
        Qwen3 atomic layer selection, split metadata,
        frozen probe, and per-layer metrics.

    qwen3_8b_r1_core/
        Qwen3 frozen-transfer results, pairwise AUROCs,
        score geometry, compound positive control,
        and cross-model comparison.

    paper_round2_checks/
        Fixed-seed qualitative samples,
        corrected surface-position analysis,
        Qwen3 bootstrap confidence intervals,
        and layer-sweep verification.
```

## Reproducing the main analyses

The large cached activation arrays are intentionally not committed because they total several GB.

The analysis pipeline is:

```text
source datasets
      ↓
activation extraction
      ↓
cached activations (.npy, gitignored)
      ↓
atomic probe selection
      ↓
freeze probe
      ↓
compound transfer evaluation
      ↓
mechanism / robustness analyses
```

### Qwen3 atomic gate

```bash
python scripts/14_qwen3_atomic_union_probe_gate.py
```

This selects the Qwen3 operational layer using atomic held-out data only and saves the selected probe and split metadata under:

```text
results/qwen3_8b_union_probe_gate/
```

### Qwen3 compound replication

```bash
python scripts/15_qwen3_r1_core_replication.py
```

This evaluates the frozen atomic probe on the R1 compound benchmark and trains the same-layer compound positive-control probe.

Outputs are written to:

```text
results/qwen3_8b_r1_core/
```

### Paper robustness checks

```bash
python scripts/16_paper_round2_checks.py
```

This reproduces the fixed-seed qualitative samples, corrected surface-position diagnostic, Qwen3 bootstrap confidence intervals, and layer-sweep checks.

Outputs are written to:

```text
results/paper_round2_checks/
```

See `scripts/mechanism/` for the Qwen2.5 decomposition, score-geometry, bootstrap, and layer-sweep analyses.

## Data and extraction provenance

Qwen3 extraction metadata is committed under:

```text
acts/qwen3_8b/
```

These files record the row-aligned input statements and extraction provenance without committing the multi-gigabyte activation arrays themselves.

The extraction convention used for the main experiments is:

- raw statement text;
- no chat template;
- tokenizer special-token behavior left consistent with the original extraction pipeline;
- `output_hidden_states=True`;
- embedding output excluded;
- activation from the last real token;
- all transformer-block activations cached as float16.

Qwen2.5 and Qwen3 use separate activation spaces and separately trained probes.

## What I verified

Because much of the implementation and analysis used AI coding agents, I separately checked several assumptions that could invalidate the result.

Among other checks:

- verified zero entity overlap across the atomic train/test split;
- reused the saved entity split for the Qwen3 replication;
- verified the R1 held-out benchmark contains exactly 200 rows per connective × truth cell;
- confirmed the probe was selected and frozen before compound evaluation;
- reconstructed AND and OR AUROC from their pairwise truth-cell comparisons and matched the direct calculation to numerical precision;
- checked Qwen2.5 and Qwen3 tokenization behavior on representative raw statements;
- inspected randomly sampled raw atomic and compound examples;
- checked the R1 generation logic and false-constituent construction;
- discovered that the original TF/FT comparison used canonical rather than surface labels and recomputed the correct positional diagnostic;
- bootstrapped canonical pairs rather than individual compound rows to preserve the benchmark's grouped structure.

See:

```text
results/paper_round2_checks/
```

for the corresponding cached-data audit outputs.

## Limitations

This project studies two related Qwen model families and therefore does not establish universality across architectures.

The main readout is logistic regression. Mass-mean/TTPD-style truth directions are an important untested baseline.

The same-layer compound positive control is split by canonical pair rather than requiring complete entity disjointness across compound train and test pairs, so per-entity information may contribute to its very high performance.

The OR sentences combine facts that may be semantically unrelated. Some of the observed score compression could therefore reflect general effects of combining multiple unrelated statements rather than an OR-specific mechanism. A null-connective control such as `A. B.` would help distinguish these possibilities.

Finally, these experiments study linear readouts of internal representations rather than whether the models behaviorally answer compound truth questions correctly.

## Related work

This project builds on work studying linear truth representations and their behavior under logical composition.

In particular:

- **Bürger et al. (2024), _Truth is Universal_** study truth directions across datasets and include conjunction/disjunction datasets. Their structured disjunction datasets exclude TT examples by construction, meaning OR evaluation on that support primarily tests the mixed-vs-FF distinction.
- **Bao et al. (2025)** already report an AND-over-OR decoding gap on Llama-3.1-8B across several probe types. The contribution here is not the observation that AND can outperform OR, but using a balanced full truth table to localize why frozen atomic transfer differs.
- **Li, Patil & Rawlins (2026)** study conjunctions with balanced truth combinations and report that preliminary disjunction experiments are less successful. This project adds a balanced disjunction benchmark and explicitly studies atomic-to-compound transfer.
- **Poulis, Crovella & Terzi (2026)** emphasize the layer dependence of truth directions, motivating the all-layer robustness analysis rather than relying solely on the selected layer.

## Author

Grace Shan

September 2026