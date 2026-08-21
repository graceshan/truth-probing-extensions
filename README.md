# Truth-Probing Extensions

Follow-on work extending my truth-probing project, investigating how linear truth
probes behave on logically compound statements.

## Motivation

Linear probes on residual-stream activations separate true from false factual
statements with high accuracy, but this accuracy can reflect factual association
rather than truth-conditionality. This repo investigates that measurement-validity
question for compound statements.

## Contributions

- Mechanism: on "and"/"or" compounds, probe scores behave like a weighted average
  of the two conjunct scores, with the connective adding a roughly constant offset,
  suggesting the probe counts true conjuncts rather than representing the operator.
- Metrics critique: within-connective evaluation cannot separate operator-reading
  from conjunct-counting, so published generalization claims for compounds are
  potentially confounded.

## Experiments

- [ ] Operator-swap on fixed fact pairs (does probe score move with and/or?)
- [ ] Nested three-fact statements mixing "and" with "or"
- [ ] Scale sweep across model sizes
- [ ] TTPD and mass-mean probes alongside logistic regression
- [ ] Cross-domain compounds beyond cities

## References

- Bürger, Hamprecht, Nadler. Truth is Universal: Robust Detection of Lies in LLMs.
  NeurIPS 2024. arXiv:2407.12831
