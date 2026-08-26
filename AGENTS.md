# Project

MATS application extension of prior truth-probing work.

Read before research changes:
- docs/01_background_and_plan_revised.md
- docs/02_implementation_walkthrough_revised.md

## Current research claim

We are testing whether a scalar truth-probe score on compounds is primarily
an affine mixture of constituent truth scores and connective identity rather
than a clean whole-proposition truth variable.

This additive form can correctly classify AND and OR, so XOR is used as the
discriminating case where the same geometry becomes insufficient.

This is a claim about probe geometry, NOT that the model fails to understand
logical operators. Do not infer representational absence from decoder or
probe failure.

## Critical rules

- Group splits by entity / entity pair. Never leak entities across train/test.
- Union probe trains on affirmative + negated examples.
- A good union probe should work on negation; do not expect inversion.
- Atom and compound scores in one regression must use the same probe and
  same affine scale.
- Never independently z-score atom and compound scores.
- Treat entity pair, not individual sentence variant, as the statistical unit.
- AND/OR additive geometry is not itself a semantic failure; XOR is the
  discriminating falsification test.
- Do not infer representational absence from decoder failure.

## Current next step

Train and evaluate the five-topic Qwen union probe across layers.

Do not extract compounds until the union-probe gate passes.

## Coding rules

- Research code should save outputs rather than only print them.
- Before implementing a new helper, inspect src/ for an existing equivalent.
- Preserve existing experiment outputs unless explicitly replacing them.
- When changing analysis methodology, state the scientific consequence of
  the change before implementing it.
- Do not add AI or Codex attribution, co-author trailers, or references to
  Codex in commits or pull-request descriptions.
