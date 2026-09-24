# Evaluation protocol

Commit `3145daa` (`3145daab52d359353b7a242c4965f6e4c49b63ca`) is the
**frozen exploratory snapshot**. All datasets and results generated from it
belong to the exploratory benchmark, including untracked activation caches.
Do not modify or overwrite existing exploratory data or result files; write
new experiment artifacts to separate paths.

Existing R1/R2 results are exploratory and must not be reported as untouched
final-test estimates. This includes outputs labeled "test", "heldout", or
"frozen", atomic layer-selection results, compound controls, replications,
and subsequent mechanism and robustness analyses.

New experiments must use **entity-first splits**: assign entities to disjoint
partitions before generating atomic variants or compound pairs. Keep every
variant of an entity in its assigned partition, and construct compounds only
from entities within the same partition. Splitting rows or canonical pairs
alone does not ensure entity disjointness.

Keep three roles separate, with disjoint data and entity membership:

1. **Atomic fitting:** fit atomic probes and any learned preprocessing using
   fitting data only.
2. **Validation:** choose methods, models, layers, and hyperparameters using
   validation data, separate from fitting and final evaluation. Any additional
   supervised controls must also keep their fitting data separate from these
   validation and final-evaluation partitions.
3. **Final evaluation:** evaluate the locked experiment suite on a new,
   untouched final-test set, without using it for selection or tuning.

Do not create the final evaluation set yet. Create it only after the experiment
suite is locked. Final-test data, examples, labels, and diagnostic results must
not be inspected while developing methods or used to revise them. Previously
inspected exploratory data cannot be relabeled as an untouched final test.
