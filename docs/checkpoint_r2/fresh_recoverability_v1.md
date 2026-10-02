# Bounded fresh CPU recoverability stage

This new runner implements the explicitly authorized early CPU stage at base
`7656a0eb68a5fd7ff7dac1c73efeab982d540cc2`. Historical configurations and code are
unchanged. Its new execution configuration is committed before any real fit.
The exact implementation commit is required by `--expected-commit` and recorded
in every result/companion receipt; the checkout must be clean at launch.

The independent acceptance receipt verifies the pinned complete-copy inventory,
every listed file and exact full producer `verify-output`, including all arrays,
token packets, ordered logical bindings and layer conventions. It describes
review/re-execution on `MacBook-Pro-7.lan`, without claiming a second physical Mac.
The runner repeats inventory/output verification before use. The documented
NumPy 2.4.6 rebuild environment additionally reproduces all frozen metadata bytes;
the fitting runtime's numerical package versions are separately frozen and logged.

`FreshAdapter` resolves `(group, logical_id)` via hashed frozen source metadata
to an extracted text ID and shard row, preserving source order and multiplicity.
For example, atomic D has 1,012 observations but 1,006 distinct text rows: the six
duplicate statements remain separate logical observations. Fitting and score
packets carry ordered composite identities and binding hashes. No text matching
or ordering repair is permitted. P15 has 2,778 rows and excludes every source
variant of all 75 A15 persons; T_C has 100 training persons/500 pairs/8,000 rows;
bare D has 483 pairs/7,728 rows. Training and D person sets are disjoint.

One layer is streamed at a time from read-only NPY mappings, into float64 logical
blocks. Fresh heads, P15 preprocessing, row-bound scores and bootstrap arrays go
to a separate new external directory. All source file size/mtime/ctime tuples
are rechecked during/after use; initial validation hashes every file. Historical
parameters, statistics and 36 conditional matches are never loaded or adopted.

Exactly 122 fits are planned: raw reduced LR at Qwen L17/C=1 and Llama L15/C=10,
plus R0 and C_clean at every saved layer (28/32). Reduced LR directly reuses the
reviewed raw LR implementation and 2,000/10,000 cold warning-retry policy.
R0 uses reviewed mean-P15 BCE plus .001 weight penalty, with P-only population
standardization, floor1e-6 and constant-coordinate masking. C_clean transforms
the full mixed T_C block with the same P statistics and calls the reviewed
single-mean-block solver on T_C alone: mean-T_C BCE plus .001 weight penalty,
not the half-P/half-compound repair objective. Both have free intercepts, zero
initialization, 2,000 initial iterations, up to 10,000 total on one warm retry,
finite/library-success/gradient-infinity<=1e-4 checks. Failures remain explicit
in the inventory, tables and curve gaps; no failed predictions are admitted.

Evaluation uses only atomic D and bare compound D. Each head reports atomic,
AND, OR and OR mixed-versus-FF AUROC, pooled, per-topic and equal-five-topic
macro. R0 selection calls the original atomic-only pooled-AUROC/exact-tie-lower-
layer implementation. C_clean follows inherited pooled D OR boundary, then
atomic D and continuous standardized OR separation within1e-12. Complete final
ties use the protocol's shared fresh PCG64(20261008) choice over sorted head IDs.
D-selected C_clean and conditional intervals are optimistic development results.
The atomic-selected R0 also has selected-on-D exposure. No compound metric enters
R0 selection, and no S_atom_all bank selector is run.

The reviewed bootstrap uses seed1729, 2,000 draws, whole-person atomic weights
and endpoint-product compound weights, shared across heads and models. Undefined
draws are retained; all-five macro propagates undefined topics. Pointwise linear
percentile intervals require 1,800 valid draws. Layer selection is not repeated
inside the bootstrap. Paired C_clean-minus-R0 and C_clean-minus-reduced-LR
contrasts include every same-layer comparison and fixed/selected layer controls.
Atomic/AND retention against fresh reduced LR uses unchanged .005/.02 margins,
reported as point retention and paired lower-bound support.

One fitting process and one BLAS/OpenMP thread are enforced and logged. Runtime,
Darwin peak RSS bytes, complete attempts, source/config/input hashes and external
artifact hashes are recorded. No outcomes trigger retuning. Wording fitting,
selection and evaluation, other atomic methods, B25, constituent fitting,
behavior/chat and E are excluded.

Reproduction from the exact implementation commit (choose a new external output):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLCONFIGDIR=/private/tmp/r2-fresh-recoverability-matplotlib
/private/tmp/clean-extraction-venv/bin/python -B -m unittest tests.test_checkpoint_r2_fresh_recoverability -v
/private/tmp/clean-extraction-venv/bin/python -B -m src.checkpoint_r2_fresh_recoverability run \
  --expected-commit EXACT_IMPLEMENTATION_SHA \
  --acceptance /Users/apple/truth-probing-backups/20261001T164224Z/r2-recoverability-results-20261002/acceptance.json \
  --output /Users/apple/truth-probing-backups/20261001T164224Z/r2-recoverability-results-20261002/NEW_RUN
```

The acceptance and numerical runtime are pinned by the new configuration; a
changed installation or handoff fails closed. These commands do not alter the
original backup or any historical fit. Completed findings are delivered in a
separate results report after execution.
