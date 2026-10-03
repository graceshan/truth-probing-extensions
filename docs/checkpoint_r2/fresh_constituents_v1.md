# Fresh raw constituent-truth transfer panel

This successor starts exactly at `1187140c7f79b0b64439eff382d420f51d974a25`
(completed B25 delivery). It adds a constituent production path without changing
historical code, the synthetic-only common adapter, frozen science or B25 solver.
No RunPod contact, atomic-bank/B25 fitting, allocations, new supervision budget,
behavior/chat, interventions or E activations/predictions are part of this panel.

Inputs resolve from the committed recoverability acceptance/delivery receipts.
Every listed complete-fresh-copy hash and the exact full `verify-output` are
checked before use. Logical `(group, logical_id)` observations retain original
order and multiplicity even when physical text is shared. FrozenBindings verifies
source/person/fact identities, truth, AB/BA labels and rendered strings; the
production adapter additionally checks fresh text/metadata hashes, partitions,
representation index, all-layer axes and FP16 files. Common synthetic score
packets remain synthetic-only. Production scores bind ordered observations,
representation, producer, head parameters, locked layer and file hashes.

The 20 TC entities/topic and all pair lists are already frozen. Internal source
validation has four A15 entities/topic, all six pairs/topic, 240 rows/operator.
The remaining 16 entities/topic and 100 frozen pairs/topic give 4,000 source-fit
rows/operator. Validation people are excluded from source fitting and P15.
Final refits use the same full-TC 100-of-190 pairs/topic as C_clean, 4,000 rows
per operator. No random sampling is repeated. Final refits deliberately include
the validation entities only after the immutable source-only selection lock.

Each model/layer/source operator fits FIRST and SECOND SURFACE truth heads:
240 initial fits. AB/BA swaps labels and surface fact/person identities. Each
objective is mean constituent BCE + .001*||w||² with an unpenalized intercept.
The unchanged reviewed mean-single-block numerical primitive is called with
that constituent block, not P BCE or half-P repair. It retains zero initialization,
float64 analytic gradients, finite/library-success/gradient-infinity<=1e-4 gates,
2,000 initial iterations and one warm retry within 10,000 cumulative iterations.
No raw-LR/TTPD amendment is imported. A fit/validation gate failure preserves
its evidence and stops downstream execution without tuning or silent exclusion.

Only source-operator internal validation chooses layers. Reuse the frozen
six-endpoint selector: maximize the minimum of the six equal-topic macros,
then their mean within 1e-12, then lower saved layer. Both marginal and all four
conditional endpoints remain separate. Missing topics propagate undefined
macros. D/target/wording scores cannot enter this function. D all-layer source-fit
curves are descriptive and separate from selected full-TC refits. They include
both evaluation operators to expose source-competent-layer transfer patterns.

At each source-selected layer refit source and matched target FIRST/SECOND heads
on full TC, in both directions. Only identical model/layer/operator/position,
ordered exposure, preprocessing and objective settings are deduplicated; at most
16 final fits. Evaluate source-on-source, frozen source-on-target and matched
 target-on-identical-target D. Evaluate all three established held-out wordings
with those locked heads. Per-topic, pooled and all-five macro six endpoints are
reported, with AB/BA descriptive splits. Fixed decisions use **logit >= 0**, i.e.
sigmoid >= .5; exact ties are **True**. Report head accuracy/BA, joint correctness,
min logits for AND/max for OR compound AUROC and relevant boundary AUROC, plus
Boolean decision composition accuracy/BA. No threshold fitting, sign flipping,
scale alignment or target recalibration is performed.

Reuse the reviewed D PCG64(1729) topic-stratified person multiplicities exactly,
with endpoint-product row weights, shared across models/heads/operators/wordings.
All 2,000 draws are retained. Zero-class/empty-topic draws remain undefined;
all-five macro propagates them. Pointwise linear 95% percentile intervals require
1,800 valid draws and condition on the observed locked fits. Surface-order splits
are descriptive points, not additional confirmatory interval families. Target
minus source transfer losses compare identical target observations. The .90
absolute target and .05 noninferiority planning margin remain separate per
endpoint. Upper paired bound strictly below .05 supports the planning margin;
lower bound above .05 supports a material deficit; crossing/insufficient bounds
remain unresolved. Small losses between weak heads are not useful transfer.

Fresh P15 statistics come from the hash-bound accepted recoverability producer
`3a5b7bb286876ff7bca4a57ddf2f337dc5d1733f`. Every preprocessing file is verified
and reproduced exactly from fresh P15 features before reuse: population ddof=0,
std floor1e-6 and constant-coordinate masking. No prior fitted head is imported.
One fitting process and one BLAS/OpenMP thread run on actual macOS; CPU runtime,
peak RSS, complete fit statuses and provenance are recorded. Large arrays and
heads stay in a fresh external directory. Existing results are never overwritten.

Implementation, configuration, tests and the metadata gate are committed before
any real fit. The exact fitting producer SHA is required at launch, with a clean
checkout. After execution the read-only audit reconstructs every prediction from
saved parameters and fresh features, independently checks BCE/gradient gates,
source-only selection, sklearn point metrics, the shared bootstrap schedule and
principal paired intervals, without refitting. `delivery.json` pins external
artifacts and compact committed results. These are development diagnostics;
no causal, historical-compatibility or final-test claim is warranted.

From the exact fitting producer (replace PRODUCER_SHA and choose a new external
RUN_DIRECTORY):

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
/private/tmp/clean-extraction-venv/bin/python -B -m unittest tests.test_checkpoint_r2_fresh_constituents -v
/private/tmp/clean-extraction-venv/bin/python -B -m src.checkpoint_r2_fresh_constituents validate --output NEW_METADATA_RECEIPT.json
/private/tmp/clean-extraction-venv/bin/python -B -m src.checkpoint_r2_fresh_constituents run --expected-commit PRODUCER_SHA --output RUN_DIRECTORY
/private/tmp/clean-extraction-venv/bin/python -B -m scripts.checkpoint_r2_audit_constituents --run RUN_DIRECTORY --expected-producer PRODUCER_SHA --receipt NEW_AUDIT_RECEIPT.json
```

The audit command is also the saved-prediction/table rebuild check from the final
delivery checkout: the frozen configuration hashes code and data, and the expected
producer binds the external completion. No fitting is invoked by this command.
