# Scientific-lock candidate, October 4, 2026

**For review only. E remains closed.** The user directed preparation and publication with the choices below. That instruction approves this task, not the exact final lock or E access. The candidate remains `DRAFT`, `scientific_lock_approved: false`, `claims_finalized: false`, and `approval: null`. No real phase authorization or accounting adoption is created.

## Exact references and preserved contract

- Reviewed base delivery: `e8f087e53b96b2172cfcde03e3a451e6a65bb0fe`.
- Proposed runnable E producer: `b2c6dde71e82f5f0af308761bf19ed3341670587`.
- Full-bank delivery: `754ff1d7d362cf76ddb2bca681b32b2aeb7118f5`.
- Branch: `checkpoint-final-e-lock-candidate-20261004`.
- Candidate: `config/final_e/lock_candidate_20261004.json`.
- Candidate SHA-256: `be5b15fe38c6c32a344e321d767d04a6f4c3b6f529ecfb338c2cc6e6edd5d1d3`.
- Governing Revision 3 document SHA-256: `ae135d188dddac61d921c0efc8b79499b6344396aae1046f5c9cce1283003035`.

The candidate is an additive copy of the original draft under the existing `final-e-scientific-lock-v1` schema. Its `resolutions` contain proposed scientific decisions, and its explicit `interpretation_contract` records the detailed rules. The unresolved independent-review/execution decision remains in `pending`. Additional candidate metadata does not change the gate, which reads the existing fields. Approval and phase authorizations remain mandatory. There is no implementation/schema conflict requiring an executable change.

The original draft, six-entry `immutable_contract`, all frozen configurations, trained heads, statistical rules and historical receipts retain their bytes. Execution modules are unchanged. The candidate delivery commit is a review artifact, never the runnable producer or a replacement for prior head/extraction provenance.

## Question and claim to test

With a fixed structured compound-supervision allocation, how much does learning a readout add beyond selecting an atomic-trained readout, and does that advantage generalize to held-out entities while retaining atomic and AND discrimination?

The proposed claim is that under the tested B25 protocol, learning a compound-supervised readout improves OR-boundary discrimination beyond selecting an atomic-trained readout, with atomic and AND retention assessed separately. B25 means 25 pair blocks, five per topic, expanded into 400 dependent compound rows per sample. Disclose the substantial atomic supervision; this is neither 25 independent binary labels nor a minimum-budget/sample-efficiency result.

## Endpoints and interpretation rules

| Decision | Proposed contract |
|---|---|
| Original primary | `R_all − S_all`, pooled **bare OR TF/FT-versus-FF AUROC**, excluding TT, separately for Qwen and Llama. |
| Samples | Ordinary mean of samples 11, 23 and 37, including the mean of metrics/paired differences inside every bootstrap draw. Never average or concatenate different fitted heads' scores. |
| Full-training sensitivity | `R_all − S_full` separately supports restored-full-atomic-training robustness. It cannot replace a failed original primary. Preserve every frozen contrast, including `S_full − S_atom_full` and `R_all − S_atom_full`. |
| Positive gain | Paired interval lower bound **> 0**. |
| Practical gain | Lower bound **> .02** establishes gain beyond the practical margin. An interval crossing or touching .02 leaves that threshold unresolved; upper bound < .02 does not reach it. An undefined interval supports neither claim. |
| Topic weighting | Pooled stays primary. A corresponding positive fixed-five-topic macro interval supports the additional equal-topic-weighting claim. Macro is not co-primary; its lower bound need not exceed .02. Neither every topic nor every sample must be individually significant. |
| Retention | `R_all − reduced_LR`, where `reduced_LR` is the frozen reduced-training canonical LR evaluated on E. Require lower bounds **≥ −.005 atomic** and **≥ −.02 AND**, at both pooled and macro scopes. |
| Retention uncertainty | A bound crossing a loss boundary, or an undefined interval, leaves preservation unresolved. Upper bound strictly below the boundary supports retention failure. Equality of the lower bound with the loss boundary satisfies the declared retention criterion. |
| Secondary analyses | Frozen wordings, topics, constituent evaluations and controls explain or qualify the primary result. They cannot rescue it. |

Full atomic training is an advantaged sensitivity comparator: atomic exposure includes A identities, and exposure, preprocessing and balanced sampling change together. This is not a pure withholding intervention. `S_full` uses compound labels; full-training repair was not tested. A claim that compound labels improve selection requires its own positive paired `S_full − S_atom_full` lower bound. Frozen D-selected bank maxima remain optimistic descriptive references, not an E search or selected procedure.

For each model, positive original-primary gain plus atomic/AND retention at both scopes supports the basic improvement-with-retention claim. Gain beyond .02, equal-topic improvement and full-training robustness each require their own specified checks. Both models satisfying the basic checks supports a replicated claim; one supports a model-specific claim. Different significance labels alone do not establish a model difference.

Original-primary success without full-bank support narrows the robustness claim. Pooled success without macro support qualifies topic weighting and does not establish improvement in every topic. Gain without retention support permits a compound-improvement statement without established preservation; distinguish unresolved retention from supported failure. Report unfavorable, contradictory, undefined and inconclusive outcomes explicitly.

## Frozen uncertainty

Preserve 2,000 topic-stratified entity draws using `numpy.random.Generator(PCG64)` with seed 1729. Atomic and compound membership frames remain separate (294 and 225 people). Within each frame, models, methods, samples, wordings and comparisons share draws. Atomic rows use entity multiplicity; compound rows use the product of endpoint multiplicities, with dependent cells/orders sharing pair weights.

Macro remains the ordinary mean of all five fixed topics. Undefined endpoints propagate; there is no surviving-topic average, redraw, imputation or topic dropping. Preserve valid/invalid counts and the requirement of at least 1,800 finite draws out of 2,000 for an interval. Use the frozen 95% pointwise percentile intervals and linear quantiles, conditional on fitted heads, selectors and evaluated support.

**No joint atomic–compound or simultaneous confidence claim is proposed.** A conjunction of passing pointwise checks does not create a simultaneous 95% guarantee. Do not replace the frames with a union population or require every sample/topic to be independently significant.

## Scientific scope and stopping rules

E establishes held-out-entity evidence within this benchmark: unseen by readout fitting/selection, not necessarily unfamiliar to the pretrained models. It does not establish unseen-topic or unrestricted compositional generalization, model reasoning failure, causal use of decoded information, or information outside the collective span of the finite bank. Retention is AUROC retention, not calibration, frozen-decision or behavioral retention.

Compound support is uneven: animals 16 entities, cities 146, elements 18, inventors 11 and Spanish 34. Report pooled and macro together, with topic uncertainty; equal weighting does not create evidence for small topics. Constituent decodability does not establish how the model uses that information when answering.

Defer behavior, matched chat, calibration, causal work and loss-deletion fits. Omit the claim that the atomic-loss term causes retention. The proposed deletion ablation concerns `R_fixed` at its fixed layers/budgets and does not automatically explain `R_all`.

Do not use E to fit, choose heads/layers/seeds, calibrate, change thresholds/exclusions, search comparators or redesign procedures. Surprising results change the claim, not the procedure. A discovered bug requires documented review and consistent correction while preserving original results and failed attempts. No optional experimental expansion follows this candidate by default.

## Resource decision and available local accounting

The proposed `limits` use exactly the accepted keys:

```json
{"model_seconds": 7200, "total_seconds": 18000, "setup_phase_seconds": 1800}
```

Keep the existing **10,800-second CPU-job cap** separately; no unsupported CPU key is added to `limits`. The 18,000-second cap is cumulative extraction-worker accounting under the existing contract: frozen carried model-active charges plus new E worker wall. It is a stopping cap, not a runtime forecast or a limit on total project elapsed time, transfer time or CPU elapsed time. Historical setup wall remains in its original receipts and is not reclassified as GPU time.

| Model | Carried seconds | Remaining from 7,200 seconds |
|---|---:|---:|
| Qwen | 160.83801023702836 | 7039.161989762972 |
| Llama | 123.82646256202133 | 7076.173537437979 |

The seed total is the ordinary sum of the two frozen carried charges. Available local original/successor attempts, adoption and execution reports reconcile exactly; no intervening charge in this campaign was found in the inspected project worktrees and backup roots. The accounting receipt records metadata paths, hashes, duplicate-copy treatment and other historical campaigns. The older full-raw ledger and full-bank CPU worker are separate historical campaigns; preserve their charges rather than relabel them into this frozen carry-forward contract.

This is a bounded local reconciliation, not certification of remote state. No budget adoption occurs here. Recheck intervening receipts before later launch. A missing/inconsistent settlement or new charge blocks adoption pending review; never reset charges or silently rewrite immutable pins.

Actual E runtime readiness remains unmeasured. Later work still requires reviewed hash-bound final-lock and phase authorizations, accepted offline runtime/snapshots, token plans and independent token audit, CUDA-memory checks, real persistent storage allocation/fsync/readback and two-copy reservations, complete shard/accounting acceptance and durable verified transfer. RunPod is not contacted during this preparation.

## Validation and review delivery

Validation evidence is under `results/final_e_lock_candidate_20261004/`. The checked-in `validate_candidate.py` verifies structure, producer ancestry, all six immutable hashes, original-draft/config preservation, registry/sample/contrast identities, the interpretation contract and rejection of the actual candidate for every declared E phase. It never changes the real candidate to FINAL. The existing positive authorization tests use synthetic fixtures only.

Run the candidate check from this worktree with the reviewed interpreter:

```bash
PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/opt/python@3.11/bin/python3.11 \
  results/final_e_lock_candidate_20261004/validate_candidate.py
```

The same validator runs the existing `tests.test_final_e_locked_path` and `tests.test_final_e_review` suites serially under `/tmp/truth-probing-cpu-exclusive.lock`, with one numerical thread. It writes the validation receipt and focused-test log. Only synthetic fixtures and metadata are used; no models, E activations/predictions/metrics, completed fits, full saved-D reproduction or wording diagnosis are rerun.

Review the diff against the exact base:

```bash
git diff e8f087e53b96b2172cfcde03e3a451e6a65bb0fe..HEAD -- \
  config/final_e/lock_candidate_20261004.json \
  docs/final_e/SCIENTIFIC_LOCK_CANDIDATE_20261004.md \
  results/final_e_lock_candidate_20261004/
```

Original worktree HEADs/statuses and pre-existing untracked-file bytes are checked again before publication. Publication is on the isolated branch with a normal non-force push and verified remote SHA; no merge is performed. This delivery stops for review. Remaining decisions are exact-lock approval, actual reviewer/instruction evidence, `claims_finalized`, `scientific_lock_approved`, FINAL status, reviewed delivery identity, accounting adoption and separate hash-bound phase authorizations/runtime checks. None is supplied by this candidate.
