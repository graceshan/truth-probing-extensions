# R2 constituent development decision

Producer `2d28c0b2733a4433aa6ceb4de020bfb815c2f7db`, config `21e99762ffb43ded228d579c9ec8757aef7326c04e40e822aa1850b68ebe8bde`.
Complete raw panel: 240 initial + 12 final valid fits, no failed fits or warm retries. Independent audit passed.

Bare D supports strong marginal and conditional constituent ranking for both models. All three matched-layer comparisons' six macro lower bounds exceed .90 (minimum .9354 Qwen, .9851 Llama). Llama supports the .05 pointwise paired transfer-loss planning margin on all six endpoints in both directions. Qwen has two unresolved conditional endpoints: AND-to-OR FIRST given SECOND true, .0290 [.0082,.0577]; OR-to-AND SECOND given FIRST true, .0252 [.0085,.0552]. No simultaneous across-endpoint guarantee is claimed.

The strongest limiting transfer result is Qwen AND-to-OR explicit-or-both FIRST marginal: target-minus-source .0768 [.0593,.0921], a material deficit; transferred AUROC .8944 [.8768,.9168]. Wording also causes major fixed-decision/composition losses despite strong ranking. Llama explicit-or-both retains >=.90 macro points, but two transfer-loss bounds cross .05 and some absolute wording bounds cross .90. Full endpoint and topic records remain in six-endpoints.csv and six-endpoint-transfer-losses.csv. Poor fixed composition does not prove constituent information absent.

Decision: retain the bare-statement constituent availability/transfer result with the stated uncertainty. Do not claim wording-general useful transfer, calibrated compound competence, internal causal use or final-test generalization. This raw panel alone is not the whole checkpoint.

One next diagnostic is warranted: a separately specified saved-score study of FIRST/SECOND offset, scale and topic/cell/order interference across operator and wording, using existing locked heads. Proposed cap: one CPU process/thread and one focused hour; no fitting, new thresholds, sign flips, new supervision, model execution, RunPod or E. Stop after the predefined score-distribution/decision-discordance packet. Any source-only calibration fit needs a separate frozen design and authorization and must retain the original panel. Joint heads and intervention preparation are deferred because the present ambiguity is wording/decision robustness rather than absent bare-D transfer. No follow-up was launched.

Rebuild commands and exact provenance are in report.md, delivery.json and the implementation contract. Frozen science and prior artifacts remain preserved.
