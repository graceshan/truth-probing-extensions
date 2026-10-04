# Full atomic-training comparator: executed and independently audited

Restoring the full atomic-training procedure **did not close the frozen reduced-P repair advantage** on the bare OR mixed-positive versus FF-negative boundary. The paired mean repair-minus-`S_full` gap is **0.122503 for Qwen** (95% interval 0.101681–0.140490) and **0.149766 for Llama** (0.125530–0.172026). Both lower endpoints exceed the predeclared 0.02 practical margin. The fixed five-topic macro gaps also remain positive: 0.125057 and 0.167244. Full-versus-original `S_all` changes are small, with intervals crossing zero on this endpoint.

This uses the exact CPU producer `b392ecec5f2136dbf6c3a4c41a7652aa7f6c49c8` in a clean detached execution checkout. All **600 candidates are valid** (280 Qwen, 320 Llama); failed, incomplete and unrun counts are zero. One worker attempt completed. No repair refitting, behavior/chat, new language-model inference, optional experiments, Pod contact or E access occurred.

## Prescribed paired contrasts

Values are AUROC differences with 95% pointwise, conditional paired intervals. Each fitted head is evaluated separately; seed-11/23/37 metrics and differences are averaged inside each draw. Scores from different heads are never concatenated or averaged. Positive `R_all` minus comparator values favor frozen repair. `S_all` means the original reduced-training selector.

| Model | Contrast | Scope | Difference [95% interval] |
| --- | --- | --- | --- |
| qwen | R_all − S_full | pooled | 0.1225 [0.1017, 0.1405] |
| qwen | R_all − S_full | topic_macro | 0.1251 [0.0979, 0.1540] |
| qwen | S_full − S_atom_full | pooled | 0.0696 [0.0509, 0.0901] |
| qwen | S_full − S_atom_full | topic_macro | 0.0405 [0.0131, 0.0665] |
| qwen | R_all − S_atom_full | pooled | 0.1921 [0.1704, 0.2118] |
| qwen | R_all − S_atom_full | topic_macro | 0.1655 [0.1333, 0.1951] |
| qwen | S_full − S_all | pooled | -0.0002 [-0.0072, 0.0073] |
| qwen | S_full − S_all | topic_macro | 0.0024 [-0.0122, 0.0155] |
| llama | R_all − S_full | pooled | 0.1498 [0.1255, 0.1720] |
| llama | R_all − S_full | topic_macro | 0.1672 [0.1333, 0.2040] |
| llama | S_full − S_atom_full | pooled | 0.2691 [0.2301, 0.3168] |
| llama | S_full − S_atom_full | topic_macro | 0.1143 [0.0564, 0.1689] |
| llama | R_all − S_atom_full | pooled | 0.4189 [0.3910, 0.4540] |
| llama | R_all − S_atom_full | topic_macro | 0.2815 [0.2472, 0.3121] |
| llama | S_full − S_all | pooled | 0.0035 [-0.0032, 0.0097] |
| llama | S_full − S_all | topic_macro | 0.0048 [-0.0012, 0.0111] |

Complete original-precision tables include all seeds, pooled/fixed-five macro/topic endpoints, atomic AUROC, AND AUROC, AND TT-versus-mixed, OR AUROC and OR mixed-versus-FF. See `metrics.csv` (5,544 rows), `contrasts.csv` (6,160 rows), `topic-contrasts.csv`, `wording-contrasts.csv` and `retention.csv`. No endpoint was omitted because of failure.

## Frozen selected heads and bank references

| Model | Seed | S_full | S_atom_full |
| --- | --- | --- | --- |
| llama | 11 | llama/full_train/L09/ttpd | llama/full_train/L20/r0 |
| llama | 23 | llama/full_train/L09/l2_logistic/C=1 | llama/full_train/L20/r0 |
| llama | 37 | llama/full_train/L09/ttpd | llama/full_train/L20/r0 |
| qwen | 11 | qwen/full_train/L18/l2_logistic/C=10 | qwen/full_train/L17/l2_logistic/C=0.1 |
| qwen | 23 | qwen/full_train/L18/difference_of_means | qwen/full_train/L17/l2_logistic/C=0.1 |
| qwen | 37 | qwen/full_train/L18/l2_logistic/C=10 | qwen/full_train/L17/l2_logistic/C=0.1 |

The atomic-only head was selected once per model. Selectors locked before wording evaluation. The common eligibility reference remains the original reduced-training LR atomic-D score (Qwen 0.9998320548672843; Llama 1.0), with the existing 0.005 margin. Full-training canonical LR is a separate reporting arm, not a replacement eligibility reference. Eligible candidate counts are 138/280 Qwen and 181/320 Llama.

| Model | Arm | Bare boundary pooled AUROC |
| --- | --- | --- |
| qwen | R_all | 0.9726 |
| qwen | S_all | 0.8503 |
| qwen | S_atom_full | 0.7805 |
| qwen | S_full | 0.8501 |
| qwen | bank_oracle | 0.8600 |
| qwen | full_bank_eligible | 0.8597 |
| qwen | full_bank_unrestricted | 0.8597 |
| qwen | full_canonical_LR | 0.7699 |
| qwen | reduced_LR | 0.7783 |
| llama | R_all | 0.9905 |
| llama | S_all | 0.8373 |
| llama | S_atom_full | 0.5716 |
| llama | S_full | 0.8407 |
| llama | bank_oracle | 0.8481 |
| llama | full_bank_eligible | 0.8495 |
| llama | full_bank_unrestricted | 0.8495 |
| llama | full_canonical_LR | 0.6430 |
| llama | reduced_LR | 0.6546 |

The eligible and unrestricted primary full-bank maxima coincide: Qwen layer 18 difference-of-means, 0.859727; Llama layer 9 raw LR C=0.1, 0.849453. `bank-maxima.csv` contains 224 eligible/unrestricted full-training and original-P15 endpoint references, with exact maximizing IDs by topic and endpoint. These endpoint maxima are descriptive. Conditional intervals for fixed primary winners are in `metrics.csv`; they do not account for winner reselection or provide simultaneous coverage across the bank.

## Atomic/AND retention

Atomic retention uses the unchanged 0.005 loss margin; AND uses 0.02. “Supported” requires the paired lower endpoint to remain above the negative margin. Tables retain seed-specific and all-topic evidence.

| Model | Arm | Endpoint | Scope | Difference to reduced LR [95% interval] | Point retained | Interval supported |
| --- | --- | --- | --- | --- | --- | --- |
| qwen | S_full | atomic_auroc | pooled | -0.0005 [-0.0009, -0.0001] | True | True |
| qwen | S_full | atomic_auroc | topic_macro | -0.0016 [-0.0034, -0.0003] | True | True |
| qwen | S_full | AND_auroc | pooled | 0.0488 [0.0339, 0.0637] | True | True |
| qwen | S_full | AND_auroc | topic_macro | 0.0721 [0.0488, 0.0983] | True | True |
| qwen | S_atom_full | atomic_auroc | pooled | 0.0000 [0.0000, 0.0001] | True | True |
| qwen | S_atom_full | atomic_auroc | topic_macro | 0.0000 [0.0000, 0.0000] | True | True |
| qwen | S_atom_full | AND_auroc | pooled | 0.0054 [-0.0004, 0.0113] | True | True |
| qwen | S_atom_full | AND_auroc | topic_macro | 0.0160 [0.0049, 0.0299] | True | True |
| llama | S_full | atomic_auroc | pooled | -0.0008 [-0.0014, -0.0004] | True | True |
| llama | S_full | atomic_auroc | topic_macro | -0.0010 [-0.0037, -0.0000] | True | True |
| llama | S_full | AND_auroc | pooled | -0.0080 [-0.0180, 0.0014] | True | True |
| llama | S_full | AND_auroc | topic_macro | -0.0201 [-0.0341, -0.0092] | False | False |
| llama | S_atom_full | atomic_auroc | pooled | 0.0000 [0.0000, 0.0000] | True | True |
| llama | S_atom_full | atomic_auroc | topic_macro | 0.0000 [0.0000, 0.0000] | True | True |
| llama | S_atom_full | AND_auroc | pooled | -0.0291 [-0.0370, -0.0208] | False | False |
| llama | S_atom_full | AND_auroc | topic_macro | -0.0091 [-0.0159, -0.0030] | True | True |

Qwen `S_full` supports atomic and AND retention in pooled and macro summaries. Llama `S_full` supports atomic retention and pooled AND retention, but **macro AND retention fails at the point**: −0.020125 (interval −0.034077 to −0.009247), against the −0.02 margin. Llama `S_atom_full` fails pooled AND retention (−0.029077; interval −0.036961 to −0.020826), although its macro AND retention is supported. The Qwen primary bank maximum has unsupported macro atomic retention, and the Llama primary bank maximum has unsupported macro AND retention; full details are retained rather than treating a pooled gate as sufficient for every endpoint.

## Wording and topic results

The repair advantage remains for both existing OR wordings after selectors were locked. These are the registered OR mixed-versus-FF differences; the AND wording and every prescribed contrast are also in the compact tables.

| Model | Wording | Scope | R_all − S_full [95% interval] |
| --- | --- | --- | --- |
| qwen | or_at_least_one_v1 | pooled | 0.2737 [0.2538, 0.2964] |
| qwen | or_at_least_one_v1 | topic_macro | 0.2724 [0.2476, 0.3025] |
| qwen | or_explicit_or_both_v1 | pooled | 0.1188 [0.0941, 0.1415] |
| qwen | or_explicit_or_both_v1 | topic_macro | 0.1255 [0.0914, 0.1548] |
| llama | or_at_least_one_v1 | pooled | 0.2614 [0.2411, 0.2845] |
| llama | or_at_least_one_v1 | topic_macro | 0.2639 [0.2197, 0.3216] |
| llama | or_explicit_or_both_v1 | pooled | 0.1694 [0.1371, 0.2042] |
| llama | or_explicit_or_both_v1 | topic_macro | 0.1972 [0.1595, 0.2298] |

Bare per-topic boundary results are shown below. The positive fixed-five macro gaps establish that the aggregate difference persists within topics; this report does not attribute the remaining wording effect to a new mechanism or repeat the saved-score diagnosis.

| Model | Topic | R_all − S_full [95% interval] |
| --- | --- | --- |
| qwen | animal_class | 0.1132 [0.0200, 0.2205] |
| qwen | cities | 0.0406 [0.0277, 0.0566] |
| qwen | element_symb | 0.1597 [0.1056, 0.2236] |
| qwen | inventors | 0.0837 [0.0286, 0.1383] |
| qwen | sp_en_trans | 0.2280 [0.1734, 0.2872] |
| llama | animal_class | 0.2254 [0.1041, 0.4054] |
| llama | cities | 0.0811 [0.0559, 0.1115] |
| llama | element_symb | 0.1422 [0.0970, 0.1709] |
| llama | inventors | 0.2001 [0.1380, 0.2523] |
| llama | sp_en_trans | 0.1874 [0.1403, 0.2261] |

## Accepted inputs and preservation

The 3,040 admitted outer-training bindings, original 2,778 P15 bindings and exact 1,000-row balanced sampler are unchanged. Each model combines 2,909 reviewed existing fresh representations with its actual 131-text supplement. t_G/TTPD use the exact verified sampler; full-admitted fitted preprocessing, method objectives, raw LR C scaling, layer ranges and convergence rules remain frozen. A identities now occur in atomic training.

| Model | Extraction producer | Acceptance SHA-256 | Transfer SHA-256 |
| --- | --- | --- | --- |
| llama | 6865bfa43c52d9c5d7d3c96df90351ad8f895aea | 604972033fb21ad582f20307d346690b5c9d08fddca51a582becf30235bf988e | ac8df4a84c77c8a68ae26924d224ef4c3914b47440f3242b07dd566f9885adc0 |
| qwen | 9b0ac1d84b35d7faf2148d43664569aa867cba52 | 96c5d1f5c064e79a58b6704763d6dbca77e4f352d48b13ae3658d9ddc8ba539d | 88918f7fcd28b37a8a593a844b1c8ab10c11e940e4144e493dc4c05809729ebc |

CPU identity is separate from extraction provenance. Original representation extraction remains `7656a0eb68a5fd7ff7dac1c73efeab982d540cc2`. The approved October 3 Revision 2 document hash remains `f81c2209c03eacf5c6fe5e14729d68069035ea3a08de3d2b9eb273b070c4cae2`.

Actual Mac 1 handoff: `/Users/apple/projects/truth-probing-extensions/.worktrees/checkpoint-next-raw-only-execution-6865bfa/mac2-handoff.json`, hash `0930955d49eb83c37470769a16419a0a31f134023b26cc0c13a8b03cbf9b344a`. The run used a distinct verified unpacking of its Llama transfer because the archive arrived before Mac 1 published that descriptor; all 557 Llama files are byte-identical to the actual handoff. All 574 original Qwen files and both source/execution handoffs remain hash-identical after fitting and audit. Adoption SHA-256 `388029a7b35e6ee5ac5a4dc16eef34cd7ebac02152263e9a8c642ebaa78b7df9` links the original and successor accounting and does not replace either extraction identity.

Accepted extraction charges are distinct from CPU worker charges: Qwen original settled campaign 160.838010 seconds (including its failed preflight); Llama original 41.157094 plus successor 82.669369 = 123.826463 seconds. No allowance was reset.

Full receipt and hash-bound descriptor: `downstream-input-acceptance.json`, `handoffs.json`, `mac1-handoff-binding.json`, `llama-transfer.json`, and `transfer-verification.json`. Authorization binds descriptor SHA-256 `e0520d308dbe4928716492df3e360cbb10b635fab7f0c8b51d6bcb8e308ba935`, accepted downstream receipt `8f772ea994f54aaaa24fdbc0b71c06890fc5a40f7210e4e56b427926d3e76a50`, preparation `c43c567b77ad1c1defed66be5d1cb6d2ca16eb5350f2f052f4d4627cca6c0c87`, and configuration `d677711d44602ba722e51a49787ba9ce295750a39a4c999d9eccd9ad0cb166d5`.

Original worktrees/deliveries, scientific files, source arrays, and the prior full-bank preparation review ZIP are preserved. This separate report branch starts at adapter delivery `1d629a84aee5217e8f422b0bc80ee0debacdf3c8` and changes only reports and receipts; the execution checkout remains detached and clean at the exact producer. All arrays and full fitting records remain outside Git at `/Users/apple/truth-probing-backups/20261001T164224Z/full-bank-execution-20261004T002800Z`. `external-output-manifest.json` binds their hashes and sizes; `candidate-statuses.csv` accounts for every candidate and its fitting/parameter/score hashes.

## Independent audit, uncertainty and resources

Independent audit **passed 1,420,047 checks with maximum absolute difference 0.0**. It independently reconstructed all 600 atomic and allocated-boundary candidate points, every locked full-head endpoint and all 2,000 saved draws, mean metrics, contrast points/intervals and undefined accounting. It performed no refitting. This is an independent calculation on the same Mac, not second-machine replication. See `independent-audit.json`.

The original paired entity schedules remain hash-bound in `bootstrap-provenance.json`. Intervals are 95% pointwise percentile intervals conditional on fitted heads, fixed selectors and support. Metrics are averaged within draws across samples; no score pooling, schedule redraw, missing-topic dropping or imputation occurs. All 5,544 metric and 6,160 contrast rows have 2,000 valid and zero undefined draws. Conditional intervals do not incorporate training-set or selector uncertainty.

| Stage | Measured seconds | Peak RSS bytes |
| --- | --- | --- |
| Acceptance | 17.3292 | 911310848 |
| Preparation | 33.3955 | 1468956672 |
| Run | 1976.3408 | 4276142080 |
| Audit | 150.7392 | 3743842304 |

The worker’s internal elapsed time was 1940.4797 seconds. Its **single cumulative charged attempt** used 1956.0689 of 10,800 seconds, leaving 8843.9311 seconds. The cap applies to the cumulative worker, including its input reopening/fits/evaluation; it is not a runtime estimate. The outer run wall time includes launch acceptance and supervision, while worker charge and internal timing have their own recorded boundaries. Audit ran sequentially in the exclusive CPU slot and does not replenish the worker cap. macOS `getrusage` RSS units are bytes.

No producer gate, fit, cap or audit failed; no fit was retried. Operational readiness diagnostics are recorded honestly in `runtime-accounting.json`: initial sandboxed process inspection was denied; an overbroad metadata classifier initially flagged an old zero-resource exiting process record; and an authorization helper exited before writing while its readiness receipt was absent. Read-only inspection confirmed those records were not active jobs, and authorization was written only after all prerequisites passed. The metadata packaging helper also initially used the wrong folder serialization; it was corrected to resolve exact IDs from the actual fit receipts and verify completion pins. None changed a scientific gate, numerical rule or output.

## Interpretation and reproduction

This compares restored full atomic training against frozen reduced-P repair. `S_full` still uses compound labels for selection; `S_atom_full` uses the original atomic-only rule once per model. The measured nonclosure applies to this frozen grid and procedure. Even matching repair would not establish that compound supervision is unnecessary or that full-training repair adds nothing; full-training repair was not tested.

The exact commands, scoped authorization, preparation receipt, handoff hashes and immutable output identity are delivered alongside the tables. See `REPRODUCTION.md`. Reuse saved outputs for review; do not launch a duplicate campaign or overwrite the completed one.
