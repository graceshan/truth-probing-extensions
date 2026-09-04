# TF/FT surface-position check

In `data/r1_quadruples.csv`, `labelA`, `labelB`, and `cell` describe canonical
`conjunctA` and `conjunctB` before surface ordering. This follows
`scripts/01_generate_r1_r2_datasets.py::gen_binary`: A and B are selected first, `_row`
stores their canonical texts and labels, and only `first, second` are swapped when
`ordering == "BA"`. Therefore BA reverses surface position relative to canonical A/B.

The existing canonical TF-vs-FT AUROC is not a valid first-surface-vs-second-surface test:
both canonical classes contain both surface configurations because AB and BA are balanced.

Exact mapping:

| Cell | Ordering | Surface class |
|---|---|---|
| TF | AB | FIRST_TRUE_SECOND_FALSE |
| TF | BA | FIRST_FALSE_SECOND_TRUE |
| FT | AB | FIRST_FALSE_SECOND_TRUE |
| FT | BA | FIRST_TRUE_SECOND_FALSE |

AUROC is oriented as the probability that FIRST_TRUE_SECOND_FALSE receives a higher frozen
atomic-probe score than FIRST_FALSE_SECOND_TRUE. The original 100-pair/1,600-row held-out
files were used without resplitting.

| model | connective | comparison | n_first_true_second_false | mean_first_true_second_false | sd_first_true_second_false | n_first_false_second_true | mean_first_false_second_true | sd_first_false_second_true | auroc |
|---|---|---|---|---|---|---|---|---|---|
| Qwen2.5-7B-Instruct | AND | FIRST_TRUE_SECOND_FALSE_vs_FIRST_FALSE_SECOND_TRUE | 200 | 0.5296245611958957 | 4.341321312502242 | 200 | 3.2770618589662335 | 4.338505263087924 | 0.34795 |
| Qwen2.5-7B-Instruct | OR | FIRST_TRUE_SECOND_FALSE_vs_FIRST_FALSE_SECOND_TRUE | 200 | -0.3933605040957777 | 4.715379361913477 | 200 | 0.9753391194151024 | 3.8118457134834225 | 0.4338 |
| Qwen3-8B | AND | FIRST_TRUE_SECOND_FALSE_vs_FIRST_FALSE_SECOND_TRUE | 200 | -2.462079231472024 | 6.071381125650979 | 200 | 1.919034649596664 | 4.0350242621351535 | 0.293625 |
| Qwen3-8B | OR | FIRST_TRUE_SECOND_FALSE_vs_FIRST_FALSE_SECOND_TRUE | 200 | -1.4153638873185719 | 5.853825189844518 | 200 | 0.23767008506918985 | 4.610752281025014 | 0.42362500000000003 |
