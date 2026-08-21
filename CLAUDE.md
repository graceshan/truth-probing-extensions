# Project: Truth-Probing Extensions

Follow-on to the `truth-probing` project (sibling repo). Extends the compound-statement
analysis toward nested three-fact statements like `(A and B) or C`, which break the
conjunct-counting degeneracy that within-connective evaluation can't detect.

## Workflow / environment
- GPU activation extraction runs on Colab via `notebooks/extract_colab.ipynb`.
  Everything else runs locally on CPU from saved `.npy` files in `data/activations/`
  (gitignored). NEVER load the model locally.
- Extraction produces, per dataset: `<name>_acts.npy` and `<name>_labels.npy` in
  `data/activations/`.

## Core conventions (the copied `src/` relies on these)
- **Activations**: shape `[n_statements, n_layers, d_model]`, `resid_post`, LAST token.
- **SIGN CONVENTION**: all directions oriented so positive = true/honest side. With
  labels `{0, 1}` (1 = true), both the diff-in-means direction and sklearn's logistic
  decision function come out correctly oriented; `src/probes.py` asserts
  `classes_ == [0, 1]` to keep this guarantee.
- **Probes**: `LogisticRegression(max_iter=2000, C=0.1)`. Split indices once and reuse
  across every layer so accuracy is comparable layer-to-layer.
- **Directions** stored as unit vectors (`d / np.linalg.norm(d)`).
- **Datasets**: Geometry of Truth CSVs under `geometry-of-truth/datasets/`, columns
  `statement`, `label` (1 = true). Only the CSVs needed here are vendored (cities,
  neg_cities, sp_en_trans) plus `make_conj_disj.py`; the full GoT repo is not.
- **Model** (extraction only): Qwen2.5-1.5B base, HF handoff +
  `from_pretrained_no_processing`, fp16.

## Probe-fitting conventions, by use (carried from the source project)
Two conventions coexist deliberately — do not describe them as uniform:
- **Within-dataset accuracy/AUROC/control**: group-level train/test split (by city /
  Spanish word, via `src.data.group_key` + `src.probes.split_indices(..., groups=...)`),
  aggregated across seeds (mean + t-distribution 95% CI, `src.stats.seed_mean_ci`).
  Group-level splitting prevents the same entity's true/false pair landing on both
  sides of a split.
- **Cross-dataset transfer / compound analysis**: the source probe fits on ALL of the
  source dataset (`src.probes.fit_probe`, no held-out split), since the evaluation set
  is a different dataset. Uncertainty comes from evaluation-side bootstrap instead
  (resample the eval set, 1000 resamples, percentile interval), source probe held fixed.
- **Conjunct regression** (`scripts/06_conjunct_regression.py`) is the odd one out: it
  fits the cities probe via `train_layer_probe` on the standard 80/20 (row-level) split.
  Kept as-is from the source rather than "fixed" to match the transfer convention.

## Uncertainty helpers (`src/stats.py`)
- `seed_mean_ci`: t-distribution CI across a small number of seed-level point estimates.
- `percentile_bootstrap`: resample-and-recompute for ratios / differences / regression
  coefficients whose sampling distribution isn't analytic. The reported point estimate
  always comes from the original unresampled computation; the bootstrap only supplies
  (lo, hi).
- `assert_unchanged`: point-estimate regression check — adding a bootstrap must never
  move the original computation (atol 1e-6, above threading-induced last-bit noise).

## Figures
Saved at `dpi=150`. Never commit `.npy` activation files.
