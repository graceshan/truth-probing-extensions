# Project: Truth-Probing Extensions

Does a linear truth probe read a compound's truth value, or a weighted average of
its conjuncts' scores plus a connective offset? See `README.md` for the four arms
(R3 audit, Figure 1, R1 decomposition, R2 XOR) and the research plan. This file
documents the working conventions; the README is the plan of record.

- **Model:** Qwen2.5-7B-Instruct (primary), Llama-3.1-8B-Instruct (reserve).
- **Status:** no model is loaded and no GPU torch build is installed yet.
  Extraction runs later on Runpod; local work is CPU-only (analysis from saved
  `acts/*.npy`). Do not install a CUDA torch wheel or download model weights
  without being asked.

## Compute workflow — tmux + ipython persistent kernel

The 7B model is expensive to load, so it lives in one long-running IPython kernel
inside a tmux session; every command is sent to that same kernel.

- **One tmux session, one IPython kernel.** Start the GPU box's session
  (e.g. `tmux new -s tiu`), launch `ipython` in it, and run everything there.
  Reattach with `tmux attach -t tiu`; never open a second kernel that reloads the
  model in parallel.
- **Load the model exactly once**, at the top of the session, into a module-level
  variable that persists across cells. Reuse it for the whole run.
- **Never restart the kernel or kill the tmux session without asking first.**
  A restart evicts the 7B weights from GPU memory and costs a full reload; if a
  restart seems necessary (OOM, wedged state), surface it and wait for the go-ahead.
- **Long extractions run detached** in the tmux session so a dropped SSH
  connection doesn't kill them.

## Plots

- **Save every plot to a PNG under `figures/`** with `plt.savefig(path, dpi=150,
  bbox_inches="tight")` — the session is headless, so nothing renders interactively.
  Never rely on `plt.show()`. Name figures for the arm/quantity they support.

## Activation checkpoints

- **Checkpoint activations to `acts/{dataset}.npy`**, all layers, **fp16**
  (`np.float16`) — shape `[n_statements, n_layers, d_model]`, hidden state at the
  **last token**, one row per statement.
- **Write a parallel CSV `acts/{dataset}.csv`** alongside each `.npy` with columns
  `statement,label` (label 1 = true), in the **same row order** as the array's
  first axis, so a checkpoint is self-describing without re-running extraction.
- `acts/*.npy` are git-ignored (large, regenerable); the parallel CSVs are
  committed so labels/order survive in git.
- Re-extraction is a GPU job — treat existing checkpoints as the source of truth
  for CPU analysis rather than regenerating them.

## Probe / analysis conventions

- **Labels** are `{0, 1}` with 1 = true. Orient every direction so positive = true;
  sklearn's logistic decision function is already oriented this way when
  `classes_ == [0, 1]`.
- **Atom scores and compound scores in any single formula come from the same probe
  on the same scale** (see README §"Which probe does what"). Name the probe behind
  every reported number.
