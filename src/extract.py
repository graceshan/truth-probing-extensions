"""Model loading and activation extraction (Hugging Face transformers).

Runs on the GPU pod for the real Qwen2.5-7B-Instruct pass; the same code drives
the CPU dry-run against a tiny stand-in model. Follows the CLAUDE.md checkpoint
contract:

  - hidden state at the LAST token, ALL layers, one row per statement
  - acts array shape [n_statements, n_layers, d_model], saved as np.float16 to
    acts/{dataset}.npy
  - a parallel acts/{dataset}.csv with columns statement,label (1 = true) in the
    SAME row order as the array's first axis, so a checkpoint is self-describing

Design decisions (documented because they change what the numbers mean):

  - `output_hidden_states=True`, not TransformerLens run_with_cache. HF returns
    hidden_states as a tuple of length n_layers+1: index 0 is the embedding
    output, indices 1..n_layers are the block (residual-stream) outputs. We drop
    index 0 and keep the n_layers block outputs, so the layer axis equals
    model.config.num_hidden_layers and matches the resid_post-per-layer
    semantics of the prior 1.5B extraction.
  - RAW statement text, no chat template. Probes read raw-statement activations
    (README "prompt-regime caveat"); behavioral/Figure-1 lines are the only
    place a chat template belongs, and that is a separate pass.
  - LAST REAL token, located via the attention mask (index sum(mask)-1), not a
    bare [-1]. With batch_size=1 there is no padding so this equals [-1]; the
    mask-based index is what keeps it correct if anyone ever batches with padding.
  - One statement per forward pass (batch_size=1) by default: sidesteps the
    left-padding / last-token footgun entirely and keeps peak memory low. The
    model is still loaded exactly once and reused across every statement.
"""

import os

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Primary model for the real run (see CLAUDE.md). NOT loaded by default -- the
# caller passes an explicit model_name, so the CPU dry-run can hand in a tiny
# stand-in without this ever pulling Qwen weights.
HF_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_ACTS_DIR = "acts"


def load_model(model_name: str, device: str | None = None, dtype=None):
    """Load a causal LM + tokenizer once, in eval mode, ready for extraction.

    dtype defaults to bfloat16 on CUDA (the real 7B run) and float32 on CPU
    (the dry-run); pass it explicitly to override. Returns (model, tokenizer).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def extract_acts(model, tokenizer, statements, batch_size: int = 1) -> np.ndarray:
    """Last-token hidden state, every layer, for each raw statement.

    Returns a float16 array of shape [len(statements), n_layers, d_model], where
    n_layers == model.config.num_hidden_layers (embedding layer dropped).
    """
    device = next(model.parameters()).device
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    acts = np.zeros((len(statements), n_layers, d_model), dtype=np.float16)

    with torch.inference_mode():
        for start in range(0, len(statements), batch_size):
            batch = statements[start : start + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", padding=batch_size > 1
            ).to(device)
            out = model(**enc, output_hidden_states=True)
            # hidden_states: tuple len n_layers+1; drop embedding (index 0).
            hs = out.hidden_states[1:]  # each [batch, seq, d_model]
            # Last REAL token per row, from the attention mask.
            last_idx = enc["attention_mask"].sum(dim=1) - 1  # [batch]
            for b in range(len(batch)):
                li = int(last_idx[b])
                # stack layers -> [n_layers, d_model] for this statement
                vec = torch.stack([h[b, li, :] for h in hs], dim=0)
                acts[start + b] = vec.float().cpu().numpy().astype(np.float16)
    return acts


def save_checkpoint(
    acts: np.ndarray, df: pd.DataFrame, dataset: str, acts_dir: str = DEFAULT_ACTS_DIR
) -> tuple[str, str]:
    """Write acts/{dataset}.npy (float16) and the parallel acts/{dataset}.csv.

    `df` must be the SAME dataframe (same row order) whose `statement` column was
    passed to extract_acts, so CSV row i describes array row i. Columns written
    are exactly statement,label (label 1 = true).
    """
    assert len(df) == acts.shape[0], (
        f"{len(df)} rows in df but {acts.shape[0]} rows in acts -- order/count "
        "would not line up"
    )
    assert acts.dtype == np.float16, f"acts must be float16, got {acts.dtype}"
    os.makedirs(acts_dir, exist_ok=True)
    npy_path = os.path.join(acts_dir, f"{dataset}.npy")
    csv_path = os.path.join(acts_dir, f"{dataset}.csv")
    np.save(npy_path, acts)
    df[["statement", "label"]].to_csv(csv_path, index=False)
    return npy_path, csv_path

def save_checkpoint_full(
    acts: np.ndarray, df: pd.DataFrame, dataset: str, acts_dir: str = DEFAULT_ACTS_DIR
) -> tuple[str, str]:
    """Like save_checkpoint but preserves ALL metadata columns, row-aligned.

    Used for compound datasets (R1 quadruples, XOR) whose CSVs carry per-row
    metadata (topic, conjunctA/B, labelA/B, cell, connective, ordering) that the
    downstream regression needs. Extraction convention is unchanged: the same
    extract_acts feeds df['statement'] in order, so array row i == metadata row i.
    """
    assert len(df) == acts.shape[0], (
        f"{len(df)} rows in df but {acts.shape[0]} rows in acts -- order/count "
        "would not line up"
    )
    assert acts.dtype == np.float16, f"acts must be float16, got {acts.dtype}"
    assert "statement" in df.columns, "df must have a 'statement' column"
    os.makedirs(acts_dir, exist_ok=True)
    npy_path = os.path.join(acts_dir, f"{dataset}.npy")
    csv_path = os.path.join(acts_dir, f"{dataset}.csv")
    np.save(npy_path, acts)
    df.to_csv(csv_path, index=False)  # full metadata, not a two-column projection
    return npy_path, csv_path

def extract_and_save(
    model,
    tokenizer,
    df: pd.DataFrame,
    dataset: str,
    acts_dir: str = DEFAULT_ACTS_DIR,
    batch_size: int = 1,
) -> tuple[str, str]:
    """Convenience: extract activations for df['statement'] and checkpoint them.

    df must have columns `statement` and `label` (1 = true). Returns the written
    (npy_path, csv_path).
    """
    acts = extract_acts(model, tokenizer, df["statement"].tolist(), batch_size=batch_size)
    return save_checkpoint(acts, df, dataset, acts_dir=acts_dir)


def extract_and_save_full(
    model, tokenizer, df, dataset, acts_dir=DEFAULT_ACTS_DIR, batch_size=1
):
    """Like extract_and_save but preserves ALL metadata columns (for compound
    datasets). Extraction convention is identical to extract_and_save — same
    extract_acts — so activations match the atomic run; only the sidecar differs."""
    acts = extract_acts(model, tokenizer, df["statement"].tolist(), batch_size=batch_size)
    return save_checkpoint_full(acts, df, dataset, acts_dir=acts_dir)
