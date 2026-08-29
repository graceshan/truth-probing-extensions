"""Loading Geometry of Truth statement datasets and saved activations."""

import numpy as np
import pandas as pd

DEFAULT_DATASETS_DIR = "geometry-of-truth/datasets"
DEFAULT_ACTS_DIR = "acts"

ENTITY_PATTERNS = {
    "cities": r"^The city of (.+?) is (?:not )?in ",
    "sp_en_trans": r"Spanish word '(.+?)'",
    "inventors": r"^(.+?) (?:lived|did not live) in ",
    "element_symb": r"^(.+?) (?:has|does not have) the symbol ",
    "animal_class": r"^The (.+?) is (?:not )?an? ",
}


def load_statements(name: str, datasets_dir: str = DEFAULT_DATASETS_DIR) -> pd.DataFrame:
    """Load a Geometry of Truth CSV. Columns are `statement` and `label` (1 = true)."""
    return pd.read_csv(f"{datasets_dir}/{name}.csv")


def entity_key_from_statements(topic: str, statements) -> np.ndarray:
    """Extract a topic's natural entity from exact atomic statement text."""
    if topic not in ENTITY_PATTERNS:
        raise ValueError(f"no entity pattern defined for topic {topic!r}")
    statements = pd.Series(statements, dtype="string")
    entities = statements.str.extract(ENTITY_PATTERNS[topic])[0]
    assert entities.notna().all(), (
        f"{topic}: some statements did not match the entity pattern"
    )
    return entities.to_numpy()


def group_key(name: str, datasets_dir: str = DEFAULT_DATASETS_DIR) -> np.ndarray:
    """Group identifier for group-level train/test splitting, so no group's
    rows land on both sides of a split (e.g. no city's true/false statement
    pair split across train and test).

    The affirmative and negated variants of each supported topic deliberately
    map to the same natural entity (city, Spanish word, inventor, element, or
    animal), so they can share one entity-disjoint split.
    """
    df = load_statements(name, datasets_dir=datasets_dir)
    topic = name.removeprefix("neg_")
    if topic == "cities" and "city" in df.columns:
        return df["city"].to_numpy()
    return entity_key_from_statements(topic, df["statement"])


def load_activations(
    name: str, acts_dir: str = DEFAULT_ACTS_DIR
) -> tuple[np.ndarray, np.ndarray]:
    """Load a saved checkpoint's activations + labels for a dataset.

    Reads the CLAUDE.md checkpoint pair written by src/extract.py: the fp16
    array acts/{name}.npy and the parallel acts/{name}.csv whose `label` column
    is in the same row order as the array's first axis.

    Returns (acts, labels) where acts has shape [n_statements, n_layers, d_model]
    (last-token hidden state, every layer) and labels is [n_statements] (1 = true).
    """
    acts = np.load(f"{acts_dir}/{name}.npy")
    labels = pd.read_csv(f"{acts_dir}/{name}.csv")["label"].to_numpy()
    assert len(labels) == acts.shape[0], (
        f"{name}: {len(labels)} CSV labels but {acts.shape[0]} activation rows"
    )
    return acts, labels
