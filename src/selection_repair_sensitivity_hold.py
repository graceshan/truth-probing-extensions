"""Fail closed after the source-correction notification; synthetic helpers remain usable."""
from pathlib import Path
import hashlib

HOLD_PATH = Path('config/clean_protocol/canonical_sensitivity_correction_hold_20261001.json')
HOLD_SHA256 = 'a7925e697049c315015d21fb9e93905e2d0b158162c8cbce690554b45d4fe7b7'


def require_current_correction(root):
    """This v2 runner cannot be resumed by editing/deleting a status flag.

    A forthcoming reviewed successor needs a new versioned binding and entry
    point. The pinned hold remains historical evidence, never an override knob.
    """
    path = Path(root) / HOLD_PATH
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != HOLD_SHA256:
        raise ValueError('Missing or changed correction hold; fitting remains blocked')
    raise ValueError('Pending reviewed source correction: inventors:23, inventors:157 '
                     'and paired negations require quarantine; v2 fitting is blocked')
