"""Optimization audit only: no test option and no selection-artifact writes."""

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from src.atomic_convergence_audit import audit_model


def main():
    config = json.loads((ROOT / "config/clean_protocol/atomic_probes.json").read_text())
    for key in ("qwen25_7b", "qwen3_8b"):
        def progress(row):
            print(f"{key}: layer={row['layer']} C={row['C']:g}, converged={row['refit_converged']}, "
                  f"n_iter={row['refit_n_iter']}, validation AUROC="
                  f"{row['original_validation_auroc']:.9f} -> {row['refit_validation_auroc']:.9f}", flush=True)
        _, summary = audit_model(ROOT, config, key, progress)
        print(json.dumps({"model": key, **summary["selection_comparison"],
                          "test_decision_scores_computed": 0}, indent=2), flush=True)


if __name__ == "__main__":
    main()
