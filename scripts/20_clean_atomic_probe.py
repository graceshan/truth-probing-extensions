"""Shared clean atomic selection; final test requires an explicit separate mode."""

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

from src.clean_atomic_probes import AtomicData, evaluate_frozen_test, fit_and_save, json_bytes
from src.clean_compounds import check
from src.entity_partitions import save_outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/clean_protocol/atomic_probes.json")
    parser.add_argument("--model", choices=("qwen25_7b", "qwen3_8b", "both"), default="both")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-only", action="store_true", help="verify structure without loading activation values")
    mode.add_argument("--evaluate-test", action="store_true", help="evaluate existing frozen probe; never fit or select")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    models = ("qwen25_7b", "qwen3_8b") if args.model == "both" else (args.model,)
    assignment_hash = None
    for key in models:
        directory = (ROOT / config["models"][key]["results_dir"]).resolve()
        check(directory.is_relative_to(ROOT / "results/clean_protocol"), "outputs must remain in results/clean_protocol")
        # Fail before authorizing test data if no frozen selection exists.
        if args.evaluate_test:
            check((directory / "selection.json").exists(), "final evaluation requires an existing frozen selection")
        data = AtomicData(ROOT, config, key, authorize_test=args.evaluate_test)
        current_hash = data.structural["assignment_sha256"]
        check(assignment_hash is None or current_hash == assignment_hash, "models disagree on manifest assignments")
        assignment_hash = current_hash
        if args.check_only:
            save_outputs(directory, {"structural_checks.json": json_bytes(data.structural)})
            print(json.dumps({"model": key, "counts": data.counts, "test_scores_computed": 0}, indent=2))
        elif args.evaluate_test:
            print(json.dumps(evaluate_frozen_test(data, directory), indent=2))
        else:
            def progress(layer, rows):
                print(f"{key}: layer {layer}, {len(rows)} C values converged "
                      f"({sum(row['retry_needed'] for row in rows)} retries); validation AUROC range "
                      f"{min(r['validation_overall_auroc'] for r in rows):.6f}–"
                      f"{max(r['validation_overall_auroc'] for r in rows):.6f}", flush=True)
            result = fit_and_save(data, directory, progress)
            print(json.dumps({k: result[k] for k in ("model", "selected_layer", "selected_C",
                                                     "selected_validation_metrics", "retried_configurations",
                                                     "all_final_fits_converged", "test_evaluated",
                                                     "test_decision_scores_computed")}, indent=2))


if __name__ == "__main__":
    main()
