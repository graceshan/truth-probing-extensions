#!/usr/bin/env python3
"""Metadata-only common checkpoint validation, with synthetic scores only.

Default is read-only. --receipt creates a new file exclusively; never overwrites.
Optional --original-worktree verifies the original local DOCX/untracked hashes.
"""
import argparse
import json
import platform
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.checkpoint_r2_common_v1 import (
    CONTRACT, CONTRACT_SHA256, FrozenBindings, contract, encoded, git,
    local_preservation, membership_receipt, preparation_inventory, preservation_receipt,
    select_atomic_synthetic, sha, shared_atomic_bank,
)
from src.checkpoint_r2_selection_v1 import AtomicCandidate, TOPICS, candidate_id


def validate(original_worktree=None):
    cfg = contract()
    memberships = membership_receipt(rebuild=True)
    frozen = FrozenBindings()
    synthetic = {}
    for operator in ('AND', 'OR'):
        b = frozen.block('constituent_source_validation', operator)
        scores = np.array([[r['surface_first_truth'], r['surface_second_truth']] for r in b['rows']], dtype=float)
        p = dict(metadata_sha256=b['metadata_sha256'], ordered_keys=b['ordered_keys'],
                 scores=scores, score_kind='synthetic_validation')
        result = frozen.select_synthetic(b, {0: p, 1: p}, source_operator=operator)
        synthetic[operator] = dict(status=result['status'], selected_layer=result['selected_layer'],
                                   rows=len(b['rows']), metadata_sha256=b['metadata_sha256'],
                                   minimum_tie_layers=result['minimum_tie_layers'],
                                   note='perfect synthetic labels-as-scores fixture; not research performance')
    for model in ('qwen', 'llama'):
        cs = [AtomicCandidate(candidate_id(model, layer, 'r0'), model, layer, 'r0', None,
                              True, .8, {t: .8 for t in TOPICS}) for layer in range(6)]
        bank = shared_atomic_bank(cs, model=model, reduced_lr_atomic_auc=.8)
        synthetic[model] = select_atomic_synthetic(cs, bank=bank, s_all_eligible_ids=bank['eligible_ids'])
    local = dict(status='not requested; no local untracked-file requirement')
    if original_worktree is not None:
        original = json.loads((ROOT / 'results/checkpoint_r2_preflight_v1/preservation.json').read_bytes())
        local = local_preservation(original_worktree, original['all14_sha256'])
        local['root_read_only'] = str(original_worktree)
    files = [CONTRACT, 'src/checkpoint_r2_common_v1.py', 'tests/test_checkpoint_r2_common_v1.py',
             'scripts/68_validate_checkpoint_r2_common.py']
    return dict(schema_version=1, status='passed_integration_metadata_and_synthetic_validation',
                generated_utc=datetime.now(timezone.utc).isoformat(),
                environment=dict(system=platform.system(), hostname=socket.gethostname(), repository=str(ROOT),
                                 branch=git(ROOT, 'branch', '--show-current'), validation_parent_head=git(ROOT, 'rev-parse', 'HEAD')),
                contract_sha256=CONTRACT_SHA256, source_commits=cfg['source_commits'],
                implementation_sha256={p: sha((ROOT / p).read_bytes()) for p in files},
                preservation=preservation_receipt(), optional_local_preservation=local,
                memberships=memberships, adapter=frozen.receipt(), preparation=preparation_inventory(),
                synthetic_only_results=synthetic, adoption_status=cfg['status'],
                bridge=cfg['bridge'], real_scores_read=False, fitting_performed=False,
                SSH_performed=False, E_scoring_enabled=False, production_execution_enabled=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-worktree', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    result = validate(args.original_worktree)
    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        with args.receipt.open('xb') as f:
            f.write(encoded(result))
    print(json.dumps(dict(status=result['status'], counts=result['memberships']['counts'],
                          source_commits=result['source_commits'], preservation=result['preservation'],
                          production_execution_enabled=False, representation_compatibility='unverified'), indent=2))


if __name__ == '__main__':
    main()
