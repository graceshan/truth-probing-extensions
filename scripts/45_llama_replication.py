#!/usr/bin/env python3
"""Fixed Llama cross-family replication stages; no data/scientific CLI overrides."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['pin', 'atomic', 'select', 'transfer', 'analysis'])
    parser.add_argument('--mode', required=True, choices=['pin', 'plan', 'smoke', 'extract', 'check-only', 'fit',
                                                        'preflight', 'freeze-spec', 'score', 'evaluate'])
    parser.add_argument('--revision', help='Required only for pin: exact 40-character HF commit SHA; never main')
    args = parser.parse_args()
    allowed = dict(pin=['pin'], atomic=['plan', 'smoke', 'extract'], select=['check-only', 'fit'],
                   transfer=['plan', 'smoke', 'extract'], analysis=['preflight', 'freeze-spec', 'score', 'evaluate'])
    if args.mode not in allowed[args.stage] or bool(args.revision) != (args.stage == 'pin'):
        parser.error('invalid stage/mode or revision outside pin stage')
    if args.stage == 'pin':
        from src.llama_replication_extraction import pin_revision
        result = pin_revision(args.revision)
    elif args.stage in ['atomic', 'transfer']:
        from src.llama_replication_extraction import run
        result = run(args.stage, args.mode)
    elif args.stage == 'select':
        from src.llama_replication_probes import select
        result = select(check_only=args.mode == 'check-only')
    elif args.mode == 'evaluate':
        from src.llama_replication_evaluation import evaluate
        result = evaluate()
    else:
        from src.llama_replication_transfer import preflight, freeze, score
        result = {'preflight': preflight, 'freeze-spec': freeze, 'score': score}[args.mode]()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
