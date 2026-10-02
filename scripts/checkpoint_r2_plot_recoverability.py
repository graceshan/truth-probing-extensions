"""Render audited CSV curves on a full AUROC axis; no fitting or selection.

Usage: python scripts/checkpoint_r2_plot_recoverability.py TABLE.csv OUTPUT.svg
"""
import argparse
import csv
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('refuse figure overwrite')
    with args.table.open() as stream:
        rows = list(csv.DictReader(stream))
    endpoints = ('atomic_auroc', 'AND_auroc', 'OR_auroc', 'OR_mixed_vs_FF_auroc')
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for row, model in enumerate(('qwen', 'llama')):
        for col, metric in enumerate(endpoints):
            ax = axes[row, col]
            for family, color in [('R0', '#0072B2'), ('C_clean', '#D55E00')]:
                for scope, style in [('pooled', '-'), ('topic_macro', '--')]:
                    subset = sorted([r for r in rows if r['model'] == model and r['family'] == family
                                     and r['metric'] == metric and r['scope'] == scope],
                                    key=lambda r: int(r['saved_layer']))
                    ax.plot([int(r['saved_layer']) for r in subset],
                            [float(r['point']) if r['point'] else float('nan') for r in subset],
                            style, color=color, label=family + ' ' + scope)
            reference = next(r for r in rows if r['model'] == model and r['family'] == 'reduced_LR'
                             and r['metric'] == metric and r['scope'] == 'pooled')
            ax.axhline(float(reference['point']), color='gray', linewidth=1, label='fresh reduced LR pooled')
            ax.set(title=model + ' ' + metric, xlabel='Saved layer', ylim=(0, 1.01))
            ax.grid(alpha=.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Fresh bare-D recoverability — pooled and equal-topic macro; full AUROC axis')
    fig.tight_layout()
    fig.savefig(args.output)
    plt.close(fig)


if __name__ == '__main__':
    main()
