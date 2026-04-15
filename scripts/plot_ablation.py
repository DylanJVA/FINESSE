"""
Generate all paper figures from an ablation parquet file.

Usage:
    python scripts/plot_ablation.py [ablation.parquet] [--out figures/]

The parquet file is produced by:
    python scripts/run_ablation.py --seeds 20 --trials 5 --out ablation.parquet
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use('Agg')

from finesse.plots import (
    fig_overview,
    fig_swap_bars,
    fig_lf_bars,
    fig_scatter,
    fig_contributions,
    fig_pareto,
    fig_depth_bars,
)


def save(fig, path: str) -> None:
    fig.savefig(path, bbox_inches='tight')
    png = path.replace('.pdf', '.png')
    fig.savefig(png, dpi=150, bbox_inches='tight')
    print(f'  saved {path}  +  {png}')
    import matplotlib.pyplot as plt
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parquet', nargs='?', default='ablation.parquet')
    parser.add_argument('--out', default='figures')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = pd.read_parquet(args.parquet)
    n_circs  = df['circuit'].nunique()
    n_seeds  = df['seed'].nunique()
    configs  = df['config'].unique().tolist()
    print(f'Loaded {args.parquet}: {n_circs} circuits, {n_seeds} seeds, '
          f'{len(configs)} configs: {configs}')

    print('\nGenerating figures...')

    save(fig_overview(df),
         os.path.join(args.out, 'ablation_overview.pdf'))

    save(fig_swap_bars(df),
         os.path.join(args.out, 'ablation_swaps.pdf'))

    save(fig_depth_bars(df),
         os.path.join(args.out, 'ablation_depth.pdf'))

    save(fig_lf_bars(df),
         os.path.join(args.out, 'ablation_lf_cost.pdf'))

    save(fig_scatter(df),
         os.path.join(args.out, 'ablation_scatter.pdf'))

    save(fig_contributions(df),
         os.path.join(args.out, 'ablation_contributions.pdf'))

    save(fig_pareto(df),
         os.path.join(args.out, 'ablation_pareto.pdf'))

    print(f'\nAll figures saved to {args.out}/')


if __name__ == '__main__':
    main()
