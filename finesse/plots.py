# finesse/plots.py
#
# Publication-quality figure functions for the FINESSE ablation study.
#
# All functions accept a pandas DataFrame with the parquet schema produced by
# scripts/run_ablation.py / finesse.ablation.run_benchmark():
#
#   columns: circuit, n_logical, n_physical, config, label, seed,
#            swap_count, gate_depth, lf_cost, fidelity
#
# Every function returns the matplotlib Figure; the caller decides whether to
# save or display it.  No plt.show() or savefig() calls inside.

from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd

# ---------------------------------------------------------------------------
# Display constants
# ---------------------------------------------------------------------------

CONFIG_ORDER: list[str] = [
    'qiskit_sabre',
    'sabre',
    'lightsabre',
    'mirage_sabre',
    'mirage',
    'sabre_fid',
    'lightsabre_fid',
    'mirage_fid_sabre',
    'mirage_fid',
    'finesse',
]

# Color grouping logic:
#   grey  = external reference (Qiskit)
#   blue  = SABRE family (no fidelity, no mirrors)
#   green = MIRAGE family (no fidelity, with mirrors)
#   amber = fidelity-aware without mirrors
#   red   = fidelity-aware with mirrors (FINESSE lineage)
COLORS: dict[str, str] = {
    'qiskit_sabre':    '#888888',
    'sabre':           '#4477AA',
    'lightsabre':      '#66AADD',
    'mirage_sabre':    '#88BB44',
    'mirage':          '#55AA22',
    'sabre_fid':       '#DDAA33',
    'lightsabre_fid':  '#FFCC55',
    'mirage_fid_sabre':'#EE7733',
    'mirage_fid':      '#CC4422',
    'finesse':         '#AA1111',
}

HATCH: dict[str, str] = {
    'qiskit_sabre':    '///',
    'sabre':           '',
    'lightsabre':      '',
    'mirage_sabre':    '',
    'mirage':          '',
    'sabre_fid':       '..',
    'lightsabre_fid':  '..',
    'mirage_fid_sabre':'..',
    'mirage_fid':      '..',
    'finesse':         '..',
}

# Vertical separator positions (after index): base | mirrors | fid | fid+mirrors
GROUP_BOUNDARIES: list[int] = [1, 3, 5, 9]


def _ordered_configs(df: pd.DataFrame, configs: list[str] | None) -> list[str]:
    present = set(df['config'].unique())
    if configs is not None:
        return [c for c in configs if c in present]
    return [c for c in CONFIG_ORDER if c in present]


def _label_of(df: pd.DataFrame, key: str) -> str:
    rows = df[df['config'] == key]['label']
    return rows.iloc[0] if len(rows) else key


def _config_means(df: pd.DataFrame, col: str) -> pd.Series:
    """Grand mean per config: average over circuits, then over seeds."""
    return df.groupby(['config', 'circuit'])[col].mean().groupby('config').mean()


def _hex_lighten(hex_color: str, factor: float = 0.45) -> str:
    """Return a lighter (toward white) version of hex_color for table backgrounds."""
    r, g, b = mcolors.to_rgb(hex_color)
    r = r + (1 - r) * factor
    g = g + (1 - g) * factor
    b = b + (1 - b) * factor
    return mcolors.to_hex((r, g, b))


# ---------------------------------------------------------------------------
# Shared horizontal-bar helper (used by fig_overview)
# ---------------------------------------------------------------------------

def _draw_hbar(
    ax: plt.Axes,
    df: pd.DataFrame,
    keys: list[str],
    values: dict[str, float],
    ref_val: float | None,
    xlabel: str,
    panel_label: str,
    ref_label: str = 'SABRE baseline',
    higher_is_better: bool = False,
    show_ylabels: bool = True,
) -> None:
    """Draw a horizontal bar chart on ax.  No error bars."""
    n = len(keys)
    y = np.arange(n)
    maxv = max((v for v in values.values() if not np.isnan(v)), default=1.0)

    # Replace NaN widths with 0 so barh doesn't produce undefined patches
    widths = [0.0 if np.isnan(values.get(k, float('nan'))) else values.get(k, 0.0)
              for k in keys]

    ax.barh(
        y,
        widths,
        color=[COLORS.get(k, '#999') for k in keys],
        hatch=[HATCH.get(k, '') for k in keys],
        height=0.6,
        edgecolor='none',
    )

    if ref_val is not None and not np.isnan(ref_val):
        ax.axvline(ref_val, color='#444', linestyle='--', linewidth=0.9,
                   label=ref_label, zorder=3)
        ax.legend(fontsize=7, loc='lower right', framealpha=0.8,
                  handlelength=1.4, borderpad=0.5)

    for i, k in enumerate(keys):
        v = values.get(k, float('nan'))
        if np.isnan(v):
            continue
        if ref_val and not np.isnan(ref_val) and ref_val != 0 and k != _ref_key(keys, ref_val, values):
            pct = (v - ref_val) / abs(ref_val) * 100
            pct_str = f' {pct:+.0f}%'
            ann_col = '#CC3311' if (pct < 0) != higher_is_better else '#555'
        else:
            pct_str = ''
            ann_col = '#333'
        ax.text(v + maxv * 0.012, i, f'{v:.1f}{pct_str}',
                va='center', ha='left', fontsize=7.5, color=ann_col)

    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([_label_of(df, k) for k in keys], fontsize=8.5)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_title(panel_label, fontsize=10, fontweight='bold', pad=7)
    ax.invert_yaxis()
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
    ax.xaxis.grid(True, alpha=0.22, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0, maxv * 1.38)


def _ref_key(keys, ref_val, values):
    """Return the key whose value is closest to ref_val (i.e. the reference config)."""
    return min(keys, key=lambda k: abs(values.get(k, float('inf')) - ref_val))


# ---------------------------------------------------------------------------
# fig_overview — the main paper figure
# ---------------------------------------------------------------------------

def fig_overview(
    df: pd.DataFrame,
    configs: list[str] | None = None,
    reference: str = 'sabre',
    title: str = 'FINESSE Ablation Study',
) -> plt.Figure:
    """
    Three-panel horizontal-bar figure + summary table, matching the paper style.

      (a) Routing overhead  — avg SWAP count, no error bars
      (b) Circuit depth     — avg gate depth, no error bars
      (c) Circuit fidelity  — avg fidelity (exp(-lf_cost)), no error bars

    Below: styled table with one row per config showing all three metrics and
    percentage deltas vs the reference config (default: SABRE).

    Args:
        df:        Benchmark DataFrame from run_benchmark() or pd.read_parquet().
        configs:   Configs to include (default: all present in df, canonical order).
        reference: Config used as the baseline for percentage annotations.
        title:     Figure suptitle.

    Returns:
        matplotlib Figure.
    """
    keys = _ordered_configs(df, configs)
    n = len(keys)

    # ── aggregate means (circuit-averaged, then seed-averaged) ───────────────
    sw  = _config_means(df, 'swap_count')
    gd  = _config_means(df, 'gate_depth')
    # Use lf_cost (lower = better) for panel (c).  exp(-lf_cost) underflows to 0
    # on large circuits, making the fidelity column useless for those benchmarks.
    lf_df = df.dropna(subset=['lf_cost'])
    lf = _config_means(lf_df, 'lf_cost') if len(lf_df) else pd.Series(dtype=float)

    ref_sw = float(sw.get(reference, sw.iloc[0]  if len(sw) else 1.0))
    ref_gd = float(gd.get(reference, gd.iloc[0]  if len(gd) else 1.0))
    ref_lf = float(lf.get(reference, lf.iloc[0]  if len(lf) else float('nan')))

    sw_vals = {k: float(sw.get(k, float('nan'))) for k in keys}
    gd_vals = {k: float(gd.get(k, float('nan'))) for k in keys}
    lf_vals = {k: float(lf.get(k, float('nan'))) for k in keys}

    # ── layout ───────────────────────────────────────────────────────────────
    n_circuits = df['circuit'].nunique()
    n_seeds    = df['seed'].nunique()
    row_h      = max(0.42, 3.6 / n)          # bar height scales with config count
    bar_h      = n * row_h + 1.2             # panel height

    fig = plt.figure(figsize=(15, bar_h + 3.8))
    gs  = fig.add_gridspec(
        2, 3,
        height_ratios=[bar_h, 3.8],
        hspace=0.55,
        wspace=0.38,
        top=0.91, bottom=0.04, left=0.08, right=0.97,
    )

    ax_sw  = fig.add_subplot(gs[0, 0])
    ax_gd  = fig.add_subplot(gs[0, 1])
    ax_fid = fig.add_subplot(gs[0, 2])
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.axis('off')

    ref_lbl = f'{_label_of(df, reference)} baseline'

    # ── bar panels ───────────────────────────────────────────────────────────
    # All three panels show the same y-tick labels (same config names) so
    # their label areas are equal width — this is what keeps the bar areas
    # left-aligned across panels.  Do NOT use sharey: it causes invert_yaxis()
    # to toggle on the shared axis and scramble the config order.
    _draw_hbar(ax_sw,  df, keys, sw_vals,  ref_sw,  'Avg SWAP count', '(a) Routing overhead',
               ref_label=ref_lbl, show_ylabels=True)
    _draw_hbar(ax_gd,  df, keys, gd_vals,  ref_gd,  'Avg gate depth', '(b) Circuit depth',
               ref_label=ref_lbl, show_ylabels=True)

    if len(lf) and not all(np.isnan(v) for v in lf_vals.values()):
        # lf_cost = −log F: lower is better.  higher_is_better=False so red = worse.
        _draw_hbar(ax_fid, df, keys, lf_vals, ref_lf,
                   'Avg −log-fidelity cost', '(c) Circuit error',
                   ref_label=ref_lbl,
                   higher_is_better=False, show_ylabels=True)
    else:
        # No fidelity_matrix was passed to run_benchmark — fill panel with
        # matching y-labels so horizontal alignment with (a) and (b) is preserved.
        y = np.arange(n)
        ax_fid.barh(y, np.zeros(n), height=0.6)
        ax_fid.set_yticks(y)
        ax_fid.set_yticklabels([_label_of(df, k) for k in keys], fontsize=8.5)
        ax_fid.invert_yaxis()
        ax_fid.set_xlabel('Avg −log-fidelity cost', fontsize=9)
        ax_fid.set_title('(c) Circuit error', fontsize=10, fontweight='bold', pad=7)
        ax_fid.spines[['top', 'right', 'left']].set_visible(False)
        ax_fid.tick_params(axis='y', length=0)
        ax_fid.set_xlim(0, 1)
        ax_fid.xaxis.grid(True, alpha=0.22, linewidth=0.6)
        ax_fid.set_axisbelow(True)
        ax_fid.text(0.5, 0.5, 'no fidelity data', ha='center', va='center',
                    transform=ax_fid.transAxes, fontsize=9, color='#aaa',
                    style='italic')

    # ── summary table ────────────────────────────────────────────────────────
    has_lf = len(lf) > 0

    ref_label_str = _label_of(df, reference)
    col_headers = ['Config', 'Avg SWAPs', f'vs {ref_label_str}',
                   'Avg Depth', f'vs {ref_label_str}\n(depth)']
    if has_lf:
        col_headers += ['Avg −log F', f'vs {ref_label_str}\n(error)']
    n_cols = len(col_headers)

    def pct(v, ref, lower_better=True):
        if np.isnan(v) or np.isnan(ref) or ref == 0:
            return '—'
        p = (v - ref) / abs(ref) * 100
        return f'{p:+.1f}%'

    table_data = []
    for k in keys:
        sw_v = sw_vals.get(k, float('nan'))
        gd_v = gd_vals.get(k, float('nan'))
        lf_v = lf_vals.get(k, float('nan'))

        row = [
            _label_of(df, k),
            f'{sw_v:.1f}' if not np.isnan(sw_v) else '—',
            '—' if k == reference else pct(sw_v, ref_sw),
            f'{gd_v:.1f}' if not np.isnan(gd_v) else '—',
            '—' if k == reference else pct(gd_v, ref_gd),
        ]
        if has_lf:
            row += [
                f'{lf_v:.1f}' if not np.isnan(lf_v) else '—',
                '—' if k == reference else pct(lf_v, ref_lf),
            ]
        table_data.append(row)

    tbl = ax_tbl.table(
        cellText=table_data,
        colLabels=col_headers,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)

    # Style header
    for col in range(n_cols):
        cell = tbl[0, col]
        cell.set_facecolor('#1a3a5c')
        cell.set_text_props(color='white', fontweight='bold')
        cell.set_edgecolor('#1a3a5c')

    # Style data rows
    for row_i, k in enumerate(keys):
        r = row_i + 1
        base_color = COLORS.get(k, '#999')
        cfg_bg = _hex_lighten(base_color, 0.55)
        is_ref  = (k == reference)
        is_best_sw  = (not np.isnan(sw_vals.get(k, float('nan'))) and
                       sw_vals.get(k) == min(v for v in sw_vals.values() if not np.isnan(v)))

        for col in range(n_cols):
            cell = tbl[r, col]
            if col == 0:
                cell.set_facecolor(cfg_bg)
                cell.set_edgecolor('#dddddd')
            else:
                # Highlight the best-value cell in swap/depth/fidelity columns
                cell.set_facecolor('#f7f7f7' if row_i % 2 == 0 else 'white')
                cell.set_edgecolor('#e0e0e0')

            # Bold + colored text for the reference and best config
            txt = cell.get_text()
            if k == 'finesse' or is_best_sw:
                txt.set_fontweight('bold')
                if col > 0:
                    txt.set_color('#990000')
            if is_ref and col > 0:
                txt.set_color('#333333')

    # Scale table to fill the axes
    tbl.scale(1, 1.55)

    # subtitle above table
    n_trials = df.groupby(['config', 'circuit', 'seed']).ngroups  # proxy
    ax_tbl.set_title(
        f'(d) Full comparison — averaged over {n_circuits} circuits, '
        f'{n_seeds} seeds',
        fontsize=9.5, pad=14, loc='center',
    )

    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.975)
    return fig


# ---------------------------------------------------------------------------
# fig_swap_bars — vertical bar chart (no error bars)
# ---------------------------------------------------------------------------

def fig_swap_bars(
    df: pd.DataFrame,
    configs: list[str] | None = None,
    reference: str = 'sabre',
    title_suffix: str = '',
) -> plt.Figure:
    keys = _ordered_configs(df, configs)
    means = {k: float(_config_means(df, 'swap_count').get(k, float('nan'))) for k in keys}
    ref_mean = means.get(reference, 1.0)

    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.3), 4.5))
    x = np.arange(len(keys))

    ax.bar(x, [means[k] for k in keys],
           color=[COLORS.get(k, '#999') for k in keys],
           hatch=[HATCH.get(k, '') for k in keys],
           edgecolor='white', linewidth=0.8)

    for i, k in enumerate(keys):
        v = means[k]
        if np.isnan(v):
            continue
        if k == reference:
            label, col = '—', '#666'
        else:
            pct = (v - ref_mean) / abs(ref_mean) * 100
            label = f'{pct:+.0f}%'
            col = '#CC3311' if pct < 0 else '#888'
        ax.text(i, v + max(means.values()) * 0.02, label,
                ha='center', va='bottom', fontsize=8, color=col,
                fontweight='bold' if label != '—' else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels([_label_of(df, k) for k in keys],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Average SWAP count', fontsize=11)
    n_seeds = df['seed'].nunique()
    n_circs = df['circuit'].nunique()
    t = f'Routing overhead ({n_circs} circuits, {n_seeds} seeds)'
    if title_suffix:
        t += f', {title_suffix}'
    ax.set_title(t, fontsize=11)
    ax.set_ylim(0, max(v for v in means.values() if not np.isnan(v)) * 1.22)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_lf_bars — lf_cost bar chart (no error bars)
# ---------------------------------------------------------------------------

def fig_lf_bars(
    df: pd.DataFrame,
    configs: list[str] | None = None,
) -> plt.Figure:
    keys = _ordered_configs(df, configs)
    fid_df = df.dropna(subset=['lf_cost'])
    keys = [k for k in keys if k in fid_df['config'].unique()]

    if not keys:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No fidelity data', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    means = {k: float(_config_means(fid_df, 'lf_cost').get(k, float('nan'))) for k in keys}
    baseline = means[keys[0]]

    fig, ax = plt.subplots(figsize=(max(5, len(keys) * 1.4), 4.5))
    x = np.arange(len(keys))

    ax.bar(x, [means[k] for k in keys],
           color=[COLORS.get(k, '#999') for k in keys],
           hatch=[HATCH.get(k, '') for k in keys],
           edgecolor='white', linewidth=0.8)

    for i, k in enumerate(keys):
        v = means[k]
        if np.isnan(v):
            continue
        pct = (v - baseline) / abs(baseline) * 100
        label = '—' if i == 0 else f'{pct:+.0f}%'
        col = '#CC3311' if pct < 0 else '#888'
        ax.text(i, v + max(means.values()) * 0.02, label,
                ha='center', va='bottom', fontsize=9, color=col,
                fontweight='bold' if label != '—' else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels([_label_of(df, k) for k in keys],
                       rotation=25, ha='right', fontsize=10)
    ax.set_ylabel('Average −log-fidelity cost', fontsize=10)
    ax.set_title('Total circuit error — fidelity-aware configs', fontsize=10)
    ax.set_ylim(0, max(v for v in means.values() if not np.isnan(v)) * 1.18)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_depth_bars — gate depth bar chart (no error bars)
# ---------------------------------------------------------------------------

def fig_depth_bars(
    df: pd.DataFrame,
    configs: list[str] | None = None,
    reference: str = 'sabre',
) -> plt.Figure:
    keys = _ordered_configs(df, configs)
    means = {k: float(_config_means(df, 'gate_depth').get(k, float('nan'))) for k in keys}
    ref_mean = means.get(reference, 1.0)

    fig, ax = plt.subplots(figsize=(max(7, len(keys) * 1.3), 4.5))
    x = np.arange(len(keys))

    ax.bar(x, [means[k] for k in keys],
           color=[COLORS.get(k, '#999') for k in keys],
           hatch=[HATCH.get(k, '') for k in keys],
           edgecolor='white', linewidth=0.8)

    for i, k in enumerate(keys):
        v = means[k]
        if np.isnan(v):
            continue
        if k == reference:
            label, col = '—', '#666'
        else:
            pct = (v - ref_mean) / abs(ref_mean) * 100
            label = f'{pct:+.0f}%'
            col = '#CC3311' if pct < 0 else '#888'
        ax.text(i, v + max(means.values()) * 0.02, label,
                ha='center', va='bottom', fontsize=8, color=col,
                fontweight='bold' if label != '—' else 'normal')

    ax.set_xticks(x)
    ax.set_xticklabels([_label_of(df, k) for k in keys],
                       rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Average gate depth', fontsize=11)
    ax.set_title('Circuit depth after routing', fontsize=11)
    ax.set_ylim(0, max(v for v in means.values() if not np.isnan(v)) * 1.22)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_swap_boxplot — distribution box plot across seeds
# ---------------------------------------------------------------------------

def fig_swap_boxplot(
    df: pd.DataFrame,
    configs: list[str] | None = None,
    circuits: list[str] | None = None,
    title: str = '',
) -> plt.Figure:
    keys = _ordered_configs(df, configs)
    if circuits is None:
        circuits = sorted(df['circuit'].unique())

    n_circs = len(circuits)
    n_cfgs  = len(keys)
    width   = 0.8 / n_cfgs
    offsets = np.linspace(-(n_cfgs - 1) / 2, (n_cfgs - 1) / 2, n_cfgs) * width

    fig, ax = plt.subplots(figsize=(max(6, n_circs * 1.6 + 1), 4.5))

    for i, cfg in enumerate(keys):
        sub = df[df['config'] == cfg]
        data = [sub[sub['circuit'] == c]['swap_count'].values for c in circuits]
        pos  = np.arange(n_circs) + offsets[i]
        ax.boxplot(
            data, positions=pos, widths=width * 0.85,
            patch_artist=True, manage_ticks=False,
            medianprops=dict(color='white', linewidth=1.5),
            whiskerprops=dict(linewidth=0.8, color='#555'),
            capprops=dict(linewidth=0.8, color='#555'),
            flierprops=dict(marker='.', markersize=3,
                            markerfacecolor='#aaa', markeredgewidth=0),
            boxprops=dict(facecolor=COLORS.get(cfg, '#999'), linewidth=0),
        )

    ax.set_xticks(np.arange(n_circs))
    ax.set_xticklabels(circuits, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('SWAP count', fontsize=11)
    ax.set_title(title or 'SWAP count distribution across seeds', fontsize=11)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    handles = [mpatches.Patch(facecolor=COLORS.get(k, '#999'), label=_label_of(df, k))
               for k in keys]
    ax.legend(handles=handles, fontsize=8, framealpha=0.7,
              loc='upper left', bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_scatter — per-circuit scatter: baseline vs comparison configs
# ---------------------------------------------------------------------------

def fig_scatter(
    df: pd.DataFrame,
    x_config: str = 'sabre',
    y_configs: list[str] | None = None,
) -> plt.Figure:
    if y_configs is None:
        present = set(df['config'].unique())
        y_configs = [c for c in ['mirage', 'mirage_fid', 'finesse'] if c in present]
    if not y_configs:
        y_configs = [c for c in _ordered_configs(df, None) if c != x_config][:2]

    circ_means = df.groupby(['config', 'circuit'])['swap_count'].mean().reset_index()
    circuits   = df['circuit'].unique()
    n_logical  = df.groupby('circuit')['n_logical'].first()

    ncols = len(y_configs)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4.5), squeeze=False)
    axes = axes[0]

    x_base = {
        circ: circ_means.loc[
            (circ_means['config'] == x_config) & (circ_means['circuit'] == circ),
            'swap_count'
        ].values[0]
        for circ in circuits
        if ((circ_means['config'] == x_config) & (circ_means['circuit'] == circ)).any()
    }

    for ax, y_key in zip(axes, y_configs):
        sx, sy, sizes, labels_pts = [], [], [], []
        for circ in circuits:
            if circ not in x_base:
                continue
            row = circ_means[(circ_means['config'] == y_key) &
                             (circ_means['circuit'] == circ)]
            if row.empty:
                continue
            sx.append(x_base[circ])
            sy.append(float(row['swap_count'].values[0]))
            sizes.append(max(30, int(n_logical.get(circ, 5)) * 10))
            labels_pts.append(circ)

        sx, sy = np.array(sx), np.array(sy)
        lo = 0
        hi = max(sx.max(), sy.max()) * 1.1 if len(sx) else 10
        ax.plot([lo, hi], [lo, hi], '--', color='#bbb', lw=1.2)
        ax.scatter(sx, sy, s=sizes, color=COLORS.get(y_key, '#999'),
                   alpha=0.82, edgecolors='white', linewidths=0.7, zorder=3)
        for circ, xi, yi in zip(labels_pts, sx, sy):
            ax.annotate(circ.split('_')[0][:10], (xi, yi),
                        fontsize=6.5, color='#333',
                        xytext=(3, 3), textcoords='offset points')

        pct = (sy - sx) / np.maximum(sx, 1) * 100
        ax.set_xlabel(f'{_label_of(df, x_config)} avg SWAPs', fontsize=10)
        ax.set_ylabel(f'{_label_of(df, y_key)} avg SWAPs', fontsize=10)
        ax.set_title(f'{_label_of(df, y_key)}\nmedian {np.median(pct):+.0f}% vs {_label_of(df, x_config)}',
                     fontsize=10)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect('equal')
        ax.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Per-circuit SWAP count (below diagonal = fewer SWAPs)',
                 fontsize=11, y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_contributions — cumulative contribution waterfall
# ---------------------------------------------------------------------------

def fig_contributions(
    df: pd.DataFrame,
    steps: list[tuple[str, str]] | None = None,
) -> plt.Figure:
    if steps is None:
        present = set(df['config'].unique())
        default = [
            ('sabre',         'SABRE baseline'),
            ('lightsabre',    '+ LightSABRE\n  (release valve)'),
            ('mirage',        '+ MIRAGE\n  (mirror absorption)'),
            ('mirage_fid',    '+ Fidelity in H\n  (SWAP + mirror aware)'),
            ('finesse',       '+ FinesseLayout\n  (fidelity-aware initial layout)'),
        ]
        steps = [(k, l) for k, l in default if k in present]

    keys  = [k for k, _ in steps]
    slabs = [l for _, l in steps]
    vals  = {k: float(_config_means(df, 'swap_count').get(k, float('nan'))) for k in keys}

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(keys) * 0.9)))
    y = np.arange(len(keys))

    ax.barh(y, [vals.get(k, 0) for k in keys],
            color=[COLORS.get(k, '#999') for k in keys],
            hatch=[HATCH.get(k, '') for k in keys],
            edgecolor='white', height=0.55)

    sabre_val = vals.get(keys[0], 1.0)
    for i, k in enumerate(keys):
        v = vals.get(k, float('nan'))
        if np.isnan(v):
            continue
        pct = (v - sabre_val) / sabre_val * 100
        pct_str = '—' if i == 0 else f'{pct:+.0f}%'
        col = '#CC3311' if pct < -2 else '#555'
        ax.text(v + sabre_val * 0.01, i, f'{v:.0f}  ({pct_str})',
                va='center', ha='left', fontsize=9, color=col,
                fontweight='bold' if abs(pct) > 15 else 'normal')

    ax.set_yticks(y)
    ax.set_yticklabels(slabs, fontsize=9)
    ax.set_xlabel('Average SWAP count', fontsize=10)
    ax.set_title(f'Cumulative contribution — {df["circuit"].nunique()} circuits, Q20 Tokyo',
                 fontsize=10)
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    maxv = max(v for v in vals.values() if not np.isnan(v))
    ax.set_xlim(0, maxv * 1.35)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# fig_pareto — SWAP vs lf_cost 2D
# ---------------------------------------------------------------------------

def fig_pareto(
    df: pd.DataFrame,
    configs: list[str] | None = None,
) -> plt.Figure:
    keys = _ordered_configs(df, configs)
    fid_df = df.dropna(subset=['lf_cost'])
    fid_keys = [k for k in keys if k in fid_df['config'].unique()]

    if not fid_keys:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No fidelity data', ha='center', va='center',
                transform=ax.transAxes)
        return fig

    sw_means  = _config_means(df, 'swap_count')
    lf_means  = _config_means(fid_df, 'lf_cost')

    fig, ax = plt.subplots(figsize=(6, 5))
    for k in fid_keys:
        sw = float(sw_means.get(k, float('nan')))
        lf = float(lf_means.get(k, float('nan')))
        if np.isnan(sw) or np.isnan(lf):
            continue
        ax.scatter(sw, lf, s=140, color=COLORS.get(k, '#999'),
                   edgecolors='white', linewidths=0.9, zorder=3)
        ax.annotate(_label_of(df, k), (sw, lf),
                    xytext=(6, 4), textcoords='offset points',
                    fontsize=8.5, color='#222')

    ax.set_xlabel('Average SWAP count', fontsize=10)
    ax.set_ylabel('Average −log-fidelity cost', fontsize=10)
    ax.set_title('SWAP overhead vs circuit error\nBetter = bottom-left', fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.xaxis.grid(True, alpha=0.25)
    ax.yaxis.grid(True, alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    return fig
