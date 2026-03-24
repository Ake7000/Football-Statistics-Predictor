"""
plot_classifier_heatmaps.py
Plots F1-macro + accuracy heatmaps where each cell is split diagonally:
  upper-left  triangle = CV F1-macro  (independent colour scale)
  lower-right triangle = CV accuracy  (independent colour scale)

  1. Per-model heatmap: rows=variant, cols=stat, one figure per model_type.
     Both cw strategies stacked vertically with no gap.
  2. Summary heatmap:  rows=model_type, cols=variant, mean across all stats.
     Both cw strategies stacked vertically with no gap.
  3. CW comparison: model × cw_strategy, CV F1-macro only.

Reads:  output/classifier_results_master.csv
Writes: output/plots/clf_heatmap_<model_type>.png
        output/plots/clf_heatmap_summary.png
        output/plots/clf_heatmap_cw_comparison.png

Usage:
    python refactor/analysis/classifier_analysis/plot_classifier_heatmaps.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as mcm
import matplotlib.colors as mcolors
import seaborn as sns

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # → refactor/
from shared_config import CLASSIFIER_STAT_PAIRS

OUTPUT_DIR = Path(__file__).parent / "output"
PLOTS_DIR  = OUTPUT_DIR / "plots"
MASTER_CSV = OUTPUT_DIR / "classifier_results_master.csv"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Canonical stat order from config
STAT_ORDER = [s for s, _, _ in CLASSIFIER_STAT_PAIRS]

F1_CMAP   = "Blues"
ACC_CMAP  = "Oranges"
ANNOT_FS  = 7   # annotation font size inside triangle cells


def load_data() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"classifier_results_master.csv not found. "
            f"Run collect_classifier_results.py first.\n  Expected: {MASTER_CSV}"
        )
    return pd.read_csv(MASTER_CSV)


def _draw_split_cells(
    ax,
    f1_pivot: pd.DataFrame,
    acc_pivot: pd.DataFrame,
    cols: list,
    f1_norm: mcolors.Normalize,
    acc_norm: mcolors.Normalize,
    show_xlabels: bool,
    ylabel: str,
) -> None:
    """
    Fill each cell with two triangles using ax.fill():
      upper-left  triangle = CV F1-macro  (Blues, darker = better)
      lower-right triangle = CV accuracy  (Oranges, darker = better)
    """
    import matplotlib
    f1_cm  = matplotlib.colormaps[F1_CMAP]
    acc_cm = matplotlib.colormaps[ACC_CMAP]
    na_color = (0.88, 0.88, 0.88, 1.0)
    n_rows = len(f1_pivot.index)
    n_cols = len(cols)

    for i, row_label in enumerate(f1_pivot.index):
        for j, col_label in enumerate(cols):
            f1_val  = f1_pivot.at[row_label, col_label]  if col_label in f1_pivot.columns  else np.nan
            acc_val = acc_pivot.at[row_label, col_label] if col_label in acc_pivot.columns else np.nan
            try:
                f1_nan  = np.isnan(float(f1_val))
            except (TypeError, ValueError):
                f1_nan  = True
            try:
                acc_nan = np.isnan(float(acc_val))
            except (TypeError, ValueError):
                acc_nan = True

            f1_color  = f1_cm(f1_norm(float(f1_val)))   if not f1_nan  else na_color
            acc_color = acc_cm(acc_norm(float(acc_val))) if not acc_nan else na_color

            # upper-left triangle (F1): top-left, top-right, bottom-left
            ax.fill([j, j + 1, j],         [i, i, i + 1],         color=f1_color,  zorder=1)
            # lower-right triangle (Acc): top-right, bottom-right, bottom-left
            ax.fill([j + 1, j + 1, j],     [i, i + 1, i + 1],     color=acc_color, zorder=1)
            # diagonal divider
            ax.plot([j, j + 1], [i + 1, i], color="white", lw=0.8, zorder=2)

            # annotations
            if not f1_nan:
                ax.text(j + 0.28, i + 0.30, f"{float(f1_val):.3f}",
                        ha="center", va="center", fontsize=ANNOT_FS,
                        color="white" if f1_norm(float(f1_val)) > 0.6 else "black", zorder=3)
            if not acc_nan:
                ax.text(j + 0.72, i + 0.72, f"{float(acc_val):.3f}",
                        ha="center", va="center", fontsize=ANNOT_FS,
                        color="white" if acc_norm(float(acc_val)) > 0.6 else "black", zorder=3)

    # cell grid
    for k in range(n_cols + 1):
        ax.axvline(k, color="white", lw=1.2, zorder=2)
    for k in range(n_rows + 1):
        ax.axhline(k, color="white", lw=1.2, zorder=2)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(f1_pivot.index.tolist(), rotation=0, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="both", length=0)
    if show_xlabels:
        ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    else:
        ax.set_xticklabels([""] * n_cols)
        ax.tick_params(axis="x", bottom=False)


def _add_dual_colorbars(fig, axes, f1_norm, acc_norm) -> None:
    """Add two colorbars (F1 = Blues left, Acc = Oranges right) spanning full height."""
    import matplotlib
    fig.subplots_adjust(right=0.82)
    # F1 colorbar
    cax_f1 = fig.add_axes([0.84, 0.10, 0.03, 0.78])
    sm_f1  = mcm.ScalarMappable(cmap=F1_CMAP, norm=f1_norm)
    sm_f1.set_array([])
    fig.colorbar(sm_f1, cax=cax_f1, label="CV F1-macro ◤ (Blues)")
    # Acc colorbar
    cax_acc = fig.add_axes([0.90, 0.10, 0.03, 0.78])
    sm_acc  = mcm.ScalarMappable(cmap=ACC_CMAP, norm=acc_norm)
    sm_acc.set_array([])
    fig.colorbar(sm_acc, cax=cax_acc, label="CV Accuracy ◢ (Oranges)")


def plot_per_model_heatmaps(df: pd.DataFrame) -> None:
    cw_strategies = sorted(df["class_weight_strategy"].unique())
    for model_type, grp in df.groupby("model_type"):
        # Build one F1 pivot and one Acc pivot per cw strategy
        pivots_f1  = {}
        pivots_acc = {}
        for cw in cw_strategies:
            sub = grp[grp["class_weight_strategy"] == cw]
            if sub.empty:
                continue
            pf1 = sub.pivot_table(
                index="variant", columns="stat", values="cv_f1_macro_mean", aggfunc="mean",
            )
            pac = sub.pivot_table(
                index="variant", columns="stat", values="cv_acc_mean", aggfunc="mean",
            )
            present = set(pf1.columns) | set(pac.columns)
            cols    = [s for s in STAT_ORDER if s in present] + sorted(present - set(STAT_ORDER))
            pivots_f1[cw]  = pf1.reindex(columns=cols)
            pivots_acc[cw] = pac.reindex(columns=cols)

        if not pivots_f1:
            continue

        # Shared norms computed across all cw strategies for this model
        all_f1  = pd.concat(pivots_f1.values()).values.flatten().astype(float)
        all_acc = pd.concat(pivots_acc.values()).values.flatten().astype(float)
        all_f1  = all_f1[~np.isnan(all_f1)]
        all_acc = all_acc[~np.isnan(all_acc)]
        f1_norm  = mcolors.Normalize(vmin=float(all_f1.min()),  vmax=float(all_f1.max()))
        acc_norm = mcolors.Normalize(vmin=float(all_acc.min()), vmax=float(all_acc.max()))

        sample = next(iter(pivots_f1.values()))
        n_cw   = len(pivots_f1)
        n_rows = len(sample)
        n_cols = len(sample.columns)
        fig_w  = max(10, n_cols * 1.4)
        fig_h  = max(4, n_rows * 1.0 + 1.0) * n_cw

        fig, axes = plt.subplots(
            n_cw, 1,
            figsize=(fig_w, fig_h),
            sharex=True,
            gridspec_kw={"hspace": 0},
        )
        if n_cw == 1:
            axes = [axes]

        active_cws = [cw for cw in cw_strategies if cw in pivots_f1]
        for i, (ax, cw) in enumerate(zip(axes, active_cws)):
            is_last = (i == n_cw - 1)
            _draw_split_cells(
                ax,
                pivots_f1[cw], pivots_acc[cw],
                list(pivots_f1[cw].columns),
                f1_norm, acc_norm,
                show_xlabels=is_last,
                ylabel=f"cw={cw}",
            )

        _add_dual_colorbars(fig, axes, f1_norm, acc_norm)
        fig.suptitle(
            f"CV F1-macro (◤ Blues) & Accuracy (◢ Oranges) by Variant × Stat  —  {model_type}",
            fontsize=13, y=1.01,
        )
        out_path = PLOTS_DIR / f"clf_heatmap_{model_type}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved {out_path}")


def plot_summary_heatmap(df: pd.DataFrame) -> None:
    cw_strategies = sorted(df["class_weight_strategy"].unique())

    pivots_f1  = {}
    pivots_acc = {}
    for cw in cw_strategies:
        sub = df[df["class_weight_strategy"] == cw]
        if sub.empty:
            continue
        f1_agg = (
            sub.groupby(["model_type", "variant"])["cv_f1_macro_mean"]
            .mean().reset_index()
        )
        acc_agg = (
            sub.groupby(["model_type", "variant"])["cv_acc_mean"]
            .mean().reset_index()
        )
        pf1 = f1_agg.pivot(index="model_type", columns="variant", values="cv_f1_macro_mean")
        pac = acc_agg.pivot(index="model_type", columns="variant", values="cv_acc_mean")
        all_variants = sorted(set(pf1.columns) | set(pac.columns))
        pivots_f1[cw]  = pf1.reindex(columns=all_variants)
        pivots_acc[cw] = pac.reindex(columns=all_variants)

    if not pivots_f1:
        return

    # Shared norms across cw strategies
    all_f1  = pd.concat(pivots_f1.values()).values.flatten().astype(float)
    all_acc = pd.concat(pivots_acc.values()).values.flatten().astype(float)
    all_f1  = all_f1[~np.isnan(all_f1)]
    all_acc = all_acc[~np.isnan(all_acc)]
    f1_norm  = mcolors.Normalize(vmin=float(all_f1.min()),  vmax=float(all_f1.max()))
    acc_norm = mcolors.Normalize(vmin=float(all_acc.min()), vmax=float(all_acc.max()))

    n_cw   = len(pivots_f1)
    sample = next(iter(pivots_f1.values()))
    fig_w  = max(8, len(sample.columns) * 1.4)
    fig_h  = max(3, len(sample) * 1.1 + 1.0) * n_cw

    fig, axes = plt.subplots(
        n_cw, 1,
        figsize=(fig_w, fig_h),
        sharex=True,
        gridspec_kw={"hspace": 0},
    )
    if n_cw == 1:
        axes = [axes]

    active_cws = [cw for cw in cw_strategies if cw in pivots_f1]
    for i, (ax, cw) in enumerate(zip(axes, active_cws)):
        is_last = (i == n_cw - 1)
        _draw_split_cells(
            ax,
            pivots_f1[cw], pivots_acc[cw],
            list(pivots_f1[cw].columns),
            f1_norm, acc_norm,
            show_xlabels=is_last,
            ylabel=f"cw={cw}  |  Model",
        )

    _add_dual_colorbars(fig, axes, f1_norm, acc_norm)
    fig.suptitle(
        "Mean CV F1-macro (◤ Blues) & Accuracy (◢ Oranges)  —  Model × Variant",
        fontsize=13, y=1.01,
    )
    out_path = PLOTS_DIR / "clf_heatmap_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def plot_cw_comparison_heatmap(df: pd.DataFrame) -> None:
    """Summary heatmap showing none vs sqrt side-by-side for each model."""
    summary = (
        df.groupby(["model_type", "class_weight_strategy"])["cv_f1_macro_mean"]
        .mean()
        .reset_index()
        .rename(columns={"cv_f1_macro_mean": "mean_cv_f1_macro"})
    )
    pivot = summary.pivot(
        index="model_type", columns="class_weight_strategy", values="mean_cv_f1_macro"
    )

    fig, ax = plt.subplots(figsize=(max(5, len(pivot.columns) * 2), max(3, len(pivot) * 0.9 + 1.5)))
    sns.heatmap(
        pivot, ax=ax, annot=True, fmt=".4f",
        cmap="RdYlGn", linewidths=0.5, linecolor="white",
        vmin=0.25, vmax=0.65,
        cbar_kws={"label": "Mean CV F1-macro"},
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=10)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    ax.set_title("Class Weight Strategy Comparison  —  Mean CV F1-macro per Model", fontsize=12, pad=12)
    ax.set_xlabel("Class Weight Strategy")
    ax.set_ylabel("Model")
    plt.tight_layout()
    out_path = PLOTS_DIR / "clf_heatmap_cw_comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def main():
    df = load_data()
    if "cv_f1_macro_mean" not in df.columns:
        print("[ERROR] 'cv_f1_macro_mean' column missing. Did collect_classifier_results.py complete?")
        return
    plot_per_model_heatmaps(df)
    plot_summary_heatmap(df)
    plot_cw_comparison_heatmap(df)


if __name__ == "__main__":
    main()
