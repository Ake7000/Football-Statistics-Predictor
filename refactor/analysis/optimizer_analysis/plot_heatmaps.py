"""
plot_heatmaps.py
Plots split-cell CV metric heatmaps:
    upper-left  triangle = CV RMSE (independent colour scale)
    lower-right triangle = CV MAE  (independent colour scale)

    1. Per-model heatmap: rows=variant, cols=target.
    2. Summary heatmap:  rows=model_type, cols=variant, averaged across targets.

Reads:  output/results_master.csv
Writes: output/plots/heatmap_<model_type>.png
        output/plots/heatmap_summary.png

Usage:
    python refactor/analysis/optimizer_analysis/plot_heatmaps.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import matplotlib.colors as mcolors

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # refactor/ → shared_config
from shared_config import TARGETS

OUTPUT_DIR = Path(__file__).parent / "output"
PLOTS_DIR  = OUTPUT_DIR / "plots"
MASTER_CSV = OUTPUT_DIR / "results_master.csv"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

RMSE_CMAP = "Blues"
MAE_CMAP  = "Oranges"
ANNOT_FS  = 7


def load_data() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"results_master.csv not found. Run collect_results.py first.\n  Expected: {MASTER_CSV}"
        )
    return pd.read_csv(MASTER_CSV)


def _draw_split_cells(
    ax,
    rmse_pivot: pd.DataFrame,
    mae_pivot: pd.DataFrame,
    cols: list,
    rmse_norm: mcolors.Normalize,
    mae_norm: mcolors.Normalize,
    show_xlabels: bool,
    ylabel: str,
) -> None:
    """
    Fill each cell with two triangles:
      upper-left  = CV RMSE  (Blues)
      lower-right = CV MAE   (Oranges)
    """
    rmse_cm  = matplotlib.colormaps[RMSE_CMAP]
    mae_cm   = matplotlib.colormaps[MAE_CMAP]
    na_color = (0.88, 0.88, 0.88, 1.0)
    n_rows   = len(rmse_pivot.index)
    n_cols   = len(cols)

    for i, row_label in enumerate(rmse_pivot.index):
        for j, col_label in enumerate(cols):
            rmse_val = rmse_pivot.at[row_label, col_label] if col_label in rmse_pivot.columns else np.nan
            mae_val  = mae_pivot.at[row_label, col_label]  if col_label in mae_pivot.columns else np.nan

            try:
                rmse_nan = np.isnan(float(rmse_val))
            except (TypeError, ValueError):
                rmse_nan = True
            try:
                mae_nan = np.isnan(float(mae_val))
            except (TypeError, ValueError):
                mae_nan = True

            rmse_color = rmse_cm(rmse_norm(float(rmse_val))) if not rmse_nan else na_color
            mae_color  = mae_cm(mae_norm(float(mae_val)))    if not mae_nan else na_color

            ax.fill([j, j + 1, j],     [i, i, i + 1],         color=rmse_color, zorder=1)
            ax.fill([j + 1, j + 1, j], [i, i + 1, i + 1],     color=mae_color, zorder=1)
            ax.plot([j, j + 1], [i + 1, i], color="white", lw=0.8, zorder=2)

            if not rmse_nan:
                ax.text(
                    j + 0.28, i + 0.30, f"{float(rmse_val):.3f}",
                    ha="center", va="center", fontsize=ANNOT_FS,
                    color="white" if rmse_norm(float(rmse_val)) > 0.6 else "black", zorder=3,
                )
            if not mae_nan:
                ax.text(
                    j + 0.72, i + 0.72, f"{float(mae_val):.3f}",
                    ha="center", va="center", fontsize=ANNOT_FS,
                    color="white" if mae_norm(float(mae_val)) > 0.6 else "black", zorder=3,
                )

    for k in range(n_cols + 1):
        ax.axvline(k, color="white", lw=1.2, zorder=2)
    for k in range(n_rows + 1):
        ax.axhline(k, color="white", lw=1.2, zorder=2)

    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.invert_yaxis()
    ax.set_xticks(np.arange(n_cols) + 0.5)
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(rmse_pivot.index.tolist(), rotation=0, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.tick_params(axis="both", length=0)
    if show_xlabels:
        ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=9)
    else:
        ax.set_xticklabels([""] * n_cols)
        ax.tick_params(axis="x", bottom=False)


def _add_dual_colorbars(fig, rmse_norm, mae_norm) -> None:
    fig.subplots_adjust(right=0.82)

    cax_rmse = fig.add_axes([0.84, 0.10, 0.03, 0.78])
    sm_rmse  = mcm.ScalarMappable(cmap=RMSE_CMAP, norm=rmse_norm)
    sm_rmse.set_array([])
    fig.colorbar(sm_rmse, cax=cax_rmse, label="CV RMSE ◤ (Blues, lower is better)")

    cax_mae = fig.add_axes([0.90, 0.10, 0.03, 0.78])
    sm_mae  = mcm.ScalarMappable(cmap=MAE_CMAP, norm=mae_norm)
    sm_mae.set_array([])
    fig.colorbar(sm_mae, cax=cax_mae, label="CV MAE ◢ (Oranges, lower is better)")


def plot_per_model_heatmaps(df: pd.DataFrame) -> None:
    for model_type, grp in df.groupby("model_type"):
        pivot_rmse = grp.pivot_table(
            index="variant", columns="target", values="cv_rmse", aggfunc="mean"
        )
        pivot_mae = grp.pivot_table(
            index="variant", columns="target", values="cv_mae", aggfunc="mean"
        )
        present = set(pivot_rmse.columns) | set(pivot_mae.columns)
        cols    = [t for t in TARGETS if t in present] + sorted(present - set(TARGETS))
        pivot_rmse = pivot_rmse.reindex(columns=cols)
        pivot_mae  = pivot_mae.reindex(columns=cols)
        labels  = [c.replace("HOME_", "H_").replace("AWAY_", "A_").replace("_", "\n") for c in cols]

        all_rmse = pivot_rmse.values.flatten().astype(float)
        all_mae  = pivot_mae.values.flatten().astype(float)
        all_rmse = all_rmse[~np.isnan(all_rmse)]
        all_mae  = all_mae[~np.isnan(all_mae)]
        if len(all_rmse) == 0 or len(all_mae) == 0:
            continue
        rmse_norm = mcolors.Normalize(vmin=float(all_rmse.min()), vmax=float(all_rmse.max()))
        mae_norm  = mcolors.Normalize(vmin=float(all_mae.min()),  vmax=float(all_mae.max()))

        fig, ax = plt.subplots(
            figsize=(max(14, len(cols) * 1.1), max(4, len(pivot_rmse) * 0.75 + 1.5))
        )
        _draw_split_cells(
            ax,
            pivot_rmse, pivot_mae,
            cols,
            rmse_norm, mae_norm,
            show_xlabels=True,
            ylabel="Variant",
        )
        ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=8)
        _add_dual_colorbars(fig, rmse_norm, mae_norm)
        ax.set_title(
            f"CV RMSE (◤ Blues) & MAE (◢ Oranges) by Variant × Target  —  {model_type}",
            fontsize=13,
            pad=12,
        )
        ax.set_xlabel("")
        fig.subplots_adjust(left=0.08, bottom=0.12, top=0.92)
        out_path = PLOTS_DIR / f"heatmap_{model_type}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved {out_path}")


def plot_summary_heatmap(df: pd.DataFrame) -> None:
    summary_rmse = (
        df.groupby(["model_type", "variant"])["cv_rmse"]
        .mean().reset_index().rename(columns={"cv_rmse": "mean_cv_rmse"})
    )
    summary_mae = (
        df.groupby(["model_type", "variant"])["cv_mae"]
        .mean().reset_index().rename(columns={"cv_mae": "mean_cv_mae"})
    )
    pivot_rmse = summary_rmse.pivot(index="model_type", columns="variant", values="mean_cv_rmse")
    pivot_mae  = summary_mae.pivot(index="model_type", columns="variant", values="mean_cv_mae")

    all_variants = sorted(set(pivot_rmse.columns) | set(pivot_mae.columns))
    pivot_rmse = pivot_rmse.reindex(columns=all_variants)
    pivot_mae  = pivot_mae.reindex(columns=all_variants)

    all_rmse = pivot_rmse.values.flatten().astype(float)
    all_mae  = pivot_mae.values.flatten().astype(float)
    all_rmse = all_rmse[~np.isnan(all_rmse)]
    all_mae  = all_mae[~np.isnan(all_mae)]
    if len(all_rmse) == 0 or len(all_mae) == 0:
        return
    rmse_norm = mcolors.Normalize(vmin=float(all_rmse.min()), vmax=float(all_rmse.max()))
    mae_norm  = mcolors.Normalize(vmin=float(all_mae.min()),  vmax=float(all_mae.max()))

    fig, ax = plt.subplots(
        figsize=(max(8, len(pivot_rmse.columns) * 1.3), max(3, len(pivot_rmse) * 1.0 + 1.5))
    )
    _draw_split_cells(
        ax,
        pivot_rmse, pivot_mae,
        all_variants,
        rmse_norm, mae_norm,
        show_xlabels=True,
        ylabel="Model",
    )
    _add_dual_colorbars(fig, rmse_norm, mae_norm)
    ax.set_title("Mean CV RMSE (◤) & MAE (◢)  —  Model × Variant", fontsize=13, pad=12)
    ax.set_xlabel("Variant")
    fig.subplots_adjust(left=0.08, bottom=0.12, top=0.92)
    out_path = PLOTS_DIR / "heatmap_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def main():
    df = load_data()
    required = {"cv_rmse", "cv_mae"}
    if not required.issubset(df.columns):
        print(f"[ERROR] Required columns missing: {sorted(required - set(df.columns))}")
        return
    plot_per_model_heatmaps(df)
    plot_summary_heatmap(df)


if __name__ == "__main__":
    main()
