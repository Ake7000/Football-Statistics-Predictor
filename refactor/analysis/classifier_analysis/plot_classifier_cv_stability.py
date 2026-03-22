"""
plot_classifier_cv_stability.py
Two complementary stability views:

  1. Val F1 vs CV F1 scatter (one subplot per stat).
     Points above the diagonal y=x have a generalisation gap (CV worse than val).
     Marker shape = model_type, colour = class_weight_strategy.

  2. CV F1-macro std bar chart per model — which model is most stable across
     folds and runs?  One figure for fold-std, one for run-std.

Reads:  output/classifier_results_master.csv
Writes: output/plots/clf_cv_stability_scatter.png
        output/plots/clf_cv_stability_std.png

Usage:
    python refactor/analysis/classifier_analysis/plot_classifier_cv_stability.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # → refactor/
from shared_config import CLASSIFIER_STAT_PAIRS

OUTPUT_DIR = Path(__file__).parent / "output"
PLOTS_DIR  = OUTPUT_DIR / "plots"
MASTER_CSV = OUTPUT_DIR / "classifier_results_master.csv"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

STAT_ORDER = [s for s, _, _ in CLASSIFIER_STAT_PAIRS]

_PALETTE = [
    "#4878CF", "#6ACC65", "#D65F5F", "#E6701A", "#9B59B6",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]
_MARKERS = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "h"]
_CW_COLORS  = {"none": "#D65F5F", "sqrt": "#4878CF"}


def _model_styles(model_types) -> dict:
    return {
        m: (_MARKERS[i % len(_MARKERS)], _PALETTE[i % len(_PALETTE)])
        for i, m in enumerate(sorted(model_types))
    }


def load_data() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"classifier_results_master.csv not found. "
            f"Run collect_classifier_results.py first.\n  Expected: {MASTER_CSV}"
        )
    return pd.read_csv(MASTER_CSV)


def plot_val_vs_cv_scatter(df: pd.DataFrame) -> None:
    """Scatter: val_f1_macro_mean (x) vs cv_f1_macro_mean (y) per stat."""
    required = {"val_f1_macro_mean", "cv_f1_macro_mean"}
    if not required.issubset(df.columns):
        print(f"[ERROR] Missing columns for scatter: {required - set(df.columns)}")
        return

    present = set(df["stat"].unique())
    stats   = [s for s in STAT_ORDER if s in present] + sorted(present - set(STAT_ORDER))
    styles  = _model_styles(df["model_type"].unique())
    n       = len(stats)
    ncols   = 2
    nrows   = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
    axes = axes.flatten()

    for idx, stat in enumerate(stats):
        ax  = axes[idx]
        sub = df[df["stat"] == stat].dropna(subset=["val_f1_macro_mean", "cv_f1_macro_mean"])
        if sub.empty:
            ax.set_visible(False)
            continue

        all_vals = pd.concat([sub["val_f1_macro_mean"], sub["cv_f1_macro_mean"]]).dropna()
        lim_lo   = max(0.0, all_vals.min() * 0.92)
        lim_hi   = min(1.0, all_vals.max() * 1.08)
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8, alpha=0.4)
        ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], [lim_hi, lim_hi],
                        alpha=0.04, color="red")

        for model_type, mgrp in sub.groupby("model_type"):
            marker, _ = styles.get(model_type, ("o", "#999"))
            for cw, cgrp in mgrp.groupby("class_weight_strategy"):
                color = _CW_COLORS.get(cw, "#888")
                xerr  = cgrp.get("val_f1_macro_run_std", pd.Series([0]*len(cgrp))).fillna(0).values
                yerr  = cgrp.get("cv_f1_macro_run_std",  pd.Series([0]*len(cgrp))).fillna(0).values
                ax.errorbar(
                    cgrp["val_f1_macro_mean"], cgrp["cv_f1_macro_mean"],
                    xerr=xerr, yerr=yerr,
                    fmt=marker, color=color, alpha=0.85,
                    markersize=7, capsize=3,
                )
                for _, row in cgrp.iterrows():
                    label = row["model_type"].replace("classifier_", "")[:20]
                    ax.annotate(
                        label, (row["val_f1_macro_mean"], row["cv_f1_macro_mean"]),
                        fontsize=5, alpha=0.65, xytext=(3, 3), textcoords="offset points",
                    )

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.axhline(1/3, color="grey", lw=0.6, linestyle=":", alpha=0.5)
        ax.axvline(1/3, color="grey", lw=0.6, linestyle=":", alpha=0.5)
        ax.set_xlabel("Val F1-macro", fontsize=8)
        ax.set_ylabel("CV F1-macro",  fontsize=8)
        ax.set_title(stat, fontsize=9)

    for idx in range(len(stats), len(axes)):
        axes[idx].set_visible(False)

    marker_handles = [
        mlines.Line2D([], [], marker=m, color=c, linestyle="None", markersize=7, label=mt)
        for mt, (m, c) in styles.items()
    ]
    cw_handles = [
        mpatches.Patch(color=c, label=f"cw={cw}") for cw, c in _CW_COLORS.items()
    ]
    fig.legend(
        handles=marker_handles + cw_handles,
        loc="upper center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Val F1-macro vs CV F1-macro  (points above diagonal = generalisation gap)\n"
        "dotted lines = random baseline (0.333)",
        fontsize=11, y=1.04,
    )
    plt.tight_layout()
    out_path = PLOTS_DIR / "clf_cv_stability_scatter.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def plot_stability_bars(df: pd.DataFrame) -> None:
    """Bar chart: average cv_f1_macro_fold_std and cv_f1_macro_run_std per model."""
    for std_col, title_suffix, fname_suffix in [
        ("cv_f1_macro_fold_std", "Fold-to-Fold Std (within CV)", "fold_std"),
        ("cv_f1_macro_run_std",  "Run-to-Run Std",               "run_std"),
    ]:
        if std_col not in df.columns:
            continue
        summary = (
            df.groupby(["model_type", "class_weight_strategy"])[std_col]
            .mean()
            .reset_index()
        )
        if summary.empty:
            continue
        if summary[std_col].fillna(0).eq(0).all():
            print(f"[SKIP] {fname_suffix}: all values are 0 "
                  f"(need n_runs > 1 to compute run-to-run std)")
            continue

        models  = sorted(summary["model_type"].unique())
        cw_list = sorted(summary["class_weight_strategy"].unique())
        x       = np.arange(len(models))
        width   = 0.35
        colors  = [_CW_COLORS.get(cw, "#888") for cw in cw_list]

        fig, ax = plt.subplots(figsize=(max(8, len(models) * 1.2), 5))
        for j, (cw, color) in enumerate(zip(cw_list, colors)):
            vals = [
                summary.loc[
                    (summary["model_type"] == m) & (summary["class_weight_strategy"] == cw),
                    std_col
                ].values[0] if any(
                    (summary["model_type"] == m) & (summary["class_weight_strategy"] == cw)
                ) else 0.0
                for m in models
            ]
            offset = (j - (len(cw_list) - 1) / 2) * width
            bars = ax.bar(x + offset, vals, width=width * 0.9, color=color,
                          alpha=0.82, label=f"cw={cw}", edgecolor="white")
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                        f"{val:.4f}", ha="center", va="bottom", fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels(
            [m.replace("classifier_", "") for m in models],
            rotation=20, ha="right", fontsize=9,
        )
        ax.set_ylabel(std_col, fontsize=9)
        ax.set_title(f"CV F1-macro Stability  —  {title_suffix}\n(lower = more stable)", fontsize=11)
        ax.legend(fontsize=9)
        plt.tight_layout()
        out_path = PLOTS_DIR / f"clf_cv_stability_{fname_suffix}.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] Saved {out_path}")


def main():
    df = load_data()
    if "cv_f1_macro_mean" not in df.columns:
        print("[ERROR] 'cv_f1_macro_mean' column missing.")
        return
    plot_val_vs_cv_scatter(df)
    plot_stability_bars(df)


if __name__ == "__main__":
    main()
