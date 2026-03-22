"""
plot_cv_stability.py
Scatter plot: val_rmse (x) vs cv_rmse (y) for each model×variant combo.
Points above the diagonal y=x have a generalisation gap (CV worse than val).
Error bars = ± run_std. Marker shape encodes model_type.

One figure with 14 subplots (one per target).

Reads:  output/results_master.csv
Writes: output/plots/cv_stability.png

Usage:
    python refactor/analysis/optimizer_analysis/plot_cv_stability.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # refactor/ → shared_config
from shared_config import TARGETS

OUTPUT_DIR = Path(__file__).parent / "output"
PLOTS_DIR  = OUTPUT_DIR / "plots"
MASTER_CSV = OUTPUT_DIR / "results_master.csv"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

_PALETTE = [
    "#4878CF", "#6ACC65", "#D65F5F", "#E6701A", "#9B59B6",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]
_MARKERS = ["o", "^", "s", "D", "P", "X", "v", "<", ">", "h"]


def _model_styles(model_types) -> dict:
    """Assign consistent (marker, color) pairs to model types."""
    return {
        m: (_MARKERS[i % len(_MARKERS)], _PALETTE[i % len(_PALETTE)])
        for i, m in enumerate(sorted(model_types))
    }


def load_data() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"results_master.csv not found. Run collect_results.py first.\n  Expected: {MASTER_CSV}"
        )
    return pd.read_csv(MASTER_CSV)


def main():
    df = load_data()
    required = {"val_rmse", "cv_rmse"}
    if not required.issubset(df.columns):
        print(f"[ERROR] Required columns {required} not all present.")
        return

    present = set(df["target"].unique())
    targets = [t for t in TARGETS if t in present] + sorted(present - set(TARGETS))
    styles  = _model_styles(df["model_type"].unique())
    n       = len(targets)
    ncols   = 2
    nrows   = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, nrows * 4))
    axes = axes.flatten()

    for idx, target in enumerate(targets):
        ax  = axes[idx]
        sub = df[df["target"] == target].dropna(subset=["val_rmse", "cv_rmse"])

        if sub.empty:
            ax.set_visible(False)
            continue

        all_vals = pd.concat([sub["val_rmse"], sub["cv_rmse"]]).dropna()
        lim_lo   = all_vals.min() * 0.93
        lim_hi   = all_vals.max() * 1.07
        ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=0.8, alpha=0.4)
        ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi], [lim_hi, lim_hi],
                        alpha=0.04, color="red", label="_nolegend_")

        for model_type, mgrp in sub.groupby("model_type"):
            marker, color = styles.get(model_type, ("o", "#999"))
            xerr = mgrp.get("val_rmse_run_std", pd.Series([0]*len(mgrp))).fillna(0).values
            yerr = mgrp.get("cv_rmse_run_std",  pd.Series([0]*len(mgrp))).fillna(0).values
            ax.errorbar(
                mgrp["val_rmse"], mgrp["cv_rmse"],
                xerr=xerr, yerr=yerr,
                fmt=marker, color=color, alpha=0.85,
                markersize=7, capsize=3, label=model_type,
            )
            # Annotate variant name next to each point
            for _, row in mgrp.iterrows():
                ax.annotate(
                    row["variant"], (row["val_rmse"], row["cv_rmse"]),
                    fontsize=5.5, alpha=0.7,
                    xytext=(3, 3), textcoords="offset points",
                )

        ax.set_xlim(lim_lo, lim_hi)
        ax.set_ylim(lim_lo, lim_hi)
        ax.set_xlabel("Val RMSE", fontsize=8)
        ax.set_ylabel("CV RMSE",  fontsize=8)
        ax.set_title(target.replace("_", " "), fontsize=9)
        ax.legend(fontsize=6)

    for idx in range(len(targets), len(axes)):
        axes[idx].set_visible(False)

    # Legend
    handles = [
        mlines.Line2D([], [], marker=m, color=c, linestyle="None",
                      markersize=7, label=mt)
        for mt, (m, c) in styles.items()
    ]
    handles.append(mlines.Line2D([], [], linestyle="--", color="black", lw=0.8, label="y=x (no gap)"))
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Val RMSE vs CV RMSE  (points above diagonal = generalisation gap)",
                 fontsize=12, y=1.04)
    plt.tight_layout()
    out_path = PLOTS_DIR / "cv_stability.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


if __name__ == "__main__":
    main()
