"""
plot_rankings.py
For each prediction target: horizontal bar chart ranking all model×variant combos
by CV RMSE (best = left). Error bars = cv_rmse_run_std. Bars coloured by model_type.

Reads:  output/results_master.csv
Writes: output/plots/rankings_all.png  (all targets as subplots)

Usage:
    python refactor/analysis/optimizer_analysis/plot_rankings.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


def _model_colors(model_types) -> dict:
    """Assign consistent colors to model types from a fixed palette."""
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(sorted(model_types))}


def load_data() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"results_master.csv not found. Run collect_results.py first.\n  Expected: {MASTER_CSV}"
        )
    return pd.read_csv(MASTER_CSV)


def plot_target_ranking(ax, target_df: pd.DataFrame, target: str, colors: dict) -> None:
    target_df  = target_df.dropna(subset=["cv_rmse"]).sort_values("cv_rmse", ascending=True)
    # best at top of horizontal bar → invert so best is at top visually
    labels     = [f"{r.model_type}\n{r.variant}" for r in target_df.itertuples()]
    values     = target_df["cv_rmse"].values
    errs       = target_df["cv_rmse_run_std"].fillna(0).values if "cv_rmse_run_std" in target_df else np.zeros(len(values))
    bar_colors = [colors.get(m, "#999999") for m in target_df["model_type"]]

    y = np.arange(len(labels))
    ax.barh(y, values, xerr=errs, color=bar_colors, alpha=0.85,
            edgecolor="white", height=0.65, capsize=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6.5)
    ax.set_xlabel("CV RMSE", fontsize=8)
    ax.set_title(target.replace("_", " "), fontsize=9, pad=4)
    ax.invert_yaxis()   # best (lowest RMSE) at top


def main():
    df = load_data()
    if "cv_rmse" not in df.columns:
        print("[ERROR] 'cv_rmse' column missing.")
        return

    present = set(df["target"].unique())
    targets = [t for t in TARGETS if t in present] + sorted(present - set(TARGETS))
    colors  = _model_colors(df["model_type"].unique())
    n       = len(targets)
    ncols   = 2
    nrows   = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 5))
    axes = axes.flatten()

    for idx, target in enumerate(targets):
        sub = df[df["target"] == target].copy()
        plot_target_ranking(axes[idx], sub, target, colors)

    for idx in range(len(targets), len(axes)):
        axes[idx].set_visible(False)

    # Legend
    handles = [mpatches.Patch(color=c, label=m) for m, c in colors.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Per-Target Rankings: CV RMSE across all model×variant combos",
                 fontsize=13, y=1.04)
    plt.tight_layout()
    out_path = PLOTS_DIR / "rankings_all.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


if __name__ == "__main__":
    main()
