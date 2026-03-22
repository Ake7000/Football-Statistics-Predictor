"""
plot_classifier_rankings.py
For each stat: horizontal bar chart ranking all model×variant×cw_strategy combos
by CV F1-macro (best = top). Error bars = cv_f1_macro_run_std. Bars coloured by
model_type; bar pattern (solid/hatched) encodes class_weight_strategy.

Reads:  output/classifier_results_master.csv
Writes: output/plots/clf_rankings_<stat>.png  (one file per stat)
        output/plots/clf_rankings_overview.png (all stats as subplots)

Usage:
    python refactor/analysis/classifier_analysis/plot_classifier_rankings.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
_CW_HATCH = {"none": "", "sqrt": "//"}


def _model_colors(model_types) -> dict:
    return {m: _PALETTE[i % len(_PALETTE)] for i, m in enumerate(sorted(model_types))}


def load_data() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"classifier_results_master.csv not found. "
            f"Run collect_classifier_results.py first.\n  Expected: {MASTER_CSV}"
        )
    return pd.read_csv(MASTER_CSV)


def _plot_stat_ranking(ax, stat_df: pd.DataFrame, stat: str, colors: dict) -> None:
    stat_df = stat_df.dropna(subset=["cv_f1_macro_mean"]).copy()
    stat_df["label"] = (
        stat_df["model_type"].str.replace("classifier_", "", regex=False)
        + "\n" + stat_df["variant"].str[:35]
        + "\n(cw=" + stat_df["class_weight_strategy"] + ")"
    )
    stat_df = stat_df.sort_values("cv_f1_macro_mean", ascending=False).reset_index(drop=True)

    labels     = stat_df["label"].tolist()
    values     = stat_df["cv_f1_macro_mean"].values
    errs       = stat_df["cv_f1_macro_run_std"].fillna(0).values if "cv_f1_macro_run_std" in stat_df else np.zeros(len(values))
    bar_colors = [colors.get(m, "#999") for m in stat_df["model_type"]]
    hatches    = [_CW_HATCH.get(cw, "") for cw in stat_df["class_weight_strategy"]]

    y = np.arange(len(labels))
    for yi, (val, err, col, hatch) in enumerate(zip(values, errs, bar_colors, hatches)):
        ax.barh(yi, val, xerr=err, color=col, hatch=hatch, alpha=0.82,
                edgecolor="white", height=0.7, capsize=3)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=6)
    ax.set_xlabel("CV F1-macro", fontsize=8)
    ax.set_title(stat, fontsize=10, pad=4)
    ax.set_xlim(0, 1.0)
    ax.invert_yaxis()   # best at top

    # Chance level reference line (1/3 for 3-class)
    ax.axvline(1 / 3, color="grey", linestyle="--", lw=0.8, alpha=0.5)


def plot_individual_stat(df: pd.DataFrame, stat: str, colors: dict) -> None:
    stat_df = df[df["stat"] == stat].copy()
    if stat_df.empty:
        return
    n_bars = len(stat_df)
    fig, ax = plt.subplots(figsize=(10, max(4, n_bars * 0.55 + 1.5)))
    _plot_stat_ranking(ax, stat_df, stat, colors)

    # Legend
    model_handles = [mpatches.Patch(color=c, label=m) for m, c in colors.items()
                     if m in stat_df["model_type"].values]
    cw_handles    = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=h, label=f"cw={cw}")
        for cw, h in _CW_HATCH.items()
    ]
    ax.legend(handles=model_handles + cw_handles, fontsize=7,
              loc="lower right", ncol=2)

    plt.tight_layout()
    out_path = PLOTS_DIR / f"clf_rankings_{stat}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def plot_overview(df: pd.DataFrame, stats: list, colors: dict) -> None:
    n = len(stats)
    ncols = 2
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 5))
    axes = axes.flatten()

    for idx, stat in enumerate(stats):
        stat_df = df[df["stat"] == stat].copy()
        _plot_stat_ranking(axes[idx], stat_df, stat, colors)

    for idx in range(len(stats), len(axes)):
        axes[idx].set_visible(False)

    # Legend
    model_handles = [mpatches.Patch(color=c, label=m) for m, c in colors.items()]
    cw_handles    = [
        mpatches.Patch(facecolor="white", edgecolor="black", hatch=h, label=f"cw={cw}")
        for cw, h in _CW_HATCH.items()
    ]
    fig.legend(handles=model_handles + cw_handles, loc="upper center",
               ncol=4, fontsize=8, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(
        "Per-Stat Rankings: CV F1-macro across all model×variant×cw combos\n"
        "(-- line = random baseline 0.333)",
        fontsize=12, y=1.04,
    )
    plt.tight_layout()
    out_path = PLOTS_DIR / "clf_rankings_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def main():
    df = load_data()
    if "cv_f1_macro_mean" not in df.columns:
        print("[ERROR] 'cv_f1_macro_mean' column missing.")
        return

    present = set(df["stat"].unique())
    stats   = [s for s in STAT_ORDER if s in present] + sorted(present - set(STAT_ORDER))
    colors  = _model_colors(df["model_type"].unique())

    for stat in stats:
        plot_individual_stat(df, stat, colors)

    plot_overview(df, stats, colors)


if __name__ == "__main__":
    main()
