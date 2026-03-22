"""
plot_ablation.py
Feature-group contribution analysis via ΔRMSE between variant pairs.

The current study variants define this progression:
    cform...stage_sum      → cform...raw_stage_sum       : adds raw slots
    cform...stage_sum      → cform...odds_stage_sum      : adds odds
    cform...raw_stage_sum  → cform...odds_raw_stage_sum  : adds odds
    cform...odds_stage_sum → cform...odds_raw_stage_sum  : adds raw slots

For each comparison and each target, ΔCVRMSE = enriched - base.
  Negative = adding the group improves predictions.
  Positive = adding the group hurts (noise / overfitting).

One figure per model_type. Also saves a summary figure averaging ΔRMSE across targets.

Reads:  output/results_master.csv
Writes: output/plots/ablation.png
        output/plots/ablation_summary.png

Usage:
    python refactor/analysis/optimizer_analysis/plot_ablation.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # refactor/ → shared_config
from shared_config import TARGETS

OUTPUT_DIR = Path(__file__).parent / "output"
PLOTS_DIR  = OUTPUT_DIR / "plots"
MASTER_CSV = OUTPUT_DIR / "results_master.csv"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# (label, base_variant, enriched_variant, color)
# Add new variant-pair comparisons here; pairs not found in data are silently skipped.
COMPARISONS = [
    ("Base → +raw",
     "cform_diffmean_diffsum_form_mean_nplayers_stage_sum",
     "cform_diffmean_diffsum_form_mean_nplayers_raw_stage_sum",
     "#4878CF"),
    ("Base → +odds",
     "cform_diffmean_diffsum_form_mean_nplayers_stage_sum",
     "cform_diffmean_diffsum_form_mean_nplayers_odds_stage_sum",
     "#6ACC65"),
    ("+raw → +raw+odds",
     "cform_diffmean_diffsum_form_mean_nplayers_raw_stage_sum",
     "cform_diffmean_diffsum_form_mean_nplayers_odds_raw_stage_sum",
     "#D65F5F"),
    ("+odds → +odds+raw",
     "cform_diffmean_diffsum_form_mean_nplayers_odds_stage_sum",
     "cform_diffmean_diffsum_form_mean_nplayers_odds_raw_stage_sum",
     "#E6AC27"),
]

_PALETTE = [
    "#4878CF", "#6ACC65", "#D65F5F", "#E6701A", "#9B59B6",
    "#8C564B", "#E377C2", "#7F7F7F", "#BCBD22", "#17BECF",
]


def load_data() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"results_master.csv not found. Run collect_results.py first.\n  Expected: {MASTER_CSV}"
        )
    return pd.read_csv(MASTER_CSV)


def compute_deltas(model_df: pd.DataFrame, targets: list) -> pd.DataFrame:
    """Return a DataFrame with columns: comparison_label, target, delta_cv_rmse."""
    rows = []
    idx_df = model_df.set_index(["variant", "target"])
    for label, base_v, enrich_v, _ in COMPARISONS:
        for t in targets:
            base_rmse   = idx_df.loc[(base_v, t),   "cv_rmse"] if (base_v,   t) in idx_df.index else np.nan
            enrich_rmse = idx_df.loc[(enrich_v, t), "cv_rmse"] if (enrich_v, t) in idx_df.index else np.nan
            if pd.notna(base_rmse) and pd.notna(enrich_rmse):
                rows.append({"comparison": label, "target": t,
                              "delta": enrich_rmse - base_rmse})
    if not rows:
        return pd.DataFrame(columns=["comparison", "target", "delta"])
    return pd.DataFrame(rows)


def plot_ablation_model(ax, model_df: pd.DataFrame, model_type: str) -> None:
    present = set(model_df["target"].unique())
    targets = [t for t in TARGETS if t in present] + sorted(present - set(TARGETS))
    x       = np.arange(len(targets))
    n_comp  = len(COMPARISONS)
    width   = 0.8 / n_comp
    offsets = np.linspace(-(n_comp - 1) * width / 2, (n_comp - 1) * width / 2, n_comp)

    deltas_df = compute_deltas(model_df, targets)

    if deltas_df.empty:
        ax.axhline(0, color="black", linewidth=0.9, alpha=0.7)
        ax.set_xticks([])
        ax.set_ylabel("ΔCVRMSE (negative = improvement)", fontsize=9)
        ax.set_title(f"Feature Group Ablation  —  {model_type}", fontsize=11)
        ax.text(
            0.5, 0.5,
            "No configured ablation comparisons\nare present in current results.",
            transform=ax.transAxes, ha="center", va="center", fontsize=9, alpha=0.8,
        )
        return

    for i, (label, base_v, enrich_v, color) in enumerate(COMPARISONS):
        sub = deltas_df[deltas_df["comparison"] == label].set_index("target")
        vals = [sub.loc[t, "delta"] if t in sub.index else np.nan for t in targets]
        ax.bar(x + offsets[i], vals, width=width, color=color, alpha=0.8,
               label=label, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [t.replace("HOME_", "H_").replace("AWAY_", "A_").replace("_", "\n")
         for t in targets],
        fontsize=7,
    )
    ax.set_ylabel("ΔCVRMSE (negative = improvement)", fontsize=9)
    ax.set_title(f"Feature Group Ablation  —  {model_type}", fontsize=11)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.7)


def plot_summary(df: pd.DataFrame) -> None:
    """Mean ΔCVRMSE across all targets, one bar per comparison per model."""
    model_types = sorted(df["model_type"].unique())
    n_comp      = len(COMPARISONS)
    n_models    = len(model_types)
    x           = np.arange(n_comp)
    width       = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(12, 5))
    offsets = np.linspace(-(n_models - 1) * width / 2, (n_models - 1) * width / 2, n_models)
    for mi, model_type in enumerate(model_types):
        mdf    = df[df["model_type"] == model_type]
        means  = []
        for label, base_v, enrich_v, _ in COMPARISONS:
            base_cv   = mdf[mdf["variant"] == base_v]["cv_rmse"].mean()
            enrich_cv = mdf[mdf["variant"] == enrich_v]["cv_rmse"].mean()
            means.append(enrich_cv - base_cv if pd.notna(base_cv) and pd.notna(enrich_cv) else np.nan)
        ax.bar(x + offsets[mi], means, width=width,
               color=_PALETTE[mi % len(_PALETTE)],
               alpha=0.85, label=model_type, edgecolor="white")

    ax.axhline(0, color="black", linewidth=0.9, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in COMPARISONS], fontsize=8)
    ax.set_ylabel("Mean ΔCVRMSE across all targets", fontsize=10)
    ax.set_title("Feature Group Contribution Summary (mean across targets)", fontsize=12)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = PLOTS_DIR / "ablation_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")


def main():
    df = load_data()
    if "cv_rmse" not in df.columns:
        print("[ERROR] 'cv_rmse' column missing.")
        return

    model_types = sorted(df["model_type"].unique())
    n           = len(model_types)
    fig, axes   = plt.subplots(n, 1, figsize=(16, n * 5.5))
    if n == 1:
        axes = [axes]

    for ax, model_type in zip(axes, model_types):
        plot_ablation_model(ax, df[df["model_type"] == model_type].copy(), model_type)

    plt.tight_layout()
    out_path = PLOTS_DIR / "ablation.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved {out_path}")

    plot_summary(df)


if __name__ == "__main__":
    main()
