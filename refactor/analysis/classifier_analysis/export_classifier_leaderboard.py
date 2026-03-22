"""
export_classifier_leaderboard.py
Rank all model × variant × class_weight_strategy × stat combinations by
cv_f1_macro_mean (higher = better).

Outputs:
  output/tables/clf_leaderboard.csv         — full ranked list
  output/tables/clf_leaderboard_top3.csv    — top-3 per stat
  output/tables/clf_leaderboard.tex         — LaTeX table of top-3 per stat
  output/tables/clf_cw_comparison.csv       — cw strategy pivot

Usage:
    python refactor/analysis/classifier_analysis/export_classifier_leaderboard.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # → refactor/
from shared_config import CLASSIFIER_STAT_PAIRS

OUTPUT_DIR   = Path(__file__).parent / "output"
TABLES_DIR   = OUTPUT_DIR / "tables"
MASTER_CSV   = OUTPUT_DIR / "classifier_results_master.csv"
TABLES_DIR.mkdir(parents=True, exist_ok=True)

LEADERBOARD_CSV      = TABLES_DIR / "clf_leaderboard.csv"
LEADERBOARD_TOP3_CSV = TABLES_DIR / "clf_leaderboard_top3.csv"
LEADERBOARD_TEX      = TABLES_DIR / "clf_leaderboard.tex"

STAT_ORDER = [s for s, _, _ in CLASSIFIER_STAT_PAIRS]

LEADERBOARD_COLS = [
    "model_type",
    "variant",
    "class_weight_strategy",
    "stat",
    "rank",
    "cv_f1_macro_mean",
    "cv_f1_macro_run_std",
    "cv_f1_macro_fold_std",
    "val_f1_macro_mean",
    "val_f1_macro_run_std",
    "n_runs",
]


def load_data() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"classifier_results_master.csv not found. "
            f"Run collect_classifier_results.py first.\n  Expected: {MASTER_CSV}"
        )
    return pd.read_csv(MASTER_CSV)


def build_leaderboard(df: pd.DataFrame) -> pd.DataFrame:
    present = set(df["stat"].unique())
    stats   = [s for s in STAT_ORDER if s in present] + sorted(present - set(STAT_ORDER))

    parts = []
    for stat in stats:
        stat_df = df[df["stat"] == stat].sort_values(
            "cv_f1_macro_mean", ascending=False, na_position="last"
        ).reset_index(drop=True)
        stat_df["rank"] = stat_df["cv_f1_macro_mean"].rank(
            method="min", ascending=False, na_option="bottom"
        ).astype(int)
        parts.append(stat_df)

    leaderboard = pd.concat(parts, ignore_index=True)
    # Keep only columns that exist
    existing = [c for c in LEADERBOARD_COLS if c in leaderboard.columns]
    return leaderboard[existing]


def export_full(lb: pd.DataFrame) -> None:
    lb.to_csv(LEADERBOARD_CSV, index=False)
    print(f"[CSV] Saved {LEADERBOARD_CSV}  ({len(lb)} rows)")


def export_top3(lb: pd.DataFrame) -> None:
    top3 = lb[lb["rank"] <= 3].sort_values(["stat", "rank"])
    top3.to_csv(LEADERBOARD_TOP3_CSV, index=False)
    print(f"[CSV] Saved {LEADERBOARD_TOP3_CSV}  ({len(top3)} rows)")
    return top3


def _short_model(name: str) -> str:
    return (name.replace("classifier_", "")
                .replace("_multioutput", "-MO")
                .replace("_torch", "")
                .replace("lstm_mlp", "LSTM-MLP")
                .replace("mlp", "MLP")
                .replace("xgb", "XGB"))


def _short_variant(name: str) -> str:
    parts = name.split("_")
    return "_".join(parts[:5]) + ("…" if len(parts) > 5 else "")


def export_latex(top3: pd.DataFrame) -> None:
    lines = []
    lines.append(r"\begin{table}[htbp]")
    lines.append(r"\centering")
    lines.append(r"\caption{Top-3 classifiers per target statistic (ranked by mean CV F1-macro)}")
    lines.append(r"\label{tab:clf_leaderboard}")
    lines.append(r"\small")
    lines.append(r"\begin{tabular}{llllrrrr}")
    lines.append(r"\toprule")
    lines.append(
        r"\textbf{Stat} & \textbf{Rank} & \textbf{Model} & \textbf{CW} & "
        r"\textbf{CV F1↑} & \textbf{CV σ\textsubscript{run}} & "
        r"\textbf{Val F1↑} & \textbf{n} \\"
    )
    lines.append(r"\midrule")

    present = set(top3["stat"].unique())
    stats   = [s for s in STAT_ORDER if s in present] + sorted(present - set(STAT_ORDER))

    for stat in stats:
        rows = top3[top3["stat"] == stat].sort_values("rank")
        for i, (_, row) in enumerate(rows.iterrows()):
            stat_cell = stat if i == 0 else ""
            model     = _short_model(row.get("model_type", ""))
            cw        = row.get("class_weight_strategy", "")
            cv_mean   = row.get("cv_f1_macro_mean", float("nan"))
            cv_std    = row.get("cv_f1_macro_run_std", float("nan"))
            val_mean  = row.get("val_f1_macro_mean", float("nan"))
            n_runs    = int(row.get("n_runs", 0))
            rank      = int(row.get("rank", 0))
            line = (
                f"{stat_cell} & {rank} & {model} & {cw} & "
                f"{cv_mean:.4f} & {cv_std:.4f} & {val_mean:.4f} & {n_runs} \\\\"
            )
            lines.append(line)
        if stat != stats[-1]:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    LEADERBOARD_TEX.write_text("\n".join(lines), encoding="utf-8")
    print(f"[TEX] Saved {LEADERBOARD_TEX}")


def cw_comparison_table(df: pd.DataFrame) -> None:
    """
    Print a pivot showing mean cv_f1_macro_mean per (model_type, class_weight_strategy)
    averaged across all stats and variants.
    """
    pivot = (
        df.groupby(["model_type", "class_weight_strategy"])["cv_f1_macro_mean"]
        .mean()
        .unstack("class_weight_strategy")
    )
    pivot.index = [_short_model(m) for m in pivot.index]
    pivot.columns = [f"cw={c}" for c in pivot.columns]
    print("\n--- CW strategy comparison (mean CV F1-macro) ---")
    print(pivot.round(4).to_string())
    print()

    cw_csv = TABLES_DIR / "clf_cw_comparison.csv"
    pivot.to_csv(cw_csv)
    print(f"[CSV] Saved {cw_csv}")


def main():
    df = load_data()
    lb = build_leaderboard(df)
    export_full(lb)
    top3 = export_top3(lb)
    export_latex(top3)
    cw_comparison_table(df)
    print("[DONE] Leaderboard export complete.")


if __name__ == "__main__":
    main()
