"""
export_leaderboard.py
Builds a leaderboard: for each target, all model×variant combos ranked by CV RMSE.
Reports mean ± std across runs, RMSE%, and MAE.

Outputs:
  output/tables/leaderboard.csv        — full ranking (all combos, all targets)
  output/tables/leaderboard_top3.csv   — top-3 per target
  output/tables/leaderboard.tex        — LaTeX booktabs table (top-3 per target)

Usage:
    python refactor/analysis/optimizer_analysis/export_leaderboard.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # refactor/ → shared_config
from shared_config import TARGETS

OUTPUT_DIR = Path(__file__).parent / "output"
TABLES_DIR = OUTPUT_DIR / "tables"
MASTER_CSV = OUTPUT_DIR / "results_master.csv"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not MASTER_CSV.exists():
        raise FileNotFoundError(
            f"results_master.csv not found. Run collect_results.py first.\n  Expected: {MASTER_CSV}"
        )
    return pd.read_csv(MASTER_CSV)


def build_leaderboard(df: pd.DataFrame, top_k: int = 3) -> pd.DataFrame:
    keep = ["model_type", "variant", "target",
            "cv_rmse", "cv_rmse_run_std", "cv_rmse_fold_std",
            "cv_mae",  "cv_mae_run_std",
            "cv_rmse_pct", "cv_mae_pct",
            "val_rmse", "val_mae", "n_runs"]
    sub = df[[c for c in keep if c in df.columns]].dropna(subset=["cv_rmse"]).copy()

    rows_top = []
    rows_all = []
    present  = set(sub["target"].values)
    ordered  = [t for t in TARGETS if t in present] + sorted(present - set(TARGETS))
    for target in ordered:
        tdf = sub[sub["target"] == target].sort_values("cv_rmse").reset_index(drop=True)
        tdf["rank"] = tdf.index + 1
        rows_all.append(tdf)
        rows_top.append(tdf.head(top_k))

    full = pd.concat(rows_all, ignore_index=True) if rows_all else pd.DataFrame()
    top3 = pd.concat(rows_top, ignore_index=True) if rows_top else pd.DataFrame()

    col_order = ["target", "rank", "model_type", "variant"] + \
                [c for c in keep if c not in ("model_type", "variant", "target")]
    for df_out in (full, top3):
        if not df_out.empty:
            existing = [c for c in col_order if c in df_out.columns]
            ordered_cols = existing + [c for c in df_out.columns if c not in existing]
            reordered = df_out[ordered_cols].copy()
            df_out.drop(columns=df_out.columns.tolist(), inplace=True)
            for c in reordered.columns:
                df_out[c] = reordered[c].values

    return full, top3


def _fmt(val, decimals=4):
    if val is None:
        return "--"
    try:
        fval = float(val)
    except (TypeError, ValueError):
        return "--"
    if pd.isna(fval):
        return "--"
    return f"{fval:.{decimals}f}"


def to_latex(top3: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[ht]",
        r"\centering",
        r"\caption{Top-3 model$\times$variant per target ranked by CV RMSE (mean across runs)}",
        r"\label{tab:leaderboard}",
        r"\small",
        r"\begin{tabular}{llllrrrr}",
        r"\toprule",
        r"Target & Rank & Model & Variant & CV RMSE & $\pm$std & CV MAE & RMSE\,\% \\",
        r"\midrule",
    ]
    prev_target = None
    for _, row in top3.iterrows():
        target = row["target"]
        if target != prev_target and prev_target is not None:
            lines.append(r"\midrule")
        prev_target = target
        tname   = target.replace("_", r"\_") if int(row.get("rank", 1)) == 1 else ""
        rank    = int(row.get("rank", ""))
        model   = str(row["model_type"]).replace("_", r"\_")
        variant = str(row["variant"])
        cv_rmse = _fmt(row.get("cv_rmse"))
        cv_std  = _fmt(row.get("cv_rmse_run_std", 0))
        cv_mae  = _fmt(row.get("cv_mae"))
        cv_pct  = f"{row['cv_rmse_pct']:.1f}\\%" if pd.notna(row.get("cv_rmse_pct")) else "--"
        lines.append(
            f"{tname} & {rank} & {model} & {variant} & "
            f"{cv_rmse} & {cv_std} & {cv_mae} & {cv_pct} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    df = load_data()
    full, top3 = build_leaderboard(df, top_k=3)

    if top3.empty:
        print("[WARN] Leaderboard is empty. Check results_master.csv.")
        return

    top3.to_csv(TABLES_DIR / "leaderboard_top3.csv", index=False)
    print(f"[TABLE] Saved {TABLES_DIR / 'leaderboard_top3.csv'}")

    if not full.empty:
        full.to_csv(TABLES_DIR / "leaderboard.csv", index=False)
        print(f"[TABLE] Saved {TABLES_DIR / 'leaderboard.csv'}")

    latex = to_latex(top3)
    (TABLES_DIR / "leaderboard.tex").write_text(latex)
    print(f"[TABLE] Saved {TABLES_DIR / 'leaderboard.tex'}")


if __name__ == "__main__":
    main()
