"""
collect_results.py
Scans regression artifact roots for run_result.json files, averages metrics
across runs with the same (model_type × variant), and writes results_master.csv.

Column schema (long format — one row per model_type × variant × target):
  model_type, variant, target, n_runs, n_features, n_train, n_val, target_mean,
  val_rmse, val_rmse_run_std, val_mae, val_mae_run_std,
  val_rmse_pct, val_mae_pct,
  cv_rmse, cv_rmse_fold_std, cv_rmse_run_std,
  cv_mae,  cv_mae_fold_std,  cv_mae_run_std,
  cv_rmse_pct, cv_mae_pct

Usage:
    python refactor/analysis/optimizer_analysis/collect_results.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))   # refactor/ → shared_config

from shared_config import TARGETS, WORKSPACE_ROOT

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MASTER_CSV = OUTPUT_DIR / "results_master.csv"


def _discover_artifact_roots(workspace_root: Path) -> list[Path]:
    """
    Discover regression artifact roots.

    Supports both layouts:
      - New:    artifacts/regression/
      - Legacy: artifacts_* (excluding classification artifacts)
    """
    roots: list[Path] = []

    new_root = workspace_root / "artifacts" / "regression"
    if new_root.exists() and new_root.is_dir():
        roots.append(new_root)

    legacy_roots = sorted(
        p for p in workspace_root.iterdir()
        if p.is_dir()
        and p.name.startswith("artifacts_")
        and "classification" not in p.name
    )
    roots.extend(legacy_roots)
    return roots


def collect_all() -> pd.DataFrame:
    # key: (model_type, variant, target) → list of per-run metric dicts
    per_run: defaultdict = defaultdict(list)

    roots = _discover_artifact_roots(WORKSPACE_ROOT)
    if not roots:
        print(
            f"[WARN] No regression artifact roots found under {WORKSPACE_ROOT} "
            f"(checked artifacts/regression and legacy artifacts_*)"
        )
        return pd.DataFrame()
    for root in roots:
        for rr_path in sorted(root.rglob("run_result.json")):
            try:
                rr = json.loads(rr_path.read_text())
            except Exception as e:
                print(f"[WARN] Skipping {rr_path}: {e}")
                continue
            model_type = rr.get("model_type") or root.name.removeprefix("artifacts_")
            variant    = rr.get("variant", "unknown")
            n_features = rr.get("n_features")
            n_train    = rr.get("n_train")
            n_val      = rr.get("n_val")
            outcome_acc_by_stat = {m["stat"]: m["accuracy"] for m in rr.get("outcome_metrics", [])}
            for tentry in rr.get("targets", []):
                target = tentry.get("target")
                if not target:
                    continue
                stat = target.replace("HOME_", "").replace("AWAY_", "")
                per_run[(model_type, variant, target)].append({
                    "n_features":     n_features,
                    "n_train":        n_train,
                    "n_val":          n_val,
                    "target_mean":    tentry.get("target_mean"),
                    # val metrics (single value per run)
                    "val_rmse":       tentry.get("val_rmse"),
                    "val_mae":        tentry.get("val_mae"),
                    "val_rmse_pct":   tentry.get("val_rmse_pct"),
                    "val_mae_pct":    tentry.get("val_mae_pct"),
                    # cv metrics (fold-averaged within each run)
                    "cv_rmse_mean":   tentry.get("cv_rmse_mean"),
                    "cv_rmse_std":    tentry.get("cv_rmse_std"),    # within-run fold std
                    "cv_mae_mean":    tentry.get("cv_mae_mean"),
                    "cv_mae_std":     tentry.get("cv_mae_std"),     # within-run fold std
                    "cv_rmse_pct":    tentry.get("cv_rmse_pct_mean"),
                    "cv_mae_pct":     tentry.get("cv_mae_pct_mean"),
                    "val_round_acc":  tentry.get("val_round_acc"),
                    "val_outcome_acc": outcome_acc_by_stat.get(stat),
                })

    if not per_run:
        return pd.DataFrame()

    out_rows = []
    for (model_type, variant, target), runs in per_run.items():
        n_runs = len(runs)

        def _avg(field):
            vals = [r[field] for r in runs if r.get(field) is not None]
            return float(np.mean(vals)) if vals else None

        def _std(field):
            vals = [r[field] for r in runs if r.get(field) is not None]
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        out = {
            "model_type":        model_type,
            "variant":           variant,
            "target":            target,
            "n_runs":            n_runs,
            "n_features":        runs[0]["n_features"],
            "n_train":           runs[0]["n_train"],
            "n_val":             runs[0]["n_val"],
            "target_mean":       runs[0]["target_mean"],   # constant across runs
            # val metrics — mean and run-to-run std
            "val_rmse":          _avg("val_rmse"),
            "val_rmse_run_std":  _std("val_rmse"),
            "val_mae":           _avg("val_mae"),
            "val_mae_run_std":   _std("val_mae"),
            "val_rmse_pct":      _avg("val_rmse_pct"),
            "val_mae_pct":       _avg("val_mae_pct"),
            # cv metrics
            # cv_rmse: average of per-run fold-means
            # cv_rmse_fold_std: average of per-run fold stds (typical fold-to-fold variance)
            # cv_rmse_run_std: std of per-run fold-means (run-to-run variance)
            "cv_rmse":           _avg("cv_rmse_mean"),
            "cv_rmse_fold_std":  _avg("cv_rmse_std"),
            "cv_rmse_run_std":   _std("cv_rmse_mean"),
            "cv_mae":            _avg("cv_mae_mean"),
            "cv_mae_fold_std":   _avg("cv_mae_std"),
            "cv_mae_run_std":    _std("cv_mae_mean"),
            "cv_rmse_pct":       _avg("cv_rmse_pct"),
            "cv_mae_pct":        _avg("cv_mae_pct"),
            "val_round_acc":     _avg("val_round_acc"),
            "val_outcome_acc":   _avg("val_outcome_acc"),
        }
        out_rows.append(out)

    df = pd.DataFrame(out_rows)
    df = df.sort_values(["model_type", "variant", "target"]).reset_index(drop=True)
    return df


def main():
    print(
        f"[COLLECT] Scanning regression artifacts under {WORKSPACE_ROOT} "
        f"for run_result.json …"
    )
    df = collect_all()
    if df.empty:
        print("[WARN] No run_result.json files found. Run the optimizers first.")
        return
    df.to_csv(MASTER_CSV, index=False)
    n_combos = df[["model_type", "variant"]].drop_duplicates().shape[0]
    print(f"[COLLECT] {len(df)} rows ({n_combos} model×variant combos) → {MASTER_CSV}")


if __name__ == "__main__":
    main()
