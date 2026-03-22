"""
collect_classifier_results.py
Scans artifacts/classification/ for run_result.json files, aggregates metrics
across runs sharing the same (model_type × variant × class_weight_strategy),
and writes classifier_results_master.csv.

Column schema (long format — one row per model_type × variant × cw_strategy × stat):
  model_type, variant, class_weight_strategy, stat, n_runs,
  n_features, n_train, n_val,
  val_acc_mean, val_acc_run_std,
  val_f1_macro_mean, val_f1_macro_run_std,
  cv_acc_mean, cv_acc_run_std,
  cv_acc_fold_std,
  cv_f1_macro_mean, cv_f1_macro_run_std,
  cv_f1_macro_fold_std

Usage:
    python refactor/analysis/classifier_analysis/collect_classifier_results.py
"""

import sys
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# refactor/ on path so shared_config is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared_config import ARTIFACTS_CLASSIFICATION_ROOT

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
MASTER_CSV = OUTPUT_DIR / "classifier_results_master.csv"


def collect_all() -> pd.DataFrame:
    # key: (model_type, variant, cw_strategy, stat) → list of per-run metric dicts
    per_run: defaultdict = defaultdict(list)

    root = ARTIFACTS_CLASSIFICATION_ROOT
    if not root.exists():
        print(f"[WARN] Classification artifacts root not found: {root}")
        return pd.DataFrame()

    for rr_path in sorted(root.rglob("run_result.json")):
        try:
            rr = json.loads(rr_path.read_text())
        except Exception as e:
            print(f"[WARN] Skipping {rr_path}: {e}")
            continue

        # Only process classifier run_result files (not regression)
        if rr.get("task") != "classification":
            continue

        model_type  = rr.get("model_type", "unknown")
        variant     = rr.get("variant", "unknown")
        cw_strategy = rr.get("class_weight_strategy", "unknown")
        n_features  = rr.get("n_features")
        n_train     = rr.get("n_train")
        n_val       = rr.get("n_val")

        for tentry in rr.get("targets", []):
            stat = tentry.get("stat")
            if not stat:
                continue

            per_run[(model_type, variant, cw_strategy, stat)].append({
                "n_features":          n_features,
                "n_train":             n_train,
                "n_val":               n_val,
                # val metrics
                "val_accuracy":        tentry.get("val_accuracy"),
                "val_f1_macro":        tentry.get("val_f1_macro"),
                # cv metrics (fold-averaged within each run)
                "cv_acc_mean":         tentry.get("cv_acc_mean"),
                "cv_acc_std":          tentry.get("cv_acc_std"),      # within-run fold std
                "cv_f1_macro_mean":    tentry.get("cv_f1_macro_mean"),
                "cv_f1_macro_std":     tentry.get("cv_f1_macro_std"), # within-run fold std
            })

    if not per_run:
        return pd.DataFrame()

    out_rows = []
    for (model_type, variant, cw_strategy, stat), runs in per_run.items():
        n_runs = len(runs)

        def _avg(field):
            vals = [r[field] for r in runs if r.get(field) is not None]
            return float(np.mean(vals)) if vals else None

        def _std(field):
            vals = [r[field] for r in runs if r.get(field) is not None]
            return float(np.std(vals)) if len(vals) > 1 else 0.0

        out_rows.append({
            "model_type":             model_type,
            "variant":                variant,
            "class_weight_strategy":  cw_strategy,
            "stat":                   stat,
            "n_runs":                 n_runs,
            "n_features":             runs[0]["n_features"],
            "n_train":                runs[0]["n_train"],
            "n_val":                  runs[0]["n_val"],
            # val metrics — mean and run-to-run std
            "val_acc_mean":           _avg("val_accuracy"),
            "val_acc_run_std":        _std("val_accuracy"),
            "val_f1_macro_mean":      _avg("val_f1_macro"),
            "val_f1_macro_run_std":   _std("val_f1_macro"),
            # cv metrics
            # cv_*_mean: average of per-run fold-means
            # cv_*_fold_std: average of per-run within-fold stds (fold-to-fold variance)
            # cv_*_run_std: std of per-run fold-means (run-to-run variance)
            "cv_acc_mean":            _avg("cv_acc_mean"),
            "cv_acc_fold_std":        _avg("cv_acc_std"),
            "cv_acc_run_std":         _std("cv_acc_mean"),
            "cv_f1_macro_mean":       _avg("cv_f1_macro_mean"),
            "cv_f1_macro_fold_std":   _avg("cv_f1_macro_std"),
            "cv_f1_macro_run_std":    _std("cv_f1_macro_mean"),
        })

    df = pd.DataFrame(out_rows)
    df = df.sort_values(
        ["model_type", "variant", "class_weight_strategy", "stat"]
    ).reset_index(drop=True)
    return df


def main():
    print(f"[COLLECT] Scanning {ARTIFACTS_CLASSIFICATION_ROOT} for run_result.json …")
    df = collect_all()
    if df.empty:
        print("[WARN] No classifier run_result.json files found. Run the classifiers first.")
        return
    df.to_csv(MASTER_CSV, index=False)
    n_combos = df[["model_type", "variant", "class_weight_strategy"]].drop_duplicates().shape[0]
    print(
        f"[COLLECT] {len(df)} rows "
        f"({n_combos} model×variant×cw combos) → {MASTER_CSV}"
    )


if __name__ == "__main__":
    main()
