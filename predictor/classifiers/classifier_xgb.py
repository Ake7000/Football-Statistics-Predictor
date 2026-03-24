# classifier_xgb.py  (refactor/classifiers/)
# Per-stat XGBoost 3-class direction classifiers optimised with Optuna.
#
# Predicts sign(HOME_X - AWAY_X) for each stat pair:
#   0 = HOME_WIN  (HOME > AWAY)
#   1 = DRAW      (HOME == AWAY)
#   2 = AWAY_WIN  (HOME < AWAY)
#
# How to run (from the workspace root):
#   python refactor/classifiers/classifier_xgb.py              # all variants, 3 repeats
#   python refactor/classifiers/classifier_xgb.py --variant form --repeats 1 --seed 42

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # classifiers/ → refactor/

import json
import pickle
import warnings
import argparse
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import xgboost as xgb
from xgboost import XGBClassifier

from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import KFold

from shared_config import (
    TARGETS,
    CLASSIFIER_STAT_PAIRS, CLASSIFIER_TARGETS, N_CLASSES,
    ARTIFACTS_CLASSIFIER_XGB_ROOT, TRAIN_TABLE_PATH,
    N_TRIALS, CV_FOLDS, GLOBAL_SEED,
    XGB_PARAM_SPACE, XGB_N_ESTIMATORS_MAX, XGB_EARLY_STOPPING_ROUNDS,
    VARIANTS,
)
from shared_features import build_feature_matrices, build_full_feature_matrix
from shared_preprocessing import load_and_prepare_dataframe
from shared_utils import set_all_seeds, get_xgb_tree_method, get_xgb_device, make_run_dir
from shared_metrics import make_direction_labels, make_stat_labels_df, clf_metrics_dict, OUTCOME_CLASSES


# =============================================================================
# ==  PARAMETER BUILDERS  =====================================================
# =============================================================================

def _build_cls_params_from_trial(
    trial: optuna.trial.Trial,
    tree_method: str,
    device: str,
    seed: int = GLOBAL_SEED,
) -> Dict:
    """Sample XGBoost classifier hyperparameters from Optuna trial."""
    space = XGB_PARAM_SPACE
    return {
        "objective":         "multi:softmax",
        "num_class":         N_CLASSES,
        "eval_metric":       "mlogloss",
        "booster":           "gbtree",
        "tree_method":       tree_method,
        "device":            device,
        "n_estimators":      XGB_N_ESTIMATORS_MAX,
        "verbosity":         0,
        "random_state":      seed,
        "max_depth":         trial.suggest_int("max_depth",         *space["max_depth"]),
        "learning_rate":     trial.suggest_float("learning_rate",   *space["learning_rate"],    log=True),
        "subsample":         trial.suggest_float("subsample",       *space["subsample"]),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",  *space["colsample_bytree"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", *space["colsample_bylevel"]),
        "reg_lambda":        trial.suggest_float("reg_lambda",      *space["reg_lambda"],       log=True),
        "reg_alpha":         trial.suggest_float("reg_alpha",       *space["reg_alpha"],        log=True),
        "min_child_weight":  trial.suggest_int("min_child_weight",  *space["min_child_weight"]),
        "gamma":             trial.suggest_float("gamma",           *space["gamma"]),
    }


def _build_cls_params_from_dict(
    params: Dict,
    tree_method: str,
    device: str,
    seed: int = GLOBAL_SEED,
) -> Dict:
    """Reconstruct XGBoost classifier params from saved best params dict."""
    space = XGB_PARAM_SPACE
    return {
        "objective":         "multi:softmax",
        "num_class":         N_CLASSES,
        "eval_metric":       "mlogloss",
        "booster":           "gbtree",
        "tree_method":       tree_method,
        "device":            device,
        "n_estimators":      XGB_N_ESTIMATORS_MAX,
        "verbosity":         0,
        "random_state":      seed,
        "max_depth":         int(params["max_depth"]),
        "learning_rate":     float(params["learning_rate"]),
        "subsample":         float(params["subsample"]),
        "colsample_bytree":  float(params["colsample_bytree"]),
        "colsample_bylevel": float(params["colsample_bylevel"]),
        "reg_lambda":        float(params["reg_lambda"]),
        "reg_alpha":         float(params["reg_alpha"]),
        "min_child_weight":  int(params["min_child_weight"]),
        "gamma":             float(params["gamma"]),
    }


# =============================================================================
# ==  FIT STRATEGIES (version-tolerant)  ======================================
# =============================================================================

def _fit_cls_new_callbacks(
    params: Dict, X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> XGBClassifier:
    """XGBoost >= 1.6: EarlyStopping via callbacks."""
    callbacks = [xgb.callback.EarlyStopping(
        rounds=XGB_EARLY_STOPPING_ROUNDS, save_best=True, maximize=False,
    )]
    clf = XGBClassifier(**params)
    clf.fit(X_tr, y_tr, sample_weight=sample_weight,
            eval_set=[(X_val, y_val)], verbose=False, callbacks=callbacks)
    return clf


def _fit_cls_old_early_stop(
    params: Dict, X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> XGBClassifier:
    """XGBoost < 1.6: early_stopping_rounds directly in fit()."""
    clf = XGBClassifier(**params)
    clf.fit(
        X_tr, y_tr, sample_weight=sample_weight,
        eval_set=[(X_val, y_val)], verbose=False,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
    )
    return clf


def _fit_cls_no_early_stop(
    params: Dict, X_tr: np.ndarray, y_tr: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> XGBClassifier:
    """Fallback: no early stopping."""
    clf = XGBClassifier(**params)
    clf.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)
    return clf


def fit_xgb_classifier(
    params: Dict,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
) -> XGBClassifier:
    """
    Fit XGBClassifier with version-tolerant early stopping.
    Priority: new callbacks API → old early_stopping_rounds → no early stopping.
    Falls back to CPU automatically on CUDA errors.
    """
    def _is_gpu_err(e: Exception) -> bool:
        return any(kw in str(e).lower() for kw in ("gpu", "cuda"))

    def _cpu_fallback(p: Dict) -> Dict:
        p = dict(p)
        p["tree_method"] = "hist"
        p["device"]      = "cpu"
        print("[WARN] XGBoost CUDA error — falling back to CPU.")
        return p

    for fit_fn in (_fit_cls_new_callbacks, _fit_cls_old_early_stop):
        try:
            return fit_fn(params, X_tr, y_tr, X_val, y_val, sample_weight)
        except TypeError:
            continue
        except xgb.core.XGBoostError as e:
            if _is_gpu_err(e):
                params = _cpu_fallback(params)
                try:
                    return fit_fn(params, X_tr, y_tr, X_val, y_val, sample_weight)
                except TypeError:
                    continue
            raise

    # Final fallback: no early stopping
    try:
        return _fit_cls_no_early_stop(params, X_tr, y_tr, sample_weight)
    except xgb.core.XGBoostError as e:
        if _is_gpu_err(e):
            params = _cpu_fallback(params)
            return _fit_cls_no_early_stop(params, X_tr, y_tr, sample_weight)
        raise

    raise RuntimeError("All XGBoost classifier fit strategies failed.")


# =============================================================================
# ==  OPTUNA OBJECTIVE  =======================================================
# =============================================================================

def objective_factory(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    stat_name: str,
    tree_method: str,
    device: str,
    seed: int = GLOBAL_SEED,
    cw_strategy: str = "none",
):
    # Compute sample weights from y_train label frequencies (normalized to mean=1
    # so absolute scale matches XGBoost's min_child_weight expectation)
    if cw_strategy == "sqrt":
        _counts = np.bincount(y_train, minlength=N_CLASSES).astype(float)
        _counts = np.where(_counts == 0, 1.0, _counts)
        _cw     = 1.0 / np.sqrt(_counts)
        sample_weight = _cw[y_train]
        sample_weight = sample_weight / sample_weight.mean()
    else:
        sample_weight = None

    def objective(trial: optuna.trial.Trial) -> float:
        params = _build_cls_params_from_trial(trial, tree_method, device, seed)

        print(
            f"[{stat_name}] Trial {trial.number:03d} ▶ START | "
            f"depth={params['max_depth']} lr={params['learning_rate']:.4f} "
            f"sub={params['subsample']:.2f} cbt={params['colsample_bytree']:.2f}"
        )

        clf    = fit_xgb_classifier(params, X_train, y_train, X_val, y_val,
                                    sample_weight=sample_weight)
        y_pred = clf.predict(X_val)
        val_f1 = float(f1_score(y_val, y_pred, average="macro", zero_division=0))

        trial.report(1.0 - val_f1, step=0)

        print(
            f"[{stat_name}] Trial {trial.number:03d} ✓ DONE  | "
            f"f1={val_f1:.4f}"
        )
        return 1.0 - val_f1   # minimize (1 - F1 macro)

    return objective


# =============================================================================
# ==  SINGLE-STAT STUDY  ======================================================
# =============================================================================

def run_stat_study(
    stat: str,
    home_col: str,
    away_col: str,
    X_train: np.ndarray,
    y_train_df: pd.DataFrame,
    X_val: np.ndarray,
    y_val_df: pd.DataFrame,
    run_dir: Path,
    tree_method: str,
    device: str,
    seed: int = GLOBAL_SEED,
    cw_strategy: str = "none",
) -> Dict:
    """
    Run Optuna study for one stat pair, retrain best config, save all artifacts.
    Returns a summary dict with metrics and best hyperparameters.
    """
    stat_dir = run_dir / stat
    stat_dir.mkdir(parents=True, exist_ok=True)

    y_train_cls = make_stat_labels_df(y_train_df, home_col, away_col)
    y_val_cls   = make_stat_labels_df(y_val_df,   home_col, away_col)

    class_counts = {
        OUTCOME_CLASSES[c]: int(np.sum(y_train_cls == c)) for c in range(N_CLASSES)
    }

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(multivariate=True, seed=seed),
        pruner=MedianPruner(n_startup_trials=max(5, N_TRIALS // 5)),
    )
    study.optimize(
        objective_factory(
            X_train, y_train_cls, X_val, y_val_cls,
            stat, tree_method, device, seed, cw_strategy,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_trial_params = dict(study.best_trial.params)
    best_params       = _build_cls_params_from_dict(best_trial_params, tree_method, device, seed)

    print(
        f"[{stat}] ▶ RETRAIN best | "
        f"depth={best_params['max_depth']} lr={best_params['learning_rate']:.4f} "
        f"sub={best_params['subsample']:.2f} col={best_params['colsample_bytree']:.2f}"
    )

    # Sample weights for retrain (normalized to mean=1)
    if cw_strategy == "sqrt":
        _ret_counts = np.bincount(y_train_cls, minlength=N_CLASSES).astype(float)
        _ret_counts = np.where(_ret_counts == 0, 1.0, _ret_counts)
        _ret_sw     = (1.0 / np.sqrt(_ret_counts))[y_train_cls]
        _ret_sw     = _ret_sw / _ret_sw.mean()
    else:
        _ret_sw = None

    clf    = fit_xgb_classifier(best_params, X_train, y_train_cls, X_val, y_val_cls,
                                sample_weight=_ret_sw)
    y_pred = clf.predict(X_val).astype(np.int64)
    y_proba = clf.predict_proba(X_val)   # (n_val, n_classes)

    metrics = clf_metrics_dict(y_val_cls, y_pred)

    # --- Save artifacts ---
    with open(stat_dir / "best_model.pkl", "wb") as f:
        pickle.dump(clf, f)
    try:
        clf.get_booster().save_model(str(stat_dir / "best_model.json"))
    except Exception:
        pass

    (stat_dir / "best_params.json").write_text(json.dumps(best_trial_params, indent=2))
    (stat_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    (stat_dir / "class_distribution_train.json").write_text(
        json.dumps(class_counts, indent=2)
    )
    np.save(str(stat_dir / "val_predictions_proba.npy"), y_proba)

    try:
        importance = clf.get_booster().get_score(importance_type="gain")
        (stat_dir / "feature_importance.json").write_text(
            json.dumps(importance, indent=2)
        )
    except Exception as e:
        print(f"[WARN] Could not save feature importance for {stat}: {e}")

    study.trials_dataframe().to_csv(stat_dir / "study_summary.csv", index=False)

    print(f"[SAVE] {stat}: acc={metrics['accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}")

    return {
        "stat":                     stat,
        "home_col":                 home_col,
        "away_col":                 away_col,
        "train_class_distribution": class_counts,
        **metrics,
        "best_trial":               int(study.best_trial.number),
        "best_params_dict":         best_trial_params,
        "val_preds_proba":          y_proba,   # popped in main() before saving CSV
    }


# =============================================================================
# ==  CROSS-VALIDATED FINAL EVALUATION  =======================================
# =============================================================================

def cross_validate_stat(
    stat: str,
    home_col: str,
    away_col: str,
    df: pd.DataFrame,
    best_trial_params: Dict,
    X_full_raw: np.ndarray,
    tree_method: str,
    device: str,
    n_folds: int = CV_FOLDS,
    cw_strategy: str = "none",
) -> Dict:
    """
    Run k-fold CV with the best hyperparameters for an honest accuracy estimate.
    XGBoost does NOT use StandardScaler (tree-based model).
    """
    params        = _build_cls_params_from_dict(best_trial_params, tree_method, device)
    y_labels_full = make_stat_labels_df(df, home_col, away_col)
    kf            = KFold(n_splits=n_folds, shuffle=True, random_state=GLOBAL_SEED)
    acc_list: List[float] = []
    f1_list:  List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full_raw), start=1):
        X_tr, X_te = X_full_raw[train_idx], X_full_raw[test_idx]
        y_tr, y_te = y_labels_full[train_idx], y_labels_full[test_idx]

        if cw_strategy == "sqrt":
            _cv_counts = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
            _cv_counts = np.where(_cv_counts == 0, 1.0, _cv_counts)
            _cv_sw     = (1.0 / np.sqrt(_cv_counts))[y_tr]
            _cv_sw     = _cv_sw / _cv_sw.mean()
        else:
            _cv_sw = None

        clf          = fit_xgb_classifier(params, X_tr, y_tr, X_te, y_te,
                                         sample_weight=_cv_sw)
        y_pred       = clf.predict(X_te).astype(np.int64)
        fold_acc     = float(accuracy_score(y_te, y_pred))
        fold_f1      = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
        acc_list.append(fold_acc)
        f1_list.append(fold_f1)
        print(f"  [CV {stat}] Fold {fold_idx}/{n_folds} → acc={fold_acc:.4f}  f1={fold_f1:.4f}")

    result = {
        "stat":             stat,
        "cv_acc_mean":      round(float(np.mean(acc_list)), 4),
        "cv_acc_std":       round(float(np.std(acc_list)),  4),
        "cv_f1_macro_mean": round(float(np.mean(f1_list)),  4),
        "cv_f1_macro_std":  round(float(np.std(f1_list)),   4),
    }
    print(f"[CV {stat}] acc = {result['cv_acc_mean']:.4f} ± {result['cv_acc_std']:.4f}")
    return result


# =============================================================================
# ==  MAIN  ===================================================================
# =============================================================================

def main(table_variant: str, seed: int, cw_strategy: str = "none") -> None:
    table_variant = table_variant.lower()
    if table_variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant '{table_variant}'. Allowed: {sorted(VARIANTS.keys())}"
        )

    set_all_seeds(seed)
    tree_method = get_xgb_tree_method()
    device      = get_xgb_device()

    run_dir = make_run_dir(ARTIFACTS_CLASSIFIER_XGB_ROOT, TRAIN_TABLE_PATH, table_variant, suffix=f"__cw_{cw_strategy}")
    print(f"[RUN] Artifacts: {run_dir}")
    print(f"[RUN] Variant: {table_variant} | device: {device} | seed: {seed}")

    df = load_and_prepare_dataframe(TRAIN_TABLE_PATH)

    # XGBoost does not need feature scaling
    X_train, X_val, y_train_df, y_val_df, _, feature_names = build_feature_matrices(
        df, table_variant, apply_scaler=False
    )

    (run_dir / "features_list.txt").write_text("\n".join(feature_names))
    (run_dir / "stat_pairs.txt").write_text(
        "\n".join(f"{s}: {h} vs {a}" for s, h, a in CLASSIFIER_STAT_PAIRS)
    )

    _X_save, _y_save, _, _feat_save = build_full_feature_matrix(df, table_variant, apply_scaler=False)
    _tt = pd.DataFrame(_X_save, columns=_feat_save)
    for _col in TARGETS:
        _tt[_col] = _y_save[_col].values
    _tt.to_csv(run_dir / "training_table.csv", index=False)

    print(f"[FEATURES] {len(feature_names)} feature columns for variant '{table_variant}'")

    summary_rows: List[Dict] = []
    cv_rows:      List[Dict] = []

    print(f"[INFO] Optimising {len(CLASSIFIER_STAT_PAIRS)} stat pairs …")
    for i, (stat, home_col, away_col) in enumerate(
        tqdm(CLASSIFIER_STAT_PAIRS, desc="Stats", unit="stat")
    ):
        print(f"\n[STAT {i+1}/{len(CLASSIFIER_STAT_PAIRS)}] {stat}  ({home_col} vs {away_col})")

        result = run_stat_study(
            stat, home_col, away_col,
            X_train, y_train_df, X_val, y_val_df,
            run_dir, tree_method, device, seed, cw_strategy,
        )
        result.pop("val_preds_proba")
        summary_rows.append(result)

        print(f"[CV] Running {CV_FOLDS}-fold CV for {stat} …")
        X_full_raw = build_full_feature_matrix(df, table_variant, apply_scaler=False)[0]
        cv_result  = cross_validate_stat(
            stat, home_col, away_col, df,
            result["best_params_dict"], X_full_raw,
            tree_method, device, cw_strategy=cw_strategy,
        )
        cv_rows.append(cv_result)
        (run_dir / stat / "cv_metrics.json").write_text(json.dumps(cv_result, indent=2))

    # --- Run-level summary CSVs ---
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary_all_stats.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(run_dir / "cv_summary_all_stats.csv", index=False)

    # --- Unified run_result.json ---
    _targets_out = []
    for _row, _cv in zip(summary_rows, cv_rows):
        _targets_out.append({
            "stat":                _row["stat"],
            "home_col":            _row["home_col"],
            "away_col":            _row["away_col"],
            "class_distribution":  _row["train_class_distribution"],
            "val_accuracy":        _row["accuracy"],
            "val_f1_macro":        _row["f1_macro"],
            "val_f1_per_class":    _row["f1_per_class"],
            "cv_acc_mean":         _cv["cv_acc_mean"],
            "cv_acc_std":          _cv["cv_acc_std"],
            "cv_f1_macro_mean":    _cv["cv_f1_macro_mean"],
            "cv_f1_macro_std":     _cv["cv_f1_macro_std"],
            "confusion_matrix":    _row["confusion_matrix"],
            "best_params":         _row["best_params_dict"],
        })

    _run_result = {
        "model_type":                "classifier_xgb",
        "task":                      "classification",
        "n_classes":                 N_CLASSES,
        "class_names":               OUTCOME_CLASSES,
        "variant":                   table_variant,
        "seed":                      seed,
        "n_features":                len(feature_names),
        "n_train":                   int(X_train.shape[0]),
        "n_val":                     int(X_val.shape[0]),
        "seq_k":                     None,
        "timestamp":                 run_dir.name.split("__")[-2],
        "class_weight_strategy":     cw_strategy,
        "targets":                   _targets_out,
        "overall_val_acc_mean":      round(float(np.mean([r["accuracy"] for r in summary_rows])), 4),
        "overall_val_f1_macro_mean": round(float(np.mean([r["f1_macro"] for r in summary_rows])), 4),
        "overall_cv_acc_mean":       round(float(np.mean([c["cv_acc_mean"] for c in cv_rows])), 4),
        "overall_cv_f1_macro_mean":  round(float(np.mean([c["cv_f1_macro_mean"] for c in cv_rows])), 4),
    }
    (run_dir / "run_result.json").write_text(json.dumps(_run_result, indent=2))
    print(f"[SAVE] run_result.json → {run_dir}")

    print(f"\n[DONE] Variant '{table_variant}' complete.")
    print(f"[ARTIFACTS] {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost 3-class direction classifier with Optuna")
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Feature table variant (e.g. form, sum, mean …). "
             "If omitted, all variants defined in VARIANTS are run.",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Number of independent runs per variant (each uses a different seed).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override the seed for this run. If omitted, uses GLOBAL_SEED + run_idx.",
    )
    parser.add_argument(
        "--class_weights", type=str, default="sqrt",
        choices=["none", "sqrt"],
        help="Class weight strategy: none=unweighted, sqrt=1/sqrt(count)",
    )
    args = parser.parse_args()

    variants_to_run = (
        [args.variant.lower()] if args.variant
        else list(VARIANTS.keys())
    )

    for variant in variants_to_run:
        for run_idx in range(1, args.repeats + 1):
            seed = args.seed if args.seed is not None else GLOBAL_SEED + run_idx
            print(f"\n{'='*70}")
            print(f"[RUN {run_idx}/{args.repeats}] variant={variant}  seed={seed}")
            print(f"{'='*70}")
            main(table_variant=variant, seed=seed, cw_strategy=args.class_weights)
