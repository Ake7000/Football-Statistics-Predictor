# optimizer_xgb.py  (refactor/)
# Per-target XGBoost regressors optimised with Optuna.
# All shared logic lives in shared_*.py — this file contains ONLY XGBoost-specific code.
#
# How to run (from the refactor/ directory):
#   python optimizer_xgb.py              # runs all variants, 3 repeats
#   python refactor/optimizers/optimizer_xgb.py --variant raw --repeats 1

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # optimizers/ → refactor/

import json
import math
import pickle
import warnings
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import xgboost as xgb

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from shared_config import (
    TARGETS, TARGET_LOSS_MAP,
    ARTIFACTS_XGB_ROOT, TRAIN_TABLE_PATH,
    N_TRIALS, CV_FOLDS, GLOBAL_SEED,
    XGB_PARAM_SPACE, XGB_N_ESTIMATORS_MAX, XGB_EARLY_STOPPING_ROUNDS,
    VARIANTS,
)
from shared_features import build_feature_matrices, build_full_feature_matrix
from shared_preprocessing import load_and_prepare_dataframe
from shared_utils import set_all_seeds, get_xgb_tree_method, get_xgb_device, make_run_dir, rmse
from shared_metrics import round_accuracy, compute_outcome_metrics_list


# =============================================================================
# ==  XGB OBJECTIVE MAP  ======================================================
# =============================================================================

def get_xgb_objective(target_name: str) -> str:
    """
    Map TARGET_LOSS_MAP entries to XGBoost objective strings.
    Configured in shared_config.py — no changes needed here when adding targets.
    """
    loss_type = TARGET_LOSS_MAP.get(target_name, "mse")
    if loss_type == "poisson":
        return "count:poisson"
    return "reg:squarederror"


# =============================================================================
# ==  BOOSTER WRAPPER  ========================================================
# =============================================================================

class BoosterWrapper:
    """
    Thin wrapper around xgb.Booster (from xgb.train) that provides:
      - A sklearn-compatible predict(X) interface.
      - best_iteration tracking for early-stopped models.
      - pickle serialisation via __getstate__ / __setstate__.
      - save_model() for JSON export.

    This wrapper exists because xgb.train (DMatrix API) is more stable across
    XGBoost versions than the sklearn wrapper for early stopping.
    """

    def __init__(
        self,
        booster: xgb.Booster,
        best_iteration: Optional[int],
        n_features: int,
    ) -> None:
        self.booster       = booster
        self.best_iteration = best_iteration
        self.n_features    = n_features

    def predict(self, X: np.ndarray) -> np.ndarray:
        dmat = xgb.DMatrix(X)
        ntree = None
        if getattr(self.booster, "best_ntree_limit", None) is not None:
            ntree = int(self.booster.best_ntree_limit)
        elif self.best_iteration is not None:
            ntree = int(self.best_iteration) + 1
        if ntree is not None:
            try:
                return self.booster.predict(dmat, ntree_limit=ntree)
            except TypeError:
                pass
        return self.booster.predict(dmat)

    def save_model(self, path: Path) -> None:
        self.booster.save_model(str(path))

    def __getstate__(self):
        raw = self.booster.save_raw(raw_format="json")
        return {"booster_raw": raw, "best_iteration": self.best_iteration,
                "n_features": self.n_features}

    def __setstate__(self, state):
        b = xgb.Booster()
        b.load_model(bytearray(state["booster_raw"]))
        self.booster        = b
        self.best_iteration = state["best_iteration"]
        self.n_features     = state["n_features"]


# =============================================================================
# ==  FIT STRATEGIES (version-tolerant)  ======================================
# =============================================================================

def _fit_new_callbacks(
    params: Dict, X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> xgb.XGBRegressor:
    """XGBoost ≥ 1.6: EarlyStopping via callbacks."""
    callbacks = [xgb.callback.EarlyStopping(
        rounds=XGB_EARLY_STOPPING_ROUNDS, save_best=True, maximize=False,
    )]
    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False, callbacks=callbacks)
    return model


def _fit_old_early_stop(
    params: Dict, X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> xgb.XGBRegressor:
    """XGBoost < 1.6: early_stopping_rounds directly in fit()."""
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False,
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
    )
    return model


def _fit_dmatrix(
    params: Dict, X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> BoosterWrapper:
    """
    Universal fallback using the DMatrix API (xgb.train).
    Works with all XGBoost versions.
    """
    booster_params = {
        "objective":        params.get("objective", "reg:squarederror"),
        "eval_metric":      "rmse",
        "booster":          "gbtree",
        "tree_method":      params.get("tree_method", "hist"),
        "device":           params.get("device", "cpu"),
        "eta":              params.get("learning_rate", 0.1),
        "max_depth":        params.get("max_depth", 6),
        "subsample":        params.get("subsample", 1.0),
        "colsample_bytree": params.get("colsample_bytree", 1.0),
        "lambda":           params.get("reg_lambda", 1.0),
        "alpha":            params.get("reg_alpha", 0.0),
        "min_child_weight": params.get("min_child_weight", 1),
        "gamma":            params.get("gamma", 0.0),
        "verbosity":        0,
        "seed":             params.get("random_state", GLOBAL_SEED),
    }
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval   = xgb.DMatrix(X_val, label=y_val)

    bst = xgb.train(
        booster_params, dtrain,
        num_boost_round=XGB_N_ESTIMATORS_MAX,
        evals=[(dtrain, "train"), (dval, "eval")],
        early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        verbose_eval=False,
    )
    best_iter = getattr(bst, "best_iteration", None)
    if best_iter is None:
        ntl = getattr(bst, "best_ntree_limit", None)
        best_iter = (int(ntl) - 1) if ntl is not None else None
    return BoosterWrapper(bst, best_iter, n_features=X_tr.shape[1])


def fit_xgb(
    params: Dict,
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
):
    """
    Attempt to fit XGBoost using the best available API, with GPU→CPU fallback.

    Priority:
      1. sklearn wrapper + EarlyStopping callback (XGB ≥ 1.6)
      2. sklearn wrapper + early_stopping_rounds in fit() (XGB < 1.6)
      3. xgb.train DMatrix API (universal fallback)
    Each step falls back to CPU automatically on CUDA errors.
    """
    def _is_gpu_err(e: Exception) -> bool:
        return any(kw in str(e).lower() for kw in ("gpu", "cuda"))

    def _cpu_fallback(params: Dict) -> Dict:
        p = dict(params)
        p["tree_method"] = "hist"
        p["device"]      = "cpu"
        print("[WARN] XGBoost CUDA error — falling back to CPU.")
        return p

    for fit_fn in (_fit_new_callbacks, _fit_old_early_stop, _fit_dmatrix):
        try:
            return fit_fn(params, X_tr, y_tr, X_val, y_val)
        except TypeError:
            continue          # API not supported in this version, try next
        except xgb.core.XGBoostError as e:
            if _is_gpu_err(e):
                params = _cpu_fallback(params)
                try:
                    return fit_fn(params, X_tr, y_tr, X_val, y_val)
                except TypeError:
                    continue
            raise             # non-GPU XGBoost error — propagate

    raise RuntimeError("All XGBoost fit strategies failed.")


# =============================================================================
# ==  PREDICT HELPER  =========================================================
# =============================================================================

def predict(model, X: np.ndarray) -> np.ndarray:
    """
    Unified predict that handles both XGBRegressor (sklearn) and BoosterWrapper.
    Always uses best_iteration when available.
    """
    if isinstance(model, BoosterWrapper):
        return model.predict(X)

    # sklearn XGBRegressor
    best_iter = getattr(model, "best_iteration", None)
    if best_iter is not None:
        try:
            return model.predict(X, iteration_range=(0, int(best_iter) + 1))
        except TypeError:
            pass
        try:
            return model.predict(X, ntree_limit=int(best_iter) + 1)
        except TypeError:
            pass
    return model.predict(X)


# =============================================================================
# ==  OPTUNA OBJECTIVE  =======================================================
# =============================================================================

def _build_params_from_trial(
    trial: optuna.trial.Trial,
    tree_method: str,
    device: str,
    objective: str,
    seed: int = GLOBAL_SEED,
) -> Dict:
    """Sample XGBoost hyperparameters from Optuna trial."""
    space = XGB_PARAM_SPACE
    return {
        "objective":        objective,
        "eval_metric":      "rmse",
        "booster":          "gbtree",
        "tree_method":      tree_method,
        "device":           device,
        "n_estimators":     XGB_N_ESTIMATORS_MAX,   # fixed ceiling; early stopping decides
        "verbosity":        0,
        "random_state":     seed,
        "max_depth":        trial.suggest_int("max_depth",        *space["max_depth"]),
        "learning_rate":    trial.suggest_float("learning_rate",  *space["learning_rate"],   log=True),
        "subsample":        trial.suggest_float("subsample",      *space["subsample"]),
        "colsample_bytree":  trial.suggest_float("colsample_bytree",  *space["colsample_bytree"]),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", *space["colsample_bylevel"]),
        "reg_lambda":        trial.suggest_float("reg_lambda",        *space["reg_lambda"],      log=True),
        "reg_alpha":        trial.suggest_float("reg_alpha",      *space["reg_alpha"],       log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", *space["min_child_weight"]),
        "gamma":            trial.suggest_float("gamma",          *space["gamma"]),
    }


def _build_params_from_dict(
    params: Dict,
    tree_method: str,
    device: str,
    objective: str,
    seed: int = GLOBAL_SEED,
) -> Dict:
    """Reconstruct XGBoost params dict from a saved params dict (for retrain)."""
    space = XGB_PARAM_SPACE
    return {
        "objective":        objective,
        "eval_metric":      "rmse",
        "booster":          "gbtree",
        "tree_method":      tree_method,
        "device":           device,
        "n_estimators":     XGB_N_ESTIMATORS_MAX,
        "verbosity":        0,
        "random_state":     seed,
        "max_depth":        int(params["max_depth"]),
        "learning_rate":    float(params["learning_rate"]),
        "subsample":        float(params["subsample"]),
        "colsample_bytree":  float(params["colsample_bytree"]),
        "colsample_bylevel": float(params["colsample_bylevel"]),
        "reg_lambda":        float(params["reg_lambda"]),
        "reg_alpha":        float(params["reg_alpha"]),
        "min_child_weight": int(params["min_child_weight"]),
        "gamma":            float(params["gamma"]),
    }


def objective_factory(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_name: str,
    tree_method: str,
    device: str,
    xgb_objective: str,
    seed: int = GLOBAL_SEED,
):
    def objective(trial: optuna.trial.Trial) -> float:
        params = _build_params_from_trial(trial, tree_method, device, xgb_objective, seed)

        print(
            f"[{target_name}] Trial {trial.number:03d} ▶ START | "
            f"depth={params['max_depth']} lr={params['learning_rate']:.4f} "
            f"sub={params['subsample']:.2f} cbt={params['colsample_bytree']:.2f} "
            f"cbl={params['colsample_bylevel']:.2f} "
            f"λ={params['reg_lambda']:.2e} α={params['reg_alpha']:.2e} "
            f"mcw={params['min_child_weight']} γ={params['gamma']:.2f}"
        )

        model      = fit_xgb(params, X_train, y_train, X_val, y_val)
        y_pred     = predict(model, X_val)
        val_rmse   = rmse(y_val, y_pred)
        best_iter  = getattr(model, "best_iteration", None)
        if isinstance(model, BoosterWrapper):
            best_iter = model.best_iteration

        # FIX: report to Optuna so MedianPruner can act
        # XGBoost doesn't expose per-round val scores here, so we report
        # the final RMSE at step=best_iter (pruner uses this as a proxy).
        trial.report(val_rmse, step=best_iter or 0)

        print(
            f"[{target_name}] Trial {trial.number:03d} ✓ DONE  | "
            f"RMSE={val_rmse:.5f} best_iter={best_iter}"
        )
        return float(val_rmse)

    return objective


# =============================================================================
# ==  SINGLE-TARGET STUDY  ====================================================
# =============================================================================

def run_target_study(
    target_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    run_dir: Path,
    tree_method: str,
    device: str,
    seed: int = GLOBAL_SEED,
) -> Dict:
    """
    Run Optuna study for one target, retrain best config, save all artifacts.
    Returns a summary dict with metrics and best hyperparameters.
    """
    target_dir    = run_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)
    xgb_objective = get_xgb_objective(target_name)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(multivariate=True, seed=seed),
        pruner=MedianPruner(n_startup_trials=max(5, N_TRIALS // 5)),
    )
    study.optimize(
        objective_factory(X_train, y_train, X_val, y_val,
                          target_name, tree_method, device, xgb_objective, seed),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_trial_params = dict(study.best_trial.params)
    best_params       = _build_params_from_dict(best_trial_params, tree_method, device, xgb_objective, seed)

    print(
        f"[{target_name}] ▶ RETRAIN best | "
        f"depth={best_params['max_depth']} lr={best_params['learning_rate']:.4f} "
        f"sub={best_params['subsample']:.2f} col={best_params['colsample_bytree']:.2f}"
    )

    model     = fit_xgb(best_params, X_train, y_train, X_val, y_val)
    y_pred    = predict(model, X_val)
    mae       = float(mean_absolute_error(y_val, y_pred))
    mse_val   = float(mean_squared_error(y_val, y_pred))
    r         = math.sqrt(mse_val)
    round_acc = round_accuracy(y_val, y_pred)
    best_iter = getattr(model, "best_iteration", None)
    if isinstance(model, BoosterWrapper):
        best_iter = model.best_iteration

    # --- Save artifacts ---
    if hasattr(model, "save_model"):
        model.save_model(target_dir / "best_model.json")
    try:
        with open(target_dir / "best_model.pkl", "wb") as f:
            pickle.dump(model, f)
    except Exception:
        pass

    export_params = dict(best_trial_params)
    if best_iter is not None:
        export_params["best_iteration"] = int(best_iter)
    (target_dir / "best_params.json").write_text(json.dumps(export_params, indent=2))
    (target_dir / "val_metrics.json").write_text(
        json.dumps({"MAE": mae, "MSE": mse_val, "RMSE": r}, indent=2)
    )

    # --- Feature importance ---
    try:
        booster = model.booster if isinstance(model, BoosterWrapper) else model.get_booster()
        importance = booster.get_score(importance_type="gain")
        (target_dir / "feature_importance.json").write_text(
            json.dumps(importance, indent=2)
        )
    except Exception as e:
        print(f"[WARN] Could not save feature importance for {target_name}: {e}")

    study.trials_dataframe().to_csv(target_dir / "study_summary.csv", index=False)

    print(f"[SAVE] {target_name}: MAE={mae:.4f}  RMSE={r:.4f}")

    return {
        "target":        target_name,
        "MAE":           mae,
        "MSE":           mse_val,
        "RMSE":          r,
        "val_round_acc": round(round_acc, 4),
        "val_preds":     y_pred,            # popped in main() before saving to CSV
        "best_trial":    int(study.best_trial.number),
        **export_params,
    }


# =============================================================================
# ==  CROSS-VALIDATED FINAL EVALUATION  =======================================
# =============================================================================

def cross_validate_target(
    target_name: str,
    best_trial_params: Dict,
    X_full: np.ndarray,
    y_full: np.ndarray,
    tree_method: str,
    device: str,
    n_folds: int = CV_FOLDS,
) -> Dict:
    """
    Run k-fold CV with the best hyperparameters for an honest generalization estimate.
    XGBoost does NOT use StandardScaler (tree-based model — scaling has no effect).
    Each fold uses early stopping against the fold's own val split.
    """
    xgb_objective = get_xgb_objective(target_name)
    params        = _build_params_from_dict(best_trial_params, tree_method, device, xgb_objective)
    kf            = KFold(n_splits=n_folds, shuffle=True, random_state=GLOBAL_SEED)
    metrics       = {"MAE": [], "MSE": [], "RMSE": []}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full), start=1):
        X_tr, X_te = X_full[train_idx], X_full[test_idx]
        y_tr, y_te = y_full[train_idx], y_full[test_idx]

        # No scaler for XGBoost
        model  = fit_xgb(params, X_tr, y_tr, X_te, y_te)
        y_pred = predict(model, X_te)

        fold_mae  = float(mean_absolute_error(y_te, y_pred))
        fold_mse  = float(mean_squared_error(y_te, y_pred))
        fold_rmse = math.sqrt(fold_mse)
        metrics["MAE"].append(fold_mae)
        metrics["MSE"].append(fold_mse)
        metrics["RMSE"].append(fold_rmse)
        print(f"  [CV {target_name}] Fold {fold_idx}/{n_folds} → RMSE={fold_rmse:.4f}")

    result = {}
    for m, vals in metrics.items():
        result[f"{m}_mean"] = float(np.mean(vals))
        result[f"{m}_std"]  = float(np.std(vals))
    print(
        f"[CV {target_name}] RMSE = {result['RMSE_mean']:.4f} ± {result['RMSE_std']:.4f}"
    )
    return result


# =============================================================================
# ==  MAIN  ===================================================================
# =============================================================================

def main(table_variant: str, seed: int) -> None:
    table_variant = table_variant.lower()
    if table_variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant '{table_variant}'. Allowed: {sorted(VARIANTS.keys())}"
        )

    set_all_seeds(seed)
    tree_method = get_xgb_tree_method()   # always 'hist' in XGBoost >= 2.0
    device      = get_xgb_device()        # 'cuda' or 'cpu'

    run_dir = make_run_dir(ARTIFACTS_XGB_ROOT, TRAIN_TABLE_PATH, table_variant)
    print(f"[RUN] Artifacts: {run_dir}")
    print(f"[RUN] Variant: {table_variant} | device: {device} | seed: {seed}")

    df = load_and_prepare_dataframe(TRAIN_TABLE_PATH)
    target_means = {t: float(df[t].mean()) for t in TARGETS}

    # FIX: XGBoost does NOT use StandardScaler (apply_scaler=False)
    X_train, X_val, y_train_df, y_val_df, _, feature_names = build_feature_matrices(
        df, table_variant, apply_scaler=False
    )

    # Save run-level artifacts (no scaler for XGB)
    (run_dir / "features_list.txt").write_text("\n".join(feature_names))
    (run_dir / "targets.txt").write_text("\n".join(TARGETS))

    # Save the exact feature+target table fed to the model (unscaled, all rows)
    _X_save, _y_save, _, _feat_save = build_full_feature_matrix(df, table_variant, apply_scaler=False)
    _training_table = pd.DataFrame(_X_save, columns=_feat_save)
    for _col in TARGETS:
        _training_table[_col] = _y_save[_col].values
    _training_table.to_csv(run_dir / "training_table.csv", index=False)

    print(f"[FEATURES] {len(feature_names)} feature columns for variant '{table_variant}'")

    # --- Per-target Optuna study ---
    summary_rows:        List[Dict]             = []
    cv_rows:             List[Dict]             = []
    val_preds_by_target: Dict[str, np.ndarray] = {}
    y_val_by_target:     Dict[str, np.ndarray] = {}

    print(f"[INFO] Optimising {len(TARGETS)} targets …")
    for i, target in enumerate(tqdm(TARGETS, desc="Targets", unit="tgt")):
        print(f"\n[TARGET {i+1}/{len(TARGETS)}] {target}")

        y_tr = y_train_df[target].values.astype(float)
        y_vl = y_val_df[target].values.astype(float)

        result = run_target_study(
            target, X_train, y_tr, X_val, y_vl,
            run_dir, tree_method, device, seed,
        )
        y_val_by_target[target]     = y_vl
        val_preds_by_target[target] = result.pop("val_preds")
        summary_rows.append(result)

        # --- 5-fold CV for honest final metric ---
        print(f"[CV] Running {CV_FOLDS}-fold CV for {target} …")
        X_full_raw = build_full_feature_matrix(df, table_variant, apply_scaler=False)[0]
        y_full     = df[target].values.astype(float)

        best_trial_params = {
            k: v for k, v in result.items()
            if k not in ["target", "MAE", "MSE", "RMSE", "best_trial", "best_iteration"]
        }
        cv_result = cross_validate_target(
            target, best_trial_params, X_full_raw, y_full, tree_method, device,
        )
        cv_result["target"] = target
        cv_rows.append(cv_result)

        target_dir = run_dir / target
        (target_dir / "cv_metrics.json").write_text(json.dumps(cv_result, indent=2))

    # --- Save run-level summaries ---
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary_all_targets.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(run_dir / "cv_summary_all_targets.csv", index=False)

    # --- Write unified run_result.json ---
    _targets_out = []
    for _row, _cv in zip(summary_rows, cv_rows):
        _t  = _row["target"]
        _tm = target_means[_t]
        _vr = _row["RMSE"]
        _vm = _row["MAE"]
        _cr = _cv.get("RMSE_mean")
        _cm = _cv.get("MAE_mean")
        _bp = {k: v for k, v in _row.items()
               if k not in ("target", "MAE", "MSE", "RMSE", "best_trial")}
        _targets_out.append({
            "target":           _t,
            "target_mean":      round(_tm, 6),
            "val_mae":          round(_vm, 6),
            "val_rmse":         round(_vr, 6),
            "val_round_acc":    _row.get("val_round_acc"),
            "val_mae_pct":      round(_vm / _tm * 100, 2) if _tm else None,
            "val_rmse_pct":     round(_vr / _tm * 100, 2) if _tm else None,
            "cv_rmse_mean":     round(_cr, 6) if _cr is not None else None,
            "cv_rmse_std":      round(_cv.get("RMSE_std", 0.0), 6),
            "cv_mae_mean":      round(_cm, 6) if _cm is not None else None,
            "cv_mae_std":       round(_cv.get("MAE_std", 0.0), 6),
            "cv_rmse_pct_mean": round(_cr / _tm * 100, 2) if (_cr and _tm) else None,
            "cv_mae_pct_mean":  round(_cm / _tm * 100, 2) if (_cm and _tm) else None,
            "best_params":      _bp,
        })
    outcome_metrics_list = compute_outcome_metrics_list(
        val_preds_by_target, y_val_by_target, TARGETS
    )
    _run_result = {
        "model_type":            "xgb",
        "variant":               table_variant,
        "seed":                  seed,
        "n_features":            len(feature_names),
        "n_train":               int(X_train.shape[0]),
        "n_val":                 int(X_val.shape[0]),
        "timestamp":             run_dir.name.split("__")[-1],
        "targets":               _targets_out,
        "overall_val_rmse_mean": round(float(np.mean([r["RMSE"] for r in summary_rows])), 6),
        "overall_cv_rmse_mean":  round(float(np.mean([c.get("RMSE_mean", 0.0) for c in cv_rows])), 6),
        "outcome_metrics":       outcome_metrics_list,
    }
    (run_dir / "run_result.json").write_text(json.dumps(_run_result, indent=2))
    print(f"[SAVE] run_result.json → {run_dir}")

    print(f"\n[DONE] Variant '{table_variant}' complete.")
    print(f"[ARTIFACTS] {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost optimizer with Optuna")
    parser.add_argument(
        "--variant", type=str, default=None,
        help="Feature table variant (e.g. sum, mean, form, agg, full …). "
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
            main(table_variant=variant, seed=seed)
