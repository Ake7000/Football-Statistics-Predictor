# optimizer_lstm_mlp_torch.py  (refactor/)
# Per-target LSTM-MLP regressors optimised with Optuna.
#
# Architecture: shared LSTM encoder for home/away sequences + static MLP encoder,
# late-fused via concatenation → single scalar output per target.
# See shared_sequence.py for architecture details.
#
# How to run (from the workspace root):
#   python refactor/optimizer_lstm_mlp_torch.py                         # default variant, 3 repeats
#   python refactor/optimizer_lstm_mlp_torch.py --variant cform_diffmean_diffsum_form_mean_nplayers_stage_sum --repeats 1
#   python refactor/optimizers/optimizer_lstm_mlp_torch.py --variant diffmean_diffsum_form_mean_nplayers_raw_stage_sum --repeats 2 --seed 42

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

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from shared_config import (
    TARGETS, TARGET_LOSS_MAP,
    ARTIFACTS_LSTM_MLP_ROOT, TRAIN_TABLE_PATH, SEQ_TABLE_PATH,
    N_TRIALS, EPOCHS, PATIENCE, RETRAIN_EPOCHS, RETRAIN_PATIENCE,
    BATCH_SIZE_OPTIONS,
    N_HIDDEN_MIN, N_HIDDEN_MAX,
    DROPOUT_MIN, DROPOUT_MAX,
    LR_MIN, LR_MAX, L2_MIN, L2_MAX,
    ACTIVATIONS, OPTIMIZERS,
    UNITS_CHOICES, DEFAULT_UNITS_CHOICES,
    VARIANTS,
    LSTM_HIDDEN_CHOICES, LSTM_LAYERS_OPTIONS,
    LSTM_DROPOUT_MIN, LSTM_DROPOUT_MAX,
    LSTM_DEFAULT_STATIC_VARIANT,
    SEQ_K, SEQ_INPUT_DIM, USE_ROLE_TOKEN,
    CV_FOLDS, GLOBAL_SEED,
)
from shared_metrics import round_accuracy, compute_outcome_metrics_list
from shared_sequence import (
    load_seq_data, split_seq_data,
    build_single_model, make_seq_dataloader,
    get_criterion, decode_predictions,
    train_lstm_mlp_model,
)
from shared_utils import (
    set_all_seeds, get_torch_device, make_run_dir,
    build_layer_sizes, make_torch_optimizer,
)


# =============================================================================
# ==  OPTUNA OBJECTIVE  =======================================================
# =============================================================================
# _snap / build_layer_sizes / make_torch_optimizer are imported from shared_utils.

def objective_factory(
    *,
    home_seq_train:  np.ndarray,
    away_seq_train:  np.ndarray,
    X_train:         np.ndarray,
    home_seq_val:    np.ndarray,
    away_seq_val:    np.ndarray,
    X_val:           np.ndarray,
    y_train_model:   np.ndarray,   # possibly target-scaled
    y_val_model:     np.ndarray,
    target_name:     str,
    device:          torch.device,
    units_choices:   List[int],
):
    criterion = get_criterion(target_name)
    static_input_dim = X_train.shape[1]

    # Pre-build tensors once to avoid repeated allocation during Optuna search
    hs_tr = torch.from_numpy(home_seq_train).float()
    as_tr = torch.from_numpy(away_seq_train).float()
    x_tr  = torch.from_numpy(X_train).float()
    y_tr  = torch.from_numpy(y_train_model).float()

    hs_vl = torch.from_numpy(home_seq_val).float()
    as_vl = torch.from_numpy(away_seq_val).float()
    x_vl  = torch.from_numpy(X_val).float()
    y_vl  = torch.from_numpy(y_val_model).float()

    def objective(trial: optuna.trial.Trial) -> float:
        # ---- Sample hyperparameters ----
        n_hidden    = trial.suggest_int("n_hidden",    N_HIDDEN_MIN, N_HIDDEN_MAX)
        base_units  = trial.suggest_categorical("base_units",  units_choices)
        activation  = trial.suggest_categorical("activation",  ACTIVATIONS)
        mlp_dropout = trial.suggest_float("mlp_dropout", DROPOUT_MIN, DROPOUT_MAX)
        lr          = trial.suggest_float("lr",        LR_MIN, LR_MAX, log=True)
        l2_reg      = trial.suggest_float("l2_reg",    L2_MIN, L2_MAX, log=True)
        opt_name    = trial.suggest_categorical("optimizer",   OPTIMIZERS)
        bs          = trial.suggest_categorical("batch_size",  BATCH_SIZE_OPTIONS)

        # LSTM branch
        lstm_hidden  = trial.suggest_categorical("lstm_hidden", LSTM_HIDDEN_CHOICES)
        lstm_layers  = trial.suggest_categorical("lstm_layers", LSTM_LAYERS_OPTIONS)
        lstm_dropout = trial.suggest_float("lstm_dropout", LSTM_DROPOUT_MIN, LSTM_DROPOUT_MAX)

        # Fusion head
        fusion_n_hid  = trial.suggest_categorical("fusion_head_n_hidden", [0, 1])
        fusion_drop   = trial.suggest_float("fusion_dropout", DROPOUT_MIN, DROPOUT_MAX)

        # Static all mult_k up to N_HIDDEN_MAX-1 (keeps search space static for TPE)
        all_mults = [
            trial.suggest_categorical(f"mult_{k}", [0.5, 1.0, 2.0])
            for k in range(1, N_HIDDEN_MAX)
        ]
        layer_sizes = build_layer_sizes(trial.params, units_choices)

        print(
            f"[{target_name}] Trial {trial.number:03d} ▶ START | "
            f"mlp={layer_sizes} lstm_h={lstm_hidden}×{lstm_layers} "
            f"act={activation} lr={lr:.2e} bs={bs}"
        )

        model     = build_single_model(
            static_input_dim=static_input_dim,
            mlp_layer_sizes=layer_sizes,
            activation=activation,
            mlp_dropout=mlp_dropout,
            lstm_hidden_size=lstm_hidden,
            lstm_num_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            fusion_head_n_hidden=fusion_n_hid,
            fusion_dropout=fusion_drop,
        )
        optimizer = make_torch_optimizer(opt_name, model.parameters(), lr, l2_reg)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(PATIENCE // 3, 3), min_lr=1e-6,
        )

        from torch.utils.data import TensorDataset, DataLoader
        from shared_sequence import LSTMMLPDataset
        train_loader = DataLoader(
            LSTMMLPDataset(hs_tr, as_tr, x_tr, y_tr),
            batch_size=bs, shuffle=True, drop_last=False,
        )
        val_loader = DataLoader(
            LSTMMLPDataset(hs_vl, as_vl, x_vl, y_vl),
            batch_size=bs, shuffle=False, drop_last=False,
        )

        best_rmse, epochs_run, _ = train_lstm_mlp_model(
            model, optimizer, train_loader, val_loader,
            device, criterion, EPOCHS, PATIENCE, scheduler,
            optuna_trial=trial,
        )
        print(
            f"[{target_name}] Trial {trial.number:03d} ✓ DONE  | "
            f"RMSE={best_rmse:.5f} epochs={epochs_run}"
        )
        return best_rmse

    return objective


# =============================================================================
# ==  SINGLE-TARGET STUDY  ====================================================
# =============================================================================

def run_target_study(
    *,
    target_name:      str,
    home_seq_train:   np.ndarray,
    away_seq_train:   np.ndarray,
    X_train:          np.ndarray,
    y_train:          np.ndarray,
    home_seq_val:     np.ndarray,
    away_seq_val:     np.ndarray,
    X_val:            np.ndarray,
    y_val:            np.ndarray,
    run_dir:          Path,
    device:           torch.device,
    units_choices:    List[int],
    seed:             int = GLOBAL_SEED,
) -> Dict:
    target_dir = run_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # Scale MSE targets for better loss conditioning
    target_scaler: Optional[StandardScaler] = None
    y_train_model = y_train.copy()
    y_val_model   = y_val.copy()
    if TARGET_LOSS_MAP.get(target_name, "mse") == "mse":
        target_scaler = StandardScaler()
        y_train_model = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_model   = target_scaler.transform(y_val.reshape(-1, 1)).ravel()

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(multivariate=True, seed=seed),
        pruner=MedianPruner(n_startup_trials=max(5, N_TRIALS // 5)),
    )
    study.optimize(
        objective_factory(
            home_seq_train=home_seq_train,
            away_seq_train=away_seq_train,
            X_train=X_train,
            home_seq_val=home_seq_val,
            away_seq_val=away_seq_val,
            X_val=X_val,
            y_train_model=y_train_model.astype(np.float32),
            y_val_model=y_val_model.astype(np.float32),
            target_name=target_name,
            device=device,
            units_choices=units_choices,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params      = dict(study.best_trial.params)
    best_layer_sizes = build_layer_sizes(best_params, units_choices)

    print(
        f"\n[{target_name}] ▶ RETRAIN best | mlp={best_layer_sizes} "
        f"lstm_h={best_params.get('lstm_hidden')}×{best_params.get('lstm_layers')} "
        f"act={best_params.get('activation')} lr={best_params.get('lr', 0):.2e}"
    )

    # ---- Retrain with best config ----
    criterion = get_criterion(target_name)
    best_model = build_single_model(
        static_input_dim=X_train.shape[1],
        mlp_layer_sizes=best_layer_sizes,
        activation=best_params["activation"],
        mlp_dropout=float(best_params["mlp_dropout"]),
        lstm_hidden_size=int(best_params["lstm_hidden"]),
        lstm_num_layers=int(best_params["lstm_layers"]),
        lstm_dropout=float(best_params["lstm_dropout"]),
        fusion_head_n_hidden=int(best_params["fusion_head_n_hidden"]),
        fusion_dropout=float(best_params["fusion_dropout"]),
    ).to(device)
    optimizer  = make_torch_optimizer(
        best_params["optimizer"], best_model.parameters(),
        float(best_params["lr"]), float(best_params["l2_reg"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=max(PATIENCE // 3, 3), min_lr=1e-6,
    )
    bs = int(best_params["batch_size"])
    from shared_sequence import LSTMMLPDataset
    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        LSTMMLPDataset(
            torch.from_numpy(home_seq_train).float(),
            torch.from_numpy(away_seq_train).float(),
            torch.from_numpy(X_train).float(),
            torch.from_numpy(y_train_model.astype(np.float32)).float(),
        ),
        batch_size=bs, shuffle=True,
    )
    val_loader = DataLoader(
        LSTMMLPDataset(
            torch.from_numpy(home_seq_val).float(),
            torch.from_numpy(away_seq_val).float(),
            torch.from_numpy(X_val).float(),
            torch.from_numpy(y_val_model.astype(np.float32)).float(),
        ),
        batch_size=bs, shuffle=False,
    )

    _, _, best_state = train_lstm_mlp_model(
        best_model, optimizer, train_loader, val_loader,
        device, criterion, RETRAIN_EPOCHS, RETRAIN_PATIENCE, scheduler,
    )
    best_model.load_state_dict(best_state)

    # ---- Evaluate on val set ----
    best_model.eval()
    preds_list: List[np.ndarray] = []
    with torch.no_grad():
        for h_seq, a_seq, x_stat, _ in val_loader:
            preds_list.append(
                best_model(h_seq.to(device), a_seq.to(device), x_stat.to(device)).cpu().numpy()
            )
    y_pred = decode_predictions(np.concatenate(preds_list), target_name)
    if target_scaler is not None:
        y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    mae       = float(mean_absolute_error(y_val, y_pred))
    mse       = float(mean_squared_error(y_val, y_pred))
    r         = math.sqrt(mse)
    round_acc = round_accuracy(y_val, y_pred)

    # ---- Save artifacts ----
    export_params = {k: v for k, v in best_params.items() if not k.startswith("mult_")}
    export_params["layer_sizes"] = best_layer_sizes

    torch.save(
        {
            "model_state_dict":    best_model.state_dict(),
            "static_input_dim":    X_train.shape[1],
            "mlp_layer_sizes":     best_layer_sizes,
            "activation":          best_params["activation"],
            "mlp_dropout":         float(best_params["mlp_dropout"]),
            "lstm_hidden_size":    int(best_params["lstm_hidden"]),
            "lstm_num_layers":     int(best_params["lstm_layers"]),
            "lstm_dropout":        float(best_params["lstm_dropout"]),
            "fusion_head_n_hidden":int(best_params["fusion_head_n_hidden"]),
            "fusion_dropout":      float(best_params["fusion_dropout"]),
            "use_shared_lstm":     True,
            "use_role_token":      USE_ROLE_TOKEN,
            "seq_k":               SEQ_K,
            "seq_input_dim":       SEQ_INPUT_DIM,
            "target":              target_name,
        },
        target_dir / "best_model.pt",
    )
    (target_dir / "val_metrics.json").write_text(
        json.dumps({"MAE": mae, "MSE": mse, "RMSE": r}, indent=2)
    )
    (target_dir / "best_params.json").write_text(json.dumps(export_params, indent=2))
    if target_scaler is not None:
        with open(target_dir / "target_scaler.pkl", "wb") as f:
            pickle.dump(target_scaler, f)

    trials_df = study.trials_dataframe()
    if not trials_df.empty:
        mult_cols = [c for c in trials_df.columns if c.startswith("params_mult_")]
        trials_df.drop(columns=mult_cols, inplace=True, errors="ignore")
    trials_df.to_csv(target_dir / "study_summary.csv", index=False)

    print(f"[SAVE] {target_name}: MAE={mae:.4f}  RMSE={r:.4f}")
    return {
        "target":         target_name,
        "MAE":            mae,
        "MSE":            mse,
        "RMSE":           r,
        "val_round_acc":  round(round_acc, 4),
        "val_preds":      y_pred,        # popped in main() before saving to CSV
        "best_trial":     int(study.best_trial.number),
        "layer_sizes":    best_layer_sizes,
        **{k: v for k, v in export_params.items() if k != "layer_sizes"},
    }


# =============================================================================
# ==  CROSS-VALIDATED FINAL EVALUATION  =======================================
# =============================================================================

def cross_validate_target(
    target_name:     str,
    best_params:     Dict,
    X_full_raw:      np.ndarray,   # unscaled static features
    y_full:          np.ndarray,   # original targets (1D)
    home_seq_full:   np.ndarray,
    away_seq_full:   np.ndarray,
    device:          torch.device,
    units_choices:   List[int],
    n_folds:         int = CV_FOLDS,
) -> Dict:
    layer_sizes = build_layer_sizes(best_params, units_choices)
    criterion   = get_criterion(target_name)
    bs          = int(best_params["batch_size"])

    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=GLOBAL_SEED)
    metrics = {"MAE": [], "MSE": [], "RMSE": []}

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_full_raw), start=1):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_full_raw[tr_idx]).astype(np.float32)
        X_te   = scaler.transform(X_full_raw[te_idx]).astype(np.float32)
        y_tr   = y_full[tr_idx].astype(np.float32)
        y_te   = y_full[te_idx].astype(np.float32)

        fold_tsc: Optional[StandardScaler] = None
        y_tr_m = y_tr.copy()
        y_te_m = y_te.copy()
        if TARGET_LOSS_MAP.get(target_name, "mse") == "mse":
            fold_tsc = StandardScaler()
            y_tr_m   = fold_tsc.fit_transform(y_tr.reshape(-1, 1)).ravel()
            y_te_m   = fold_tsc.transform(y_te.reshape(-1, 1)).ravel()

        model = build_single_model(
            static_input_dim=X_tr.shape[1],
            mlp_layer_sizes=layer_sizes,
            activation=best_params["activation"],
            mlp_dropout=float(best_params["mlp_dropout"]),
            lstm_hidden_size=int(best_params["lstm_hidden"]),
            lstm_num_layers=int(best_params["lstm_layers"]),
            lstm_dropout=float(best_params["lstm_dropout"]),
            fusion_head_n_hidden=int(best_params["fusion_head_n_hidden"]),
            fusion_dropout=float(best_params["fusion_dropout"]),
        ).to(device)
        optimizer = make_torch_optimizer(
            best_params["optimizer"], model.parameters(),
            float(best_params["lr"]), float(best_params["l2_reg"]),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(PATIENCE // 3, 3), min_lr=1e-6,
        )

        from shared_sequence import LSTMMLPDataset
        from torch.utils.data import DataLoader
        train_loader = DataLoader(
            LSTMMLPDataset(
                torch.from_numpy(home_seq_full[tr_idx]).float(),
                torch.from_numpy(away_seq_full[tr_idx]).float(),
                torch.from_numpy(X_tr).float(),
                torch.from_numpy(y_tr_m).float(),
            ),
            batch_size=bs, shuffle=True,
        )
        val_loader = DataLoader(
            LSTMMLPDataset(
                torch.from_numpy(home_seq_full[te_idx]).float(),
                torch.from_numpy(away_seq_full[te_idx]).float(),
                torch.from_numpy(X_te).float(),
                torch.from_numpy(y_te_m).float(),
            ),
            batch_size=bs, shuffle=False,
        )

        _, _, best_state = train_lstm_mlp_model(
            model, optimizer, train_loader, val_loader,
            device, criterion, RETRAIN_EPOCHS, RETRAIN_PATIENCE, scheduler,
        )
        model.load_state_dict(best_state)
        model.eval()

        preds_list: List[np.ndarray] = []
        with torch.no_grad():
            for h_seq, a_seq, x_stat, _ in val_loader:
                preds_list.append(
                    model(h_seq.to(device), a_seq.to(device), x_stat.to(device)).cpu().numpy()
                )
        y_pred = decode_predictions(np.concatenate(preds_list), target_name)
        if fold_tsc is not None:
            y_pred = fold_tsc.inverse_transform(y_pred.reshape(-1, 1)).ravel()

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
    print(f"[CV {target_name}] RMSE = {result['RMSE_mean']:.4f} ± {result['RMSE_std']:.4f}")
    return result


# =============================================================================
# ==  MAIN  ===================================================================
# =============================================================================

def main(table_variant: str, seed: int) -> None:
    table_variant = table_variant.lower()
    if table_variant not in VARIANTS:
        raise ValueError(f"Unknown variant '{table_variant}'. Allowed: {sorted(VARIANTS.keys())}")

    set_all_seeds(seed)
    device        = get_torch_device()
    units_choices = UNITS_CHOICES.get(table_variant, DEFAULT_UNITS_CHOICES)

    run_dir = make_run_dir(ARTIFACTS_LSTM_MLP_ROOT, TRAIN_TABLE_PATH, table_variant)
    print(f"[RUN] Artifacts : {run_dir}")
    print(f"[RUN] Variant   : {table_variant} | seq K={SEQ_K} F={SEQ_INPUT_DIM} "
          f"role_token={USE_ROLE_TOKEN} | seed={seed}")

    # ---- Load data (static + sequences) ----
    data    = load_seq_data(table_variant)
    split   = split_seq_data(data)
    X_train = split["X_train"]
    X_val   = split["X_val"]

    with open(run_dir / "scaler.pkl", "wb") as f:
        pickle.dump(split["scaler"], f)
    (run_dir / "features_list.txt").write_text("\n".join(data["feature_names"]))

    # ---- Per-target Optuna study ----
    summary_rows:        List[Dict]             = []
    cv_rows:             List[Dict]             = []
    best_params_all:     Dict[str, Dict]        = {}
    val_preds_by_target: Dict[str, np.ndarray]  = {}
    y_val_by_target:     Dict[str, np.ndarray]  = {}
    target_means = {t: float(split["y_train"][:, i].mean()) for i, t in enumerate(TARGETS)}

    for target in tqdm(TARGETS, desc="targets"):
        y_train_t = split["y_train"][:, TARGETS.index(target)]
        y_val_t   = split["y_val"][:, TARGETS.index(target)]

        result = run_target_study(
            target_name=target,
            home_seq_train=split["home_seq_train"],
            away_seq_train=split["away_seq_train"],
            X_train=X_train,
            y_train=y_train_t,
            home_seq_val=split["home_seq_val"],
            away_seq_val=split["away_seq_val"],
            X_val=X_val,
            y_val=y_val_t,
            run_dir=run_dir,
            device=device,
            units_choices=units_choices,
            seed=seed,
        )
        y_val_by_target[target]     = y_val_t
        val_preds_by_target[target] = result.pop("val_preds")
        summary_rows.append(result)
        best_params_all[target] = {
            k: v for k, v in result.items()
            if k not in ("target", "MAE", "MSE", "RMSE", "best_trial")
        }

        # --- 5-fold CV ---
        print(f"[CV] Running {CV_FOLDS}-fold CV for {target} …")
        best_params_cv = {k: v for k, v in result.items()
                          if k not in ("target", "MAE", "MSE", "RMSE", "best_trial")}
        y_full_t = split["y_full"][:, TARGETS.index(target)]
        cv_result = cross_validate_target(
            target_name=target,
            best_params=best_params_cv,
            X_full_raw=split["X_full_raw"],
            y_full=y_full_t,
            home_seq_full=split["home_seq_full"],
            away_seq_full=split["away_seq_full"],
            device=device,
            units_choices=units_choices,
        )
        cv_result["target"] = target
        cv_rows.append(cv_result)
        target_dir = run_dir / target
        (target_dir / "cv_metrics.json").write_text(json.dumps(cv_result, indent=2))

    # ---- Save run-level summary ----
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary_all_targets.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(run_dir / "cv_summary_all_targets.csv", index=False)
    (run_dir / "best_params_all.json").write_text(json.dumps(best_params_all, indent=2))

    # ---- Write unified run_result.json ----
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
        "model_type":            "lstm_mlp_torch",
        "variant":               table_variant,
        "seed":                  seed,
        "n_features":            len(data["feature_names"]),
        "n_train":               int(split["X_train"].shape[0]),
        "n_val":                 int(split["X_val"].shape[0]),
        "timestamp":             run_dir.name.split("__")[-1],
        "targets":               _targets_out,
        "overall_val_rmse_mean": round(float(np.mean([r["RMSE"] for r in summary_rows])), 6),
        "overall_cv_rmse_mean":  round(float(np.mean([c.get("RMSE_mean", 0.0) for c in cv_rows])), 6),
        "outcome_metrics":       outcome_metrics_list,
    }
    (run_dir / "run_result.json").write_text(json.dumps(_run_result, indent=2))
    print(f"[SAVE] run_result.json → {run_dir}")

    print(f"\n[DONE] Run artifacts saved to: {run_dir}")
    print(f"{'Target':30s}  {'RMSE':>8s}  {'MAE':>8s}")
    for row in summary_rows:
        print(f"{row['target']:30s}  {row['RMSE']:8.4f}  {row['MAE']:8.4f}")


# =============================================================================
# ==  ENTRY POINT  ============================================================
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LSTM-MLP single-target optimizer")
    parser.add_argument(
        "--variant", type=str, default=LSTM_DEFAULT_STATIC_VARIANT,
        help=f"Static feature variant (e.g. {LSTM_DEFAULT_STATIC_VARIANT}). "
             f"Default: {LSTM_DEFAULT_STATIC_VARIANT}",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Independent runs per variant (each uses a different seed).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override seed. If omitted, uses GLOBAL_SEED + run_idx.",
    )
    args = parser.parse_args()

    for run_idx in range(1, args.repeats + 1):
        seed = args.seed if args.seed is not None else GLOBAL_SEED + run_idx
        print(f"\n{'='*70}")
        print(f"[RUN {run_idx}/{args.repeats}] variant={args.variant}  seed={seed}")
        print(f"{'='*70}")
        main(table_variant=args.variant.lower(), seed=seed)
