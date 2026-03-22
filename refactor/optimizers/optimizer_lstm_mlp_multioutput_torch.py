# optimizer_lstm_mlp_multioutput_torch.py  (refactor/)
# Multi-target LSTM-MLP: shared LSTM encoder + static MLP backbone,
# one independent output head per prediction target.
#
# One Optuna study per static-feature variant (minimises mean val-RMSE
# across all 14 targets).  Same approach as optimizer_mlp_multioutput_torch.py
# but with the LSTM sequence branch added.
# See shared_sequence.py for architecture details.
#
# How to run (from the workspace root):
#   python refactor/optimizer_lstm_mlp_multioutput_torch.py                          # default variant
#   python refactor/optimizers/optimizer_lstm_mlp_multioutput_torch.py --variant cform_diffmean_diffsum_form_mean_nplayers_stage_sum --repeats 1

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
    ARTIFACTS_LSTM_MLP_MULTIOUTPUT_ROOT, TRAIN_TABLE_PATH, SEQ_TABLE_PATH,
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
from shared_sequence import (
    load_seq_data, split_seq_data,
    build_multi_model, make_seq_dataloader,
    get_criteria, decode_all_predictions,
    build_target_scalers, decode_and_unscale,
    train_lstm_multioutput_model,
    LSTMMLPDataset,
)
from shared_utils import (
    set_all_seeds, get_torch_device, make_run_dir,
    build_layer_sizes, make_torch_optimizer,
)
from shared_metrics import round_accuracy, compute_outcome_metrics_list


# =============================================================================
# ==  EVALUATION HELPER  ======================================================
# =============================================================================
# _snap / build_layer_sizes / make_torch_optimizer are imported from shared_utils.

def evaluate_model(
    model:          nn.Module,
    val_loader,
    device:         torch.device,
    y_val_original: np.ndarray,
    target_scalers: Dict[int, StandardScaler],
) -> Dict:
    model.eval()
    preds_list: List[np.ndarray] = []
    with torch.no_grad():
        for h_seq, a_seq, x_stat, _ in val_loader:
            preds_list.append(
                model(h_seq.to(device), a_seq.to(device), x_stat.to(device)).cpu().numpy()
            )
    raw_preds = np.vstack(preds_list)
    decoded   = decode_and_unscale(raw_preds, target_scalers)

    per_target: Dict[str, Dict] = {}
    all_rmse: List[float] = []
    for i, t in enumerate(TARGETS):
        mae  = float(mean_absolute_error(y_val_original[:, i], decoded[:, i]))
        mse  = float(mean_squared_error(y_val_original[:, i], decoded[:, i]))
        rmse = math.sqrt(mse)
        per_target[t] = {"MAE": mae, "MSE": mse, "RMSE": rmse}
        all_rmse.append(rmse)

    return {"mean_RMSE": float(np.mean(all_rmse)), "per_target": per_target, "val_preds": decoded}


# =============================================================================
# ==  OPTUNA OBJECTIVE  =======================================================
# =============================================================================

def objective_factory(
    *,
    home_seq_train:  np.ndarray,
    away_seq_train:  np.ndarray,
    X_train:         np.ndarray,
    home_seq_val:    np.ndarray,
    away_seq_val:    np.ndarray,
    X_val:           np.ndarray,
    y_train_model:   np.ndarray,   # (N_train, n_targets), MSE cols already scaled
    y_val_model:     np.ndarray,
    device:          torch.device,
    units_choices:   List[int],
    criteria:        List[nn.Module],
):
    n_targets        = len(TARGETS)
    static_input_dim = X_train.shape[1]

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

        lstm_hidden  = trial.suggest_categorical("lstm_hidden", LSTM_HIDDEN_CHOICES)
        lstm_layers  = trial.suggest_categorical("lstm_layers", LSTM_LAYERS_OPTIONS)
        lstm_dropout = trial.suggest_float("lstm_dropout", LSTM_DROPOUT_MIN, LSTM_DROPOUT_MAX)

        fusion_drop  = trial.suggest_float("fusion_dropout", DROPOUT_MIN, DROPOUT_MAX)
        head_hidden  = trial.suggest_categorical("head_hidden", [True, False])

        all_mults = [
            trial.suggest_categorical(f"mult_{k}", [0.5, 1.0, 2.0])
            for k in range(1, N_HIDDEN_MAX)
        ]
        layer_sizes = build_layer_sizes(trial.params, units_choices)

        print(
            f"[MULTI] Trial {trial.number:03d} ▶ START | "
            f"mlp={layer_sizes} lstm_h={lstm_hidden}×{lstm_layers} "
            f"act={activation} lr={lr:.2e} bs={bs}"
        )

        model     = build_multi_model(
            static_input_dim=static_input_dim,
            mlp_layer_sizes=layer_sizes,
            activation=activation,
            mlp_dropout=mlp_dropout,
            lstm_hidden_size=lstm_hidden,
            lstm_num_layers=lstm_layers,
            lstm_dropout=lstm_dropout,
            n_targets=n_targets,
            head_hidden=head_hidden,
            fusion_dropout=fusion_drop,
        )
        optimizer = make_torch_optimizer(opt_name, model.parameters(), lr, l2_reg)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(PATIENCE // 3, 3), min_lr=1e-6,
        )
        from torch.utils.data import DataLoader
        train_loader = DataLoader(LSTMMLPDataset(hs_tr, as_tr, x_tr, y_tr), batch_size=bs, shuffle=True)
        val_loader   = DataLoader(LSTMMLPDataset(hs_vl, as_vl, x_vl, y_vl), batch_size=bs, shuffle=False)

        best_rmse, epochs_run, _ = train_lstm_multioutput_model(
            model, optimizer, train_loader, val_loader,
            device, criteria, EPOCHS, PATIENCE, scheduler,
            optuna_trial=trial,
        )
        print(
            f"[MULTI] Trial {trial.number:03d} ✓ DONE  | "
            f"mean_RMSE={best_rmse:.5f} epochs={epochs_run}"
        )
        return best_rmse

    return objective


# =============================================================================
# ==  VARIANT STUDY  ==========================================================
# =============================================================================

def run_variant_study(
    *,
    table_variant:    str,
    home_seq_train:   np.ndarray,
    away_seq_train:   np.ndarray,
    X_train:          np.ndarray,
    y_train:          np.ndarray,   # (N_train, n_targets) original (unscaled)
    home_seq_val:     np.ndarray,
    away_seq_val:     np.ndarray,
    X_val:            np.ndarray,
    y_val:            np.ndarray,   # (N_val, n_targets)  original (unscaled)
    run_dir:          Path,
    device:           torch.device,
    units_choices:    List[int],
    criteria:         List[nn.Module],
    seed:             int = GLOBAL_SEED,
) -> Dict:
    run_dir.mkdir(parents=True, exist_ok=True)
    n_targets = len(TARGETS)

    # Scale MSE targets for loss conditioning (Poisson targets are skipped)
    target_scalers = build_target_scalers(y_train)
    y_train_model  = y_train.copy()
    y_val_model    = y_val.copy()
    for idx, sc in target_scalers.items():
        y_train_model[:, idx] = sc.transform(y_train_model[:, idx:idx+1]).ravel()
        y_val_model[:, idx]   = sc.transform(y_val_model[:, idx:idx+1]).ravel()

    hs_tr = torch.from_numpy(home_seq_train).float()
    as_tr = torch.from_numpy(away_seq_train).float()
    x_tr  = torch.from_numpy(X_train).float()
    y_tr  = torch.from_numpy(y_train_model.astype(np.float32)).float()

    hs_vl = torch.from_numpy(home_seq_val).float()
    as_vl = torch.from_numpy(away_seq_val).float()
    x_vl  = torch.from_numpy(X_val).float()
    y_vl  = torch.from_numpy(y_val_model.astype(np.float32)).float()

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(multivariate=True, seed=seed),
        study_name=f"lstm_multioutput_{table_variant}",
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
            device=device,
            units_choices=units_choices,
            criteria=criteria,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params      = dict(study.best_trial.params)
    best_layer_sizes = build_layer_sizes(best_params, units_choices)
    head_hidden      = bool(best_params.get("head_hidden", True))

    print(
        f"\n[MULTI] ▶ RETRAIN best | mlp={best_layer_sizes} "
        f"lstm_h={best_params.get('lstm_hidden')}×{best_params.get('lstm_layers')} "
        f"act={best_params.get('activation')} lr={best_params.get('lr', 0):.2e}"
    )

    best_model = build_multi_model(
        static_input_dim=X_train.shape[1],
        mlp_layer_sizes=best_layer_sizes,
        activation=best_params["activation"],
        mlp_dropout=float(best_params["mlp_dropout"]),
        lstm_hidden_size=int(best_params["lstm_hidden"]),
        lstm_num_layers=int(best_params["lstm_layers"]),
        lstm_dropout=float(best_params["lstm_dropout"]),
        n_targets=n_targets,
        head_hidden=head_hidden,
        fusion_dropout=float(best_params["fusion_dropout"]),
    ).to(device)
    optimizer = make_torch_optimizer(
        best_params["optimizer"], best_model.parameters(),
        float(best_params["lr"]), float(best_params["l2_reg"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=max(PATIENCE // 3, 3), min_lr=1e-6,
    )
    bs = int(best_params["batch_size"])
    from torch.utils.data import DataLoader
    train_loader = DataLoader(LSTMMLPDataset(hs_tr, as_tr, x_tr, y_tr), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(LSTMMLPDataset(hs_vl, as_vl, x_vl, y_vl), batch_size=bs, shuffle=False)

    _, _, best_state = train_lstm_multioutput_model(
        best_model, optimizer, train_loader, val_loader,
        device, criteria, RETRAIN_EPOCHS, RETRAIN_PATIENCE, scheduler,
    )
    best_model.load_state_dict(best_state)

    metrics              = evaluate_model(best_model, val_loader, device, y_val, target_scalers)
    val_preds            = metrics.pop("val_preds")
    round_accs           = {t: round(round_accuracy(y_val[:, i], val_preds[:, i]), 4)
                             for i, t in enumerate(TARGETS)}
    outcome_metrics_list = compute_outcome_metrics_list(
        {t: val_preds[:, i] for i, t in enumerate(TARGETS)},
        {t: y_val[:, i]     for i, t in enumerate(TARGETS)},
        TARGETS,
    )

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
            "fusion_dropout":      float(best_params["fusion_dropout"]),
            "head_hidden":         head_hidden,
            "n_targets":           n_targets,
            "use_shared_lstm":     True,
            "use_role_token":      USE_ROLE_TOKEN,
            "seq_k":               SEQ_K,
            "seq_input_dim":       SEQ_INPUT_DIM,
            "targets":             TARGETS,
        },
        run_dir / "best_model.pt",
    )
    (run_dir / "val_metrics_per_target.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "best_params.json").write_text(json.dumps(export_params, indent=2))
    with open(run_dir / "target_scalers.pkl", "wb") as f:
        pickle.dump(target_scalers, f)

    summary_rows = [
        {
            "target": t,
            "MAE":    metrics["per_target"][t]["MAE"],
            "MSE":    metrics["per_target"][t]["MSE"],
            "RMSE":   metrics["per_target"][t]["RMSE"],
        }
        for t in TARGETS
    ]
    pd.DataFrame(summary_rows).to_csv(run_dir / "summary_all_targets.csv", index=False)

    trials_df = study.trials_dataframe()
    if not trials_df.empty:
        mult_cols = [c for c in trials_df.columns if c.startswith("params_mult_")]
        trials_df.drop(columns=mult_cols, inplace=True, errors="ignore")
    trials_df.to_csv(run_dir / "study_summary.csv", index=False)

    print(f"\n[MULTI] Val mean_RMSE={metrics['mean_RMSE']:.4f}")
    for t in TARGETS:
        m = metrics["per_target"][t]
        print(f"  {t:30s}: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}")

    return {
        "layer_sizes":     best_layer_sizes,
        "head_hidden":     head_hidden,
        "mean_RMSE":       metrics["mean_RMSE"],
        "per_target":      metrics["per_target"],
        "round_accs":      round_accs,
        "outcome_metrics": outcome_metrics_list,
        **{k: v for k, v in export_params.items() if k != "layer_sizes"},
    }


# =============================================================================
# ==  CROSS-VALIDATED FINAL EVALUATION  =======================================
# =============================================================================

def cross_validate_variant(
    best_params:     Dict,
    X_full_raw:      np.ndarray,   # unscaled static features
    y_full:          np.ndarray,   # (N, n_targets) original targets
    home_seq_full:   np.ndarray,
    away_seq_full:   np.ndarray,
    device:          torch.device,
    units_choices:   List[int],
    criteria:        List[nn.Module],
    n_folds:         int = CV_FOLDS,
) -> Dict:
    layer_sizes = build_layer_sizes(best_params, units_choices)
    head_hidden = bool(best_params.get("head_hidden", True))
    bs          = int(best_params["batch_size"])
    n_targets   = len(TARGETS)

    kf         = KFold(n_splits=n_folds, shuffle=True, random_state=GLOBAL_SEED)
    fold_rmses: Dict[str, List[float]] = {t: [] for t in TARGETS}
    fold_maes:  Dict[str, List[float]] = {t: [] for t in TARGETS}

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_full_raw), start=1):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_full_raw[tr_idx]).astype(np.float32)
        X_te   = scaler.transform(X_full_raw[te_idx]).astype(np.float32)
        y_tr   = y_full[tr_idx].astype(np.float32)
        y_te   = y_full[te_idx].astype(np.float32)

        fold_tscalers: Dict[int, StandardScaler] = {}
        y_tr_m = y_tr.copy()
        y_te_m = y_te.copy()
        for i, t in enumerate(TARGETS):
            if TARGET_LOSS_MAP.get(t, "mse") == "mse":
                sc = StandardScaler()
                y_tr_m[:, i] = sc.fit_transform(y_tr[:, i:i+1]).ravel()
                y_te_m[:, i] = sc.transform(y_te[:, i:i+1]).ravel()
                fold_tscalers[i] = sc

        model = build_multi_model(
            static_input_dim=X_tr.shape[1],
            mlp_layer_sizes=layer_sizes,
            activation=best_params["activation"],
            mlp_dropout=float(best_params["mlp_dropout"]),
            lstm_hidden_size=int(best_params["lstm_hidden"]),
            lstm_num_layers=int(best_params["lstm_layers"]),
            lstm_dropout=float(best_params["lstm_dropout"]),
            n_targets=n_targets,
            head_hidden=head_hidden,
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

        _, _, best_state = train_lstm_multioutput_model(
            model, optimizer, train_loader, val_loader,
            device, criteria, RETRAIN_EPOCHS, RETRAIN_PATIENCE, scheduler,
        )
        model.load_state_dict(best_state)
        model.eval()

        preds_list: List[np.ndarray] = []
        with torch.no_grad():
            for h_seq, a_seq, x_stat, _ in val_loader:
                preds_list.append(
                    model(h_seq.to(device), a_seq.to(device), x_stat.to(device)).cpu().numpy()
                )
        decoded = decode_and_unscale(np.vstack(preds_list), fold_tscalers)

        fold_mean_rmse = 0.0
        for i, t in enumerate(TARGETS):
            fold_rmse = math.sqrt(float(mean_squared_error(y_te[:, i], decoded[:, i])))
            fold_mae  = float(mean_absolute_error(y_te[:, i], decoded[:, i]))
            fold_rmses[t].append(fold_rmse)
            fold_maes[t].append(fold_mae)
            fold_mean_rmse += fold_rmse
        fold_mean_rmse /= n_targets
        print(f"  [CV] Fold {fold_idx}/{n_folds} → mean_RMSE={fold_mean_rmse:.4f}")

    result: Dict = {}
    all_means: List[float] = []
    for t in TARGETS:
        result[f"{t}_RMSE_mean"] = float(np.mean(fold_rmses[t]))
        result[f"{t}_RMSE_std"]  = float(np.std(fold_rmses[t]))
        result[f"{t}_MAE_mean"]  = float(np.mean(fold_maes[t]))
        result[f"{t}_MAE_std"]   = float(np.std(fold_maes[t]))
        all_means.append(result[f"{t}_RMSE_mean"])
    result["overall_RMSE_mean"] = float(np.mean(all_means))

    print(f"\n[CV] overall_RMSE_mean = {result['overall_RMSE_mean']:.4f}")
    for t in TARGETS:
        print(f"  {t:30s}: {result[f'{t}_RMSE_mean']:.4f} ± {result[f'{t}_RMSE_std']:.4f}")
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
    criteria      = get_criteria()

    run_dir = make_run_dir(ARTIFACTS_LSTM_MLP_MULTIOUTPUT_ROOT, TRAIN_TABLE_PATH, table_variant)
    print(f"[RUN] Artifacts : {run_dir}")
    print(f"[RUN] Variant   : {table_variant} | seq K={SEQ_K} F={SEQ_INPUT_DIM} "
          f"role_token={USE_ROLE_TOKEN} | seed={seed}")

    data  = load_seq_data(table_variant)
    split = split_seq_data(data)
    target_means = {t: float(split["y_train"][:, i].mean()) for i, t in enumerate(TARGETS)}

    with open(run_dir / "scaler.pkl", "wb") as f:
        pickle.dump(split["scaler"], f)
    (run_dir / "features_list.txt").write_text("\n".join(data["feature_names"]))

    result = run_variant_study(
        table_variant=table_variant,
        home_seq_train=split["home_seq_train"],
        away_seq_train=split["away_seq_train"],
        X_train=split["X_train"],
        y_train=split["y_train"],
        home_seq_val=split["home_seq_val"],
        away_seq_val=split["away_seq_val"],
        X_val=split["X_val"],
        y_val=split["y_val"],
        run_dir=run_dir,
        device=device,
        units_choices=units_choices,
        criteria=criteria,
        seed=seed,
    )
    _round_accs           = result.pop("round_accs", {})
    _outcome_metrics_list = result.pop("outcome_metrics", [])

    # --- 5-fold CV ---
    best_params_cv = {k: v for k, v in result.items() if k not in ("mean_RMSE", "per_target")}
    print(f"\n[CV] Running {CV_FOLDS}-fold CV …")
    cv_result = cross_validate_variant(
        best_params=best_params_cv,
        X_full_raw=split["X_full_raw"],
        y_full=split["y_full"],
        home_seq_full=split["home_seq_full"],
        away_seq_full=split["away_seq_full"],
        device=device,
        units_choices=units_choices,
        criteria=criteria,
    )
    pd.DataFrame([
        {
            "target":    t,
            "RMSE_mean": cv_result[f"{t}_RMSE_mean"],
            "RMSE_std":  cv_result[f"{t}_RMSE_std"],
            "MAE_mean":  cv_result[f"{t}_MAE_mean"],
            "MAE_std":   cv_result[f"{t}_MAE_std"],
        }
        for t in TARGETS
    ]).to_csv(run_dir / "cv_summary_all_targets.csv", index=False)

    # ---- Write unified run_result.json ----
    _shared_bp = {k: v for k, v in result.items() if k not in ("per_target", "mean_RMSE")}
    _targets_out = []
    for _t in TARGETS:
        _tm = target_means[_t]
        _vr = result["per_target"][_t]["RMSE"]
        _vm = result["per_target"][_t]["MAE"]
        _cr = cv_result.get(f"{_t}_RMSE_mean")
        _cm = cv_result.get(f"{_t}_MAE_mean")
        _targets_out.append({
            "target":           _t,
            "target_mean":      round(_tm, 6),
            "val_mae":          round(_vm, 6),
            "val_rmse":         round(_vr, 6),
            "val_round_acc":    _round_accs.get(_t),
            "val_mae_pct":      round(_vm / _tm * 100, 2) if _tm else None,
            "val_rmse_pct":     round(_vr / _tm * 100, 2) if _tm else None,
            "cv_rmse_mean":     round(_cr, 6) if _cr is not None else None,
            "cv_rmse_std":      round(cv_result.get(f"{_t}_RMSE_std", 0.0), 6),
            "cv_mae_mean":      round(_cm, 6) if _cm is not None else None,
            "cv_mae_std":       round(cv_result.get(f"{_t}_MAE_std", 0.0), 6),
            "cv_rmse_pct_mean": round(_cr / _tm * 100, 2) if (_cr and _tm) else None,
            "cv_mae_pct_mean":  round(_cm / _tm * 100, 2) if (_cm and _tm) else None,
            "best_params":      dict(_shared_bp),
        })
    _run_result = {
        "model_type":            "lstm_mlp_multioutput_torch",
        "variant":               table_variant,
        "seed":                  seed,
        "n_features":            len(data["feature_names"]),
        "n_train":               int(split["X_train"].shape[0]),
        "n_val":                 int(split["X_val"].shape[0]),
        "timestamp":             run_dir.name.split("__")[-1],
        "targets":               _targets_out,
        "overall_val_rmse_mean": round(result["mean_RMSE"], 6),
        "overall_cv_rmse_mean":  round(cv_result["overall_RMSE_mean"], 6),
        "outcome_metrics":       _outcome_metrics_list,
    }
    (run_dir / "run_result.json").write_text(json.dumps(_run_result, indent=2))
    print(f"[SAVE] run_result.json → {run_dir}")

    print(f"\n[DONE] Run artifacts saved to: {run_dir}")
    print(f"{'Target':30s}  {'RMSE':>8s}  {'MAE':>8s}")
    for t in TARGETS:
        m = result["per_target"][t]
        print(f"{t:30s}  {m['RMSE']:8.4f}  {m['MAE']:8.4f}")


# =============================================================================
# ==  ENTRY POINT  ============================================================
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LSTM-MLP multi-target optimizer (one shared model for all 14 targets)"
    )
    parser.add_argument(
        "--variant", type=str, default=LSTM_DEFAULT_STATIC_VARIANT,
        help=f"Static feature variant. Default: {LSTM_DEFAULT_STATIC_VARIANT}",
    )
    parser.add_argument(
        "--repeats", type=int, default=3,
        help="Independent runs (each uses a different seed).",
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
