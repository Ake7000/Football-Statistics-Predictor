# optimizer_mlp_torch.py  (refactor/)
# Per-target PyTorch MLP regressors optimised with Optuna.
# All shared logic lives in shared_*.py — this file contains ONLY MLP-specific code.
#
# How to run (from the refactor/ directory):
#   python optimizer_mlp_torch.py              # runs all variants, 3 repeats
#   python refactor/optimizers/optimizer_mlp_torch.py --variant raw --repeats 1

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
from torch.utils.data import TensorDataset, DataLoader

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from shared_config import (
    TARGETS, TARGET_LOSS_MAP,
    ARTIFACTS_MLP_ROOT, TRAIN_TABLE_PATH,
    N_TRIALS, EPOCHS, PATIENCE, RETRAIN_EPOCHS, RETRAIN_PATIENCE, BATCH_SIZE_OPTIONS,
    N_HIDDEN_MIN, N_HIDDEN_MAX,
    DROPOUT_MIN, DROPOUT_MAX,
    LR_MIN, LR_MAX, L2_MIN, L2_MAX,
    ACTIVATIONS, OPTIMIZERS,
    UNITS_CHOICES, DEFAULT_UNITS_CHOICES,
    VARIANTS,
    CV_FOLDS, GLOBAL_SEED,
)
from shared_features import build_feature_matrices, build_full_feature_matrix
from shared_preprocessing import load_and_prepare_dataframe
from shared_utils import (
    set_all_seeds, get_torch_device, make_run_dir, rmse,
    get_activation, build_layer_sizes, make_torch_optimizer, ResidualMLP,
)
from shared_metrics import round_accuracy, compute_outcome_metrics_list


# =============================================================================
# ==  LOSS FACTORY  ===========================================================
# =============================================================================

def get_criterion(target_name: str) -> nn.Module:
    """
    Return the appropriate loss function for a given target.
    Configured via TARGET_LOSS_MAP in shared_config.py.
    """
    loss_type = TARGET_LOSS_MAP.get(target_name, "mse")
    if loss_type == "poisson":
        # log_input=True: model outputs log(rate) — any real value is valid.
        # Loss = exp(pred) - target * pred  (always finite, no NaN risk).
        # At inference, apply exp() to convert back to count predictions.
        return nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")
    return nn.MSELoss()


def decode_predictions(preds: np.ndarray, target_name: str) -> np.ndarray:
    """
    Convert raw model outputs to count-space predictions.
    For Poisson targets (log_input=True), the model outputs log-rates,
    so we exponentiate.  For MSE targets, outputs are used as-is.
    Clipping prevents overflow from very large log-rate predictions.
    """
    if TARGET_LOSS_MAP.get(target_name, "mse") == "poisson":
        return np.exp(np.clip(preds, -10.0, 10.0))
    return preds


# =============================================================================
# ==  CORE TRAINING LOOP  =====================================================
# =============================================================================
# ResidualMLP, get_activation, build_layer_sizes, make_torch_optimizer
# are imported from shared_utils — not redefined here.

def train_one_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    epochs: int,
    patience: int,
    lr_scheduler=None,
    optuna_trial: Optional[optuna.trial.Trial] = None,
) -> Tuple[float, int, Dict[str, torch.Tensor]]:
    """
    Train `model` with early stopping.  Saves best weights by val loss.

    Fixes applied vs original:
      - Weighted batch val loss (accounts for unequal last-batch size).
      - Gradient clipping (max_norm=1.0) prevents exploding gradients.
      - Early-stopping counter reset after LR reduction by ReduceLROnPlateau.
      - Optuna pruning: trial.report + trial.should_prune() called each epoch.

    Returns:
        best_val_rmse   float
        epochs_run      int
        best_state      model state_dict at best validation epoch
    """
    best_val_rmse    = float("inf")
    best_state: Optional[Dict] = None
    epochs_no_improve = 0
    last_lr           = optimizer.param_groups[0]["lr"]
    epochs_run        = 0

    # Pre-compute whether criterion uses log-space output (PoissonNLLLoss).
    _is_poisson = isinstance(criterion, nn.PoissonNLLLoss)

    model.to(device)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)
            loss  = criterion(preds, yb)
            loss.backward()
            # FIX: gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ---- Validate (weighted by batch size) ----
        model.eval()
        val_loss_acc   = 0.0
        total_samples  = 0
        preds_chunks:   List[torch.Tensor] = []
        targets_chunks: List[torch.Tensor] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                n = xb.size(0)
                p = model(xb)
                val_loss_acc  += criterion(p, yb).item() * n
                total_samples += n
                preds_chunks.append(p.cpu())
                targets_chunks.append(yb.cpu())

        # val_loss: used only for LR scheduler (valid plateau signal).
        val_loss = val_loss_acc / total_samples

        # Proper RMSE: convert Poisson log-space output → count-space, then MSE.
        all_preds   = torch.cat(preds_chunks,   dim=0)
        all_targets = torch.cat(targets_chunks, dim=0)
        if _is_poisson:
            all_preds = torch.exp(all_preds).clamp(max=1e6)
        if torch.isnan(all_preds).any() or torch.isinf(all_preds).any():
            val_rmse = float("inf")
        else:
            val_rmse = float(((all_preds - all_targets) ** 2).mean().sqrt())
            if not math.isfinite(val_rmse):
                val_rmse = float("inf")
        # Guard: NaN loss (exploding gradients) must NOT be treated as 0.
        # max(nan, 0.0) == 0.0 in Python, so we must check first.
        val_rmse   = float("inf") if (not math.isfinite(val_loss)) else val_rmse
        epochs_run = epoch

        # ---- LR scheduler ----
        if lr_scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                # FIX: give the model a fresh window at the lower LR
                epochs_no_improve = 0
                last_lr = new_lr

        # ---- Optuna pruning (active per epoch) ----
        if optuna_trial is not None:
            optuna_trial.report(val_rmse, step=epoch)
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        # ---- Early stopping (on proper RMSE) ----
        if val_rmse < best_val_rmse - 1e-7:
            best_val_rmse     = val_rmse
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # Safety fallback: if every val loss was NaN (e.g. bad random init on a
    # rare Poisson edge case), best_state was never assigned.  Use the final
    # model weights rather than crashing — Optuna will score this trial low.
    if best_state is None:
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_val_rmse = float("inf")

    return best_val_rmse, epochs_run, best_state


# =============================================================================
# ==  OPTUNA OBJECTIVE  =======================================================
# =============================================================================

def make_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Build a DataLoader from numpy arrays."""
    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def objective_factory(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    target_name: str,
    device: torch.device,
    units_choices: List[int],
    # FIX: build tensors ONCE outside the objective closure
    train_tensor_X: torch.Tensor,
    train_tensor_y: torch.Tensor,
    val_tensor_X: torch.Tensor,
    val_tensor_y: torch.Tensor,
):
    criterion = get_criterion(target_name)

    def objective(trial: optuna.trial.Trial) -> float:
        # --- Hyperparameter sampling ---
        n_hidden   = trial.suggest_int("n_hidden", N_HIDDEN_MIN, N_HIDDEN_MAX)
        base_units = trial.suggest_categorical("base_units", units_choices)
        activation = trial.suggest_categorical("activation", ACTIVATIONS)
        l2_reg     = trial.suggest_float("l2_reg", L2_MIN, L2_MAX, log=True)
        dropout    = trial.suggest_float("dropout", DROPOUT_MIN, DROPOUT_MAX)
        lr         = trial.suggest_float("lr", LR_MIN, LR_MAX, log=True)
        opt_name   = trial.suggest_categorical("optimizer", OPTIMIZERS)
        bs         = trial.suggest_categorical("batch_size", BATCH_SIZE_OPTIONS)

        # Sample all mult_k up to N_HIDDEN_MAX-1 unconditionally so the search
        # space is static — required for multivariate=True TPESampler.
        # Only the first (n_hidden-1) values are used to build layer_sizes.
        all_mults = [
            trial.suggest_categorical(f"mult_{k}", [0.5, 1.0, 2.0])
            for k in range(1, N_HIDDEN_MAX)
        ]
        layer_sizes = build_layer_sizes(trial.params, units_choices)

        print(
            f"[{target_name}] Trial {trial.number:03d} ▶ START | "
            f"layers={layer_sizes} act={activation} opt={opt_name} "
            f"lr={lr:.2e} bs={bs} drop={dropout:.2f} l2={l2_reg:.2e}"
        )

        # --- Build model & optimizer ---
        model     = ResidualMLP(X_train.shape[1], layer_sizes, activation, dropout, out_dim=1, squeeze=True)
        optimizer = make_torch_optimizer(opt_name, model.parameters(), lr, l2_reg)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(PATIENCE // 3, 3), min_lr=1e-6,
        )

        # --- DataLoaders (reuse pre-built tensors) ---
        train_ds     = TensorDataset(train_tensor_X, train_tensor_y)
        val_ds       = TensorDataset(val_tensor_X, val_tensor_y)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False)

        best_rmse, epochs_run, _ = train_one_model(
            model, optimizer, train_loader, val_loader,
            device, criterion, EPOCHS, PATIENCE, scheduler,
            optuna_trial=trial,
        )

        print(
            f"[{target_name}] Trial {trial.number:03d} ✓ DONE  | "
            f"RMSE={best_rmse:.5f} epochs={epochs_run} layers={layer_sizes}"
        )
        return best_rmse

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
    device: torch.device,
    units_choices: List[int],
    seed: int = GLOBAL_SEED,
) -> Dict:
    """
    Run Optuna study for one target, retrain the best config, save all artifacts.
    Returns a summary dict with metrics and best hyperparameters.
    """
    target_dir = run_dir / target_name
    target_dir.mkdir(parents=True, exist_ok=True)

    # --- Target scaling for MSE targets ---
    # Normalise y so MSE loss gradients are well-conditioned and
    # don't blow up for high-valued targets (CORNERS, SHOTS_ON_TARGET).
    # Poisson targets skip — PoissonNLLLoss works in log-space and is scale-invariant.
    target_scaler: Optional[StandardScaler] = None
    y_train_model = y_train.copy()
    y_val_model   = y_val.copy()
    if TARGET_LOSS_MAP.get(target_name, "mse") == "mse":
        target_scaler = StandardScaler()
        y_train_model = target_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
        y_val_model   = target_scaler.transform(y_val.reshape(-1, 1)).ravel()

    # Build tensors once, shared across all N_TRIALS
    train_tx = torch.from_numpy(X_train).float()
    train_ty = torch.from_numpy(y_train_model).float()
    val_tx   = torch.from_numpy(X_val).float()
    val_ty   = torch.from_numpy(y_val_model).float()

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(multivariate=True, seed=seed),
        pruner=MedianPruner(n_startup_trials=max(5, N_TRIALS // 5)),
    )
    study.optimize(
        objective_factory(
            X_train, y_train, X_val, y_val,
            target_name, device, units_choices,
            train_tx, train_ty, val_tx, val_ty,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params     = dict(study.best_trial.params)
    best_layer_sizes = build_layer_sizes(best_params, units_choices)

    print(
        f"[{target_name}] ▶ RETRAIN best | layers={best_layer_sizes} "
        f"act={best_params.get('activation')} lr={best_params.get('lr', 0):.2e} "
        f"bs={best_params.get('batch_size')} drop={best_params.get('dropout', 0):.2f} "
        f"l2={best_params.get('l2_reg', 0):.2e}"
    )

    criterion = get_criterion(target_name)
    best_model = ResidualMLP(
        X_train.shape[1], best_layer_sizes,
        best_params["activation"], float(best_params["dropout"]),
        out_dim=1, squeeze=True,
    ).to(device)
    optimizer  = make_torch_optimizer(
        best_params["optimizer"], best_model.parameters(),
        float(best_params["lr"]), float(best_params["l2_reg"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=max(PATIENCE // 3, 3), min_lr=1e-6,
    )
    bs           = int(best_params["batch_size"])
    train_loader = DataLoader(TensorDataset(train_tx, train_ty), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_tx,   val_ty),   batch_size=bs, shuffle=False)

    _, _, best_state = train_one_model(
        best_model, optimizer, train_loader, val_loader,
        device, criterion, RETRAIN_EPOCHS, RETRAIN_PATIENCE, scheduler,
    )
    best_model.load_state_dict(best_state)

    # --- Evaluate on val set ---
    best_model.eval()
    preds_list = []
    with torch.no_grad():
        for xb, _ in val_loader:
            preds_list.append(best_model(xb.to(device)).cpu().numpy())
    # Decode raw outputs: exp() for Poisson, inverse_transform for scaled MSE.
    y_val_pred = decode_predictions(np.concatenate(preds_list), target_name)
    if target_scaler is not None:
        y_val_pred = target_scaler.inverse_transform(
            y_val_pred.reshape(-1, 1)
        ).ravel()
    # Compare against original (unscaled) y_val.
    mae       = float(mean_absolute_error(y_val, y_val_pred))
    mse       = float(mean_squared_error(y_val, y_val_pred))
    r         = math.sqrt(mse)
    round_acc = round_accuracy(y_val, y_val_pred)

    # --- Save artifacts ---
    export_params = {k: v for k, v in best_params.items() if not k.startswith("mult_")}
    export_params["layer_sizes"] = best_layer_sizes

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "input_dim":   X_train.shape[1],
            "layer_sizes": best_layer_sizes,
            "activation":  best_params["activation"],
            "dropout":     float(best_params["dropout"]),
        },
        target_dir / "best_model.pt",
    )
    (target_dir / "val_metrics.json").write_text(
        json.dumps({"MAE": mae, "MSE": mse, "RMSE": r}, indent=2)
    )
    (target_dir / "best_params.json").write_text(
        json.dumps(export_params, indent=2)
    )
    if target_scaler is not None:
        with open(target_dir / "target_scaler.pkl", "wb") as f:
            pickle.dump(target_scaler, f)

    # Clean up trials dataframe: add layer_sizes column, drop mult_* columns
    trials_df = study.trials_dataframe()
    if not trials_df.empty:
        def _row_layers(row) -> str:
            try:
                p = {col.replace("params_", ""): row[col]
                     for col in trials_df.columns if col.startswith("params_")}
                return json.dumps(build_layer_sizes(p, units_choices))
            except Exception:
                return "[]"
        trials_df["params_layer_sizes"] = trials_df.apply(_row_layers, axis=1)
        mult_cols = [c for c in trials_df.columns if c.startswith("params_mult_")]
        trials_df.drop(columns=mult_cols, inplace=True, errors="ignore")
    trials_df.to_csv(target_dir / "study_summary.csv", index=False)

    print(f"[SAVE] {target_name}: MAE={mae:.4f}  RMSE={r:.4f}")

    return {
        "target":        target_name,
        "MAE":           mae,
        "MSE":           mse,
        "RMSE":          r,
        "val_round_acc": round(round_acc, 4),
        "val_preds":     y_val_pred,       # popped in main() before saving to CSV
        "best_trial":    int(study.best_trial.number),
        "layer_sizes":   best_layer_sizes,
        **{k: v for k, v in export_params.items() if k != "layer_sizes"},
    }


# =============================================================================
# ==  CROSS-VALIDATED FINAL EVALUATION  =======================================
# =============================================================================

def cross_validate_target(
    target_name: str,
    best_params: Dict,
    X_full: np.ndarray,
    y_full: np.ndarray,
    device: torch.device,
    units_choices: List[int],
    n_folds: int = CV_FOLDS,
) -> Dict:
    """
    Run k-fold CV with the best hyperparameters to produce an honest
    generalization estimate.  Each fold trains from scratch.

    Returns a dict with mean and std of MAE, MSE, RMSE across folds.
    """
    layer_sizes = build_layer_sizes(best_params, units_choices)
    criterion   = get_criterion(target_name)
    bs          = int(best_params["batch_size"])

    kf      = KFold(n_splits=n_folds, shuffle=True, random_state=GLOBAL_SEED)
    metrics = {"MAE": [], "MSE": [], "RMSE": []}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full), start=1):
        X_tr, X_te = X_full[train_idx], X_full[test_idx]
        y_tr, y_te = y_full[train_idx], y_full[test_idx]

        # Scale features per fold (fit on train only — no leakage)
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        # Scale MSE targets per fold (fit on train only)
        fold_target_scaler: Optional[StandardScaler] = None
        y_tr_model = y_tr.copy()
        y_te_model = y_te.copy()   # scaled val targets for the training-loop loss
        if TARGET_LOSS_MAP.get(target_name, "mse") == "mse":
            fold_target_scaler = StandardScaler()
            y_tr_model = fold_target_scaler.fit_transform(y_tr.reshape(-1, 1)).ravel()
            y_te_model = fold_target_scaler.transform(y_te.reshape(-1, 1)).ravel()

        model     = ResidualMLP(X_tr.shape[1], layer_sizes,
                                best_params["activation"], float(best_params["dropout"]),
                                out_dim=1, squeeze=True).to(device)
        optimizer = make_torch_optimizer(
            best_params["optimizer"], model.parameters(),
            float(best_params["lr"]), float(best_params["l2_reg"]),
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(PATIENCE // 3, 3), min_lr=1e-6,
        )

        tr_tx = torch.from_numpy(X_tr).float()
        tr_ty = torch.from_numpy(y_tr_model).float()   # scaled for MSE, unchanged for Poisson
        te_tx = torch.from_numpy(X_te).float()
        te_ty = torch.from_numpy(y_te_model).float()   # scaled val targets for training loop

        train_loader = DataLoader(TensorDataset(tr_tx, tr_ty), batch_size=bs, shuffle=True)
        val_loader   = DataLoader(TensorDataset(te_tx, te_ty), batch_size=bs, shuffle=False)

        _, _, best_state = train_one_model(
            model, optimizer, train_loader, val_loader,
            device, criterion, RETRAIN_EPOCHS, RETRAIN_PATIENCE, scheduler,
        )
        model.load_state_dict(best_state)
        model.eval()

        preds = []
        with torch.no_grad():
            for xb, _ in val_loader:
                preds.append(model(xb.to(device)).cpu().numpy())
        # Decode raw outputs: exp() for Poisson, inverse_transform for scaled MSE.
        y_pred = decode_predictions(np.concatenate(preds), target_name)
        if fold_target_scaler is not None:
            y_pred = fold_target_scaler.inverse_transform(
                y_pred.reshape(-1, 1)
            ).ravel()
        # Compare against original (unscaled) y_te.
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
    device        = get_torch_device()
    units_choices = UNITS_CHOICES.get(table_variant, DEFAULT_UNITS_CHOICES)

    run_dir = make_run_dir(ARTIFACTS_MLP_ROOT, TRAIN_TABLE_PATH, table_variant)
    print(f"[RUN] Artifacts: {run_dir}")
    print(f"[RUN] Variant: {table_variant} | units_choices: {units_choices} | seed: {seed}")

    df = load_and_prepare_dataframe(TRAIN_TABLE_PATH)
    target_means = {t: float(df[t].mean()) for t in TARGETS}

    # MLP uses StandardScaler (apply_scaler=True)
    X_train, X_val, y_train_df, y_val_df, scaler, feature_names = build_feature_matrices(
        df, table_variant, apply_scaler=True
    )

    # Save run-level artifacts
    with open(run_dir / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
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
            run_dir, device, units_choices, seed,
        )
        y_val_by_target[target]     = y_vl
        val_preds_by_target[target] = result.pop("val_preds")
        summary_rows.append(result)

        # --- 5-fold CV for honest final metric ---
        print(f"[CV] Running {CV_FOLDS}-fold CV for {target} …")
        # Full feature matrix (no train/val split; scaler applied per fold inside CV)
        X_full_raw = build_full_feature_matrix(df, table_variant, apply_scaler=False)[0]
        y_full     = df[target].values.astype(float)

        best_params = {k: v for k, v in result.items()
                       if k not in ["target", "MAE", "MSE", "RMSE", "best_trial", "layer_sizes"]}
        best_params["layer_sizes"] = result["layer_sizes"]
        # Reconstruct n_hidden + base_units for build_layer_sizes compatibility
        best_params["n_hidden"]   = len(result["layer_sizes"])
        best_params["base_units"] = result["layer_sizes"][0]

        cv_result = cross_validate_target(
            target, best_params, X_full_raw, y_full,
            device, units_choices,
        )
        cv_result["target"] = target
        cv_rows.append(cv_result)

        # Save per-target CV metrics
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
        "model_type":            "mlp_torch",
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
    parser = argparse.ArgumentParser(description="MLP optimizer with Optuna")
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
