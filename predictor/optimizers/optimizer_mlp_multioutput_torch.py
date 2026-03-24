# optimizer_mlp_multioutput_torch.py  (refactor/)
# Multi-output PyTorch MLP: one shared backbone, 14 independent output heads.
# One Optuna study per feature-table variant (minimises mean val-RMSE across all targets).
# Reuses all shared_*.py modules unchanged.
#
# Key idea:
#   Instead of 14 independent models, a single model learns a shared representation
#   and each head specialises for one target.  Correlated targets (goals/corners,
#   fouls/cards) share backbone gradients, which acts as a regulariser.
#
# How to run (from the refactor/ directory):
#   python optimizer_mlp_multioutput_torch.py                  # all variants, 3 repeats
#   python refactor/optimizers/optimizer_mlp_multioutput_torch.py --variant sum --repeats 1

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
    ARTIFACTS_MLP_MULTIOUTPUT_ROOT, TRAIN_TABLE_PATH,
    N_TRIALS, EPOCHS, PATIENCE, RETRAIN_EPOCHS, RETRAIN_PATIENCE, BATCH_SIZE_OPTIONS,
    N_HIDDEN_MIN, N_HIDDEN_MAX,
    DROPOUT_MIN, DROPOUT_MAX,
    LR_MIN, LR_MAX, L2_MIN, L2_MAX,
    ACTIVATIONS, OPTIMIZERS,
    UNITS_CHOICES,
    DEFAULT_UNITS_CHOICES,
    VARIANTS,
    CV_FOLDS, GLOBAL_SEED,
)
from shared_features import build_feature_matrices, build_full_feature_matrix
from shared_preprocessing import load_and_prepare_dataframe
from shared_utils import (
    set_all_seeds, get_torch_device, make_run_dir,
    get_activation, build_layer_sizes, make_torch_optimizer,
)
from shared_metrics import round_accuracy, compute_outcome_metrics_list


# =============================================================================
# ==  LOSS FACTORY  ===========================================================
# =============================================================================

def get_criteria() -> List[nn.Module]:
    """
    Build one loss function per target (in TARGETS order).

    MSE targets  → nn.MSELoss()
    Poisson targets → nn.PoissonNLLLoss(log_input=True)
        Model outputs log-rates; exp() applied at inference.
    """
    result = []
    for t in TARGETS:
        if TARGET_LOSS_MAP.get(t, "mse") == "poisson":
            result.append(nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean"))
        else:
            result.append(nn.MSELoss())
    return result


def decode_all_predictions(preds: np.ndarray) -> np.ndarray:
    """
    Convert raw model outputs to interpretable count/regression values.

    Args:
        preds: (N, n_targets) raw model outputs.

    Returns:
        Decoded (N, n_targets) array:
          - Poisson targets: exp(clip(pred, -10, 10))  — model outputs log-rates
          - MSE targets:     pred as-is
    """
    out = preds.copy()
    for i, t in enumerate(TARGETS):
        if TARGET_LOSS_MAP.get(t, "mse") == "poisson":
            out[:, i] = np.exp(np.clip(preds[:, i], -10.0, 10.0))
    return out


def build_target_scalers(y_train: np.ndarray) -> Dict[int, StandardScaler]:
    """
    Fit a StandardScaler for each MSE target column.
    Poisson targets are skipped — PoissonNLLLoss works in log-space and is scale-invariant.
    Always fit on the training split only (call before any val/test look-ahead).
    Returns: {column_index: fitted_scaler} for MSE targets only.
    """
    scalers: Dict[int, StandardScaler] = {}
    for i, t in enumerate(TARGETS):
        if TARGET_LOSS_MAP.get(t, "mse") == "mse":
            sc = StandardScaler()
            sc.fit(y_train[:, i:i+1])
            scalers[i] = sc
    return scalers


def decode_and_unscale(
    preds: np.ndarray,
    target_scalers: Dict[int, StandardScaler],
) -> np.ndarray:
    """
    Convert raw model outputs to interpretable count/regression values.
      - Poisson targets: exp(clip(pred, -10, 10))  — log-rates → counts
      - MSE targets with scaler: inverse_transform   — scaled values → original space
      - MSE targets without scaler: as-is
    """
    out = preds.copy()
    for i, t in enumerate(TARGETS):
        if TARGET_LOSS_MAP.get(t, "mse") == "poisson":
            out[:, i] = np.exp(np.clip(preds[:, i], -10.0, 10.0))
        elif i in target_scalers:
            out[:, i] = target_scalers[i].inverse_transform(
                out[:, i].reshape(-1, 1)
            ).ravel()
    return out


# =============================================================================
# ==  MODEL DEFINITION  =======================================================
# =============================================================================
# get_activation is imported from shared_utils — not redefined here.


class MultiOutputMLP(nn.Module):
    """
    Shared-backbone MLP with one independent output head per prediction target.

    Architecture:
        Input (n_features)
            └─► Backbone: [Linear → Activation → Dropout] × backbone_depth
                      └─► n_targets independent heads
                                head (head_hidden=False): Linear(width → 1)
                                head (head_hidden=True):  Linear(width → width//2)
                                                          → Activation
                                                          → Linear(width//2 → 1)

    Note on Poisson targets:
        Heads for Poisson targets output log-rates (any real value).
        exp() is applied at inference time (see decode_all_predictions).
        MSE heads output raw regression values.
        Both are valid; the backbone learns to serve both simultaneously.
    """

    def __init__(
        self,
        input_dim: int,
        backbone_sizes: List[int],
        activation: str,
        dropout: float,
        n_targets: int,
        head_hidden: bool,
    ) -> None:
        super().__init__()

        # ---- Backbone with residual skip connections every 2 layers ----
        self.bb_norms:   nn.ModuleList = nn.ModuleList()   # one LayerNorm per layer (pre-norm)
        self.bb_linears: nn.ModuleList = nn.ModuleList()
        self.bb_acts:    nn.ModuleList = nn.ModuleList()
        self.bb_drops:   nn.ModuleList = nn.ModuleList()
        self.bb_projs:   nn.ModuleList = nn.ModuleList()   # one per complete 2-layer block
        prev = input_dim
        block_in_dim = input_dim
        for i, units in enumerate(backbone_sizes):
            self.bb_norms.append(nn.LayerNorm(prev))
            self.bb_linears.append(nn.Linear(prev, units))
            self.bb_acts.append(get_activation(activation))
            self.bb_drops.append(nn.Dropout(dropout) if dropout > 0.0 else nn.Identity())
            if i % 2 == 0:
                block_in_dim = prev          # record input dim at start of each block
            if i % 2 == 1:
                self.bb_projs.append(
                    nn.Linear(block_in_dim, units, bias=False)
                    if block_in_dim != units else nn.Identity()
                )
            prev = units
        self.backbone_out_dim = prev
        self._n_bb_layers = len(backbone_sizes)

        # ---- Heads ----
        # head_hidden_dim: small per-head hidden layer to let each head
        # specialise from the shared representation.
        head_hidden_dim = max(16, prev // 2)
        self.heads = nn.ModuleList()
        for _ in range(n_targets):
            if head_hidden:
                self.heads.append(nn.Sequential(
                    nn.Linear(prev, head_hidden_dim),
                    get_activation(activation),
                    nn.Linear(head_hidden_dim, 1),
                ))
            else:
                self.heads.append(nn.Linear(prev, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, n_features)
        Returns:
            (batch, n_targets) raw predictions
        """
        # Backbone with residual skip connections every 2 linear layers.
        skip = x
        proj_idx = 0
        for i in range(self._n_bb_layers):
            if i % 2 == 0:
                skip = x           # capture input at the start of each 2-layer block
            x = self.bb_drops[i](self.bb_acts[i](self.bb_linears[i](self.bb_norms[i](x))))
            if i % 2 == 1:
                x = x + self.bb_projs[proj_idx](skip)
                proj_idx += 1
        z = x
        # Each head outputs (batch, 1); cat along dim=1 → (batch, n_targets)
        return torch.cat([head(z) for head in self.heads], dim=1)


# =============================================================================
# ==  COMBINED TRAINING LOOP  =================================================
# =============================================================================
# build_layer_sizes and make_torch_optimizer are imported from shared_utils.

def train_multioutput_model(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    criteria: List[nn.Module],
    epochs: int,
    patience: int,
    lr_scheduler=None,
    optuna_trial: Optional[optuna.trial.Trial] = None,
) -> Tuple[float, int, Optional[Dict]]:
    """
    Train the multi-output model with early stopping.

    Combined loss = uniform mean over all per-target losses.

    Includes all fixes from the single-target training loop:
      - Weighted batch val loss
      - Gradient clipping (max_norm=1.0)
      - Early-stopping counter reset after LR reduction
      - Optuna per-epoch pruning

    Returns:
        best_mean_val_rmse  float
        epochs_run          int
        best_state          state_dict at best validation epoch
    """
    best_val_rmse     = float("inf")
    best_state: Optional[Dict] = None
    epochs_no_improve = 0
    epochs_run        = 0
    n_targets         = len(criteria)

    # Pre-compute which targets use log-space (PoissonNLLLoss) for RMSE conversion.
    is_log_space: List[bool] = [isinstance(c, nn.PoissonNLLLoss) for c in criteria]

    _tw = torch.full((n_targets,), 1.0 / n_targets, dtype=torch.float32, device=device)

    model.to(device)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)   # yb: (batch, n_targets)
            optimizer.zero_grad(set_to_none=True)
            preds = model(xb)                         # (batch, n_targets)
            per_loss = torch.stack([
                criteria[i](preds[:, i], yb[:, i]) for i in range(n_targets)
            ])
            loss = (per_loss * _tw).sum()
            loss.backward()
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
                n     = xb.size(0)
                preds = model(xb)
                per_bl = torch.stack([
                    criteria[i](preds[:, i], yb[:, i]) for i in range(n_targets)
                ])
                batch_loss = (per_bl * _tw).sum()
                val_loss_acc  += batch_loss.item() * n
                total_samples += n
                preds_chunks.append(preds.cpu())
                targets_chunks.append(yb.cpu())

        # val_loss: used only for LR scheduler (valid plateau signal).
        val_loss = val_loss_acc / total_samples

        # Proper RMSE: convert Poisson log-space outputs → count-space, then MSE.
        all_preds   = torch.cat(preds_chunks,   dim=0)   # (N, n_targets)
        all_targets = torch.cat(targets_chunks, dim=0)   # (N, n_targets)
        for i, log_sp in enumerate(is_log_space):
            if log_sp:
                all_preds[:, i] = torch.exp(all_preds[:, i]).clamp(max=1e6)
        if torch.isnan(all_preds).any() or torch.isinf(all_preds).any():
            val_rmse = float("inf")
        else:
            val_rmse = float(((all_preds - all_targets) ** 2).mean(dim=0).mean().sqrt())
            if not math.isfinite(val_rmse):
                val_rmse = float("inf")
        epochs_run = epoch

        # ---- LR scheduler ----
        if lr_scheduler is not None:
            current_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]["lr"]
            if new_lr < current_lr:
                epochs_no_improve = 0   # fresh window at lower LR

        # ---- Optuna pruning (per-epoch) ----
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

    # Safety fallback: NaN loss on every epoch → use final weights
    if best_state is None:
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_val_rmse = float("inf")

    return best_val_rmse, epochs_run, best_state


# =============================================================================
# ==  OPTUNA OBJECTIVE  =======================================================
# =============================================================================

def objective_factory(
    X_train: np.ndarray,
    y_train: np.ndarray,          # (N_train, n_targets)
    X_val: np.ndarray,
    y_val: np.ndarray,            # (N_val,   n_targets)
    device: torch.device,
    units_choices: List[int],
    criteria: List[nn.Module],
    # Tensors built once outside the closure (avoid repeated allocation)
    train_tensor_X: torch.Tensor,
    train_tensor_y: torch.Tensor,
    val_tensor_X: torch.Tensor,
    val_tensor_y: torch.Tensor,
):
    n_targets = len(TARGETS)

    def objective(trial: optuna.trial.Trial) -> float:
        # ---- Hyperparameter sampling ----
        n_hidden    = trial.suggest_int("n_hidden", N_HIDDEN_MIN, N_HIDDEN_MAX)
        base_units  = trial.suggest_categorical("base_units", units_choices)
        activation  = trial.suggest_categorical("activation", ACTIVATIONS)
        l2_reg      = trial.suggest_float("l2_reg", L2_MIN, L2_MAX, log=True)
        dropout     = trial.suggest_float("dropout", DROPOUT_MIN, DROPOUT_MAX)
        lr          = trial.suggest_float("lr", LR_MIN, LR_MAX, log=True)
        opt_name    = trial.suggest_categorical("optimizer", OPTIMIZERS)
        bs          = trial.suggest_categorical("batch_size", BATCH_SIZE_OPTIONS)
        head_hidden = trial.suggest_categorical("head_hidden", [True, False])

        # Sample all mult_k up to N_HIDDEN_MAX-1 unconditionally so the search
        # space is static — required for multivariate=True TPESampler.
        # Only the first (n_hidden-1) values are used to build layer_sizes.
        all_mults = [
            trial.suggest_categorical(f"mult_{k}", [0.5, 1.0, 2.0])
            for k in range(1, N_HIDDEN_MAX)
        ]
        layer_sizes = build_layer_sizes(trial.params, units_choices)

        print(
            f"[MULTIOUTPUT] Trial {trial.number:03d} ▶ START | "
            f"layers={layer_sizes} act={activation} lr={lr:.2e} "
            f"bs={bs} drop={dropout:.2f} l2={l2_reg:.2e} head_hidden={head_hidden}"
        )

        model     = MultiOutputMLP(
            X_train.shape[1], layer_sizes, activation, dropout,
            n_targets, head_hidden,
        )
        optimizer = make_torch_optimizer(opt_name, model.parameters(), lr, l2_reg)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(PATIENCE // 3, 3), min_lr=1e-6,
        )

        train_ds     = TensorDataset(train_tensor_X, train_tensor_y)
        val_ds       = TensorDataset(val_tensor_X,   val_tensor_y)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False)

        best_rmse, epochs_run, _ = train_multioutput_model(
            model, optimizer, train_loader, val_loader,
            device, criteria, EPOCHS, PATIENCE, scheduler,
            optuna_trial=trial,
        )

        print(
            f"[MULTIOUTPUT] Trial {trial.number:03d} ✓ DONE  | "
            f"mean_RMSE={best_rmse:.5f} epochs={epochs_run} layers={layer_sizes}"
        )
        return best_rmse

    return objective


# =============================================================================
# ==  EVALUATION HELPER  ======================================================
# =============================================================================

def evaluate_model(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    y_val: np.ndarray,   # (N_val, n_targets) ORIGINAL (unscaled) targets
    target_scalers: Optional[Dict[int, StandardScaler]] = None,
) -> Dict:
    """
    Evaluate a trained MultiOutputMLP.

    y_val must contain original (unscaled) target values.
    target_scalers: per-column scalers fitted on the train split (MSE targets only).
                    Used to invert scaling before computing RMSE/MAE.

    Returns a dict with:
      "per_target": {target_name: {"MAE": ..., "MSE": ..., "RMSE": ...}}
      "mean_RMSE":  float  (mean RMSE across all targets)
    """
    model.eval()
    all_preds: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in val_loader:
            all_preds.append(model(xb.to(device)).cpu().numpy())  # (batch, n_targets)

    raw_preds = np.vstack(all_preds)        # (N_val, n_targets)
    decoded   = decode_and_unscale(raw_preds, target_scalers or {})

    per_target: Dict[str, Dict[str, float]] = {}
    rmse_list: List[float] = []
    for i, target in enumerate(TARGETS):
        mae  = float(mean_absolute_error(y_val[:, i], decoded[:, i]))
        mse  = float(mean_squared_error(y_val[:, i],  decoded[:, i]))
        r    = math.sqrt(mse)
        per_target[target] = {"MAE": mae, "MSE": mse, "RMSE": r}
        rmse_list.append(r)

    return {
        "per_target": per_target,
        "mean_RMSE":  float(np.mean(rmse_list)),
        "val_preds":  decoded,
    }


# =============================================================================
# ==  VARIANT STUDY (ONE STUDY FOR ALL 14 TARGETS)  ===========================
# =============================================================================

def run_variant_study(
    table_variant: str,
    X_train: np.ndarray,
    y_train: np.ndarray,   # (N_train, n_targets) — original, unscaled targets
    X_val: np.ndarray,
    y_val: np.ndarray,     # (N_val,   n_targets) — original, unscaled targets
    run_dir: Path,
    device: torch.device,
    units_choices: List[int],
    criteria: List[nn.Module],
    seed: int = GLOBAL_SEED,
) -> Dict:
    """
    Run one Optuna study for all 14 targets simultaneously.
    Retrain best configuration and save all artifacts.

    Returns a summary dict with best hyperparameters and val metrics.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    n_targets = len(TARGETS)

    # Fit one StandardScaler per MSE target column — normalises y for loss conditioning.
    # Poisson targets skip (PoissonNLLLoss operates in log-space; no scaling needed).
    target_scalers = build_target_scalers(y_train)
    y_train_model  = y_train.copy()
    y_val_model    = y_val.copy()
    for idx, sc in target_scalers.items():
        y_train_model[:, idx] = sc.transform(y_train_model[:, idx:idx+1]).ravel()
        y_val_model[:, idx]   = sc.transform(y_val_model[:, idx:idx+1]).ravel()

    # Build tensors once; shared across all N_TRIALS
    train_tx = torch.from_numpy(X_train).float()
    train_ty = torch.from_numpy(y_train_model).float()
    val_tx   = torch.from_numpy(X_val).float()
    val_ty   = torch.from_numpy(y_val_model).float()

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(multivariate=True, seed=seed),
        study_name=f"multioutput_{table_variant}",
        pruner=MedianPruner(n_startup_trials=max(5, N_TRIALS // 5)),
    )
    study.optimize(
        objective_factory(
            X_train, y_train, X_val, y_val,
            device, units_choices, criteria,
            train_tx, train_ty, val_tx, val_ty,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params      = dict(study.best_trial.params)
    best_layer_sizes = build_layer_sizes(best_params, units_choices)
    head_hidden      = bool(best_params.get("head_hidden", False))

    print(
        f"\n[MULTIOUTPUT] ▶ RETRAIN best | "
        f"layers={best_layer_sizes} act={best_params.get('activation')} "
        f"lr={best_params.get('lr', 0):.2e} bs={best_params.get('batch_size')} "
        f"drop={best_params.get('dropout', 0):.2f} l2={best_params.get('l2_reg', 0):.2e} "
        f"head_hidden={head_hidden}"
    )

    best_model = MultiOutputMLP(
        X_train.shape[1], best_layer_sizes,
        best_params["activation"], float(best_params["dropout"]),
        n_targets, head_hidden,
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

    _, _, best_state = train_multioutput_model(
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
            "model_state_dict": best_model.state_dict(),
            "input_dim":    X_train.shape[1],
            "layer_sizes":  best_layer_sizes,
            "activation":   best_params["activation"],
            "dropout":      float(best_params["dropout"]),
            "head_hidden":  head_hidden,
            "n_targets":    n_targets,
            "targets":      TARGETS,
        },
        run_dir / "best_model.pt",
    )
    (run_dir / "val_metrics_per_target.json").write_text(json.dumps(metrics, indent=2))
    (run_dir / "best_params.json").write_text(json.dumps(export_params, indent=2))
    with open(run_dir / "target_scalers.pkl", "wb") as f:
        pickle.dump(target_scalers, f)

    # Flat CSV: one row per target
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

    # Optuna trials dataframe
    trials_df = study.trials_dataframe()
    if not trials_df.empty:
        def _row_layers(row) -> str:
            try:
                p = {
                    col.replace("params_", ""): row[col]
                    for col in trials_df.columns if col.startswith("params_")
                }
                return json.dumps(build_layer_sizes(p, units_choices))
            except Exception:
                return "[]"
        trials_df["params_layer_sizes"] = trials_df.apply(_row_layers, axis=1)
        mult_cols = [c for c in trials_df.columns if c.startswith("params_mult_")]
        trials_df.drop(columns=mult_cols, inplace=True, errors="ignore")
    trials_df.to_csv(run_dir / "study_summary.csv", index=False)

    print(f"\n[MULTIOUTPUT] Val results — mean_RMSE={metrics['mean_RMSE']:.4f}")
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
    best_params: Dict,
    X_full: np.ndarray,    # raw (not yet scaled) full feature matrix
    y_full: np.ndarray,    # (N, n_targets) original targets
    device: torch.device,
    units_choices: List[int],
    criteria: List[nn.Module],
    n_folds: int = CV_FOLDS,
) -> Dict:
    """
    Run k-fold CV with the best hyperparameters for an honest generalisation estimate.
    Each fold fits its own StandardScaler on train split only.

    Returns per-target RMSE mean/std across folds, plus an overall mean.
    """
    layer_sizes  = build_layer_sizes(best_params, units_choices)
    head_hidden  = bool(best_params.get("head_hidden", False))
    bs           = int(best_params["batch_size"])
    n_targets    = len(TARGETS)

    kf         = KFold(n_splits=n_folds, shuffle=True, random_state=GLOBAL_SEED)
    # fold_rmses[target] = list of per-fold RMSE values
    fold_rmses: Dict[str, List[float]] = {t: [] for t in TARGETS}
    fold_maes:  Dict[str, List[float]] = {t: [] for t in TARGETS}

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full), start=1):
        X_tr, X_te = X_full[train_idx], X_full[test_idx]
        y_tr, y_te = y_full[train_idx], y_full[test_idx]

        # Scale features per fold (fit on train only — no leakage)
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr)
        X_te   = scaler.transform(X_te)

        # Scale MSE target columns per fold (fit on train only)
        fold_target_scalers: Dict[int, StandardScaler] = {}
        y_tr_model = y_tr.copy()
        y_te_model = y_te.copy()   # scaled val targets for the training-loop loss
        for i, t in enumerate(TARGETS):
            if TARGET_LOSS_MAP.get(t, "mse") == "mse":
                sc = StandardScaler()
                y_tr_model[:, i] = sc.fit_transform(y_tr[:, i:i+1]).ravel()
                y_te_model[:, i] = sc.transform(y_te[:, i:i+1]).ravel()
                fold_target_scalers[i] = sc

        model     = MultiOutputMLP(
            X_tr.shape[1], layer_sizes,
            best_params["activation"], float(best_params["dropout"]),
            n_targets, head_hidden,
        ).to(device)
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

        _, _, best_state = train_multioutput_model(
            model, optimizer, train_loader, val_loader,
            device, criteria, RETRAIN_EPOCHS, RETRAIN_PATIENCE, scheduler,
        )
        model.load_state_dict(best_state)
        model.eval()

        preds_list: List[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in val_loader:
                preds_list.append(model(xb.to(device)).cpu().numpy())

        raw_preds = np.vstack(preds_list)        # (N_test, n_targets)
        decoded   = decode_and_unscale(raw_preds, fold_target_scalers)

        fold_mean_rmse = 0.0
        for i, target in enumerate(TARGETS):
            fold_rmse = math.sqrt(float(mean_squared_error(y_te[:, i], decoded[:, i])))
            fold_mae  = float(mean_absolute_error(y_te[:, i], decoded[:, i]))
            fold_rmses[target].append(fold_rmse)
            fold_maes[target].append(fold_mae)
            fold_mean_rmse += fold_rmse
        fold_mean_rmse /= n_targets
        print(f"  [CV] Fold {fold_idx}/{n_folds} → mean_RMSE={fold_mean_rmse:.4f}")

    result: Dict = {}
    all_means: List[float] = []
    for target in TARGETS:
        vals  = fold_rmses[target]
        mvals = fold_maes[target]
        result[f"{target}_RMSE_mean"] = float(np.mean(vals))
        result[f"{target}_RMSE_std"]  = float(np.std(vals))
        result[f"{target}_MAE_mean"]  = float(np.mean(mvals))
        result[f"{target}_MAE_std"]   = float(np.std(mvals))
        all_means.append(float(np.mean(vals)))

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
        raise ValueError(
            f"Unknown variant '{table_variant}'. Allowed: {sorted(VARIANTS.keys())}"
        )

    set_all_seeds(seed)
    device        = get_torch_device()
    units_choices = UNITS_CHOICES.get(table_variant, DEFAULT_UNITS_CHOICES)
    criteria      = get_criteria()

    run_dir = make_run_dir(ARTIFACTS_MLP_MULTIOUTPUT_ROOT, TRAIN_TABLE_PATH, table_variant)
    print(f"[RUN] Artifacts: {run_dir}")
    print(f"[RUN] Variant: {table_variant} | units: {units_choices} | seed: {seed}")

    df = load_and_prepare_dataframe(TRAIN_TABLE_PATH)
    target_means = {t: float(df[t].mean()) for t in TARGETS}

    # MLP uses StandardScaler (apply_scaler=True); scaling is on features only,
    # targets are never scaled.
    X_train, X_val, y_train_df, y_val_df, scaler, feature_names = build_feature_matrices(
        df, table_variant, apply_scaler=True
    )

    # Stack all 14 targets into 2D arrays (targets are NOT scaled)
    y_train = y_train_df[TARGETS].values.astype(float)   # (N_train, 14)
    y_val   = y_val_df[TARGETS].values.astype(float)     # (N_val,   14)

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

    print(f"[FEATURES] {len(feature_names)} features | {len(TARGETS)} targets")
    print(f"[DATA]     train={X_train.shape[0]}  val={X_val.shape[0]}")

    # ---- One Optuna study for all 14 targets ----
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    print(f"\n[INFO] Running multi-output Optuna study ({N_TRIALS} trials, variant='{table_variant}') …")

    result = run_variant_study(
        table_variant, X_train, y_train, X_val, y_val,
        run_dir, device, units_choices, criteria, seed,
    )
    _round_accs           = result.pop("round_accs", {})
    _outcome_metrics_list = result.pop("outcome_metrics", [])

    # ---- 5-fold CV for honest generalisation estimate ----
    print(f"\n[CV] Running {CV_FOLDS}-fold CV …")
    X_full_raw = build_full_feature_matrix(df, table_variant, apply_scaler=False)[0]
    y_full     = df[TARGETS].values.astype(float)   # (N, 14)

    # Reconstruct best_params in the format expected by build_layer_sizes
    best_params = {
        k: v for k, v in result.items()
        if k not in ["layer_sizes", "mean_RMSE", "per_target"]
    }
    best_params["layer_sizes"] = result["layer_sizes"]
    best_params["n_hidden"]    = len(result["layer_sizes"])
    best_params["base_units"]  = result["layer_sizes"][0]

    cv_result = cross_validate_variant(
        best_params, X_full_raw, y_full, device, units_choices, criteria,
    )

    (run_dir / "cv_metrics.json").write_text(json.dumps(cv_result, indent=2))
    pd.DataFrame(
        [
            {
                "target":    t,
                "RMSE_mean": cv_result[f"{t}_RMSE_mean"],
                "RMSE_std":  cv_result[f"{t}_RMSE_std"],
                "MAE_mean":  cv_result[f"{t}_MAE_mean"],
                "MAE_std":   cv_result[f"{t}_MAE_std"],
            }
            for t in TARGETS
        ]
    ).to_csv(run_dir / "cv_summary_all_targets.csv", index=False)

    # --- Write unified run_result.json ---
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
        "model_type":            "mlp_multioutput_torch",
        "variant":               table_variant,
        "seed":                  seed,
        "n_features":            len(feature_names),
        "n_train":               int(X_train.shape[0]),
        "n_val":                 int(X_val.shape[0]),
        "timestamp":             run_dir.name.split("__")[-1],
        "targets":               _targets_out,
        "overall_val_rmse_mean": round(result["mean_RMSE"], 6),
        "overall_cv_rmse_mean":  round(cv_result["overall_RMSE_mean"], 6),
        "outcome_metrics":       _outcome_metrics_list,
    }
    (run_dir / "run_result.json").write_text(json.dumps(_run_result, indent=2))
    print(f"[SAVE] run_result.json → {run_dir}")

    print(f"\n[DONE] Variant '{table_variant}' complete.")
    print(f"[ARTIFACTS] {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-output MLP optimizer with Optuna")
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
