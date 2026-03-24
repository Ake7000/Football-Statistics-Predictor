# classifier_mlp_multioutput_torch.py  (refactor/classifiers/)
# Multi-stat PyTorch MLP: one shared backbone, 7 independent 3-class heads.
# One Optuna study per feature-table variant (minimises mean val CE across all stats).
#
# Predicts sign(HOME_X - AWAY_X) for each of the 7 stat pairs:
#   0 = HOME_WIN, 1 = DRAW, 2 = AWAY_WIN
#
# How to run (from the workspace root):
#   python refactor/classifiers/classifier_mlp_multioutput_torch.py              # all variants
#   python refactor/classifiers/classifier_mlp_multioutput_torch.py --variant form --repeats 1

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # classifiers/ → refactor/

import json
import pickle
import warnings
import argparse
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

from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from shared_config import (
    TARGETS,
    CLASSIFIER_STAT_PAIRS, CLASSIFIER_TARGETS, N_CLASSES,
    ARTIFACTS_CLASSIFIER_MLP_MULTIOUTPUT_ROOT, TRAIN_TABLE_PATH,
    N_TRIALS, EPOCHS, PATIENCE, RETRAIN_EPOCHS, RETRAIN_PATIENCE, BATCH_SIZE_OPTIONS,
    N_HIDDEN_MIN, N_HIDDEN_MAX,
    DROPOUT_MIN, DROPOUT_MAX,
    LR_MIN, LR_MAX, L2_MIN, L2_MAX,
    ACTIVATIONS, OPTIMIZERS,
    UNITS_CHOICES, DEFAULT_UNITS_CHOICES,
    VARIANTS, CV_FOLDS, GLOBAL_SEED,
)
from shared_features import build_feature_matrices, build_full_feature_matrix
from shared_preprocessing import load_and_prepare_dataframe
from shared_utils import (
    set_all_seeds, get_torch_device, make_run_dir,
    get_activation, build_layer_sizes, make_torch_optimizer, ResidualMLP,
)
from shared_metrics import (
    make_direction_labels, make_all_stat_labels_df,
    clf_metrics_dict, OUTCOME_CLASSES,
)

N_STATS = len(CLASSIFIER_STAT_PAIRS)


# =============================================================================
# ==  MODEL DEFINITION  =======================================================
# =============================================================================

class MultiOutputMLPClassifier(nn.Module):
    """
    Shared-backbone MLP with N_STATS independent 3-class output heads.

    Architecture:
        Input (n_features)
          └─► Backbone: ResidualMLP(out_dim=None)  — pre-norm residual blocks
                    └─► N_STATS heads, each: Linear → Act → Linear(n_classes)

    Uses ResidualMLP from shared_utils (same backbone as MLPRegressor /
    MLPClassifier) to avoid duplicating backbone architecture code.
    """

    def __init__(
        self,
        input_dim:      int,
        backbone_sizes: List[int],
        activation:     str,
        dropout:        float,
        n_stats:        int = N_STATS,
        n_classes:      int = N_CLASSES,
    ) -> None:
        super().__init__()
        self.backbone         = ResidualMLP(input_dim, backbone_sizes, activation, dropout)
        self.backbone_out_dim = self.backbone.out_dim
        head_dim = max(16, self.backbone.out_dim // 2)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.backbone.out_dim, head_dim),
                get_activation(activation),
                nn.Linear(head_dim, n_classes),
            )
            for _ in range(n_stats)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, N_STATS, N_CLASSES) logits."""
        z = self.backbone(x)
        return torch.stack([head(z) for head in self.heads], dim=1)


# =============================================================================
# ==  TRAINING LOOP  ==========================================================
# =============================================================================

def train_multiout_classifier(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    patience: int,
    lr_scheduler=None,
    optuna_trial: Optional[optuna.trial.Trial] = None,
    class_weights: Optional[torch.Tensor] = None,   # (N_STATS, N_CLASSES)
) -> Tuple[float, int, Dict]:
    """
    Train MultiOutputMLPClassifier with early stopping on mean val F1 macro.

    Training loss: per-stat weighted CE, averaged across heads.
    Early stopping + Optuna objective: mean F1 macro across N_STATS heads.
    LR scheduler: driven by CE loss (smoother signal).
    Returns (best_val_f1, epochs_run, best_state_dict).
    """
    if class_weights is not None:
        criteria = [
            nn.CrossEntropyLoss(weight=class_weights[i].to(device))
            for i in range(N_STATS)
        ]
    else:
        criteria = [nn.CrossEntropyLoss() for _ in range(N_STATS)]
    ce_unweighted = nn.CrossEntropyLoss()   # for LR scheduler signal

    best_val_f1 = -1.0
    best_state: Optional[Dict] = None
    no_improve  = 0
    epochs_run  = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.long().to(device)   # yb: (B, N_STATS)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)                              # (B, N_STATS, N_CLASSES)
            loss   = sum(
                criteria[i](logits[:, i, :], yb[:, i]) for i in range(N_STATS)
            ) / N_STATS
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ---- Validate ----
        model.eval()
        val_loss_acc = 0.0
        total_samples = 0
        all_preds_per_stat:  List[List[np.ndarray]] = [[] for _ in range(N_STATS)]
        all_labels_per_stat: List[List[np.ndarray]] = [[] for _ in range(N_STATS)]
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.long().to(device)
                n      = xb.size(0)
                logits = model(xb)
                loss   = sum(
                    ce_unweighted(logits[:, i, :], yb[:, i]) for i in range(N_STATS)
                ) / N_STATS
                val_loss_acc  += loss.item() * n
                total_samples += n
                preds = logits.argmax(dim=-1).cpu().numpy()  # (B, N_STATS)
                for i in range(N_STATS):
                    all_preds_per_stat[i].append(preds[:, i])
                    all_labels_per_stat[i].append(yb[:, i].cpu().numpy())

        val_loss = val_loss_acc / total_samples
        f1_per_stat = [
            float(f1_score(
                np.concatenate(all_labels_per_stat[i]),
                np.concatenate(all_preds_per_stat[i]),
                average="macro", zero_division=0,
            ))
            for i in range(N_STATS)
        ]
        val_f1     = float(np.mean(f1_per_stat))
        epochs_run = epoch

        if lr_scheduler is not None:
            lr_scheduler.step(val_loss)   # CE loss → smoother LR signal

        if optuna_trial is not None:
            optuna_trial.report(1.0 - val_f1, step=epoch)
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is None:
        best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_val_f1 = 0.0

    return best_val_f1, epochs_run, best_state


# =============================================================================
# ==  OPTUNA OBJECTIVE  =======================================================
# =============================================================================

def objective_factory(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    units_choices: List[int],
    train_tensor_X: torch.Tensor,
    train_tensor_y: torch.Tensor,
    val_tensor_X: torch.Tensor,
    val_tensor_y: torch.Tensor,
    cw_strategy: str = "none",
):
    # Compute per-stat class weights from y_train  (shape: N_STATS, N_CLASSES)
    if cw_strategy == "sqrt":
        _cw_rows = []
        for _i in range(N_STATS):
            _c = np.bincount(y_train[:, _i], minlength=N_CLASSES).astype(float)
            _c = np.where(_c == 0, 1.0, _c)
            _cw_rows.append(1.0 / np.sqrt(_c))
        class_weights = torch.tensor(np.stack(_cw_rows), dtype=torch.float32)
    else:
        class_weights = None

    def objective(trial: optuna.trial.Trial) -> float:
        n_hidden   = trial.suggest_int("n_hidden",    N_HIDDEN_MIN, N_HIDDEN_MAX)
        base_units = trial.suggest_categorical("base_units", units_choices)
        activation = trial.suggest_categorical("activation", ACTIVATIONS)
        l2_reg     = trial.suggest_float("l2_reg",    L2_MIN, L2_MAX, log=True)
        dropout    = trial.suggest_float("dropout",   DROPOUT_MIN, DROPOUT_MAX)
        lr         = trial.suggest_float("lr",        LR_MIN, LR_MAX, log=True)
        opt_name   = trial.suggest_categorical("optimizer",   OPTIMIZERS)
        bs         = trial.suggest_categorical("batch_size",  BATCH_SIZE_OPTIONS)

        all_mults = [
            trial.suggest_categorical(f"mult_{k}", [0.5, 1.0, 2.0])
            for k in range(1, N_HIDDEN_MAX)
        ]
        layer_sizes = build_layer_sizes(trial.params, units_choices)

        print(
            f"[multi] Trial {trial.number:03d} ▶ START | "
            f"layers={layer_sizes} act={activation} lr={lr:.2e} bs={bs} drop={dropout:.2f}"
        )

        model     = MultiOutputMLPClassifier(X_train.shape[1], layer_sizes, activation, dropout)
        optimizer = make_torch_optimizer(opt_name, model.parameters(), lr, l2_reg)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(PATIENCE // 3, 3), min_lr=1e-6,
        )
        train_ds     = TensorDataset(train_tensor_X, train_tensor_y)
        val_ds       = TensorDataset(val_tensor_X,   val_tensor_y)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False)

        best_f1, epochs_run, _ = train_multiout_classifier(
            model, optimizer, train_loader, val_loader,
            device, EPOCHS, PATIENCE, scheduler, optuna_trial=trial,
            class_weights=class_weights,
        )

        print(
            f"[multi] Trial {trial.number:03d} ✓ DONE  | "
            f"f1={best_f1:.5f} epochs={epochs_run} layers={layer_sizes}"
        )
        return 1.0 - best_f1  # minimize (1 - mean F1 macro)

    return objective


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
    device        = get_torch_device()
    units_choices = UNITS_CHOICES.get(table_variant, DEFAULT_UNITS_CHOICES)

    run_dir = make_run_dir(ARTIFACTS_CLASSIFIER_MLP_MULTIOUTPUT_ROOT, TRAIN_TABLE_PATH, table_variant, suffix=f"__cw_{cw_strategy}")
    print(f"[RUN] Artifacts: {run_dir}")
    print(f"[RUN] Variant: {table_variant} | device: {device} | seed: {seed}")

    df = load_and_prepare_dataframe(TRAIN_TABLE_PATH)

    X_train, X_val, y_train_df, y_val_df, _, feature_names = build_feature_matrices(
        df, table_variant, apply_scaler=True
    )

    y_train_cls = make_all_stat_labels_df(y_train_df, CLASSIFIER_STAT_PAIRS)  # (n_train, N_STATS)
    y_val_cls   = make_all_stat_labels_df(y_val_df,   CLASSIFIER_STAT_PAIRS)  # (n_val,   N_STATS)

    (run_dir / "features_list.txt").write_text("\n".join(feature_names))
    (run_dir / "stat_pairs.txt").write_text(
        "\n".join(f"{s}: {h} vs {a}" for s, h, a in CLASSIFIER_STAT_PAIRS)
    )

    _X_save, _y_save, _, _feat_save = build_full_feature_matrix(df, table_variant, apply_scaler=False)
    _tt = pd.DataFrame(_X_save, columns=_feat_save)
    for _col in TARGETS:
        _tt[_col] = _y_save[_col].values
    _tt.to_csv(run_dir / "training_table.csv", index=False)

    print(f"[FEATURES] {len(feature_names)} features | units_choices: {units_choices}")

    # Pre-build tensors (shared across all N_TRIALS)
    X_train_f32  = X_train.astype(np.float32)
    X_val_f32    = X_val.astype(np.float32)
    train_tensor_X = torch.from_numpy(X_train_f32)
    train_tensor_y = torch.from_numpy(y_train_cls)
    val_tensor_X   = torch.from_numpy(X_val_f32)
    val_tensor_y   = torch.from_numpy(y_val_cls)

    # ---- Optuna study (one for the whole variant) ----
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(multivariate=True, seed=seed),
        pruner=MedianPruner(n_startup_trials=max(5, N_TRIALS // 5)),
    )
    study.optimize(
        objective_factory(
            X_train_f32, y_train_cls, X_val_f32, y_val_cls,
            device, units_choices,
            train_tensor_X, train_tensor_y, val_tensor_X, val_tensor_y,
            cw_strategy,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params      = dict(study.best_trial.params)
    best_layer_sizes = build_layer_sizes(best_params, units_choices)
    export_params    = {k: v for k, v in best_params.items() if not k.startswith("mult_")}
    export_params["layer_sizes"] = best_layer_sizes

    print(
        f"[RETRAIN] best | layers={best_layer_sizes} "
        f"act={best_params.get('activation')} lr={best_params.get('lr', 0):.2e}"
    )

    # ---- Retrain with best config ----
    best_model = MultiOutputMLPClassifier(
        X_train.shape[1], best_layer_sizes,
        best_params["activation"], float(best_params["dropout"]),
    ).to(device)
    optimizer = make_torch_optimizer(
        best_params["optimizer"], best_model.parameters(),
        float(best_params["lr"]), float(best_params["l2_reg"]),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5,
        patience=max(PATIENCE // 3, 3), min_lr=1e-6,
    )
    bs           = int(best_params["batch_size"])
    train_loader = DataLoader(TensorDataset(train_tensor_X, train_tensor_y), batch_size=bs, shuffle=True)
    val_loader   = DataLoader(TensorDataset(val_tensor_X,   val_tensor_y),   batch_size=bs, shuffle=False)

    # Class weights for retrain (same y_train_cls)
    if cw_strategy == "sqrt":
        _retrain_cw_rows = []
        for _i in range(N_STATS):
            _c = np.bincount(y_train_cls[:, _i], minlength=N_CLASSES).astype(float)
            _c = np.where(_c == 0, 1.0, _c)
            _retrain_cw_rows.append(1.0 / np.sqrt(_c))
        _retrain_class_weights = torch.tensor(np.stack(_retrain_cw_rows), dtype=torch.float32)
    else:
        _retrain_class_weights = None

    _, _, best_state = train_multiout_classifier(
        best_model, optimizer, train_loader, val_loader,
        device, RETRAIN_EPOCHS, RETRAIN_PATIENCE, scheduler,
        class_weights=_retrain_class_weights,
    )
    best_model.load_state_dict(best_state)

    # ---- Evaluate on val set (per-stat) ----
    best_model.eval()
    logits_list: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, _ in val_loader:
            logits_list.append(best_model(xb.to(device)).cpu())
    all_logits = torch.cat(logits_list, dim=0)             # (n_val, N_STATS, N_CLASSES)
    y_pred_all = all_logits.argmax(dim=-1).numpy()         # (n_val, N_STATS)
    y_proba_all = torch.softmax(all_logits, dim=-1).numpy()  # (n_val, N_STATS, N_CLASSES)

    # ---- Per-stat metrics ----
    per_stat_metrics = []
    for i, (stat, home_col, away_col) in enumerate(CLASSIFIER_STAT_PAIRS):
        m = clf_metrics_dict(y_val_cls[:, i], y_pred_all[:, i].astype(np.int64))
        stat_dir = run_dir / stat
        stat_dir.mkdir(exist_ok=True)
        (stat_dir / "val_metrics.json").write_text(json.dumps(m, indent=2))
        np.save(str(stat_dir / "val_predictions_proba.npy"), y_proba_all[:, i, :])
        # Class distribution on train
        y_tr_i       = y_train_cls[:, i]
        class_counts = {OUTCOME_CLASSES[c]: int(np.sum(y_tr_i == c)) for c in range(N_CLASSES)}
        (stat_dir / "class_distribution_train.json").write_text(json.dumps(class_counts, indent=2))
        per_stat_metrics.append({"stat": stat, "home_col": home_col, "away_col": away_col, "class_distribution": class_counts, **m})

    # ---- Save shared model ----
    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "input_dim":        X_train.shape[1],
            "backbone_sizes":   best_layer_sizes,
            "activation":       best_params["activation"],
            "dropout":          float(best_params["dropout"]),
            "n_stats":          N_STATS,
            "n_classes":        N_CLASSES,
            "stat_pairs":       CLASSIFIER_STAT_PAIRS,
        },
        run_dir / "best_model.pt",
    )
    (run_dir / "best_params.json").write_text(json.dumps(export_params, indent=2))

    # Clean up study CSV
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
    trials_df.to_csv(run_dir / "study_summary.csv", index=False)

    # ---- Cross-validated evaluation ----
    X_full_raw, y_full_df, _, _ = build_full_feature_matrix(df, table_variant, apply_scaler=False)
    y_full_cls = make_all_stat_labels_df(y_full_df, CLASSIFIER_STAT_PAIRS)

    kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=GLOBAL_SEED)
    cv_acc_per_stat  = [[] for _ in range(N_STATS)]
    cv_f1_per_stat   = [[] for _ in range(N_STATS)]

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_full_raw), start=1):
        X_tr_raw, X_te_raw = X_full_raw[tr_idx], X_full_raw[te_idx]
        y_tr,     y_te     = y_full_cls[tr_idx],  y_full_cls[te_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr_raw).astype(np.float32)
        X_te   = scaler.transform(X_te_raw).astype(np.float32)

        # Per-stat class weights for this fold
        if cw_strategy == "sqrt":
            _fold_cw_rows = []
            for _i in range(N_STATS):
                _c = np.bincount(y_tr[:, _i], minlength=N_CLASSES).astype(float)
                _c = np.where(_c == 0, 1.0, _c)
                _fold_cw_rows.append(1.0 / np.sqrt(_c))
            _fold_cw = torch.tensor(np.stack(_fold_cw_rows), dtype=torch.float32)
        else:
            _fold_cw = None

        tr_tx = torch.from_numpy(X_tr)
        tr_ty = torch.from_numpy(y_tr)
        te_tx = torch.from_numpy(X_te)
        te_ty = torch.from_numpy(y_te)

        fold_model = MultiOutputMLPClassifier(
            X_tr.shape[1], best_layer_sizes,
            best_params["activation"], float(best_params["dropout"]),
        ).to(device)
        fold_opt   = make_torch_optimizer(
            best_params["optimizer"], fold_model.parameters(),
            float(best_params["lr"]), float(best_params["l2_reg"]),
        )
        fold_sched = optim.lr_scheduler.ReduceLROnPlateau(
            fold_opt, mode="min", factor=0.5,
            patience=max(PATIENCE // 3, 3), min_lr=1e-6,
        )
        tr_loader = DataLoader(TensorDataset(tr_tx, tr_ty), batch_size=bs, shuffle=True)
        te_loader = DataLoader(TensorDataset(te_tx, te_ty), batch_size=bs, shuffle=False)

        _, _, fold_state = train_multiout_classifier(
            fold_model, fold_opt, tr_loader, te_loader,
            device, RETRAIN_EPOCHS, RETRAIN_PATIENCE, fold_sched,
            class_weights=_fold_cw,
        )
        fold_model.load_state_dict(fold_state)

        fold_model.eval()
        fold_logits_list: List[torch.Tensor] = []
        with torch.no_grad():
            for xb, _ in te_loader:
                fold_logits_list.append(fold_model(xb.to(device)).cpu())
        fold_logits = torch.cat(fold_logits_list, dim=0)
        fold_preds  = fold_logits.argmax(dim=-1).numpy()   # (n_te, N_STATS)

        for i in range(N_STATS):
            fold_acc = float(accuracy_score(y_te[:, i], fold_preds[:, i]))
            fold_f1  = float(f1_score(y_te[:, i], fold_preds[:, i], average="macro", zero_division=0))
            cv_acc_per_stat[i].append(fold_acc)
            cv_f1_per_stat[i].append(fold_f1)

        print(f"  [CV fold {fold_idx}/{CV_FOLDS}] mean_acc={np.mean([cv_acc_per_stat[i][-1] for i in range(N_STATS)]):.4f}")

    # Save per-stat CV metrics
    cv_rows = []
    for i, (stat, _, _) in enumerate(CLASSIFIER_STAT_PAIRS):
        cv_result = {
            "stat":             stat,
            "cv_acc_mean":      round(float(np.mean(cv_acc_per_stat[i])),  4),
            "cv_acc_std":       round(float(np.std(cv_acc_per_stat[i])),   4),
            "cv_f1_macro_mean": round(float(np.mean(cv_f1_per_stat[i])),   4),
            "cv_f1_macro_std":  round(float(np.std(cv_f1_per_stat[i])),    4),
        }
        cv_rows.append(cv_result)
        (run_dir / stat / "cv_metrics.json").write_text(json.dumps(cv_result, indent=2))
        print(f"[CV {stat}] acc = {cv_result['cv_acc_mean']:.4f} ± {cv_result['cv_acc_std']:.4f}")

    # ---- Run-level summaries ----
    pd.DataFrame(per_stat_metrics).to_csv(run_dir / "summary_all_stats.csv", index=False)
    pd.DataFrame(cv_rows).to_csv(run_dir / "cv_summary_all_stats.csv", index=False)

    _targets_out = []
    for _m, _cv in zip(per_stat_metrics, cv_rows):
        _targets_out.append({
            "stat":               _m["stat"],
            "home_col":           _m["home_col"],
            "away_col":           _m["away_col"],
            "class_distribution": _m["class_distribution"],
            "val_accuracy":       _m["accuracy"],
            "val_f1_macro":       _m["f1_macro"],
            "val_f1_per_class":   _m["f1_per_class"],
            "cv_acc_mean":        _cv["cv_acc_mean"],
            "cv_acc_std":         _cv["cv_acc_std"],
            "cv_f1_macro_mean":   _cv["cv_f1_macro_mean"],
            "cv_f1_macro_std":    _cv["cv_f1_macro_std"],
            "confusion_matrix":   _m["confusion_matrix"],
        })

    _run_result = {
        "model_type":                "classifier_mlp_multioutput",
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
        "best_params":               export_params,
        "targets":                   _targets_out,
        "overall_val_acc_mean":      round(float(np.mean([m["accuracy"] for m in per_stat_metrics])), 4),
        "overall_val_f1_macro_mean": round(float(np.mean([m["f1_macro"] for m in per_stat_metrics])), 4),
        "overall_cv_acc_mean":       round(float(np.mean([c["cv_acc_mean"] for c in cv_rows])), 4),
        "overall_cv_f1_macro_mean":  round(float(np.mean([c["cv_f1_macro_mean"] for c in cv_rows])), 4),
    }
    (run_dir / "run_result.json").write_text(json.dumps(_run_result, indent=2))
    print(f"[SAVE] run_result.json → {run_dir}")

    print(f"\n[DONE] Variant '{table_variant}' complete.")
    print(f"[ARTIFACTS] {run_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-output MLP 3-class direction classifier with Optuna"
    )
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed",    type=int, default=None)
    parser.add_argument("--class_weights", type=str, default="sqrt",
                        choices=["none", "sqrt"],
                        help="Class weight strategy: none=unweighted, sqrt=1/sqrt(count)")
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
