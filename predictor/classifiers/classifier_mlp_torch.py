# classifier_mlp_torch.py  (refactor/classifiers/)
# Per-stat PyTorch MLP 3-class direction classifiers optimised with Optuna.
#
# Predicts sign(HOME_X - AWAY_X) for each stat pair:
#   0 = HOME_WIN  (HOME > AWAY)
#   1 = DRAW      (HOME == AWAY)
#   2 = AWAY_WIN  (HOME < AWAY)
#
# How to run (from the workspace root):
#   python refactor/classifiers/classifier_mlp_torch.py              # all variants, 3 repeats
#   python refactor/classifiers/classifier_mlp_torch.py --variant form --repeats 1 --seed 42

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))  # classifiers/ → refactor/

import json
import math
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
    ARTIFACTS_CLASSIFIER_MLP_ROOT, TRAIN_TABLE_PATH,
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
from shared_metrics import make_direction_labels, make_stat_labels_df, clf_metrics_dict, OUTCOME_CLASSES


# =============================================================================
# ==  TRAINING LOOP  ==========================================================
# =============================================================================
# ResidualMLP, get_activation, build_layer_sizes, make_torch_optimizer,
# make_stat_labels_df are imported from shared_utils / shared_metrics.


def train_classifier_model(
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
) -> Tuple[float, int, Dict]:
    """
    Train MLPClassifier with early stopping on val F1 macro.

    Training loss: CrossEntropyLoss with class weights (gradient signal).
    Early stopping + Optuna objective: val F1 macro.
    LR scheduler: driven by CE loss (smoother signal).
    Returns (best_val_f1, epochs_run, best_state_dict).
    """
    best_val_f1  = -1.0
    best_state: Optional[Dict] = None
    no_improve      = 0
    epochs_run      = 0

    model.to(device)

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.long().to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # ---- Validate ----
        model.eval()
        val_loss_acc = 0.0
        total_samples = 0
        all_preds:  List[np.ndarray] = []
        all_labels: List[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.long().to(device)
                n             = xb.size(0)
                logits        = model(xb)
                val_loss_acc += criterion(logits, yb).item() * n
                total_samples += n
                all_preds.append(logits.argmax(dim=-1).cpu().numpy())
                all_labels.append(yb.cpu().numpy())

        val_loss   = val_loss_acc / total_samples
        y_p        = np.concatenate(all_preds)
        y_t        = np.concatenate(all_labels)
        val_f1     = float(f1_score(y_t, y_p, average="macro", zero_division=0))
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
    stat_name: str,
    device: torch.device,
    units_choices: List[int],
    train_tensor_X: torch.Tensor,
    train_tensor_y: torch.Tensor,
    val_tensor_X: torch.Tensor,
    val_tensor_y: torch.Tensor,
    cw_strategy: str = "none",
):
    if cw_strategy == "sqrt":
        counts = np.bincount(y_train, minlength=N_CLASSES).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w      = torch.tensor(1.0 / np.sqrt(counts), dtype=torch.float32)
    else:
        w = None

    def objective(trial: optuna.trial.Trial) -> float:
        criterion = nn.CrossEntropyLoss(weight=w.to(device) if w is not None else None)
        n_hidden   = trial.suggest_int("n_hidden",    N_HIDDEN_MIN, N_HIDDEN_MAX)
        base_units = trial.suggest_categorical("base_units", units_choices)
        activation = trial.suggest_categorical("activation", ACTIVATIONS)
        l2_reg     = trial.suggest_float("l2_reg",    L2_MIN, L2_MAX, log=True)
        dropout    = trial.suggest_float("dropout",   DROPOUT_MIN, DROPOUT_MAX)
        lr         = trial.suggest_float("lr",        LR_MIN, LR_MAX, log=True)
        opt_name   = trial.suggest_categorical("optimizer", OPTIMIZERS)
        bs         = trial.suggest_categorical("batch_size", BATCH_SIZE_OPTIONS)

        all_mults = [
            trial.suggest_categorical(f"mult_{k}", [0.5, 1.0, 2.0])
            for k in range(1, N_HIDDEN_MAX)
        ]
        layer_sizes = build_layer_sizes(trial.params, units_choices)

        print(
            f"[{stat_name}] Trial {trial.number:03d} ▶ START | "
            f"layers={layer_sizes} act={activation} lr={lr:.2e} bs={bs} drop={dropout:.2f}"
        )

        model     = ResidualMLP(X_train.shape[1], layer_sizes, activation, dropout,
                                out_dim=N_CLASSES, squeeze=False)
        optimizer = make_torch_optimizer(opt_name, model.parameters(), lr, l2_reg)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5,
            patience=max(PATIENCE // 3, 3), min_lr=1e-6,
        )

        train_ds     = TensorDataset(train_tensor_X, train_tensor_y)
        val_ds       = TensorDataset(val_tensor_X,   val_tensor_y)
        train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  drop_last=False)
        val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, drop_last=False)

        best_f1, epochs_run, _ = train_classifier_model(
            model, optimizer, train_loader, val_loader,
            device, criterion, EPOCHS, PATIENCE, scheduler,
            optuna_trial=trial,
        )

        print(
            f"[{stat_name}] Trial {trial.number:03d} ✓ DONE  | "
            f"f1={best_f1:.5f} epochs={epochs_run} layers={layer_sizes}"
        )
        return 1.0 - best_f1  # minimize (1 - F1 macro)

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
    device: torch.device,
    units_choices: List[int],
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

    # Pre-build tensors once (shared across all N_TRIALS)
    train_tx = torch.from_numpy(X_train).float()
    train_ty = torch.from_numpy(y_train_cls)
    val_tx   = torch.from_numpy(X_val).float()
    val_ty   = torch.from_numpy(y_val_cls)

    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(multivariate=True, seed=seed),
        pruner=MedianPruner(n_startup_trials=max(5, N_TRIALS // 5)),
    )
    study.optimize(
        objective_factory(
            X_train, y_train_cls, X_val, y_val_cls,
            stat, device, units_choices,
            train_tx, train_ty, val_tx, val_ty,
            cw_strategy,
        ),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best_params      = dict(study.best_trial.params)
    best_layer_sizes = build_layer_sizes(best_params, units_choices)

    print(
        f"[{stat}] ▶ RETRAIN best | layers={best_layer_sizes} "
        f"act={best_params.get('activation')} lr={best_params.get('lr', 0):.2e} "
        f"bs={best_params.get('batch_size')} drop={best_params.get('dropout', 0):.2f}"
    )

    if cw_strategy == "sqrt":
        _counts   = np.bincount(y_train_cls, minlength=N_CLASSES).astype(float)
        _counts   = np.where(_counts == 0, 1.0, _counts)
        _w        = torch.tensor(1.0 / np.sqrt(_counts), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=_w)
    else:
        criterion = nn.CrossEntropyLoss()
    best_model = ResidualMLP(
        X_train.shape[1], best_layer_sizes,
        best_params["activation"], float(best_params["dropout"]),
        out_dim=N_CLASSES, squeeze=False,
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

    _, _, best_state = train_classifier_model(
        best_model, optimizer, train_loader, val_loader,
        device, criterion, RETRAIN_EPOCHS, RETRAIN_PATIENCE, scheduler,
    )
    best_model.load_state_dict(best_state)

    # --- Evaluate on val set ---
    best_model.eval()
    logits_list: List[torch.Tensor] = []
    with torch.no_grad():
        for xb, _ in val_loader:
            logits_list.append(best_model(xb.to(device)).cpu())
    all_logits = torch.cat(logits_list, dim=0)
    y_pred     = all_logits.argmax(dim=1).numpy().astype(np.int64)
    y_proba    = torch.softmax(all_logits, dim=1).numpy()

    metrics = clf_metrics_dict(y_val_cls, y_pred)

    # --- Save artifacts ---
    export_params = {k: v for k, v in best_params.items() if not k.startswith("mult_")}
    export_params["layer_sizes"] = best_layer_sizes

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "input_dim":        X_train.shape[1],
            "layer_sizes":      best_layer_sizes,
            "activation":       best_params["activation"],
            "dropout":          float(best_params["dropout"]),
            "n_classes":        N_CLASSES,
            "stat":             stat,
        },
        stat_dir / "best_model.pt",
    )
    (stat_dir / "val_metrics.json").write_text(json.dumps(metrics, indent=2))
    (stat_dir / "best_params.json").write_text(json.dumps(export_params, indent=2))
    (stat_dir / "class_distribution_train.json").write_text(
        json.dumps(class_counts, indent=2)
    )
    np.save(str(stat_dir / "val_predictions_proba.npy"), y_proba)

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
    trials_df.to_csv(stat_dir / "study_summary.csv", index=False)

    print(f"[SAVE] {stat}: acc={metrics['accuracy']:.4f}  f1_macro={metrics['f1_macro']:.4f}")

    return {
        "stat":                     stat,
        "home_col":                 home_col,
        "away_col":                 away_col,
        "train_class_distribution": class_counts,
        **metrics,
        "best_trial":               int(study.best_trial.number),
        "best_params_dict":         {k: v for k, v in best_params.items() if not k.startswith("mult_")},
        "val_preds_proba":          y_proba,   # popped in main() before saving CSV
    }


# =============================================================================
# ==  CROSS-VALIDATED FINAL EVALUATION  =======================================
# =============================================================================

def cross_validate_stat(
    stat: str,
    home_col: str,
    away_col: str,
    best_params: Dict,
    X_full_raw: np.ndarray,
    y_labels_full: np.ndarray,
    device: torch.device,
    units_choices: List[int],
    n_folds: int = CV_FOLDS,
    cw_strategy: str = "none",
) -> Dict:
    """
    Run k-fold CV with the best hyperparameters to produce an honest
    generalization estimate.  Each fold fits a fresh StandardScaler on train split.
    """
    layer_sizes = build_layer_sizes(best_params, units_choices)
    bs          = int(best_params["batch_size"])
    kf          = KFold(n_splits=n_folds, shuffle=True, random_state=GLOBAL_SEED)
    acc_list:   List[float] = []
    f1_list:    List[float] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_full_raw), start=1):
        X_tr_raw, X_te_raw = X_full_raw[train_idx], X_full_raw[test_idx]
        y_tr,     y_te     = y_labels_full[train_idx], y_labels_full[test_idx]

        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X_tr_raw).astype(np.float32)
        X_te   = scaler.transform(X_te_raw).astype(np.float32)

        if cw_strategy == "sqrt":
            _cv_counts = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
            _cv_counts = np.where(_cv_counts == 0, 1.0, _cv_counts)
            _cv_w      = torch.tensor(1.0 / np.sqrt(_cv_counts), dtype=torch.float32).to(device)
            criterion  = nn.CrossEntropyLoss(weight=_cv_w)
        else:
            criterion  = nn.CrossEntropyLoss()

        train_tx = torch.from_numpy(X_tr)
        train_ty = torch.from_numpy(y_tr)
        val_tx   = torch.from_numpy(X_te)
        val_ty   = torch.from_numpy(y_te)

        model     = ResidualMLP(
            X_tr.shape[1], layer_sizes,
            best_params["activation"], float(best_params["dropout"]),
            out_dim=N_CLASSES, squeeze=False,
        ).to(device)
        opt       = make_torch_optimizer(
            best_params["optimizer"], model.parameters(),
            float(best_params["lr"]), float(best_params["l2_reg"]),
        )
        sched     = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5,
            patience=max(PATIENCE // 3, 3), min_lr=1e-6,
        )
        tr_loader = DataLoader(TensorDataset(train_tx, train_ty), batch_size=bs, shuffle=True)
        vl_loader = DataLoader(TensorDataset(val_tx,   val_ty),   batch_size=bs, shuffle=False)

        _, _, best_state = train_classifier_model(
            model, opt, tr_loader, vl_loader,
            device, criterion, RETRAIN_EPOCHS, RETRAIN_PATIENCE, sched,
        )  # criterion carries per-fold class weights
        model.load_state_dict(best_state)

        model.eval()
        preds_list: List[np.ndarray] = []
        with torch.no_grad():
            for xb, _ in vl_loader:
                preds_list.append(model(xb.to(device)).cpu().argmax(dim=1).numpy())
        y_pred = np.concatenate(preds_list).astype(np.int64)

        fold_acc = float(accuracy_score(y_te, y_pred))
        fold_f1  = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
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
    device        = get_torch_device()
    units_choices = UNITS_CHOICES.get(table_variant, DEFAULT_UNITS_CHOICES)

    run_dir = make_run_dir(ARTIFACTS_CLASSIFIER_MLP_ROOT, TRAIN_TABLE_PATH, table_variant, suffix=f"__cw_{cw_strategy}")
    print(f"[RUN] Artifacts: {run_dir}")
    print(f"[RUN] Variant: {table_variant} | device: {device} | seed: {seed}")

    df = load_and_prepare_dataframe(TRAIN_TABLE_PATH)

    X_train, X_val, y_train_df, y_val_df, _, feature_names = build_feature_matrices(
        df, table_variant, apply_scaler=True
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

    print(f"[FEATURES] {len(feature_names)} features | units_choices: {units_choices}")

    summary_rows: List[Dict] = []
    cv_rows:      List[Dict] = []

    # Pre-compute full feature matrix (unscaled) for CV once
    X_full_raw = build_full_feature_matrix(df, table_variant, apply_scaler=False)[0]

    print(f"[INFO] Optimising {len(CLASSIFIER_STAT_PAIRS)} stat pairs …")
    for i, (stat, home_col, away_col) in enumerate(
        tqdm(CLASSIFIER_STAT_PAIRS, desc="Stats", unit="stat")
    ):
        print(f"\n[STAT {i+1}/{len(CLASSIFIER_STAT_PAIRS)}] {stat}  ({home_col} vs {away_col})")

        result = run_stat_study(
            stat, home_col, away_col,
            X_train.astype(np.float32), y_train_df,
            X_val.astype(np.float32),   y_val_df,
            run_dir, device, units_choices, seed, cw_strategy,
        )
        result.pop("val_preds_proba")
        summary_rows.append(result)

        print(f"[CV] Running {CV_FOLDS}-fold CV for {stat} …")
        y_labels_full = make_stat_labels_df(df, home_col, away_col)
        cv_result     = cross_validate_stat(
            stat, home_col, away_col,
            result["best_params_dict"], X_full_raw, y_labels_full,
            device, units_choices, cw_strategy=cw_strategy,
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
        "model_type":                "classifier_mlp",
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
    parser = argparse.ArgumentParser(description="MLP 3-class direction classifier with Optuna")
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
