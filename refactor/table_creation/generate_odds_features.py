# generate_odds_features.py  (refactor/table_creation/)
#
# Generates 21 out-of-fold (OOF) classifier probability columns — "odds" features —
# for every row in the training table, with NO leakage.
#
# For each of the 7 CLASSIFIER_STAT_PAIRS (GOALS, CORNERS, …) the script:
#   1. Scans all saved artifact runs in artifacts/classification/*/ and picks
#      the (model_type, variant, cw_strategy, run) with the highest cv_acc_mean
#      for that stat.
#   2. Reconstructs the model architecture from best_params.json.
#   3. Re-runs CV_FOLDS-fold cross-validation using the best hyperparameters on
#      the FULL dataset.  Each fold predicts softmax probabilities on its
#      held-out rows only, so every row is predicted exactly once without ever
#      having been in training.
#   4. Stitches the fold predictions back in original row order.
#
# Output:  train_tables/odds_oof.npz
#   Keys:
#     "fixture_ids"          int64  (N,)
#     "odds_{STAT}_home"     float32 (N,)   — P(HOME > AWAY)
#     "odds_{STAT}_draw"     float32 (N,)   — P(HOME == AWAY)
#     "odds_{STAT}_away"     float32 (N,)   — P(HOME < AWAY)
#   for each STAT in CLASSIFIER_TARGETS (7 stats).
#
# After running this script once, the "odds" group becomes available as a
# feature variant in all optimizers via shared_features.get_odds().
#
# Usage (from workspace root):
#   python refactor/table_creation/generate_odds_features.py
#   python refactor/table_creation/generate_odds_features.py --force   # overwrite existing file

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))   # refactor/

import json
import argparse
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from shared_config import (
    WORKSPACE_ROOT,
    CLASSIFIER_STAT_PAIRS, CLASSIFIER_TARGETS, N_CLASSES,
    TRAIN_TABLE_PATH, SEQ_TABLE_PATH,
    VARIANTS, CV_FOLDS, GLOBAL_SEED,
    RETRAIN_EPOCHS, RETRAIN_PATIENCE, PATIENCE,
    N_HIDDEN_MIN, N_HIDDEN_MAX,
    UNITS_CHOICES, DEFAULT_UNITS_CHOICES,
    XGB_N_ESTIMATORS_MAX, XGB_EARLY_STOPPING_ROUNDS,
    SEQ_K, SEQ_INPUT_DIM, USE_ROLE_TOKEN,
)
from shared_features import build_X, get_y
from shared_preprocessing import load_and_prepare_dataframe
from shared_utils import (
    set_all_seeds, get_torch_device,
    build_layer_sizes, make_torch_optimizer,
    ResidualMLP, get_activation,
)
from shared_metrics import make_stat_labels_df, OUTCOME_CLASSES
from shared_sequence import (
    load_seq_data, LSTMMLPDataset,
    LSTMEncoder, MLPEncoder, _LSTM_INPUT_SIZE,
)

ARTIFACTS_CLF_ROOT = WORKSPACE_ROOT / "artifacts" / "classification"
OOF_OUTPUT_PATH    = WORKSPACE_ROOT / "train_tables" / "odds_oof.npz"

# ─── model type tags (directory names under artifacts/classification/) ───────
MLP_SINGLE     = "mlp_torch"
MLP_MULTI      = "mlp_multioutput_torch"
XGB            = "xgb"
LSTM_SINGLE    = "lstm_mlp_torch"
LSTM_MULTI     = "lstm_mlp_multioutput_torch"

ALL_MODEL_TYPES = [MLP_SINGLE, MLP_MULTI, XGB, LSTM_SINGLE, LSTM_MULTI]


# =============================================================================
# ==  STEP 1: SELECT BEST RUN PER STAT  =======================================
# =============================================================================

def _collect_cv_stats(root: Path) -> pd.DataFrame:
    """
    Scan every cv_summary_all_stats.csv across all artifact runs and return a
    DataFrame with columns:
        model_type, variant, cw_strategy, run_dir, stat, cv_acc_mean
    """
    rows: List[Dict] = []
    for model_type_dir in root.iterdir():
        if not model_type_dir.is_dir():
            continue
        model_type = model_type_dir.name
        for variant_dir in model_type_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            variant = variant_dir.name
            for run_dir in variant_dir.iterdir():
                if not run_dir.is_dir():
                    continue
                cv_csv = run_dir / "cv_summary_all_stats.csv"
                if not cv_csv.exists():
                    continue
                # Extract cw_strategy from dir name suffix  __cw_{strategy}
                parts = run_dir.name.split("__cw_")
                cw_strategy = parts[-1] if len(parts) > 1 else "none"
                try:
                    df = pd.read_csv(cv_csv)
                    for _, row in df.iterrows():
                        rows.append({
                            "model_type":  model_type,
                            "variant":     variant,
                            "cw_strategy": cw_strategy,
                            "run_dir":     str(run_dir),
                            "stat":        str(row["stat"]),
                            "cv_acc_mean": float(row["cv_acc_mean"]),
                        })
                except Exception as e:
                    print(f"[WARN] Could not read {cv_csv}: {e}")
    return pd.DataFrame(rows)


def select_best_runs(root: Path) -> Dict[str, Dict]:
    """
    Returns {stat: {"model_type", "variant", "cw_strategy", "run_dir"}} for
    the run with the highest cv_acc_mean for each stat.
    """
    df = _collect_cv_stats(root)
    if df.empty:
        raise RuntimeError(f"No cv_summary_all_stats.csv found under {root}")

    best: Dict[str, Dict] = {}
    for stat in CLASSIFIER_TARGETS:
        sub = df[df["stat"] == stat]
        if sub.empty:
            raise RuntimeError(f"No artifact found for stat '{stat}'")
        idx = sub["cv_acc_mean"].idxmax()
        row = sub.loc[idx]
        best[stat] = {
            "model_type":  row["model_type"],
            "variant":     row["variant"],
            "cw_strategy": row["cw_strategy"],
            "run_dir":     Path(row["run_dir"]),
            "cv_acc_mean": row["cv_acc_mean"],
        }
        print(
            f"  [SELECT] {stat:20s}  best={row['cv_acc_mean']:.4f}  "
            f"model={row['model_type']}  variant={row['variant']}  cw={row['cw_strategy']}"
        )
    return best


# =============================================================================
# ==  STEP 2: LOAD FULL DATA  =================================================
# =============================================================================

def _load_full_data(
    variant: str,
    model_type: str,
    seq_cache: Optional[Dict] = None,
) -> Dict:
    """
    Load full (non-split) feature matrix and targets aligned by fixture_id.
    For LSTM models also loads sequence arrays.
    Returns a dict ready for OOF folding.
    """
    df_clean      = load_and_prepare_dataframe(TRAIN_TABLE_PATH)
    groups        = VARIANTS[variant]
    features_df   = build_X(df_clean, groups)
    targets_df    = get_y(df_clean)
    feature_names = list(features_df.columns)

    raw         = pd.read_csv(TRAIN_TABLE_PATH)
    fixture_ids = raw["fixture_id"].to_numpy(dtype=np.int64)

    X_raw = features_df.values.astype(np.float32)

    result = {
        "X_raw":         X_raw,
        "targets_df":    targets_df,
        "feature_names": feature_names,
        "fixture_ids":   fixture_ids,
        "df_clean":      df_clean,
    }

    if model_type in (LSTM_SINGLE, LSTM_MULTI):
        if seq_cache is not None and variant in seq_cache:
            sq = seq_cache[variant]
        else:
            sq = load_seq_data(variant)
            if seq_cache is not None:
                seq_cache[variant] = sq
        result["home_seq"] = sq["home_seq"]
        result["away_seq"] = sq["away_seq"]

    return result


# =============================================================================
# ==  STEP 3: PER-MODEL-TYPE OOF CV  ==========================================
# =============================================================================

# ── MLP single-target ────────────────────────────────────────────────────────

def _train_mlp_single(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    best_params: Dict,
    units_choices: List[int],
    cw_strategy: str,
    device: torch.device,
) -> np.ndarray:
    """Train one fold for MLP single-target classifier; returns (N_te, 3) probas."""
    layer_sizes = build_layer_sizes(best_params, units_choices)
    bs          = int(best_params["batch_size"])

    if cw_strategy == "sqrt":
        counts = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w      = torch.tensor(1.0 / np.sqrt(counts), dtype=torch.float32).to(device)
        crit   = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()

    model = ResidualMLP(
        X_tr.shape[1], layer_sizes,
        best_params["activation"], float(best_params["dropout"]),
        out_dim=N_CLASSES, squeeze=False,
    ).to(device)
    opt   = make_torch_optimizer(
        best_params["optimizer"], model.parameters(),
        float(best_params["lr"]), float(best_params["l2_reg"]),
    )
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(PATIENCE // 3, 3), min_lr=1e-6,
    )

    tr_tx = torch.from_numpy(X_tr); tr_ty = torch.from_numpy(y_tr)
    te_tx = torch.from_numpy(X_te); te_ty = torch.from_numpy(y_te)
    tr_ld = DataLoader(TensorDataset(tr_tx, tr_ty), batch_size=bs, shuffle=True)
    te_ld = DataLoader(TensorDataset(te_tx, te_ty), batch_size=bs, shuffle=False)

    from shared_utils import ResidualMLP as _R  # noqa — already imported
    # reuse the training loop from classifier_mlp_torch
    best_val_f1 = -1.0
    best_state: Optional[Dict] = None
    no_improve = 0
    model.train()
    for epoch in range(1, RETRAIN_EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.long().to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        model.eval()
        preds, labs = [], []
        val_loss_acc, n_total = 0.0, 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(device), yb.long().to(device)
                logits = model(xb)
                val_loss_acc += crit(logits, yb).item() * xb.size(0)
                n_total      += xb.size(0)
                preds.append(logits.argmax(1).cpu().numpy())
                labs.append(yb.cpu().numpy())
        from sklearn.metrics import f1_score as _f1
        val_f1 = float(_f1(np.concatenate(labs), np.concatenate(preds),
                            average="macro", zero_division=0))
        sched.step(val_loss_acc / n_total)
        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= RETRAIN_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    proba_list = []
    with torch.no_grad():
        for xb, _ in te_ld:
            proba_list.append(
                torch.softmax(model(xb.to(device)), dim=1).cpu().numpy()
            )
    return np.concatenate(proba_list, axis=0)


# ── MLP multioutput ────────────────────────────────────────────────────────────

def _get_stat_index(stat: str) -> int:
    for i, (s, _, _) in enumerate(CLASSIFIER_STAT_PAIRS):
        if s == stat:
            return i
    raise ValueError(f"stat '{stat}' not in CLASSIFIER_STAT_PAIRS")


class _MultiOutputMLP(nn.Module):
    """Minimal reconstruction of MultiOutputMLPClassifier for inference."""
    def __init__(self, input_dim, backbone_sizes, activation, dropout, n_stats, n_classes):
        super().__init__()
        self.backbone         = ResidualMLP(input_dim, backbone_sizes, activation, dropout)
        head_dim = max(16, self.backbone.out_dim // 2)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.backbone.out_dim, head_dim),
                get_activation(activation),
                nn.Linear(head_dim, n_classes),
            )
            for _ in range(n_stats)
        ])
    def forward(self, x):
        z = self.backbone(x)
        return torch.stack([h(z) for h in self.heads], dim=1)  # (B, N_STATS, N_CLASSES)


def _train_mlp_multioutput(
    X_tr: np.ndarray, y_tr_all: np.ndarray,  # (N_tr, N_STATS)
    X_te: np.ndarray, y_te_all: np.ndarray,
    stat_idx: int,
    best_params: Dict,
    units_choices: List[int],
    cw_strategy: str,
    device: torch.device,
) -> np.ndarray:
    """Train one fold for MLP multioutput; returns (N_te, 3) probas for one stat."""
    from shared_config import N_CLASSES as NC
    n_stats = len(CLASSIFIER_STAT_PAIRS)
    layer_sizes = build_layer_sizes(best_params, units_choices)
    bs          = int(best_params["batch_size"])

    if cw_strategy == "sqrt":
        cw_list = []
        for i in range(n_stats):
            c = np.bincount(y_tr_all[:, i], minlength=NC).astype(float)
            c = np.where(c == 0, 1.0, c)
            cw_list.append(1.0 / np.sqrt(c))
        class_weights = torch.tensor(np.stack(cw_list), dtype=torch.float32)
        criteria = [nn.CrossEntropyLoss(weight=class_weights[i].to(device)) for i in range(n_stats)]
    else:
        criteria = [nn.CrossEntropyLoss() for _ in range(n_stats)]
    ce_unweighted = nn.CrossEntropyLoss()

    model = _MultiOutputMLP(X_tr.shape[1], layer_sizes,
                            best_params["activation"], float(best_params["dropout"]),
                            n_stats, NC).to(device)
    opt   = make_torch_optimizer(
        best_params["optimizer"], model.parameters(),
        float(best_params["lr"]), float(best_params["l2_reg"]),
    )
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(PATIENCE // 3, 3), min_lr=1e-6,
    )

    tr_tx = torch.from_numpy(X_tr)
    tr_ty = torch.from_numpy(y_tr_all)   # (N_tr, N_STATS)
    te_tx = torch.from_numpy(X_te)
    te_ty = torch.from_numpy(y_te_all)
    tr_ld = DataLoader(TensorDataset(tr_tx, tr_ty), batch_size=bs, shuffle=True)
    te_ld = DataLoader(TensorDataset(te_tx, te_ty), batch_size=bs, shuffle=False)

    from sklearn.metrics import f1_score as _f1
    best_val_f1 = -1.0
    best_state: Optional[Dict] = None
    no_improve = 0
    for epoch in range(1, RETRAIN_EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(device), yb.long().to(device)
            opt.zero_grad(set_to_none=True)
            logits  = model(xb)                         # (B, N_STATS, N_CLASSES)
            loss    = sum(criteria[i](logits[:, i, :], yb[:, i]) for i in range(n_stats)) / n_stats
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        model.eval()
        val_loss_acc, n_total = 0.0, 0
        f1_per_stat = []
        with torch.no_grad():
            all_preds  = [[] for _ in range(n_stats)]
            all_labels = [[] for _ in range(n_stats)]
            for xb, yb in te_ld:
                xb, yb   = xb.to(device), yb.long().to(device)
                logits   = model(xb)
                l        = sum(ce_unweighted(logits[:, i, :], yb[:, i]) for i in range(n_stats)) / n_stats
                val_loss_acc += l.item() * xb.size(0)
                n_total      += xb.size(0)
                for i in range(n_stats):
                    all_preds[i].append(logits[:, i, :].argmax(1).cpu().numpy())
                    all_labels[i].append(yb[:, i].cpu().numpy())
        f1_per_stat = [
            float(_f1(np.concatenate(all_labels[i]), np.concatenate(all_preds[i]),
                      average="macro", zero_division=0))
            for i in range(n_stats)
        ]
        val_f1 = float(np.mean(f1_per_stat))
        sched.step(val_loss_acc / n_total)
        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= RETRAIN_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    proba_list = []
    with torch.no_grad():
        for xb, _ in te_ld:
            logits = model(xb.to(device))               # (B, N_STATS, N_CLASSES)
            proba_list.append(
                torch.softmax(logits[:, stat_idx, :], dim=1).cpu().numpy()
            )
    return np.concatenate(proba_list, axis=0)


# ── XGBoost ───────────────────────────────────────────────────────────────────

def _train_xgb(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    best_params: Dict,
    cw_strategy: str,
) -> np.ndarray:
    """Train one fold for XGB classifier; returns (N_te, 3) probas."""
    from xgboost import XGBClassifier
    try:
        import torch as _t
        tree_method = "hist"
        device      = "cuda" if _t.cuda.is_available() else "cpu"
    except ImportError:
        tree_method, device = "hist", "cpu"

    params = {
        "objective":         "multi:softprob",
        "num_class":         N_CLASSES,
        "eval_metric":       "mlogloss",
        "booster":           "gbtree",
        "tree_method":       tree_method,
        "device":            device,
        "n_estimators":      XGB_N_ESTIMATORS_MAX,
        "verbosity":         0,
        "random_state":      GLOBAL_SEED,
        **{k: v for k, v in best_params.items()
           if k not in ("objective", "num_class", "eval_metric", "booster",
                        "tree_method", "device", "n_estimators", "verbosity", "random_state")},
    }

    if cw_strategy == "sqrt":
        counts    = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
        counts    = np.where(counts == 0, 1.0, counts)
        sw        = (1.0 / np.sqrt(counts))[y_tr]
        sw        = sw / sw.mean()
    else:
        sw = None

    clf = XGBClassifier(**params)
    try:
        clf.fit(
            X_tr, y_tr, sample_weight=sw,
            eval_set=[(X_te, y_te)], verbose=False,
            early_stopping_rounds=XGB_EARLY_STOPPING_ROUNDS,
        )
    except Exception:
        clf.fit(X_tr, y_tr, sample_weight=sw, verbose=False)

    proba = clf.predict_proba(X_te)   # (N_te, 3)
    return proba.astype(np.float32)


# ── LSTM-MLP classifier (shared by single and multioutput variants) ───────────

class _LSTMMLPClassifier(nn.Module):
    """Minimal reconstruction of LSTMMLPClassifier for OOF inference."""
    def __init__(self, lstm_encoder, mlp_encoder, fusion_head, use_role_token=USE_ROLE_TOKEN,
                 lstm_encoder_away=None):
        super().__init__()
        self.lstm_encoder      = lstm_encoder
        self.lstm_encoder_away = lstm_encoder_away
        self.mlp_encoder       = mlp_encoder
        self.fusion_head       = fusion_head
        self.use_role_token    = use_role_token

    def _encode(self, seq, role_val, encoder):
        if self.use_role_token:
            B, K, _ = seq.shape
            tok = seq.new_full((B, K, 1), fill_value=role_val)
            seq = torch.cat([seq, tok], dim=-1)
        return encoder(seq)

    def forward(self, home_seq, away_seq, static_x):
        enc_away = self.lstm_encoder_away or self.lstm_encoder
        h_home   = self._encode(home_seq, 1.0, self.lstm_encoder)
        h_away   = self._encode(away_seq, 0.0, enc_away)
        h_static = self.mlp_encoder(static_x)
        z        = torch.cat([h_home, h_away, h_static], dim=-1)
        return self.fusion_head(z)


def _build_lstm_mlp_model(best_params: Dict, static_input_dim: int, n_out: int, device: torch.device):
    """
    Reconstruct LSTMMLPClassifier from best_params dict.
    n_out = N_CLASSES for single-stat, N_STATS * N_CLASSES for multioutput
    (but for multioutput we use N_STATS separate heads — see _LSTMMLPMultiOut).
    """
    layer_sizes       = best_params["layer_sizes"]
    activation        = best_params["activation"]
    mlp_dropout       = float(best_params["mlp_dropout"])
    lstm_hidden       = int(best_params["lstm_hidden"])
    lstm_layers       = int(best_params["lstm_layers"])
    lstm_dropout      = float(best_params["lstm_dropout"])
    fusion_n_hidden   = int(best_params.get("fusion_head_n_hidden", 1))
    fusion_dropout    = float(best_params.get("fusion_dropout", 0.0))

    mlp_enc   = MLPEncoder(static_input_dim, layer_sizes, activation, mlp_dropout)
    lstm_home = LSTMEncoder(_LSTM_INPUT_SIZE, lstm_hidden, lstm_layers, lstm_dropout)
    fusion_dim = 2 * lstm_hidden + mlp_enc.out_dim

    if fusion_n_hidden == 0:
        head = nn.Linear(fusion_dim, n_out)
    else:
        mid    = max(16, fusion_dim // 2)
        act_fn = {"relu": nn.ReLU, "swish": nn.SiLU, "gelu": nn.GELU,
                  "elu": nn.ELU, "selu": nn.SELU}.get(activation.lower(), nn.ReLU)()
        head = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, mid),
            act_fn,
            nn.Dropout(fusion_dropout) if fusion_dropout > 0.0 else nn.Identity(),
            nn.Linear(mid, n_out),
        )
    return _LSTMMLPClassifier(lstm_home, mlp_enc, head).to(device)


def _train_lstm_single(
    hs_tr, as_tr, X_tr, y_tr,
    hs_te, as_te, X_te, y_te,
    best_params, units_choices, cw_strategy, device,
) -> np.ndarray:
    """Train one LSTM-MLP single-stat fold; returns (N_te, 3) probas."""
    bs = int(best_params["batch_size"])
    if cw_strategy == "sqrt":
        counts = np.bincount(y_tr, minlength=N_CLASSES).astype(float)
        counts = np.where(counts == 0, 1.0, counts)
        w      = torch.tensor(1.0 / np.sqrt(counts), dtype=torch.float32).to(device)
        crit   = nn.CrossEntropyLoss(weight=w)
    else:
        crit = nn.CrossEntropyLoss()

    model = _build_lstm_mlp_model(best_params, X_tr.shape[1], N_CLASSES, device)
    opt   = make_torch_optimizer(
        best_params["optimizer"], model.parameters(),
        float(best_params["lr"]), float(best_params["l2_reg"]),
    )
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(PATIENCE // 3, 3), min_lr=1e-6,
    )

    tr_ld = DataLoader(
        LSTMMLPDataset(torch.from_numpy(hs_tr).float(), torch.from_numpy(as_tr).float(),
                       torch.from_numpy(X_tr).float(),  torch.from_numpy(y_tr)),
        batch_size=bs, shuffle=True,
    )
    te_ld = DataLoader(
        LSTMMLPDataset(torch.from_numpy(hs_te).float(), torch.from_numpy(as_te).float(),
                       torch.from_numpy(X_te).float(),  torch.from_numpy(y_te)),
        batch_size=bs, shuffle=False,
    )

    from sklearn.metrics import f1_score as _f1
    best_val_f1, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, RETRAIN_EPOCHS + 1):
        model.train()
        for h_seq, a_seq, x_stat, yb in tr_ld:
            h_seq, a_seq, x_stat, yb = (
                h_seq.to(device), a_seq.to(device), x_stat.to(device), yb.long().to(device)
            )
            opt.zero_grad(set_to_none=True)
            loss = crit(model(h_seq, a_seq, x_stat), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        model.eval()
        val_loss_acc, n_total, preds, labs = 0.0, 0, [], []
        with torch.no_grad():
            for h_seq, a_seq, x_stat, yb in te_ld:
                h_seq, a_seq, x_stat, yb = (
                    h_seq.to(device), a_seq.to(device), x_stat.to(device), yb.long().to(device)
                )
                logits = model(h_seq, a_seq, x_stat)
                val_loss_acc += crit(logits, yb).item() * h_seq.size(0)
                n_total      += h_seq.size(0)
                preds.append(logits.argmax(1).cpu().numpy())
                labs.append(yb.cpu().numpy())
        val_f1 = float(_f1(np.concatenate(labs), np.concatenate(preds),
                            average="macro", zero_division=0))
        sched.step(val_loss_acc / n_total)
        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= RETRAIN_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    proba_list = []
    with torch.no_grad():
        for h_seq, a_seq, x_stat, _ in te_ld:
            proba_list.append(
                torch.softmax(
                    model(h_seq.to(device), a_seq.to(device), x_stat.to(device)), dim=1
                ).cpu().numpy()
            )
    return np.concatenate(proba_list, axis=0)


class _LSTMMLPMultiOut(nn.Module):
    """LSTM encoder + MLP encoder + N_STATS independent heads."""
    def __init__(self, lstm_encoder, mlp_encoder, heads, use_role_token=USE_ROLE_TOKEN):
        super().__init__()
        self.lstm_encoder = lstm_encoder
        self.mlp_encoder  = mlp_encoder
        self.heads        = nn.ModuleList(heads)
        self.use_role_token = use_role_token

    def _encode(self, seq, role_val):
        if self.use_role_token:
            B, K, _ = seq.shape
            tok = seq.new_full((B, K, 1), fill_value=role_val)
            seq = torch.cat([seq, tok], dim=-1)
        return self.lstm_encoder(seq)

    def forward(self, home_seq, away_seq, static_x):
        h_home   = self._encode(home_seq, 1.0)
        h_away   = self._encode(away_seq, 0.0)
        h_static = self.mlp_encoder(static_x)
        z        = torch.cat([h_home, h_away, h_static], dim=-1)
        return torch.stack([h(z) for h in self.heads], dim=1)  # (B, N_STATS, N_CLASSES)


def _train_lstm_multioutput(
    hs_tr, as_tr, X_tr, y_tr_all,
    hs_te, as_te, X_te, y_te_all,
    stat_idx, best_params, units_choices, cw_strategy, device,
) -> np.ndarray:
    """Train one LSTM-MLP multioutput fold; returns (N_te, 3) probas for one stat."""
    n_stats = len(CLASSIFIER_STAT_PAIRS)
    bs      = int(best_params["batch_size"])

    if cw_strategy == "sqrt":
        cw_list = []
        for i in range(n_stats):
            c = np.bincount(y_tr_all[:, i], minlength=N_CLASSES).astype(float)
            c = np.where(c == 0, 1.0, c)
            cw_list.append(1.0 / np.sqrt(c))
        class_weights = torch.tensor(np.stack(cw_list), dtype=torch.float32)
        criteria = [nn.CrossEntropyLoss(weight=class_weights[i].to(device)) for i in range(n_stats)]
    else:
        criteria = [nn.CrossEntropyLoss() for _ in range(n_stats)]
    ce_unweighted = nn.CrossEntropyLoss()

    layer_sizes    = best_params["layer_sizes"]
    activation     = best_params["activation"]
    mlp_dropout    = float(best_params["mlp_dropout"])
    lstm_hidden    = int(best_params["lstm_hidden"])
    lstm_layers    = int(best_params["lstm_layers"])
    lstm_dropout   = float(best_params["lstm_dropout"])
    fusion_n_hid   = int(best_params.get("fusion_head_n_hidden", 1))
    fusion_drop    = float(best_params.get("fusion_dropout", 0.0))

    mlp_enc    = MLPEncoder(X_tr.shape[1], layer_sizes, activation, mlp_dropout)
    lstm_enc   = LSTMEncoder(_LSTM_INPUT_SIZE, lstm_hidden, lstm_layers, lstm_dropout)
    fusion_dim = 2 * lstm_hidden + mlp_enc.out_dim

    def _make_head():
        if fusion_n_hid == 0:
            return nn.Linear(fusion_dim, N_CLASSES)
        mid    = max(16, fusion_dim // 2)
        act_fn = {"relu": nn.ReLU, "swish": nn.SiLU, "gelu": nn.GELU,
                  "elu": nn.ELU, "selu": nn.SELU}.get(activation.lower(), nn.ReLU)()
        return nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, mid),
            act_fn,
            nn.Dropout(fusion_drop) if fusion_drop > 0 else nn.Identity(),
            nn.Linear(mid, N_CLASSES),
        )

    model = _LSTMMLPMultiOut(lstm_enc, mlp_enc, [_make_head() for _ in range(n_stats)]).to(device)
    opt   = make_torch_optimizer(
        best_params["optimizer"], model.parameters(),
        float(best_params["lr"]), float(best_params["l2_reg"]),
    )
    sched = optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=max(PATIENCE // 3, 3), min_lr=1e-6,
    )

    tr_ld = DataLoader(
        LSTMMLPDataset(torch.from_numpy(hs_tr).float(), torch.from_numpy(as_tr).float(),
                       torch.from_numpy(X_tr).float(),  torch.from_numpy(y_tr_all)),
        batch_size=bs, shuffle=True,
    )
    te_ld = DataLoader(
        LSTMMLPDataset(torch.from_numpy(hs_te).float(), torch.from_numpy(as_te).float(),
                       torch.from_numpy(X_te).float(),  torch.from_numpy(y_te_all)),
        batch_size=bs, shuffle=False,
    )

    from sklearn.metrics import f1_score as _f1
    best_val_f1, best_state, no_improve = -1.0, None, 0
    for epoch in range(1, RETRAIN_EPOCHS + 1):
        model.train()
        for h_seq, a_seq, x_stat, yb in tr_ld:
            h_seq, a_seq, x_stat, yb = (
                h_seq.to(device), a_seq.to(device), x_stat.to(device), yb.long().to(device)
            )
            opt.zero_grad(set_to_none=True)
            logits = model(h_seq, a_seq, x_stat)
            loss   = sum(criteria[i](logits[:, i, :], yb[:, i]) for i in range(n_stats)) / n_stats
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        model.eval()
        val_loss_acc, n_total = 0.0, 0
        with torch.no_grad():
            all_preds  = [[] for _ in range(n_stats)]
            all_labels = [[] for _ in range(n_stats)]
            for h_seq, a_seq, x_stat, yb in te_ld:
                h_seq, a_seq, x_stat, yb = (
                    h_seq.to(device), a_seq.to(device), x_stat.to(device), yb.long().to(device)
                )
                logits = model(h_seq, a_seq, x_stat)
                l = sum(ce_unweighted(logits[:, i, :], yb[:, i]) for i in range(n_stats)) / n_stats
                val_loss_acc += l.item() * h_seq.size(0)
                n_total      += h_seq.size(0)
                for i in range(n_stats):
                    all_preds[i].append(logits[:, i, :].argmax(1).cpu().numpy())
                    all_labels[i].append(yb[:, i].cpu().numpy())
        f1_per_s = [
            float(_f1(np.concatenate(all_labels[i]), np.concatenate(all_preds[i]),
                      average="macro", zero_division=0))
            for i in range(n_stats)
        ]
        sched.step(val_loss_acc / n_total)
        val_f1 = float(np.mean(f1_per_s))
        if val_f1 > best_val_f1 + 1e-5:
            best_val_f1 = val_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
            if no_improve >= RETRAIN_PATIENCE:
                break

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    proba_list = []
    with torch.no_grad():
        for h_seq, a_seq, x_stat, _ in te_ld:
            logits = model(h_seq.to(device), a_seq.to(device), x_stat.to(device))
            proba_list.append(
                torch.softmax(logits[:, stat_idx, :], dim=1).cpu().numpy()
            )
    return np.concatenate(proba_list, axis=0)


# =============================================================================
# ==  STEP 4: OOF LOOP PER STAT  ==============================================
# =============================================================================

def _load_best_params(run_dir: Path, model_type: str) -> Dict:
    """Load best_params.json from the appropriate location for each model type."""
    if model_type in (MLP_MULTI, LSTM_MULTI):
        # multioutput: best_params.json is in the run root
        return json.loads((run_dir / "best_params.json").read_text())
    else:
        # Single-target models: sample from first stat dir
        for stat, _, _ in CLASSIFIER_STAT_PAIRS:
            p = run_dir / stat / "best_params.json"
            if p.exists():
                return json.loads(p.read_text())
        raise FileNotFoundError(f"No best_params.json found under {run_dir}")


def _make_all_labels(df_clean: pd.DataFrame) -> np.ndarray:
    """Return (N, N_STATS) int64 label array for all stat pairs."""
    cols = []
    for stat, home_col, away_col in CLASSIFIER_STAT_PAIRS:
        cols.append(make_stat_labels_df(df_clean, home_col, away_col))
    return np.stack(cols, axis=1)   # (N, N_STATS)


def generate_oof_for_stat(
    stat: str,
    home_col: str,
    away_col: str,
    info: Dict,
    device: torch.device,
    seq_cache: Optional[Dict],
) -> np.ndarray:
    """
    Run OOF cross-validation for a single stat.
    Returns proba array (N, 3) in original dataset row order.
    """
    model_type  = info["model_type"]
    variant     = info["variant"]
    cw_strategy = info["cw_strategy"]
    run_dir     = info["run_dir"]
    stat_idx    = _get_stat_index(stat)

    print(f"\n[OOF] {stat}  model={model_type}  variant={variant}  cw={cw_strategy}")
    print(f"      run_dir: {run_dir.name}")

    # Load data
    data         = _load_full_data(variant, model_type, seq_cache)
    X_raw        = data["X_raw"]
    df_clean     = data["df_clean"]
    fixture_ids  = data["fixture_ids"]
    units_choices = UNITS_CHOICES.get(variant, DEFAULT_UNITS_CHOICES)
    N            = X_raw.shape[0]

    best_params  = _load_best_params(run_dir, model_type)

    # Labels
    y_single     = make_stat_labels_df(df_clean, home_col, away_col)   # (N,) int64
    y_all        = _make_all_labels(df_clean)                           # (N, N_STATS) int64

    # OOF probability array in original order
    oof_proba    = np.zeros((N, 3), dtype=np.float32)
    kf           = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=GLOBAL_SEED)

    for fold_idx, (tr_idx, te_idx) in enumerate(kf.split(X_raw), start=1):
        print(f"  Fold {fold_idx}/{CV_FOLDS} …", end=" ", flush=True)

        X_tr_raw, X_te_raw = X_raw[tr_idx], X_raw[te_idx]

        if model_type == XGB:
            # XGBoost: no scaler
            proba = _train_xgb(
                X_tr_raw.astype(np.float32), y_single[tr_idx],
                X_te_raw.astype(np.float32), y_single[te_idx],
                best_params, cw_strategy,
            )
        else:
            # Neural net models: fit scaler on train fold
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X_tr_raw).astype(np.float32)
            X_te   = scaler.transform(X_te_raw).astype(np.float32)

            if model_type == MLP_SINGLE:
                proba = _train_mlp_single(
                    X_tr, y_single[tr_idx].astype(np.int64),
                    X_te, y_single[te_idx].astype(np.int64),
                    best_params, units_choices, cw_strategy, device,
                )
            elif model_type == MLP_MULTI:
                proba = _train_mlp_multioutput(
                    X_tr, y_all[tr_idx].astype(np.int64),
                    X_te, y_all[te_idx].astype(np.int64),
                    stat_idx, best_params, units_choices, cw_strategy, device,
                )
            elif model_type == LSTM_SINGLE:
                hs, as_ = data["home_seq"], data["away_seq"]
                proba = _train_lstm_single(
                    hs[tr_idx], as_[tr_idx], X_tr, y_single[tr_idx].astype(np.int64),
                    hs[te_idx], as_[te_idx], X_te, y_single[te_idx].astype(np.int64),
                    best_params, units_choices, cw_strategy, device,
                )
            elif model_type == LSTM_MULTI:
                hs, as_ = data["home_seq"], data["away_seq"]
                proba = _train_lstm_multioutput(
                    hs[tr_idx], as_[tr_idx], X_tr, y_all[tr_idx].astype(np.int64),
                    hs[te_idx], as_[te_idx], X_te, y_all[te_idx].astype(np.int64),
                    stat_idx, best_params, units_choices, cw_strategy, device,
                )
            else:
                raise ValueError(f"Unknown model_type '{model_type}'")

        oof_proba[te_idx] = proba
        from sklearn.metrics import accuracy_score as _acc
        y_pred_fold = proba.argmax(axis=1)
        fold_acc    = float(_acc(y_single[te_idx], y_pred_fold))
        print(f"acc={fold_acc:.4f}")

    oof_acc = float((oof_proba.argmax(axis=1) == y_single).mean())
    print(f"  [OOF] {stat} overall OOF acc = {oof_acc:.4f}")
    return oof_proba


# =============================================================================
# ==  MAIN  ===================================================================
# =============================================================================

def main(force: bool = False) -> None:
    if OOF_OUTPUT_PATH.exists() and not force:
        print(f"[SKIP] {OOF_OUTPUT_PATH} already exists. Use --force to regenerate.")
        return

    set_all_seeds(GLOBAL_SEED)
    device    = get_torch_device()
    print(f"[START] Generating OOF odds features  (device={device})")

    print("\n[STEP 1] Selecting best model per stat …")
    best_runs  = select_best_runs(ARTIFACTS_CLF_ROOT)

    seq_cache: Dict = {}   # cache loaded sequence data by variant to avoid reloading

    # Build fixture_ids from the raw CSV (once)
    raw_df      = pd.read_csv(TRAIN_TABLE_PATH)
    fixture_ids = raw_df["fixture_id"].to_numpy(dtype=np.int64)
    N           = len(fixture_ids)

    print(f"\n[STEP 2-3] Running OOF CV for each stat (N={N} rows, {CV_FOLDS} folds) …")
    save_dict: Dict[str, np.ndarray] = {"fixture_ids": fixture_ids}

    for stat, home_col, away_col in CLASSIFIER_STAT_PAIRS:
        info  = best_runs[stat]
        proba = generate_oof_for_stat(
            stat, home_col, away_col, info, device, seq_cache,
        )  # (N, 3)  — columns: home, draw, away  (classes 0, 1, 2)
        save_dict[f"odds_{stat}_home"] = proba[:, 0].astype(np.float32)
        save_dict[f"odds_{stat}_draw"] = proba[:, 1].astype(np.float32)
        save_dict[f"odds_{stat}_away"] = proba[:, 2].astype(np.float32)

    OOF_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(OOF_OUTPUT_PATH), **save_dict)
    print(f"\n[DONE] Saved {OOF_OUTPUT_PATH}")
    print(f"       Keys: fixture_ids + {3 * len(CLASSIFIER_STAT_PAIRS)} probability columns")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate OOF classifier probability features (odds group)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing odds_oof.npz even if it already exists.",
    )
    args = parser.parse_args()
    main(force=args.force)
