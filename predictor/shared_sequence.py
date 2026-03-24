# shared_sequence.py
# Shared LSTM-MLP architecture, dataset, loss helpers, and training utilities.
#
# Imported by:
#   optimizer_lstm_mlp_torch.py              (single-target)
#   optimizer_lstm_mlp_multioutput_torch.py  (multi-target)
#
# Architecture overview:
#   home_seq (B,K,F) → role token → LSTMEncoder ─────────────────────┐
#   away_seq (B,K,F) → role token → LSTMEncoder (shared weights) ─────┤
#   static_x (B,D)              → MLPEncoder ────────────────────────┘
#        → concat(h_home, h_away, h_static)  (B, 2H+E)
#        → FusionHead
#        → (B,1) single-target  OR  (B,T) multi-target
#
# Design choices:
#   - Shared LSTM encoder for home/away teams: fewer params, less overfit at
#     N=1806.  To switch to separate encoders: pass use_shared_lstm=False to
#     the factory function — no architectural change required.
#   - Role token (1.0=home / 0.0=away) appended per step inside the forward
#     pass.  Controlled by USE_ROLE_TOKEN in shared_config.py.
#   - Sequences are NOT scaled (raw outcome counts are on a natural scale for
#     the LSTM).  Static features are StandardScaler-normalised by the caller.
#   - Pre-norm residual blocks in MLPEncoder (mirrors MLPRegressor design).

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import optuna

from shared_utils import get_activation, ResidualMLP
from shared_config import (
    TARGETS, TARGET_LOSS_MAP,
    TRAIN_TABLE_PATH, SEQ_TABLE_PATH,
    SEQ_INPUT_DIM, SEQ_K, USE_ROLE_TOKEN,
    VARIANTS,
    TEST_SIZE, RANDOM_STATE, SHUFFLE,
    LSTM_HIDDEN_CHOICES, LSTM_LAYERS_OPTIONS,
    LSTM_DROPOUT_MIN, LSTM_DROPOUT_MAX,
)
from shared_features import build_X, get_y
from shared_preprocessing import load_and_prepare_dataframe

# Effective LSTM input size: raw stats + optional 1-hot role token
_LSTM_INPUT_SIZE: int = SEQ_INPUT_DIM + int(USE_ROLE_TOKEN)


# =============================================================================
# ==  DATA LOADING  ===========================================================
# =============================================================================

def load_seq_data(table_variant: str) -> dict:
    """
    Load and align all data for the LSTM-MLP optimizers.

    Returns a dict with:
        features_df   (N, D)      pd.DataFrame  — unscaled static features
        targets_df    (N, T)      pd.DataFrame  — regression targets
        feature_names list[str]
        home_seq      (N, K, F)   np.float32    — home team sequences
        away_seq      (N, K, F)   np.float32    — away team sequences

    Static features are returned unscaled so callers can fit a fresh
    StandardScaler on each train fold (prevents data leakage in CV).

    Sequences are returned as raw outcome counts (no normalisation).
    The LSTM handles natural scale; zero-padded steps stay at zero.
    """
    if not SEQ_TABLE_PATH.exists():
        raise FileNotFoundError(
            f"Sequence table not found: {SEQ_TABLE_PATH}\n"
            f"Generate it first:\n"
            f"  python refactor/table_creation/build_sequence_table.py"
        )

    # --- Static features ---
    df_clean      = load_and_prepare_dataframe(TRAIN_TABLE_PATH)
    groups        = VARIANTS[table_variant]
    features_df   = build_X(df_clean, groups)
    targets_df    = get_y(df_clean)
    feature_names = list(features_df.columns)

    # --- Sequences (align to original CSV row order by fixture_id) ---
    # load_and_prepare_dataframe drops meta cols, so we re-read for fixture_id.
    # Row order is preserved (load_and_prepare_dataframe never reorders rows).
    raw              = pd.read_csv(TRAIN_TABLE_PATH)
    fixture_ids_df   = raw["fixture_id"].to_numpy(dtype=np.int64)

    seq              = np.load(SEQ_TABLE_PATH)
    fixture_ids_seq  = seq["fixture_ids"]      # saved in original CSV row order

    if np.array_equal(fixture_ids_df, fixture_ids_seq):
        home_seq = seq["home_seq"].astype(np.float32)
        away_seq = seq["away_seq"].astype(np.float32)
    else:
        # Sequences are not in the same order — reindex by fixture_id.
        id_to_idx = {int(fid): i for i, fid in enumerate(fixture_ids_seq)}
        order     = np.array([id_to_idx[int(fid)] for fid in fixture_ids_df])
        home_seq  = seq["home_seq"][order].astype(np.float32)
        away_seq  = seq["away_seq"][order].astype(np.float32)

    return {
        "features_df":   features_df,
        "targets_df":    targets_df,
        "feature_names": feature_names,
        "home_seq":      home_seq,   # (N, K, F)
        "away_seq":      away_seq,   # (N, K, F)
    }


def split_seq_data(data: dict) -> dict:
    """
    Apply the standard 80/20 train/val split to the loaded data.

    Uses the same TEST_SIZE, SHUFFLE, RANDOM_STATE as build_feature_matrices()
    so that a pure-MLP run and an LSTM-MLP run on the same variant see
    identical train/val partitions.

    Returns a dict with:
        X_train_raw, X_val_raw   np.ndarray  (unscaled static features)
        y_train, y_val           np.ndarray  (original target values)
        home_seq_train/val       np.ndarray  (K-step outcome sequences)
        away_seq_train/val       np.ndarray
        scaler                   StandardScaler (fit on X_train_raw)
        X_train, X_val           np.ndarray  (scaled static features)
        feature_names            list[str]
    """
    X_full        = data["features_df"].values.astype(np.float64)
    y_full        = data["targets_df"][TARGETS].values.astype(np.float64)
    home_seq_full = data["home_seq"]   # (N, K, F)  float32
    away_seq_full = data["away_seq"]   # (N, K, F)  float32
    N             = X_full.shape[0]

    indices           = np.arange(N)
    train_idx, val_idx = train_test_split(
        indices, test_size=TEST_SIZE, shuffle=SHUFFLE, random_state=RANDOM_STATE,
    )
    print(
        f"[SPLIT] Train: {len(train_idx)} rows | Val: {len(val_idx)} rows "
        f"| Static features: {X_full.shape[1]} | Seq K={SEQ_K} F={SEQ_INPUT_DIM}"
    )

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_full[train_idx])
    X_val   = scaler.transform(X_full[val_idx])

    return {
        "train_idx":        train_idx,
        "val_idx":          val_idx,
        "X_train":          X_train.astype(np.float32),
        "X_val":            X_val.astype(np.float32),
        "y_train":          y_full[train_idx].astype(np.float32),
        "y_val":            y_full[val_idx].astype(np.float32),
        "home_seq_train":   home_seq_full[train_idx],
        "home_seq_val":     home_seq_full[val_idx],
        "away_seq_train":   away_seq_full[train_idx],
        "away_seq_val":     away_seq_full[val_idx],
        "scaler":           scaler,
        "feature_names":    data["feature_names"],
        "home_seq_full":    home_seq_full,
        "away_seq_full":    away_seq_full,
        "X_full_raw":       X_full.astype(np.float32),
        "y_full":           y_full.astype(np.float32),
    }


# =============================================================================
# ==  DATASET  ================================================================
# =============================================================================

class LSTMMLPDataset(Dataset):
    """
    Dataset that bundles the three model inputs with the regression targets.

    Args:
        home_seq  (N, K, F) — home team's K-step outcome sequence
        away_seq  (N, K, F) — away team's K-step outcome sequence
        static_x  (N, D)    — pre-scaled static feature vector
        y         (N,) or (N, T) — regression targets
    """

    def __init__(
        self,
        home_seq:  torch.Tensor,
        away_seq:  torch.Tensor,
        static_x:  torch.Tensor,
        y:         torch.Tensor,
    ) -> None:
        assert home_seq.shape[0] == away_seq.shape[0] == static_x.shape[0] == y.shape[0]
        self.home_seq = home_seq
        self.away_seq = away_seq
        self.static_x = static_x
        self.y        = y

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.home_seq[idx], self.away_seq[idx], self.static_x[idx], self.y[idx]


def make_seq_dataloader(
    home_seq:   np.ndarray,
    away_seq:   np.ndarray,
    static_x:   np.ndarray,
    y:          np.ndarray,
    batch_size: int,
    shuffle:    bool,
) -> DataLoader:
    """Build a DataLoader for the LSTM-MLP fusion model."""
    ds = LSTMMLPDataset(
        torch.from_numpy(home_seq).float(),
        torch.from_numpy(away_seq).float(),
        torch.from_numpy(static_x).float(),
        torch.from_numpy(y).float(),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


# =============================================================================
# ==  LSTM ENCODER  ===========================================================
# =============================================================================

class LSTMEncoder(nn.Module):
    """
    Encodes a team's K-step sequence into a fixed-size vector.

    Input : (B, K, input_size) — batch-first sequences
    Output: (B, hidden_size)   — last-layer hidden state at final time step

    The same LSTMEncoder instance is shared for home and away in the fusion
    model (weight-tied).  To use separate encoders instead, pass
    use_shared_lstm=False to the factory functions — no other change needed.
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int,
        num_layers:  int   = 1,
        dropout:     float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # PyTorch warns if dropout > 0 on a single-layer LSTM; guard here.
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, hidden_size) — last time-step, last-layer hidden state."""
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


# =============================================================================
# ==  MLP ENCODER (static branch)  ============================================
# =============================================================================

class MLPEncoder(ResidualMLP):
    """
    Pre-norm residual MLP trunk for the static feature branch of LSTM-MLP models.
    No output head — returns the final hidden vector (B, out_dim) for fusion.

    Thin subclass of ResidualMLP(out_dim=None) — all backbone logic lives in
    shared_utils.ResidualMLP, which keeps shared_sequence.py DRY.
    The state_dict keys (norms.*, linears.*, …) are identical to the previous
    standalone MLPEncoder, so existing .pt checkpoints load without changes.
    """

    def __init__(
        self,
        input_dim:   int,
        layer_sizes: List[int],
        activation:  str,
        dropout:     float,
    ) -> None:
        super().__init__(input_dim, layer_sizes, activation, dropout, out_dim=None)


# =============================================================================
# ==  SINGLE-TARGET FUSION MODEL  =============================================
# =============================================================================

class LSTMMLPModel(nn.Module):
    """
    Single-target LSTM-MLP fusion model.

    Forward:
        home_seq (B,K,F) → [role token] → lstm_encoder → h_home (B,H)
        away_seq (B,K,F) → [role token] → lstm_encoder → h_away (B,H)  (shared weights)
        static_x (B,D)               → mlp_encoder  → h_static (B,E)
        concat(h_home, h_away, h_static)  (B, 2H+E)  → fusion_head → scalar (B,)

    Args:
        lstm_encoder      : shared LSTMEncoder for home; also used for away
                            unless lstm_encoder_away is provided.
        lstm_encoder_away : optional separate LSTMEncoder for away team.
                            None = use shared lstm_encoder (default).
        use_role_token    : if True, appends 1.0 (home) / 0.0 (away) to each
                            sequence step before feeding to the encoder.
    """

    def __init__(
        self,
        lstm_encoder:       LSTMEncoder,
        mlp_encoder:        MLPEncoder,
        fusion_head:        nn.Module,
        use_role_token:     bool                    = USE_ROLE_TOKEN,
        lstm_encoder_away:  Optional[LSTMEncoder]   = None,
    ) -> None:
        super().__init__()
        self.lstm_encoder      = lstm_encoder
        self.lstm_encoder_away = lstm_encoder_away
        self.mlp_encoder       = mlp_encoder
        self.fusion_head       = fusion_head
        self.use_role_token    = use_role_token

    def _encode(
        self,
        seq:      torch.Tensor,
        role_val: float,
        encoder:  nn.Module,
    ) -> torch.Tensor:
        if self.use_role_token:
            B, K, _ = seq.shape
            tok = seq.new_full((B, K, 1), fill_value=role_val)
            seq = torch.cat([seq, tok], dim=-1)
        return encoder(seq)

    def forward(
        self,
        home_seq:  torch.Tensor,   # (B, K, F)
        away_seq:  torch.Tensor,   # (B, K, F)
        static_x:  torch.Tensor,   # (B, D)
    ) -> torch.Tensor:
        enc_away = self.lstm_encoder_away or self.lstm_encoder
        h_home   = self._encode(home_seq, 1.0, self.lstm_encoder)
        h_away   = self._encode(away_seq, 0.0, enc_away)
        h_static = self.mlp_encoder(static_x)
        z        = torch.cat([h_home, h_away, h_static], dim=-1)
        return self.fusion_head(z)


# =============================================================================
# ==  MULTI-TARGET FUSION MODEL  ==============================================
# =============================================================================

class LSTMMLPMultiModel(nn.Module):
    """
    Multi-target LSTM-MLP fusion model.

    Identical feature extraction to LSTMMLPModel, but instead of a single
    fusion head there is one independent head per prediction target.

    Returns (B, n_targets) raw predictions.

    Args:
        head_hidden: if True each head has an intermediate hidden layer
                     (Linear → Activation → Linear), else a direct Linear.
    """

    def __init__(
        self,
        lstm_encoder:       LSTMEncoder,
        mlp_encoder:        MLPEncoder,
        fusion_dim:         int,
        activation:         str,
        dropout:            float,
        n_targets:          int,
        head_hidden:        bool                    = True,
        use_role_token:     bool                    = USE_ROLE_TOKEN,
        lstm_encoder_away:  Optional[LSTMEncoder]   = None,
    ) -> None:
        super().__init__()
        self.lstm_encoder      = lstm_encoder
        self.lstm_encoder_away = lstm_encoder_away
        self.mlp_encoder       = mlp_encoder
        self.use_role_token    = use_role_token
        head_dim = max(16, fusion_dim // 2)
        self.heads = nn.ModuleList()
        for _ in range(n_targets):
            if head_hidden:
                self.heads.append(nn.Sequential(
                    nn.LayerNorm(fusion_dim),
                    nn.Linear(fusion_dim, head_dim),
                    get_activation(activation),
                    nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
                    nn.Linear(head_dim, 1),
                ))
            else:
                self.heads.append(nn.Linear(fusion_dim, 1))

    def _encode(
        self,
        seq:      torch.Tensor,
        role_val: float,
        encoder:  nn.Module,
    ) -> torch.Tensor:
        if self.use_role_token:
            B, K, _ = seq.shape
            tok = seq.new_full((B, K, 1), fill_value=role_val)
            seq = torch.cat([seq, tok], dim=-1)
        return encoder(seq)

    def forward(
        self,
        home_seq:  torch.Tensor,
        away_seq:  torch.Tensor,
        static_x:  torch.Tensor,
    ) -> torch.Tensor:
        enc_away = self.lstm_encoder_away or self.lstm_encoder
        h_home   = self._encode(home_seq, 1.0, self.lstm_encoder)
        h_away   = self._encode(away_seq, 0.0, enc_away)
        h_static = self.mlp_encoder(static_x)
        z        = torch.cat([h_home, h_away, h_static], dim=-1)
        return torch.cat([head(z) for head in self.heads], dim=1)


# =============================================================================
# ==  MODEL FACTORIES  ========================================================
# =============================================================================

def _build_fusion_head(
    fusion_dim:    int,
    activation:    str,
    dropout:       float,
    n_hidden:      int,          # 0 = single Linear; 1 = Linear→Act→Drop→Linear
) -> nn.Module:
    if n_hidden == 0:
        return nn.Sequential(nn.Linear(fusion_dim, 1), nn.Flatten(0))
    mid = max(16, fusion_dim // 2)
    return nn.Sequential(
        nn.LayerNorm(fusion_dim),
        nn.Linear(fusion_dim, mid),
        get_activation(activation),
        nn.Dropout(dropout) if dropout > 0.0 else nn.Identity(),
        nn.Linear(mid, 1),
    )


class _SqueezeHead(nn.Module):
    """Wraps a sequential head and squeezes the trailing dim to get (B,)."""
    def __init__(self, seq: nn.Sequential) -> None:
        super().__init__()
        self.seq = seq
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x).squeeze(-1)


def build_single_model(
    *,
    static_input_dim:     int,
    mlp_layer_sizes:      List[int],
    activation:           str,
    mlp_dropout:          float,
    lstm_hidden_size:     int,
    lstm_num_layers:      int,
    lstm_dropout:         float,
    fusion_head_n_hidden: int   = 1,
    fusion_dropout:       float = 0.0,
    use_shared_lstm:      bool  = True,
) -> LSTMMLPModel:
    """
    Factory for the single-target LSTMMLPModel.

    use_shared_lstm=True  → one LSTMEncoder shared for home and away.
    use_shared_lstm=False → two independent LSTMEncoders (separate weights).
    """
    mlp_enc   = MLPEncoder(static_input_dim, mlp_layer_sizes, activation, mlp_dropout)
    lstm_home = LSTMEncoder(_LSTM_INPUT_SIZE, lstm_hidden_size, lstm_num_layers, lstm_dropout)
    lstm_away = (
        None if use_shared_lstm
        else LSTMEncoder(_LSTM_INPUT_SIZE, lstm_hidden_size, lstm_num_layers, lstm_dropout)
    )
    fusion_dim  = 2 * lstm_hidden_size + mlp_enc.out_dim
    head        = _SqueezeHead(_build_fusion_head(fusion_dim, activation, fusion_dropout, fusion_head_n_hidden))
    return LSTMMLPModel(lstm_home, mlp_enc, head, lstm_encoder_away=lstm_away)


def build_multi_model(
    *,
    static_input_dim:  int,
    mlp_layer_sizes:   List[int],
    activation:        str,
    mlp_dropout:       float,
    lstm_hidden_size:  int,
    lstm_num_layers:   int,
    lstm_dropout:      float,
    n_targets:         int,
    head_hidden:       bool  = True,
    fusion_dropout:    float = 0.0,
    use_shared_lstm:   bool  = True,
) -> LSTMMLPMultiModel:
    """Factory for the multi-target LSTMMLPMultiModel."""
    mlp_enc   = MLPEncoder(static_input_dim, mlp_layer_sizes, activation, mlp_dropout)
    lstm_home = LSTMEncoder(_LSTM_INPUT_SIZE, lstm_hidden_size, lstm_num_layers, lstm_dropout)
    lstm_away = (
        None if use_shared_lstm
        else LSTMEncoder(_LSTM_INPUT_SIZE, lstm_hidden_size, lstm_num_layers, lstm_dropout)
    )
    fusion_dim = 2 * lstm_hidden_size + mlp_enc.out_dim
    return LSTMMLPMultiModel(
        lstm_home, mlp_enc, fusion_dim, activation, fusion_dropout,
        n_targets, head_hidden, lstm_encoder_away=lstm_away,
    )


# =============================================================================
# ==  LOSS HELPERS  ===========================================================
# =============================================================================

def get_criterion(target_name: str) -> nn.Module:
    if TARGET_LOSS_MAP.get(target_name, "mse") == "poisson":
        return nn.PoissonNLLLoss(log_input=True, full=False, reduction="mean")
    return nn.MSELoss()


def get_criteria() -> List[nn.Module]:
    return [get_criterion(t) for t in TARGETS]


def decode_predictions(preds: np.ndarray, target_name: str) -> np.ndarray:
    if TARGET_LOSS_MAP.get(target_name, "mse") == "poisson":
        return np.exp(np.clip(preds, -10.0, 10.0))
    return preds


def decode_all_predictions(preds: np.ndarray) -> np.ndarray:
    out = preds.copy()
    for i, t in enumerate(TARGETS):
        if TARGET_LOSS_MAP.get(t, "mse") == "poisson":
            out[:, i] = np.exp(np.clip(preds[:, i], -10.0, 10.0))
    return out


def build_target_scalers(y_train: np.ndarray) -> Dict[int, StandardScaler]:
    """Fit StandardScaler for each MSE target (training data only)."""
    scalers: Dict[int, StandardScaler] = {}
    for i, t in enumerate(TARGETS):
        if TARGET_LOSS_MAP.get(t, "mse") == "mse":
            sc = StandardScaler()
            sc.fit(y_train[:, i:i+1])
            scalers[i] = sc
    return scalers


def decode_and_unscale(
    preds:          np.ndarray,
    target_scalers: Dict[int, StandardScaler],
) -> np.ndarray:
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
# ==  TRAINING LOOPS  =========================================================
# =============================================================================

def _collect_preds(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    criterion:  nn.Module,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Run one validation pass.  Returns (weighted_loss, all_preds, all_targets).
    Handles both single-target (y: (B,)) and multi-target (y: (B,T)) loaders.
    """
    loss_acc      = 0.0
    total         = 0
    preds_list:   List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    with torch.no_grad():
        for h_seq, a_seq, x_stat, yb in loader:
            h_seq, a_seq, x_stat, yb = (
                h_seq.to(device), a_seq.to(device),
                x_stat.to(device), yb.to(device),
            )
            n     = yb.size(0)
            preds = model(h_seq, a_seq, x_stat)
            loss_acc += criterion(preds, yb).item() * n
            total    += n
            preds_list.append(preds.cpu())
            targets_list.append(yb.cpu())
    return loss_acc / total, torch.cat(preds_list), torch.cat(targets_list)


def train_lstm_mlp_model(
    model:        nn.Module,
    optimizer:    optim.Optimizer,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    criterion:    nn.Module,
    epochs:       int,
    patience:     int,
    lr_scheduler  = None,
    optuna_trial: Optional[optuna.trial.Trial] = None,
) -> Tuple[float, int, Optional[Dict]]:
    """
    Single-target training loop with early stopping, gradient clipping, and
    optional Optuna pruning.  Mirrors train_one_model() in optimizer_mlp_torch.py.
    """
    best_val_rmse     = float("inf")
    best_state:       Optional[Dict] = None
    epochs_no_improve = 0
    epochs_run        = 0
    _is_poisson       = isinstance(criterion, nn.PoissonNLLLoss)
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        for h_seq, a_seq, x_stat, yb in train_loader:
            h_seq, a_seq, x_stat, yb = (
                h_seq.to(device), a_seq.to(device),
                x_stat.to(device), yb.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(h_seq, a_seq, x_stat), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss, all_preds, all_targets = _collect_preds(model, val_loader, device, criterion)

        if _is_poisson:
            all_preds = torch.exp(all_preds).clamp(max=1e6)
        if not math.isfinite(val_loss) or torch.isnan(all_preds).any() or torch.isinf(all_preds).any():
            val_rmse = float("inf")
        else:
            val_rmse = float(((all_preds - all_targets) ** 2).mean().sqrt())
            if not math.isfinite(val_rmse):
                val_rmse = float("inf")

        epochs_run = epoch
        if lr_scheduler is not None:
            old_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step(val_loss)
            if optimizer.param_groups[0]["lr"] < old_lr:
                epochs_no_improve = 0

        if optuna_trial is not None:
            optuna_trial.report(val_rmse, step=epoch)
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if val_rmse < best_val_rmse - 1e-7:
            best_val_rmse     = val_rmse
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is None:
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_val_rmse = float("inf")
    return best_val_rmse, epochs_run, best_state


def train_lstm_multioutput_model(
    model:        nn.Module,
    optimizer:    optim.Optimizer,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    device:       torch.device,
    criteria:     List[nn.Module],
    epochs:       int,
    patience:     int,
    lr_scheduler  = None,
    optuna_trial: Optional[optuna.trial.Trial] = None,
) -> Tuple[float, int, Optional[Dict]]:
    """
    Multi-target training loop.  Combined loss = uniform mean of per-target
    losses.  Mirrors train_multioutput_model() in optimizer_mlp_multioutput_torch.py.
    """
    best_val_rmse     = float("inf")
    best_state:       Optional[Dict] = None
    epochs_no_improve = 0
    epochs_run        = 0
    n_targets         = len(criteria)
    is_log_space      = [isinstance(c, nn.PoissonNLLLoss) for c in criteria]
    _tw = torch.full((n_targets,), 1.0 / n_targets, dtype=torch.float32, device=device)

    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        for h_seq, a_seq, x_stat, yb in train_loader:
            h_seq, a_seq, x_stat, yb = (
                h_seq.to(device), a_seq.to(device),
                x_stat.to(device), yb.to(device),
            )
            optimizer.zero_grad(set_to_none=True)
            preds    = model(h_seq, a_seq, x_stat)
            per_loss = torch.stack([criteria[i](preds[:, i], yb[:, i]) for i in range(n_targets)])
            (per_loss * _tw).sum().backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss_acc  = 0.0
        total         = 0
        preds_list:   List[torch.Tensor] = []
        targets_list: List[torch.Tensor] = []
        with torch.no_grad():
            for h_seq, a_seq, x_stat, yb in val_loader:
                h_seq, a_seq, x_stat, yb = (
                    h_seq.to(device), a_seq.to(device),
                    x_stat.to(device), yb.to(device),
                )
                n     = yb.size(0)
                preds = model(h_seq, a_seq, x_stat)
                per_bl = torch.stack([criteria[i](preds[:, i], yb[:, i]) for i in range(n_targets)])
                val_loss_acc += (per_bl * _tw).sum().item() * n
                total        += n
                preds_list.append(preds.cpu())
                targets_list.append(yb.cpu())

        val_loss    = val_loss_acc / total
        all_preds   = torch.cat(preds_list)
        all_targets = torch.cat(targets_list)
        for i, log_sp in enumerate(is_log_space):
            if log_sp:
                all_preds[:, i] = torch.exp(all_preds[:, i]).clamp(max=1e6)

        if not math.isfinite(val_loss) or torch.isnan(all_preds).any() or torch.isinf(all_preds).any():
            val_rmse = float("inf")
        else:
            val_rmse = float(((all_preds - all_targets) ** 2).mean(dim=0).mean().sqrt())
            if not math.isfinite(val_rmse):
                val_rmse = float("inf")

        epochs_run = epoch
        if lr_scheduler is not None:
            old_lr = optimizer.param_groups[0]["lr"]
            lr_scheduler.step(val_loss)
            if optimizer.param_groups[0]["lr"] < old_lr:
                epochs_no_improve = 0

        if optuna_trial is not None:
            optuna_trial.report(val_rmse, step=epoch)
            if optuna_trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        if val_rmse < best_val_rmse - 1e-7:
            best_val_rmse     = val_rmse
            best_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is None:
        best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_val_rmse = float("inf")
    return best_val_rmse, epochs_run, best_state
