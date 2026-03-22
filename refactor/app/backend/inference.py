# inference.py  (rewritten — clean pipeline)
# Central inference: raw row → classifiers → regression models → predictions.
#
# Public API:
#   predict_all(home_team_id, away_team_id, home_sel, away_sel,
#               season_dir, data_root, workspace_root,
#               reg_registry, clf_registry) -> Dict[str, float]
#
# Pipeline:
#   1. Load player stats + build raw row (training-CSV schema)
#   2. Strip non-features → 177 cols for build_X()
#   3. Run 7 classifiers (each with its own variant) → 21 odds columns
#   4. Attach odds to feature row
#   5. Run 14 regression models (each with its own variant) → predictions
#   6. Return predictions dict

import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_BACKEND_DIR  = Path(__file__).resolve().parent
_APP_DIR      = _BACKEND_DIR.parent
_REFACTOR_DIR = _APP_DIR.parent
for _p in [str(_REFACTOR_DIR), str(_APP_DIR), str(_BACKEND_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

from shared_config import (
    TARGETS, TARGET_LOSS_MAP, VARIANTS, CLASSIFIER_TARGETS,
    TRAIN_TABLE_PATH,
)
from shared_utils import ResidualMLP, get_activation
from shared_sequence import (
    build_single_model as _build_single_lstm,
    build_multi_model  as _build_multi_lstm,
    LSTMMLPModel, LSTMEncoder, MLPEncoder,
)
from shared_features import build_X

from optimizers.optimizer_mlp_multioutput_torch import MultiOutputMLP

from raw_row_builder   import build_raw_row, strip_non_features
from feature_builder   import load_player_stats_bulk
from sequence_builder  import build_team_sequence
from data_layer        import DATA_ROOT


# ====================================================================
# Helpers
# ====================================================================

def _read_features_list(run_dir: Path) -> List[str]:
    p = run_dir / "features_list.txt"
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def _align(df: pd.DataFrame, feature_names: List[str]) -> pd.DataFrame:
    """Align 1-row DataFrame to model's feature list (fill missing 0, drop extra)."""
    for col in feature_names:
        if col not in df.columns:
            df = df.assign(**{col: 0.0})
    return df[feature_names]


def _load_scaler(run_dir: Path) -> Optional[StandardScaler]:
    p = run_dir / "scaler.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def _refit_scaler_from_csv(
    run_dir: Path,
    feature_names: List[str],
    variant: str,
) -> Optional[StandardScaler]:
    """
    Build a scaler by re-fitting from training_table.csv (MLP classifiers)
    or from the main training CSV (LSTM classifiers).
    """
    # Try run-local training_table.csv first (MLP classifiers save this)
    tt_path = run_dir / "training_table.csv"
    if tt_path.exists():
        tt_df = pd.read_csv(tt_path)
        for fc in feature_names:
            if fc not in tt_df.columns:
                tt_df[fc] = 0.0
        scaler = StandardScaler()
        scaler.fit(tt_df[feature_names].fillna(0.0).values.astype(np.float32))
        return scaler

    # Fallback: re-derive from main training CSV + build_X
    if TRAIN_TABLE_PATH.exists():
        from shared_features import build_full_feature_matrix
        main_df = pd.read_csv(TRAIN_TABLE_PATH)
        # Drop meta/id columns as the training pipeline does
        from shared_config import META_COLS, DROP_IF_CONTAINS
        drop_cols = set(META_COLS)
        for c in main_df.columns:
            for sub in DROP_IF_CONTAINS:
                if sub in c:
                    drop_cols.add(c)
        main_df = main_df.drop(columns=[c for c in drop_cols if c in main_df.columns])
        groups = VARIANTS.get(variant, [])
        if groups:
            feat_df = build_X(main_df, [g for g in groups if g != "odds"])
            # Align to the feature_names we need
            for fc in feature_names:
                if fc not in feat_df.columns:
                    feat_df[fc] = 0.0
            scaler = StandardScaler()
            scaler.fit(feat_df[feature_names].fillna(0.0).values.astype(np.float32))
            return scaler

    return None


def _decode(raw: float, target: str, model_type: str) -> float:
    """Poisson decode (exp) for neural models; XGB outputs already decoded."""
    if model_type == "xgb":
        return max(0.0, raw)
    if TARGET_LOSS_MAP.get(target, "mse") == "poisson":
        return float(np.exp(np.clip(raw, -10.0, 10.0)))
    return max(0.0, raw)


# ====================================================================
# LSTM sequence helpers (lazy)
# ====================================================================

def _prepare_seq(
    seq: np.ndarray, seq_k: int, seq_dim: int,
) -> torch.Tensor:
    """Pad/truncate one sequence to (1, K, dim) tensor."""
    arr = seq.astype(np.float32)
    if arr.shape[0] < seq_k:
        pad = np.zeros((seq_k - arr.shape[0], seq_dim), dtype=np.float32)
        arr = np.concatenate([pad, arr], axis=0)
    arr = arr[:seq_k, :seq_dim]
    return torch.tensor(arr[np.newaxis, :, :], dtype=torch.float32)


# ====================================================================
# Per-model-type predictors
# ====================================================================

def _predict_mlp(target: str, run_dir: Path, X: np.ndarray) -> float:
    ck = torch.load(str(run_dir / target / "best_model.pt"),
                    map_location="cpu", weights_only=False)
    model = ResidualMLP(
        input_dim=ck["input_dim"], layer_sizes=ck["layer_sizes"],
        activation=ck["activation"], dropout=ck["dropout"],
        out_dim=1, squeeze=True,
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    with torch.no_grad():
        raw = model(torch.tensor(X, dtype=torch.float32)).item()
    # Unscale MSE targets (single-output models save per-target scaler)
    ts_path = run_dir / target / "target_scaler.pkl"
    if ts_path.exists() and TARGET_LOSS_MAP.get(target, "mse") == "mse":
        with open(str(ts_path), "rb") as f:
            tscaler = pickle.load(f)
        raw = float(tscaler.inverse_transform([[raw]])[0, 0])
    return raw


def _predict_mlp_multi(target: str, run_dir: Path, X: np.ndarray) -> float:
    ck = torch.load(str(run_dir / "best_model.pt"),
                    map_location="cpu", weights_only=False)
    model = MultiOutputMLP(
        input_dim=ck["input_dim"], backbone_sizes=ck["layer_sizes"],
        activation=ck["activation"], dropout=ck["dropout"],
        n_targets=ck["n_targets"], head_hidden=ck["head_hidden"],
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    targets_list: List[str] = ck["targets"]
    if target not in targets_list:
        return 0.0
    idx = targets_list.index(target)
    with torch.no_grad():
        raw = model(torch.tensor(X, dtype=torch.float32))[0, idx].item()
    # Unscale MSE targets
    ts_path = run_dir / "target_scalers.pkl"
    if ts_path.exists() and TARGET_LOSS_MAP.get(target, "mse") == "mse":
        with open(str(ts_path), "rb") as f:
            tscalers = pickle.load(f)
        global_idx = TARGETS.index(target) if target in TARGETS else -1
        if global_idx >= 0 and global_idx in tscalers:
            raw = float(tscalers[global_idx].inverse_transform([[raw]])[0, 0])
    return raw


def _predict_xgb(target: str, run_dir: Path, X: np.ndarray) -> float:
    booster = xgb.Booster()
    booster.load_model(str(run_dir / target / "best_model.json"))
    return float(booster.predict(xgb.DMatrix(X))[0])


def _predict_lstm(
    target: str, run_dir: Path, X_static: np.ndarray,
    home_seq: np.ndarray, away_seq: np.ndarray,
) -> float:
    ck = torch.load(str(run_dir / target / "best_model.pt"),
                    map_location="cpu", weights_only=False)
    model = _build_single_lstm(
        static_input_dim=ck["static_input_dim"],
        mlp_layer_sizes=ck["mlp_layer_sizes"], activation=ck["activation"],
        mlp_dropout=ck["mlp_dropout"], lstm_hidden_size=ck["lstm_hidden_size"],
        lstm_num_layers=ck["lstm_num_layers"], lstm_dropout=ck["lstm_dropout"],
        fusion_head_n_hidden=ck["fusion_head_n_hidden"],
        fusion_dropout=ck["fusion_dropout"],
        use_shared_lstm=ck["use_shared_lstm"],
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    seq_k   = ck.get("seq_k", 5)
    seq_dim = ck.get("seq_input_dim", 14)
    ht = _prepare_seq(home_seq, seq_k, seq_dim)
    at = _prepare_seq(away_seq, seq_k, seq_dim)
    st = torch.tensor(X_static, dtype=torch.float32)
    with torch.no_grad():
        raw = model(ht, at, st).item()
    # Unscale MSE targets (single-output models save per-target scaler)
    ts_path = run_dir / target / "target_scaler.pkl"
    if ts_path.exists() and TARGET_LOSS_MAP.get(target, "mse") == "mse":
        with open(str(ts_path), "rb") as f:
            tscaler = pickle.load(f)
        raw = float(tscaler.inverse_transform([[raw]])[0, 0])
    return raw


def _predict_lstm_multi(
    target: str, run_dir: Path, X_static: np.ndarray,
    home_seq: np.ndarray, away_seq: np.ndarray,
) -> float:
    ck = torch.load(str(run_dir / "best_model.pt"),
                    map_location="cpu", weights_only=False)
    targets_list: List[str] = ck["targets"]
    if target not in targets_list:
        return 0.0
    idx = targets_list.index(target)
    model = _build_multi_lstm(
        static_input_dim=ck["static_input_dim"],
        mlp_layer_sizes=ck["mlp_layer_sizes"], activation=ck["activation"],
        mlp_dropout=ck["mlp_dropout"], lstm_hidden_size=ck["lstm_hidden_size"],
        lstm_num_layers=ck["lstm_num_layers"], lstm_dropout=ck["lstm_dropout"],
        n_targets=ck["n_targets"], head_hidden=ck["head_hidden"],
        fusion_dropout=ck["fusion_dropout"],
        use_shared_lstm=ck["use_shared_lstm"],
    )
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    seq_k   = ck.get("seq_k", 5)
    seq_dim = ck.get("seq_input_dim", 14)
    ht = _prepare_seq(home_seq, seq_k, seq_dim)
    at = _prepare_seq(away_seq, seq_k, seq_dim)
    st = torch.tensor(X_static, dtype=torch.float32)
    with torch.no_grad():
        raw = model(ht, at, st)[0, idx].item()
    ts_path = run_dir / "target_scalers.pkl"
    if ts_path.exists() and TARGET_LOSS_MAP.get(target, "mse") == "mse":
        with open(str(ts_path), "rb") as f:
            tscalers = pickle.load(f)
        global_idx = TARGETS.index(target) if target in TARGETS else -1
        if global_idx >= 0 and global_idx in tscalers:
            raw = float(tscalers[global_idx].inverse_transform([[raw]])[0, 0])
    return raw


# ====================================================================
# Classifier runner  (produces 21 odds columns)
# ====================================================================

def _run_classifier(
    stat: str,
    info: dict,
    features_df: pd.DataFrame,
    home_seq: Optional[np.ndarray],
    away_seq: Optional[np.ndarray],
) -> np.ndarray:
    """
    Run one classifier and return (3,) softmax probabilities [home, draw, away].
    """
    run_dir    = Path(info["run_dir"])
    model_type = info["model_type"]
    variant    = info["variant"]

    feature_names = _read_features_list(run_dir)
    if not feature_names:
        return np.array([1/3, 1/3, 1/3], dtype=np.float32)

    # Build variant-specific features for this classifier
    groups = VARIANTS.get(variant, [])
    groups_no_odds = [g for g in groups if g != "odds"]
    if groups_no_odds:
        X_df = build_X(features_df, groups_no_odds)
    else:
        X_df = features_df.copy()

    aligned = _align(X_df, feature_names)
    X = aligned.values.astype(np.float32)

    # Scale (not for XGB)
    if model_type != "xgb":
        scaler = _load_scaler(run_dir)
        if scaler is None:
            scaler = _refit_scaler_from_csv(run_dir, feature_names, variant)
        if scaler is not None:
            X = scaler.transform(X)

    if model_type == "mlp_torch":
        ck = torch.load(str(run_dir / stat / "best_model.pt"),
                        map_location="cpu", weights_only=False)
        n_classes = int(ck.get("n_classes", 3))
        model = ResidualMLP(
            input_dim=ck["input_dim"], layer_sizes=ck["layer_sizes"],
            activation=ck["activation"], dropout=ck["dropout"],
            out_dim=n_classes, squeeze=False,
        )
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor(X, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()[0]

    elif model_type == "lstm_mlp_torch":
        if home_seq is None or away_seq is None:
            return np.array([1/3, 1/3, 1/3], dtype=np.float32)
        ck = torch.load(str(run_dir / stat / "best_model.pt"),
                        map_location="cpu", weights_only=False)
        n_classes = int(ck.get("n_classes", 3))

        # Build model components manually — classifiers were trained
        # WITHOUT _SqueezeHead, so fusion_head is a plain Sequential/Linear.
        use_role   = ck.get("use_role_token", True)
        seq_dim_ck = ck.get("seq_input_dim", 14)
        lstm_in    = seq_dim_ck + int(use_role)

        lstm_enc = LSTMEncoder(lstm_in, ck["lstm_hidden_size"],
                               ck["lstm_num_layers"], ck["lstm_dropout"])
        use_shared = ck.get("use_shared_lstm", True)
        lstm_away  = (None if use_shared
                      else LSTMEncoder(lstm_in, ck["lstm_hidden_size"],
                                       ck["lstm_num_layers"], ck["lstm_dropout"]))

        mlp_enc = MLPEncoder(ck["static_input_dim"], ck["mlp_layer_sizes"],
                             ck["activation"], ck["mlp_dropout"])
        fusion_dim = 2 * ck["lstm_hidden_size"] + mlp_enc.out_dim
        n_hidden   = ck.get("fusion_head_n_hidden", 1)
        f_drop     = ck.get("fusion_dropout", 0.0)

        if n_hidden == 0:
            fusion_head = torch.nn.Linear(fusion_dim, n_classes)
        else:
            mid = max(16, fusion_dim // 2)
            fusion_head = torch.nn.Sequential(
                torch.nn.LayerNorm(fusion_dim),
                torch.nn.Linear(fusion_dim, mid),
                get_activation(ck["activation"]),
                torch.nn.Dropout(f_drop) if f_drop > 0 else torch.nn.Identity(),
                torch.nn.Linear(mid, n_classes),
            )

        model = LSTMMLPModel(lstm_enc, mlp_enc, fusion_head,
                              use_role_token=use_role,
                              lstm_encoder_away=lstm_away)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()

        seq_k   = ck.get("seq_k", 5)
        ht = _prepare_seq(home_seq, seq_k, seq_dim_ck)
        at = _prepare_seq(away_seq, seq_k, seq_dim_ck)
        st = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = model(ht, at, st)
            return torch.softmax(logits, dim=1).numpy()[0]

    # Fallback: uniform
    return np.array([1/3, 1/3, 1/3], dtype=np.float32)


# ====================================================================
# Debug CSV saver
# ====================================================================

def _save_debug_csv(
    name: str,
    columns: List[str],
    values: np.ndarray,
    debug_dir: Path,
) -> None:
    try:
        debug_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(values, columns=columns).to_csv(
            debug_dir / f"{name}.csv", index=False,
        )
    except Exception as e:
        logger.warning(f"[DEBUG] Failed to save {name}: {e}")


# ====================================================================
# Public API
# ====================================================================

def predict_all(
    home_team_id:  int,
    away_team_id:  int,
    home_sel:      Dict[str, List[int]],
    away_sel:      Dict[str, List[int]],
    season_dir,
    data_root      = DATA_ROOT,
    workspace_root = None,
    reg_registry:  dict = None,
    clf_registry:  dict = None,
    # backward compat aliases
    registry:      dict = None,
    classifiers:   dict = None,
) -> Dict[str, float]:
    """
    Run all predictions for a match.

    Returns dict mapping target names (e.g. "HOME_GOALS") to float predictions,
    plus odds_* keys for classifier probabilities.
    """
    # Handle backward-compat aliases
    if reg_registry is None:
        reg_registry = registry or {}
    if clf_registry is None:
        clf_registry = classifiers or {}
    if workspace_root is None:
        workspace_root = _REFACTOR_DIR.parent

    season_dir = Path(season_dir)
    data_root  = Path(data_root)

    debug_dir = _APP_DIR / "debug_inference"

    # ----------------------------------------------------------------
    # 1. Load player stats
    # ----------------------------------------------------------------
    all_pids: List[int] = []
    for sel in (home_sel, away_sel):
        for pids in sel.values():
            all_pids.extend(pids)
    player_stats = load_player_stats_bulk(all_pids, season_dir)
    logger.info(f"[1] Loaded stats for {len(player_stats)} players")

    # ----------------------------------------------------------------
    # 2. Build raw row (216 cols, training-CSV schema minus targets)
    # ----------------------------------------------------------------
    raw_row = build_raw_row(
        home_team_id, away_team_id,
        home_sel, away_sel,
        player_stats, season_dir, data_root,
    )
    logger.info(f"[2] Raw row built: {raw_row.shape[1]} columns")
    _save_debug_csv("00_raw_row", list(raw_row.columns), raw_row.values, debug_dir)

    # ----------------------------------------------------------------
    # 3. Strip meta + player_id → feature-only DF for build_X
    # ----------------------------------------------------------------
    features_df = strip_non_features(raw_row)
    logger.info(f"[3] Feature-only row: {features_df.shape[1]} columns")
    _save_debug_csv("01_features_only", list(features_df.columns), features_df.values, debug_dir)

    # ----------------------------------------------------------------
    # 4. Build LSTM sequences (needed if any model is LSTM-based)
    # ----------------------------------------------------------------
    _home_seq: Optional[np.ndarray] = None
    _away_seq: Optional[np.ndarray] = None

    def _get_seqs():
        nonlocal _home_seq, _away_seq
        if _home_seq is None:
            _home_seq = build_team_sequence(home_team_id, data_root)
            _away_seq = build_team_sequence(away_team_id, data_root)
        return _home_seq, _away_seq

    # ----------------------------------------------------------------
    # 5. Run classifiers → 21 odds columns
    # ----------------------------------------------------------------
    odds: Dict[str, float] = {}
    for stat in CLASSIFIER_TARGETS:
        if stat in clf_registry:
            info = clf_registry[stat]
            model_type = info["model_type"]
            needs_seq  = "lstm" in model_type
            h_seq, a_seq = _get_seqs() if needs_seq else (None, None)
            try:
                proba = _run_classifier(stat, info, features_df, h_seq, a_seq)
                logger.info(f"[CLF] {stat}: home={proba[0]:.3f} draw={proba[1]:.3f} away={proba[2]:.3f}")
            except Exception as e:
                logger.error(f"[CLF] {stat} failed: {e}", exc_info=True)
                proba = np.array([1/3, 1/3, 1/3])
        else:
            proba = np.array([1/3, 1/3, 1/3])
        odds[f"odds_{stat}_home"] = float(proba[0])
        odds[f"odds_{stat}_draw"] = float(proba[1])
        odds[f"odds_{stat}_away"] = float(proba[2])

    logger.info(f"[5] Classifiers done: {len(odds)} odds columns")

    # Attach odds to features_df so build_X(..., "odds") can pick them up
    odds_df = pd.DataFrame([odds], index=features_df.index)
    features_with_odds = pd.concat([features_df, odds_df], axis=1)
    _save_debug_csv("02_features_with_odds", list(features_with_odds.columns),
                    features_with_odds.values, debug_dir)

    # ----------------------------------------------------------------
    # 6. Run regression models → 14 predicted stats
    # ----------------------------------------------------------------
    predictions: Dict[str, float] = {}

    for target in TARGETS:
        if target not in reg_registry:
            logger.warning(f"[REG] {target}: no model in registry, skipping")
            continue

        info       = reg_registry[target]
        model_type = info["model_type"]
        variant    = info["variant"]
        run_dir    = Path(info["run_dir"])

        feature_names = _read_features_list(run_dir)
        if not feature_names:
            logger.warning(f"[REG] {target}: no features_list.txt, skipping")
            continue

        # Determine which source DF to use (with or without odds)
        has_odds_features = any(f.startswith("odds_") for f in feature_names)
        source_df = features_with_odds if has_odds_features else features_df

        # Build variant-specific features
        groups = VARIANTS.get(variant, [])
        groups_for_build = [g for g in groups if g != "odds"]
        if groups_for_build:
            X_df = build_X(source_df, groups_for_build)
        else:
            X_df = source_df.copy()

        # If model expects odds but build_X didn't produce them, add manually
        if has_odds_features:
            odds_feature_names = [f for f in feature_names if f.startswith("odds_")]
            for oc in odds_feature_names:
                if oc not in X_df.columns and oc in source_df.columns:
                    X_df[oc] = source_df[oc].values

        aligned = _align(X_df, feature_names)
        X = aligned.values.astype(np.float64)

        # Save debug CSV for this target (pre-scaling)
        _save_debug_csv(f"reg_{target}_{model_type}", feature_names, X, debug_dir)

        # Scale (not for XGB)
        if model_type != "xgb":
            scaler = _load_scaler(run_dir)
            if scaler is not None:
                if X.shape[1] != scaler.n_features_in_:
                    logger.error(
                        f"[REG] {target}: scaler expects {scaler.n_features_in_} "
                        f"features but got {X.shape[1]}"
                    )
                    predictions[target] = 0.0
                    continue
                X = scaler.transform(X)
        X = X.astype(np.float32)

        try:
            if model_type == "mlp_torch":
                raw = _predict_mlp(target, run_dir, X)
                predictions[target] = _decode(raw, target, model_type)

            elif model_type == "mlp_multioutput_torch":
                raw = _predict_mlp_multi(target, run_dir, X)
                if TARGET_LOSS_MAP.get(target, "mse") == "poisson":
                    predictions[target] = float(np.exp(np.clip(raw, -10, 10)))
                else:
                    predictions[target] = max(0.0, raw)

            elif model_type == "xgb":
                raw = _predict_xgb(target, run_dir, X)
                predictions[target] = _decode(raw, target, model_type)

            elif model_type == "lstm_mlp_torch":
                h_seq, a_seq = _get_seqs()
                raw = _predict_lstm(target, run_dir, X, h_seq, a_seq)
                predictions[target] = _decode(raw, target, model_type)

            elif model_type == "lstm_mlp_multioutput_torch":
                h_seq, a_seq = _get_seqs()
                raw = _predict_lstm_multi(target, run_dir, X, h_seq, a_seq)
                if TARGET_LOSS_MAP.get(target, "mse") == "poisson":
                    predictions[target] = float(np.exp(np.clip(raw, -10, 10)))
                else:
                    predictions[target] = max(0.0, raw)

            else:
                logger.warning(f"[REG] {target}: unknown model_type '{model_type}'")
                predictions[target] = 0.0

            logger.info(f"[REG] {target}: {predictions[target]:.4f}  ({model_type})")

        except Exception as exc:
            logger.error(f"[REG] {target}: prediction failed: {exc}", exc_info=True)
            predictions[target] = 0.0

    # ----------------------------------------------------------------
    # 7. Merge odds into predictions dict (for UI display)
    # ----------------------------------------------------------------
    predictions.update(odds)

    # ----------------------------------------------------------------
    # 8. Attach per-target CV metrics for confidence intervals
    # ----------------------------------------------------------------
    for target in TARGETS:
        info = reg_registry.get(target)
        if info is None:
            continue
        if "cv_mae" in info:
            predictions[f"mae_{target}"] = info["cv_mae"]
        if "cv_rmse_mean" in info:
            predictions[f"rmse_{target}"] = info["cv_rmse_mean"]

    return predictions
