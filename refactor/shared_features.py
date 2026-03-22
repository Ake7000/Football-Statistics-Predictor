# shared_features.py
# Composable feature group selection for all optimizer scripts.
#
# GROUPS
# ------
#   "raw"      — per-slot numeric stat columns as scraped (no player IDs, no NO_OF_*)
#   "sum"      — per-role summed stats across all player slots
#   "mean"     — per-role averaged stats (sum / player count)
#   "nplayers" — number of players per role per side (NO_OF_*)
#   "form"     — rolling form averages for home + away (last-N matches)
#   "stage"    — match stage as normalised float [0.0, 1.0]
#   "diffsum"  — home-minus-away differences of sum columns
#   "diffmean" — home-minus-away differences of mean columns
#
# USAGE
# -----
#   from shared_features import build_X, get_y, build_feature_matrices, build_full_feature_matrix
#   X  = build_X(df, ["sum", "form", "stage"])
#   y  = get_y(df)
#
# To add a new variant, edit VARIANTS in shared_config.py — nothing else changes.

from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shared_config import (
    ROLE_CFG,
    SIDES,
    TARGETS,
    TEST_SIZE,
    RANDOM_STATE,
    SHUFFLE,
    VARIANTS,
    WORKSPACE_ROOT,
    CLASSIFIER_TARGETS,
)


# =============================================================================
# ==  MODULE CONSTANTS  =======================================================
# =============================================================================

_FORM_SUBSTR  = "_FORM_"        # substring present in all form column names
_CFORM_SUBSTR = "_CFORM_"       # substring present in all continuous-form column names
_STAGE_COL    = "STAGE_NORMALIZED"
_NPLAYERS_PFX = "NO_OF_"        # prefix of player-count columns

# Public list of all supported group names (for validation / introspection).
ALL_GROUPS: List[str] = [
    "raw", "sum", "mean", "nplayers",
    "form", "cform", "stage", "diffsum", "diffmean",
    "odds",
]

_OOF_PATH = WORKSPACE_ROOT / "train_tables" / "odds_oof.npz"


# =============================================================================
# ==  INTERNAL: SUM / MEAN COMPUTATION  =======================================
# =============================================================================

def _sum_dict(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Compute per-role × per-side × per-stat SUM series.
    Returns  { "SUM_{ROLE}_{SIDE}_{STAT}": pd.Series, ... }

    carry_over roles (GK): single column taken directly.
    multi-slot roles      : summed across slot 1 … max_slots (NaN → 0).
    """
    result: Dict[str, pd.Series] = {}
    for side in SIDES:
        for role, cfg in ROLE_CFG.items():
            for stat in cfg["stats"]:
                if cfg.get("carry_over", False):
                    col = f"{role}_{side}_{stat}"
                    s = (df[col].fillna(0.0) if col in df.columns
                         else pd.Series(0.0, index=df.index))
                else:
                    cols    = [f"{role}{i}_{side}_{stat}"
                               for i in range(1, cfg["max_slots"] + 1)]
                    present = [c for c in cols if c in df.columns]
                    s = (df[present].fillna(0.0).sum(axis=1) if present
                         else pd.Series(0.0, index=df.index))
                result[f"SUM_{role}_{side}_{stat}"] = s
    return result


def _mean_dict(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Compute per-role × per-side × per-stat MEAN series.
    Returns  { "MEAN_{ROLE}_{SIDE}_{STAT}": pd.Series, ... }

    carry_over roles: mean == sum (only 1 player; no division needed).
    multi-slot roles: sum / NO_OF_{ROLE}_{SIDE}   (0 when count = 0).
    """
    sums   = _sum_dict(df)
    result: Dict[str, pd.Series] = {}
    for side in SIDES:
        for role, cfg in ROLE_CFG.items():
            if cfg.get("carry_over", False):
                for stat in cfg["stats"]:
                    result[f"MEAN_{role}_{side}_{stat}"] = sums[f"SUM_{role}_{side}_{stat}"]
            else:
                count_col = f"NO_OF_{role}_{side}"
                count = (df[count_col].fillna(0.0) if count_col in df.columns
                         else pd.Series(0.0, index=df.index))
                denom = count.replace(0.0, np.nan)   # avoid division by zero
                for stat in cfg["stats"]:
                    s = sums[f"SUM_{role}_{side}_{stat}"]
                    result[f"MEAN_{role}_{side}_{stat}"] = (s / denom).fillna(0.0)
    return result


# =============================================================================
# ==  GROUP SELECTORS  ========================================================
# =============================================================================

def get_raw(df: pd.DataFrame) -> pd.DataFrame:
    """
    All per-slot numeric stat columns as-is.
    Excludes: targets, NO_OF_* columns, form columns, STAGE_NORMALIZED.
    Player ID columns are assumed already removed by the data loader.
    """
    exclude = set(TARGETS) | {_STAGE_COL}
    cols = [
        c for c in df.columns
        if c not in exclude
        and not c.startswith(_NPLAYERS_PFX)
        and _FORM_SUBSTR not in c
        and _CFORM_SUBSTR not in c
    ]
    return df[cols].fillna(0.0)


def get_sum(df: pd.DataFrame) -> pd.DataFrame:
    """24 SUM-aggregated stat columns  (SUM_{ROLE}_{SIDE}_{STAT})."""
    return pd.DataFrame(_sum_dict(df), index=df.index)


def get_mean(df: pd.DataFrame) -> pd.DataFrame:
    """24 MEAN-aggregated stat columns  (MEAN_{ROLE}_{SIDE}_{STAT})."""
    return pd.DataFrame(_mean_dict(df), index=df.index)


def get_nplayers(df: pd.DataFrame) -> pd.DataFrame:
    """
    6 player-count columns for non-GK roles: NO_OF_{ROLE}_{SIDE}.
    GK is excluded because it is always 1 (carries no information).
    """
    cols = [
        f"NO_OF_{role}_{side}"
        for role, cfg in ROLE_CFG.items()
        if not cfg.get("carry_over", False)
        for side in SIDES
        if f"NO_OF_{role}_{side}" in df.columns
    ]
    return df[cols].fillna(0.0)


def get_form(df: pd.DataFrame) -> pd.DataFrame:
    """28 rolling form columns  ({SIDE}_FORM_{STAT}_{FOR|AGAINST})."""
    cols = [c for c in df.columns if _FORM_SUBSTR in c and _CFORM_SUBSTR not in c]
    return df[cols].fillna(0.0)


def get_cform(df: pd.DataFrame) -> pd.DataFrame:
    """28 continuous-form columns ({SIDE}_CFORM_{STAT}_{FOR|AGAINST}), no season reset."""
    cols = [c for c in df.columns if _CFORM_SUBSTR in c]
    return df[cols].fillna(0.0)


def get_stage(df: pd.DataFrame) -> pd.DataFrame:
    """1 stage-normalisation column  (STAGE_NORMALIZED, float 0.0 → 1.0)."""
    if _STAGE_COL in df.columns:
        return df[[_STAGE_COL]].fillna(0.0)
    return pd.DataFrame(index=df.index)


def get_diffsum(df: pd.DataFrame) -> pd.DataFrame:
    """
    12 home-minus-away difference columns from SUM values.
    DIFFSUM_{ROLE}_{STAT}  =  SUM_{ROLE}_HOME_{STAT} − SUM_{ROLE}_AWAY_{STAT}
    """
    sums   = _sum_dict(df)
    result: Dict[str, pd.Series] = {}
    for role, cfg in ROLE_CFG.items():
        for stat in cfg["stats"]:
            h = sums.get(f"SUM_{role}_HOME_{stat}", pd.Series(0.0, index=df.index))
            a = sums.get(f"SUM_{role}_AWAY_{stat}", pd.Series(0.0, index=df.index))
            result[f"DIFFSUM_{role}_{stat}"] = h - a
    return pd.DataFrame(result, index=df.index)


def get_diffmean(df: pd.DataFrame) -> pd.DataFrame:
    """
    12 home-minus-away difference columns from MEAN values.
    DIFFMEAN_{ROLE}_{STAT}  =  MEAN_{ROLE}_HOME_{STAT} − MEAN_{ROLE}_AWAY_{STAT}
    """
    means  = _mean_dict(df)
    result: Dict[str, pd.Series] = {}
    for role, cfg in ROLE_CFG.items():
        for stat in cfg["stats"]:
            h = means.get(f"MEAN_{role}_HOME_{stat}", pd.Series(0.0, index=df.index))
            a = means.get(f"MEAN_{role}_AWAY_{stat}", pd.Series(0.0, index=df.index))
            result[f"DIFFMEAN_{role}_{stat}"] = h - a
    return pd.DataFrame(result, index=df.index)


def get_odds(df: pd.DataFrame) -> pd.DataFrame:
    """
    21 out-of-fold classifier probability columns (3 per stat × 7 stats):
      odds_{STAT}_home  — P(HOME > AWAY)
      odds_{STAT}_draw  — P(HOME == AWAY)
      odds_{STAT}_away  — P(HOME < AWAY)

    Requires train_tables/odds_oof.npz to exist.  Generate it once with:
      python refactor/table_creation/generate_odds_features.py
    Rows are aligned to df by fixture_id so row order doesn't matter.
    """
    if not _OOF_PATH.exists():
        raise FileNotFoundError(
            f"odds_oof.npz not found at {_OOF_PATH}\n"
            "Generate it first:\n"
            "  python refactor/table_creation/generate_odds_features.py"
        )
    from shared_config import TRAIN_TABLE_PATH as _TTP

    npz_data        = np.load(str(_OOF_PATH))
    oof_fixture_ids = npz_data["fixture_ids"].astype(np.int64)
    id_to_idx: Dict[int, int] = {int(fid): i for i, fid in enumerate(oof_fixture_ids)}

    # Re-read fixture_ids from the raw CSV in original row order
    raw_df  = pd.read_csv(_TTP, usecols=["fixture_id"])
    df_fids = raw_df["fixture_id"].to_numpy(dtype=np.int64)

    # Map each training-table row to its position in the npz
    order = np.array([id_to_idx[int(fid)] for fid in df_fids], dtype=np.int64)

    col_names = (
        [f"odds_{stat}_home" for stat in CLASSIFIER_TARGETS] +
        [f"odds_{stat}_draw" for stat in CLASSIFIER_TARGETS] +
        [f"odds_{stat}_away" for stat in CLASSIFIER_TARGETS]
    )
    arrays = {col: npz_data[col][order].astype(np.float32) for col in col_names}
    return pd.DataFrame(arrays, index=df.index)


# =============================================================================
# ==  GROUP DISPATCH  =========================================================
# =============================================================================

_GROUP_FN: Dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "raw":      get_raw,
    "sum":      get_sum,
    "mean":     get_mean,
    "nplayers": get_nplayers,
    "form":     get_form,
    "cform":    get_cform,
    "stage":    get_stage,
    "diffsum":  get_diffsum,
    "diffmean": get_diffmean,
    "odds":     get_odds,
}


def build_X(df: pd.DataFrame, groups: List[str]) -> pd.DataFrame:
    """
    Concatenate the feature DataFrames for the requested groups.

    Args:
        df:     DataFrame produced by load_and_prepare_dataframe()
                (targets still present as columns is fine).
        groups: ordered list of group name strings,
                e.g. ["sum", "form", "stage"].

    Returns:
        pd.DataFrame with only the selected feature columns.
    """
    if not groups:
        raise ValueError("groups must contain at least one group name.")
    unknown = [g for g in groups if g not in _GROUP_FN]
    if unknown:
        raise ValueError(
            f"Unknown group(s): {unknown}. "
            f"Allowed: {sorted(_GROUP_FN.keys())}"
        )
    parts: List[pd.DataFrame] = [_GROUP_FN[g](df) for g in groups]
    return pd.concat(parts, axis=1)


def get_y(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 14 target columns as a DataFrame."""
    return df[TARGETS].copy()


# =============================================================================
# ==  MATRIX BUILDERS  ========================================================
# =============================================================================

def _resolve_groups(variant: str) -> List[str]:
    variant = variant.lower()
    if variant not in VARIANTS:
        raise ValueError(
            f"Unknown variant '{variant}'. "
            f"Allowed: {sorted(VARIANTS.keys())}"
        )
    return VARIANTS[variant]


def build_feature_matrices(
    df: pd.DataFrame,
    variant: str,
    apply_scaler: bool = True,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame, "StandardScaler | None", List[str]]:
    """
    Build train / val feature matrices for the requested variant.

    Args:
        df:           DataFrame produced by load_and_prepare_dataframe().
        variant:      key from VARIANTS in shared_config.py.
        apply_scaler: if True, fit StandardScaler on train and transform both.
                      Pass False for tree-based models (XGBoost).

    Returns:
        X_train, X_val        np.ndarray
        y_train_df, y_val_df  pd.DataFrame
        scaler                StandardScaler | None
        feature_names         List[str]
    """
    groups      = _resolve_groups(variant)
    features_df = build_X(df, groups)
    targets_df  = get_y(df)

    X_train_df, X_val_df, y_train_df, y_val_df = train_test_split(
        features_df, targets_df,
        test_size=TEST_SIZE,
        shuffle=SHUFFLE,
        random_state=RANDOM_STATE,
    )
    print(
        f"[SPLIT] Train: {len(X_train_df)} rows | Val: {len(X_val_df)} rows "
        f"| Features: {features_df.shape[1]} | Groups: {groups}"
    )

    if apply_scaler:
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_train_df.values)
        X_val   = scaler.transform(X_val_df.values)
    else:
        scaler  = None
        X_train = X_train_df.values
        X_val   = X_val_df.values

    return X_train, X_val, y_train_df, y_val_df, scaler, list(features_df.columns)


def build_full_feature_matrix(
    df: pd.DataFrame,
    variant: str,
    apply_scaler: bool = True,
) -> Tuple[np.ndarray, pd.DataFrame, "StandardScaler | None", List[str]]:
    """
    Build feature matrix over the FULL dataset (no train/val split).
    Used for cross-validated final evaluation.

    Returns:
        X             np.ndarray
        y_df          pd.DataFrame
        scaler        StandardScaler | None
        feature_names List[str]
    """
    groups      = _resolve_groups(variant)
    features_df = build_X(df, groups)
    targets_df  = get_y(df)

    if apply_scaler:
        scaler = StandardScaler()
        X      = scaler.fit_transform(features_df.values)
    else:
        scaler = None
        X      = features_df.values

    return X, targets_df, scaler, list(features_df.columns)
