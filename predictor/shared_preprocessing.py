# shared_preprocessing.py
# Data loading and cleaning shared across all optimizer scripts.
# All decisions about what to drop / impute live here.

import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from shared_config import TARGETS, META_COLS, DROP_IF_CONTAINS

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# =============================================================================
# ==  COLUMN HELPERS  =========================================================
# =============================================================================

def _is_drop_col(col: str) -> bool:
    """Return True if the column name contains any of the DROP_IF_CONTAINS substrings."""
    return any(sub in col for sub in DROP_IF_CONTAINS)


def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast every non-numeric column to numeric.
    Values that cannot be parsed become NaN (errors='coerce').
    Operates on a copy — does not modify the input DataFrame.
    """
    df = df.copy()
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def split_features_targets(
    df: pd.DataFrame,
    targets: list[str] = TARGETS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into (X, y).

    Returns:
        X: all columns NOT in `targets`
        y: columns in `targets`
    """
    y = df[targets].copy()
    X = df.drop(columns=targets, errors="ignore").copy()
    return X, y


# =============================================================================
# ==  MAIN LOADER  ============================================================
# =============================================================================

def load_and_prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    """
    Load the training CSV and apply all cleaning steps:

    1. Drop metadata columns (META_COLS).
    2. Drop player-ID columns (any col containing DROP_IF_CONTAINS substrings).
    3. Cast all remaining columns to numeric.
    4. Impute NaN in TARGET columns with 0.
       Rationale: a NaN target means the stat was not recorded for that match;
       treating it as 0 is a conservative assumption that avoids losing the row.
    5. Impute NaN in FEATURE columns with 0.
       Rationale: slot-based features (e.g. DF3_HOME_MINUTES_PLAYED) are NaN
       when the slot is unfilled — 0 is the correct semantic value.
    6. Drop constant-zero feature columns (carry no information).

    Returns:
        Cleaned DataFrame with all TARGETS still present as columns.
    """
    csv_path = Path(csv_path)
    print(f"[LOAD] Reading {csv_path.name}")
    df = pd.read_csv(csv_path)
    print(f"[LOAD] Raw shape: {df.shape[0]} rows x {df.shape[1]} cols")

    # --- 1. Drop metadata ---
    meta_present = [c for c in META_COLS if c in df.columns]
    if meta_present:
        df.drop(columns=meta_present, inplace=True)
        print(f"[CLEAN] Dropped {len(meta_present)} metadata columns.")

    # --- 2. Drop player-ID columns ---
    id_cols = [c for c in df.columns if _is_drop_col(c)]
    if id_cols:
        df.drop(columns=id_cols, inplace=True)
        print(f"[CLEAN] Dropped {len(id_cols)} player-ID columns.")

    # --- 3. Cast to numeric ---
    df = _to_numeric_df(df)

    # --- 4. Impute NaN targets with 0 ---
    target_cols_present = [t for t in TARGETS if t in df.columns]
    target_nan = int(df[target_cols_present].isna().sum().sum())
    if target_nan > 0:
        print(f"[IMPUTE] Replaced {target_nan} NaN(s) in target columns with 0.")
        df[target_cols_present] = df[target_cols_present].fillna(0)

    # --- 5. Impute NaN features with 0 ---
    feature_cols = [c for c in df.columns if c not in TARGETS]
    feat_nan = int(df[feature_cols].isna().sum().sum())
    if feat_nan > 0:
        print(f"[IMPUTE] Replaced {feat_nan} NaN(s) in feature columns with 0.")
        df[feature_cols] = df[feature_cols].fillna(0)

    # --- 6. Drop constant-zero feature columns ---
    const_zero = [c for c in feature_cols if df[c].abs().sum() == 0]
    if const_zero:
        df.drop(columns=const_zero, inplace=True)
        print(f"[CLEAN] Dropped {len(const_zero)} constant-zero feature columns.")

    print(f"[CLEAN] Final shape: {df.shape[0]} rows x {df.shape[1]} cols")
    return df
