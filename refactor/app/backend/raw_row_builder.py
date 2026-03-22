# raw_row_builder.py
# Builds a single-row DataFrame whose columns match the training CSV schema
# (excluding targets).  This gives build_X() the exact same input shape it
# sees during training.
#
# Public API:
#   build_raw_row(...)      -> 1-row pd.DataFrame  (216 cols: meta + features)
#   strip_non_features(df)  -> pd.DataFrame         (177 cols: stat + NO_OF + form + cform + stage)
#
# Column order mirrors build_table_v2.py exactly:
#   meta (5) → player_id + slot stats + NO_OF per side (34+114+6) → form (28) → cform (28) → stage (1)

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_BACKEND_DIR  = Path(__file__).resolve().parent
_APP_DIR      = _BACKEND_DIR.parent
_REFACTOR_DIR = _APP_DIR.parent
for _p in [str(_REFACTOR_DIR), str(_APP_DIR), str(_BACKEND_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared_config import ROLE_CFG, SIDES, META_COLS, DROP_IF_CONTAINS
from data_layer import (
    DATA_ROOT, get_all_season_dirs, _fixture_ts, _read_json,
)
from feature_builder import (
    load_player_stats_bulk,
    _scan_completed_fixtures_for_team,
    _compute_form_for_team,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TARGET_KEYS = [
    "GOALS", "CORNERS", "YELLOWCARDS", "SHOTS_ON_TARGET",
    "FOULS", "OFFSIDES", "REDCARDS",
]


# ---------------------------------------------------------------------------
# Column order helpers  (mirrors build_table_v2.py → build_header())
# ---------------------------------------------------------------------------

def _slot_id_col(role: str, side: str, slot_i: int = 0) -> str:
    """Player ID column name matching training convention (lowercase)."""
    r = role.lower()
    s = side.lower()
    if slot_i == 0:
        return f"{r}_{s}_player_id"
    return f"{r}{slot_i}_{s}_player_id"


def _build_feature_header() -> List[str]:
    """
    Exact ordered list of feature columns (no targets).
    Matches build_table_v2.py → build_header() minus target columns.
    """
    meta = list(META_COLS)   # 5 columns
    features: List[str] = []

    for side in SIDES:
        s = side   # HOME or AWAY
        for role, cfg in ROLE_CFG.items():
            max_slots = cfg["max_slots"]
            stats     = cfg["stats"]
            carry     = cfg.get("carry_over", False)

            if carry:
                # GK: one player_id + stats, no index
                features.append(_slot_id_col(role, s))
                for stat in stats:
                    features.append(f"{role}_{s}_{stat}")
            else:
                for i in range(1, max_slots + 1):
                    features.append(_slot_id_col(role, s, i))
                    for stat in stats:
                        features.append(f"{role}{i}_{s}_{stat}")
                features.append(f"NO_OF_{role}_{s}")

    # Form (28): side → stat → direction
    for side in SIDES:
        for stat in _TARGET_KEYS:
            features.append(f"{side}_FORM_{stat}_FOR")
        for stat in _TARGET_KEYS:
            features.append(f"{side}_FORM_{stat}_AGAINST")

    # CForm (28): same layout
    for side in SIDES:
        for stat in _TARGET_KEYS:
            features.append(f"{side}_CFORM_{stat}_FOR")
        for stat in _TARGET_KEYS:
            features.append(f"{side}_CFORM_{stat}_AGAINST")

    # Stage (1)
    features.append("STAGE_NORMALIZED")

    return meta + features


# Cache header (it's static and derived purely from ROLE_CFG)
_HEADER: List[str] = _build_feature_header()


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_raw_row(
    home_team_id: int,
    away_team_id: int,
    home_sel: Dict[str, List[int]],
    away_sel: Dict[str, List[int]],
    player_stats: Dict[int, Dict[str, float]],
    season_dir: Path,
    data_root: Path = DATA_ROOT,
    fixture_id: int = 0,
) -> pd.DataFrame:
    """
    Build a 1-row DataFrame matching the training CSV schema (no targets).

    Parameters
    ----------
    home_sel / away_sel : {"GK": [pid], "DF": [pid,...], "MF": [...], "ATK": [...]}
    player_stats        : {pid: {"GOALS_CONCEDED": float, ...}}  (from load_player_stats_bulk)
    season_dir          : current season directory
    data_root           : root data/ directory (for cross-season cform)
    fixture_id          : optional fixture ID (for debug)
    """
    row: Dict[str, Any] = {}

    # --- Meta (placeholder values for inference) ---
    row["season_label"]  = season_dir.name.split("_")[0] if season_dir else ""
    row["fixture_id"]    = fixture_id
    row["fixture_ts"]    = datetime.now().isoformat()
    row["home_team_id"]  = home_team_id
    row["away_team_id"]  = away_team_id

    # --- Player slots: player_id + stats per slot ---
    sels = {"HOME": home_sel, "AWAY": away_sel}
    for side in SIDES:
        sel = sels[side]
        for role, cfg in ROLE_CFG.items():
            players   = sel.get(role, [])
            max_slots = cfg["max_slots"]
            stats     = cfg["stats"]
            carry     = cfg.get("carry_over", False)

            if carry:
                # GK: single slot
                pid = players[0] if players else None
                row[_slot_id_col(role, side)] = pid if pid is not None else ""
                pst = player_stats.get(pid, {}) if pid is not None else {}
                for stat in stats:
                    row[f"{role}_{side}_{stat}"] = pst.get(stat, 0.0)
            else:
                for i in range(1, max_slots + 1):
                    pid = players[i - 1] if i <= len(players) else None
                    row[_slot_id_col(role, side, i)] = pid if pid is not None else ""
                    pst = player_stats.get(pid, {}) if pid is not None else {}
                    for stat in stats:
                        col = f"{role}{i}_{side}_{stat}"
                        row[col] = pst.get(stat, 0.0) if pid is not None else 0.0
                row[f"NO_OF_{role}_{side}"] = float(len(players))

    # --- Form / CForm ---
    all_season_dirs = get_all_season_dirs(data_root)

    for team_id, side in [(home_team_id, "HOME"), (away_team_id, "AWAY")]:
        # Form: current season only
        cur_fixtures = _scan_completed_fixtures_for_team(team_id, [season_dir])
        form_vals    = _compute_form_for_team(team_id, cur_fixtures, 5, False, season_dir)

        # CForm: all seasons
        all_fixtures = _scan_completed_fixtures_for_team(team_id, all_season_dirs)
        cform_vals   = _compute_form_for_team(team_id, all_fixtures, 5, False, season_dir)

        for stat in _TARGET_KEYS:
            for direction in ("FOR", "AGAINST"):
                key = f"{stat}_{direction}"
                row[f"{side}_FORM_{stat}_{direction}"]  = form_vals.get(key, 0.0)
                row[f"{side}_CFORM_{stat}_{direction}"] = cform_vals.get(key, 0.0)

    # --- Stage ---
    row["STAGE_NORMALIZED"] = 1.0

    # Build DataFrame in exact header order
    values = [row.get(col, 0.0) for col in _HEADER]
    return pd.DataFrame([values], columns=_HEADER)


# ---------------------------------------------------------------------------
# Feature-only helper (drop meta + player_id before build_X)
# ---------------------------------------------------------------------------

def strip_non_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop meta columns and player_id columns so the DataFrame is ready
    for build_X().  Returns ~177 feature columns.
    """
    drop = set(META_COLS)
    for col in df.columns:
        for substr in DROP_IF_CONTAINS:
            if substr in col:
                drop.add(col)
                break
    keep = [c for c in df.columns if c not in drop]
    return df[keep]
