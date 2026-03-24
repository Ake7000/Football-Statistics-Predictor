# feature_builder.py
# Builds the static feature row (1-row DataFrame in training-table schema) from
# user's player selections + rolling form computed from past fixture data.
#
# Public API:
#   build_static_row(home_sel, away_sel, player_stats, season_dir, data_root)
#     -> pd.DataFrame  (1 row, columns = training schema, no targets, no player IDs)
#
# Inputs:
#   home_sel / away_sel: {
#       "GK":  [player_id, ...],  # exactly 1
#       "DF":  [player_id, ...],  # 1..6
#       "MF":  [player_id, ...],  # 1..6
#       "ATK": [player_id, ...],  # 1..4
#   }
#   player_stats: { player_id: { "GOALS_CONCEDED": float, ... } }
#     (pre-loaded via load_player_stats_bulk)
#
# Outputs a DataFrame whose columns exactly mirror the training CSV
# (without player_id cols, without targets), plus STAGE_NORMALIZED and
# form / cform columns.

import json
import math
import re
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_BACKEND_DIR  = Path(__file__).resolve().parent           # licenta/refactor/app/backend/
_APP_DIR      = _BACKEND_DIR.parent                      # licenta/refactor/app/
_REFACTOR_DIR = _APP_DIR.parent                          # licenta/refactor/
for _p in [str(_REFACTOR_DIR), str(_APP_DIR), str(_BACKEND_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared_config import ROLE_CFG, SIDES, WORKSPACE_ROOT
from data_layer import (
    DATA_ROOT, _is_valid_season, _season_sort_key, _fixture_ts,
    _read_json, POSITION_ID_TO_CODE, get_all_season_dirs,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FORM_WINDOW  = 5
_TARGET_KEYS  = ["GOALS", "CORNERS", "YELLOWCARDS", "SHOTS_ON_TARGET",
                  "FOULS", "OFFSIDES", "REDCARDS"]

# ROLE_CFG stat lists (what to put in slot columns)
_GK_STATS  = ROLE_CFG["GK"]["stats"]
_DF_STATS  = ROLE_CFG["DF"]["stats"]
_MF_STATS  = ROLE_CFG["MF"]["stats"]
_ATK_STATS = ROLE_CFG["ATK"]["stats"]

_ROLE_STATS = {
    "GK":  _GK_STATS,
    "DF":  _DF_STATS,
    "MF":  _MF_STATS,
    "ATK": _ATK_STATS,
}


# ---------------------------------------------------------------------------
# Player stat loader
# ---------------------------------------------------------------------------

def _accumulate_stat(out: Dict[str, float], key: str, val) -> None:
    try:
        v = float(val)
        out[key] = out.get(key, 0.0) + v
    except Exception:
        pass


def _accumulate_from_detail(out: Dict[str, float], detail: Dict) -> None:
    if not isinstance(detail, dict):
        return
    t = detail.get("type") or {}
    raw_key = t.get("developer_name") or t.get("code") or t.get("name")
    if not raw_key:
        return
    raw_key = str(raw_key)
    key_n = "_".join(x for x in raw_key.replace("-", " ").replace("/", " ").split() if x).upper()

    val = detail.get("value")
    if isinstance(val, dict):
        if "total" in val:
            _accumulate_stat(out, key_n, val["total"])
        elif "in" in val or "out" in val:
            vin  = val.get("in",  0) or 0
            vout = val.get("out", 0) or 0
            try: vin = float(vin)
            except: vin = 0.0
            try: vout = float(vout)
            except: vout = 0.0
            _accumulate_stat(out, f"{key_n}_IN",  vin)
            _accumulate_stat(out, f"{key_n}_OUT", vout)
            _accumulate_stat(out, key_n, vin + vout)
        elif "value" in val:
            _accumulate_stat(out, key_n, val["value"])
    elif isinstance(val, (int, float, str)):
        _accumulate_stat(out, key_n, val)
    # fallback: try 'data' key
    if key_n not in out:
        d2 = detail.get("data")
        if isinstance(d2, (int, float, str)):
            _accumulate_stat(out, key_n, d2)
        elif isinstance(d2, dict) and "value" in d2:
            _accumulate_stat(out, key_n, d2["value"])


def _find_player_dir(player_id: int, season_dir: Path) -> Optional[Path]:
    """Find the player directory for a given player_id in a season."""
    players_dir = season_dir / "players"
    if not players_dir.is_dir():
        return None
    for entry in players_dir.iterdir():
        if not entry.is_dir():
            continue
        _, _, suffix = entry.name.rpartition("_")
        if suffix.isdigit() and int(suffix) == player_id:
            return entry
    return None


def _load_stats_from_json(path: Path) -> Dict[str, float]:
    """Parse a statistics JSON file into {STAT_KEY: float}."""
    payload = _read_json(str(path))
    if not payload:
        return {}
    out: Dict[str, float] = {}
    entries = [payload] if isinstance(payload, dict) else (payload if isinstance(payload, list) else [])
    for entry in entries:
        details = entry.get("details") or entry.get("statistics") or []
        if isinstance(details, list):
            for d in details:
                _accumulate_from_detail(out, d)
    return out


def load_player_stats(player_id: int, season_dir: Path) -> Dict[str, float]:
    """
    Load statistics for a player, matching training logic:
    Start from last_year_statistics.json, fill missing keys from
    current_statistics.json.
    """
    player_dir = _find_player_dir(player_id, season_dir)
    if player_dir is None:
        return {}

    base = _load_stats_from_json(player_dir / "last_year_statistics.json")
    cur  = _load_stats_from_json(player_dir / "current_statistics.json")
    for k, v in cur.items():
        if k not in base:
            base[k] = v
    return base


def load_player_stats_bulk(
    player_ids: List[int],
    season_dir: Path,
) -> Dict[int, Dict[str, float]]:
    """Batch load current stats for all selected players."""
    return {pid: load_player_stats(pid, season_dir) for pid in player_ids}


# ---------------------------------------------------------------------------
# Slot column builders
# ---------------------------------------------------------------------------

def _build_slot_columns(
    selection: Dict[str, List[int]],
    side: str,
    player_stats: Dict[int, Dict[str, float]],
) -> Dict[str, float]:
    """
    Build per-slot feature columns for one side (HOME or AWAY).

    Returns a flat dict:
        { "GK_HOME_GOALS_CONCEDED": float, "DF1_HOME_MINUTES_PLAYED": float, ... }
    Also sets NO_OF_{ROLE}_{SIDE} columns.
    """
    row: Dict[str, float] = {}

    for role, cfg in ROLE_CFG.items():
        players = selection.get(role, [])
        max_slots = cfg["max_slots"]
        stats = cfg["stats"]
        is_carry = cfg.get("carry_over", False)

        if is_carry:
            # Single slot, no index
            pid = players[0] if players else None
            pstats = player_stats.get(pid, {}) if pid is not None else {}
            for stat in stats:
                col = f"{role}_{side}_{stat}"
                row[col] = pstats.get(stat, 0.0)
        else:
            for slot_i in range(1, max_slots + 1):
                pid = players[slot_i - 1] if slot_i <= len(players) else None
                pstats = player_stats.get(pid, {}) if pid is not None else {}
                for stat in stats:
                    col = f"{role}{slot_i}_{side}_{stat}"
                    row[col] = pstats.get(stat, 0.0)
            # NO_OF_* column
            row[f"NO_OF_{role}_{side}"] = float(len(players))

    return row


# ---------------------------------------------------------------------------
# Form / cform computation from past fixtures
# ---------------------------------------------------------------------------

def _scan_completed_fixtures_for_team(
    team_id: int,
    season_dirs: List[Path],
) -> List[Tuple[datetime, bool, Dict[str, float]]]:
    """
    Return a chronologically sorted list of (timestamp, was_home, stats_dict)
    for all completed past fixtures involving team_id across given season_dirs.

    stats_dict has keys like HOME_GOALS, AWAY_GOALS, ... (from statistics.json).
    """
    results: List[Tuple[datetime, bool, Dict[str, float]]] = []
    now = datetime.now()

    for season_dir in season_dirs:
        fixtures_dir = season_dir / "fixtures"
        if not fixtures_dir.is_dir():
            continue
        for fix_entry in fixtures_dir.iterdir():
            if not fix_entry.is_dir():
                continue
            ts = _fixture_ts(fix_entry.name)
            if ts is None or ts >= now:
                continue
            data_path  = fix_entry / "data.json"
            stats_path = fix_entry / "statistics.json"
            if not data_path.exists() or not stats_path.exists():
                continue

            # Check team participation and determine home/away
            data_payload = _read_json(str(data_path))
            if not data_payload or not isinstance(data_payload.get("data"), dict):
                continue
            participants = data_payload["data"].get("participants", [])
            home_id = away_id = None
            for p in (participants if isinstance(participants, list) else []):
                if not isinstance(p, dict):
                    continue
                pid_raw = p.get("id")
                if pid_raw is None:
                    continue
                try: pid_val = int(pid_raw)
                except: continue
                meta = p.get("meta") or {}
                loc = (meta.get("location") or "").lower()
                if loc == "home":
                    home_id = pid_val
                elif loc == "away":
                    away_id = pid_val

            if home_id is None or away_id is None:
                continue
            if team_id not in (home_id, away_id):
                continue

            was_home = (team_id == home_id)

            # Extract target stats from statistics.json
            stats_payload = _read_json(str(stats_path)) or {}
            stats = _extract_target_stats(stats_payload, home_id, away_id)
            results.append((ts, was_home, stats))

    results.sort(key=lambda x: x[0])
    return results


def _extract_target_stats(
    stat_payload: Dict,
    home_team_id: int,
    away_team_id: int,
) -> Dict[str, float]:
    """Extract HOME/AWAY target stats from statistics.json payload."""
    out = {f"HOME_{k}": math.nan for k in _TARGET_KEYS}
    out.update({f"AWAY_{k}": math.nan for k in _TARGET_KEYS})

    def _take_numeric(entry: Dict) -> Optional[float]:
        val = entry.get("value")
        if isinstance(val, dict):
            if "total" in val:
                try: return float(val["total"])
                except: pass
            if "value" in val:
                try: return float(val["value"])
                except: pass
        if isinstance(val, (int, float, str)):
            try: return float(val)
            except: pass
        # Fallback: value nested in entry["data"]["value"] or entry["data"]["total"]
        dct = entry.get("data")
        if isinstance(dct, dict):
            if "value" in dct:
                try: return float(dct["value"])
                except: pass
            if "total" in dct:
                try: return float(dct["total"])
                except: pass
        return None

    def _consider(entry: Dict) -> None:
        t = entry.get("type") or {}
        raw_key = t.get("developer_name") or t.get("code") or t.get("name")
        if not raw_key:
            return
        key_n = "_".join(x for x in str(raw_key).replace("-", " ").replace("/", " ").split() if x).upper()
        if key_n not in _TARGET_KEYS:
            return
        loc = (entry.get("location") or "").lower().strip()
        if loc not in ("home", "away"):
            pid = entry.get("participant_id", entry.get("team_id"))
            try: pid = int(pid) if pid is not None else None
            except: pid = None
            if pid is None:
                return
            loc = "home" if pid == home_team_id else ("away" if pid == away_team_id else "")
        if loc not in ("home", "away"):
            return
        v = _take_numeric(entry)
        if v is not None:
            out[f"{loc.upper()}_{key_n}"] = v

    data = stat_payload.get("data")
    if isinstance(data, dict):
        for e in (data.get("statistics") or []):
            if isinstance(e, dict):
                _consider(e)
    elif isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                _consider(e)

    return out


def _impute_stats(stats: Dict[str, float]) -> Dict[str, float]:
    _FILLS = {
        "HOME_REDCARDS": 0.0, "AWAY_REDCARDS": 0.0,
        "HOME_OFFSIDES": 1.93, "AWAY_OFFSIDES": 1.73,
        "HOME_FOULS": 11.65, "AWAY_FOULS": 11.71,
    }
    out = dict(stats)
    for key in _TARGET_KEYS:
        for side in ("HOME", "AWAY"):
            full_key = f"{side}_{key}"
            v = out.get(full_key, math.nan)
            try: v = float(v)
            except: v = math.nan
            if math.isnan(v):
                out[full_key] = _FILLS.get(full_key, 0.0)
    return out


def _compute_form_for_team(
    team_id: int,
    completed_fixtures: List[Tuple[datetime, bool, Dict[str, float]]],
    window: int,
    current_season_only: bool,
    current_season_dir: Path,
) -> Dict[str, float]:
    """
    Compute rolling average form from up to `window` most recent fixtures.
    Returns {"{STAT}_FOR": float, "{STAT}_AGAINST": float, ...} for 7 stats.
    If no history: all 0.0.
    """
    # Filter to current season only if requested
    if current_season_only:
        current_fixtures_dir = current_season_dir / "fixtures"
        filtered = [
            (ts, was_home, stats) for ts, was_home, stats in completed_fixtures
            if (current_season_dir / "fixtures").is_dir() and
               any(
                   True
                   for fix in (current_season_dir / "fixtures").iterdir()
                   if _fixture_ts(fix.name) == ts
               )
        ]
        # Simpler approach: just scan timeline
        # For form: all fixtures are already sorted, take last `window` from current season
        # We already filter by season directory in the scan, so just take all
        recent = completed_fixtures[-window:]
    else:
        recent = completed_fixtures[-window:]

    if not recent:
        return {f"{stat}_{dir_}": 0.0 for stat in _TARGET_KEYS for dir_ in ("FOR", "AGAINST")}

    form: Dict[str, List[float]] = {
        f"{stat}_FOR":     [] for stat in _TARGET_KEYS
    }
    form.update({
        f"{stat}_AGAINST": [] for stat in _TARGET_KEYS
    })

    for _, was_home, stats in recent:
        stats = _impute_stats(stats)
        for stat in _TARGET_KEYS:
            h_val = stats.get(f"HOME_{stat}", 0.0)
            a_val = stats.get(f"AWAY_{stat}", 0.0)
            if was_home:
                form[f"{stat}_FOR"].append(h_val)
                form[f"{stat}_AGAINST"].append(a_val)
            else:
                form[f"{stat}_FOR"].append(a_val)
                form[f"{stat}_AGAINST"].append(h_val)

    return {
        k: (sum(v) / len(v)) if v else 0.0
        for k, v in form.items()
    }


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_static_row(
    home_sel: Dict[str, List[int]],
    away_sel: Dict[str, List[int]],
    player_stats: Dict[int, Dict[str, float]],
    season_dir: Path,
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    """
    Build a single-row pd.DataFrame whose columns mirror the training table schema
    (no player_id columns, no targets).

    Args:
        home_sel    : {"GK": [pid], "DF": [pid,...], "MF": [...], "ATK": [...]}
        away_sel    : same for away team
        player_stats: {player_id: {"GOALS_CONCEDED": float, ...}}  (pre-loaded)
        season_dir  : current season directory (for form computation in current season)
        data_root   : root data/ directory (for cross-season cform)
    """
    row: Dict[str, Any] = {}

    # --- Player slot columns ---
    row.update(_build_slot_columns(home_sel, "HOME", player_stats))
    row.update(_build_slot_columns(away_sel, "AWAY", player_stats))

    # --- Form and cform ---
    all_season_dirs = get_all_season_dirs(data_root)

    home_team_ids = (home_sel.get("GK") or []) + (home_sel.get("DF") or [])
    # We need team IDs, not player IDs. We don't have team IDs here directly,
    # so we scan for the home/away team IDs from the selection.
    # Actually we need to pass team IDs separately. Let's add home_team_id / away_team_id params.
    # For now, we skip form computation here – it's added by a wrapper.

    # --- Stage ---
    row["STAGE_NORMALIZED"] = 1.0

    df = pd.DataFrame([row])
    return df


def build_static_row_with_form(
    home_team_id: int,
    away_team_id: int,
    home_sel: Dict[str, List[int]],
    away_sel: Dict[str, List[int]],
    player_stats: Dict[int, Dict[str, float]],
    season_dir: Path,
    data_root: Path = DATA_ROOT,
) -> pd.DataFrame:
    """
    Full static row builder with form/cform columns included.
    home_team_id / away_team_id are used to look up past fixtures for form.
    """
    row: Dict[str, Any] = {}

    # --- Player slot columns ---
    row.update(_build_slot_columns(home_sel, "HOME", player_stats))
    row.update(_build_slot_columns(away_sel, "AWAY", player_stats))

    # --- STAGE ---
    row["STAGE_NORMALIZED"] = 1.0

    # --- Form / cform ---
    all_season_dirs = get_all_season_dirs(data_root)

    for team_id, side in [(home_team_id, "HOME"), (away_team_id, "AWAY")]:
        # Collect all completed fixtures for this team across all seasons
        all_fixtures = _scan_completed_fixtures_for_team(team_id, all_season_dirs)

        # Form: current season only
        cur_fixtures = _scan_completed_fixtures_for_team(team_id, [season_dir])
        form_vals  = _compute_form_for_team(team_id, cur_fixtures, 5, False, season_dir)
        cform_vals = _compute_form_for_team(team_id, all_fixtures, 5, False, season_dir)

        for stat in _TARGET_KEYS:
            for direction in ("FOR", "AGAINST"):
                key = f"{stat}_{direction}"
                # Form column: e.g. HOME_FORM_GOALS_FOR
                row[f"{side}_FORM_{stat}_{direction}"]  = form_vals.get(key, 0.0)
                row[f"{side}_CFORM_{stat}_{direction}"] = cform_vals.get(key, 0.0)

    return pd.DataFrame([row])


if __name__ == "__main__":
    # Quick sanity check
    season_dir = DATA_ROOT.parent / "data" / "2025-2026_25580"
    print(f"Season dir exists: {season_dir.exists()}")
