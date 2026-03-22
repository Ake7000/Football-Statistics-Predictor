# shared_table_utils.py  (refactor/table_creation/)
#
# Self-contained utility library for table-building scripts.
# Contains all constants and helper functions needed by:
#   - form_stage_utils.py
#   - build_table_v2.py
#
# This file is independent — it does NOT import from train_fixed_slots.py.
# The logic here is identical to train_fixed_slots.py utilities, consolidated
# so the new scripts can run even if the original is deleted.

import os
import re
import json
import math
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# ==  CONSTANTS  ==============================================================
# =============================================================================

SEASON_DIR_RE = re.compile(r"^(?P<y1>\d{4})-(?P<y2>\d{4})_(?P<season_id>\d+)$")

# Position mapping based on Sportmonks ids
POSITION_ID_TO_CODE: Dict[int, str] = {
    24: "GK",
    25: "DF",
    26: "MF",
    27: "ATK",
}

# Prediction target stat keys (used for both targets and form tracking)
TARGET_KEYS: List[str] = [
    "GOALS",
    "CORNERS",
    "YELLOWCARDS",
    "SHOTS_ON_TARGET",
    "FOULS",
    "OFFSIDES",
    "REDCARDS",
]


# =============================================================================
# ==  STAT IMPUTATION  ========================================================
# =============================================================================

# Fill values for missing (NaN) match statistics.
# Keys use the HOME_*/AWAY_* prefix format produced by extract_targets().
# Any HOME_*/AWAY_* key not listed here is filled with 0.0.
STAT_FILL_VALUES: Dict[str, float] = {
    "HOME_REDCARDS":  0.0,
    "AWAY_REDCARDS":  0.0,
    "HOME_OFFSIDES":  1.93,
    "AWAY_OFFSIDES":  1.73,
    "HOME_FOULS":    11.65,
    "AWAY_FOULS":    11.71,
}


def impute_stats(stats: Dict[str, float]) -> Dict[str, float]:
    """
    Return a copy of `stats` with every NaN value replaced by the
    appropriate fill value from STAT_FILL_VALUES (default 0.0).

    Only HOME_*/AWAY_* keys for TARGET_KEYS are considered; any other
    key in the input dict is passed through unchanged.
    """
    out: Dict[str, float] = dict(stats)
    for key in TARGET_KEYS:
        for side in ("HOME", "AWAY"):
            full_key = f"{side}_{key}"
            raw = out.get(full_key, math.nan)
            try:
                v = float(raw)
            except (TypeError, ValueError):
                v = math.nan
            if math.isnan(v):
                out[full_key] = STAT_FILL_VALUES.get(full_key, 0.0)
    return out


# =============================================================================
# ==  SMALL UTILITIES  ========================================================
# =============================================================================

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def is_valid_season_dir(name: str) -> bool:
    m = SEASON_DIR_RE.match(name)
    if not m:
        return False
    y1 = int(m.group("y1"))
    y2 = int(m.group("y2"))
    return y2 == y1 + 1


def season_key(name: str) -> Tuple[int, int, int]:
    """Return a sortable key (y1, y2, season_id) for a season folder name."""
    m = SEASON_DIR_RE.match(name)
    if not m:
        return (0, 0, 0)
    return (int(m.group("y1")), int(m.group("y2")), int(m.group("season_id")))


def norm_key(s: str) -> str:
    """Normalize stat keys to stable UPPER_SNAKE_CASE."""
    s = str(s)
    s = s.replace("-", " ").replace("/", " ").replace("\t", " ")
    s = "_".join([x for x in s.strip().split() if x])
    return s.upper()


def read_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def find_player_dirs(season_players_dir: str) -> Dict[int, str]:
    """Scan <season>/players/ and return {player_id: player_folder_path}."""
    mapping: Dict[int, str] = {}
    if not os.path.isdir(season_players_dir):
        return mapping
    for entry in os.scandir(season_players_dir):
        if not entry.is_dir():
            continue
        name_part, sep, suffix = entry.name.rpartition("_")
        if sep and suffix.isdigit():
            mapping[int(suffix)] = entry.path
    return mapping


# =============================================================================
# ==  PLAYER STAT ACCUMULATORS  ===============================================
# =============================================================================

def _accumulate_stat(out: Dict[str, float], key_n: str, val: Optional[float]) -> None:
    if val is None:
        return
    try:
        v = float(val)
    except Exception:
        return
    out[key_n] = out.get(key_n, 0.0) + v


def _extract_numeric_from_dict(
    dct: Dict,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (total, vin, vout). Typically only one will be non-None."""
    if isinstance(dct, dict) and "total" in dct:
        try:
            return (float(dct["total"]), None, None)
        except Exception:
            pass
    if isinstance(dct, dict) and "value" in dct and isinstance(dct["value"], (int, float, str)):
        try:
            return (float(dct["value"]), None, None)
        except Exception:
            pass
    if isinstance(dct, dict) and (("in" in dct) or ("out" in dct)):
        vin  = dct.get("in",  0) or 0
        vout = dct.get("out", 0) or 0
        try:
            vin = float(vin)
        except Exception:
            vin = 0.0
        try:
            vout = float(vout)
        except Exception:
            vout = 0.0
        return (None, vin, vout)
    return (None, None, None)


def _accumulate_from_detail(out: Dict[str, float], detail: Dict) -> None:
    if not isinstance(detail, dict):
        return
    t   = detail.get("type") or {}
    key = t.get("developer_name") or t.get("code") or t.get("name")
    if not key:
        return
    key_n = norm_key(key)

    val     = detail.get("value")
    handled = False
    if isinstance(val, dict):
        total, vin, vout = _extract_numeric_from_dict(val)
        if total is not None:
            _accumulate_stat(out, key_n, total)
            handled = True
        elif (vin is not None) or (vout is not None):
            _accumulate_stat(out, f"{key_n}_IN",  vin  or 0.0)
            _accumulate_stat(out, f"{key_n}_OUT", vout or 0.0)
            _accumulate_stat(out, key_n, (vin or 0.0) + (vout or 0.0))
            handled = True
    elif isinstance(val, (int, float, str)):
        _accumulate_stat(out, key_n, val)
        handled = True

    if not handled:
        d2 = detail.get("data")
        if isinstance(d2, dict):
            total, vin, vout = _extract_numeric_from_dict(d2)
            if total is not None:
                _accumulate_stat(out, key_n, total)
            elif (vin is not None) or (vout is not None):
                _accumulate_stat(out, f"{key_n}_IN",  vin  or 0.0)
                _accumulate_stat(out, f"{key_n}_OUT", vout or 0.0)
                _accumulate_stat(out, key_n, (vin or 0.0) + (vout or 0.0))
        elif isinstance(d2, (int, float, str)):
            _accumulate_stat(out, key_n, d2)


# =============================================================================
# ==  PLAYER STATS LOADERS  ===================================================
# =============================================================================

def load_player_last_year_stats(player_dir: str) -> Dict[str, float]:
    """Read last_year_statistics.json → {STAT_KEY: numeric total}."""
    out: Dict[str, float] = {}
    if not player_dir:
        return out
    path    = os.path.join(player_dir, "last_year_statistics.json")
    payload = read_json(path)
    if not payload:
        return out

    data       = payload.get("data", payload)
    stats_list = None
    if isinstance(data, dict):
        stats_list = data.get("statistics")
    elif isinstance(data, list):
        stats_list = data
    if not isinstance(stats_list, list):
        stats_list = payload.get("statistics", [])
    if not isinstance(stats_list, list):
        return out

    for stat_obj in stats_list:
        if not isinstance(stat_obj, dict):
            continue
        details = stat_obj.get("details") or stat_obj.get("statistics") or []
        if not isinstance(details, list):
            continue
        for d in details:
            _accumulate_from_detail(out, d)

    return out


def load_player_current_stats(player_dir: str) -> Dict[str, float]:
    """Read current_statistics.json → {STAT_KEY: numeric total}."""
    out: Dict[str, float] = {}
    if not player_dir:
        return out
    path    = os.path.join(player_dir, "current_statistics.json")
    payload = read_json(path)
    if not payload:
        return out

    entries = [payload] if isinstance(payload, dict) else (
        [e for e in payload if isinstance(e, dict)] if isinstance(payload, list) else []
    )
    for entry in entries:
        details = entry.get("details") or entry.get("statistics") or []
        if not isinstance(details, list):
            continue
        for d in details:
            _accumulate_from_detail(out, d)

    return out


def load_player_stats_with_current_fallback(
    player_dir: str,
    use_current_fallback: bool = True,
) -> Dict[str, float]:
    """
    Start from last_year stats; fill any missing keys from current stats.
    """
    base = load_player_last_year_stats(player_dir)
    if not use_current_fallback:
        return base
    cur = load_player_current_stats(player_dir)
    for k, v in cur.items():
        if k not in base:
            base[k] = v
    return base


# =============================================================================
# ==  LINEUP PARSING  =========================================================
# =============================================================================

def parse_lineup_players(lineup_payload: Dict) -> List[Dict[str, Any]]:
    """
    Extract a flat list of starting lineup entries (type_id == 11).
    Each entry: {player_id, team_id, position_id, formation_position, type_id}
    """
    players: List[Dict[str, Any]] = []

    def consider(entry: Dict) -> None:
        type_id = entry.get("type_id")
        try:
            type_id = int(type_id)
        except Exception:
            return
        if type_id != 11:
            return
        pid  = entry.get("player_id")
        tid  = entry.get("team_id")
        pos  = entry.get("position_id")
        fpos = entry.get("formation_position", None)
        if pid is None or tid is None or pos is None:
            return
        try:
            pid = int(pid); tid = int(tid); pos = int(pos)
        except Exception:
            return
        ffpos = None
        if isinstance(fpos, int):
            ffpos = fpos
        elif isinstance(fpos, str) and fpos.isdigit():
            ffpos = int(fpos)
        players.append({
            "player_id":         pid,
            "team_id":           tid,
            "position_id":       pos,
            "formation_position": ffpos,
            "type_id":           11,
        })

    if isinstance(lineup_payload, dict):
        d = lineup_payload.get("data")
        if isinstance(d, dict):
            lineups = d.get("lineups") or d.get("lineup")
            if isinstance(lineups, list):
                for e in lineups:
                    if isinstance(e, dict):
                        consider(e)
        for key in ("lineups", "lineup", "players"):
            arr = lineup_payload.get(key)
            if isinstance(arr, list):
                for e in arr:
                    if isinstance(e, dict):
                        consider(e)

    uniq: Dict[Tuple[int, int], Dict] = {}
    for p in players:
        uniq[(p["player_id"], p["team_id"])] = p
    return list(uniq.values())


# =============================================================================
# ==  FIXTURE HELPERS  ========================================================
# =============================================================================

def resolve_home_away_team_ids(fixture_dir: str) -> Optional[Tuple[int, int]]:
    """Return (home_team_id, away_team_id) from data.json or statistics.json."""
    data_path = os.path.join(fixture_dir, "data.json")
    payload   = read_json(data_path)
    if payload and isinstance(payload.get("data"), dict):
        participants = payload["data"].get("participants")
        if isinstance(participants, list):
            home_id = away_id = None
            for p in participants:
                if not isinstance(p, dict):
                    continue
                tid  = p.get("id")
                meta = p.get("meta") or {}
                loc  = (meta.get("location") or "").lower()
                if tid is None:
                    continue
                if loc == "home":
                    home_id = int(tid)
                elif loc == "away":
                    away_id = int(tid)
            if home_id is not None and away_id is not None:
                return (home_id, away_id)

    stat_path = os.path.join(fixture_dir, "statistics.json")
    sp = read_json(stat_path)
    if sp:
        parts = sp.get("participants")
        if isinstance(parts, list):
            home_id = away_id = None
            for p in parts:
                if not isinstance(p, dict):
                    continue
                tid  = p.get("id") or p.get("participant_id")
                meta = p.get("meta") or {}
                loc  = (meta.get("location") or p.get("location") or "").lower()
                if tid is None:
                    continue
                if loc == "home":
                    home_id = int(tid)
                elif loc == "away":
                    away_id = int(tid)
            if home_id is not None and away_id is not None:
                return (home_id, away_id)

    return None


def extract_targets(
    stat_payload: Dict,
    home_team_id: int,
    away_team_id: int,
) -> Dict[str, Optional[float]]:
    """
    Extract HOME_/AWAY_ values for TARGET_KEYS from statistics.json.
    Returns NaN for any missing stat.
    """
    results: Dict[str, Optional[float]] = {}
    for key in TARGET_KEYS:
        results[f"HOME_{key}"] = math.nan
        results[f"AWAY_{key}"] = math.nan

    def take_numeric(entry: Dict) -> Optional[float]:
        val = entry.get("value")
        if isinstance(val, dict) and "total" in val:
            try: return float(val["total"])
            except Exception: pass
        if isinstance(val, (int, float, str)):
            try: return float(val)
            except Exception: pass
        dct = entry.get("data")
        if isinstance(dct, dict):
            if "value" in dct:
                try: return float(dct["value"])
                except Exception: pass
            if "total" in dct:
                try: return float(dct["total"])
                except Exception: pass
        if isinstance(dct, (int, float, str)):
            try: return float(dct)
            except Exception: pass
        return None

    def consider(entry: Dict) -> None:
        t     = entry.get("type") or {}
        key   = t.get("developer_name") or t.get("code") or t.get("name")
        if not key:
            return
        key_n = norm_key(key)
        if key_n not in TARGET_KEYS:
            return
        loc = (entry.get("location") or "").lower().strip()
        if loc not in ("home", "away"):
            pid = entry.get("participant_id", entry.get("team_id"))
            try:
                pid = int(pid) if pid is not None else None
            except Exception:
                pid = None
            if pid is None:
                return
            loc = "home" if pid == home_team_id else (
                  "away" if pid == away_team_id  else "")
        if loc not in ("home", "away"):
            return
        v = take_numeric(entry)
        if v is None:
            return
        results[f"{loc.upper()}_{key_n}"] = v

    data = stat_payload.get("data")
    if isinstance(data, dict):
        stats_list = data.get("statistics")
        if isinstance(stats_list, list):
            for e in stats_list:
                if isinstance(e, dict):
                    consider(e)
    elif isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                consider(e)

    return results


# =============================================================================
# ==  COLUMN / SLOT HELPERS  ==================================================
# =============================================================================

def choose_sort_key_for_line(last_year_stats: Dict[str, float]) -> tuple:
    """Score for ordering players in slots (fallback when no formation_position)."""
    if "MINUTES_PLAYED" in last_year_stats:
        return (1, float(last_year_stats["MINUTES_PLAYED"]))
    if "APPEARANCES" in last_year_stats:
        return (0, float(last_year_stats["APPEARANCES"]))
    return (-1, 0.0)


def slot_id_col(pos: str, side: str, idx: Optional[int] = None) -> str:
    """
    Build the player-id column name.
    Examples:
        slot_id_col("gk", "home")    -> "gk_home_player_id"
        slot_id_col("df", "home", 2) -> "df2_home_player_id"
    """
    pos  = pos.lower()
    side = side.lower()
    if pos == "gk":
        return f"gk_{side}_player_id"
    if idx is None:
        idx = 1
    return f"{pos}{idx}_{side}_player_id"


def _windows_safe(s: str) -> str:
    """Remove characters illegal on Windows filesystems."""
    return re.sub(r'[<>:"/\\|?*\[\], ]+', "-", s).strip("-._")
