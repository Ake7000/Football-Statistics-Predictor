# sequence_builder.py
# Builds the (K, SEQ_INPUT_DIM) LSTM input sequence for a team from past fixture history.
#
# Feature layout per step (SEQ_INPUT_DIM = 14):
#   [GOALS_FOR, GOALS_AGAINST,
#    CORNERS_FOR, CORNERS_AGAINST,
#    YELLOWCARDS_FOR, YELLOWCARDS_AGAINST,
#    SHOTS_ON_TARGET_FOR, SHOTS_ON_TARGET_AGAINST,
#    FOULS_FOR, FOULS_AGAINST,
#    OFFSIDES_FOR, OFFSIDES_AGAINST,
#    REDCARDS_FOR, REDCARDS_AGAINST]
#
# Zero-padding fills the BEGINNING of the sequence if fewer than K past matches exist.
#
# Public API:
#   build_team_sequence(team_id, data_root, K=5) -> np.ndarray  (K, 14)  float32

import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

_BACKEND_DIR  = Path(__file__).resolve().parent           # licenta/refactor/app/backend/
_APP_DIR      = _BACKEND_DIR.parent                      # licenta/refactor/app/
_REFACTOR_DIR = _APP_DIR.parent                          # licenta/refactor/
for _p in [str(_REFACTOR_DIR), str(_APP_DIR), str(_BACKEND_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared_config import WORKSPACE_ROOT, SEQ_K, SEQ_STATS, SEQ_INPUT_DIM
from data_layer import (
    DATA_ROOT, _is_valid_season, _season_sort_key, _fixture_ts, _read_json,
    get_all_season_dirs,
)

# Stats in the order they appear in the sequence (must match build_sequence_table.py)
_SEQ_STATS_ORDER = SEQ_STATS   # ["GOALS", "CORNERS", "YELLOWCARDS", "SHOTS_ON_TARGET",
                               #   "FOULS", "OFFSIDES", "REDCARDS"]

# Fill defaults for missing stats (imputation)
_STAT_FILLS: Dict[str, float] = {
    "REDCARDS":     0.0,
    "OFFSIDES":     1.8,
    "FOULS":        11.7,
}


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _collect_completed_fixtures(
    team_id: int,
    season_dirs: List[Path],
) -> List[Tuple[datetime, bool, Dict[str, float]]]:
    """
    Collect all completed past fixtures for team_id across given season dirs.
    Returns list of (timestamp, was_home, target_stats_dict) sorted oldest-first.
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

            # Resolve home/away IDs
            data_payload = _read_json(str(data_path))
            if not data_payload or not isinstance(data_payload.get("data"), dict):
                continue
            participants = data_payload["data"].get("participants") or []
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
            stats = _extract_stats(stats_path, home_id, away_id)
            results.append((ts, was_home, stats))

    results.sort(key=lambda x: x[0])
    return results


def _extract_stats(
    stats_path: Path,
    home_id: int,
    away_id: int,
) -> Dict[str, float]:
    """Extract GOALS/CORNERS/etc. from statistics.json for both teams."""
    out = {f"HOME_{k}": 0.0 for k in _SEQ_STATS_ORDER}
    out.update({f"AWAY_{k}": 0.0 for k in _SEQ_STATS_ORDER})

    payload = _read_json(str(stats_path))
    if not payload:
        return out

    def _take_num(entry: Dict) -> Optional[float]:
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
        return None

    def _consider(entry: Dict) -> None:
        t = entry.get("type") or {}
        raw_key = t.get("developer_name") or t.get("code") or t.get("name")
        if not raw_key:
            return
        key_n = "_".join(x for x in str(raw_key).replace("-", " ").replace("/", " ").split() if x).upper()
        if key_n not in _SEQ_STATS_ORDER:
            return
        loc = (entry.get("location") or "").lower().strip()
        if loc not in ("home", "away"):
            pid = entry.get("participant_id", entry.get("team_id"))
            try: pid = int(pid) if pid is not None else None
            except: pid = None
            if pid is None:
                return
            loc = "home" if pid == home_id else ("away" if pid == away_id else "")
        if loc not in ("home", "away"):
            return
        v = _take_num(entry)
        if v is not None:
            out[f"{loc.upper()}_{key_n}"] = v

    data = payload.get("data")
    if isinstance(data, dict):
        for e in (data.get("statistics") or []):
            if isinstance(e, dict):
                _consider(e)
    elif isinstance(data, list):
        for e in data:
            if isinstance(e, dict):
                _consider(e)

    # Impute missing
    for stat in _SEQ_STATS_ORDER:
        for side in ("HOME", "AWAY"):
            key = f"{side}_{stat}"
            v = out.get(key, 0.0)
            if v != v:  # NaN check
                out[key] = _STAT_FILLS.get(stat, 0.0)

    return out


def _build_step_vector(was_home: bool, stats: Dict[str, float]) -> np.ndarray:
    """
    Build a (SEQ_INPUT_DIM,) = (14,) feature vector for one past match.
    Layout: [STAT0_FOR, STAT0_AGAINST, STAT1_FOR, STAT1_AGAINST, ...]
    """
    vec = np.zeros(SEQ_INPUT_DIM, dtype=np.float32)
    for k, stat in enumerate(_SEQ_STATS_ORDER):
        h_val = float(stats.get(f"HOME_{stat}", 0.0))
        a_val = float(stats.get(f"AWAY_{stat}", 0.0))
        if was_home:
            vec[2 * k]     = h_val   # FOR
            vec[2 * k + 1] = a_val   # AGAINST
        else:
            vec[2 * k]     = a_val   # FOR
            vec[2 * k + 1] = h_val   # AGAINST
    return vec


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_team_sequence(
    team_id: int,
    data_root: Path = DATA_ROOT,
    K: int = SEQ_K,
) -> np.ndarray:
    """
    Build the (K, SEQ_INPUT_DIM) LSTM sequence for a team, using last K completed
    fixtures from ALL seasons in data_root (chronological order, oldest → newest).
    Zero-pads the beginning if fewer than K matches are available.

    Args:
        team_id   : integer team identifier.
        data_root : path to the data root directory.
        K         : number of sequence steps.

    Returns:
        np.ndarray of shape (K, SEQ_INPUT_DIM) and dtype float32.
    """
    all_dirs = get_all_season_dirs(data_root)
    all_fixtures = _collect_completed_fixtures(team_id, all_dirs)

    # Take the K most recent fixtures (already sorted oldest-first)
    recent = all_fixtures[-K:] if len(all_fixtures) >= K else all_fixtures

    # Build step vectors
    steps = [_build_step_vector(was_home, stats) for _, was_home, stats in recent]

    # Zero-pad beginning
    n_real = len(steps)
    seq = np.zeros((K, SEQ_INPUT_DIM), dtype=np.float32)
    if n_real > 0:
        seq[K - n_real:] = np.stack(steps, axis=0)

    return seq


if __name__ == "__main__":
    # Quick test with a known team
    import sys
    season_dir = DATA_ROOT / "2025-2026_25580"
    from data_layer import get_team_list
    teams = get_team_list(season_dir) if season_dir.exists() else []
    if teams:
        name, tid = teams[0]
        print(f"Building sequence for {name} (id={tid}) ...")
        seq = build_team_sequence(tid)
        print(f"  shape={seq.shape}  mean={seq.mean():.4f}  nonzero={np.count_nonzero(seq)}")
