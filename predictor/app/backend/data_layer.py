# data_layer.py
# Season/team/roster discovery from the data/ directory.
#
# Public API:
#   get_current_season_dir(data_root)   -> Path | None
#   get_team_list(season_dir)           -> list[tuple[str, int]]
#   get_team_roster(team_id, season_dir) -> dict[str, list[tuple[str, int]]]
#     position code -> [(player_display_name, player_id), ...]

import os
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_REFACTOR_DIR = Path(__file__).resolve().parent.parent.parent   # licenta/refactor/
if str(_REFACTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_REFACTOR_DIR))

from shared_config import WORKSPACE_ROOT

_SEASON_RE = re.compile(r"^(?P<y1>\d{4})-(?P<y2>\d{4})_(?P<sid>\d+)$")
_FIXTURE_RE = re.compile(r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_\d+$")

POSITION_ID_TO_CODE: Dict[int, str] = {24: "GK", 25: "DF", 26: "MF", 27: "ATK"}
POSITION_CODES = ["GK", "DF", "MF", "ATK"]

DATA_ROOT: Path = WORKSPACE_ROOT / "data"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _season_sort_key(name: str) -> Tuple[int, int, int]:
    m = _SEASON_RE.match(name)
    if not m:
        return (0, 0, 0)
    return (int(m.group("y1")), int(m.group("y2")), int(m.group("sid")))


def _is_valid_season(name: str) -> bool:
    m = _SEASON_RE.match(name)
    if not m:
        return False
    return int(m.group("y2")) == int(m.group("y1")) + 1


def _fixture_ts(dirname: str) -> Optional[datetime]:
    m = _FIXTURE_RE.match(dirname)
    if not m:
        return None
    try:
        return datetime.strptime(m.group("ts"), "%Y-%m-%dT%H-%M-%S")
    except ValueError:
        return None


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_current_season_dir(data_root: Path = DATA_ROOT) -> Optional[Path]:
    """Return the Path to the most recent valid season directory."""
    if not data_root.is_dir():
        return None
    seasons = [
        e.name for e in data_root.iterdir()
        if e.is_dir() and _is_valid_season(e.name)
    ]
    if not seasons:
        return None
    latest = max(seasons, key=_season_sort_key)
    return data_root / latest


def get_team_list(season_dir: Path) -> List[Tuple[str, int]]:
    """
    Scan <season>/teams/ and return [(display_name, team_id), ...] sorted by name.
    Folder naming convention: '{TeamName}_{team_id}/'.
    """
    teams_dir = season_dir / "teams"
    if not teams_dir.is_dir():
        return []
    result: List[Tuple[str, int]] = []
    for entry in teams_dir.iterdir():
        if not entry.is_dir():
            continue
        name, _, sid = entry.name.rpartition("_")
        if not sid.isdigit():
            continue
        display = name.replace("-", " ")
        result.append((display, int(sid)))
    return sorted(result, key=lambda x: x[0].lower())


def get_team_roster(
    team_id: int,
    season_dir: Path,
) -> Dict[str, List[Tuple[str, int]]]:
    """
    Scan all lineup.json files in <season>/fixtures/ for the given team_id.
    Returns only type_id==11 (starting lineup) players, grouped by position:
        {"GK": [(name, player_id), ...], "DF": [...], "MF": [...], "ATK": [...]}
    Player name is taken from the lineup entry's player_name field.
    """
    roster: Dict[str, Dict[int, str]] = {pos: {} for pos in POSITION_CODES}
    fixtures_dir = season_dir / "fixtures"
    if not fixtures_dir.is_dir():
        return {pos: [] for pos in POSITION_CODES}

    for fix_entry in fixtures_dir.iterdir():
        if not fix_entry.is_dir():
            continue
        lineup_path = fix_entry / "lineup.json"
        if not lineup_path.exists():
            continue
        payload = _read_json(str(lineup_path))
        if not payload:
            continue

        lineups = None
        d = payload.get("data")
        if isinstance(d, dict):
            lineups = d.get("lineups") or d.get("lineup")
        if lineups is None:
            lineups = payload.get("lineups") or payload.get("lineup") or []

        if not isinstance(lineups, list):
            continue

        for entry in lineups:
            if not isinstance(entry, dict):
                continue
            try:
                t_id = int(entry["team_id"])
                type_id = int(entry["type_id"])
                pos_id = int(entry["position_id"])
                p_id = int(entry["player_id"])
            except (KeyError, TypeError, ValueError):
                continue
            if t_id != team_id:
                continue
            if type_id != 11:
                continue
            pos_code = POSITION_ID_TO_CODE.get(pos_id)
            if pos_code is None:
                continue
            p_name = str(entry.get("player_name") or f"Player_{p_id}")
            if p_id not in roster[pos_code]:
                roster[pos_code][p_id] = p_name

    return {
        pos: sorted([(name, pid) for pid, name in players.items()], key=lambda x: x[0])
        for pos, players in roster.items()
    }


def get_all_season_dirs(data_root: Path = DATA_ROOT) -> List[Path]:
    """Return all valid season directories in chronological order."""
    if not data_root.is_dir():
        return []
    seasons = [
        e.name for e in data_root.iterdir()
        if e.is_dir() and _is_valid_season(e.name)
    ]
    return [data_root / s for s in sorted(seasons, key=_season_sort_key)]


def get_jersey_numbers(season_dir: Path) -> Dict[int, int]:
    """
    Scan all players in <season>/players/ and return a mapping of
    player_id -> jersey_number from each player's current_statistics.json.
    """
    result: Dict[int, int] = {}
    players_dir = season_dir / "players"
    if not players_dir.is_dir():
        return result
    for folder in players_dir.iterdir():
        if not folder.is_dir():
            continue
        parts = folder.name.rsplit("_", 1)
        if len(parts) != 2 or not parts[1].isdigit():
            continue
        pid = int(parts[1])
        stats_path = folder / "current_statistics.json"
        if not stats_path.exists():
            continue
        data = _read_json(str(stats_path))
        if not data:
            continue
        if isinstance(data, list):
            data = data[0] if data and isinstance(data[0], dict) else None
        if not isinstance(data, dict):
            continue
        jn = data.get("jersey_number")
        if jn is not None:
            try:
                result[pid] = int(jn)
            except (TypeError, ValueError):
                pass
    return result


if __name__ == "__main__":
    season = get_current_season_dir()
    print(f"Current season: {season}")
    if season:
        teams = get_team_list(season)
        print(f"Teams ({len(teams)}):")
        for name, tid in teams[:5]:
            print(f"  {tid}: {name}")
        if teams:
            roster = get_team_roster(teams[0][1], season)
            print(f"\nRoster for {teams[0][0]}:")
            for pos, players in roster.items():
                print(f"  {pos}: {[p[0] for p in players[:3]]}")
