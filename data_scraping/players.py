# players.py
import os
import re
import json
import time
import tempfile
from typing import Dict, List, Optional, Set, Tuple, Union, Any

# Optional: load .env automatically if present (so SPORTMONKS_API_TOKEN is picked up)
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

import requests  # pip install requests

SEASON_DIR_RE = re.compile(r"^(?P<y1>\d{4})-(?P<y2>\d{4})_(?P<season_id>\d+)$")
INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*]')  # Windows-illegal; safe across OSes
WS_RE = re.compile(r"\s+")


# -------------------------------
# Season / naming helpers
# -------------------------------
def is_valid_season_dir(name: str) -> bool:
    """
    Accept season directories named like: yyyy-[yyyy+1]_<season_id>, e.g., 2024-2025_23619.
    """
    m = SEASON_DIR_RE.match(name)
    if not m:
        return False
    y1 = int(m.group("y1"))
    y2 = int(m.group("y2"))
    return y2 == y1 + 1


def sanitize_dir_name(s: str) -> str:
    """
    Match teams.py behavior:
      - keep diacritics
      - replace any whitespace with '-'
      - replace OS-illegal characters with '_'
      - collapse sequences of '-' and '_'
      - trim leading/trailing spaces, dots, hyphens, underscores
    """
    s = str(s)
    s = WS_RE.sub("-", s)               # whitespace -> hyphen
    s = INVALID_CHARS_RE.sub("_", s)    # illegal -> underscore
    s = re.sub(r"[-_]{2,}", lambda m: "-" if "-" in m.group(0) else "_", s)
    s = s.strip(" .-_")
    if not s:
        s = "_"
    return s


def atomic_write_json(path: str, payload: Union[Dict, List]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


# -------------------------------
# Players discovery helpers (current_statistics.json)
# -------------------------------
def extract_players_from_squad_json(path: str) -> List[Tuple[int, str, Dict]]:
    """
    Read a teams/*/squad.json and return a list of:
      (player_id, player_name_from_player.name, full_entry_object)
    IMPORTANT: player_name is taken STRICTLY from player['name'].
    The full_entry_object is copied EXACTLY as in squad.json (no mutations).
    """
    out: List[Tuple[int, str, Dict]] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return out

    data = payload.get("data")
    if not isinstance(data, list):
        return out

    for entry in data:
        if not isinstance(entry, dict):
            continue
        player = entry.get("player")
        if not isinstance(player, dict):
            continue

        pid = player.get("id", entry.get("player_id"))
        try:
            pid = int(pid)
        except Exception:
            continue

        pname = player.get("name")
        if not isinstance(pname, str) or not pname.strip():
            # strict requirement: must come from player['name']
            continue

        out.append((pid, pname, entry))
    return out


def parse_existing_player_ids(players_dir: str) -> Tuple[Set[int], Dict[int, str]]:
    """
    Scan an existing <season>/players directory and return:
      - a set of player_ids already present
      - a map player_id -> folder_path
    """
    existing_ids: Set[int] = set()
    id_to_folder: Dict[int, str] = {}
    if not os.path.isdir(players_dir):
        return existing_ids, id_to_folder

    for entry in os.scandir(players_dir):
        if not entry.is_dir():
            continue
        name = entry.name
        name_part, sep, suffix = name.rpartition("_")
        if sep and suffix.isdigit():
            pid = int(suffix)
            existing_ids.add(pid)
            id_to_folder[pid] = entry.path
    return existing_ids, id_to_folder


def load_current_statistics(path: str) -> Optional[Union[Dict, List]]:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def json_equal(a: Union[Dict, List], b: Union[Dict, List]) -> bool:
    # structural equality via dumps with sorted keys
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == json.dumps(b, sort_keys=True, ensure_ascii=False)


# -------------------------------
# Main: write current_statistics.json
# -------------------------------
def write_current_statistics_from_squads(data_root: str, verbose: bool = True) -> Dict:
    """
    For each season (yyyy-[yyyy+1]_<season_id>):
      - scan <season>/teams/*/squad.json
      - for each player entry in data[], ensure <season>/players/<player-name>_<player_id>/ exists
      - write / update <season>/players/.../current_statistics.json by copying the EXACT entry object

    Behavior when current_statistics.json already exists:
      - if equal to the new entry → skip
      - if different:
          * if file is a single object → convert to list [old, new]
          * if file is a list → append new only if not already present (by structural equality)
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"[players] DATA_ROOT does not exist or is not a directory: {data_root}")

    seasons_processed = 0
    players_touched = 0
    written_new = 0
    appended_new = 0
    skipped_equal = 0
    create_errors = 0

    for season_entry in os.scandir(data_root):
        if not season_entry.is_dir() or not is_valid_season_dir(season_entry.name):
            continue

        season_dir = season_entry.path
        teams_dir = os.path.join(season_dir, "teams")
        if not os.path.isdir(teams_dir):
            continue

        players_dir = os.path.join(season_dir, "players")
        os.makedirs(players_dir, exist_ok=True)

        seasons_processed += 1
        if verbose:
            print(f"[{season_entry.name}] writing current_statistics.json from squads...")

        # Build quick index of existing player folders
        existing_ids, id_to_folder = parse_existing_player_ids(players_dir)

        for team_entry in os.scandir(teams_dir):
            if not team_entry.is_dir():
                continue
            squad_json = os.path.join(team_entry.path, "squad.json")
            if not os.path.isfile(squad_json):
                continue

            triples = extract_players_from_squad_json(squad_json)
            if not triples:
                continue

            for pid, pname, entry_obj in triples:
                players_touched += 1

                # ensure player folder path
                folder_path = id_to_folder.get(pid)
                if folder_path is None:
                    safe_name = sanitize_dir_name(pname)
                    dirname = f"{safe_name}_{pid}"
                    folder_path = os.path.join(players_dir, dirname)
                    try:
                        os.makedirs(folder_path, exist_ok=True)
                        existing_ids.add(pid)
                        id_to_folder[pid] = folder_path
                    except Exception as e:
                        create_errors += 1
                        if verbose:
                            print(f"[WARN] cannot create '{folder_path}': {e}")
                        continue

                # write/update current_statistics.json
                stats_path = os.path.join(folder_path, "current_statistics.json")
                existing = load_current_statistics(stats_path)

                if existing is None:
                    # brand new file: write as the exact object
                    try:
                        atomic_write_json(stats_path, entry_obj)
                        written_new += 1
                    except Exception as e:
                        create_errors += 1
                        if verbose:
                            print(f"[WARN] write failed '{stats_path}': {e}")
                    continue

                # file exists: compare / merge
                if isinstance(existing, list):
                    # append only if not already present
                    if any(json_equal(entry_obj, x) for x in existing):
                        skipped_equal += 1
                    else:
                        existing.append(entry_obj)
                        try:
                            atomic_write_json(stats_path, existing)
                            appended_new += 1
                        except Exception as e:
                            create_errors += 1
                            if verbose:
                                print(f"[WARN] append failed '{stats_path}': {e}")
                elif isinstance(existing, dict):
                    if json_equal(existing, entry_obj):
                        skipped_equal += 1
                    else:
                        new_payload = [existing, entry_obj]
                        try:
                            atomic_write_json(stats_path, new_payload)
                            appended_new += 1
                        except Exception as e:
                            create_errors += 1
                            if verbose:
                                print(f"[WARN] upgrade-to-list failed '{stats_path}': {e}")
                else:
                    # unexpected type: overwrite with the exact object to recover
                    try:
                        atomic_write_json(stats_path, entry_obj)
                        written_new += 1
                    except Exception as e:
                        create_errors += 1
                        if verbose:
                            print(f"[WARN] overwrite (unexpected type) failed '{stats_path}': {e}")

    return {
        "seasons_processed": seasons_processed,
        "players_touched": players_touched,
        "written_new_files": written_new,
        "appended_entries": appended_new,
        "skipped_equal": skipped_equal,
        "create_or_write_errors": create_errors,
    }


# -------------------------------
# Helpers for previous season resolution
# -------------------------------
def parse_season_parts(name: str) -> Optional[Tuple[int, int, int]]:
    """
    Return (y1, y2, season_id) if name matches 'yyyy-[yyyy+1]_<season_id>', else None.
    """
    m = SEASON_DIR_RE.match(name)
    if not m:
        return None
    try:
        return int(m.group("y1")), int(m.group("y2")), int(m.group("season_id"))
    except Exception:
        return None


def find_previous_season_id(data_root: str, current_season_folder: str) -> Optional[int]:
    """
    Given the current season folder name 'yyyy-[yyyy+1]_<season_id>',
    find sibling folder for previous year '(y1-1)-(y2-1)_{season_id_prev}' and return season_id_prev.
    """
    parts = parse_season_parts(current_season_folder)
    if not parts:
        return None
    y1, y2, _sid_cur = parts
    prev_y1, prev_y2 = y1 - 1, y2 - 1
    pat = re.compile(rf"^{prev_y1}-{prev_y2}_(\d+)$")
    try:
        for entry in os.scandir(data_root):
            if not entry.is_dir():
                continue
            m = pat.match(entry.name)
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    continue
    except FileNotFoundError:
        return None
    return None


# -------------------------------
# Robust HTTP client for last_year_statistics.json
# -------------------------------
class SportmonksClientPlayers:
    """
    Focused client for player last-season stats.
    Reliable long-run behavior:
      - delay between calls
      - retries on 429/5xx/timeouts
      - respects Retry-After when present
      - logs and skips on non-429 4xx
    """
    def __init__(
        self,
        api_token: Optional[str] = None,
        api_base: str = "https://api.sportmonks.com",
        request_delay: float = 1.4,
        hard_rate_sleep: int = 3605,
        verbose: bool = True,
    ):
        tok = (
            api_token
            or os.environ.get("SPORTMONKS_API_TOKEN")
            or os.environ.get("SPORTMONKS_TOKEN")
        )
        if not tok:
            raise ValueError("SPORTMONKS_API_TOKEN (or SPORTMONKS_TOKEN) not set.")
        self.api_token = tok
        self.api_base = api_base.rstrip("/")
        self.request_delay = request_delay
        self.hard_rate_sleep = hard_rate_sleep
        self.verbose = verbose
        self.session = requests.Session()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _robust_get(self, url: str, params: Dict[str, Any]) -> Optional[Dict]:
        """
        Robust GET loop for long runs:
          - infinite retries for 429 (rate-limit), 5xx, and network errors with backoff
          - applies per-call delay
          - on other 4xx (401/403/404/422, etc.), logs and returns None (skip)
        """
        params = dict(params)
        params["api_token"] = self.api_token

        backoff = 5
        while True:
            try:
                resp = self.session.get(url, params=params, timeout=60)
            except requests.RequestException as e:
                self._log(f"[NET] {e} → sleep {backoff}s and retry")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue

            # Rate limited
            if resp.status_code == 429:
                ra = resp.headers.get("Retry-After")
                if ra:
                    try:
                        wait_s = int(ra)
                    except Exception:
                        wait_s = self.hard_rate_sleep
                    self._log(f"[429] Retry-After={wait_s}s")
                    time.sleep(wait_s + 1)
                else:
                    self._log(f"[429] No Retry-After → sleep {self.hard_rate_sleep}s")
                    time.sleep(self.hard_rate_sleep)
                continue

            # Transient server errors
            if 500 <= resp.status_code < 600:
                self._log(f"[{resp.status_code}] server error → sleep {backoff}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue

            # Other client errors (fatal for our purposes): log and skip
            if not resp.ok:
                self._log(f"[SKIP] HTTP {resp.status_code} for {url} → {resp.text[:200]}")
                return None

            try:
                data = resp.json()
            except ValueError:
                self._log("[WARN] JSON decode failed, retry in 5s")
                time.sleep(5)
                continue

            # Soft rate limit message embedded in JSON
            if isinstance(data, dict) and isinstance(data.get("message"), str) and "rate limit" in data["message"].lower():
                self._log(f"[API msg] 'rate limit' in payload → sleep {self.hard_rate_sleep}s")
                time.sleep(self.hard_rate_sleep)
                continue

            # Normal successful path
            time.sleep(self.request_delay)
            return data

    def fetch_player_last_season_stats(self, player_id: int, season_id_prev: int) -> Optional[Dict]:
        """
        GET /v3/football/players/{player_id}
            ?include=statistics.position;statistics.details.type
            &filters=playerStatisticSeasons:{season_id_prev}
        """
        url = f"{self.api_base}/v3/football/players/{player_id}"
        params = {
            "include": "statistics.position;statistics.details.type",
            "filters": f"playerStatisticSeasons:{season_id_prev}",
        }
        return self._robust_get(url, params)


# -------------------------------
# Main: write last_year_statistics.json
# -------------------------------
def write_last_year_statistics(
    data_root: str,
    api_token: Optional[str] = None,
    request_delay: float = 1.3,
    verbose: bool = True,
) -> Dict:
    """
    For every season (yyyy-[yyyy+1]_<season_id_cur>):
      - find previous season's ID by sibling folder '(y1-1)-(y2-1)_{season_id_prev}'
      - for every player folder in <season_cur>/players/[name]_[player_id], if
        last_year_statistics.json is missing, call the API and save the exact response.

    No 'force' mode by design. Existing files are skipped.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"[players] DATA_ROOT does not exist or is not a directory: {data_root}")

    client = SportmonksClientPlayers(
        api_token=api_token,
        request_delay=request_delay,
        verbose=verbose,
    )

    seasons_processed = 0
    players_seen = 0
    written = 0
    skipped_existing = 0
    skipped_no_prev = 0
    api_skips = 0  # non-OK 4xx like 401/403/404 etc.

    for season_entry in os.scandir(data_root):
        if not season_entry.is_dir() or not is_valid_season_dir(season_entry.name):
            continue

        prev_season_id = find_previous_season_id(data_root, season_entry.name)
        if prev_season_id is None:
            if verbose:
                print(f"[{season_entry.name}] previous season folder not found → skip this season")
            skipped_no_prev += 1
            continue

        players_dir = os.path.join(season_entry.path, "players")
        if not os.path.isdir(players_dir):
            continue

        seasons_processed += 1
        if verbose:
            print(f"[{season_entry.name}] prev season id = {prev_season_id} → fetching last_year_statistics...")

        for p in os.scandir(players_dir):
            if not p.is_dir():
                continue
            # parse player_id from '[name]_[id]'
            name_part, sep, suffix = p.name.rpartition("_")
            if not (sep and suffix.isdigit()):
                if verbose:
                    print(f"[WARN] cannot parse player_id from '{p.name}'")
                continue

            player_id = int(suffix)
            # Use the folder's name prefix as the player's display name in logs.
            # Replace hyphens with spaces to make it easier to read.
            player_name = name_part.replace("-", " ")

            players_seen += 1

            out_path = os.path.join(p.path, "last_year_statistics.json")
            if os.path.exists(out_path):
                skipped_existing += 1
                continue

            # Call API
            payload = client.fetch_player_last_season_stats(player_id, prev_season_id)
            if payload is None:
                api_skips += 1
                continue

            # Write exact response
            try:
                atomic_write_json(out_path, payload)
                written += 1
                if verbose:
                    print(f"[OK] last_year_statistics.json ← [{player_name}] {player_id} (prev season {prev_season_id})")
            except Exception as e:
                if verbose:
                    print(f"[WARN] write failed '{out_path}': {e}")

    return {
        "seasons_processed": seasons_processed,
        "players_seen": players_seen,
        "written_files": written,
        "skipped_existing_files": skipped_existing,
        "skipped_no_prev_season": skipped_no_prev,
        "api_skips_non_ok": api_skips,
    }


# --- Minimal entrypoint (no argparse, no prints) ---
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "data")

if __name__ == "__main__":
    # Minimal call: generate last_year_statistics.json for all seasons/players
    write_last_year_statistics(DEFAULT_DATA_ROOT)
    # If you need to (re)build current_statistics.json instead, comment the above line and use:
    write_current_statistics_from_squads(DEFAULT_DATA_ROOT)
