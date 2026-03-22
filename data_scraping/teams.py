# teams.py
import os
import re
import json
import time
import shutil
import tempfile
from typing import Dict, Set, Tuple, List, Optional

import requests  # pip install requests

try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()  # loads .env from current working directory
except Exception:
    pass

SEASON_DIR_RE = re.compile(r"^(?P<y1>\d{4})-(?P<y2>\d{4})_(?P<season_id>\d+)$")
INVALID_CHARS_RE = re.compile(r'[<>:"/\\|?*]')  # Windows-illegal; safe across OSes
WS_RE = re.compile(r'\s+')


# -------------------------------
# Season / naming helpers
# -------------------------------
def is_valid_season_dir(name: str) -> bool:
    """Accept season directories named like: yyyy-[yyyy+1]_<season_id>, e.g., 2024-2025_23619."""
    m = SEASON_DIR_RE.match(name)
    if not m:
        return False
    y1 = int(m.group("y1"))
    y2 = int(m.group("y2"))
    return y2 == y1 + 1


def season_id_from_dir(name: str) -> Optional[int]:
    """Extract season_id from 'yyyy-[yyyy+1]_<season_id>' folder name."""
    m = SEASON_DIR_RE.match(name)
    if not m:
        return None
    try:
        return int(m.group("season_id"))
    except Exception:
        return None


def read_season_finished_flag(season_path: str, default_if_missing: bool = False) -> bool:
    """
    Read <season_path>/data/data.json and return the boolean 'finished'.
    If the file is missing/corrupt, return default_if_missing (False by default).
    """
    info_path = os.path.join(season_path, "data", "data.json")
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        finished = obj.get("finished")
        if isinstance(finished, bool):
            return finished
    except Exception:
        pass
    return default_if_missing


def sanitize_dir_name(s: str) -> str:
    """
    Normalize folder names:
      - keep diacritics
      - replace any whitespace with '-'
      - replace OS-illegal characters with '_'
      - collapse sequences of '-' and '_'
      - trim leading/trailing spaces, dots, hyphens, underscores
    """
    s = str(s)
    s = WS_RE.sub("-", s)               # whitespace -> hyphen
    s = INVALID_CHARS_RE.sub("_", s)    # illegal -> underscore
    s = re.sub(r'[-_]{2,}', lambda m: "-" if "-" in m.group(0) else "_", s)
    s = s.strip(" .-_")
    if not s:
        s = "_"
    return s


# -------------------------------
# Migration: spaces -> hyphens
# -------------------------------
def migrate_team_dirs_spaces_to_hyphens(
    data_root: str,
    dry_run: bool = True,
    verbose: bool = True
) -> List[Tuple[str, str]]:
    """
    Rename existing <season>/teams/<Team Name>_<id> folders so that whitespace in the
    <Team Name> part is replaced with hyphens (and normalized by sanitize_dir_name()).

    If the target (hyphenated) folder already exists, the old folder is REMOVED.
    Otherwise, the old folder is RENAMED to the target.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"[migrate] data_root does not exist or is not a directory: {data_root}")

    changes: List[Tuple[str, str]] = []

    for season_entry in os.scandir(data_root):
        if not season_entry.is_dir() or not is_valid_season_dir(season_entry.name):
            continue

        teams_dir = os.path.join(season_entry.path, "teams")
        if not os.path.isdir(teams_dir):
            continue

        if verbose:
            print(f"[{season_entry.name}] scanning teams/ for migration...")

        for t in os.scandir(teams_dir):
            if not t.is_dir():
                continue

            old_name = t.name
            # Split "<name>_<id>" keeping the last underscore as separator
            name_part, sep, suffix = old_name.rpartition("_")
            if sep and suffix.isdigit():
                new_name_part = sanitize_dir_name(name_part)  # spaces -> hyphens, etc.
                new_name = f"{new_name_part}_{suffix}"
            else:
                new_name = sanitize_dir_name(old_name)

            if new_name == old_name:
                continue  # nothing to do

            old_path = os.path.join(teams_dir, old_name)
            new_path = os.path.join(teams_dir, new_name)

            if os.path.exists(new_path):
                changes.append((old_path, new_path))
                if dry_run:
                    if verbose:
                        print(f"[WOULD DELETE] {old_name} (target exists: '{new_name}')")
                else:
                    try:
                        shutil.rmtree(old_path)
                        if verbose:
                            print(f"[DELETED] {old_name} (kept existing '{new_name}')")
                    except Exception as e:
                        if verbose:
                            print(f"[WARN] delete failed '{old_path}': {e}")
                continue

            changes.append((old_path, new_path))
            if dry_run:
                if verbose:
                    print(f"[WOULD RENAME] {old_name} → {new_name}")
            else:
                try:
                    os.rename(old_path, new_path)
                    if verbose:
                        print(f"[RENAMED] {old_name} → {new_name}")
                except Exception as e:
                    if verbose:
                        print(f"[WARN] rename failed '{old_path}' → '{new_path}': {e}")

    if verbose:
        print(f"Migration summary: {len(changes)} item(s) {'to process' if dry_run else 'processed'}.")

    return changes


# -------------------------------
# Extract teams from fixtures
# -------------------------------
def extract_teams_from_match_obj(match_obj: Dict) -> List[Tuple[int, str]]:
    """Return a list of (team_id, team_name) found in 'participants' of a single match object."""
    out: List[Tuple[int, str]] = []
    participants = match_obj.get("participants")
    if isinstance(participants, list):
        for p in participants:
            if not isinstance(p, dict):
                continue
            tid = p.get("id")
            tname = p.get("name")
            if tid is None or tname is None:
                continue
            try:
                tid = int(tid)
            except Exception:
                continue
            out.append((tid, str(tname)))
    return out


def extract_teams_from_data_json(path: str) -> Set[Tuple[int, str]]:
    """
    Open a fixtures data.json and collect all unique (team_id, team_name) pairs.
    Accepts both shapes:
      - {"data": {...}}     # a single match object
      - {"data": [...]}     # a list of match objects
    """
    teams: Set[Tuple[int, str]] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return teams  # skip missing/corrupt files

    data = payload.get("data")
    if isinstance(data, dict):
        for tid, tname in extract_teams_from_match_obj(data):
            teams.add((tid, tname))
    elif isinstance(data, list):
        for match_obj in data:
            if not isinstance(match_obj, dict):
                continue
            for tid, tname in extract_teams_from_match_obj(match_obj):
                teams.add((tid, tname))
    return teams


def sync_teams_folders_from_fixtures(data_root: str, verbose: bool = True) -> Dict:
    """
    Walk all season folders under DATA_ROOT that match yyyy-[yyyy+1]_<season_id>.
    For each season: scan fixtures/**/data.json, extract teams, and create
    <season>/teams/[team-name]_[team_id] folders (idempotent).
    """
    seasons_processed = 0
    total_created = 0
    per_season: Dict[str, Dict[str, int]] = {}

    for entry in os.scandir(data_root):
        if not entry.is_dir():
            continue
        season_name = entry.name
        if not is_valid_season_dir(season_name):
            continue

        season_dir = entry.path
        fixtures_dir = os.path.join(season_dir, "fixtures")
        if not os.path.isdir(fixtures_dir):
            continue

        seasons_processed += 1
        if verbose:
            print(f"[{season_name}] scanning fixtures...")

        teams_dir = os.path.join(season_dir, "teams")
        os.makedirs(teams_dir, exist_ok=True)

        # Collect season's teams to avoid redundant mkdir calls
        season_teams: Set[Tuple[int, str]] = set()
        for root, _, files in os.walk(fixtures_dir):
            if "data.json" in files:
                data_json_path = os.path.join(root, "data.json")
                season_teams |= extract_teams_from_data_json(data_json_path)

        created_here = 0
        for tid, tname in sorted(season_teams, key=lambda x: (x[1].lower(), x[0])):
            safe_name = sanitize_dir_name(tname)
            dirname = f"{safe_name}_{tid}"
            target = os.path.join(teams_dir, dirname)

            existed = os.path.isdir(target)
            try:
                os.makedirs(target, exist_ok=True)
            except Exception as e:
                if verbose:
                    print(f"[WARN] cannot create '{target}': {e}")
                continue

            if not existed:
                created_here += 1

        total_created += created_here
        per_season[season_name] = {
            "teams_found": len(season_teams),
            "folders_created": created_here,
        }
        if verbose:
            print(f"[{season_name}] teams_found={len(season_teams)}, created={created_here}")

    return {
        "seasons_processed": seasons_processed,
        "total_folders_created": total_created,
        "per_season": per_season,
    }


# -------------------------------
# HTTP client with rate-limit safety
# -------------------------------
class SportmonksClient:
    """
    Minimal client with backoff:
      - sleeps 'request_delay' between calls
      - on 429, waits Retry-After if present; otherwise sleeps 'hard_rate_sleep'
      - retries 5xx with a short backoff
    """
    def __init__(
        self,
        api_token: str,
        api_base: str = "https://api.sportmonks.com",
        request_delay: float = 1.3,
        hard_rate_sleep: int = 3605,
        verbose: bool = True,
    ):
        if not api_token:
            raise ValueError("SPORTMONKS_TOKEN is required.")
        self.api_token = api_token
        self.api_base = api_base.rstrip("/")
        self.request_delay = request_delay
        self.hard_rate_sleep = hard_rate_sleep
        self.verbose = verbose
        self.session = requests.Session()

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    def _get(self, url: str, params: Optional[Dict] = None) -> Dict:
        params = dict(params or {})
        params["api_token"] = self.api_token

        while True:
            resp = self.session.get(url, params=params, timeout=60)
            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    wait_s = int(retry_after)
                    self._log(f"[429] Retry-After={wait_s}s → sleeping...")
                    time.sleep(wait_s + 1)
                else:
                    self._log(f"[429] No Retry-After → sleeping {self.hard_rate_sleep}s...")
                    time.sleep(self.hard_rate_sleep)
                continue

            if 500 <= resp.status_code < 600:
                self._log(f"[{resp.status_code}] Server error → retry in 5s...")
                time.sleep(5)
                continue

            if not resp.ok:
                raise RuntimeError(f"HTTP {resp.status_code} for {url} → {resp.text[:400]}")

            data = resp.json()
            if isinstance(data, dict) and isinstance(data.get("message"), str) and "rate limit" in data["message"].lower():
                self._log(f"[API msg] 'rate limit' in payload → sleeping {self.hard_rate_sleep}s...")
                time.sleep(self.hard_rate_sleep)
                continue

            time.sleep(self.request_delay)
            return data

    # --- endpoints you need ---
    def fetch_team_statistics(self, team_id: int, season_id: int) -> Dict:
        """
        GET /v3/football/teams/{team_id}?include=statistics.details.type&filters=teamStatisticSeasons:{season_id}
        """
        url = f"{self.api_base}/v3/football/teams/{team_id}"
        params = {
            "include": "statistics.details.type",
            "filters": f"teamStatisticSeasons:{season_id}",
        }
        return self._get(url, params=params)

    def fetch_team_squad(self, season_id: int, team_id: int) -> Dict:
        """
        GET /v3/football/squads/seasons/{season_id}/teams/{team_id}?include=position;player;details.type
        """
        url = f"{self.api_base}/v3/football/squads/seasons/{season_id}/teams/{team_id}"
        params = {
            "include": "position;player;details.type",
        }
        return self._get(url, params=params)


# -------------------------------
# Atomic write helper
# -------------------------------
def atomic_write_json(path: str, payload: Dict):
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
# Fetch & write statistics.json + squad.json (with season 'finished' logic)
# -------------------------------
def update_teams_payloads(
    data_root: str,
    api_token: Optional[str] = None,
    api_base: str = "https://api.sportmonks.com",
    request_delay: float = 1.3,
    force: bool = False,
    refresh_unfinished: bool = True,
    verbose: bool = True,
) -> Dict:
    """
    For every season folder and every <season>/teams/<Team-Name>_<team_id> folder, write:
      - statistics.json (team statistics for that specific season)
      - squad.json (squad for that specific season & team)

    Rules:
      - If file does NOT exist → fetch and write.
      - If file exists:
          - If season 'finished' is False and refresh_unfinished=True → fetch and overwrite.
          - If season 'finished' is True → skip re-fetch.
      - 'force=True' overrides and fetches regardless of existing files or 'finished' flag.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"[update] data_root does not exist or is not a directory: {data_root}")

    token = api_token or os.environ.get("SPORTMONKS_API_TOKEN", "")
    if not token:
        raise ValueError("SPORTMONKS_API_TOKEN not set. Set env var or pass api_token.")

    client = SportmonksClient(
        api_token=token,
        api_base=api_base,
        request_delay=request_delay,
        hard_rate_sleep=3605,
        verbose=verbose,
    )

    seasons_processed = 0
    teams_seen = 0
    written_stats = 0
    written_squad = 0
    skipped_stats = 0
    skipped_squad = 0
    refreshed_stats = 0
    refreshed_squad = 0

    for season_entry in os.scandir(data_root):
        if not season_entry.is_dir() or not is_valid_season_dir(season_entry.name):
            continue

        season_id = season_id_from_dir(season_entry.name)
        if season_id is None:
            continue

        season_finished = read_season_finished_flag(season_entry.path, default_if_missing=False)

        teams_dir = os.path.join(season_entry.path, "teams")
        if not os.path.isdir(teams_dir):
            continue

        seasons_processed += 1
        if verbose:
            print(f"[{season_entry.name}] updating teams payloads... (finished={season_finished})")

        for t in os.scandir(teams_dir):
            if not t.is_dir():
                continue

            folder_name = t.name
            name_part, sep, suffix = folder_name.rpartition("_")
            if not (sep and suffix.isdigit()):
                if verbose:
                    print(f"[WARN] cannot parse team_id from folder '{folder_name}'")
                continue

            team_id = int(suffix)
            teams_seen += 1

            # paths
            stats_path = os.path.join(t.path, "statistics.json")
            squad_path = os.path.join(t.path, "squad.json")

            # --- statistics.json ---
            stats_exists = os.path.exists(stats_path)
            should_fetch_stats = (
                force
                or (not stats_exists)
                or (refresh_unfinished and not season_finished and stats_exists)
            )
            if should_fetch_stats:
                try:
                    payload = client.fetch_team_statistics(team_id=team_id, season_id=season_id)
                    atomic_write_json(stats_path, payload)
                    if stats_exists and not force:
                        refreshed_stats += 1
                    else:
                        written_stats += 1
                    if verbose:
                        print(f"[OK] statistics.json ← team {team_id} (season {season_id})")
                except Exception as e:
                    if verbose:
                        print(f"[WARN] statistics fetch/write failed (team {team_id}, season {season_id}): {e}")
            else:
                skipped_stats += 1

            # --- squad.json ---
            squad_exists = os.path.exists(squad_path)
            should_fetch_squad = (
                force
                or (not squad_exists)
                or (refresh_unfinished and not season_finished and squad_exists)
            )
            if should_fetch_squad:
                try:
                    payload = client.fetch_team_squad(season_id=season_id, team_id=team_id)
                    atomic_write_json(squad_path, payload)
                    if squad_exists and not force:
                        refreshed_squad += 1
                    else:
                        written_squad += 1
                    if verbose:
                        print(f"[OK] squad.json ← team {team_id} (season {season_id})")
                except Exception as e:
                    if verbose:
                        print(f"[WARN] squad fetch/write failed (team {team_id}, season {season_id}): {e}")
            else:
                skipped_squad += 1

    return {
        "seasons_processed": seasons_processed,
        "teams_seen": teams_seen,
        "written": {
            "statistics.json": written_stats,
            "squad.json": written_squad,
        },
        "refreshed_existing_unfinished": {
            "statistics.json": refreshed_stats,
            "squad.json": refreshed_squad,
        },
        "skipped_existing_finished": {
            "statistics.json": skipped_stats,
            "squad.json": skipped_squad,
        },
    }


# --- Minimal entrypoint (no argparse, no prints) ---
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "data")

if __name__ == "__main__":
    # Call the sync function and that's it.
    sync_teams_folders_from_fixtures(DEFAULT_DATA_ROOT)
    #changes = migrate_team_dirs_spaces_to_hyphens(DEFAULT_DATA_ROOT, dry_run=False, verbose=True)
    # Single, minimal call as requested
    update_teams_payloads(DEFAULT_DATA_ROOT)

