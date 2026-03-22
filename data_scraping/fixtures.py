# data_scraping/fixtures.py
from __future__ import annotations

import os
import sys
import json
import time
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import requests
from datetime import datetime, timezone, timedelta

# Load env from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Config (env or sensible defaults) ---
API_TOKEN = os.getenv("SPORTMONKS_API_TOKEN")
BASE_URL = os.getenv("SPORTMONKS_BASE_URL", "https://api.sportmonks.com/v3/football")
DATA_ROOT = os.getenv("DATA_ROOT", "data")
FIXTURES_PER_PAGE = int(os.getenv("FIXTURES_PER_PAGE", "50"))  # tweak in .env if needed
FIXTURES_INCLUDE = os.getenv("FIXTURES_INCLUDE", "")            # keep empty for base fields

FIXTURES_ENDPOINT = f"{BASE_URL.rstrip('/')}/fixtures"
SEASON_DIR_REGEX = re.compile(r"^\d{4}-\d{4}_(\d+)$")  # data/YYYY-YYYY_<id>

SLEEP_BETWEEN_CALLS_SEC = float(os.getenv("SLEEP_BETWEEN_CALLS_SEC", "1.3"))
HARD_RATE_LIMIT_SLEEP_SEC = int(os.getenv("HARD_RATE_LIMIT_SLEEP_SEC", "3600"))

# --- HTTP helper with tiny retry/backoff for 429/5xx ---

def _http_get(url: str, params: Dict[str, Any], max_retries: int = 6) -> Dict[str, Any]:
    """
    Simple + resilient:
      - fixed sleep before each request (keeps us under hourly cap)
      - if 429 (rate limit): sleep for the reset window (from API if available) or 1 hour, then retry
      - if transient 5xx: small incremental backoff
    """
    attempt = 0
    while True:
        attempt += 1

        # Pace every call to avoid hitting the cap again
        time.sleep(SLEEP_BETWEEN_CALLS_SEC)

        resp = requests.get(url, params=params, timeout=45)
 
        if resp.status_code == 200:
            return resp.json()

        if resp.status_code == 429:
            # Try to read reset window from JSON payload
            sleep_s = HARD_RATE_LIMIT_SLEEP_SEC
            try:
                j = resp.json()
                rl = (j.get("rate_limit") or {})
                resets = rl.get("resets_in_seconds")
                if isinstance(resets, (int, float)) and resets > 0:
                    sleep_s = int(resets)
            except Exception:
                pass

            # Log & sleep, then retry
            wake_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + sleep_s))
            print(f"[rate] 429 hit. Sleeping {sleep_s}s (until {wake_at}) then retrying...")
            time.sleep(sleep_s)
            continue

        if resp.status_code in (500, 502, 503, 504) and attempt <= max_retries:
            delay = 5 * attempt
            print(f"[http] {resp.status_code} transient. Sleeping {delay}s then retrying...")
            time.sleep(delay)
            continue

        # Other errors -> raise so you can see what's wrong
        raise RuntimeError(f"HTTP {resp.status_code} for {url} | body: {resp.text[:500]}")

def iter_season_dirs(base_dir: str = DATA_ROOT):
    """Yield (season_id:int, season_dir:Path) for directories matching 'YYYY-YYYY_<id>'."""
    base = Path(base_dir)
    if not base.exists():
        return
    for p in base.iterdir():
        if not p.is_dir():
            continue
        m = SEASON_DIR_REGEX.fullmatch(p.name)
        if not m:
            continue
        season_id = int(m.group(1))
        yield season_id, p

def _load_state_id_from_file(path: Path) -> Optional[int]:
    """
    Read a JSON file and try to extract data.state_id (Sportmonks response shape).
    Returns int or None if not found/unreadable.
    """
    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        data = payload.get("data", {})
        sid = data.get("state_id")
        return int(sid) if isinstance(sid, (int, float)) else None
    except Exception:
        return None


def read_season_finished_flag(season_dir: Path):
    """Return True/False if found, or None if missing/unreadable."""
    data_file = season_dir / "data" / "data.json"
    if not data_file.exists():
        return None
    try:
        with data_file.open("r", encoding="utf-8") as f:
            obj = json.load(f)
        val = obj.get("finished", None)
        return val if isinstance(val, bool) else (None if val is None else bool(val))
    except Exception:
        return None

def fetch_all_fixtures_for_season(season_id: int) -> List[Dict[str, Any]]:
    """Paginate through /fixtures filtered by fixtureSeasons:{season_id} and return ALL fixtures."""
    all_items: List[Dict[str, Any]] = []
    seen: set[int] = set()
    page = 1
    while True:
        params = {
            "api_token": API_TOKEN,
            "filter": f"fixtureSeasons:{season_id}",
            "per_page": FIXTURES_PER_PAGE,
            "page": page,
        }
        if FIXTURES_INCLUDE:
            params["include"] = FIXTURES_INCLUDE
        payload = _http_get(FIXTURES_ENDPOINT, params=params)
        items = payload.get("data") or []
        for it in items:
            fid = it.get("id")
            if fid is None or fid in seen:
                continue
            seen.add(int(fid))
            all_items.append(it)
        pagination = payload.get("pagination") or {}
        if not bool(pagination.get("has_more")):
            break
        page = int(pagination.get("current_page", page)) + 1
    try:
        all_items.sort(key=lambda x: (x.get("starting_at_timestamp") or 0, x.get("id") or 0))
    except Exception:
        pass
    return all_items

def should_fetch_fixtures(season_dir: Path, out_file: Path, force: bool = False) -> bool:
    """
    Fetch if:
      - force=True, or
      - out_file missing, or
      - finished == False
    Else skip.
    """
    if force:
        return True
    if not out_file.exists():
        return True
    finished = read_season_finished_flag(season_dir)
    return finished is False

def write_ucl_fixtures_full_for_all_seasons(
    base_dir: str = DATA_ROOT,
    filename: str = "ucl_fixtures.json",
    force: bool = False,
) -> List[str]:
    """
    For every season dir (data/YYYY-YYYY_<id>), (re)fetch ALL fixtures and write:
        data/YYYY-YYYY_<id>/fixtures/ucl_fixtures.json
    Skip if file exists AND season finished, unless force=True.
    """
    if not API_TOKEN:
        raise RuntimeError("Missing SPORTMONKS_API_TOKEN. Put it in your .env or OS env.")
    written: List[str] = []
    for season_id, season_dir in iter_season_dirs(base_dir):
        fixtures_dir = season_dir / "fixtures"
        fixtures_dir.mkdir(parents=True, exist_ok=True)
        out_file = fixtures_dir / filename
        if not should_fetch_fixtures(season_dir, out_file, force=False):
            continue
        fixtures = fetch_all_fixtures_for_season(season_id)
        with out_file.open("w", encoding="utf-8") as f:
            json.dump({"data": fixtures}, f, ensure_ascii=False, indent=2)
        written.append(str(out_file))
    return written

# Helper: format "starting_at" safely for folder names
def _format_starting_at_for_dir(item: Dict[str, Any]) -> str:
    """
    Prefer 'starting_at' (e.g., '2024-07-31 18:30:00') -> '2024-07-31T18-30-00'.
    Fallback to 'starting_at_timestamp' (UTC). If both missing -> 'unknown-start'.
    """
    s = item.get("starting_at")
    if isinstance(s, str) and s.strip():
        return s.strip().replace(" ", "T").replace(":", "-")
    ts = item.get("starting_at_timestamp")
    if isinstance(ts, (int, float)) and ts:
        try:
            # UTC formatting; payloads sunt în UTC
            return time.strftime("%Y-%m-%dT%H-%M-%S", time.gmtime(int(ts)))
        except Exception:
            pass
    return "unknown-start"

def create_per_fixture_folders(
    base_dir: str = DATA_ROOT,
    fixtures_filename: str = "ucl_fixtures.json",
) -> list[str]:
    """
    For each season (data/YYYY-YYYY_<id>), read fixtures/ucl_fixtures.json
    and create a folder per fixture named: 'YYYY-MM-DDTHH-MM-SS_<fixture_id>'.

    Returns the list of created/ensured directories.
    """
    created: list[str] = []
    for _, season_dir in iter_season_dirs(base_dir):
        fixtures_dir = season_dir / "fixtures"
        json_file = fixtures_dir / fixtures_filename
        if not json_file.exists():
            # skip seasons without fixtures file
            continue

        try:
            with json_file.open("r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            # bad/partial file — skip gracefully
            continue

        items = (payload or {}).get("data") or []
        for it in items:
            fid = it.get("id")
            if fid is None:
                continue
            start = _format_starting_at_for_dir(it)
            folder_name = f"{start}_{int(fid)}"  # ensure numeric id
            target = fixtures_dir / folder_name
            target.mkdir(parents=True, exist_ok=True)
            created.append(str(target))

    return created

# pattern pentru directoarele de fixture: "YYYY-MM-DDTHH-MM-SS_<fixture_id>"
FIXTURE_DIR_REGEX = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_(\d+)$")

def _parse_fixture_dir_entry(dir_name: str) -> Optional[Tuple[datetime, int]]:
    """
    Extract (start_utc, fixture_id) from a fixture folder name like:
      '2024-07-31T18-30-00_19135802'
    Returns None if it doesn't match.
    """
    m = FIXTURE_DIR_REGEX.fullmatch(dir_name)
    if not m:
        return None
    ts_str, fid_str = m.group(1), m.group(2)
    try:
        start_utc = datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S").replace(tzinfo=timezone.utc)
        fixture_id = int(fid_str)
        return start_utc, fixture_id
    except Exception:
        return None

def _fixture_has_happened(start_utc: datetime) -> bool:
    """Return True if the fixture start datetime is in the past (UTC)."""
    now_utc = datetime.now(timezone.utc)
    return start_utc <= now_utc

def _fetch_fixture_with_include(fixture_id: int, include: str) -> dict:
    """
    GET /fixtures/{id}?include=...
    Returns parsed JSON (dict). Raises on HTTP error via _http_get.
    """
    url = f"{BASE_URL.rstrip('/')}/fixtures/{fixture_id}"
    params = {"api_token": API_TOKEN}
    if include:
        params["include"] = include
    return _http_get(url, params)

def populate_fixture_files_for_all_seasons(
    base_dir: str = DATA_ROOT,
    overwrite: bool = False,
) -> list[str]:
    """
    For every season dir (data/YYYY-YYYY_<id>), scan fixtures subfolders
    named 'YYYY-MM-DDTHH-MM-SS_<fixture_id>' and write the 3 files:
      - data.json         (include=participants;scores;formations)
      - lineup.json       (include=lineups)
      - statistics.json   (include=statistics.type)

    Rules:
      - Always fetch for past fixtures.
      - For future fixtures: fetch ONLY if the kick-off is within the next 7 days.
      - If a file exists and overwrite=False:
          * If fixture is in the past AND state_id in file is 1 (or missing) -> refresh (refetch & overwrite)
          * Else keep the file (skip).
    """
    if not API_TOKEN:
        raise RuntimeError("Missing SPORTMONKS_API_TOKEN. Put it in your .env or OS env.")

    written: list[str] = []
    now_utc = datetime.now(timezone.utc)
    week_ahead = now_utc + timedelta(days=7)

    for _, season_dir in iter_season_dirs(base_dir):
        fixtures_dir = season_dir / "fixtures"
        if not fixtures_dir.exists() or not fixtures_dir.is_dir():
            continue

        for entry in fixtures_dir.iterdir():
            if not entry.is_dir():
                continue

            parsed = _parse_fixture_dir_entry(entry.name)
            if not parsed:
                continue  # ignore unrelated folders
            start_utc, fixture_id = parsed
            start_print = start_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

            # Decide whether we are allowed to fetch for FUTURE fixtures
            if start_utc > now_utc and start_utc > week_ahead:
                # Far future: skip to save calls
                print(f"[fixtures] Skip far-future fixture id={fixture_id} date={start_print} (>7d ahead)")
                continue

            targets = [
                (entry / "data.json",        "participants;scores;formations", "data"),
                (entry / "lineup.json",      "lineups",                        "lineup"),
                (entry / "statistics.json",  "statistics.type",                "statistics"),
            ]

            for out_path, include, label in targets:
                # If file is missing -> fetch regardless (past or within next week)
                if not out_path.exists():
                    payload = _fetch_fixture_with_include(fixture_id, include)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    print(f"[fixtures] Got {label} for fixture id={fixture_id} date={start_print} -> {out_path.name}")
                    written.append(str(out_path))
                    continue

                # File exists and overwrite=False -> maybe refresh if the match is in the past and looks unfinished.
                if not overwrite:
                    if start_utc <= now_utc:
                        # Past fixture – check if file looks "not started"/incomplete
                        sid = _load_state_id_from_file(out_path)
                        if sid is None or sid == 1:
                            payload = _fetch_fixture_with_include(fixture_id, include)
                            with out_path.open("w", encoding="utf-8") as f:
                                json.dump(payload, f, ensure_ascii=False, indent=2)
                            print(f"[fixtures] Refresh {label} for fixture id={fixture_id} date={start_print} (state_id={sid})")
                            written.append(str(out_path))
                        else:
                            # Looks finalized -> keep it
                            continue
                    else:
                        # Future (<= 7 days) AND file already exists -> keep it as-is
                        continue
                else:
                    # overwrite=True -> always refetch
                    payload = _fetch_fixture_with_include(fixture_id, include)
                    with out_path.open("w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    print(f"[fixtures] Overwrite {label} for fixture id={fixture_id} date={start_print}")
                    written.append(str(out_path))

    return written

# --- Minimal CLI (no flags) ---
if __name__ == "__main__":
    if not API_TOKEN:
        print("Missing SPORTMONKS_API_TOKEN. Put it in your .env or OS env.", file=sys.stderr)
        sys.exit(1)
    files = write_ucl_fixtures_full_for_all_seasons(DATA_ROOT, filename="ucl_fixtures.json", force=False)
    print(f"Wrote/updated fixtures for {len(files)} seasons.")

    create_per_fixture_folders(DATA_ROOT, "ucl_fixtures.json")

    written = populate_fixture_files_for_all_seasons(DATA_ROOT, overwrite=False)
    print(f"Wrote {len(written)} JSON files across fixture folders.")