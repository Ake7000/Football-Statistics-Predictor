# data_scraping/seasons.py
import os
import sys
import json
import re
from pathlib import Path
import requests

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Config from environment ---
SUBFOLDERS = ["data", "fixtures", "teams", "players"]
API_TOKEN = os.getenv("SPORTMONKS_API_TOKEN")
BASE_URL = os.getenv("SPORTMONKS_BASE_URL", "https://api.sportmonks.com/v3/football")
LEAGUE_ID = os.getenv("UCL_LEAGUE_ID", "2")
OUTPUT_FILE = os.getenv("UCL_SEASONS_OUTPUT", "seasons_ids.json")  # saved in project root
DATA_ROOT = os.getenv("DATA_ROOT", "data")  # base folder for season directories

URL = f"{BASE_URL.rstrip('/')}/leagues/{LEAGUE_ID}"

def fetch_and_save_seasons_json(api_token: str, url: str = URL, output_path: str = OUTPUT_FILE) -> None:
    """
    Fetch raw JSON for league {LEAGUE_ID} including seasons and save to a file (no transformation).
    """
    params = {"api_token": api_token, "include": "seasons"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def _season_base_name(raw_name: str) -> str:
    """
    Convert 'YYYY/YYYY' → 'YYYY-YYYY'. If unexpected, sanitize and replace '/' with '-'.
    """
    raw_name = (raw_name or "").strip()
    m = re.fullmatch(r"(\d{4})/(\d{4})", raw_name)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    # Fallback sanitation for odd names
    safe = re.sub(r"[\\/:\*\?\"<>\|]", "_", raw_name)  # common illegal chars
    safe = safe.replace("/", "-")
    return safe or "unknown-season"

def _season_folder_name(raw_name: str, season_id) -> str:
    """
    Compose final folder name: 'YYYY-YYYY_[season_id]'.
    """
    base = _season_base_name(raw_name)
    sid = str(season_id).strip() if season_id is not None else "unknownid"
    return f"{base}_{sid}"

def create_season_folders_flat(json_path: str = OUTPUT_FILE, base_dir: str = DATA_ROOT) -> list[str]:
    """
    Create ONE folder per season under `data/`, named like '2024-2025_23619',
    and inside each season folder create subfolders: 'data', 'fixtures', 'teams', 'players'.
    Returns the list of created/ensured season directories.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    seasons = (payload or {}).get("data", {}).get("seasons", [])
    created: list[str] = []
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    for s in seasons:
        name = str(s.get("name", "")).strip()
        season_id = s.get("id")
        if not name:
            continue

        folder = _season_folder_name(name, season_id)
        target = base / folder
        target.mkdir(parents=True, exist_ok=True)

        # ensure required subfolders exist
        for sub in SUBFOLDERS:
            (target / sub).mkdir(parents=True, exist_ok=True)

        created.append(str(target))

    return created

def update_season_metadata_files(
    json_path: str = OUTPUT_FILE,
    base_dir: str = DATA_ROOT,
    overwrite: bool = False,
) -> list[str]:
    """
    For each season in the seasons JSON, write its raw metadata into:
      data/YYYY-YYYY_<season_id>/data/data.json

    If overwrite=False, existing data.json files are left untouched.
    Returns the list of files written (or touched if overwritten).
    """
    # Load payload with seasons
    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    seasons = (payload or {}).get("data", {}).get("seasons", [])
    written: list[str] = []

    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    for s in seasons:
        name = str(s.get("name", "")).strip()
        season_id = s.get("id")
        if not name or season_id is None:
            continue

        # Compute season folder name "YYYY-YYYY_<id>" and ensure path exists
        season_folder = _season_folder_name(name, season_id)
        season_data_dir = base / season_folder / "data"
        season_data_dir.mkdir(parents=True, exist_ok=True)

        # Prepare destination file
        out_file = season_data_dir / "data.json"

        # Skip if exists and not overwriting
        if out_file.exists() and not overwrite:
            continue

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)

        written.append(str(out_file))

    return written



if __name__ == "__main__":
    if not API_TOKEN:
        print("Missing SPORTMONKS_API_TOKEN. Put it in your .env or OS env.", file=sys.stderr)
        sys.exit(1)

    # 1) Fetch & save raw seasons JSON
    fetch_and_save_seasons_json(API_TOKEN)
    # print(f"Saved raw JSON to {OUTPUT_FILE}")

    # 2) Create flat season folders like data/2024-2025_23619
    dirs = create_season_folders_flat(OUTPUT_FILE, DATA_ROOT)
    # print(f"Created/ensured {len(dirs)} season folders under '{DATA_ROOT}'.")

    # 3) Write per-season data.json files
    files = update_season_metadata_files(OUTPUT_FILE, DATA_ROOT, overwrite=False)
    print(f"Wrote {len(files)} season metadata files.")
