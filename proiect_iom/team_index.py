"""
team_index.py

Builds and queries a team index for the UI:
- Scans the data/ directory for seasons starting in 2023–2026.
- For each season, reads the teams/ folder and extracts (team_id, team_name).
- Deduplicates by team_id.
- Saves the result to teams_index.json for fast loading.
- Provides a fuzzy search helper for the autocomplete UI.
"""

import json
import os
import re
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional

# ---------- Paths & constants ----------

BASE_DIR = Path(__file__).resolve().parent
# Same logic as the rest of your project: default data root is "data" at project root.
DEFAULT_DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

TEAM_INDEX_PATH = BASE_DIR / "teams_index.json"

# Seasons we care about (by starting year)
ALLOWED_START_YEARS = {2023, 2024, 2025, 2026}

# Example season dir name: "2024-2025_25580"
SEASON_DIR_RE = re.compile(r"^(?P<y1>\d{4})-(?P<y2>\d{4})_(?P<season_id>\d+)$")


@dataclass
class TeamEntry:
    team_id: int
    name: str        # original display name
    search_name: str # normalized for matching (lowercase, no extra punctuation)


# ---------- Helpers ----------

def _normalize_search_text(s: str) -> str:
    """
    Normalize a string for fuzzy search:
    - lowercase
    - replace non-alphanumeric with spaces
    - collapse multiple spaces
    """
    s = s.lower()
    # Replace anything that's not letter/digit with space
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = " ".join(s.split())
    return s


def _is_allowed_season_dir(name: str) -> bool:
    """Check if folder name matches season pattern and start year is in allowed set."""
    m = SEASON_DIR_RE.match(name)
    if not m:
        return False
    y1 = int(m.group("y1"))
    # We only care about seasons starting in specific years
    return y1 in ALLOWED_START_YEARS


def build_team_index(data_root: Path = DEFAULT_DATA_ROOT) -> List[TeamEntry]:
    """
    Scan data_root for seasons and build a deduplicated list of teams.

    Expected structure:
        data/
          {season_year}-{season_year+1}_{season_id}/
              teams/
                  {team_name}_{team_id}/

    Returns:
        List[TeamEntry] with unique team_id.
    """
    if not data_root.is_dir():
        raise FileNotFoundError(f"Data root not found: {data_root}")

    teams_by_id: Dict[int, TeamEntry] = {}

    for entry in data_root.iterdir():
        if not entry.is_dir():
            continue
        if not _is_allowed_season_dir(entry.name):
            continue

        season_dir = entry
        teams_dir = season_dir / "teams"
        if not teams_dir.is_dir():
            continue

        for t_entry in teams_dir.iterdir():
            if not t_entry.is_dir():
                continue
            # Folder name expected: "{team_name}_{team_id}"
            name_part, sep, suffix = t_entry.name.rpartition("_")
            if not sep or not suffix.isdigit():
                # If format is unexpected, skip
                continue

            team_id = int(suffix)
            team_name = name_part if name_part else t_entry.name

            if team_id in teams_by_id:
                # Same team_id already seen in another season; skip duplicates
                continue

            search_name = _normalize_search_text(team_name)
            teams_by_id[team_id] = TeamEntry(
                team_id=team_id,
                name=team_name,
                search_name=search_name,
            )

    teams = list(teams_by_id.values())
    # Sort alphabetically by display name for nicer suggestion lists
    teams.sort(key=lambda t: t.name.lower())
    return teams


def save_team_index(teams: List[TeamEntry], path: Path = TEAM_INDEX_PATH) -> None:
    """Save team index as JSON to disk."""
    data = [asdict(t) for t in teams]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_team_index(path: Path = TEAM_INDEX_PATH) -> List[TeamEntry]:
    """
    Load team index from JSON.
    Raises FileNotFoundError if the file does not exist.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    teams: List[TeamEntry] = []
    for item in raw:
        teams.append(
            TeamEntry(
                team_id=int(item["team_id"]),
                name=str(item["name"]),
                search_name=str(item["search_name"]),
            )
        )
    return teams


def ensure_team_index(
    data_root: Path = DEFAULT_DATA_ROOT, path: Path = TEAM_INDEX_PATH
) -> List[TeamEntry]:
    """
    Load existing index if present; otherwise build from data_root and save to disk.
    """
    if path.is_file():
        return load_team_index(path)
    teams = build_team_index(data_root=data_root)
    save_team_index(teams, path=path)
    return teams


# ---------- Fuzzy search for UI ----------

def search_teams(
    query: str,
    teams: List[TeamEntry],
    limit: int = 10,
    min_ratio: float = 0.35,
) -> List[TeamEntry]:
    """
    Fuzzy search for teams given a text query.

    Strategy:
    - normalize query
    - for each team:
        - score_sub: how many query tokens appear as substrings in team.search_name
        - score_ratio: global similarity ratio (SequenceMatcher)
    - filter out results with too low ratio
    - sort by:
        - descending score_sub
        - descending ratio
        - alphabetical name
    - return top 'limit' results

    This should handle:
        "real masrid" -> "Real Madrid"
        "dinamo" -> "Dinamo Zagreb", "Dinamo Tbilisi", ...
    """
    q_norm = _normalize_search_text(query)
    if not q_norm:
        return []

    q_tokens = q_norm.split()
    q_joined = " ".join(q_tokens)

    scored: List[tuple] = []

    for t in teams:
        # Count how many query tokens appear in the team name
        sub_hits = sum(1 for tok in q_tokens if tok in t.search_name)
        score_sub = sub_hits / max(1, len(q_tokens))

        # Global fuzzy ratio between full query and team name
        ratio = SequenceMatcher(None, q_joined, t.search_name).ratio()

        # Rough filter for very bad matches
        if ratio < min_ratio and score_sub == 0:
            continue

        scored.append((t, score_sub, ratio))

    # Sort by:
    # 1) more token matches
    # 2) better fuzzy ratio
    # 3) alphabetical name
    scored.sort(
        key=lambda item: (-item[1], -item[2], item[0].name.lower())
    )

    # Keep only the team objects, up to 'limit'
    return [item[0] for item in scored[:limit]]


# ---------- Small CLI for debugging ----------

def _debug_repl():
    """
    Simple interactive REPL to test the index & search from terminal:
    python team_index.py
    """
    print(f"[INFO] Using data root: {DEFAULT_DATA_ROOT}")
    teams = ensure_team_index()
    print(f"[INFO] Loaded {len(teams)} unique teams.")
    print("Type a search query (or 'exit' to quit):")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[INFO] Exiting.")
            break

        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("[INFO] Exiting.")
            break

        results = search_teams(q, teams, limit=10)
        if not results:
            print("  (no matches)")
            continue

        for t in results:
            print(f"  - {t.name} (id={t.team_id})")


def main():
    # Build or load index, then enter debug REPL.
    teams = ensure_team_index()
    print(f"[OK] Team index ready with {len(teams)} unique teams.")
    _debug_repl()


if __name__ == "__main__":
    main()
