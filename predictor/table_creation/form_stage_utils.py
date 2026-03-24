# form_stage_utils.py  (refactor/table_creation/)
#
# Pure-computation utilities for:
#   - Season stage normalisation (stage_id → float 0.0–1.0)
#   - Rolling per-team form vectors (last-N-match statistics)
#
# No CSV writing happens here.  All I/O is read-only (JSON files).
# Imported by build_table_v2.py.

import math
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from shared_table_utils import (
    read_json,
    norm_key,
    is_valid_season_dir,
    season_key,
    resolve_home_away_team_ids,
    extract_targets,
    TARGET_KEYS,
    impute_stats,
)


# =============================================================================
# ==  CONSTANTS  ==============================================================
# =============================================================================

# Stats tracked in the form window — same as the prediction targets.
FORM_STAT_KEYS: List[str] = TARGET_KEYS   # e.g. ["GOALS", "CORNERS", ...]

# Number of form column names per side (home/away):
#   len(FORM_STAT_KEYS) * 2  (FOR + AGAINST each key)
# Total form columns = 2 sides * len(FORM_STAT_KEYS) * 2 = 4 * 7 = 28

# Column name helpers
def form_col(side: str, stat: str, direction: str) -> str:
    """
    Build a form column name.
    Examples:
        form_col("HOME", "GOALS",   "FOR")     -> "HOME_FORM_GOALS_FOR"
        form_col("AWAY", "CORNERS", "AGAINST") -> "AWAY_FORM_CORNERS_AGAINST"
    """
    return f"{side.upper()}_FORM_{stat.upper()}_{direction.upper()}"


def all_form_columns() -> List[str]:
    """
    Return the ordered list of all 28 form column names.
    Order: HOME_FOR … HOME_AGAINST … AWAY_FOR … AWAY_AGAINST …
    This order is used both in the CSV header and in row assembly.
    """
    cols: List[str] = []
    for side in ("HOME", "AWAY"):
        for direction in ("FOR", "AGAINST"):
            for stat in FORM_STAT_KEYS:
                cols.append(form_col(side, stat, direction))
    return cols


# --- Continuous-form column helpers (no season reset) ---

def cform_col(side: str, stat: str, direction: str) -> str:
    """
    Build a continuous-form column name (cross-season rolling window).
    Examples:
        cform_col("HOME", "GOALS",   "FOR")     -> "HOME_CFORM_GOALS_FOR"
        cform_col("AWAY", "CORNERS", "AGAINST") -> "AWAY_CFORM_CORNERS_AGAINST"
    """
    return f"{side.upper()}_CFORM_{stat.upper()}_{direction.upper()}"


def all_cform_columns() -> List[str]:
    """
    Return the ordered list of all 28 continuous-form column names.
    Same order as all_form_columns() but with 'CFORM' instead of 'FORM'.
    """
    cols: List[str] = []
    for side in ("HOME", "AWAY"):
        for direction in ("FOR", "AGAINST"):
            for stat in FORM_STAT_KEYS:
                cols.append(cform_col(side, stat, direction))
    return cols


STAGE_COL = "STAGE_NORMALIZED"


# =============================================================================
# ==  PASS 1: SCAN SEASON  ====================================================
# =============================================================================

class FixtureInfo:
    """
    Lightweight container for everything we need from one fixture during pass 1.
    `stats` maps  "HOME_GOALS" / "AWAY_CORNERS" / … → float | nan
    `state_id` == 5 means the match is finished (only these rows are written).
    """
    __slots__ = ("fix_dir", "ts", "fixture_id", "home_tid", "away_tid",
                 "stage_id", "state_id", "stats")

    def __init__(
        self,
        fix_dir: str,
        ts: str,
        fixture_id,
        home_tid: int,
        away_tid: int,
        stage_id,
        state_id,
        stats: Dict[str, float],
    ):
        self.fix_dir    = fix_dir
        self.ts         = ts
        self.fixture_id = fixture_id
        self.home_tid   = home_tid
        self.away_tid   = away_tid
        self.stage_id   = stage_id
        self.state_id   = state_id
        self.stats      = stats   # {HOME_GOALS: float, AWAY_GOALS: float, …}


def scan_season_fixtures(season_dir: str) -> List[FixtureInfo]:
    """
    Pass-1 scan: read every fixture's data.json + statistics.json
    in `season_dir/fixtures/`.

    Returns a list of FixtureInfo objects (one per readable fixture).
    Unreadable or incomplete fixtures are silently skipped — the main
    loop will skip them for the same reason via the original logic.
    """
    fixtures_dir = os.path.join(season_dir, "fixtures")
    if not os.path.isdir(fixtures_dir):
        return []

    results: List[FixtureInfo] = []

    for entry in os.scandir(fixtures_dir):
        if not entry.is_dir():
            continue
        fix_dir    = entry.path
        data_path  = os.path.join(fix_dir, "data.json")
        stats_path = os.path.join(fix_dir, "statistics.json")

        if not (os.path.isfile(data_path) and os.path.isfile(stats_path)):
            continue

        # --- fixture metadata ---
        data_payload = read_json(data_path)
        if not data_payload:
            continue
        d = data_payload.get("data", {})
        if not isinstance(d, dict):
            continue

        stage_id   = d.get("stage_id")
        state_id   = d.get("state_id")
        fixture_id = d.get("id")
        ts = str(d.get("starting_at") or d.get("starting_at_timestamp") or "")
        if not ts:
            continue

        # --- home / away team ids ---
        ha = resolve_home_away_team_ids(fix_dir)
        if not ha:
            continue
        home_tid, away_tid = ha

        # --- target stats (used later for form vectors) ---
        stats_payload = read_json(stats_path) or {}
        tgt = extract_targets(stats_payload, home_tid, away_tid)

        results.append(FixtureInfo(
            fix_dir    = fix_dir,
            ts         = ts,
            fixture_id = fixture_id,
            home_tid   = home_tid,
            away_tid   = away_tid,
            stage_id   = stage_id,
            state_id   = state_id,
            stats      = tgt,
        ))

    return results


# =============================================================================
# ==  STAGE NORMALISATION  ====================================================
# =============================================================================

def build_stage_map(fixture_infos: List[FixtureInfo]) -> Dict:
    """
    Given all fixtures in a season (from scan_season_fixtures), build a mapping:
        stage_id → float in [0.0, 1.0]

    Stages are ordered by their earliest fixture timestamp.
    0.0 = very first qualifying round, 1.0 = final.
    If only one stage exists (degenerate), returns {stage_id: 0.0}.

    The normalisation is:  rank / (n_stages - 1)
    where rank is 0-indexed chronological order of stage_id.
    """
    stage_earliest: Dict = {}     # stage_id → earliest timestamp string

    for fi in fixture_infos:
        if fi.stage_id is None:
            continue
        ts = fi.ts
        prev = stage_earliest.get(fi.stage_id)
        if prev is None or ts < prev:
            stage_earliest[fi.stage_id] = ts

    if not stage_earliest:
        return {}

    ordered = sorted(stage_earliest.items(), key=lambda x: x[1])
    n = len(ordered)

    stage_map: Dict = {}
    for rank, (sid, _) in enumerate(ordered):
        stage_map[sid] = 0.0 if n == 1 else rank / (n - 1)

    return stage_map


# =============================================================================
# ==  FORM TRACKING  ==========================================================
# =============================================================================

class TeamFormTracker:
    """
    Maintains a rolling window of the last `window` matches for every team.

    Each match entry stored per team is a dict:
        {stat_key: (value_for, value_against), ...}
    where value_for   = stat K from the team's own perspective,
          value_against = stat K from the opponent's perspective.

    Example: if team T played HOME and HOME_GOALS=2, AWAY_GOALS=1:
        stored["GOALS"] = (2.0, 1.0)   # scored 2, conceded 1

    Usage pattern (called BEFORE writing each row, updated AFTER):
        form_home = tracker.get_form(home_tid)
        form_away = tracker.get_form(away_tid)
        # ... write row using form_home, form_away ...
        tracker.update(home_tid, away_tid, stats_dict)
    """

    def __init__(self, window: int = 5):
        self.window = window
        # team_id → deque of {stat_key: (for, against)} dicts
        self._history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=window))

    def reset(self):
        """Call between seasons."""
        self._history.clear()

    def get_form(self, team_id: int) -> Dict[str, Optional[float]]:
        """
        Compute the mean FOR and AGAINST for each stat over the rolling window.

        Returns:
            {"{stat}_FOR": float|nan, "{stat}_AGAINST": float|nan, ...}

        NaN is returned for a stat if the team has NO history yet
        (very first match in this season).  If history exists but some stat
        values are NaN (not recorded in statistics.json), those slots are
        excluded from the mean; if all slots are NaN the result is NaN.
        """
        history = self._history[team_id]
        result: Dict[str, Optional[float]] = {}

        # No history yet (first match of the season): return 0 for all columns.
        if not history:
            for stat in FORM_STAT_KEYS:
                result[f"{stat}_FOR"]     = 0.0
                result[f"{stat}_AGAINST"] = 0.0
            return result

        for stat in FORM_STAT_KEYS:
            for_vals:     List[float] = []
            against_vals: List[float] = []
            for entry in history:
                v_for, v_against = entry.get(stat, (math.nan, math.nan))
                if not math.isnan(v_for):
                    for_vals.append(v_for)
                if not math.isnan(v_against):
                    against_vals.append(v_against)
            result[f"{stat}_FOR"]     = (sum(for_vals)     / len(for_vals))     if for_vals     else math.nan
            result[f"{stat}_AGAINST"] = (sum(against_vals) / len(against_vals)) if against_vals else math.nan

        return result

    def update(
        self,
        home_tid: int,
        away_tid: int,
        stats: Dict[str, float],
    ) -> None:
        """
        Record match results for both teams.
        `stats` is the extract_targets() output:
            {"HOME_GOALS": 2.0, "AWAY_GOALS": 1.0, "HOME_CORNERS": 5.0, ...}
        """
        # Impute missing values before storing — ensures the deque never
        # accumulates NaN slots that would pollute future form averages.
        stats = impute_stats(stats)

        home_entry: Dict[str, Tuple[float, float]] = {}
        away_entry: Dict[str, Tuple[float, float]] = {}

        for stat in FORM_STAT_KEYS:
            h_val = stats.get(f"HOME_{stat}", math.nan)
            a_val = stats.get(f"AWAY_{stat}", math.nan)
            try:
                h_val = float(h_val)
            except (TypeError, ValueError):
                h_val = math.nan
            try:
                a_val = float(a_val)
            except (TypeError, ValueError):
                a_val = math.nan

            # home team: FOR = home stat, AGAINST = away stat
            home_entry[stat] = (h_val, a_val)
            # away team: FOR = away stat, AGAINST = home stat
            away_entry[stat] = (a_val, h_val)

        self._history[home_tid].append(home_entry)
        self._history[away_tid].append(away_entry)


# =============================================================================
# ==  ROW ASSEMBLY HELPER  ====================================================
# =============================================================================

def form_cells_for_row(
    home_form: Dict[str, Optional[float]],
    away_form: Dict[str, Optional[float]],
) -> List:
    """
    Assemble the 28 form cell values in the canonical column order
    defined by all_form_columns().

    Order mirrors all_form_columns() exactly:
        HOME × FOR  × [all stats]
        HOME × AGAINST × [all stats]
        AWAY × FOR  × [all stats]
        AWAY × AGAINST × [all stats]
    """
    cells: List = []
    for side_form in (home_form, away_form):
        for direction in ("FOR", "AGAINST"):
            for stat in FORM_STAT_KEYS:
                cells.append(side_form.get(f"{stat}_{direction}", math.nan))
    return cells


def cform_cells_for_row(
    home_cform: Dict[str, Optional[float]],
    away_cform: Dict[str, Optional[float]],
) -> List:
    """
    Assemble the 28 continuous-form cell values in the canonical column order
    defined by all_cform_columns().  Identical logic to form_cells_for_row.
    """
    cells: List = []
    for side_form in (home_cform, away_cform):
        for direction in ("FOR", "AGAINST"):
            for stat in FORM_STAT_KEYS:
                cells.append(side_form.get(f"{stat}_{direction}", math.nan))
    return cells
