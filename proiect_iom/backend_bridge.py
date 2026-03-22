"""
backend_bridge.py

Glue code between the PyQt UI and the existing backend pipeline:

- Builds input.csv for a given (home_team_id, away_team_id) using
  build_single_row_team_ids() from build_single_input_row.py
- Calls:
    build_aggregated_inputs_main()
    predict_from_artifacts_main()
- Parses predictions/<TARGET>/report.txt files and returns the
  middle value from the 'medie:' line for each target.

Public function:

    run_prediction_pipeline(home_team_id: int, away_team_id: int) -> dict

Return format:

{
    "HOME": {
        "GOALS": float,
        "CORNERS": float,
        "YELLOWCARDS": float,
        "SHOTS_ON_TARGET": float,
        "FOULS": float,
        "OFFSIDES": float,
        "REDCARDS": float,
    },
    "AWAY": { ... same keys ... }
}
"""

import os
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional

from build_single_input_row import (
    build_single_row_for_team_ids,
    _write_single_csv,
    DEFAULT_DATA_ROOT,
)
from build_aggregated_inputs import main as build_aggregated_inputs_main
from predict_from_artifacts import main as predict_from_artifacts_main

BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_DIR = BASE_DIR / "predictions"

# same targets as in predict_from_artifacts.py
TARGETS: List[str] = [
    "HOME_GOALS", "AWAY_GOALS",
    "HOME_CORNERS", "AWAY_CORNERS",
    "HOME_YELLOWCARDS", "AWAY_YELLOWCARDS",
    "HOME_SHOTS_ON_TARGET", "AWAY_SHOTS_ON_TARGET",
    "HOME_FOULS", "AWAY_FOULS",
    "HOME_OFFSIDES", "AWAY_OFFSIDES",
    "HOME_REDCARDS", "AWAY_REDCARDS",
]


class BackendError(Exception):
    """Custom exception for backend pipeline errors."""


def _env_list(name: str, default: List[str]) -> List[str]:
    """
    Helper similar to the one in build_single_input_row.py:
    reads a comma-separated env var or falls back to default.
    """
    v = os.environ.get(name, "")
    if not v.strip():
        return default
    parts = [p.strip() for p in v.split(",")]
    return [p for p in parts if p]


def _build_input_csv(home_team_id: int, away_team_id: int) -> Path:
    """
    Use build_single_row_team_ids(...) + _write_single_csv(...)
    to create input.csv next to this file.
    """

    # same defaults as main() in build_single_input_row.py
    GK_STATS = _env_list("GK_STATS", ["GOALS_CONCEDED"])
    DF_STATS = _env_list("DF_STATS", ["GOALS_CONCEDED", "MINUTES_PLAYED", "APPEARANCES"])
    MF_STATS = _env_list("MF_STATS", ["GOALS_CONCEDED", "MINUTES_PLAYED", "APPEARANCES"])
    ATK_STATS = _env_list(
        "ATK_STATS",
        ["MINUTES_PLAYED", "APPEARANCES", "GOALS_CONCEDED", "SUBSTITUTIONS_IN", "SUBSTITUTIONS_OUT"],
    )

    MAX_DF = int(os.environ.get("MAX_DF", "6"))
    MAX_MF = int(os.environ.get("MAX_MF", "6"))
    MAX_ATK = int(os.environ.get("MAX_ATK", "4"))

    # build a single row for this (home_team_id, away_team_id)
    header, row = build_single_row_for_team_ids(
        data_root=DEFAULT_DATA_ROOT,
        home_team_id=home_team_id,
        away_team_id=away_team_id,
        gk_stats=GK_STATS,
        df_stats=DF_STATS,
        mf_stats=MF_STATS,
        atk_stats=ATK_STATS,
        max_df=MAX_DF,
        max_mf=MAX_MF,
        max_atk=MAX_ATK,
    )

    out_path = BASE_DIR / "input.csv"
    _write_single_csv(header, row, str(out_path))
    return out_path


def _parse_report_file(report_path: Path) -> Optional[float]:
    """
    Parse predictions/.../report.txt and return the middle value
    from the line starting with 'medie:'.

    Format expected:
        ...
        medie: 2.552, 4.644, 6.737

    We return 4.644 (the second value).
    """
    if not report_path.is_file():
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
    except Exception:
        return None

    medie_line = None
    for ln in reversed(lines):
        if ln.lower().startswith("medie"):
            medie_line = ln
            break

    if medie_line is None:
        return None

    try:
        _, rest = medie_line.split(":", 1)
    except ValueError:
        return None

    parts = [p.strip() for p in rest.split(",")]
    if len(parts) < 2:
        return None

    try:
        mid = float(parts[1])
    except Exception:
        return None

    return mid


def _collect_predictions() -> Dict[str, Dict[str, float]]:
    """
    Read predictions/<TARGET>/report.txt for all TARGETS and build
    a nested dict:

        {"HOME": {"GOALS": ..., ...}, "AWAY": {...}}
    """
    result: Dict[str, Dict[str, float]] = {"HOME": {}, "AWAY": {}}

    for target in TARGETS:
        target_dir = PREDICTIONS_DIR / target
        report_path = target_dir / "report.txt"
        mid_val = _parse_report_file(report_path)
        if mid_val is None:
            # skip missing / malformed targets
            continue

        side, metric = target.split("_", 1)  # e.g. HOME, GOALS
        side = side.upper()
        metric = metric.upper()
        if side not in result:
            result[side] = {}
        result[side][metric] = mid_val

    return result


def run_prediction_pipeline(home_team_id: int, away_team_id: int) -> Dict[str, Dict[str, float]]:
    """
    High-level entry point for the UI.

    Steps:
    1) Build input.csv using build_single_row_team_ids(...)
    2) Run build_aggregated_inputs_main()
    3) Run predict_from_artifacts_main()
    4) Parse predictions/*/report.txt and return the nested dict.
    """
    try:
        # 1) build input.csv
        _build_input_csv(home_team_id, away_team_id)

        # 2) aggregated inputs (raw_input.csv, sum_input.csv, etc.)
        build_aggregated_inputs_main()

        # 3) run prediction pipeline (writes predictions/*/report.txt)
        predict_from_artifacts_main()

        # 4) collect medie values
        summary = _collect_predictions()
        return summary

    except Exception as e:
        raise BackendError(f"Backend pipeline failed: {e}") from e
