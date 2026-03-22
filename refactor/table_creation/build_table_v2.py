# build_table_v2.py  (refactor/table_creation/)
#
# Drop-in replacement for table_creation/train_fixed_slots.py that produces
# the SAME columns as the original PLUS:
#   - 28 rolling-form columns  (HOME/AWAY × 7 stats × FOR/AGAINST)
#   - 1  stage-normalisation column  (STAGE_NORMALIZED, 0.0=qualifying → 1.0=final)
#
# Column order in the output CSV:
#   <meta>  <player-slot features (original)>  <form cols>  <STAGE_NORMALIZED>  <targets>
#
# Key design decisions:
#   - Original train_fixed_slots.py is NEVER modified.  All its helpers are
#     imported directly.
#   - Two-pass processing per season:
#       Pass 1 — scan every fixture's data.json + statistics.json to build the
#                stage map and the fixture stats cache.
#       Pass 2 — iterate fixtures CHRONOLOGICALLY, read form BEFORE the match,
#                write the row, then update the form tracker.
#   - Form resets between seasons (as requested).
#   - Fixtures that are skipped for row-writing (missing lineup / exceeds slots)
#     still update the form tracker, because the match happened and affected
#     the teams' form.
#   - Partial history: if a team has played k < N games this season, we average
#     those k games.  NaN only for the team's very first match in the season.
#
# How to run (from the refactor/table_creation/ directory):
#   python build_table_v2.py
# Or via env vars (same interface as original):
#   DATA_ROOT=../../data TRAIN_START_SEASON=2017-2018_7907 \
#   TRAIN_END_SEASON=2025-2026_25580 python build_table_v2.py

import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Path setup: ensure refactor/table_creation/ is importable.
# ---------------------------------------------------------------------------
_THIS_DIR       = Path(__file__).parent               # refactor/table_creation/
_REFACTOR_DIR   = _THIS_DIR.parent                    # refactor/
_WORKSPACE_ROOT = _REFACTOR_DIR.parent                # licenta/

if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# ---------------------------------------------------------------------------
# Imports from shared_table_utils (self-contained, no dependency on original)
# ---------------------------------------------------------------------------
from shared_table_utils import (      # noqa: E402
    TARGET_KEYS,
    POSITION_ID_TO_CODE,
    is_valid_season_dir,
    season_key,
    norm_key,
    ensure_dir,
    read_json,
    find_player_dirs,
    load_player_last_year_stats,
    load_player_stats_with_current_fallback,
    parse_lineup_players,
    resolve_home_away_team_ids,
    extract_targets,
    impute_stats,
    choose_sort_key_for_line,
    slot_id_col,
    _windows_safe,
)

# ---------------------------------------------------------------------------
# Imports from this package
# ---------------------------------------------------------------------------
from form_stage_utils import (        # noqa: E402
    FORM_STAT_KEYS,
    STAGE_COL,
    all_form_columns,
    all_cform_columns,
    scan_season_fixtures,
    build_stage_map,
    TeamFormTracker,
    form_cells_for_row,
    cform_cells_for_row,
    FixtureInfo,
)


# =============================================================================
# ==  HEADER BUILDER  =========================================================
# =============================================================================

def build_header(
    gk_stats_n: List[str],
    df_stats_n: List[str],
    mf_stats_n: List[str],
    atk_stats_n: List[str],
    max_df: int,
    max_mf: int,
    max_atk: int,
) -> List[str]:
    """
    Construct the full CSV header.
    Identical to the original logic for meta + feature + target sections.
    New sections (form, stage) are inserted between features and targets.
    """
    meta_cols = ["season_label", "fixture_id", "fixture_ts",
                 "home_team_id", "away_team_id"]

    feature_cols: List[str] = []

    # ---- HOME ----
    feature_cols += [slot_id_col("gk", "home")]
    feature_cols += [f"GK_HOME_{s}" for s in gk_stats_n]

    for i in range(1, max_df + 1):
        feature_cols += [slot_id_col("df", "home", i)]
        for s in df_stats_n:
            feature_cols.append(f"DF{i}_HOME_{s}")
    feature_cols.append("NO_OF_DF_HOME")

    for i in range(1, max_mf + 1):
        feature_cols += [slot_id_col("mf", "home", i)]
        for s in mf_stats_n:
            feature_cols.append(f"MF{i}_HOME_{s}")
    feature_cols.append("NO_OF_MF_HOME")

    for i in range(1, max_atk + 1):
        feature_cols += [slot_id_col("atk", "home", i)]
        for s in atk_stats_n:
            feature_cols.append(f"ATK{i}_HOME_{s}")
    feature_cols.append("NO_OF_ATK_HOME")

    # ---- AWAY ----
    feature_cols += [slot_id_col("gk", "away")]
    feature_cols += [f"GK_AWAY_{s}" for s in gk_stats_n]

    for i in range(1, max_df + 1):
        feature_cols += [slot_id_col("df", "away", i)]
        for s in df_stats_n:
            feature_cols.append(f"DF{i}_AWAY_{s}")
    feature_cols.append("NO_OF_DF_AWAY")

    for i in range(1, max_mf + 1):
        feature_cols += [slot_id_col("mf", "away", i)]
        for s in mf_stats_n:
            feature_cols.append(f"MF{i}_AWAY_{s}")
    feature_cols.append("NO_OF_MF_AWAY")

    for i in range(1, max_atk + 1):
        feature_cols += [slot_id_col("atk", "away", i)]
        for s in atk_stats_n:
            feature_cols.append(f"ATK{i}_AWAY_{s}")
    feature_cols.append("NO_OF_ATK_AWAY")

    # ---- Form (new) ----
    form_cols = all_form_columns()   # 28 columns

    # ---- Continuous-form (cross-season rolling window) ----
    cform_cols = all_cform_columns()  # 28 columns

    # ---- Stage (new) ----
    stage_cols = [STAGE_COL]

    # ---- Targets ----
    target_cols = []
    for k in TARGET_KEYS:
        target_cols.append(f"HOME_{k}")
        target_cols.append(f"AWAY_{k}")

    return meta_cols + feature_cols + form_cols + cform_cols + stage_cols + target_cols


# =============================================================================
# ==  SIDE FEATURE CELLS  =====================================================
# =============================================================================

def _sort_line(line_players: List[Dict], pid_to_path: Dict[int, str]) -> List[Dict]:
    """
    Re-implementation of the nested sort_line() from train_fixed_slots.py.
    Identical logic, exposed as a module-level function.

    Sorts players by:
      1. formation_position (if available) ascending
      2. fallback: MINUTES_PLAYED then APPEARANCES descending
    """
    decorated = []
    for p in line_players:
        last_stats = {}
        player_dir = pid_to_path.get(p["player_id"])
        if player_dir:
            last_stats = load_player_last_year_stats(player_dir)
        fb   = choose_sort_key_for_line(last_stats)
        fpos = p.get("formation_position")
        if isinstance(fpos, int):
            decorated.append((0, fpos, fb, p))
        else:
            decorated.append((1, 10 ** 9, fb, p))
    decorated.sort(key=lambda x: (x[0], x[1], -x[2][0], -x[2][1]))
    return [t[-1] for t in decorated]


def _build_side_feature_cells(
    side_line: Dict[str, List[Dict]],
    pid_to_path: Dict[int, str],
    max_df: int,
    max_mf: int,
    max_atk: int,
    gk_stats_n: List[str],
    df_stats_n: List[str],
    mf_stats_n: List[str],
    atk_stats_n: List[str],
) -> List[Any]:
    """
    Re-implementation of the nested side_slots() from train_fixed_slots.py.
    Identical logic, exposed as a module-level function.

    Returns the flat list of cell values for one side's player-feature columns.
    """
    row_vals: List[Any] = []

    # ---- GK ----
    gk_list = side_line["GK"][:1]
    if gk_list:
        pid   = gk_list[0]["player_id"]
        pdir  = pid_to_path.get(pid)
        row_vals.append(pid)
        stats = load_player_stats_with_current_fallback(pdir, True) if pdir else {}
        for s in gk_stats_n:
            row_vals.append(stats.get(s, math.nan))
    else:
        row_vals.append("")
        for _ in gk_stats_n:
            row_vals.append(math.nan)

    # ---- DF ----
    df_list = side_line["DF"][:max_df]
    for slot in range(max_df):
        if slot < len(df_list):
            pid   = df_list[slot]["player_id"]
            pdir  = pid_to_path.get(pid)
            stats = load_player_stats_with_current_fallback(pdir, True) if pdir else {}
            row_vals.append(pid)
            for s in df_stats_n:
                row_vals.append(stats.get(s, math.nan))
        else:
            row_vals.append("")
            for _ in df_stats_n:
                row_vals.append(math.nan)
    row_vals.append(len(side_line["DF"]))

    # ---- MF ----
    mf_list = side_line["MF"][:max_mf]
    for slot in range(max_mf):
        if slot < len(mf_list):
            pid   = mf_list[slot]["player_id"]
            pdir  = pid_to_path.get(pid)
            stats = load_player_stats_with_current_fallback(pdir, True) if pdir else {}
            row_vals.append(pid)
            for s in mf_stats_n:
                row_vals.append(stats.get(s, math.nan))
        else:
            row_vals.append("")
            for _ in mf_stats_n:
                row_vals.append(math.nan)
    row_vals.append(len(side_line["MF"]))

    # ---- ATK ----
    atk_list = side_line["ATK"][:max_atk]
    for slot in range(max_atk):
        if slot < len(atk_list):
            pid   = atk_list[slot]["player_id"]
            pdir  = pid_to_path.get(pid)
            stats = load_player_stats_with_current_fallback(pdir, True) if pdir else {}
            row_vals.append(pid)
            for s in atk_stats_n:
                row_vals.append(stats.get(s, math.nan))
        else:
            row_vals.append("")
            for _ in atk_stats_n:
                row_vals.append(math.nan)
    row_vals.append(len(side_line["ATK"]))

    return row_vals


# =============================================================================
# ==  OUTPUT FILENAME  ========================================================
# =============================================================================

def _build_output_slug(
    start_season: str,
    end_season: str,
    gk_stats_n: List[str],
    df_stats_n: List[str],
    mf_stats_n: List[str],
    atk_stats_n: List[str],
    max_df: int,
    max_mf: int,
    max_atk: int,
    form_window: int,
) -> str:
    import hashlib, json as _json
    cfg_str = _json.dumps({
        "start": start_season, "end": end_season,
        "slots": {"GK": 1, "DF": max_df, "MF": max_mf, "ATK": max_atk},
        "gk": gk_stats_n, "df": df_stats_n, "mf": mf_stats_n, "atk": atk_stats_n,
        "form_window": form_window,
        "version": "v2",
    }, sort_keys=True, ensure_ascii=True)
    cfg_hash = hashlib.md5(cfg_str.encode("utf-8")).hexdigest()[:10]

    start_tag  = start_season.split("_")[0]
    end_tag    = end_season.split("_")[0]
    stats_tag  = f"g{len(gk_stats_n)}-d{len(df_stats_n)}-m{len(mf_stats_n)}-a{len(atk_stats_n)}"
    slots_tag  = f"1-{max_df}-{max_mf}-{max_atk}"

    slug = (
        f"fixedslots_v2__{start_tag}_to_{end_tag}"
        f"__stats({stats_tag})__slots({slots_tag})"
        f"__form{form_window}__cfg_{cfg_hash}.csv"
    )
    return _windows_safe(slug)


# =============================================================================
# ==  MAIN BUILD FUNCTION  ====================================================
# =============================================================================

def build_train_table_v2(
    data_root: str,
    start_season: str,
    end_season: str,
    gk_stats: List[str],
    df_stats: List[str],
    mf_stats: List[str],
    atk_stats: List[str],
    max_df: int = 6,
    max_mf: int = 6,
    max_atk: int = 4,
    form_window: int = 5,
    out_dir: str = "train_tables",
    verbose: bool = True,
) -> str:
    """
    Build a training CSV with all original player-slot columns PLUS form and
    stage columns.

    Args:
        data_root:      Path to the data/ directory.
        start_season:   e.g. "2017-2018_7907"
        end_season:     e.g. "2025-2026_25580"
        gk_stats / df_stats / mf_stats / atk_stats:
                        Per-role stat names (same as original script).
        max_df / max_mf / max_atk:
                        Maximum player slots per role (same as original).
        form_window:    N last matches to include in rolling form features.
        out_dir:        Output directory for the CSV.
        verbose:        Print progress.

    Returns:
        Absolute path to the written CSV.
    """
    # ---- Normalise stat key lists ----
    gk_stats_n  = [norm_key(s) for s in gk_stats]
    df_stats_n  = [norm_key(s) for s in df_stats]
    mf_stats_n  = [norm_key(s) for s in mf_stats]
    atk_stats_n = [norm_key(s) for s in atk_stats]

    ensure_dir(out_dir)
    slug     = _build_output_slug(start_season, end_season,
                                  gk_stats_n, df_stats_n, mf_stats_n, atk_stats_n,
                                  max_df, max_mf, max_atk, form_window)
    out_path = os.path.join(out_dir, slug)

    header = build_header(gk_stats_n, df_stats_n, mf_stats_n, atk_stats_n,
                          max_df, max_mf, max_atk)

    # ---- Collect seasons in range ----
    all_season_entries = [
        e for e in os.scandir(data_root)
        if e.is_dir() and is_valid_season_dir(e.name)
    ]
    all_season_entries.sort(key=lambda e: season_key(e.name))

    bounds   = (season_key(start_season), season_key(end_season))
    y1_min, y1_max = bounds[0][0], bounds[1][0]
    seasons_in_range = [
        e for e in all_season_entries
        if y1_min <= season_key(e.name)[0] <= y1_max
    ]

    if verbose:
        print(f"[RANGE] Using seasons: {[e.name for e in seasons_in_range]}")
        print(f"[INFO]  Form window N={form_window}  |  Output: {out_path}")

    tracker       = TeamFormTracker(window=form_window)
    cform_tracker = TeamFormTracker(window=form_window)  # never reset — cross-season form

    total_processed = 0
    total_written   = 0
    total_skipped   = 0

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for season_entry in seasons_in_range:
            season_label = season_entry.name
            season_dir   = season_entry.path
            fixtures_dir = os.path.join(season_dir, "fixtures")
            players_dir  = os.path.join(season_dir, "players")

            if not os.path.isdir(fixtures_dir):
                if verbose:
                    print(f"[WARN] No fixtures dir for {season_label}")
                continue

            pid_to_path = find_player_dirs(players_dir)

            # ------------------------------------------------------------------
            # Pass 1: scan all fixtures in this season to build stage map and
            #         fixture stats cache.  This reads data.json + statistics.json
            #         for every fixture.
            # ------------------------------------------------------------------
            if verbose:
                print(f"[PASS 1] Scanning {season_label} …")

            fixture_infos: List[FixtureInfo] = scan_season_fixtures(season_dir)
            stage_map = build_stage_map(fixture_infos)

            # Build a lookup: fixture_dir_path → FixtureInfo for quick access
            fi_by_dir: Dict[str, FixtureInfo] = {fi.fix_dir: fi for fi in fixture_infos}

            # ------------------------------------------------------------------
            # Between seasons: reset form history so each season starts fresh.
            # cform_tracker intentionally NOT reset — cross-season continuity.
            # ------------------------------------------------------------------
            tracker.reset()

            # ------------------------------------------------------------------
            # Pass 2: iterate fixtures CHRONOLOGICALLY, build rows.
            # ------------------------------------------------------------------
            # Sort by timestamp string (ISO format sorts correctly lexicographically).
            fixture_infos_sorted = sorted(fixture_infos, key=lambda fi: fi.ts)

            season_written  = 0
            season_skipped  = 0
            season_processed = 0

            for fi in fixture_infos_sorted:
                season_processed += 1

                # --- Only process finished matches (state_id == 5) ---
                if fi.state_id != 5:
                    season_skipped += 1
                    continue

                # Read form BEFORE this match (team's history so far this season)
                home_form  = tracker.get_form(fi.home_tid)
                away_form  = tracker.get_form(fi.away_tid)
                home_cform = cform_tracker.get_form(fi.home_tid)
                away_cform = cform_tracker.get_form(fi.away_tid)

                # --- Try to load lineup ---
                lineup_path = os.path.join(fi.fix_dir, "lineup.json")
                if not os.path.isfile(lineup_path):
                    # Fixture happened but we have no lineup → skip row, update form
                    if verbose:
                        base = os.path.basename(fi.fix_dir)
                        print(f"[SKIP] Missing lineup for {base} in {season_label}")
                    tracker.update(fi.home_tid, fi.away_tid, fi.stats)
                    cform_tracker.update(fi.home_tid, fi.away_tid, fi.stats)
                    season_skipped += 1
                    continue

                lineup_payload  = read_json(lineup_path)
                lineup_players  = parse_lineup_players(lineup_payload or {})

                home_line: Dict[str, List[Dict]] = {"GK": [], "DF": [], "MF": [], "ATK": []}
                away_line: Dict[str, List[Dict]] = {"GK": [], "DF": [], "MF": [], "ATK": []}

                for p in lineup_players:
                    pos_code = POSITION_ID_TO_CODE.get(p["position_id"])
                    if not pos_code:
                        continue
                    if p["team_id"] == fi.home_tid:
                        home_line[pos_code].append(p)
                    elif p["team_id"] == fi.away_tid:
                        away_line[pos_code].append(p)

                # Sort each line
                for k in list(home_line.keys()):
                    home_line[k] = _sort_line(home_line[k], pid_to_path)
                for k in list(away_line.keys()):
                    away_line[k] = _sort_line(away_line[k], pid_to_path)

                # --- Check slot limits (same skip logic as original) ---
                exceeds = (
                    len(home_line["DF"]) > max_df or
                    len(home_line["MF"]) > max_mf or
                    len(home_line["ATK"]) > max_atk or
                    len(away_line["DF"]) > max_df or
                    len(away_line["MF"]) > max_mf or
                    len(away_line["ATK"]) > max_atk
                )
                if exceeds:
                    if verbose:
                        base = os.path.basename(fi.fix_dir)
                        print(
                            f"[SKIP] Exceeds slot limits in {base} ({season_label}) | "
                            f"HOME: DF={len(home_line['DF'])}/{max_df}, "
                            f"MF={len(home_line['MF'])}/{max_mf}, "
                            f"ATK={len(home_line['ATK'])}/{max_atk} | "
                            f"AWAY: DF={len(away_line['DF'])}/{max_df}, "
                            f"MF={len(away_line['MF'])}/{max_mf}, "
                            f"ATK={len(away_line['ATK'])}/{max_atk}"
                        )
                    # Match happened → update form even though we skip the row
                    tracker.update(fi.home_tid, fi.away_tid, impute_stats(fi.stats))
                    cform_tracker.update(fi.home_tid, fi.away_tid, impute_stats(fi.stats))
                    season_skipped += 1
                    continue

                # --- Build player feature cells (original logic) ---
                home_feat = _build_side_feature_cells(
                    home_line, pid_to_path,
                    max_df, max_mf, max_atk,
                    gk_stats_n, df_stats_n, mf_stats_n, atk_stats_n,
                )
                away_feat = _build_side_feature_cells(
                    away_line, pid_to_path,
                    max_df, max_mf, max_atk,
                    gk_stats_n, df_stats_n, mf_stats_n, atk_stats_n,
                )

                # --- Form cells (28 values) ---
                form_cells  = form_cells_for_row(home_form, away_form)
                cform_cells = cform_cells_for_row(home_cform, away_cform)

                # --- Stage cell (1 value, NaN if stage_id not in map) ---
                stage_val = stage_map.get(fi.stage_id, math.nan)

                # --- Impute missing stats once (used for targets + form update) ---
                imputed = impute_stats(fi.stats)

                # --- Target cells ---
                tgt_cells = []
                for k in TARGET_KEYS:
                    tgt_cells.append(imputed[f"HOME_{k}"])
                    tgt_cells.append(imputed[f"AWAY_{k}"])

                # --- Assemble row ---
                fixture_ts_str = os.path.basename(fi.fix_dir)   # dir name is timestamp_id
                row = (
                    [season_label, fi.fixture_id, fixture_ts_str,
                     fi.home_tid,  fi.away_tid]
                    + home_feat
                    + away_feat
                    + form_cells
                    + cform_cells
                    + [stage_val]
                    + tgt_cells
                )
                writer.writerow(row)
                season_written += 1

                # --- Update form AFTER writing the row (imputed stats) ---
                tracker.update(fi.home_tid, fi.away_tid, imputed)
                cform_tracker.update(fi.home_tid, fi.away_tid, imputed)

            total_processed += season_processed
            total_written   += season_written
            total_skipped   += season_skipped

            if verbose:
                print(
                    f"[SEASON {season_label}] written={season_written}  "
                    f"skipped={season_skipped}  processed={season_processed}"
                )

    if verbose:
        print(f"\n[DONE] Total rows written: {total_written}  "
              f"(processed={total_processed}, skipped={total_skipped})")
        print(f"[OUTPUT] {os.path.abspath(out_path)}")

    return os.path.abspath(out_path)


# =============================================================================
# ==  CLI ENTRY POINT  ========================================================
# =============================================================================

DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", str(_WORKSPACE_ROOT / "data"))

def _env_list(name: str, default: List[str]) -> List[str]:
    v = os.environ.get(name, "")
    if not v.strip():
        return default
    parts = [p.strip() for p in v.split(",")]
    return [p for p in parts if p]


if __name__ == "__main__":
    START_SEASON = os.environ.get("TRAIN_START_SEASON", "2017-2018_7907").strip()
    END_SEASON   = os.environ.get("TRAIN_END_SEASON",   "2025-2026_25580").strip()
    if not START_SEASON or not END_SEASON:
        raise ValueError(
            "Please set TRAIN_START_SEASON and TRAIN_END_SEASON, "
            "e.g. '2017-2018_7907' and '2025-2026_25580'."
        )

    GK_STATS  = _env_list("GK_STATS",  ["GOALS_CONCEDED"])
    DF_STATS  = _env_list("DF_STATS",  ["GOALS_CONCEDED", "MINUTES_PLAYED", "APPEARANCES"])
    MF_STATS  = _env_list("MF_STATS",  ["GOALS_CONCEDED", "MINUTES_PLAYED", "APPEARANCES"])
    ATK_STATS = _env_list("ATK_STATS", [
        "MINUTES_PLAYED", "APPEARANCES", "GOALS_CONCEDED",
        "SUBSTITUTIONS_IN", "SUBSTITUTIONS_OUT",
    ])

    MAX_DF      = int(os.environ.get("MAX_DF",       "6"))
    MAX_MF      = int(os.environ.get("MAX_MF",       "6"))
    MAX_ATK     = int(os.environ.get("MAX_ATK",      "4"))
    FORM_WINDOW = int(os.environ.get("FORM_WINDOW",  "5"))
    OUT_DIR     = os.environ.get("TRAIN_OUT_DIR", str(_WORKSPACE_ROOT / "train_tables"))

    print(f"[CONFIG] DATA_ROOT={DEFAULT_DATA_ROOT}")
    print(f"[CONFIG] Seasons: {START_SEASON} → {END_SEASON}")
    print(f"[CONFIG] GK={GK_STATS} DF={DF_STATS} MF={MF_STATS} ATK={ATK_STATS}")
    print(f"[CONFIG] slots(1-{MAX_DF}-{MAX_MF}-{MAX_ATK})  form_window={FORM_WINDOW}")

    out_csv = build_train_table_v2(
        data_root    = DEFAULT_DATA_ROOT,
        start_season = START_SEASON,
        end_season   = END_SEASON,
        gk_stats     = GK_STATS,
        df_stats     = DF_STATS,
        mf_stats     = MF_STATS,
        atk_stats    = ATK_STATS,
        max_df       = MAX_DF,
        max_mf       = MAX_MF,
        max_atk      = MAX_ATK,
        form_window  = FORM_WINDOW,
        out_dir      = OUT_DIR,
        verbose      = True,
    )
    print(f"[OUTPUT] {out_csv}")
