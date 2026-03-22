# build_single_input_row.py
# - Writes input.csv next to this file (same folder).
# - Backend entry: we receive home_team_id and away_team_id.
# - For each team we find the most recent *past* fixture (played, any location),
#   extract that team's lineup & player stats, and build a single-row CSV.
# - The CSV schema is 1:1 with the old training schema (same header, including targets),
#   but metadata (except team IDs) and targets are filled with NaN.

import os, re, csv, json, math
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

# --------- Config & mapping (same as train_fixed_slots.py) ---------
SEASON_DIR_RE = re.compile(r"^(?P<y1>\d{4})-(?P<y2>\d{4})_(?P<season_id>\d+)$")
# fixtures folder name: "YYYY-MM-DDThh-mm-ss_fixtureId"
FIXTURE_DIR_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})_(?P<fixture_id>\d+)$"
)

POSITION_ID_TO_CODE = {24: "GK", 25: "DF", 26: "MF", 27: "ATK"}
TARGET_KEYS = [
    "GOALS",
    "CORNERS",
    "YELLOWCARDS",
    "SHOTS_ON_TARGET",
    "FOULS",
    "OFFSIDES",
    "REDCARDS",
]

DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "data")


# ---------- helpers for seasons / fixtures ----------

def is_valid_season_dir(name: str) -> bool:
    m = SEASON_DIR_RE.match(name)
    if not m:
        return False
    y1, y2 = int(m.group("y1")), int(m.group("y2"))
    return y2 == y1 + 1


def season_key(name: str) -> Tuple[int, int, int]:
    m = SEASON_DIR_RE.match(name)
    if not m:
        return (0, 0, 0)
    return (int(m.group("y1")), int(m.group("y2")), int(m.group("season_id")))


def norm_key(s: str) -> str:
    s = str(s).replace("-", " ").replace("/", " ").replace("\t", " ")
    return "_".join([x for x in s.strip().split() if x]).upper()


def read_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_fixture_ts(dirname: str) -> Optional[datetime]:
    """Extract datetime from fixture folder name."""
    m = FIXTURE_DIR_RE.match(dirname)
    if not m:
        return None
    ts_str = m.group("ts")
    try:
        return datetime.strptime(ts_str, "%Y-%m-%dT%H-%M-%S")
    except Exception:
        return None


def fixture_has_team(data_path: str, team_id: int) -> bool:
    """Return True if the team with team_id participates in this fixture."""
    payload = read_json(data_path)
    if not payload:
        return False
    data = payload.get("data")
    if not isinstance(data, dict):
        return False
    parts = data.get("participants")
    if not isinstance(parts, list):
        return False
    for p in parts:
        if not isinstance(p, dict):
            continue
        tid = p.get("id")
        try:
            tid_int = int(tid) if tid is not None else None
        except Exception:
            continue
        if tid_int == team_id:
            return True
    return False


def find_latest_fixture_for_team(
    data_root: str, team_id: int
) -> Tuple[str, str, str]:
    """
    Search all seasons/fixtures in reverse chronological order and return:
    (season_label, season_dir, fixture_dir_path) for the most recent *past* fixture
    in which team_id appears (home or away). Raises FileNotFoundError if none found.
    """
    now = datetime.now()
    seasons = [
        e for e in os.scandir(data_root) if e.is_dir() and is_valid_season_dir(e.name)
    ]
    # reverse chronological (latest seasons first)
    seasons.sort(key=lambda e: season_key(e.name), reverse=True)

    for s in seasons:
        season_label = s.name
        season_dir = s.path
        fixtures_dir = os.path.join(season_dir, "fixtures")
        if not os.path.isdir(fixtures_dir):
            continue

        # collect fixtures with valid timestamp in the *past*
        fixtures: List[Tuple[datetime, os.DirEntry]] = []
        for fe in os.scandir(fixtures_dir):
            if not fe.is_dir():
                continue
            ts = parse_fixture_ts(fe.name)
            if ts is None:
                continue
            if ts > now:
                # skip fixtures in the future
                continue
            fixtures.append((ts, fe))

        # newest first
        fixtures.sort(key=lambda x: x[0], reverse=True)

        for ts, fe in fixtures:
            fix_dir = fe.path
            data_path = os.path.join(fix_dir, "data.json")
            lineup_path = os.path.join(fix_dir, "lineup.json")

            if not (os.path.isfile(data_path) and os.path.isfile(lineup_path)):
                continue

            if not fixture_has_team(data_path, team_id):
                continue

            # found most recent past fixture for this team
            return (season_label, season_dir, fix_dir)

    raise FileNotFoundError(
        f"No past fixture found for team_id={team_id} under {data_root}"
    )


def find_player_dirs(season_players_dir: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if not os.path.isdir(season_players_dir):
        return mapping
    for entry in os.scandir(season_players_dir):
        if not entry.is_dir():
            continue
        name_part, sep, suffix = entry.name.rpartition("_")
        if sep and suffix.isdigit():
            mapping[int(suffix)] = entry.path
    return mapping


# ---------- player stats loaders (same as train) ----------

def _accumulate_stat(out: Dict[str, float], key_n: str, val: Optional[float]):
    if val is None:
        return
    try:
        v = float(val)
    except Exception:
        return
    out[key_n] = out.get(key_n, 0.0) + v


def _extract_numeric_from_dict(
    dct: Dict,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if isinstance(dct, dict) and "total" in dct:
        try:
            return (float(dct["total"]), None, None)
        except Exception:
            pass
    if isinstance(dct, dict) and "value" in dct and isinstance(
        dct["value"], (int, float, str)
    ):
        try:
            return (float(dct["value"]), None, None)
        except Exception:
            pass
    if isinstance(dct, dict) and (("in" in dct) or ("out" in dct)):
        vin = dct.get("in", 0) or 0
        vout = dct.get("out", 0) or 0
        try:
            vin = float(vin)
        except Exception:
            vin = 0.0
        try:
            vout = float(vout)
        except Exception:
            vout = 0.0
        return (None, vin, vout)
    return (None, None, None)


def _accumulate_from_detail(out: Dict[str, float], detail: Dict):
    if not isinstance(detail, dict):
        return
    t = detail.get("type") or {}
    key = t.get("developer_name") or t.get("code") or t.get("name")
    if not key:
        return
    key_n = norm_key(key)

    val = detail.get("value")
    handled = False
    if isinstance(val, dict):
        total, vin, vout = _extract_numeric_from_dict(val)
        if total is not None:
            _accumulate_stat(out, key_n, total)
            handled = True
        elif (vin is not None) or (vout is not None):
            _accumulate_stat(out, f"{key_n}_IN", vin or 0.0)
            _accumulate_stat(out, f"{key_n}_OUT", vout or 0.0)
            _accumulate_stat(out, key_n, (vin or 0.0) + (vout or 0.0))
            handled = True
    elif isinstance(val, (int, float, str)):
        _accumulate_stat(out, key_n, val)
        handled = True

    if not handled:
        d2 = detail.get("data")
        if isinstance(d2, dict):
            total, vin, vout = _extract_numeric_from_dict(d2)
            if total is not None:
                _accumulate_stat(out, key_n, total)
            elif (vin is not None) or (vout is not None):
                _accumulate_stat(out, f"{key_n}_IN", vin or 0.0)
                _accumulate_stat(out, f"{key_n}_OUT", vout or 0.0)
                _accumulate_stat(out, key_n, (vin or 0.0) + (vout or 0.0))
        elif isinstance(d2, (int, float, str)):
            _accumulate_stat(out, key_n, d2)


def load_player_last_year_stats(player_dir: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not player_dir:
        return out
    payload = read_json(os.path.join(player_dir, "last_year_statistics.json"))
    if not payload:
        return out
    data = payload.get("data", payload)
    stats_list = None
    if isinstance(data, dict):
        stats_list = data.get("statistics")
    elif isinstance(data, list):
        stats_list = data
    if not isinstance(stats_list, list):
        stats_list = payload.get("statistics", [])
    if not isinstance(stats_list, list):
        return out
    for stat_obj in stats_list:
        if not isinstance(stat_obj, dict):
            continue
        details = stat_obj.get("details") or stat_obj.get("statistics") or []
        if not isinstance(details, list):
            continue
        for d in details:
            _accumulate_from_detail(out, d)
    return out


def load_player_current_stats(player_dir: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not player_dir:
        return out
    payload = read_json(os.path.join(player_dir, "current_statistics.json"))
    if not payload:
        return out
    entries = (
        [payload]
        if isinstance(payload, dict)
        else [e for e in payload if isinstance(e, dict)]
        if isinstance(payload, list)
        else []
    )
    for entry in entries:
        details = entry.get("details") or entry.get("statistics") or []
        if not isinstance(details, list):
            continue
        for d in details:
            _accumulate_from_detail(out, d)
    return out


def load_player_stats_with_current_fallback(
    player_dir: str, use_current_fallback: bool = True
) -> Dict[str, float]:
    base = load_player_last_year_stats(player_dir)
    if not use_current_fallback:
        return base
    cur = load_player_current_stats(player_dir)
    for k, v in cur.items():
        if k not in base:
            base[k] = v
    return base


# ---------- lineups ----------

def parse_lineup_players(lineup_payload: Dict) -> List[Dict[str, Any]]:
    players: List[Dict[str, Any]] = []

    def consider(entry: Dict):
        try:
            type_id = int(entry.get("type_id"))
            if type_id != 11:
                return
            pid = int(entry.get("player_id"))
            tid = int(entry.get("team_id"))
            pos = int(entry.get("position_id"))
        except Exception:
            return
        fpos = entry.get("formation_position")
        ffpos = (
            int(fpos) if isinstance(fpos, (int, str)) and str(fpos).isdigit() else None
        )
        players.append(
            {
                "player_id": pid,
                "team_id": tid,
                "position_id": pos,
                "formation_position": ffpos,
                "type_id": 11,
            }
        )

    if isinstance(lineup_payload, dict):
        d = lineup_payload.get("data")
        if isinstance(d, dict):
            lineups = d.get("lineups") or d.get("lineup")
            if isinstance(lineups, list):
                for e in lineups:
                    if isinstance(e, dict):
                        consider(e)
        for key in ("lineups", "lineup", "players"):
            arr = lineup_payload.get(key)
            if isinstance(arr, list):
                for e in arr:
                    if isinstance(e, dict):
                        consider(e)

    uniq: Dict[Tuple[int, int], Dict] = {}
    for p in players:
        uniq[(p["player_id"], p["team_id"])] = p
    return list(uniq.values())


def choose_sort_key_for_line(last_year_stats: Dict[str, float]) -> tuple:
    if "MINUTES_PLAYED" in last_year_stats:
        return (1, float(last_year_stats["MINUTES_PLAYED"]))
    if "APPEARANCES" in last_year_stats:
        return (0, float(last_year_stats["APPEARANCES"]))
    return (-1, 0.0)


def _sort_line(line_players: List[Dict], pid_to_path: Dict[int, str]) -> List[Dict]:
    decorated = []
    for p in line_players:
        last_stats = {}
        pdir = pid_to_path.get(p["player_id"])
        if pdir:
            last_stats = load_player_last_year_stats(pdir)
        fb = choose_sort_key_for_line(last_stats)
        fpos = p.get("formation_position")
        if isinstance(fpos, int):
            decorated.append((0, fpos, fb, p))
        else:
            decorated.append((1, 10**9, fb, p))
    decorated.sort(key=lambda x: (x[0], x[1], -x[2][0], -x[2][1]))
    return [t[-1] for t in decorated]


def slot_id_col(pos: str, side: str, idx: Optional[int] = None) -> str:
    pos = pos.lower()
    side = side.lower()
    if pos == "gk":
        return f"gk_{side}_player_id"
    if idx is None:
        idx = 1
    return f"{pos}{idx}_{side}_player_id"


# ---------- header builder ----------

def _env_list(name: str, default: List[str]) -> List[str]:
    v = os.environ.get(name, "")
    if not v.strip():
        return default
    parts = [p.strip() for p in v.split(",")]
    return [p for p in parts if p]


def _build_header(
    gk_stats_n, df_stats_n, mf_stats_n, atk_stats_n, max_df, max_mf, max_atk
) -> List[str]:
    meta = ["season_label", "fixture_id", "fixture_ts", "home_team_id", "away_team_id"]
    feats: List[str] = []
    # HOME
    feats += [slot_id_col("gk", "home")]
    feats += [f"GK_HOME_{s}" for s in gk_stats_n]
    for i in range(1, max_df + 1):
        feats += [slot_id_col("df", "home", i)]
        feats += [f"DF{i}_HOME_{s}" for s in df_stats_n]
    feats += ["NO_OF_DF_HOME"]
    for i in range(1, max_mf + 1):
        feats += [slot_id_col("mf", "home", i)]
        feats += [f"MF{i}_HOME_{s}" for s in mf_stats_n]
    feats += ["NO_OF_MF_HOME"]
    for i in range(1, max_atk + 1):
        feats += [slot_id_col("atk", "home", i)]
        feats += [f"ATK{i}_HOME_{s}" for s in atk_stats_n]
    feats += ["NO_OF_ATK_HOME"]
    # AWAY
    feats += [slot_id_col("gk", "away")]
    feats += [f"GK_AWAY_{s}" for s in gk_stats_n]
    for i in range(1, max_df + 1):
        feats += [slot_id_col("df", "away", i)]
        feats += [f"DF{i}_AWAY_{s}" for s in df_stats_n]
    feats += ["NO_OF_DF_AWAY"]
    for i in range(1, max_mf + 1):
        feats += [slot_id_col("mf", "away", i)]
        feats += [f"MF{i}_AWAY_{s}" for s in mf_stats_n]
    feats += ["NO_OF_MF_AWAY"]
    for i in range(1, max_atk + 1):
        feats += [slot_id_col("atk", "away", i)]
        feats += [f"ATK{i}_AWAY_{s}" for s in atk_stats_n]
    feats += ["NO_OF_ATK_AWAY"]
    targets = [f"HOME_{k}" for k in TARGET_KEYS] + [
        f"AWAY_{k}" for k in TARGET_KEYS
    ]
    return meta + feats + targets


def _side_slots(
    side_line: Dict[str, List[Dict]],
    pid_to_path: Dict[int, str],
    gk_stats_n,
    df_stats_n,
    mf_stats_n,
    atk_stats_n,
    max_df,
    max_mf,
    max_atk,
) -> List[Any]:
    row_vals: List[Any] = []
    # GK
    gk_list = side_line["GK"][:1]
    if gk_list:
        pid = gk_list[0]["player_id"]
        pdir = pid_to_path.get(pid)
        row_vals.append(pid)
        stats_map = (
            load_player_stats_with_current_fallback(pdir, True) if pdir else {}
        )
        for s in gk_stats_n:
            row_vals.append(stats_map.get(s, math.nan))
    else:
        row_vals.append("")
        for _ in gk_stats_n:
            row_vals.append(math.nan)

    # DF
    df_list = side_line["DF"][:max_df]
    for slot in range(max_df):
        if slot < len(df_list):
            pid = df_list[slot]["player_id"]
            pdir = pid_to_path.get(pid)
            stats_map = (
                load_player_stats_with_current_fallback(pdir, True) if pdir else {}
            )
            row_vals.append(pid)
            for s in df_stats_n:
                row_vals.append(stats_map.get(s, math.nan))
        else:
            row_vals.append("")
            for _ in df_stats_n:
                row_vals.append(math.nan)
    row_vals.append(len(side_line["DF"]))

    # MF
    mf_list = side_line["MF"][:max_mf]
    for slot in range(max_mf):
        if slot < len(mf_list):
            pid = mf_list[slot]["player_id"]
            pdir = pid_to_path.get(pid)
            stats_map = (
                load_player_stats_with_current_fallback(pdir, True) if pdir else {}
            )
            row_vals.append(pid)
            for s in mf_stats_n:
                row_vals.append(stats_map.get(s, math.nan))
        else:
            row_vals.append("")
            for _ in mf_stats_n:
                row_vals.append(math.nan)
    row_vals.append(len(side_line["MF"]))

    # ATK
    atk_list = side_line["ATK"][:max_atk]
    for slot in range(max_atk):
        if slot < len(atk_list):
            pid = atk_list[slot]["player_id"]
            pdir = pid_to_path.get(pid)
            stats_map = (
                load_player_stats_with_current_fallback(pdir, True) if pdir else {}
            )
            row_vals.append(pid)
            for s in atk_stats_n:
                row_vals.append(stats_map.get(s, math.nan))
        else:
            row_vals.append("")
            for _ in atk_stats_n:
                row_vals.append(math.nan)
    row_vals.append(len(side_line["ATK"]))
    return row_vals


def build_side_from_latest_fixture(
    data_root: str,
    team_id: int,
    gk_stats_n: List[str],
    df_stats_n: List[str],
    mf_stats_n: List[str],
    atk_stats_n: List[str],
    max_df: int,
    max_mf: int,
    max_atk: int,
) -> List[Any]:
    """
    For a given team_id, find its most recent past fixture and build the
    feature vector (GK/DF/MF/ATK slots) for that team only.
    """
    season_label, season_dir, fix_dir = find_latest_fixture_for_team(
        data_root, team_id
    )
    # season_label is not used here directly, but returned for possible debugging.
    _ = season_label

    lineup_path = os.path.join(fix_dir, "lineup.json")
    lineup_payload = read_json(lineup_path) or {}
    lineup_players = parse_lineup_players(lineup_payload)

    side_line = {"GK": [], "DF": [], "MF": [], "ATK": []}
    for p in lineup_players:
        if p.get("team_id") != team_id:
            continue
        pos_code = POSITION_ID_TO_CODE.get(p.get("position_id"))
        if not pos_code:
            continue
        side_line[pos_code].append(p)

    # sort each line according to formation position / minutes played
    for k in list(side_line.keys()):
        side_line[k] = _sort_line(side_line[k], find_player_dirs(os.path.join(season_dir, "players")))

    pid_to_path = find_player_dirs(os.path.join(season_dir, "players"))

    return _side_slots(
        side_line,
        pid_to_path,
        gk_stats_n,
        df_stats_n,
        mf_stats_n,
        atk_stats_n,
        max_df,
        max_mf,
        max_atk,
    )


def build_single_row_for_team_ids(
    data_root: str,
    home_team_id: int,
    away_team_id: int,
    gk_stats: List[str],
    df_stats: List[str],
    mf_stats: List[str],
    atk_stats: List[str],
    max_df: int,
    max_mf: int,
    max_atk: int,
) -> Tuple[List[str], List[Any]]:
    """
    Main builder: given home_team_id and away_team_id, construct one row
    for input.csv with the same header as the training schema.

    - HOME features are built from the last past fixture of home_team_id.
    - AWAY features are built from the last past fixture of away_team_id.
    - Meta columns (season_label, fixture_id, fixture_ts) are filled with NaN.
    - home_team_id / away_team_id are filled with the provided IDs.
    - All target columns are set to NaN.
    """
    gk_stats_n = [norm_key(s) for s in gk_stats]
    df_stats_n = [norm_key(s) for s in df_stats]
    mf_stats_n = [norm_key(s) for s in mf_stats]
    atk_stats_n = [norm_key(s) for s in atk_stats]

    header = _build_header(
        gk_stats_n, df_stats_n, mf_stats_n, atk_stats_n, max_df, max_mf, max_atk
    )

    # Build side features
    home_vals = build_side_from_latest_fixture(
        data_root,
        home_team_id,
        gk_stats_n,
        df_stats_n,
        mf_stats_n,
        atk_stats_n,
        max_df,
        max_mf,
        max_atk,
    )
    away_vals = build_side_from_latest_fixture(
        data_root,
        away_team_id,
        gk_stats_n,
        df_stats_n,
        mf_stats_n,
        atk_stats_n,
        max_df,
        max_mf,
        max_atk,
    )

    row: List[Any] = []
    # Meta: season_label, fixture_id, fixture_ts -> NaN; team IDs real
    row.append(math.nan)          # season_label
    row.append(math.nan)          # fixture_id
    row.append(math.nan)          # fixture_ts
    row.append(home_team_id)      # home_team_id
    row.append(away_team_id)      # away_team_id

    # Features
    row += home_vals
    row += away_vals

    # Targets: we don't have a real home vs away match here, so all NaN
    for _ in TARGET_KEYS:
        row.append(math.nan)  # HOME_*
    for _ in TARGET_KEYS:
        row.append(math.nan)  # AWAY_*

    return header, row


# -------------- CSV writer + main (useful for testing) --------------

def _write_single_csv(header: List[str], row: List[Any], out_path: str):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerow(row)


def main():
    # Stats lists & max slots remain configurable via env (same style as before)
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

    # For testing from CLI you can set HOME_TEAM_ID and AWAY_TEAM_ID in env.
    home_team_id_str = 3444
    away_team_id_str = 8
    if not home_team_id_str or not away_team_id_str:
        raise RuntimeError(
            "HOME_TEAM_ID and AWAY_TEAM_ID must be set in environment for CLI usage."
        )

    home_team_id = 3444
    away_team_id = 8

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

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.csv")
    _write_single_csv(header, row, out_path)
    print(f"[OK] Wrote one-line CSV: {out_path}")


if __name__ == "__main__":
    main()
