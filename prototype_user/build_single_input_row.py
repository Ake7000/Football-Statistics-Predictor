# build_single_input_row.py
# - Scrie raw_input.csv lângă acest fișier (același folder).
# - Fără CLI: FIXTURE_DIRNAME este hardcodat mai jos.
# - Schema și logica sunt aliniate 1:1 cu train_fixed_slots.py.

import os, re, csv, json, math, hashlib
from typing import Dict, List, Tuple, Optional, Any

# --------- INPUT HARDCODAT (schimbă după nevoie) ---------
FIXTURE_DIRNAME = "2025-11-05T20-00-00_19568506"  # ex.: "YYYY-MM-DDThh-mm-ss_fixtureId"

# --------- Config și mapping identic cu train_fixed_slots.py ---------
SEASON_DIR_RE = re.compile(r"^(?P<y1>\d{4})-(?P<y2>\d{4})_(?P<season_id>\d+)$")
POSITION_ID_TO_CODE = {24: "GK", 25: "DF", 26: "MF", 27: "ATK"}
TARGET_KEYS = ["GOALS","CORNERS","YELLOWCARDS","SHOTS_ON_TARGET","FOULS","OFFSIDES","REDCARDS"]

def is_valid_season_dir(name: str) -> bool:
    m = SEASON_DIR_RE.match(name)
    if not m: return False
    y1, y2 = int(m.group("y1")), int(m.group("y2"))
    return y2 == y1 + 1

def season_key(name: str) -> Tuple[int,int,int]:
    m = SEASON_DIR_RE.match(name)
    if not m: return (0,0,0)
    return (int(m.group("y1")), int(m.group("y2")), int(m.group("season_id")))

def norm_key(s: str) -> str:
    s = str(s).replace("-", " ").replace("/", " ").replace("\t"," ")
    return "_".join([x for x in s.strip().split() if x]).upper()

def read_json(path: str) -> Optional[Dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def find_player_dirs(season_players_dir: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if not os.path.isdir(season_players_dir): return mapping
    for entry in os.scandir(season_players_dir):
        if not entry.is_dir(): continue
        name_part, sep, suffix = entry.name.rpartition("_")
        if sep and suffix.isdigit():
            mapping[int(suffix)] = entry.path
    return mapping

# ---------- player stats loaders (ca în train) ----------
def _accumulate_stat(out: Dict[str,float], key_n: str, val: Optional[float]):
    if val is None: return
    try: v = float(val)
    except Exception: return
    out[key_n] = out.get(key_n, 0.0) + v

def _extract_numeric_from_dict(dct: Dict) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if isinstance(dct, dict) and "total" in dct:
        try: return (float(dct["total"]), None, None)
        except Exception: pass
    if isinstance(dct, dict) and "value" in dct and isinstance(dct["value"], (int,float,str)):
        try: return (float(dct["value"]), None, None)
        except Exception: pass
    if isinstance(dct, dict) and (("in" in dct) or ("out" in dct)):
        vin = dct.get("in", 0) or 0
        vout = dct.get("out", 0) or 0
        try: vin = float(vin)
        except Exception: vin = 0.0
        try: vout = float(vout)
        except Exception: vout = 0.0
        return (None, vin, vout)
    return (None,None,None)

def _accumulate_from_detail(out: Dict[str,float], detail: Dict):
    if not isinstance(detail, dict): return
    t = detail.get("type") or {}
    key = t.get("developer_name") or t.get("code") or t.get("name")
    if not key: return
    key_n = norm_key(key)

    val = detail.get("value")
    handled = False
    if isinstance(val, dict):
        total, vin, vout = _extract_numeric_from_dict(val)
        if total is not None:
            _accumulate_stat(out, key_n, total); handled = True
        elif (vin is not None) or (vout is not None):
            _accumulate_stat(out, f"{key_n}_IN", vin or 0.0)
            _accumulate_stat(out, f"{key_n}_OUT", vout or 0.0)
            _accumulate_stat(out, key_n, (vin or 0.0)+(vout or 0.0)); handled = True
    elif isinstance(val, (int,float,str)):
        _accumulate_stat(out, key_n, val); handled = True

    if not handled:
        d2 = detail.get("data")
        if isinstance(d2, dict):
            total, vin, vout = _extract_numeric_from_dict(d2)
            if total is not None:
                _accumulate_stat(out, key_n, total)
            elif (vin is not None) or (vout is not None):
                _accumulate_stat(out, f"{key_n}_IN", vin or 0.0)
                _accumulate_stat(out, f"{key_n}_OUT", vout or 0.0)
                _accumulate_stat(out, key_n, (vin or 0.0)+(vout or 0.0))
        elif isinstance(d2, (int,float,str)):
            _accumulate_stat(out, key_n, d2)

def load_player_last_year_stats(player_dir: str) -> Dict[str,float]:
    out: Dict[str,float] = {}
    if not player_dir: return out
    payload = read_json(os.path.join(player_dir, "last_year_statistics.json"))
    if not payload: return out
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
        if not isinstance(stat_obj, dict): continue
        details = stat_obj.get("details") or stat_obj.get("statistics") or []
        if not isinstance(details, list): continue
        for d in details:
            _accumulate_from_detail(out, d)
    return out

def load_player_current_stats(player_dir: str) -> Dict[str,float]:
    out: Dict[str,float] = {}
    if not player_dir: return out
    payload = read_json(os.path.join(player_dir, "current_statistics.json"))
    if not payload: return out
    entries = [payload] if isinstance(payload, dict) else [e for e in payload if isinstance(e, dict)] if isinstance(payload, list) else []
    for entry in entries:
        details = entry.get("details") or entry.get("statistics") or []
        if not isinstance(details, list): continue
        for d in details:
            _accumulate_from_detail(out, d)
    return out

def load_player_stats_with_current_fallback(player_dir: str, use_current_fallback: bool=True) -> Dict[str,float]:
    base = load_player_last_year_stats(player_dir)
    if not use_current_fallback: return base
    cur = load_player_current_stats(player_dir)
    for k, v in cur.items():
        if k not in base: base[k] = v
    return base

# ---------- lineups & fixtures ----------
def parse_lineup_players(lineup_payload: Dict) -> List[Dict[str,Any]]:
    players: List[Dict[str,Any]] = []
    def consider(entry: Dict):
        try:
            type_id = int(entry.get("type_id"))
            if type_id != 11: return
            pid = int(entry.get("player_id"))
            tid = int(entry.get("team_id"))
            pos = int(entry.get("position_id"))
        except Exception:
            return
        fpos = entry.get("formation_position")
        ffpos = int(fpos) if isinstance(fpos, (int,str)) and str(fpos).isdigit() else None
        players.append({"player_id": pid, "team_id": tid, "position_id": pos, "formation_position": ffpos, "type_id": 11})

    if isinstance(lineup_payload, dict):
        d = lineup_payload.get("data")
        if isinstance(d, dict):
            lineups = d.get("lineups") or d.get("lineup")
            if isinstance(lineups, list):
                for e in lineups:
                    if isinstance(e, dict): consider(e)
        for key in ("lineups","lineup","players"):
            arr = lineup_payload.get(key)
            if isinstance(arr, list):
                for e in arr:
                    if isinstance(e, dict): consider(e)

    uniq: Dict[Tuple[int,int], Dict] = {}
    for p in players:
        uniq[(p["player_id"], p["team_id"])] = p
    return list(uniq.values())

def resolve_home_away_team_ids(fixture_dir: str) -> Optional[Tuple[int,int]]:
    data_path = os.path.join(fixture_dir, "data.json")
    payload = read_json(data_path)
    if payload and isinstance(payload.get("data"), dict):
        participants = payload["data"].get("participants")
        if isinstance(participants, list):
            home_id = away_id = None
            for p in participants:
                tid = p.get("id"); meta = p.get("meta") or {}
                loc = (meta.get("location") or "").lower()
                if tid is None: continue
                if loc == "home": home_id = int(tid)
                elif loc == "away": away_id = int(tid)
            if home_id is not None and away_id is not None:
                return (home_id, away_id)

    stat_path = os.path.join(fixture_dir, "statistics.json")
    sp = read_json(stat_path)
    if sp:
        parts = sp.get("participants")
        if isinstance(parts, list):
            home_id = away_id = None
            for p in parts:
                tid = p.get("id") or p.get("participant_id")
                loc = (p.get("meta",{}).get("location") or p.get("location") or "").lower()
                if tid is None: continue
                if loc == "home": home_id = int(tid)
                elif loc == "away": away_id = int(tid)
            if home_id is not None and away_id is not None:
                return (home_id, away_id)
    return None

def extract_targets(stat_payload: Dict, home_team_id: int, away_team_id: int) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {f"HOME_{k}": math.nan for k in TARGET_KEYS}
    results.update({f"AWAY_{k}": math.nan for k in TARGET_KEYS})

    def take_numeric(entry: Dict) -> Optional[float]:
        val = entry.get("value")
        if isinstance(val, dict) and "total" in val:
            try: return float(val["total"])
            except Exception: pass
        if isinstance(val, (int,float,str)):
            try: return float(val)
            except Exception: pass
        dct = entry.get("data")
        if isinstance(dct, dict):
            for key in ("value","total"):
                if key in dct:
                    try: return float(dct[key])
                    except Exception: pass
        elif isinstance(dct, (int,float,str)):
            try: return float(dct)
            except Exception: pass
        return None

    def consider(entry: Dict):
        t = entry.get("type") or {}
        key = t.get("developer_name") or t.get("code") or t.get("name")
        if not key: return
        key_n = norm_key(key)
        if key_n not in TARGET_KEYS: return

        loc = (entry.get("location") or "").lower().strip()
        if loc not in ("home","away"):
            pid = entry.get("participant_id", entry.get("team_id"))
            try: pid = int(pid) if pid is not None else None
            except Exception: pid = None
            if pid is None: return
            loc = "home" if pid == home_team_id else ("away" if pid == away_team_id else "")
        if loc not in ("home","away"): return

        v = take_numeric(entry)
        if v is None: return
        results[f"{loc.upper()}_{key_n}"] = v

    data = stat_payload.get("data")
    if isinstance(data, dict):
        stats_list = data.get("statistics")
        if isinstance(stats_list, list):
            for e in stats_list:
                if isinstance(e, dict): consider(e)
    elif isinstance(data, list):
        for e in data:
            if isinstance(e, dict): consider(e)
    return results

def choose_sort_key_for_line(last_year_stats: Dict[str,float]) -> tuple:
    if "MINUTES_PLAYED" in last_year_stats:
        return (1, float(last_year_stats["MINUTES_PLAYED"]))
    if "APPEARANCES" in last_year_stats:
        return (0, float(last_year_stats["APPEARANCES"]))
    return (-1, 0.0)

def slot_id_col(pos: str, side: str, idx: Optional[int]=None) -> str:
    pos = pos.lower(); side = side.lower()
    if pos == "gk": return f"gk_{side}_player_id"
    if idx is None: idx = 1
    return f"{pos}{idx}_{side}_player_id"

# ---------- header builder ----------
def _env_list(name: str, default: List[str]) -> List[str]:
    v = os.environ.get(name, "")
    if not v.strip(): return default
    parts = [p.strip() for p in v.split(",")]
    return [p for p in parts if p]

def _build_header(gk_stats_n, df_stats_n, mf_stats_n, atk_stats_n, max_df, max_mf, max_atk) -> List[str]:
    meta = ["season_label","fixture_id","fixture_ts","home_team_id","away_team_id"]
    feats: List[str] = []
    # HOME
    feats += [slot_id_col("gk","home")]
    feats += [f"GK_HOME_{s}" for s in gk_stats_n]
    for i in range(1, max_df+1):
        feats += [slot_id_col("df","home",i)]
        feats += [f"DF{i}_HOME_{s}" for s in df_stats_n]
    feats += ["NO_OF_DF_HOME"]
    for i in range(1, max_mf+1):
        feats += [slot_id_col("mf","home",i)]
        feats += [f"MF{i}_HOME_{s}" for s in mf_stats_n]
    feats += ["NO_OF_MF_HOME"]
    for i in range(1, max_atk+1):
        feats += [slot_id_col("atk","home",i)]
        feats += [f"ATK{i}_HOME_{s}" for s in atk_stats_n]
    feats += ["NO_OF_ATK_HOME"]
    # AWAY
    feats += [slot_id_col("gk","away")]
    feats += [f"GK_AWAY_{s}" for s in gk_stats_n]
    for i in range(1, max_df+1):
        feats += [slot_id_col("df","away",i)]
        feats += [f"DF{i}_AWAY_{s}" for s in df_stats_n]
    feats += ["NO_OF_DF_AWAY"]
    for i in range(1, max_mf+1):
        feats += [slot_id_col("mf","away",i)]
        feats += [f"MF{i}_AWAY_{s}" for s in mf_stats_n]
    feats += ["NO_OF_MF_AWAY"]
    for i in range(1, max_atk+1):
        feats += [slot_id_col("atk","away",i)]
        feats += [f"ATK{i}_AWAY_{s}" for s in atk_stats_n]
    feats += ["NO_OF_ATK_AWAY"]
    targets = [f"HOME_{k}" for k in TARGET_KEYS] + [f"AWAY_{k}" for k in TARGET_KEYS]
    return meta + feats + targets

def _sort_line(line_players: List[Dict], pid_to_path: Dict[int,str]) -> List[Dict]:
    decorated = []
    for p in line_players:
        last_stats = {}
        pdir = pid_to_path.get(p["player_id"])
        if pdir: last_stats = load_player_last_year_stats(pdir)
        fb = choose_sort_key_for_line(last_stats)
        fpos = p.get("formation_position")
        if isinstance(fpos, int):
            decorated.append((0, fpos, fb, p))
        else:
            decorated.append((1, 10**9, fb, p))
    decorated.sort(key=lambda x: (x[0], x[1], -x[2][0], -x[2][1]))
    return [t[-1] for t in decorated]

def _side_slots(side_line: Dict[str, List[Dict]], pid_to_path: Dict[int,str],
                gk_stats_n, df_stats_n, mf_stats_n, atk_stats_n,
                max_df, max_mf, max_atk) -> List[Any]:
    row_vals: List[Any] = []
    # GK
    gk_list = side_line["GK"][:1]
    if gk_list:
        pid = gk_list[0]["player_id"]; pdir = pid_to_path.get(pid)
        row_vals.append(pid)
        stats_map = load_player_stats_with_current_fallback(pdir, True) if pdir else {}
        for s in gk_stats_n: row_vals.append(stats_map.get(s, math.nan))
    else:
        row_vals.append("")
        for _ in gk_stats_n: row_vals.append(math.nan)

    # DF
    df_list = side_line["DF"][:max_df]
    for slot in range(max_df):
        if slot < len(df_list):
            pid = df_list[slot]["player_id"]; pdir = pid_to_path.get(pid)
            stats_map = load_player_stats_with_current_fallback(pdir, True) if pdir else {}
            row_vals.append(pid)
            for s in df_stats_n: row_vals.append(stats_map.get(s, math.nan))
        else:
            row_vals.append("")
            for _ in df_stats_n: row_vals.append(math.nan)
    row_vals.append(len(side_line["DF"]))

    # MF
    mf_list = side_line["MF"][:max_mf]
    for slot in range(max_mf):
        if slot < len(mf_list):
            pid = mf_list[slot]["player_id"]; pdir = pid_to_path.get(pid)
            stats_map = load_player_stats_with_current_fallback(pdir, True) if pdir else {}
            row_vals.append(pid)
            for s in mf_stats_n: row_vals.append(stats_map.get(s, math.nan))
        else:
            row_vals.append("")
            for _ in mf_stats_n: row_vals.append(math.nan)
    row_vals.append(len(side_line["MF"]))

    # ATK
    atk_list = side_line["ATK"][:max_atk]
    for slot in range(max_atk):
        if slot < len(atk_list):
            pid = atk_list[slot]["player_id"]; pdir = pid_to_path.get(pid)
            stats_map = load_player_stats_with_current_fallback(pdir, True) if pdir else {}
            row_vals.append(pid)
            for s in atk_stats_n: row_vals.append(stats_map.get(s, math.nan))
        else:
            row_vals.append("")
            for _ in atk_stats_n: row_vals.append(math.nan)
    row_vals.append(len(side_line["ATK"]))
    return row_vals

def build_single_row_for_fixture(
    data_root: str,
    fixture_dirname: str,
    gk_stats: List[str],
    df_stats: List[str],
    mf_stats: List[str],
    atk_stats: List[str],
    max_df: int,
    max_mf: int,
    max_atk: int,
) -> Tuple[List[str], List[Any]]:
    gk_stats_n = [norm_key(s) for s in gk_stats]
    df_stats_n = [norm_key(s) for s in df_stats]
    mf_stats_n = [norm_key(s) for s in mf_stats]
    atk_stats_n = [norm_key(s) for s in atk_stats]
    header = _build_header(gk_stats_n, df_stats_n, mf_stats_n, atk_stats_n, max_df, max_mf, max_atk)

    # localizează fixture-ul
    seasons = [e for e in os.scandir(data_root) if e.is_dir() and is_valid_season_dir(e.name)]
    seasons.sort(key=lambda e: season_key(e.name))
    found = None
    for s in seasons:
        fdir = os.path.join(s.path, "fixtures", fixture_dirname)
        if os.path.isdir(fdir):
            found = (s.name, s.path, fdir)
            break
    if not found:
        raise FileNotFoundError(f"Fixture folder '{fixture_dirname}' not found under {data_root}/<season>/fixtures/")

    season_label, season_dir, fix_dir = found
    data_path = os.path.join(fix_dir, "data.json")
    lineup_path = os.path.join(fix_dir, "lineup.json")
    stats_path = os.path.join(fix_dir, "statistics.json")

    if not (os.path.isfile(data_path) and os.path.isfile(lineup_path) and os.path.isfile(stats_path)):
        raise FileNotFoundError(f"Missing one of lineup/statistics/data in {fix_dir}")

    ha = resolve_home_away_team_ids(fix_dir)
    if not ha:
        raise RuntimeError(f"Cannot resolve home/away team ids for fixture {fixture_dirname}")
    home_tid, away_tid = ha

    pid_to_path = find_player_dirs(os.path.join(season_dir, "players"))
    lineup_payload = read_json(lineup_path) or {}
    lineup_players = parse_lineup_players(lineup_payload)
    home_line = {"GK": [], "DF": [], "MF": [], "ATK": []}
    away_line = {"GK": [], "DF": [], "MF": [], "ATK": []}
    for p in lineup_players:
        pos_code = POSITION_ID_TO_CODE.get(p["position_id"])
        if not pos_code: continue
        if p["team_id"] == home_tid: home_line[pos_code].append(p)
        elif p["team_id"] == away_tid: away_line[pos_code].append(p)

    for k in list(home_line.keys()):
        home_line[k] = _sort_line(home_line[k], pid_to_path)
    for k in list(away_line.keys()):
        away_line[k] = _sort_line(away_line[k], pid_to_path)

    data_payload = read_json(data_path) or {}
    fixture_id = None
    try:
        d = data_payload.get("data", {})
        fixture_id = d.get("id")
    except Exception:
        pass

    row: List[Any] = []
    row += [season_label, fixture_id, os.path.basename(fix_dir), home_tid, away_tid]
    row += _side_slots(home_line, pid_to_path, gk_stats_n, df_stats_n, mf_stats_n, atk_stats_n, max_df, max_mf, max_atk)
    row += _side_slots(away_line, pid_to_path, gk_stats_n, df_stats_n, mf_stats_n, atk_stats_n, max_df, max_mf, max_atk)
    stat_payload = read_json(stats_path) or {}
    tgt = extract_targets(stat_payload, home_tid, away_tid)
    for k in TARGET_KEYS: row.append(tgt.get(f"HOME_{k}", math.nan))
    for k in TARGET_KEYS: row.append(tgt.get(f"AWAY_{k}", math.nan))
    return header, row

# -------------- rulare fără CLI: scrie raw_input.csv lângă fișier --------------
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "data")

def _write_single_csv(header: List[str], row: List[Any], out_path: str):
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header); w.writerow(row)

def main():
    GK_STATS = _env_list("GK_STATS", ["GOALS_CONCEDED"])
    DF_STATS = _env_list("DF_STATS", ["GOALS_CONCEDED","MINUTES_PLAYED","APPEARANCES"])
    MF_STATS = _env_list("MF_STATS", ["GOALS_CONCEDED","MINUTES_PLAYED","APPEARANCES"])
    ATK_STATS = _env_list("ATK_STATS", ["MINUTES_PLAYED","APPEARANCES","GOALS_CONCEDED","SUBSTITUTIONS_IN","SUBSTITUTIONS_OUT"])

    MAX_DF = int(os.environ.get("MAX_DF", "6"))
    MAX_MF = int(os.environ.get("MAX_MF", "6"))
    MAX_ATK = int(os.environ.get("MAX_ATK", "4"))

    header, row = build_single_row_for_fixture(
        data_root=DEFAULT_DATA_ROOT,
        fixture_dirname=FIXTURE_DIRNAME,
        gk_stats=GK_STATS, df_stats=DF_STATS, mf_stats=MF_STATS, atk_stats=ATK_STATS,
        max_df=MAX_DF, max_mf=MAX_MF, max_atk=MAX_ATK,
    )

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input.csv")
    _write_single_csv(header, row, out_path)
    print(f"[OK] Wrote one-line CSV: {out_path}")

if __name__ == "__main__":
    main()
