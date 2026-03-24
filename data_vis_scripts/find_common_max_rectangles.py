# find_windows_fixed_fixtures_with_relax.py
import os
import re
import csv
from typing import List, Dict, Tuple, Optional

# ------------------------------
# Configuration
# ------------------------------
CATEGORIES = ["ATTACKER", "DEFENDER", "GOALKEEPER", "MIDFIELDER"]
REQUIRED_FIXTURE_STATS = [
    "GOALS",
    "CORNERS",
    "YELLOWCARDS",
    "SHOTS_ON_TARGET",
    "FOULS",
    "OFFSIDES",
    "REDCARDS",
]
SEASON_LABEL_RE = re.compile(r"^(?P<y1>\d{4})-(?P<y2>\d{4})_(?P<sid>\d+)$")


# ------------------------------
# I/O helpers
# ------------------------------
def read_pivot_csv(path: str) -> Tuple[List[str], List[str], List[List[float]]]:
    """
    Read a pivot CSV of shape:
      header: stat_key, <season1>, <season2>, ..., <seasonN>, avg_percent
    Returns:
      seasons (column order, without 'avg_percent'),
      stat_keys (row order),
      matrix (len(stat_keys) x len(seasons)) with float percents.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Pivot CSV not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or header[0].lower() != "stat_key":
            raise ValueError(f"Unexpected header in {path}: {header}")

        seasons = header[1:]
        if seasons and seasons[-1].lower() == "avg_percent":
            seasons = seasons[:-1]

        stat_keys: List[str] = []
        matrix: List[List[float]] = []
        for row in r:
            if not row:
                continue
            key = row[0].strip()
            if not key:
                continue
            vals = row[1:1+len(seasons)]
            try:
                perc = [float(x) if x != "" else 0.0 for x in vals]
            except Exception:
                perc = [0.0 for _ in vals]
            stat_keys.append(key)
            matrix.append(perc)

    return seasons, stat_keys, matrix


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_csv(path: str, header: List[str], rows: List[List[str]]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for row in rows:
            w.writerow(row)


# ------------------------------
# Season filtering / alignment
# ------------------------------
def season_start_year(season_label: str) -> Optional[int]:
    m = SEASON_LABEL_RE.match(season_label)
    return int(m.group("y1")) if m else None


def filter_seasons_since(
    seasons: List[str], matrix_cols_major: List[List[int]], min_start_y1: int
) -> Tuple[List[str], List[List[int]]]:
    """
    Keep only columns whose season start year >= min_start_y1. Preserve order.
    matrix_cols_major is rows x cols (rows = stats, cols = seasons).
    """
    keep = [i for i, s in enumerate(seasons) if (season_start_year(s) or 0) >= min_start_y1]
    if not keep:
        return [], []
    filt_seasons = [seasons[i] for i in keep]
    filt_matrix = [[row[i] for i in keep] for row in matrix_cols_major]
    return filt_seasons, filt_matrix


def filter_seasons_since_float(
    seasons: List[str], matrix_cols_major: List[List[float]], min_start_y1: int
) -> Tuple[List[str], List[List[float]]]:
    keep = [i for i, s in enumerate(seasons) if (season_start_year(s) or 0) >= min_start_y1]
    if not keep:
        return [], []
    filt_seasons = [seasons[i] for i in keep]
    filt_matrix = [[row[i] for i in keep] for row in matrix_cols_major]
    return filt_seasons, filt_matrix


def build_boolean_matrix(matrix: List[List[float]], threshold_percent: float) -> List[List[int]]:
    thr = float(threshold_percent)
    return [[1 if v >= thr else 0 for v in row] for row in matrix]


# ------------------------------
# Core logic
# ------------------------------
def longest_true_run(flags: List[int]) -> Tuple[Optional[int], Optional[int], int]:
    """
    Return (l, r, length) of the longest contiguous run of 1s in flags.
    If no 1s, returns (None, None, 0).
    """
    best_len = 0
    best_l = None
    best_r = None

    cur_len = 0
    cur_l = 0
    for i, v in enumerate(flags):
        if v:
            if cur_len == 0:
                cur_l = i
            cur_len += 1
            if cur_len > best_len:
                best_len = cur_len
                best_l = cur_l
                best_r = i
        else:
            cur_len = 0

    return best_l, best_r, best_len


def find_fixtures_max_window_fixed_stats(
    seasons: List[str],
    stat_keys: List[str],
    bool_matrix: List[List[int]],
    min_start_y1: int,
    required_stats: List[str],
) -> Dict:
    """
    Fix the set of required fixture stats. For each season, require all required stats to be 1.
    Find the longest contiguous season window (>= min_start_y1) where all required stats are present (1).
    """
    # stat_key -> row index
    idx = {k: i for i, k in enumerate(stat_keys)}
    # require rows; if a required stat is missing, treat as all zeros
    required_rows: List[List[int]] = []
    missing: List[str] = []
    for key in required_stats:
        if key in idx:
            required_rows.append(bool_matrix[idx[key]])
        else:
            missing.append(key)
            required_rows.append([0] * (len(seasons)))

    # AND across required rows, per season
    season_ok = []
    for c in range(len(seasons)):
        all_ok = 1
        for row in required_rows:
            if row[c] == 0:
                all_ok = 0
                break
        season_ok.append(all_ok)

    # filter seasons since min_start_y1
    seasons_f, matrix_f = filter_seasons_since(seasons, [season_ok], min_start_y1)
    if not seasons_f:
        return {
            "start_idx": None, "end_idx": None,
            "start_season": None, "end_season": None,
            "width": 0, "missing_stats": missing, "seasons": seasons_f,
        }
    season_ok_f = matrix_f[0]

    l, r, length = longest_true_run(season_ok_f)
    if length == 0:
        return {
            "start_idx": None, "end_idx": None,
            "start_season": None, "end_season": None,
            "width": 0, "missing_stats": missing, "seasons": seasons_f,
        }

    return {
        "start_idx": l, "end_idx": r,
        "start_season": seasons_f[l], "end_season": seasons_f[r],
        "width": length, "missing_stats": missing, "seasons": seasons_f,
    }


def counts_for_all_windows(bool_matrix: List[List[int]]) -> List[List[int]]:
    """
    For each window [l..r] over columns, return the count of rows that are all-ones in that window.
    Complexity: O(R*C^2) — fine for our sizes.
    """
    if not bool_matrix:
        return []
    R = len(bool_matrix)
    C = len(bool_matrix[0])
    counts = [[0]*C for _ in range(C)]
    for l in range(C):
        active = [1]*R
        for r in range(l, C):
            for i in range(R):
                active[i] = 1 if (active[i] and bool_matrix[i][r]) else 0
            counts[l][r] = sum(active)
    return counts


def active_rows_in_window(bool_matrix: List[List[int]], l: int, r: int) -> List[int]:
    rows = []
    for i, row in enumerate(bool_matrix):
        if all(row[c] == 1 for c in range(l, r+1)):
            rows.append(i)
    return rows


def best_rectangle_with_min_width(
    sub_bool: List[List[int]],
    stat_keys: List[str],
    sub_seasons: List[str],
    min_width: int,
) -> Dict:
    """
    Max-area rectangle with row reordering allowed and contiguous columns,
    restricted to windows with width >= min_width.
    """
    if not sub_seasons:
        return {"area": 0, "height": 0, "width": 0, "start_season": None, "end_season": None, "stats": []}
    C = len(sub_seasons)
    counts = counts_for_all_windows(sub_bool)

    best = {"l": 0, "r": 0, "area": -1, "height": 0}
    any_valid = False
    for l in range(C):
        for r in range(l, C):
            width = r - l + 1
            if width < min_width:
                continue
            height = counts[l][r]
            area = height * width
            if area > best["area"] or (area == best["area"] and height > best["height"]):
                best.update({"l": l, "r": r, "area": area, "height": height})
                any_valid = True

    if not any_valid or best["area"] <= 0:
        return {"area": 0, "height": 0, "width": 0, "start_season": None, "end_season": None, "stats": []}

    l, r = best["l"], best["r"]
    rows_idx = active_rows_in_window(sub_bool, l, r)
    stats = [stat_keys[i] for i in rows_idx]
    return {
        "area": best["area"],
        "height": len(rows_idx),
        "width": (r - l + 1),
        "start_season": sub_seasons[l],
        "end_season": sub_seasons[r],
        "stats": stats,
    }


def category_best_with_relaxation(
    seasons_axis: List[str],
    stat_keys: List[str],
    float_matrix: List[List[float]],
    window_seasons: List[str],
    base_threshold: float,
    min_width_target: int,
    allow_relax: bool = True,
    relax_step: float = 2.5,
    relax_cap_pp: float = 10.0,
    threshold_floor: float = 50.0,
) -> Dict:
    """
    Find best rectangle inside window with width >= min_width_target.
    If no valid rectangle at base_threshold, relax threshold stepwise (down to floor or cap).
    Returns dict with effective_threshold, relaxed_by_pp, width_min_target, width_min_used, width_min_feasible.
    """
    # Align columns to intersection (preserving window order)
    idx = {s: i for i, s in enumerate(seasons_axis)}
    cols = [idx[s] for s in window_seasons if s in idx]
    sub_seasons = [seasons_axis[j] for j in cols]
    if not cols:
        return {
            "area": 0, "height": 0, "width": 0, "start_season": None, "end_season": None, "stats": [],
            "effective_threshold": base_threshold, "relaxed_by_pp": 0.0,
            "width_min_target": min_width_target, "width_min_used": 0, "width_min_feasible": False,
            "available_cols": 0,
        }
    sub_float = [[row[j] for j in cols] for row in float_matrix]

    # Width feasibility
    available_cols = len(sub_seasons)
    width_min_used = min_width_target
    width_min_feasible = True
    if available_cols < min_width_target:
        width_min_feasible = False
        width_min_used = available_cols  # best we can try

    # Relaxation loop
    def try_with_threshold(thr: float) -> Dict:
        sub_bool = build_boolean_matrix(sub_float, thr)
        return best_rectangle_with_min_width(sub_bool, stat_keys, sub_seasons, width_min_used)

    # compute relaxation bound
    min_thr_allowed = max(threshold_floor, base_threshold - relax_cap_pp)
    current_thr = base_threshold
    relaxed_by = 0.0

    result = try_with_threshold(current_thr)
    if result["area"] <= 0 and allow_relax:
        while current_thr - relax_step >= min_thr_allowed - 1e-9:
            current_thr -= relax_step
            relaxed_by += relax_step
            result = try_with_threshold(current_thr)
            if result["area"] > 0:
                break

    # Attach meta
    result.update({
        "effective_threshold": round(current_thr, 4),
        "relaxed_by_pp": round(relaxed_by, 4),
        "width_min_target": int(min_width_target),
        "width_min_used": int(width_min_used),
        "width_min_feasible": bool(width_min_feasible),
        "available_cols": int(available_cols),
    })
    return result


# ------------------------------
# Driver
# ------------------------------
def run_fixed_fixtures_then_categories_with_relax(
    vis_root: str = "data_visualisation",
    threshold_percent: float = 80.0,
    min_start_y1: int = 2017,
    allow_relax: bool = True,
    relax_step: float = 2.5,
    relax_cap_pp: float = 10.0,
    threshold_floor: float = 50.0,
    verbose: bool = True,
) -> Dict:
    # Paths
    fixtures_pivot = os.path.join(vis_root, "ALL_SEASONS", "fixtures", "overall_pivot.csv")
    players_base = os.path.join(vis_root, "ALL_SEASONS", "players_last_year")

    per_cat_pivots = {
        "ATTACKER": os.path.join(players_base, "by_position_ATTACKER_pivot.csv"),
        "DEFENDER": os.path.join(players_base, "by_position_DEFENDER_pivot.csv"),
        "GOALKEEPER": os.path.join(players_base, "by_position_GOALKEEPER_pivot.csv"),
        "MIDFIELDER": os.path.join(players_base, "by_position_MIDFIELDER_pivot.csv"),
    }

    # --- Load fixtures pivot ---
    if verbose:
        print(f"[LOAD] fixtures pivot: {fixtures_pivot}")
    f_seasons, f_stat_keys, f_matrix_f = read_pivot_csv(fixtures_pivot)
    f_bool = build_boolean_matrix(f_matrix_f, threshold_percent)

    # --- Find fixtures window with fixed required stats ---
    if verbose:
        print(f"[FIXTURES] required stats: {REQUIRED_FIXTURE_STATS} | threshold={threshold_percent}% | min_start_y1={min_start_y1}")
    fix_win = find_fixtures_max_window_fixed_stats(
        seasons=f_seasons,
        stat_keys=f_stat_keys,
        bool_matrix=f_bool,
        min_start_y1=min_start_y1,
        required_stats=[s.upper() for s in REQUIRED_FIXTURE_STATS],
    )
    out_dir = os.path.join(vis_root, "ALL_SEASONS", "analysis"); ensure_dir(out_dir)

    if fix_win["width"] == 0:
        if verbose:
            print("[RESULT] No contiguous fixtures window where all required stats pass the threshold.")
            if fix_win["missing_stats"]:
                print(f"[NOTE] Missing rows in pivot for: {fix_win['missing_stats']}")
        write_csv(os.path.join(out_dir, "fixtures_fixedstats_window_summary_since_2017.csv"),
                  ["start_season","end_season","width","threshold_percent","min_start_y1","required_stats","missing_stats"],
                  [[None, None, 0, int(threshold_percent), min_start_y1, "|".join(REQUIRED_FIXTURE_STATS), "|".join(fix_win["missing_stats"])]])
        return {"fixtures_window": fix_win, "categories_within_fixtures": {}}

    if verbose:
        print(f"[FIXTURES] window: {fix_win['start_season']} → {fix_win['end_season']}  (width={fix_win['width']})")
        if fix_win["missing_stats"]:
            print(f"[WARN] Some required stats are missing in pivot (treated as 0): {fix_win['missing_stats']}")

    write_csv(os.path.join(out_dir, "fixtures_fixedstats_window_summary_since_2017.csv"),
              ["start_season","end_season","width","threshold_percent","min_start_y1","required_stats","missing_stats"],
              [[fix_win["start_season"], fix_win["end_season"], fix_win["width"], int(threshold_percent), min_start_y1,
                "|".join(REQUIRED_FIXTURE_STATS), "|".join(fix_win["missing_stats"])]]
              )

    # --- Prepare fixtures window seasons ---
    window_seasons = fix_win["seasons"][fix_win["start_idx"]:fix_win["end_idx"]+1]
    W_fix = fix_win["width"]
    W_min_target = max(1, W_fix - 2)

    # --- For each category, find max-area rectangle inside the fixtures window with min width (and optional relaxation) ---
    results = {}
    for cat, path in per_cat_pivots.items():
        if verbose:
            print(f"[LOAD] {cat}: {path}")
        if not os.path.isfile(path):
            if verbose:
                print(f"[WARN] Missing pivot for {cat}: {path}")
            results[cat] = {
                "area":0,"height":0,"width":0,"start_season":None,"end_season":None,"stats":[],
                "effective_threshold": threshold_percent, "relaxed_by_pp": 0.0,
                "width_min_target": W_min_target, "width_min_used": 0, "width_min_feasible": False,
                "available_cols": 0,
            }
            continue

        c_seasons, c_stat_keys, c_matrix_f = read_pivot_csv(path)
        # Filter category seasons to >= min_start_y1 (consistency), though window_seasons are already filtered
        c_seasons_f, c_matrix_f_filt = filter_seasons_since_float(c_seasons, c_matrix_f, min_start_y1)

        res = category_best_with_relaxation(
            seasons_axis=c_seasons_f,
            stat_keys=c_stat_keys,
            float_matrix=c_matrix_f_filt,
            window_seasons=window_seasons,
            base_threshold=threshold_percent,
            min_width_target=W_min_target,
            allow_relax=allow_relax,
            relax_step=relax_step,
            relax_cap_pp=relax_cap_pp,
            threshold_floor=threshold_floor,
        )
        results[cat] = res
        if verbose:
            print(f"[{cat}] area={res['area']} (h={res['height']}, w={res['width']})  "
                  f"{res['start_season']} → {res['end_season']} | stats={len(res['stats'])} "
                  f"| eff_thr={res['effective_threshold']}% (relaxed {res['relaxed_by_pp']}pp) "
                  f"| Wmin target={res['width_min_target']} used={res['width_min_used']} feasible={res['width_min_feasible']}")

    # Write per-category summaries and stat lists
    rows = []
    for cat in CATEGORIES:
        x = results[cat]
        rows.append([
            cat, x["start_season"], x["end_season"], x["area"], x["height"], x["width"],
            int(threshold_percent), min_start_y1, W_min_target,
            x["effective_threshold"], x["relaxed_by_pp"], x["width_min_used"], int(x["width_min_feasible"]), x["available_cols"]
        ])

    write_csv(os.path.join(out_dir, "within_fixtures_window_max_area_summary_since_2017.csv"),
              ["category","start_season","end_season","area","height","width",
               "base_threshold","min_start_y1","width_min_target",
               "effective_threshold","relaxed_by_pp","width_min_used","width_min_feasible","available_cols"],
              rows)

    for cat in CATEGORIES:
        write_csv(os.path.join(out_dir, f"within_fixtures_window_stats_{cat}_since_2017.csv"),
                  ["stat_key"], [[s] for s in results[cat]["stats"]])

    return {"fixtures_window": fix_win, "categories_within_fixtures": results}


# --- minimal entrypoint ---
DEFAULT_VIS_ROOT = os.environ.get("VIS_ROOT", "data_visualisation")

def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default

def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name, "")
    if v == "": return default
    return v.strip() not in {"0", "false", "False", "FALSE", "no", "No", "NO"}

if __name__ == "__main__":
    threshold = _env_float("RECT_THRESHOLD", 80.0)       # e.g., 75, 80, 90
    min_y1 = int(os.environ.get("RECT_MIN_START_Y1", "2016"))
    allow_relax = _env_bool("RELAX_ALLOW", True)
    relax_step = _env_float("RELAX_STEP", 2.5)           # in percentage points
    relax_cap_pp = _env_float("RELAX_CAP_PP", 10.0)      # max total relaxation in pp
    threshold_floor = _env_float("THRESHOLD_FLOOR", 50.0)

    print(f"[CONFIG] VIS_ROOT={DEFAULT_VIS_ROOT} | base_threshold={threshold}% | min_start_y1={min_y1} "
          f"| relax={'ON' if allow_relax else 'OFF'} (step={relax_step}pp, cap={relax_cap_pp}pp, floor={threshold_floor}%)")

    res = run_fixed_fixtures_then_categories_with_relax(
        vis_root=DEFAULT_VIS_ROOT,
        threshold_percent=threshold,
        min_start_y1=min_y1,
        allow_relax=allow_relax,
        relax_step=relax_step,
        relax_cap_pp=relax_cap_pp,
        threshold_floor=threshold_floor,
        verbose=True,
    )

    fw = res["fixtures_window"]
    if fw["width"] == 0:
        print("\n=== FIXTURES FIXED-STATS WINDOW ===\nNo valid window found.")
    else:
        print("\n=== FIXTURES FIXED-STATS WINDOW ===")
        print(f"{fw['start_season']} → {fw['end_season']} (width={fw['width']}) | required={REQUIRED_FIXTURE_STATS}")

    print("\n=== WITHIN FIXTURES WINDOW (per category, max area w/ min-width + relaxation) ===")
    for cat, x in res["categories_within_fixtures"].items():
        print(f"{cat:11s}: area={x['area']:4d} (h={x['height']}, w={x['width']})  {x['start_season']} → {x['end_season']} "
              f"| stats={len(x['stats'])} | eff_thr={x['effective_threshold']}% (relaxed {x['relaxed_by_pp']}pp) "
              f"| target_wmin={x['width_min_target']}, used={x['width_min_used']}, feasible={x['width_min_feasible']} "
              f"| available_cols={x['available_cols']}")
