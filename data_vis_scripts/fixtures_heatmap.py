# plot_fixtures_statistics_heatmap.py
import os
import re
import json
import csv
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt  # one figure per chart

# ---------- tiny logger ----------
def _log(msg: str, verbose: bool = True):
    if verbose:
        print(msg, flush=True)

# ---------- season helpers ----------
SEASON_DIR_RE = re.compile(r"^(?P<y1>\d{4})-(?P<y2>\d{4})_(?P<season_id>\d+)$")

def is_valid_season_dir(name: str) -> bool:
    m = SEASON_DIR_RE.match(name)
    if not m:
        return False
    y1 = int(m.group("y1"))
    y2 = int(m.group("y2"))
    return y2 == y1 + 1

def safe_load_json(path: str) -> Optional[Union[Dict, List]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

# ---------- parsing helpers ----------
def canonical_stat_key(type_obj: Optional[dict]) -> Optional[str]:
    """
    Preferred key: developer_name → code → NAME_WITH_UNDERSCORES
    """
    if not isinstance(type_obj, dict):
        return None
    for k in ("developer_name", "code", "name"):
        v = type_obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip().upper().replace(" ", "_") if k == "name" else v.strip().upper()
    return None

def get_stat_group(type_obj: Optional[dict]) -> str:
    if isinstance(type_obj, dict):
        sg = type_obj.get("stat_group")
        if isinstance(sg, str) and sg.strip():
            return sg.strip().upper()
    return "UNKNOWN"

def iter_stats_from_statistics_payload(payload: Union[Dict, List]):
    """
    Yield each details-like node under any 'statistics' node found under 'data'.
    Robust to both dict/list wrapper shapes.
    Expected shapes from samples:
      payload -> data (dict or list) -> statistics (list of dict) -> type, location, data/value, etc.
    """
    if not payload:
        return
    def _walk_container(container: dict):
        stats = container.get("statistics")
        if isinstance(stats, list):
            for s in stats:
                if isinstance(s, dict):
                    yield s

    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            yield from _walk_container(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield from _walk_container(item)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                data = item.get("data")
                if isinstance(data, dict):
                    yield from _walk_container(data)
                elif isinstance(data, list):
                    for sub in data:
                        if isinstance(sub, dict):
                            yield from _walk_container(sub)

# ---------- aggregation across ALL seasons ----------
def aggregate_all_fixtures(data_root: str, verbose: bool = True, progress_every: int = 300):
    """
    Walks DATA_ROOT/<season>/fixtures/**/statistics.json and aggregates:
      - overall_sets: Dict[season -> Dict[stat_key -> Set[fixture_id_or_path]]]
      - totals_overall: Dict[season -> Set[fixture_id_or_path]] (denominator = fixtures having statistics.json)
      - stat_key_group: Dict[stat_key -> stat_group]
    """
    season_entries = [e for e in os.scandir(data_root) if e.is_dir() and is_valid_season_dir(e.name)]

    def season_sort_key(n: str):
        m = SEASON_DIR_RE.match(n)
        y1 = int(m.group("y1")); sid = int(m.group("season_id"))
        return (y1, sid)

    season_entries.sort(key=lambda e: season_sort_key(e.name))
    _log(f"[scan] seasons found: {len(season_entries)}", verbose)

    seasons = []
    overall_sets: Dict[str, Dict[str, Set[str]]] = {}
    totals_overall: Dict[str, Set[str]] = {}
    stat_key_group: Dict[str, str] = {}

    for e in season_entries:
        season = e.name
        fixtures_dir = os.path.join(e.path, "fixtures")
        if not os.path.isdir(fixtures_dir):
            _log(f"[{season}] fixtures/ not found → skipping season", verbose)
            continue

        _log(f"[{season}] start aggregation…", verbose)
        seasons.append(season)
        overall_sets[season] = {}
        totals_overall[season] = set()

        processed = 0
        found_files = 0

        # walk all subfolders recursively
        for root, _, files in os.walk(fixtures_dir):
            if "statistics.json" not in files:
                continue
            stats_path = os.path.join(root, "statistics.json")
            payload = safe_load_json(stats_path)
            if payload is None:
                continue

            # fixture identifier (prefer id from folder name suffix; fallback to path)
            fixture_id = None
            base = os.path.basename(root)  # e.g., '2024-07-09T15-30-00_19135224'
            _, sep, suffix = base.rpartition("_")
            if sep and suffix.isdigit():
                fixture_id = suffix
            fixture_key = fixture_id or stats_path  # unique within season

            totals_overall[season].add(fixture_key)
            found_files += 1

            # per-fixture unique stat keys (avoid double-counting duplicate entries)
            seen_stats_for_fixture: Set[str] = set()

            for stat_item in iter_stats_from_statistics_payload(payload):
                t = stat_item.get("type") if isinstance(stat_item, dict) else None
                key = canonical_stat_key(t)
                if not key:
                    continue
                group = get_stat_group(t)
                if key not in stat_key_group or stat_key_group[key] == "UNKNOWN":
                    stat_key_group[key] = group

                if key in seen_stats_for_fixture:
                    continue
                seen_stats_for_fixture.add(key)

                overall_sets[season].setdefault(key, set()).add(fixture_key)

            processed += 1
            if processed % progress_every == 0:
                _log(f"[{season}] processed fixtures: {processed} (files: {found_files}) | stats keys so far: {len(stat_key_group)}", verbose)

        _log(f"[{season}] done. fixtures with file: {len(totals_overall[season])}, "
             f"unique stats this season: {len(overall_sets[season])}", verbose)

    _log(f"[scan] aggregation across all seasons complete. unique stat keys (global): {len({k for s in seasons for k in overall_sets.get(s, {}).keys()})}", verbose)
    return seasons, overall_sets, totals_overall, stat_key_group

# ---------- matrix building ----------
def build_pivot_percent_matrix(
    seasons: List[str],
    counts_by_season: Dict[str, Dict[str, Set[str]]],
    denominators: Dict[str, Set[str]],
    all_stat_keys: List[str],
) -> List[List[float]]:
    """
    Returns matrix with shape [len(all_stat_keys)] x [len(seasons)] of percents (0..100).
    """
    matrix: List[List[float]] = []
    for stat in all_stat_keys:
        row: List[float] = []
        for season in seasons:
            denom = len(denominators.get(season, set()))
            if denom == 0:
                row.append(0.0)
                continue
            num = len(counts_by_season.get(season, {}).get(stat, set()))
            row.append(100.0 * num / denom)
        matrix.append(row)
    return matrix

def reorder_stats_by_global_average(matrix: List[List[float]], stat_keys: List[str]) -> Tuple[List[List[float]], List[str], List[float]]:
    """
    Compute average percent across columns and order rows DESC by average.
    First row in the ordered matrix will be rendered at the BOTTOM (origin='lower').
    """
    averages = [sum(row) / (len(row) if row else 1) for row in matrix]
    order = sorted(range(len(stat_keys)), key=lambda i: -averages[i])
    matrix_ord = [matrix[i] for i in order]
    stats_ord = [stat_keys[i] for i in order]
    avgs_ord = [averages[i] for i in order]
    return matrix_ord, stats_ord, avgs_ord

# ---------- plotting ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_csv(path: str, header: List[str], rows: List[List[Union[str, float, int]]], verbose: bool = True):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow(r)
    _log(f"[write] {path}", verbose)

def plot_heatmap(
    out_path: str,
    title: str,
    seasons: List[str],
    stat_rows: List[str],
    matrix_percent: List[List[float]],
    note: Optional[str] = None,
    overwrite: bool = False,
    verbose: bool = True,
):
    if (not overwrite) and os.path.exists(out_path):
        _log(f"[skip] PNG exists (overwrite=False): {out_path}", verbose)
        return

    ensure_dir(os.path.dirname(out_path))
    width = max(10, len(seasons) * 0.6)
    height = max(8, len(stat_rows) * 0.25)
    _log(f"[plot] rendering heatmap → size=({width:.1f}, {height:.1f}) rows={len(stat_rows)} cols={len(seasons)}", verbose)

    fig = plt.figure(figsize=(width, height))
    ax = plt.gca()

    im = ax.imshow(matrix_percent, aspect="auto", origin="lower")  # default colormap
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Fixtures with statistic (%)")

    ax.set_title(title)
    ax.set_xlabel("Season")
    ax.set_ylabel("Statistic (developer_name)")

    ax.set_xticks(range(len(seasons)))
    ax.set_xticklabels(seasons, rotation=75, ha="right", fontsize=8)

    ax.set_yticks(range(len(stat_rows)))
    ax.set_yticklabels(stat_rows, fontsize=8)

    if note:
        ax.text(
            0.99, 0.01, note,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.6)
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    _log(f"[write] {out_path}", verbose)

# ---------- driver ----------
def generate_all_seasons_fixtures_statistics_heatmap(
    data_root: str,
    vis_root: str = "data_visualisation",
    overwrite: bool = False,
    verbose: bool = True,
    progress_every: int = 300,
) -> Dict:
    """
    Builds ALL-SEASONS heatmap for fixtures/statistics.json presence of each statistic key.
    Saves under: data_visualisation/ALL_SEASONS/fixtures/
      - overall_pivot.csv, overall_totals.csv, overall_heatmap.png
    Skips PNG generation if file exists and overwrite=False.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"[heatmap] DATA_ROOT does not exist or is not a directory: {data_root}")

    base_out = os.path.join(vis_root, "ALL_SEASONS", "fixtures")
    ensure_dir(base_out)
    _log(f"[start] data_root={data_root}  vis_root={vis_root}  overwrite={overwrite}", verbose)

    seasons, overall_sets, totals_overall, stat_key_group = aggregate_all_fixtures(
        data_root, verbose=verbose, progress_every=progress_every
    )

    if not seasons:
        _log("[done] no valid seasons found.", verbose)
        return {"seasons": 0, "heatmaps": 0, "csvs": 0}

    # universe of stats across ALL seasons:
    all_stats = sorted({stat for s in seasons for stat in overall_sets.get(s, {}).keys()})
    _log(f"[overall] building matrix… seasons={len(seasons)} stats={len(all_stats)}", verbose)
    overall_matrix = build_pivot_percent_matrix(seasons, overall_sets, totals_overall, all_stats)
    overall_matrix, overall_stats_ord, overall_avgs = reorder_stats_by_global_average(overall_matrix, all_stats)
    _log(f"[overall] matrix ready. ordering by global average done.", verbose)

    # CSV pivot with avg
    pivot_csv = os.path.join(base_out, "overall_pivot.csv")
    header = ["stat_key"] + seasons + ["avg_percent"]
    rows = []
    for i, stat in enumerate(overall_stats_ord):
        rows.append([stat] + [round(v, 4) for v in overall_matrix[i]] + [round(overall_avgs[i], 4)])
    save_csv(pivot_csv, header, rows, verbose)

    # CSV totals per season (denominators)
    totals_csv = os.path.join(base_out, "overall_totals.csv")
    totals_rows = [["season", "fixtures_with_statistics_json"]] + [[s, len(totals_overall.get(s, set()))] for s in seasons]
    save_csv(totals_csv, totals_rows[0], totals_rows[1:], verbose)

    # Heatmap
    overall_png = os.path.join(base_out, "overall_heatmap.png")
    plot_heatmap(
        out_path=overall_png,
        title="Fixtures — statistics coverage by stat (ALL SEASONS)",
        seasons=seasons,
        stat_rows=overall_stats_ord,
        matrix_percent=overall_matrix,
        note="Value = % of fixtures (with statistics.json) having that stat",
        overwrite=overwrite,
        verbose=verbose,
    )

    _log(f"[ALL_SEASONS] heatmaps: 1, csvs: 2", verbose)
    return {"seasons": len(seasons), "heatmaps": 1, "csvs": 2}

# --- minimal entrypoint ---
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "data")
DEFAULT_VIS_ROOT = os.environ.get("VIS_ROOT", "data_visualisation")

if __name__ == "__main__":
    generate_all_seasons_fixtures_statistics_heatmap(
        data_root=DEFAULT_DATA_ROOT,
        vis_root=DEFAULT_VIS_ROOT,
        overwrite=False,      # set True to re-render PNGs
        verbose=True,
        progress_every=300,   # heartbeat frequency
    )
