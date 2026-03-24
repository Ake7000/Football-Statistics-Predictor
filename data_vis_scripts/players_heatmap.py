# plot_players_last_year_heatmap.py
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
def _norm_pos_code(pos_obj) -> str:
    if isinstance(pos_obj, dict):
        for k in ("code", "developer_name", "name"):
            v = pos_obj.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip().upper()
    return "UNKNOWN"

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

def iter_details_from_last_year_payload(payload: Union[Dict, List]):
    """
    Yield (position_code, detail_dict) for each details[] node under any statistics[] node.
    Robust to both dict/list wrapper shapes.
    """
    if not payload:
        return
    def _walk_stats(container: dict):
        stats = container.get("statistics")
        if isinstance(stats, list):
            for s in stats:
                if not isinstance(s, dict):
                    continue
                pos_code = _norm_pos_code(s.get("position"))
                details = s.get("details")
                if isinstance(details, list):
                    for d in details:
                        if isinstance(d, dict):
                            yield pos_code, d

    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            yield from _walk_stats(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield from _walk_stats(item)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                data = item.get("data")
                if isinstance(data, dict):
                    yield from _walk_stats(data)
                elif isinstance(data, list):
                    for sub in data:
                        if isinstance(sub, dict):
                            yield from _walk_stats(sub)

# ---------- aggregation across ALL seasons ----------
def aggregate_all_seasons(data_root: str, verbose: bool = True, progress_every: int = 200):
    """
    Returns:
      seasons: List[str] ordered chronologically
      overall_sets: Dict[season -> Dict[stat_key -> Set[player_id]]]
      pos_sets: Dict[season -> Dict[pos_code -> Dict[stat_key -> Set[player_id]]]]
      totals_overall: Dict[season -> Set[player_id]]   # denominators (players having last_year_statistics.json)
      totals_by_pos: Dict[season -> Dict[pos_code -> Set[player_id]]]  # denominators per position
      stat_key_group: Dict[stat_key -> stat_group]
    """
    season_entries = [e for e in os.scandir(data_root) if e.is_dir() and is_valid_season_dir(e.name)]

    def season_sort_key(n: str):
        m = SEASON_DIR_RE.match(n)
        y1 = int(m.group("y1")); sid = int(m.group("season_id"))
        return (y1, sid)

    season_entries.sort(key=lambda e: season_sort_key(e.name))
    _log(f"[scan] seasons found: {len(season_entries)}", verbose)

    seasons = []
    overall_sets: Dict[str, Dict[str, Set[int]]] = {}
    pos_sets: Dict[str, Dict[str, Dict[str, Set[int]]]] = {}
    totals_overall: Dict[str, Set[int]] = {}
    totals_by_pos: Dict[str, Dict[str, Set[int]]] = {}
    stat_key_group: Dict[str, str] = {}

    for e in season_entries:
        season = e.name
        players_dir = os.path.join(e.path, "players")
        if not os.path.isdir(players_dir):
            _log(f"[{season}] players/ not found → skipping season", verbose)
            continue

        _log(f"[{season}] start aggregation…", verbose)
        seasons.append(season)
        overall_sets[season] = {}
        pos_sets[season] = {}
        totals_overall[season] = set()
        totals_by_pos[season] = {}

        processed = 0
        found_files = 0

        for p in os.scandir(players_dir):
            if not p.is_dir():
                continue
            _, sep, suffix = p.name.rpartition("_")
            if not (sep and suffix.isdigit()):
                continue
            player_id = int(suffix)

            last_path = os.path.join(p.path, "last_year_statistics.json")
            if not os.path.isfile(last_path):
                continue

            found_files += 1
            payload = safe_load_json(last_path)
            if payload is None:
                continue

            totals_overall[season].add(player_id)

            per_player_seen_overall: Set[str] = set()
            per_player_seen_by_pos: Dict[str, Set[str]] = {}

            for pos_code, detail in iter_details_from_last_year_payload(payload):
                t = detail.get("type") if isinstance(detail, dict) else None
                key = canonical_stat_key(t)
                if not key:
                    continue
                group = get_stat_group(t)
                if key not in stat_key_group or stat_key_group[key] == "UNKNOWN":
                    stat_key_group[key] = group

                if key not in per_player_seen_overall:
                    overall_sets[season].setdefault(key, set()).add(player_id)
                    per_player_seen_overall.add(key)

                if pos_code not in per_player_seen_by_pos:
                    per_player_seen_by_pos[pos_code] = set()
                if key not in per_player_seen_by_pos[pos_code]:
                    pos_sets[season].setdefault(pos_code, {}).setdefault(key, set()).add(player_id)
                    per_player_seen_by_pos[pos_code].add(key)

                totals_by_pos[season].setdefault(pos_code, set()).add(player_id)

            processed += 1
            if processed % progress_every == 0:
                _log(f"[{season}] processed players: {processed} (files: {found_files}) | stats keys so far: {len(stat_key_group)}", verbose)

        _log(f"[{season}] done. players with file: {len(totals_overall[season])}, "
             f"unique stats this season: {len(overall_sets[season])}, positions: {len(pos_sets[season])}", verbose)

    _log(f"[scan] aggregation across all seasons complete. unique stat keys (global): {len(stat_key_group)}", verbose)
    return seasons, overall_sets, pos_sets, totals_overall, totals_by_pos, stat_key_group

# ---------- matrix building ----------
def build_pivot_percent_matrix(
    seasons: List[str],
    counts_by_season: Dict[str, Dict[str, Set[int]]],
    denominators: Dict[str, Set[int]],
    all_stat_keys: List[str],
) -> List[List[float]]:
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

    im = ax.imshow(matrix_percent, aspect="auto", origin="lower")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Players with statistic (%)")

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
def generate_all_seasons_players_last_year_heatmaps(
    data_root: str,
    vis_root: str = "data_visualisation",
    overwrite: bool = False,
    verbose: bool = True,
    progress_every: int = 200,
) -> Dict:
    """
    Builds ALL-SEASONS heatmaps (overall + per position) for players/last_year_statistics.json.
    Saves under: data_visualisation/ALL_SEASONS/players_last_year/
    Skips PNG generation if file exists and overwrite=False.
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"[heatmap] DATA_ROOT does not exist or is not a directory: {data_root}")

    base_out = os.path.join(vis_root, "ALL_SEASONS", "players_last_year")
    ensure_dir(base_out)
    _log(f"[start] data_root={data_root}  vis_root={vis_root}  overwrite={overwrite}", verbose)

    seasons, overall_sets, pos_sets, totals_overall, totals_by_pos, stat_key_group = aggregate_all_seasons(
        data_root, verbose=verbose, progress_every=progress_every
    )

    if not seasons:
        _log("[done] no valid seasons found.", verbose)
        return {"seasons": 0, "heatmaps": 0, "csvs": 0}

    # ---- OVERALL ----
    all_stats = sorted({stat for s in seasons for stat in overall_sets.get(s, {}).keys()})
    _log(f"[overall] building matrix… seasons={len(seasons)} stats={len(all_stats)}", verbose)
    overall_matrix = build_pivot_percent_matrix(seasons, overall_sets, totals_overall, all_stats)
    overall_matrix, overall_stats_ord, overall_avgs = reorder_stats_by_global_average(overall_matrix, all_stats)
    _log(f"[overall] matrix ready. ordering by global average done.", verbose)

    pivot_csv = os.path.join(base_out, "overall_pivot.csv")
    header = ["stat_key"] + seasons + ["avg_percent"]
    rows = []
    for i, stat in enumerate(overall_stats_ord):
        rows.append([stat] + [round(v, 4) for v in overall_matrix[i]] + [round(overall_avgs[i], 4)])
    save_csv(pivot_csv, header, rows, verbose)

    totals_csv = os.path.join(base_out, "overall_totals.csv")
    totals_rows = [["season", "players_total"]] + [[s, len(totals_overall.get(s, set()))] for s in seasons]
    save_csv(totals_csv, totals_rows[0], totals_rows[1:], verbose)

    overall_png = os.path.join(base_out, "overall_heatmap.png")
    plot_heatmap(
        out_path=overall_png,
        title="Players — last_year_statistics coverage by stat (ALL SEASONS, OVERALL)",
        seasons=seasons,
        stat_rows=overall_stats_ord,
        matrix_percent=overall_matrix,
        note="Value = % of players with last_year_statistics.json having that stat",
        overwrite=overwrite,
        verbose=verbose,
    )

    charts = 1
    csvs = 2

    # ---- BY POSITION ----
    all_positions = sorted({pos for s in seasons for pos in pos_sets.get(s, {}).keys()})
    _log(f"[by-position] positions found: {all_positions}", verbose)

    for pos in all_positions:
        pos_all_stats = sorted({stat for s in seasons for stat in pos_sets.get(s, {}).get(pos, {}).keys()})
        if not pos_all_stats:
            _log(f"[{pos}] no stats found across seasons → skip", verbose)
            continue

        _log(f"[{pos}] building matrix… seasons={len(seasons)} stats={len(pos_all_stats)}", verbose)
        counts_pos_by_season = {s: pos_sets.get(s, {}).get(pos, {}) for s in seasons}
        denoms_pos_by_season = {s: totals_by_pos.get(s, {}).get(pos, set()) for s in seasons}
        pos_matrix = build_pivot_percent_matrix(seasons, counts_pos_by_season, denoms_pos_by_season, pos_all_stats)
        pos_matrix, pos_stats_ord, pos_avgs = reorder_stats_by_global_average(pos_matrix, pos_all_stats)
        _log(f"[{pos}] matrix ready. ordering by global average done.", verbose)

        pos_pivot_csv = os.path.join(base_out, f"by_position_{pos}_pivot.csv")
        pos_rows = []
        for i, stat in enumerate(pos_stats_ord):
            pos_rows.append([stat] + [round(v, 4) for v in pos_matrix[i]] + [round(pos_avgs[i], 4)])
        save_csv(pos_pivot_csv, header, pos_rows, verbose)

        pos_totals_csv = os.path.join(base_out, f"by_position_{pos}_totals.csv")
        pos_tot_rows = [["season", f"{pos}_players_total"]]
        pos_tot_rows += [[s, len(denoms_pos_by_season.get(s, set()))] for s in seasons]
        save_csv(pos_totals_csv, pos_tot_rows[0], pos_tot_rows[1:], verbose)

        pos_png = os.path.join(base_out, f"by_position_{pos}_heatmap.png")
        plot_heatmap(
            out_path=pos_png,
            title=f"Players — last_year_statistics coverage by stat (ALL SEASONS, {pos})",
            seasons=seasons,
            stat_rows=pos_stats_ord,
            matrix_percent=pos_matrix,
            note=f"Value = % of {pos} players (with last_year_statistics.json) having that stat",
            overwrite=overwrite,
            verbose=verbose,
        )

        charts += 1
        csvs += 2

    _log(f"[ALL_SEASONS] heatmaps: {charts}, csvs: {csvs}", verbose)
    return {"seasons": len(seasons), "heatmaps": charts, "csvs": csvs}

# --- minimal entrypoint ---
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "data")
DEFAULT_VIS_ROOT = os.environ.get("VIS_ROOT", "data_visualisation")

if __name__ == "__main__":
    # tune progress_every if you want more/less frequent heartbeats
    generate_all_seasons_players_last_year_heatmaps(
        data_root=DEFAULT_DATA_ROOT,
        vis_root=DEFAULT_VIS_ROOT,
        overwrite=False,
        verbose=True,
        progress_every=200,
    )
