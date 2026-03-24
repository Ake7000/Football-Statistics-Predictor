# players.py
import os
import re
import json
import csv
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt  # matplotlib only; one chart per figure

# ---- Season helpers ----
SEASON_DIR_RE = re.compile(r"^(?P<y1>\d{4})-(?P<y2>\d{4})_(?P<season_id>\d+)$")


def is_valid_season_dir(name: str) -> bool:
    m = SEASON_DIR_RE.match(name)
    if not m:
        return False
    y1 = int(m.group("y1"))
    y2 = int(m.group("y2"))
    return y2 == y1 + 1


# ---- JSON helpers ----
def safe_load_json(path: str) -> Optional[Union[Dict, List]]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def iter_player_last_year_stats_details(payload: Union[Dict, List]):
    """
    Yield tuples (position_code_or_UNKNOWN, details_item) for every 'details' entry
    found under any 'statistics' item in the payload. Handles both dict and list shapes.
    """
    if not payload:
        return

    def _normalize_pos_code(pos_obj) -> str:
        if isinstance(pos_obj, dict):
            code = pos_obj.get("code") or pos_obj.get("developer_name") or pos_obj.get("name")
            if isinstance(code, str) and code.strip():
                return code.upper()
        return "UNKNOWN"

    def _iter_statistics(obj):
        if not isinstance(obj, dict):
            return
        stats = obj.get("statistics")
        if isinstance(stats, list):
            for s in stats:
                if not isinstance(s, dict):
                    continue
                pos_code = _normalize_pos_code(s.get("position"))
                details = s.get("details")
                if isinstance(details, list):
                    for d in details:
                        if isinstance(d, dict):
                            yield pos_code, d

    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            yield from _iter_statistics(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    yield from _iter_statistics(item)
    elif isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                data = item.get("data")
                if isinstance(data, dict):
                    yield from _iter_statistics(data)
                elif isinstance(data, list):
                    for sub in data:
                        if isinstance(sub, dict):
                            yield from _iter_statistics(sub)


def canonical_stat_key(type_obj: Optional[dict]) -> Optional[str]:
    """
    Preferred key: developer_name → code → name (upper, spaces -> underscores).
    """
    if not isinstance(type_obj, dict):
        return None
    for k in ("developer_name", "code", "name"):
        v = type_obj.get(k)
        if isinstance(v, str) and v.strip():
            if k == "name":
                return v.strip().upper().replace(" ", "_")
            return v.strip().upper()
    return None


def get_stat_group(type_obj: Optional[dict]) -> str:
    if isinstance(type_obj, dict):
        sg = type_obj.get("stat_group")
        if isinstance(sg, str) and sg.strip():
            return sg.strip().upper()
    return "UNKNOWN"


# ---- Aggregation (per season) ----
def aggregate_season_players_last_year(season_players_dir: str):
    """
    Walk <season>/players/*/last_year_statistics.json and aggregate:
      - overall: stat_key -> set(player_ids)
      - per position: pos_code -> (stat_key -> set(player_ids))
      - stat_key -> stat_group map
      - total_players_with_file: unique player_ids counted once per season (only those with last_year_statistics.json)
      - pos_code -> set(player_ids) (for denominators per position)
    """
    overall_sets: Dict[str, Set[int]] = defaultdict(set)
    pos_stat_sets: Dict[str, Dict[str, Set[int]]] = defaultdict(lambda: defaultdict(set))
    stat_key_group: Dict[str, str] = {}
    total_players_with_file: Set[int] = set()
    pos_players: Dict[str, Set[int]] = defaultdict(set)

    if not os.path.isdir(season_players_dir):
        return overall_sets, pos_stat_sets, stat_key_group, total_players_with_file, pos_players

    for entry in os.scandir(season_players_dir):
        if not entry.is_dir():
            continue
        # parse player_id from "[name]_[id]"
        _, sep, suffix = entry.name.rpartition("_")
        if not (sep and suffix.isdigit()):
            continue
        player_id = int(suffix)

        last_path = os.path.join(entry.path, "last_year_statistics.json")
        if not os.path.isfile(last_path):
            continue

        payload = safe_load_json(last_path)
        if payload is None:
            continue

        total_players_with_file.add(player_id)

        # We'll collect a set of stats per player to avoid double-counting the same stat for the same player
        player_overall_stats_seen: Set[str] = set()
        # For per-position, we also avoid duplicates per player & stat & position
        player_pos_stats_seen: Dict[str, Set[str]] = defaultdict(set)

        for pos_code, detail in iter_player_last_year_stats_details(payload):
            t = detail.get("type") if isinstance(detail, dict) else None
            key = canonical_stat_key(t)
            if not key:
                continue
            group = get_stat_group(t)
            # remember group for key (first non-UNKNOWN wins)
            if key not in stat_key_group or stat_key_group[key] == "UNKNOWN":
                stat_key_group[key] = group

            # overall (per player, per stat, unique)
            if key not in player_overall_stats_seen:
                overall_sets[key].add(player_id)
                player_overall_stats_seen.add(key)

            # per-position (per player, per stat, per position, unique)
            if key not in player_pos_stats_seen[pos_code]:
                pos_stat_sets[pos_code][key].add(player_id)
                player_pos_stats_seen[pos_code].add(key)

            # track player presence in this position (for denominators)
            pos_players[pos_code].add(player_id)

    return overall_sets, pos_stat_sets, stat_key_group, total_players_with_file, pos_players


# ---- Output helpers ----
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_csv_counts(path: str, rows: List[Dict[str, Union[str, int, float]]], fieldnames: List[str]):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def build_color_map(groups: List[str]) -> Dict[str, str]:
    """
    Assign distinct colors per stat_group using matplotlib's tab20 cycle.
    Returns group -> color hex.
    """
    # 20 distinguishable colors
    from itertools import cycle
    tab_colors = plt.get_cmap("tab20").colors  # tuples of RGBA
    hex_colors = ["#" + "".join(f"{int(c*255):02X}" for c in rgba[:3]) for rgba in tab_colors]
    cmap = {}
    for g, col in zip(groups, cycle(hex_colors)):
        cmap[g] = col
    return cmap


def plot_bar_counts(
    out_path: str,
    title: str,
    x_labels: List[str],
    counts: List[int],
    percents: List[float],
    groups_for_bars: List[str],
    group_to_color: Dict[str, str],
    total_players_note: Optional[str] = None,
    rotate_xticks: int = 75,
):
    ensure_dir(os.path.dirname(out_path))

    fig = plt.figure(figsize=(max(10, len(x_labels) * 0.4), 7))  # widen with #bars
    ax = plt.gca()

    # colors per bar by stat_group
    bar_colors = [group_to_color.get(g, "#999999") for g in groups_for_bars]

    bars = ax.bar(range(len(x_labels)), counts, color=bar_colors)
    ax.set_title(title)
    ax.set_xlabel("Statistic (developer_name)")
    ax.set_ylabel("Players with the statistic")

    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=rotate_xticks, ha="right")

    # annotate each bar with "count (pct%)"
    for rect, c, p in zip(bars, counts, percents):
        height = rect.get_height()
        label = f"{c} ({p:.1f}%)"
        ax.annotate(
            label,
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    # legend for stat_group
    unique_groups = []
    for g in groups_for_bars:
        if g not in unique_groups:
            unique_groups.append(g)
    legend_handles = []
    from matplotlib.patches import Patch
    for g in unique_groups:
        legend_handles.append(Patch(color=group_to_color.get(g, "#999999"), label=g))
    if legend_handles:
        ax.legend(handles=legend_handles, title="stat_group", loc="upper right")

    # total players note
    if total_players_note:
        ax.text(
            0.99, 0.01, total_players_note,
            transform=ax.transAxes,
            ha="right", va="bottom",
            fontsize=9, bbox=dict(boxstyle="round", facecolor="white", alpha=0.6)
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---- Main generator ----
def generate_players_last_year_stats_visuals(
    data_root: str,
    vis_root: str = "data_visualisation",
    verbose: bool = True,
) -> Dict:
    """
    For each season (yyyy-[yyyy+1]_<season_id>), generate:
      - data_visualisation/<season>/players_last_year_stat_presence_counts.csv
      - data_visualisation/<season>/players_last_year_stat_presence_bar.png
      - data_visualisation/<season>/players_last_year_stat_presence_by_position_<POS>.csv (per existing POS)
      - data_visualisation/<season>/players_last_year_stat_presence_by_position_<POS>.png (per existing POS)
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(f"[plot] DATA_ROOT does not exist or is not a directory: {data_root}")

    ensure_dir(vis_root)

    seasons_done = 0
    outputs: Dict[str, Dict[str, int]] = {}

    for season_entry in os.scandir(data_root):
        if not season_entry.is_dir() or not is_valid_season_dir(season_entry.name):
            continue

        season_name = season_entry.name
        season_players_dir = os.path.join(season_entry.path, "players")
        season_vis_dir = os.path.join(vis_root, season_name)
        ensure_dir(season_vis_dir)

        if verbose:
            print(f"[{season_name}] aggregating players' last_year_statistics...")

        (overall_sets,
         pos_stat_sets,
         stat_key_group,
         total_players_with_file,
         pos_players) = aggregate_season_players_last_year(season_players_dir)

        total_players = len(total_players_with_file)
        if total_players == 0:
            if verbose:
                print(f"[{season_name}] no last_year_statistics.json found → writing empty CSV, skipping charts.")
            # write empty CSVs to mark processed season
            write_csv_counts(
                os.path.join(season_vis_dir, "players_last_year_stat_presence_counts.csv"),
                [],
                fieldnames=["stat_key", "stat_group", "players_count", "percent"]
            )
            seasons_done += 1
            outputs[season_name] = {"charts": 0, "csvs": 1}
            continue

        # ---- Overall chart & CSV ----
        counts = {k: len(v) for k, v in overall_sets.items()}
        # sort by frequency desc, then stat_key asc
        ordered_stats = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        x_labels = [k for k, _ in ordered_stats]
        y_counts = [c for _, c in ordered_stats]
        y_perc = [c * 100.0 / total_players for c in y_counts]
        groups_for_bars = [stat_key_group.get(k, "UNKNOWN") for k in x_labels]

        # color map by stat_group (deterministic ordering)
        all_groups_sorted = sorted({g for g in groups_for_bars})
        group_to_color = build_color_map(all_groups_sorted)

        # CSV
        csv_rows = [
            {"stat_key": k, "stat_group": stat_key_group.get(k, "UNKNOWN"),
             "players_count": c, "percent": round(c * 100.0 / total_players, 4)}
            for k, c in ordered_stats
        ]
        write_csv_counts(
            os.path.join(season_vis_dir, "players_last_year_stat_presence_counts.csv"),
            csv_rows,
            fieldnames=["stat_key", "stat_group", "players_count", "percent"]
        )

        # Plot
        plot_bar_counts(
            out_path=os.path.join(season_vis_dir, "players_last_year_stat_presence_bar.png"),
            title=f"{season_name} — Players: last_year_statistics coverage by stat",
            x_labels=x_labels,
            counts=y_counts,
            percents=y_perc,
            groups_for_bars=groups_for_bars,
            group_to_color=group_to_color,
            total_players_note=f"Total players with last_year_statistics.json = {total_players}",
        )

        charts_count = 1
        csvs_count = 1

        # ---- Position split: per POS chart & CSV ----
        for pos_code, stat_map in sorted(pos_stat_sets.items(), key=lambda kv: kv[0]):
            pos_players_count = len(pos_players.get(pos_code, set()))
            if pos_players_count == 0:
                continue

            pos_counts = {k: len(vset) for k, vset in stat_map.items()}
            pos_ordered = sorted(pos_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            pos_x = [k for k, _ in pos_ordered]
            pos_y = [c for _, c in pos_ordered]
            pos_y_perc = [c * 100.0 / pos_players_count for c in pos_y]
            pos_groups = [stat_key_group.get(k, "UNKNOWN") for k in pos_x]

            # Reuse color map; ensure any new group is added with a default color
            pos_groups_unique = sorted({g for g in pos_groups})
            for g in pos_groups_unique:
                if g not in group_to_color:
                    group_to_color[g] = "#999999"

            # CSV per position
            csv_rows_pos = [
                {
                    "position_code": pos_code,
                    "stat_key": k,
                    "stat_group": stat_key_group.get(k, "UNKNOWN"),
                    "players_count": c,
                    "percent": round(c * 100.0 / pos_players_count, 4),
                    "position_total_players": pos_players_count,
                }
                for k, c in pos_ordered
            ]
            write_csv_counts(
                os.path.join(season_vis_dir, f"players_last_year_stat_presence_by_position_{pos_code}.csv"),
                csv_rows_pos,
                fieldnames=["position_code", "stat_key", "stat_group", "players_count", "percent", "position_total_players"]
            )

            # Plot per position
            plot_bar_counts(
                out_path=os.path.join(season_vis_dir, f"players_last_year_stat_presence_by_position_{pos_code}.png"),
                title=f"{season_name} — Players ({pos_code}): last_year_statistics coverage by stat",
                x_labels=pos_x,
                counts=pos_y,
                percents=pos_y_perc,
                groups_for_bars=pos_groups,
                group_to_color=group_to_color,
                total_players_note=f"Total {pos_code} players (with last_year_statistics.json) = {pos_players_count}",
            )

            charts_count += 1
            csvs_count += 1

        seasons_done += 1
        outputs[season_name] = {"charts": charts_count, "csvs": csvs_count}
        if verbose:
            print(f"[{season_name}] done → charts: {charts_count}, csvs: {csvs_count}")

    return {"seasons_processed": seasons_done, "outputs": outputs}


# --- Minimal entrypoint (no argparse, no prints) ---
DEFAULT_DATA_ROOT = os.environ.get("DATA_ROOT", "data")
DEFAULT_VIS_ROOT = os.environ.get("VIS_ROOT", "data_visualisation")

if __name__ == "__main__":
    generate_players_last_year_stats_visuals(DEFAULT_DATA_ROOT, DEFAULT_VIS_ROOT, verbose=True)
