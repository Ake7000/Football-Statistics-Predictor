# app.py — Streamlit match-prediction app  (tactical-pitch UI)
# Run: streamlit run refactor/app/app.py

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_APP_DIR      = Path(__file__).resolve().parent          # licenta/refactor/app/
_REFACTOR_DIR = _APP_DIR.parent                          # licenta/refactor/
_WORKSPACE    = _REFACTOR_DIR.parent                     # licenta/

for _p in [str(_REFACTOR_DIR), str(_APP_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import logging
import streamlit as st
import pandas as pd

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("watchdog").setLevel(logging.INFO)
logging.getLogger("streamlit.runtime").setLevel(logging.INFO)
logging.getLogger("streamlit.watcher").setLevel(logging.INFO)

from shared_config import CLASSIFIER_STAT_PAIRS, ROLE_CFG
from backend.model_registry  import build_regression_registry, build_classification_registry
from backend.data_layer      import get_current_season_dir, get_team_list, get_team_roster, get_jersey_numbers, DATA_ROOT
from backend.inference       import predict_all

# ---------------------------------------------------------------------------
# Page config & theme CSS
# ---------------------------------------------------------------------------
st.set_page_config(page_title="TACTICAL COMMAND | Match Predictor", layout="wide")

# Colors from the mockup
_C = dict(
    surface="#0b0e14", surface_low="#10131a", surface_container="#161a21",
    surface_high="#1c2028", surface_highest="#22262f",
    primary="#8eff71", primary_dim="#2be800", on_primary="#0d6100",
    secondary="#929bfa", secondary_dim="#8d96f4",
    on_surface="#ecedf6", on_surface_var="#a9abb3",
    outline_var="#45484f", outline="#73757d",
    pitch="#0e5800", pitch_line="rgba(255,255,255,0.45)",
    error="#ff7351",
)

_JERSEY_SVG = (_APP_DIR / "assets" / "jersey.svg").read_text()

# Build data-URI for jersey in each team color (using CSS filter)
_JERSEY_HOME_COLOR = _C["primary"]
_JERSEY_AWAY_COLOR = _C["secondary"]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;900&family=Inter:wght@400;500;600;700&display=swap');

/* ---------- Global overrides ---------- */
.stApp {{
    background-color: {_C['surface']};
    color: {_C['on_surface']};
    font-family: 'Inter', sans-serif;
}}
h1, h2, h3 {{
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.02em;
}}

/* Side-panel cards */
.side-card {{
    background: {_C['surface_low']};
    border: 1px solid {_C['outline_var']}22;
    border-radius: 0.75rem;
    padding: 1.2rem;
}}
.side-card-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-weight: 700;
    font-size: 1.05rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

/* Pitch container — applied via keyed st.container */
.st-key-pitch {{
    background-color: {_C['pitch']} !important;
    background-image: linear-gradient(
        90deg,
        rgba(255,255,255,0.04) 50%, transparent 50%
    ) !important;
    background-size: 200px 100% !important;
    background-position: 50% 0 !important;
    border-radius: 1rem !important;
    border: 1px solid {_C['outline_var']}22 !important;
    padding: 1rem 0.5rem !important;
    box-sizing: border-box;
    position: relative !important;
    overflow: hidden;
}}
/* Center vertical line */
.st-key-pitch::before {{
    content: '';
    position: absolute;
    top: 0; bottom: 0;
    left: 50%;
    width: 3px;
    background: {_C['pitch_line']};
    pointer-events: none;
    z-index: 0;
}}
/* Center circle */
.st-key-pitch::after {{
    content: '';
    position: absolute;
    top: 50%; left: 50%;
    transform: translate(-50%, -50%);
    width: 110px; height: 110px;
    border: 3px solid {_C['pitch_line']};
    border-radius: 50%;
    pointer-events: none;
    z-index: 0;
}}
/* Goal boxes — left (home) and right (away) */
.st-key-pitch > div:first-child::before {{
    content: '';
    position: absolute;
    top: 50%; left: -0.5rem;
    transform: translateY(-50%);
    width: 60px; height: 180px;
    border: 3px solid {_C['pitch_line']};
    border-left: none;
    pointer-events: none;
    z-index: 0;
}}
.st-key-pitch > div:first-child::after {{
    content: '';
    position: absolute;
    top: 50%; right: -0.5rem;
    transform: translateY(-50%);
    width: 60px; height: 180px;
    border: 3px solid {_C['pitch_line']};
    border-right: none;
    pointer-events: none;
    z-index: 0;
}}
/* Ensure the inner wrapper is positioned for goal box pseudo-elements */
.st-key-pitch > div:first-child {{
    position: relative !important;
}}
/* Vertically center jerseys in each pitch column */
.st-key-pitch [data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {{
    display: flex !important;
    flex-direction: column !important;
    justify-content: space-evenly !important;
}}
/* Ensure pitch columns sit above the field markings */
.st-key-pitch [data-testid="stColumn"] {{
    position: relative !important;
    z-index: 2 !important;
}}

/* Jersey node in pitch */
.jersey-node {{
    position: relative;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 2px;
    margin: 4px 0;
}}
.jersey-node svg {{
    width: 36px; height: 36px;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.5));
}}
.jersey-number {{
    position: absolute;
    top: 55%;
    left: 50%;
    transform: translate(-50%, -50%);
    color: white;
    font-family: Arial, sans-serif;
    font-weight: 900;
    line-height: 1;
    text-shadow: 0 1px 3px rgba(0,0,0,0.6);
    pointer-events: none;
}}
.jersey-name {{
    font-size: 9px;
    font-weight: 600;
    color: white;
    text-shadow: 0 1px 3px rgba(0,0,0,0.8);
    max-width: 70px;
    text-align: center;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    line-height: 1.1;
}}

/* Results table */
.res-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    background: {_C['surface_low']};
    border-radius: 0.75rem;
    overflow: hidden;
    border: 1px solid {_C['outline_var']}22;
}}
.res-table thead th {{
    background: {_C['surface_high']};
    padding: 10px 12px;
    font-size: 10px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {_C['on_surface_var']}aa;
    text-align: center;
    white-space: nowrap;
    border-bottom: 1px solid {_C['outline_var']}22;
}}
.res-table thead th:first-child {{
    text-align: left;
    padding-left: 20px;
}}
.res-table tbody td {{
    padding: 10px 12px;
    text-align: center;
    border-bottom: 1px solid {_C['outline_var']}0d;
    color: {_C['on_surface']};
}}
.res-table tbody td:first-child {{
    text-align: left;
    font-weight: 700;
    padding-left: 20px;
}}
.res-table tbody tr:nth-child(even) {{
    background: {_C['surface_highest']}18;
}}
.res-table tbody tr:hover {{
    background: {_C['surface_highest']}44;
}}
.val-home {{ font-weight: 900; color: {_C['primary']}; }}
.val-away {{ font-weight: 900; color: {_C['secondary']}; }}
.val-safe {{ color: {_C['on_surface_var']}; font-weight: 500; }}
.val-safer {{ color: {_C['on_surface_var']}99; }}

/* Probability bars */
.prob-bar {{
    display: flex;
    height: 22px;
    border-radius: 999px;
    overflow: hidden;
    background: {_C['surface_highest']};
}}
.prob-bar .seg-home, .prob-bar .seg-draw, .prob-bar .seg-away {{
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    color: #1a1a2e;
    text-shadow: none;
    overflow: hidden;
    white-space: nowrap;
}}
.prob-bar .seg-home {{ background: {_C['primary']}; }}
.prob-bar .seg-draw {{ background: {_C['outline']}; color: white; }}
.prob-bar .seg-away {{ background: {_C['secondary']}; }}

/* Popover styling tweaks */
div[data-testid="stPopover"] > div {{
    z-index: 50;
}}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached resources  (same as before)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading regression models…")
def _registry():
    logger.info("Building regression model registry...")
    registry = build_regression_registry(_WORKSPACE)
    logger.info(f"Registry loaded: {len(registry)} targets")
    for target, info in sorted(registry.items()):
        logger.info(f"[{target:25s}] {info['model_type']:25s} | cv_mae={info['cv_mae']:.4f} | {info['variant']}")
    return registry

@st.cache_resource(show_spinner="Loading classifiers…")
def _classifiers():
    logger.info("Building classification registry...")
    clfs = build_classification_registry(_WORKSPACE)
    logger.info(f"Classifiers loaded: {len(clfs)} stats")
    for stat, info in sorted(clfs.items()):
        logger.info(f"[{stat:25s}] {info['model_type']:25s} | cv_f1={info['cv_f1']:.4f} | {info['variant']}")
    return clfs

@st.cache_resource(show_spinner="Loading season data…")
def _season_dir():
    season = get_current_season_dir(DATA_ROOT)
    if not season:
        logger.error("No valid season directory found")
    return season

@st.cache_resource(show_spinner="Loading team list…")
def _team_list(season_dir_str: str):
    return get_team_list(Path(season_dir_str))

@st.cache_data(show_spinner=False)
def _roster(team_id: int, season_dir_str: str) -> Dict[str, List[Tuple[str, int]]]:
    return get_team_roster(team_id, Path(season_dir_str))

@st.cache_data(show_spinner=False)
def _jersey_nums(season_dir_str: str) -> Dict[int, int]:
    return get_jersey_numbers(Path(season_dir_str))


registry    = _registry()
classifiers = _classifiers()
season_dir  = _season_dir()

if season_dir is None:
    st.error("No valid season directory found in data/. Check your data folder.")
    st.stop()

season_dir_str = str(season_dir)
teams          = _team_list(season_dir_str)

if not teams:
    st.error(f"No teams found in {season_dir}.")
    st.stop()

team_names      = [name for name, _ in teams]
team_id_by_name = {name: tid for name, tid in teams}

DF_MAX  = ROLE_CFG["DF"]["max_slots"]
MF_MAX  = ROLE_CFG["MF"]["max_slots"]
ATK_MAX = ROLE_CFG["ATK"]["max_slots"]
OUTFIELD_TOTAL = 10


def _smart_defaults(
    caps: Tuple[int, int, int],
    preferred: Tuple[int, int, int] = (4, 3, 3),
    target: int = OUTFIELD_TOTAL,
) -> Tuple[int, int, int]:
    """Return (df, mf, atk) defaults that sum to *target* and respect *caps*."""
    vals = [min(p, c) for p, c in zip(preferred, caps)]
    deficit = target - sum(vals)
    # Distribute surplus slots round-robin to positions with headroom
    while deficit > 0:
        distributed = False
        for i in range(len(vals)):
            if deficit <= 0:
                break
            if vals[i] < caps[i]:
                vals[i] += 1
                deficit -= 1
                distributed = True
        if not distributed:
            break  # all positions at their cap, can't reach target
    return (vals[0], vals[1], vals[2])


# ---------------------------------------------------------------------------
# Jersey HTML helper
# ---------------------------------------------------------------------------

def _jersey_html(color: str, number: int | None = None) -> str:
    svg = _JERSEY_SVG.replace('fill="currentColor"', f'fill="{color}"')
    num_html = ""
    if number is not None:
        num_str = str(number)
        font_size = "13px" if len(num_str) <= 2 else "10px"
        num_html = (
            f'<span class="jersey-number" style="font-size:{font_size};">'
            f'{num_str}</span>'
        )
    return f'<div class="jersey-node">{svg}{num_html}</div>'


# ---------------------------------------------------------------------------
# Pitch-column helper:  render N player slots in one st.column
# ---------------------------------------------------------------------------

def _pitch_col(
    col,
    side_key: str,
    pos: str,
    count: int,
    roster: Dict[str, List[Tuple[str, int]]],
    color: str,
    jersey_numbers: Dict[int, int],
    selection_out: Dict[str, List[int]],
) -> bool:
    """
    Render `count` jersey + selectbox slots inside `col` for a given position.
    Appends chosen player ids to selection_out[pos].
    Returns True iff all slots have a valid player.
    """
    opts = roster.get(pos, [])
    if not opts:
        col.caption(f"No {pos}")
        return False

    names_map  = {name: pid for name, pid in opts}
    names_list = list(names_map.keys())
    all_ok = True

    for i in range(count):
        sk = f"{side_key}_{pos}_{i}"
        # Default to different player per slot
        default_idx = min(i, len(names_list) - 1)

        # Resolve currently-selected name from session state (or default)
        current_name = st.session_state.get(sk, names_list[default_idx])
        if current_name not in names_map:
            current_name = names_list[default_idx]

        with col:
            # Jersey visual with number above the selectbox
            pid_for_jersey = names_map.get(current_name)
            jnum = jersey_numbers.get(pid_for_jersey) if pid_for_jersey else None
            st.markdown(_jersey_html(color, jnum), unsafe_allow_html=True)
            st.selectbox(
                f"Pick {pos} {i+1}",
                options=names_list,
                index=names_list.index(current_name) if current_name in names_list else default_idx,
                key=sk,
                label_visibility="collapsed",
            )

        # Resolve after widget runs
        final_name = st.session_state.get(sk, names_list[default_idx])
        pid = names_map.get(final_name)
        if pid is not None:
            selection_out[pos].append(pid)
        else:
            all_ok = False

    return all_ok


# ===========================================================================
# MAIN LAYOUT
# ===========================================================================

# ---- Side-panel column spec: 2 | 8 | 2 ----
with st.container(key="pitch_row"):
    panel_left, pitch_area, panel_right = st.columns([2, 8, 2], gap="medium")

# =====================  HOME SIDE PANEL  =====================
with panel_left:
    st.markdown(
        f'<div class="side-card">'
        f'<div style="display:flex;align-items:center;gap:8px;">'
        f'<div style="width:4px;height:22px;background:{_C["primary"]};border-radius:2px;"></div>'
        f'<span class="side-card-title">Home Side</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    home_team_name = st.selectbox("Home Team", options=team_names, key="home_team")
    home_team_id: int = team_id_by_name[home_team_name]
    home_roster = _roster(home_team_id, season_dir_str)
    h_df_max  = min(DF_MAX,  max(len(home_roster.get("DF",  [])), 1))
    h_mf_max  = min(MF_MAX,  max(len(home_roster.get("MF",  [])), 1))
    h_atk_max = min(ATK_MAX, max(len(home_roster.get("ATK", [])), 1))
    h_def_df, h_def_mf, h_def_atk = _smart_defaults((h_df_max, h_mf_max, h_atk_max))

    # Reset formation when team changes (or first load)
    if st.session_state.get("_prev_home_team") != home_team_name:
        st.session_state["_prev_home_team"] = home_team_name
        st.session_state["home_n_df"]  = h_def_df
        st.session_state["home_n_mf"]  = h_def_mf
        st.session_state["home_n_atk"] = h_def_atk

    st.caption("Formation  (DF + MF + ATK = 10)")
    h_df  = st.number_input("Defenders",  min_value=1, max_value=h_df_max,  step=1, key="home_n_df")
    h_mf  = st.number_input("Midfielders",min_value=1, max_value=h_mf_max,  step=1, key="home_n_mf")
    h_atk = st.number_input("Attackers",  min_value=1, max_value=h_atk_max, step=1, key="home_n_atk")
    h_total = int(h_df) + int(h_mf) + int(h_atk)
    home_formation_ok = h_total == OUTFIELD_TOTAL
    if home_formation_ok:
        st.success(f"{int(h_df)}-{int(h_mf)}-{int(h_atk)} ✓")
    else:
        st.error(f"Sum = {h_total} / {OUTFIELD_TOTAL}")

# =====================  AWAY SIDE PANEL  =====================
with panel_right:
    st.markdown(
        f'<div class="side-card">'
        f'<div style="display:flex;align-items:center;gap:8px;justify-content:flex-end;">'
        f'<span class="side-card-title">Away Side</span>'
        f'<div style="width:4px;height:22px;background:{_C["secondary"]};border-radius:2px;"></div>'
        f'</div></div>',
        unsafe_allow_html=True,
    )
    away_team_name = st.selectbox("Away Team", options=team_names, key="away_team",
                                  index=min(1, len(team_names)-1))
    away_team_id: int = team_id_by_name[away_team_name]
    away_roster = _roster(away_team_id, season_dir_str)
    a_df_max  = min(DF_MAX,  max(len(away_roster.get("DF",  [])), 1))
    a_mf_max  = min(MF_MAX,  max(len(away_roster.get("MF",  [])), 1))
    a_atk_max = min(ATK_MAX, max(len(away_roster.get("ATK", [])), 1))
    a_def_df, a_def_mf, a_def_atk = _smart_defaults((a_df_max, a_mf_max, a_atk_max))

    # Reset formation when team changes (or first load)
    if st.session_state.get("_prev_away_team") != away_team_name:
        st.session_state["_prev_away_team"] = away_team_name
        st.session_state["away_n_df"]  = a_def_df
        st.session_state["away_n_mf"]  = a_def_mf
        st.session_state["away_n_atk"] = a_def_atk

    st.caption("Formation  (DF + MF + ATK = 10)")
    a_df  = st.number_input("Defenders",  min_value=1, max_value=a_df_max,  step=1, key="away_n_df")
    a_mf  = st.number_input("Midfielders",min_value=1, max_value=a_mf_max,  step=1, key="away_n_mf")
    a_atk = st.number_input("Attackers",  min_value=1, max_value=a_atk_max, step=1, key="away_n_atk")
    a_total = int(a_df) + int(a_mf) + int(a_atk)
    away_formation_ok = a_total == OUTFIELD_TOTAL
    if away_formation_ok:
        st.success(f"{int(a_df)}-{int(a_mf)}-{int(a_atk)} ✓")
    else:
        st.error(f"Sum = {a_total} / {OUTFIELD_TOTAL}")


# =====================  TACTICAL PITCH  =====================
with pitch_area:
    pitch = st.container(key="pitch")

    home_sel: Dict[str, List[int]] = {"GK": [], "DF": [], "MF": [], "ATK": []}
    away_sel: Dict[str, List[int]] = {"GK": [], "DF": [], "MF": [], "ATK": []}
    all_players_ok = True

    # Rosters already loaded above for formation capping; reuse them
    if not home_formation_ok:
        home_roster = {}
    if not away_formation_ok:
        away_roster = {}
    jersey_nums = _jersey_nums(season_dir_str)

    # 8 columns inside pitch container
    with pitch:
        c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8, gap="small")

    if home_formation_ok:
        ok1 = _pitch_col(c1, "home", "GK",  1,          home_roster, _JERSEY_HOME_COLOR, jersey_nums, home_sel)
        ok2 = _pitch_col(c2, "home", "DF",  int(h_df),  home_roster, _JERSEY_HOME_COLOR, jersey_nums, home_sel)
        ok3 = _pitch_col(c3, "home", "MF",  int(h_mf),  home_roster, _JERSEY_HOME_COLOR, jersey_nums, home_sel)
        ok4 = _pitch_col(c4, "home", "ATK", int(h_atk), home_roster, _JERSEY_HOME_COLOR, jersey_nums, home_sel)
        all_players_ok = all_players_ok and ok1 and ok2 and ok3 and ok4
    else:
        all_players_ok = False

    if away_formation_ok:
        ok5 = _pitch_col(c5, "away", "ATK", int(a_atk), away_roster, _JERSEY_AWAY_COLOR, jersey_nums, away_sel)
        ok6 = _pitch_col(c6, "away", "MF",  int(a_mf),  away_roster, _JERSEY_AWAY_COLOR, jersey_nums, away_sel)
        ok7 = _pitch_col(c7, "away", "DF",  int(a_df),  away_roster, _JERSEY_AWAY_COLOR, jersey_nums, away_sel)
        ok8 = _pitch_col(c8, "away", "GK",  1,          away_roster, _JERSEY_AWAY_COLOR, jersey_nums, away_sel)
        all_players_ok = all_players_ok and ok5 and ok6 and ok7 and ok8
    else:
        all_players_ok = False


# ---- JS: force pitch to fill its column height ----
# The column is already stretch-filled to match side panels by Streamlit's flex.
# But the inner wrappers don't propagate that height.
# This script directly sets inline styles on the exact DOM chain.
import streamlit.components.v1 as _stcomp
_stcomp.html("""
<script>
(function(){
  const doc = window.parent.document;
  function fill(){
    const pitch = doc.querySelector('.st-key-pitch');
    if(!pitch) return;
    // Walk UP from .st-key-pitch to the stColumn
    // Chain: stColumn > stVerticalBlock > stLayoutWrapper > div > .st-key-pitch
    let el = pitch;
    const chain = [el];
    while(el.parentElement){
      el = el.parentElement;
      chain.unshift(el);
      if(el.getAttribute('data-testid') === 'stColumn') break;
    }
    // chain[0] = stColumn, chain[last] = .st-key-pitch
    const col = chain[0];
    if(col.getAttribute('data-testid') !== 'stColumn') return;
    // The column height is already correct (stretched by flex row).
    const colH = col.getBoundingClientRect().height;
    if(colH < 50) return;

    // Set display:flex + flex-direction:column on the column
    col.style.display = 'flex';
    col.style.flexDirection = 'column';

    // Every element in the chain (except the column itself and the pitch):
    // flex: 1 to fill parent
    for(let i = 1; i < chain.length; i++){
      chain[i].style.flex = '1 1 0%';
      chain[i].style.minHeight = '0';
      if(chain[i] !== pitch){
        chain[i].style.display = 'flex';
        chain[i].style.flexDirection = 'column';
      }
    }

    // Now make pitch inner elements fill too
    // pitch > stLayoutWrapper > stHorizontalBlock (the 8-col row)
    const pitchLayout = pitch.querySelector('[data-testid="stLayoutWrapper"]');
    if(pitchLayout){
      pitchLayout.style.flex = '1 1 0%';
      pitchLayout.style.minHeight = '0';
    }
    const pitchHBlock = pitch.querySelector('[data-testid="stHorizontalBlock"]');
    if(pitchHBlock){
      pitchHBlock.style.height = '100%';
      pitchHBlock.style.alignItems = 'stretch';
    }
    // Each inner pitch column: fill height + evenly space content
    const innerCols = pitchHBlock ? pitchHBlock.querySelectorAll(':scope > [data-testid="stColumn"]') : [];
    innerCols.forEach(function(ic){
      const vb = ic.querySelector('[data-testid="stVerticalBlock"]');
      if(vb){
        vb.style.justifyContent = 'space-evenly';
        vb.style.height = '100%';
      }
    });
  }
  // Run multiple times to catch Streamlit re-renders
  setTimeout(fill, 200);
  setTimeout(fill, 500);
  setTimeout(fill, 1000);
  setTimeout(fill, 2000);
  // Also observe mutations for re-renders
  const row = doc.querySelector('.st-key-pitch_row');
  if(row){
    let t = null;
    new MutationObserver(function(){
      clearTimeout(t);
      t = setTimeout(fill, 100);
    }).observe(row, {childList:true, subtree:true});
  }
})();
</script>
""", height=0)

# ---------------------------------------------------------------------------
# Predict button
# ---------------------------------------------------------------------------
predict_ready = home_formation_ok and away_formation_ok and all_players_ok

_, btn_col, _ = st.columns([4, 2, 4])
with btn_col:
    if not predict_ready:
        st.info("Set valid formations and select all players.")
    predict_clicked = st.button("⚡  PREDICT", disabled=not predict_ready,
                                type="primary", use_container_width=True)

if predict_clicked:
    with st.spinner("Running predictions…"):
        logger.info(f"Starting prediction: {home_team_id} (HOME) vs {away_team_id} (AWAY)")
        logger.debug(f"Home selection: {home_sel}")
        logger.debug(f"Away selection: {away_sel}")
        try:
            preds = predict_all(
                home_team_id  = home_team_id,
                away_team_id  = away_team_id,
                home_sel      = home_sel,
                away_sel      = away_sel,
                season_dir    = season_dir,
                data_root     = DATA_ROOT,
                workspace_root= _WORKSPACE,
                reg_registry  = registry,
                clf_registry  = classifiers,
            )
            logger.info("Prediction completed successfully")
            logger.debug(f"Predictions: {preds}")
        except Exception as exc:
            import traceback
            logger.error(f"Prediction failed: {exc}", exc_info=True)
            st.error(f"Prediction failed: {exc}")
            st.code(traceback.format_exc())
            st.stop()

    # ==================================================================
    # PREDICTED MATCH STATISTICS  (custom HTML table)
    # ==================================================================
    base_stats = ["GOALS", "CORNERS", "YELLOWCARDS", "SHOTS_ON_TARGET",
                  "FOULS", "OFFSIDES", "REDCARDS"]

    def _fi(pred, err):
        lo = max(0.0, pred - err)
        return f"{lo:.2f} – {pred + err:.2f}"

    rows_html = ""
    for stat in base_stats:
        hv  = preds.get(f"HOME_{stat}", float("nan"))
        av  = preds.get(f"AWAY_{stat}", float("nan"))
        hm  = preds.get(f"mae_HOME_{stat}")
        hr  = preds.get(f"rmse_HOME_{stat}")
        am  = preds.get(f"mae_AWAY_{stat}")
        ar  = preds.get(f"rmse_AWAY_{stat}")
        label = stat.replace("_", " ").title()
        rows_html += (
            f'<tr>'
            f'<td>{label}</td>'
            f'<td class="val-home">{hv:.2f}</td>'
            f'<td class="val-safe">{_fi(hv,hm) if hm else "—"}</td>'
            f'<td class="val-safer">{_fi(hv,hr) if hr else "—"}</td>'
            f'<td class="val-away">{av:.2f}</td>'
            f'<td class="val-safe">{_fi(av,am) if am else "—"}</td>'
            f'<td class="val-safer">{_fi(av,ar) if ar else "—"}</td>'
            f'</tr>'
        )

    st.markdown(f"""
    <h2 style="font-family:'Space Grotesk',sans-serif;font-weight:900;font-size:1.5rem;
       text-transform:uppercase;font-style:italic;letter-spacing:-0.03em;margin-top:2rem;">
       Predicted Match Statistics</h2>
    <table class="res-table">
    <thead><tr>
        <th>Stat</th>
        <th style="color:{_C['primary']}">Home</th><th>Home Safe</th><th>Home Safer</th>
        <th style="color:{_C['secondary']}">Away</th><th>Away Safe</th><th>Away Safer</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)

    # ==================================================================
    # OUTCOME PROBABILITIES  (stacked bar chart)
    # ==================================================================
    if classifiers:
        prob_rows = ""
        for stat, _, _ in CLASSIFIER_STAT_PAIRS:
            label = stat.replace("_", " ").title()
            hp = preds.get(f"odds_{stat}_home", 0.0)
            dp = preds.get(f"odds_{stat}_draw", 0.0)
            ap = preds.get(f"odds_{stat}_away", 0.0)
            hp_pct, dp_pct, ap_pct = hp*100, dp*100, ap*100
            prob_rows += (
                f'<tr>'
                f'<td>{label}</td>'
                f'<td style="padding:8px 20px;">'
                f'  <div class="prob-bar">'
                f'    <div class="seg-home" style="width:{hp_pct:.1f}%" title="Home: {hp_pct:.1f}%">{hp_pct:.1f}%</div>'
                f'    <div class="seg-draw" style="width:{dp_pct:.1f}%" title="Draw: {dp_pct:.1f}%">{dp_pct:.1f}%</div>'
                f'    <div class="seg-away" style="width:{ap_pct:.1f}%" title="Away: {ap_pct:.1f}%">{ap_pct:.1f}%</div>'
                f'  </div>'
                f'</td>'
                f'</tr>'
            )

        st.markdown(f"""
        <h2 style="font-family:'Space Grotesk',sans-serif;font-weight:900;font-size:1.5rem;
           text-transform:uppercase;font-style:italic;letter-spacing:-0.03em;margin-top:2rem;">
           Outcome Probabilities</h2>
        <table class="res-table">
        <thead><tr><th style="width:180px">Stat</th><th>Probability Distribution (Home / Draw / Away)</th></tr></thead>
        <tbody>{prob_rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
