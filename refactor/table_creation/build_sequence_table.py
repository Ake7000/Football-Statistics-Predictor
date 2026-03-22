"""
build_sequence_table.py
=======================
Pre-computes per-fixture LSTM sequence tensors and saves them to SEQ_TABLE_PATH.

For each fixture in the training CSV, extracts the K=SEQ_K most recent prior
fixtures (from each team's perspective) and assembles a (K, SEQ_INPUT_DIM)
float32 array.  Zero-padding fills in missing history for early fixtures.

Feature layout per sequence step (SEQ_INPUT_DIM = 14):
    [GOALS_FOR, GOALS_AGAINST,
     CORNERS_FOR, CORNERS_AGAINST,
     YELLOWCARDS_FOR, YELLOWCARDS_AGAINST,
     SHOTS_ON_TARGET_FOR, SHOTS_ON_TARGET_AGAINST,
     FOULS_FOR, FOULS_AGAINST,
     OFFSIDES_FOR, OFFSIDES_AGAINST,
     REDCARDS_FOR, REDCARDS_AGAINST]

"FOR"     = stat achieved BY the team of interest in that past match.
"AGAINST" = stat suffered BY the team of interest in that past match.

Output saved to SEQ_TABLE_PATH (.npz):
    fixture_ids  (N,)        int64    — fixture_id for row alignment
    home_seq     (N, K, F)   float32  — home team's last K outcomes
    away_seq     (N, K, F)   float32  — away team's last K outcomes
    home_mask    (N, K)      bool     — True = real step, False = zero-pad
    away_mask    (N, K)      bool     — True = real step, False = zero-pad

Sequences are saved in the SAME ORDER as the CSV rows (not sorted order),
so that alignment with load_and_prepare_dataframe() is trivial.

Run from workspace root:
    python refactor/table_creation/build_sequence_table.py
"""

import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure refactor/ is on sys.path so shared_config imports correctly.
# ---------------------------------------------------------------------------
_HERE    = Path(__file__).resolve().parent      # table_creation/
_REFACTOR = _HERE.parent                         # refactor/
if str(_REFACTOR) not in sys.path:
    sys.path.insert(0, str(_REFACTOR))

from shared_config import (
    TRAIN_TABLE_PATH,
    SEQ_TABLE_PATH,
    SEQ_K,
    SEQ_STATS,
    SEQ_INPUT_DIM,
)

# (home_col, away_col) pairs in the order they appear in the feature vector
_STAT_PAIRS: list[tuple[str, str]] = [
    (f"HOME_{s}", f"AWAY_{s}") for s in SEQ_STATS
]


# =============================================================================
# ==  CORE HELPERS  ===========================================================
# =============================================================================

def _extract_vec(row: pd.Series, team_was_home: bool) -> np.ndarray:
    """
    Build a (SEQ_INPUT_DIM,) = (14,) feature vector for one past match,
    from the perspective of the team of interest.

    If the team was HOME in that past match: FOR = HOME_X, AGAINST = AWAY_X
    If the team was AWAY in that past match: FOR = AWAY_X, AGAINST = HOME_X
    """
    vec = np.empty(SEQ_INPUT_DIM, dtype=np.float32)
    for k, (home_col, away_col) in enumerate(_STAT_PAIRS):
        h_val = float(row[home_col]) if pd.notna(row.get(home_col)) else 0.0
        a_val = float(row[away_col]) if pd.notna(row.get(away_col)) else 0.0
        if team_was_home:
            vec[2 * k]     = h_val   # FOR
            vec[2 * k + 1] = a_val   # AGAINST
        else:
            vec[2 * k]     = a_val   # FOR
            vec[2 * k + 1] = h_val   # AGAINST
    return vec


# =============================================================================
# ==  MAIN BUILDER  ===========================================================
# =============================================================================

def build_sequence_table(
    csv_path: Path = TRAIN_TABLE_PATH,
    out_path: Path = SEQ_TABLE_PATH,
    K: int         = SEQ_K,
) -> None:
    print(f"[SEQ] Reading {csv_path.name} …")
    raw = pd.read_csv(csv_path)
    N   = len(raw)
    F   = SEQ_INPUT_DIM
    print(f"[SEQ] {N} fixtures, K={K}, F={F}")

    # ---- Sort by fixture_ts to determine chronological order ----------------
    # fixture_ts format: "2017-06-27T16-00-00_1818602"
    # Lexicographic sort is correct for this ISO-like format.
    sort_order  = raw["fixture_ts"].argsort(kind="stable").values  # indices of raw → sorted
    df_sorted   = raw.iloc[sort_order].reset_index(drop=True)

    # Inverted mapping: original row index → position in sorted df
    orig_to_sorted = np.empty(N, dtype=np.int64)
    orig_to_sorted[sort_order] = np.arange(N, dtype=np.int64)

    # ---- Build per-team chronological history (in sorted order) -------------
    # team_history[team_id] = sorted list of (sorted_pos, is_home) tuples
    team_history: dict[int, list[tuple[int, bool]]] = defaultdict(list)
    for spos in range(N):
        row = df_sorted.iloc[spos]
        team_history[int(row["home_team_id"])].append((spos, True))
        team_history[int(row["away_team_id"])].append((spos, False))
    # Lists are already in ascending sorted_pos order (we iterate 0..N-1).

    # ---- Pre-allocate output arrays (indexed in ORIGINAL row order) ---------
    home_seq  = np.zeros((N, K, F), dtype=np.float32)
    away_seq  = np.zeros((N, K, F), dtype=np.float32)
    home_mask = np.zeros((N, K),    dtype=bool)
    away_mask = np.zeros((N, K),    dtype=bool)

    # ---- Fill sequences for each original row --------------------------------
    for orig_idx in range(N):
        spos     = int(orig_to_sorted[orig_idx])
        row_orig = raw.iloc[orig_idx]
        home_tid = int(row_orig["home_team_id"])
        away_tid = int(row_orig["away_team_id"])

        for team_id, seq_arr, mask_arr in [
            (home_tid, home_seq, home_mask),
            (away_tid, away_seq, away_mask),
        ]:
            # All matches for this team that occurred BEFORE the current fixture
            # in chronological order (sorted_pos < spos → earlier in time).
            history = [
                (p, is_home)
                for p, is_home in team_history[team_id]
                if p < spos
            ]
            history = history[-K:]      # keep at most K most recent
            n_real  = len(history)

            # Right-align: zero-padding at the front, real steps at the back.
            # Positions [0 : K-n_real] remain zeros (already zero-initialised).
            for j, (past_spos, is_home) in enumerate(history):
                past_row = df_sorted.iloc[past_spos]
                vec      = _extract_vec(past_row, is_home)
                seq_arr [orig_idx, K - n_real + j] = vec
                mask_arr[orig_idx, K - n_real + j] = True

    # ---- Save ----------------------------------------------------------------
    fixture_ids = raw["fixture_id"].to_numpy(dtype=np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        fixture_ids=fixture_ids,
        home_seq=home_seq,
        away_seq=away_seq,
        home_mask=home_mask,
        away_mask=away_mask,
    )

    # Human-readable info file for quick debugging
    pad_home = int((~home_mask).sum())
    pad_away = int((~away_mask).sum())
    total    = N * K
    info = "\n".join([
        f"CSV             : {csv_path.name}",
        f"N fixtures      : {N}",
        f"K (seq len)     : {K}",
        f"F (input dim)   : {F}",
        f"Stats (for/ag.) : {SEQ_STATS}",
        f"home_seq shape  : {home_seq.shape}",
        f"away_seq shape  : {away_seq.shape}",
        f"Padded steps    : home={pad_home}/{total} ({100*pad_home/total:.1f}%)"
        f"  away={pad_away}/{total} ({100*pad_away/total:.1f}%)",
    ])
    out_path.with_suffix(".info.txt").write_text(info)

    print(f"[SEQ] Saved → {out_path}")
    print(f"      home_seq={home_seq.shape}  away_seq={away_seq.shape}")
    print(f"      Padded steps: home={pad_home}/{total}  away={pad_away}/{total}")


# =============================================================================
# ==  ENTRY POINT  ============================================================
# =============================================================================

if __name__ == "__main__":
    build_sequence_table()
