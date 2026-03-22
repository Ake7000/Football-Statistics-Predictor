# build_aggregated_inputs.py
# Citește input.csv (o linie sau mai multe) și produce:
#   - raw_input.csv     (120 coloane, ordonate exact conform specificației)
#   - sum_input.csv     (30 coloane)
#   - mean_input.csv    (30 coloane)
#   - summean_input.csv (52 coloane)
# Toate fișierele se salvează lângă acest script.

import os
import pandas as pd
import numpy as np

# ---------- Config hardcodat ----------
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
INPUT_CSV  = os.path.join(BASE_DIR, "input.csv")

RAW_OUT      = os.path.join(BASE_DIR, "raw_input.csv")
SUM_OUT      = os.path.join(BASE_DIR, "sum_input.csv")
MEAN_OUT     = os.path.join(BASE_DIR, "mean_input.csv")
SUMMEAN_OUT  = os.path.join(BASE_DIR, "summean_input.csv")

# Coloane meta/ID de eliminat (dacă apar în input.csv)
META_COLS = ["season_label", "fixture_id", "fixture_ts", "home_team_id", "away_team_id"]
DROP_IF_CONTAINS = ["_player_id"]

# Config roluri (aliniat cu proiectul tău)
ROLE_CFG = {
    "DF":  {"max_slots": 6, "stats": ["GOALS_CONCEDED", "MINUTES_PLAYED", "APPEARANCES"]},
    "MF":  {"max_slots": 6, "stats": ["GOALS_CONCEDED", "MINUTES_PLAYED", "APPEARANCES"]},
    "ATK": {"max_slots": 4, "stats": ["MINUTES_PLAYED", "APPEARANCES", "GOALS_CONCEDED", "SUBSTITUTIONS_IN", "SUBSTITUTIONS_OUT"]},
}
SIDES = ["HOME", "AWAY"]

# ---------- Lista fixă pentru RAW (120 coloane) ----------
RAW_FEATURES = [
    # HOME
    "GK_HOME_GOALS_CONCEDED",
    "DF1_HOME_GOALS_CONCEDED","DF1_HOME_MINUTES_PLAYED","DF1_HOME_APPEARANCES",
    "DF2_HOME_GOALS_CONCEDED","DF2_HOME_MINUTES_PLAYED","DF2_HOME_APPEARANCES",
    "DF3_HOME_GOALS_CONCEDED","DF3_HOME_MINUTES_PLAYED","DF3_HOME_APPEARANCES",
    "DF4_HOME_GOALS_CONCEDED","DF4_HOME_MINUTES_PLAYED","DF4_HOME_APPEARANCES",
    "DF5_HOME_GOALS_CONCEDED","DF5_HOME_MINUTES_PLAYED","DF5_HOME_APPEARANCES",
    "DF6_HOME_GOALS_CONCEDED","DF6_HOME_MINUTES_PLAYED","DF6_HOME_APPEARANCES",
    "NO_OF_DF_HOME",
    "MF1_HOME_GOALS_CONCEDED","MF1_HOME_MINUTES_PLAYED","MF1_HOME_APPEARANCES",
    "MF2_HOME_GOALS_CONCEDED","MF2_HOME_MINUTES_PLAYED","MF2_HOME_APPEARANCES",
    "MF3_HOME_GOALS_CONCEDED","MF3_HOME_MINUTES_PLAYED","MF3_HOME_APPEARANCES",
    "MF4_HOME_GOALS_CONCEDED","MF4_HOME_MINUTES_PLAYED","MF4_HOME_APPEARANCES",
    "MF5_HOME_GOALS_CONCEDED","MF5_HOME_MINUTES_PLAYED","MF5_HOME_APPEARANCES",
    "MF6_HOME_GOALS_CONCEDED","MF6_HOME_MINUTES_PLAYED","MF6_HOME_APPEARANCES",
    "NO_OF_MF_HOME",
    "ATK1_HOME_MINUTES_PLAYED","ATK1_HOME_APPEARANCES","ATK1_HOME_GOALS_CONCEDED","ATK1_HOME_SUBSTITUTIONS_IN","ATK1_HOME_SUBSTITUTIONS_OUT",
    "ATK2_HOME_MINUTES_PLAYED","ATK2_HOME_APPEARANCES","ATK2_HOME_GOALS_CONCEDED","ATK2_HOME_SUBSTITUTIONS_IN","ATK2_HOME_SUBSTITUTIONS_OUT",
    "ATK3_HOME_MINUTES_PLAYED","ATK3_HOME_APPEARANCES","ATK3_HOME_GOALS_CONCEDED","ATK3_HOME_SUBSTITUTIONS_IN","ATK3_HOME_SUBSTITUTIONS_OUT",
    "ATK4_HOME_MINUTES_PLAYED","ATK4_HOME_APPEARANCES","ATK4_HOME_GOALS_CONCEDED","ATK4_HOME_SUBSTITUTIONS_IN","ATK4_HOME_SUBSTITUTIONS_OUT",
    "NO_OF_ATK_HOME",
    # AWAY
    "GK_AWAY_GOALS_CONCEDED",
    "DF1_AWAY_GOALS_CONCEDED","DF1_AWAY_MINUTES_PLAYED","DF1_AWAY_APPEARANCES",
    "DF2_AWAY_GOALS_CONCEDED","DF2_AWAY_MINUTES_PLAYED","DF2_AWAY_APPEARANCES",
    "DF3_AWAY_GOALS_CONCEDED","DF3_AWAY_MINUTES_PLAYED","DF3_AWAY_APPEARANCES",
    "DF4_AWAY_GOALS_CONCEDED","DF4_AWAY_MINUTES_PLAYED","DF4_AWAY_APPEARANCES",
    "DF5_AWAY_GOALS_CONCEDED","DF5_AWAY_MINUTES_PLAYED","DF5_AWAY_APPEARANCES",
    "DF6_AWAY_GOALS_CONCEDED","DF6_AWAY_MINUTES_PLAYED","DF6_AWAY_APPEARANCES",
    "NO_OF_DF_AWAY",
    "MF1_AWAY_GOALS_CONCEDED","MF1_AWAY_MINUTES_PLAYED","MF1_AWAY_APPEARANCES",
    "MF2_AWAY_GOALS_CONCEDED","MF2_AWAY_MINUTES_PLAYED","MF2_AWAY_APPEARANCES",
    "MF3_AWAY_GOALS_CONCEDED","MF3_AWAY_MINUTES_PLAYED","MF3_AWAY_APPEARANCES",
    "MF4_AWAY_GOALS_CONCEDED","MF4_AWAY_MINUTES_PLAYED","MF4_AWAY_APPEARANCES",
    "MF5_AWAY_GOALS_CONCEDED","MF5_AWAY_MINUTES_PLAYED","MF5_AWAY_APPEARANCES",
    "MF6_AWAY_GOALS_CONCEDED","MF6_AWAY_MINUTES_PLAYED","MF6_AWAY_APPEARANCES",
    "NO_OF_MF_AWAY",
    "ATK1_AWAY_MINUTES_PLAYED","ATK1_AWAY_APPEARANCES","ATK1_AWAY_GOALS_CONCEDED","ATK1_AWAY_SUBSTITUTIONS_IN","ATK1_AWAY_SUBSTITUTIONS_OUT",
    "ATK2_AWAY_MINUTES_PLAYED","ATK2_AWAY_APPEARANCES","ATK2_AWAY_GOALS_CONCEDED","ATK2_AWAY_SUBSTITUTIONS_IN","ATK2_AWAY_SUBSTITUTIONS_OUT",
    "ATK3_AWAY_MINUTES_PLAYED","ATK3_AWAY_APPEARANCES","ATK3_AWAY_GOALS_CONCEDED","ATK3_AWAY_SUBSTITUTIONS_IN","ATK3_AWAY_SUBSTITUTIONS_OUT",
    "ATK4_AWAY_MINUTES_PLAYED","ATK4_AWAY_APPEARANCES","ATK4_AWAY_GOALS_CONCEDED","ATK4_AWAY_SUBSTITUTIONS_IN","ATK4_AWAY_SUBSTITUTIONS_OUT",
    "NO_OF_ATK_AWAY",
]

# ---------- Utilitare ----------
def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _sum_over_slots(df: pd.DataFrame, role: str, side: str, stat: str, max_slots: int) -> pd.Series:
    cols = [f"{role}{i}_{side}_{stat}" for i in range(1, max_slots + 1)]
    exists = [c for c in cols if c in df.columns]
    if not exists:
        return pd.Series(0.0, index=df.index)
    return df[exists].sum(axis=1)

def _build_aggregated_df(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    mode in {"sum","mean"}:
      - păstrează GK_*_GOALS_CONCEDED și NO_OF_*_*;
      - agregă pe rol (DF/MF/ATK) pentru stats din ROLE_CFG.
    """
    out = pd.DataFrame(index=df.index)

    # GK (copiere simplă, fallback 0)
    for side in SIDES:
        gk_col = f"GK_{side}_GOALS_CONCEDED"
        out[gk_col] = df[gk_col].fillna(0.0) if gk_col in df.columns else 0.0

    for side in SIDES:
        for role, cfg in ROLE_CFG.items():
            count_col = f"NO_OF_{role}_{side}"
            out[count_col] = df[count_col].fillna(0.0) if count_col in df.columns else 0.0

            for stat in cfg["stats"]:
                s = _sum_over_slots(df, role, side, stat, cfg["max_slots"]).fillna(0.0)
                if mode == "mean":
                    denom = out[count_col].replace(0.0, np.nan)
                    val = (s / denom).fillna(0.0)
                    col_name = f"{role}_MEAN_{side}_{stat}"
                else:
                    val = s
                    col_name = f"{role}_SUM_{side}_{stat}"
                out[col_name] = val
    return out

# ---------- Reordonări pentru SUM / MEAN / SUMMEAN ----------
def _reorder_columns_sum(df_sum: pd.DataFrame) -> pd.DataFrame:
    order = [
        "GK_HOME_GOALS_CONCEDED","GK_AWAY_GOALS_CONCEDED",
        "NO_OF_DF_HOME","DF_SUM_HOME_GOALS_CONCEDED","DF_SUM_HOME_MINUTES_PLAYED","DF_SUM_HOME_APPEARANCES",
        "NO_OF_MF_HOME","MF_SUM_HOME_GOALS_CONCEDED","MF_SUM_HOME_MINUTES_PLAYED","MF_SUM_HOME_APPEARANCES",
        "NO_OF_ATK_HOME","ATK_SUM_HOME_MINUTES_PLAYED","ATK_SUM_HOME_APPEARANCES","ATK_SUM_HOME_GOALS_CONCEDED","ATK_SUM_HOME_SUBSTITUTIONS_IN","ATK_SUM_HOME_SUBSTITUTIONS_OUT",
        "NO_OF_DF_AWAY","DF_SUM_AWAY_GOALS_CONCEDED","DF_SUM_AWAY_MINUTES_PLAYED","DF_SUM_AWAY_APPEARANCES",
        "NO_OF_MF_AWAY","MF_SUM_AWAY_GOALS_CONCEDED","MF_SUM_AWAY_MINUTES_PLAYED","MF_SUM_AWAY_APPEARANCES",
        "NO_OF_ATK_AWAY","ATK_SUM_AWAY_MINUTES_PLAYED","ATK_SUM_AWAY_APPEARANCES","ATK_SUM_AWAY_GOALS_CONCEDED","ATK_SUM_AWAY_SUBSTITUTIONS_IN","ATK_SUM_AWAY_SUBSTITUTIONS_OUT",
    ]
    for c in order:
        if c not in df_sum.columns:
            df_sum[c] = 0.0
    return df_sum[order]

def _reorder_columns_mean(df_mean: pd.DataFrame) -> pd.DataFrame:
    order = [
        "GK_HOME_GOALS_CONCEDED","GK_AWAY_GOALS_CONCEDED",
        "NO_OF_DF_HOME","DF_MEAN_HOME_GOALS_CONCEDED","DF_MEAN_HOME_MINUTES_PLAYED","DF_MEAN_HOME_APPEARANCES",
        "NO_OF_MF_HOME","MF_MEAN_HOME_GOALS_CONCEDED","MF_MEAN_HOME_MINUTES_PLAYED","MF_MEAN_HOME_APPEARANCES",
        "NO_OF_ATK_HOME","ATK_MEAN_HOME_MINUTES_PLAYED","ATK_MEAN_HOME_APPEARANCES","ATK_MEAN_HOME_GOALS_CONCEDED","ATK_MEAN_HOME_SUBSTITUTIONS_IN","ATK_MEAN_HOME_SUBSTITUTIONS_OUT",
        "NO_OF_DF_AWAY","DF_MEAN_AWAY_GOALS_CONCEDED","DF_MEAN_AWAY_MINUTES_PLAYED","DF_MEAN_AWAY_APPEARANCES",
        "NO_OF_MF_AWAY","MF_MEAN_AWAY_GOALS_CONCEDED","MF_MEAN_AWAY_MINUTES_PLAYED","MF_MEAN_AWAY_APPEARANCES",
        "NO_OF_ATK_AWAY","ATK_MEAN_AWAY_MINUTES_PLAYED","ATK_MEAN_AWAY_APPEARANCES","ATK_MEAN_AWAY_GOALS_CONCEDED","ATK_MEAN_AWAY_SUBSTITUTIONS_IN","ATK_MEAN_AWAY_SUBSTITUTIONS_OUT",
    ]
    for c in order:
        if c not in df_mean.columns:
            df_mean[c] = 0.0
    return df_mean[order]

def _reorder_columns_summean(df_sum: pd.DataFrame, df_mean: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "GK_HOME_GOALS_CONCEDED","GK_AWAY_GOALS_CONCEDED",
        "NO_OF_DF_HOME","NO_OF_MF_HOME","NO_OF_ATK_HOME",
        "NO_OF_DF_AWAY","NO_OF_MF_AWAY","NO_OF_ATK_AWAY",
    ]
    for c in base_cols:
        if c not in df_sum.columns:
            df_sum[c] = 0.0
    out = df_sum[base_cols].copy()

    sum_order = [
        "DF_SUM_HOME_GOALS_CONCEDED","DF_SUM_HOME_MINUTES_PLAYED","DF_SUM_HOME_APPEARANCES",
        "MF_SUM_HOME_GOALS_CONCEDED","MF_SUM_HOME_MINUTES_PLAYED","MF_SUM_HOME_APPEARANCES",
        "ATK_SUM_HOME_MINUTES_PLAYED","ATK_SUM_HOME_APPEARANCES","ATK_SUM_HOME_GOALS_CONCEDED","ATK_SUM_HOME_SUBSTITUTIONS_IN","ATK_SUM_HOME_SUBSTITUTIONS_OUT",
        "DF_SUM_AWAY_GOALS_CONCEDED","DF_SUM_AWAY_MINUTES_PLAYED","DF_SUM_AWAY_APPEARANCES",
        "MF_SUM_AWAY_GOALS_CONCEDED","MF_SUM_AWAY_MINUTES_PLAYED","MF_SUM_AWAY_APPEARANCES",
        "ATK_SUM_AWAY_MINUTES_PLAYED","ATK_SUM_AWAY_APPEARANCES","ATK_SUM_AWAY_GOALS_CONCEDED","ATK_SUM_AWAY_SUBSTITUTIONS_IN","ATK_SUM_AWAY_SUBSTITUTIONS_OUT",
    ]
    for c in sum_order:
        if c not in df_sum.columns:
            df_sum[c] = 0.0
        out[c] = df_sum[c]

    mean_order = [
        "DF_MEAN_HOME_GOALS_CONCEDED","DF_MEAN_HOME_MINUTES_PLAYED","DF_MEAN_HOME_APPEARANCES",
        "MF_MEAN_HOME_GOALS_CONCEDED","MF_MEAN_HOME_MINUTES_PLAYED","MF_MEAN_HOME_APPEARANCES",
        "ATK_MEAN_HOME_MINUTES_PLAYED","ATK_MEAN_HOME_APPEARANCES","ATK_MEAN_HOME_GOALS_CONCEDED","ATK_MEAN_HOME_SUBSTITUTIONS_IN","ATK_MEAN_HOME_SUBSTITUTIONS_OUT",
        "DF_MEAN_AWAY_GOALS_CONCEDED","DF_MEAN_AWAY_MINUTES_PLAYED","DF_MEAN_AWAY_APPEARANCES",
        "MF_MEAN_AWAY_GOALS_CONCEDED","MF_MEAN_AWAY_MINUTES_PLAYED","MF_MEAN_AWAY_APPEARANCES",
        "ATK_MEAN_AWAY_MINUTES_PLAYED","ATK_MEAN_AWAY_APPEARANCES","ATK_MEAN_AWAY_GOALS_CONCEDED","ATK_MEAN_AWAY_SUBSTITUTIONS_IN","ATK_MEAN_AWAY_SUBSTITUTIONS_OUT",
    ]
    for c in mean_order:
        if c not in df_mean.columns:
            df_mean[c] = 0.0
        out[c] = df_mean[c]

    final_order = base_cols + sum_order + mean_order
    return out[final_order]

# ---------- Builder RAW (120 coloane) ----------
def _make_raw_from_input(df: pd.DataFrame) -> pd.DataFrame:
    """
    Selectează DOAR coloanele din RAW_FEATURES, în ordinea exactă.
    - Dacă lipsește vreo coloană, o creează cu 0.0
    - Convertește numeric și umple NaN cu 0.0
    """
    tmp = df.copy()

    # eliminăm meta și *_player_id (dacă apar)
    for c in META_COLS:
        if c in tmp.columns:
            tmp.drop(columns=c, inplace=True, errors="ignore")
    id_cols = [c for c in tmp.columns if any(substr in c for substr in DROP_IF_CONTAINS)]
    if id_cols:
        tmp.drop(columns=id_cols, inplace=True, errors="ignore")

    # garantăm existența tuturor coloanelor
    for c in RAW_FEATURES:
        if c not in tmp.columns:
            tmp[c] = 0.0

    tmp = _to_numeric_df(tmp)
    tmp = tmp.fillna(0.0)

    # păstrăm exclusiv cele 120 coloane în ordinea cerută
    raw_df = tmp[RAW_FEATURES].copy()
    return raw_df

# ---------- Main (fără CLI) ----------
def main():
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"Nu găsesc {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # 1) RAW (120 coloane fix)
    raw_df = _make_raw_from_input(df)
    raw_df.to_csv(RAW_OUT, index=False)

    # Pentru agregări, lucrăm din input dar curățăm minimal la fel:
    work = df.copy()
    for c in META_COLS:
        if c in work.columns:
            work.drop(columns=c, inplace=True, errors="ignore")
    id_cols = [c for c in work.columns if any(substr in c for substr in DROP_IF_CONTAINS)]
    if id_cols:
        work.drop(columns=id_cols, inplace=True, errors="ignore")
    work = _to_numeric_df(work).fillna(0.0)

    # 2) SUM / MEAN
    df_sum  = _build_aggregated_df(work, mode="sum")
    df_mean = _build_aggregated_df(work, mode="mean")

    sum_ordered     = _reorder_columns_sum(df_sum.copy())
    mean_ordered    = _reorder_columns_mean(df_mean.copy())
    summean_ordered = _reorder_columns_summean(df_sum.copy(), df_mean.copy())

    # 3) scriem CSV-urile
    sum_ordered.to_csv(SUM_OUT, index=False)
    mean_ordered.to_csv(MEAN_OUT, index=False)
    summean_ordered.to_csv(SUMMEAN_OUT, index=False)

    print(f"[OK] Wrote: {RAW_OUT}")
    print(f"[OK] Wrote: {SUM_OUT}")
    print(f"[OK] Wrote: {MEAN_OUT}")
    print(f"[OK] Wrote: {SUMMEAN_OUT}")

if __name__ == "__main__":
    main()
