# prototype_user/predict_from_artifacts.py
# Repaired: folosește structura pe care ai descris-o.
# - Citește raw_input.csv, sum_input.csv, mean_input.csv, summean_input.csv (deja existente)
# - Scanează artifacts_mlp_torch și artifacts_xgb
# - Pentru fiecare target și fiecare (mlp/xgb × raw/sum/mean/summean):
#   - alege runda cu MAE minim (din <run>/<TARGET>/val_metrics.json)
#   - încarcă modelul și prezice pe inputul corespunzător
# - Scrie predictions/<TARGET>/report.txt în formatul cerut și printează în terminal

import os, json, math, pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILES = {
    "raw": BASE_DIR / "raw_input.csv",
    "sum": BASE_DIR / "sum_input.csv",
    "mean": BASE_DIR / "mean_input.csv",
    "summean": BASE_DIR / "summean_input.csv",
}

ARTIFACTS = {
    "mlp": BASE_DIR.parent / "artifacts_mlp_torch",
    "xgb": BASE_DIR.parent / "artifacts_xgb",
}

VARIANTS = ["raw", "sum", "mean", "summean"]

TARGETS = [
    "HOME_GOALS","AWAY_GOALS",
    "HOME_CORNERS","AWAY_CORNERS",
    "HOME_YELLOWCARDS","AWAY_YELLOWCARDS",
    "HOME_SHOTS_ON_TARGET","AWAY_SHOTS_ON_TARGET",
    "HOME_FOULS","AWAY_FOULS",
    "HOME_OFFSIDES","AWAY_OFFSIDES",
    "HOME_REDCARDS","AWAY_REDCARDS",
]

META_COLS = ["season_label", "fixture_id", "fixture_ts", "home_team_id", "away_team_id"]
DROP_IF_CONTAINS = ["_player_id"]

PREDICTIONS_DIR = BASE_DIR / "predictions"

# ----- utils
def fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "nan"
    return f"{v:.3f}"

def safe_nanmean(vals: List[float]) -> float:
    arr = np.array(vals, dtype=float)
    return float(np.nanmean(arr)) if arr.size else float("nan")

def to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_input_variant(variant: str) -> pd.DataFrame:
    path = INPUT_FILES[variant]
    if not path.exists():
        raise FileNotFoundError(f"Missing input file for '{variant}': {path}")
    df = pd.read_csv(path)
    # defensiv: drop meta & *_player_id dacă apar
    for c in META_COLS:
        if c in df.columns:
            df.drop(columns=c, inplace=True, errors="ignore")
    id_cols = [c for c in df.columns if any(s in c for s in DROP_IF_CONTAINS)]
    if id_cols:
        df.drop(columns=id_cols, inplace=True, errors="ignore")
    df = to_numeric(df).fillna(0.0)
    return df

def align_with_feature_list(df: pd.DataFrame, feat_list_path: Optional[Path]) -> pd.DataFrame:
    """Dacă feat_list există, aliniez la el; altfel, folosesc df ca atare."""
    if feat_list_path and feat_list_path.exists():
        with open(feat_list_path, "r", encoding="utf-8") as f:
            feats = [ln.strip() for ln in f if ln.strip()]
        # adaug lipsuri cu 0, tai surplusul
        for c in feats:
            if c not in df.columns:
                df[c] = 0.0
        df = df[feats]
    # numeric + NaN->0
    df = to_numeric(df).fillna(0.0)
    return df

# ----- torch MLP
import torch
import torch.nn as nn

def get_activation(name: str) -> nn.Module:
    name = (name or "relu").lower()
    return {
        "relu": nn.ReLU, "gelu": nn.GELU, "selu": nn.SELU, "elu": nn.ELU, "swish": nn.SiLU
    }.get(name, nn.ReLU)()

class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, layer_sizes: List[int], activation: str, dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for units in layer_sizes:
            layers += [nn.Linear(prev, units), get_activation(activation)]
            if dropout and dropout > 0:
                layers += [nn.Dropout(dropout)]
            prev = units
        layers += [nn.Linear(prev, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ----- xgboost (optional)
try:
    import xgboost as xgb  # type: ignore
    XGB_OK = True
except Exception:
    XGB_OK = False

# ----- best run finder (MAE min)
def find_best_run(artifacts_root: Path, variant: str, target: str) -> Optional[Tuple[Path, float]]:
    var_dir = artifacts_root / variant
    if not var_dir.exists():
        print(f"[WARN] Variant dir missing: {var_dir}")
        return None
    best = None
    for run in sorted(p for p in var_dir.iterdir() if p.is_dir()):
        tdir = run / target
        metrics = tdir / "val_metrics.json"
        if not metrics.exists():
            # dacă nu există, sărim runda (structura ta spune că există)
            continue
        try:
            mae = float(json.load(open(metrics, "r", encoding="utf-8"))["MAE"])
        except Exception:
            continue
        if (best is None) or (mae < best[1]):
            best = (run, mae)
    if best is None:
        print(f"[WARN] No valid runs for {artifacts_root.name}/{variant}/{target}")
    return best

def pick_feat_and_scaler(run_dir: Path, target: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Caută features_list.txt & scaler.pkl:
    - întâi în rădăcina rundei
    - apoi în folderul targetului
    """
    root_feat = run_dir / "features_list.txt"
    root_scaler = run_dir / "scaler.pkl"
    targ_feat = run_dir / target / "features_list.txt"
    targ_scaler = run_dir / target / "scaler.pkl"

    feat = root_feat if root_feat.exists() else (targ_feat if targ_feat.exists() else None)
    scaler = root_scaler if root_scaler.exists() else (targ_scaler if targ_scaler.exists() else None)
    return feat, scaler

# ----- predictors
def mlp_predict_from_run(run_dir: Path, target: str, X_df: pd.DataFrame) -> Tuple[float, float]:
    tdir = run_dir / target
    metrics_path = tdir / "val_metrics.json"
    model_path   = tdir / "best_model.pt"
    mae = float(json.load(open(metrics_path, "r", encoding="utf-8"))["MAE"])

    feat_path, scaler_path = pick_feat_and_scaler(run_dir, target)
    X = align_with_feature_list(X_df.copy(), feat_path).values.astype(np.float32)

    # ✅ încercăm weights_only=True; dacă nu e suportat, revenim
    try:
        bundle = torch.load(model_path, map_location="cpu", weights_only=True)  # type: ignore
    except TypeError:
        bundle = torch.load(model_path, map_location="cpu")

    input_dim  = int(bundle["input_dim"])
    layer_sizes = list(bundle["layer_sizes"])
    activation  = bundle.get("activation", "relu")
    dropout     = float(bundle.get("dropout", 0.0))
    state_dict  = bundle["model_state_dict"]

    if scaler_path and scaler_path.exists():
        try:
            scaler = pickle.load(open(scaler_path, "rb"))
            X = scaler.transform(X)
        except Exception as e:
            print(f"[WARN] scaler load/transform failed: {e}")

    if X.shape[1] != input_dim:
        raise RuntimeError(f"Feature count mismatch for {target}: X={X.shape[1]} vs model input_dim={input_dim}")

    model = MLPRegressor(input_dim=input_dim, layer_sizes=layer_sizes, activation=activation, dropout=dropout)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        pred = float(model(torch.from_numpy(X)).cpu().numpy().reshape(-1)[0])
    return pred, mae

def xgb_predict_from_run(run_dir: Path, target: str, X_df: pd.DataFrame) -> Tuple[float, float]:
    tdir = run_dir / target
    metrics_path = tdir / "val_metrics.json"
    mae = float(json.load(open(metrics_path, "r", encoding="utf-8"))["MAE"])

    feat_path, scaler_path = pick_feat_and_scaler(run_dir, target)
    X_aligned = align_with_feature_list(X_df.copy(), feat_path).values

    # scaler dacă există (raro la XGB)
    if scaler_path and scaler_path.exists():
        try:
            scaler = pickle.load(open(scaler_path, "rb"))
            X_aligned = scaler.transform(X_aligned)
        except Exception as e:
            print(f"[WARN] scaler load/transform failed (xgb): {e}")

    json_path = tdir / "best_model.json"
    pkl_path  = tdir / "best_model.pkl"

    # ✅ 1) prefer JSON (Booster)
    if json_path.exists():
        if not XGB_OK:
            raise RuntimeError("xgboost not available to load best_model.json")
        booster = xgb.Booster()
        booster.load_model(str(json_path))
        dtest = xgb.DMatrix(X_aligned)
        pred = float(np.asarray(booster.predict(dtest)).reshape(-1)[0])
        return pred, mae

    # 2) fallback pe PKL doar dacă reușește unpickling într-un estimator utilizabil
    if pkl_path.exists():
        try:
            model = pickle.load(open(pkl_path, "rb"))
            # suportă atât XGBRegressor cât și pipeline-uri cu .predict
            if hasattr(model, "predict"):
                pred = float(np.asarray(model.predict(X_aligned)).reshape(-1)[0])
                return pred, mae
            else:
                raise TypeError("Unpickled object has no .predict()")
        except Exception as e:
            raise RuntimeError(
                f"Cannot unpickle XGB model ({pkl_path.name}): {e}. "
                f"Prefer JSON export when training (best_model.json)."
            )

    raise FileNotFoundError(f"No XGB model file in {tdir} (neither JSON nor PKL)")


# ----- report
def write_target_report(target: str, rows: List[Tuple[str, float, float]]) -> Tuple[float, float, float]:
    """
    rows: list of (label, predict, mae)

    - Scrie predictions/<target>/report.txt (detaliat, ca înainte).
    - NU mai printează nimic aici (ca să evităm dublarea).
    - Returnează (pred_mean, diff_mean, sum_mean) pentru consum în main().
    """
    target_dir = PREDICTIONS_DIR / target
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / "report.txt"

    preds = [p for _, p, _ in rows if not (p is None or math.isnan(p))]
    maes  = [m for _, _, m in rows if not (m is None or math.isnan(m) or math.isinf(m))]
    pred_mean = safe_nanmean(preds)
    mae_mean  = safe_nanmean(maes)

    lines = []
    for label, pred, mae in rows:
        if (pred is None or math.isnan(pred)) or (mae is None or math.isnan(mae) or math.isinf(mae)):
            diff = float('nan')
            sum_pm = float('nan')
        else:
            diff = max(pred - mae, 0.0)
            sum_pm = pred + mae
        lines.append(f"{label}: {fmt(diff)}, {fmt(pred)}, {fmt(sum_pm)}")

    if math.isnan(pred_mean) or math.isnan(mae_mean):
        diff_mean = float('nan')
        sum_mean  = float('nan')
    else:
        diff_mean = max(pred_mean - mae_mean, 0.0)
        sum_mean  = pred_mean + mae_mean

    lines.append(f"medie: {fmt(diff_mean)}, {fmt(pred_mean)}, {fmt(sum_mean)}")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return pred_mean, diff_mean, sum_mean


# ----- main
def main():
    # încarc toate cele 4 input-uri odată
    inputs: Dict[str, pd.DataFrame] = {v: load_input_variant(v) for v in ["raw","sum","mean","summean"]}

    # cache pentru a construi blocurile HOME_/AWAY_/TOTAL_ pe aceeași pereche
    # structure: totals[<base_target>] = {"HOME": (pred, minv, maxv), "AWAY": (...)}
    totals: Dict[str, Dict[str, Tuple[float, float, float]]] = {}

    for target in TARGETS:
        rows: List[Tuple[str, float, float]] = []
        combos = [("mlp","raw"),("mlp","sum"),("mlp","mean"),("mlp","summean"),
                  ("xgb","raw"),("xgb","sum"),("xgb","mean"),("xgb","summean")]

        for fw, variant in combos:
            label = f"{fw}_{variant}"
            root = ARTIFACTS[fw]
            best = find_best_run(root, variant, target)
            if best is None:
                rows.append((label, float("nan"), float("nan")))
                continue
            run_dir, _ = best
            try:
                if fw == "mlp":
                    pred, mae = mlp_predict_from_run(run_dir, target, inputs[variant])
                else:
                    pred, mae = xgb_predict_from_run(run_dir, target, inputs[variant])
            except Exception as e:
                print(f"[WARN] {label} {target}: {e}")
                rows.append((label, float("nan"), float("nan")))
                continue
            rows.append((label, pred, mae))

        # scriem raport + obținem sumarul pentru targetul curent
        pred_mean, minv, maxv = write_target_report(target, rows)

        # dacă targetul este HOME_* sau AWAY_*, păstrăm pentru total
        if target.startswith("HOME_") or target.startswith("AWAY_"):
            side = "HOME" if target.startswith("HOME_") else "AWAY"
            base = target.split("_", 1)[1]  # ex: GOALS, CORNERS ...
            totals.setdefault(base, {})
            totals[base][side] = (pred_mean, minv, maxv)

            # dacă ambele părți sunt prezente, afișăm blocurile cerute
            if "HOME" in totals[base] and "AWAY" in totals[base]:
                h_pred, h_min, h_max = totals[base]["HOME"]
                a_pred, a_min, a_max = totals[base]["AWAY"]

                # TOTAL = sume (rezultate și capete de interval)
                t_pred = h_pred + a_pred if not (math.isnan(h_pred) or math.isnan(a_pred)) else float("nan")
                t_min  = (h_min + a_min) if not (math.isnan(h_min) or math.isnan(a_min)) else float("nan")
                t_max  = (h_max + a_max) if not (math.isnan(h_max) or math.isnan(a_max)) else float("nan")

                # printăm blocurile suplimentare
                print(f"\n===== HOME_{base} =====")
                print(f"Rezultat: {fmt(h_pred)}, Interval: [{fmt(h_min)}; {fmt(h_max)}]")

                print(f"\n===== AWAY_{base} =====")
                print(f"Rezultat: {fmt(a_pred)}, Interval: [{fmt(a_min)}; {fmt(a_max)}]")

                print(f"\n===== TOTAL_{base} =====")
                print(f"Rezultat: {fmt(t_pred)}, Interval: [{fmt(t_min)}; {fmt(t_max)}]")

                print("\n")


if __name__ == "__main__":
    main()
