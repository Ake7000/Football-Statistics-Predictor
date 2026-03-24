"""
shared_metrics.py

Model-agnostic metrics for regression-to-event and classification prediction.

Regression helpers:
    round_accuracy            — per-target: exact integer match rate on the val set.
    outcome_confusion_metrics — per HOME/AWAY pair: 3-class direction accuracy and
                                full confusion matrix (for later plotting).
    compute_outcome_metrics_list — batch version for all targets.

Classification helpers (used by classifier_*.py scripts):
    make_direction_labels     — derive 3-class label from HOME/AWAY values.
    clf_metrics_dict          — accuracy, macro F1, per-class precision/recall/F1,
                                and confusion matrix in dict form.

All functions accept plain NumPy arrays and are independent of model type.
"""

import numpy as np
from typing import Dict, List

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix as _sk_confusion_matrix,
)

# Class labels for the outcome direction classification.
OUTCOME_CLASSES: List[str] = ["HOME_WIN", "DRAW", "AWAY_WIN"]


def _direction(y_home: np.ndarray, y_away: np.ndarray) -> np.ndarray:
    """
    Round each value to nearest integer, then classify the pair:
      0 = HOME_WIN  (round(home) >  round(away))
      1 = DRAW      (round(home) == round(away))
      2 = AWAY_WIN  (round(home) <  round(away))
    """
    h = np.round(y_home).astype(int)
    a = np.round(y_away).astype(int)
    return np.where(h > a, 0, np.where(h == a, 1, 2))


def round_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Exact integer match accuracy.

    Both arrays are rounded to the nearest integer independently.
    Returns the fraction of samples where round(pred) == round(true), in [0, 1].
    """
    return float(np.mean(np.round(y_pred) == np.round(y_true)))


def make_direction_labels(y_home: np.ndarray, y_away: np.ndarray) -> np.ndarray:
    """
    Derive 3-class direction labels from a HOME/AWAY value pair.

    Values are rounded to the nearest integer before comparison, so this
    works correctly for both raw integer counts and regression predictions.

    Returns an int ndarray with:
      0 = HOME_WIN  (round(home) >  round(away))
      1 = DRAW      (round(home) == round(away))
      2 = AWAY_WIN  (round(home) <  round(away))
    """
    return _direction(y_home, y_away)


def clf_metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Classification metrics for 3-class direction prediction.

    Args:
        y_true: integer array of true labels  (0, 1, or 2).
        y_pred: integer array of predicted labels.

    Returns a dict with:
        accuracy            (float)
        f1_macro            (float)
        f1_per_class        {class_name: float}
        precision_per_class {class_name: float}
        recall_per_class    {class_name: float}
        confusion_matrix    list[list[int]]  — matrix[true_class][pred_class]
    """
    classes  = OUTCOME_CLASSES
    acc      = float(accuracy_score(y_true, y_pred))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1       = f1_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    prec     = precision_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    rec      = recall_score(y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0)
    cm       = _sk_confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist()
    return {
        "accuracy":            round(acc, 4),
        "f1_macro":            round(f1_macro, 4),
        "f1_per_class":        {c: round(float(f1[i]),   4) for i, c in enumerate(classes)},
        "precision_per_class": {c: round(float(prec[i]), 4) for i, c in enumerate(classes)},
        "recall_per_class":    {c: round(float(rec[i]),  4) for i, c in enumerate(classes)},
        "confusion_matrix":    cm,
    }


def outcome_confusion_metrics(
    y_true_home: np.ndarray,
    y_true_away: np.ndarray,
    y_pred_home: np.ndarray,
    y_pred_away: np.ndarray,
) -> Dict:
    """
    3-class outcome direction accuracy and confusion matrix.

    Each sample gets a class based on the rounded home vs away value:
      HOME_WIN : round(home) >  round(away)
      DRAW     : round(home) == round(away)
      AWAY_WIN : round(home) <  round(away)

    Returns:
        accuracy (float): fraction of correctly classified outcomes, in [0, 1].
        classes  (list):  ["HOME_WIN", "DRAW", "AWAY_WIN"] — index → label mapping.
        matrix   (list):  3×3 nested int list.
                          matrix[true_class][pred_class] = count.
                          Example: matrix[0][1] = samples truly HOME_WIN
                                   but predicted as DRAW.
    """
    true_cls = _direction(y_true_home, y_true_away)
    pred_cls = _direction(y_pred_home, y_pred_away)

    matrix = [
        [int(np.sum((true_cls == t) & (pred_cls == p))) for p in range(3)]
        for t in range(3)
    ]
    accuracy = float(np.mean(true_cls == pred_cls))

    return {
        "accuracy": round(accuracy, 4),
        "classes":  OUTCOME_CLASSES,
        "matrix":   matrix,  # matrix[true][pred]
    }


def compute_outcome_metrics_list(
    val_preds_by_target: Dict[str, np.ndarray],
    y_val_by_target: Dict[str, np.ndarray],
    targets: List[str],
) -> List[Dict]:
    """
    Auto-derive HOME/AWAY pairs from `targets` and compute outcome_confusion_metrics
    for each pair that has both predictions available.

    A pair (home_t, away_t) is recognised whenever:
      - home_t starts with "HOME_"
      - away_t = home_t.replace("HOME_", "AWAY_") is also in targets

    Returns a list of dicts, one per pair:
        [{"stat": "GOALS", "accuracy": ..., "classes": [...], "matrix": [...]}, ...]
    """
    pairs = [
        (t.replace("HOME_", ""), t, t.replace("HOME_", "AWAY_"))
        for t in targets
        if t.startswith("HOME_") and t.replace("HOME_", "AWAY_") in targets
    ]

    result = []
    for stat, home_t, away_t in pairs:
        if home_t not in val_preds_by_target or away_t not in val_preds_by_target:
            continue
        metrics = outcome_confusion_metrics(
            y_val_by_target[home_t], y_val_by_target[away_t],
            val_preds_by_target[home_t], val_preds_by_target[away_t],
        )
        result.append({"stat": stat, **metrics})
    return result


# =============================================================================
# ==  CLASSIFICATION LABEL HELPERS  ===========================================
# =============================================================================
# Thin wrappers used by all classifier_*.py scripts.  Centralised here so they
# are not duplicated across classifier_xgb.py, classifier_mlp_torch.py, etc.

def make_stat_labels_df(y_df, home_col: str, away_col: str) -> "np.ndarray":
    """
    Derive 3-class direction labels from a HOME/AWAY column pair.

    Accepts either a pd.DataFrame (column access) or any object supporting
    [home_col].values  (e.g. a labelled DataFrame slice).

    Returns int64 ndarray of 0/1/2 (HOME_WIN / DRAW / AWAY_WIN).
    """
    return make_direction_labels(y_df[home_col].values, y_df[away_col].values).astype(np.int64)


def make_stat_labels_arr(y_arr: "np.ndarray", home_col: str, away_col: str, all_targets) -> "np.ndarray":
    """
    Derive 3-class direction labels from a (N, n_targets) numpy array.

    Used by LSTM-based single-stat classifiers where targets are stored as
    a numpy array (not a DataFrame).

    Parameters
    ----------
    y_arr       : (N, n_targets) float array in all_targets column order.
    home_col    : name of the home-team stat column.
    away_col    : name of the away-team stat column.
    all_targets : list[str] — column names corresponding to y_arr columns.
    """
    h_idx = all_targets.index(home_col)
    a_idx = all_targets.index(away_col)
    return make_direction_labels(y_arr[:, h_idx], y_arr[:, a_idx]).astype(np.int64)


def make_all_stat_labels_df(y_df, stat_pairs) -> "np.ndarray":
    """
    Build (N, N_STATS) int64 label matrix from a targets DataFrame.

    Parameters
    ----------
    y_df       : pd.DataFrame with HOME_* / AWAY_* columns.
    stat_pairs : iterable of (stat_name, home_col, away_col) tuples
                 (typically CLASSIFIER_STAT_PAIRS from shared_config).
    """
    n     = len(y_df)
    n_s   = len(stat_pairs)
    labels = np.zeros((n, n_s), dtype=np.int64)
    for i, (_, home_col, away_col) in enumerate(stat_pairs):
        labels[:, i] = make_direction_labels(y_df[home_col].values, y_df[away_col].values)
    return labels


def make_all_stat_labels_arr(y_arr: "np.ndarray", stat_pairs, all_targets) -> "np.ndarray":
    """
    Build (N, N_STATS) int64 label matrix from a (N, n_targets) numpy array.

    Used by LSTM-based classifiers where split_seq_data() returns targets as
    a numpy array (not a DataFrame).

    Parameters
    ----------
    y_arr       : (N, n_targets) float array in TARGETS column order.
    stat_pairs  : iterable of (stat_name, home_col, away_col) tuples.
    all_targets : list[str]  — column names corresponding to y_arr columns.
    """
    n      = len(y_arr)
    n_s    = len(stat_pairs)
    labels = np.zeros((n, n_s), dtype=np.int64)
    for i, (_, home_col, away_col) in enumerate(stat_pairs):
        h_idx = all_targets.index(home_col)
        a_idx = all_targets.index(away_col)
        labels[:, i] = make_direction_labels(y_arr[:, h_idx], y_arr[:, a_idx])
    return labels
