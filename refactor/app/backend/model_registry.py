# model_registry.py
# Unified registry: scans artifacts/regression/ AND artifacts/classification/
# to pick the best run per regression target (lowest cv_mae_mean) and
# the best run per classifier stat (highest cv_f1_macro_mean).
#
# Public API:
#   build_regression_registry(workspace_root)   -> {target: info}
#   build_classification_registry(workspace_root) -> {stat: info}

import json
import sys
from pathlib import Path
from typing import Dict

_REFACTOR_DIR = Path(__file__).resolve().parent.parent.parent   # licenta/refactor/
if str(_REFACTOR_DIR) not in sys.path:
    sys.path.insert(0, str(_REFACTOR_DIR))

from shared_config import WORKSPACE_ROOT, TARGETS, CLASSIFIER_TARGETS


# ---------------------------------------------------------------------------
# Generic artifact scanner
# ---------------------------------------------------------------------------

def _scan_artifacts(
    arts_root: Path,
    key_field: str,
    metric_field: str,
    allowed_keys: list,
    higher_is_better: bool,
    extra_fields: list = None,
) -> dict:
    """
    Walk arts_root/{model_type}/{variant}/{run_dir}/run_result.json.
    For each target entry whose `key_field` is in `allowed_keys`, keep the
    run with the best `metric_field` value.

    Returns:
        {key_value: {"model_type", "variant", "run_dir", "metric", ...extra_fields}}
    """
    extra_fields = extra_fields or []
    best: dict = {}
    if not arts_root.exists():
        return best

    for model_type_dir in sorted(arts_root.iterdir()):
        if not model_type_dir.is_dir():
            continue
        model_type = model_type_dir.name

        for variant_dir in sorted(model_type_dir.iterdir()):
            if not variant_dir.is_dir():
                continue
            variant = variant_dir.name

            for run_dir in sorted(variant_dir.iterdir()):
                if not run_dir.is_dir():
                    continue
                rr_path = run_dir / "run_result.json"
                if not rr_path.exists():
                    continue
                try:
                    rr = json.loads(rr_path.read_text(encoding="utf-8"))
                except Exception:
                    continue

                for tgt in rr.get("targets", []):
                    key_val = tgt.get(key_field)
                    if key_val not in allowed_keys:
                        continue
                    metric = tgt.get(metric_field)
                    if metric is None:
                        continue
                    metric = float(metric)
                    existing = best.get(key_val)
                    if existing is None:
                        is_better = True
                    elif higher_is_better:
                        is_better = metric > existing["metric"]
                    else:
                        is_better = metric < existing["metric"]
                    if is_better:
                        entry = {
                            "model_type": model_type,
                            "variant":    variant,
                            "run_dir":    run_dir,
                            "metric":     metric,
                        }
                        for ef in extra_fields:
                            val = tgt.get(ef)
                            if val is not None:
                                entry[ef] = float(val)
                        best[key_val] = entry
    return best


# ---------------------------------------------------------------------------
# Regression registry
# ---------------------------------------------------------------------------

def build_regression_registry(workspace_root: Path = WORKSPACE_ROOT) -> Dict[str, dict]:
    """
    Best run per regression target by lowest cv_mae_mean.

    Returns:
        {target: {"model_type", "variant", "run_dir": Path, "cv_mae": float}}
    """
    arts = workspace_root / "artifacts" / "regression"
    raw = _scan_artifacts(
        arts, "target", "cv_mae_mean", TARGETS,
        higher_is_better=False,
        extra_fields=["cv_rmse_mean"],
    )
    # Rename "metric" key to "cv_mae" for clarity
    return {
        k: {**v, "cv_mae": v.pop("metric")}
        for k, v in raw.items()
    }


# ---------------------------------------------------------------------------
# Classification registry
# ---------------------------------------------------------------------------

def build_classification_registry(workspace_root: Path = WORKSPACE_ROOT) -> Dict[str, dict]:
    """
    Best run per classifier stat by highest cv_f1_macro_mean.

    Returns:
        {stat: {"model_type", "variant", "run_dir": Path, "cv_f1": float}}
    """
    arts = workspace_root / "artifacts" / "classification"
    raw = _scan_artifacts(arts, "stat", "cv_f1_macro_mean", CLASSIFIER_TARGETS, higher_is_better=True)
    return {
        k: {**v, "cv_f1": v.pop("metric")}
        for k, v in raw.items()
    }


# ---------------------------------------------------------------------------
# Convenience: backward-compatible name used by app.py
# ---------------------------------------------------------------------------

def build_registry(workspace_root: Path = WORKSPACE_ROOT) -> Dict[str, dict]:
    """Alias for build_regression_registry (backward compat)."""
    return build_regression_registry(workspace_root)


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------

def print_regression_registry(registry: dict) -> None:
    for target, info in sorted(registry.items()):
        print(
            f"  {target:<28}  {info['model_type']:<32}  "
            f"cv_mae={info['cv_mae']:.4f}  variant={info['variant']}"
        )


def print_classification_registry(registry: dict) -> None:
    for stat, info in sorted(registry.items()):
        print(
            f"  {stat:<20}  {info['model_type']:<32}  "
            f"cv_f1={info['cv_f1']:.4f}  variant={info['variant']}"
        )


if __name__ == "__main__":
    print("=== Regression Registry ===")
    reg = build_regression_registry()
    print(f"  {len(reg)} / {len(TARGETS)} targets found")
    print_regression_registry(reg)
    missing = [t for t in TARGETS if t not in reg]
    if missing:
        print(f"  WARNING: no run found for {missing}")

    print()
    print("=== Classification Registry ===")
    clf = build_classification_registry()
    print(f"  {len(clf)} / {len(CLASSIFIER_TARGETS)} stats found")
    print_classification_registry(clf)
    missing_clf = [s for s in CLASSIFIER_TARGETS if s not in clf]
    if missing_clf:
        print(f"  WARNING: no run found for {missing_clf}")
