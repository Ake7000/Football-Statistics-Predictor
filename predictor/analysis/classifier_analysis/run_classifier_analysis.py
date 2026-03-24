"""
run_classifier_analysis.py
Single entry point: runs the full classifier analysis pipeline in order.

  1. collect_classifier_results  → output/classifier_results_master.csv
  2. plot_classifier_heatmaps    → output/plots/clf_heatmap_*.png
  3. plot_classifier_rankings    → output/plots/clf_rankings_*.png
  4. plot_classifier_cv_stability→ output/plots/clf_cv_stability_*.png
  5. export_classifier_leaderboard → output/clf_leaderboard*.{csv,tex}

Usage:
    python refactor/analysis/classifier_analysis/run_classifier_analysis.py
"""

import sys
import importlib
import traceback
from pathlib import Path

# Ensure classifier_analysis/ scripts can import each other and shared_config
sys.path.insert(0, str(Path(__file__).parent))               # classifier_analysis/
sys.path.insert(0, str(Path(__file__).parent.parent.parent)) # refactor/

_PIPELINE = [
    "collect_classifier_results",
    "plot_classifier_heatmaps",
    "plot_classifier_rankings",
    "plot_classifier_cv_stability",
    "export_classifier_leaderboard",
]


def main():
    output_dir = Path(__file__).parent / "output"
    failed = []

    for script_name in _PIPELINE:
        print(f"\n{'='*60}")
        print(f"[PIPELINE] {script_name}")
        print(f"{'='*60}")
        try:
            mod = importlib.import_module(script_name)
            mod.main()
        except Exception as e:
            print(f"[ERROR] {script_name} failed: {e}")
            traceback.print_exc()
            failed.append(script_name)

    print(f"\n{'='*60}")
    print(f"[PIPELINE] Done. Output → {output_dir}")
    if failed:
        print(f"[FAILED]  {', '.join(failed)}")
    else:
        print("[STATUS]  All steps completed successfully.")


if __name__ == "__main__":
    main()
