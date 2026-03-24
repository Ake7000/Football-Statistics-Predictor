"""
run_analysis.py
Single entry point: runs the full analysis pipeline in order.

  1. collect_results   → output/results_master.csv
  2. plot_heatmaps     → output/plots/heatmap_*.png
  3. plot_rankings     → output/plots/rankings_all.png
  4. plot_cv_stability → output/plots/cv_stability.png
  5. plot_ablation     → output/plots/ablation*.png
  6. export_leaderboard → output/tables/leaderboard*.{csv,tex}

Usage:
    python refactor/analysis/optimizer_analysis/run_analysis.py
"""

import sys
import importlib
import traceback
from pathlib import Path

# Ensure optimizer_analysis scripts can import each other and shared_config
sys.path.insert(0, str(Path(__file__).parent))                    # refactor/analysis/optimizer_analysis/
sys.path.insert(0, str(Path(__file__).parent.parent.parent))      # refactor/

_PIPELINE = [
    "collect_results",
    "plot_heatmaps",
    "plot_rankings",
    "plot_cv_stability",
    "plot_ablation",
    "export_leaderboard",
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
