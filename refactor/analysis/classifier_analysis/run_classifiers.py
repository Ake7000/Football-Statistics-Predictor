"""
run_classifiers.py
Runs all classifier scripts for all specified variants × class-weight strategies.

Each model × variant × cw_strategy combination runs REPEATS=2 times.  The 2
repeats are launched as parallel subprocesses (different seeds), then waited
on together before moving to the next combination.

Edit VARIANTS_TO_RUN, MODELS_TO_RUN, CW_STRATEGIES, REPEATS at the top.

Usage (from repo root or any directory):
    python refactor/analysis/classifier_analysis/run_classifiers.py
    python refactor/analysis/classifier_analysis/run_classifiers.py --dry-run
"""

import argparse
import random
import sys
import time
import subprocess
from pathlib import Path

# =============================================================================
# == CONFIGURE HERE ===========================================================
# =============================================================================
VARIANTS_TO_RUN = [
    "cform_diffmean_diffsum_form_mean_nplayers_raw_stage_sum",
    "cform_diffmean_diffsum_form_mean_nplayers_stage_sum",
]
MODELS_TO_RUN = [
    "xgb",
    "mlp_torch",
    "mlp_multioutput_torch",
    "lstm_mlp_torch",
    "lstm_mlp_multioutput_torch",
]
CW_STRATEGIES = ["none", "sqrt"]
REPEATS       = 1       # repeats per model × variant × cw (launched in parallel)
# =============================================================================

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent  # licenta/
REFACTOR_DIR   = WORKSPACE_ROOT / "refactor"


def run_parallel(model: str, variant: str, cw_strategy: str, dry_run: bool, base_seed: int) -> bool:
    """Launch REPEATS subprocesses in parallel (each with --repeats 1) and wait for all."""
    script = REFACTOR_DIR / "classifiers" / f"classifier_{model}.py"
    if not script.exists():
        print(f"[ERROR] Script not found: {script}")
        return False

    base_cmd = [
        sys.executable, str(script),
        "--variant", variant,
        "--repeats", "1",
        "--class_weights", cw_strategy,
    ]

    print(f"\n{'='*70}")
    print(f"[RUN] model={model}  variant={variant}  cw={cw_strategy}  parallel_repeats={REPEATS}")
    print(f"{'='*70}")

    cmds = [base_cmd + ["--seed", str(base_seed + i)] for i in range(REPEATS)]

    if dry_run:
        for cmd in cmds:
            print("CMD:", " ".join(cmd))
        return True

    t0    = time.time()
    procs = [subprocess.Popen(cmd, cwd=str(WORKSPACE_ROOT)) for cmd in cmds]
    return_codes = [p.wait() for p in procs]
    elapsed = time.time() - t0

    failed_count = sum(rc != 0 for rc in return_codes)
    if failed_count:
        print(
            f"[ERROR] {model}/{variant}/cw={cw_strategy}: "
            f"{failed_count}/{REPEATS} repeat(s) failed "
            f"(exit codes: {return_codes})"
        )
        return False

    print(f"[DONE]  {model}/{variant}/cw={cw_strategy} — all repeats finished in {elapsed / 60:.1f} min")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all classifiers in parallel batches.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands that would be run without executing them.")
    args = parser.parse_args()

    base_seed = random.randint(1, 100_000)
    print(f"[SEED] Using base_seed={base_seed} for this run (seeds: {base_seed}…{base_seed + REPEATS - 1})")

    combos = [
        (model, variant, cw)
        for model in MODELS_TO_RUN
        for variant in VARIANTS_TO_RUN
        for cw in CW_STRATEGIES
    ]

    total   = len(combos)
    done    = 0
    failed  = []
    t_start = time.time()

    for model, variant, cw in combos:
        ok = run_parallel(model, variant, cw, args.dry_run, base_seed)
        done += 1
        if not ok:
            failed.append(f"{model}/{variant}/cw={cw}")
        if not args.dry_run:
            elapsed = time.time() - t_start
            print(f"[PROGRESS] {done}/{total} combos | wall={elapsed / 60:.1f} min")

    print(f"\n{'='*70}")
    print(f"[ALL DONE] {total} combos — {REPEATS} parallel repeats each")
    if not args.dry_run:
        print(f"  Total wall time: {(time.time() - t_start) / 60:.1f} min")
    if failed:
        print(f"[FAILED]   {len(failed)} combos failed: {', '.join(failed)}")
    else:
        print("[STATUS]   All combos completed successfully.")


if __name__ == "__main__":
    main()
