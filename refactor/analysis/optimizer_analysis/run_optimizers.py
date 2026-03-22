"""
run_optimizers.py
Runs all optimizers for all specified variants.

Each model×variant combination runs REPEATS=2 times. The 2 repeats are
launched as parallel subprocesses (to better utilize GPU time-slicing),
then waited on together before moving to the next combination.

Edit VARIANTS_TO_RUN, MODELS_TO_RUN, REPEATS at the top to control which
combinations are executed.

Usage:
    python refactor/analysis/optimizer_analysis/run_optimizers.py
"""

import random
import sys
import time
import subprocess
from pathlib import Path

# =============================================================================
# == CONFIGURE HERE ===========================================================
# =============================================================================
VARIANTS_TO_RUN = [
    "cform_diffmean_diffsum_form_mean_nplayers_odds_stage_sum",
    "cform_diffmean_diffsum_form_mean_nplayers_odds_raw_stage_sum",
    "cform_diffmean_diffsum_form_mean_nplayers_stage_sum",
    "cform_diffmean_diffsum_form_mean_nplayers_raw_stage_sum",
]
MODELS_TO_RUN   = [
    "mlp_torch",
    "mlp_multioutput_torch",
    "xgb",
    "lstm_mlp_torch",
    "lstm_mlp_multioutput_torch",
]
REPEATS         = 2   # total repeats per model×variant (run in parallel batches of 2)

# Per-model variant override.  If a model appears here, only these variants
# are run for it instead of VARIANTS_TO_RUN.
MODEL_VARIANTS_OVERRIDE = {
}
# =============================================================================

WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent   # optimizer_analysis/ → analysis/ → refactor/ → licenta/
REFACTOR_DIR   = WORKSPACE_ROOT / "refactor"


def run_parallel(model: str, variant: str, base_seed: int) -> bool:
    """Launch REPEATS subprocesses in parallel (each with --repeats 1) and wait for all."""
    script = REFACTOR_DIR / "optimizers" / f"optimizer_{model}.py"
    if not script.exists():
        print(f"[ERROR] Script not found: {script}")
        return False

    cmd = [sys.executable, str(script), "--variant", variant, "--repeats", "1"]

    print(f"\n{'='*70}")
    print(f"[RUN] model={model}  variant={variant}  parallel_repeats={REPEATS}")
    print(f"{'='*70}")

    t0 = time.time()
    procs = [
        subprocess.Popen(
            cmd + ["--seed", str(base_seed + i)],
            cwd=str(WORKSPACE_ROOT)
        )
        for i in range(REPEATS)
    ]

    # Wait for all and collect return codes
    return_codes = [p.wait() for p in procs]
    elapsed = time.time() - t0

    failed_count = sum(rc != 0 for rc in return_codes)
    if failed_count:
        print(f"[ERROR] {model}/{variant}: {failed_count}/{REPEATS} repeat(s) failed "
              f"(exit codes: {return_codes})")
        return False

    print(f"[DONE]  {model}/{variant} — both repeats finished in {elapsed / 60:.1f} min")
    return True


def main():
    base_seed = random.randint(1, 100_000)
    print(f"[SEED] Using base_seed={base_seed} for this run (seeds: {base_seed}\u2026{base_seed + REPEATS - 1})")

    total   = sum(len(MODEL_VARIANTS_OVERRIDE.get(m, VARIANTS_TO_RUN)) for m in MODELS_TO_RUN)
    done    = 0
    failed  = []
    t_start = time.time()

    for model in MODELS_TO_RUN:
        variants = MODEL_VARIANTS_OVERRIDE.get(model, VARIANTS_TO_RUN)
        for variant in variants:
            ok = run_parallel(model, variant, base_seed)
            done += 1
            if not ok:
                failed.append(f"{model}/{variant}")
            elapsed = time.time() - t_start
            print(f"[PROGRESS] {done}/{total} combos | wall={elapsed / 60:.1f} min")

    print(f"\n{'='*70}")
    print(f"[ALL DONE] {total} combos in {(time.time() - t_start) / 60:.1f} min")
    if failed:
        print(f"[FAILED]   {len(failed)} combos failed: {', '.join(failed)}")
    else:
        print("[STATUS]   All combos completed successfully.")


if __name__ == "__main__":
    main()
