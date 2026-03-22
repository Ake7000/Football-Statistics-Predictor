# run_all.py
# Script that sequentially runs the main() functions from:
#   1. build_single_inputs.py
#   2. build_aggregated_inputs.py
#   3. predict_from_artifacts.py

from build_single_input_row import main as build_single_input_row_main
from build_aggregated_inputs import main as build_aggregated_inputs_main
from predict_from_artifacts import main as predict_from_artifacts_main

def main():
    print("=== Step 1: Building single inputs ===")
    build_single_input_row_main()

    print("\n=== Step 2: Building aggregated inputs ===")
    build_aggregated_inputs_main()

    print("\n=== Step 3: Predicting from artifacts ===")
    predict_from_artifacts_main()

if __name__ == "__main__":
    main()
