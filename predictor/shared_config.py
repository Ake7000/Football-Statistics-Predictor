# shared_config.py
# Central configuration for all optimizer scripts.
# This is the PRIMARY file to edit when:
#   - Adding new prediction targets
#   - Changing training table path
#   - Adjusting Optuna / training budget
#   - Changing loss function per target
#   - Adding new player roles or stats (see ROLE_CFG below)

from pathlib import Path

# =============================================================================
# ==  PATHS  ==================================================================
# =============================================================================
# Anchored to workspace root so scripts run correctly from any working directory.
WORKSPACE_ROOT = Path(__file__).parent.parent   # refactor/ -> licenta/

TRAIN_TABLE_PATH = (
    WORKSPACE_ROOT
    / "train_tables"
    / "fixedslots_v2__2017-2018_to_2025-2026__stats(g1-d3-m3-a5)__slots(1-6-6-4)__form5__cfg_2a5258cfb0.csv"
)

ARTIFACTS_REGRESSION_ROOT      = WORKSPACE_ROOT / "artifacts" / "regression"
ARTIFACTS_MLP_ROOT             = ARTIFACTS_REGRESSION_ROOT / "mlp_torch"
ARTIFACTS_MLP_MULTIOUTPUT_ROOT = ARTIFACTS_REGRESSION_ROOT / "mlp_multioutput_torch"
ARTIFACTS_XGB_ROOT             = ARTIFACTS_REGRESSION_ROOT / "xgb"


# =============================================================================
# ==  TARGETS  ================================================================
# =============================================================================
# Add or remove prediction targets here.  Every other script reads from this list.
TARGETS: list[str] = [
    "HOME_GOALS",           "AWAY_GOALS",
    "HOME_CORNERS",         "AWAY_CORNERS",
    "HOME_YELLOWCARDS",     "AWAY_YELLOWCARDS",
    "HOME_SHOTS_ON_TARGET", "AWAY_SHOTS_ON_TARGET",
    "HOME_FOULS",           "AWAY_FOULS",
    "HOME_OFFSIDES",        "AWAY_OFFSIDES",
    "HOME_REDCARDS",        "AWAY_REDCARDS",
]

# Classification: 7 HOME/AWAY stat pairs → 3-class direction labels each.
# Classes: 0 = HOME_WIN (HOME > AWAY), 1 = DRAW (HOME == AWAY), 2 = AWAY_WIN (HOME < AWAY).
CLASSIFIER_STAT_PAIRS: list[tuple] = [
    ("GOALS",           "HOME_GOALS",           "AWAY_GOALS"),
    ("CORNERS",         "HOME_CORNERS",         "AWAY_CORNERS"),
    ("YELLOWCARDS",     "HOME_YELLOWCARDS",     "AWAY_YELLOWCARDS"),
    ("SHOTS_ON_TARGET", "HOME_SHOTS_ON_TARGET", "AWAY_SHOTS_ON_TARGET"),
    ("FOULS",           "HOME_FOULS",           "AWAY_FOULS"),
    ("OFFSIDES",        "HOME_OFFSIDES",        "AWAY_OFFSIDES"),
    ("REDCARDS",        "HOME_REDCARDS",        "AWAY_REDCARDS"),
]
CLASSIFIER_TARGETS: list[str] = [stat for stat, _, _ in CLASSIFIER_STAT_PAIRS]
N_CLASSES: int = 3

# Loss function per target.
# Allowed values:
#   "mse"     -> nn.MSELoss()              / xgb objective "reg:squarederror"
#   "poisson" -> nn.PoissonNLLLoss()       / xgb objective "count:poisson"
# Verified against actual data distributions (1806 rows):
#   Poisson: right-skewed, notable zero% → goals (23-33% zeros, skew~1.2),
#            yellow cards, offsides, red cards.
#   MSE:     near-symmetric, very low zero% → corners, shots on target,
#            fouls (skew~0.2, only 1.7% zeros, mean~11.7).
TARGET_LOSS_MAP: dict[str, str] = {
    "HOME_GOALS":           "poisson",
    "AWAY_GOALS":           "poisson",
    "HOME_CORNERS":         "mse",
    "AWAY_CORNERS":         "mse",
    "HOME_YELLOWCARDS":     "poisson",
    "AWAY_YELLOWCARDS":     "poisson",
    "HOME_SHOTS_ON_TARGET": "mse",
    "AWAY_SHOTS_ON_TARGET": "mse",
    "HOME_FOULS":           "mse",
    "AWAY_FOULS":           "mse",
    "HOME_OFFSIDES":        "poisson",
    "AWAY_OFFSIDES":        "poisson",
    "HOME_REDCARDS":        "poisson",
    "AWAY_REDCARDS":        "poisson",
}


# =============================================================================
# ==  FEATURE ENGINEERING  ====================================================
# =============================================================================
# SIDES: which sides to build features for.
SIDES: list[str] = ["HOME", "AWAY"]

# ROLE_CFG: defines which player roles exist, how many slots each has,
# and which per-slot stats to aggregate.
#
# TO ADD A NEW ROLE: just add a new key with max_slots and stats.
# TO ADD A NEW STAT TO A ROLE: append the stat name to its "stats" list.
# The aggregation code in shared_features.py is fully driven by this dict —
# no other file needs to change.
ROLE_CFG: dict[str, dict] = {
    "GK": {
        "max_slots": 1,
        "stats": ["GOALS_CONCEDED"],
        "carry_over": True,   # GK stats are taken as-is (not summed/meaned over slots)
    },
    "DF": {
        "max_slots": 6,
        "stats": ["GOALS_CONCEDED", "MINUTES_PLAYED", "APPEARANCES"],
        "carry_over": False,
    },
    "MF": {
        "max_slots": 6,
        "stats": ["GOALS_CONCEDED", "MINUTES_PLAYED", "APPEARANCES"],
        "carry_over": False,
    },
    "ATK": {
        "max_slots": 4,
        "stats": [
            "MINUTES_PLAYED",
            "APPEARANCES",
            "GOALS_CONCEDED",
            "SUBSTITUTIONS_IN",
            "SUBSTITUTIONS_OUT",
        ],
        "carry_over": False,
    },
}


# =============================================================================
# ==  FEATURE VARIANTS  =======================================================
# =============================================================================
# Each variant is a named list of feature group names.
# Groups available: "raw", "sum", "mean", "nplayers",
#                   "form", "stage", "diffsum", "diffmean", "odds"
#
# TO ADD A NEW VARIANT: just append one entry to this dict.
# No other file needs to change.
VARIANTS: dict[str, list[str]] = {
    # --- single groups ---
    "raw":          ["raw"],
    "sum":          ["sum"],
    "mean":         ["mean"],
    "nplayers":     ["nplayers"],
    "form":         ["form"],
    "cform":        ["cform"],
    "stage":        ["stage"],
    "diffsum":      ["diffsum"],
    "diffmean":     ["diffmean"],
    "odds":         ["odds"],
    # --- combined (key = group names sorted alphabetically, joined by _) ---
    "mean_sum":                                                ["mean", "sum"],
    "mean_nplayers_sum":                                       ["mean", "nplayers", "sum"],
    "form_sum":                                                ["form", "sum"],
    "form_mean":                                               ["form", "mean"],
    "form_mean_sum":                                           ["form", "mean", "sum"],
    "diffsum_sum":                                             ["diffsum", "sum"],
    "diffmean_mean":                                           ["diffmean", "mean"],
    "diffmean_diffsum_mean_sum":                               ["diffmean", "diffsum", "mean", "sum"],
    "diffmean_diffsum_form_mean_nplayers_stage_sum":           ["diffmean", "diffsum", "form", "mean", "nplayers", "stage", "sum"],
    "diffmean_diffsum_form_mean_nplayers_raw_stage_sum":       ["diffmean", "diffsum", "form", "mean", "nplayers", "raw", "stage", "sum"],
    "cform_diffmean_diffsum_form_mean_nplayers_raw_stage_sum": ["cform", "diffmean", "diffsum", "form", "mean", "nplayers", "raw", "stage", "sum"],
    "diffmean_diffsum_mean_nplayers_sum":                      ["diffmean", "diffsum", "mean", "nplayers", "sum"],
    "form_nplayers_stage":                                     ["form", "nplayers", "stage"],
    "cform_diffmean_diffsum_form_mean_nplayers_stage_sum":     ["cform", "diffmean", "diffsum", "form", "mean", "nplayers", "stage", "sum"],
    # --- combined with odds ---
    "cform_diffmean_diffsum_form_mean_nplayers_odds_stage_sum":              ["cform", "diffmean", "diffsum", "form", "mean", "nplayers", "odds", "stage", "sum"],
    "cform_diffmean_diffsum_form_mean_nplayers_odds_raw_stage_sum":          ["cform", "diffmean", "diffsum", "form", "mean", "nplayers", "odds", "raw", "stage", "sum"],
    "diffmean_diffsum_form_mean_nplayers_odds_stage_sum":                    ["diffmean", "diffsum", "form", "mean", "nplayers", "odds", "stage", "sum"],
}


# =============================================================================
# ==  DATA CLEANING  ==========================================================
# =============================================================================
# Metadata columns present in the CSV that are not features.
META_COLS: list[str] = [
    "season_label",
    "fixture_id",
    "fixture_ts",
    "home_team_id",
    "away_team_id",
]

# Any column whose name contains one of these substrings will be dropped.
DROP_IF_CONTAINS: list[str] = ["_player_id"]


# =============================================================================
# ==  TRAIN / VAL SPLIT  ======================================================
# =============================================================================
TEST_SIZE: float   = 0.20
RANDOM_STATE: int  = 42
SHUFFLE: bool      = True

# Number of folds for final cross-validated evaluation (honest metric for reporting).
CV_FOLDS: int = 5


# =============================================================================
# ==  OPTUNA / TRAINING BUDGET  ===============================================
# =============================================================================
N_TRIALS: int  = 40
EPOCHS: int    = 200
PATIENCE: int  = 15   # early-stopping patience during Optuna search (epochs)

# Final retrain / CV folds get a longer budget — search trials already pruned bad configs.
RETRAIN_EPOCHS: int   = 400
RETRAIN_PATIENCE: int = 30

# Batch sizes Optuna can choose from (MLP only).
BATCH_SIZE_OPTIONS: list[int] = [64, 128, 256, 384, 512, 768, 1024]


# =============================================================================
# ==  MLP HYPERPARAMETER SEARCH SPACE  ========================================
# =============================================================================
N_HIDDEN_MIN: int   = 1
N_HIDDEN_MAX: int   = 4
DROPOUT_MIN: float  = 0.0
DROPOUT_MAX: float  = 0.5
LR_MIN: float       = 1e-5
LR_MAX: float       = 5e-3
L2_MIN: float       = 1e-6
L2_MAX: float       = 1e-3
ACTIVATIONS: list[str]  = ["relu", "swish"]
OPTIMIZERS: list[str]   = ["adam"]

# Hidden-layer unit choices per variant, scaled to approximate feature count.
# For variants not listed, DEFAULT_UNITS_CHOICES is used as a fallback.
# TO ADD A NEW VARIANT: add an entry below OR rely on the default.
DEFAULT_UNITS_CHOICES: list[int] = [32, 64, 128, 192]

UNITS_CHOICES: dict[str, list[int]] = {
    # ~114 raw slot features
    "raw":          [32, 64, 128, 256],
    # ~24 aggregated features
    "sum":          [8, 16, 32, 48],
    "mean":         [8, 16, 32, 48],
    # ~6
    "nplayers":     [4, 8, 16],
    # ~28 form features
    "form":         [8, 16, 32, 48],
    # ~28 continuous-form features (same scale as form)
    "cform":        [8, 16, 32, 48],
    # ~1
    "stage":        [2, 4,  8],
    # ~12 diff features
    "diffsum":      [8,  16, 32],
    "diffmean":     [8,  16, 32],
    # ~48
    "mean_sum":                                                [12, 24, 48, 72, 96],
    # ~54
    "mean_nplayers_sum":                                       [24, 32, 64, 96],
    # ~52
    "form_sum":                                                [24, 32, 64, 96],
    "form_mean":                                               [24, 32, 64, 96],
    # ~76
    "form_mean_sum":                                           [24, 48, 96, 128],
    # ~36
    "diffsum_sum":                                             [12, 24, 48, 72],
    "diffmean_mean":                                           [12, 24, 48, 72],
    # ~60
    "diffmean_diffsum_mean_sum":                               [24, 32, 64, 96, 128],
    # ~107
    "diffmean_diffsum_form_mean_nplayers_stage_sum":           [32, 64, 128, 192],
    # ~135
    "cform_diffmean_diffsum_form_mean_nplayers_stage_sum":     [32, 64, 128, 192, 256],
    # ~221
    "diffmean_diffsum_form_mean_nplayers_raw_stage_sum":       [64, 128, 256, 384],
    # ~249
    "cform_diffmean_diffsum_form_mean_nplayers_raw_stage_sum": [64, 128, 256, 384, 512],
    # ~78
    "diffmean_diffsum_mean_nplayers_sum":                      [24, 48, 96, 128],
    # ~35
    "form_nplayers_stage":                                     [8, 16, 32, 48],
}


# =============================================================================
# ==  XGBOOST HYPERPARAMETER SEARCH SPACE  ====================================
# =============================================================================
# n_estimators is intentionally NOT here — early stopping determines tree count.
XGB_N_ESTIMATORS_MAX: int = 3000   # fixed ceiling; early stopping decides actual count
XGB_EARLY_STOPPING_ROUNDS: int = 50

XGB_PARAM_SPACE: dict[str, tuple] = {
    "max_depth":         (2,    12),
    "learning_rate":     (1e-4, 0.3),    # log scale
    "subsample":         (0.5,  1.0),
    "colsample_bytree":  (0.5,  1.0),
    "colsample_bylevel": (0.5,  1.0),
    "reg_lambda":        (1e-9, 100.0),  # log scale
    "reg_alpha":         (1e-9, 10.0),   # log scale
    "min_child_weight":  (1,    20),
    "gamma":             (0.0,  10.0),
}


# =============================================================================
# ==  LSTM-MLP CONFIGURATION  =================================================
# =============================================================================

# Sequence parameters
SEQ_K: int = 5                        # past matches per team in the LSTM sequence
SEQ_STATS: list[str] = [              # stat types extracted from each past match
    "GOALS", "CORNERS", "YELLOWCARDS",
    "SHOTS_ON_TARGET", "FOULS", "OFFSIDES", "REDCARDS",
]
SEQ_INPUT_DIM: int = len(SEQ_STATS) * 2   # for + against = 14 features per step

# If True, a role token (1.0=home / 0.0=away) is appended to each LSTM input
# step, giving the shared encoder a per-team context signal at zero extra
# parameter cost.  Effective LSTM input_size = SEQ_INPUT_DIM + 1 when True.
USE_ROLE_TOKEN: bool = True

SEQ_TABLE_PATH = (
    WORKSPACE_ROOT / "train_tables" / f"seq_K{SEQ_K}.npz"
)

# Default static branch variant for LSTM-MLP optimizers.
# Override via --variant CLI argument.
LSTM_DEFAULT_STATIC_VARIANT: str = "cform_diffmean_diffsum_form_mean_nplayers_stage_sum"

# Artifact roots for LSTM-MLP optimizers (regression)
ARTIFACTS_LSTM_MLP_ROOT             = ARTIFACTS_REGRESSION_ROOT / "lstm_mlp_torch"
ARTIFACTS_LSTM_MLP_MULTIOUTPUT_ROOT = ARTIFACTS_REGRESSION_ROOT / "lstm_mlp_multioutput_torch"

# Artifact roots for classifiers
ARTIFACTS_CLASSIFICATION_ROOT                  = WORKSPACE_ROOT / "artifacts" / "classification"
ARTIFACTS_CLASSIFIER_MLP_ROOT                  = ARTIFACTS_CLASSIFICATION_ROOT / "mlp_torch"
ARTIFACTS_CLASSIFIER_MLP_MULTIOUTPUT_ROOT      = ARTIFACTS_CLASSIFICATION_ROOT / "mlp_multioutput_torch"
ARTIFACTS_CLASSIFIER_XGB_ROOT                  = ARTIFACTS_CLASSIFICATION_ROOT / "xgb"
ARTIFACTS_CLASSIFIER_LSTM_MLP_ROOT             = ARTIFACTS_CLASSIFICATION_ROOT / "lstm_mlp_torch"
ARTIFACTS_CLASSIFIER_LSTM_MLP_MULTIOUTPUT_ROOT = ARTIFACTS_CLASSIFICATION_ROOT / "lstm_mlp_multioutput_torch"

# LSTM hyperparameter search space
LSTM_HIDDEN_CHOICES: list[int] = [32, 64, 128]
LSTM_LAYERS_OPTIONS: list[int] = [1, 2]
LSTM_DROPOUT_MIN: float        = 0.0
LSTM_DROPOUT_MAX: float        = 0.4


# =============================================================================
# ==  REPRODUCIBILITY  ========================================================
# =============================================================================
GLOBAL_SEED: int = 1337
