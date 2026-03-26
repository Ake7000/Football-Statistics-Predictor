# UCL Match Statistics Predictor

A full end-to-end machine learning system for predicting UEFA Champions League (UCL) match statistics, combined with a systematic research platform for finding the best models, architectures, and feature combinations for that task.

---

## Overview

This project has two intertwined goals:

**1. Build a working predictor.**
Given two teams and their selected lineups, predict the statistical outcome of a Champions League match across 14 targets:

| Side   | Statistics predicted |
|--------|----------------------|
| Home   | Goals, Corners, Yellow Cards, Shots on Target, Fouls, Offsides, Red Cards |
| Away   | Goals, Corners, Yellow Cards, Shots on Target, Fouls, Offsides, Red Cards |

On top of the 14 regression predictions, 7 **direction classifiers** (one per stat pair) predict which side will "win" each statistic — Home Win / Draw / Away Win — adding a tactical layer on top of the raw numbers.

**2. Research which approach works best.**
The project systematically experiments with:
- **5 model architectures**: single-output MLP, multi-output MLP, single-output LSTM-MLP, multi-output LSTM-MLP, and XGBoost.
- **20+ feature group combinations** (player stats, rolling form, stage context, inter-team differences, OOF classifier probabilities as features, etc.).
- **Loss function assignment**: Poisson NLL for heavy-tailed count stats (goals, yellow cards, offsides, red cards) vs. MSE for near-symmetric stats (corners, shots on target, fouls).

Everything — from raw API scraping through feature engineering, model training, comparative analysis, and a live prediction app — lives in this repository.

---

### Dataset scope

- **Source:** [Sportmonks Football API v3](https://www.sportmonks.com/) — UEFA Champions League (league ID = 2)
- **Seasons scraped:** 25 seasons, from 2000–2001 through 2025–2026
- **Training window:** 2017–2018 → 2025–2026 (justified by data coverage analysis — see §5)
- **Training rows:** 1,806 fixtures
- **Player data per fixture:** up to 17 players per side (GK ×1, DF ×6, MF ×6, ATK ×4) with per-player career statistics from the prior season

---

### Technology stack

| Layer | Tools |
|-------|-------|
| Data collection | Python, `requests`, Sportmonks API v3 |
| Data processing | `pandas`, `numpy` |
| Deep learning | PyTorch (MLP, LSTM-MLP with residual blocks) |
| Gradient boosting | XGBoost |
| Hyperparameter search | Optuna (TPE sampler + Median pruner) |
| Cross-validation | Scikit-learn (K-Fold, StandardScaler) |
| Visualisation | Matplotlib, Seaborn |
| Web application | Streamlit |

---

## 2. Repository Structure

```
licenta/
│
├── data_scraping/              # Scripts that pull raw data from the Sportmonks API
│
├── data/                       # Raw scraped data — one folder per UCL season
│   └── YYYY-YYYY_<season_id>/
│       ├── data/               # Season metadata
│       ├── fixtures/           # Per-fixture: match stats, lineups, scores
│       ├── teams/              # Per-team: squad rosters, team statistics
│       └── players/            # Per-player: current + prior season statistics
│
├── data_vis_scripts/           # Visualisation & coverage-analysis scripts
├── data_visualisation/         # Output plots and CSVs from the above scripts
│
├── train_tables/               # Generated training CSV + sequence/odds NPZ files
│
├── predictor/                  # ★ Main project package
│   ├── shared_config.py        # Central config: targets, loss types, roles, variants
│   ├── shared_features.py      # Feature group assembler
│   ├── shared_metrics.py       # Evaluation helpers (RMSE, round accuracy, etc.)
│   ├── shared_preprocessing.py # Data loading & cleaning pipeline
│   ├── shared_sequence.py      # LSTM-MLP architecture + sequence dataset utilities
│   ├── shared_utils.py         # Seeds, device detection, ResidualMLP, optimizer factory
│   │
│   ├── table_creation/         # Training table builders (CSV, sequences, OOF odds)
│   │
│   ├── optimizers/             # Regression model trainers (5 architectures × N variants)
│   ├── classifiers/            # Direction classifier trainers (5 architectures × N variants)
│   │
│   ├── analysis/               # Post-training artifact analysis & plotting pipelines
│   │   ├── optimizer_analysis/ # Regression: collect results → heatmaps → rankings → leaderboard
│   │   └── classifier_analysis/# Classification: same pipeline structure
│   │
│   └── app/                    # Streamlit prediction application
│       ├── app.py              # Main app entry point (UI layout, pitch, controls)
│       └── backend/            # Inference pipeline modules
│           ├── data_layer.py       # Team lists & player rosters from current season
│           ├── feature_builder.py  # Assembles player stats + rolling form
│           ├── raw_row_builder.py  # Builds full-width training-schema DataFrame row
│           ├── sequence_builder.py # Builds LSTM input sequences for a team
│           ├── model_registry.py   # Scans artifacts → best model per target
│           └── inference.py        # End-to-end: row → classifiers → odds → regressors → output
│
├── artifacts/                  # Saved model runs (auto-generated, not committed)
│   ├── regression/             # One subfolder per model_type / variant / timestamped run
│   └── classification/         # Same structure for classifiers
│
└── ucl_seasons_ids.json        # Mapping of season labels to Sportmonks season IDs
```

### How the folders relate to the pipeline

```
data_scraping/  →  data/  →  data_vis_scripts/  →  (choose training window)
                                                          ↓
                              predictor/table_creation/  →  train_tables/
                                                          ↓
                              predictor/classifiers/     →  artifacts/classification/
                                                          ↓
                              predictor/table_creation/generate_odds_features.py
                                                          ↓
                              predictor/optimizers/      →  artifacts/regression/
                                                          ↓
                              predictor/analysis/        →  plots, leaderboards
                                                          ↓
                              predictor/app/             →  live predictions
```

Each stage is independent and rerunnable. The only strict ordering constraints are:
1. `data_scraping/` must run before `predictor/table_creation/`
2. Classifiers must be trained before `generate_odds_features.py`
3. `generate_odds_features.py` must run before any optimizer that uses the `odds` feature group

---

## 3. Data Collection — `data_scraping/`

All scraping targets the **Sportmonks Football API v3** (`https://api.sportmonks.com/v3/football`).

### Configuration

API credentials and endpoint settings are loaded from a `.env` file in the project root:

```
SPORTMONKS_API_TOKEN=<your_token>
SPORTMONKS_BASE_URL=https://api.sportmonks.com/v3/football
UCL_LEAGUE_ID=2
```

All four scripts use a shared pattern:
- Paginated GET requests with `per_page=50` (Sportmonks default)
- Automatic **429 rate-limit handling**: detects the `Retry-After` header, sleeps the required time, then retries
- **5xx backoff**: exponential retry on server errors
- **Atomic file writes**: data is written to a temp file first, then renamed — preventing corrupt JSON if the script is interrupted mid-write
- **Skip-if-exists logic**: already-fetched files are not re-requested, making scripts safe to re-run after interruptions

---

### `seasons.py` — Discover all UCL seasons

**What it does:**
Queries the Sportmonks API for all seasons belonging to the UCL league (ID=2). For each season found, it:
1. Records the season ID, label (e.g. `"2023/2024"`), start/end dates, and whether it is finished.
2. Saves the full list to `ucl_seasons_ids.json` in the project root.
3. Creates the directory skeleton under `data/` for every season:

```
data/
└── 2023-2024_21638/
    ├── data/        ← season metadata will go here
    ├── fixtures/    ← match data will go here
    ├── teams/       ← team rosters will go here
    └── players/     ← player stats will go here
```

**Run order:** This must be the first script executed. Everything else depends on the `data/` structure it creates.

```bash
python data_scraping/seasons.py
```

---

### `fixtures.py` — Collect all match data

**What it does:**
Iterates over every season folder created by `seasons.py`. For each season, it fetches all fixtures from the API and for each fixture creates a timestamped subdirectory:

```
data/2023-2024_21638/fixtures/
└── 2023-09-19T21-00-00_19156735/
    ├── data.json        ← fixture metadata
    ├── statistics.json  ← match-level statistics
    └── lineup.json      ← starting lineups + formation
```

**`data.json` contents:**
- `fixture_id`, `state_id` (scheduled / live / finished), `stage_id`, `round_id`
- `home_team_id`, `away_team_id`
- Final score (home goals, away goals)
- Kick-off date/time

**`statistics.json` contents — the prediction targets:**

| Statistic | API key |
|-----------|---------|
| Goals | `GOALS` |
| Corners | `CORNERS` |
| Yellow Cards | `YELLOWCARDS` |
| Shots on Target | `SHOTS_ON_TARGET` |
| Fouls | `FOULS` |
| Offsides | `OFFSIDES` |
| Red Cards | `REDCARDS` |

Each stat is stored for both `home` and `away` participants, giving 14 values per fixture.

**`lineup.json` contents:**
- List of starting players per team
- Each entry: `player_id`, `position_id` (1=GK, 2=DF, 3=MF, 4=ATK), `type_id` (starter vs. substitute), jersey number
- Formation string (e.g. `"4-3-3"`)

**Run:**
```bash
python data_scraping/fixtures.py
```

---

### `teams.py` — Collect squad rosters and team statistics

**What it does:**
For every team that appears across all seasons, creates a folder under the season's `teams/` directory and fetches two resources:

```
data/2023-2024_21638/teams/
└── Manchester_City_9/
    ├── squad.json        ← full player roster with jersey numbers + positions
    └── statistics.json   ← team-level aggregate statistics for the season
```

**`squad.json`** contains an array of player entries, each with:
- `player_id`, `player.name`
- `position_id` (role: GK / DF / MF / ATK)
- `jersey_number`
- `details[]` — per-stat values for the current season (appearances, minutes played, goals, etc.)

**`statistics.json`** contains team-level aggregates (not used directly in the ML pipeline but retained for potential future features).

**Run:**
```bash
python data_scraping/teams.py
```

---

### `players.py` — Collect per-player career statistics

**What it does:**
This is the most granular scraping step. For every player in every team's squad, it creates a folder under `players/` and fetches two files:

```
data/2023-2024_21638/players/
└── Erling_Haaland_123456/
    ├── current_statistics.json    ← stats for the season being scraped
    └── last_year_statistics.json  ← stats from the prior season (used as ML input)
```

**Why `last_year_statistics.json` matters:**
At prediction time, we do not know what a player will do *in* the current season — we only know what they did *last* season. This is what gets used as the input feature for each player slot. It reflects the player's form going into the season: minutes played, appearances, goals conceded (for defenders/keepers), and substitution patterns.

The specific fields extracted per player, per position:

| Position | Stats used as features |
|----------|----------------------|
| GK (×1) | `GOALS_CONCEDED` |
| DF (×6) | `GOALS_CONCEDED`, `MINUTES_PLAYED`, `APPEARANCES` |
| MF (×6) | `GOALS_CONCEDED`, `MINUTES_PLAYED`, `APPEARANCES` |
| ATK (×4) | `MINUTES_PLAYED`, `APPEARANCES`, `GOALS_CONCEDED`, `SUBSTITUTIONS_IN`, `SUBSTITUTIONS_OUT` |

**Run:**
```bash
python data_scraping/players.py
```

---

### Scraping order

Scripts must be executed in this sequence, as each one depends on the directory structure created by the previous:

```
seasons.py  →  fixtures.py  →  teams.py  →  players.py
```

The full dataset (25 seasons, 200+ fixtures each, ~3,000 players per season) takes several hours to collect due to API rate limits.

---

## 4. Raw Data — `data/`

The `data/` folder is the direct output of the scraping step. It contains one subfolder per UCL season, named using the format `YYYY-YYYY_<season_id>` (e.g. `2023-2024_21638`). In total, **25 seasons** are stored, spanning 2000–2001 through 2025–2026.

### Season folder layout

```
data/
├── 2000-2001_24218/
├── 2001-2002_24217/
│   ...
├── 2023-2024_21638/
│   ├── data/
│   │   └── data.json
│   ├── fixtures/
│   │   ├── 2023-09-19T21-00-00_19156735/
│   │   │   ├── data.json
│   │   │   ├── statistics.json
│   │   │   └── lineup.json
│   │   └── ... (200+ fixture folders)
│   ├── teams/
│   │   ├── Manchester_City_9/
│   │   │   ├── squad.json
│   │   │   └── statistics.json
│   │   └── ... (80+ team folders)
│   └── players/
│       ├── Erling_Haaland_123456/
│       │   ├── current_statistics.json
│       │   └── last_year_statistics.json
│       └── ... (~3,000 player folders)
└── 2025-2026_<id>/
```

---

### `data/data.json` — Season metadata

Stores high-level information about the season itself:

```json
{
  "id": 21638,
  "name": "2023/2024",
  "finished": true,
  "is_current": false,
  "starting_at": "2023-06-27",
  "ending_at": "2024-06-01"
}
```

The `finished` and `is_current` flags are used by some scraping scripts to skip processing of incomplete seasons gracefully.

---

### `fixtures/<timestamp>_<id>/data.json` — Match metadata

```json
{
  "fixture_id": 19156735,
  "state_id": 5,
  "stage_id": 77457863,
  "round_id": 274151,
  "home_team_id": 9,
  "away_team_id": 14,
  "scores": { "home": 3, "away": 1 },
  "starting_at": "2023-09-19T21:00:00+00:00"
}
```

`state_id = 5` means the match is finished. Other states (scheduled, live, postponed) are also present in the data but filtered out during table construction.

---

### `fixtures/<timestamp>_<id>/statistics.json` — Match statistics

The core prediction targets. Stored as an array of participant objects, one for home and one for away:

```json
[
  {
    "location": "home",
    "details": [
      { "type": { "developer_name": "GOALS" },            "value": { "total": 3 } },
      { "type": { "developer_name": "CORNERS" },          "value": { "total": 7 } },
      { "type": { "developer_name": "YELLOWCARDS" },      "value": { "total": 1 } },
      { "type": { "developer_name": "SHOTS_ON_TARGET" },  "value": { "total": 6 } },
      { "type": { "developer_name": "FOULS" },            "value": { "total": 11 } },
      { "type": { "developer_name": "OFFSIDES" },         "value": { "total": 2 } },
      { "type": { "developer_name": "REDCARDS" },         "value": { "total": 0 } }
    ]
  },
  { "location": "away", "details": [ ... ] }
]
```

Not every statistic is present in every fixture — especially for older seasons. The data coverage analysis (§5) was done specifically to determine which seasons have reliable data for each stat.

---

### `fixtures/<timestamp>_<id>/lineup.json` — Starting lineups

```json
[
  {
    "team_id": 9,
    "formation": "4-3-3",
    "players": [
      { "player_id": 123456, "position_id": 1, "type_id": 11, "jersey_number": 31 },
      { "player_id": 789012, "position_id": 2, "type_id": 11, "jersey_number": 3  },
      ...
    ]
  },
  { "team_id": 14, "formation": "4-2-3-1", "players": [ ... ] }
]
```

`type_id = 11` means starting XI; substitutes are also present but not used in training. `position_id` maps to the role system: 1=GK, 2=DF, 3=MF, 4=ATK.

---

### `teams/<TeamName>_<id>/squad.json` — Player roster

Contains the full registered squad for that team in that season (~25–32 players). Each entry includes:

```json
{
  "player_id": 123456,
  "player": { "id": 123456, "name": "Erling Haaland" },
  "position_id": 4,
  "jersey_number": 9,
  "details": [
    { "type": { "developer_name": "GOALS" }, "value": { "total": 36 } },
    ...
  ]
}
```

This file is used by the app to populate the player-picker dropdowns for the current season.

---

### `players/<PlayerName>_<id>/last_year_statistics.json` — Prior-season player stats

This is the file used directly as **ML input features**. It contains the player's statistics from the season *before* the one being modelled — the only information realistically available before a match is played.

```json
{
  "position_id": 4,
  "statistics": [
    {
      "details": [
        { "type": { "developer_name": "MINUTES_PLAYED" },      "value": { "total": 2765 } },
        { "type": { "developer_name": "APPEARANCES" },         "value": { "total": 35  } },
        { "type": { "developer_name": "GOALS_CONCEDED" },      "value": { "total": 26  } },
        { "type": { "developer_name": "SUBSTITUTIONS_IN" },    "value": { "total": 3   } },
        { "type": { "developer_name": "SUBSTITUTIONS_OUT" },   "value": { "total": 7   } }
      ]
    }
  ]
}
```

Missing stats (e.g. a goalkeeper with no `SUBSTITUTIONS_IN` entry) are imputed to 0 during table construction.

---

### Scale summary

| Dimension | Count |
|-----------|-------|
| Seasons scraped | 25 (2000–2001 → 2025–2026) |
| Seasons used for training | 9 (2017–2018 → 2025–2026) |
| Fixtures in training set | 1,806 |
| Teams per season (approx.) | 32–80 (qualifying rounds increase this) |
| Players per season (approx.) | ~3,000 |
| JSON files total | ~500,000+ |

---

## 5. Data Visualisation — `data_vis_scripts/`

Before building any training table, it is necessary to understand the quality and coverage of the raw data. Not all seasons have all statistics recorded, and not all player positions have all stat fields populated. The scripts in `data_vis_scripts/` answer two key questions:

1. **Which seasons have reliable data for each match statistic?**
2. **Which seasons have reliable player statistics for each position?**

The answers directly determined the choice of **2017–2018 as the start of the training window**. Outputs (plots + CSV files) are saved to `data_visualisation/`.

---

### `fixtures_heatmap.py` — Match statistics coverage

**What it does:**
Scans every `statistics.json` file across all 25 seasons. For each (season, statistic) pair it computes the **fraction of finished fixtures that have that statistic recorded** — a value between 0.0 (never present) and 1.0 (always present).

The result is a pivot table (seasons × stats) saved as `fixture_stat_coverage.csv`, and visualised as a colour-coded heatmap where:
- **Dark green** → stat present in nearly all matches
- **Yellow/red** → stat frequently missing

**Key insight it provides:**
Early seasons (pre-2010) are missing CORNERS, SHOTS_ON_TARGET, FOULS and OFFSIDES in a large fraction of fixtures. This heatmap makes the coverage drop-off visually obvious and provides the justification for not training on those seasons, even though the fixture results (scores) are available.

**Run:**
```bash
python data_vis_scripts/fixtures_heatmap.py
```

---

### `players_heatmap.py` — Player statistics coverage per position

**What it does:**
Scans every `last_year_statistics.json` file across all seasons, grouped by player position (GK, DF, MF, ATK). For each (season, position, stat) combination it computes the fraction of players who have that stat recorded.

Produces one heatmap per position (saved as PNG), and a summary CSV. For example, the GK heatmap shows when `GOALS_CONCEDED` became reliably present; the ATK heatmap shows when `SUBSTITUTIONS_IN` / `SUBSTITUTIONS_OUT` became available.

**Key insight it provides:**
Player stat coverage is even patchier than fixture coverage in early seasons. In some pre-2015 seasons, fewer than 30% of defenders have `MINUTES_PLAYED` recorded at all. These gaps would create very noisy training rows — essentially rows where most of the player features are imputed to zero and carry no signal. This heatmap highlights which seasons are safe to include.

**Run:**
```bash
python data_vis_scripts/players_heatmap.py
```

---

### `players.py` — Player statistics distributions

**What it does:**
For each position and each stat, produces bar charts showing the **distribution of values** across all players in the dataset — e.g., the distribution of `APPEARANCES` for midfielders, or of `GOALS_CONCEDED` for goalkeepers.

This is used to understand whether a feature has a meaningful spread (useful for the model) or is near-constant across players (less informative). It also helps spot outliers and confirms that the imputation value of 0 for missing stats is reasonable relative to the true distribution.

**Run:**
```bash
python data_vis_scripts/players.py
```

---

### `find_common_max_rectangles.py` — Optimal training window selection

**What it does:**
This is the most analytical of the four scripts. It formalises the training-window selection problem as finding the **largest contiguous block of seasons** where *all* required statistics meet a minimum coverage threshold, simultaneously across all required player positions.

The algorithm:
1. Loads the coverage pivot CSVs produced by `fixtures_heatmap.py` and `players_heatmap.py`.
2. For a configurable set of required stats and player categories, builds a binary matrix: `1` if coverage ≥ threshold (e.g. 0.80), `0` otherwise.
3. Finds the maximum contiguous column range (seasons) where every required cell is `1` — essentially a "maximum rectangle of ones" problem.
4. Outputs candidate windows ranked by length, with per-stat coverage summaries.

**Key insight it provides:**
The analysis confirmed that the largest window where all 7 match stats and all 4 player-position stat groups meet an 80% coverage threshold starts at **2017–2018**. Including earlier seasons would require accepting either very sparse feature rows or dropping certain stats from the prediction task entirely. The result is saved as a summary CSV in `data_visualisation/`.

**Run:**
```bash
python data_vis_scripts/find_common_max_rectangles.py
```

---

### Summary: what these scripts decided

| Decision | Justified by |
|----------|-------------|
| Training starts at 2017–2018 | `find_common_max_rectangles.py` — all 7 stats + all 4 positions reach ≥80% coverage from that season onward |
| GK uses only `GOALS_CONCEDED` | `players_heatmap.py` — it is the only GK stat with consistent coverage; minutes/appearances for GKs are unreliable |
| ATK uses 5 stats including substitution counts | `players_heatmap.py` — `SUBSTITUTIONS_IN/OUT` coverage for attackers is reliable from 2017 onward |
| OFFSIDES and FOULS included despite lower early coverage | `fixtures_heatmap.py` — within the 2017–2026 window coverage is >90% for both |

---

## 6. Training Table Construction — `predictor/table_creation/`

This folder contains the scripts that transform the raw JSON data in `data/` into structured, model-ready files in `train_tables/`. Three artefacts are produced, in a fixed order:

```
build_table_v2.py          →  train_tables/*.csv          (main feature table)
build_sequence_table.py    →  train_tables/seq_K5.npz     (LSTM input sequences)
generate_odds_features.py  →  train_tables/odds_oof.npz   (OOF classifier probabilities)
```

Two helper modules support the builders: `shared_table_utils.py` and `form_stage_utils.py`.

---

### `shared_table_utils.py` — Shared utilities

A self-contained library used by all other scripts in this folder. Key components:

- **`TARGET_KEYS`** — the 7 stats tracked per match: `GOALS`, `CORNERS`, `YELLOWCARDS`, `SHOTS_ON_TARGET`, `FOULS`, `OFFSIDES`, `REDCARDS`
- **`STAT_FILL_VALUES`** — imputation constants for missing fixture stats. Rare stats like `OFFSIDES` and `REDCARDS` that are absent from some records are filled with sensible defaults rather than dropped.
- **JSON helpers** — safe readers for all four file types (`data.json`, `statistics.json`, `lineup.json`, `squad.json`)
- **Lineup parsing** — extracts starting XI players per team, grouped by position
- **Fixture resolution** — determines which fixtures are finished and usable for training, filtering out postponed, cancelled, or live matches

---

### `form_stage_utils.py` — Rolling form and stage normalisation

**Stage normalisation:**
Maps each `stage_id` from the API to a float in `[0.0, 1.0]`:
- `0.0` → qualifying rounds (earliest stage)
- `1.0` → final

This scalar is included in training as the `STAGE_NORMALIZED` feature, giving the model context about how deep into the competition the match is.

**`TeamFormTracker`:**
Maintains a per-team rolling deque of the last N matches (default N=5). For each team, it tracks both the FOR (scored/conceded from that team's perspective) and AGAINST values for all 7 stats.

Before each fixture is processed, the tracker produces a snapshot of that team's form *at that point in time*. This snapshot becomes 28 columns in the training row:

```
FORM_HOME_GOALS_FOR,   FORM_HOME_GOALS_AGAINST,
FORM_HOME_CORNERS_FOR, FORM_HOME_CORNERS_AGAINST,
... (7 stats × 2 directions × 2 sides = 28 columns)
```

A second variant, **`cform`** (continuous form), uses exponentially weighted averaging instead of a simple rolling mean — giving more weight to recent matches.

**Critical design detail:** form values are read *before* each fixture is added to the tracker, then the tracker is updated *after* writing the row. This ensures no data leakage — the model never sees the outcome of the match it is being asked to predict.

---

### `build_table_v2.py` — Main training table builder

**What it produces:**
A single CSV file saved to `train_tables/`, named to encode the full configuration:
```
fixedslots_v2__2017-2018_to_2025-2026__stats(g1-d3-m3-a5)__slots(1-6-6-4)__form5__cfg_<hash>.csv
```
Where:
- `stats(g1-d3-m3-a5)` — stats per role: GK=1, DF=3, MF=3, ATK=5
- `slots(1-6-6-4)` — player slots per role: GK=1, DF=6, MF=6, ATK=4
- `form5` — rolling form window of 5 matches
- `cfg_<hash>` — MD5 hash of the full config dict for reproducibility

**Two-pass build process:**

*Pass 1* — scans all fixtures in all seasons to build:
- A `stage_id → STAGE_NORMALIZED` mapping
- A fixture stats cache (all match outcomes)
- Chronological fixture ordering within each season

*Pass 2* — iterates fixtures in chronological order. For each finished fixture with a valid lineup:
1. Reads home and away team form snapshots from `TeamFormTracker` (no leakage)
2. Looks up each player in the starting XI, loads their `last_year_statistics.json`
3. Fills player slots in fixed order: GK slot 1, then DF slots 1–6, MF slots 1–6, ATK slots 1–4
4. Computes `NO_OF_DF_HOME`, `NO_OF_MF_HOME` etc. (actual number of players filling each role)
5. Writes one CSV row
6. Updates `TeamFormTracker` with this fixture's result

Fixtures with missing lineups are **not** written as rows, but they **do** update the form tracker — a team's form is informed by all their results, not only the ones where lineups are available.

**Output schema (230 columns):**

| Column group | Example columns | Count |
|---|---|---|
| Meta | `season_label`, `fixture_id`, `fixture_ts`, `home_team_id`, `away_team_id` | 5 |
| GK player stats | `GK_{side}_GOALS_CONCEDED` | 2 |
| DF player stats | `DF{n}_{side}_GOALS_CONCEDED`, `DF{n}_{side}_MINUTES_PLAYED`, `DF{n}_{side}_APPEARANCES` | 36 |
| MF player stats | `MF{n}_{side}_GOALS_CONCEDED`, `MF{n}_{side}_MINUTES_PLAYED`, `MF{n}_{side}_APPEARANCES` | 36 |
| ATK player stats | `ATK{n}_{side}_MINUTES_PLAYED`, `ATK{n}_{side}_APPEARANCES`, `ATK{n}_{side}_GOALS_CONCEDED`, `ATK{n}_{side}_SUBSTITUTIONS_IN`, `ATK{n}_{side}_SUBSTITUTIONS_OUT` | 40 |
| Player counts | `NO_OF_GK_{side}`, `NO_OF_DF_{side}`, `NO_OF_MF_{side}`, `NO_OF_ATK_{side}` | 8 |
| Player IDs | `GK_{side}_player_id`, `DF{n}_{side}_player_id`, `MF{n}_{side}_player_id`, `ATK{n}_{side}_player_id` | 34 |
| Form (rolling) | `FORM_{side}_GOALS_FOR`, `FORM_{side}_GOALS_AGAINST`, `FORM_{side}_CORNERS_FOR`, `FORM_{side}_CORNERS_AGAINST`, `FORM_{side}_YELLOWCARDS_FOR`, `FORM_{side}_YELLOWCARDS_AGAINST`, `FORM_{side}_SHOTS_ON_TARGET_FOR`, `FORM_{side}_SHOTS_ON_TARGET_AGAINST`, `FORM_{side}_FOULS_FOR`, `FORM_{side}_FOULS_AGAINST`, `FORM_{side}_OFFSIDES_FOR`, `FORM_{side}_OFFSIDES_AGAINST`, `FORM_{side}_REDCARDS_FOR`, `FORM_{side}_REDCARDS_AGAINST` | 28 |
| Continuous form | `CFORM_{side}_GOALS_FOR`, `CFORM_{side}_GOALS_AGAINST`, `CFORM_{side}_CORNERS_FOR`, `CFORM_{side}_CORNERS_AGAINST`, `CFORM_{side}_YELLOWCARDS_FOR`, `CFORM_{side}_YELLOWCARDS_AGAINST`, `CFORM_{side}_SHOTS_ON_TARGET_FOR`, `CFORM_{side}_SHOTS_ON_TARGET_AGAINST`, `CFORM_{side}_FOULS_FOR`, `CFORM_{side}_FOULS_AGAINST`, `CFORM_{side}_OFFSIDES_FOR`, `CFORM_{side}_OFFSIDES_AGAINST`, `CFORM_{side}_REDCARDS_FOR`, `CFORM_{side}_REDCARDS_AGAINST` | 28 |
| Stage | `STAGE_NORMALIZED` | 1 |
| Targets | `{side}_GOALS`, `{side}_CORNERS`, `{side}_YELLOWCARDS`, `{side}_SHOTS_ON_TARGET`, `{side}_FOULS`, `{side}_OFFSIDES`, `{side}_REDCARDS` | 14 |

**Run:**
```bash
python predictor/table_creation/build_table_v2.py
```

---

### `build_sequence_table.py` — LSTM sequence tensors

**What it produces:**
`train_tables/seq_K5.npz` — a NumPy archive with the following arrays:

| Key | Shape | Description |
|-----|-------|-------------|
| `fixture_ids` | `(1806,)` | Fixture IDs in the same row order as the CSV |
| `home_seq` | `(1806, 5, 14)` | Home team's last 5 match feature vectors |
| `away_seq` | `(1806, 5, 14)` | Away team's last 5 match feature vectors |
| `home_mask` | `(1806, 5)` bool | `True` = real match, `False` = zero-padding |
| `away_mask` | `(1806, 5)` bool | Same for away |

**Feature vector per sequence step (14 floats):**
For each of the 7 stats, two values from the team's perspective: `FOR` (what they scored/committed) and `AGAINST` (what the opponent scored/committed against them). This gives the LSTM a time-series view of how a team has been performing across their 5 most recent matches.

**Zero-padding:**
If a team has played fewer than K=5 prior matches at the time of a given fixture (e.g. early in their first season in the training window), missing steps are filled with zeros and flagged in the mask. Approximately 16.9% of all sequence steps are zero-padded.

**Why sequences in addition to form columns?**
The rolling form columns in the CSV capture aggregated statistics (mean over 5 matches). The sequence tensors preserve the *temporal structure* — the model can learn that a team's performance in the most recent match matters more than one from 5 matches ago, and detect trends (improving form, declining form). This is the feature that the LSTM-MLP architectures exploit.

**Run:**
```bash
python predictor/table_creation/build_sequence_table.py
```

---

### `generate_odds_features.py` — OOF classifier probabilities

**What it produces:**
`train_tables/odds_oof.npz` — 21 out-of-fold probability columns, 3 per stat:

```
odds_GOALS_home,   odds_GOALS_draw,   odds_GOALS_away,
odds_CORNERS_home, odds_CORNERS_draw, odds_CORNERS_away,
... (7 stats × 3 classes = 21 columns)
```

**What "OOF" means and why it matters:**
OOF stands for *Out-Of-Fold*. When adding classifier predictions as features for a regression model, a naive approach would be to train the classifier on all data and use those predictions as features — but the classifier would have already seen the labels it is predicting, causing severe data leakage.

Instead, this script uses **K-fold cross-validation** to generate held-out predictions:
- For each fold, the classifier is trained on the other K−1 folds
- Predictions (softmax probabilities) are collected only for the held-out fold
- The K sets of held-out predictions are stitched back into the original row order

The result is 21 columns where every row's prediction was made by a model that **never saw that row during training** — a realistic estimate of what the classifier would produce at inference time.

**How the best classifier is selected:**
The script scans `artifacts/classification/` for saved runs, finds the model with the highest `cv_f1_macro_mean` for each of the 7 stats, reconstructs that model configuration, and re-runs K-fold CV to generate the OOF probabilities.

**Run (after classifiers have been trained):**
```bash
python predictor/table_creation/generate_odds_features.py
```

---

### Output files in `train_tables/`

| File | Size | Description |
|------|------|-------------|
| `fixedslots_v2__...cfg_<hash>.csv` | ~1,806 rows × 230 cols | Main training table |
| `seq_K5.npz` | ~16 MB | LSTM sequence tensors |
| `seq_K5.info.txt` | text | Human-readable summary of the NPZ |
| `odds_oof.npz` | ~140 KB | OOF classifier probabilities |

---

## 7. Shared Modules — `predictor/shared_*.py`

These six files form the backbone of the entire ML pipeline. Every optimizer, classifier, and app backend module imports from them. Centralising configuration and logic here ensures that a single change (e.g. adding a new prediction target or feature group) propagates consistently everywhere.

---

### `shared_config.py` — Central configuration

The single source of truth for the entire project. All other modules import their constants from here.

**Prediction targets:**

14 regression targets — 7 stats × 2 sides:
```python
REGRESSION_TARGETS = [
    "HOME_GOALS", "AWAY_GOALS",
    "HOME_CORNERS", "AWAY_CORNERS",
    "HOME_YELLOWCARDS", "AWAY_YELLOWCARDS",
    "HOME_SHOTS_ON_TARGET", "AWAY_SHOTS_ON_TARGET",
    "HOME_FOULS", "AWAY_FOULS",
    "HOME_OFFSIDES", "AWAY_OFFSIDES",
    "HOME_REDCARDS", "AWAY_REDCARDS",
]
```

7 classification targets — one direction label per stat pair:
```python
CLASSIFICATION_TARGETS = [
    "GOALS", "CORNERS", "YELLOWCARDS", "SHOTS_ON_TARGET",
    "FOULS", "OFFSIDES", "REDCARDS",
]
# Each predicts: 0 = HOME_WIN, 1 = DRAW, 2 = AWAY_WIN
```

**Loss function assignment:**
A key modelling decision — which loss function to use per target:

| Target group | Loss | Rationale |
|---|---|---|
| GOALS, YELLOWCARDS, OFFSIDES, REDCARDS | Poisson NLL | Right-skewed distributions, high zero% — classic count data |
| CORNERS, SHOTS_ON_TARGET, FOULS | MSE | More symmetric distributions, few zeros |

**Player role configuration (`ROLE_CFG`):**

Defines how many slots are reserved per position and which stats to read for each:

| Role | Slots | Stats |
|------|-------|-------|
| GK | 1 | `GOALS_CONCEDED` |
| DF | 6 | `GOALS_CONCEDED`, `MINUTES_PLAYED`, `APPEARANCES` |
| MF | 6 | `GOALS_CONCEDED`, `MINUTES_PLAYED`, `APPEARANCES` |
| ATK | 4 | `MINUTES_PLAYED`, `APPEARANCES`, `GOALS_CONCEDED`, `SUBSTITUTIONS_IN`, `SUBSTITUTIONS_OUT` |

If a team plays fewer than 6 defenders (e.g. a 3-5-2 formation), the unused DF slots are filled with zeros. The `NO_OF_DF_HOME` column records how many slots are actually filled.

**Feature variants:**
Named combinations of feature groups used for the ablation study. Examples:

```python
VARIANTS = {
    "raw":    ["raw"],
    "sum":    ["sum"],
    "mean":   ["mean"],
    "form":   ["form"],
    "diffsum_diffmean_form_mean_nplayers_stage_sum": [
        "diffsum", "diffmean", "form", "mean", "nplayers", "stage", "sum"
    ],
    "cform_diffmean_diffsum_form_mean_nplayers_odds_raw_stage_sum": [
        "cform", "diffmean", "diffsum", "form", "mean", "nplayers", "odds", "raw", "stage", "sum"
    ],
    # ... 20+ total variants
}
```

**Training parameters:**
```python
CV_FOLDS     = 5
TEST_SIZE    = 0.20
RANDOM_STATE = 42
```

---

### `shared_features.py` — Feature group assembler

`build_X(df, groups)` takes a DataFrame (the training CSV) and a list of group names, and assembles the input feature matrix by selecting and concatenating the relevant columns. `get_y(df)` returns the 14-column target matrix.

**Available feature groups:**

| Group | What it contains | Approx. columns |
|-------|-----------------|-----------------|
| `raw` | Per-slot player stats as-scraped (e.g. `GK_HOME_GOALS_CONCEDED`, `DF3_AWAY_MINUTES_PLAYED`) | ~114 |
| `sum` | Sum of each stat across all slots per role per side (e.g. `SUM_DF_HOME_GOALS_CONCEDED`) | ~18 |
| `mean` | `sum` ÷ number of filled slots | ~18 |
| `nplayers` | Filled slot count per role per side (`NO_OF_GK_HOME`, `NO_OF_DF_AWAY`, ...) | 8 |
| `form` | Rolling 5-match averages: FOR/AGAINST × 7 stats × 2 sides | 28 |
| `cform` | Continuous (exponentially weighted) form — same structure as `form` | 28 |
| `diffsum` | `SUM_*_HOME - SUM_*_AWAY` for each stat | ~9 |
| `diffmean` | `MEAN_*_HOME - MEAN_*_AWAY` for each stat | ~9 |
| `stage` | `STAGE_NORMALIZED` (single float, 0.0=qualifying → 1.0=final) | 1 |
| `odds` | 21 OOF classifier softmax probabilities (3 classes × 7 stats) | 21 |

Combining all groups produces ~177 features. The ablation study (§10) evaluates which subsets produce the best models.

---

### `shared_metrics.py` — Evaluation helpers

Provides consistent metric computation across all optimizers and classifiers:

- **`round_accuracy(y_true, y_pred)`** — fraction of predictions where `round(pred) == round(true)`. For example, predicting 2.7 when the answer is 3 counts as correct. This is the metric most meaningful for end users.
- **`make_direction_labels(home_vals, away_vals)`** — converts raw target pairs into 3-class direction labels (0=HOME_WIN, 1=DRAW, 2=AWAY_WIN). Used to evaluate whether regressors implicitly get the direction right.
- **`outcome_confusion_metrics(y_true, y_pred)`** — 3×3 confusion matrix + per-class accuracy for a stat pair. Applied to regression outputs after rounding.
- **`clf_metrics_dict(y_true, y_pred)`** — accuracy, macro F1, and per-class precision/recall/F1 for classifier evaluation.

---

### `shared_preprocessing.py` — Data loading and cleaning

`load_and_prepare_dataframe(csv_path)` is the single entry point for loading the training CSV. Steps:

1. Load CSV with pandas
2. Drop meta columns (`season_label`, `fixture_id`, `fixture_ts`, team ID columns)
3. Drop all `*_player_id` columns (identifiers, not features)
4. Cast all remaining columns to numeric (coerces bad values to NaN)
5. Impute NaN in **target** columns → `0` (a missing stat is treated as zero)
6. Impute NaN in **feature** columns → `0` (unfilled player slots carry no information)
7. Drop columns that are constant zero across the entire dataset (they add no signal)

The cleaned DataFrame is what all optimizers and classifiers receive as input.

---

### `shared_sequence.py` — LSTM-MLP architecture

Defines the full LSTM-MLP hybrid model and its dataset utilities.

**Architecture:**

```
home_seq  (B, K, F) ──► LSTMEncoder ──► h_home  (B, H)  ─┐
away_seq  (B, K, F) ──► LSTMEncoder ──► h_away  (B, H)  ─┤
                         (shared or separate weights)      │
static_x  (B, D)    ──► MLPEncoder  ──► h_stat  (B, S)  ─┤
                                                           ▼
                                             concat → FusionHead → output
```

- **K = 5** sequence steps, **F = 14** features per step (7 stats × FOR/AGAINST)
- `LSTMEncoder`: stacked LSTM layers + optional role token appended to each step input
- `MLPEncoder`: residual MLP blocks on static features (StandardScaler-normalised)
- `FusionHead`: MLP that takes the concatenated LSTM and MLP representations
- `use_shared_lstm=True/False`: when `True`, the same LSTM weights are used for both home and away sequences (parameter sharing); when `False`, two independent encoders are used

**Role token:** An optional scalar (1.0 for home team, 0.0 for away team) appended to each LSTM step. This lets a single shared encoder distinguish which side it is processing.

**Dataset class:**
`SequenceDataset` wraps the static feature matrix, sequence tensors, masks, and targets into a PyTorch `Dataset`, handling the alignment between CSV rows and NPZ rows by fixture ID.

---

### `shared_utils.py` — General utilities

- **`set_all_seeds(seed)`** — sets Python, NumPy, and PyTorch (CPU + CUDA) seeds for full reproducibility
- **`get_torch_device()`** — returns `"cuda"` if a GPU is available, otherwise `"cpu"`
- **`get_xgb_device()`** — same, returning the XGBoost-compatible device string
- **`ResidualMLP`** — a pre-norm residual block: `LayerNorm → Linear → Activation → Dropout → Linear → residual add`. Stacked to build all MLP-based models in the project
- **`build_layer_sizes(base_units, n_hidden)`** — generates a list of layer widths (halving each layer)
- **`make_torch_optimizer(name, params, lr, l2)`** — factory for Adam / AdamW / SGD from a name string; used by Optuna to include optimizer type in the search space
- **`make_run_dir(artifacts_root, csv_path, variant, suffix)`** — creates a timestamped, uniquely-named output directory for each training run, encoding the config in the folder name

---

## 8. Regression Optimizers — `predictor/optimizers/`

The optimizers are the training scripts for the **regression task**: predicting the 14 raw numeric target values (goals, corners, etc.) for both sides of a match.

Five architectures are implemented. Each is driven by **Optuna** for automated hyperparameter search and evaluated using **K-fold cross-validation** (K=5). Results are saved to `artifacts/regression/`.

---

### Common training loop design

All five optimizers follow the same outer structure:

1. **Load data** via `shared_preprocessing.load_and_prepare_dataframe()`
2. **Build feature matrix** via `shared_features.build_X(df, groups)` for the selected variant
3. **Define Optuna objective**: for each trial, sample hyperparameters, run K-fold CV, return the mean CV RMSE (minimised)
4. **Run study**: `optuna.create_study(direction="minimize", sampler=TPESampler(), pruner=MedianPruner())`
5. **Retrain on full data** with best hyperparameters found
6. **Save artefacts** to a timestamped run directory

Each optimizer is launched with command-line arguments:
```bash
python predictor/optimizers/optimizer_<type>.py \
    --variant <variant_name> \
    --repeats <N> \
    --seed <int>
```

`--repeats` runs the full optimisation N times with different seeds, producing multiple independent runs for later averaging in the analysis step.

---

### Artefact structure

Every completed run produces a directory under `artifacts/regression/<model_type>/<variant>/<timestamp>/`:

```
artifacts/regression/mlp_torch/diffmean_form_mean_stage_sum/20251014-022646/
├── run_result.json      ← all metrics, hyperparameters, config
├── scaler.pkl           ← fitted StandardScaler
├── model.pt             ← saved PyTorch model (or model.pkl for XGBoost)
└── features_list.txt    ← exact ordered feature names used during training
```

**`run_result.json` contains:**
- `model_type`, `variant`, `seed`, `n_features`, `n_train`, `n_val`
- Overall val RMSE and CV RMSE
- Per-target metrics: `val_mae`, `val_rmse`, `round_accuracy`, `cv_mae_mean`, `cv_rmse_mean`, `cv_rmse_std`, `best_params`
- Per-stat direction metrics: `outcome_accuracy`, `confusion_matrix` (3×3)

`features_list.txt` is critical for inference — it ensures that the model always receives features in exactly the order it was trained on, regardless of any future changes to the feature assembler.

---

### `optimizer_mlp_torch.py` — Single-output MLP

**Model:** 14 **independent** `ResidualMLP` regressors, one per target. Each is trained and optimised separately — the hyperparameters for `HOME_GOALS` need not match those for `AWAY_REDCARDS`.

**Hyperparameter search space:**

| Parameter | Search range |
|-----------|-------------|
| `n_hidden` | 1 – 5 residual blocks |
| `base_units` | {32, 64, 128, 256, 512} |
| `activation` | swish, relu, elu, selu |
| `dropout` | 0.0 – 0.5 |
| `learning_rate` | 1e-4 – 1e-2 (log scale) |
| `weight_decay` | 1e-6 – 1e-2 (log scale) |
| `optimizer` | Adam, AdamW, SGD |
| `batch_size` | {32, 64, 128, 256} |

**Loss per target:** Poisson NLL or MSE, as defined in `shared_config.LOSS_MAP`.

**Run:**
```bash
python predictor/optimizers/optimizer_mlp_torch.py --variant sum --repeats 3
```

---

### `optimizer_mlp_multioutput_torch.py` — Multi-output MLP

**Model:** A **shared backbone** `ResidualMLP` followed by 14 independent output heads (one per target). All heads are trained jointly in a single pass, with the total loss being the mean of all 14 per-target losses.

**Difference from single-output:** The shared backbone forces the model to learn a common representation useful for predicting all 14 targets simultaneously. This can improve performance when targets are correlated (e.g. `HOME_GOALS` and `HOME_SHOTS_ON_TARGET` are related), but can also hurt if targets are only loosely related.

**Same hyperparameter space** as the single-output MLP, with the addition of a `head_hidden` parameter controlling whether each output head has extra layers.

**Run:**
```bash
python predictor/optimizers/optimizer_mlp_multioutput_torch.py --variant mean --repeats 3
```

---

### `optimizer_lstm_mlp_torch.py` — Single-output LSTM-MLP

**Model:** For each of the 14 targets, an independent LSTM-MLP hybrid (architecture defined in `shared_sequence.py`). The LSTM processes the K=5 historical match sequences for home and away teams; the MLP processes the static features; a fusion head combines both representations.

**Additional hyperparameters vs. MLP:**

| Parameter | Search range |
|-----------|-------------|
| `lstm_hidden` | {32, 64, 128, 256} |
| `lstm_layers` | 1 – 3 |
| `lstm_dropout` | 0.0 – 0.4 |
| `use_shared_lstm` | True / False |
| `use_role_token` | True / False |

Requires `train_tables/seq_K5.npz` to be present.

**Run:**
```bash
python predictor/optimizers/optimizer_lstm_mlp_torch.py --variant form --repeats 3
```

---

### `optimizer_lstm_mlp_multioutput_torch.py` — Multi-output LSTM-MLP

**Model:** Same LSTM-MLP hybrid architecture, but with a **shared backbone + 14 output heads**, trained jointly. Combines the temporal modelling capability of the LSTM with the multi-task learning benefit of shared representations.

This is the most parameter-rich model in the project and has the largest search space. It is the most expensive to train.

**Run:**
```bash
python predictor/optimizers/optimizer_lstm_mlp_multioutput_torch.py --variant cform_diffmean_diffsum_form_mean_nplayers_stage_sum --repeats 2
```

---

### `optimizer_xgb.py` — XGBoost

**Model:** 14 independent **XGBoost** gradient-boosted tree regressors, one per target.

**Objective per target:**
- `count:poisson` for GOALS, YELLOWCARDS, OFFSIDES, REDCARDS
- `reg:squarederror` for CORNERS, SHOTS_ON_TARGET, FOULS

**Hyperparameter search space:**

| Parameter | Search range |
|-----------|-------------|
| `max_depth` | 3 – 10 |
| `learning_rate` | 0.01 – 0.3 (log scale) |
| `n_estimators` | 100 – 1000 |
| `subsample` | 0.5 – 1.0 |
| `colsample_bytree` | 0.5 – 1.0 |
| `colsample_bylevel` | 0.5 – 1.0 |
| `gamma` | 0.0 – 5.0 |
| `min_child_weight` | 1 – 10 |
| `reg_alpha` | 1e-8 – 10.0 (log scale) |
| `reg_lambda` | 1e-8 – 10.0 (log scale) |

Automatically uses GPU (`device='cuda'`) when available.

**Run:**
```bash
python predictor/optimizers/optimizer_xgb.py --variant raw --repeats 3
```

---

### Running all optimizers at once

The analysis pipeline (§10) includes a `run_optimizers.py` script that launches all model × variant combinations in parallel subprocesses:

```bash
python predictor/analysis/optimizer_analysis/run_optimizers.py
```

This is the recommended way to run a full experiment sweep.

---

## 9. Direction Classifiers — `predictor/classifiers/`

The classifiers solve a parallel but distinct problem from the regressors: instead of predicting *how many* goals/corners/etc. will occur, they predict *which side will win each statistic*.

### The classification task

For each of the 7 stat pairs, the classifier predicts one of three outcomes:

| Class | Label | Meaning |
|-------|-------|---------|
| 0 | `HOME_WIN` | Home side had a higher value |
| 1 | `DRAW` | Both sides had equal values |
| 2 | `AWAY_WIN` | Away side had a higher value |

This gives **7 independent 3-class classification problems**, one per stat. The classifiers are trained and evaluated separately, and the best model per stat is later used to generate the OOF odds features (§6).

### Why classify at all?

Two reasons:

1. **As a standalone prediction:** Knowing that a team is more likely to win the corner battle or the shot battle is tactically meaningful, even without knowing the exact numbers.

2. **As features for regressors:** The OOF softmax probabilities from the classifiers (21 columns: 3 classes × 7 stats) are fed as input features to the regression models under the `odds` feature group. This gives the regressors a pre-computed signal about likely match direction, which can improve regression accuracy — especially for targets whose distributions are heavily skewed by match context.

---

### Common design

The classifiers mirror the optimizer structure exactly:
- Same Optuna + K-fold CV training loop
- Same `--variant`, `--repeats`, `--seed` CLI arguments
- Results saved to `artifacts/classification/<model_type>/<variant>/<timestamp>/`

**Key difference from regressors:** classifiers use `CrossEntropyLoss` instead of Poisson/MSE, and report `accuracy` + `macro F1` rather than RMSE/MAE.

**Class imbalance handling:**  
`DRAW` (equal values for both sides) is a less common outcome for most stats, especially goals and corners. Classifiers expose an optional `--class-weight balanced` flag, which computes inverse-frequency weights for each class and passes them to `CrossEntropyLoss`. This prevents the model from simply predicting HOME_WIN or AWAY_WIN for every row.

**Artefact contents** (same structure as regressors):
```
artifacts/classification/mlp_torch/form_mean_stage_sum/20251014-110538/
├── run_result.json      ← cv_f1_macro_mean, per-class precision/recall/F1, confusion matrix
├── scaler.pkl
├── model.pt / model.pkl
└── features_list.txt
```

---

### `classifier_mlp_torch.py` — Single-output MLP classifier

**Model:** 7 independent `ResidualMLP` classifiers, one per stat. Each outputs 3 logits fed to `CrossEntropyLoss`.

Same hyperparameter search space as `optimizer_mlp_torch.py`, with the addition of `class_weight` (balanced / none).

**Run:**
```bash
python predictor/classifiers/classifier_mlp_torch.py --variant form --repeats 3
```

---

### `classifier_mlp_multioutput_torch.py` — Multi-output MLP classifier

**Model:** Shared backbone + 7 classification heads (one per stat), trained jointly. Total loss = mean of 7 `CrossEntropyLoss` values.

Useful when the classification targets are correlated — for example, a team that dominates shots is also likely to win the corner count, so a shared representation may capture this covariance.

**Run:**
```bash
python predictor/classifiers/classifier_mlp_multioutput_torch.py --variant mean --repeats 3
```

---

### `classifier_lstm_mlp_torch.py` — Single-output LSTM-MLP classifier

**Model:** 7 independent LSTM-MLP hybrids, one per stat. The LSTM processes the K=5 match history sequences; the classification head replaces the regression head.

**Run:**
```bash
python predictor/classifiers/classifier_lstm_mlp_torch.py --variant cform_form_stage --repeats 2
```

---

### `classifier_lstm_mlp_multioutput_torch.py` — Multi-output LSTM-MLP classifier

**Model:** Shared LSTM-MLP backbone + 7 classification heads. The most expressive classifier variant.

**Run:**
```bash
python predictor/classifiers/classifier_lstm_mlp_multioutput_torch.py --variant cform_diffmean_form_mean_stage --repeats 2
```

---

### `classifier_xgb.py` — XGBoost classifier

**Model:** 7 independent XGBoost classifiers, one per stat. Uses `multi:softmax` objective with 3 classes and `mlogloss` as the evaluation metric. Automatically uses GPU when available.

Same hyperparameter search space as `optimizer_xgb.py`.

**Run:**
```bash
python predictor/classifiers/classifier_xgb.py --variant form --repeats 3
```

---

### Running all classifiers at once

```bash
python predictor/analysis/classifier_analysis/run_classifiers.py
```

This should be done **before** running `generate_odds_features.py`, since it needs trained classifier artefacts to generate the OOF probabilities.

---

### Classifier performance context

Some stats are inherently easier to classify than others:

| Stat | Difficulty | Reason |
|------|-----------|--------|
| GOALS | Hard | Outcomes are noisy and near-random even for strong teams |
| REDCARDS | Easy | Draws (0–0 red cards for both sides) dominate → high accuracy but low F1 |
| CORNERS | Medium | Correlated with possession and attacking style, somewhat predictable |
| SHOTS_ON_TARGET | Medium | Correlated with team quality and formation |

This is why macro F1 (which weights all three classes equally) is used as the primary metric rather than accuracy — it avoids rewarding models that achieve high accuracy by only predicting the majority class.

---

## 10. Analysis & Research — `predictor/analysis/`

After training runs are complete, the `analysis/` folder provides tools to systematically compare all saved artefacts, identify the best configurations, and produce publication-quality visualisations. It contains two parallel sub-pipelines — one for regression, one for classification — each following the same logical flow.

```
predictor/analysis/
├── optimizer_analysis/    ← regression model comparison
└── classifier_analysis/   ← classifier model comparison
```

---

### 10.1 Regression analysis — `optimizer_analysis/`

#### Step 1 — `collect_results.py` — Aggregate all artefacts

Recursively scans `artifacts/regression/` for every `run_result.json`. For runs with multiple repeats (same model_type + variant, different seeds), it averages the metrics across all seeds to produce a stable estimate.

Outputs: **`results_master.csv`** — one row per (model_type, variant, target) combination, with columns:
- `cv_rmse_mean`, `cv_rmse_std`, `cv_mae_mean`
- `val_rmse`, `val_mae`, `round_accuracy`
- `outcome_accuracy` (direction classification from regression outputs)
- `n_runs` (how many seeds were averaged)

This CSV is the input for all subsequent plotting and export scripts.

**Run:**
```bash
python predictor/analysis/optimizer_analysis/collect_results.py
```

---

#### Step 2 — `plot_heatmaps.py` — CV-RMSE heatmaps

Produces one heatmap per prediction target (14 plots). Each heatmap has:
- **Rows:** model types (mlp_torch, mlp_multioutput_torch, lstm_mlp_torch, lstm_mlp_multioutput_torch, xgb)
- **Columns:** feature variants
- **Cell colour:** CV-RMSE (lower = better, darker green)

This gives an immediate visual overview of which model × variant combination works best for each target, and makes it easy to spot whether LSTM models consistently outperform MLP models or vice versa.

---

#### Step 3 — `plot_rankings.py` — Overall model rankings

Aggregates CV-RMSE across all 14 targets (mean or sum) and ranks all model × variant configurations from best to worst. Produces a bar chart and a ranked table.

This is the primary output used to answer the research question *"which combination is best overall?"*

---

#### Step 4 — `plot_cv_stability.py` — Cross-validation fold stability

For each configuration, plots the per-fold RMSE values across the K=5 folds (plus mean and standard deviation). Configurations with high fold-to-fold variance are flagged as potentially unstable — their good average performance may not generalise reliably.

This is important because CV-RMSE alone can be misleading: a model with mean RMSE 1.20 but fold RMSEs of {0.80, 1.60, 0.90, 1.55, 1.25} is less reliable than one with consistent {1.18, 1.21, 1.22, 1.19, 1.20}.

---

#### Step 5 — `plot_ablation.py` — Feature group ablation

Produces incremental ablation charts: starting from the best full-feature variant, it shows how performance changes as feature groups are removed one by one. This answers *"which feature groups contribute the most?"*

For example, it might show that removing `odds` features increases RMSE by 0.05 for GOALS but has negligible effect on CORNERS, indicating that classifier probabilities are especially valuable for predicting goals.

---

#### Step 6 — `export_leaderboard.py` — Best model per target

Reads `results_master.csv` and selects the single best configuration for each of the 14 regression targets (lowest `cv_mae_mean`). Exports:
- A **CSV leaderboard** with model_type, variant, and all metrics per target
- A **LaTeX table** formatted for inclusion in a thesis or paper

---

#### Full pipeline in one command — `run_optimizers_analysis.py`

Orchestrates all 6 steps above in sequence:

```bash
python predictor/analysis/optimizer_analysis/run_optimizers_analysis.py
```

Alternatively, to first train all models and then analyse:
```bash
python predictor/analysis/optimizer_analysis/run_optimizers.py       # trains all models
python predictor/analysis/optimizer_analysis/run_optimizers_analysis.py  # analyses results
```

---

### 10.2 Classifier analysis — `classifier_analysis/`

An exact mirror of the regression analysis pipeline, adapted for classification metrics.

| Script | Output |
|--------|--------|
| `collect_classifier_results.py` | `classifier_results_master.csv` — cv_f1_macro_mean, per-class F1, accuracy |
| `plot_classifier_heatmaps.py` | Heatmap of macro F1 per model_type × variant, per stat |
| `plot_classifier_rankings.py` | Ranked bar chart of overall macro F1 across all 7 stats |
| `plot_classifier_cv_stability.py` | Per-fold F1 stability across K=5 folds |
| `export_classifier_leaderboard.py` | Best classifier per stat as CSV + LaTeX |
| `run_classifiers.py` | Launches all classifier training runs in parallel |
| `run_classifier_analysis.py` | Orchestrates the full analysis pipeline |

**Run:**
```bash
python predictor/analysis/classifier_analysis/run_classifiers.py         # trains all classifiers
python predictor/analysis/classifier_analysis/run_classifier_analysis.py # analyses results
```

---

### What the analysis answers

The combination of both pipelines allows the following research questions to be answered systematically:

| Research question | Tool |
|-------------------|------|
| Which model architecture performs best overall? | `plot_rankings.py` |
| Does adding LSTM sequences improve over static MLP? | `plot_rankings.py` + `plot_heatmaps.py` |
| Which feature groups contribute the most? | `plot_ablation.py` |
| Does adding OOF odds features help regression? | Variant comparison in heatmaps |
| Are LSTM models more or less stable than MLP? | `plot_cv_stability.py` |
| Which is the single best model for each target? | `export_leaderboard.py` |
| Which stat is easiest/hardest to classify? | `export_classifier_leaderboard.py` |

---

## 11. Prediction App — `predictor/app/`

The app is the user-facing culmination of the entire pipeline. It allows anyone to select two Champions League teams, compose their starting lineups, and receive a full statistical prediction for the hypothetical match — powered by the best trained models discovered during the research phase.

**Launch:**
```bash
streamlit run predictor/app/app.py
```

---

### UI Overview

The app is built with **Streamlit** and styled as a dark tactical interface titled *"Tactical Command | Match Predictor"*. The layout is divided into three vertical panels:

```
┌──────────────────┬────────────────────┬──────────────────┐
│   HOME TEAM      │    PITCH           │   AWAY TEAM      │
│                  │                    │                  │
│  Team selector   │  CSS-rendered      │  Team selector   │
│                  │  green pitch with  │                  │
│  GK picker       │  centre line,      │  GK picker       │
│  DF pickers ×6   │  circle, and       │  DF pickers ×6   │
│  MF pickers ×6   │  goal boxes        │  MF pickers ×6   │
│  ATK pickers ×4  │                    │  ATK pickers ×4  │
│                  │  [Predict] button  │                  │
└──────────────────┴────────────────────┴──────────────────┘
```

**Left and right panels:** Team colour-coded. Each player slot shows a jersey SVG icon with the player's jersey number printed on the shirt, paired with a dropdown populated from the current season's squad. Slots are organised by role: GK, then DF, MF, ATK.

**Centre panel:** A tactically styled pitch (green background, white centre line, centre circle, goal boxes rendered via CSS pseudo-elements). The pitch height is fixed to match the side panels regardless of how many player slots are active. The **Predict** button sits in the centre.

**Formation caps:** The number of active slots per role is capped by the formation detected from the team's most recent match (e.g. a 4-3-3 activates 4 DF, 3 MF, 3 ATK). This prevents nonsensical lineups (e.g. 6 attackers, 1 midfielder).

**Smart defaults:** When a team is selected, the dropdowns are pre-filled with the players from that team's last recorded starting lineup for the same role, in jersey-number order.

---

### Backend architecture — `predictor/app/backend/`

The backend is split into six focused modules that form a clean inference pipeline:

---

#### `data_layer.py` — Data access

Responsible for reading from the `data/` directory.

- `get_current_season_dir()` — scans `data/` for season folders, returns the most recent one that is marked as either current or finished
- `get_team_list()` — returns a sorted list of `(team_name, team_id)` tuples from the current season's `teams/` folder
- `get_team_roster(team_id)` — reads `squad.json` for the given team and returns a dict organised by role:
  ```python
  {
    "GK": [("Ederson", "123456"), ...],
    "DF": [("Rúben Dias", "789012"), ...],
    "MF": [...],
    "ATK": [...]
  }
  ```
- `get_last_lineup(team_id)` — scans the team's most recent finished fixture to extract the actual starting XI used, including formation and jersey numbers

---

#### `feature_builder.py` — Player stats and form

Assembles per-player statistics and rolling team form — the two main feature sources.

- `load_player_stats_bulk(player_ids)` — reads `last_year_statistics.json` for each player in the selection. Returns a dict mapping `player_id → {stat_name: value}`. Missing stats are filled with 0.
- `build_static_row(home_players, away_players, home_team_id, away_team_id, data_root)` — builds the player-slot feature block (GK/DF/MF/ATK × HOME/AWAY) by assigning players to fixed slots in role order, then fetches each team's rolling 5-match form from recent completed fixtures in `data/`

---

#### `raw_row_builder.py` — Full training-schema row

- `build_raw_row(home_players, away_players, ...)` — assembles a complete DataFrame row with all 230 columns matching the training CSV schema exactly: player slot features, NO_OF_* counts, form, cform, stage, and targets (targets are set to NaN since they are what we want to predict)
- `strip_non_features(df)` — removes meta columns and `*_player_id` columns, leaving only the 177 feature columns that `build_X()` expects

This strict schema matching ensures the inference row is byte-for-byte compatible with what the models were trained on.

---

#### `sequence_builder.py` — LSTM input sequences

- `build_team_sequence(team_id, data_root, K=5)` — scans all historical fixture data for the given team across all seasons, collects the K=5 most recent completed fixtures, and builds a `(5, 14)` float32 array using the same feature extraction logic as `build_sequence_table.py`
- Steps with fewer than K prior matches are zero-padded (with mask set to `False`)
- Called separately for home and away teams, producing the two sequence inputs for LSTM-based models

---

#### `model_registry.py` — Best model selection

Scans `artifacts/` to find the best saved run for each target:

- **Regression:** scans `artifacts/regression/` for all `run_result.json` files, groups by target, and selects the run with the lowest `cv_mae_mean`. Returns a dict mapping each target name to its best run directory, model type, and feature list.
- **Classification:** scans `artifacts/classification/` similarly, selecting the run with the highest `cv_f1_macro_mean` per stat.

This is computed once at app startup and cached in Streamlit session state, so the disk scan does not repeat on every interaction.

---

#### `inference.py` — End-to-end prediction

The orchestrator that ties all backend modules together:

```
1. build_raw_row()           → full 230-column DataFrame row
2. strip_non_features()      → 177-column feature row
3. For each of 7 stats:
     load classifier (scaler.pkl + model)
     align features to features_list.txt
     scaler.transform() → StandardScaler-normalised input
     model.predict_proba() → 3 softmax probabilities
4. Attach 21 odds columns to feature row
5. For each of 14 targets:
     load regressor (scaler.pkl + model + features_list.txt)
     select only the features this model was trained on
     scaler.transform()
     model.predict() → scalar prediction
     if Poisson target: prediction = exp(raw_output)  (log-rate → rate)
6. Return dict: { "HOME_GOALS": 1.4, "AWAY_CORNERS": 5.8, ... }
```

**Feature alignment:** Each model was trained on a specific ordered subset of features (recorded in `features_list.txt`). During inference, the full feature row is subset and reordered to match exactly what that model expects — preventing any column mismatch.

**Poisson decoding:** Models trained with Poisson NLL loss output the log of the predicted rate. The `exp()` is applied at inference time to recover the actual expected count.

---

### Prediction output

After clicking **Predict**, the app displays:

- **14 regression predictions** — one per (side, stat) pair, shown as colour-coded numeric cards
- **7 direction outcomes** — derived from comparing home vs. away predictions: `HOME WIN`, `DRAW`, or `AWAY WIN` for each stat, shown as a styled outcome bar with probability-like colouring based on the margin
- **Outcome probability bars** — visual bars styled by the predicted direction, giving an at-a-glance summary of which team is expected to dominate each statistical category

---

### App run command

```bash
streamlit run predictor/app/app.py --server.port 8502
```

---

## 12. End-to-End Pipeline

### Full training pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 1 — DATA COLLECTION                                              │
│                                                                         │
│  Sportmonks API v3                                                      │
│       │                                                                 │
│       ▼                                                                 │
│  data_scraping/seasons.py    →  data/<YYYY-YYYY_id>/  (folder skeleton) │
│  data_scraping/fixtures.py   →  data/.../fixtures/    (stats + lineups) │
│  data_scraping/teams.py      →  data/.../teams/       (squad rosters)   │
│  data_scraping/players.py    →  data/.../players/     (career stats)    │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 2 — COVERAGE ANALYSIS                                            │
│                                                                         │
│  data_vis_scripts/fixtures_heatmap.py   →  stat coverage per season     │
│  data_vis_scripts/players_heatmap.py    →  player stat coverage         │
│  data_vis_scripts/players.py            →  feature distributions        │
│  data_vis_scripts/find_common_max_rectangles.py                         │
│                           →  training window: 2017-2018 → 2025-2026     │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 3 — TABLE CONSTRUCTION                                           │
│                                                                         │
│  predictor/table_creation/build_table_v2.py                             │
│       →  train_tables/fixedslots_v2_*.csv   (1806 rows × 230 cols)      │
│                                                                         │
│  predictor/table_creation/build_sequence_table.py                       │
│       →  train_tables/seq_K5.npz            (1806 × 5 × 14 sequences)   │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 4 — CLASSIFIER TRAINING  [run before generate_odds_features]     │
│                                                                         │
│  predictor/classifiers/classifier_*.py  (5 architectures × N variants)  │
│       →  artifacts/classification/<model>/<variant>/<run>/              │
│             run_result.json  scaler.pkl  model.pt  features_list.txt    │
│                                                                         │
│  predictor/analysis/classifier_analysis/run_classifiers.py              │
│       (launches all combinations in parallel)                           │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 5 — OOF ODDS GENERATION                                          │
│                                                                         │
│  predictor/table_creation/generate_odds_features.py                     │
│       →  train_tables/odds_oof.npz   (1806 × 21 OOF probabilities)      │
│          (no leakage — K-fold held-out predictions only)                │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 6 — REGRESSION TRAINING                                          │
│                                                                         │
│  predictor/optimizers/optimizer_*.py  (5 architectures × N variants)    │
│       →  artifacts/regression/<model>/<variant>/<run>/                  │
│             run_result.json  scaler.pkl  model.pt  features_list.txt    │
│                                                                         │
│  predictor/analysis/optimizer_analysis/run_optimizers.py                │
│       (launches all combinations in parallel)                           │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 7 — ANALYSIS & COMPARISON                                        │
│                                                                         │
│  predictor/analysis/optimizer_analysis/run_optimizers_analysis.py       │
│       →  collect → heatmaps → rankings → stability → ablation →         │
│          leaderboard (CSV + LaTeX)                                      │
│                                                                         │
│  predictor/analysis/classifier_analysis/run_classifier_analysis.py      │
│       →  same pipeline for classifiers                                  │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  PHASE 8 — LIVE PREDICTION APP                                          │
│                                                                         │
│  streamlit run predictor/app/app.py                                     │
│                                                                         │
│  User selects teams + lineups                                           │
│       │                                                                 │
│       ▼                                                                 │
│  data_layer  →  feature_builder  →  raw_row_builder                     │
│  sequence_builder  →  model_registry (loads best run per target)        │
│       │                                                                 │
│       ▼                                                                 │
│  inference.py:                                                          │
│    7 classifiers  →  21 odds columns                                    │
│    14 regressors  →  14 numeric predictions                             │
│       │                                                                 │
│       ▼                                                                 │
│  App displays:                                                          │
│    Goals, Corners, Yellow Cards, Shots on Target,                       │
│    Fouls, Offsides, Red Cards  ×  Home + Away                           │
│    + 7 direction outcomes (HOME WIN / DRAW / AWAY WIN)                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Data flow at inference time

When a user submits a prediction in the app, no training data is re-read. Instead:

```
squad.json (current season)
    │
    ▼
Player selection (UI dropdowns)
    │
    ▼
last_year_statistics.json  ←── feature_builder.py
recent fixtures (form)     ←── feature_builder.py
seq_K5 equivalent           ←── sequence_builder.py
    │
    ▼
raw_row_builder.py  →  230-col DataFrame row  →  strip to 177 features
    │
    ▼  (+ odds columns from classifiers)
inference.py  →  14 regression predictions
    │
    ▼
Displayed in app UI
```

The app is entirely self-contained: once the models are trained and artefacts saved, it runs offline with no API calls required.

---

## 13. Setup & Usage

### Prerequisites

- **Conda** (Anaconda or Miniconda)
- A **Sportmonks API token** (required only if re-scraping data)
- A CUDA-capable GPU is optional but strongly recommended for training PyTorch models

---

### 1. Clone the repository

```bash
git clone <repo-url>
cd licenta
```

---

### 2. Create the conda environment

```bash
conda create -n licenta python=3.10
conda activate licenta
```

Install dependencies:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # GPU
# or: pip install torch  (CPU only)

pip install xgboost optuna scikit-learn pandas numpy matplotlib seaborn streamlit python-dotenv tqdm
```

---

### 3. Configure the API key (scraping only)

Create a `.env` file in the project root:

```
SPORTMONKS_API_TOKEN=your_token_here
SPORTMONKS_BASE_URL=https://api.sportmonks.com/v3/football
UCL_LEAGUE_ID=2
```

This file is listed in `.gitignore` and will not be committed.

---

### 4. Run the data scraping pipeline

> Skip this step if the `data/` folder is already populated.

```bash
python data_scraping/seasons.py
python data_scraping/fixtures.py
python data_scraping/teams.py
python data_scraping/players.py
```

Each script is safe to re-run — it skips already-fetched files.

---

### 5. Explore data coverage (optional)

```bash
python data_vis_scripts/fixtures_heatmap.py
python data_vis_scripts/players_heatmap.py
python data_vis_scripts/players.py
python data_vis_scripts/find_common_max_rectangles.py
```

Plots and CSVs are saved to `data_visualisation/`.

---

### 6. Build the training tables

```bash
# Main feature table
python predictor/table_creation/build_table_v2.py

# LSTM sequence tensors
python predictor/table_creation/build_sequence_table.py
```

Outputs go to `train_tables/`.

---

### 7. Train classifiers

```bash
# Train all classifier architectures across all variants (parallel)
python predictor/analysis/classifier_analysis/run_classifiers.py

# Or train a specific classifier manually
python predictor/classifiers/classifier_mlp_torch.py --variant form --repeats 3
```

---

### 8. Generate OOF odds features

> Must run after classifiers are trained.

```bash
python predictor/table_creation/generate_odds_features.py
```

Output: `train_tables/odds_oof.npz`

---

### 9. Train regression models

```bash
# Train all optimizer architectures across all variants (parallel)
python predictor/analysis/optimizer_analysis/run_optimizers.py

# Or train a specific optimizer manually
python predictor/optimizers/optimizer_mlp_torch.py --variant sum --repeats 3
python predictor/optimizers/optimizer_xgb.py --variant form_mean_stage --repeats 3
```

---

### 10. Run analysis

```bash
# Regression analysis (collect → heatmaps → rankings → stability → ablation → leaderboard)
python predictor/analysis/optimizer_analysis/run_optimizers_analysis.py

# Classifier analysis
python predictor/analysis/classifier_analysis/run_classifier_analysis.py
```

---

### 11. Launch the prediction app

```bash
conda activate licenta
streamlit run predictor/app/app.py --server.port 8502
```

Then open `http://localhost:8502` in your browser.

The app automatically picks the best trained model per target from `artifacts/`. No configuration needed.

---

### Quick-start: app only (models already trained)

If the `artifacts/` folder is already populated with trained models:

```bash
conda activate licenta
streamlit run predictor/app/app.py
```

That's all that's needed to run the predictor.

---

### Directory permissions note

The `train_tables/` and `artifacts/` directories are created automatically by the scripts. The `data/` directory must be writable for the scraping scripts to populate it.

---

### `.gitignore` notes

The following are excluded from version control by default:
- `data/` — raw scraped data (too large, re-creatable from API)
- `artifacts/` — trained model artefacts (re-creatable from training)
- `train_tables/` — generated tables (re-creatable from data)
- `.env` — API credentials
- `__pycache__/`, `*.pyc` — Python bytecode
- `data_visualisation/` — generated plots

