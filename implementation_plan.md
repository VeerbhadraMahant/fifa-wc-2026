# FIFA World Cup 2026 ML Prediction Platform
## V1 Implementation Plan

---

## 1. Project Overview

This project is a pre-tournament prediction platform for the 2026 FIFA World Cup.

The goal of v1 is to train a model on historical international football data, give more weight to recent matches from 2022-2026, predict the 72 fixed group-stage fixtures, simulate the 32 knockout matches, and estimate the most likely tournament winner before the World Cup begins.

The v1 modelling stack is:

```text
Historical results + team features
-> weighted match prediction model
-> 10,000 Monte Carlo tournament simulations
-> match, group, bracket, and winner probabilities
```

For this project, "10,000 epochs" means 10,000 full tournament simulations, not neural-network training epochs.

---

## 2. V1 Scope

### In Scope

- Train only on completed historical matches before `2026-06-11`.
- Use the 2026 World Cup fixture rows in `results.csv` as prediction targets, not training data.
- Predict all 72 fixed group-stage matches.
- Simulate all 32 knockout-stage matches.
- Run 10,000 full tournament simulations.
- Predict:
  - match outcome probabilities
  - expected scores
  - group standings and qualification odds
  - likely knockout bracket paths
  - team stage-reach probabilities
  - tournament winner probability
  - most likely champion.
- Produce static JSON outputs for a web app.
- Build the web app as a consumer of generated JSON files.

### Deferred From V1

- Live football APIs.
- GitHub Actions live refresh.
- In-tournament retraining.
- Automated result ingestion after real matches.
- Vercel automation beyond reading static output files.

---

## 3. Tournament Structure

The 2026 FIFA World Cup has 48 teams and 104 matches.

| Stage | Matches |
| --- | ---: |
| Group Stage | 72 |
| Round of 32 | 16 |
| Round of 16 | 8 |
| Quarter-finals | 4 |
| Semi-finals | 2 |
| Third-place play-off | 1 |
| Final | 1 |
| **Total** | **104** |

Group-stage fixtures are fixed before the tournament. Knockout fixtures are not fixed because they depend on group-stage results, so v1 will represent knockout matches as simulated bracket slots.

Group advancement rules:

- Top two teams from each of the 12 groups advance.
- The eight best third-place teams also advance.
- Total teams entering the Round of 32: 32.

---

## 4. Raw Datasets

The project uses the raw files currently stored in `data/raw/`.

### `results.csv`

Primary match-level dataset.

Expected columns:

- `date`
- `home_team`
- `away_team`
- `home_score`
- `away_score`
- `tournament`
- `city`
- `country`
- `neutral`

Used for:

- identifying completed historical training matches
- identifying the 72 fixed 2026 World Cup group fixtures
- training Elo ratings
- fitting expected-goals and outcome models
- deriving form, result, and tournament-weight features.

Important rule:

- Rows dated `2026-06-11` or later must not be used for model training.
- 2026 World Cup rows with `NA` scores are prediction targets.

### `former_names.csv`

Team name normalization dataset.

Expected columns:

- `current`
- `former`
- `start_date`
- `end_date`

Used for:

- mapping historical team names to current canonical names
- keeping team identities consistent across all datasets.

### `goalscorers.csv`

Goal-level dataset.

Expected columns:

- `date`
- `home_team`
- `away_team`
- `team`
- `scorer`
- `minute`
- `own_goal`
- `penalty`

Used for team-level attacking and penalty features:

- penalty conversion rate
- distinct scorer count
- late goal rate
- own goal rate.

### `shootouts.csv`

Penalty shootout dataset.

Expected columns:

- `date`
- `home_team`
- `away_team`
- `winner`
- `first_shooter`

Used for:

- historical shootout win rate
- total shootouts per team
- Bayesian-smoothed shootout probability
- knockout tiebreaker simulation.

---

## 5. Repository Structure

```text
fifa-wc-2026/
|-- data/
|   |-- raw/
|   |   |-- results.csv
|   |   |-- former_names.csv
|   |   |-- goalscorers.csv
|   |   `-- shootouts.csv
|   `-- processed/
|-- ml/
|   |-- preprocessing/
|   |-- models/
|   |-- simulation/
|   |-- pipeline/
|   `-- outputs/
|-- notebooks/
|-- web/
|-- requirements.txt
|-- README.md
`-- implementation_plan.md
```

---

## 6. Phase 1: Data Understanding And Cleaning

### Goal

Create clean, leakage-safe datasets for training and prediction.

### Tasks

- Load all raw CSV files from `data/raw/`.
- Parse all date columns as dates.
- Normalize team names using `former_names.csv`.
- Split `results.csv` into:
  - training matches: completed matches before `2026-06-11`
  - fixture targets: 2026 World Cup matches dated `2026-06-11` onward.
- Drop training rows with:
  - missing dates
  - missing team names
  - missing scores
  - non-numeric scores
  - `NA` scores
  - dates on or after `2026-06-11`.
- Confirm exactly 72 fixed group-stage fixtures are detected.
- Store cleaned outputs in `data/processed/`.

### Deliverables

- `data/processed/results_clean.csv`
- `data/processed/world_cup_2026_fixtures.csv`
- `data/processed/goalscorers_clean.csv`
- `data/processed/shootouts_clean.csv`
- `data/processed/team_name_map.csv`

### Acceptance Checks

- No training row has `date >= 2026-06-11`.
- No training row has missing or invalid scores.
- Exactly 72 group-stage fixtures are available for prediction.

---

## 7. Phase 2: Feature Engineering And Weighting

### Goal

Build match-level and team-level features, with extra emphasis on recent international form.

### Match-Level Features

From `results_clean.csv`, create:

- `home_goals`
- `away_goals`
- `goal_difference`
- `result_label`
- `is_draw`
- `is_home_win`
- `is_away_win`
- `is_neutral`
- `year`
- `days_before_cutoff`
- `tournament_category`
- `recency_weight`
- `tournament_weight`
- `final_training_weight`.

### Recency Weighting

Use this default weighting policy:

| Match Date | Recency Weight |
| --- | ---: |
| `2022-01-01` to `2026-06-10` | 1.00 |
| `2018-01-01` to `2021-12-31` | 0.70 |
| `2010-01-01` to `2017-12-31` | 0.40 |
| before `2010-01-01` | 0.15 |

### Tournament Weighting

Use this default weighting policy:

| Tournament Type | Tournament Weight |
| --- | ---: |
| FIFA World Cup | 1.00 |
| World Cup qualification | 0.85 |
| Continental championship | 0.80 |
| Continental qualification | 0.70 |
| Nations League or similar competitive tournament | 0.60 |
| Friendly | 0.30 |
| Other | 0.50 |

Final match weight:

```text
final_training_weight = recency_weight * tournament_weight
```

### Team-Level Features

From `goalscorers_clean.csv`, compute per team:

- penalty conversion rate
- total penalties scored
- total penalties attempted
- distinct scorer count
- late goal rate for goals after the 80th minute
- own goal rate.

From `shootouts_clean.csv`, compute per team:

- shootout wins
- total shootouts
- raw shootout win rate
- Bayesian-smoothed shootout win probability.

Default Bayesian smoothing:

```text
smoothed_rate = (wins + 2) / (total_shootouts + 4)
```

### Deliverables

- `data/processed/match_features.csv`
- `data/processed/team_features.csv`
- `data/processed/shootout_features.csv`

### Acceptance Checks

- Every training match has a positive `final_training_weight`.
- 2022-2026 matches receive higher default weight than older matches.
- Missing team-level features are filled with sensible global defaults.

---

## 8. Phase 3: Match Prediction Model

### Goal

Produce pre-match probabilities and scoreline estimates for any matchup.

### Model Components

#### Elo Rating Model

- Initialize teams at 1500 Elo.
- Iterate through training matches chronologically.
- Apply match update strength using `final_training_weight`.
- Use margin-of-victory adjustment.
- Apply home advantage only for non-neutral historical matches.
- Treat 2026 World Cup fixtures as neutral prediction matches.

Output:

- current Elo rating per team as of `2026-06-10`.
- Elo-based win/draw/loss probabilities.

#### Expected-Goals Model

- Train a Poisson-style expected-goals model using weighted historical results.
- Estimate team attack and defense strength.
- Include neutral-site handling.
- Regularize team parameters to avoid extreme values for teams with sparse data.

Output:

- expected goals for each team in a matchup.
- scoreline probability matrix.
- Poisson-based win/draw/loss probabilities.

#### Blended Match Prediction

Blend Elo and expected-goals probabilities.

Default blend:

```text
final_probability = 0.40 * elo_probability + 0.60 * poisson_probability
```

### Group Fixture Predictions

For each of the 72 fixed group-stage fixtures, output:

- match id
- date
- group
- home team
- away team
- home win probability
- draw probability
- away win probability
- expected home goals
- expected away goals
- most likely scoreline
- scoreline probability matrix.

### Deliverables

- `ml/models/elo.py`
- `ml/models/poisson_model.py`
- `ml/pipeline/train_models.py`
- `ml/outputs/match_predictions.json`

### Acceptance Checks

- Each match's win/draw/loss probabilities sum to approximately 1.
- Every probability is between 0 and 1.
- All 72 group-stage fixtures receive predictions.

---

## 9. Phase 4: Group-Stage Simulation

### Goal

Simulate the 72 fixed group-stage matches and determine group standings.

### Tasks

- Load group-stage fixture predictions.
- Sample scorelines from each match's scoreline probability matrix.
- Track per-team group stats:
  - matches played
  - wins
  - draws
  - losses
  - points
  - goals for
  - goals against
  - goal difference.
- Rank each group using FIFA-style tiebreakers:
  1. points
  2. goal difference
  3. goals scored
  4. head-to-head points among tied teams
  5. head-to-head goal difference
  6. head-to-head goals scored
  7. random draw fallback.
- Advance:
  - top two teams from each group
  - eight best third-place teams.

### Deliverables

- `ml/simulation/group_stage.py`
- group standings data inside `ml/outputs/groups.json`.

### Acceptance Checks

- Every simulated group has exactly four teams.
- Exactly 24 teams advance as top-two finishers.
- Exactly 8 third-place teams advance.
- Exactly 32 teams enter the knockout stage.

---

## 10. Phase 5: Knockout-Stage Simulation

### Goal

Simulate the 32 knockout matches after group-stage outcomes are known.

### Tasks

- Build Round of 32 bracket slots from simulated group rankings.
- Predict matchup probabilities for each possible knockout matchup.
- Simulate each knockout match:
  - sample regular-time scoreline
  - if one team wins, advance that team
  - if the match is drawn, use the shootout model to select a winner.
- Simulate:
  - Round of 32: 16 matches
  - Round of 16: 8 matches
  - Quarter-finals: 4 matches
  - Semi-finals: 2 matches
  - Third-place play-off: 1 match
  - Final: 1 match.

### Knockout Outputs

Because knockout teams are not known before the tournament, output two views:

1. Probabilistic bracket slots:
   - most likely teams to occupy each slot
   - probability each team reaches each stage
   - probability each team wins each stage.
2. Most-likely projected bracket:
   - one readable bracket using the most likely qualifiers and winners.

### Deliverables

- `ml/simulation/knockout_stage.py`
- bracket data inside `ml/outputs/bracket.json`.

### Acceptance Checks

- Exactly 32 knockout matches are simulated per tournament run.
- Every knockout match has exactly one winner.
- The final produces exactly one champion.

---

## 11. Phase 6: 10,000-Run Monte Carlo Tournament Runner

### Goal

Run 10,000 complete tournament simulations and aggregate probabilities.

### Tasks

- Run the full tournament simulation 10,000 times.
- Each simulation must include:
  - 72 group-stage matches
  - 32 knockout-stage matches
  - 104 total matches.
- Aggregate team probabilities for:
  - finishing 1st, 2nd, 3rd, or 4th in group
  - advancing from group
  - reaching Round of 32
  - reaching Round of 16
  - reaching Quarter-finals
  - reaching Semi-finals
  - reaching Final
  - winning the tournament.
- Aggregate match and bracket probabilities:
  - most likely final matchups
  - most likely champion
  - most likely bracket path
  - top contenders by tournament win probability.

### Deliverables

- `ml/simulation/monte_carlo.py`
- `ml/pipeline/run_pre_tournament.py`
- `ml/outputs/team_probabilities.json`
- `ml/outputs/metadata.json`.

### Acceptance Checks

- Exactly 10,000 simulations run.
- Every simulation contains exactly 104 matches.
- Winner probabilities sum to approximately 1.
- All stage probabilities are between 0 and 1.
- Stage probabilities are monotonic, for example:
  - `win_tournament <= reach_final <= reach_semi <= reach_quarter`.

---

## 12. Phase 7: JSON Outputs And Reporting

### Goal

Create stable outputs that can be inspected directly and consumed by the web app.

### JSON Outputs

Write these files to `ml/outputs/`:

#### `match_predictions.json`

Contains predictions for the 72 fixed group-stage fixtures and summary projections for simulated knockout slots.

#### `groups.json`

Contains:

- group teams
- group fixtures
- predicted group tables
- qualification odds
- best third-place advancement odds.

#### `bracket.json`

Contains:

- probabilistic bracket slots
- most likely projected bracket
- stage-by-stage team probabilities.

#### `team_probabilities.json`

Contains one record per team:

- Elo rating
- attack strength
- defense strength
- group advancement probability
- stage-reach probabilities
- tournament win probability.

#### `metadata.json`

Contains:

- generated timestamp
- training cutoff date
- simulation count
- model version
- data files used
- weighting policy summary.

### Notebook / Report

Create a validation and explanation notebook that summarizes:

- data cutoff
- training data size
- fixture count
- weighting strategy
- model design
- top predicted teams
- predicted champion
- basic validation against prior World Cup results where possible.

### Acceptance Checks

- All JSON files are valid JSON.
- Required top-level fields exist.
- Web app can load outputs without running ML code.

---

## 13. Phase 8: Web App Consuming Static Outputs

### Goal

Build a frontend that presents the prediction results clearly.

### Data Strategy

- The web app reads static JSON generated by the Python pipeline.
- The web app does not train models.
- The web app does not run tournament simulations.
- ML outputs can be copied or exposed under `web/public/data/`.

### Pages

#### Homepage

Shows:

- predicted winner
- top tournament contenders
- simulation count
- training cutoff date
- last generated timestamp.

#### Groups Page

Shows:

- all 12 groups
- predicted group standings
- group-stage match probabilities
- qualification odds.

#### Bracket Page

Shows:

- probabilistic bracket slots
- most likely projected bracket
- team probability to reach each stage.

#### Matches Page

Shows:

- 72 fixed group-stage match predictions
- simulated knockout-stage slot projections
- expected scores
- win/draw/loss probabilities.

#### Teams Page

Shows:

- team-level tournament odds
- Elo rating
- attack and defense strengths
- group and knockout advancement probabilities.

### Acceptance Checks

- The app loads from static JSON only.
- The homepage clearly displays the predicted champion.
- Group and bracket pages render without missing data.
- The app can be rebuilt without rerunning the ML pipeline.

---

## 14. Test Plan

### Data Tests

- Verify no training data includes rows dated `2026-06-11` or later.
- Verify training data contains only completed matches.
- Verify all required raw files exist.
- Verify the pipeline detects exactly 72 fixed group-stage fixtures.

### Model Tests

- Verify match probabilities are between 0 and 1.
- Verify each match's win/draw/loss probabilities sum to approximately 1.
- Verify scoreline matrices sum to approximately 1.
- Verify missing team-level features fall back to defaults.

### Simulation Tests

- Verify each tournament simulation contains exactly 104 matches.
- Verify exactly 32 teams enter the knockout stage.
- Verify every knockout match has one winner.
- Verify each simulation produces one champion.

### Output Tests

- Verify all generated JSON files are valid.
- Verify winner probabilities sum to approximately 1.
- Verify stage probabilities are monotonic.
- Verify output schemas match what the web app expects.

### Validation

- Back-test the modelling approach against the 2022 World Cup where possible.
- Track:
  - match result accuracy
  - Brier score
  - goal prediction error
  - group advancement accuracy.

---

## 15. V1 Success Criteria

V1 is complete when:

- The pipeline trains only on completed matches before `2026-06-11`.
- The pipeline predicts all 72 fixed group-stage fixtures.
- The simulator runs 10,000 full tournament simulations.
- Each simulation includes exactly 104 matches.
- The system outputs a most likely World Cup winner.
- The system outputs team probabilities for every major tournament stage.
- Static JSON outputs are generated successfully.
- The web app displays the predicted winner, group predictions, bracket projections, and team odds from static JSON.

---

## 16. Assumptions

- V1 is pre-tournament only.
- "10,000 epochs" means 10,000 Monte Carlo tournament simulations.
- The 2026 World Cup begins on `2026-06-11`, so this is the training cutoff.
- Rows in `results.csv` dated `2026-06-11` or later are prediction targets, not training examples.
- `results.csv` contains the 72 fixed group-stage fixtures.
- Knockout fixtures must be simulated because their teams are unknown before the tournament.
- Live updates and in-tournament retraining may be added later, but are not part of v1.
