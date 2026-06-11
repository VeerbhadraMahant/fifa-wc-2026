# FIFA World Cup 2026 ML Prediction Platform

A probabilistic prediction platform for the 2026 FIFA World Cup. The project combines
historical international match data, team-strength models, and tournament simulation to
estimate match outcomes, group qualification odds, knockout paths, and tournament winner
probabilities.

The planned modelling stack is:

```text
Elo ratings -> Bivariate Poisson expected-goals model -> Monte Carlo tournament simulation
```

The Python pipeline produces static JSON outputs that are consumed by a Next.js frontend.
This keeps the web app lightweight while allowing the ML pipeline to run offline or through
scheduled GitHub Actions workflows.

## Project Goals

- Predict all 104 matches in the expanded 2026 World Cup format.
- Support pre-tournament predictions from historical data.
- Support in-tournament updates after real match results are available.
- Generate transparent probabilities for matches, groups, bracket slots, and teams.
- Build a portfolio-quality ML system with validation, automation, and a polished web UI.

## Tournament Format

The 2026 World Cup has 48 teams and 104 matches:

| Stage | Matches |
| --- | ---: |
| Group stage | 72 |
| Round of 32 | 16 |
| Round of 16 | 8 |
| Quarter-finals | 4 |
| Semi-finals | 2 |
| Third-place play-off | 1 |
| Final | 1 |

Group advancement follows the 2026 format: the top two teams from each of 12 groups plus
the eight best third-place teams advance to the Round of 32.

## Repository Structure

```text
fifa-wc-2026/
|-- data/
|   |-- raw/                 # Source CSV datasets
|   `-- processed/           # Cleaned datasets and feature tables
|-- ml/
|   |-- preprocessing/       # Name normalization and feature engineering
|   |-- models/              # Elo, Poisson, and shootout models
|   |-- simulation/          # Group, knockout, and Monte Carlo simulation
|   |-- pipeline/            # Pre-tournament and live-update entrypoints
|   `-- outputs/             # JSON consumed by the web app
|-- notebooks/               # EDA and validation notebooks
|-- web/                     # Next.js frontend
|-- .github/workflows/       # Automation workflows
|-- implementation_plan.md   # Full build plan
|-- requirements.txt         # Python dependencies
`-- README.md
```

## Data Sources

The pipeline expects these CSV files under `data/raw/`:

| File | Purpose |
| --- | --- |
| `results.csv` | Historical match results for model training |
| `former_names.csv` | Historical-to-current team name mappings |
| `goalscores.csv` | Goal-level data for penalty and attacking features |
| `shootout.csv` | Historical penalty shootout records |

Large raw datasets are intentionally ignored by Git. Keep source data in `data/raw/`
locally, then write cleaned outputs to `data/processed/`.

## ML Pipeline

The pipeline is planned in these phases:

1. Data cleaning and team name normalization.
2. Elo rating training with tournament weights and margin-of-victory adjustment.
3. Bivariate Poisson expected-goals modelling.
4. Penalty shootout probability modelling with Bayesian shrinkage.
5. Monte Carlo tournament simulation.
6. Static JSON export for the frontend.

Primary outputs will be written to `ml/outputs/`:

- `groups.json`
- `bracket.json`
- `team_probabilities.json`
- `match_predictions.json`
- `metadata.json`

## Local Setup

Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

When the Next.js app is scaffolded under `web/`, install frontend dependencies there:

```powershell
cd web
npm install
npm run dev
```

## Development Notes

- See `implementation_plan.md` for the full architecture and phased roadmap.
- Treat `data/raw/` as local input data, not source-controlled application code.
- Keep generated model artifacts and simulation outputs reproducible through scripts.
- Use notebooks for exploration and validation, then move stable logic into `ml/`.

## Validation Targets

The target validation benchmarks are:

| Metric | Target |
| --- | ---: |
| 2022 World Cup group advancement accuracy | > 65% |
| 2022 World Cup match W/D/L Brier score | < 0.24 |
| Poisson MAE on goals | < 0.9 |
| 100k Monte Carlo runtime | < 5 minutes |
