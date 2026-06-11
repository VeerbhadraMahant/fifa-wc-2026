from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ml.models.elo import EloModel
from ml.models.poisson_model import PoissonGoalModel


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "ml" / "outputs"
ELO_WEIGHT = 0.40
POISSON_WEIGHT = 0.60


def _read_processed(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data file: {path}")
    return pd.read_csv(path)


def _round_probability(value: float) -> float:
    return round(float(value), 6)


def _most_likely_scoreline(matrix: list[list[float]]) -> dict[str, float | int]:
    best_home = 0
    best_away = 0
    best_probability = -1.0
    for home_goals, row in enumerate(matrix):
        for away_goals, probability in enumerate(row):
            if probability > best_probability:
                best_home = home_goals
                best_away = away_goals
                best_probability = probability

    return {
        "home_goals": best_home,
        "away_goals": best_away,
        "probability": _round_probability(best_probability),
    }


def _blend_probabilities(
    elo_probabilities: dict[str, float],
    poisson_probabilities: dict[str, float],
) -> dict[str, float]:
    blended = {
        key: (ELO_WEIGHT * elo_probabilities[key]) + (POISSON_WEIGHT * poisson_probabilities[key])
        for key in ("home_win", "draw", "away_win")
    }
    total = sum(blended.values())
    return {key: value / total for key, value in blended.items()}


def build_predictions() -> dict[str, object]:
    match_features = _read_processed("match_features.csv")
    team_features = _read_processed("team_features.csv")
    fixtures = _read_processed("world_cup_2026_fixtures.csv")

    elo_model = EloModel.fit(match_features)
    poisson_model = PoissonGoalModel.fit(match_features, team_features)

    predictions = []
    for fixture in fixtures.itertuples(index=False):
        home_team = fixture.home_team
        away_team = fixture.away_team
        expected_home_goals, expected_away_goals = poisson_model.expected_goals(
            home_team,
            away_team,
        )
        scoreline_matrix = poisson_model.scoreline_matrix(home_team, away_team)
        poisson_probabilities = poisson_model.outcome_probabilities(home_team, away_team)
        elo_probabilities = elo_model.outcome_probabilities(home_team, away_team)
        blended_probabilities = _blend_probabilities(elo_probabilities, poisson_probabilities)

        prediction = {
            "match_id": fixture.fixture_id,
            "fixture_number": int(fixture.fixture_number),
            "group": fixture.group,
            "date": fixture.date,
            "city": fixture.city,
            "country": fixture.country,
            "home_team": home_team,
            "away_team": away_team,
            "home_elo": round(elo_model.rating(home_team), 2),
            "away_elo": round(elo_model.rating(away_team), 2),
            "expected_home_goals": round(expected_home_goals, 3),
            "expected_away_goals": round(expected_away_goals, 3),
            "home_win_probability": _round_probability(blended_probabilities["home_win"]),
            "draw_probability": _round_probability(blended_probabilities["draw"]),
            "away_win_probability": _round_probability(blended_probabilities["away_win"]),
            "elo_probabilities": {
                key: _round_probability(value)
                for key, value in elo_probabilities.items()
            },
            "poisson_probabilities": {
                key: _round_probability(value)
                for key, value in poisson_probabilities.items()
            },
            "most_likely_scoreline": _most_likely_scoreline(scoreline_matrix),
            "scoreline_matrix": [
                [_round_probability(cell) for cell in row]
                for row in scoreline_matrix
            ],
        }
        predictions.append(prediction)

    return {
        "metadata": {
            "model_version": "phase3_baseline_v1",
            "training_cutoff": "2026-06-11",
            "match_count": len(predictions),
            "elo_weight": ELO_WEIGHT,
            "poisson_weight": POISSON_WEIGHT,
            "scoreline_max_goals": 9,
        },
        "matches": predictions,
    }


def validate_predictions(payload: dict[str, object]) -> None:
    matches = payload["matches"]
    if len(matches) != 72:
        raise ValueError(f"Expected 72 match predictions, found {len(matches)}")

    for match in matches:
        probabilities = [
            match["home_win_probability"],
            match["draw_probability"],
            match["away_win_probability"],
        ]
        if any(probability < 0 or probability > 1 for probability in probabilities):
            raise ValueError(f"Invalid probability in {match['match_id']}")

        probability_sum = sum(probabilities)
        if abs(probability_sum - 1.0) > 0.00001:
            raise ValueError(
                f"Probabilities for {match['match_id']} sum to {probability_sum}"
            )

        matrix_sum = sum(sum(row) for row in match["scoreline_matrix"])
        if abs(matrix_sum - 1.0) > 0.0001:
            raise ValueError(f"Scoreline matrix for {match['match_id']} sums to {matrix_sum}")


def write_outputs() -> dict[str, int]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_predictions()
    validate_predictions(payload)

    output_path = OUTPUT_DIR / "match_predictions.json"
    output_path.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    return {
        "match_predictions": len(payload["matches"]),
    }


def main() -> None:
    summary = write_outputs()
    print("Phase 3 match prediction complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
