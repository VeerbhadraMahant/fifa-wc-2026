from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.preprocessing.feature_engineering import (
    build_match_features,
    build_shootout_features,
    build_team_features,
    infer_fixture_groups,
)


ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"


def _read_processed(name: str) -> pd.DataFrame:
    path = PROCESSED_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing processed data file: {path}")
    return pd.read_csv(path)


def write_outputs() -> dict[str, int]:
    results = _read_processed("results_clean.csv")
    goalscorers = _read_processed("goalscorers_clean.csv")
    shootouts = _read_processed("shootouts_clean.csv")
    fixtures = _read_processed("world_cup_2026_fixtures.csv")

    match_features = build_match_features(results)
    team_features = build_team_features(goalscorers)
    shootout_features = build_shootout_features(shootouts)
    fixtures_with_groups = infer_fixture_groups(fixtures)

    if (match_features["date"] >= "2026-06-11").any():
        raise ValueError("Match features contain rows on or after 2026-06-11")
    if not match_features["final_training_weight"].gt(0).all():
        raise ValueError("Every training match must have a positive final_training_weight")
    if len(fixtures_with_groups) != 72:
        raise ValueError(f"Expected 72 fixtures, found {len(fixtures_with_groups)}")

    match_features.to_csv(PROCESSED_DIR / "match_features.csv", index=False)
    team_features.to_csv(PROCESSED_DIR / "team_features.csv", index=False)
    shootout_features.to_csv(PROCESSED_DIR / "shootout_features.csv", index=False)
    fixtures_with_groups.to_csv(
        PROCESSED_DIR / "world_cup_2026_fixtures.csv",
        index=False,
    )

    return {
        "match_feature_rows": len(match_features),
        "team_feature_rows": len(team_features),
        "shootout_feature_rows": len(shootout_features),
        "fixture_rows": len(fixtures_with_groups),
    }


def main() -> None:
    summary = write_outputs()
    print("Phase 2 feature engineering complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
