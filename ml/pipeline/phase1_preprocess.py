from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.preprocessing.name_normalizer import load_name_map, normalize_team_columns


CUTOFF_DATE = pd.Timestamp("2026-06-11")
ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"


def _read_csv(name: str) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data file: {path}")
    return pd.read_csv(path, dtype=str)


def _parse_date_column(frame: pd.DataFrame, source_name: str) -> pd.DataFrame:
    parsed = frame.copy()
    if "date" not in parsed.columns:
        raise ValueError(f"{source_name} is missing required column: date")
    parsed["date"] = pd.to_datetime(parsed["date"], errors="coerce")
    return parsed


def _clean_boolean_column(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    cleaned = frame.copy()
    if column not in cleaned.columns:
        return cleaned

    bool_map = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
        "yes": True,
        "no": False,
    }
    cleaned[column] = (
        cleaned[column]
        .astype("string")
        .str.strip()
        .str.lower()
        .map(bool_map)
        .astype("boolean")
    )
    return cleaned


def clean_results(results: pd.DataFrame, name_map: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    results = _parse_date_column(results, "results.csv")
    results = normalize_team_columns(results, name_map)
    results = _clean_boolean_column(results, "neutral")

    for column in ("home_score", "away_score"):
        results[column] = pd.to_numeric(results[column], errors="coerce")

    required_text_columns = ["home_team", "away_team", "tournament", "city", "country"]
    for column in required_text_columns:
        if column in results.columns:
            results[column] = results[column].astype("string").str.strip()

    fixture_mask = (
        (results["date"] >= CUTOFF_DATE)
        & (results["tournament"] == "FIFA World Cup")
    )
    fixtures = results.loc[fixture_mask].copy()
    fixtures = fixtures.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)
    fixtures.insert(0, "fixture_id", [f"GS-{index + 1:02d}" for index in range(len(fixtures))])

    training_mask = (
        (results["date"] < CUTOFF_DATE)
        & results["home_score"].notna()
        & results["away_score"].notna()
        & results["home_team"].notna()
        & results["away_team"].notna()
    )
    training = results.loc[training_mask].copy()
    training["home_score"] = training["home_score"].astype(int)
    training["away_score"] = training["away_score"].astype(int)
    training = training.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)

    return training, fixtures


def clean_goalscorers(goalscorers: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    goalscorers = _parse_date_column(goalscorers, "goalscorers.csv")
    goalscorers = normalize_team_columns(goalscorers, name_map)
    goalscorers = _clean_boolean_column(goalscorers, "own_goal")
    goalscorers = _clean_boolean_column(goalscorers, "penalty")

    goalscorers["minute"] = pd.to_numeric(goalscorers["minute"], errors="coerce")
    required = ["date", "home_team", "away_team", "team"]
    cleaned = goalscorers.dropna(subset=required).copy()
    return cleaned.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def clean_shootouts(shootouts: pd.DataFrame, name_map: dict[str, str]) -> pd.DataFrame:
    shootouts = _parse_date_column(shootouts, "shootouts.csv")
    shootouts = normalize_team_columns(shootouts, name_map)
    required = ["date", "home_team", "away_team", "winner"]
    cleaned = shootouts.dropna(subset=required).copy()
    return cleaned.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def write_outputs() -> dict[str, int]:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    name_map = load_name_map(RAW_DIR / "former_names.csv")
    results = _read_csv("results.csv")
    goalscorers = _read_csv("goalscorers.csv")
    shootouts = _read_csv("shootouts.csv")

    results_clean, fixtures = clean_results(results, name_map)
    goalscorers_clean = clean_goalscorers(goalscorers, name_map)
    shootouts_clean = clean_shootouts(shootouts, name_map)
    team_name_map = pd.DataFrame(
        sorted(name_map.items()),
        columns=["former", "current"],
    )

    if len(fixtures) != 72:
        raise ValueError(f"Expected 72 2026 World Cup group fixtures, found {len(fixtures)}")
    if (results_clean["date"] >= CUTOFF_DATE).any():
        raise ValueError("Training data contains rows on or after 2026-06-11")

    results_clean.to_csv(PROCESSED_DIR / "results_clean.csv", index=False)
    fixtures.to_csv(PROCESSED_DIR / "world_cup_2026_fixtures.csv", index=False)
    goalscorers_clean.to_csv(PROCESSED_DIR / "goalscorers_clean.csv", index=False)
    shootouts_clean.to_csv(PROCESSED_DIR / "shootouts_clean.csv", index=False)
    team_name_map.to_csv(PROCESSED_DIR / "team_name_map.csv", index=False)

    return {
        "training_matches": len(results_clean),
        "world_cup_2026_fixtures": len(fixtures),
        "goalscorer_rows": len(goalscorers_clean),
        "shootout_rows": len(shootouts_clean),
        "team_name_mappings": len(team_name_map),
    }


def main() -> None:
    summary = write_outputs()
    print("Phase 1 preprocessing complete")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
