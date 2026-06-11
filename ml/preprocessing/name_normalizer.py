from __future__ import annotations

from pathlib import Path

import pandas as pd


TEAM_COLUMNS = ("home_team", "away_team", "team", "winner", "first_shooter")


def load_name_map(path: Path) -> dict[str, str]:
    """Load former-to-current team name mappings."""
    names = pd.read_csv(path, dtype=str).fillna("")
    required_columns = {"current", "former"}
    missing_columns = required_columns.difference(names.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"{path} is missing required columns: {missing}")

    mapping = {}
    for row in names.itertuples(index=False):
        current = str(row.current).strip()
        former = str(row.former).strip()
        if former and current:
            mapping[former] = current
    return mapping


def normalize_team_name(value: object, name_map: dict[str, str]) -> object:
    if pd.isna(value):
        return value

    name = str(value).strip()
    if not name:
        return pd.NA
    return name_map.get(name, name)


def normalize_team_columns(
    frame: pd.DataFrame,
    name_map: dict[str, str],
    columns: tuple[str, ...] = TEAM_COLUMNS,
) -> pd.DataFrame:
    normalized = frame.copy()
    for column in columns:
        if column in normalized.columns:
            normalized[column] = normalized[column].map(
                lambda value: normalize_team_name(value, name_map)
            )
    return normalized
