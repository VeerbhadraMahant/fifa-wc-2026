from __future__ import annotations

import re

import pandas as pd


CUTOFF_DATE = pd.Timestamp("2026-06-11")


def categorize_tournament(tournament: object) -> str:
    name = "" if pd.isna(tournament) else str(tournament).strip().lower()

    if name == "fifa world cup":
        return "fifa_world_cup"
    if "world cup" in name and "qualification" in name:
        return "world_cup_qualification"
    if name == "friendly":
        return "friendly"
    if "qualification" in name:
        return "continental_qualification"
    if "nations league" in name:
        return "nations_league"

    continental_patterns = (
        "uefa euro",
        "copa américa",
        "copa america",
        "african cup of nations",
        "afc asian cup",
        "gold cup",
        "concacaf championship",
        "oceania nations cup",
    )
    if any(pattern in name for pattern in continental_patterns):
        return "continental_championship"

    return "other"


def tournament_weight(category: str) -> float:
    weights = {
        "fifa_world_cup": 1.00,
        "world_cup_qualification": 0.85,
        "continental_championship": 0.80,
        "continental_qualification": 0.70,
        "nations_league": 0.60,
        "friendly": 0.30,
        "other": 0.50,
    }
    return weights.get(category, 0.50)


def recency_weight(date: pd.Timestamp) -> float:
    if date >= pd.Timestamp("2022-01-01"):
        return 1.00
    if date >= pd.Timestamp("2018-01-01"):
        return 0.70
    if date >= pd.Timestamp("2010-01-01"):
        return 0.40
    return 0.15


def build_match_features(results: pd.DataFrame) -> pd.DataFrame:
    features = results.copy()
    features["date"] = pd.to_datetime(features["date"], errors="coerce")
    features["home_score"] = pd.to_numeric(features["home_score"], errors="coerce").astype(int)
    features["away_score"] = pd.to_numeric(features["away_score"], errors="coerce").astype(int)

    features = features.rename(
        columns={
            "home_score": "home_goals",
            "away_score": "away_goals",
        }
    )
    features["goal_difference"] = features["home_goals"] - features["away_goals"]
    features["result_label"] = features["goal_difference"].map(
        lambda diff: "home_win" if diff > 0 else "away_win" if diff < 0 else "draw"
    )
    features["is_draw"] = features["goal_difference"].eq(0)
    features["is_home_win"] = features["goal_difference"].gt(0)
    features["is_away_win"] = features["goal_difference"].lt(0)
    features["is_neutral"] = (
        features["neutral"].astype("string").str.strip().str.lower().eq("true")
    )
    features["year"] = features["date"].dt.year
    features["days_before_cutoff"] = (CUTOFF_DATE - features["date"]).dt.days
    features["tournament_category"] = features["tournament"].map(categorize_tournament)
    features["recency_weight"] = features["date"].map(recency_weight)
    features["tournament_weight"] = features["tournament_category"].map(tournament_weight)
    features["final_training_weight"] = (
        features["recency_weight"] * features["tournament_weight"]
    )

    return features.sort_values(["date", "home_team", "away_team"]).reset_index(drop=True)


def _to_bool(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().eq("true")


def build_team_features(goalscorers: pd.DataFrame) -> pd.DataFrame:
    goals = goalscorers.copy()
    goals["date"] = pd.to_datetime(goals["date"], errors="coerce")
    goals = goals[goals["date"] < CUTOFF_DATE].copy()
    goals["minute"] = pd.to_numeric(goals["minute"], errors="coerce")
    goals["own_goal"] = _to_bool(goals["own_goal"])
    goals["penalty"] = _to_bool(goals["penalty"])

    team_rows = []
    for team, team_goals in goals.groupby("team", dropna=True):
        total_goals = len(team_goals)
        penalties_scored = int(team_goals["penalty"].sum())
        penalties_attempted_observed = penalties_scored
        late_goal_count = int(team_goals["minute"].ge(80).sum())
        own_goals_for = int(team_goals["own_goal"].sum())
        distinct_scorers = int(
            team_goals.loc[~team_goals["own_goal"], "scorer"].dropna().nunique()
        )

        team_rows.append(
            {
                "team": team,
                "total_goals": total_goals,
                "penalties_scored": penalties_scored,
                "penalties_attempted_observed": penalties_attempted_observed,
                "penalty_conversion_rate": 1.0 if penalties_scored > 0 else pd.NA,
                "distinct_scorer_count": distinct_scorers,
                "late_goal_count": late_goal_count,
                "late_goal_rate": late_goal_count / total_goals if total_goals else 0.0,
                "own_goals_for": own_goals_for,
                "own_goal_rate": own_goals_for / total_goals if total_goals else 0.0,
            }
        )

    features = pd.DataFrame(team_rows)
    if features.empty:
        return features

    features["penalty_conversion_rate"] = pd.to_numeric(
        features["penalty_conversion_rate"],
        errors="coerce",
    )
    global_penalty_rate = features["penalty_conversion_rate"].dropna().mean()
    if pd.isna(global_penalty_rate):
        global_penalty_rate = 0.0
    features["penalty_conversion_rate"] = features["penalty_conversion_rate"].fillna(
        global_penalty_rate
    )

    return features.sort_values("team").reset_index(drop=True)


def build_shootout_features(shootouts: pd.DataFrame) -> pd.DataFrame:
    shootouts = shootouts.copy()
    shootouts["date"] = pd.to_datetime(shootouts["date"], errors="coerce")
    shootouts = shootouts[shootouts["date"] < CUTOFF_DATE].copy()

    teams = pd.concat(
        [shootouts["home_team"], shootouts["away_team"], shootouts["winner"]],
        ignore_index=True,
    ).dropna()

    rows = []
    for team in sorted(teams.unique()):
        participated = shootouts[
            (shootouts["home_team"] == team) | (shootouts["away_team"] == team)
        ]
        total = len(participated)
        wins = int((participated["winner"] == team).sum())
        rows.append(
            {
                "team": team,
                "shootout_wins": wins,
                "total_shootouts": total,
                "raw_shootout_win_rate": wins / total if total else pd.NA,
                "smoothed_shootout_win_probability": (wins + 2) / (total + 4),
            }
        )

    return pd.DataFrame(rows).sort_values("team").reset_index(drop=True)


def infer_fixture_groups(fixtures: pd.DataFrame) -> pd.DataFrame:
    """Infer groups from the fixture order: 72 matches, 12 groups, 6 matches each."""
    grouped = fixtures.copy()
    grouped["fixture_number"] = range(1, len(grouped) + 1)
    grouped["group"] = grouped["fixture_number"].map(
        lambda value: chr(ord("A") + ((value - 1) % 24) // 2)
    )
    return grouped
