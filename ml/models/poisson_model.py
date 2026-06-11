from __future__ import annotations

from dataclasses import dataclass
from math import exp, factorial

import pandas as pd


MAX_GOALS = 9


def _poisson_pmf(rate: float, goals: int) -> float:
    return (exp(-rate) * (rate ** goals)) / factorial(goals)


@dataclass
class TeamStrength:
    attack: float
    defense: float
    weighted_matches: float


@dataclass
class PoissonGoalModel:
    strengths: dict[str, TeamStrength]
    team_features: dict[str, dict[str, float]]
    global_goals_per_team: float
    default_strength: TeamStrength
    global_late_goal_rate: float
    global_own_goal_rate: float

    @classmethod
    def fit(
        cls,
        matches: pd.DataFrame,
        team_features: pd.DataFrame,
        shrinkage_matches: float = 18.0,
    ) -> "PoissonGoalModel":
        team_rows = []
        for row in matches.itertuples(index=False):
            weight = float(row.final_training_weight)
            team_rows.append(
                {
                    "team": row.home_team,
                    "goals_for": float(row.home_goals),
                    "goals_against": float(row.away_goals),
                    "weight": weight,
                }
            )
            team_rows.append(
                {
                    "team": row.away_team,
                    "goals_for": float(row.away_goals),
                    "goals_against": float(row.home_goals),
                    "weight": weight,
                }
            )

        long_matches = pd.DataFrame(team_rows)
        weighted_goals = (long_matches["goals_for"] * long_matches["weight"]).sum()
        total_weight = long_matches["weight"].sum()
        global_goals = weighted_goals / total_weight

        strengths: dict[str, TeamStrength] = {}
        for team, rows in long_matches.groupby("team"):
            weight_sum = rows["weight"].sum()
            goals_for = (rows["goals_for"] * rows["weight"]).sum() / weight_sum
            goals_against = (rows["goals_against"] * rows["weight"]).sum() / weight_sum
            shrink = weight_sum / (weight_sum + shrinkage_matches)
            attack = 1.0 + shrink * ((goals_for / global_goals) - 1.0)
            defense = 1.0 + shrink * ((goals_against / global_goals) - 1.0)
            strengths[team] = TeamStrength(
                attack=max(0.35, min(2.75, attack)),
                defense=max(0.35, min(2.75, defense)),
                weighted_matches=weight_sum,
            )

        feature_map: dict[str, dict[str, float]] = {}
        if not team_features.empty:
            numeric_features = team_features.copy()
            for column in numeric_features.columns:
                if column != "team":
                    numeric_features[column] = pd.to_numeric(
                        numeric_features[column],
                        errors="coerce",
                    )
            feature_map = numeric_features.set_index("team").to_dict(orient="index")
            global_late_goal_rate = float(numeric_features["late_goal_rate"].mean())
            global_own_goal_rate = float(numeric_features["own_goal_rate"].mean())
        else:
            global_late_goal_rate = 0.0
            global_own_goal_rate = 0.0

        return cls(
            strengths=strengths,
            team_features=feature_map,
            global_goals_per_team=global_goals,
            default_strength=TeamStrength(attack=1.0, defense=1.0, weighted_matches=0.0),
            global_late_goal_rate=global_late_goal_rate,
            global_own_goal_rate=global_own_goal_rate,
        )

    def expected_goals(self, team_a: str, team_b: str) -> tuple[float, float]:
        strength_a = self.strengths.get(team_a, self.default_strength)
        strength_b = self.strengths.get(team_b, self.default_strength)

        goals_a = self.global_goals_per_team * strength_a.attack * strength_b.defense
        goals_b = self.global_goals_per_team * strength_b.attack * strength_a.defense

        goals_a *= self._feature_adjustment(team_a, team_b)
        goals_b *= self._feature_adjustment(team_b, team_a)

        return (
            max(0.15, min(5.0, goals_a)),
            max(0.15, min(5.0, goals_b)),
        )

    def _feature_adjustment(self, attacking_team: str, defending_team: str) -> float:
        attack_features = self.team_features.get(attacking_team, {})
        defense_features = self.team_features.get(defending_team, {})
        late_delta = float(attack_features.get("late_goal_rate", self.global_late_goal_rate)) - self.global_late_goal_rate
        own_goal_delta = float(defense_features.get("own_goal_rate", self.global_own_goal_rate)) - self.global_own_goal_rate
        adjustment = 1.0 + (0.08 * late_delta) + (0.10 * own_goal_delta)
        return max(0.90, min(1.10, adjustment))

    def scoreline_matrix(self, team_a: str, team_b: str) -> list[list[float]]:
        goals_a, goals_b = self.expected_goals(team_a, team_b)
        probabilities_a = [_poisson_pmf(goals_a, goals) for goals in range(MAX_GOALS + 1)]
        probabilities_b = [_poisson_pmf(goals_b, goals) for goals in range(MAX_GOALS + 1)]
        matrix = [
            [probability_a * probability_b for probability_b in probabilities_b]
            for probability_a in probabilities_a
        ]
        total = sum(sum(row) for row in matrix)
        return [[cell / total for cell in row] for row in matrix]

    def outcome_probabilities(self, team_a: str, team_b: str) -> dict[str, float]:
        matrix = self.scoreline_matrix(team_a, team_b)
        home_win = 0.0
        draw = 0.0
        away_win = 0.0
        for home_goals, row in enumerate(matrix):
            for away_goals, probability in enumerate(row):
                if home_goals > away_goals:
                    home_win += probability
                elif home_goals < away_goals:
                    away_win += probability
                else:
                    draw += probability

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
        }
