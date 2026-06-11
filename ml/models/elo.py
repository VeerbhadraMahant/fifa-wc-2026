from __future__ import annotations

from dataclasses import dataclass
from math import log

import pandas as pd


@dataclass
class EloModel:
    ratings: dict[str, float]
    default_rating: float = 1500.0
    home_advantage: float = 75.0

    @classmethod
    def fit(
        cls,
        matches: pd.DataFrame,
        default_rating: float = 1500.0,
        base_k: float = 40.0,
        home_advantage: float = 75.0,
    ) -> "EloModel":
        ratings: dict[str, float] = {}
        ordered = matches.sort_values("date")

        for row in ordered.itertuples(index=False):
            home_team = row.home_team
            away_team = row.away_team
            home_rating = ratings.get(home_team, default_rating)
            away_rating = ratings.get(away_team, default_rating)
            is_neutral = bool(row.is_neutral)
            home_offset = 0.0 if is_neutral else home_advantage

            expected_home = cls.expected_score(
                home_rating + home_offset,
                away_rating,
            )
            home_goals = int(row.home_goals)
            away_goals = int(row.away_goals)
            actual_home = 1.0 if home_goals > away_goals else 0.0 if home_goals < away_goals else 0.5
            goal_diff = abs(home_goals - away_goals)
            mov_multiplier = log(goal_diff + 1.0) * (
                2.2 / ((abs(home_rating - away_rating) * 0.001) + 2.2)
            )
            if goal_diff == 0:
                mov_multiplier = 1.0

            match_weight = float(row.final_training_weight)
            adjustment = base_k * match_weight * mov_multiplier * (actual_home - expected_home)
            ratings[home_team] = home_rating + adjustment
            ratings[away_team] = away_rating - adjustment

        return cls(
            ratings=ratings,
            default_rating=default_rating,
            home_advantage=home_advantage,
        )

    @staticmethod
    def expected_score(rating_a: float, rating_b: float) -> float:
        return 1.0 / (1.0 + (10.0 ** ((rating_b - rating_a) / 400.0)))

    def rating(self, team: str) -> float:
        return self.ratings.get(team, self.default_rating)

    def outcome_probabilities(self, team_a: str, team_b: str) -> dict[str, float]:
        rating_a = self.rating(team_a)
        rating_b = self.rating(team_b)
        expected_a = self.expected_score(rating_a, rating_b)
        rating_gap = abs(rating_a - rating_b)
        draw = max(0.16, min(0.30, 0.28 - (rating_gap / 2500.0)))
        decisive = 1.0 - draw
        home_win = decisive * expected_a
        away_win = decisive * (1.0 - expected_a)

        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
        }
