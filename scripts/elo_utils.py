"""
Classic soccer Elo: win=1, draw=0.5, loss=0 with home-advantage offset.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_RATING = 1500.0
DEFAULT_K = 20.0
DEFAULT_HOME_ADV = 80.0
SCALE = 400.0


def expected_score(rating_a: float, rating_b: float) -> float:
    """P(A beats B) on the 0–1 Elo scale (draws as 0.5 targets)."""
    return 1.0 / (1.0 + 10.0 ** (-(rating_a - rating_b) / SCALE))


@dataclass
class EloState:
    k: float = DEFAULT_K
    home_adv: float = DEFAULT_HOME_ADV
    initial: float = DEFAULT_RATING
    ratings: dict[int, float] = field(default_factory=dict)

    def rating(self, team_id: int) -> float:
        return self.ratings.get(int(team_id), self.initial)

    def pre_match(
        self, home_id: int, away_id: int
    ) -> tuple[float, float, float, float]:
        """
        Returns (r_home, r_away, elo_diff, e_home).

        elo_diff = (R_home + home_adv) - R_away
        e_home = expected home score in {0, 0.5, 1} space.
        """
        r_home = self.rating(home_id)
        r_away = self.rating(away_id)
        elo_diff = (r_home + self.home_adv) - r_away
        e_home = expected_score(r_home + self.home_adv, r_away)
        return r_home, r_away, elo_diff, e_home

    def reset(self) -> None:
        """Clear all ratings back to the initial value (season restart)."""
        self.ratings.clear()

    def update(self, home_id: int, away_id: int, result: str) -> None:
        """Update ratings after observing result in {H, D, A}."""
        r_home, r_away, _diff, e_home = self.pre_match(home_id, away_id)
        if result == "H":
            actual_home = 1.0
        elif result == "A":
            actual_home = 0.0
        elif result == "D":
            actual_home = 0.5
        else:
            raise ValueError(f"Unknown result: {result}")

        delta = self.k * (actual_home - e_home)
        self.ratings[int(home_id)] = r_home + delta
        self.ratings[int(away_id)] = r_away - delta


def elo_edge_for_favorite(
    elo_diff: float, favorite_side: str
) -> float:
    """
    Positive when the market favorite is stronger on Elo.

    elo_diff is (R_home + H) - R_away.
    """
    if favorite_side == "H":
        return float(elo_diff)
    if favorite_side == "A":
        return float(-elo_diff)
    raise ValueError(f"Unknown favorite_side: {favorite_side}")


def clip_prob(p: float, eps: float = 1e-6) -> float:
    return float(np.clip(p, eps, 1.0 - eps))
