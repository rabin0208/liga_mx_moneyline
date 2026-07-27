"""
Odds helpers for Liga MX 1X2 → two-way (favorite vs underdog double chance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def implied_prob(decimal_odds: float) -> float:
    o = float(decimal_odds)
    if o <= 1.0:
        return float("nan")
    return 1.0 / o


def remove_vig_3way(p_h: float, p_d: float, p_a: float) -> tuple[float, float, float]:
    total = p_h + p_d + p_a
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    return p_h / total, p_d / total, p_a / total


def double_chance_odds(draw_odds: float, dog_odds: float) -> float:
    """Approximate DC price from 1X2 decimal odds (vig still embedded)."""
    return 1.0 / (implied_prob(draw_odds) + implied_prob(dog_odds))


def flat_bet_profit(decimal_odds: float, won: bool) -> float:
    if not won:
        return -1.0
    return float(decimal_odds) - 1.0


def add_two_way_market(df: pd.DataFrame) -> pd.DataFrame:
    """
    From AvgCH/AvgCD/AvgCA closing averages, add favorite / DC legs and fair probs.

    Skips pick'ems (AvgCH == AvgCA).
    """
    out = df.copy()
    for c in ("AvgCH", "AvgCD", "AvgCA"):
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["AvgCH", "AvgCD", "AvgCA", "Res"])
    out = out.loc[out["AvgCH"] != out["AvgCA"]].copy()

    home_fav = out["AvgCH"] < out["AvgCA"]
    out["favorite_side"] = np.where(home_fav, "H", "A")
    out["underdog_side"] = np.where(home_fav, "A", "H")
    out["fav_odds"] = np.where(home_fav, out["AvgCH"], out["AvgCA"])
    out["dog_odds"] = np.where(home_fav, out["AvgCA"], out["AvgCH"])
    out["dc_odds"] = [
        double_chance_odds(d, dog) for d, dog in zip(out["AvgCD"], out["dog_odds"])
    ]

    out["fav_won"] = (out["Res"] == out["favorite_side"]).astype(int)
    out["dc_won"] = 1 - out["fav_won"]

    p_h = out["AvgCH"].map(implied_prob)
    p_d = out["AvgCD"].map(implied_prob)
    p_a = out["AvgCA"].map(implied_prob)
    fair = [remove_vig_3way(h, d, a) for h, d, a in zip(p_h, p_d, p_a)]
    out["p_home_fair"] = [t[0] for t in fair]
    out["p_draw_fair"] = [t[1] for t in fair]
    out["p_away_fair"] = [t[2] for t in fair]
    out["p_fav_mkt"] = np.where(home_fav, out["p_home_fair"], out["p_away_fair"])
    out["p_dc_mkt"] = 1.0 - out["p_fav_mkt"]
    return out.reset_index(drop=True)
