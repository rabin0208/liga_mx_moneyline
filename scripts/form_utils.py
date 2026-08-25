"""
Short-run form overlay on a long-run Poisson 1X2 model.

Long-run λ, μ come from Bayesian Dixon–Coles (ρ kept). Each team's last N
games contribute attack/defence residuals (goals scored/conceded minus
expected). An EWMA of those residuals shifts λ, μ; the 1X2 is a mix of
long-run and form-adjusted probabilities, both using Dixon–Coles τ(ρ):

    λ' = clip(λ + scale * (att_home + def_away))
    μ' = clip(μ + scale * (att_away + def_home))
    P  = (1 − w) P_long(λ, μ, ρ) + w P(λ', μ', ρ)

Defence residual is leakiness (conceded − expected). MLS expanding 2025
preferred w = 0, so the live default weight is 0 until Liga MX evidence
says otherwise.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from dixon_coles_utils import intercept_home_from_model, match_lambdas, outcome_probs

DEFAULT_FORM_GAMES = 8
DEFAULT_FORM_HALFLIFE = 3.0
DEFAULT_FORM_WEIGHT = 0.0
DEFAULT_FORM_SCALE = 1.0
MIN_FORM_GAMES = 3
LAM_MIN = 0.20
LAM_MAX = 4.50


def ewma_weights(n: int, halflife: float) -> np.ndarray:
    """Newest observation gets the largest weight. Length n, sums to 1."""
    if n <= 0:
        return np.zeros(0)
    ages = np.arange(n - 1, -1, -1, dtype=float)
    w = np.exp(-np.log(2.0) / max(halflife, 1e-6) * ages)
    return w / w.sum()


def _clip_lambda(x: float) -> float:
    return float(np.clip(x, LAM_MIN, LAM_MAX))


def expected_goals(model, home: str, away: str) -> tuple[float, float] | None:
    idx = model.team_to_idx
    if home not in idx or away not in idx:
        return None
    intercept, home_adv = intercept_home_from_model(model)
    return match_lambdas(
        model.attack, model.defence, intercept, home_adv, idx[home], idx[away]
    )


def team_form(
    hist: pd.DataFrame,
    model,
    *,
    n_games: int = DEFAULT_FORM_GAMES,
    halflife: float = DEFAULT_FORM_HALFLIFE,
    teams: set[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """
    Per-team (attack_form, defence_form) in goal units.

    Attack residual = goals scored − expected scored.
    Defence residual = goals conceded − expected conceded (positive = leaky).
    Only the last ``n_games`` appearances per team are used (walked newest-first).
    Pass ``teams`` to skip clubs that are not on the current slate.
    """
    if hist.empty or n_games <= 0:
        return {}
    if teams is None:
        teams = set(hist["home_name"].astype(str)) | set(hist["away_name"].astype(str))
    remaining = {str(t): n_games for t in teams}
    if not remaining:
        return {}
    work = hist.sort_values("game_date")
    att: dict[str, list[float]] = defaultdict(list)
    deff: dict[str, list[float]] = defaultdict(list)

    for row in work.iloc[::-1].itertuples(index=False):
        if not remaining:
            break
        home = str(row.home_name)
        away = str(row.away_name)
        take_h = remaining.get(home, 0) > 0
        take_a = remaining.get(away, 0) > 0
        if not take_h and not take_a:
            continue
        exp = expected_goals(model, home, away)
        if exp is None:
            continue
        lam, mu = exp
        hg = float(row.home_goals)
        ag = float(row.away_goals)
        if take_h:
            att[home].append(hg - lam)
            deff[home].append(ag - mu)
            remaining[home] -= 1
            if remaining[home] <= 0:
                remaining.pop(home, None)
        if take_a:
            att[away].append(ag - mu)
            deff[away].append(hg - lam)
            remaining[away] -= 1
            if remaining[away] <= 0:
                remaining.pop(away, None)

    form: dict[str, tuple[float, float]] = {}
    for team in set(att) | set(deff):
        a = np.asarray(att[team][::-1], dtype=float)
        d = np.asarray(deff[team][::-1], dtype=float)
        if len(a) < MIN_FORM_GAMES:
            form[team] = (0.0, 0.0)
            continue
        w = ewma_weights(len(a), halflife)
        form[team] = (float(np.dot(w, a)), float(np.dot(w, d)))
    return form


def form_lambdas(
    lam: float,
    mu: float,
    home: str,
    away: str,
    form: dict[str, tuple[float, float]],
    *,
    scale: float = 1.0,
) -> tuple[float, float]:
    att_h, def_h = form.get(home, (0.0, 0.0))
    att_a, def_a = form.get(away, (0.0, 0.0))
    s = float(scale)
    lam_f = _clip_lambda(lam + s * (att_h + def_a))
    mu_f = _clip_lambda(mu + s * (att_a + def_h))
    return lam_f, mu_f


def mix_probs(
    p_long: tuple[float, float, float],
    p_form: tuple[float, float, float],
    weight: float,
) -> tuple[float, float, float]:
    w = min(max(float(weight), 0.0), 1.0)
    p_h = (1.0 - w) * p_long[0] + w * p_form[0]
    p_d = (1.0 - w) * p_long[1] + w * p_form[1]
    p_a = (1.0 - w) * p_long[2] + w * p_form[2]
    total = p_h + p_d + p_a
    if total <= 0:
        return p_long
    return p_h / total, p_d / total, p_a / total


def predict_match_with_form(
    model,
    home: str,
    away: str,
    form: dict[str, tuple[float, float]],
    *,
    weight: float = DEFAULT_FORM_WEIGHT,
    scale: float = 1.0,
    use_posterior: bool = False,
) -> dict[str, float]:
    pred = model.predict_match(home, away, use_posterior=use_posterior)
    lam = float(pred["lam"])
    mu = float(pred["mu"])
    p_long = (float(pred["p_home"]), float(pred["p_draw"]), float(pred["p_away"]))
    if weight <= 0.0 or not form:
        pred["form_att_H"] = 0.0
        pred["form_def_H"] = 0.0
        pred["form_att_A"] = 0.0
        pred["form_def_A"] = 0.0
        pred["form_weight"] = 0.0
        pred["form_scale"] = 0.0
        return pred

    rho = float(getattr(model, "rho", 0.0))
    lam_f, mu_f = form_lambdas(lam, mu, home, away, form, scale=scale)
    p_form = outcome_probs(lam_f, mu_f, rho)
    p_h, p_d, p_a = mix_probs(p_long, p_form, weight)
    w = min(max(float(weight), 0.0), 1.0)
    att_h, def_h = form.get(home, (0.0, 0.0))
    att_a, def_a = form.get(away, (0.0, 0.0))
    pred["p_home"] = p_h
    pred["p_draw"] = p_d
    pred["p_away"] = p_a
    pred["lam"] = (1.0 - w) * lam + w * lam_f
    pred["mu"] = (1.0 - w) * mu + w * mu_f
    pred["form_att_H"] = att_h
    pred["form_def_H"] = def_h
    pred["form_att_A"] = att_a
    pred["form_def_A"] = def_a
    pred["form_weight"] = w
    pred["form_scale"] = float(scale)
    return pred
