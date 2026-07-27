"""
Dixon-Coles bivariate Poisson model for soccer scores.

Home goals ~ Poisson(λ), away goals ~ Poisson(μ), with low-score correlation ρ:

  λ = exp(attack_home - defence_away + home_adv)
  μ = exp(attack_away - defence_home)

Likelihood is time-weighted with exponential decay.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson


DEFAULT_XI = 0.0019  # ~1-year half-life: exp(-xi * 365) ≈ 0.5
MAX_GOALS = 10
RHO_BOUNDS = (-0.2, 0.2)


def dc_tau(hg: int, ag: int, lam: float, mu: float, rho: float) -> float:
    """Dixon-Coles adjustment for (0,0), (1,0), (0,1), (1,1)."""
    if hg == 0 and ag == 0:
        return 1.0 - lam * mu * rho
    if hg == 0 and ag == 1:
        return 1.0 + lam * rho
    if hg == 1 and ag == 0:
        return 1.0 + mu * rho
    if hg == 1 and ag == 1:
        return 1.0 - rho
    return 1.0


def score_prob(hg: int, ag: int, lam: float, mu: float, rho: float) -> float:
    tau = dc_tau(hg, ag, lam, mu, rho)
    return tau * poisson.pmf(hg, lam) * poisson.pmf(ag, mu)


def match_lambdas(
    attack: np.ndarray,
    defence: np.ndarray,
    home_adv: float,
    home_idx: int,
    away_idx: int,
) -> tuple[float, float]:
    lam = float(np.exp(attack[home_idx] - defence[away_idx] + home_adv))
    mu = float(np.exp(attack[away_idx] - defence[home_idx]))
    return lam, mu


def outcome_probs(
    lam: float,
    mu: float,
    rho: float,
    max_goals: int = MAX_GOALS,
) -> tuple[float, float, float]:
    """Return (P_home_win, P_draw, P_away_win), renormalized over 0..max_goals."""
    p_h = p_d = p_a = 0.0
    for hg in range(max_goals + 1):
        for ag in range(max_goals + 1):
            p = score_prob(hg, ag, lam, mu, rho)
            if hg > ag:
                p_h += p
            elif hg == ag:
                p_d += p
            else:
                p_a += p
    total = p_h + p_d + p_a
    if total <= 0:
        return float("nan"), float("nan"), float("nan")
    return p_h / total, p_d / total, p_a / total


@dataclass
class DixonColesModel:
    teams: list[str]
    attack: np.ndarray
    defence: np.ndarray
    home_adv: float
    rho: float
    xi: float = DEFAULT_XI

    @property
    def team_to_idx(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    def predict_match(self, home: str, away: str) -> dict[str, float]:
        idx = self.team_to_idx
        if home not in idx or away not in idx:
            # Unseen team → neutral priors
            return {"p_home": 1 / 3, "p_draw": 1 / 3, "p_away": 1 / 3, "lam": 1.2, "mu": 1.0}
        lam, mu = match_lambdas(
            self.attack, self.defence, self.home_adv, idx[home], idx[away]
        )
        p_h, p_d, p_a = outcome_probs(lam, mu, self.rho)
        return {"p_home": p_h, "p_draw": p_d, "p_away": p_a, "lam": lam, "mu": mu}


def _unpack_params(x: np.ndarray, n_teams: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Parameter vector:
      attack[0..n-2], defence[0..n-2], home_adv, rho
    with attack[n-1] = -sum(attack[:-1]), same for defence (identifiability).
    """
    a_free = x[: n_teams - 1]
    d_free = x[n_teams - 1 : 2 * (n_teams - 1)]
    home_adv = float(x[-2])
    rho = float(x[-1])
    attack = np.zeros(n_teams)
    defence = np.zeros(n_teams)
    attack[:-1] = a_free
    defence[:-1] = d_free
    attack[-1] = -a_free.sum()
    defence[-1] = -d_free.sum()
    return attack, defence, home_adv, rho


def _pack_params(
    attack: np.ndarray, defence: np.ndarray, home_adv: float, rho: float
) -> np.ndarray:
    return np.concatenate([attack[:-1], defence[:-1], [home_adv, rho]])


def _weighted_loglik(
    x: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    attack, defence, home_adv, rho = _unpack_params(x, n_teams)
    if not (RHO_BOUNDS[0] <= rho <= RHO_BOUNDS[1]):
        return 1e12

    lam = np.exp(attack[home_idx] - defence[away_idx] + home_adv)
    mu = np.exp(attack[away_idx] - defence[home_idx])

    # Vectorized DC tau for common cases
    tau = np.ones(len(hg), dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho

    # Invalid tau (non-positive) → heavy penalty
    if np.any(tau <= 0):
        return 1e12

    log_p = (
        np.log(tau)
        + poisson.logpmf(hg, lam)
        + poisson.logpmf(ag, mu)
    )
    # Guard against -inf
    log_p = np.where(np.isfinite(log_p), log_p, -50.0)
    return float(-np.sum(weights * log_p))


def fit_dixon_coles(
    homes: list[str] | np.ndarray,
    aways: list[str] | np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    dates: np.ndarray,
    *,
    xi: float = DEFAULT_XI,
    x0: np.ndarray | None = None,
    teams: list[str] | None = None,
) -> DixonColesModel:
    """
    Fit Dixon-Coles on a match sample.

    `dates` should be timezone-naive datetimes; newest match gets weight 1,
    older matches get exp(-xi * days_ago).
    """
    homes = np.asarray(homes)
    aways = np.asarray(aways)
    hg = np.asarray(home_goals, dtype=int)
    ag = np.asarray(away_goals, dtype=int)
    dates = pd_to_datetime64(dates)

    if teams is None:
        teams = sorted(set(homes.tolist()) | set(aways.tolist()))
    team_to_idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)

    home_idx = np.array([team_to_idx[t] for t in homes], dtype=int)
    away_idx = np.array([team_to_idx[t] for t in aways], dtype=int)

    max_date = dates.max()
    days_ago = ((max_date - dates) / np.timedelta64(1, "D")).astype(float)
    weights = np.exp(-xi * days_ago)

    if x0 is None:
        x0 = np.zeros(2 * (n - 1) + 2)
        x0[-2] = 0.25  # home advantage in log-intensity space
        x0[-1] = -0.03  # typical small negative rho

    bounds = [(-3.0, 3.0)] * (2 * (n - 1)) + [(-1.0, 1.5), RHO_BOUNDS]

    res = minimize(
        _weighted_loglik,
        x0,
        args=(home_idx, away_idx, hg, ag, weights, n),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-8},
    )
    attack, defence, home_adv, rho = _unpack_params(res.x, n)
    return DixonColesModel(
        teams=teams,
        attack=attack,
        defence=defence,
        home_adv=float(home_adv),
        rho=float(rho),
        xi=xi,
    )


def pd_to_datetime64(dates: np.ndarray) -> np.ndarray:
    import pandas as pd

    return pd.to_datetime(dates).to_numpy(dtype="datetime64[ns]")


def model_to_x0(model: DixonColesModel, teams: list[str]) -> np.ndarray:
    """Warm-start vector for a (possibly expanded) team list."""
    n = len(teams)
    attack = np.zeros(n)
    defence = np.zeros(n)
    old = model.team_to_idx
    for i, t in enumerate(teams):
        if t in old:
            attack[i] = model.attack[old[t]]
            defence[i] = model.defence[old[t]]
    # re-center for constraint
    attack = attack - attack.mean()
    defence = defence - defence.mean()
    return _pack_params(attack, defence, model.home_adv, model.rho)
