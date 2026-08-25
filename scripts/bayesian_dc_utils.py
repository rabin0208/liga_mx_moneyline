"""
Bayesian hierarchical Dixon-Coles model (Laplace posterior).

Team attack/defence strengths have Gaussian shrinkage priors:

  attack_i  ~ N(0, σ_att²)
  defence_i ~ N(0, σ_def²)
  intercept ~ N(μ_int, σ_int²)     # log average scoring rate
  home_adv  ~ N(μ_home, σ_home²)   # extra home boost only
  rho       ~ N(μ_rho, σ_rho²) truncated to RHO_BOUNDS

Likelihood matches the frequentist Dixon-Coles model (independent Poisson
goals with low-score correlation τ), optionally time-weighted:

  λ = exp(intercept + attack_home - defence_away + home_adv)
  μ = exp(intercept + attack_away - defence_home)

Inference:
  1. MAP via L-BFGS-B on the negative log posterior
  2. Laplace approximation: posterior ≈ N(MAP, H^{-1})
  3. Posterior-predictive 1X2 probs by averaging score matrices over draws
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

from dixon_coles_utils import (
    DEFAULT_INTERCEPT,
    DEFAULT_XI,
    MAX_GOALS,
    RHO_BOUNDS,
    intercept_home_from_model,
    match_lambdas,
    n_free_params,
    outcome_probs,
    pd_to_datetime64,
    _pack_params,
    _unpack_params,
)


# Prior hyperparameters (log-intensity scale)
DEFAULT_SIGMA_ATT = 0.35
DEFAULT_SIGMA_DEF = 0.35
DEFAULT_MU_INTERCEPT = DEFAULT_INTERCEPT
DEFAULT_SIGMA_INTERCEPT = 0.25
DEFAULT_MU_HOME = 0.25
DEFAULT_SIGMA_HOME = 0.40
DEFAULT_MU_RHO = -0.05
DEFAULT_SIGMA_RHO = 0.08
DEFAULT_N_POSTERIOR = 250


@dataclass
class BayesianDCModel:
    teams: list[str]
    attack: np.ndarray  # MAP
    defence: np.ndarray  # MAP
    home_adv: float
    rho: float
    xi: float
    sigma_att: float
    sigma_def: float
    # Laplace posterior draws of free params (n_samples, n_free)
    posterior_samples: np.ndarray | None = None
    map_x: np.ndarray | None = None
    cov: np.ndarray | None = None
    intercept: float = 0.0

    @property
    def team_to_idx(self) -> dict[str, int]:
        return {t: i for i, t in enumerate(self.teams)}

    def predict_match(
        self,
        home: str,
        away: str,
        *,
        use_posterior: bool = True,
        max_goals: int = MAX_GOALS,
    ) -> dict[str, float]:
        """Posterior-mean 1X2 (or MAP if no samples / use_posterior=False)."""
        idx = self.team_to_idx
        if home not in idx or away not in idx:
            return {
                "p_home": 1 / 3,
                "p_draw": 1 / 3,
                "p_away": 1 / 3,
                "lam": 1.2,
                "mu": 1.0,
                "p_home_lo": 1 / 3,
                "p_home_hi": 1 / 3,
            }

        hi, ai = idx[home], idx[away]
        if (
            use_posterior
            and self.posterior_samples is not None
            and len(self.posterior_samples) > 0
        ):
            return _predict_from_samples(
                self.posterior_samples,
                len(self.teams),
                hi,
                ai,
                max_goals=max_goals,
            )

        intercept, home_adv = intercept_home_from_model(self)
        lam, mu = match_lambdas(
            self.attack, self.defence, intercept, home_adv, hi, ai
        )
        p_h, p_d, p_a = outcome_probs(lam, mu, self.rho, max_goals=max_goals)
        return {
            "p_home": p_h,
            "p_draw": p_d,
            "p_away": p_a,
            "lam": lam,
            "mu": mu,
            "p_home_lo": p_h,
            "p_home_hi": p_h,
        }


def _neg_log_prior(
    x: np.ndarray,
    n_teams: int,
    sigma_att: float,
    sigma_def: float,
    mu_intercept: float,
    sigma_intercept: float,
    mu_home: float,
    sigma_home: float,
    mu_rho: float,
    sigma_rho: float,
) -> float:
    attack, defence, intercept, home_adv, rho = _unpack_params(x, n_teams)
    # Soft sum-to-zero already hard-coded; prior on all team params
    nlp = 0.5 * np.sum((attack / sigma_att) ** 2)
    nlp += 0.5 * np.sum((defence / sigma_def) ** 2)
    nlp += 0.5 * ((intercept - mu_intercept) / sigma_intercept) ** 2
    nlp += 0.5 * ((home_adv - mu_home) / sigma_home) ** 2
    nlp += 0.5 * ((rho - mu_rho) / sigma_rho) ** 2
    # Keep rho inside valid DC region
    if rho < RHO_BOUNDS[0] or rho > RHO_BOUNDS[1]:
        return 1e12
    return float(nlp)


def _neg_log_lik(
    x: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    attack, defence, intercept, home_adv, rho = _unpack_params(x, n_teams)
    if not (RHO_BOUNDS[0] <= rho <= RHO_BOUNDS[1]):
        return 1e12

    lam = np.exp(intercept + attack[home_idx] - defence[away_idx] + home_adv)
    mu = np.exp(intercept + attack[away_idx] - defence[home_idx])

    tau = np.ones(len(hg), dtype=float)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    if np.any(tau <= 0):
        return 1e12

    log_p = np.log(tau) + poisson.logpmf(hg, lam) + poisson.logpmf(ag, mu)
    log_p = np.where(np.isfinite(log_p), log_p, -50.0)
    return float(-np.sum(weights * log_p))


def _neg_log_posterior(
    x: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    hg: np.ndarray,
    ag: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    sigma_att: float,
    sigma_def: float,
    mu_intercept: float,
    sigma_intercept: float,
    mu_home: float,
    sigma_home: float,
    mu_rho: float,
    sigma_rho: float,
) -> float:
    nll = _neg_log_lik(x, home_idx, away_idx, hg, ag, weights, n_teams)
    if nll >= 1e12:
        return nll
    nlp = _neg_log_prior(
        x,
        n_teams,
        sigma_att,
        sigma_def,
        mu_intercept,
        sigma_intercept,
        mu_home,
        sigma_home,
        mu_rho,
        sigma_rho,
    )
    return nll + nlp


def _numerical_hessian(
    fun,
    x: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """Central-difference Hessian of scalar `fun` at `x`."""
    n = len(x)
    hess = np.zeros((n, n), dtype=float)
    f0 = fun(x)
    # Diagonal
    for i in range(n):
        e = np.zeros(n)
        e[i] = eps
        f_pp = fun(x + e)
        f_mm = fun(x - e)
        hess[i, i] = (f_pp - 2.0 * f0 + f_mm) / (eps ** 2)
    # Off-diagonal
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = eps
            ej[j] = eps
            f_pp = fun(x + ei + ej)
            f_pm = fun(x + ei - ej)
            f_mp = fun(x - ei + ej)
            f_mm = fun(x - ei - ej)
            val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * eps * eps)
            hess[i, j] = val
            hess[j, i] = val
    return hess


def _outcome_probs_many(
    lam: np.ndarray,
    mu: np.ndarray,
    rho: np.ndarray,
    max_goals: int = MAX_GOALS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized 1X2 probs for many (λ, μ, ρ) draws."""
    n = len(lam)
    p_h = np.zeros(n)
    p_d = np.zeros(n)
    p_a = np.zeros(n)
    goals = np.arange(max_goals + 1)
    # P(goals) per draw: shape (n, G)
    # Use log-pmf for stability then exp
    for i in range(n):
        ph = poisson.pmf(goals, lam[i])
        pa = poisson.pmf(goals, mu[i])
        # Outer product with DC tau on low scores
        mat = np.outer(ph, pa)
        r = float(rho[i])
        mat[0, 0] *= 1.0 - lam[i] * mu[i] * r
        mat[0, 1] *= 1.0 + lam[i] * r
        mat[1, 0] *= 1.0 + mu[i] * r
        mat[1, 1] *= 1.0 - r
        mat = np.clip(mat, 0.0, None)
        # outcomes
        ih, ia = np.indices(mat.shape)
        s = mat.sum()
        if s <= 0:
            p_h[i] = p_d[i] = p_a[i] = np.nan
            continue
        p_h[i] = mat[ih > ia].sum() / s
        p_d[i] = mat[ih == ia].sum() / s
        p_a[i] = mat[ih < ia].sum() / s
    return p_h, p_d, p_a


def _predict_from_samples(
    samples: np.ndarray,
    n_teams: int,
    home_idx: int,
    away_idx: int,
    *,
    max_goals: int = MAX_GOALS,
) -> dict[str, float]:
    n_s = len(samples)
    lam = np.empty(n_s)
    mu = np.empty(n_s)
    rho = np.empty(n_s)
    for i, x in enumerate(samples):
        attack, defence, intercept, home_adv, r = _unpack_params(x, n_teams)
        r = float(np.clip(r, RHO_BOUNDS[0] + 1e-4, RHO_BOUNDS[1] - 1e-4))
        lam[i], mu[i] = match_lambdas(
            attack, defence, intercept, home_adv, home_idx, away_idx
        )
        rho[i] = r
    p_h, p_d, p_a = _outcome_probs_many(lam, mu, rho, max_goals=max_goals)
    return {
        "p_home": float(np.nanmean(p_h)),
        "p_draw": float(np.nanmean(p_d)),
        "p_away": float(np.nanmean(p_a)),
        "lam": float(np.mean(lam)),
        "mu": float(np.mean(mu)),
        "p_home_lo": float(np.nanquantile(p_h, 0.1)),
        "p_home_hi": float(np.nanquantile(p_h, 0.9)),
    }


def fit_bayesian_dc(
    homes: list[str] | np.ndarray,
    aways: list[str] | np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    dates: np.ndarray,
    *,
    xi: float = DEFAULT_XI,
    sigma_att: float = DEFAULT_SIGMA_ATT,
    sigma_def: float = DEFAULT_SIGMA_DEF,
    mu_intercept: float = DEFAULT_MU_INTERCEPT,
    sigma_intercept: float = DEFAULT_SIGMA_INTERCEPT,
    mu_home: float = DEFAULT_MU_HOME,
    sigma_home: float = DEFAULT_SIGMA_HOME,
    mu_rho: float = DEFAULT_MU_RHO,
    sigma_rho: float = DEFAULT_SIGMA_RHO,
    n_posterior: int = DEFAULT_N_POSTERIOR,
    x0: np.ndarray | None = None,
    teams: list[str] | None = None,
    random_state: int = 42,
    compute_laplace: bool = True,
) -> BayesianDCModel:
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

    if x0 is not None and len(x0) != n_free_params(n):
        x0 = None
    if x0 is None:
        x0 = np.zeros(n_free_params(n))
        x0[-3] = mu_intercept
        x0[-2] = mu_home
        x0[-1] = mu_rho

    bounds = [(-3.0, 3.0)] * (2 * (n - 1)) + [(-1.0, 1.5), (-1.0, 1.5), RHO_BOUNDS]

    def objective(x: np.ndarray) -> float:
        return _neg_log_posterior(
            x,
            home_idx,
            away_idx,
            hg,
            ag,
            weights,
            n,
            sigma_att,
            sigma_def,
            mu_intercept,
            sigma_intercept,
            mu_home,
            sigma_home,
            mu_rho,
            sigma_rho,
        )

    res = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 400, "ftol": 1e-9},
    )
    map_x = res.x
    attack, defence, intercept, home_adv, rho = _unpack_params(map_x, n)

    samples = None
    cov = None
    if compute_laplace and n_posterior > 0:
        hess = _numerical_hessian(objective, map_x, eps=1e-4)
        # Stabilize: add jitter, project to PSD
        hess = 0.5 * (hess + hess.T)
        jitter = 1e-5 * np.eye(len(map_x))
        try:
            cov = np.linalg.inv(hess + jitter)
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(hess + jitter)
        # Ensure PSD for sampling
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-8, None)
        cov = (eigvecs * eigvals) @ eigvecs.T

        rng = np.random.default_rng(random_state)
        raw = rng.multivariate_normal(map_x, cov, size=n_posterior)
        # Reject / clip invalid rho draws
        samples_list = []
        for draw in raw:
            d = draw.copy()
            d[-1] = float(np.clip(d[-1], RHO_BOUNDS[0] + 1e-4, RHO_BOUNDS[1] - 1e-4))
            # Reject extreme attack/defence
            if np.any(np.abs(d[:-3]) > 4.0):
                continue
            if objective(d) >= 1e11:
                continue
            samples_list.append(d)
        if len(samples_list) < max(20, n_posterior // 5):
            # Fall back: keep clipped raw draws
            samples = raw.copy()
            samples[:, -1] = np.clip(
                samples[:, -1], RHO_BOUNDS[0] + 1e-4, RHO_BOUNDS[1] - 1e-4
            )
        else:
            samples = np.asarray(samples_list)

    return BayesianDCModel(
        teams=teams,
        attack=attack,
        defence=defence,
        home_adv=float(home_adv),
        rho=float(rho),
        xi=xi,
        sigma_att=sigma_att,
        sigma_def=sigma_def,
        posterior_samples=samples,
        map_x=map_x,
        cov=cov,
        intercept=float(intercept),
    )


def model_to_x0(model: BayesianDCModel, teams: list[str]) -> np.ndarray:
    n = len(teams)
    attack = np.zeros(n)
    defence = np.zeros(n)
    old = model.team_to_idx
    for i, t in enumerate(teams):
        if t in old:
            attack[i] = model.attack[old[t]]
            defence[i] = model.defence[old[t]]
    attack = attack - attack.mean()
    defence = defence - defence.mean()
    intercept, home_adv = intercept_home_from_model(model)
    return _pack_params(attack, defence, intercept, home_adv, model.rho)


def edge_posterior_prob(
    model: BayesianDCModel,
    home: str,
    away: str,
    p_home_fair: float,
    p_draw_fair: float,
    p_away_fair: float,
    side: str,
    *,
    edge_threshold: float = 0.0,
) -> float:
    """
    Fraction of posterior draws where model edge on `side` ≥ edge_threshold.
    Useful selective-betting filter.
    """
    if model.posterior_samples is None or home not in model.team_to_idx or away not in model.team_to_idx:
        pred = model.predict_match(home, away, use_posterior=False)
        edge = {
            "H": pred["p_home"] - p_home_fair,
            "D": pred["p_draw"] - p_draw_fair,
            "A": pred["p_away"] - p_away_fair,
        }[side]
        return 1.0 if edge >= edge_threshold else 0.0

    hi = model.team_to_idx[home]
    ai = model.team_to_idx[away]
    fair = {"H": p_home_fair, "D": p_draw_fair, "A": p_away_fair}[side]
    hits = 0
    n = 0
    for x in model.posterior_samples:
        attack, defence, intercept, home_adv, rho = _unpack_params(x, len(model.teams))
        rho = float(np.clip(rho, RHO_BOUNDS[0] + 1e-4, RHO_BOUNDS[1] - 1e-4))
        lam, mu = match_lambdas(attack, defence, intercept, home_adv, hi, ai)
        p_h, p_d, p_a = outcome_probs(lam, mu, rho)
        p = {"H": p_h, "D": p_d, "A": p_a}[side]
        if (p - fair) >= edge_threshold:
            hits += 1
        n += 1
    return hits / n if n else 0.0
