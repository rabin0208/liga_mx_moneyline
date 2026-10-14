"""
Fit Bayesian hierarchical Dixon-Coles and evaluate vs Liga MX closing 1X2 market.

Default: MAP + Laplace posterior on all seasons before TEST_SEASONS, then
score the holdout with posterior-predictive P(H/D/A). Flat edge bets use the
same rule as the MLS Bayesian DC pipeline (best side among H/D/A).

Usage:
  python scripts/fit_bayesian_dc.py
  python scripts/fit_bayesian_dc.py --test-from 2026-01-01   # train through 2025
  python scripts/fit_bayesian_dc.py --map-only --test-from 2026-01-01 --expanding
  python scripts/fit_bayesian_dc.py --n-posterior 150 --sigma-att 0.30
  python scripts/fit_bayesian_dc.py --map-only          # skip Laplace
  python scripts/fit_bayesian_dc.py --conf 0.7          # require P(edge>0)≥0.7
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from bayesian_dc_utils import (
    DEFAULT_N_POSTERIOR,
    DEFAULT_SIGMA_ATT,
    DEFAULT_SIGMA_DEF,
    DEFAULT_XI,
    BayesianDCModel,
    _outcome_probs_many,
    fit_bayesian_dc,
    model_to_x0,
)
from dixon_coles_utils import (
    DEFAULT_XI as DC_XI,
    RHO_BOUNDS,
    match_lambdas,
    _unpack_params,
)
from model_utils import TEST_SEASONS, train_test_season_sets
from odds_utils import flat_bet_profit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

MATCHES_CSV = DATA_DIR / "liga_mx_matches.csv"
CLASS_ORDER = ["H", "D", "A"]
ODDS_COL = {"H": "AvgCH", "D": "AvgCD", "A": "AvgCA"}
FAIR_COL = {"H": "p_home_fair", "D": "p_draw_fair", "A": "p_away_fair"}
DEFAULT_TAG = "_".join(sorted(TEST_SEASONS))


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(MATCHES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["season"] = df["season"].astype(str)
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    needed = [
        "game_id",
        "game_date",
        "season",
        "home_name",
        "away_name",
        "home_goals",
        "away_goals",
        "Res",
        "AvgCH",
        "AvgCD",
        "AvgCA",
        "p_home_fair",
        "p_draw_fair",
        "p_away_fair",
    ]
    return df.dropna(subset=needed).reset_index(drop=True)


def fit_on_history(
    hist: pd.DataFrame,
    *,
    xi: float,
    sigma_att: float,
    sigma_def: float,
    n_posterior: int,
    compute_laplace: bool,
    prev: BayesianDCModel | None,
) -> BayesianDCModel:
    teams = sorted(set(hist["home_name"]) | set(hist["away_name"]))
    x0 = model_to_x0(prev, teams) if prev is not None else None
    return fit_bayesian_dc(
        hist["home_name"].to_numpy(),
        hist["away_name"].to_numpy(),
        hist["home_goals"].to_numpy(),
        hist["away_goals"].to_numpy(),
        hist["game_date"].to_numpy(),
        xi=xi,
        sigma_att=sigma_att,
        sigma_def=sigma_def,
        n_posterior=n_posterior,
        x0=x0,
        teams=teams,
        compute_laplace=compute_laplace,
    )


def _posterior_side_probs(
    model: BayesianDCModel,
    home: str,
    away: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Return per-draw (p_H, p_D, p_A) arrays, or None if unavailable."""
    if model.posterior_samples is None:
        return None
    if home not in model.team_to_idx or away not in model.team_to_idx:
        return None
    hi = model.team_to_idx[home]
    ai = model.team_to_idx[away]
    n_s = len(model.posterior_samples)
    lam = np.empty(n_s)
    mu = np.empty(n_s)
    rho = np.empty(n_s)
    for i, x in enumerate(model.posterior_samples):
        attack, defence, intercept, home_adv, r = _unpack_params(x, len(model.teams))
        r = float(np.clip(r, RHO_BOUNDS[0] + 1e-4, RHO_BOUNDS[1] - 1e-4))
        lam[i], mu[i] = match_lambdas(attack, defence, intercept, home_adv, hi, ai)
        rho[i] = r
    return _outcome_probs_many(lam, mu, rho)


def score_rows(
    df: pd.DataFrame,
    model: BayesianDCModel,
    *,
    use_posterior: bool,
    conf: float,
    conf_edge: float,
) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        home = str(row["home_name"])
        away = str(row["away_name"])
        need_conf = conf > 0 and use_posterior and model.posterior_samples is not None
        side_probs = _posterior_side_probs(model, home, away) if need_conf else None

        if side_probs is not None:
            p_h_s, p_d_s, p_a_s = side_probs
            pred = {
                "p_home": float(np.nanmean(p_h_s)),
                "p_draw": float(np.nanmean(p_d_s)),
                "p_away": float(np.nanmean(p_a_s)),
                "lam": float("nan"),
                "mu": float("nan"),
                "p_home_lo": float(np.nanquantile(p_h_s, 0.1)),
                "p_home_hi": float(np.nanquantile(p_h_s, 0.9)),
            }
            map_pred = model.predict_match(home, away, use_posterior=False)
            pred["lam"] = map_pred["lam"]
            pred["mu"] = map_pred["mu"]
        else:
            pred = model.predict_match(home, away, use_posterior=use_posterior)

        rec = {
            "game_id": int(row["game_id"]),
            "p_home_model": pred["p_home"],
            "p_draw_model": pred["p_draw"],
            "p_away_model": pred["p_away"],
            "lam": pred["lam"],
            "mu": pred["mu"],
            "p_home_lo": pred["p_home_lo"],
            "p_home_hi": pred["p_home_hi"],
        }
        if side_probs is not None:
            p_h_s, p_d_s, p_a_s = side_probs
            fair = {
                "H": float(row["p_home_fair"]),
                "D": float(row["p_draw_fair"]),
                "A": float(row["p_away_fair"]),
            }
            draws = {"H": p_h_s, "D": p_d_s, "A": p_a_s}
            for side in CLASS_ORDER:
                rec[f"p_edge_pos_{side}"] = float(
                    np.mean(draws[side] - fair[side] >= conf_edge)
                )
        else:
            for side in CLASS_ORDER:
                rec[f"p_edge_pos_{side}"] = 1.0
        rows.append(rec)

    out = df.merge(pd.DataFrame(rows), on="game_id", how="left")
    out["model_pick"] = out[
        ["p_home_model", "p_draw_model", "p_away_model"]
    ].idxmax(axis=1).map(
        {
            "p_home_model": "H",
            "p_draw_model": "D",
            "p_away_model": "A",
        }
    )
    out["mkt_pick"] = out[
        ["p_home_fair", "p_draw_fair", "p_away_fair"]
    ].idxmax(axis=1).map(
        {
            "p_home_fair": "H",
            "p_draw_fair": "D",
            "p_away_fair": "A",
        }
    )
    for side in CLASS_ORDER:
        out[f"edge_{side}"] = out[f"p_{ {'H':'home','D':'draw','A':'away'}[side] }_model"] - out[
            FAIR_COL[side]
        ]
    return out


def pick_bets(
    df: pd.DataFrame,
    edge: float,
    sides: list[str],
    *,
    conf: float,
    odds_col: dict[str, str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    price = odds_col or ODDS_COL
    bet_side: list[str | None] = []
    bet_odds: list[float | None] = []
    bet_won: list[bool | None] = []
    bet_profit: list[float] = []
    bet_edge: list[float | None] = []

    for _, row in out.iterrows():
        best_side = None
        best_edge = -1e9
        for side in sides:
            e = float(row[f"edge_{side}"])
            p_pos = float(row.get(f"p_edge_pos_{side}", 1.0))
            if conf > 0 and p_pos < conf:
                continue
            if e > best_edge:
                best_edge = e
                best_side = side
        if best_side is None or best_edge < edge:
            bet_side.append(None)
            bet_odds.append(None)
            bet_won.append(None)
            bet_profit.append(0.0)
            bet_edge.append(None)
            continue
        odds = float(row[price[best_side]])
        won = str(row["Res"]) == best_side
        bet_side.append(best_side)
        bet_odds.append(odds)
        bet_won.append(won)
        bet_profit.append(flat_bet_profit(odds, won))
        bet_edge.append(best_edge)

    out["bet_side"] = bet_side
    out["bet_odds"] = bet_odds
    out["bet_won"] = bet_won
    out["bet_profit"] = bet_profit
    out["bet_edge"] = bet_edge
    return out


def roi_row(bets: pd.DataFrame, edge: float, sides_label: str) -> dict:
    placed = bets.dropna(subset=["bet_side"])
    if placed.empty:
        return {
            "sides": sides_label,
            "edge": edge,
            "n_bets": 0,
            "n_wins": 0,
            "hit_rate": float("nan"),
            "profit": 0.0,
            "roi": float("nan"),
            "n_H": 0,
            "n_D": 0,
            "n_A": 0,
            "avg_odds": float("nan"),
        }
    return {
        "sides": sides_label,
        "edge": edge,
        "n_bets": int(len(placed)),
        "n_wins": int(placed["bet_won"].sum()),
        "hit_rate": float(placed["bet_won"].mean()),
        "profit": float(placed["bet_profit"].sum()),
        "roi": float(placed["bet_profit"].sum() / len(placed)),
        "n_H": int((placed["bet_side"] == "H").sum()),
        "n_D": int((placed["bet_side"] == "D").sum()),
        "n_A": int((placed["bet_side"] == "A").sum()),
        "avg_odds": float(placed["bet_odds"].mean()),
    }


def summarize_1x2(df: pd.DataFrame) -> None:
    y = df["Res"]
    y_idx = y.map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    p_model = df[["p_home_model", "p_draw_model", "p_away_model"]].to_numpy()
    p_mkt = df[["p_home_fair", "p_draw_fair", "p_away_fair"]].to_numpy()
    print("\nProbability quality (1X2):")
    print(
        f"  {'bayes_dc':12s}  acc={accuracy_score(y, df['model_pick']):.4f}  "
        f"logloss={log_loss(y_idx, p_model, labels=[0, 1, 2]):.4f}"
    )
    print(
        f"  {'market':12s}  acc={accuracy_score(y, df['mkt_pick']):.4f}  "
        f"logloss={log_loss(y_idx, p_mkt, labels=[0, 1, 2]):.4f}"
    )


def print_strengths(model: BayesianDCModel, limit: int = 10) -> pd.DataFrame:
    rows = []
    for i, team in enumerate(model.teams):
        rows.append(
            {
                "team": team,
                "attack": model.attack[i],
                "defence": model.defence[i],
                "strength": model.attack[i] + model.defence[i],
            }
        )
    tab = pd.DataFrame(rows).sort_values("strength", ascending=False)
    print("\nTop teams by attack + defence (MAP):")
    print(tab.head(limit).to_string(index=False))
    return tab


def output_suffix(
    test_seasons: set[str] | list[str],
    *,
    test_from: str | None = None,
    expanding: bool = False,
) -> str:
    if test_from:
        suffix = f"_{pd.Timestamp(test_from).year}"
    else:
        tag = "_".join(sorted(test_seasons))
        suffix = "" if tag == DEFAULT_TAG else "_" + tag.replace("/", "-")
    if expanding:
        suffix = f"{suffix}_expanding" if suffix else "_expanding"
    return suffix


def score_expanding(
    df: pd.DataFrame,
    test: pd.DataFrame,
    *,
    xi: float,
    sigma_att: float,
    sigma_def: float,
    n_posterior: int,
    compute_laplace: bool,
    use_posterior: bool,
    conf: float,
    conf_edge: float,
) -> tuple[BayesianDCModel, pd.DataFrame]:
    """Refit on all games strictly before each test matchday, then score that day."""
    test_dates = sorted(test["game_date"].unique())
    print(
        f"Expanding-window MAP refit over {len(test_dates)} matchdays "
        f"({len(test)} games)..."
    )
    model: BayesianDCModel | None = None
    parts: list[pd.DataFrame] = []
    t0 = time.time()
    for i, dt in enumerate(test_dates, start=1):
        hist = df.loc[df["game_date"] < dt]
        if len(hist) < 50:
            continue
        model = fit_on_history(
            hist,
            xi=xi,
            sigma_att=sigma_att,
            sigma_def=sigma_def,
            n_posterior=n_posterior,
            compute_laplace=compute_laplace,
            prev=model,
        )
        day = test.loc[test["game_date"] == dt]
        parts.append(
            score_rows(
                day,
                model,
                use_posterior=use_posterior,
                conf=conf,
                conf_edge=conf_edge,
            )
        )
        if i % 10 == 0 or i == len(test_dates):
            print(
                f"  {i}/{len(test_dates)}  {pd.Timestamp(dt).date()}  "
                f"hist={len(hist)}  {time.time() - t0:.1f}s"
            )
    if model is None or not parts:
        raise SystemExit("Expanding window produced no scored games.")
    return model, pd.concat(parts, ignore_index=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Bayesian Dixon-Coles vs Liga MX market.")
    p.add_argument("--xi", type=float, default=DEFAULT_XI, help="Time-decay per day")
    p.add_argument("--sigma-att", type=float, default=DEFAULT_SIGMA_ATT)
    p.add_argument("--sigma-def", type=float, default=DEFAULT_SIGMA_DEF)
    p.add_argument("--n-posterior", type=int, default=DEFAULT_N_POSTERIOR)
    p.add_argument(
        "--map-only",
        action="store_true",
        help="Use MAP predictions only (skip Laplace posterior).",
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.0,
        help="Min posterior P(edge ≥ conf-edge) to allow a bet (0=off).",
    )
    p.add_argument(
        "--conf-edge",
        type=float,
        default=0.0,
        help="Edge threshold inside the posterior confidence filter.",
    )
    p.add_argument(
        "--edge",
        type=str,
        default="0,0.02,0.03,0.05,0.08,0.10",
        help="Comma-separated mean-edge thresholds",
    )
    p.add_argument("--sides", type=str, default="H,D,A")
    p.add_argument(
        "--test-seasons",
        type=str,
        default=",".join(TEST_SEASONS),
        help="Comma-separated holdout seasons (train = all earlier seasons).",
    )
    p.add_argument(
        "--test-from",
        type=str,
        default="",
        help="If set (YYYY-MM-DD), train on games before this date and test on/after it.",
    )
    p.add_argument(
        "--expanding",
        action="store_true",
        help=(
            "Refit on all history strictly before each test matchday "
            "(same-day games share a fit). Use with --map-only."
        ),
    )
    args = p.parse_args()

    edges = [float(x) for x in args.edge.split(",") if x.strip()]
    sides = [s.strip().upper() for s in args.sides.split(",") if s.strip()]
    sides_label = ",".join(sides)
    use_posterior = not args.map_only
    compute_laplace = use_posterior
    test_season_list = [s.strip() for s in args.test_seasons.split(",") if s.strip()]

    if not MATCHES_CSV.exists():
        raise SystemExit(f"Missing {MATCHES_CSV}. Run: python scripts/prepare_data.py")

    df = load_matches()
    test_from = args.test_from.strip() or None
    if test_from:
        cut = pd.Timestamp(test_from)
        train = df.loc[df["game_date"] < cut].copy()
        test = df.loc[df["game_date"] >= cut].copy()
        train_seasons = sorted(train["season"].astype(str).unique())
        test_seasons = sorted(test["season"].astype(str).unique())
        print(f"Matches: {len(df)}")
        print(
            f"Train: through {cut.date()} "
            f"({sorted(train_seasons)[0]} … {sorted(train_seasons)[-1]}, "
            f"{len(train)} games)"
        )
        print(
            f"Test: from {cut.date()} "
            f"({sorted(test_seasons)}, {len(test)} games)"
        )
    else:
        all_seasons = sorted(df["season"].astype(str).unique())
        train_seasons, test_seasons = train_test_season_sets(
            all_seasons, test_seasons=test_season_list
        )
        train = df.loc[df["season"].isin(train_seasons)].copy()
        test = df.loc[df["season"].isin(test_seasons)].copy()
        print(f"Matches: {len(df)}")
        print(
            f"Train: {sorted(train_seasons)[0]} … {sorted(train_seasons)[-1]} "
            f"({len(train)} games)"
        )
        print(f"Test: {sorted(test_seasons)} ({len(test)} games)")
    print(
        f"xi={args.xi}  σ_att={args.sigma_att}  σ_def={args.sigma_def}  "
        f"posterior={'MAP-only' if args.map_only else f'Laplace n={args.n_posterior}'}  "
        f"conf={args.conf}  expanding={args.expanding}"
    )
    if args.expanding and not args.map_only:
        print(
            "  Warning: expanding + Laplace is slow (Hessian every matchday). "
            "Prefer --map-only."
        )

    t0 = time.time()
    if args.expanding:
        print(
            f"\nInitial history before first test matchday: {len(train)} games"
        )
        model, test_s = score_expanding(
            df,
            test,
            xi=args.xi,
            sigma_att=args.sigma_att,
            sigma_def=args.sigma_def,
            n_posterior=args.n_posterior,
            compute_laplace=compute_laplace,
            use_posterior=use_posterior,
            conf=args.conf,
            conf_edge=args.conf_edge,
        )
        print(
            f"  Done in {time.time() - t0:.1f}s  "
            f"final intercept={model.intercept:.3f}  "
            f"home_adv={model.home_adv:.3f}  rho={model.rho:.3f}  "
            f"teams={len(model.teams)}  scored={len(test_s)}"
        )
    else:
        print(f"\nFitting Bayesian DC on {len(train)} train matches...")
        model = fit_on_history(
            train,
            xi=args.xi,
            sigma_att=args.sigma_att,
            sigma_def=args.sigma_def,
            n_posterior=args.n_posterior,
            compute_laplace=compute_laplace,
            prev=None,
        )
        print(
            f"  Done in {time.time() - t0:.1f}s  "
            f"intercept={model.intercept:.3f}  home_adv={model.home_adv:.3f}  "
            f"rho={model.rho:.3f}  teams={len(model.teams)}"
        )
        if model.posterior_samples is not None:
            print(f"  Posterior draws kept: {len(model.posterior_samples)}")

        print("\nScoring holdout...")
        t1 = time.time()
        test_s = score_rows(
            test,
            model,
            use_posterior=use_posterior,
            conf=args.conf,
            conf_edge=args.conf_edge,
        )
        print(f"  Scored {len(test_s)} games in {time.time() - t1:.1f}s")
    summarize_1x2(test_s)
    print(
        f"  Mean P(H/D/A) model="
        f"{test_s['p_home_model'].mean():.3f}/"
        f"{test_s['p_draw_model'].mean():.3f}/"
        f"{test_s['p_away_model'].mean():.3f}  "
        f"market="
        f"{test_s['p_home_fair'].mean():.3f}/"
        f"{test_s['p_draw_fair'].mean():.3f}/"
        f"{test_s['p_away_fair'].mean():.3f}"
    )
    print(
        f"  Mean λ/μ={test_s['lam'].mean():.2f}/{test_s['mu'].mean():.2f}  "
        f"model_pick H/D/A="
        f"{(test_s['model_pick']=='H').sum()}/"
        f"{(test_s['model_pick']=='D').sum()}/"
        f"{(test_s['model_pick']=='A').sum()}  "
        f"market H/D/A="
        f"{(test_s['mkt_pick']=='H').sum()}/"
        f"{(test_s['mkt_pick']=='D').sum()}/"
        f"{(test_s['mkt_pick']=='A').sum()}"
    )

    rows = []
    print(
        f"\nFlat 1u edge bets (best side among [{sides_label}], "
        f"payout closing Avg*, conf≥{args.conf:g}):"
    )
    print(
        f"  {'edge':>6}  {'bets':>5}  {'wins':>5}  {'hit%':>6}  "
        f"{'P/L':>8}  {'ROI':>7}  H/D/A  avg_odds"
    )
    for e in edges:
        scored = pick_bets(test_s, e, sides, conf=args.conf)
        row = roi_row(scored, e, sides_label)
        rows.append(row)
        if row["n_bets"] == 0:
            print(f"  {e:6.2f}      0      0     n/a     +0.0      n/a  0/0/0      n/a")
            continue
        print(
            f"  {e:6.2f}  {row['n_bets']:5d}  {row['n_wins']:5d}  "
            f"{100 * row['hit_rate']:5.1f}%  {row['profit']:+8.1f}  "
            f"{100 * row['roi']:+6.1f}%  "
            f"{row['n_H']}/{row['n_D']}/{row['n_A']}  {row['avg_odds']:.3f}"
        )

    print(f"\nFlat 1u edge bets (best side among [H,A], conf≥{args.conf:g}):")
    print(
        f"  {'edge':>6}  {'bets':>5}  {'wins':>5}  {'hit%':>6}  "
        f"{'P/L':>8}  {'ROI':>7}  H/D/A  avg_odds"
    )
    for e in edges:
        scored = pick_bets(test_s, e, ["H", "A"], conf=args.conf)
        row = roi_row(scored, e, "H,A")
        rows.append(row)
        if row["n_bets"] == 0:
            print(f"  {e:6.2f}      0      0     n/a     +0.0      n/a  0/0/0      n/a")
            continue
        print(
            f"  {e:6.2f}  {row['n_bets']:5d}  {row['n_wins']:5d}  "
            f"{100 * row['hit_rate']:5.1f}%  {row['profit']:+8.1f}  "
            f"{100 * row['roi']:+6.1f}%  "
            f"{row['n_H']}/{row['n_D']}/{row['n_A']}  {row['avg_odds']:.3f}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    suffix = output_suffix(
        test_seasons, test_from=test_from, expanding=args.expanding
    )

    strengths = print_strengths(model)
    strengths.to_csv(TABLES_DIR / f"bayesian_dc_team_strengths{suffix}.csv", index=False)

    joblib.dump(model, MODELS_DIR / f"bayesian_dc_model{suffix}.pkl")

    y_idx = test_s["Res"].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    p_model = test_s[["p_home_model", "p_draw_model", "p_away_model"]].to_numpy()
    p_mkt = test_s[["p_home_fair", "p_draw_fair", "p_away_fair"]].to_numpy()
    metrics = {
        "test_seasons": sorted(test_seasons),
        "test_from": test_from,
        "expanding": args.expanding,
        "xi": args.xi,
        "sigma_att": args.sigma_att,
        "sigma_def": args.sigma_def,
        "n_posterior": 0 if args.map_only else args.n_posterior,
        "conf": args.conf,
        "intercept_map": model.intercept,
        "home_adv_map": model.home_adv,
        "rho_map": model.rho,
        "n_train": int(len(train)),
        "n_test": int(len(test_s)),
        "test_logloss": float(log_loss(y_idx, p_model, labels=[0, 1, 2])),
        "market_logloss": float(log_loss(y_idx, p_mkt, labels=[0, 1, 2])),
        "test_acc": float(accuracy_score(test_s["Res"], test_s["model_pick"])),
        "market_acc": float(accuracy_score(test_s["Res"], test_s["mkt_pick"])),
        "dc_xi_ref": DC_XI,
    }
    with open(MODELS_DIR / f"bayesian_dc_params{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    roi_path = TABLES_DIR / f"bayesian_dc_edge_roi{suffix}.csv"
    pd.DataFrame(rows).to_csv(roi_path, index=False, float_format="%.6f")

    scored0 = pick_bets(test_s, 0.0, sides, conf=args.conf)
    keep = [
        "game_id",
        "game_date",
        "season",
        "home_name",
        "away_name",
        "Res",
        "p_home_fair",
        "p_draw_fair",
        "p_away_fair",
        "p_home_model",
        "p_draw_model",
        "p_away_model",
        "p_home_lo",
        "p_home_hi",
        "lam",
        "mu",
        "edge_H",
        "edge_D",
        "edge_A",
        "p_edge_pos_H",
        "p_edge_pos_D",
        "p_edge_pos_A",
        "model_pick",
        "mkt_pick",
        "bet_side",
        "bet_odds",
        "bet_edge",
        "bet_won",
        "bet_profit",
    ]
    pred_path = TABLES_DIR / f"bayesian_dc_predictions_holdout{suffix}.csv"
    scored0[keep].to_csv(pred_path, index=False)

    print(f"\nWrote {pred_path}")
    print(f"Wrote {roi_path}")
    print(f"Saved model to {MODELS_DIR / f'bayesian_dc_model{suffix}.pkl'}")
    print(
        f"\nHoldout log-loss: Bayesian DC {metrics['test_logloss']:.4f} vs "
        f"market {metrics['market_logloss']:.4f}"
    )


if __name__ == "__main__":
    main()
