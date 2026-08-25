"""
Fit Dixon-Coles Poisson model and evaluate vs Liga MX closing market.

Default backtest: fit on all history before each test matchday (expanding
window, warm-started), predict 1X2, map to favorite-win probability, then
simulate edge bets like the Elo pipeline.

Usage:
  python scripts/fit_dixon_coles.py
  python scripts/fit_dixon_coles.py --static          # fit once on train only
  python scripts/fit_dixon_coles.py --xi 0.003
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from dixon_coles_utils import (
    DEFAULT_XI,
    DixonColesModel,
    fit_dixon_coles,
    model_to_x0,
)
from eval_vs_market import pick_bets, roi_row, summarize_probs
from model_utils import TEST_SEASONS

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

MATCHES_CSV = DATA_DIR / "liga_mx_matches.csv"


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(MATCHES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
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
        "favorite_side",
        "fav_won",
        "dc_won",
        "p_fav_mkt",
        "p_dc_mkt",
        "fav_odds",
        "dc_odds",
    ]
    return df.dropna(subset=needed).reset_index(drop=True)


def fav_prob_from_1x2(p_home: float, p_away: float, favorite_side: str) -> float:
    if favorite_side == "H":
        return float(p_home)
    if favorite_side == "A":
        return float(p_away)
    raise ValueError(favorite_side)


def fit_on_history(
    hist: pd.DataFrame,
    *,
    xi: float,
    prev: DixonColesModel | None,
) -> DixonColesModel:
    teams = sorted(set(hist["home_name"]) | set(hist["away_name"]))
    x0 = model_to_x0(prev, teams) if prev is not None else None
    return fit_dixon_coles(
        hist["home_name"].to_numpy(),
        hist["away_name"].to_numpy(),
        hist["home_goals"].to_numpy(),
        hist["away_goals"].to_numpy(),
        hist["game_date"].to_numpy(),
        xi=xi,
        x0=x0,
        teams=teams,
    )


def score_rows(df: pd.DataFrame, model: DixonColesModel) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        pred = model.predict_match(str(row["home_name"]), str(row["away_name"]))
        p_fav = fav_prob_from_1x2(pred["p_home"], pred["p_away"], str(row["favorite_side"]))
        rows.append(
            {
                "game_id": int(row["game_id"]),
                "p_home_dc": pred["p_home"],
                "p_draw_dc": pred["p_draw"],
                "p_away_dc": pred["p_away"],
                "lam": pred["lam"],
                "mu": pred["mu"],
                "p_fav_model": p_fav,
            }
        )
    out = df.merge(pd.DataFrame(rows), on="game_id", how="left")
    out["p_dc_model"] = 1.0 - out["p_fav_model"]
    out["edge_fav"] = out["p_fav_model"] - out["p_fav_mkt"]
    out["edge_dc"] = out["p_dc_model"] - out["p_dc_mkt"]
    return out


def backtest_static(df: pd.DataFrame, train_seasons: set[str], test_seasons: set[str], xi: float):
    train = df.loc[df["season"].isin(train_seasons)].copy()
    test = df.loc[df["season"].isin(test_seasons)].copy()
    print(f"Static fit on {len(train)} train matches...")
    model = fit_on_history(train, xi=xi, prev=None)
    train_s = score_rows(train, model)
    test_s = score_rows(test, model)
    return model, train_s, test_s


def backtest_expanding(df: pd.DataFrame, train_seasons: set[str], test_seasons: set[str], xi: float):
    train = df.loc[df["season"].isin(train_seasons)].copy()
    test = df.loc[df["season"].isin(test_seasons)].copy()

    print(f"Initial fit on {len(train)} train matches...")
    model = fit_on_history(train, xi=xi, prev=None)
    train_s = score_rows(train, model)

    pred_parts = []
    test_dates = sorted(test["game_date"].unique())
    print(f"Expanding-window refit over {len(test_dates)} test matchdays...")
    for i, dt in enumerate(test_dates, start=1):
        hist = df.loc[df["game_date"] < dt]
        if len(hist) < 50:
            continue
        model = fit_on_history(hist, xi=xi, prev=model)
        day = test.loc[test["game_date"] == dt]
        pred_parts.append(score_rows(day, model))
        if i % 10 == 0 or i == len(test_dates):
            print(f"  {i}/{len(test_dates)} dates  (through {pd.Timestamp(dt).date()})")

    test_s = pd.concat(pred_parts, ignore_index=True)
    return model, train_s, test_s


def print_strengths(model: DixonColesModel, limit: int = 10) -> None:
    rows = []
    for i, team in enumerate(model.teams):
        rows.append(
            {
                "team": team,
                "attack": model.attack[i],
                "defence": model.defence[i],
                # Higher attack and higher defence (harder to score against) are better.
                "strength": model.attack[i] + model.defence[i],
            }
        )
    tab = pd.DataFrame(rows).sort_values("strength", ascending=False)
    print("\nTop teams by attack + defence:")
    print(tab.head(limit).to_string(index=False))
    return tab


def main() -> None:
    p = argparse.ArgumentParser(description="Fit Dixon-Coles and evaluate vs market.")
    p.add_argument("--xi", type=float, default=DEFAULT_XI, help="Time-decay rate (per day)")
    p.add_argument(
        "--static",
        action="store_true",
        help="Fit once on train seasons (no expanding refit on test).",
    )
    p.add_argument(
        "--edge",
        type=str,
        default="0,0.02,0.03,0.05,0.08,0.10",
        help="Comma-separated edge thresholds",
    )
    args = p.parse_args()
    edges = [float(x) for x in args.edge.split(",") if x.strip()]

    if not MATCHES_CSV.exists():
        raise SystemExit(f"Missing {MATCHES_CSV}. Run: python scripts/prepare_data.py")

    df = load_matches()
    all_seasons = sorted(df["season"].unique())
    test_seasons = set(TEST_SEASONS)
    train_seasons = set(all_seasons) - test_seasons
    print(f"Matches: {len(df)}")
    print(
        f"Train: {sorted(train_seasons)[0]} … {sorted(train_seasons)[-1]} "
        f"({len(train_seasons)} seasons)"
    )
    print(f"Test: {sorted(test_seasons)}")
    print(f"xi={args.xi}  mode={'static' if args.static else 'expanding'}")

    if args.static:
        model, train_s, test_s = backtest_static(df, train_seasons, test_seasons, args.xi)
    else:
        model, train_s, test_s = backtest_expanding(
            df, train_seasons, test_seasons, args.xi
        )

    print("\nProbability quality (favorite wins) — TRAIN (from final/init model):")
    summarize_probs("dc", train_s["fav_won"].astype(int), train_s["p_fav_model"])
    summarize_probs("market", train_s["fav_won"].astype(int), train_s["p_fav_mkt"])

    print("\nProbability quality (favorite wins) — TEST:")
    summarize_probs("dc", test_s["fav_won"].astype(int), test_s["p_fav_model"])
    summarize_probs("market", test_s["fav_won"].astype(int), test_s["p_fav_mkt"])

    y = test_s["fav_won"].astype(int)
    p = test_s["p_fav_model"]
    metrics = {
        "mode": "static" if args.static else "expanding",
        "xi": args.xi,
        "intercept": model.intercept,
        "home_adv": model.home_adv,
        "rho": model.rho,
        "n_train": int(len(train_s)),
        "n_test": int(len(test_s)),
        "test_acc": float(((p >= 0.5).astype(int) == y).mean()),
        "test_auc": float(roc_auc_score(y, p)),
        "test_logloss": float(log_loss(y, p)),
        "test_brier": float(brier_score_loss(y, p)),
        "market_auc": float(roc_auc_score(y, test_s["p_fav_mkt"])),
        "market_logloss": float(log_loss(y, test_s["p_fav_mkt"])),
    }

    rows = []
    print("\nFlat 1u edge bets on TEST (Dixon-Coles vs closing market):")
    print(f"  {'edge':>6}  {'bets':>5}  {'wins':>5}  {'hit%':>6}  {'P/L':>8}  {'ROI':>7}  fav/dc")
    for e in edges:
        scored = pick_bets(test_s, e)
        row = roi_row(scored, e)
        rows.append(row)
        hit = f"{100 * row['hit_rate']:.1f}%" if row["n_bets"] else "n/a"
        roi = f"{100 * row['roi']:+.1f}%" if row["n_bets"] else "n/a"
        print(
            f"  {e:6.2f}  {row['n_bets']:5d}  {row['n_wins']:5d}  {hit:>6}  "
            f"{row['profit']:+8.1f}  {roi:>7}  {row['n_fav']}/{row['n_dc']}"
        )
        metrics[f"roi_edge_{e:g}"] = row["roi"]
        metrics[f"profit_edge_{e:g}"] = row["profit"]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, MODELS_DIR / "dixon_coles_model.pkl")
    with open(MODELS_DIR / "dixon_coles_params.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    strengths = print_strengths(model)
    strengths.to_csv(TABLES_DIR / "dixon_coles_team_strengths.csv", index=False)
    pd.DataFrame(rows).to_csv(
        TABLES_DIR / "dixon_coles_market_edge_roi.csv", index=False, float_format="%.6f"
    )

    keep = [
        "game_id",
        "game_date",
        "season",
        "home_name",
        "away_name",
        "favorite_side",
        "fav_won",
        "lam",
        "mu",
        "p_home_dc",
        "p_draw_dc",
        "p_away_dc",
        "p_fav_mkt",
        "p_fav_model",
        "edge_fav",
        "edge_dc",
        "fav_odds",
        "dc_odds",
    ]
    test_s[keep].to_csv(TABLES_DIR / "dixon_coles_predictions_holdout.csv", index=False)
    print(f"\nWrote {TABLES_DIR / 'dixon_coles_predictions_holdout.csv'}")
    print(f"Wrote {TABLES_DIR / 'dixon_coles_market_edge_roi.csv'}")
    print(f"Saved model to {MODELS_DIR / 'dixon_coles_model.pkl'}")
    print(
        f"\nHoldout log-loss: Dixon-Coles {metrics['test_logloss']:.4f} vs "
        f"market {metrics['market_logloss']:.4f}"
    )


if __name__ == "__main__":
    main()
