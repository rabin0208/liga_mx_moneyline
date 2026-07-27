"""
Compare model P(favorite wins) to closing market and simulate edge betting.

Bets when |model - market| exceeds a threshold:
  - model favors favorite → bet favorite at fav_odds
  - model favors DC → bet underdog double chance at dc_odds
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score

from model_utils import FEATURE_COLS, TEST_SEASONS
from odds_utils import flat_bet_profit

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

FEATURED_CSV = DATA_DIR / "liga_mx_featured.csv"
MODEL_PATH = MODELS_DIR / "logistic_regression.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


def score_games(seasons: list[str] | None) -> pd.DataFrame:
    df = pd.read_csv(FEATURED_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    if seasons is not None:
        df = df.loc[df["season"].isin(seasons)].copy()

    for c in ("home_rest_days", "away_rest_days", "week_of_year", "season_index"):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    df = df.dropna(subset=FEATURE_COLS + ["fav_won", "p_fav_mkt", "fav_odds", "dc_odds"])
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    X = scaler.transform(df[FEATURE_COLS].astype(float))
    df["p_fav_model"] = model.predict_proba(X)[:, 1]
    df["p_dc_model"] = 1.0 - df["p_fav_model"]
    df["edge_fav"] = df["p_fav_model"] - df["p_fav_mkt"]
    df["edge_dc"] = df["p_dc_model"] - df["p_dc_mkt"]
    return df.reset_index(drop=True)


def summarize_probs(name: str, y: pd.Series, p: pd.Series) -> None:
    pred = (p >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    try:
        auc = roc_auc_score(y, p)
        auc_s = f"{auc:.4f}"
    except ValueError:
        auc_s = "n/a"
    print(
        f"  {name:12s}  acc={acc:.4f}  auc={auc_s}  "
        f"logloss={log_loss(y, p):.4f}  brier={brier_score_loss(y, p):.4f}"
    )


def pick_bets(df: pd.DataFrame, edge: float) -> pd.DataFrame:
    out = df.copy()
    side: list[str | None] = []
    odds: list[float | None] = []
    won: list[bool | None] = []
    profit: list[float] = []

    for _, row in out.iterrows():
        if row["edge_fav"] >= edge and row["edge_fav"] >= row["edge_dc"]:
            s, o, w = "favorite", float(row["fav_odds"]), bool(row["fav_won"])
        elif row["edge_dc"] >= edge:
            s, o, w = "underdog_dc", float(row["dc_odds"]), bool(row["dc_won"])
        else:
            side.append(None)
            odds.append(None)
            won.append(None)
            profit.append(0.0)
            continue
        side.append(s)
        odds.append(o)
        won.append(w)
        profit.append(flat_bet_profit(o, w))

    out["bet_side"] = side
    out["bet_odds"] = odds
    out["bet_won"] = won
    out["bet_profit"] = profit
    return out


def roi_row(bets: pd.DataFrame, edge: float) -> dict:
    placed = bets.dropna(subset=["bet_side"])
    if placed.empty:
        return {
            "edge": edge,
            "n_bets": 0,
            "n_wins": 0,
            "hit_rate": float("nan"),
            "profit": 0.0,
            "roi": float("nan"),
            "n_fav": 0,
            "n_dc": 0,
        }
    return {
        "edge": edge,
        "n_bets": int(len(placed)),
        "n_wins": int(placed["bet_won"].sum()),
        "hit_rate": float(placed["bet_won"].mean()),
        "profit": float(placed["bet_profit"].sum()),
        "roi": float(placed["bet_profit"].sum() / len(placed)),
        "n_fav": int((placed["bet_side"] == "favorite").sum()),
        "n_dc": int((placed["bet_side"] == "underdog_dc").sum()),
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Model vs market edge evaluation.")
    p.add_argument(
        "--seasons",
        type=str,
        default=",".join(TEST_SEASONS),
        help="Comma-separated seasons to evaluate (default: holdout TEST_SEASONS).",
    )
    p.add_argument(
        "--edge",
        type=str,
        default="0,0.02,0.03,0.05,0.08",
        help="Comma-separated edge thresholds.",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Optional per-game predictions CSV.",
    )
    args = p.parse_args()
    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    edges = [float(x) for x in args.edge.split(",") if x.strip()]

    print(f"Scoring seasons {seasons}...")
    games = score_games(seasons)
    print(f"  Games: {len(games)}")

    y = games["fav_won"].astype(int)
    print("\nProbability quality (favorite wins):")
    summarize_probs("model", y, games["p_fav_model"])
    summarize_probs("market", y, games["p_fav_mkt"])

    rows = []
    print("\nFlat 1u edge bets (model vs closing market):")
    print(f"  {'edge':>6}  {'bets':>5}  {'wins':>5}  {'hit%':>6}  {'P/L':>8}  {'ROI':>7}  fav/dc")
    for e in edges:
        scored = pick_bets(games, e)
        row = roi_row(scored, e)
        rows.append(row)
        hit = f"{100 * row['hit_rate']:.1f}%" if row["n_bets"] else "n/a"
        roi = f"{100 * row['roi']:+.1f}%" if row["n_bets"] else "n/a"
        print(
            f"  {e:6.2f}  {row['n_bets']:5d}  {row['n_wins']:5d}  {hit:>6}  "
            f"{row['profit']:+8.1f}  {roi:>7}  {row['n_fav']}/{row['n_dc']}"
        )

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    roi_path = TABLES_DIR / "market_edge_roi.csv"
    pd.DataFrame(rows).to_csv(roi_path, index=False, float_format="%.6f")
    print(f"\nWrote {roi_path}")

    out_path = args.output or (TABLES_DIR / "predictions_holdout.csv")
    keep = [
        "game_id",
        "game_date",
        "season",
        "home_name",
        "away_name",
        "favorite_side",
        "fav_won",
        "p_fav_mkt",
        "p_fav_model",
        "edge_fav",
        "edge_dc",
        "fav_odds",
        "dc_odds",
    ]
    games[keep].to_csv(out_path, index=False)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
