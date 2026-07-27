"""
Fit and evaluate a Liga MX Elo model.

1. Grid-search K and home advantage on train seasons
2. Calibrate P(favorite wins) with logistic regression on elo_edge
3. Score holdout seasons vs closing market and simulate edge bets

Usage:
  python scripts/fit_elo_model.py
  python scripts/fit_elo_model.py --season-reset
  python scripts/fit_elo_model.py --compare-reset
  python scripts/fit_elo_model.py --k 20 --home-adv 80
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from elo_utils import (
    DEFAULT_HOME_ADV,
    DEFAULT_K,
    EloState,
    elo_edge_for_favorite,
)
from eval_vs_market import pick_bets, roi_row, summarize_probs
from model_utils import TEST_SEASONS

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

MATCHES_CSV = DATA_DIR / "liga_mx_matches.csv"

K_GRID = [10, 15, 20, 25, 30, 40]
HOME_ADV_GRID = [40, 60, 80, 100, 120]


def load_matches() -> pd.DataFrame:
    df = pd.read_csv(MATCHES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    needed = [
        "game_id",
        "game_date",
        "season",
        "home_id",
        "away_id",
        "home_name",
        "away_name",
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


def walk_elo(
    df: pd.DataFrame,
    k: float,
    home_adv: float,
    *,
    season_reset: bool = False,
) -> tuple[pd.DataFrame, EloState]:
    """Chronological Elo walk; rows get pre-match ratings / elo_edge."""
    state = EloState(k=k, home_adv=home_adv)
    rows: list[dict] = []
    prev_season: str | None = None

    for _, row in df.iterrows():
        season = str(row["season"])
        if season_reset and prev_season is not None and season != prev_season:
            state.reset()
        prev_season = season

        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        r_home, r_away, elo_diff, e_home = state.pre_match(home_id, away_id)
        fav_side = str(row["favorite_side"])
        elo_edge = elo_edge_for_favorite(elo_diff, fav_side)

        rows.append(
            {
                "game_id": int(row["game_id"]),
                "elo_home": r_home,
                "elo_away": r_away,
                "elo_diff": elo_diff,
                "elo_e_home": e_home,
                "elo_edge": elo_edge,
            }
        )
        state.update(home_id, away_id, str(row["Res"]))

    out = df.copy()
    elo_df = pd.DataFrame(rows)
    out = out.merge(elo_df, on="game_id", how="left")
    return out, state


def fit_calibrator(train: pd.DataFrame) -> LogisticRegression:
    X = train[["elo_edge"]].astype(float)
    y = train["fav_won"].astype(int)
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    return model


def score_calibrator(model: LogisticRegression, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    X = out[["elo_edge"]].astype(float)
    out["p_fav_model"] = model.predict_proba(X)[:, 1]
    out["p_dc_model"] = 1.0 - out["p_fav_model"]
    out["edge_fav"] = out["p_fav_model"] - out["p_fav_mkt"]
    out["edge_dc"] = out["p_dc_model"] - out["p_dc_mkt"]
    return out


def train_logloss(
    df: pd.DataFrame,
    k: float,
    home_adv: float,
    train_seasons: set[str],
    *,
    season_reset: bool,
) -> float:
    walked, _ = walk_elo(df, k=k, home_adv=home_adv, season_reset=season_reset)
    train = walked.loc[walked["season"].isin(train_seasons)]
    if train.empty:
        return float("inf")
    model = fit_calibrator(train)
    p = model.predict_proba(train[["elo_edge"]].astype(float))[:, 1]
    return float(log_loss(train["fav_won"].astype(int), p))


def grid_search(
    df: pd.DataFrame,
    train_seasons: set[str],
    *,
    season_reset: bool,
) -> tuple[float, float, float]:
    best = (float("inf"), DEFAULT_K, DEFAULT_HOME_ADV)
    mode = "season-reset" if season_reset else "continuous"
    print(f"Grid-searching K × home_adv on train log-loss ({mode})...")
    for k in K_GRID:
        for h in HOME_ADV_GRID:
            ll = train_logloss(
                df, k, h, train_seasons, season_reset=season_reset
            )
            print(f"  K={k:5.1f}  H={h:5.1f}  train_logloss={ll:.4f}")
            if ll < best[0]:
                best = (ll, float(k), float(h))
    print(f"Best: K={best[1]}, home_adv={best[2]} (train logloss={best[0]:.4f})")
    return best[1], best[2], best[0]


def evaluate_mode(
    df: pd.DataFrame,
    train_seasons: set[str],
    test_seasons: set[str],
    *,
    season_reset: bool,
    k: float | None,
    home_adv: float | None,
    edges: list[float],
) -> dict:
    label = "season_reset" if season_reset else "continuous"
    print(f"\n===== Elo mode: {label} =====")

    if k is not None and home_adv is not None:
        best_k, best_h = float(k), float(home_adv)
        print(f"Using fixed params: K={best_k}, home_adv={best_h}")
    else:
        best_k, best_h, _ = grid_search(
            df, train_seasons, season_reset=season_reset
        )

    walked, final_state = walk_elo(
        df, k=best_k, home_adv=best_h, season_reset=season_reset
    )
    train = walked.loc[walked["season"].isin(train_seasons)].copy()
    test = walked.loc[walked["season"].isin(test_seasons)].copy()
    if test.empty:
        raise SystemExit(f"No test rows for {TEST_SEASONS}")

    calibrator = fit_calibrator(train)
    train_s = score_calibrator(calibrator, train)
    test_s = score_calibrator(calibrator, test)

    print("\nProbability quality (favorite wins) — TRAIN:")
    summarize_probs("elo", train_s["fav_won"].astype(int), train_s["p_fav_model"])
    summarize_probs("market", train_s["fav_won"].astype(int), train_s["p_fav_mkt"])

    print("\nProbability quality (favorite wins) — TEST:")
    summarize_probs("elo", test_s["fav_won"].astype(int), test_s["p_fav_model"])
    summarize_probs("market", test_s["fav_won"].astype(int), test_s["p_fav_mkt"])

    y = test_s["fav_won"].astype(int)
    p = test_s["p_fav_model"]
    metrics = {
        "mode": label,
        "k": best_k,
        "home_adv": best_h,
        "test_logloss": float(log_loss(y, p)),
        "test_auc": float(roc_auc_score(y, p)),
        "test_brier": float(brier_score_loss(y, p)),
        "test_acc": float(((p >= 0.5).astype(int) == y).mean()),
        "market_logloss": float(log_loss(y, test_s["p_fav_mkt"])),
        "market_auc": float(roc_auc_score(y, test_s["p_fav_mkt"])),
    }

    rows = []
    print("\nFlat 1u edge bets on TEST (Elo vs closing market):")
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
        metrics[f"nbets_edge_{e:g}"] = row["n_bets"]

    return {
        "metrics": metrics,
        "walked": walked,
        "test_s": test_s,
        "calibrator": calibrator,
        "final_state": final_state,
        "roi_rows": rows,
        "best_k": best_k,
        "best_h": best_h,
        "train": train,
        "test": test,
    }


def save_artifacts(result: dict, df: pd.DataFrame, test_seasons: set[str], *, season_reset: bool) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    suffix = "_season_reset" if season_reset else ""
    calibrator = result["calibrator"]
    final_state: EloState = result["final_state"]
    walked = result["walked"]
    test_s = result["test_s"]
    train = result["train"]
    test = result["test"]
    k = result["best_k"]
    home_adv = result["best_h"]

    params = {
        "k": k,
        "home_adv": home_adv,
        "season_reset": season_reset,
        "initial_rating": final_state.initial,
        "test_seasons": sorted(test_seasons),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "coef_elo_edge": float(calibrator.coef_[0][0]),
        "intercept": float(calibrator.intercept_[0]),
    }
    joblib.dump(calibrator, MODELS_DIR / f"elo_calibrator{suffix}.pkl")
    joblib.dump(
        {
            "k": k,
            "home_adv": home_adv,
            "season_reset": season_reset,
            "ratings": dict(final_state.ratings),
        },
        MODELS_DIR / f"elo_state{suffix}.pkl",
    )
    with open(MODELS_DIR / f"elo_params{suffix}.json", "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    pd.DataFrame(result["roi_rows"]).to_csv(
        TABLES_DIR / f"elo_market_edge_roi{suffix}.csv",
        index=False,
        float_format="%.6f",
    )

    keep = [
        "game_id",
        "game_date",
        "season",
        "home_name",
        "away_name",
        "favorite_side",
        "fav_won",
        "elo_home",
        "elo_away",
        "elo_diff",
        "elo_edge",
        "p_fav_mkt",
        "p_fav_model",
        "edge_fav",
        "edge_dc",
        "fav_odds",
        "dc_odds",
    ]
    test_s[keep].to_csv(
        TABLES_DIR / f"elo_predictions_holdout{suffix}.csv", index=False
    )
    walked[
        [
            "game_id",
            "game_date",
            "season",
            "home_id",
            "away_id",
            "home_name",
            "away_name",
            "elo_home",
            "elo_away",
            "elo_diff",
            "elo_edge",
            "elo_e_home",
        ]
    ].to_csv(DATA_DIR / f"liga_mx_elo{suffix}.csv", index=False)

    id_to_name: dict[int, str] = {}
    for _, row in df.iterrows():
        id_to_name[int(row["home_id"])] = str(row["home_name"])
        id_to_name[int(row["away_id"])] = str(row["away_name"])
    ratings_rows = [
        {"team_id": tid, "team": id_to_name.get(tid, str(tid)), "elo": rating}
        for tid, rating in final_state.ratings.items()
    ]
    ratings_df = pd.DataFrame(ratings_rows).sort_values("elo", ascending=False)
    ratings_df.to_csv(TABLES_DIR / f"elo_ratings_final{suffix}.csv", index=False)
    print(f"Saved artifacts with suffix '{suffix or '(none)'}'")
    print("\nTop ratings:")
    print(ratings_df.head(10).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser(description="Fit Liga MX Elo + calibrate vs market.")
    p.add_argument("--k", type=float, default=None, help="Fixed K (skip grid if set with --home-adv)")
    p.add_argument(
        "--home-adv",
        type=float,
        default=None,
        help="Fixed home advantage Elo points (skip grid if set with --k)",
    )
    p.add_argument(
        "--season-reset",
        action="store_true",
        help="Reset all ratings to 1500 at each Season boundary.",
    )
    p.add_argument(
        "--compare-reset",
        action="store_true",
        help="Run continuous and season-reset Elo, print a side-by-side summary.",
    )
    p.add_argument(
        "--edge",
        type=str,
        default="0,0.02,0.03,0.05,0.08",
        help="Comma-separated edge thresholds for holdout ROI",
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
        f"Train seasons ({len(train_seasons)}): "
        f"{sorted(train_seasons)[0]} … {sorted(train_seasons)[-1]}"
    )
    print(f"Test seasons: {sorted(test_seasons)}")

    modes = [False, True] if args.compare_reset else [args.season_reset]
    results = []
    for season_reset in modes:
        result = evaluate_mode(
            df,
            train_seasons,
            test_seasons,
            season_reset=season_reset,
            k=args.k,
            home_adv=args.home_adv,
            edges=edges,
        )
        save_artifacts(result, df, test_seasons, season_reset=season_reset)
        results.append(result["metrics"])

    if len(results) > 1:
        cmp = pd.DataFrame(results)
        cmp_path = TABLES_DIR / "elo_continuous_vs_season_reset.csv"
        cmp.to_csv(cmp_path, index=False, float_format="%.6f")
        print("\n===== Continuous vs season-reset (TEST) =====")
        show = cmp[
            [
                "mode",
                "k",
                "home_adv",
                "test_auc",
                "test_logloss",
                "test_brier",
                "market_auc",
                "market_logloss",
                "roi_edge_0",
                "roi_edge_0.05",
            ]
        ].copy()
        print(show.to_string(index=False))
        print(f"\nWrote {cmp_path}")

        cont = cmp.loc[cmp["mode"] == "continuous"].iloc[0]
        reset = cmp.loc[cmp["mode"] == "season_reset"].iloc[0]
        better = "season_reset" if reset["test_logloss"] < cont["test_logloss"] else "continuous"
        print(
            f"\nVerdict: {better} wins on holdout log-loss "
            f"({reset['test_logloss']:.4f} reset vs {cont['test_logloss']:.4f} continuous)."
        )


if __name__ == "__main__":
    main()
