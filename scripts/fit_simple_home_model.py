"""
Home-win model: recent form + H2H + leak-free season context.

Predicts P(home win) only. Closing odds are not model inputs; they are used
afterward to decide bets when model probability exceeds the fair market by
an edge threshold (default 10%).

Features (all strictly pre-match):
  - Home / away last-5 form: win rate, draw rate, avg goal difference
  - Last-5 H2H (from current home team's perspective) + meetings available
  - Season-to-date form for home / away (only games already played this Season)
  - Season progress for each side (games already played / typical season length)

Chronological evaluation:
  - Train: seasons through 2023/2024
  - Validation (optional threshold check): 2024/2025
  - Untouched test: 2025/2026
  - 2026/2027 excluded (incomplete)

Usage:
  python scripts/fit_simple_home_model.py
  python scripts/fit_simple_home_model.py --edge 0.10
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

MATCHES_CSV = DATA_DIR / "liga_mx_matches.csv"
MODEL_PATH = MODELS_DIR / "simple_home_win_model.joblib"
MODEL_CARD_PATH = MODELS_DIR / "simple_home_win_model.json"
COEFFICIENTS_PATH = TABLES_DIR / "simple_home_win_coefficients.csv"
VALIDATION_EDGE_PATH = TABLES_DIR / "simple_home_win_validation_edges.csv"
HOLDOUT_EDGE_PATH = TABLES_DIR / "simple_home_win_holdout_edges.csv"
PREDICTIONS_PATH = TABLES_DIR / "simple_home_win_predictions_holdout.csv"

WINDOW = 5
# Liga MX Apertura/Clausura-style length used only for scaling progress in [0, 1].
# Does not use future match counts from the current season.
TYPICAL_SEASON_GAMES = 17
VALIDATION_SEASON = "2024/2025"
TEST_SEASON = "2025/2026"
INCOMPLETE_SEASONS = {"2026/2027"}
DEFAULT_EDGE = 0.10
EDGE_SWEEP = np.round(np.arange(0.0, 0.201, 0.01), 2)

FEATURE_COLS = [
    "home_win_rate_l5",
    "home_draw_rate_l5",
    "home_goal_diff_l5",
    "away_win_rate_l5",
    "away_draw_rate_l5",
    "away_goal_diff_l5",
    "h2h_home_win_rate_l5",
    "h2h_draw_rate_l5",
    "h2h_home_goal_diff_l5",
    "h2h_matches_available",
    "home_sotd_win_rate",
    "home_sotd_draw_rate",
    "home_sotd_goal_diff",
    "home_sotd_games",
    "away_sotd_win_rate",
    "away_sotd_draw_rate",
    "away_sotd_goal_diff",
    "away_sotd_games",
    "home_season_progress",
    "away_season_progress",
]


def _form_rates(history: list[tuple[int, int, float]]) -> tuple[float, float, float]:
    recent = history[-WINDOW:]
    return (
        float(np.mean([m[0] for m in recent])),
        float(np.mean([m[1] for m in recent])),
        float(np.mean([m[2] for m in recent])),
    )


def _sotd_rates(
    history: list[tuple[int, int, float]],
) -> tuple[float, float, float, int, float]:
    """Season-to-date rates + game count + progress in [0, 1]."""
    n = len(history)
    progress = min(n / float(TYPICAL_SEASON_GAMES), 1.0)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0, progress
    return (
        float(np.mean([m[0] for m in history])),
        float(np.mean([m[1] for m in history])),
        float(np.mean([m[2] for m in history])),
        n,
        progress,
    )


def _h2h_form(
    history: list[tuple[int, int, float, float]], current_home_id: int
) -> tuple[float, float, float, int]:
    home_wins: list[int] = []
    draws: list[int] = []
    goal_diffs: list[float] = []

    for past_home_id, _past_away_id, past_hg, past_ag in history[-WINDOW:]:
        if past_home_id == current_home_id:
            goals_for, goals_against = past_hg, past_ag
        else:
            goals_for, goals_against = past_ag, past_hg
        home_wins.append(int(goals_for > goals_against))
        draws.append(int(goals_for == goals_against))
        goal_diffs.append(goals_for - goals_against)

    count = len(home_wins)
    if count == 0:
        return float("nan"), float("nan"), float("nan"), 0
    return (
        float(np.mean(home_wins)),
        float(np.mean(draws)),
        float(np.mean(goal_diffs)),
        count,
    )


def build_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Build strictly pre-match rolling + season-to-date features chronologically."""
    df = matches.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # Career rolling (cross-season) for last-5 form.
    team_history: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    # Season-to-date only; cleared when Season label changes for that team.
    team_sotd: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
    team_sotd_season: dict[int, str] = {}
    h2h_history: dict[
        tuple[int, int], list[tuple[int, int, float, float]]
    ] = defaultdict(list)
    rows: list[dict] = []

    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        season = str(row["season"])
        h2h_key = (min(home_id, away_id), max(home_id, away_id))

        for tid in (home_id, away_id):
            if team_sotd_season.get(tid) != season:
                team_sotd[tid] = []
                team_sotd_season[tid] = season

        if (
            len(team_history[home_id]) >= WINDOW
            and len(team_history[away_id]) >= WINDOW
        ):
            home_form = _form_rates(team_history[home_id])
            away_form = _form_rates(team_history[away_id])
            h2h_form = _h2h_form(h2h_history[h2h_key], home_id)
            home_sotd = _sotd_rates(team_sotd[home_id])
            away_sotd = _sotd_rates(team_sotd[away_id])
            rows.append(
                {
                    "game_id": int(row["game_id"]),
                    "game_date": row["game_date"],
                    "season": season,
                    "home_name": row["home_name"],
                    "away_name": row["away_name"],
                    "home_win": int(row["home_win"]),
                    "home_odds": float(row["AvgCH"]),
                    "p_home_market_fair": float(row["p_home_fair"]),
                    "home_win_rate_l5": home_form[0],
                    "home_draw_rate_l5": home_form[1],
                    "home_goal_diff_l5": home_form[2],
                    "away_win_rate_l5": away_form[0],
                    "away_draw_rate_l5": away_form[1],
                    "away_goal_diff_l5": away_form[2],
                    "h2h_home_win_rate_l5": h2h_form[0],
                    "h2h_draw_rate_l5": h2h_form[1],
                    "h2h_home_goal_diff_l5": h2h_form[2],
                    "h2h_matches_available": h2h_form[3],
                    "home_sotd_win_rate": home_sotd[0],
                    "home_sotd_draw_rate": home_sotd[1],
                    "home_sotd_goal_diff": home_sotd[2],
                    "home_sotd_games": home_sotd[3],
                    "away_sotd_win_rate": away_sotd[0],
                    "away_sotd_draw_rate": away_sotd[1],
                    "away_sotd_goal_diff": away_sotd[2],
                    "away_sotd_games": away_sotd[3],
                    "home_season_progress": home_sotd[4],
                    "away_season_progress": away_sotd[4],
                }
            )

        home_goals = float(row["home_goals"])
        away_goals = float(row["away_goals"])
        is_draw = int(home_goals == away_goals)
        home_rec = (
            int(home_goals > away_goals),
            is_draw,
            home_goals - away_goals,
        )
        away_rec = (
            int(away_goals > home_goals),
            is_draw,
            away_goals - home_goals,
        )
        team_history[home_id].append(home_rec)
        team_history[away_id].append(away_rec)
        team_sotd[home_id].append(home_rec)
        team_sotd[away_id].append(away_rec)
        h2h_history[h2h_key].append((home_id, away_id, home_goals, away_goals))

    return pd.DataFrame(rows)


def make_pipeline() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=42)),
        ]
    )


def add_predictions(model: Pipeline, games: pd.DataFrame) -> pd.DataFrame:
    out = games.copy()
    out["p_home_model"] = model.predict_proba(out[FEATURE_COLS])[:, 1]
    out["market_probability_edge"] = (
        out["p_home_model"] - out["p_home_market_fair"]
    )
    out["modeled_expected_roi"] = out["p_home_model"] * out["home_odds"] - 1.0
    out["bet_profit"] = np.where(
        out["home_win"] == 1, out["home_odds"] - 1.0, -1.0
    )
    return out


def edge_results(
    games: pd.DataFrame,
    *,
    thresholds: np.ndarray = EDGE_SWEEP,
    edge_col: str = "market_probability_edge",
) -> pd.DataFrame:
    rows: list[dict] = []
    for threshold in thresholds:
        bets = games.loc[games[edge_col] >= threshold]
        rows.append(
            {
                "min_probability_edge": float(threshold),
                "bets": int(len(bets)),
                "wins": int(bets["home_win"].sum()) if len(bets) else 0,
                "hit_rate": (
                    float(bets["home_win"].mean()) if len(bets) else float("nan")
                ),
                "profit": float(bets["bet_profit"].sum()) if len(bets) else 0.0,
                "roi": (
                    float(bets["bet_profit"].mean()) if len(bets) else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_edge(games: pd.DataFrame, edge: float) -> dict[str, float | int]:
    bets = games.loc[games["market_probability_edge"] >= edge]
    if bets.empty:
        return {"bets": 0, "wins": 0, "hit_rate": float("nan"), "profit": 0.0, "roi": float("nan")}
    return {
        "bets": int(len(bets)),
        "wins": int(bets["home_win"].sum()),
        "hit_rate": float(bets["home_win"].mean()),
        "profit": float(bets["bet_profit"].sum()),
        "roi": float(bets["bet_profit"].mean()),
    }


def probability_metrics(games: pd.DataFrame) -> dict[str, float]:
    y = games["home_win"]
    model_p = games["p_home_model"]
    market_p = games["p_home_market_fair"]
    return {
        "model_auc": float(roc_auc_score(y, model_p)),
        "model_log_loss": float(log_loss(y, model_p)),
        "model_brier": float(brier_score_loss(y, model_p)),
        "market_log_loss": float(log_loss(y, market_p)),
        "market_brier": float(brier_score_loss(y, market_p)),
    }


def coefficient_table(model: Pipeline) -> pd.DataFrame:
    coefficients = model.named_steps["model"].coef_[0]
    return pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "standardized_coefficient": coefficients,
        }
    ).sort_values("standardized_coefficient", key=abs, ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit home-win model; bet when probability edge ≥ threshold."
    )
    parser.add_argument(
        "--edge",
        type=float,
        default=DEFAULT_EDGE,
        help="Minimum (model − fair market) probability edge to bet home (default 0.10).",
    )
    args = parser.parse_args()
    edge = float(args.edge)

    print(f"Loading {MATCHES_CSV}...")
    matches = pd.read_csv(MATCHES_CSV)
    featured = build_features(matches)
    featured = featured.loc[
        ~featured["season"].isin(INCOMPLETE_SEASONS)
    ].reset_index(drop=True)

    validation = featured.loc[featured["season"] == VALIDATION_SEASON].copy()
    test = featured.loc[featured["season"] == TEST_SEASON].copy()
    train = featured.loc[
        ~featured["season"].isin([VALIDATION_SEASON, TEST_SEASON])
    ].copy()
    if train.empty or validation.empty or test.empty:
        raise RuntimeError("Train, validation, and test sets must all be non-empty.")

    print(
        f"Samples: train={len(train)}, validation={len(validation)}, "
        f"test={len(test)}"
    )
    print(
        f"Features: last-{WINDOW} form, H2H, season-to-date, season progress "
        f"(no season ID / no odds in the model)"
    )
    print(f"Bet rule: home only when market_probability_edge ≥ {edge:.0%}")

    # Fit on pre-validation seasons; score val + holdout with a model that
    # never sees the test season.
    selection_model = make_pipeline()
    selection_model.fit(train[FEATURE_COLS], train["home_win"])
    validation_scored = add_predictions(selection_model, validation)
    validation_edges = edge_results(validation_scored)
    val_at_edge = summarize_edge(validation_scored, edge)

    pre_test = pd.concat([train, validation], ignore_index=True)
    evaluation_model = make_pipeline()
    evaluation_model.fit(pre_test[FEATURE_COLS], pre_test["home_win"])
    test_scored = add_predictions(evaluation_model, test)
    holdout_edges = edge_results(test_scored)
    test_at_edge = summarize_edge(test_scored, edge)
    metrics = probability_metrics(test_scored)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    validation_edges.to_csv(VALIDATION_EDGE_PATH, index=False, float_format="%.6f")
    holdout_edges.to_csv(HOLDOUT_EDGE_PATH, index=False, float_format="%.6f")

    prediction_cols = [
        "game_id",
        "game_date",
        "season",
        "home_name",
        "away_name",
        "home_win",
        "home_odds",
        "p_home_market_fair",
        "p_home_model",
        "market_probability_edge",
        "modeled_expected_roi",
        "bet_profit",
    ] + FEATURE_COLS
    test_scored[prediction_cols].to_csv(
        PREDICTIONS_PATH, index=False, float_format="%.6f"
    )

    production_model = make_pipeline()
    production_model.fit(featured[FEATURE_COLS], featured["home_win"])
    joblib.dump(production_model, MODEL_PATH)
    coefficient_table(production_model).to_csv(
        COEFFICIENTS_PATH, index=False, float_format="%.6f"
    )

    profitable = (
        test_at_edge["bets"] > 0
        and not np.isnan(test_at_edge["roi"])
        and test_at_edge["roi"] > 0
    )
    model_card = {
        "target": "home_win",
        "window": WINDOW,
        "typical_season_games": TYPICAL_SEASON_GAMES,
        "features": FEATURE_COLS,
        "odds_in_model": False,
        "bet_side": "home",
        "bet_rule": "market_probability_edge = p_home_model - p_home_market_fair",
        "edge_threshold": edge,
        "validation_season": VALIDATION_SEASON,
        "test_season": TEST_SEASON,
        "leakage_notes": [
            "Form/H2H/season-to-date use only games strictly before kickoff.",
            "Season-to-date resets at each Season boundary; no future results.",
            "Season progress uses games already played / typical length, not "
            "the eventual season length.",
            "Season label is never a model feature (avoids one-hot leakage).",
            "Odds used only for edge/ROI after prediction.",
        ],
        "validation_at_edge": val_at_edge,
        "holdout_at_edge": test_at_edge,
        "holdout_probability_metrics": metrics,
        "production_training_seasons": sorted(featured["season"].unique().tolist()),
        "warning": (
            None
            if profitable
            else (
                "Strategy was not profitable on the untouched holdout season "
                f"at {edge:.0%} probability edge. Do not treat as a validated edge."
            )
        ),
    }
    MODEL_CARD_PATH.write_text(json.dumps(model_card, indent=2) + "\n")

    print("\nProbability quality — untouched holdout:")
    print(
        f"  AUC={metrics['model_auc']:.3f}  "
        f"model Brier={metrics['model_brier']:.3f}  "
        f"market Brier={metrics['market_brier']:.3f}"
    )

    def _print_slice(title: str, stats: dict) -> None:
        if stats["bets"] == 0:
            print(f"\n{title}: 0 bets")
            return
        print(
            f"\n{title}:\n"
            f"  Bets: {stats['bets']}  wins: {stats['wins']}  "
            f"hit: {stats['hit_rate']:.1%}  "
            f"P/L: {stats['profit']:+.2f}u  ROI: {stats['roi']:+.1%}"
        )

    _print_slice(f"Validation {VALIDATION_SEASON} @ {edge:.0%} edge", val_at_edge)
    _print_slice(f"Holdout {TEST_SEASON} @ {edge:.0%} edge", test_at_edge)

    print("\nHoldout edge sweep (probability edge):")
    print(f"  {'edge':>6}  {'bets':>5}  {'hit%':>6}  {'P/L':>8}  {'ROI':>7}")
    for _, row in holdout_edges.iterrows():
        if row["bets"] == 0:
            continue
        if row["min_probability_edge"] not in (0.0, 0.05, 0.08, 0.10, 0.12, 0.15):
            continue
        print(
            f"  {row['min_probability_edge']:6.2f}  {int(row['bets']):5d}  "
            f"{100 * row['hit_rate']:5.1f}%  {row['profit']:+8.1f}  "
            f"{100 * row['roi']:+6.1f}%"
        )

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved model card: {MODEL_CARD_PATH}")
    print(f"Saved holdout predictions: {PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
