"""
Feature engineering for Liga MX matches.

Adds lagged team form, H2H, and rest days. Writes data/liga_mx_featured.csv.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from model_utils import (
    LAG_WINDOW,
    FEATURE_COLS,
    h2h_lag_column_names,
    lag_vector,
    team_lag_column_names,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

MATCHES_CSV = DATA_DIR / "liga_mx_matches.csv"
OUT_CSV = DATA_DIR / "liga_mx_featured.csv"


def add_lagged_team_features(df: pd.DataFrame, window: int = LAG_WINDOW) -> pd.DataFrame:
    team_wins: dict[int, list[int]] = defaultdict(list)
    team_goals: dict[int, list[float]] = defaultdict(list)
    team_goals_allowed: dict[int, list[float]] = defaultdict(list)
    team_goal_diff: dict[int, list[float]] = defaultdict(list)

    cnames = team_lag_column_names()
    cols: dict[str, list[float]] = {c: [] for c in cnames}

    def append_side(team_id: int, prefix: str) -> None:
        tw = lag_vector(team_wins[team_id], window, 0.0)
        tg = lag_vector(team_goals[team_id], window, 0.0)
        tga = lag_vector(team_goals_allowed[team_id], window, 0.0)
        tgd = lag_vector(team_goal_diff[team_id], window, 0.0)
        for k in range(window):
            cols[f"{prefix}_win_lag_{k + 1}"].append(tw[k])
            cols[f"{prefix}_goals_lag_{k + 1}"].append(tg[k])
            cols[f"{prefix}_goals_allowed_lag_{k + 1}"].append(tga[k])
            cols[f"{prefix}_goal_diff_lag_{k + 1}"].append(tgd[k])

    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        home_won = int(row["home_win"])
        away_won = int(row["away_win"])
        hg = float(row["home_goals"])
        ag = float(row["away_goals"])

        append_side(home_id, "home")
        append_side(away_id, "away")

        team_wins[home_id].append(home_won)
        team_wins[away_id].append(away_won)
        team_goals[home_id].append(hg)
        team_goals[away_id].append(ag)
        team_goals_allowed[home_id].append(ag)
        team_goals_allowed[away_id].append(hg)
        team_goal_diff[home_id].append(hg - ag)
        team_goal_diff[away_id].append(ag - hg)

    out = df.copy()
    for c in cnames:
        out[c] = cols[c]
    return out


def add_lagged_h2h(df: pd.DataFrame, window: int = LAG_WINDOW) -> pd.DataFrame:
    h2h: dict[tuple[int, int], list[tuple[pd.Timestamp, int, int]]] = defaultdict(list)
    cnames = h2h_lag_column_names()
    cols: dict[str, list[float]] = {c: [] for c in cnames}

    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        game_date = row["game_date"]
        home_won = int(row["home_win"])

        key = (min(home_id, away_id), max(home_id, away_id))
        past = [p for p in h2h[key] if p[0] < game_date][-window:]
        wins_chrono: list[int] = []
        for _past_date, past_home_id, past_home_won in past:
            if past_home_id == home_id:
                wins_chrono.append(past_home_won)
            else:
                wins_chrono.append(1 - past_home_won)

        lags = lag_vector(wins_chrono, window, 0.0)
        for k in range(window):
            cols[f"home_h2h_win_lag_{k + 1}"].append(lags[k])

        h2h[key].append((game_date, home_id, home_won))

    out = df.copy()
    for c in cnames:
        out[c] = cols[c]
    return out


def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    last_played: dict[int, pd.Timestamp] = {}
    home_rest: list[float] = []
    away_rest: list[float] = []

    for _, row in df.iterrows():
        home_id = int(row["home_id"])
        away_id = int(row["away_id"])
        game_date = row["game_date"]

        if home_id in last_played:
            home_rest.append(float((game_date - last_played[home_id]).days))
        else:
            home_rest.append(7.0)
        if away_id in last_played:
            away_rest.append(float((game_date - last_played[away_id]).days))
        else:
            away_rest.append(7.0)

        last_played[home_id] = game_date
        last_played[away_id] = game_date

    out = df.copy()
    out["home_rest_days"] = home_rest
    out["away_rest_days"] = away_rest
    return out


def add_season_index(df: pd.DataFrame) -> pd.DataFrame:
    seasons = sorted(df["season"].unique())
    mapping = {s: i for i, s in enumerate(seasons)}
    out = df.copy()
    out["season_index"] = out["season"].map(mapping).astype(int)
    return out


def main() -> None:
    print(f"Loading {MATCHES_CSV}...")
    df = pd.read_csv(MATCHES_CSV)
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    print(f"  Building lags (window={LAG_WINDOW})...")
    df = add_lagged_team_features(df)
    df = add_lagged_h2h(df)
    df = add_rest_days(df)
    df = add_season_index(df)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing feature columns: {missing}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"  Rows: {len(df)}  features: {len(FEATURE_COLS)}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
