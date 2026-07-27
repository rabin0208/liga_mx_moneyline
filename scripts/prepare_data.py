"""
Clean football-data Mexico CSV into a modeling table.

Reads data/MEX.csv, parses dates, adds outcome flags and two-way market columns,
writes data/liga_mx_matches.csv.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from odds_utils import add_two_way_market

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"

RAW_CSV = DATA_DIR / "MEX.csv"
OUT_CSV = DATA_DIR / "liga_mx_matches.csv"


def main() -> None:
    print(f"Loading {RAW_CSV}...")
    df = pd.read_csv(RAW_CSV)
    df["game_date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["game_date", "Home", "Away", "Res", "HG", "AG"])
    df = df.loc[df["Res"].isin(["H", "D", "A"])].copy()

    df = df.rename(
        columns={
            "Home": "home_name",
            "Away": "away_name",
            "HG": "home_goals",
            "AG": "away_goals",
            "Season": "season",
        }
    )
    df["home_win"] = (df["Res"] == "H").astype(int)
    df["draw"] = (df["Res"] == "D").astype(int)
    df["away_win"] = (df["Res"] == "A").astype(int)
    df["week_of_year"] = df["game_date"].dt.isocalendar().week.astype(int)

    # Stable team ids from name (across seasons).
    teams = sorted(set(df["home_name"]) | set(df["away_name"]))
    team_to_id = {name: i for i, name in enumerate(teams)}
    df["home_id"] = df["home_name"].map(team_to_id)
    df["away_id"] = df["away_name"].map(team_to_id)

    df = add_two_way_market(df)
    df = df.sort_values(["game_date", "home_name", "away_name"]).reset_index(drop=True)
    df["game_id"] = df.index.astype(int)

    keep = [
        "game_id",
        "game_date",
        "season",
        "week_of_year",
        "home_id",
        "away_id",
        "home_name",
        "away_name",
        "home_goals",
        "away_goals",
        "Res",
        "home_win",
        "draw",
        "away_win",
        "favorite_side",
        "underdog_side",
        "fav_odds",
        "dog_odds",
        "dc_odds",
        "fav_won",
        "dc_won",
        "AvgCH",
        "AvgCD",
        "AvgCA",
        "p_home_fair",
        "p_draw_fair",
        "p_away_fair",
        "p_fav_mkt",
        "p_dc_mkt",
    ]
    out = df[keep]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"  Rows: {len(out)}  seasons: {out['season'].nunique()}  teams: {len(teams)}")
    print(f"  Favorite win rate: {out['fav_won'].mean():.3f}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
