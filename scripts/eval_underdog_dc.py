"""
Backtest two-way framing of Liga MX 1X2 closing odds:

  - favorite: always bet the closing favorite to win
  - underdog_dc: always bet draw OR underdog (underdog double chance)

DC prices from market-average closing 1X2:
  odds_dc = 1 / (1/odds_draw + 1/odds_underdog)

Flat 1-unit stakes. Reports ROI by season and overall.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = PROJECT_ROOT / "data" / "MEX.csv"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "tables"


def load_matches(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"Season", "Home", "Away", "Res", "AvgCH", "AvgCD", "AvgCA"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    out = df.copy()
    for col in ("AvgCH", "AvgCD", "AvgCA", "HG", "AG"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    out = out.dropna(subset=["Season", "Res", "AvgCH", "AvgCD", "AvgCA"])
    out = out[out["Res"].isin(["H", "D", "A"])]
    out = out[(out["AvgCH"] > 1) & (out["AvgCD"] > 1) & (out["AvgCA"] > 1)]
    return out.reset_index(drop=True)


def annotate_bets(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Label favorite / underdog DC and compute flat-bet P/L for the chosen side."""
    out = df.copy()

    # Skip true pick'ems (equal home/away closing price).
    out = out.loc[out["AvgCH"] != out["AvgCA"]].copy()

    home_fav = out["AvgCH"] < out["AvgCA"]
    out["favorite_side"] = home_fav.map({True: "H", False: "A"})
    out["underdog_side"] = home_fav.map({True: "A", False: "H"})
    out["fav_odds"] = out["AvgCH"].where(home_fav, out["AvgCA"])
    out["dog_odds"] = out["AvgCA"].where(home_fav, out["AvgCH"])

    # Approx book double-chance price from 1X2 market averages (vig still in).
    out["dc_odds"] = 1.0 / (1.0 / out["AvgCD"] + 1.0 / out["dog_odds"])

    out["fav_won"] = out["Res"] == out["favorite_side"]
    out["dc_won"] = ~out["fav_won"]  # draw or underdog

    if side == "favorite":
        out["bet_won"] = out["fav_won"]
        out["bet_odds"] = out["fav_odds"]
    elif side == "underdog_dc":
        out["bet_won"] = out["dc_won"]
        out["bet_odds"] = out["dc_odds"]
    else:
        raise ValueError(f"Unknown side: {side}")

    out["profit"] = -1.0
    out.loc[out["bet_won"], "profit"] = out.loc[out["bet_won"], "bet_odds"] - 1.0

    # Market-implied P(favorite wins) after removing 1X2 vig (for calibration).
    p_h = 1.0 / out["AvgCH"]
    p_d = 1.0 / out["AvgCD"]
    p_a = 1.0 / out["AvgCA"]
    total = p_h + p_d + p_a
    p_h_fair = p_h / total
    p_a_fair = p_a / total
    out["p_fav_fair"] = p_h_fair.where(home_fav, p_a_fair)
    out["p_dc_fair"] = 1.0 - out["p_fav_fair"]

    return out.reset_index(drop=True)


def season_summary(bets: pd.DataFrame) -> pd.DataFrame:
    g = bets.groupby("Season", sort=True)
    summary = g.agg(
        n_bets=("profit", "size"),
        n_wins=("bet_won", "sum"),
        win_rate=("bet_won", "mean"),
        avg_bet_odds=("bet_odds", "mean"),
        profit=("profit", "sum"),
        fav_win_rate=("fav_won", "mean"),
        avg_p_fav_fair=("p_fav_fair", "mean"),
        avg_fav_odds=("fav_odds", "mean"),
        avg_dc_odds=("dc_odds", "mean"),
    )
    summary["roi"] = summary["profit"] / summary["n_bets"]
    summary["n_wins"] = summary["n_wins"].astype(int)

    overall = pd.DataFrame(
        {
            "n_bets": [len(bets)],
            "n_wins": [int(bets["bet_won"].sum())],
            "win_rate": [bets["bet_won"].mean()],
            "avg_bet_odds": [bets["bet_odds"].mean()],
            "profit": [bets["profit"].sum()],
            "fav_win_rate": [bets["fav_won"].mean()],
            "avg_p_fav_fair": [bets["p_fav_fair"].mean()],
            "avg_fav_odds": [bets["fav_odds"].mean()],
            "avg_dc_odds": [bets["dc_odds"].mean()],
            "roi": [bets["profit"].sum() / len(bets)],
        },
        index=pd.Index(["ALL"], name="Season"),
    )
    return pd.concat([summary, overall])


def format_pct(x: float) -> str:
    return f"{100.0 * x:+.1f}%" if pd.notna(x) else ""


def print_report(summary: pd.DataFrame, side: str) -> None:
    title = {
        "favorite": "Always bet the closing favorite, 1u flat",
        "underdog_dc": "Always bet underdog double chance (draw + dog), 1u flat",
    }[side]
    odds_note = {
        "favorite": "Favorite odds = min(AvgCH, AvgCA) closing averages",
        "underdog_dc": "DC odds approximated from AvgCH/AvgCD/AvgCA closing averages",
    }[side]

    cols = [
        "n_bets",
        "n_wins",
        "win_rate",
        "avg_bet_odds",
        "profit",
        "roi",
        "fav_win_rate",
        "avg_p_fav_fair",
    ]
    view = summary[cols].copy()
    view["win_rate"] = view["win_rate"].map(lambda x: f"{100 * x:.1f}%")
    view["roi"] = view["roi"].map(format_pct)
    view["profit"] = view["profit"].map(lambda x: f"{x:+.1f}")
    view["avg_bet_odds"] = view["avg_bet_odds"].map(lambda x: f"{x:.3f}")
    view["fav_win_rate"] = view["fav_win_rate"].map(lambda x: f"{100 * x:.1f}%")
    view["avg_p_fav_fair"] = view["avg_p_fav_fair"].map(lambda x: f"{100 * x:.1f}%")
    view = view.rename(
        columns={
            "n_bets": "bets",
            "n_wins": "wins",
            "win_rate": "hit%",
            "avg_bet_odds": "avg_odds",
            "profit": "P/L (u)",
            "roi": "ROI",
            "fav_win_rate": "fav_hit%",
            "avg_p_fav_fair": "mkt_p_fav",
        }
    )
    print(f"\n{title}")
    print(f"{odds_note}\n")
    print(view.to_string())
    print()
    all_row = summary.loc["ALL"]
    print(
        f"Verdict: {'PROFITABLE' if all_row['profit'] > 0 else 'NOT profitable'} "
        f"overall — {all_row['profit']:+.1f}u on {int(all_row['n_bets'])} bets "
        f"({format_pct(all_row['roi'])} ROI)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate always-betting Liga MX favorite or underdog DC by season."
    )
    parser.add_argument(
        "--side",
        choices=["favorite", "underdog_dc"],
        default="underdog_dc",
        help="Which side of the two-way market to bet (default: underdog_dc)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=DEFAULT_CSV,
        help=f"Path to football-data Mexico CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Season summary CSV path (default depends on --side)",
    )
    args = parser.parse_args()
    out_path = args.output or (DEFAULT_OUT_DIR / f"{args.side}_by_season.csv")

    matches = load_matches(args.csv)
    bets = annotate_bets(matches, side=args.side)
    summary = season_summary(bets)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, float_format="%.6f")
    print_report(summary, side=args.side)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
