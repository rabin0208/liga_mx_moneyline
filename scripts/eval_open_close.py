"""
Compare Bayesian DC probabilities to ESPN BET opening and closing 1X2.

Football-data's Mexico file only has closing averages (AvgCH/AvgCD/AvgCA).
ESPN's core odds API keeps ESPN BET open + close moneyline (including draw)
for Liga MX, which we join onto an existing predictions table.

Usage:
  python scripts/eval_open_close.py
  python scripts/eval_open_close.py --predictions results/tables/bayesian_dc_predictions_holdout_2026.csv
  python scripts/eval_open_close.py --predictions results/tables/bayesian_dc_predictions_walkforward.csv --since 2025-07-01
  python scripts/eval_open_close.py --refresh
"""
from __future__ import annotations

import argparse
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import accuracy_score, log_loss

from odds_utils import flat_bet_profit, implied_prob, remove_vig_3way
from predict_upcoming import (
    ESPN_HEADER,
    ESPN_LEAGUE,
    ESPN_UA,
    american_to_decimal,
    map_team,
)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"

DEFAULT_PREDICTIONS = TABLES_DIR / "bayesian_dc_predictions_holdout.csv"
ODDS_CACHE = DATA_DIR / "espn_open_close.csv"
CLASS_ORDER = ["H", "D", "A"]
ESPN_CORE_ODDS = (
    "https://sports.core.api.espn.com/v2/sports/soccer/leagues/"
    f"{ESPN_LEAGUE}/events/{{eid}}/competitions/{{eid}}/odds"
)
SIDES = ("home", "draw", "away")
ODDS_OPEN = {"H": "open_H", "D": "open_D", "A": "open_A"}
ODDS_CLOSE = {"H": "close_H", "D": "close_D", "A": "close_A"}
FAIR_OPEN = {"H": "p_home_open", "D": "p_draw_open", "A": "p_away_open"}
FAIR_CLOSE = {"H": "p_home_espn_close", "D": "p_draw_espn_close", "A": "p_away_espn_close"}
FAIR_AVGC = {"H": "p_home_fair", "D": "p_draw_fair", "A": "p_away_fair"}
MODEL = {"H": "p_home_model", "D": "p_draw_model", "A": "p_away_model"}
# football-data Date is the UK calendar date of kickoff (23:00 UTC → next day).
FD_DATE_TZ = "Europe/London"


def _finite_odds(value) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(x) and x > 1.0)


def _price_decimal(price) -> float:
    if not isinstance(price, dict):
        return float("nan")
    if price.get("decimal") is not None:
        try:
            return float(price["decimal"])
        except (TypeError, ValueError):
            return float("nan")
    if price.get("american") is not None:
        return american_to_decimal(price["american"])
    return float("nan")


def _block_ml_decimal(block: dict | None, snap: str) -> float:
    if not isinstance(block, dict):
        return float("nan")
    nested = block.get(snap)
    if isinstance(nested, dict):
        ml = nested.get("moneyLine")
        if isinstance(ml, dict):
            return _price_decimal(ml)
        if ml is not None:
            return american_to_decimal(ml)
    if snap == "close" and block.get("moneyLine") is not None:
        ml = block.get("moneyLine")
        if isinstance(ml, dict):
            return _price_decimal(ml)
        return american_to_decimal(ml)
    return float("nan")


def parse_core_item(item: dict) -> dict | None:
    home = item.get("homeTeamOdds") or {}
    away = item.get("awayTeamOdds") or {}
    top_open = item.get("open") or {}
    top_close = item.get("close") or {}
    oh = _block_ml_decimal(home, "open")
    ch = _block_ml_decimal(home, "close")
    oa = _block_ml_decimal(away, "open")
    ca = _block_ml_decimal(away, "close")
    od = _price_decimal(top_open.get("draw") or {})
    cd = _price_decimal(top_close.get("draw") or {})
    if not _finite_odds(od):
        od = _block_ml_decimal(item.get("drawOdds") or {}, "open")
    if not _finite_odds(cd):
        cd = _block_ml_decimal(item.get("drawOdds") or {}, "close")
    if not all(_finite_odds(x) for x in (oh, od, oa, ch, cd, ca)):
        return None
    return {
        "book": (item.get("provider") or {}).get("name"),
        "open_H": oh,
        "open_D": od,
        "open_A": oa,
        "close_H": ch,
        "close_D": cd,
        "close_A": ca,
    }


def _book_rank(name: str | None) -> int:
    n = str(name or "").lower()
    if "live" in n:
        return 9
    if n == "espn bet":
        return 0
    if n == "draftkings":
        return 1
    return 5


def fetch_core_odds(espn_id: str) -> dict | None:
    url = ESPN_CORE_ODDS.format(eid=espn_id)
    r = requests.get(url, headers=ESPN_UA, timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    parsed: list[dict] = []
    for item in r.json().get("items") or []:
        row = parse_core_item(item)
        if row is not None:
            parsed.append(row)
    if not parsed:
        return None
    parsed.sort(key=lambda t: _book_rank(t.get("book")))
    return parsed[0]


def _espn_header_events(lo: str, hi: str) -> list[dict]:
    url = f"{ESPN_HEADER}?sport=soccer&league={ESPN_LEAGUE}&dates={lo}-{hi}"
    r = requests.get(url, headers=ESPN_UA, timeout=30)
    r.raise_for_status()
    sports = r.json().get("sports") or []
    if not sports:
        return []
    leagues = sports[0].get("leagues") or []
    if not leagues:
        return []
    return list(leagues[0].get("events") or [])


def _competitor_name(comp: dict) -> str | None:
    return (
        comp.get("displayName")
        or (comp.get("team") or {}).get("displayName")
        or comp.get("name")
    )


def parse_header_event(event: dict) -> dict | None:
    comps = event.get("competitors") or []
    home = away = None
    for c in comps:
        name = _competitor_name(c)
        if c.get("homeAway") == "home":
            home = name
        else:
            away = name
    if not home or not away or not event.get("id"):
        return None
    kick = pd.to_datetime(event.get("date"), utc=True)
    return {
        "espn_id": str(event.get("id")),
        "kickoff_utc": kick,
        "espn_home": home,
        "espn_away": away,
        "status": event.get("status") or event.get("summary") or "",
    }


def date_windows(start: pd.Timestamp, end: pd.Timestamp, days: int = 6) -> list[tuple[str, str]]:
    windows = []
    cur = start.normalize()
    last = end.normalize()
    while cur <= last:
        hi = min(cur + timedelta(days=days - 1), last)
        windows.append((cur.strftime("%Y%m%d"), hi.strftime("%Y%m%d")))
        cur = hi + timedelta(days=1)
    return windows


def fetch_espn_events(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    seen = set()
    for lo, hi in date_windows(start, end):
        for event in _espn_header_events(lo, hi):
            parsed = parse_header_event(event)
            if parsed is None or parsed["espn_id"] in seen:
                continue
            seen.add(parsed["espn_id"])
            rows.append(parsed)
        time.sleep(0.05)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def load_odds_cache(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "espn_id" in df.columns:
        df["espn_id"] = df["espn_id"].astype(str)
    if "kickoff_utc" in df.columns:
        df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    return df


def collect_open_close(
    events: pd.DataFrame,
    cache: pd.DataFrame,
    *,
    refresh: bool,
    pause: float,
) -> pd.DataFrame:
    odds_cols = ["espn_id", "book", "open_H", "open_D", "open_A", "close_H", "close_D", "close_A"]
    have = set()
    cached_odds = pd.DataFrame(columns=odds_cols)
    if not refresh and not cache.empty:
        present = [c for c in odds_cols if c in cache.columns]
        cached_odds = cache[present].drop_duplicates("espn_id")
        ok = cached_odds.dropna(subset=["open_H", "open_D", "open_A", "close_H", "close_D", "close_A"])
        have = set(ok["espn_id"].astype(str))
    todo = [eid for eid in events["espn_id"].astype(str) if eid not in have]
    new_rows = []
    for i, eid in enumerate(todo, start=1):
        try:
            odds = fetch_core_odds(eid)
        except requests.RequestException as exc:
            print(f"  odds fail {eid}: {exc}")
            odds = None
        rec = {"espn_id": eid}
        if odds:
            rec.update(odds)
        new_rows.append(rec)
        if i % 25 == 0 or i == len(todo):
            print(f"  odds {i}/{len(todo)}")
        time.sleep(pause)
    parts = []
    if not cached_odds.empty:
        parts.append(cached_odds)
    if new_rows:
        parts.append(pd.DataFrame(new_rows))
    odds = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=odds_cols)
    odds["espn_id"] = odds["espn_id"].astype(str)
    odds = odds.drop_duplicates("espn_id", keep="last")
    return events.merge(odds, on="espn_id", how="left")


def add_fair(df: pd.DataFrame, odds: dict[str, str], prefix: str) -> pd.DataFrame:
    out = df.copy()
    p_h, p_d, p_a = [], [], []
    for _, row in out.iterrows():
        h = implied_prob(row[odds["H"]])
        d = implied_prob(row[odds["D"]])
        a = implied_prob(row[odds["A"]])
        fh, fd, fa = remove_vig_3way(h, d, a)
        p_h.append(fh)
        p_d.append(fd)
        p_a.append(fa)
    out[f"p_home_{prefix}"] = p_h
    out[f"p_draw_{prefix}"] = p_d
    out[f"p_away_{prefix}"] = p_a
    return out


def football_data_date(kickoff_utc: pd.Series) -> pd.Series:
    return (
        kickoff_utc.dt.tz_convert(FD_DATE_TZ)
        .dt.tz_localize(None)
        .dt.normalize()
    )


def join_predictions(pred: pd.DataFrame, espn: pd.DataFrame, known: set[str]) -> pd.DataFrame:
    rows = []
    skipped = 0
    for _, row in espn.iterrows():
        try:
            home = map_team(str(row["espn_home"]), known)
            away = map_team(str(row["espn_away"]), known)
        except KeyError:
            skipped += 1
            continue
        rec = row.to_dict()
        rec["home_name"] = home
        rec["away_name"] = away
        rows.append(rec)
    mapped = pd.DataFrame(rows)
    if mapped.empty:
        print(f"  no ESPN rows mapped (skipped {skipped})")
        return pred.iloc[0:0].copy()
    mapped["fd_date"] = football_data_date(mapped["kickoff_utc"])
    pred = pred.copy()
    pred["game_date"] = pd.to_datetime(pred["game_date"]).dt.normalize()
    merged = pred.merge(
        mapped,
        left_on=["home_name", "away_name", "game_date"],
        right_on=["home_name", "away_name", "fd_date"],
        how="inner",
        suffixes=("", "_espn"),
    )
    used_espn = set(merged["espn_id"].astype(str)) if "espn_id" in merged.columns else set()
    used_pred = set(zip(merged["home_name"], merged["away_name"], merged["game_date"]))
    leftover_pred = pred.loc[
        ~pred.apply(
            lambda r: (r["home_name"], r["away_name"], r["game_date"]) in used_pred,
            axis=1,
        )
    ]
    leftover_espn = mapped.loc[~mapped["espn_id"].astype(str).isin(used_espn)].copy()
    extra = []
    for _, row in leftover_pred.iterrows():
        cand = leftover_espn[
            (leftover_espn["home_name"] == row["home_name"])
            & (leftover_espn["away_name"] == row["away_name"])
        ]
        if cand.empty:
            continue
        cand = cand.assign(_dd=(cand["fd_date"] - row["game_date"]).dt.days.abs())
        cand = cand.loc[cand["_dd"] <= 1]
        if cand.empty:
            continue
        best = cand.sort_values("_dd").iloc[0]
        rec = {**row.to_dict(), **best.drop(labels=["_dd"]).to_dict()}
        extra.append(rec)
        leftover_espn = leftover_espn.loc[
            leftover_espn["espn_id"].astype(str) != str(best["espn_id"])
        ]
    if extra:
        merged = pd.concat([merged, pd.DataFrame(extra)], ignore_index=True)
    if skipped:
        print(f"  ESPN events with unmapped teams: {skipped}")
    return merged


def attach_open_close(pred: pd.DataFrame, cache: pd.DataFrame | None = None) -> pd.DataFrame:
    """Left-join ESPN BET / DraftKings open+close 1X2 onto a predictions table.

    ESPN 1X2 coverage starts around July 2025; earlier games stay missing.
    """
    out = pred.copy()
    out["game_date"] = pd.to_datetime(out["game_date"])
    extra = ["OpenH", "OpenD", "OpenA", "CloseH", "CloseD", "CloseA", "espn_book"]
    espn = load_odds_cache(ODDS_CACHE) if cache is None else cache
    if espn.empty or "game_id" not in out.columns:
        for col in extra:
            if col not in out.columns:
                out[col] = pd.NA
        return out
    cutoff = pd.Timestamp("2025-07-01")
    recent = out.loc[out["game_date"] >= cutoff].copy()
    older = out.loc[out["game_date"] < cutoff].copy()
    if recent.empty:
        for col in extra:
            out[col] = pd.NA
        return out
    known = set(recent["home_name"].astype(str)) | set(recent["away_name"].astype(str))
    joined = join_predictions(recent, espn, known)
    keep = ["game_id", "open_H", "open_D", "open_A", "close_H", "close_D", "close_A"]
    if "book" in joined.columns:
        keep.append("book")
    have = [c for c in keep if c in joined.columns]
    slim = joined[have].drop_duplicates("game_id")
    rename = {
        "open_H": "OpenH",
        "open_D": "OpenD",
        "open_A": "OpenA",
        "close_H": "CloseH",
        "close_D": "CloseD",
        "close_A": "CloseA",
        "book": "espn_book",
    }
    slim = slim.rename(columns={k: v for k, v in rename.items() if k in slim.columns})
    slim_cols = [c for c in slim.columns if c != "game_id"]
    drop_existing = [c for c in slim_cols if c in recent.columns]
    if drop_existing:
        recent = recent.drop(columns=drop_existing)
    recent = recent.merge(slim, on="game_id", how="left")
    for col in extra:
        if col not in recent.columns:
            recent[col] = pd.NA
        if col not in older.columns:
            older[col] = pd.NA
    return pd.concat([older, recent], ignore_index=True)


def pick_from(
    df: pd.DataFrame,
    fair: dict[str, str],
    odds: dict[str, str],
    edge: float,
    sides: list[str],
) -> pd.DataFrame:
    out = df.copy()
    bet_side: list[str | None] = []
    bet_odds: list[float | None] = []
    bet_won: list[bool | None] = []
    bet_profit: list[float] = []
    bet_edge: list[float | None] = []
    close_odds: list[float | None] = []
    for _, row in out.iterrows():
        best_side = None
        best_edge = -1e9
        for side in sides:
            e = float(row[MODEL[side]]) - float(row[fair[side]])
            if e > best_edge:
                best_edge = e
                best_side = side
        if best_side is None or best_edge < edge:
            bet_side.append(None)
            bet_odds.append(None)
            bet_won.append(None)
            bet_profit.append(0.0)
            bet_edge.append(None)
            close_odds.append(None)
            continue
        o = float(row[odds[best_side]])
        won = str(row["Res"]) == best_side
        bet_side.append(best_side)
        bet_odds.append(o)
        bet_won.append(won)
        bet_profit.append(flat_bet_profit(o, won))
        bet_edge.append(best_edge)
        close_odds.append(float(row[ODDS_CLOSE[best_side]]))
    out["bet_side"] = bet_side
    out["bet_odds"] = bet_odds
    out["bet_won"] = bet_won
    out["bet_profit"] = bet_profit
    out["bet_edge"] = bet_edge
    out["close_odds_same_side"] = close_odds
    return out


def roi_summary(bets: pd.DataFrame, edge: float, label: str) -> dict:
    placed = bets.dropna(subset=["bet_side"])
    if placed.empty:
        return {
            "book": label,
            "edge": edge,
            "n": 0,
            "hit": float("nan"),
            "profit": 0.0,
            "roi": float("nan"),
            "clv": float("nan"),
        }
    clv = np.nan
    if "close_odds_same_side" in placed.columns:
        ok = placed.dropna(subset=["bet_odds", "close_odds_same_side"])
        if not ok.empty:
            clv = float((ok["bet_odds"] / ok["close_odds_same_side"] - 1.0).mean())
    return {
        "book": label,
        "edge": edge,
        "n": int(len(placed)),
        "hit": float(placed["bet_won"].mean()),
        "profit": float(placed["bet_profit"].sum()),
        "roi": float(placed["bet_profit"].sum() / len(placed)),
        "clv": clv,
    }


def quality_row(df: pd.DataFrame, name: str, p_cols: list[str]) -> dict:
    y = df["Res"]
    y_idx = y.map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    p = df[p_cols].to_numpy(dtype=float)
    pick = df[p_cols].idxmax(axis=1).map(
        {p_cols[0]: "H", p_cols[1]: "D", p_cols[2]: "A"}
    )
    return {
        "price": name,
        "n": int(len(df)),
        "accuracy": float(accuracy_score(y, pick)),
        "log_loss": float(log_loss(y_idx, p, labels=[0, 1, 2])),
        "mean_p_H": float(df[p_cols[0]].mean()),
        "mean_p_D": float(df[p_cols[1]].mean()),
        "mean_p_A": float(df[p_cols[2]].mean()),
    }


def movement_stats(df: pd.DataFrame) -> dict:
    moved_toward = 0
    n = 0
    abs_open = []
    abs_close = []
    abs_avgc = []
    for _, row in df.iterrows():
        for side in CLASS_ORDER:
            pm = float(row[MODEL[side]])
            po = float(row[FAIR_OPEN[side]])
            pc = float(row[FAIR_CLOSE[side]])
            pa = float(row[FAIR_AVGC[side]])
            abs_open.append(abs(pm - po))
            abs_close.append(abs(pm - pc))
            abs_avgc.append(abs(pm - pa))
            delta_mkt = pc - po
            delta_model = pm - po
            if abs(delta_mkt) < 1e-9:
                continue
            n += 1
            if delta_mkt * delta_model > 0:
                moved_toward += 1
    return {
        "mean_abs_model_vs_open": float(np.mean(abs_open)),
        "mean_abs_model_vs_espn_close": float(np.mean(abs_close)),
        "mean_abs_model_vs_avgc": float(np.mean(abs_avgc)),
        "close_moved_toward_model": moved_toward / n if n else float("nan"),
        "n_side_moves": n,
    }


def print_quality(rows: list[dict]) -> None:
    print("\nProbability quality (1X2, same games):")
    print(f"  {'price':22s}  {'n':>4}  acc     log-loss  mean P(H/D/A)")
    for r in rows:
        print(
            f"  {r['price']:22s}  {r['n']:4d}  {r['accuracy']:.3f}  "
            f"{r['log_loss']:.3f}     "
            f"{r['mean_p_H']:.3f}/{r['mean_p_D']:.3f}/{r['mean_p_A']:.3f}"
        )


def print_roi(rows: list[dict]) -> None:
    print("\nFlat 1u, best of H/D/A:")
    print(f"  {'vs':28s}  edge   n   hit     P/L    ROI    CLV vs ESPN close")
    for r in rows:
        clv = "—" if not np.isfinite(r["clv"]) else f"{r['clv']:+.1%}"
        hit = "—" if not np.isfinite(r["hit"]) else f"{r['hit']:.1%}"
        roi = "—" if not np.isfinite(r["roi"]) else f"{r['roi']:+.1%}"
        print(
            f"  {r['book']:28s}  {r['edge']:.2f}  {r['n']:3d}  {hit:>6}  "
            f"{r['profit']:+6.1f}  {roi:>6}  {clv}"
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Model vs ESPN BET open/close 1X2.")
    p.add_argument(
        "--predictions",
        type=str,
        default=str(DEFAULT_PREDICTIONS),
        help="Bayesian DC predictions CSV (must have p_*_model and AvgC fair cols).",
    )
    p.add_argument("--refresh", action="store_true", help="Re-fetch ESPN odds.")
    p.add_argument("--pause", type=float, default=0.08, help="Seconds between odds calls.")
    p.add_argument("--edge", type=str, default="0,0.05,0.10")
    p.add_argument(
        "--since",
        type=str,
        default="",
        help="Only fetch/join games on or after this date (YYYY-MM-DD). ESPN 1X2 starts ~Jul 2025.",
    )
    args = p.parse_args()

    pred_path = Path(args.predictions)
    if not pred_path.is_file():
        raise SystemExit(f"Missing {pred_path}")
    pred = pd.read_csv(pred_path)
    pred["game_date"] = pd.to_datetime(pred["game_date"])
    needed = [
        "game_date",
        "home_name",
        "away_name",
        "Res",
        "p_home_model",
        "p_draw_model",
        "p_away_model",
        "p_home_fair",
        "p_draw_fair",
        "p_away_fair",
    ]
    missing = [c for c in needed if c not in pred.columns]
    if missing:
        raise SystemExit(f"{pred_path.name} missing {missing}")
    matches_csv = DATA_DIR / "liga_mx_matches.csv"
    if any(c not in pred.columns for c in ("AvgCH", "AvgCD", "AvgCA")):
        if not matches_csv.is_file():
            raise SystemExit(f"Need AvgCH/D/A on the predictions file or {matches_csv}")
        matches = pd.read_csv(matches_csv, usecols=["game_id", "AvgCH", "AvgCD", "AvgCA"])
        if "game_id" not in pred.columns:
            raise SystemExit("Predictions need game_id to join closing AvgC odds.")
        pred = pred.merge(matches, on="game_id", how="left")
    pred = pred.dropna(subset=needed + ["AvgCH", "AvgCD", "AvgCA"]).reset_index(drop=True)
    if args.since:
        since = pd.Timestamp(args.since)
        pred = pred.loc[pred["game_date"] >= since].reset_index(drop=True)
        if pred.empty:
            raise SystemExit(f"No predictions on or after {args.since}.")
    known = set(pred["home_name"]) | set(pred["away_name"])
    start = pred["game_date"].min() - timedelta(days=1)
    end = pred["game_date"].max() + timedelta(days=1)
    print(f"Predictions: {pred_path}  n={len(pred)}  {start.date()} → {end.date()}")

    print("Fetching ESPN Liga MX events...")
    events = fetch_espn_events(start, end)
    print(f"  events={len(events)}")
    cache = pd.DataFrame() if args.refresh else load_odds_cache(ODDS_CACHE)
    print(f"Fetching ESPN BET open/close ({'refresh' if args.refresh else 'cache+' + str(len(cache))} )...")
    espn = collect_open_close(events, cache, refresh=args.refresh, pause=args.pause)
    odds_cols = ["espn_id", "kickoff_utc", "espn_home", "espn_away", "book",
                 "open_H", "open_D", "open_A", "close_H", "close_D", "close_A"]
    save = espn[[c for c in odds_cols if c in espn.columns]].drop_duplicates("espn_id")
    ODDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    save.to_csv(ODDS_CACHE, index=False)
    print(f"  wrote {ODDS_CACHE}  rows={len(save)}")

    joined = join_predictions(pred, espn, known)
    have_odds = joined.dropna(
        subset=["open_H", "open_D", "open_A", "close_H", "close_D", "close_A"]
    ).copy()
    print(
        f"Joined {len(joined)} / {len(pred)} games; "
        f"{len(have_odds)} with full ESPN open+close 1X2"
    )
    if have_odds.empty:
        raise SystemExit("No overlapping games with ESPN open and close odds.")

    have_odds = add_fair(have_odds, ODDS_OPEN, "open")
    have_odds = add_fair(have_odds, ODDS_CLOSE, "espn_close")
    # add_fair used p_home_open etc; rename to FAIR_OPEN keys already match.
    # close prefix espn_close → p_home_espn_close. Good.

    qrows = [
        quality_row(have_odds, "model", ["p_home_model", "p_draw_model", "p_away_model"]),
        quality_row(have_odds, "ESPN BET open", ["p_home_open", "p_draw_open", "p_away_open"]),
        quality_row(
            have_odds,
            "ESPN BET close",
            ["p_home_espn_close", "p_draw_espn_close", "p_away_espn_close"],
        ),
        quality_row(have_odds, "football-data AvgC", ["p_home_fair", "p_draw_fair", "p_away_fair"]),
    ]
    print_quality(qrows)

    mv = movement_stats(have_odds)
    print("\nModel vs market (mean |P_model − P_fair| across H/D/A):")
    print(f"  vs ESPN open:        {mv['mean_abs_model_vs_open']:.3f}")
    print(f"  vs ESPN close:       {mv['mean_abs_model_vs_espn_close']:.3f}")
    print(f"  vs football-data AvgC: {mv['mean_abs_model_vs_avgc']:.3f}")
    print(
        f"  When the ESPN 1X2 moved, close went toward the model "
        f"{mv['close_moved_toward_model']:.1%} of side-moves "
        f"(n={mv['n_side_moves']})."
    )
    books = (
        have_odds["book"].value_counts().to_dict()
        if "book" in have_odds.columns
        else {}
    )
    print(f"  Books: {books}")

    have_odds = have_odds.copy()
    have_odds["window"] = np.where(
        have_odds["game_date"] < pd.Timestamp("2026-01-01"),
        "Apertura 2025",
        np.where(
            have_odds["game_date"] < pd.Timestamp("2026-07-01"),
            "Clausura 2026",
            "Apertura 2026",
        ),
    )

    edges = [float(x) for x in args.edge.split(",") if x.strip()]
    roi_rows = []
    for edge in edges:
        open_bets = pick_from(have_odds, FAIR_OPEN, ODDS_OPEN, edge, CLASS_ORDER)
        close_bets = pick_from(have_odds, FAIR_CLOSE, ODDS_CLOSE, edge, CLASS_ORDER)
        avgc_bets = pick_from(have_odds, FAIR_AVGC, {"H": "AvgCH", "D": "AvgCD", "A": "AvgCA"}, edge, CLASS_ORDER)
        roi_rows.append(roi_summary(open_bets, edge, "ESPN BET open"))
        roi_rows.append(roi_summary(close_bets, edge, "ESPN BET close"))
        roi_rows.append(roi_summary(avgc_bets, edge, "football-data AvgC"))
    print_roi(roi_rows)

    print("\nBy tournament (edge ≥ 5%, best of H/D/A):")
    for window, g in have_odds.groupby("window", sort=True):
        open_bets = pick_from(g, FAIR_OPEN, ODDS_OPEN, 0.05, CLASS_ORDER)
        avgc_bets = pick_from(
            g, FAIR_AVGC, {"H": "AvgCH", "D": "AvgCD", "A": "AvgCA"}, 0.05, CLASS_ORDER
        )
        r_open = roi_summary(open_bets, 0.05, "open")
        r_avg = roi_summary(avgc_bets, 0.05, "avgc")
        print(
            f"  {window:16s}  n={len(g):3d}  "
            f"open {r_open['n']:3d} bets ROI={r_open['roi']:+.1%}  "
            f"AvgC {r_avg['n']:3d} bets ROI={r_avg['roi']:+.1%}"
        )

    out = have_odds.copy()
    out["edge_open_H"] = out["p_home_model"] - out["p_home_open"]
    out["edge_open_D"] = out["p_draw_model"] - out["p_draw_open"]
    out["edge_open_A"] = out["p_away_model"] - out["p_away_open"]
    out["edge_espn_close_H"] = out["p_home_model"] - out["p_home_espn_close"]
    out["edge_espn_close_D"] = out["p_draw_model"] - out["p_draw_espn_close"]
    out["edge_espn_close_A"] = out["p_away_model"] - out["p_away_espn_close"]
    keep = [
        c
        for c in [
            "game_id",
            "game_date",
            "season",
            "home_name",
            "away_name",
            "Res",
            "book",
            "open_H",
            "open_D",
            "open_A",
            "close_H",
            "close_D",
            "close_A",
            "AvgCH",
            "AvgCD",
            "AvgCA",
            "p_home_model",
            "p_draw_model",
            "p_away_model",
            "p_home_open",
            "p_draw_open",
            "p_away_open",
            "p_home_espn_close",
            "p_draw_espn_close",
            "p_away_espn_close",
            "p_home_fair",
            "p_draw_fair",
            "p_away_fair",
            "edge_open_H",
            "edge_open_D",
            "edge_open_A",
            "edge_espn_close_H",
            "edge_espn_close_D",
            "edge_espn_close_A",
            "edge_H",
            "edge_D",
            "edge_A",
        ]
        if c in out.columns
    ]
    dest = TABLES_DIR / "open_close_vs_model.csv"
    out[keep].to_csv(dest, index=False)
    pd.DataFrame(qrows).to_csv(TABLES_DIR / "open_close_quality.csv", index=False)
    pd.DataFrame(roi_rows).to_csv(TABLES_DIR / "open_close_edge_roi.csv", index=False)
    print(f"\nWrote {dest}")
    print(f"Wrote {TABLES_DIR / 'open_close_quality.csv'}")
    print(f"Wrote {TABLES_DIR / 'open_close_edge_roi.csv'}")


if __name__ == "__main__":
    main()
