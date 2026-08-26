"""
Score upcoming Liga MX games with Bayesian Dixon-Coles (fair 1X2).

Fits on all completed matches in data/liga_mx_matches.csv (warm-starts from a
saved model when possible), pulls the next ESPN slate + live 1X2, and
writes model probabilities / fair odds vs the current book.

Usage:
  python scripts/predict_upcoming.py
  python scripts/predict_upcoming.py --dates 20260815,20260816,20260817
  python scripts/predict_upcoming.py --refit
  python scripts/predict_upcoming.py --map-only
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests

from bayesian_dc_utils import (
    DEFAULT_N_POSTERIOR,
    DEFAULT_SIGMA_ATT,
    DEFAULT_SIGMA_DEF,
    DEFAULT_XI,
    BayesianDCModel,
)
from dixon_coles_utils import is_legacy_dc_model
from fit_bayesian_dc import fit_on_history, load_matches
from form_utils import (
    DEFAULT_FORM_GAMES,
    DEFAULT_FORM_HALFLIFE,
    DEFAULT_FORM_SCALE,
    DEFAULT_FORM_WEIGHT,
    predict_match_with_form,
    team_form,
)
from odds_utils import implied_prob, remove_vig_3way

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
MODELS_DIR = PROJECT_ROOT / "results" / "models"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
LIVE_MODEL = MODELS_DIR / "bayesian_dc_model_live.pkl"

ESPN_LEAGUE = "mex.1"
ESPN_SCOREBOARD = f"https://site.api.espn.com/apis/site/v2/sports/soccer/{ESPN_LEAGUE}/scoreboard"
ESPN_HEADER = "https://site.web.api.espn.com/apis/v2/scoreboard/header"
ESPN_CORE_ODDS = (
    "https://sports.core.api.espn.com/v2/sports/soccer/leagues/"
    f"{ESPN_LEAGUE}/events/{{eid}}/competitions/{{eid}}/odds"
)
DISPLAY_TZ = "America/Mexico_City"
FIXTURES_CSV = PROJECT_ROOT / "data" / "upcoming_fixtures.csv"
ODDS_CACHE_HOURS = 12
ESPN_UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/plain,*/*",
    "Referer": "https://www.espn.com/soccer/",
}

# ESPN / common names → football-data MEX.csv names
NAME_ALIASES = {
    "América": "Club America",
    "Club América": "Club America",
    "CF América": "Club America",
    "León": "Club Leon",
    "Club León": "Club Leon",
    "FC Juarez": "Juarez",
    "FC Juárez": "Juarez",
    "Juárez": "Juarez",
    "Querétaro": "Queretaro",
    "Querétaro FC": "Queretaro",
    "Atlético San Luis": "Atl. San Luis",
    "Atletico San Luis": "Atl. San Luis",
    "Atlético de San Luis": "Atl. San Luis",
    "Atletico de San Luis": "Atl. San Luis",
    "San Luis": "Atl. San Luis",
    "Chivas": "Guadalajara Chivas",
    "Guadalajara": "Guadalajara Chivas",
    "CD Guadalajara": "Guadalajara Chivas",
    "Pumas": "UNAM Pumas",
    "Pumas UNAM": "UNAM Pumas",
    "Tigres": "Tigres UANL",
    "Tijuana": "Club Tijuana",
    "Xolos": "Club Tijuana",
    "Club Tijuana Xoloitzcuintles": "Club Tijuana",
    "Mazatlán": "Mazatlan FC",
    "Mazatlán FC": "Mazatlan FC",
    "Santos": "Santos Laguna",
    "Morelia": "Monarcas",
    "Atlante FC": "Atlante",
    "CF Atlante": "Atlante",
}


def _strip_accents(s: str) -> str:
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def _norm_name(s: str) -> str:
    s = _strip_accents(s).lower()
    s = s.replace("&", "and").replace(".", " ")
    s = re.sub(r"\b(fc|sc|cf|cd|de|club)\b", " ", s)
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())


def map_team(name: str, known: set[str]) -> str:
    if name in known:
        return name
    if name in NAME_ALIASES and NAME_ALIASES[name] in known:
        return NAME_ALIASES[name]
    target = _norm_name(name)
    hits = [t for t in known if _norm_name(t) == target]
    if len(hits) == 1:
        return hits[0]
    # looser: token containment
    hits = [t for t in known if target and (target in _norm_name(t) or _norm_name(t) in target)]
    if len(hits) == 1:
        return hits[0]
    raise KeyError(f"Cannot map team {name!r} to football-data names")


def american_to_decimal(odds) -> float:
    if odds is None or (isinstance(odds, float) and np.isnan(odds)):
        return float("nan")
    s = str(odds).strip().upper().replace("−", "-")
    if s in {"EVEN", "EV", "PK"}:
        return 2.0
    n = float(s)
    if n > 0:
        return 1.0 + n / 100.0
    if n < 0:
        return 1.0 + 100.0 / abs(n)
    return float("nan")


def _finite_odds(value) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(x) and x > 1.0)


def _ml_american(block: dict | None, side: str) -> str | None:
    if not block:
        return None
    side_block = block.get(side) or {}
    for key in ("close", "open"):
        odds = (side_block.get(key) or {}).get("odds")
        if odds:
            return str(odds)
    return None


def _ml_snap(block: dict | None, side: str, snap: str) -> str | None:
    if not block:
        return None
    odds = ((block.get(side) or {}).get(snap) or {}).get("odds")
    return str(odds) if odds is not None else None


def _espn_header_events(dates: list[str]) -> list[dict]:
    """site.web.api header scoreboard — works when site.api scoreboard 403s."""
    lo, hi = min(dates), max(dates)
    url = f"{ESPN_HEADER}?sport=soccer&league={ESPN_LEAGUE}&dates={lo}-{hi}"
    r = requests.get(url, headers=ESPN_UA, timeout=30)
    r.raise_for_status()
    sports = (r.json().get("sports") or [])
    if not sports:
        return []
    leagues = sports[0].get("leagues") or []
    if not leagues:
        return []
    return list(leagues[0].get("events") or [])


def _espn_scoreboard_events(dates: list[str]) -> list[dict]:
    """Legacy site.api scoreboard (often 403)."""
    payloads = []
    for d in dates:
        url = f"{ESPN_SCOREBOARD}?dates={d}&limit=50"
        r = requests.get(url, headers=ESPN_UA, timeout=30)
        if r.status_code == 403:
            break
        r.raise_for_status()
        payloads.append(r.json())
    if not payloads and len(dates) >= 1:
        lo, hi = min(dates), max(dates)
        url = f"{ESPN_SCOREBOARD}?dates={lo}-{hi}&limit=50"
        r = requests.get(url, headers=ESPN_UA, timeout=30)
        r.raise_for_status()
        payloads.append(r.json())
    events = []
    seen = set()
    for payload in payloads:
        for event in payload.get("events") or []:
            eid = event.get("id")
            if eid in seen:
                continue
            seen.add(eid)
            events.append(event)
    return events


def _espn_events(dates: list[str]) -> list[dict]:
    """Prefer the header API; fall back to site.api scoreboard."""
    try:
        events = _espn_header_events(dates)
        if events:
            return events
    except Exception:
        events = []
    return _espn_scoreboard_events(dates)


def _competitor_name(comp: dict) -> str | None:
    return (
        comp.get("displayName")
        or (comp.get("team") or {}).get("displayName")
        or comp.get("name")
    )


def _header_ml(odds: dict, side: str) -> str | None:
    if not odds:
        return None
    direct = (odds.get(side) or {}).get("moneyLine")
    if direct is not None:
        return str(direct)
    key = {"home": "homeTeamOdds", "away": "awayTeamOdds", "draw": "drawOdds"}.get(side)
    nested = (odds.get(key) or {}).get("moneyLine") if key else None
    if nested is not None:
        return str(nested)
    ml = odds.get("moneyline") or {}
    close = ((ml.get(side) or {}).get("close") or {}).get("odds")
    if close is not None:
        return str(close)
    open_ = ((ml.get(side) or {}).get("open") or {}).get("odds")
    return str(open_) if open_ is not None else None


def _header_open_ml(odds: dict, side: str) -> str | None:
    if not odds:
        return None
    return _ml_snap(odds.get("moneyline") or {}, side, "open")


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
    if snap in {"close", "current"} and block.get("moneyLine") is not None:
        ml = block.get("moneyLine")
        if isinstance(ml, dict):
            return _price_decimal(ml)
        return american_to_decimal(ml)
    return float("nan")


def _parse_core_item(item: dict) -> dict | None:
    name = str((item.get("provider") or {}).get("name") or "")
    if "live" in name.lower():
        return None
    home = item.get("homeTeamOdds") or {}
    away = item.get("awayTeamOdds") or {}
    top_open = item.get("open") or {}
    top_live = item.get("current") or item.get("close") or {}
    oh = _block_ml_decimal(home, "open")
    od = _price_decimal(top_open.get("draw") or {})
    oa = _block_ml_decimal(away, "open")
    lh = _block_ml_decimal(home, "current")
    if not _finite_odds(lh):
        lh = _block_ml_decimal(home, "close")
    ld = _price_decimal(top_live.get("draw") or {})
    if not _finite_odds(ld):
        ld = _block_ml_decimal(item.get("drawOdds") or {}, "close")
    la = _block_ml_decimal(away, "current")
    if not _finite_odds(la):
        la = _block_ml_decimal(away, "close")
    if not any(_finite_odds(x) for x in (oh, od, oa, lh, ld, la)):
        return None
    return {
        "book": name or None,
        "odds_open_H": oh,
        "odds_open_D": od,
        "odds_open_A": oa,
        "odds_H": lh,
        "odds_D": ld,
        "odds_A": la,
    }


def fetch_core_open_live(espn_id: str) -> dict | None:
    url = ESPN_CORE_ODDS.format(eid=espn_id)
    r = requests.get(url, headers=ESPN_UA, timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    parsed: list[dict] = []
    for item in r.json().get("items") or []:
        row = _parse_core_item(item)
        if row is not None:
            parsed.append(row)
    if not parsed:
        return None
    parsed.sort(key=lambda t: 0 if str(t.get("book") or "").lower() == "espn bet" else 1)
    return parsed[0]


def fill_open_odds(slate: pd.DataFrame) -> pd.DataFrame:
    """Fill missing opening 1X2 from the ESPN core odds API."""
    out = slate.copy()
    for col in ("odds_open_H", "odds_open_D", "odds_open_A"):
        if col not in out.columns:
            out[col] = np.nan
    if "espn_id" not in out.columns or out.empty:
        return out
    missing = []
    for idx, row in out.iterrows():
        if not all(_finite_odds(row.get(c)) for c in ("odds_open_H", "odds_open_D", "odds_open_A")):
            missing.append(idx)
    for idx in missing:
        eid = out.at[idx, "espn_id"]
        if eid is None or (isinstance(eid, float) and np.isnan(eid)):
            continue
        try:
            extra = fetch_core_open_live(str(eid).replace(".0", ""))
        except requests.RequestException:
            extra = None
        if not extra:
            continue
        for col in ("odds_open_H", "odds_open_D", "odds_open_A"):
            if not _finite_odds(out.at[idx, col]) and _finite_odds(extra.get(col)):
                out.at[idx, col] = extra[col]
        for col in ("odds_H", "odds_D", "odds_A"):
            if col in out.columns and not _finite_odds(out.at[idx, col]) and _finite_odds(extra.get(col)):
                out.at[idx, col] = extra[col]
        if not out.at[idx, "book"] and extra.get("book"):
            out.at[idx, "book"] = extra["book"]
        time.sleep(0.05)
    return out


def _parse_event(event: dict) -> dict | None:
    """Normalize header-API or site.api scoreboard events into one row."""
    competitors = event.get("competitors") or []
    odds_obj: dict = {}
    status_type: dict = {}
    if competitors:
        odds_raw = event.get("odds")
        odds_obj = odds_raw if isinstance(odds_raw, dict) else {}
        status_type = (event.get("fullStatus") or {}).get("type") or {}
        if not status_type and event.get("status"):
            status_type = {
                "name": str(event.get("summary") or event.get("status")),
                "completed": str(event.get("status")).lower() in {"post", "final"},
            }
        ah = _header_ml(odds_obj, "home")
        ad = _header_ml(odds_obj, "draw")
        aa = _header_ml(odds_obj, "away")
        oh = _header_open_ml(odds_obj, "home")
        od = _header_open_ml(odds_obj, "draw")
        oa = _header_open_ml(odds_obj, "away")
        book = (odds_obj.get("provider") or {}).get("name")
    else:
        comps = (event.get("competitions") or [{}])[0]
        competitors = comps.get("competitors") or []
        status_type = (event.get("status") or {}).get("type") or {}
        book = None
        ah = ad = aa = None
        oh = od = oa = None
        odds_list = comps.get("odds") or []
        if odds_list:
            o0 = odds_list[0]
            book = (o0.get("provider") or {}).get("name")
            ml = o0.get("moneyline") or {}
            ah = _ml_american(ml, "home")
            ad = _ml_american(ml, "draw")
            aa = _ml_american(ml, "away")
            oh = _ml_snap(ml, "home", "open")
            od = _ml_snap(ml, "draw", "open")
            oa = _ml_snap(ml, "away", "open")
            if ah is None and o0.get("homeTeamOdds"):
                ah = (o0.get("homeTeamOdds") or {}).get("moneyLine")
            if aa is None and o0.get("awayTeamOdds"):
                aa = (o0.get("awayTeamOdds") or {}).get("moneyLine")
            if ad is None and o0.get("drawOdds"):
                ad = (o0.get("drawOdds") or {}).get("moneyLine")

    home = away = None
    for c in competitors:
        display = _competitor_name(c)
        if c.get("homeAway") == "home":
            home = display
        else:
            away = display
    if not home or not away:
        return None
    kick = pd.to_datetime(event.get("date"), utc=True)
    return {
        "espn_id": event.get("id"),
        "kickoff_utc": kick,
        "espn_home": home,
        "espn_away": away,
        "status": status_type.get("name") or status_type.get("description") or event.get("summary") or "",
        "completed": bool(status_type.get("completed")),
        "book": book,
        "odds_H": american_to_decimal(ah),
        "odds_D": american_to_decimal(ad),
        "odds_A": american_to_decimal(aa),
        "odds_open_H": american_to_decimal(oh),
        "odds_open_D": american_to_decimal(od),
        "odds_open_A": american_to_decimal(oa),
    }


def load_fixtures_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["kickoff_utc"] = pd.to_datetime(df["kickoff_utc"], utc=True)
    now = pd.Timestamp.now(tz="UTC")
    df["completed"] = df["kickoff_utc"] < now
    if "status" not in df.columns:
        df["status"] = np.where(df["completed"], "Final", "Scheduled")
    else:
        df.loc[df["completed"] & df["status"].isna(), "status"] = "Final"
    return df.sort_values("kickoff_utc").reset_index(drop=True)


def fetch_espn_slate(dates: list[str]) -> pd.DataFrame:
    """Pull ESPN events (and 1X2 when present) for the given YYYYMMDD dates."""
    wanted = {datetime.strptime(d, "%Y%m%d").date() for d in dates}
    rows = []
    seen = set()
    for event in _espn_events(dates):
        eid = event.get("id")
        if eid in seen:
            continue
        seen.add(eid)
        row = _parse_event(event)
        if row is None:
            continue
        kick_local = row["kickoff_utc"].tz_convert(DISPLAY_TZ).date()
        if kick_local not in wanted:
            continue
        rows.append(row)
    if not rows:
        raise RuntimeError(f"No ESPN Liga MX events for dates {dates}")
    return fill_open_odds(pd.DataFrame(rows).sort_values("kickoff_utc").reset_index(drop=True))


def _cache_age_hours(path: Path) -> float | None:
    if not path.is_file():
        return None
    mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() / 3600.0


def write_fixtures_cache(slate: pd.DataFrame, path: Path = FIXTURES_CSV) -> None:
    cols = [
        "espn_id",
        "kickoff_utc",
        "espn_home",
        "espn_away",
        "status",
        "completed",
        "book",
        "odds_H",
        "odds_D",
        "odds_A",
        "odds_open_H",
        "odds_open_D",
        "odds_open_A",
    ]
    out = slate.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = pd.NA
    path.parent.mkdir(parents=True, exist_ok=True)
    out[cols].to_csv(path, index=False)


def load_cached_upcoming(path: Path = FIXTURES_CSV, *, max_age_hours: float = ODDS_CACHE_HOURS) -> pd.DataFrame | None:
    age = _cache_age_hours(path)
    if age is None or age > max_age_hours:
        return None
    df = load_fixtures_csv(path)
    if not {"odds_open_H", "odds_open_D", "odds_open_A"}.issubset(df.columns):
        return None
    live = df.loc[~df["completed"].astype(bool)]
    if live.empty:
        return None
    return df


def fetch_upcoming_slate(n_days: int = 4, *, use_cache: bool = True) -> pd.DataFrame:
    """
    Next Liga MX games in a window starting today.

    Uses a 12h fixtures cache when it still has unplayed games. If the requested
    span is empty (midweek gap), looks ahead up to 14 days so Wednesday/weekend
    slates still appear.
    """
    if use_cache:
        cached = load_cached_upcoming()
        if cached is not None:
            live = cached.loc[~cached["completed"].astype(bool)]
            if not live.empty:
                return live.reset_index(drop=True)

    last_err: Exception | None = None
    for span in range(max(n_days, 1), 15):
        dates = default_dates(span)
        try:
            slate = fetch_espn_slate(dates)
        except Exception as exc:
            last_err = exc
            continue
        live = slate.loc[~slate["completed"].astype(bool)].copy()
        if live.empty:
            continue
        write_fixtures_cache(live)
        return live.reset_index(drop=True)

    cached = load_fixtures_csv(FIXTURES_CSV) if FIXTURES_CSV.exists() else None
    if cached is not None:
        live = cached.loc[~cached["completed"].astype(bool)]
        if not live.empty:
            return live.reset_index(drop=True)
    if last_err is not None:
        raise last_err
        raise RuntimeError("No upcoming Liga MX games found in the next 14 days.")


def load_or_fit(
    hist: pd.DataFrame,
    *,
    refit: bool,
    map_only: bool,
    n_posterior: int,
    xi: float,
    sigma_att: float,
    sigma_def: float,
) -> BayesianDCModel:
    if (not refit) and LIVE_MODEL.exists():
        try:
            saved = joblib.load(LIVE_MODEL)
            if is_legacy_dc_model(saved):
                print("Saved live model has no league intercept; refitting.")
            elif abs(float(getattr(saved, "xi", xi)) - xi) > 1e-9:
                print(
                    f"Saved live model xi={getattr(saved, 'xi', None)} != {xi}; refitting."
                )
            else:
                print(f"Using saved live model ({LIVE_MODEL.name}). Pass --refit to rebuild.")
                return saved
        except Exception as exc:
            print(f"Could not load {LIVE_MODEL.name}: {exc}")

    prev = None
    for path in (LIVE_MODEL, MODELS_DIR / "bayesian_dc_model_2026.pkl", MODELS_DIR / "bayesian_dc_model.pkl"):
        if path.exists():
            try:
                prev = joblib.load(path)
                print(f"Warm-start from {path.name}")
                break
            except Exception as exc:
                print(f"Could not load {path.name}: {exc}")

    print(f"Fitting Bayesian DC on {len(hist)} completed matches (through {hist['game_date'].max().date()})...")
    t0 = time.time()
    model = fit_on_history(
        hist,
        xi=xi,
        sigma_att=sigma_att,
        sigma_def=sigma_def,
        n_posterior=n_posterior,
        compute_laplace=not map_only,
        prev=prev,
    )
    print(
        f"  Done in {time.time() - t0:.1f}s  "
        f"intercept={model.intercept:.3f}  home_adv={model.home_adv:.3f}  "
        f"rho={model.rho:.3f}  teams={len(model.teams)}"
    )
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, LIVE_MODEL)
    print(f"  Saved {LIVE_MODEL}")
    return model


def score_slate(
    slate: pd.DataFrame,
    model: BayesianDCModel,
    known: set[str],
    *,
    use_posterior: bool,
    hist: pd.DataFrame | None = None,
    form_weight: float = DEFAULT_FORM_WEIGHT,
    form_games: int = DEFAULT_FORM_GAMES,
    form_halflife: float = DEFAULT_FORM_HALFLIFE,
    form_scale: float = DEFAULT_FORM_SCALE,
) -> pd.DataFrame:
    form: dict[str, tuple[float, float]] = {}
    if form_weight > 0.0 and hist is not None and not hist.empty:
        slate_teams = {
            map_team(str(row["espn_home"]), known) for _, row in slate.iterrows()
        } | {
            map_team(str(row["espn_away"]), known) for _, row in slate.iterrows()
        }
        form = team_form(
            hist,
            model,
            n_games=form_games,
            halflife=form_halflife,
            teams=slate_teams,
        )
    rows = []
    for _, row in slate.iterrows():
        home = map_team(str(row["espn_home"]), known)
        away = map_team(str(row["espn_away"]), known)
        pred = predict_match_with_form(
            model,
            home,
            away,
            form,
            weight=form_weight,
            scale=form_scale,
            use_posterior=use_posterior,
        )
        p_h, p_d, p_a = pred["p_home"], pred["p_draw"], pred["p_away"]
        fair_h, fair_d, fair_a = 1.0 / p_h, 1.0 / p_d, 1.0 / p_a

        def _odds(key: str) -> float:
            try:
                return float(row[key])
            except (KeyError, TypeError, ValueError):
                return float("nan")

        oh, od, oa = _odds("odds_H"), _odds("odds_D"), _odds("odds_A")
        ooh, ood, ooa = _odds("odds_open_H"), _odds("odds_open_D"), _odds("odds_open_A")
        has_book = _finite_odds(oh) and _finite_odds(od) and _finite_odds(oa)
        has_open = _finite_odds(ooh) and _finite_odds(ood) and _finite_odds(ooa)
        if has_book:
            mkt = remove_vig_3way(implied_prob(oh), implied_prob(od), implied_prob(oa))
        else:
            mkt = (float("nan"), float("nan"), float("nan"))
        if has_open:
            opened = remove_vig_3way(implied_prob(ooh), implied_prob(ood), implied_prob(ooa))
        else:
            opened = (float("nan"), float("nan"), float("nan"))

        edges = {
            "H": p_h - mkt[0],
            "D": p_d - mkt[1],
            "A": p_a - mkt[2],
        }
        if has_book:
            best = max(edges, key=edges.get)
            best_edge = edges[best]
        else:
            best, best_edge = None, float("nan")

        model_pick = max(
            [("H", p_h), ("D", p_d), ("A", p_a)],
            key=lambda t: t[1],
        )[0]
        kick_utc = pd.to_datetime(row["kickoff_utc"], utc=True)
        kick_local = kick_utc.tz_convert(DISPLAY_TZ)
        rows.append(
            {
                "espn_id": row.get("espn_id"),
                "kickoff_utc": kick_utc.isoformat(),
                "kickoff_et": kick_local.strftime("%Y-%m-%d %H:%M"),
                "status": row.get("status") or "",
                "completed": bool(row.get("completed", False)),
                "home_name": home,
                "away_name": away,
                "p_home": p_h,
                "p_draw": p_d,
                "p_away": p_a,
                "p_home_lo": pred.get("p_home_lo"),
                "p_home_hi": pred.get("p_home_hi"),
                "lam": pred.get("lam"),
                "mu": pred.get("mu"),
                "fair_odds_H": fair_h,
                "fair_odds_D": fair_d,
                "fair_odds_A": fair_a,
                "book": row.get("book"),
                "odds_H": oh,
                "odds_D": od,
                "odds_A": oa,
                "odds_open_H": ooh,
                "odds_open_D": ood,
                "odds_open_A": ooa,
                "p_home_mkt": mkt[0],
                "p_draw_mkt": mkt[1],
                "p_away_mkt": mkt[2],
                "p_home_open": opened[0],
                "p_draw_open": opened[1],
                "p_away_open": opened[2],
                "edge_H": edges["H"],
                "edge_D": edges["D"],
                "edge_A": edges["A"],
                "edge_open_H": p_h - opened[0],
                "edge_open_D": p_d - opened[1],
                "edge_open_A": p_a - opened[2],
                "best_side": best,
                "best_edge": best_edge,
                "model_pick": model_pick,
                "form_att_H": pred.get("form_att_H", 0.0),
                "form_def_H": pred.get("form_def_H", 0.0),
                "form_att_A": pred.get("form_att_A", 0.0),
                "form_def_A": pred.get("form_def_A", 0.0),
                "form_weight": pred.get("form_weight", 0.0),
                "form_scale": pred.get("form_scale", 0.0),
            }
        )
    return pd.DataFrame(rows)


UPCOMING_ODDS = {"H": "odds_H", "D": "odds_D", "A": "odds_A"}
SIDE_NAME = {"H": "Home", "D": "Draw", "A": "Away"}


def pick_upcoming_bets(
    df: pd.DataFrame,
    edge: float,
    sides: list[str],
) -> pd.DataFrame:
    """Mark the best qualifying 1X2 side per game (flat 1u). No result yet."""
    out = df.copy()
    bet_side: list[str | None] = []
    bet_odds: list[float | None] = []
    bet_edge: list[float | None] = []
    bet_team: list[str | None] = []

    for _, row in out.iterrows():
        best_s = None
        best_e = -1e9
        for side in sides:
            e = row.get(f"edge_{side}")
            if e is None or not np.isfinite(e):
                continue
            e = float(e)
            if e > best_e:
                best_e = e
                best_s = side
        if best_s is None or best_e < edge:
            bet_side.append(None)
            bet_odds.append(None)
            bet_edge.append(None)
            bet_team.append(None)
            continue
        odds = float(row[UPCOMING_ODDS[best_s]])
        if not np.isfinite(odds) or odds <= 1.0:
            bet_side.append(None)
            bet_odds.append(None)
            bet_edge.append(None)
            bet_team.append(None)
            continue
        if best_s == "H":
            team = str(row["home_name"])
        elif best_s == "A":
            team = str(row["away_name"])
        else:
            team = "Draw"
        bet_side.append(best_s)
        bet_odds.append(odds)
        bet_edge.append(best_e)
        bet_team.append(team)

    out["bet_side"] = bet_side
    out["bet_odds"] = bet_odds
    out["bet_edge"] = bet_edge
    out["bet_team"] = bet_team
    return out


def upcoming_only(df: pd.DataFrame) -> pd.DataFrame:
    """Drop completed / already-kicked-off matches from a scored slate."""
    out = df.copy()
    if "completed" in out.columns:
        raw = out["completed"]
        if raw.dtype == bool:
            done = raw.fillna(False)
        else:
            done = raw.astype(str).str.lower().isin(["1", "true", "yes"])
        out = out.loc[~done]
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=2)
    kick = None
    if "kickoff_utc" in out.columns and not out.empty:
        kick = pd.to_datetime(out["kickoff_utc"], utc=True, errors="coerce")
    elif "kickoff_et" in out.columns and not out.empty:
        kick = pd.to_datetime(out["kickoff_et"], errors="coerce")
        if kick.dt.tz is None:
            kick = kick.dt.tz_localize(DISPLAY_TZ)
        kick = kick.dt.tz_convert("UTC")
    if kick is not None:
        out = out.loc[kick >= cutoff]
    return out.reset_index(drop=True)


def load_scored_slate(
    *,
    n_days: int = 4,
    map_only: bool = True,
    refit: bool = False,
    allow_fixture_fallback: bool = True,
) -> pd.DataFrame:
    """Fetch the next ESPN window, fit/load Bayesian DC, score 1X2 vs the live book."""
    hist = load_matches()
    known = set(hist["home_name"]) | set(hist["away_name"])

    try:
        slate = fetch_upcoming_slate(n_days, use_cache=True)
    except Exception:
        if not allow_fixture_fallback:
            raise
        cached = TABLES_DIR / "bayesian_dc_upcoming.csv"
        if cached.exists():
            return pd.read_csv(cached)
        raise

    model = load_or_fit(
        hist,
        refit=refit,
        map_only=map_only,
        n_posterior=min(120, DEFAULT_N_POSTERIOR),
        xi=DEFAULT_XI,
        sigma_att=DEFAULT_SIGMA_ATT,
        sigma_def=DEFAULT_SIGMA_DEF,
    )
    scored = score_slate(
        slate, model, known, use_posterior=not map_only, hist=hist
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    scored.to_csv(TABLES_DIR / "bayesian_dc_upcoming.csv", index=False, float_format="%.6f")
    return scored


def _fmt_odds(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "  n/a"
    return f"{x:5.2f}"


def _fmt_pct(x: float) -> str:
    if x is None or not np.isfinite(x):
        return "  n/a"
    return f"{100 * x:4.1f}%"


def print_table(df: pd.DataFrame) -> None:
    print("\nUpcoming Liga MX — Bayesian Dixon–Coles fair 1X2")
    print("P = model. Open / Live = vig-free ESPN 1X2. Edge = P_model − P_live.")
    print("Edge = P_model − P_fair_book. Large edges were overconfident on 2025; shop, don't auto-bet.\n")
    header = (
        f"{'Kickoff CT':<16} {'Match':<42} "
        f"{'P(H)':>6} {'P(D)':>6} {'P(A)':>6}  "
        f"{'fair H/D/A':>16}  {'book H/D/A':>16}  "
        f"{'xG':>7} {'pick':>4} {'edge':>10}"
    )
    print(header)
    print("-" * len(header))
    for _, r in df.iterrows():
        match = f"{r['home_name']} vs {r['away_name']}"
        fair = f"{_fmt_odds(r['fair_odds_H'])}/{_fmt_odds(r['fair_odds_D'])}/{_fmt_odds(r['fair_odds_A'])}"
        book = f"{_fmt_odds(r['odds_H'])}/{_fmt_odds(r['odds_D'])}/{_fmt_odds(r['odds_A'])}"
        xg = f"{r['lam']:.1f}-{r['mu']:.1f}" if np.isfinite(r["lam"]) else "  n/a"
        if r["best_side"] and np.isfinite(r["best_edge"]):
            edge = f"{r['best_side']} {_fmt_pct(r['best_edge'])}"
        else:
            edge = "n/a"
        print(
            f"{r['kickoff_et']:<16} {match:<42} "
            f"{_fmt_pct(r['p_home']):>6} {_fmt_pct(r['p_draw']):>6} {_fmt_pct(r['p_away']):>6}  "
            f"{fair:>16}  {book:>16}  "
            f"{xg:>7} {r['model_pick']:>4} {edge:>10}"
        )


def default_dates(n_days: int = 4) -> list[str]:
    start = datetime.now(timezone.utc).date()
    return [(start + timedelta(days=i)).strftime("%Y%m%d") for i in range(n_days)]


def main() -> None:
    p = argparse.ArgumentParser(description="Predict upcoming Liga MX games with Bayesian DC.")
    p.add_argument(
        "--dates",
        type=str,
        default="",
        help="Comma-separated ESPN dates YYYYMMDD (default: today UTC through +3 days).",
    )
    p.add_argument(
        "--fixtures",
        type=str,
        default="",
        help="CSV with espn_home/espn_away/kickoff_utc and optional odds_H/D/A (skips ESPN).",
    )
    p.add_argument("--refit", action="store_true", help="Refit on all completed matches.")
    p.add_argument("--map-only", action="store_true", help="MAP only (skip Laplace posterior).")
    p.add_argument("--n-posterior", type=int, default=min(120, DEFAULT_N_POSTERIOR))
    p.add_argument("--xi", type=float, default=DEFAULT_XI)
    p.add_argument("--sigma-att", type=float, default=DEFAULT_SIGMA_ATT)
    p.add_argument("--sigma-def", type=float, default=DEFAULT_SIGMA_DEF)
    p.add_argument(
        "--form-weight",
        type=float,
        default=DEFAULT_FORM_WEIGHT,
        help="Mix weight on recent-form 1X2 (0 = long-run Dixon–Coles only).",
    )
    p.add_argument("--form-games", type=int, default=DEFAULT_FORM_GAMES)
    p.add_argument("--form-halflife", type=float, default=DEFAULT_FORM_HALFLIFE)
    p.add_argument("--form-scale", type=float, default=DEFAULT_FORM_SCALE)
    args = p.parse_args()

    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    hist = load_matches()
    known = set(hist["home_name"]) | set(hist["away_name"])
    print(f"History: {len(hist)} matches through {hist['game_date'].max().date()}")

    fixtures_path = Path(args.fixtures) if args.fixtures else FIXTURES_CSV
    if args.fixtures:
        print(f"Loading fixtures from {fixtures_path}...")
        slate = load_fixtures_csv(fixtures_path)
    elif dates:
        print(f"Fetching ESPN slate for {', '.join(dates)}...")
        slate = fetch_espn_slate(dates)
        live = slate.loc[~slate["completed"].astype(bool)]
        slate = live if not live.empty else slate
        write_fixtures_cache(slate)
    else:
        print("Fetching next ESPN slate...")
        slate = fetch_upcoming_slate(n_days=7, use_cache=True)
    print(f"  {len(slate)} games")

    model = load_or_fit(
        hist,
        refit=args.refit,
        map_only=args.map_only,
        n_posterior=args.n_posterior,
        xi=args.xi,
        sigma_att=args.sigma_att,
        sigma_def=args.sigma_def,
    )

    scored = score_slate(
        slate,
        model,
        known,
        use_posterior=not args.map_only,
        hist=hist,
        form_weight=args.form_weight,
        form_games=args.form_games,
        form_halflife=args.form_halflife,
        form_scale=args.form_scale,
    )
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = TABLES_DIR / "bayesian_dc_upcoming.csv"
    scored.to_csv(out_path, index=False, float_format="%.6f")
    print_table(scored)
    print(f"\nWrote {out_path}")
    print(
        "Use: compare book odds to fair odds. Bet only when a book is worse than the "
        "model *and* you would still like the price vs the eventual close (CLV)."
    )


if __name__ == "__main__":
    main()
