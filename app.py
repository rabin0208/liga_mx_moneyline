"""Streamlit dashboard: upcoming Liga MX 1X2 vs live book, with edge recommendations."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
WALKFORWARD_PATH = TABLES_DIR / "bayesian_dc_predictions_walkforward.csv"

SIDE_CHOICES = {
    "Away only": ["A"],
    "Home & away": ["H", "A"],
    "All 1X2": ["H", "D", "A"],
}


def _load_scripts_module(name: str):
    """Load a file from scripts/ by path so a similarly named package cannot win."""
    path = SCRIPTS_DIR / f"{name}.py"
    if not path.is_file():
        listing = sorted(p.name for p in SCRIPTS_DIR.glob("*.py")) if SCRIPTS_DIR.is_dir() else []
        raise ImportError(
            f"Missing {path}. scripts/ contains: {listing or 'no .py files (directory missing?)'}"
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

_load_scripts_module("odds_utils")
_load_scripts_module("dixon_coles_utils")
_load_scripts_module("bayesian_dc_utils")
_load_scripts_module("model_utils")
_load_scripts_module("fit_bayesian_dc")
_load_scripts_module("form_utils")
_load_scripts_module("predict_upcoming")
_load_scripts_module("eval_open_close")

from eval_open_close import attach_open_close  # noqa: E402
from fit_bayesian_dc import pick_bets  # noqa: E402
from odds_utils import fair_probs_from_odds  # noqa: E402
from predict_upcoming import (  # noqa: E402
    load_scored_slate,
    pick_upcoming_bets,
    upcoming_only,
)

ESPN_OPEN_CLOSE = PROJECT_ROOT / "data" / "espn_open_close.csv"
OPEN_ODDS_COL = {"H": "OpenH", "D": "OpenD", "A": "OpenA"}
ESPN_CLOSE_ODDS_COL = {"H": "CloseH", "D": "CloseD", "A": "CloseA"}

st.set_page_config(
    page_title="Liga MX Moneyline Edge",
    page_icon="⚽",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root {
      --bm-bg: #07101f;
      --bm-panel: #0b1424;
      --bm-line: #203049;
      --bm-text: #eef4ff;
      --bm-muted: #9aa9bd;
      --bm-accent: #3ee0a0;
      --bm-green: #35d07f;
      --bm-orange: #ffad5c;
    }
    .stApp {
      background:
        radial-gradient(circle at 74% 5%, rgba(62, 224, 160, 0.12), transparent 26rem),
        linear-gradient(180deg, #111a2c 0, var(--bm-bg) 24rem);
      color: var(--bm-text);
    }
    [data-testid="stSidebar"] {
      background: rgba(11, 20, 36, 0.98);
      border-right: 1px solid var(--bm-line);
    }
    [data-testid="stMetric"] {
      background: rgba(11, 20, 36, 0.78);
      border: 1px solid var(--bm-line);
      border-radius: 0.75rem;
      padding: 0.85rem 1rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
      background: rgba(11, 20, 36, 0.76);
      border-color: var(--bm-line);
      border-radius: 0.8rem;
      box-shadow: 0 14px 34px rgba(0, 0, 0, 0.18);
    }
    .bm-hero {
      margin: 0 0 1.5rem;
      padding: clamp(1.25rem, 3vw, 2.25rem);
      background:
        linear-gradient(90deg, rgba(62, 224, 160, 0.20), transparent 62%),
        linear-gradient(135deg, #0b6b45 0%, #12855a 52%, #1aa37a 100%);
      border: 1px solid rgba(126, 255, 196, 0.38);
      border-radius: 0.8rem;
    }
    .bm-hero h1 {
      margin: 0;
      color: white;
      font-size: clamp(2.2rem, 5vw, 4rem);
      line-height: 1;
    }
    .bm-hero p {
      margin: 0.65rem 0 0;
      color: rgba(255, 255, 255, 0.88);
      font-size: 1.05rem;
    }
    .bm-game-title {
      color: var(--bm-text);
      font-size: 1.2rem;
      font-weight: 750;
    }
    .bm-meta {
      color: var(--bm-muted);
      font-size: 0.83rem;
    }
    .bm-pick {
      display: inline-block;
      margin: 0.15rem 0.5rem 0.8rem 0;
      padding: 0.28rem 0.6rem;
      color: #03140c;
      background: var(--bm-green);
      border-radius: 0.3rem;
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .bm-stake {
      display: inline-block;
      margin: 0.15rem 0 0.8rem;
      padding: 0.28rem 0.6rem;
      color: #1a1206;
      background: var(--bm-orange);
      border-radius: 0.3rem;
      font-size: 0.76rem;
      font-weight: 800;
      letter-spacing: 0.02em;
      text-transform: uppercase;
    }
    .bm-no-pick {
      color: var(--bm-muted);
      font-size: 0.8rem;
    }
    h1, h2, h3, label, [data-testid="stMetricLabel"] {
      color: var(--bm-text) !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(ttl=12 * 3600, show_spinner=False)
def load_slate(n_days: int) -> pd.DataFrame:
    return load_scored_slate(n_days=n_days, map_only=True)


@st.cache_data(ttl=3600, show_spinner=False)
def load_walkforward() -> pd.DataFrame:
    if not WALKFORWARD_PATH.is_file():
        raise FileNotFoundError(
            "No walk-forward table. Run "
            "`python scripts/eval_tournament_dc.py --predictions-only`."
        )
    df = pd.read_csv(WALKFORWARD_PATH)
    needed = ["AvgCH", "AvgCD", "AvgCA", "Res", "p_home_model", "p_draw_model", "p_away_model"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"{WALKFORWARD_PATH.name} missing {missing}")
    df = df.dropna(subset=needed).reset_index(drop=True)
    return attach_open_close(df)


def apply_book_edges(df: pd.DataFrame, odds_h: str, odds_d: str, odds_a: str) -> pd.DataFrame:
    """Replace AvgC fair probs / edges with de-vigged prices from another 1X2 book."""
    out = df.dropna(subset=[odds_h, odds_d, odds_a]).copy()
    fair = [
        fair_probs_from_odds(h, d, a)
        for h, d, a in zip(out[odds_h], out[odds_d], out[odds_a])
    ]
    out["p_home_fair"] = [t[0] for t in fair]
    out["p_draw_fair"] = [t[1] for t in fair]
    out["p_away_fair"] = [t[2] for t in fair]
    out = out.dropna(subset=["p_home_fair", "p_draw_fair", "p_away_fair"])
    out["edge_H"] = out["p_home_model"] - out["p_home_fair"]
    out["edge_D"] = out["p_draw_model"] - out["p_draw_fair"]
    out["edge_A"] = out["p_away_model"] - out["p_away_fair"]
    return out


def percent(value: object) -> str:
    return "—" if pd.isna(value) else f"{float(value):.1%}"


def decimal_odds(value: object) -> str:
    return "—" if pd.isna(value) else f"{float(value):.2f}"


def dollars(amount: object) -> str:
    return "—" if pd.isna(amount) else f"${float(amount):,.0f}"


def slate_profit_if_all_win(recs: pd.DataFrame, unit: float) -> float:
    profit = 0.0
    for _, rec in recs.iterrows():
        odds = rec.get("bet_odds")
        if pd.isna(odds) or float(odds) <= 1.0:
            continue
        profit += unit * (float(odds) - 1.0)
    return profit


def render_outcome(
    *,
    title: str,
    model_prob: object,
    open_prob: object,
    live_prob: object,
    edge: object,
    open_odds: object,
    live_odds: object,
) -> None:
    st.subheader(title)
    st.metric(
        "Model probability",
        percent(model_prob),
        delta=(
            f"{float(edge):+.1%} vs live"
            if pd.notna(edge)
            else None
        ),
        delta_color="normal",
    )
    if pd.notna(model_prob):
        st.progress(min(max(float(model_prob), 0.0), 1.0), text=f"Model {percent(model_prob)}")
    if pd.notna(open_prob):
        st.progress(
            min(max(float(open_prob), 0.0), 1.0),
            text=f"Open {percent(open_prob)} · {decimal_odds(open_odds)}",
        )
    else:
        st.caption("Open line not posted yet.")
    if pd.notna(live_prob):
        st.progress(
            min(max(float(live_prob), 0.0), 1.0),
            text=f"Live {percent(live_prob)} · {decimal_odds(live_odds)}",
        )
    elif pd.notna(model_prob):
        st.caption("No live 1X2 on ESPN yet.")


def render_slate(
    *,
    n_days: int,
    recommendations_only: bool,
    edge_threshold: float,
    sides: list[str],
    unit: float,
) -> None:
    with st.spinner(f"Scoring the next {n_days} day(s) and fetching live odds…"):
        try:
            games = load_slate(n_days)
        except Exception as exc:
            st.error(str(exc))
            st.info(
                "Need a live model (`results/models/bayesian_dc_model_live.pkl`) and either "
                "network access to ESPN or `data/upcoming_fixtures.csv`. "
                "Refit with `python scripts/predict_upcoming.py --refit --map-only`."
            )
            return

    if games.empty:
        st.info("No Liga MX games in this window.")
        return

    live = upcoming_only(games)
    is_past = live.empty
    if is_past:
        st.warning(
            "No remaining kickoffs in this window (ESPN may also be blocking). "
            "Showing the last scored slate as completed — recommendations are off."
        )
        visible_source = games.sort_values("kickoff_et").reset_index(drop=True)
    else:
        visible_source = live

    games = pick_upcoming_bets(visible_source, edge_threshold, sides)
    games = games.sort_values(
        ["bet_edge", "kickoff_et"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    rec_mask = games["bet_side"].notna() if not is_past else pd.Series(False, index=games.index)
    recommendations = games.loc[rec_mask]
    matched = games.dropna(subset=["odds_H", "odds_A"])
    n_units = int(len(recommendations))
    stake = n_units * unit
    all_win = slate_profit_if_all_win(recommendations, unit)

    kpi = st.columns(5)
    kpi[0].metric("Games", len(games))
    kpi[1].metric("With odds", f"{len(matched)}/{len(games)}")
    if is_past:
        kpi[2].metric("Recommended", "—")
        kpi[3].metric("Slate stake", "—")
        kpi[4].metric("If all win", "—")
    else:
        kpi[2].metric("Recommended", len(recommendations))
        if unit > 0:
            kpi[3].metric(
                "Slate stake",
                dollars(stake),
                delta=f"{n_units} × {dollars(unit)}",
                delta_color="off",
            )
            kpi[4].metric("If all win", dollars(all_win), delta_color="off")
        else:
            kpi[3].metric("Slate stake", f"{n_units}u")
            kpi[4].metric("If all win", f"{all_win:+.1f}u")

    visible = recommendations if (recommendations_only and not is_past) else games
    if visible.empty:
        st.info(
            f"No bets clear {edge_threshold:.0%} edge on sides {','.join(sides)}."
        )
        return

    sides_label = ",".join(sides)
    st.subheader("Last slate" if is_past else "Upcoming matchups")
    if is_past:
        st.caption(
            f"Showing {len(visible)} completed games. Open and live are the last "
            "ESPN lines, not a betting slate."
        )
    else:
        st.caption(
            f"Showing {len(visible)} of {len(games)} games · "
            f"flat 1u · sides [{sides_label}] · threshold {edge_threshold:.0%}. "
            "Bars are model / ESPN open / ESPN live (de-vigged). "
            "Recommendations use edge vs the live book, not the open. "
            "Walk-forward 2018–2026 at 10% H/A is about −3% vs the close."
        )

    for _, game in visible.iterrows():
        with st.container(border=True):
            header, status = st.columns([4, 1])
            header.markdown(
                f"<div class='bm-game-title'>{game['home_name']} vs {game['away_name']}</div>",
                unsafe_allow_html=True,
            )
            book = game.get("book") or "ESPN book"
            xg = ""
            if pd.notna(game.get("lam")) and pd.notna(game.get("mu")):
                xg = f"<br>xG {float(game['lam']):.1f}–{float(game['mu']):.1f}"
            game_status = game.get("status") or ("Final" if is_past else "")
            status.markdown(
                f"<div class='bm-meta'>{game.get('kickoff_et', '')} CT"
                f"<br>{game_status} · {book}{xg}</div>",
                unsafe_allow_html=True,
            )

            if is_past:
                st.markdown(
                    "<span class='bm-no-pick'>Final — model vs last posted book, not a live bet.</span>",
                    unsafe_allow_html=True,
                )
            elif pd.notna(game.get("bet_side")):
                stake_html = (
                    f"<span class='bm-stake'>Stake: {dollars(unit)} (1u)</span>"
                    if unit > 0
                    else "<span class='bm-stake'>Stake: 1u</span>"
                )
                st.markdown(
                    f"<span class='bm-pick'>Model edge: {game['bet_team']} "
                    f"{decimal_odds(game['bet_odds'])} · {percent(game['bet_edge'])}</span>"
                    f"{stake_html}",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    "<span class='bm-no-pick'>No side clears the selected edge threshold.</span>",
                    unsafe_allow_html=True,
                )

            home_col, draw_col, away_col = st.columns(3, gap="large")
            with home_col:
                render_outcome(
                    title=str(game["home_name"]),
                    model_prob=game["p_home"],
                    open_prob=game.get("p_home_open"),
                    live_prob=game.get("p_home_mkt"),
                    edge=game.get("edge_H"),
                    open_odds=game.get("odds_open_H"),
                    live_odds=game.get("odds_H"),
                )
            with draw_col:
                render_outcome(
                    title="Draw",
                    model_prob=game["p_draw"],
                    open_prob=game.get("p_draw_open"),
                    live_prob=game.get("p_draw_mkt"),
                    edge=game.get("edge_D"),
                    open_odds=game.get("odds_open_D"),
                    live_odds=game.get("odds_D"),
                )
            with away_col:
                render_outcome(
                    title=str(game["away_name"]),
                    model_prob=game["p_away"],
                    open_prob=game.get("p_away_open"),
                    live_prob=game.get("p_away_mkt"),
                    edge=game.get("edge_A"),
                    open_odds=game.get("odds_open_A"),
                    live_odds=game.get("odds_A"),
                )


def tournament_roi_table(bets: pd.DataFrame) -> pd.DataFrame:
    cols = ["Tournament", "Bets", "Wins", "Hit rate", "Profit (u)", "ROI"]
    if bets.empty or "tournament" not in bets.columns:
        return pd.DataFrame(columns=cols)
    work = bets.copy()
    grouped = (
        work.groupby("tournament", as_index=False)
        .agg(
            Bets=("bet_profit", "count"),
            Wins=("bet_won", "sum"),
            profit=("bet_profit", "sum"),
        )
    )
    grouped["Wins"] = grouped["Wins"].astype(int)
    grouped["Hit rate"] = grouped["Wins"] / grouped["Bets"]
    grouped["ROI"] = grouped["profit"] / grouped["Bets"]
    grouped["Tournament"] = grouped["tournament"]
    grouped["Profit (u)"] = grouped["profit"]
    order = sorted(
        grouped["Tournament"].unique(),
        key=lambda name: (
            int(str(name).split()[-1]),
            0 if str(name).startswith("Clausura") else 1,
        ),
    )
    grouped["Tournament"] = pd.Categorical(
        grouped["Tournament"], categories=order, ordered=True
    )
    return grouped.sort_values("Tournament")[cols]


def render_roi_tab(*, edge_threshold: float, sides: list[str], tournament: str) -> None:
    st.subheader("Walk-forward flat 1u ROI")
    market = st.radio(
        "Edge vs",
        options=["Closing AvgC*", "ESPN close", "Opening (ESPN/DK)"],
        index=2,
        horizontal=True,
        help=(
            "Closing AvgC* uses football-data (full 2018–2026). "
            "ESPN close and Opening use ESPN BET / DraftKings 1X2 from ESPN’s "
            "core odds API (Jul 2025 onward) and pay at those prices."
        ),
    )
    use_open = market.startswith("Opening")
    use_espn_close = market.startswith("ESPN close")
    if use_open:
        st.caption(
            f"Each tournament is scored with a MAP fit on **all history before it**. "
            f"Best side among [{','.join(sides)}] when model − fair **opening** ≥ "
            f"{edge_threshold:.0%}, payout at ESPN BET / DraftKings open. "
            "Coverage is Apertura 2025 onward."
        )
    elif use_espn_close:
        st.caption(
            f"Each tournament is scored with a MAP fit on **all history before it**. "
            f"Best side among [{','.join(sides)}] when model − fair **ESPN close** ≥ "
            f"{edge_threshold:.0%}, payout at ESPN BET / DraftKings close. "
            "Coverage is Apertura 2025 onward."
        )
    else:
        st.caption(
            f"Each tournament is scored with a MAP fit on **all history before it** "
            f"(not previous-tournament-only). Same rule as the slate: best side among "
            f"[{','.join(sides)}] when model − fair close ≥ {edge_threshold:.0%}, "
            "payout at AvgC*."
        )

    with st.spinner("Loading walk-forward predictions…"):
        try:
            games = load_walkforward()
        except Exception as exc:
            st.error(str(exc))
            return

    if tournament != "All":
        games = games.loc[games["tournament"] == tournament].copy()

    odds_col = None
    if use_open or use_espn_close:
        if not ESPN_OPEN_CLOSE.is_file() or "OpenH" not in games.columns:
            st.warning(
                "No ESPN open/close cache. Run "
                "`python scripts/eval_open_close.py "
                "--predictions results/tables/bayesian_dc_predictions_walkforward.csv "
                "--since 2025-07-01`."
            )
            return
        if use_open:
            games = apply_book_edges(games, "OpenH", "OpenD", "OpenA")
            odds_col = OPEN_ODDS_COL
            missing_msg = (
                "No games in this filter have ESPN opening 1X2. "
                "Coverage starts Apertura 2025."
            )
        else:
            games = apply_book_edges(games, "CloseH", "CloseD", "CloseA")
            odds_col = ESPN_CLOSE_ODDS_COL
            missing_msg = (
                "No games in this filter have ESPN closing 1X2. "
                "Coverage starts Apertura 2025."
            )
        if games.empty:
            st.info(missing_msg)
            return

    scored = pick_bets(games, edge_threshold, sides, conf=0.0, odds_col=odds_col)
    bets = scored.dropna(subset=["bet_side"]).copy()
    by_tourn = tournament_roi_table(bets)

    kpi = st.columns(4)
    if bets.empty:
        kpi[0].metric("Bets", 0)
        kpi[1].metric("Hit rate", "—")
        kpi[2].metric("Profit", "—")
        kpi[3].metric("ROI", "—")
        st.info(f"No bets clear {edge_threshold:.0%} on [{','.join(sides)}].")
        return

    n = len(bets)
    hit = float(bets["bet_won"].mean())
    profit = float(bets["bet_profit"].sum())
    roi = profit / n
    kpi[0].metric("Bets", n)
    kpi[1].metric("Hit rate", f"{hit:.1%}")
    kpi[2].metric("Profit", f"{profit:+.1f}u")
    kpi[3].metric("ROI", f"{roi:+.1%}")

    display = by_tourn.copy()
    display["Hit rate"] = display["Hit rate"].map(lambda x: f"{x:.1%}")
    display["Profit (u)"] = display["Profit (u)"].map(lambda x: f"{x:+.2f}")
    display["ROI"] = display["ROI"].map(lambda x: f"{x:+.1%}")
    st.dataframe(display, use_container_width=True, hide_index=True)

    if len(by_tourn) >= 2:
        chart_df = by_tourn.copy()
        chart_df["Tournament"] = chart_df["Tournament"].astype(str)
        chart = (
            alt.Chart(chart_df)
            .mark_bar()
            .encode(
                x=alt.X(
                    "Tournament:N",
                    title="Tournament",
                    sort=list(chart_df["Tournament"]),
                    axis=alt.Axis(labelAngle=-35),
                ),
                y=alt.Y(
                    "ROI:Q",
                    title="Flat 1u ROI",
                    axis=alt.Axis(format=".0%"),
                ),
                tooltip=[
                    alt.Tooltip("Tournament:N"),
                    alt.Tooltip("ROI:Q", format=".1%"),
                    alt.Tooltip("Bets:Q"),
                ],
            )
            .properties(
                title=(
                    f"Walk-forward ROI by tournament "
                    f"(edge ≥ {edge_threshold:.0%}, sides {','.join(sides)})"
                ),
                height=320,
            )
            .configure_title(fontSize=16, anchor="start", color="#eef4ff")
            .configure_axis(labelColor="#9aa9bd", titleColor="#eef4ff")
        )
        st.altair_chart(chart, use_container_width=True)


st.markdown(
    """
    <section class="bm-hero">
      <h1>Liga MX Moneyline Edge</h1>
      <p>Bayesian Dixon–Coles 1X2 vs ESPN open and live. Live model: all history through yesterday.
      Recommendations: edge vs the live book. Walk-forward ROI can use AvgC close or ESPN open/close. Default: home &amp; away, 10% edge, flat 1u.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    n_days = st.slider("Upcoming window (days)", min_value=1, max_value=14, value=7)
    recommendations_only = st.toggle(
        "Recommendations only",
        value=False,
        help="Hide games that do not clear the edge threshold on the selected sides.",
    )
    side_label = st.radio(
        "Sides",
        options=list(SIDE_CHOICES.keys()),
        index=1,
        help="Walk-forward at 10% is almost the same for H/A vs H/D/A. Draws are rare at that bar.",
    )
    sides = SIDE_CHOICES[side_label]
    edge_percent = st.slider(
        "Minimum edge",
        min_value=0,
        max_value=20,
        value=10,
        step=1,
        format="%d%%",
        help="Model probability minus vig-free live ESPN 1X2 on the slate, or closing AvgC* / ESPN open-close on the walk-forward tab.",
    )
    edge_threshold = edge_percent / 100.0
    unit = st.number_input(
        "Unit size ($)",
        min_value=0,
        value=25,
        step=5,
        help="Each recommendation is 1 unit. Set 0 to show units only. Not Kelly — selected edges are overconfident.",
    )
    if st.button(
        "Refresh slate",
        type="primary",
        use_container_width=True,
        help="Clears the 12-hour cache and re-fetches ESPN.",
    ):
        load_slate.clear()
        load_walkforward.clear()
        st.rerun()
    st.divider()
    st.caption(
        "Unconditional probabilities track the market. A 10% selected edge is not a "
        "calibrated 10% — walk-forward vs the close is about −3% at this filter."
    )

slate_tab, roi_tab = st.tabs(["Upcoming slate", "Walk-forward ROI"])

with slate_tab:
    render_slate(
        n_days=n_days,
        recommendations_only=recommendations_only,
        edge_threshold=edge_threshold,
        sides=sides,
        unit=float(unit),
    )

with roi_tab:
    try:
        wf = load_walkforward()
        tournaments = ["All"] + sorted(
            wf["tournament"].dropna().unique().tolist(),
            key=lambda name: (
                int(str(name).split()[-1]),
                0 if str(name).startswith("Clausura") else 1,
            ),
        )
    except Exception:
        tournaments = ["All"]
    roi_tournament = st.selectbox("Tournament", options=tournaments, index=0)
    render_roi_tab(
        edge_threshold=edge_threshold,
        sides=sides,
        tournament=roi_tournament,
    )
