"""
Expanding-window backtest of long-run Dixon–Coles 1X2 plus a recent-form mix.

Each matchday: fit MAP on all earlier matches (ρ kept), score that day's
games, then roll forward. Form residuals use history strictly before kickoff.

Unlike MLS, this does not drop the Dixon–Coles low-score correction: form-
adjusted λ, μ are converted with the same ρ as the long-run model.

Usage:
  python scripts/eval_expanding_form.py
  python scripts/eval_expanding_form.py --seasons 2025/2026,2026/2027
  python scripts/eval_expanding_form.py --test-from 2026-01-01
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from bayesian_dc_utils import DEFAULT_SIGMA_ATT, DEFAULT_SIGMA_DEF
from dixon_coles_utils import DEFAULT_XI, outcome_probs
from fit_bayesian_dc import fit_on_history, load_matches
from form_utils import (
    DEFAULT_FORM_GAMES,
    DEFAULT_FORM_HALFLIFE,
    DEFAULT_FORM_WEIGHT,
    mix_probs,
    form_lambdas,
    team_form,
)
from model_utils import TEST_SEASONS

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
CLASS_ORDER = ["H", "D", "A"]


def _logloss(y: np.ndarray, p: np.ndarray) -> float:
    return float(log_loss(y, p, labels=[0, 1, 2]))


def _summarize(df: pd.DataFrame, p_h: np.ndarray, p_d: np.ndarray, p_a: np.ndarray) -> dict:
    y = df["Res"].map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    p = np.column_stack([p_h, p_d, p_a])
    pick = np.array(["H", "D", "A"])[p.argmax(axis=1)]
    mkt = df[["p_home_fair", "p_draw_fair", "p_away_fair"]].to_numpy()
    return {
        "n": int(len(df)),
        "logloss": _logloss(y, p),
        "market_logloss": _logloss(y, mkt),
        "acc": float(accuracy_score(df["Res"], pick)),
        "market_acc": float(
            accuracy_score(
                df["Res"],
                np.array(["H", "D", "A"])[mkt.argmax(axis=1)],
            )
        ),
    }


def expanding_components(
    df: pd.DataFrame,
    test: pd.DataFrame,
    *,
    xi: float,
    n_games: int,
    halflife: float,
    label: str,
) -> pd.DataFrame:
    """Walk matchday-by-matchday; store long-run 1X2, λ/μ, ρ, and form residuals."""
    work = df.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    days = list(test.groupby("game_date", sort=True))
    print(
        f"Expanding {label}: {len(test)} games in {len(days)} matchdays  "
        f"xi={xi}  form last {n_games} (halflife {halflife:g})"
    )

    model = None
    parts: list[pd.DataFrame] = []
    t0 = time.time()
    for i, (dt, chunk) in enumerate(days, start=1):
        hist = work.loc[work["game_date"] < dt]
        if len(hist) < 50:
            continue
        model = fit_on_history(
            hist,
            xi=xi,
            sigma_att=DEFAULT_SIGMA_ATT,
            sigma_def=DEFAULT_SIGMA_DEF,
            n_posterior=0,
            compute_laplace=False,
            prev=model,
        )
        rows = []
        rho = float(model.rho)
        for row in chunk.itertuples(index=False):
            home = str(row.home_name)
            away = str(row.away_name)
            pred = model.predict_match(home, away, use_posterior=False)
            if n_games > 0:
                form_hist = work.loc[work["game_date"] < row.game_date]
                form = team_form(
                    form_hist,
                    model,
                    n_games=n_games,
                    halflife=halflife,
                    teams={home, away},
                )
            else:
                form = {}
            att_h, def_h = form.get(home, (0.0, 0.0))
            att_a, def_a = form.get(away, (0.0, 0.0))
            rows.append(
                {
                    "game_id": int(row.game_id),
                    "game_date": row.game_date,
                    "season": str(row.season),
                    "home_name": home,
                    "away_name": away,
                    "Res": row.Res,
                    "p_home_fair": float(row.p_home_fair),
                    "p_draw_fair": float(row.p_draw_fair),
                    "p_away_fair": float(row.p_away_fair),
                    "p_home_long": float(pred["p_home"]),
                    "p_draw_long": float(pred["p_draw"]),
                    "p_away_long": float(pred["p_away"]),
                    "lam": float(pred["lam"]),
                    "mu": float(pred["mu"]),
                    "rho": rho,
                    "form_att_H": att_h,
                    "form_def_H": def_h,
                    "form_att_A": att_a,
                    "form_def_A": def_a,
                }
            )
        parts.append(pd.DataFrame(rows))
        if i % 10 == 0 or i == len(days):
            print(
                f"  {i}/{len(days)}  {pd.Timestamp(dt).date()}  "
                f"hist={len(hist)}  ({time.time() - t0:.0f}s)"
            )

    if not parts:
        raise RuntimeError(f"No expanding predictions for {label}")
    return pd.concat(parts, ignore_index=True)


def score_weight(
    components: pd.DataFrame,
    weight: float,
    *,
    scale: float = 1.0,
) -> dict:
    p_h = np.empty(len(components))
    p_d = np.empty(len(components))
    p_a = np.empty(len(components))
    for i, row in enumerate(components.itertuples(index=False)):
        p_long = (row.p_home_long, row.p_draw_long, row.p_away_long)
        if weight <= 0:
            p_h[i], p_d[i], p_a[i] = p_long
            continue
        form = {
            row.home_name: (row.form_att_H, row.form_def_H),
            row.away_name: (row.form_att_A, row.form_def_A),
        }
        lam_f, mu_f = form_lambdas(
            row.lam, row.mu, row.home_name, row.away_name, form, scale=scale
        )
        p_form = outcome_probs(lam_f, mu_f, float(row.rho))
        p_h[i], p_d[i], p_a[i] = mix_probs(p_long, p_form, weight)
    stats = _summarize(components, p_h, p_d, p_a)
    stats["form_weight"] = weight
    stats["form_scale"] = 0.0 if weight <= 0 else scale
    stats["gap"] = stats["logloss"] - stats["market_logloss"]
    return stats


def test_slice(
    df: pd.DataFrame,
    *,
    seasons: list[str] | None,
    test_from: str | None,
) -> tuple[pd.DataFrame, str]:
    if test_from:
        cut = pd.Timestamp(test_from)
        test = df.loc[df["game_date"] >= cut].copy()
        return test, f"from {cut.date()}"
    test = df.loc[df["season"].astype(str).isin(seasons or TEST_SEASONS)].copy()
    return test, ",".join(seasons or TEST_SEASONS)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Expanding-window form-mix backtest (Dixon–Coles ρ kept)."
    )
    p.add_argument(
        "--seasons",
        type=str,
        default=",".join(TEST_SEASONS),
        help="Comma-separated holdout seasons (ignored if --test-from is set).",
    )
    p.add_argument(
        "--test-from",
        type=str,
        default="",
        help="If set (YYYY-MM-DD), score all games on/after this date.",
    )
    p.add_argument("--xi", type=str, default=str(DEFAULT_XI))
    p.add_argument("--weights", type=str, default="0,0.15,0.25,0.40")
    p.add_argument("--scales", type=str, default="0.25,0.50,1.0")
    p.add_argument("--form-games", type=int, default=DEFAULT_FORM_GAMES)
    p.add_argument("--form-halflife", type=float, default=DEFAULT_FORM_HALFLIFE)
    args = p.parse_args()

    seasons = [s.strip() for s in args.seasons.split(",") if s.strip()]
    xis = [float(x) for x in args.xi.split(",") if x.strip()]
    weights = [float(x) for x in args.weights.split(",") if x.strip()]
    scales = [float(x) for x in args.scales.split(",") if x.strip()]
    test_from = args.test_from.strip() or None

    df = load_matches()
    test, label = test_slice(df, seasons=seasons, test_from=test_from)
    if test.empty:
        raise SystemExit(f"No test games for {label}.")

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for xi in xis:
        comps = expanding_components(
            df,
            test,
            xi=xi,
            n_games=args.form_games,
            halflife=args.form_halflife,
            label=label,
        )
        print(f"\n{label}  xi={xi:.4f}  n={len(comps)}  (Dixon–Coles ρ kept)")
        print(
            f"  {'w':>6}  {'s':>5}  {'n':>4}  {'logloss':>8}  {'mkt':>8}  "
            f"{'gap':>7}  {'acc':>6}  {'mkt acc':>7}"
        )
        seen_zero = False
        for w in weights:
            run_scales = [1.0] if w <= 0 else scales
            for scale in run_scales:
                if w <= 0:
                    if seen_zero:
                        continue
                    seen_zero = True
                stats = score_weight(comps, w, scale=scale)
                stats["window"] = label
                stats["xi"] = xi
                stats["form_games"] = args.form_games
                stats["form_halflife"] = args.form_halflife
                rows.append(stats)
                print(
                    f"  {w:6.2f}  {stats['form_scale']:5.2f}  {stats['n']:4d}  "
                    f"{stats['logloss']:8.4f}  {stats['market_logloss']:8.4f}  "
                    f"{stats['gap']:+7.4f}  {100 * stats['acc']:5.1f}%  "
                    f"{100 * stats['market_acc']:6.1f}%"
                )

    out = pd.DataFrame(rows)
    path = TABLES_DIR / "expanding_form_grid.csv"
    out.to_csv(path, index=False, float_format="%.6f")
    print(f"\nWrote {path}")
    print(
        f"Live default form weight is {DEFAULT_FORM_WEIGHT:.2f}. "
        "Keep ρ; pick the expanding row with the lowest log-loss."
    )


if __name__ == "__main__":
    main()
