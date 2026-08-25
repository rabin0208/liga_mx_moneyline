"""
Train Bayesian DC on Liga MX tournament windows (Apertura / Clausura).

Liga MX form is often treated as resetting every ~6 months. This compares:
  - previous tournament only (e.g. Apertura 2025 → Clausura 2026)
  - previous same-type tournament (Clausura → next Clausura)
  - last two tournaments
  - all history before the test tournament (current default)

Usage:
  python scripts/eval_tournament_dc.py
  python scripts/eval_tournament_dc.py --target "Clausura 2026"
  python scripts/eval_tournament_dc.py --predictions-only
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

from bayesian_dc_utils import DEFAULT_SIGMA_ATT, DEFAULT_SIGMA_DEF, DEFAULT_XI
from fit_bayesian_dc import (
    CLASS_ORDER,
    TABLES_DIR,
    fit_on_history,
    load_matches,
    pick_bets,
    score_rows,
)

TABLES_DIR.mkdir(parents=True, exist_ok=True)


def tournament_label(dt: pd.Timestamp) -> str:
    if dt.month >= 7:
        return f"Apertura {dt.year}"
    return f"Clausura {dt.year}"


def add_tournament(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["tournament"] = out["game_date"].map(tournament_label)
    return out


def tournament_order(df: pd.DataFrame) -> list[str]:
    first = df.groupby("tournament")["game_date"].min().sort_values()
    return list(first.index)


def summarize(test_s: pd.DataFrame) -> dict:
    y = test_s["Res"]
    y_idx = y.map({c: i for i, c in enumerate(CLASS_ORDER)}).to_numpy()
    p_model = test_s[["p_home_model", "p_draw_model", "p_away_model"]].to_numpy()
    p_mkt = test_s[["p_home_fair", "p_draw_fair", "p_away_fair"]].to_numpy()
    scored = pick_bets(test_s, 0.0, CLASS_ORDER, conf=0.0)
    placed = scored.dropna(subset=["bet_side"])
    n = len(placed)
    return {
        "n_test": int(len(test_s)),
        "model_acc": float(accuracy_score(y, test_s["model_pick"])),
        "market_acc": float(accuracy_score(y, test_s["mkt_pick"])),
        "model_logloss": float(log_loss(y_idx, p_model, labels=[0, 1, 2])),
        "market_logloss": float(log_loss(y_idx, p_mkt, labels=[0, 1, 2])),
        "n_bets": int(n),
        "profit": float(placed["bet_profit"].sum()) if n else 0.0,
        "roi": float(placed["bet_profit"].sum() / n) if n else float("nan"),
    }


def fit_and_score(train: pd.DataFrame, test: pd.DataFrame, *, xi: float) -> dict:
    if len(train) < 40 or len(test) == 0:
        return {}
    model = fit_on_history(
        train,
        xi=xi,
        sigma_att=DEFAULT_SIGMA_ATT,
        sigma_def=DEFAULT_SIGMA_DEF,
        n_posterior=0,
        compute_laplace=False,
        prev=None,
    )
    test_s = score_rows(
        test, model, use_posterior=False, conf=0.0, conf_edge=0.0
    )
    out = summarize(test_s)
    out["n_train"] = int(len(train))
    out["intercept"] = float(model.intercept)
    out["home_adv"] = float(model.home_adv)
    out["logloss_gap"] = out["model_logloss"] - out["market_logloss"]
    return out


def print_row(label: str, m: dict) -> None:
    if not m:
        print(f"  {label:28s}  (skipped)")
        return
    print(
        f"  {label:28s}  train={m['n_train']:4d}  test={m['n_test']:3d}  "
        f"acc={m['model_acc']:.3f}/{m['market_acc']:.3f}  "
        f"ll={m['model_logloss']:.3f}/{m['market_logloss']:.3f}  "
        f"gap={m['logloss_gap']:+.3f}  "
        f"P/L={m['profit']:+7.1f}  ROI={100 * m['roi']:+5.1f}%"
    )


PRED_KEEP = [
    "game_id",
    "game_date",
    "season",
    "tournament",
    "home_name",
    "away_name",
    "Res",
    "AvgCH",
    "AvgCD",
    "AvgCA",
    "p_home_fair",
    "p_draw_fair",
    "p_away_fair",
    "p_home_model",
    "p_draw_model",
    "p_away_model",
    "lam",
    "mu",
    "edge_H",
    "edge_D",
    "edge_A",
    "model_pick",
    "mkt_pick",
]


def dump_walkforward_predictions(
    df: pd.DataFrame,
    order: list[str],
    *,
    walk_from: str,
) -> Path:
    start_i = order.index(walk_from) if walk_from in order else 8
    print(
        f"\n=== Per-game walk-forward (all history, MAP) from {order[start_i]} ==="
    )
    parts: list[pd.DataFrame] = []
    prev_model = None
    for t in order[start_i:]:
        wins_t = windows_for_target(df, t, order)
        train, test = wins_t["all_history"], df.loc[df["tournament"] == t]
        if len(train) < 40 or test.empty:
            continue
        model = fit_on_history(
            train,
            xi=DEFAULT_XI,
            sigma_att=DEFAULT_SIGMA_ATT,
            sigma_def=DEFAULT_SIGMA_DEF,
            n_posterior=0,
            compute_laplace=False,
            prev=prev_model,
        )
        prev_model = model
        test_s = score_rows(
            test, model, use_posterior=False, conf=0.0, conf_edge=0.0
        )
        test_s["tournament"] = t
        parts.append(test_s)
        print(f"  {t:18s}  train={len(train):4d}  test={len(test_s):3d}")
    if not parts:
        raise SystemExit("Walk-forward produced no games.")
    out = pd.concat(parts, ignore_index=True)
    keep = [c for c in PRED_KEEP if c in out.columns]
    path = TABLES_DIR / "bayesian_dc_predictions_walkforward.csv"
    out[keep].to_csv(path, index=False)
    print(f"Wrote {path} ({len(out)} games)")
    return path


def windows_for_target(df: pd.DataFrame, target: str, order: list[str]) -> dict[str, pd.DataFrame]:
    if target not in order:
        raise SystemExit(f"Unknown tournament {target}. Have: {order[-6:]}")
    i = order.index(target)
    test = df.loc[df["tournament"] == target]
    start = test["game_date"].min()
    before = df.loc[df["game_date"] < start]
    prev = df.loc[df["tournament"] == order[i - 1]] if i else before.iloc[0:0]
    kind = target.split()[0]
    same = [t for t in order[:i] if t.startswith(kind)]
    same_prev = df.loc[df["tournament"] == same[-1]] if same else before.iloc[0:0]
    last2_names = order[max(0, i - 2) : i]
    last2 = df.loc[df["tournament"].isin(last2_names)]
    return {
        "prev_tournament": prev,
        "same_type_prev": same_prev,
        "last_two": last2,
        "all_history": before,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Tournament-window Bayesian DC.")
    p.add_argument("--target", type=str, default="Clausura 2026")
    p.add_argument("--walk-from", type=str, default="Apertura 2018")
    p.add_argument(
        "--predictions-only",
        action="store_true",
        help="Write per-game all-history walk-forward predictions and exit.",
    )
    args = p.parse_args()

    df = add_tournament(load_matches())
    order = tournament_order(df)
    print("Tournaments:", " → ".join(order[-6:]), "…")

    if args.predictions_only:
        dump_walkforward_predictions(df, order, walk_from=args.walk_from)
        return
    print(
        f"\n=== {args.target}: train-window comparison (MAP, no Laplace) ==="
    )
    print("  acc/ll are model/market. gap = model log-loss − market (lower/closer to 0 is better).")

    wins = windows_for_target(df, args.target, order)
    target_rows = []
    for name, train in wins.items():
        xi = 0.0 if name != "all_history" else DEFAULT_XI
        m = fit_and_score(train, df.loc[df["tournament"] == args.target], xi=xi)
        print_row(name, m)
        if m:
            target_rows.append({"target": args.target, "window": name, **m})

    pd.DataFrame(target_rows).to_csv(
        TABLES_DIR / "tournament_dc_clausura_2026.csv", index=False, float_format="%.6f"
    )

    # Walk-forward: previous tournament vs all history, each tournament.
    start_i = order.index(args.walk_from) if args.walk_from in order else 8
    print(
        f"\n=== Walk-forward from {order[start_i]} "
        f"(prev tournament vs all history) ==="
    )
    walk_rows = []
    for t in order[start_i:]:
        if t == "Apertura 2026":
            continue  # incomplete
        wins_t = windows_for_target(df, t, order)
        test = df.loc[df["tournament"] == t]
        for window, xi in (("prev_tournament", 0.0), ("all_history", DEFAULT_XI)):
            m = fit_and_score(wins_t[window], test, xi=xi)
            if not m:
                continue
            walk_rows.append({"target": t, "window": window, **m})
        prev = next(r for r in walk_rows if r["target"] == t and r["window"] == "prev_tournament")
        hist = next(r for r in walk_rows if r["target"] == t and r["window"] == "all_history")
        print(
            f"  {t:18s}  prev gap={prev['logloss_gap']:+.3f} ROI={100*prev['roi']:+5.1f}%   "
            f"hist gap={hist['logloss_gap']:+.3f} ROI={100*hist['roi']:+5.1f}%"
        )

    walk = pd.DataFrame(walk_rows)
    walk.to_csv(
        TABLES_DIR / "tournament_dc_walkforward.csv", index=False, float_format="%.6f"
    )
    print("\nWalk-forward means (complete tournaments):")
    for window, g in walk.groupby("window"):
        print(
            f"  {window:18s}  mean gap={g['logloss_gap'].mean():+.3f}  "
            f"mean ROI={100 * g['roi'].mean():+.1f}%  "
            f"median ROI={100 * g['roi'].median():+.1f}%  "
            f"n_tourn={len(g)}"
        )
    print(f"\nWrote {TABLES_DIR / 'tournament_dc_clausura_2026.csv'}")
    print(f"Wrote {TABLES_DIR / 'tournament_dc_walkforward.csv'}")


if __name__ == "__main__":
    main()
