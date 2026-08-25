# Liga MX Moneyline

ML project to estimate Liga MX match probabilities from recent form, compare them to closing market odds, and bet when the model finds an edge.

Uses [football-data.co.uk](https://www.football-data.co.uk/mexico.php) Mexico CSV (`data/MEX.csv`).

**Results & how to use the models in practice:** see [FINDINGS.md](FINDINGS.md)
(Bayesian Dixon–Coles with league intercept; 10% edge is +20% on the first
holdout and ~0% walk-forward 2018–2026 — not a found edge at the close).

## Live app

Same idea as a through-yesterday refit (all completed games), then score the
next ESPN Liga MX slate. The **Walk-forward ROI** tab uses one MAP fit per
Apertura/Clausura on history before that tournament.

```bash
python scripts/eval_tournament_dc.py --predictions-only   # once, for the ROI tab
python scripts/predict_upcoming.py --refit --map-only     # live model + slate
streamlit run app.py
```

## Setup

```bash
conda env create -f environment.yml
conda activate liga_mx_moneyline
```

Run all commands from the project root.

## Two-way market

1X2 closing odds are mapped to:

| Side | Wins when |
|------|-----------|
| Favorite | Favorite wins |
| Underdog DC | Draw or underdog wins |

## Pipeline

### 1. Prepare matches + market columns
```bash
python scripts/prepare_data.py
```
Writes `data/liga_mx_matches.csv`.

### 2. Feature engineering
```bash
python scripts/build_features.py
```
Adds lagged form (last 5 games: wins, goals, goals allowed, goal diff), H2H, rest days. Writes `data/liga_mx_featured.csv`.

### 3. Fit logistic model
```bash
python scripts/fit_logistic_model.py
```
Predicts `fav_won`. Train = all seasons before holdout; test = `2025/2026` and `2026/2027` (`TEST_SEASONS` in `scripts/model_utils.py`). Saves model to `results/models/`.

### 4. Model vs market (edge bets)
```bash
python scripts/eval_vs_market.py
```
Compares logistic model vs closing fair favorite probability and simulates flat 1u bets when edge ≥ threshold.

### Simple home-win model
```bash
python scripts/fit_simple_home_model.py
python scripts/fit_simple_home_model.py --edge 0.10
```
Predicts `P(home win)` from last-5 home/away form, H2H, and leak-free season
context (season-to-date form + progress). Odds are excluded from the model and
used only afterward; bets home when `p_model - p_market_fair ≥ edge` (default
10%). Train through `2023/2024`, sanity-check on `2024/2025`, evaluate once on
`2025/2026`. Writes model, model card, edge tables, coefficients, and holdout
predictions under `results/`.

### 5. Elo model
```bash
python scripts/fit_elo_model.py
python scripts/fit_elo_model.py --season-reset          # reset ratings each Season
python scripts/fit_elo_model.py --compare-reset          # continuous vs season-reset
python scripts/fit_elo_model.py --mid-season             # score only 2nd half of holdout
python scripts/fit_elo_model.py --mid-season --season-reset
python scripts/fit_elo_model.py --mid-season --compare-reset
```
Grid-searches Elo `K` and home advantage on train seasons, calibrates `P(favorite wins)` from the pre-match Elo edge, then reports holdout quality and edge-bet ROI vs the market. Writes ratings to `results/tables/elo_ratings_final.csv` and per-game Elo columns to `data/liga_mx_elo.csv`.

`--mid-season` is a within-season stress test: ratings still walk through the full schedule (H1 warm-up, then keep updating), but metrics and edge bets use only the second half of each holdout season. `K`, home advantage, and the calibrator are never fit on that second half.

### 6. Dixon–Coles Poisson model
```bash
python scripts/fit_dixon_coles.py              # expanding-window refit on test matchdays
python scripts/fit_dixon_coles.py --static     # fit once on train only (faster)
```
Estimates attack/defence strengths with a low-score correlation (`ρ`) and time decay, converts scorelines to 1X2 / favorite-win probabilities, then evaluates edge bets vs the closing market.

### 7. Bayesian Dixon–Coles (hierarchical Poisson)
```bash
python scripts/fit_bayesian_dc.py
python scripts/fit_bayesian_dc.py --test-from 2026-01-01   # train through 2025, score 2026 YTD
python scripts/fit_bayesian_dc.py --map-only --test-from 2026-01-01 --expanding
python scripts/fit_bayesian_dc.py --test-seasons 2025/2026
python scripts/fit_bayesian_dc.py --conf 0.7          # only bet when P(edge>0) ≥ 0.7
python scripts/fit_bayesian_dc.py --map-only           # shrinkage MAP, no Laplace
```
Same approach as the MLS moneyline project: attack/defence with Gaussian shrinkage
priors, a **league intercept** so home advantage is only the extra home boost
(not the scoring rate), Laplace posterior at the MAP, then posterior-predictive
1X2. Bets the side with the largest edge vs fair closing probs. Writes
`results/tables/bayesian_dc_edge_roi.csv`. The MLS 10% edge result used
`--map-only` and also reports an H,A-only slice (skip draws).

Tournament windows (Apertura / Clausura; previous tournament vs all history):

```bash
python scripts/eval_tournament_dc.py
python scripts/eval_tournament_dc.py --target "Clausura 2026"
```

Matchday expanding + optional recent-form mix (Dixon–Coles ρ is kept; live
form weight defaults to 0, same as MLS after their 2025 grid):

```bash
python scripts/eval_expanding_form.py
python scripts/eval_expanding_form.py --test-from 2026-01-01
```

### Baseline (no model)
```bash
python scripts/eval_underdog_dc.py --side favorite
python scripts/eval_underdog_dc.py --side underdog_dc
```

## Environment

- **Python** 3.11
- **Data / modeling:** pandas, numpy, scikit-learn, joblib
- **Exploration:** jupyter, matplotlib, seaborn
