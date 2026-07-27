# Liga MX Moneyline

ML project to estimate Liga MX match probabilities from recent form, compare them to closing market odds, and bet when the model finds an edge.

Uses [football-data.co.uk](https://www.football-data.co.uk/mexico.php) Mexico CSV (`data/MEX.csv`).

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

### 5. Elo model
```bash
python scripts/fit_elo_model.py
python scripts/fit_elo_model.py --season-reset          # reset ratings each Season
python scripts/fit_elo_model.py --compare-reset          # continuous vs season-reset
```
Grid-searches Elo `K` and home advantage on train seasons, calibrates `P(favorite wins)` from the pre-match Elo edge, then reports holdout quality and edge-bet ROI vs the market. Writes ratings to `results/tables/elo_ratings_final.csv` and per-game Elo columns to `data/liga_mx_elo.csv`.

### 6. Dixon–Coles Poisson model
```bash
python scripts/fit_dixon_coles.py              # expanding-window refit on test matchdays
python scripts/fit_dixon_coles.py --static     # fit once on train only (faster)
```
Estimates attack/defence strengths with a low-score correlation (`ρ`) and time decay, converts scorelines to 1X2 / favorite-win probabilities, then evaluates edge bets vs the closing market.

### Baseline (no model)
```bash
python scripts/eval_underdog_dc.py --side favorite
python scripts/eval_underdog_dc.py --side underdog_dc
```

## Environment

- **Python** 3.11
- **Data / modeling:** pandas, numpy, scikit-learn, joblib
- **Exploration:** jupyter, matplotlib, seaborn
