# Liga MX Moneyline — Findings & Notes

What worked, what didn’t, and how to use the model in practice (soft books +
promos). Odds throughout are football-data closing averages `AvgCH` / `AvgCD` /
`AvgCA` unless noted. Data refreshed through **2026-09-16**.

Bayesian Dixon–Coles was ported from `Documents/semantic/mls_moneyline`,
including the **league intercept** spec that produced the MLS 10% edge result.

---

## Bottom line

1. **Reference price = the closing market.** When the model and the close
   disagree, the close has been more accurate.
2. **Best model so far = Bayesian Dixon–Coles with league intercept**
   (hierarchical Poisson, MAP). Mean 1X2 now matches the close; log-loss is
   still worse (1.003 vs 0.983 on the first holdout).
3. **The MLS 10% edge rule does not replicate as a long-run edge here.**
   Same rule (MAP, best of H/A if `P_model − P_fair ≥ 0.10`, skip draws):
   first holdout **+12.3% / 82 bets**, calendar 2026 **−18.1% / 70 bets**,
   walk-forward 2018–2026 **+0.3% / 600 bets**. Same pattern as MLS 2025 vs 2026.
4. **Do not** treat the first-holdout plus as a found edge. Walk-forward at
   the same 10% rule is +80% in Apertura 2025, −1.5% in Clausura 2026, and
   −18.1% on calendar 2026.
5. **Do not** train on the previous tournament only, and **do not** bet
   draws-only at 5% edge. Both are worse.

---

## Model ladder

First holdout: train on **all seasons from 2012/2013 through 2024/2025**
(4,312 games), test **2025/2026 + 2026/2027** (406 matches, 12 Jul 2025 –
16 Sep 2026). Time decay (`xi=0.0038`, ~6-month half-life) downweights old
games without dropping them. Numbers below are **MAP + league intercept**.

| Price | 1X2 accuracy | 1X2 log-loss |
|-------|--------------|--------------|
| Closing market (de-vigged) | **52.5%** | **0.983** |
| Bayesian Dixon–Coles (intercept, MAP) | 51.7% | 1.003 |

Mean P(H/D/A): model **0.454 / 0.254 / 0.292** vs market **0.446 / 0.247 / 0.307**.
Mean λ/μ = 1.58 / 1.19. MAP: γ = **0.142**, δ = **0.277**,
ρ = −0.075. `model_pick` 279 H / 127 A vs market 262 / 143.

Edge=0 (best of H/D/A every game at AvgC): **−3.7%** ROI (−15.2u / 406).

Artifacts:

- `results/tables/bayesian_dc_edge_roi.csv`
- `results/tables/bayesian_dc_predictions_holdout.csv`
- `results/tables/bayesian_dc_edge10_walkforward.csv`
- `results/models/bayesian_dc_params.json`

```bash
python scripts/fit_bayesian_dc.py --map-only
python scripts/fit_bayesian_dc.py --map-only --test-from 2026-01-01
python scripts/fit_bayesian_dc.py            # MAP + Laplace posterior
```

### What Bayesian DC does

- Home/away goals ~ Poisson with Dixon–Coles low-score correlation ρ
- League intercept γ + home advantage δ (δ is *only* the extra home boost)
- Team attack/defence ~ hierarchical Gaussian priors (shrinkage)
- MAP via L-BFGS-B, optional Laplace posterior → average scoreline matrix → P(H), P(D), P(A)
- Edge rule: bet the side among H/D/A (or H,A) with largest `P_model − P_fair`
  if that gap ≥ threshold; payout at closing Avg*

---

## League intercept (MLS spec)

Same identifiability fix as MLS. Without γ, sum-to-zero attack/defence forces
geometric-mean away xG ≈ 1.0, so `home_adv` absorbs **all home scoring**.

```
λ = exp(γ + attack_home − defence_away + δ)
μ = exp(γ + attack_away − defence_home)
```

γ ~ N(0.15, 0.25²) (Liga MX away-goals log-mean), δ ~ N(0.25, 0.40²).

Pre-intercept pickles (`intercept = 0`, `home_adv ≳ 0.4`) are legacy; refit.

---

## MLS 10% edge rule on Liga MX

MLS 2025 holdout: MAP intercept model, best of H/A, edge ≥ 0.10 → **+11.9% /
117 bets** (H/D/A +8.5% / 117). 2026 YTD there was **−6.9%**. Same rule here.

First holdout (406 games, train through 2024/2025):

| Filter | n | Hit | ROI @ AvgC* |
|--------|---|-----|-------------|
| H/D/A edge ≥ 0 | 406 | 34.7% | **−3.7%** |
| H/D/A edge ≥ 0.05 | 216 | 34.7% | −2.8% |
| H/D/A edge ≥ 0.10 | 86 | 39.5% | **+7.1%** |
| **H,A edge ≥ 0.10** | **82** | **41.5%** | **+12.3%** |

Calendar **2026 YTD** (train through 2025-12-31, 237 games through 16 Sep):

| Filter | n | Hit | ROI |
|--------|---|-----|-----|
| H/D/A edge ≥ 0 | 237 | 33.8% | **−18.6%** |
| H/D/A edge ≥ 0.10 | 70 | 32.9% | **−18.1%** |
| H,A edge ≥ 0.10 | 70 | 32.9% | **−18.1%** |

**Expanding matchday refits** (MAP, history strictly before each of 66
2026 dates, 12.9s). Log-loss **1.036 vs 1.011** — same as the static fit.
Edge ≥ 0 **−10.8%**; edge ≥ 0.10 **−33.6% / 25 bets**. Mid-season scores
do not close the gap to the close.

MLS weekly expanding + a goals-residual form mix is already covered here
by matchday expanding (finer than MLS weeks). Porting the form overlay
**kept Dixon–Coles ρ**. First holdout (363 games, 124 matchdays): **w = 0
wins** (log-loss 1.0023 vs market 0.985). Any w > 0 is worse, same as MLS
2025. Live form weight stays 0.

MLS also moved time decay from `xi=0.0019` (~1-year half-life) to `0.0038`
(~6 months). Expanding matchday log-loss was slightly worse here (1.0023 →
1.0028 on the first holdout; 1.0357 → 1.0360 on 2026 YTD) while accuracy
ticked up ~1–2pp. The **static** MAP holdout went the other way: log-loss
1.004 → **1.001** and accuracy 51.3% → **51.6%**. Default is now `0.0038`
so live ratings track the last ~6 months more tightly. Artifact:
`results/tables/expanding_xi_grid.csv`.

```bash
python scripts/fit_bayesian_dc.py --map-only --test-from 2026-01-01 --expanding
python scripts/eval_expanding_form.py
```

Walk-forward all-history MAP, 17 tournaments 2018–2026, edge ≥ 0.10:

| Sides | Bets | P/L | ROI | Positive tournaments |
|-------|------|-----|-----|----------------------|
| H,A | 600 | +1.7u | **+0.3%** | 8 / 17 |
| H/D/A | 601 | +0.7u | **+0.1%** | 8 / 17 |

First-holdout +12% is still not a found edge: Apertura 2025 walk-forward
is +80% / 20 bets against Clausura 2026 **−1.5% / 47** and calendar 2026
**−18.1%**. Overconfidence on selected bets remains; do not Kelly on raw
`P_model`. Artifact:
`results/tables/bayesian_dc_edge10_walkforward.csv`.

---

## Two different numbers (do not mix them)

Pre-intercept holdout (345 games). The intercept refit above is the current
model; this section is the reason the old +5% was not a found edge.

**“Close to the market”** is probability quality (log-loss / accuracy). On the
first holdout the model was nearby (0.998 vs 0.978) but the close still won.
Betting the model’s side at the close every game is the shopping baseline:
**−2.1%**.

**The +5.2%** is a *disagreement filter*, not closeness:

- Rule: best of H/D/A only if `P_model − P_fair ≥ 0.05`
- First holdout: 174 bets, +9.1u, **+5.2%**
- Of that: **16 draws at ~5.0 odds, 6 wins, +12.7u**. Homes: 113 bets, **−8.7u**
- Apertura 2025 on that filter: +27.5%. Clausura 2026: **−12.0%**
- Mean `P_model` on those bets 0.41; hit rate 0.34 (same as the market’s 0.32)

So the plus is overconfidence + a hot Apertura draw streak, averaged with a
losing Clausura.

---

## The first holdout hid a split

The “close to market, −2.1%” headline mixed two tournaments:

| Window | Games | Model acc | Market acc | Log-loss gap | Edge-0 ROI |
|--------|-------|-----------|------------|--------------|------------|
| Apertura 2025 (Jul–Dec) | 169 | 55.6% | 53.8% | **+0.005** | **+2.9%** |
| Clausura 2026 (Jan–May) | 167 | 47.3% | 52.7% | +0.037 | **−10.6%** |
| Blended first holdout | 345 | 51.6% | 53.0% | +0.020 | −2.1% |

Apertura 2025’s +2.9% has a standard error of about **±13 pp**. Almost all of
it is September (+13.3u); October was −8.7u. Same tournament in the
walk-forward (MAP, all history) is **−5.8%**. It is not a profitable Apertura
regime.

Calendar **2026 YTD** (train through 2025-12-31, 237 games through 16 Sep):
log-loss 1.040 vs market 1.003; edge=0 **−18.6%**; edge≥0.05 **−20.3%**.
Retraining through Apertura 2025 did not fix Clausura.

Walk-forward all-history, 16 tournaments 2018–2026: **2 of 16** finished
positive at edge 0 (Clausura 2020 +5.8%, Apertura 2021 +0.6%). Aperturas
average **−7.4%**, Clausuras **−11.6%**.

---

## Tournament-only training does not help

Hypothesis: Liga MX resets every 6 months, so train Apertura 2025 → predict
Clausura 2026.

Same 18 teams both tournaments (no roster change). MAP, no Laplace:

| Train window | Train games | Log-loss gap | Edge-0 ROI |
|--------------|-------------|--------------|------------|
| Apertura 2025 only | 169 | **+0.060** | −13.3% |
| Clausura 2025 only | 170 | +0.062 | −7.1% |
| Last two tournaments | 339 | +0.055 | −16.2% |
| All history through Apertura 2025 | 4,481 | **+0.035** | −12.9% |

Walk-forward 2018–2026: all history beats previous-tournament on **13 / 16**
tournaments (mean gap +0.021 vs +0.035). The close already prices the reset;
dropping history overfits ~170 games (home advantage jumped to 0.54 on
Apertura 2025 alone). Time decay (`xi=0.0038`) already downweights last year.

```bash
python scripts/eval_tournament_dc.py
python scripts/eval_tournament_dc.py --target "Clausura 2026"
```

---

## Draw-only at 5% edge — do not use

Bet 1u on the draw at `AvgCD` whenever `P_model(draw) − P_fair(draw) ≥ 0.05`.

| Sample | Bets | Wins | Model P | Hit | Fair close | ROI |
|--------|------|------|---------|-----|------------|-----|
| First holdout | 29 | 7 | 0.271 | 0.241 | 0.200 | **+16%** |
| 2026 YTD | 16 | 6 | 0.257 | 0.375 | 0.191 | **+61%** |
| Walk-forward 2018–26 | 125 | 21 | **0.278** | **0.168** | 0.209 | **−26%** |

2026’s 6/16 (37.5%) is luckier than the model, not a found edge. Stricter
draw thresholds are worse (edge 0: −11%; 5%: −26%; 8%: −66%). 3 of 14
tournaments with any bets were positive; two of those are Apertura 2025 and
Clausura 2026.

---

## Least-bad filter vs the close: edge ≥ 5–8%

Best of H/D/A, walk-forward all history, 16 tournaments, 2,650 games:

| Filter | Bets | Hit | P/L | ROI |
|--------|------|-----|-----|-----|
| Edge ≥ 0 (every game) | 2,650 | 32.9% | −263u | **−9.9%** |
| Edge ≥ 2% | 2,212 | 33.8% | −192u | −8.7% |
| Edge ≥ 3% | 1,893 | 33.7% | −163u | −8.6% |
| Edge ≥ 5% | 1,378 | 34.5% | −72u | **−5.2%** |
| Edge ≥ 8% | 813 | 34.4% | −41u | **−5.0%** |
| Edge ≥ 10% | 545 | 34.3% | −39u | −7.1% |

Hit rate barely moves (33% → 34%). You lose fewer units because you bet less,
not because you found more winners. **2026** is the exception: edge ≥ 5% was
*worse* than betting every game (−17% vs −10%).

This is the least-bad *close-betting* filter. Sitting out still beats −5% if
you are paying AvgC.

Artifacts: `results/tables/edge_filter_walkforward_pooled.csv`,
`results/tables/tournament_dc_edge05_walkforward.csv`.

---

## Practical use: soft book + bonuses

The −5.2% is versus **closing averages**, which already carry ~7% 1X2
overround on this bet mix. A small hold plus volume is how you clear promos.

On the 1,378 walk-forward bets at edge ≥ 5%:

| Your price vs AvgC | ROI on this mix | 8× rollover leftover |
|--------------------|-----------------|----------------------|
| At the close | −5.2% | ~58% of bonus |
| +3% longer | −2.4% | ~81% |
| +5% longer | −0.5% | ~96% |
| +6% longer | **+0.5%** | — |
| +8% longer | +2.4% | ~119% |

Break-even vs AvgC is about **5.5%** longer prices. Clausura 2026 needed ~8%.

**Caveat:** this filter is **67% homes** (925 H / 409 A / 44 D). Soft books
often shade favorites shorter and dogs longer. If you are mostly on homes you
may be getting *worse* than AvgCH, not better. Only bet when your screen is
actually longer than the close. Track **CLV** (your odds vs eventual AvgC),
not just win/loss.

Intended workflow:

1. Fit / refresh Bayesian DC on history through yesterday.
2. For upcoming games, get **current** soft/early 1X2 (not only historical closes).
3. De-vig → compare to `P_model`. Prefer sides with edge ≥ 5–8%.
4. Bet only if the book is longer than `1 / P_model` **and** longer than the
   likely close. Skip draw-only 5%.
5. Log CLV + calibration. Promos can make a −2% to −5% hold +EV; they will
   not save a −17% tournament.

---

## How to refresh 2026 and re-run

```bash
curl -sL -A "Mozilla/5.0" "https://www.football-data.co.uk/new/MEX.csv" -o data/MEX.csv
python scripts/prepare_data.py
python scripts/fit_bayesian_dc.py --map-only
python scripts/fit_bayesian_dc.py --map-only --test-from 2026-01-01
python scripts/eval_tournament_dc.py
python scripts/eval_tournament_dc.py --predictions-only
python scripts/eval_open_close.py --predictions results/tables/bayesian_dc_predictions_walkforward.csv --since 2025-07-01
```

Holdout seasons default to `2025/2026` and `2026/2027` (`TEST_SEASONS` in
`scripts/model_utils.py`). `--test-from YYYY-MM-DD` trains on all games before
that date.

---

## Suggested next improvements

Higher leverage than another estimator on the same 1X2 closes:

- Record **your** odds vs AvgC (real CLV), especially on homes
- Lineups / absences / xG (information the books already use)
- Lineups / absences / xG (information the books already use)
- Walk-forward mid-tournament refits (not previous-tournament-only) —
  matchday expanding + form overlay already tried; w = 0 is best
- Bonus math with your actual rollover and stake mix
