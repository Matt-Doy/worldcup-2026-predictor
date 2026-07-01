# 2026 FIFA World Cup — Probabilistic Prediction Model

A from-scratch prediction system for the 2026 FIFA World Cup (USA / Canada / Mexico): an expected-goals model trained on 125 years of international football, sampled through a Poisson Monte Carlo tournament simulation to produce per-team probabilities of reaching each stage — not a single deterministic bracket guess.

Built as a personal/learning project to combine football data, machine learning, and Monte Carlo simulation methodology.

## How it works

1. **Elo ratings** (`data/raw/elo_ratings_wc2026.csv`) give a point-in-time strength rating for each of the 48 qualified teams, scraped from [eloratings.net](https://www.eloratings.net/) with full historical bridging (e.g. Czechoslovakia → Czechia) — see `data/raw/README.md` for the full data dictionary.
2. **XGBoost expected-goals model**: two `XGBRegressor`s (home/away expected goals) trained on `elo_diff` + `neutral` (whether the match is played at neither team's home federation, relevant with 3 co-hosts), `objective="count:poisson"`, `max_depth=3`, with monotonic constraints on `elo_diff` to prevent regression-tree noise from producing non-monotonic win probabilities.
3. **Poisson sampling**: each match's expected goals are drawn as independent Poisson random variables; draws in the knockout stage are broken via an Elo-based win-probability estimate (proxy for extra-time/penalties).
4. **Monte Carlo tournament simulation**: the entire 48-team bracket (12 groups → Round of 32 → Round of 16 → QF → SF → Final) is simulated thousands of times; per-team probabilities of reaching each stage are the fraction of simulations in which that team got there.

The group draw is fixed (it already happened), but the **Round of 32 knockout pairing is asymmetric** — not a simple "1st place vs. 2nd place of the next group" scheme. It was reverse-engineered from the official FIFA match numbering (Wikipedia) and cross-validated against real results before being hardcoded into the simulation (see `scripts/run_simulation_live.py`).

## Results (live forecast, as of 2026-07-01)

Group stage + Round of 32 both complete (79/79 real matches). Top 10 championship probabilities from `n_sim=5000` Monte Carlo draws on the remaining bracket:

| Team | Champion % |
|---|---|
| France | 28.3% |
| Argentina | 24.2% |
| Spain | 21.7% |
| Brazil | 6.5% |
| England | 5.1% |
| Colombia | 4.5% |
| Portugal | 3.8% |
| Norway | 2.1% |
| Morocco | 1.4% |
| Paraguay | 1.1% |

Full table: `data/processed/wc2026_stage_probabilities_live.csv`. A styled PDF report (championship probability chart, group standings, and full bracket tree) is generated at `data/processed/wc2026_bracket_prediction.pdf`.

**Bookmaker comparison (2026-07-01):** France 28.3% vs bookmakers ~28.6% — essentially identical. Argentina 24.2% vs ~22%, Spain 21.7% vs ~13.3% (Spain slightly over-rated by the model, likely due to injury/tactical information not captured by Elo). Overall the model tracks bookmaker consensus closely for the top contenders.

## Model validation

Backtested out-of-sample on real 2014/2018/2022/2026 World Cup matches, with the model trained **only** on matches before 2014 (`scripts/backtest_worldcups.py`):

- **62.0%** outcome-prediction accuracy on the 71 real 2026 group-stage matches played so far, vs a **46.5%** naive baseline (always predict the most frequent outcome).
- **58.8%** accuracy across all backtested 2014–2026 World Cup matches.
- Brier score 0.562 (0 = perfect, 0.667 = chance level on 3 outcomes).
- Calibration is reasonable but slightly overconfident in the 70%+ predicted-probability bucket.

The model objective (`count:poisson` vs `squarederror`) and tree depth were chosen via a proper train/validation split in `scripts/model_selection.py`, not tuned on the test set. Two additional features (rolling team form, point-in-time Elo rank) were tested and **rejected** — their apparent gains weren't statistically significant on bootstrap resampling (`scripts/feature_test*.py`).

**Honest take on reliability:** the edge over a naive baseline is real and statistically significant, but modest — football has a lot of irreducible variance. Treat the relative ranking between teams as more trustworthy than the exact probability values.

## Repository structure

```
data/
  raw/                    source data (Elo ratings, match history, 2026 teams/fixtures)
  processed/              model outputs: probability tables, backtest results, PDF report
notebooks/
  01_exploration.ipynb    exploratory data analysis and initial prototyping (data merging,
                           feature engineering, first group/knockout simulation logic)
scripts/
  model_utils.py          shared bracket constants + live Elo computation (single update point)
  model_selection.py      XGBoost objective/depth grid search, validated on a real WC backtest
  backtest_worldcups.py   out-of-sample validation on 2014-2026 World Cup matches
  feature_test.py         tests rolling-form and rank-diff features (rejected, kept for reference)
  feature_test_bootstrap.py   statistical significance check for feature_test.py's results
  run_simulation_live.py  current forecast: real 2026 results locked in, only remaining
                           matches simulated, using the verified real FIFA bracket structure
  generate_bracket_pdf.py builds the PDF report from run_simulation_live.py's output
  update_elo.py           scrapes eloratings.net and updates elo_ratings_wc2026.csv (run daily)
```

## Running it

```bash
pip install -r requirements.txt
cd scripts
python model_selection.py        # optional: re-validate the model config
python backtest_worldcups.py     # optional: re-run the out-of-sample validation
python run_simulation_live.py    # generates the current forecast
python generate_bracket_pdf.py   # builds the PDF report (run after the line above)
```

## Data sources

- Historical match results: [martj42/international_results](https://github.com/martj42/international_results)
- Elo ratings: [eloratings.net](https://www.eloratings.net/) (full provenance and methodology in `data/raw/README.md`)
- 2026 squads, groups, and fixtures: official FIFA World Cup 2026 draw
