"""
Bootstrap sur le gain de Brier (par match) entre le modele baseline
(elo_diff, neutral) et le modele + form_diff, pour verifier si le gain
observe dans feature_test.py est statistiquement reel.

Lancer depuis scripts/ : python feature_test_bootstrap.py
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

np.random.seed(42)

results = pd.read_csv("../data/raw/results.csv")
elo = pd.read_csv("../data/raw/elo_ratings_wc2026.csv")
teams = pd.read_csv("../data/raw/wc_2026_teams.csv")
teams_ref = set(teams["team"])

name_mapping = {"United States": "USA", "Turkey": "Türkiye", "Czech Republic": "Czechia"}
results["home_team"] = results["home_team"].replace(name_mapping)
results["away_team"] = results["away_team"].replace(name_mapping)
results["date"] = pd.to_datetime(results["date"])
results["year"] = results["date"].dt.year

home_rows = results[["date", "home_team", "home_score", "away_score"]].rename(
    columns={"home_team": "team", "home_score": "gf", "away_score": "ga"})
away_rows = results[["date", "away_team", "away_score", "home_score"]].rename(
    columns={"away_team": "team", "away_score": "gf", "home_score": "ga"})
long_form = pd.concat([home_rows, away_rows], ignore_index=True).dropna(subset=["gf", "ga"])
long_form["points"] = np.where(long_form["gf"] > long_form["ga"], 3, np.where(long_form["gf"] < long_form["ga"], 1, 0))
long_form = long_form.sort_values(["team", "date"])
long_form["recent_form"] = (
    long_form.groupby("team")["points"].apply(lambda s: s.shift(1).rolling(window=10, min_periods=5).mean())
    .reset_index(level=0, drop=True)
)
form_lookup = long_form[["team", "date", "recent_form"]].drop_duplicates(subset=["team", "date"], keep="last")

mask = (results["home_team"].isin(teams_ref) & results["away_team"].isin(teams_ref)
        & (results["year"] >= 1901) & results["home_score"].notna())
matches = results[mask].copy()
matches["join_year"] = matches["year"] - 1
elo_simple = elo[["country", "year", "rating"]]
matches = matches.merge(elo_simple.rename(columns={"country": "home_team", "year": "join_year", "rating": "home_elo"}),
                         on=["home_team", "join_year"], how="left")
matches = matches.merge(elo_simple.rename(columns={"country": "away_team", "year": "join_year", "rating": "away_elo"}),
                         on=["away_team", "join_year"], how="left")
matches = matches.merge(form_lookup.rename(columns={"team": "home_team", "recent_form": "home_form"}),
                         on=["home_team", "date"], how="left")
matches = matches.merge(form_lookup.rename(columns={"team": "away_team", "recent_form": "away_form"}),
                         on=["away_team", "date"], how="left")

mask_complete = (matches["home_elo"].notna() & matches["away_elo"].notna()
                  & matches["home_form"].notna() & matches["away_form"].notna())
matches_clean = matches[mask_complete].copy()
matches_clean["elo_diff"] = matches_clean["home_elo"] - matches_clean["away_elo"]
matches_clean["form_diff"] = matches_clean["home_form"] - matches_clean["away_form"]
matches_clean["result"] = np.where(matches_clean["home_score"] > matches_clean["away_score"], "home_win",
    np.where(matches_clean["home_score"] < matches_clean["away_score"], "away_win", "draw"))

train = matches_clean[matches_clean["year"] < 2014].copy()
wc = matches_clean[(matches_clean["tournament"] == "FIFA World Cup") & (matches_clean["year"] >= 2014)].copy()

MODEL_PARAMS = dict(n_estimators=100, random_state=42, objective="count:poisson", max_depth=3)
labels = np.array(["home_win", "draw", "away_win"])


def get_outcome_probs(feature_list, monotone, n_sim=2000):
    X_train, X_wc = train[feature_list], wc[feature_list]
    mh = XGBRegressor(**MODEL_PARAMS, monotone_constraints=monotone)
    mh.fit(X_train, train["home_score"])
    ma = XGBRegressor(**MODEL_PARAMS, monotone_constraints=tuple(-c if c != 0 else 0 for c in monotone))
    ma.fit(X_train, train["away_score"])
    lam_h, lam_a = mh.predict(X_wc), ma.predict(X_wc)
    sim_h = np.random.poisson(lam_h[:, None], size=(len(wc), n_sim))
    sim_a = np.random.poisson(lam_a[:, None], size=(len(wc), n_sim))
    p_h = (sim_h > sim_a).mean(axis=1)
    p_d = (sim_h == sim_a).mean(axis=1)
    p_a = (sim_h < sim_a).mean(axis=1)
    return np.vstack([p_h, p_d, p_a]).T


probs_baseline = get_outcome_probs(["elo_diff", "neutral"], (1, 0))
probs_form = get_outcome_probs(["elo_diff", "neutral", "form_diff"], (1, 0, 1))

actual_onehot = np.zeros_like(probs_baseline)
for i, lab in enumerate(labels):
    actual_onehot[:, i] = (wc["result"].to_numpy() == lab).astype(int)

per_match_brier_baseline = np.sum((probs_baseline - actual_onehot) ** 2, axis=1)
per_match_brier_form = np.sum((probs_form - actual_onehot) ** 2, axis=1)
diff = per_match_brier_baseline - per_match_brier_form  # positif = form_diff ameliore

print(f"Brier baseline   : {per_match_brier_baseline.mean():.4f}")
print(f"Brier + form_diff: {per_match_brier_form.mean():.4f}")
print(f"Gain moyen par match : {diff.mean():.4f}")

n_boot = 2000
boot_means = []
n = len(diff)
for _ in range(n_boot):
    sample = np.random.choice(diff, size=n, replace=True)
    boot_means.append(sample.mean())
boot_means = np.array(boot_means)
ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
print(f"IC 95% bootstrap du gain moyen : [{ci_low:.4f}, {ci_high:.4f}]")
print("Gain statistiquement significatif (CI exclut 0) :", ci_low > 0 or ci_high < 0)
