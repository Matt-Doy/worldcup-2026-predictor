"""
Backtest du modele sur les vraies Coupes du Monde 2014, 2018 et 2022.

Ces trois tournois sont entierement hors-echantillon : le modele est
entraine uniquement sur des matchs avant 2014 (train = year < 2014).
On verifie ici si les probabilites qu'il produit sont fiables sur des
matchs reels jamais vus, pas seulement si le MAE est bon en moyenne.

Lancer depuis le dossier scripts/ : python backtest_worldcups.py
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Chargement + preparation (identique a run_simulation.py)
# ---------------------------------------------------------------------------

results = pd.read_csv("../data/raw/results.csv")
elo = pd.read_csv("../data/raw/elo_ratings_wc2026.csv")
teams = pd.read_csv("../data/raw/wc_2026_teams.csv")

teams_ref = set(teams["team"])

name_mapping = {
    "United States": "USA",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
}

results["home_team"] = results["home_team"].replace(name_mapping)
results["away_team"] = results["away_team"].replace(name_mapping)

results["date"] = pd.to_datetime(results["date"])
results["year"] = results["date"].dt.year

mask = (
    results["home_team"].isin(teams_ref)
    & results["away_team"].isin(teams_ref)
    & (results["year"] >= 1901)
    & results["home_score"].notna()
)
matches = results[mask].copy()
matches["join_year"] = matches["year"] - 1

elo_simple = elo[["country", "year", "rating"]]

matches = matches.merge(
    elo_simple.rename(columns={"country": "home_team", "year": "join_year", "rating": "home_elo"}),
    on=["home_team", "join_year"],
    how="left",
)
matches = matches.merge(
    elo_simple.rename(columns={"country": "away_team", "year": "join_year", "rating": "away_elo"}),
    on=["away_team", "join_year"],
    how="left",
)

mask_complete = matches["home_elo"].notna() & matches["away_elo"].notna()
matches_clean = matches[mask_complete].copy()

matches_clean["elo_diff"] = matches_clean["home_elo"] - matches_clean["away_elo"]
matches_clean["result"] = np.where(
    matches_clean["home_score"] > matches_clean["away_score"],
    "home_win",
    np.where(matches_clean["home_score"] < matches_clean["away_score"], "away_win", "draw"),
)

train = matches_clean[matches_clean["year"] < 2014].copy()
test = matches_clean[matches_clean["year"] >= 2014].copy()

# Config choisie via model_selection.py (meilleure accuracy/Brier sur ce
# meme backtest, parmi {squarederror, count:poisson} x {depth 3,4,6}).
features = ["elo_diff", "neutral"]
X_train = train[features]

MODEL_PARAMS = dict(n_estimators=100, random_state=42, objective="count:poisson", max_depth=3)

model_home = XGBRegressor(**MODEL_PARAMS, monotone_constraints=(1, 0))
model_home.fit(X_train, train["home_score"])

model_away = XGBRegressor(**MODEL_PARAMS, monotone_constraints=(-1, 0))
model_away.fit(X_train, train["away_score"])

# ---------------------------------------------------------------------------
# 2. Isoler les vrais matchs de Coupe du Monde 2014 / 2018 / 2022
#    (jamais vus a l'entrainement : train = year < 2014)
# ---------------------------------------------------------------------------

wc = matches_clean[
    (matches_clean["tournament"] == "FIFA World Cup") & (matches_clean["year"] >= 2014)
].copy()

print(f"Matchs de Coupe du Monde 2014/2018/2022 disponibles (entre equipes qualifiees 2026) : {len(wc)}")
print(wc["year"].value_counts().sort_index())

X_wc = wc[features]
wc["lambda_home"] = model_home.predict(X_wc)
wc["lambda_away"] = model_away.predict(X_wc)

# ---------------------------------------------------------------------------
# 3. MAE specifiquement sur ces matchs (vs baseline naive)
# ---------------------------------------------------------------------------

mae_home = mean_absolute_error(wc["home_score"], wc["lambda_home"])
mae_away = mean_absolute_error(wc["away_score"], wc["lambda_away"])
baseline_home = mean_absolute_error(wc["home_score"], [train["home_score"].mean()] * len(wc))
baseline_away = mean_absolute_error(wc["away_score"], [train["away_score"].mean()] * len(wc))

print(f"\nMAE buts domicile (modele vs baseline) : {mae_home:.3f} vs {baseline_home:.3f}")
print(f"MAE buts exterieur (modele vs baseline) : {mae_away:.3f} vs {baseline_away:.3f}")

# ---------------------------------------------------------------------------
# 4. Probabilites implicites (victoire/nul/defaite) par tirage Poisson
#    et accuracy de l'issue predite vs l'issue reelle
# ---------------------------------------------------------------------------

n_sim = 2000
lambda_home_arr = wc["lambda_home"].to_numpy()
lambda_away_arr = wc["lambda_away"].to_numpy()

sim_home = np.random.poisson(lambda_home_arr[:, None], size=(len(wc), n_sim))
sim_away = np.random.poisson(lambda_away_arr[:, None], size=(len(wc), n_sim))

p_home_win = (sim_home > sim_away).mean(axis=1)
p_draw = (sim_home == sim_away).mean(axis=1)
p_away_win = (sim_home < sim_away).mean(axis=1)

wc["p_home_win"] = p_home_win
wc["p_draw"] = p_draw
wc["p_away_win"] = p_away_win

outcome_probs = np.vstack([p_home_win, p_draw, p_away_win]).T
outcome_labels = np.array(["home_win", "draw", "away_win"])
predicted_outcome = outcome_labels[outcome_probs.argmax(axis=1)]
wc["predicted_outcome"] = predicted_outcome

accuracy = (wc["predicted_outcome"] == wc["result"]).mean()
print(f"\nAccuracy issue predite (favori du modele) vs issue reelle : {accuracy:.1%}")

# baseline naif : toujours predire l'issue la plus frequente dans le train
most_common = train["result"].value_counts().idxmax()
baseline_accuracy = (wc["result"] == most_common).mean()
print(f"Baseline (toujours predire '{most_common}')           : {baseline_accuracy:.1%}")

print("\n--- Accuracy par edition ---")
for year, group in wc.groupby("year"):
    acc = (group["predicted_outcome"] == group["result"]).mean()
    print(f"{year} ({len(group)} matchs) : {acc:.1%}")

# ---------------------------------------------------------------------------
# 5. Calibration : la probabilite annoncee pour le favori correspond-elle
#    au taux de reussite reel ?
# ---------------------------------------------------------------------------

wc["p_favorite"] = outcome_probs.max(axis=1)
wc["favorite_correct"] = wc["predicted_outcome"] == wc["result"]

bins = [0.0, 0.4, 0.5, 0.6, 0.7, 1.0]
labels = ["<40%", "40-50%", "50-60%", "60-70%", "70%+"]
wc["p_favorite_bucket"] = pd.cut(wc["p_favorite"], bins=bins, labels=labels)

calibration = wc.groupby("p_favorite_bucket", observed=True).agg(
    n_matchs=("favorite_correct", "size"),
    p_moyenne_annoncee=("p_favorite", "mean"),
    taux_reussite_reel=("favorite_correct", "mean"),
)
print("\n--- Calibration (probabilite annoncee vs taux de reussite reel) ---")
print(calibration.round(3))

# ---------------------------------------------------------------------------
# 6. Brier score multi-classe (qualite globale de calibration, plus bas = mieux)
# ---------------------------------------------------------------------------

actual_onehot = np.zeros_like(outcome_probs)
for i, label in enumerate(outcome_labels):
    actual_onehot[:, i] = (wc["result"] == label).astype(int)

brier = np.mean(np.sum((outcome_probs - actual_onehot) ** 2, axis=1))
print(f"\nBrier score (modele) : {brier:.3f}  (0 = parfait, 0.667 = niveau hasard a 3 issues)")

wc_export = wc[
    ["date", "home_team", "away_team", "home_score", "away_score", "result",
     "lambda_home", "lambda_away", "p_home_win", "p_draw", "p_away_win", "predicted_outcome"]
]
wc_export.to_csv("../data/processed/backtest_worldcups_2014_2026.csv", index=False)
print("\nSauvegarde : data/processed/backtest_worldcups_2014_2026.csv")
