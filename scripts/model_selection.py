"""
Comparaison de configurations de modele (objectif + profondeur d'arbre)
sur le jeu de validation honnete (train < 2014, test >= 2014, matchs de
Coupe du Monde reels uniquement). Sert a choisir la config a utiliser
pour le modele final, qui sera ensuite reentraine sur TOUTES les donnees.

Lancer depuis scripts/ : python model_selection.py
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

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
    on=["home_team", "join_year"], how="left",
)
matches = matches.merge(
    elo_simple.rename(columns={"country": "away_team", "year": "join_year", "rating": "away_elo"}),
    on=["away_team", "join_year"], how="left",
)
mask_complete = matches["home_elo"].notna() & matches["away_elo"].notna()
matches_clean = matches[mask_complete].copy()
matches_clean["elo_diff"] = matches_clean["home_elo"] - matches_clean["away_elo"]
matches_clean["result"] = np.where(
    matches_clean["home_score"] > matches_clean["away_score"], "home_win",
    np.where(matches_clean["home_score"] < matches_clean["away_score"], "away_win", "draw"),
)

train = matches_clean[matches_clean["year"] < 2014].copy()
features = ["elo_diff", "neutral"]
X_train = train[features]

wc = matches_clean[(matches_clean["tournament"] == "FIFA World Cup") & (matches_clean["year"] >= 2014)].copy()
X_wc = wc[features]

configs = [
    {"name": "squarederror_depth6", "objective": "reg:squarederror", "max_depth": 6},
    {"name": "squarederror_depth4", "objective": "reg:squarederror", "max_depth": 4},
    {"name": "squarederror_depth3", "objective": "reg:squarederror", "max_depth": 3},
    {"name": "poisson_depth6", "objective": "count:poisson", "max_depth": 6},
    {"name": "poisson_depth4", "objective": "count:poisson", "max_depth": 4},
    {"name": "poisson_depth3", "objective": "count:poisson", "max_depth": 3},
]

n_sim_proba = 2000
rows = []

for cfg in configs:
    model_home = XGBRegressor(
        n_estimators=100, random_state=42, monotone_constraints=(1, 0),
        objective=cfg["objective"], max_depth=cfg["max_depth"],
    )
    model_home.fit(X_train, train["home_score"])

    model_away = XGBRegressor(
        n_estimators=100, random_state=42, monotone_constraints=(-1, 0),
        objective=cfg["objective"], max_depth=cfg["max_depth"],
    )
    model_away.fit(X_train, train["away_score"])

    lambda_home = model_home.predict(X_wc)
    lambda_away = model_away.predict(X_wc)

    mae_home = mean_absolute_error(wc["home_score"], lambda_home)
    mae_away = mean_absolute_error(wc["away_score"], lambda_away)

    lambda_home = np.clip(lambda_home, 1e-6, None)
    lambda_away = np.clip(lambda_away, 1e-6, None)

    sim_home = np.random.poisson(lambda_home[:, None], size=(len(wc), n_sim_proba))
    sim_away = np.random.poisson(lambda_away[:, None], size=(len(wc), n_sim_proba))
    p_home = (sim_home > sim_away).mean(axis=1)
    p_draw = (sim_home == sim_away).mean(axis=1)
    p_away = (sim_home < sim_away).mean(axis=1)
    outcome_probs = np.vstack([p_home, p_draw, p_away]).T
    outcome_labels = np.array(["home_win", "draw", "away_win"])
    predicted = outcome_labels[outcome_probs.argmax(axis=1)]
    accuracy = (predicted == wc["result"].values).mean()

    actual_onehot = np.zeros_like(outcome_probs)
    for i, label in enumerate(outcome_labels):
        actual_onehot[:, i] = (wc["result"].values == label).astype(int)
    brier = np.mean(np.sum((outcome_probs - actual_onehot) ** 2, axis=1))

    # monotonie : verifier que lambda_home ne redescend jamais sur une grille croissante d'elo_diff
    grid = pd.DataFrame({"elo_diff": list(range(-800, 801, 50)), "neutral": [True] * 33})
    grid_preds = model_home.predict(grid)
    is_monotonic = np.all(np.diff(grid_preds) >= -1e-6)

    rows.append({
        "config": cfg["name"], "mae_home": round(mae_home, 4), "mae_away": round(mae_away, 4),
        "accuracy": round(accuracy, 4), "brier": round(brier, 4), "monotonic": is_monotonic,
    })

comparison = pd.DataFrame(rows).sort_values("brier")
print(comparison.to_string(index=False))
