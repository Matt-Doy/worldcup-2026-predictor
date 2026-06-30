"""
Teste si des features supplementaires (forme recente, rang Elo) ameliorent
le modele de base (elo_diff + neutral), sur le meme backtest honnete que
model_selection.py (train < 2014, test >= 2014, matchs de Coupe du Monde).

Lancer depuis scripts/ : python feature_test.py
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Chargement
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

# ---------------------------------------------------------------------------
# 2. Feature "forme recente" : points moyens (3/1/0) sur les 10 derniers
#    matchs joues par l'equipe, TOUS adversaires confondus, calcules en
#    excluant le match courant (shift) pour eviter toute fuite temporelle.
# ---------------------------------------------------------------------------

home_rows = results[["date", "home_team", "home_score", "away_score"]].rename(
    columns={"home_team": "team", "home_score": "gf", "away_score": "ga"}
)
away_rows = results[["date", "away_team", "away_score", "home_score"]].rename(
    columns={"away_team": "team", "away_score": "gf", "home_score": "ga"}
)
long_form = pd.concat([home_rows, away_rows], ignore_index=True)
long_form = long_form.dropna(subset=["gf", "ga"])
long_form["points"] = np.where(
    long_form["gf"] > long_form["ga"], 3, np.where(long_form["gf"] < long_form["ga"], 1, 0)
)
long_form = long_form.sort_values(["team", "date"])

# rolling sur les 10 matchs PRECEDENTS (shift avant rolling => le match
# courant n'est jamais inclus dans sa propre feature)
long_form["recent_form"] = (
    long_form.groupby("team")["points"]
    .apply(lambda s: s.shift(1).rolling(window=10, min_periods=5).mean())
    .reset_index(level=0, drop=True)
)

form_lookup = long_form[["team", "date", "recent_form"]].drop_duplicates(subset=["team", "date"], keep="last")

# ---------------------------------------------------------------------------
# 3. Dataset d'entrainement (identique a model_selection.py) + nouvelles features
# ---------------------------------------------------------------------------

mask = (
    results["home_team"].isin(teams_ref)
    & results["away_team"].isin(teams_ref)
    & (results["year"] >= 1901)
    & results["home_score"].notna()
)
matches = results[mask].copy()
matches["join_year"] = matches["year"] - 1

elo_simple = elo[["country", "year", "rating", "rank"]]

matches = matches.merge(
    elo_simple.rename(columns={"country": "home_team", "year": "join_year", "rating": "home_elo", "rank": "home_rank"}),
    on=["home_team", "join_year"], how="left",
)
matches = matches.merge(
    elo_simple.rename(columns={"country": "away_team", "year": "join_year", "rating": "away_elo", "rank": "away_rank"}),
    on=["away_team", "join_year"], how="left",
)

matches = matches.merge(
    form_lookup.rename(columns={"team": "home_team", "recent_form": "home_form"}),
    on=["home_team", "date"], how="left",
)
matches = matches.merge(
    form_lookup.rename(columns={"team": "away_team", "recent_form": "away_form"}),
    on=["away_team", "date"], how="left",
)

mask_complete = (
    matches["home_elo"].notna() & matches["away_elo"].notna()
    & matches["home_form"].notna() & matches["away_form"].notna()
)
matches_clean = matches[mask_complete].copy()
print(f"Matchs avec elo + forme recente disponibles : {len(matches_clean)} (vs sans la contrainte forme : {matches['home_elo'].notna().sum()})")

matches_clean["elo_diff"] = matches_clean["home_elo"] - matches_clean["away_elo"]
matches_clean["rank_diff"] = matches_clean["away_rank"] - matches_clean["home_rank"]  # positif = home mieux classe
matches_clean["form_diff"] = matches_clean["home_form"] - matches_clean["away_form"]
matches_clean["result"] = np.where(
    matches_clean["home_score"] > matches_clean["away_score"], "home_win",
    np.where(matches_clean["home_score"] < matches_clean["away_score"], "away_win", "draw"),
)

train = matches_clean[matches_clean["year"] < 2014].copy()
wc = matches_clean[(matches_clean["tournament"] == "FIFA World Cup") & (matches_clean["year"] >= 2014)].copy()
print(f"Matchs WC backtest : {len(wc)}")

MODEL_PARAMS = dict(n_estimators=100, random_state=42, objective="count:poisson", max_depth=3)


def evaluate(feature_list, monotone):
    X_train = train[feature_list]
    X_wc = wc[feature_list]

    model_home = XGBRegressor(**MODEL_PARAMS, monotone_constraints=monotone)
    model_home.fit(X_train, train["home_score"])
    model_away = XGBRegressor(**MODEL_PARAMS, monotone_constraints=tuple(-c if c != 0 else 0 for c in monotone))
    model_away.fit(X_train, train["away_score"])

    lambda_home = model_home.predict(X_wc)
    lambda_away = model_away.predict(X_wc)

    mae_home = mean_absolute_error(wc["home_score"], lambda_home)
    mae_away = mean_absolute_error(wc["away_score"], lambda_away)

    n_sim = 2000
    sim_home = np.random.poisson(lambda_home[:, None], size=(len(wc), n_sim))
    sim_away = np.random.poisson(lambda_away[:, None], size=(len(wc), n_sim))
    p_home = (sim_home > sim_away).mean(axis=1)
    p_draw = (sim_home == sim_away).mean(axis=1)
    p_away = (sim_home < sim_away).mean(axis=1)
    outcome_probs = np.vstack([p_home, p_draw, p_away]).T
    labels = np.array(["home_win", "draw", "away_win"])
    predicted = labels[outcome_probs.argmax(axis=1)]
    accuracy = (predicted == wc["result"]).mean()

    actual_onehot = np.zeros_like(outcome_probs)
    for i, lab in enumerate(labels):
        actual_onehot[:, i] = (wc["result"] == lab).astype(int)
    brier = np.mean(np.sum((outcome_probs - actual_onehot) ** 2, axis=1))

    return dict(mae_home=mae_home, mae_away=mae_away, accuracy=accuracy, brier=brier)


configs = {
    "baseline (elo_diff, neutral)": (["elo_diff", "neutral"], (1, 0)),
    "+ rank_diff": (["elo_diff", "neutral", "rank_diff"], (1, 0, 1)),
    "+ form_diff": (["elo_diff", "neutral", "form_diff"], (1, 0, 1)),
    "+ rank_diff + form_diff": (["elo_diff", "neutral", "rank_diff", "form_diff"], (1, 0, 1, 1)),
}

rows = []
for name, (feats, mono) in configs.items():
    res = evaluate(feats, mono)
    res["config"] = name
    rows.append(res)

table = pd.DataFrame(rows).set_index("config")[["mae_home", "mae_away", "accuracy", "brier"]]
print("\n--- Comparaison features (sur le meme backtest honnete que model_selection.py) ---")
print(table.round(4).sort_values("brier"))
