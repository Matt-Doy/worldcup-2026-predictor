"""
Simulation Monte Carlo complete de la Coupe du Monde 2026, "from scratch"
(aucun resultat reel pris en compte -- phase de groupes ET phases finales
sont entierement simulees).

Pipeline : donnees historiques -> elo_diff -> modeles XGBoost (buts attendus)
-> tirage Poisson -> simulation de groupe -> simulation des phases finales
-> probabilites de chaque equipe d'atteindre chaque stade du tournoi.

NB : le tableau final (Round of 32 -> Champion) utilise ici un schema
generique base sur les positions de groupe ("1A vs 2B", tires depuis
wc_2026_fixtures.csv), pas le vrai tableau officiel FIFA 2026 (qui est
asymetrique -- voir run_simulation_live.py). Ce script est conserve comme
version "pre-tournoi" / premiere iteration du pipeline ; pour une
prevision a jour et fidele au vrai tableau, utiliser run_simulation_live.py.

Lancer depuis le dossier scripts/ : python run_simulation.py
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Chargement des donnees
# ---------------------------------------------------------------------------

results = pd.read_csv("../data/raw/results.csv")
elo = pd.read_csv("../data/raw/elo_ratings_wc2026.csv")
teams = pd.read_csv("../data/raw/wc_2026_teams.csv")
fixtures = pd.read_csv("../data/raw/wc_2026_fixtures.csv")

latest_date = elo["snapshot_date"].max()
elo_latest = elo[elo["snapshot_date"] == latest_date].copy()

teams_ref = set(teams["team"])

# ---------------------------------------------------------------------------
# 2. Harmonisation des noms d'equipes
# ---------------------------------------------------------------------------

name_mapping = {
    "United States": "USA",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
}

elo_latest["country"] = elo_latest["country"].replace(name_mapping)
results["home_team"] = results["home_team"].replace(name_mapping)
results["away_team"] = results["away_team"].replace(name_mapping)

# ---------------------------------------------------------------------------
# 3. Construction du dataset d'entrainement (elo_diff, neutral, scores)
# ---------------------------------------------------------------------------

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

test = matches_clean[matches_clean["year"] >= 2014].copy()

# ---------------------------------------------------------------------------
# 4. Entrainement des modeles de production
#    Config choisie via model_selection.py (objectif Poisson + profondeur
#    3 + contrainte de monotonie sur elo_diff) : meilleure accuracy/Brier
#    sur le backtest honnete (voir backtest_worldcups.py, train < 2014).
#    Ici on reentraine sur TOUTES les donnees (matches_clean) puisque la
#    config a deja ete validee hors-echantillon ailleurs -- ne pas
#    gaspiller les ~20% de matchs 2014+ pour le modele final.
# ---------------------------------------------------------------------------

features = ["elo_diff", "neutral"]
train = matches_clean
X_train = train[features]
X_test = test[features]

MODEL_PARAMS = dict(n_estimators=100, random_state=42, objective="count:poisson", max_depth=3)

model_home = XGBRegressor(**MODEL_PARAMS, monotone_constraints=(1, 0))
model_home.fit(X_train, train["home_score"])
preds_home = model_home.predict(X_test)
print("MAE model_home (sur 2014+, indicatif) :", mean_absolute_error(test["home_score"], preds_home))

model_away = XGBRegressor(**MODEL_PARAMS, monotone_constraints=(-1, 0))
model_away.fit(X_train, train["away_score"])
preds_away = model_away.predict(X_test)
print("MAE model_away (sur 2014+, indicatif) :", mean_absolute_error(test["away_score"], preds_away))

# Verification rapide de la monotonie
print("\nVerification monotonie (elo_diff -> lambda_home / lambda_away) :")
for diff in [0, 100, 200, 300, 400, 500, 600]:
    X_check = pd.DataFrame({"elo_diff": [diff], "neutral": [True]})
    lh = model_home.predict(X_check)[0]
    la = model_away.predict(X_check)[0]
    print(diff, round(lh, 2), round(la, 2))

# ---------------------------------------------------------------------------
# 5. Preparation pour la simulation 2026
# ---------------------------------------------------------------------------

elo_lookup = elo_latest.set_index("country")["rating"]
group_letters = sorted(teams["group"].unique())

stage_order = [
    "Group Stage",
    "Round of 32",
    "Round of 16",
    "Quarter-final",
    "Semi-final",
    "Final",
    "Champion",
]


def resolve_host(team1, team2, country):
    if team1 == country:
        return team1, team2, False
    elif team2 == country:
        return team2, team1, False
    else:
        return team1, team2, True


def play_knockout_match(team1, team2, country):
    host, other, neutral = resolve_host(team1, team2, country)
    elo_diff = elo_lookup[host] - elo_lookup[other]
    X_new = pd.DataFrame({"elo_diff": [elo_diff], "neutral": [neutral]})
    lambda_host = model_home.predict(X_new)[0]
    lambda_other = model_away.predict(X_new)[0]

    score_host = np.random.poisson(lambda_host)
    score_other = np.random.poisson(lambda_other)

    if score_host > score_other:
        return host, other
    elif score_host < score_other:
        return other, host
    else:
        p_host_wins = 1 / (1 + 10 ** (-elo_diff / 400))
        if np.random.random() < p_host_wins:
            return host, other
        else:
            return other, host


# --- Pre-calcul des lambdas de phase de groupes (fixes, hors boucle Monte Carlo) ---

all_group_rows = []
for group_letter in group_letters:
    group_fixtures = fixtures[fixtures["group"] == group_letter]
    for _, row in group_fixtures.iterrows():
        host, other, neutral = resolve_host(row["team1"], row["team2"], row["country"])
        elo_diff = elo_lookup[host] - elo_lookup[other]
        all_group_rows.append(
            {"group": group_letter, "host": host, "other": other, "elo_diff": elo_diff, "neutral": neutral}
        )

all_group_rows_df = pd.DataFrame(all_group_rows)
X_all = all_group_rows_df[["elo_diff", "neutral"]]
all_group_rows_df["lambda_host"] = model_home.predict(X_all)
all_group_rows_df["lambda_other"] = model_away.predict(X_all)

group_match_params = {}
for group_letter, group_df in all_group_rows_df.groupby("group"):
    group_match_params[group_letter] = list(
        group_df[["host", "other", "lambda_host", "lambda_other"]].itertuples(index=False, name=None)
    )


def play_group_fast(group_letter):
    group_teams = teams[teams["group"] == group_letter]["team"].tolist()
    standings = {team: {"points": 0, "gf": 0, "ga": 0} for team in group_teams}

    for host, other, lambda_host, lambda_other in group_match_params[group_letter]:
        score_host = np.random.poisson(lambda_host)
        score_other = np.random.poisson(lambda_other)

        standings[host]["gf"] += score_host
        standings[host]["ga"] += score_other
        standings[other]["gf"] += score_other
        standings[other]["ga"] += score_host

        if score_host > score_other:
            standings[host]["points"] += 3
        elif score_host < score_other:
            standings[other]["points"] += 3
        else:
            standings[host]["points"] += 1
            standings[other]["points"] += 1

    return sorted(standings.items(), key=lambda x: (x[1]["points"], x[1]["gf"] - x[1]["ga"]), reverse=True)


def play_round(stage_name, label_prefix, prev_map):
    round_fixtures = fixtures[fixtures["stage"] == stage_name].copy()
    round_fixtures["team1_real"] = round_fixtures["team1"].map(prev_map)
    round_fixtures["team2_real"] = round_fixtures["team2"].map(prev_map)

    hosts, others, neutrals = [], [], []
    for _, row in round_fixtures.iterrows():
        host, other, neutral = resolve_host(row["team1_real"], row["team2_real"], row["country"])
        hosts.append(host)
        others.append(other)
        neutrals.append(neutral)

    elo_diffs = [elo_lookup[h] - elo_lookup[o] for h, o in zip(hosts, others)]
    X_round = pd.DataFrame({"elo_diff": elo_diffs, "neutral": neutrals})
    lambda_hosts = model_home.predict(X_round)
    lambda_others = model_away.predict(X_round)

    winners, losers = [], []
    for host, other, lambda_host, lambda_other in zip(hosts, others, lambda_hosts, lambda_others):
        score_host = np.random.poisson(lambda_host)
        score_other = np.random.poisson(lambda_other)

        if score_host > score_other:
            winners.append(host)
            losers.append(other)
        elif score_host < score_other:
            winners.append(other)
            losers.append(host)
        else:
            elo_diff = elo_lookup[host] - elo_lookup[other]
            p_host_wins = 1 / (1 + 10 ** (-elo_diff / 400))
            if np.random.random() < p_host_wins:
                winners.append(host)
                losers.append(other)
            else:
                winners.append(other)
                losers.append(host)

    round_fixtures["winner"] = winners
    round_fixtures["loser"] = losers
    winners_map = {f"{label_prefix}{i}": w for i, w in enumerate(winners, start=1)}
    return round_fixtures, winners_map


def simulate_tournament():
    stage_reached = {team: "Group Stage" for team in teams["team"]}
    group_rankings = {}
    third_placed = []

    for group_letter in group_letters:
        ranking = play_group_fast(group_letter)
        group_rankings[group_letter] = ranking
        stage_reached[ranking[0][0]] = "Round of 32"
        stage_reached[ranking[1][0]] = "Round of 32"
        third_team, third_stats = ranking[2]
        third_placed.append((group_letter, third_team, third_stats))

    third_placed_sorted = sorted(
        third_placed,
        key=lambda x: (x[2]["points"], x[2]["gf"] - x[2]["ga"], x[2]["gf"]),
        reverse=True,
    )

    placeholder_map = {}
    for group_letter, ranking in group_rankings.items():
        placeholder_map[f"1{group_letter}"] = ranking[0][0]
        placeholder_map[f"2{group_letter}"] = ranking[1][0]
    for position, (group_letter, team, stats) in enumerate(third_placed_sorted[:8], start=1):
        placeholder_map[f"Best 3rd #{position}"] = team
        stage_reached[team] = "Round of 32"

    r32_results, r32_winners_map = play_round("Round of 32", "R32 W", placeholder_map)
    for team in pd.concat([r32_results["team1_real"], r32_results["team2_real"]]):
        stage_reached[team] = "Round of 32"

    r16_results, qf_map = play_round("Round of 16", "QF", r32_winners_map)
    for team in pd.concat([r16_results["team1_real"], r16_results["team2_real"]]):
        stage_reached[team] = "Round of 16"

    qf_results, sf_map = play_round("Quarter-final", "SF", qf_map)
    for team in pd.concat([qf_results["team1_real"], qf_results["team2_real"]]):
        stage_reached[team] = "Quarter-final"

    sf_results, _ = play_round("Semi-final", "Finalist ", sf_map)
    for team in pd.concat([sf_results["team1_real"], sf_results["team2_real"]]):
        stage_reached[team] = "Semi-final"

    final_map = {f"Finalist {i}": w for i, w in enumerate(sf_results["winner"], start=1)}
    final_fixture = fixtures[fixtures["stage"] == "Final"].iloc[0]
    team1_final = final_map[final_fixture["team1"]]
    team2_final = final_map[final_fixture["team2"]]
    stage_reached[team1_final] = "Final"
    stage_reached[team2_final] = "Final"

    champion, runner_up = play_knockout_match(team1_final, team2_final, final_fixture["country"])
    stage_reached[champion] = "Champion"

    return stage_reached


# ---------------------------------------------------------------------------
# 6. Boucle Monte Carlo complete
# ---------------------------------------------------------------------------

n_sim = 1000
reach_counts = {team: {stage: 0 for stage in stage_order} for team in teams["team"]}

for sim in range(n_sim):
    result = simulate_tournament()
    for team, stage in result.items():
        reach_counts[team][stage] += 1

summary = pd.DataFrame.from_dict(reach_counts, orient="index")
summary = summary[stage_order]
summary = (summary / n_sim * 100).round(1)
summary = summary.sort_values("Champion", ascending=False)

print("\n--- Resultats (probabilites en %) ---")
print(summary)

summary.to_csv("../data/processed/wc2026_stage_probabilities.csv")
print("\nSauvegarde : data/processed/wc2026_stage_probabilities.csv")
