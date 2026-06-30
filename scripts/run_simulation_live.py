"""
Simulation 'live' de la Coupe du Monde 2026 : utilise les vrais resultats
de phase de groupes deja joues (presents dans results.csv) et ne simule
que les matchs restants (fin de phase de groupes + phases finales).

Variante de run_simulation.py, qui repartait de zero meme pour des
matchs deja joues. Ici les matchs reels comptent comme des points/buts
fixes, identiques dans toutes les simulations Monte Carlo ; seul ce qui
n'a pas encore ete joue est tire au hasard.

Lancer depuis le dossier scripts/ : python run_simulation_live.py
"""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Chargement + harmonisation (identique a run_simulation.py)
# ---------------------------------------------------------------------------

results = pd.read_csv("../data/raw/results.csv")
elo = pd.read_csv("../data/raw/elo_ratings_wc2026.csv")
teams = pd.read_csv("../data/raw/wc_2026_teams.csv")
fixtures = pd.read_csv("../data/raw/wc_2026_fixtures.csv")

latest_date = elo["snapshot_date"].max()
elo_latest = elo[elo["snapshot_date"] == latest_date].copy()

teams_ref = set(teams["team"])

name_mapping = {
    "United States": "USA",
    "Turkey": "Türkiye",
    "Czech Republic": "Czechia",
}

elo_latest["country"] = elo_latest["country"].replace(name_mapping)
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

# Config (objectif Poisson, profondeur 3, contrainte de monotonie) choisie
# via model_selection.py sur un vrai split train<2014/test>=2014 -- voir
# backtest_worldcups.py pour la validation hors-echantillon. Ici on
# reentraine sur TOUTES les donnees pour le modele de production.
features = ["elo_diff", "neutral"]
train = matches_clean
X_train = train[features]

MODEL_PARAMS = dict(n_estimators=100, random_state=42, objective="count:poisson", max_depth=3)

model_home = XGBRegressor(**MODEL_PARAMS, monotone_constraints=(1, 0))
model_home.fit(X_train, train["home_score"])

model_away = XGBRegressor(**MODEL_PARAMS, monotone_constraints=(-1, 0))
model_away.fit(X_train, train["away_score"])

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


# ---------------------------------------------------------------------------
# 2. Recuperer les vrais matchs de Coupe du Monde 2026 deja joues
# ---------------------------------------------------------------------------

real_2026 = results[
    (results["tournament"] == "FIFA World Cup")
    & (results["year"] == 2026)
    & results["home_score"].notna()
].copy()
real_lookup = {
    frozenset([row["home_team"], row["away_team"]]): (row["home_team"], row["away_team"], row["home_score"], row["away_score"])
    for _, row in real_2026.iterrows()
}
# NB : on matche par PAIRE D'EQUIPES, pas par date exacte -- la date reelle
# du match dans results.csv peut differer de quelques jours (parfois +5j)
# de la date programmee dans wc_2026_fixtures.csv. Un matching par date exacte
# faisait passer a tort des matchs deja joues pour "a venir".

print(f"Matchs reels de phase de groupes 2026 deja joues trouves : {len(real_lookup)}")


def get_real_score(team1, team2):
    """Renvoie (score_team1, score_team2) si ce match a deja ete joue, sinon None."""
    key = frozenset([team1, team2])
    if key not in real_lookup:
        return None
    home_team, away_team, home_score, away_score = real_lookup[key]
    if home_team == team1:
        return home_score, away_score
    else:
        return away_score, home_score


# ---------------------------------------------------------------------------
# 3. Separer les fixtures de groupe en "deja joues" (resultat fixe) et
#    "a venir" (a simuler), et pre-calculer les lambdas pour ces dernieres
# ---------------------------------------------------------------------------

base_standings = {
    group_letter: {team: {"points": 0, "gf": 0, "ga": 0} for team in teams[teams["group"] == group_letter]["team"]}
    for group_letter in group_letters
}

pending_rows = []
n_played, n_pending = 0, 0

for group_letter in group_letters:
    group_fixtures = fixtures[fixtures["group"] == group_letter]
    for _, row in group_fixtures.iterrows():
        host, other, neutral = resolve_host(row["team1"], row["team2"], row["country"])
        real_score = get_real_score(row["team1"], row["team2"])

        if real_score is not None:
            score_team1, score_team2 = real_score
            score_host = score_team1 if host == row["team1"] else score_team2
            score_other = score_team2 if host == row["team1"] else score_team1

            base_standings[group_letter][host]["gf"] += score_host
            base_standings[group_letter][host]["ga"] += score_other
            base_standings[group_letter][other]["gf"] += score_other
            base_standings[group_letter][other]["ga"] += score_host

            if score_host > score_other:
                base_standings[group_letter][host]["points"] += 3
            elif score_host < score_other:
                base_standings[group_letter][other]["points"] += 3
            else:
                base_standings[group_letter][host]["points"] += 1
                base_standings[group_letter][other]["points"] += 1
            n_played += 1
        else:
            elo_diff = elo_lookup[host] - elo_lookup[other]
            pending_rows.append(
                {"group": group_letter, "host": host, "other": other, "elo_diff": elo_diff, "neutral": neutral}
            )
            n_pending += 1

print(f"Matchs de groupe : {n_played} deja joues (resultat fixe), {n_pending} a venir (simules)")

pending_df = pd.DataFrame(pending_rows)
if len(pending_df) > 0:
    X_pending = pending_df[["elo_diff", "neutral"]]
    pending_df["lambda_host"] = model_home.predict(X_pending)
    pending_df["lambda_other"] = model_away.predict(X_pending)

group_match_params = {group_letter: [] for group_letter in group_letters}
if len(pending_df) > 0:
    for group_letter, group_df in pending_df.groupby("group"):
        group_match_params[group_letter] = list(
            group_df[["host", "other", "lambda_host", "lambda_other"]].itertuples(index=False, name=None)
        )


def play_group_live(group_letter):
    standings = {team: dict(stats) for team, stats in base_standings[group_letter].items()}

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


# ---------------------------------------------------------------------------
# 4. Phases finales : rien n'est encore joue, simulation classique
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Structure reelle du tableau final 2026 (Round of 32 -> Champion), verifiee
# depuis le wikicode source de Wikipedia (cf. memoire projet). Remplace
# l'ancien schema "1A v 2B" de wc_2026_fixtures.csv, qui s'est revele FAUX
# (le vrai tableau FIFA n'est pas un simple "1er de groupe contre 2eme du
# groupe suivant" -- certains matchs opposent deux 2emes entre eux, et les
# meilleurs 3emes sont distribues individuellement contre des 1ers de groupe,
# jamais entre eux). La phase de groupes 2026 est entierement terminee
# (72/72 matchs reels connus), donc ces paires de Round of 32 sont des faits
# fixes, plus une consequence d'un tirage de groupe simule.
# ---------------------------------------------------------------------------

R32_INFO = {
    73: ("South Africa", "Canada", "United States"),
    74: ("Germany", "Paraguay", "United States"),
    75: ("Netherlands", "Morocco", "Mexico"),
    76: ("Brazil", "Japan", "United States"),
    77: ("France", "Sweden", "United States"),
    78: ("Ivory Coast", "Norway", "United States"),
    79: ("Mexico", "Ecuador", "Mexico"),
    80: ("England", "DR Congo", "United States"),
    81: ("USA", "Bosnia and Herzegovina", "United States"),
    82: ("Belgium", "Senegal", "United States"),
    83: ("Portugal", "Croatia", "Canada"),
    84: ("Spain", "Austria", "United States"),
    85: ("Switzerland", "Algeria", "Canada"),
    86: ("Argentina", "Cape Verde", "United States"),
    87: ("Colombia", "Ghana", "United States"),
    88: ("Australia", "Egypt", "United States"),
}
# Vainqueurs deja connus (matchs joues fin juin 2026) ; 74 et 75 se sont
# joues nul (1-1) puis decides aux tirs au but, donc le vainqueur ne peut
# pas se deduire du score seul (non stocke dans results.csv).
REAL_R32_WINNERS = {73: "Canada", 74: "Paraguay", 75: "Morocco", 76: "Brazil"}

R16_WIRING = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
              93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
R16_VENUE = {89: "United States", 90: "United States", 91: "United States", 92: "Mexico",
             93: "United States", 94: "United States", 95: "United States", 96: "Canada"}

QF_WIRING = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
QF_VENUE = {97: "United States", 98: "United States", 99: "United States", 100: "United States"}

SF_WIRING = {101: (97, 98), 102: (99, 100)}
SF_VENUE = {101: "United States", 102: "United States"}

FINAL_WIRING = (101, 102)
FINAL_VENUE = "United States"

ALL_R32_TEAMS = {team for pair in R32_INFO.values() for team in pair[:2]}


def simulate_tournament():
    stage_reached = {team: "Group Stage" for team in teams["team"]}

    # Phase de groupes deja entierement jouee (72/72 matchs reels) ; ce
    # tirage est donc deterministe d'une simulation a l'autre, conserve ici
    # uniquement par coherence avec base_standings (aucune influence sur le
    # tableau final, dont les paires sont fixees dans R32_INFO ci-dessus).
    for group_letter in group_letters:
        play_group_live(group_letter)

    for team in ALL_R32_TEAMS:
        stage_reached[team] = "Round of 32"

    winner_of = {}
    for num, (team1, team2, country) in R32_INFO.items():
        if num in REAL_R32_WINNERS:
            winner_of[num] = REAL_R32_WINNERS[num]
        else:
            winner, loser = play_knockout_match(team1, team2, country)
            winner_of[num] = winner

    for num, (a, b) in R16_WIRING.items():
        team1, team2 = winner_of[a], winner_of[b]
        stage_reached[team1] = "Round of 16"
        stage_reached[team2] = "Round of 16"
        winner, loser = play_knockout_match(team1, team2, R16_VENUE[num])
        winner_of[num] = winner

    for num, (a, b) in QF_WIRING.items():
        team1, team2 = winner_of[a], winner_of[b]
        stage_reached[team1] = "Quarter-final"
        stage_reached[team2] = "Quarter-final"
        winner, loser = play_knockout_match(team1, team2, QF_VENUE[num])
        winner_of[num] = winner

    for num, (a, b) in SF_WIRING.items():
        team1, team2 = winner_of[a], winner_of[b]
        stage_reached[team1] = "Semi-final"
        stage_reached[team2] = "Semi-final"
        winner, loser = play_knockout_match(team1, team2, SF_VENUE[num])
        winner_of[num] = winner

    a, b = FINAL_WIRING
    team1_final, team2_final = winner_of[a], winner_of[b]
    stage_reached[team1_final] = "Final"
    stage_reached[team2_final] = "Final"

    champion, runner_up = play_knockout_match(team1_final, team2_final, FINAL_VENUE)
    stage_reached[champion] = "Champion"

    return stage_reached


# ---------------------------------------------------------------------------
# 5. Boucle Monte Carlo
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

print("\n--- Resultats live (probabilites en %, a partir des resultats reels connus) ---")
print(summary)

summary.to_csv("../data/processed/wc2026_stage_probabilities_live.csv")
print("\nSauvegarde : data/processed/wc2026_stage_probabilities_live.csv")
