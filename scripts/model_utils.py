"""
Constantes et fonctions partagees entre run_simulation_live.py et
generate_bracket_pdf.py.

SEUL ENDROIT A METTRE A JOUR apres chaque journee de matchs :
  -> ajouter les nouveaux vainqueurs dans REAL_R32_WINNERS
"""

import pandas as pd

# ---------------------------------------------------------------------------
# Tableau officiel FIFA 2026 — Round of 32 -> Final
# (match number -> (team1, team2, host_nation))
# Source : wikicode Wikipedia, valide contre 16/16 resultats reels.
# ---------------------------------------------------------------------------

R32_INFO = {
    73: ("South Africa", "Canada",                "United States"),
    74: ("Germany",      "Paraguay",               "United States"),
    75: ("Netherlands",  "Morocco",                "Mexico"),
    76: ("Brazil",       "Japan",                  "United States"),
    77: ("France",       "Sweden",                 "United States"),
    78: ("Ivory Coast",  "Norway",                 "United States"),
    79: ("Mexico",       "Ecuador",                "Mexico"),
    80: ("England",      "DR Congo",               "United States"),
    81: ("USA",          "Bosnia and Herzegovina", "United States"),
    82: ("Belgium",      "Senegal",                "United States"),
    83: ("Portugal",     "Croatia",                "Canada"),
    84: ("Spain",        "Austria",                "United States"),
    85: ("Switzerland",  "Algeria",                "Canada"),
    86: ("Argentina",    "Cape Verde",             "United States"),
    87: ("Colombia",     "Ghana",                  "United States"),
    88: ("Australia",    "Egypt",                  "United States"),
}

# Vainqueurs confirmes — mettre a jour apres chaque journee de matchs.
# Matches 74 et 75 : termines 1-1, decides aux tirs au but (score non
# disponible dans results.csv, d'ou l'entree manuelle du vainqueur).
REAL_R32_WINNERS = {
    73: "Canada",
    74: "Paraguay",
    75: "Morocco",
    76: "Brazil",
    77: "France",
    78: "Norway",
    79: "Mexico",
}

R16_WIRING = {89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
              93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87)}
R16_VENUE   = {89: "United States", 90: "United States", 91: "United States",
               92: "Mexico", 93: "United States", 94: "United States",
               95: "United States", 96: "Canada"}

QF_WIRING = {97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96)}
QF_VENUE  = {97: "United States", 98: "United States",
             99: "United States", 100: "United States"}

SF_WIRING = {101: (97, 98), 102: (99, 100)}
SF_VENUE  = {101: "United States", 102: "United States"}

FINAL_WIRING = (101, 102)
FINAL_VENUE  = "United States"

ALL_R32_TEAMS = {team for entry in R32_INFO.values() for team in entry[:2]}

# Ordres d'affichage pour le PDF (paires consecutives = match suivant)
R32_ORDER = [74, 77, 73, 75, 83, 84, 81, 82, 76, 78, 79, 80, 86, 88, 85, 87]
R16_ORDER = [89, 90, 93, 94, 91, 92, 95, 96]
QF_ORDER  = [97, 98, 99, 100]
SF_ORDER  = [101, 102]


# ---------------------------------------------------------------------------
# Elo en temps reel — calcule depuis results.csv, pas de dependance externe
# ---------------------------------------------------------------------------

def compute_tournament_elo(wc_matches: pd.DataFrame, pre_elo: dict, K: int = 40) -> dict:
    """
    Applique les mises a jour Elo pour chaque match WC2026 deja joue
    (ordre chronologique) sur les ratings pre-tournoi (snapshot mai 2026).

    Formule identique a eloratings.net : K=40 pour les matchs de Coupe
    du Monde, facteur G de difference de buts, tirage = 0.5.
    Avantage : capture les matchs du jour immediatement, sans attendre
    la mise a jour d'eloratings.net (delai de 24-48h).
    """
    elo_dict = dict(pre_elo)
    for _, m in wc_matches.sort_values("date").iterrows():
        home, away = m["home_team"], m["away_team"]
        if home not in elo_dict or away not in elo_dict:
            continue
        elo_h, elo_a = elo_dict[home], elo_dict[away]
        expected_h = 1 / (1 + 10 ** ((elo_a - elo_h) / 400))
        gd = abs(m["home_score"] - m["away_score"])
        G = 1 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        if m["home_score"] > m["away_score"]:
            result_h = 1.0
        elif m["home_score"] < m["away_score"]:
            result_h = 0.0
        else:
            result_h = 0.5
        delta = K * G * (result_h - expected_h)
        elo_dict[home] = elo_h + delta
        elo_dict[away] = elo_a - delta
    return elo_dict
