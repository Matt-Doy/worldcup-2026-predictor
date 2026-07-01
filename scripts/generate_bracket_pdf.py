"""
Genere un PDF de l'arbre de predictions (phase finale, Round of 32 -> Champion)
de la Coupe du Monde 2026, en partant des vrais resultats deja connus
(comme run_simulation_live.py) puis en propageant, a chaque match restant,
le vainqueur le PLUS PROBABLE (pas un tirage aleatoire unique) pour obtenir
un seul arbre coherent.

Phase de groupes : les matchs deja joues comptent tels quels ; les matchs
pas encore joues sont resolus par l'issue la plus probable (victoire/nul/
defaite) selon le modele.

Phase finale (Round of 32 -> Champion) : pas de match nul possible (prolongation
+ tirs au but), donc le "vainqueur le plus probable" inclut une estimation de
qui gagnerait une seance de tirs au but (via le delta Elo), comme dans
run_simulation_live.py.

Lancer depuis scripts/ : python generate_bracket_pdf.py
Sortie : ../data/processed/wc2026_bracket_prediction.pdf
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch, Rectangle
from xgboost import XGBRegressor
from model_utils import (
    compute_tournament_elo,
    R32_INFO, REAL_R32_WINNERS,
    R16_WIRING, R16_VENUE,
    QF_WIRING, QF_VENUE,
    SF_WIRING, SF_VENUE,
    FINAL_WIRING, FINAL_VENUE,
    R32_ORDER, R16_ORDER, QF_ORDER, SF_ORDER,
)

NAVY = "#1a3a5c"
NAVY_LIGHT = "#3d6ea5"
GOLD = "#c9a227"
GREEN_BG = "#eaf6ec"
GREY_BG = "#f2f2f2"
GREY_TEXT = "#9a9a9a"

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. Chargement + entrainement (identique a run_simulation_live.py)
# ---------------------------------------------------------------------------

results = pd.read_csv("../data/raw/results.csv")
elo = pd.read_csv("../data/raw/elo_ratings_wc2026.csv")
teams = pd.read_csv("../data/raw/wc_2026_teams.csv")
fixtures = pd.read_csv("../data/raw/wc_2026_fixtures.csv")

latest_date = elo["snapshot_date"].max()
elo_latest = elo[elo["snapshot_date"] == latest_date].copy()
teams_ref = set(teams["team"])

name_mapping = {"United States": "USA", "Turkey": "Türkiye", "Czech Republic": "Czechia"}
elo_latest["country"] = elo_latest["country"].replace(name_mapping)
results["home_team"] = results["home_team"].replace(name_mapping)
results["away_team"] = results["away_team"].replace(name_mapping)
results["date"] = pd.to_datetime(results["date"])
results["year"] = results["date"].dt.year

mask = (
    results["home_team"].isin(teams_ref) & results["away_team"].isin(teams_ref)
    & (results["year"] >= 1901) & results["home_score"].notna()
)
matches = results[mask].copy()
matches["join_year"] = matches["year"] - 1
elo_simple = elo[["country", "year", "rating"]]
matches = matches.merge(elo_simple.rename(columns={"country": "home_team", "year": "join_year", "rating": "home_elo"}),
                         on=["home_team", "join_year"], how="left")
matches = matches.merge(elo_simple.rename(columns={"country": "away_team", "year": "join_year", "rating": "away_elo"}),
                         on=["away_team", "join_year"], how="left")

mask_complete = matches["home_elo"].notna() & matches["away_elo"].notna()
matches_clean = matches[mask_complete].copy()
matches_clean["elo_diff"] = matches_clean["home_elo"] - matches_clean["away_elo"]

features = ["elo_diff", "neutral"]
train = matches_clean
MODEL_PARAMS = dict(n_estimators=100, random_state=42, objective="count:poisson", max_depth=3)
model_home = XGBRegressor(**MODEL_PARAMS, monotone_constraints=(1, 0))
model_home.fit(train[features], train["home_score"])
model_away = XGBRegressor(**MODEL_PARAMS, monotone_constraints=(-1, 0))
model_away.fit(train[features], train["away_score"])

group_letters = sorted(teams["group"].unique())


def resolve_host(team1, team2, country):
    if team1 == country:
        return team1, team2, False
    elif team2 == country:
        return team2, team1, False
    else:
        return team1, team2, True


# ---------------------------------------------------------------------------
# 2. Vrais resultats 2026 + Elo mis a jour match par match
# ---------------------------------------------------------------------------

real_2026 = results[
    (results["tournament"] == "FIFA World Cup") & (results["year"] == 2026) & results["home_score"].notna()
].copy()
real_lookup = {
    frozenset([row["home_team"], row["away_team"]]): (row["home_team"], row["away_team"], row["home_score"], row["away_score"])
    for _, row in real_2026.iterrows()
}
# NB : on matche par PAIRE D'EQUIPES, pas par date exacte.


pre_tournament = elo[elo["snapshot_date"] == "2026-05-27"].copy()
pre_tournament["country"] = pre_tournament["country"].replace(name_mapping)
pre_elo_base = pre_tournament.set_index("country")["rating"].to_dict()
elo_lookup = pd.Series(compute_tournament_elo(real_2026, pre_elo_base))


def get_real_score(team1, team2):
    key = frozenset([team1, team2])
    if key not in real_lookup:
        return None
    home_team, away_team, home_score, away_score = real_lookup[key]
    if home_team == team1:
        return home_score, away_score
    else:
        return away_score, home_score


# ---------------------------------------------------------------------------
# 3. Phase de groupes : resultats reels (fixes) + issue la plus probable
#    pour les matchs restants (pas de tirage aleatoire : on prend l'issue
#    de plus forte probabilite, pour produire UN SEUL arbre coherent).
# ---------------------------------------------------------------------------

standings = {
    g: {t: {"points": 0, "gf": 0, "ga": 0} for t in teams[teams["group"] == g]["team"]}
    for g in group_letters
}
match_log = []  # pour la page recap (reel ou predit)

for group_letter in group_letters:
    group_fixtures = fixtures[fixtures["group"] == group_letter]
    for _, row in group_fixtures.iterrows():
        host, other, neutral = resolve_host(row["team1"], row["team2"], row["country"])
        real_score = get_real_score(row["team1"], row["team2"])

        if real_score is not None:
            s_team1, s_team2 = real_score
            score_host = s_team1 if host == row["team1"] else s_team2
            score_other = s_team2 if host == row["team1"] else s_team1
            is_real = True
        else:
            elo_diff = elo_lookup[host] - elo_lookup[other]
            X_new = pd.DataFrame({"elo_diff": [elo_diff], "neutral": [neutral]})
            lambda_host = model_home.predict(X_new)[0]
            lambda_other = model_away.predict(X_new)[0]

            n_sim = 5000
            sim_h = np.random.poisson(lambda_host, n_sim)
            sim_o = np.random.poisson(lambda_other, n_sim)
            p_h, p_d, p_o = (sim_h > sim_o).mean(), (sim_h == sim_o).mean(), (sim_h < sim_o).mean()
            outcome = max([("home", p_h), ("draw", p_d), ("away", p_o)], key=lambda x: x[1])[0]

            score_host, score_other = round(lambda_host), round(lambda_other)
            if outcome == "home" and score_host <= score_other:
                score_host = score_other + 1
            elif outcome == "away" and score_other <= score_host:
                score_other = score_host + 1
            elif outcome == "draw" and score_host != score_other:
                score_other = score_host
            is_real = False

        standings[group_letter][host]["gf"] += score_host
        standings[group_letter][host]["ga"] += score_other
        standings[group_letter][other]["gf"] += score_other
        standings[group_letter][other]["ga"] += score_host
        if score_host > score_other:
            standings[group_letter][host]["points"] += 3
        elif score_host < score_other:
            standings[group_letter][other]["points"] += 3
        else:
            standings[group_letter][host]["points"] += 1
            standings[group_letter][other]["points"] += 1

        match_log.append(dict(group=group_letter, team1=host, team2=other,
                               score1=score_host, score2=score_other, real=is_real))

group_rankings = {}
third_placed = []
for group_letter in group_letters:
    ranking = sorted(standings[group_letter].items(), key=lambda x: (x[1]["points"], x[1]["gf"] - x[1]["ga"], x[1]["gf"]), reverse=True)
    group_rankings[group_letter] = ranking
    third_placed.append((group_letter, ranking[2][0], ranking[2][1]))

third_placed_sorted = sorted(third_placed, key=lambda x: (x[2]["points"], x[2]["gf"] - x[2]["ga"], x[2]["gf"]), reverse=True)

placeholder_map = {}
for group_letter, ranking in group_rankings.items():
    placeholder_map[f"1{group_letter}"] = ranking[0][0]
    placeholder_map[f"2{group_letter}"] = ranking[1][0]
for position, (group_letter, team, stats) in enumerate(third_placed_sorted[:8], start=1):
    placeholder_map[f"Best 3rd #{position}"] = team

qualified_thirds = {g for g, _, _ in third_placed_sorted[:8]}

# ---------------------------------------------------------------------------
# 4. Phase finale : vainqueur le plus probable a chaque match (victoire +
#    nul x probabilite de gagner aux tirs au but, via Elo).
#
#    IMPORTANT : la structure du tableau final (qui affronte qui a partir
#    du Round of 32) NE SUIT PAS le schema simpliste "1A contre 2B" present
#    dans wc_2026_fixtures.csv -- ce schema s'est revele FAUX (verifie : les
#    vrais resultats connus, ex. Canada-Morocco en 8e, ne correspondent pas
#    du tout a ce que ce schema produit). Le vrai tableau officiel FIFA 2026
#    a ete recupere et verifie directement depuis le wikicode source de
#    Wikipedia (pages "2026 FIFA World Cup knockout stage" et
#    "...round of 32"), recoupe avec les 4 resultats reels deja connus
#    (cf. memoire projet pour le detail de la verification).
# ---------------------------------------------------------------------------

def predict_knockout(team1, team2, country, n_sim=5000):
    host, other, neutral = resolve_host(team1, team2, country)
    elo_diff = elo_lookup[host] - elo_lookup[other]
    X_new = pd.DataFrame({"elo_diff": [elo_diff], "neutral": [neutral]})
    lambda_host = model_home.predict(X_new)[0]
    lambda_other = model_away.predict(X_new)[0]

    sim_h = np.random.poisson(lambda_host, n_sim)
    sim_o = np.random.poisson(lambda_other, n_sim)
    p_host = (sim_h > sim_o).mean()
    p_draw = (sim_h == sim_o).mean()
    p_other = (sim_h < sim_o).mean()

    p_host_penalty = 1 / (1 + 10 ** (-elo_diff / 400))
    p_host_total = p_host + p_draw * p_host_penalty
    p_other_total = p_other + p_draw * (1 - p_host_penalty)

    if p_host_total >= p_other_total:
        return host, other, p_host_total
    else:
        return other, host, p_other_total


winner_of = {}

r32_records = []
for num in R32_ORDER:
    team1, team2, country = R32_INFO[num]
    if num in REAL_R32_WINNERS:
        winner = REAL_R32_WINNERS[num]
        loser = team2 if winner == team1 else team1
        r32_records.append(dict(team1=team1, team2=team2, winner=winner, prob=1.0, is_real=True))
    else:
        winner, loser, prob = predict_knockout(team1, team2, country)
        r32_records.append(dict(team1=team1, team2=team2, winner=winner, prob=prob, is_real=False))
    winner_of[num] = winner

r16_records = []
for num in R16_ORDER:
    a, b = R16_WIRING[num]
    team1, team2 = winner_of[a], winner_of[b]
    winner, loser, prob = predict_knockout(team1, team2, R16_VENUE[num])
    r16_records.append(dict(team1=team1, team2=team2, winner=winner, prob=prob, is_real=False))
    winner_of[num] = winner

qf_records = []
for num in QF_ORDER:
    a, b = QF_WIRING[num]
    team1, team2 = winner_of[a], winner_of[b]
    winner, loser, prob = predict_knockout(team1, team2, QF_VENUE[num])
    qf_records.append(dict(team1=team1, team2=team2, winner=winner, prob=prob, is_real=False))
    winner_of[num] = winner

sf_records = []
for num in SF_ORDER:
    a, b = SF_WIRING[num]
    team1, team2 = winner_of[a], winner_of[b]
    winner, loser, prob = predict_knockout(team1, team2, SF_VENUE[num])
    sf_records.append(dict(team1=team1, team2=team2, winner=winner, prob=prob, is_real=False))
    winner_of[num] = winner

a, b = FINAL_WIRING
team1_f, team2_f = winner_of[a], winner_of[b]
champion, runner_up, champ_prob = predict_knockout(team1_f, team2_f, FINAL_VENUE)

print(f"Champion predit : {champion} ({champ_prob:.1%} contre {runner_up})")

# ---------------------------------------------------------------------------
# 5. Page 1 : classement des chances de titre (probabilites Monte Carlo,
#    issues de run_simulation_live.py -- a executer avant ce script). Ce
#    classement reflete TOUTES les facons de gagner/perdre a chaque tour
#    (1000 tirages), contrairement a l'arbre page 3 qui ne montre qu'UN
#    seul chemin (le plus probable a chaque match, enchaine).
# ---------------------------------------------------------------------------

prob_path = "../data/processed/wc2026_stage_probabilities_live.csv"
prob_df = pd.read_csv(prob_path, index_col=0)
champion_ranked = prob_df["Champion"].sort_values(ascending=False)
top_n = champion_ranked[champion_ranked > 0].head(20)

fig3, ax3 = plt.subplots(figsize=(11, 11))
y_pos = list(range(len(top_n)))[::-1]

bar_colors = []
for i in range(len(top_n)):
    if i == 0:
        bar_colors.append(GOLD)
    elif i == 1:
        bar_colors.append("#9aa7b5")
    elif i == 2:
        bar_colors.append("#a9734c")
    else:
        bar_colors.append(NAVY_LIGHT)

ax3.barh(y_pos, top_n.values, color=bar_colors, edgecolor="white", height=0.68, zorder=3)
ax3.set_yticks(y_pos)
ax3.set_yticklabels(top_n.index, fontsize=10.5)
ax3.set_xlim(0, top_n.values.max() * 1.18)
for y, v in zip(y_pos, top_n.values):
    ax3.text(v + top_n.values.max() * 0.015, y, f"{v:.1f}%", va="center", fontsize=9.5,
              color="#333333", fontweight="bold")

ax3.set_xlabel("Probabilite de remporter la Coupe du Monde 2026 (%)", fontsize=10, color="#555555")
ax3.spines[["top", "right", "left"]].set_visible(False)
ax3.tick_params(left=False)
ax3.xaxis.grid(True, color="#e3e3e3", lw=0.8, zorder=0)
ax3.set_axisbelow(True)
ax3.set_title(
    "Coupe du Monde 2026 — Probabilite de titre par equipe\n"
    "Simulation Monte Carlo (1000 tirages), conditionnee sur tous les vrais resultats connus au 30/06/2026.",
    fontsize=13, fontweight="bold", color=NAVY, loc="left", pad=14
)

# ---------------------------------------------------------------------------
# 6. Page 2 : phase de groupes (grille de 12 mini-tableaux + meilleurs 3emes)
# ---------------------------------------------------------------------------

fig2, ax2 = plt.subplots(figsize=(15, 11))
ax2.set_axis_off()
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_title(
    "Phase de groupes — qualifies pour le Round of 32\n"
    "(resultats reels + matchs restants resolus par l'issue la plus probable du modele)",
    fontsize=13, fontweight="bold", color=NAVY, loc="left", pad=14
)


def draw_group_cell(ax, x0, y0, w, h, group_letter, ranking, qualified_thirds):
    header_h = h * 0.20
    ax.add_patch(Rectangle((x0, y0 - header_h), w, header_h, facecolor=NAVY, edgecolor="none", zorder=2))
    ax.text(x0 + w / 2, y0 - header_h / 2, f"Groupe {group_letter}", ha="center", va="center",
            fontsize=10.5, fontweight="bold", color="white", zorder=3)

    row_h = (h - header_h) / 4
    for i, (team, stats) in enumerate(ranking[:4]):
        qualified = i < 2 or (i == 2 and group_letter in qualified_thirds)
        ry = y0 - header_h - (i + 1) * row_h
        facecolor = GREEN_BG if qualified else GREY_BG
        textcolor = "black" if qualified else GREY_TEXT
        weight = "bold" if qualified else "normal"
        marker = "* " if (i == 2 and group_letter in qualified_thirds) else ""
        ax.add_patch(Rectangle((x0, ry), w, row_h, facecolor=facecolor, edgecolor="white", lw=1, zorder=2))
        ax.text(x0 + 0.008, ry + row_h / 2, f"{i + 1}. {marker}{team}", ha="left", va="center",
                fontsize=8.7, fontweight=weight, color=textcolor, zorder=3)


cell_w, cell_h = 0.305, 0.165
gutter_x, gutter_y = 0.025, 0.028
x0_positions = [0.0, cell_w + gutter_x, 2 * (cell_w + gutter_x)]
y_top = 0.88
row_ys = [y_top - r * (cell_h + gutter_y) for r in range(4)]

for idx, group_letter in enumerate(group_letters):
    col, row = idx % 3, idx // 3
    draw_group_cell(ax2, x0_positions[col], row_ys[row], cell_w, cell_h,
                     group_letter, group_rankings[group_letter], qualified_thirds)

thirds_y = row_ys[-1] - cell_h - 0.07
ax2.text(0.0, thirds_y + 0.045, "Meilleurs 3emes qualifies (8 sur 12), classes :",
          fontsize=10.5, fontweight="bold", color=NAVY)

third_cell_w = (1.0 - 3 * gutter_x) / 4
for pos, (g, team, stats) in enumerate(third_placed_sorted[:8], start=1):
    col, row = (pos - 1) % 4, (pos - 1) // 4
    x0 = col * (third_cell_w + gutter_x)
    y0 = thirds_y - row * 0.045
    ax2.add_patch(Rectangle((x0, y0 - 0.035), third_cell_w, 0.035, facecolor=GREEN_BG, edgecolor="white", lw=1))
    ax2.text(x0 + 0.008, y0 - 0.0175, f"#{pos}  {g} - {team}", ha="left", va="center",
              fontsize=8.3, fontweight="bold", color="black")

ax2.text(0.0, thirds_y - 0.10, "Vert = qualifie pour le Round of 32 (1er, 2eme, ou meilleur 3e)   |   "
          "* = qualifie via le classement des meilleurs 3emes", fontsize=8, color="#777777", style="italic")

# ---------------------------------------------------------------------------
# 7. Page 3 : arbre de predictions (Round of 32 -> Champion), un seul
#    chemin coherent (vainqueur le plus probable enchaine a chaque match).
# ---------------------------------------------------------------------------

rounds = [
    ("Round of 32", r32_records),
    ("Round of 16", r16_records),
    ("Quarts de finale", qf_records),
    ("Demi-finales", sf_records),
    ("Finale", [dict(team1=team1_f, team2=team2_f, winner=champion, prob=champ_prob, is_real=False)]),
]

n_leaves = len(r32_records) * 2  # 32
row_spacing = 1.5
leaf_y = {i: -i * row_spacing for i in range(n_leaves)}
level_y = [leaf_y[i] for i in range(n_leaves)]

box_w, box_h = 1.55, 0.62
gap = 0.55
x_step = box_w + gap
fig_w, fig_h = 23, 25

fig, ax = plt.subplots(figsize=(fig_w, fig_h))
ax.set_axis_off()

col_x = 0


def draw_team_box(ax, x, y, w, h, name, is_winner, is_champion=False):
    if is_champion:
        facecolor, edgecolor, textcolor = "#fdf3d6", GOLD, NAVY
    elif is_winner:
        facecolor, edgecolor, textcolor = GREEN_BG, "#8fc99a", "black"
    else:
        facecolor, edgecolor, textcolor = GREY_BG, "#dddddd", GREY_TEXT
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                  facecolor=facecolor, edgecolor=edgecolor, lw=1.1, zorder=3))
    ax.text(x + 0.08, y, name, va="center", ha="left", fontsize=9.3,
            fontweight="bold" if (is_winner or is_champion) else "normal", color=textcolor, zorder=4)


for round_name, records in rounds:
    ax.text(col_x + box_w / 2, row_spacing * 0.6, round_name, ha="center", va="bottom",
              fontsize=11, fontweight="bold", color=NAVY)
    next_y = []
    for i, rec in enumerate(records):
        y1, y2 = level_y[2 * i], level_y[2 * i + 1]
        y_mid = (y1 + y2) / 2
        x_in, x_out = col_x, col_x + x_step
        x_mid = x_in + box_w + gap / 2

        draw_team_box(ax, x_in, y1, box_w, box_h, rec["team1"], rec["team1"] == rec["winner"])
        draw_team_box(ax, x_in, y2, box_w, box_h, rec["team2"], rec["team2"] == rec["winner"])

        ax.plot([x_in + box_w, x_mid], [y1, y1], color=NAVY_LIGHT, lw=1.3, zorder=2)
        ax.plot([x_in + box_w, x_mid], [y2, y2], color=NAVY_LIGHT, lw=1.3, zorder=2)
        ax.plot([x_mid, x_mid], [y1, y2], color=NAVY_LIGHT, lw=1.3, zorder=2)
        ax.plot([x_mid, x_out], [y_mid, y_mid], color=NAVY_LIGHT, lw=1.3, zorder=2)

        if rec.get("is_real"):
            ax.text(x_mid, y_mid + 0.20, "REEL", va="center", ha="center", fontsize=7,
                      fontweight="bold", color="white",
                      bbox=dict(boxstyle="round,pad=0.18", facecolor=GOLD, edgecolor="none"), zorder=5)
        else:
            ax.text(x_mid, y_mid + 0.20, f"{rec['prob']:.0%}", va="center", ha="center",
                      fontsize=7.5, color=NAVY, style="italic", zorder=5)

        next_y.append(y_mid)
    level_y = next_y
    col_x += x_step

draw_team_box(ax, col_x + 0.1, level_y[0], box_w + 0.5, box_h * 1.3, f"CHAMPION : {champion}",
              is_winner=False, is_champion=True)

ax.set_xlim(-0.1, col_x + box_w + 1.0)
ax.set_ylim(min(leaf_y.values()) - row_spacing, row_spacing * 1.3)
ax.set_title(
    "Coupe du Monde 2026 — Arbre de predictions (Round of 32 -> Champion)\n"
    "Modele XGBoost (Poisson, elo_diff) conditionne sur les vrais resultats connus au 30/06/2026. "
    "Vert = vainqueur predit a chaque match. Badge \"REEL\" = match deja joue. % = probabilite estimee (tirs au but inclus).",
    fontsize=11.5, fontweight="bold", color=NAVY, loc="left", pad=16
)

# ---------------------------------------------------------------------------
# 8. Export PDF (probabilites, puis groupes, puis arbre detaille)
# ---------------------------------------------------------------------------

out_path = "../data/processed/wc2026_bracket_prediction.pdf"
with PdfPages(out_path) as pdf:
    pdf.savefig(fig3, bbox_inches="tight")
    pdf.savefig(fig2, bbox_inches="tight")
    pdf.savefig(fig, bbox_inches="tight")

print(f"\nPDF sauvegarde : {out_path}")
