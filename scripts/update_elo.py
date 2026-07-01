"""
Met a jour elo_ratings_wc2026.csv avec un nouveau snapshot live depuis
eloratings.net. A relancer apres chaque journee de matchs pendant le
tournoi pour que le modele utilise des ratings a jour.

Lancer depuis le dossier scripts/ : python update_elo.py
"""

import subprocess
import pandas as pd
from datetime import date

SNAPSHOT_DATE = str(date.today())

# eloratings.net/World.tsv name  ->  our name in wc_2026_teams.csv
ELO_WEB_TO_OURS = {
    "United States": "USA",
    "Turkey":        "Türkiye",
    "Czech Republic":"Czechia",
    "Bosnia-Herzegovina": "Bosnia and Herzegovina",
    "Cape Verde Islands": "Cape Verde",
    "Korea Republic":     "South Korea",
}

# our name  ->  name stored in elo_ratings_wc2026.csv (raw elo names)
# Only entries that differ; everything else maps to itself.
OURS_TO_CSV = {
    "USA":     "United States",
    "Türkiye": "Turkey",
}


def fetch_tsv(url):
    return subprocess.check_output(["curl", "-s", url], text=True, encoding="utf-8")


def parse_current_ratings():
    teams_tsv = fetch_tsv("https://www.eloratings.net/en.teams.tsv")
    world_tsv  = fetch_tsv("https://www.eloratings.net/World.tsv")

    code_to_name = {}
    for line in teams_tsv.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 2:
            code_to_name[parts[0]] = parts[1]

    ratings = {}
    for line in world_tsv.strip().splitlines():
        parts = line.split("\t")
        if len(parts) >= 4:
            code = parts[2]
            name = code_to_name.get(code, code)
            try:
                ratings[name] = int(parts[3])
            except ValueError:
                pass
    return ratings


def main():
    elo_path   = "../data/raw/elo_ratings_wc2026.csv"
    teams_path = "../data/raw/wc_2026_teams.csv"

    elo  = pd.read_csv(elo_path)
    our_team_names = list(pd.read_csv(teams_path)["team"])

    current = parse_current_ratings()

    # Use the most recent YEAR-END snapshot as a stable template (guaranteed
    # to have all 48 teams with the correct raw CSV column names / structure).
    year_end_dates = sorted([d for d in elo["snapshot_date"].unique() if d.endswith("-12-31")])
    template_date  = year_end_dates[-1]  # e.g. "2025-12-31"
    template = elo[elo["snapshot_date"] == template_date].set_index("country")

    # Also grab the pre-tournament snapshot for delta reporting
    pre_tournament = elo[elo["snapshot_date"] == "2026-05-27"].set_index("country")
    ref_ratings = pre_tournament["rating"] if len(pre_tournament) else template["rating"]

    # Build web-name lookup (our_name -> eloratings.net/World.tsv name)
    ours_to_web = {v: k for k, v in ELO_WEB_TO_OURS.items()}

    new_rows = []
    missing  = []

    for our_name in our_team_names:
        csv_name = OURS_TO_CSV.get(our_name, our_name)   # key in the CSV file
        web_name = ours_to_web.get(our_name, our_name)   # key on eloratings.net

        # 1. Get new rating from eloratings.net
        new_rating = current.get(web_name) or current.get(our_name)

        # 2. Fall back to pre-tournament snapshot if not found online (e.g. Curaçao)
        if new_rating is None:
            if csv_name in template.index:
                new_rating = int(template.loc[csv_name, "rating"])
                missing.append(our_name)
            else:
                print(f"  [WARNING] {our_name}: not found anywhere, skipping")
                continue

        # 3. Copy the template row and overwrite rating + date
        if csv_name not in template.index:
            print(f"  [WARNING] {our_name} ({csv_name}): not in template snapshot, skipping")
            continue

        row = template.loc[csv_name].copy()
        row["rating"]        = new_rating
        row["snapshot_date"] = SNAPSHOT_DATE
        row.name = csv_name
        new_rows.append(row)

    if missing:
        print(f"  [fallback — not on eloratings.net] {', '.join(missing)}")

    new_df = pd.DataFrame(new_rows)
    new_df.index.name = "country"
    new_df = new_df.reset_index()

    # Remove any existing rows for today, then append
    elo = elo[elo["snapshot_date"] != SNAPSHOT_DATE]
    elo = pd.concat([elo, new_df], ignore_index=True)
    elo.to_csv(elo_path, index=False)

    print(f"\nSnapshot {SNAPSHOT_DATE} : {len(new_rows)} equipes mises a jour.")
    print(f"(template : {template_date}, reference delta : 2026-05-27)\n")
    print("Top changements vs pre-tournoi (2026-05-27) :")
    changes = []
    for our_name in our_team_names:
        csv_name = OURS_TO_CSV.get(our_name, our_name)
        row = next((r for r in new_rows if r.name == csv_name), None)
        if row is not None and csv_name in ref_ratings.index:
            old_r = int(ref_ratings[csv_name])
            new_r = int(row["rating"])
            changes.append((our_name, old_r, new_r, new_r - old_r))
    for team, old_r, new_r, delta in sorted(changes, key=lambda x: abs(x[3]), reverse=True)[:12]:
        print(f"  {team:30s} {old_r} -> {new_r}  ({delta:+d})")


if __name__ == "__main__":
    main()
