import json
import numpy as np
import pandas as pd
from datetime import date

from common import CLUSTERS_FILE, ROOT_DIR


# ============================================================
# CONFIGURATION
# ============================================================
#
# Builds docs/players.js for the player builder game in
# docs/index.html. GitHub Pages serves the docs/ folder.
#
# The data is written as a JS file (not JSON) so the page
# also works when index.html is opened straight from disk.

PLAYERS_INPUT = ROOT_DIR / "player_adjusted.csv"
SITE_OUTPUT = ROOT_DIR / "docs" / "players.js"

ATTRIBUTES = [
    "Firepower",
    "Entrying",
    "Trading",
    "Opening",
    "Clutching",
    "Sniping",
    "Utility"
]

# Teams need this many active players to be rolled,
# so there is a real choice of player
MIN_PLAYERS_PER_TEAM = 3

# How many teams can be rolled. Teams are taken strongest
# cluster first, then by Elo strength within a cluster, so
# the last cluster included may only be partly used. Teams
# without a cluster come after all clustered teams.
MAX_TEAMS = 50

# Adjusted attributes can go past 100, so they are capped
ATTRIBUTE_CEILING = 100

# Sniping splits into two groups: AWPers (~85-100) and
# riflers (~0). Players at or above this raw Sniping value
# count as AWPers.
AWPER_THRESHOLD = 40

# How many AWPers get an S (95+) in Sniping. The players
# listed are always included (a manual choice, not from the
# data); the rest of the S tier is the top-ranked AWPers.
SNIPING_S_TIER_SIZE = 5
SNIPING_S_TIER_INCLUDE = ["s1mple", "device"]

# The lowest Sniping an AWPer can get. The rest of the AWPers
# are spread by rank between this and 94. Riflers stay at 0.
SNIPING_AWPER_FLOOR = 50


# ============================================================
# LOAD PLAYERS
# ============================================================

players = pd.read_csv(PLAYERS_INPUT)

clusters = pd.read_csv(CLUSTERS_FILE)
cluster_lookup = dict(zip(clusters["team"], clusters["cluster"]))

players["cluster"] = players["team"].map(cluster_lookup)

players = players[
    (players["status"] == "active") &
    players["team"].notna()
].dropna(subset=[f"{a} adjusted" for a in ATTRIBUTES])

team_order = (
    players
    .groupby("team")
    .agg(
        active_players=("playername", "size"),
        cluster=("cluster", "first"),
        strength=("team_strength", "first")
    )
    .query("active_players >= @MIN_PLAYERS_PER_TEAM")
    .sort_values(
        ["cluster", "strength"],
        ascending=[True, False],
        na_position="last"
    )
)

selected_teams = team_order.head(MAX_TEAMS)

if len(selected_teams) < MAX_TEAMS:
    print(f"WARNING: only {len(selected_teams)} teams have "
          f"{MIN_PLAYERS_PER_TEAM}+ active players")

players = players[players["team"].isin(selected_teams.index)]


# ============================================================
# NORMALIZE SNIPING
# ============================================================
#
# Raw Sniping mostly says whether a player is an AWPer, not
# how good they are. AWPers are rescaled by their rank among
# the game's AWPers instead:
#
#   S tier    the included players plus the top-ranked AWPers,
#             spread from 100 down to 95
#   the rest  spread by rank from 94 down to SNIPING_AWPER_FLOOR
#
# Riflers get 0.

players["Sniping adjusted"] = players["Sniping adjusted"].astype(float)

awpers = players["Sniping"] >= AWPER_THRESHOLD

included = players["playername"].str.lower().isin(
    [name.lower() for name in SNIPING_S_TIER_INCLUDE]
)

missing = set(n.lower() for n in SNIPING_S_TIER_INCLUDE) - set(
    players.loc[included & awpers, "playername"].str.lower()
)

if missing:
    print(f"WARNING: not AWPers in the game pool: {sorted(missing)}")

# Included players first, then by adjusted Sniping
order = (
    players[awpers]
    .assign(included=included[awpers])
    .sort_values(["included", "Sniping adjusted"], ascending=False)
    .index
)

# Within the S tier, order by the data so the manual picks
# aren't placed above better AWPers
s_tier = (
    players.loc[order[:SNIPING_S_TIER_SIZE], "Sniping adjusted"]
    .sort_values(ascending=False)
    .index
)
rest = order[SNIPING_S_TIER_SIZE:]

players.loc[s_tier, "Sniping adjusted"] = np.linspace(100, 95, len(s_tier))
players.loc[rest, "Sniping adjusted"] = np.linspace(94, SNIPING_AWPER_FLOOR, len(rest))

players.loc[~awpers, "Sniping adjusted"] = 0


# ============================================================
# BUILD TEAMS
# ============================================================

teams = []

for team_name, group in players.groupby("team"):

    cluster = group["cluster"].iloc[0]

    teams.append({
        "name": team_name,
        "cluster": None if pd.isna(cluster) else int(cluster),
        "multiplier": round(float(group["multiplier"].iloc[0]), 3),
        "players": [
            {
                "name": row["playername"],
                "age": None if pd.isna(row["age"]) else int(row["age"]),
                "stats": {
                    a: round(min(float(row[f"{a} adjusted"]), ATTRIBUTE_CEILING), 1)
                    for a in ATTRIBUTES
                },
                "raw": {
                    a: int(row[a])
                    for a in ATTRIBUTES
                }
            }
            for _, row in group.sort_values("playername").iterrows()
        ]
    })


# ============================================================
# SAVE
# ============================================================

data = {
    "generated": date.today().isoformat(),
    "attributes": ATTRIBUTES,
    "teams": teams
}

SITE_OUTPUT.parent.mkdir(exist_ok=True)

SITE_OUTPUT.write_text(
    "// Generated by scripts/build_site.py from player_adjusted.csv\n"
    f"window.GAME_DATA = {json.dumps(data, ensure_ascii=False)};\n",
    encoding="utf-8"
)

print(f"Teams:   {len(teams)}")
print(f"AWPers:  {awpers.sum()}")
print(f"Sniping S tier: {', '.join(players.loc[s_tier, 'playername'])}")
print(f"Players: {sum(len(t['players']) for t in teams)}")
print(f"Saved to {SITE_OUTPUT.relative_to(ROOT_DIR)}")
