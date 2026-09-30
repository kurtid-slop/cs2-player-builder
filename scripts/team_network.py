import pandas as pd
import numpy as np
import networkx as nx
from pyvis.network import Network
import re
import os

from common import CLUSTERS_FILE, MATCHES_FILE, OUTPUT_DIR, RANKINGS_FILE


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_FILE = MATCHES_FILE
OUTPUT_FILE = OUTPUT_DIR / "team_network.html"
STATS_FILE = OUTPUT_DIR / "team_network_stats.csv"

# Minimum number of times two teams must have played
# for the connection to be displayed.
#
# 1 = every relationship
# 3 = useful starting point
# 5 = stronger relationships
MIN_MATCHES_FOR_EDGE = 5


# ============================================================
# ELO SETTINGS
# ============================================================

INITIAL_RATING = 1500

K_FACTOR = 32

# We only need one chronological pass through the matches.
ELO_ITERATIONS = 1

# Recency decay.
#
# Higher = older matches lose importance faster.
DECAY = 0.01

# How much recency affects the Elo calculation.
#
# 0.0 = no recency
# 1.0 = maximum recency effect
RECENCY_INFLUENCE = 0.30


# ============================================================
# TIER SETTINGS
# ============================================================

TIER_GAP = 50

MIN_TEAMS_PER_TIER = 5


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.isfile(INPUT_FILE):

    print(
        f"ERROR: '{INPUT_FILE}' was not found."
    )

    print(
        "Put matches.csv in the data folder."
    )

    raise SystemExit


# ============================================================
# LOAD DATA
# ============================================================

print("Loading matches.csv...")

data = pd.read_csv(
    INPUT_FILE
)


print()
print("CSV columns:")
print(data.columns.tolist())


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "teamId",
    "team",
    "matchId",
    "date",
    "teamA",
    "teamB",
    "score",
    "matchUrl"
]


missing = [
    column
    for column in required_columns
    if column not in data.columns
]


if missing:

    print()
    print(
        "ERROR: Missing columns:"
    )

    print(
        missing
    )

    raise SystemExit


# ============================================================
# CLEAN TEAM NAMES
# ============================================================

data["teamA"] = (
    data["teamA"]
    .astype(str)
    .str.strip()
)

data["teamB"] = (
    data["teamB"]
    .astype(str)
    .str.strip()
)


# ============================================================
# CLEAN MATCH IDS
# ============================================================

data["matchId"] = (
    data["matchId"]
    .astype(str)
    .str.strip()
)


# ============================================================
# EXTRACT DATE
# ============================================================

# Your CSV contains:
#
# Results for August 22nd 2026
#
# We extract:
#
# August 22nd 2026

data["date_clean"] = (
    data["date"]
    .astype(str)
    .str.extract(
        r"Results for (.+)"
    )[0]
)


# ============================================================
# REMOVE ORDINAL SUFFIX
# ============================================================

data["date_clean"] = (
    data["date_clean"]
    .str.replace(
        r"(\d+)(st|nd|rd|th)",
        r"\1",
        regex=True
    )
)


# ============================================================
# CONVERT TO PANDAS DATE
# ============================================================

data["date"] = pd.to_datetime(
    data["date_clean"],
    format="%B %d %Y",
    errors="coerce"
)


# ============================================================
# CHECK INVALID DATES
# ============================================================

invalid_dates = data["date"].isna().sum()


if invalid_dates > 0:

    print(
        f"WARNING: {invalid_dates} rows "
        "have invalid dates."
    )


data = data.dropna(
    subset=["date"]
).copy()


# ============================================================
# REMOVE INVALID MATCHES
# ============================================================

data = data[
    (data["teamA"] != "") &
    (data["teamB"] != "") &
    (data["teamA"] != data["teamB"])
].copy()


# ============================================================
# DEDUPLICATE MATCHES
# ============================================================

print()
print(
    f"Rows before deduplication: {len(data)}"
)


# Because your scraper collects matches from
# multiple teams, the same match can appear
# multiple times.
#
# matchId uniquely identifies the match.

data = data.drop_duplicates(
    subset=["matchId"]
).copy()


print(
    f"Unique matches: {len(data)}"
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

data = data.sort_values(
    "date"
).reset_index(
    drop=True
)


# ============================================================
# GET UNIQUE TEAMS
# ============================================================

teams = sorted(
    set(data["teamA"])
    |
    set(data["teamB"])
)


print()
print("==========================================")
print("DATASET")
print("==========================================")

print(
    f"Matches: {len(data)}"
)

print(
    f"Teams: {len(teams)}"
)

print(
    f"First match: {data['date'].min().date()}"
)

print(
    f"Last match: {data['date'].max().date()}"
)

print()


# ============================================================
# SCORE PARSER
# ============================================================

def parse_score(score):

    numbers = re.findall(
        r"\d+",
        str(score)
    )

    if len(numbers) < 2:

        return None

    return (
        int(numbers[0]),
        int(numbers[1])
    )


# ============================================================
# GET WINNER
# ============================================================

def get_winner(
    team_a,
    team_b,
    score
):

    parsed = parse_score(
        score
    )

    if parsed is None:

        return None

    score_a, score_b = parsed


    if score_a > score_b:

        return team_a


    if score_b > score_a:

        return team_b


    return None


# ============================================================
# INITIALIZE ELO
# ============================================================

ratings = {
    team: INITIAL_RATING
    for team in teams
}


# ============================================================
# EXPECTED SCORE
# ============================================================

def expected_score(
    rating_a,
    rating_b
):

    return (
        1 /
        (
            1 +
            10 ** (
                (rating_b - rating_a)
                / 400
            )
        )
    )


# ============================================================
# RECENCY WEIGHT
# ============================================================

latest_date = data["date"].max()


def recency_weight(
    match_date
):

    age_days = (
        latest_date -
        match_date
    ).days


    return np.exp(
        -DECAY * age_days
    )


# ============================================================
# CALCULATE ELO
# ============================================================

print(
    "Calculating team strength..."
)


for _, match in data.iterrows():

    team_a = match["teamA"]

    team_b = match["teamB"]

    winner = get_winner(
        team_a,
        team_b,
        match["score"]
    )


    # Skip matches where the score
    # could not be interpreted.

    if winner is None:

        continue


    rating_a = ratings[
        team_a
    ]

    rating_b = ratings[
        team_b
    ]


    expected_a = expected_score(
        rating_a,
        rating_b
    )

    expected_b = (
        1 -
        expected_a
    )


    if winner == team_a:

        actual_a = 1
        actual_b = 0

    else:

        actual_a = 0
        actual_b = 1


    # --------------------------------------------------------
    # RECENCY
    # --------------------------------------------------------

    weight = recency_weight(
        match["date"]
    )


    effective_weight = (
        (1 - RECENCY_INFLUENCE)
        +
        (
            RECENCY_INFLUENCE
            *
            weight
        )
    )


    effective_k = (
        K_FACTOR
        *
        effective_weight
    )


    # --------------------------------------------------------
    # UPDATE RATINGS
    # --------------------------------------------------------

    ratings[team_a] = (
        rating_a
        +
        effective_k
        *
        (
            actual_a
            -
            expected_a
        )
    )


    ratings[team_b] = (
        rating_b
        +
        effective_k
        *
        (
            actual_b
            -
            expected_b
        )
    )


# ============================================================
# BUILD NETWORK
# ============================================================

print(
    "Building network..."
)


G = nx.Graph()


# ============================================================
# COUNT HEAD-TO-HEAD MATCHES
# ============================================================

match_counts = {}

for _, match in data.iterrows():

    team_a = match["teamA"]
    team_b = match["teamB"]

    pair = tuple(
        sorted(
            [
                team_a,
                team_b
            ]
        )
    )

    if pair not in match_counts:
        match_counts[pair] = 0

    match_counts[pair] += 1


# ============================================================
# FIND TEAMS WITH QUALIFYING EDGES
# ============================================================

connected_teams = set()

for (
    team_a,
    team_b
), count in match_counts.items():

    if count >= MIN_MATCHES_FOR_EDGE:

        connected_teams.add(team_a)
        connected_teams.add(team_b)


# ============================================================
# ADD ONLY CONNECTED TEAMS
# ============================================================

# Sorted so the graph is built in the same order every run.
# Set order changes between runs, which would make the
# cluster detection give different results each time.
for team in sorted(connected_teams):

    G.add_node(
        team,
        strength=ratings[team]
    )


print(
    f"Teams with qualifying edges: "
    f"{len(connected_teams)}"
)


print(
    f"Teams removed from graph: "
    f"{len(teams) - len(connected_teams)}"
)


# ============================================================
# ADD EDGES
# ============================================================

for (
    team_a,
    team_b
), count in match_counts.items():

    if count >= MIN_MATCHES_FOR_EDGE:

        G.add_edge(
            team_a,
            team_b,
            weight=count
        )

print()
print(
    f"Network nodes: {G.number_of_nodes()}"
)

print(
    f"Network edges: {G.number_of_edges()}"
)


# ============================================================
# NETWORK STATISTICS
# ============================================================

degree = dict(
    G.degree()
)


weighted_degree = dict(
    G.degree(
        weight="weight"
    )
)


# ============================================================
# CONNECTED COMPONENTS
# ============================================================

components = sorted(
    nx.connected_components(G),
    key=len,
    reverse=True
)


component_lookup = {}


for component_number, component in enumerate(
    components,
    start=1
):

    for team in component:

        component_lookup[
            team
        ] = component_number


print()
print(
    "Network components:"
)


for i, component in enumerate(
    components[:20],
    start=1
):

    print(
        f"  Component {i}: "
        f"{len(component)} teams"
    )


# ============================================================
# CENTRALITY
# ============================================================

if G.number_of_edges() == 0:

    print()
    print(
        "WARNING: No network edges."
    )

    betweenness = {
        team: 0
        for team in teams
    }

    eigenvector = {
        team: 0
        for team in teams
    }

    pagerank = {
        team: 0
        for team in teams
    }

else:

    print(
        "Calculating centrality..."
    )


    # --------------------------------------------------------
    # Betweenness
    # --------------------------------------------------------

    betweenness = (
        nx.betweenness_centrality(
            G,
            weight="weight"
        )
    )


    # --------------------------------------------------------
    # Eigenvector
    # --------------------------------------------------------

    try:

        eigenvector = (
            nx.eigenvector_centrality(
                G,
                max_iter=1000,
                weight="weight"
            )
        )

    except nx.PowerIterationFailedConvergence:

        print(
            "WARNING: Eigenvector "
            "centrality did not converge."
        )

        eigenvector = {
            team: 0
            for team in teams
        }


    # --------------------------------------------------------
    # PageRank
    # --------------------------------------------------------

    pagerank = nx.pagerank(
        G,
        weight="weight"
    )


# ============================================================
# BUILD TEAM STATISTICS
# ============================================================

team_stats = []


for team in teams:

    team_matches = data[
        (data["teamA"] == team)
        |
        (data["teamB"] == team)
    ]


    wins = 0

    losses = 0


    for _, match in team_matches.iterrows():

        winner = get_winner(
            match["teamA"],
            match["teamB"],
            match["score"]
        )


        if winner == team:

            wins += 1

        elif winner is not None:

            losses += 1


    matches = (
        wins +
        losses
    )


    if matches > 0:

        win_rate = (
            wins /
            matches
        )

    else:

        win_rate = 0


    team_stats.append({

        "team": team,

        "strength":
            ratings[team],

        "matches":
            matches,

        "wins":
            wins,

        "losses":
            losses,

        "win_rate":
            win_rate,

        "unique_opponents":
            degree.get(
                team,
                0
            ),

        "network_matches":
            weighted_degree.get(
                team,
                0
            ),

        "network_component":
            component_lookup.get(
                team,
                None
            ),

        "betweenness":
            betweenness.get(
                team,
                0
            ),

        "eigenvector":
            eigenvector.get(
                team,
                0
            ),

        "pagerank":
            pagerank.get(
                team,
                0
            )
    })


stats = pd.DataFrame(
    team_stats
)


# ============================================================
# RANK TEAMS
# ============================================================

stats = stats.sort_values(
    "strength",
    ascending=False
).reset_index(
    drop=True
)


stats["rank"] = (
    np.arange(
        1,
        len(stats) + 1
    )
)


# ============================================================
# STRENGTH GAP
# ============================================================

stats["strength_gap"] = (
    stats["strength"]
    -
    stats["strength"].shift(-1)
)


stats["strength_gap"] = (
    stats["strength_gap"]
    .fillna(0)
)


# ============================================================
# CALCULATE TIERS
# ============================================================

tier_boundaries = []

current_tier_size = 0


for i in range(
    len(stats) - 1
):

    current_tier_size += 1


    gap = stats.loc[
        i,
        "strength_gap"
    ]


    if (
        gap >= TIER_GAP
        and
        current_tier_size
        >= MIN_TEAMS_PER_TIER
    ):

        tier_boundaries.append(
            i
        )

        current_tier_size = 0


boundary_set = set(
    tier_boundaries
)


current_tier = 1

tiers = []


for i in range(
    len(stats)
):

    tiers.append(
        current_tier
    )


    if i in boundary_set:

        current_tier += 1


stats["tier"] = tiers

# ============================================================
# COMMUNITY / CLUSTER DETECTION
# ============================================================

print()
print("Detecting competitive clusters...")


communities = nx.community.louvain_communities(
    G,
    weight="weight",
    resolution=2,
    seed=42
)


# ============================================================
# ORDER CLUSTERS BY HLTV POINTS
# ============================================================
#
# Clusters are ordered by the average HLTV points of their
# teams, so cluster order lines up with the HLTV ranking.
#
# Unranked teams count as 0 points. This stops a cluster
# of mostly unranked teams from ranking highly because of
# one ranked member. Average Elo strength breaks ties.

hltv = pd.read_csv(RANKINGS_FILE)

hltv_points = dict(
    zip(hltv["team"], hltv["hltvPoints"])
)

stats["hltv_points"] = (
    stats["team"].map(hltv_points).fillna(0)
)

cluster_rankings = []

for community in communities:

    cluster_data = stats[
        stats["team"].isin(community)
    ]

    average_strength = cluster_data["strength"].mean()
    average_rank = cluster_data["rank"].mean()
    median_rank = cluster_data["rank"].median()
    average_hltv_points = cluster_data["hltv_points"].mean()

    cluster_rankings.append({
        "community": community,
        "average_strength": average_strength,
        "average_rank": average_rank,
        "median_rank": median_rank,
        "average_hltv_points": average_hltv_points
    })


cluster_rankings.sort(
    key=lambda x: (
        x["average_hltv_points"],
        x["average_strength"]
    ),
    reverse=True
)


communities = [
    x["community"]
    for x in cluster_rankings
]


print()
print("==========================================")
print("COMPETITIVE CLUSTERS")
print("==========================================")


for i, community in enumerate(
    communities,
    start=1
):

    print()
    print(
        f"Cluster {i} "
        f"({len(community)} teams)"
    )

    print("-" * 50)

    for team in sorted(community):

        print(team)

# ============================================================
# SAVE TEAM CLUSTERS TO CSV
# ============================================================

cluster_lookup = {}

for cluster_id, community in enumerate(communities, start=1):

    for team in community:
        cluster_lookup[team] = cluster_id


# Add cluster assignment to team stats
stats["cluster"] = stats["team"].map(cluster_lookup)


# Calculate cluster-level statistics
cluster_info = (
    stats
    .groupby("cluster")
    .agg(
        cluster_size=("team", "count"),
        average_rank=("rank", "mean"),
        median_rank=("rank", "median"),
        average_strength=("strength", "mean"),
        median_strength=("strength", "median"),
        average_hltv_points=("hltv_points", "mean")
    )
    .reset_index()
)


# Add cluster information to every team
clusters_csv = stats[
    [
        "team",
        "rank",
        "strength",
        "hltv_points",
        "cluster"
    ]
].merge(
    cluster_info,
    on="cluster",
    how="left"
)


# Sort by cluster, then strength
clusters_csv = clusters_csv.sort_values(
    by=["cluster", "strength"],
    ascending=[True, False]
).reset_index(drop=True)


# Put cluster as the second column
columns = [
    "team",
    "cluster",
    "rank",
    "strength",
    "hltv_points",
    "cluster_size",
    "average_rank",
    "median_rank",
    "average_strength",
    "median_strength",
    "average_hltv_points"
]

clusters_csv = clusters_csv[columns]


# Save CSV
clusters_csv.to_csv(
    CLUSTERS_FILE,
    index=False
)

print(f"Saved cluster assignments to {CLUSTERS_FILE.name}")


# ============================================================
# TEAM LOOKUP
# ============================================================

stats_lookup = (
    stats
    .set_index("team")
    .to_dict("index")
)


# ============================================================
# NODE SIZE
# ============================================================

min_strength = (
    stats["strength"].min()
)

max_strength = (
    stats["strength"].max()
)


def node_size(strength):

    if max_strength == min_strength:
        return 30

    normalized = (
        strength - min_strength
    ) / (
        max_strength - min_strength
    )

    return (
        25 +
        normalized * 75
    )

# ============================================================
# TIER COLORS
# ============================================================

tier_colors = {

    1: "#ff4d4d",

    2: "#ff9f43",

    3: "#feca57",

    4: "#48dbfb",

    5: "#1dd1a1",

    6: "#a55eea",

    7: "#c8d6e5"
}


# ============================================================
# CREATE PYVIS GRAPH
# ============================================================

print()
print(
    "Creating interactive graph..."
)


net = Network(
    height="900px",
    width="100%",
    bgcolor="#111111",
    font_color="white",
    directed=False
)


# ============================================================
# GRAPH PHYSICS
# ============================================================

net.set_options(
    """
    {
        "nodes": {
            "shape": "dot",

            "font": {
                "size": 14,
                "color": "white"
            },

            "borderWidth": 1
        },

        "edges": {

            "smooth": {
                "type": "continuous"
            }
        },

        "physics": {

            "enabled": true,

            "forceAtlas2Based": {

                "gravitationalConstant": -150,

                "centralGravity": 0.005,

                "springLength": 300,

                "springConstant": 0.03,

                "damping": 0.5,

                "avoidOverlap": 2
            },

            "solver": "forceAtlas2Based",

            "stabilization": {

                "enabled": true,

                "iterations": 1000
            }
        },

        "interaction": {

            "hover": true,

            "navigationButtons": true,

            "keyboard": true,

            "zoomView": true
        }
    }
    """
)


# ============================================================
# ADD NODES
# ============================================================

for team in G.nodes:

    info = stats_lookup[
        team
    ]


    strength = info[
        "strength"
    ]


    tier = int(
        info["tier"]
    )


    color = tier_colors.get(
        tier,
        "#aaaaaa"
    )


    size = node_size(
        strength
    )


    tooltip = (
    f"{team}\n"
    f"Rank: {info['rank']}\n"
    f"Tier: {tier}\n"
    f"Strength: {strength:.2f}\n"
    f"Matches: {info['matches']}\n"
    f"Wins: {info['wins']}\n"
    f"Losses: {info['losses']}\n"
    f"Win rate: {info['win_rate'] * 100:.1f}%\n"
    f"Unique opponents: {info['unique_opponents']}\n"
    f"Network matches: {info['network_matches']}\n"
    f"Network component: {info['network_component']}\n"
    f"Betweenness: {info['betweenness']:.4f}\n"
    f"Eigenvector: {info['eigenvector']:.4f}\n"
    f"PageRank: {info['pagerank']:.4f}"
)


    net.add_node(

        team,

        label=team,

        title=tooltip,

        size=size,

        color=color
    )


# ============================================================
# ADD EDGES
# ============================================================

for team_a, team_b, edge_data in G.edges(
    data=True
):

    count = edge_data[
        "weight"
    ]


    # Logarithmic scaling keeps very
    # frequent matchups from dominating.

    width = (
        1 +
        np.log1p(count) * 1.5
    )


    tooltip = (
        f"<b>{team_a}</b>"
        f" vs "
        f"<b>{team_b}</b>"
        f"<br>"
        f"Matches: {count}"
    )


    net.add_edge(

        team_a,

        team_b,

        width=width,

        title=tooltip
    )


# ============================================================
# WRITE HTML
# ============================================================

# pyvis copies its lib/ folder into the current directory,
# so write from the outputs folder to keep lib/ next to the
# HTML file.
working_dir = os.getcwd()
os.chdir(OUTPUT_DIR)

net.write_html(
    OUTPUT_FILE.name,
    open_browser=False
)

os.chdir(working_dir)


# ============================================================
# SAVE STATISTICS
# ============================================================

stats.to_csv(
    STATS_FILE,
    index=False
)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("==========================================")
print("NETWORK COMPLETE")
print("==========================================")
print()

print(
    f"Unique matches: {len(data)}"
)

print(
    f"Teams: {G.number_of_nodes()}"
)

print(
    f"Connections: {G.number_of_edges()}"
)

print(
    f"Minimum matches per connection: "
    f"{MIN_MATCHES_FOR_EDGE}"
)

print()

print(
    f"Graph saved to:"
)

print(
    f"  {OUTPUT_FILE}"
)

print()

print(
    f"Statistics saved to:"
)

print(
    f"  {STATS_FILE}"
)

print()

print(
    "Top 20 teams by strength:"
)

print()

print(
    stats[
        [
            "rank",
            "team",
            "strength",
            "tier",
            "matches",
            "unique_opponents",
            "network_matches",
            "network_component",
            "eigenvector",
            "pagerank"
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)

print()