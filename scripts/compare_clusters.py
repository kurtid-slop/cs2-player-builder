import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau

from common import OUTPUT_DIR, RANKINGS_FILE, load_clusters, load_matches


# ============================================================
# CONFIGURATION
# ============================================================

TEAMS_OUTPUT = OUTPUT_DIR / "cluster_ranking_comparison.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "cluster_ranking_summary.csv"
WIN_RATE_OUTPUT = OUTPUT_DIR / "cluster_win_rate_matrix.csv"
ROUND_DIFFERENTIAL_OUTPUT = OUTPUT_DIR / "cluster_round_differential_matrix.csv"

# Scores above this are round scores (13 - 7), not map scores (2 - 0)
MAX_MAP_SCORE = 3

# How many misplaced teams to print
TOP_OUTLIERS = 15


# ============================================================
# LOAD DATA
# ============================================================

# Teams that never made it into the network have no cluster
clusters = load_clusters()
rankings = pd.read_csv(RANKINGS_FILE)

teams = clusters[
    ["team", "cluster", "rank", "strength"]
].merge(
    rankings[["team", "ranking", "hltvPoints"]],
    on="team",
    how="left"
)

teams = teams.rename(columns={
    "rank": "elo_rank",
    "ranking": "hltv_rank",
    "hltvPoints": "hltv_points"
})

teams["hltv_points"] = teams["hltv_points"].fillna(0)

ranked = teams.dropna(subset=["hltv_rank"]).copy()
ranked["hltv_rank"] = ranked["hltv_rank"].astype(int)

print()
print(f"Clustered teams:      {len(teams)}")
print(f"HLTV ranked teams:    {len(rankings)}")
print(f"Ranked and clustered: {len(ranked)}")


# ============================================================
# OVERALL AGREEMENT
# ============================================================
#
# If cluster order lines up with the HLTV ranking, teams in
# lower-numbered clusters should have better (lower) ranks.

spearman = spearmanr(ranked["cluster"], ranked["hltv_rank"])[0]
kendall = kendalltau(ranked["cluster"], ranked["hltv_rank"])[0]

print()
print("==========================================")
print("OVERALL AGREEMENT")
print("==========================================")
print(f"Spearman correlation: {spearman:.3f}")
print(f"Kendall tau:          {kendall:.3f}")
print("(1.0 = cluster order matches HLTV order exactly)")


# ============================================================
# EXPECTED CLUSTER
# ============================================================
#
# Take the ranked teams in HLTV order and fill the clusters
# in order, keeping each cluster's number of ranked teams.
#
# This is the cluster each team "should" be in if clusters
# followed the HLTV ranking perfectly.

cluster_sizes = (
    ranked.groupby("cluster").size().sort_index()
)

expected = np.repeat(
    cluster_sizes.index.to_numpy(),
    cluster_sizes.to_numpy()
)

ranked = ranked.sort_values("hltv_rank")
ranked["expected_cluster"] = expected
ranked["cluster_difference"] = (
    ranked["cluster"] - ranked["expected_cluster"]
)

teams = teams.merge(
    ranked[["team", "expected_cluster", "cluster_difference"]],
    on="team",
    how="left"
)


# ============================================================
# CLUSTER SUMMARY
# ============================================================

summary = (
    teams
    .groupby("cluster")
    .agg(
        teams=("team", "count"),
        ranked_teams=("hltv_rank", "count"),
        best_hltv_rank=("hltv_rank", "min"),
        median_hltv_rank=("hltv_rank", "median"),
        worst_hltv_rank=("hltv_rank", "max"),
        average_hltv_points=("hltv_points", "mean"),
        median_elo_rank=("elo_rank", "median"),
        in_expected_cluster=(
            "cluster_difference",
            lambda s: (s == 0).sum()
        )
    )
    .reset_index()
)


# ============================================================
# OVERLAP WITH NEXT CLUSTER
# ============================================================
#
# For each cluster, compare every ranked team with every
# ranked team in the next cluster down. The share of pairs
# where this cluster's team has the better HLTV rank shows
# how cleanly the two clusters are separated.
#
# 100% = no overlap, 50% = no real difference.

cluster_ids = summary["cluster"].tolist()
separation = []

for current, following in zip(cluster_ids, cluster_ids[1:] + [None]):

    a = ranked.loc[ranked["cluster"] == current, "hltv_rank"].to_numpy()

    if following is None:
        separation.append(np.nan)
        continue

    b = ranked.loc[ranked["cluster"] == following, "hltv_rank"].to_numpy()

    if len(a) == 0 or len(b) == 0:
        separation.append(np.nan)
        continue

    separation.append((a[:, None] < b[None, :]).mean())

summary["beats_next_cluster"] = separation

print()
print("==========================================")
print("CLUSTERS VS HLTV RANKING")
print("==========================================")
print()

display = summary.copy()
display["average_hltv_points"] = display["average_hltv_points"].round(1)
display["beats_next_cluster"] = (
    display["beats_next_cluster"]
    .map(lambda x: "" if pd.isna(x) else f"{x:.0%}")
)

print(display.to_string(index=False))


# ============================================================
# TEAMS IN EACH CLUSTER
# ============================================================

print()
print("==========================================")
print("RANKED TEAMS PER CLUSTER")
print("==========================================")

for cluster_id, group in ranked.sort_values("hltv_rank").groupby("cluster"):

    listing = ", ".join(
        f"{row.team} (#{row.hltv_rank})"
        for row in group.itertuples()
    )

    print()
    print(f"Cluster {cluster_id}: {listing}")


# ============================================================
# MISPLACED TEAMS
# ============================================================
#
# Positive difference = team sits in a lower cluster than its
# HLTV rank suggests. Negative = it sits in a higher one.

outliers = ranked.reindex(
    ranked["cluster_difference"].abs().sort_values(ascending=False).index
).head(TOP_OUTLIERS)

print()
print("==========================================")
print("MOST MISPLACED TEAMS")
print("==========================================")
print()
print(
    outliers[
        [
            "team",
            "hltv_rank",
            "elo_rank",
            "cluster",
            "expected_cluster",
            "cluster_difference"
        ]
    ].to_string(index=False)
)


# ============================================================
# CLUSTER VS CLUSTER MATRIX
# ============================================================
#
# An n x n matrix where cell [x][y] holds how cluster x
# performed against cluster y:
#
#   win_rate            share of matches won by x (draws = 0.5)
#   round_differential  average rounds won minus rounds lost
#                       per map for x
#
# Most matches only have a series score in maps (2 - 0), so
# the round differential comes from single-map matches, where
# the score is in rounds (13 - 7). Win rate uses every match.
#
# The diagonal (a cluster against itself) is left empty.

class ClusterMatchup:

    def __init__(self, win_rate, round_differential):

        self.win_rate = win_rate
        self.round_differential = round_differential

    def __repr__(self):

        return (
            f"ClusterMatchup(win_rate={self.win_rate}, "
            f"round_differential={self.round_differential})"
        )


matches = load_matches()

cluster_lookup = dict(zip(teams["team"], teams["cluster"]))

matches["cluster_a"] = matches["teamA"].map(cluster_lookup)
matches["cluster_b"] = matches["teamB"].map(cluster_lookup)

# Only matches between two teams in different clusters
matches = matches.dropna(subset=["cluster_a", "cluster_b"])

matches = matches[
    matches["cluster_a"] != matches["cluster_b"]
].copy()

matches["cluster_a"] = matches["cluster_a"].astype(int)
matches["cluster_b"] = matches["cluster_b"].astype(int)

# A series never goes past 3 maps, so anything higher is rounds
matches["is_round_score"] = (
    matches[["score_a", "score_b"]].max(axis=1) > MAX_MAP_SCORE
)

n = len(cluster_ids)
index = {cluster_id: i for i, cluster_id in enumerate(cluster_ids)}

wins = np.zeros((n, n))
played = np.zeros((n, n))
rounds = np.zeros((n, n))
maps = np.zeros((n, n))

for row in matches.itertuples():

    a = index[row.cluster_a]
    b = index[row.cluster_b]

    result = row.result_a

    # Record the match from both sides
    wins[a, b] += result
    wins[b, a] += 1 - result
    played[a, b] += 1
    played[b, a] += 1

    if row.is_round_score:

        difference = row.score_a - row.score_b

        rounds[a, b] += difference
        rounds[b, a] -= difference
        maps[a, b] += 1
        maps[b, a] += 1


matchup_matrix = [[None] * n for _ in range(n)]

for x in range(n):

    for y in range(n):

        if x == y or played[x, y] == 0:
            continue

        win_rate = wins[x, y] / played[x, y]

        round_differential = (
            rounds[x, y] / maps[x, y]
            if maps[x, y] > 0
            else None
        )

        matchup_matrix[x][y] = ClusterMatchup(
            win_rate=round(win_rate, 3),
            round_differential=(
                None
                if round_differential is None
                else round(round_differential, 2)
            )
        )


# ============================================================
# DISPLAY MATRIX
# ============================================================

def matrix_table(value, fmt):

    return pd.DataFrame(
        [
            [
                "" if cell is None or value(cell) is None
                else fmt(value(cell))
                for cell in row
            ]
            for row in matchup_matrix
        ],
        index=cluster_ids,
        columns=cluster_ids
    )


win_rate_table = matrix_table(
    lambda cell: cell.win_rate,
    lambda v: f"{v:.0%}"
)

round_differential_table = matrix_table(
    lambda cell: cell.round_differential,
    lambda v: f"{v:+.1f}"
)

matches_table = pd.DataFrame(
    played.astype(int),
    index=cluster_ids,
    columns=cluster_ids
)

print()
print("==========================================")
print("WIN RATE (row cluster vs column cluster)")
print("==========================================")
print()
print(win_rate_table.to_string())

print()
print("==========================================")
print("ROUND DIFFERENTIAL PER MAP (row vs column)")
print("==========================================")
print()
print(round_differential_table.to_string())

print()
print("==========================================")
print("MATCHES PLAYED (row vs column)")
print("==========================================")
print()
print(matches_table.to_string())

# Each cell shows one ClusterMatchup as "win rate / round differential"
def format_matchup(cell):

    if cell is None:
        return "-"

    differential = (
        "n/a"
        if cell.round_differential is None
        else f"{cell.round_differential:+.1f}"
    )

    return f"{cell.win_rate:.0%} / {differential}"


matrix_display = pd.DataFrame(
    [[format_matchup(cell) for cell in row] for row in matchup_matrix],
    index=cluster_ids,
    columns=cluster_ids
)

print()
print("==========================================")
print("MATCHUP MATRIX (win rate / round differential)")
print("==========================================")
print()

with pd.option_context("display.width", None, "display.max_columns", None):
    print(matrix_display.to_string())

print()
print(f"Cross-cluster matches: {len(matches)}")
print(f"With round scores:     {matches['is_round_score'].sum()}")


# ============================================================
# SAVE
# ============================================================

win_rate_table.to_csv(WIN_RATE_OUTPUT)
round_differential_table.to_csv(ROUND_DIFFERENTIAL_OUTPUT)

teams = teams.sort_values(
    by=["cluster", "hltv_rank"],
    na_position="last"
)

teams.to_csv(TEAMS_OUTPUT, index=False)
summary.to_csv(SUMMARY_OUTPUT, index=False)

print()
print(f"Saved team comparison to {TEAMS_OUTPUT}")
print(f"Saved cluster summary to {SUMMARY_OUTPUT}")
