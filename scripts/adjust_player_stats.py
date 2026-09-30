import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import spearmanr

from common import CLUSTERS_FILE, PLAYERS_FILE, ROOT_DIR, load_matches


# ============================================================
# CONFIGURATION
# ============================================================

# The final ratings go in the project root
PLAYERS_OUTPUT = ROOT_DIR / "player_adjusted.csv"
RATINGS_OUTPUT = ROOT_DIR / "cluster_ratings.csv"

STATS = [
    "Firepower",
    "Entrying",
    "Trading",
    "Opening",
    "Clutching",
    "Sniping",
    "Utility",
    "Rating 3.0"
]

# The stat used to calibrate alpha
CALIBRATION_STAT = "Rating 3.0"

# Values of alpha to try
ALPHA_GRID = np.round(np.arange(0, 3.01, 0.05), 2)

# Teams need at least this many active players to be used
# when calibrating alpha
MIN_PLAYERS_PER_TEAM = 3

# Pulls cluster ratings towards the average so a cluster
# with a handful of 100% or 0% results doesn't get an
# extreme rating. Higher = stronger pull.
RATING_PRIOR = 0.05

# Elo scale: 400 points = 10 to 1 odds
ELO_SCALE = 400


# ============================================================
# LOAD CLUSTERS
# ============================================================

clusters = pd.read_csv(CLUSTERS_FILE)

team_strength = dict(zip(clusters["team"], clusters["strength"]))

clusters = clusters.dropna(subset=["cluster"])
clusters["cluster"] = clusters["cluster"].astype(int)

cluster_lookup = dict(zip(clusters["team"], clusters["cluster"]))
cluster_ids = sorted(clusters["cluster"].unique())
index = {cluster_id: i for i, cluster_id in enumerate(cluster_ids)}
n = len(cluster_ids)


# ============================================================
# LOAD MATCHES
# ============================================================

matches = load_matches()

matches["cluster_a"] = matches["teamA"].map(cluster_lookup)
matches["cluster_b"] = matches["teamB"].map(cluster_lookup)


# ============================================================
# CLUSTER RATINGS (BRADLEY-TERRY)
# ============================================================
#
# Fits one rating per cluster from cluster-vs-cluster results,
# on the Elo scale:
#
#   P(x beats y) = 1 / (1 + 10 ^ ((R_y - R_x) / 400))
#
# Clusters that never played each other are still placed on
# the same scale through the clusters they both played.

cross = matches.dropna(subset=["cluster_a", "cluster_b"])
cross = cross[cross["cluster_a"] != cross["cluster_b"]]

wins = np.zeros((n, n))
played = np.zeros((n, n))

for row in cross.itertuples():

    a = index[int(row.cluster_a)]
    b = index[int(row.cluster_b)]

    wins[a, b] += row.result_a
    wins[b, a] += 1 - row.result_a
    played[a, b] += 1
    played[b, a] += 1


def negative_log_likelihood(strengths):

    difference = strengths[:, None] - strengths[None, :]
    log_p = -np.logaddexp(0, -difference)

    likelihood = (wins * log_p).sum()
    prior = RATING_PRIOR * (strengths ** 2).sum()

    return -likelihood + prior


fit = minimize(negative_log_likelihood, np.zeros(n), method="L-BFGS-B")

# Natural-log strengths -> Elo points
ratings = fit.x * ELO_SCALE / np.log(10)
ratings -= ratings.mean()

cluster_rating = dict(zip(cluster_ids, ratings))


# ============================================================
# SCHEDULE RATING PER TEAM
# ============================================================
#
# Average rating of the clusters a team's opponents belong
# to, over every match it played. Player stats are earned
# against these opponents, so this is what they get adjusted
# for, not the team's own cluster.
#
# Opponents without a cluster are skipped. A team with no
# clustered opponents falls back to its own cluster's rating.

side_a = matches[["teamA", "cluster_b"]].rename(
    columns={"teamA": "team", "cluster_b": "opponent_cluster"}
)

side_b = matches[["teamB", "cluster_a"]].rename(
    columns={"teamB": "team", "cluster_a": "opponent_cluster"}
)

schedule = pd.concat([side_a, side_b]).dropna(subset=["opponent_cluster"])
schedule["opponent_rating"] = (
    schedule["opponent_cluster"].astype(int).map(cluster_rating)
)

schedule_rating = schedule.groupby("team")["opponent_rating"].mean().to_dict()
schedule_matches = schedule.groupby("team").size().to_dict()


def team_schedule(team):

    if team in schedule_rating:
        return schedule_rating[team], "opponents"

    if team in cluster_lookup:
        return cluster_rating[cluster_lookup[team]], "own cluster"

    return np.nan, "none"


# ============================================================
# LOAD PLAYERS
# ============================================================

players = pd.read_csv(PLAYERS_FILE)

players["age"] = pd.to_numeric(
    players["age"].astype(str).str.replace(" years", "", regex=False),
    errors="coerce"
)

for stat in STATS:
    players[stat] = pd.to_numeric(
        players[stat].astype(str).str.replace("/100", "", regex=False),
        errors="coerce"
    )

# "FaZe (benched)" -> team "FaZe", status "benched"
parts = players["team"].str.extract(r"^(.*?)\s*(?:\((.+)\))?$")
players["team"] = parts[0]
players["status"] = parts[1].fillna("active")

players.loc[players["team"] == "No team", "team"] = np.nan

schedule_info = players["team"].map(
    lambda team: team_schedule(team) if pd.notna(team) else (np.nan, "none")
)

players["schedule_rating"] = schedule_info.map(lambda x: x[0])
players["schedule_source"] = schedule_info.map(lambda x: x[1])
players["team_strength"] = players["team"].map(team_strength)

# Players without any schedule get the average, i.e. no change
average_schedule = players["schedule_rating"].mean()

players["schedule_rating"] = players["schedule_rating"].fillna(
    average_schedule
)

# Centre on the average player so their stats don't change
players["schedule_relative"] = (
    players["schedule_rating"] - average_schedule
) / ELO_SCALE


# ============================================================
# CALIBRATE ALPHA
# ============================================================
#
# multiplier = 1 + alpha * (schedule_rating - average) / 400
#
# For each alpha, average the adjusted calibration stat for
# every team's active players and check how well it lines up
# with the team's Elo strength. The best alpha is the one
# where adjusted stats best reflect how good teams really are.

calibration = players[
    (players["status"] == "active") &
    players["team_strength"].notna() &
    players[CALIBRATION_STAT].notna()
]

team_sizes = calibration.groupby("team").size()
calibration = calibration[
    calibration["team"].isin(
        team_sizes[team_sizes >= MIN_PLAYERS_PER_TEAM].index
    )
]

alpha_scores = []

for alpha in ALPHA_GRID:

    multiplier = 1 + alpha * calibration["schedule_relative"]
    adjusted = calibration[CALIBRATION_STAT] * multiplier

    team_average = adjusted.groupby(calibration["team"]).mean()
    strength = calibration.groupby("team")["team_strength"].first()

    alpha_scores.append(spearmanr(team_average, strength)[0])

alpha_scores = pd.Series(alpha_scores, index=ALPHA_GRID)
best_alpha = alpha_scores.idxmax()


# ============================================================
# ADJUST STATS
# ============================================================

players["multiplier"] = 1 + best_alpha * players["schedule_relative"]

for stat in STATS:
    players[f"{stat} adjusted"] = (
        players[stat] * players["multiplier"]
    ).round(2)


# ============================================================
# DISPLAY
# ============================================================

print()
print("==========================================")
print("CLUSTER RATINGS (Elo scale, average = 0)")
print("==========================================")
print()

ratings_table = pd.DataFrame({
    "cluster": cluster_ids,
    "rating": np.round(ratings, 0).astype(int),
    "cross_cluster_matches": played.sum(axis=1).astype(int)
}).sort_values("rating", ascending=False)

print(ratings_table.to_string(index=False))

print()
print("==========================================")
print("ALPHA CALIBRATION")
print("==========================================")
print()
print(f"Teams used: {calibration['team'].nunique()}")
print(
    f"Spearman of team average {CALIBRATION_STAT} vs Elo strength:"
)
print(f"  alpha = 0 (raw stats):   {alpha_scores.loc[0.0]:.3f}")
print(f"  alpha = {best_alpha:<4} (best):     {alpha_scores.loc[best_alpha]:.3f}")

if best_alpha == ALPHA_GRID.max():
    print("  WARNING: best alpha is at the edge of the grid.")

print()
print("==========================================")
print("MULTIPLIERS")
print("==========================================")
print()
print(
    f"Range: {players['multiplier'].min():.3f} "
    f"to {players['multiplier'].max():.3f}"
)
print(
    "Players without a schedule (no team or unknown team): "
    f"{(players['schedule_source'] == 'none').sum()}"
)
print()

teams_table = (
    players.dropna(subset=["team"])
    .groupby("team")
    .agg(
        multiplier=("multiplier", "first"),
        schedule_rating=("schedule_rating", "first"),
        source=("schedule_source", "first")
    )
    .sort_values("multiplier", ascending=False)
)

teams_table["multiplier"] = teams_table["multiplier"].round(3)
teams_table["schedule_rating"] = teams_table["schedule_rating"].round(0)

print("Highest multipliers (toughest schedules):")
print(teams_table.head(10).to_string())
print()
print("Lowest multipliers (easiest schedules):")
print(teams_table.tail(10).to_string())

print()
print("==========================================")
print(f"TOP 15 PLAYERS BY ADJUSTED {CALIBRATION_STAT.upper()}")
print("==========================================")
print()

top = players.sort_values(
    f"{CALIBRATION_STAT} adjusted", ascending=False
).head(15)

print(
    top[
        [
            "playername",
            "team",
            CALIBRATION_STAT,
            f"{CALIBRATION_STAT} adjusted",
            "multiplier"
        ]
    ].to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

ratings_table.to_csv(RATINGS_OUTPUT, index=False)

columns = (
    ["playername", "age", "team", "status", "team_strength",
     "schedule_rating", "schedule_source", "multiplier"]
    + [c for stat in STATS for c in (stat, f"{stat} adjusted")]
)

players[columns].to_csv(PLAYERS_OUTPUT, index=False)

print()
print(f"Saved cluster ratings to {RATINGS_OUTPUT.name}")
print(f"Saved adjusted player stats to {PLAYERS_OUTPUT.name}")
