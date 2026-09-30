import pandas as pd
import numpy as np
import sys

from common import CLUSTERS_FILE, OUTPUT_DIR, RANKINGS_FILE, load_clusters, load_matches


# ============================================================
# USAGE
# ============================================================
#
# python team_fit.py <team> [target_cluster] [--apply]
#
# Lists the clusters the team fits best, then shows what
# happens to the cluster rankings if the team moves to
# target_cluster. Without a target, the best-fitting cluster
# other than its current one is used.
#
# --apply writes the move to team_clusters.csv. A backup of
# the file is saved first.

apply_changes = "--apply" in sys.argv
arguments = [a for a in sys.argv[1:] if a != "--apply"]

if len(arguments) not in (1, 2):
    print("Usage: python team_fit.py <team> [target_cluster] [--apply]")
    sys.exit(1)

team_name = arguments[0]
target_cluster = int(arguments[1]) if len(arguments) == 2 else None


# ============================================================
# CONFIGURATION
# ============================================================

BACKUP_FILE = OUTPUT_DIR / "team_clusters.backup.csv"

# Fewer matches than this against a cluster and the win rate
# is too noisy to use for the fit score
MIN_MATCHES = 5


# ============================================================
# LOAD DATA
# ============================================================

clusters = load_clusters()

teams = clusters[["team", "cluster", "strength", "hltv_points"]].copy()

hltv_rank = dict(
    pd.read_csv(RANKINGS_FILE)[["team", "ranking"]].values
)

# Case-insensitive team lookup
lookup = {name.lower(): name for name in teams["team"]}

if team_name.lower() not in lookup:
    print(f"ERROR: {team_name} is not in any cluster.")
    sys.exit(1)

team_name = lookup[team_name.lower()]
team = teams[teams["team"] == team_name].iloc[0]
current_cluster = int(team["cluster"])

if target_cluster is not None and target_cluster not in set(teams["cluster"]):
    print(f"ERROR: cluster {target_cluster} does not exist.")
    sys.exit(1)


# ============================================================
# TEAM RESULTS AGAINST EACH CLUSTER
# ============================================================

matches = load_matches()

# Put every match of the team from its own side
as_a = matches[matches["teamA"] == team_name]
as_b = matches[matches["teamB"] == team_name]

team_matches = pd.DataFrame({
    "opponent": pd.concat([as_a["teamB"], as_b["teamA"]]),
    "result": pd.concat([as_a["result_a"], 1 - as_b["result_a"]])
})

cluster_lookup = dict(zip(teams["team"], teams["cluster"]))
team_matches["opponent_cluster"] = team_matches["opponent"].map(cluster_lookup)

results = (
    team_matches
    .dropna(subset=["opponent_cluster"])
    .groupby("opponent_cluster")
    .agg(
        matches=("result", "count"),
        win_rate=("result", "mean")
    )
)

results.index = results.index.astype(int)


# ============================================================
# FIT SCORE
# ============================================================
#
# Two distances, lower = better fit:
#
#   strength_distance  how far the team's Elo strength is from
#                      the cluster's average, in standard
#                      deviations of all clustered teams
#
#   results_distance   how far its win rate against the cluster
#                      is from 50%, scaled so 0% or 100% = 1.
#                      A team that wins about half its games
#                      against a cluster plays at its level.
#
# The team itself is left out of its current cluster's
# average so it isn't compared against itself.
#
# fit_score is the average of whichever distances exist.

strength_std = teams["strength"].std()

fit_rows = []

for cluster_id, group in teams.groupby("cluster"):

    members = group[group["team"] != team_name]

    if members.empty:
        continue

    average_strength = members["strength"].mean()

    strength_distance = (
        abs(team["strength"] - average_strength) / strength_std
    )

    played = results["matches"].get(cluster_id, 0)
    win_rate = results["win_rate"].get(cluster_id, np.nan)

    results_distance = (
        abs(win_rate - 0.5) * 2
        if played >= MIN_MATCHES
        else np.nan
    )

    fit_rows.append({
        "cluster": cluster_id,
        "teams": len(members),
        "average_strength": average_strength,
        "strength_distance": strength_distance,
        "matches": played,
        "win_rate": win_rate,
        "results_distance": results_distance,
        "fit_score": np.nanmean([strength_distance, results_distance])
    })

fit = pd.DataFrame(fit_rows).sort_values("fit_score").reset_index(drop=True)
fit.index += 1

print()
print("==========================================")
print(f"CLUSTER FIT FOR {team_name.upper()}")
print("==========================================")
print()
print(f"Current cluster: {current_cluster}")
print(f"Elo strength:    {team['strength']:.0f}")
print(f"HLTV rank:       {hltv_rank.get(team_name, 'unranked')}")
print(f"HLTV points:     {team['hltv_points']:.0f}")
print()

display = fit.copy()
display["average_strength"] = display["average_strength"].round(0).astype(int)
display["strength_distance"] = display["strength_distance"].round(2)
display["win_rate"] = display["win_rate"].map(
    lambda v: "" if pd.isna(v) else f"{v:.0%}"
)
display["results_distance"] = display["results_distance"].map(
    lambda v: "" if pd.isna(v) else f"{v:.2f}"
)
display["fit_score"] = display["fit_score"].round(2)
display["current"] = np.where(display["cluster"] == current_cluster, "<--", "")

print(display.to_string())
print()
print("Lower fit_score = better fit.")
print(f"Win rates from fewer than {MIN_MATCHES} matches are not scored.")


# ============================================================
# PICK TARGET CLUSTER
# ============================================================

if target_cluster is None:

    target_cluster = int(
        fit.loc[fit["cluster"] != current_cluster, "cluster"].iloc[0]
    )

    print()
    print(f"No target given, using best fit: cluster {target_cluster}")

if target_cluster == current_cluster:
    print()
    print(f"{team_name} is already in cluster {target_cluster}.")
    sys.exit(0)


# ============================================================
# RANK CLUSTERS
# ============================================================
#
# Same ordering as team_network.py: average HLTV points,
# with average Elo strength as the tie-breaker.

def rank_clusters(assignments):

    summary = (
        assignments
        .groupby("cluster")
        .agg(
            teams=("team", "count"),
            average_hltv_points=("hltv_points", "mean"),
            average_strength=("strength", "mean")
        )
        .sort_values(
            ["average_hltv_points", "average_strength"],
            ascending=False
        )
    )

    summary["position"] = range(1, len(summary) + 1)

    return summary


before = rank_clusters(teams)

moved = teams.copy()
moved.loc[moved["team"] == team_name, "cluster"] = target_cluster

after = rank_clusters(moved)


# ============================================================
# COMPARE RANKINGS
# ============================================================
#
# Clusters keep their current number as a label so each one
# can be followed from before to after the move.

comparison = before.join(
    after,
    lsuffix="_before",
    rsuffix="_after",
    how="outer"
)

comparison["change"] = (
    comparison["position_before"] - comparison["position_after"]
)

comparison = comparison.sort_values("position_after", na_position="last")

print()
print("==========================================")
print(
    f"MOVING {team_name.upper()}: "
    f"CLUSTER {current_cluster} -> CLUSTER {target_cluster}"
)
print("==========================================")
print()

rows = []

for cluster_id, row in comparison.iterrows():

    if pd.isna(row["position_after"]):
        change = "disbanded"
    elif row["change"] > 0:
        change = f"up {int(row['change'])}"
    elif row["change"] < 0:
        change = f"down {int(-row['change'])}"
    else:
        change = ""

    note = ""
    if cluster_id == current_cluster:
        note = f"loses {team_name}"
    elif cluster_id == target_cluster:
        note = f"gains {team_name}"

    rows.append({
        "cluster": cluster_id,
        "position_before": int(row["position_before"]),
        "position_after": (
            "" if pd.isna(row["position_after"])
            else int(row["position_after"])
        ),
        "change": change,
        "avg_points_before": round(row["average_hltv_points_before"], 1),
        "avg_points_after": (
            "" if pd.isna(row["average_hltv_points_after"])
            else round(row["average_hltv_points_after"], 1)
        ),
        "note": note
    })

print(pd.DataFrame(rows).to_string(index=False))


# ============================================================
# SUMMARY
# ============================================================

print()

old_position = int(before.loc[current_cluster, "position"])

if current_cluster in after.index:

    new_position = int(after.loc[current_cluster, "position"])

    if new_position == old_position:
        print(
            f"Cluster {current_cluster} stays at #{old_position} "
            f"without {team_name}."
        )
    else:
        print(
            f"Without {team_name}, cluster {current_cluster} moves "
            f"from #{old_position} to #{new_position}."
        )

else:
    print(
        f"Cluster {current_cluster} is empty without {team_name} "
        "and disappears."
    )

moved_clusters = comparison[comparison["change"].fillna(1) != 0]

print(
    f"{len(moved_clusters)} of {len(before)} clusters change position."
)


# ============================================================
# APPLY MOVE TO CSV
# ============================================================
#
# Moves the team in team_clusters.csv, then renumbers the
# clusters so each number is the cluster's new position
# (cluster 1 = strongest). All cluster-level columns are
# recalculated.
#
# Note: running team_network.py again overwrites this file.

def apply_move(team_name, target_cluster, new_ranking):

    data = pd.read_csv(CLUSTERS_FILE)
    columns = data.columns.tolist()

    data.to_csv(BACKUP_FILE, index=False)

    data.loc[data["team"] == team_name, "cluster"] = target_cluster

    # Old cluster number -> new position
    renumber = new_ranking["position"].to_dict()
    data["cluster"] = data["cluster"].map(renumber)

    cluster_info = (
        data
        .dropna(subset=["cluster"])
        .groupby("cluster")
        .agg(
            cluster_size=("team", "count"),
            average_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            average_strength=("strength", "mean"),
            median_strength=("strength", "median"),
            average_hltv_points=("hltv_points", lambda s: s.fillna(0).mean())
        )
        .reset_index()
    )

    data = data.drop(
        columns=[c for c in cluster_info.columns if c != "cluster"]
    ).merge(
        cluster_info,
        on="cluster",
        how="left"
    )

    data = data.sort_values(
        by=["cluster", "strength"],
        ascending=[True, False],
        na_position="last"
    ).reset_index(drop=True)

    data[columns].to_csv(CLUSTERS_FILE, index=False)

    return renumber


if apply_changes:

    renumber = apply_move(team_name, target_cluster, after)

    print()
    print("==========================================")
    print(f"SAVED TO {CLUSTERS_FILE.name.upper()}")
    print("==========================================")
    print()
    print(f"Backup of the previous file: {BACKUP_FILE.name}")
    print()

    renamed = {
        old: new for old, new in renumber.items() if old != new
    }

    if renamed:
        print("Clusters renumbered to match their new positions:")
        for old, new in sorted(renamed.items(), key=lambda x: x[1]):
            print(f"  cluster {old} -> cluster {new}")
    else:
        print("No clusters were renumbered.")

    print()
    print(
        f"{team_name} is now in cluster "
        f"{renumber[target_cluster]}."
    )

else:

    print()
    print("Run again with --apply to save this move to "
          f"{CLUSTERS_FILE.name}.")
