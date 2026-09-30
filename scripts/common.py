import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# PROJECT PATHS
# ============================================================
#
# Paths are based on this file's location, so the scripts
# work no matter which directory they are run from.
#
#   data/      scraped input data
#   outputs/   intermediate results
#   (root)     final adjusted player ratings

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "outputs"

MATCHES_FILE = DATA_DIR / "matches.csv"
PLAYERS_FILE = DATA_DIR / "player.csv"
RANKINGS_FILE = DATA_DIR / "teams.csv"

CLUSTERS_FILE = OUTPUT_DIR / "team_clusters.csv"


# ============================================================
# LOAD MATCHES
# ============================================================
#
# Cleaned, deduplicated matches with the score split into
# score_a / score_b and result_a (1 = teamA won, 0 = lost,
# 0.5 = draw).

def load_matches():

    matches = pd.read_csv(MATCHES_FILE)

    matches["teamA"] = matches["teamA"].astype(str).str.strip()
    matches["teamB"] = matches["teamB"].astype(str).str.strip()

    # The same match appears once for each scraped team
    matches = matches.drop_duplicates(subset=["matchId"])

    scores = matches["score"].str.extract(r"(\d+)\s*-\s*(\d+)").astype(float)
    matches["score_a"] = scores[0]
    matches["score_b"] = scores[1]

    matches = matches.dropna(subset=["score_a", "score_b"]).copy()

    matches["result_a"] = np.select(
        [
            matches["score_a"] > matches["score_b"],
            matches["score_a"] < matches["score_b"]
        ],
        [1.0, 0.0],
        default=0.5
    )

    return matches


# ============================================================
# LOAD CLUSTERS
# ============================================================
#
# Only teams that have a cluster, with cluster as an int and
# unranked teams at 0 HLTV points.

def load_clusters():

    clusters = pd.read_csv(CLUSTERS_FILE)

    clusters = clusters.dropna(subset=["cluster"]).copy()
    clusters["cluster"] = clusters["cluster"].astype(int)
    clusters["hltv_points"] = clusters["hltv_points"].fillna(0)

    return clusters
