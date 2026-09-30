import pandas as pd
import numpy as np
import sys
import os


# ============================================================
# USAGE
# ============================================================

if len(sys.argv) != 3:
    print(
        "Usage: python team_tiers.py "
        "<input.csv> <output.csv>"
    )
    print(
        "Example: python scripts/team_tiers.py "
        "outputs/team_strength.csv outputs/team_tiered.csv"
    )
    sys.exit(1)


input_file = sys.argv[1]
output_file = sys.argv[2]


# ============================================================
# SETTINGS
# ============================================================

# Number of broad competitive tiers
NUM_TIERS = 5

# Minimum number of teams in a tier
MIN_TIER_SIZE = 3

# How strongly we favour unusually large gaps
# Higher = fewer boundaries
GAP_WEIGHT = 1.5


# ============================================================
# CHECK FILE
# ============================================================

if not os.path.isfile(input_file):

    print(f"Error: '{input_file}' not found.")
    sys.exit(1)


# ============================================================
# LOAD DATA
# ============================================================

data = pd.read_csv(input_file)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "team",
    "strength",
    "strength_gap"
]

for column in required_columns:

    if column not in data.columns:

        print(
            f"Error: Missing required column "
            f"'{column}'"
        )

        sys.exit(1)


# ============================================================
# SORT BY STRENGTH
# ============================================================

data = data.sort_values(
    "strength",
    ascending=False
).reset_index(drop=True)


# ============================================================
# CLEAN STRENGTH GAP
# ============================================================

data["strength_gap"] = pd.to_numeric(
    data["strength_gap"],
    errors="coerce"
)

data["strength"] = pd.to_numeric(
    data["strength"],
    errors="coerce"
)


# Remove invalid rows
data = data.dropna(
    subset=[
        "strength",
        "strength_gap"
    ]
).reset_index(drop=True)


# ============================================================
# GET GAPS
# ============================================================

gaps = data["strength_gap"].iloc[:-1].values


# ============================================================
# GAP DISTRIBUTION
# ============================================================

median_gap = np.median(gaps)

mean_gap = np.mean(gaps)

std_gap = np.std(gaps)


print()
print("==========================================")
print("STRENGTH DISTRIBUTION")
print("==========================================")

print(
    f"Teams:        {len(data)}"
)

print(
    f"Mean gap:     {mean_gap:.2f}"
)

print(
    f"Median gap:   {median_gap:.2f}"
)

print(
    f"Std deviation:{std_gap:.2f}"
)


# ============================================================
# CALCULATE GAP SCORE
# ============================================================
#
# We don't simply say:
#
#     gap > X = tier boundary
#
# Instead we determine how unusual each gap is relative
# to the rest of the strength distribution.
#
# A gap of 5 points is normal.
# A gap of 20 points might be meaningful.
# A gap of 70 points is extremely meaningful.
#
# ============================================================

gap_zscores = (
    gaps - mean_gap
) / (
    std_gap if std_gap > 0 else 1
)


# ============================================================
# BUILD CANDIDATE BOUNDARIES
# ============================================================

candidates = []


for i, gap in enumerate(gaps):

    teams_above = i + 1

    teams_below = (
        len(data) - teams_above
    )


    # --------------------------------------------------------
    # Never create tiny tiers
    # --------------------------------------------------------

    if teams_above < MIN_TIER_SIZE:
        continue

    if teams_below < MIN_TIER_SIZE:
        continue


    # --------------------------------------------------------
    # Calculate how unusual the gap is
    # --------------------------------------------------------

    z = gap_zscores[i]


    # --------------------------------------------------------
    # Position score
    #
    # We don't want all boundaries concentrated at the
    # bottom of the ranking.
    #
    # This favours reasonably sized competitive groups.
    # --------------------------------------------------------

    position = (
        teams_above / len(data)
    )


    # Distance from the centre
    #
    # Values near 0.5 are more neutral.
    #

    balance_score = (
        1 -
        abs(
            position - 0.5
        )
    )


    # --------------------------------------------------------
    # Combined boundary score
    # --------------------------------------------------------

    boundary_score = (
        z
        +
        GAP_WEIGHT * balance_score
    )


    candidates.append({

        "index": i,

        "gap": gap,

        "zscore": z,

        "balance": balance_score,

        "score": boundary_score

    })


# ============================================================
# SELECT TIER BOUNDARIES
# ============================================================
#
# We need NUM_TIERS - 1 boundaries.
#
# But boundaries cannot be too close together.
#
# ============================================================

required_boundaries = NUM_TIERS - 1


# Sort by boundary quality
candidates = sorted(
    candidates,
    key=lambda x: x["score"],
    reverse=True
)


selected = []


# Minimum distance between boundaries.
#
# With 49 teams and 5 tiers, this prevents things like:
#
# Tier 1 = 3 teams
# Tier 2 = 3 teams
# Tier 3 = 3 teams
# Tier 4 = 35 teams
#
# while still allowing natural variation.
#

minimum_boundary_distance = max(
    MIN_TIER_SIZE,
    int(len(data) / (NUM_TIERS * 2))
)


for candidate in candidates:

    index = candidate["index"]


    # Check distance from existing boundaries

    too_close = False

    for selected_boundary in selected:

        if abs(
            index - selected_boundary["index"]
        ) < minimum_boundary_distance:

            too_close = True
            break


    if too_close:
        continue


    selected.append(
        candidate
    )


    if len(selected) >= required_boundaries:
        break


# ============================================================
# SORT BOUNDARIES BY RANK
# ============================================================

selected = sorted(
    selected,
    key=lambda x: x["index"]
)


# ============================================================
# FALLBACK
# ============================================================
#
# If there aren't enough statistically strong boundaries,
# choose the strongest remaining valid gaps.
#
# ============================================================

if len(selected) < required_boundaries:

    for candidate in candidates:

        if candidate in selected:
            continue


        index = candidate["index"]


        too_close = False

        for selected_boundary in selected:

            if abs(
                index -
                selected_boundary["index"]
            ) < MIN_TIER_SIZE:

                too_close = True
                break


        if too_close:
            continue


        selected.append(
            candidate
        )


        if len(selected) >= required_boundaries:
            break


# Final ordering
selected = sorted(
    selected,
    key=lambda x: x["index"]
)


# ============================================================
# DISPLAY BOUNDARIES
# ============================================================

print()
print("==========================================")
print("SELECTED TIER BOUNDARIES")
print("==========================================")

for boundary in selected:

    index = boundary["index"]

    upper_team = data.loc[
        index,
        "team"
    ]

    lower_team = data.loc[
        index + 1,
        "team"
    ]

    upper_strength = data.loc[
        index,
        "strength"
    ]

    lower_strength = data.loc[
        index + 1,
        "strength"
    ]

    gap = boundary["gap"]

    print()

    print(
        f"Tier {len([
            b for b in selected
            if b["index"] <= index
        ])} boundary"
    )

    print(
        f"{upper_team} "
        f"({upper_strength:.2f})"
    )

    print(
        f"        ↓ {gap:.2f} strength gap"
    )

    print(
        f"{lower_team} "
        f"({lower_strength:.2f})"
    )


# ============================================================
# ASSIGN TIERS
# ============================================================

data["tier"] = 1


current_tier = 1


for boundary in selected:

    boundary_index = boundary["index"]

    current_tier += 1

    data.loc[
        boundary_index + 1:,
        "tier"
    ] = current_tier


# ============================================================
# TIER NAMES
# ============================================================

tier_names = {

    1: "Elite",

    2: "Contender",

    3: "Upper Competitive",

    4: "Competitive",

    5: "Lower Competitive"

}


data["tier_name"] = (
    data["tier"]
    .map(tier_names)
)


# ============================================================
# CALCULATE RANK
# ============================================================

data["rank"] = (
    np.arange(
        1,
        len(data) + 1
    )
)


# ============================================================
# CALCULATE WITHIN-TIER POSITION
# ============================================================
#
# This is important.
#
# We don't want to throw away the fact that:
#
# G2 = 2094
# BetBoom = 2042
#
# even though they're both Tier 3.
#
# ============================================================

data["tier_min_strength"] = (
    data.groupby("tier")["strength"]
    .transform("min")
)

data["tier_max_strength"] = (
    data.groupby("tier")["strength"]
    .transform("max")
)


strength_range = (
    data["tier_max_strength"]
    -
    data["tier_min_strength"]
)


data["within_tier_strength"] = np.where(

    strength_range > 0,

    (
        data["strength"]
        -
        data["tier_min_strength"]
    )
    /
    strength_range,

    1.0
)


# ============================================================
# CLEAN UP
# ============================================================

data = data.drop(
    columns=[
        "tier_min_strength",
        "tier_max_strength"
    ]
)


# ============================================================
# REORDER COLUMNS
# ============================================================

first_columns = [
    "rank",
    "tier",
    "tier_name",
    "team",
    "strength",
    "within_tier_strength",
    "strength_gap"
]


remaining_columns = [
    column
    for column in data.columns
    if column not in first_columns
]


data = data[
    first_columns +
    remaining_columns
]


# ============================================================
# SAVE
# ============================================================

data.to_csv(
    output_file,
    index=False
)


# ============================================================
# DISPLAY TIERS
# ============================================================

print()
print("==========================================")
print("FINAL TEAM TIERS")
print("==========================================")


for tier in sorted(
    data["tier"].unique()
):

    tier_data = data[
        data["tier"] == tier
    ]


    print()

    print(
        f"Tier {tier} — "
        f"{tier_names.get(tier, '')}"
    )

    print(
        "-" * 50
    )


    for _, row in tier_data.iterrows():

        print(
            f"{int(row['rank']):2d}. "
            f"{row['team']:<25} "
            f"{row['strength']:7.2f}"
        )


# ============================================================
# SUMMARY
# ============================================================

print()
print("==========================================")
print("TIER SIZES")
print("==========================================")


tier_counts = (
    data["tier"]
    .value_counts()
    .sort_index()
)


for tier, count in tier_counts.items():

    print(
        f"Tier {tier}: {count} teams"
    )


print()
print(
    f"Saved to '{output_file}'"
)