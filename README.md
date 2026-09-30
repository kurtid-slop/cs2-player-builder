# CS2 Player Builder

**Play:** https://kurtid-slop.github.io/cs2-player-builder/

Spin a random CS2 team, lock in one of its players and take one of their attributes. Seven rounds, seven attributes, one respin.

The site lives in `docs/`. Player attributes are adjusted for strength of schedule using team clusters built from match results.

## Pipeline

```
python scripts/team_network.py          # Elo, match network, clusters -> outputs/
python scripts/compare_clusters.py      # clusters vs HLTV ranking, cluster matchup matrix
python scripts/team_fit.py <team> [cluster] [--apply]   # test moving a team between clusters
python scripts/adjust_player_stats.py   # schedule-adjusted stats -> player_adjusted.csv
python scripts/build_site.py            # game data -> docs/players.js
```

Add `?debug` to the site URL to show clusters and data details.
