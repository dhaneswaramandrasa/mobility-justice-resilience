# Who gets cut off when the network breaks

**Live demo: https://dhaneswaramandrasa.github.io/mobility-justice-resilience/**

[![Demo: a map of a Jakarta river corridor. Hexagons show where older people live, a flood blocks the streets along the river and blocks turn red where older people can no longer walk to a clinic, the blocked streets they rely on most glow red, the ranking switches to the busiest streets (amber, different ones), one street is blocked on its own, and fixes made before the disaster bring blocks back into reach](docs/demo-3d.gif)](https://dhaneswaramandrasa.github.io/mobility-justice-resilience/)

A proof of concept for **vulnerability-weighted network criticality**: when a disaster cuts the street network, which streets matter most, and for whom? Most resilience studies rank streets by how much traffic they carry. This method ranks them by how much access older people and children lose when each one is blocked, and then screens what to do about it, before the disaster and on the day.

The demo answers three plain questions for a **flood, a fire or an earthquake**:

1. **Who gets cut off?** On a normal day, how many older people can walk to a clinic, children to a school, and working-age adults to a station or market? How many still can once the streets are blocked? The map turns a block red when its people lose that walk.
2. **Which streets matter most?** Each blocked street is tested on its own, and ranked by who relies on it, next to how busy it is.
3. **What can we do before it happens?** Strengthen the street that matters most (raise it, widen it into a fire lane, retrofit it), add a new connection, or plan a shuttle for the day, and see how many people are back in reach.

The method is general. The pipeline takes any city from one config file (a bounding box, the river name, the hazards and optional district totals). The demo runs on an **example city**: a 5 by 6 km stretch of East and South Jakarta along the Ciliwung, which floods most years and has dense kampungs where fires spread fast.

The result that motivates the method: **of the 50 blocked streets that matter most for older people and children, only 1 to 8 are among the 50 busiest** (2 for the flood, 8 for the fire, 1 for the earthquake). The links vulnerable groups depend on are short alleys and paths to a clinic or school, which traffic counts never see.

## What is real and what is synthetic

| Layer | Source | Real or synthetic |
| --- | --- | --- |
| Streets, alleys and footpaths (22,731 links) | OpenStreetMap via Overpass | Real |
| Clinics, hospitals, schools, markets, stations (796 places) | OpenStreetMap | Real |
| Building footprints (120,000+) used to place people | OpenStreetMap | Real |
| River centreline | OpenStreetMap | Real |
| Age cohorts per grid cell | Built from buildings and district totals, see Layer 1 | **Synthetic** |
| Flood zone | Distance to the river plus a smooth random field | **Synthetic** |
| Fire zones | Ten fires in the densest blocks of buildings, 170 m reach | **Synthetic** |
| Earthquake debris | Narrow lanes in dense blocks blocked at random (55%), bridges closed (60%) | **Synthetic** |

The synthetic layers are stand-ins with a documented plug-in format. The numbers in the demo show how the method behaves; they are not findings about Jakarta.

## The method in three layers

### 1. Cohort population grid (pluggable)

A table of grid cells, each with counts of older people (65+), children (under 15) and working-age adults, plus optional mobility-dependence attributes such as the share of people without a private vehicle.

The shipped stand-in spreads BPS-style district totals (density and age shares per kecamatan, in `config.json`) over a hexagonal grid in proportion to OpenStreetMap building footprints. It then shifts the age mix with distance decay: older residents are more common in the old riverbank kampungs, children in the densest blocks. All of this is synthetic and labelled so on the page.

**Bring your own cohort map.** Any finer source drops in without code changes: a dasymetric age-cohort map built from POIs, footfall and buildings, census blocks, or archetypes derived from mobile traces. Save it as CSV:

| Column | Meaning |
| --- | --- |
| `cell_id` | any unique id (an H3 index works) |
| `lon`, `lat` | cell centre, WGS84 |
| `elderly`, `children`, `working` | head counts per cohort |
| `no_vehicle_share` | optional, 0 to 1 |

and run `python scripts/build_data.py --cohorts your_grid.csv`. `data/cohort_grid_synthetic.csv` is the stand-in written in this exact format, as a template. Cells of any shape and size work: each cell is attached to its nearest street junction.

**Archetypes (optional).** The cells are also grouped into four mobility justice archetypes with k-means on four variables, following Haxhija et al. (2024): walking accessibility, services within 800 m, frequency of use (a synthetic proxy here), and hazard exposure (how many of the three disasters hit the block). Switch the hexagons to "archetype" in step 1 of the demo.

### 2. Network criticality

1. **Network.** Walkable OSM ways are split at junctions into links (motorways excluded). Each cell and each destination snaps to its nearest junction.
2. **Access per cohort.** For each type of place (health, school, market, transit) a multi-source shortest-path search gives the walking distance from every cell. A cohort's access is a gravity score: the sum over place types of *need* times *exp(-distance / decay)*. Need and decay differ by cohort:

   | Cohort | Health | School | Market | Transit | Walk decay |
   | --- | --- | --- | --- | --- | --- |
   | Older people | 0.50 | 0 | 0.35 | 0.15 | 350 m |
   | Children | 0.20 | 0.65 | 0.05 | 0.10 | 500 m |
   | Working age | 0.10 | 0.10 | 0.30 | 0.50 | 900 m |

3. **Walking reach.** For the plain-language counts, a cohort is *within walking reach* if its main place is close along the streets: older people within 500 m of a clinic, children within 800 m of a school, working-age adults within 1,000 m of a station or market. The reach map compares a normal day with the disaster day.
4. **Hazards.** Each hazard marks which streets are blocked: flood (3,246 streets near the river), fire (692 streets near ten fires in the densest blocks), earthquake (3,271 narrow lanes in dense blocks and some bridges).
5. **Block one street at a time.** Every blocked street is removed on its own and access is recomputed. Links not on any shortest-path tree are skipped, since removing them changes nothing. The loss is counted in *people's worth of access*: each cell's population times the share of its access that is gone.
6. **Weight by vulnerability.** The weighted criticality of a link is 3 x older people's loss + 2 x children's loss + 1 x working-age loss. Two baselines rank the same links: *everyone the same* (one need profile, one walk distance, no weights) and *how busy* (population-weighted shortest-path flow through the link, from 400 sampled origins).
7. **Compare the rankings.** Top-50 overlap between weighted and busiest is 2, 8 and 1 of 50 for flood, fire and earthquake; between weighted and everyone-the-same it is 42, 47 and 46. So most of the gap comes from looking at access at all, and the weights then reorder the top.

All weights live at the top of `scripts/build_data.py`, in one place, so a team can set them from its own survey or trace data.

### 3. Policy screening

With every exposed street blocked at once, the model searches for three kinds of low-regret action and measures how many people each brings back within walking reach:

| When | Action | Flood / fire / earthquake | How it is found |
| --- | --- | --- | --- |
| Before | Strengthen a street | raise it / widen the alley into a fire lane / retrofit and widen it | the blocked streets on the shortest route from a cut-off hot spot to its nearest safe clinic, school or market (up to 700 m) |
| Before | Add a connection | raised footbridge / a second way out / a new connecting street | a new link between two safe junctions under 300 m apart that the disaster pushes far apart on the network |
| On the day | Shuttle | a minibus from a safe pickup point to the nearest safe clinic, counted as 250 m of walking | the pickup that brings back the most weighted access |

Click any blocked street in the demo to see the "strengthen this one" case: who would keep their access if just that street were made to stay open.

All combinations are precomputed so the page can toggle them freely. The best quick win is the fix with the most weighted access restored per unit of rough cost.

## Run it on another city

```bash
pip install -r requirements.txt          # numpy and scipy
# edit config.json: name, bbox [south, west, north, east], river_names, districts
python scripts/fetch_osm.py              # streets, places, buildings into data/raw/
python scripts/build_data.py             # writes web/data/city.json (about 2 minutes)
python -m http.server -d web 8000        # open http://localhost:8000
```

Options:

- `--cohorts grid.csv` uses your cohort map instead of the synthetic one.
- A real hazard layer: in `config.json` give any hazard `"source": "file", "path": "zone.geojson"`, with polygons of the exposed areas. Streets and cells inside the polygons count as hit. Add or remove hazards in the same block.
- `districts` can be left out of a real run if you pass `--cohorts`.

`web/index.html` is one static page (MapLibre GL 4.7.1 and deck.gl 9.0.38 from unpkg, basemap from OpenFreeMap). `?step=0` to `4` opens a step; `?hz=fire`, `?cohort=children` and `?rank=traffic` set the view, and `?embed` shows a tap-to-load button first (phones get it too). GitHub Actions deploys `web/` to Pages on every push to `main`.

## Limits

- The cohort grid and all three hazards are synthetic, so no number here describes real people or real risk.
- Walking only. Motorbike, angkot and KRL access during a disaster would need their own networks and GTFS feeds.
- One link at a time misses combinations; the disaster day covers the all-at-once case, but not the order in which links fail.
- Need weights and walking distances are assumptions, set in one place to be replaced with survey or trace estimates.

## Layout

```
config.json                 the city: bbox, river, hazards, district totals, grid size
scripts/fetch_osm.py        Overpass downloads into data/raw/
scripts/build_data.py       cohort grid, network, criticality, fixes, archetypes, JSON export
data/raw/                   OSM extracts for the example city
data/cohort_grid_synthetic.csv   the stand-in cohort grid, in the plug-in format
web/index.html              the demo
web/data/city.json          model output read by the demo
assets/card.mp4             short clip of the demo
```

MIT licensed. Map data © OpenStreetMap contributors (ODbL).
