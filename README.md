# Who gets cut off when the network breaks

**Live demo: https://dhaneswaramandrasa.github.io/mobility-justice-resilience/**

[![Demo: a 3D map of a Jakarta river corridor. Hexagons show where older people live, the river floods the streets along it, then red walls rise on the flooded links older people rely on most, the ranking switches to the busiest links (amber, different streets), one link is cut to show who loses access, and three cheap fixes are switched on](docs/demo-3d.gif)](https://dhaneswaramandrasa.github.io/mobility-justice-resilience/)

A proof of concept for **vulnerability-weighted network criticality**: when a flood cuts the street network, which links matter most, and for whom? Most resilience studies rank links by how much traffic they carry. This method ranks them by how much access older people and children lose when each link goes under water, and then screens cheap fixes by how much of that access they bring back.

The method is general. The pipeline takes any city from one config file (a bounding box, the river name and optional district totals). The demo runs on an **example city**: a 5 by 6 km stretch of East and South Jakarta along the Ciliwung, which floods most years.

The result that motivates the method: **of the 50 flooded links that matter most for older people and children, only 1 is among the 50 busiest.** The links vulnerable groups depend on are short alleys and paths to a clinic or school, which traffic counts never see.

## What is real and what is synthetic

| Layer | Source | Real or synthetic |
| --- | --- | --- |
| Streets, alleys and footpaths (22,731 links) | OpenStreetMap via Overpass | Real |
| Clinics, hospitals, schools, markets, stations (796 places) | OpenStreetMap | Real |
| Building footprints (120,000+) used to place people | OpenStreetMap | Real |
| River centreline | OpenStreetMap | Real |
| Age cohorts per grid cell | Built from buildings and district totals, see Layer 1 | **Synthetic** |
| Flood zone | Distance to the river plus a smooth random field | **Synthetic** |

Both synthetic layers are stand-ins with a documented plug-in format. The numbers in the demo show how the method behaves; they are not findings about Jakarta.

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

**Archetypes (optional).** The cells are also grouped into four mobility justice archetypes with k-means on four variables, following Haxhija et al. (2024): walking accessibility, services within 800 m, frequency of use (a synthetic proxy here), and flood exposure. Switch the hexagons to "archetype" in step 1 of the demo.

### 2. Network criticality

1. **Network.** Walkable OSM ways are split at junctions into links (motorways excluded). Each cell and each destination snaps to its nearest junction.
2. **Access per cohort.** For each type of place (health, school, market, transit) a multi-source shortest-path search gives the walking distance from every cell. A cohort's access is a gravity score: the sum over place types of *need* times *exp(-distance / decay)*. Need and decay differ by cohort:

   | Cohort | Health | School | Market | Transit | Walk decay |
   | --- | --- | --- | --- | --- | --- |
   | Older people | 0.50 | 0 | 0.35 | 0.15 | 350 m |
   | Children | 0.20 | 0.65 | 0.05 | 0.10 | 500 m |
   | Working age | 0.10 | 0.10 | 0.30 | 0.50 | 900 m |

3. **Cut one link at a time.** Every link in the flood zone (3,344 here) is removed on its own and access is recomputed. Links not on any shortest-path tree are skipped, since removing them changes nothing. The loss is counted in *people's worth of access*: each cell's population times the share of its access that is gone.
4. **Weight by vulnerability.** The weighted criticality of a link is 3 x older people's loss + 2 x children's loss + 1 x working-age loss. Two baselines rank the same links: *everyone the same* (one need profile, one walk distance, no weights) and *how busy* (population-weighted shortest-path flow through the link, from 400 sampled origins).
5. **Compare the rankings.** Top-50 overlap between weighted and busiest is 1 of 50; between weighted and everyone-the-same it is 39 of 50. So most of the gap comes from looking at access at all, and the weights then reorder the top.

All weights live at the top of `scripts/build_data.py`, in one place, so a team can set them from its own survey or trace data.

### 3. Policy screening

With every flooded link cut at once, the model searches for three kinds of low-regret fix and measures the access each brings back per cohort:

| Fix | How it is found | Rough cost |
| --- | --- | --- |
| Raise one street segment | the flooded links on the shortest route from a cut-off hot spot to its nearest dry clinic, school or market (up to 700 m) | high |
| Raised footbridge | a new walkway between two dry junctions under 300 m apart that the flood pushes far apart on the network | medium |
| Pop-up clinic shuttle | a flood-day minibus from a dry pickup point to the nearest dry clinic, counted as 250 m of walking | low |

All combinations are precomputed so the page can toggle them freely. The best quick win is the fix with the most weighted access restored per unit of rough cost.

## Run it on another city

```bash
pip install -r requirements.txt          # numpy and scipy
# edit config.json: name, bbox [south, west, north, east], river_names, districts
python scripts/fetch_osm.py              # streets, places, buildings into data/raw/
python scripts/build_data.py             # writes web/data/city.json (about 1 minute)
python -m http.server -d web 8000        # open http://localhost:8000
```

Options:

- `--cohorts grid.csv` uses your cohort map instead of the synthetic one.
- A real flood layer: set `"flood": {"source": "file", "path": "flood.geojson"}` in `config.json`, with polygons of flood-prone areas. Links and cells inside the polygons count as flooded.
- `districts` can be left out of a real run if you pass `--cohorts`.

`web/index.html` is one static page (MapLibre GL 4.7.1 and deck.gl 9.0.38 from unpkg, basemap from OpenFreeMap). `?step=0` to `4` opens a step, `?cohort=children` and `?rank=traffic` set the view, and `?embed` shows a tap-to-load button first (phones get it too). GitHub Actions deploys `web/` to Pages on every push to `main`.

## Limits

- The cohort grid and the flood zone are synthetic, so no number here describes real people.
- Walking only. Motorbike, angkot and KRL access during floods would need their own networks and GTFS feeds.
- One link at a time misses combinations; the flood scenario covers the all-at-once case, but not the order in which links fail.
- Need weights and walking distances are assumptions, set in one place to be replaced with survey or trace estimates.

## Layout

```
config.json                 the city: bbox, river, district totals, grid size
scripts/fetch_osm.py        Overpass downloads into data/raw/
scripts/build_data.py       cohort grid, network, criticality, fixes, archetypes, JSON export
data/raw/                   OSM extracts for the example city
data/cohort_grid_synthetic.csv   the stand-in cohort grid, in the plug-in format
web/index.html              the demo
web/data/city.json          model output read by the demo
assets/card.mp4             short clip of the demo
```

MIT licensed. Map data © OpenStreetMap contributors (ODbL).
