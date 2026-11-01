"""Download the open inputs for one city from OpenStreetMap (Overpass API).

Reads the bounding box from config.json and writes three files to data/raw/:
  roads.json         walkable streets, alleys and paths (way geometry)
  places.json        rivers/canals plus essential destinations
  buildings.json     building footprint centroids [[lon, lat], ...], used to place people

Usage:  python scripts/fetch_osm.py [config.json]
"""
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://overpass-api.de/api/interpreter"

QUERIES = {
    "roads": """way[highway][highway!~"^(motorway|motorway_link|construction|proposed|raceway|bus_guideway)$"];
out geom qt;""",
    "places": """(way[waterway~"^(river|canal)$"];);out geom qt;
(nwr[amenity~"^(clinic|hospital|doctors|school|kindergarten|marketplace)$"];
 nwr[healthcare~"^(clinic|centre|hospital)$"];
 nwr[shop=supermarket];
 nwr[railway~"^(station|halt)$"];
 nwr[public_transport=station];);
out center qt;""",
    "buildings": """way[building];out center qt;""",
}


def fetch(query, bbox):
    s, w, n, e = bbox
    q = f"[out:json][timeout:180][bbox:{s},{w},{n},{e}];{query}"
    data = urllib.parse.urlencode({"data": q}).encode()
    req = urllib.request.Request(ENDPOINT, data=data, headers={
        "User-Agent": "mobility-justice-resilience/1.0 (open-data research demo)",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=240) as r:
        return json.load(r)


def main():
    cfg = json.loads(Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "config.json").read_text())
    out = ROOT / "data" / "raw"
    out.mkdir(parents=True, exist_ok=True)
    for name, q in QUERIES.items():
        d = fetch(q, cfg["bbox"])
        if name == "buildings":  # keep centroids only, rounded to ~1 m
            d = {"elements": [[round(e["center"]["lon"], 5), round(e["center"]["lat"], 5)]
                              for e in d["elements"] if "center" in e]}
        (out / f"{name}.json").write_text(json.dumps(d, separators=(",", ":")))
        print(f"{name}: {len(d['elements'])} elements")
        time.sleep(2)


if __name__ == "__main__":
    main()
