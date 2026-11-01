"""Vulnerability-weighted network criticality for any city.

Three layers:
  1. cohort population grid  (pluggable: --cohorts my_grid.csv, else a synthetic stand-in)
  2. network criticality      (OSM walking network, essential destinations, flood hazard)
  3. policy screening         (raised segment, footbridge, pop-up shuttle)

Usage:
  python scripts/fetch_osm.py                      # once per city, writes data/raw/
  python scripts/build_data.py                     # writes web/data/city.json
  python scripts/build_data.py --cohorts grid.csv  # use your own cohort map

Only numpy and scipy are needed.
"""
import argparse
import csv
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
RNG = np.random.default_rng(7)

COHORTS = ["elderly", "children", "working"]
DEST_TYPES = ["health", "school", "market", "transit"]

# How much each cohort needs each kind of place (rows sum to 1), how far they
# comfortably walk (decay distance, m), and how much weight a lost trip carries.
# These are the method's assumptions; change them in one place.
NEED = {
    "elderly":  {"health": 0.50, "school": 0.00, "market": 0.35, "transit": 0.15},
    "children": {"health": 0.20, "school": 0.65, "market": 0.05, "transit": 0.10},
    "working":  {"health": 0.10, "school": 0.10, "market": 0.30, "transit": 0.50},
}
DECAY_M = {"elderly": 350.0, "children": 500.0, "working": 900.0}
VULN_WEIGHT = {"elderly": 3.0, "children": 2.0, "working": 1.0}
# The plain baseline treats everyone the same: one need profile, one walk distance.
PLAIN_NEED = {t: 0.25 for t in DEST_TYPES}
PLAIN_DECAY = 800.0
BIG = 1e7


# ---------------------------------------------------------------- geometry
class Proj:
    """Local equirectangular projection in metres around the study area centre."""

    def __init__(self, bbox):
        s, w, n, e = bbox
        self.lat0, self.lon0 = (s + n) / 2, (w + e) / 2
        self.kx = 111320.0 * math.cos(math.radians(self.lat0))
        self.ky = 110540.0

    def xy(self, lon, lat):
        return (np.asarray(lon) - self.lon0) * self.kx, (np.asarray(lat) - self.lat0) * self.ky

    def ll(self, x, y):
        return np.asarray(x) / self.kx + self.lon0, np.asarray(y) / self.ky + self.lat0


def seg_point_dist(px, py, ax, ay, bx, by):
    """Distance from points (px, py) to the nearest of many segments a-b."""
    dx, dy = bx - ax, by - ay
    L2 = np.maximum(dx * dx + dy * dy, 1e-9)
    t = np.clip(((px[:, None] - ax) * dx + (py[:, None] - ay) * dy) / L2, 0, 1)
    qx, qy = ax + t * dx, ay + t * dy
    return np.sqrt((px[:, None] - qx) ** 2 + (py[:, None] - qy) ** 2).min(axis=1)


def dist_to_lines(px, py, lines, chunk=4000):
    ax = np.concatenate([l[:-1, 0] for l in lines]); ay = np.concatenate([l[:-1, 1] for l in lines])
    bx = np.concatenate([l[1:, 0] for l in lines]); by = np.concatenate([l[1:, 1] for l in lines])
    out = np.empty(len(px))
    for i in range(0, len(px), chunk):
        out[i:i + chunk] = seg_point_dist(px[i:i + chunk], py[i:i + chunk], ax, ay, bx, by)
    return out


def point_in_polys(px, py, polys):
    """Even-odd ray casting against a list of rings (each an (n, 2) array)."""
    inside = np.zeros(len(px), bool)
    for ring in polys:
        x1, y1 = ring[:, 0], ring[:, 1]
        x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
        for a, b, c, d in zip(x1, y1, x2, y2):
            cross = ((b > py) != (d > py)) & (px < (c - a) * (py - b) / (d - b + 1e-12) + a)
            inside ^= cross
    return inside


def noise_field(scale, n=6):
    """A smooth random field in about [-1, 1], used for the synthetic layers."""
    waves = [(2 * math.pi / (scale * RNG.uniform(0.6, 1.6)), RNG.uniform(0, 2 * math.pi), RNG.uniform(0, 2 * math.pi))
             for _ in range(n)]

    def f(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        return sum(np.sin(k * (x * math.cos(th) + y * math.sin(th)) + ph) for k, th, ph in waves) / math.sqrt(n / 2) / 2
    return f


def smooth_noise(x, y, scale):
    return noise_field(scale)(x, y)


# ---------------------------------------------------------------- inputs
def load_network(raw, P):
    ways = [e for e in json.loads((raw / "roads.json").read_text())["elements"]
            if e["type"] == "way" and "geometry" in e]
    use = defaultdict(int)
    for w in ways:
        for i, nid in enumerate(w["nodes"]):
            use[nid] += 2 if i in (0, len(w["nodes"]) - 1) else 1
    coord = {}
    edges = []  # (u_osm, v_osm, polyline lonlat, highway)
    for w in ways:
        hw = w["tags"].get("highway", "")
        nodes, geom = w["nodes"], w["geometry"]
        for nid, g in zip(nodes, geom):
            coord[nid] = (g["lon"], g["lat"])
        start = 0
        for i in range(1, len(nodes)):
            if use[nodes[i]] >= 2 or i == len(nodes) - 1:
                if nodes[start] != nodes[i]:
                    edges.append((nodes[start], nodes[i], [coord[n] for n in nodes[start:i + 1]], hw))
                start = i
    ids = sorted({u for e in edges for u in e[:2]})
    idx = {n: i for i, n in enumerate(ids)}
    lon = np.array([coord[n][0] for n in ids]); lat = np.array([coord[n][1] for n in ids])
    nx, ny = P.xy(lon, lat)

    best = {}
    for u, v, line, hw in edges:
        a, b = idx[u], idx[v]
        x, y = P.xy([p[0] for p in line], [p[1] for p in line])
        L = float(np.hypot(np.diff(x), np.diff(y)).sum())
        key = (min(a, b), max(a, b))
        if key not in best or L < best[key][2]:
            best[key] = (a, b, L, line, hw)
    E = list(best.values())

    # keep the largest connected piece of the network
    n = len(ids)
    A = csr_matrix((np.ones(len(E)), ([e[0] for e in E], [e[1] for e in E])), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    keep_lab = np.bincount(lab).argmax()
    keep = np.where(lab == keep_lab)[0]
    remap = -np.ones(n, int); remap[keep] = np.arange(len(keep))
    E = [(remap[a], remap[b], L, line, hw) for a, b, L, line, hw in E if lab[a] == keep_lab]
    return {"x": nx[keep], "y": ny[keep], "lon": lon[keep], "lat": lat[keep], "edges": E}


def load_places(raw, P, river_names):
    els = json.loads((raw / "places.json").read_text())["elements"]
    rivers, dests = [], {t: [] for t in DEST_TYPES}
    for e in els:
        t = e.get("tags", {})
        if t.get("waterway") in ("river", "canal") and "geometry" in e:
            if t.get("name") in river_names:
                rivers.append(np.column_stack(P.xy([g["lon"] for g in e["geometry"]], [g["lat"] for g in e["geometry"]])))
            continue
        c = e.get("center") or ({"lon": e["lon"], "lat": e["lat"]} if "lon" in e else None)
        if not c:
            continue
        am, hc = t.get("amenity"), t.get("healthcare")
        kind = ("health" if am in ("clinic", "hospital", "doctors") or hc else
                "school" if am in ("school", "kindergarten") else
                "market" if am == "marketplace" or t.get("shop") == "supermarket" else
                "transit" if t.get("railway") in ("station", "halt") or t.get("public_transport") == "station" or am == "bus_station" else None)
        if kind:
            dests[kind].append((c["lon"], c["lat"], t.get("name", "")))
    return rivers, dests


def hex_cells(P, bbox, r):
    """Pointy-top hex lattice covering the bbox; returns centre x, y in metres."""
    x0, y0 = P.xy(bbox[1], bbox[0]); x1, y1 = P.xy(bbox[3], bbox[2])
    dx, dy = math.sqrt(3) * r, 1.5 * r
    xs, ys = [], []
    for j, y in enumerate(np.arange(y0, y1 + dy, dy)):
        off = dx / 2 if j % 2 else 0
        for x in np.arange(x0 + off, x1 + dx, dx):
            xs.append(x); ys.append(y)
    return np.array(xs), np.array(ys)


def synthetic_cohorts(cfg, P, raw, river_d_fn):
    """Layer 1 stand-in: OSM buildings + district totals, with distance decay.

    Every number here is synthetic. Replace it with a real cohort map via --cohorts.
    """
    r = cfg["cell_radius_m"]
    b = np.array(json.loads((raw / "buildings.json").read_text())["elements"])
    bx, by = P.xy(b[:, 0], b[:, 1])
    cx, cy = hex_cells(P, cfg["bbox"], r)
    _, cell_of = cKDTree(np.column_stack([cx, cy])).query(np.column_stack([bx, by]))
    nb = np.bincount(cell_of, minlength=len(cx)).astype(float)
    live = nb >= 3
    cx, cy, nb = cx[live], cy[live], nb[live]

    D = cfg["districts"]
    dx, dy = P.xy([d["lon"] for d in D], [d["lat"] for d in D])
    _, dist_of = cKDTree(np.column_stack([dx, dy])).query(np.column_stack([cx, cy]))
    area_km2 = 2.598 * r * r / 1e6

    river_d = river_d_fn(cx, cy)
    dens = (nb - nb.mean()) / nb.std()
    noise = smooth_noise(cx, cy, 1500)
    pop = np.zeros(len(cx)); el = np.zeros(len(cx)); ch = np.zeros(len(cx))
    for k, d in enumerate(D):
        m = dist_of == k
        if not m.any():
            continue
        total = d["density"] * area_km2 * m.sum()
        pop[m] = total * nb[m] / nb[m].sum()
        # older riverbank kampungs skew older; the densest blocks skew younger
        we = (1 + 0.8 * np.exp(-river_d[m] / 350)) * (1 + 0.25 * noise[m])
        wc = (1 + 0.25 * np.clip(dens[m], -2, 2)) * (1 - 0.15 * noise[m])
        el[m] = d["elderly"] * we * pop[m].sum() / (we * pop[m]).sum()
        ch[m] = d["children"] * wc * pop[m].sum() / (wc * pop[m]).sum()
    elderly = np.round(pop * el); children = np.round(pop * ch)
    working = np.round(pop) - elderly - children
    no_vehicle = np.clip(0.25 + 0.2 * np.exp(-river_d / 500) + 0.08 * noise, 0.05, 0.8)
    lon, lat = P.ll(cx, cy)
    return [{"cell_id": f"c{i}", "lon": float(lon[i]), "lat": float(lat[i]),
             "elderly": int(elderly[i]), "children": int(children[i]), "working": int(working[i]),
             "no_vehicle_share": round(float(no_vehicle[i]), 3)} for i in range(len(cx))]


def read_cohorts(path):
    """Plug-in point for a real cohort map. See README, 'Bring your own cohort map'."""
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {"cell_id": r["cell_id"], "lon": float(r["lon"]), "lat": float(r["lat"])}
            for c in COHORTS:
                row[c] = float(r[c])
            if r.get("no_vehicle_share"):
                row["no_vehicle_share"] = float(r["no_vehicle_share"])
            rows.append(row)
    return rows


# ---------------------------------------------------------------- the model
class Network:
    def __init__(self, net):
        self.n = len(net["x"])
        E = net["edges"]
        self.u = np.array([e[0] for e in E]); self.v = np.array([e[1] for e in E])
        self.L = np.array([e[2] for e in E])
        self.base = csr_matrix((np.r_[self.L, self.L], (np.r_[self.u, self.v], np.r_[self.v, self.u])),
                               shape=(self.n, self.n))
        self.base.sort_indices()
        # position of each edge's two entries in the csr data array
        ip, ix = self.base.indptr, self.base.indices
        self.pos_uv = np.array([ip[a] + np.searchsorted(ix[ip[a]:ip[a + 1]], b) for a, b in zip(self.u, self.v)])
        self.pos_vu = np.array([ip[b] + np.searchsorted(ix[ip[b]:ip[b + 1]], a) for a, b in zip(self.u, self.v)])

    def graph(self, cut=(), extra=()):
        G = self.base.copy()
        cut = np.asarray(list(cut), int)
        if len(cut):
            G.data[self.pos_uv[cut]] = BIG; G.data[self.pos_vu[cut]] = BIG
        if extra:
            a = np.array([e[0] for e in extra]); b = np.array([e[1] for e in extra]); w = np.array([e[2] for e in extra])
            X = csr_matrix((np.r_[w, w], (np.r_[a, b], np.r_[b, a])), shape=G.shape)
            G = _min_merge(G, X)
        return G


def _min_merge(G, X):
    """Element-wise min of two sparse graphs, treating missing entries as no edge."""
    G = G.tolil(); X = X.tocoo()
    for a, b, w in zip(X.row, X.col, X.data):
        if G[a, b] == 0 or w < G[a, b]:
            G[a, b] = w
    return G.tocsr()


class Access:
    """Gravity-style access per cell and cohort: sum over place types of need x exp(-d / decay)."""

    def __init__(self, net, cells, dest_nodes, cell_node, cell_snap):
        self.net, self.dest_nodes = net, dest_nodes
        self.cell_node, self.cell_snap = cell_node, cell_snap
        self.pop = {c: np.array([r[c] for r in cells], float) for c in COHORTS}
        self.pop_all = sum(self.pop.values())
        self.base_d = self.dists(net.graph())
        self.base = self.scores(self.base_d)

    def dists(self, G, extra_by_type=None):
        out = {}
        for t in DEST_TYPES:
            g = G
            if extra_by_type and extra_by_type.get(t):
                g = _min_merge(G, csr_matrix(
                    (np.r_[[e[2] for e in extra_by_type[t]] * 2],
                     (np.r_[[e[0] for e in extra_by_type[t]], [e[1] for e in extra_by_type[t]]],
                      np.r_[[e[1] for e in extra_by_type[t]], [e[0] for e in extra_by_type[t]]])),
                    shape=G.shape))
            out[t] = dijkstra(g, directed=False, indices=self.dest_nodes[t], min_only=True)
        return out

    def scores(self, d):
        dc = {t: np.minimum(d[t][self.cell_node] + self.cell_snap, BIG) for t in DEST_TYPES}
        s = {c: sum(NEED[c][t] * np.exp(-dc[t] / DECAY_M[c]) for t in DEST_TYPES) for c in COHORTS}
        s["plain"] = sum(PLAIN_NEED[t] * np.exp(-dc[t] / PLAIN_DECAY) for t in DEST_TYPES)
        return s

    def loss(self, s):
        """Access lost per cell, in people: population x share of their access that is gone."""
        out = {}
        for c in COHORTS:
            out[c] = self.pop[c] * np.clip(1 - s[c] / np.maximum(self.base[c], 1e-12), 0, 1)
        out["plain"] = self.pop_all * np.clip(1 - s["plain"] / np.maximum(self.base["plain"], 1e-12), 0, 1)
        return out


def weighted(loss_tot):
    return sum(VULN_WEIGHT[c] * loss_tot[c] for c in COHORTS)


def traffic(net, node_w, n_src=400):
    """Busyness proxy: population-weighted shortest-path flow through each link (sampled)."""
    src = RNG.choice(np.where(node_w > 0)[0], size=min(n_src, (node_w > 0).sum()), replace=False,
                     p=node_w[node_w > 0] / node_w.sum())
    eid = {}
    for k, (a, b) in enumerate(zip(net.u, net.v)):
        eid[(a, b)] = k; eid[(b, a)] = k
    flow = np.zeros(len(net.u))
    G = net.graph()
    D, Pr = dijkstra(G, directed=False, indices=src, return_predecessors=True)
    for s_i in range(len(src)):
        d, pr = D[s_i], Pr[s_i]
        order = np.argsort(-d)
        sub = node_w.copy()
        for x in order:
            p = pr[x]
            if p >= 0 and np.isfinite(d[x]):
                sub[p] += sub[x]
                flow[eid[(p, x)]] += sub[x]
    return flow


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--cohorts", help="CSV with cell_id, lon, lat, elderly, children, working[, no_vehicle_share]")
    ap.add_argument("--out", default=str(ROOT / "web" / "data" / "city.json"))
    args = ap.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    raw = ROOT / "data" / "raw"
    P = Proj(cfg["bbox"])

    print("network ...")
    netd = load_network(raw, P)
    net = Network(netd)
    nx, ny = netd["x"], netd["y"]
    rivers, dests = load_places(raw, P, cfg["river_names"])
    river_d = lambda x, y: dist_to_lines(np.asarray(x, float), np.asarray(y, float), rivers)
    print(f"  {net.n} junctions, {len(net.u)} links, {sum(map(len, dests.values()))} places, {len(rivers)} river pieces")

    # Layer 1: cohort grid
    cells = read_cohorts(args.cohorts) if args.cohorts else synthetic_cohorts(cfg, P, raw, river_d)
    cohort_source = "your file" if args.cohorts else "synthetic"
    if not args.cohorts:  # write the stand-in out in the plug-in format, as a template
        with open(ROOT / "data" / "cohort_grid_synthetic.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["cell_id", "lon", "lat"] + COHORTS + ["no_vehicle_share"])
            w.writeheader()
            w.writerows({**c, "lon": round(c["lon"], 6), "lat": round(c["lat"], 6)} for c in cells)
    cx, cy = P.xy([c["lon"] for c in cells], [c["lat"] for c in cells])
    tree = cKDTree(np.column_stack([nx, ny]))
    cell_snap, cell_node = tree.query(np.column_stack([cx, cy]))
    print(f"  {len(cells)} cells, {sum(c[k] for c in cells for k in COHORTS):,.0f} people ({cohort_source})")

    dest_nodes, dest_out = {}, []
    for t in DEST_TYPES:
        px, py = P.xy([d[0] for d in dests[t]], [d[1] for d in dests[t]])
        _, nn = tree.query(np.column_stack([px, py]))
        dest_nodes[t] = np.unique(nn)
        dest_out += [[t, round(d[0], 5), round(d[1], 5), d[2]] for d in dests[t]]

    # Flood hazard
    f = cfg["flood"]
    mx = np.array([(nx[a] + nx[b]) / 2 for a, b in zip(net.u, net.v)])
    my = np.array([(ny[a] + ny[b]) / 2 for a, b in zip(net.u, net.v)])
    if f["source"] == "synthetic":
        field = noise_field(900)

        def depth(x, y):
            return (f["band_m"] - river_d(x, y)) / f["band_m"] + f["depth_noise"] * field(x, y)
        flooded = depth(mx, my) > 0
        cell_depth = depth(cx, cy)
        cell_flooded = cell_depth > 0
        node_wet = depth(nx, ny) > 0
    else:  # a real hazard layer: GeoJSON polygons of flood-prone areas
        gj = json.loads(Path(f["path"]).read_text())
        rings = []
        for feat in gj["features"]:
            g = feat["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            for poly in polys:
                rx, ry = P.xy([p[0] for p in poly[0]], [p[1] for p in poly[0]])
                rings.append(np.column_stack([rx, ry]))
        flooded = point_in_polys(mx, my, rings)
        cell_flooded = point_in_polys(cx, cy, rings)
        node_wet = point_in_polys(nx, ny, rings)
        cell_depth = cell_flooded.astype(float)
    cand = np.where(flooded)[0]
    print(f"  {len(cand)} links in the flood zone")

    # Layer 2: criticality of each exposed link, one at a time
    acc = Access(net, cells, dest_nodes, cell_node, cell_snap)
    base_d = acc.base_d
    res = {}
    for k, e in enumerate(cand):
        a, b, L = net.u[e], net.v[e], net.L[e]
        hit = [t for t in DEST_TYPES if abs(abs(base_d[t][a] - base_d[t][b]) - L) < 1e-6]
        if not hit:
            continue
        G = net.graph(cut=[e])
        d = dict(base_d)
        for t in hit:
            d[t] = dijkstra(G, directed=False, indices=dest_nodes[t], min_only=True)
        lo = acc.loss(acc.scores(d))
        res[e] = lo
        if k % 500 == 0:
            print(f"  link {k}/{len(cand)}")
    node_w = np.bincount(cell_node, weights=acc.pop_all, minlength=net.n)
    flow = traffic(net, node_w)

    rows = []
    for e in cand:
        lo = res.get(e)
        tot = {c: float(lo[c].sum()) if lo is not None else 0.0 for c in COHORTS + ["plain"]}
        rows.append((e, tot, weighted(tot), flow[e]))
    w_rank = {e: i + 1 for i, (e, *_r) in enumerate(sorted(rows, key=lambda r: -r[2]))}
    p_rank = {e: i + 1 for i, (e, *_r) in enumerate(sorted(rows, key=lambda r: -r[1]["plain"]))}
    t_rank = {e: i + 1 for i, (e, *_r) in enumerate(sorted(rows, key=lambda r: -r[3]))}

    # Flood scenario: all exposed links cut at once, then the policy options
    print("flood scenario and fixes ...")
    flood_d = acc.dists(net.graph(cut=cand))
    flood_loss = acc.loss(acc.scores(flood_d))
    flood_tot = {c: float(flood_loss[c].sum()) for c in COHORTS}

    dry = ~node_wet
    hot = flood_loss["elderly"] * 3 + flood_loss["children"] * 2 + flood_loss["working"]

    def evaluate(fix):
        cut = set(cand) - set(fix.get("raise", []))
        G = net.graph(cut=sorted(cut), extra=fix.get("bridge", []))
        d = acc.dists(G, extra_by_type=fix.get("shuttle"))
        return acc.loss(acc.scores(d))

    def gain(lo):
        return weighted({c: flood_tot[c] - float(lo[c].sum()) for c in COHORTS})

    G_flood = net.graph(cut=cand)
    hot_nodes = np.argsort(-np.bincount(cell_node, weights=hot, minlength=net.n))[:60]

    # Raised segment: raise the flooded links on the shortest dry-to-dry route past a hot spot
    best_raise = None
    dry_idx = np.where(dry)[0]
    dtree = cKDTree(np.column_stack([nx[dry_idx], ny[dry_idx]]))
    eid = {}
    for k, (a, b) in enumerate(zip(net.u, net.v)):
        eid[(a, b)] = k; eid[(b, a)] = k
    D0, Pr0 = dijkstra(net.graph(), directed=False, indices=hot_nodes[:25], return_predecessors=True)
    for i, h in enumerate(hot_nodes[:25]):
        # walk the intact shortest-path tree from the hot spot to the nearest dry clinic, school or market
        targets = np.concatenate([dest_nodes["health"], dest_nodes["school"], dest_nodes["market"]])
        targets = targets[dry[targets]]
        t = targets[np.argmin(D0[i][targets])]
        path, x = [], t
        while x != h and x >= 0:
            p = Pr0[i][x]
            if p < 0:
                break
            path.append(eid[(p, x)]); x = p
        wet_links = [e for e in path if flooded[e]]
        length = float(net.L[wet_links].sum())
        if not wet_links or length > 700:
            continue
        lo = evaluate({"raise": wet_links})
        g = gain(lo)
        if best_raise is None or g / max(length, 50) > best_raise[0] / max(best_raise[2], 50):
            best_raise = (g, wet_links, length, lo)

    # Footbridge: a new raised walkway between two dry junctions that the flood pushes far apart
    best_bridge = None
    dF = dijkstra(G_flood, directed=False, indices=hot_nodes[:15])
    for i, h in enumerate(hot_nodes[:15]):
        near = dtree.query_ball_point([nx[h], ny[h]], 350)
        for j in near:
            a = dry_idx[j]
            if a == h:
                continue
            # pair the dry junction nearest the hot spot with a dry one across the flooded stretch
            far = dtree.query_ball_point([nx[a], ny[a]], 300)
            for k in far[:: max(1, len(far) // 40)]:
                b = dry_idx[k]
                s = math.hypot(nx[a] - nx[b], ny[a] - ny[b])
                if s < 80:
                    continue
                if dF[i][b] < 4 * s + dF[i][a]:
                    continue
                cand_b = (a, b, s)
                lo = evaluate({"bridge": [cand_b]})
                g = gain(lo)
                if best_bridge is None or g > best_bridge[0]:
                    best_bridge = (g, cand_b, s, lo)
            break

    # Pop-up shuttle: a flood-day pickup point to the nearest dry clinic, counted as 250 m of walking
    best_shuttle = None
    for h in hot_nodes[:20]:
        j = dtree.query([nx[h], ny[h]])[1]
        a = dry_idx[j]
        clinics = dest_nodes["health"][dry[dest_nodes["health"]]]
        dd = np.hypot(nx[clinics] - nx[a], ny[clinics] - ny[a])
        b = clinics[np.argmin(np.where(dd > 300, dd, BIG))]
        lo = evaluate({"shuttle": {"health": [(a, b, 250.0)]}})
        g = gain(lo)
        if best_shuttle is None or g > best_shuttle[0]:
            best_shuttle = (g, (a, b), float(np.hypot(nx[a] - nx[b], ny[a] - ny[b])), lo)

    fixes = []
    if best_raise:
        fixes.append({"id": "raise", "name": "Raise one street segment", "cost": "high",
                      "what": f"Lift {best_raise[2]:.0f} m of street above flood level on the route from a cut-off neighbourhood to its nearest services",
                      "spec": {"raise": [int(e) for e in best_raise[1]]},
                      "path": [[[round(float(p[0]), 5), round(float(p[1]), 5)] for p in netd["edges"][e][3]] for e in best_raise[1]]})
    if best_bridge:
        a, b, s = best_bridge[1]
        fixes.append({"id": "bridge", "name": "Build a raised footbridge", "cost": "medium",
                      "what": f"A {s:.0f} m raised walkway between two dry streets that the flood otherwise separates",
                      "spec": {"bridge": [(int(a), int(b), float(s))]},
                      "path": [[[round(float(netd['lon'][a]), 5), round(float(netd['lat'][a]), 5)], [round(float(netd['lon'][b]), 5), round(float(netd['lat'][b]), 5)]]]})
    if best_shuttle:
        a, b = best_shuttle[1]
        fixes.append({"id": "shuttle", "name": "Run a pop-up clinic shuttle", "cost": "low",
                      "what": f"On flood days, a minibus from a dry pickup point to a clinic {best_shuttle[2]:.0f} m away",
                      "spec": {"shuttle": {"health": [(int(a), int(b), 250.0)]}},
                      "path": [[[round(float(netd['lon'][a]), 5), round(float(netd['lat'][a]), 5)], [round(float(netd['lon'][b]), 5), round(float(netd['lat'][b]), 5)]]]})

    # every combination of fixes, so the page can toggle them freely
    combos = {}
    for mask in range(1 << len(fixes)):
        spec = {"raise": [], "bridge": [], "shuttle": {}}
        for i, fx in enumerate(fixes):
            if mask >> i & 1:
                spec["raise"] += fx["spec"].get("raise", []); spec["bridge"] += fx["spec"].get("bridge", [])
                for t, v in fx["spec"].get("shuttle", {}).items():
                    spec["shuttle"].setdefault(t, []).extend(v)
        lo = flood_loss if mask == 0 else evaluate(spec)
        combos[mask] = {"tot": {c: round(float(lo[c].sum()), 1) for c in COHORTS},
                        "cell": {c: np.round(lo[c], 1).tolist() for c in COHORTS}}
    for i, fx in enumerate(fixes):
        t = combos[1 << i]["tot"]
        fx["restored"] = {c: round(flood_tot[c] - t[c], 1) for c in COHORTS}
        fx.pop("spec")
    cost_n = {"low": 1, "medium": 2, "high": 4}
    if fixes:
        quick = max(fixes, key=lambda fx: weighted(fx["restored"]) / cost_n[fx["cost"]])
        for fx in fixes:
            fx["quick_win"] = fx is quick

    # Optional: mobility justice archetypes on four variables
    walk = acc.base["plain"]
    avail = np.zeros(len(cells))
    for t in DEST_TYPES:
        avail += (base_d[t][cell_node] + cell_snap < 800)
    nv = np.array([c.get("no_vehicle_share", 0.3) for c in cells])
    freq = np.clip(0.5 * nv + 0.5 * (acc.pop["elderly"] + acc.pop["children"]) / np.maximum(acc.pop_all, 1) + 0.05 * smooth_noise(cx, cy, 1200), 0, 1)
    expo = np.clip(cell_depth, -1, 2)
    X = np.column_stack([walk, avail, freq, expo])
    Z = (X - X.mean(0)) / X.std(0)
    arche, arche_names = kmeans_archetypes(Z, X)

    # ---------------------------------------------------------- export
    lon_n, lat_n = netd["lon"], netd["lat"]
    links = []
    for e, tot, wv, fl in rows:
        line = netd["edges"][e][3]
        line = [line[0]] + line[1:-1][:: max(1, (len(line) - 2) // 6 or 1)] + [line[-1]] if len(line) > 8 else line
        lo = res.get(e)
        top = []
        if lo is not None and (w_rank[e] <= 300 or t_rank[e] <= 300):
            score = 3 * lo["elderly"] + 2 * lo["children"] + lo["working"]
            for i in np.argsort(-score)[:60]:
                if score[i] < 0.05:
                    break
                top.append([int(i), round(float(lo["elderly"][i]), 2), round(float(lo["children"][i]), 2), round(float(lo["working"][i]), 2)])
        links.append({"id": int(e), "path": [[round(p[0], 5), round(p[1], 5)] for p in line],
                      "len": round(float(net.L[e])), "hw": netd["edges"][e][4],
                      "loss": {c: round(tot[c], 2) for c in COHORTS + ["plain"]},
                      "w": round(wv, 2), "flow": round(float(fl)),
                      "rank": {"weighted": w_rank[e], "plain": p_rank[e], "traffic": t_rank[e]},
                      "cells": top})
    bg = []
    flood_set = set(cand.tolist())
    for k, (a, b, L, line, hw) in enumerate(netd["edges"]):
        if k in flood_set:
            continue
        pts = line if len(line) <= 4 else [line[0], line[len(line) // 2], line[-1]]
        bg.append([[round(p[0], 5), round(p[1], 5)] for p in pts])

    top50 = sorted(rows, key=lambda r: -r[2])[:50]
    overlap = len({r[0] for r in top50} & {e for e, r in t_rank.items() if r <= 50})
    overlap_p = len({r[0] for r in top50} & {e for e, r in p_rank.items() if r <= 50})
    out = {
        "meta": {"name": cfg["name"], "short": cfg["short"], "bbox": cfg["bbox"],
                 "cohort_source": cohort_source, "flood_source": f["source"],
                 "need": NEED, "decay_m": DECAY_M, "vuln_weight": VULN_WEIGHT,
                 "n_links": int(len(net.u)), "n_flooded": int(len(cand)), "cell_radius_m": cfg["cell_radius_m"],
                 "top50_overlap_traffic": overlap, "top50_overlap_plain": overlap_p,
                 "flood_loss": {c: round(flood_tot[c], 1) for c in COHORTS},
                 "pop": {c: int(acc.pop[c].sum()) for c in COHORTS},
                 "archetypes": arche_names},
        "cells": [[round(c["lon"], 5), round(c["lat"], 5), int(c["elderly"]), int(c["children"]), int(c["working"]),
                   int(cell_flooded[i]), int(arche[i]), round(float(acc.base["elderly"][i]), 3)] for i, c in enumerate(cells)],
        "rivers": [[[round(float(v), 5) for v in P.ll(*p)] for p in r[:: max(1, len(r) // 200)]] for r in rivers],
        "dests": dest_out,
        "links": links, "bg": bg, "fixes": fixes, "combos": combos,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, separators=(",", ":")))
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1e6:.1f} MB)")
    print(f"top-50 overlap: weighted vs traffic {overlap}/50, weighted vs plain {overlap_p}/50")
    print("flood loss (people):", out["meta"]["flood_loss"])
    for fx in fixes:
        print(f"  {fx['name']}: restores {fx['restored']}{'  <- quick win' if fx.get('quick_win') else ''}")


def kmeans_archetypes(Z, X, k=4, iters=60):
    """k-means on the four standardised variables, with plain-language names from the centroids."""
    C = Z[RNG.choice(len(Z), k, replace=False)]
    for _ in range(iters):
        lab = np.argmin(((Z[:, None] - C[None]) ** 2).sum(-1), 1)
        C = np.array([Z[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(k)])
    # name clusters from their centres: the most exposed first, the rest by walking access
    order = sorted(range(k), key=lambda j: -C[j][3])
    rest = sorted(order[1:], key=lambda j: -C[j][0])
    label = {order[0]: "Flood-exposed, walk-dependent" if C[order[0]][2] > 0 else "Flood-exposed"}
    for j, n in zip(rest, ["Dry and well served", "Dry, fewer services in reach", "Dry, far from services"]):
        label[j] = n
    names = []
    for j in range(k):
        m = lab == j
        names.append({"name": label[j], "n": int(m.sum()),
                      "walk": round(float(X[m, 0].mean()), 3), "services": round(float(X[m, 1].mean()), 2),
                      "use": round(float(X[m, 2].mean()), 2), "exposure": round(float((X[m, 3] > 0).mean()), 2)})
    return lab, names


if __name__ == "__main__":
    main()
