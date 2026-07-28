"""
micro/geometry.py

Channel network geometry and laminar hydraulics for microchannel heat sinks.

At microscale the mask inverts from the macro model — silicon is solid and etched
channels are the void. Supports straight parallel channels, branching
Murray-tree networks, and arbitrary freeform segment lists from the Studio's
etch tools. Hydraulics are exact for laminar flow: each channel is a linear
resistor, so any network is a resistor circuit solved by nodal analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from . import correlations as corr
from .materials import Coolant


@dataclass
class Segment:
    """One straight channel run, w (width) x H (depth), between two node ids."""
    a: int
    b: int
    p0: tuple
    p1: tuple
    w: float
    H: float
    level: int = 0
    Rmu: float = 0.0     # resistance-per-viscosity override [Pa·s/m³ per Pa·s];
                         # 0 = derive from geometry. Used by virtual manifold links.

    @property
    def length(self) -> float:
        return float(np.hypot(self.p1[0] - self.p0[0], self.p1[1] - self.p0[1]))

    @property
    def area(self) -> float:
        return self.w * self.H

    @property
    def Dh(self) -> float:
        return corr.hydraulic_diameter(self.w, self.H)

    def resistance(self, mu: float) -> float:
        """Laminar hydraulic resistance R = dP/Q [Pa.s/m^3]."""
        if self.Rmu > 0.0:
            return self.Rmu * mu
        fRe = corr.fRe_fanning(corr.aspect_ratio(self.w, self.H))
        L = max(self.length, 1e-9)
        Dh = max(self.Dh, 1e-12)
        return 2.0 * fRe * mu * L / (Dh * Dh * self.area)


@dataclass
class Network:
    nodes: list
    segments: list
    inlet: int
    outlet: int
    die_L: float = 0.0
    die_W: float = 0.0
    meta: dict = dc_field(default_factory=dict)

    def solve_flow(self, coolant: Coolant, Q_total: float) -> dict:
        """Inject Q_total [m^3/s] at the inlet, ground the outlet, solve nodal
        pressures, return per-segment flow + total dP. Linear, single solve."""
        n = len(self.nodes)
        G = np.zeros((n, n))
        R = [s.resistance(coolant.mu) for s in self.segments]
        for s, r in zip(self.segments, R):
            g = 1.0 / r
            G[s.a, s.a] += g; G[s.b, s.b] += g
            G[s.a, s.b] -= g; G[s.b, s.a] -= g
        b = np.zeros(n)
        b[self.inlet] = Q_total
        b[self.outlet] -= Q_total
        keep = [i for i in range(n) if i != self.outlet]   # ground outlet (p=0)
        pr = np.linalg.solve(G[np.ix_(keep, keep)], b[keep])
        p = np.zeros(n); p[keep] = pr
        seg_flow = [(p[s.a] - p[s.b]) / r for s, r in zip(self.segments, R)]
        dP = float(p[self.inlet] - p[self.outlet])
        lvl = max((s.level for s in self.segments), default=0)
        leaf_q = np.array([abs(q) for q, s in zip(seg_flow, self.segments) if s.level == lvl])
        cv = float(np.std(leaf_q) / np.mean(leaf_q)) if leaf_q.size and np.mean(leaf_q) > 0 else 0.0
        return {"pressure": p, "seg_flow": np.array(seg_flow), "dP": dP,
                "pumping_power": dP * Q_total, "maldistribution_cv": cv,
                "n_through": int(leaf_q.size)}


class _NodeStore:
    """Snaps coordinates to 0.1 um so shared endpoints merge into one node."""
    def __init__(self):
        self.nodes = []
        self._idx = {}

    def get(self, x, y) -> int:
        key = (round(x * 1e7), round(y * 1e7))
        if key not in self._idx:
            self._idx[key] = len(self.nodes)
            self.nodes.append((x, y))
        return self._idx[key]


def _dist_to_segment(px, py, p0, p1):
    """Vectorised point-to-segment distance (used by rasterize)."""
    x0, y0 = p0; x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy
    if L2 < 1e-20:
        return np.hypot(px - x0, py - y0)
    t = np.clip(((px - x0) * dx + (py - y0) * dy) / L2, 0.0, 1.0)
    return np.hypot(px - (x0 + t * dx), py - (y0 + t * dy))


def straight_parallel(n_ch=20, w=100e-6, H=300e-6, L=10e-3, pitch=None,
                      header_w=600e-6) -> Network:
    """N parallel channels fed/drained by inlet+outlet headers (U-type manifold).
    Header resistance is modelled explicitly -- it is what causes maldistribution."""
    if pitch is None:
        pitch = 2 * w
    ns = _NodeStore()
    W = max(n_ch * pitch, pitch)
    ys = (np.arange(n_ch) + 0.5) * pitch + (W - n_ch * pitch) / 2
    segs = []
    in_nodes = [ns.get(0.0, y) for y in ys]
    out_nodes = [ns.get(L, y) for y in ys]
    for yi, (a, b) in enumerate(zip(in_nodes, out_nodes)):
        segs.append(Segment(a, b, (0.0, ys[yi]), (L, ys[yi]), w, H, level=1))
    # header centrelines sit just OFF the die (real manifolds live outside the
    # active area): identical hydraulics (same length/width), but on-die heat is
    # owned by the channels, matching the conjugate reference model.
    xh_in, xh_out = -header_w / 2, L + header_w / 2
    for i in range(n_ch - 1):
        segs.append(Segment(in_nodes[i], in_nodes[i + 1], (xh_in, ys[i]), (xh_in, ys[i + 1]), header_w, H, level=0))
        segs.append(Segment(out_nodes[i], out_nodes[i + 1], (xh_out, ys[i]), (xh_out, ys[i + 1]), header_w, H, level=0))
    net = Network(ns.nodes, segs, inlet=in_nodes[0], outlet=out_nodes[0], die_L=L, die_W=W)
    net.meta = {"kind": "straight_parallel", "n_ch": n_ch, "w": w, "H": H, "L": L, "pitch": pitch}
    return net


def murray_tree(levels=4, w_trunk=400e-6, H=300e-6, die_L=12e-3, die_W=12e-3,
                exponent=3.0) -> Network:
    """Branching distribution tree + mirrored collection tree (the 'vein' network).
    Child widths follow Murray's law (symmetric split: w_child = w_parent/2^(1/n)).
    CONNECTED: inlet -> dist tree -> through-channels -> collection tree -> outlet."""
    ns = _NodeStore()
    segs = []
    mf = 0.5 ** (1.0 / exponent)
    seg_len = (die_L / 2) / (levels + 1)
    nbranch = max(1, levels - 1)

    def grow(x, y, nid, x_dir, w, lvl, span, leaves):
        if lvl >= levels:
            leaves.append((x, y, nid)); return
        x2 = x + x_dir * seg_len * (0.72 ** (lvl - 1))
        cw = w * mf
        for s in (+1, -1):
            yc = y + s * span / 2
            cid = ns.get(x2, yc)
            segs.append(Segment(nid, cid, (x, y), (x2, yc), cw, H, level=lvl))
            grow(x2, yc, cid, x_dir, cw, lvl + 1, span / 2, leaves)

    inlet = ns.get(0.0, die_W / 2)
    rid = ns.get(seg_len, die_W / 2)
    segs.append(Segment(inlet, rid, (0.0, die_W / 2), (seg_len, die_W / 2), w_trunk, H, level=0))
    dist_leaves = []
    grow(seg_len, die_W / 2, rid, +1, w_trunk, 1, die_W / 2, dist_leaves)

    outlet = ns.get(die_L, die_W / 2)
    lid = ns.get(die_L - seg_len, die_W / 2)
    segs.append(Segment(outlet, lid, (die_L, die_W / 2), (die_L - seg_len, die_W / 2), w_trunk, H, level=0))
    coll_leaves = []
    grow(die_L - seg_len, die_W / 2, lid, -1, w_trunk, 1, die_W / 2, coll_leaves)

    dist_leaves.sort(key=lambda p: p[1])
    coll_leaves.sort(key=lambda p: p[1])
    w_leaf = w_trunk * (mf ** nbranch)
    for (xd, yd, ad), (xc, yc, bc) in zip(dist_leaves, coll_leaves):
        segs.append(Segment(ad, bc, (xd, yd), (xc, yc), w_leaf, H, level=levels + 1))

    net = Network(ns.nodes, segs, inlet=inlet, outlet=outlet, die_L=die_L, die_W=die_W)
    net.meta = {"kind": "murray_tree", "levels": levels, "exponent": exponent,
                "w_trunk": w_trunk, "H": H, "n_through": len(dist_leaves)}
    return net


def rasterize(net: Network, grid_n=192, seg_flow=None, coolant=None, shape=None):
    """Burn the network onto a grid: solid silicon (1) with channels carved (0),
    plus a per-cell velocity field along the local segment. Mask is INVERTED vs the
    macro model (pivot 3). Arrays are [row=y, col=x] per physics/field.py.

    `shape=(ny, nx)` rasterizes onto a rectangular grid matching the die aspect;
    default keeps the legacy square grid_n × grid_n."""
    ny, nx = shape if shape is not None else (grid_n, grid_n)
    L = net.die_L or 1.0
    W = net.die_W or 1.0
    mask = np.ones((ny, nx), dtype=np.uint8)
    vx = np.zeros((ny, nx)); vy = np.zeros((ny, nx)); chan = np.zeros((ny, nx), np.int32)
    xs = (np.arange(nx) + 0.5) / nx * L
    ys = (np.arange(ny) + 0.5) / ny * W
    gx, gy = np.meshgrid(xs, ys)
    for si, s in enumerate(net.segments):
        if s.w <= 0:                       # virtual manifold links never rasterize
            continue
        inside = _dist_to_segment(gx, gy, s.p0, s.p1) <= (s.w / 2)
        mask[inside] = 0
        chan[inside] = si + 1
        if seg_flow is not None and coolant is not None:
            u = abs(seg_flow[si]) / max(s.area, 1e-15)
            ux, uy = (s.p1[0] - s.p0[0]), (s.p1[1] - s.p0[1])
            nrm = np.hypot(ux, uy) or 1.0
            sign = np.sign(seg_flow[si]) or 1.0
            vx[inside] = sign * u * ux / nrm
            vy[inside] = sign * u * uy / nrm
    return {"mask": mask, "vx": vx, "vy": vy, "channel": chan, "dx": L / nx, "dy": W / ny}


# ── freeform networks (the Studio's etched vein designs) ─────────────────

def _split_at_intersections(raw):
    """Split segments wherever they geometrically cross, so crossing channels
    JOIN hydraulically (etched slots that overlap really do merge). Input/
    output: list of dicts with x0, y0, x1, y1 + carried fields. O(n²) pair
    test — fine for design-scale networks (guarded upstream at 20k segments)."""
    def get(s, k):
        return float(s[k]) if isinstance(s, dict) else float(getattr(s, k))
    n = len(raw)
    cuts = [set() for _ in range(n)]
    P = [(get(s, "x0"), get(s, "y0"), get(s, "x1"), get(s, "y1")) for s in raw]
    for i in range(n):
        x0, y0, x1, y1 = P[i]
        dxi, dyi = x1 - x0, y1 - y0
        for j in range(i + 1, n):
            u0, v0, u1, v1 = P[j]
            dxj, dyj = u1 - u0, v1 - v0
            den = dxi * dyj - dyi * dxj
            if abs(den) < 1e-18:
                continue                        # parallel (colinear overlap merges via rasters anyway)
            t = ((u0 - x0) * dyj - (v0 - y0) * dxj) / den
            s2 = ((u0 - x0) * dyi - (v0 - y0) * dxi) / den
            if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= s2 <= 1 + 1e-9:
                if 1e-6 < t < 1 - 1e-6:
                    cuts[i].add(round(t, 9))
                if 1e-6 < s2 < 1 - 1e-6:
                    cuts[j].add(round(s2, 9))
    out = []
    for i, s in enumerate(raw):
        ts = sorted(cuts[i] | {0.0, 1.0})
        x0, y0, x1, y1 = P[i]
        base = dict(s) if isinstance(s, dict) else {
            "w": get(s, "w"), "depth": getattr(s, "depth", getattr(s, "H", None)),
            "level": int(getattr(s, "level", 0))}
        for a, b in zip(ts[:-1], ts[1:]):
            if b - a < 1e-6:
                continue
            piece = dict(base)
            piece.update({"x0": x0 + (x1 - x0) * a, "y0": y0 + (y1 - y0) * a,
                          "x1": x0 + (x1 - x0) * b, "y1": y0 + (y1 - y0) * b})
            out.append(piece)
    return out



def network_from_segments(segments, die_L, die_W, inlet_edge="left",
                          outlet_edge="right") -> tuple:
    """Build a solvable Network from a raw centreline segment list — the shape the
    Studio's etch/vein tools emit. All lengths in METERS (SI); segments are dicts
    or objects with x0, y0, x1, y1, w, and depth (or H).

    Boundary handling: every node lying within tolerance of the inlet edge is tied
    to one virtual inlet node through a near-zero-resistance link (an external
    plenum/manifold feeding each channel that reaches the edge), and likewise for
    the outlet. This makes both single-trunk trees and open parallel-channel
    fields solvable with the same rule.

    Returns (Network, info) where info reports dead segments (not hydraulically
    connected inlet→outlet; excluded from the solve) and the virtual node ids.
    Raises ValueError if the inlet cannot reach the outlet at all.
    """
    def _get(s, k, default=None):
        if isinstance(s, dict):
            v = s.get(k, default)
        else:
            v = getattr(s, k, default)
        if v is None:
            raise ValueError(f"segment missing required field {k!r}")
        return float(v)

    ns = _NodeStore()
    segs = []
    segments = _split_at_intersections(list(segments))
    for s in segments:
        H = _get(s, "depth", None) if (isinstance(s, dict) and "depth" in s) or hasattr(s, "depth") else None
        if H is None:
            H = _get(s, "H")
        a = ns.get(_get(s, "x0"), _get(s, "y0"))
        b = ns.get(_get(s, "x1"), _get(s, "y1"))
        if a == b:
            continue                                    # zero-length after snapping
        lvl = int(s.get("level", 0)) if isinstance(s, dict) else int(getattr(s, "level", 0))
        segs.append(Segment(a, b, tuple(ns.nodes[a]), tuple(ns.nodes[b]),
                            w=_get(s, "w"), H=H, level=lvl))
    if not segs:
        raise ValueError("no usable segments (all zero-length?)")

    # edge coordinate accessor + tolerance
    tol = 0.02 * max(die_L, die_W)
    def edge_dist(node_xy, edge):
        x, y = node_xy
        return {"left": x, "right": die_L - x, "top": y, "bottom": die_W - y}[edge]

    def edge_nodes(edge, what):
        d = [(edge_dist(ns.nodes[i], edge), i) for i in range(len(ns.nodes))]
        near = [i for dist, i in d if dist <= tol]
        if not near:
            raise ValueError(
                f"no channel reaches the {edge!r} ({what}) edge — extend a "
                f"channel to the die edge so coolant can enter/leave")
        return near

    inlet_nodes = edge_nodes(inlet_edge, "inlet")
    outlet_nodes = [n for n in edge_nodes(outlet_edge, "outlet") if n not in inlet_nodes]
    if not outlet_nodes:
        raise ValueError(f"no distinct nodes near the {outlet_edge!r} (outlet) edge")

    # virtual plenum nodes tied by near-zero-resistance links (w<0 => never rasterized)
    v_in = len(ns.nodes); ns.nodes.append((-tol, die_W / 2))
    v_out = len(ns.nodes); ns.nodes.append((die_L + tol, die_W / 2))
    w_ref = max(s.w for s in segs); H_ref = max(s.H for s in segs)
    for n in inlet_nodes:
        segs.append(Segment(v_in, n, ns.nodes[v_in], ns.nodes[n], w=-1.0, H=H_ref, level=-1))
    for n in outlet_nodes:
        segs.append(Segment(n, v_out, ns.nodes[n], ns.nodes[v_out], w=-1.0, H=H_ref, level=-1))

    # connectivity from the inlet; drop segments the flow can never reach
    adj = {}
    for i, s in enumerate(segs):
        adj.setdefault(s.a, []).append((s.b, i))
        adj.setdefault(s.b, []).append((s.a, i))
    seen = {v_in}
    stack = [v_in]
    reachable_segs = set()
    while stack:
        n = stack.pop()
        for m, si in adj.get(n, []):
            reachable_segs.add(si)
            if m not in seen:
                seen.add(m); stack.append(m)
    if v_out not in seen:
        raise ValueError("network does not connect the inlet edge to the outlet edge")
    dead = len(segs) - len(reachable_segs)
    live = [segs[i] for i in sorted(reachable_segs)]

    # virtual links: resistance = 1e-6 × the least-resistive real segment (ratio is
    # viscosity-independent because laminar R is linear in mu)
    real_Rmu = [s.resistance(1.0) for s in live if s.w > 0]
    if not real_Rmu:
        raise ValueError("network has no real (positive-width) segments")
    Rmu_virtual = min(real_Rmu) * 1e-6
    for s in live:
        if s.w <= 0:
            s.Rmu = Rmu_virtual

    # compact node ids to the live sub-network
    remap = {}
    nodes = []
    for s in live:
        for n in (s.a, s.b):
            if n not in remap:
                remap[n] = len(nodes); nodes.append(ns.nodes[n])
        s.a = remap[s.a]; s.b = remap[s.b]
    net = Network(nodes, live, inlet=remap[v_in], outlet=remap[v_out],
                  die_L=die_L, die_W=die_W)
    net.meta = {"kind": "from_segments", "n_segments": len(live), "n_dead": dead,
                "virtual_inlet": remap[v_in], "virtual_outlet": remap[v_out]}
    return net, {"n_dead": dead, "w_ref": w_ref}
