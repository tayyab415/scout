"""A walker that only uses what it has seen.

The body plans paths on the floor it has seen so far, looks where it walks, and stops early if
something it is looking for comes into view. Movement here is kinematic (1 m/s); the physics body
(Motivo) replays the same routes in `scout.body`.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .memory.explored import Memory
from .world.grid import descend, geodesic, path_length

STEP_M = 0.5        # look every half metre
SPEED = 1.0         # m/s
LOOK_AROUND_S = 2.0
REACH_M = 1.2       # close enough to an object to count as "at it"


@dataclass
class Body:
    xy: np.ndarray
    yaw: float = 0.0
    walked: float = 0.0
    t: float = 0.0
    route: list[np.ndarray] = field(default_factory=list)


def known_distances(memory: Memory, body: Body) -> np.ndarray:
    """Walking distance from the body to every cell of seen floor."""
    g = memory.layout.grid
    return geodesic(memory.known_free, [g.cell(body.xy)], g.res)


def goal_region(memory: Memory, xy, radius: float = REACH_M) -> list[tuple[int, int]]:
    """Seen floor cells within reach of a point."""
    g = memory.layout.grid
    ci, cj = g.cell(xy)
    r = int(math.ceil(radius / g.res))
    kf = memory.known_free
    out = []
    for i in range(ci - r, ci + r + 1):
        for j in range(cj - r, cj + r + 1):
            if g.inside((i, j)) and kf[i, j] and math.hypot(i - ci, j - cj) * g.res <= radius:
                out.append((i, j))
    return out


def walk(memory: Memory, body: Body, targets: list[tuple[int, int]], budget_m: float,
         stop_when=None) -> str:
    """Walk to the nearest target cell over seen floor. Returns 'arrived', 'stopped', 'budget' or 'no_path'."""
    g = memory.layout.grid
    if not targets:
        return "no_path"
    dist = geodesic(memory.known_free, targets, g.res)
    path = descend(dist, g.cell(body.xy))
    if not path:
        return "no_path"
    pts = np.array([g.pos(c) for c in path])
    seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total = seg[-1]
    s = 0.0
    while s < total - 1e-9:
        s_next = min(s + STEP_M, total)
        if body.walked + (s_next - s) > budget_m:
            return "budget"
        k = int(np.searchsorted(seg, s_next))
        xy = pts[min(k, len(pts) - 1)]
        d = xy - body.xy
        if np.linalg.norm(d) > 1e-6:
            body.yaw = math.atan2(d[1], d[0])
        body.walked += s_next - s
        body.t += (s_next - s) / SPEED
        body.xy = xy.copy()
        body.route.append(body.xy.copy())
        memory.look(body.xy, body.yaw, body.t)
        s = s_next
        if stop_when is not None and stop_when():
            return "stopped"
    return "arrived"


def look_around(memory: Memory, body: Body) -> None:
    memory.look_around(body.xy, body.t)
    body.t += LOOK_AROUND_S


def frontier_table(memory: Memory, body: Body):
    """Reachable frontiers with walking distance, plus the distance map used."""
    dist = known_distances(memory, body)
    out = []
    for f in memory.frontiers():
        d = float(dist[f.cell])
        if np.isfinite(d):
            out.append((f, d))
    return out, dist


def path_len_full(layout, start_xy, goal_type: str) -> float:
    """Shortest walk from start to within reach of any instance of goal_type, on the full map."""
    g = layout.grid
    src = []
    for o in layout.objects_of(goal_type):
        ci, cj = g.cell(o.xy)
        r = int(math.ceil(REACH_M / g.res))
        for i in range(ci - r, ci + r + 1):
            for j in range(cj - r, cj + r + 1):
                if g.is_free((i, j)) and math.hypot(i - ci, j - cj) * g.res <= REACH_M:
                    src.append((i, j))
    if not src:
        return float("inf")
    d = geodesic(g.free, src, g.res)
    return float(d[g.cell(start_xy)])
