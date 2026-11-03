"""A floor map the body builds from its own head depth camera. Pure numpy.

The body knows its own position (perfect odometry, disclosed). Everything about the house comes from depth pixels:
floor hits make cells walkable, hits between knee and head height make cells blocked.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..world.grid import descend, geodesic

FLOOR_Z = 0.06        # a hit this close to the floor counts as floor
OBST_Z = (0.15, 1.8)  # hits in this height band block the body
MAX_RANGE = 6.0       # depth beyond this is too noisy to map
BODY_RADIUS = 0.25   # planning clearance; wider closes ProcTHOR doorways (~0.9 m)
CARVE_TO = 0.6       # carve only the first 60% of each eye-to-floor ray (see integrate)


def depth_to_points(depth: np.ndarray, cam_pos, cam_mat, fovy_deg: float, stride: int = 4) -> np.ndarray:
    """MuJoCo depth image (metres along the view axis) -> world points (N, 3).

    MuJoCo cameras look down their -z axis with +y up and +x right."""
    h, w = depth.shape
    f = (h / 2) / math.tan(math.radians(fovy_deg) / 2)
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    d = depth[vs, us]
    ok = (d > 0.05) & (d < MAX_RANGE)
    x = (us[ok] + 0.5 - w / 2) / f * d[ok]
    y = -(vs[ok] + 0.5 - h / 2) / f * d[ok]
    pc = np.stack([x, y, -d[ok]], 1)
    return pc @ np.asarray(cam_mat).reshape(3, 3).T + np.asarray(cam_pos)


@dataclass
class SelfMap:
    origin: np.ndarray
    size_m: float = 40.0
    res: float = 0.1
    floor: np.ndarray = field(init=False)
    obst: np.ndarray = field(init=False)
    visited: np.ndarray = field(init=False)

    def __post_init__(self):
        n = int(round(self.size_m / self.res))
        self.origin = np.asarray(self.origin, float) - self.size_m / 2
        self.floor = np.zeros((n, n), np.int32)
        self.obst = np.zeros((n, n), np.int32)
        self.visited = np.zeros((n, n), bool)

    # --- geometry --------------------------------------------------------------------------------
    def cell(self, xy) -> tuple[int, int]:
        i, j = np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(int)
        return int(i), int(j)

    def pos(self, ij) -> np.ndarray:
        return self.origin + (np.asarray(ij, float) + 0.5) * self.res

    def _cells(self, pts: np.ndarray):
        ij = np.floor((pts[:, :2] - self.origin) / self.res).astype(int)
        n = self.floor.shape[0]
        ok = (ij >= 0).all(1) & (ij < n).all(1)
        return ij[ok], ok

    # --- updates ---------------------------------------------------------------------------------
    def integrate(self, pts: np.ndarray, cam_xy=None) -> None:
        """Add depth points. With cam_xy, the floor under the first part of each ray from the eye to a floor hit
        also counts as seen floor. Near the eye the ray is high, so only tall things could block it there; the far
        part runs low over furniture it may have missed, so it is not carved."""
        z = pts[:, 2]
        for mask, arr in ((np.abs(z) < FLOOR_Z, self.floor), ((z > OBST_Z[0]) & (z < OBST_Z[1]), self.obst)):
            ij, _ = self._cells(pts[mask])
            np.add.at(arr, (ij[:, 0], ij[:, 1]), 1)
        if cam_xy is not None:
            fl = pts[np.abs(z) < FLOOR_Z][:, :2]
            if len(fl):
                ts = np.linspace(0.0, CARVE_TO, 24)[:, None, None]
                seg = np.asarray(cam_xy, float)[None, None] + ts * (fl[None] - np.asarray(cam_xy, float)[None, None])
                ij, _ = self._cells(np.concatenate([seg.reshape(-1, 2), np.zeros((seg.shape[0] * seg.shape[1], 1))], 1))
                np.add.at(self.floor, (ij[:, 0], ij[:, 1]), 1)

    def stand(self, xy) -> None:
        """The body stands here, so a body-sized disc around it is walkable."""
        c = self.cell(xy)
        r = int(round(BODY_RADIUS / self.res))
        self.visited[max(c[0] - r, 0):c[0] + r + 1, max(c[1] - r, 0):c[1] + r + 1] = True

    # --- what the body knows -----------------------------------------------------------------------
    @property
    def blocked(self) -> np.ndarray:
        return self.obst >= 2

    @property
    def seen(self) -> np.ndarray:
        return (self.floor > 0) | self.blocked | self.visited

    def inflated(self) -> np.ndarray:
        """Cells whose centre is within BODY_RADIUS + half a cell of an obstacle cell's centre, so every path cell
        keeps at least BODY_RADIUS from the nearest edge of a blocked cell."""
        r_m = BODY_RADIUS + self.res / 2
        r = int(math.ceil(r_m / self.res))
        b = self.blocked
        grown = b.copy()
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if (di * di + dj * dj) * self.res ** 2 <= r_m ** 2 + 1e-9:
                    grown |= np.roll(np.roll(b, di, 0), dj, 1)
        return grown

    def walkable(self) -> np.ndarray:
        """Seen floor, minus anything closer than the body's clearance to an obstacle."""
        return ((self.floor > 0) | self.visited) & ~self.inflated()

    def frontiers(self, min_size: int = 3) -> list[tuple[np.ndarray, int]]:
        """Walkable cells next to never-seen cells, clustered. Returns (centre xy, size) per cluster."""
        wk = self.walkable()
        unseen = ~self.seen
        edge = np.zeros_like(wk)
        edge[1:, :] |= unseen[:-1, :]
        edge[:-1, :] |= unseen[1:, :]
        edge[:, 1:] |= unseen[:, :-1]
        edge[:, :-1] |= unseen[:, 1:]
        cand = wk & edge
        lab = np.zeros(cand.shape, np.int32)
        out, k = [], 0
        nx, ny = cand.shape
        for i, j in zip(*np.nonzero(cand)):
            if lab[i, j]:
                continue
            k += 1
            stack, cells = [(i, j)], []
            lab[i, j] = k
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da in (-1, 0, 1):
                    for db in (-1, 0, 1):
                        u, v = a + da, b + db
                        if 0 <= u < nx and 0 <= v < ny and cand[u, v] and not lab[u, v]:
                            lab[u, v] = k
                            stack.append((u, v))
            if len(cells) >= min_size:
                c = np.mean(cells, axis=0)
                rep = min(cells, key=lambda q: (q[0] - c[0]) ** 2 + (q[1] - c[1]) ** 2)
                out.append((self.pos(rep), len(cells)))
        return out

    def path(self, start_xy, goal_xy, goal_radius: float = 0.3) -> np.ndarray:
        """Shortest walkable path (world xy points), empty if unreachable on what the body has seen."""
        wk = self.walkable()
        s = self.cell(start_xy)
        # the head camera cannot see the floor right at its feet: treat unblocked cells within 0.8 m as walkable
        r = int(round(0.8 / self.res))
        ii, jj = np.mgrid[-r:r + 1, -r:r + 1]
        near = (ii ** 2 + jj ** 2 <= r * r)
        a0, b0 = s[0] - r, s[1] - r
        sl = (slice(max(a0, 0), s[0] + r + 1), slice(max(b0, 0), s[1] + r + 1))
        sub = near[max(-a0, 0):, max(-b0, 0):][:wk[sl].shape[0], :wk[sl].shape[1]]
        wk[sl] |= sub & ~self.inflated()[sl]  # unseen near floor, but never inside the clearance of an obstacle
        wk[s] = True
        g = self.cell(goal_xy)
        r = max(int(round(goal_radius / self.res)), 0)
        goals = [(i, j) for i in range(g[0] - r, g[0] + r + 1) for j in range(g[1] - r, g[1] + r + 1)
                 if 0 <= i < wk.shape[0] and 0 <= j < wk.shape[1] and wk[i, j]]
        if not goals:
            return np.zeros((0, 2))
        cells = descend(geodesic(wk, goals, self.res), s)
        return np.array([self.pos(c) for c in cells]) if cells else np.zeros((0, 2))
