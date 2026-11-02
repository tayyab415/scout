"""What the body knows about a house: where it has looked, what it saw and where things are.

Only information the body could have gathered itself goes in here: cells inside its view cone
with a clear line of sight, and objects it has looked at. The full house map never leaks in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..world.layout import GOAL_TYPES, Layout

FOV_DEG = 75.0
VIEW_RANGE = 6.0  # m


@dataclass
class Sighting:
    type: str
    xy: np.ndarray
    room: str | None
    t: float


@dataclass
class Frontier:
    """Edge between known walkable floor and unexplored space."""
    id: int
    xy: np.ndarray
    cell: tuple[int, int]
    size: int  # cells


@dataclass
class Memory:
    layout: Layout
    seen: np.ndarray = field(init=False)
    sightings: dict[str, Sighting] = field(default_factory=dict)  # one entry per object instance key
    trail: list[np.ndarray] = field(default_factory=list)

    def __post_init__(self):
        self.seen = np.zeros(self.layout.grid.shape, bool)

    # --- sensing --------------------------------------------------------------------------------
    def _cast(self, xy: np.ndarray, angles: np.ndarray, length: float):
        """Cells along many rays at once, cut at the first wall. Returns (i, j) arrays and per-ray reach (m)."""
        g = self.layout.grid
        step = g.res * 0.5
        n = int(length / step)
        r = np.arange(n) * step
        pts = xy[None, None, :] + r[None, :, None] * np.stack([np.cos(angles), np.sin(angles)], 1)[:, None, :]
        ij = np.floor((pts - g.origin) / g.res).astype(int)
        i, j = ij[..., 0], ij[..., 1]
        inside = (i >= 0) & (i < g.shape[0]) & (j >= 0) & (j < g.shape[1])
        ic, jc = np.clip(i, 0, g.shape[0] - 1), np.clip(j, 0, g.shape[1] - 1)
        blocked = ~inside
        if self.layout.walls is not None:
            blocked |= self.layout.walls[ic, jc]
        first = np.where(blocked.any(1), blocked.argmax(1), n)
        keep = np.arange(n)[None, :] < first[:, None]
        return ic[keep], jc[keep], first * step

    def look(self, xy, yaw: float, t: float = 0.0, fov_deg: float = FOV_DEG) -> list[str]:
        """Update memory from one view. Returns keys of objects seen for the first time."""
        xy = np.asarray(xy, float)
        self.trail.append(xy.copy())
        half = math.radians(fov_deg) / 2
        i, j, _ = self._cast(xy, np.linspace(yaw - half, yaw + half, 48), VIEW_RANGE)
        self.seen[i, j] = True
        cand = []
        for k, o in enumerate(self.layout.objects):
            if o.type not in GOAL_TYPES:
                continue
            v = o.xy - xy
            dist = float(np.hypot(*v))
            if dist > VIEW_RANGE or dist < 1e-6:
                continue
            off = (math.atan2(v[1], v[0]) - yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(off) <= half:
                cand.append((k, o, dist, math.atan2(v[1], v[0])))
        new = []
        if cand:
            # line of sight to within 0.4 m of the object (objects sit against walls)
            need = np.array([max(d - 0.4, 0.0) for _, _, d, _ in cand])
            _, _, reach = self._cast(xy, np.array([a for _, _, _, a in cand]), max(float(need.max()), self.layout.grid.res))
            for (k, o, _, _), nd, rc in zip(cand, need, reach):
                if rc + 1e-6 < nd - self.layout.grid.res:
                    continue
                key = f"{o.type}#{k}"
                if key not in self.sightings:
                    new.append(key)
                self.sightings[key] = Sighting(o.type, o.xy.copy(), o.room, t)
        return new

    def look_around(self, xy, t: float = 0.0) -> list[str]:
        new = []
        for k in range(4):
            new += self.look(xy, k * math.pi / 2, t)
        return new

    # --- what the body knows ----------------------------------------------------------------------
    @property
    def known_free(self) -> np.ndarray:
        return self.seen & self.layout.grid.free

    def frontiers(self, min_size: int = 6) -> list[Frontier]:
        """Known walkable cells that border unseen cells, grouped into clusters."""
        kf = self.known_free
        unseen = ~self.seen
        edge = np.zeros_like(kf)
        edge[1:, :] |= unseen[:-1, :]
        edge[:-1, :] |= unseen[1:, :]
        edge[:, 1:] |= unseen[:, :-1]
        edge[:, :-1] |= unseen[:, 1:]
        cand = kf & edge
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
            if len(cells) < min_size:
                continue
            c = np.mean(cells, axis=0)
            rep = min(cells, key=lambda q: (q[0] - c[0]) ** 2 + (q[1] - c[1]) ** 2)
            out.append(Frontier(len(out), self.layout.grid.pos(rep), rep, len(cells)))
        return out

    def seen_of(self, type_: str) -> list[Sighting]:
        return [s for s in self.sightings.values() if s.type == type_]

    def rooms_seen(self, min_frac: float = 0.3) -> list[tuple[str, str, np.ndarray]]:
        """Rooms the body has looked into enough to know what they are: (id, type, centre)."""
        out = []
        g = self.layout.grid
        for r in self.layout.rooms:
            lo, hi = r.polygon.min(0), r.polygon.max(0)
            (i0, j0), (i1, j1) = g.cell(lo), g.cell(hi)
            sub = (slice(max(i0, 0), max(i1, 0)), slice(max(j0, 0), max(j1, 0)))
            inside = g.free[sub]
            if inside.sum() == 0:
                continue
            if (self.seen[sub] & inside).sum() / inside.sum() >= min_frac:
                out.append((r.id, r.type, r.polygon.mean(0)))
        return out

    def coverage(self) -> float:
        f = self.layout.grid.free
        return float((self.seen & f).sum() / max(f.sum(), 1))
