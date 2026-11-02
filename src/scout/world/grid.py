"""Walkable-space grid for a standing body, and geodesic distances on it. Pure numpy."""
from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

SQRT2 = 2 ** 0.5
NEIGHBOURS = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
              (1, 1, SQRT2), (1, -1, SQRT2), (-1, 1, SQRT2), (-1, -1, SQRT2)]


@dataclass
class Grid:
    """Boolean mask of cells where the body fits. Cell (i, j) covers origin + (i, j) * res."""
    free: np.ndarray
    origin: np.ndarray
    res: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.free.shape

    def cell(self, xy) -> tuple[int, int]:
        i, j = np.floor((np.asarray(xy, float) - self.origin) / self.res).astype(int)
        return int(i), int(j)

    def pos(self, ij) -> np.ndarray:
        return self.origin + (np.asarray(ij, float) + 0.5) * self.res

    def inside(self, ij) -> bool:
        return 0 <= ij[0] < self.free.shape[0] and 0 <= ij[1] < self.free.shape[1]

    def is_free(self, ij) -> bool:
        return self.inside(ij) and bool(self.free[ij])


def largest_component(free: np.ndarray) -> np.ndarray:
    """Keep the largest 4-connected region of free cells."""
    lab = np.zeros(free.shape, np.int32)
    best, best_n, k = 0, 0, 0
    nx, ny = free.shape
    for i, j in zip(*np.nonzero(free)):
        if lab[i, j]:
            continue
        k += 1
        stack, n = [(i, j)], 0
        lab[i, j] = k
        while stack:
            a, b = stack.pop()
            n += 1
            for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                u, v = a + da, b + db
                if 0 <= u < nx and 0 <= v < ny and free[u, v] and not lab[u, v]:
                    lab[u, v] = k
                    stack.append((u, v))
        if n > best_n:
            best, best_n = k, n
    return lab == best if best else np.zeros_like(free)


def geodesic(free: np.ndarray, sources, res: float) -> np.ndarray:
    """Shortest walking distance (m) from any source cell, 8-connected, no corner cutting."""
    dist = np.full(free.shape, np.inf)
    heap = []
    for s in sources:
        s = tuple(int(v) for v in s)
        if free[s]:
            dist[s] = 0.0
            heap.append((0.0, s))
    heapq.heapify(heap)
    nx, ny = free.shape
    while heap:
        c, (i, j) = heapq.heappop(heap)
        if c > dist[i, j]:
            continue
        for di, dj, w in NEIGHBOURS:
            u, v = i + di, j + dj
            if not (0 <= u < nx and 0 <= v < ny and free[u, v]):
                continue
            if w > 1 and not (free[i, v] and free[u, j]):
                continue
            nc = c + w * res
            if nc < dist[u, v]:
                dist[u, v] = nc
                heapq.heappush(heap, (nc, (u, v)))
    return dist


def descend(dist: np.ndarray, start, max_steps: int = 5000) -> list[tuple[int, int]]:
    """Follow the distance field downhill from start to a source. Empty if unreachable."""
    c = tuple(int(v) for v in start)
    if not np.isfinite(dist[c]):
        return []
    path = [c]
    nx, ny = dist.shape
    while dist[c] > 0 and len(path) < max_steps:
        i, j = c
        nbs = [(i + a, j + b) for a, b, _ in NEIGHBOURS if 0 <= i + a < nx and 0 <= j + b < ny]
        nxt = min(nbs, key=lambda q: dist[q])
        if dist[nxt] >= dist[c]:
            break
        c = nxt
        path.append(c)
    return path


def path_length(path, res: float) -> float:
    if len(path) < 2:
        return 0.0
    p = np.asarray(path, float)
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum() * res)
