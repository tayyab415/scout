"""Ways to decide where to explore next. Each picks one reachable frontier."""
from __future__ import annotations

import numpy as np

from .agent import Body
from .memory.explored import Frontier, Memory


class Policy:
    name = "policy"

    def choose(self, memory: Memory, body: Body, frontiers: list[tuple[Frontier, float]],
               goal: str | None, budget_left: float) -> int:
        """Return an index into `frontiers`."""
        raise NotImplementedError


class RandomFrontier(Policy):
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def choose(self, memory, body, frontiers, goal, budget_left):
        return int(self.rng.integers(len(frontiers)))


class NearestFrontier(Policy):
    """Classic frontier exploration (Yamauchi 1997): always go to the closest unexplored edge."""
    name = "nearest"

    def choose(self, memory, body, frontiers, goal, budget_left):
        return int(np.argmin([d for _, d in frontiers]))


class LargestNearFrontier(Policy):
    """Prefer big openings, discounted by distance. A stronger hand-written baseline."""
    name = "size_over_dist"

    def choose(self, memory, body, frontiers, goal, budget_left):
        return int(np.argmax([f.size / (1.0 + d) for f, d in frontiers]))


class OracleFrontier(Policy):
    """Privileged upper bound: knows the full map and picks the opening with the shortest true walk to the goal.
    With no goal it falls back to the nearest opening."""
    name = "oracle_frontier"

    def __init__(self):
        self._cache = {}

    def _dist(self, layout, goal):
        from .agent import REACH_M
        from .world.grid import geodesic
        key = (layout.name, goal)
        if key not in self._cache:
            g = layout.grid
            src = []
            for o in layout.objects_of(goal):
                ci, cj = g.cell(o.xy)
                r = int(np.ceil(REACH_M / g.res))
                src += [(i, j) for i in range(ci - r, ci + r + 1) for j in range(cj - r, cj + r + 1)
                        if g.is_free((i, j)) and np.hypot(i - ci, j - cj) * g.res <= REACH_M]
            self._cache[key] = geodesic(g.free, src, g.res)
        return self._cache[key]

    def choose(self, memory, body, frontiers, goal, budget_left):
        if goal is None:
            return int(np.argmin([d for _, d in frontiers]))
        dg = self._dist(memory.layout, goal)
        return int(np.argmin([d + dg[f.cell] for f, d in frontiers]))
