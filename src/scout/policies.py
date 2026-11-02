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
