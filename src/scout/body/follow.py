"""Steer a body along a route using a fixed set of egocentric walking directions. Pure numpy."""
from __future__ import annotations

import math

import numpy as np

# Egocentric directions (degrees, 0 = straight ahead, positive = to the left) with Motivo skills for each.
DIRECTIONS = {0: "move-ego-0-1", 45: "move-ego-45-1", 90: "move-ego-90-1", 135: "move-ego-135-1",
              180: "move-ego-180-1", -45: "move-ego--45-1", -90: "move-ego--90-1", -135: "move-ego--135-1"}
STAND = "move-ego-0-0"


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


def lookahead(route: np.ndarray, xy: np.ndarray, progress: int, ahead_m: float = 1.0) -> tuple[np.ndarray, int]:
    """Target point `ahead_m` along the route past the closest point at or after `progress`.

    Returns (target, new progress). Progress never moves backwards, so loops in the route are kept.
    """
    window = route[progress:progress + 40]
    k = progress + int(np.argmin(np.linalg.norm(window - xy, axis=1)))
    s = 0.0
    j = k
    while j + 1 < len(route) and s < ahead_m:
        s += float(np.linalg.norm(route[j + 1] - route[j]))
        j += 1
    return route[j], k


def pick_direction(rel_deg: float, available=DIRECTIONS) -> int:
    """Closest available egocentric direction to the wanted one."""
    return min(available, key=lambda a: abs(wrap_deg(rel_deg - a)))


def steer(xy, yaw: float, route: np.ndarray, progress: int, stop_m: float = 0.5, available=DIRECTIONS):
    """One control decision. Returns (skill name, new progress, done)."""
    xy = np.asarray(xy, float)
    if np.linalg.norm(route[-1] - xy) < stop_m and progress >= len(route) - 5:
        return STAND, progress, True
    target, progress = lookahead(route, xy, progress)
    d = target - xy
    if np.linalg.norm(d) < 1e-6:
        return STAND, progress, progress >= len(route) - 1
    rel = wrap_deg(math.degrees(math.atan2(d[1], d[0]) - yaw))
    return available[pick_direction(rel, available)], progress, False


def densify(points, step: float = 0.1) -> np.ndarray:
    """Resample a polyline to roughly even spacing."""
    p = np.asarray(points, float)
    out = [p[0]]
    for a, b in zip(p[:-1], p[1:]):
        n = max(int(np.ceil(np.linalg.norm(b - a) / step)), 1)
        out += [a + (b - a) * t for t in np.linspace(0, 1, n + 1)[1:]]
    return np.array(out)
