"""Episodes: find an object in an unseen house, and explore first then find from memory."""
from __future__ import annotations

import math

import numpy as np

from ..agent import Body, goal_region, look_around, path_len_full, walk, frontier_table, REACH_M
from ..memory.explored import Memory
from ..policies import Policy
from ..world.layout import Layout


def _nearest_seen(memory: Memory, body: Body, goal: str):
    seen = memory.seen_of(goal)
    if not seen:
        return None
    return min(seen, key=lambda s: float(np.hypot(*(s.xy - body.xy))))


def _at_goal(layout: Layout, xy, goal: str) -> bool:
    """True if the body is within reach of a real instance (checked against the full house)."""
    return any(float(np.hypot(*(o.xy - xy))) <= REACH_M + 0.15 for o in layout.objects_of(goal))


def find(memory: Memory, body: Body, policy: Policy, goal: str, budget_m: float, max_decisions: int = 60) -> dict:
    """Search for `goal` from the body's current state. Uses and extends the existing memory."""
    start_walked, decisions, status = body.walked, 0, "budget"
    spent = lambda: body.walked - start_walked
    while decisions < max_decisions:
        target = _nearest_seen(memory, body, goal)
        if target is not None:
            res = walk(memory, body, goal_region(memory, target.xy), start_walked + budget_m)
            status = "found" if res == "arrived" and _at_goal(memory.layout, body.xy, goal) else res
            if status == "found" or res in ("budget",):
                break
            if res == "no_path":  # seen but not reachable over seen floor yet: keep exploring
                pass
        fr, _ = frontier_table(memory, body)
        if not fr:
            status = "exhausted"
            break
        k = policy.choose(memory, body, fr, goal, budget_m - spent())
        decisions += 1
        res = walk(memory, body, [fr[k][0].cell], start_walked + budget_m,
                   stop_when=lambda: bool(memory.seen_of(goal)))
        if res == "budget":
            status = "budget"
            break
        if res == "arrived":
            look_around(memory, body)
    return {"status": status, "success": status == "found", "walked_m": round(spent(), 2), "decisions": decisions}


def explore(memory: Memory, body: Body, policy: Policy, budget_m: float, max_decisions: int = 60) -> dict:
    """Explore with no goal until the walking budget runs out or nothing is left to see."""
    start_walked, decisions = body.walked, 0
    while decisions < max_decisions:
        fr, _ = frontier_table(memory, body)
        if not fr:
            break
        k = policy.choose(memory, body, fr, None, budget_m - (body.walked - start_walked))
        decisions += 1
        res = walk(memory, body, [fr[k][0].cell], start_walked + budget_m)
        if res == "budget":
            break
        if res == "arrived":
            look_around(memory, body)
    return {"walked_m": round(body.walked - start_walked, 2), "decisions": decisions,
            "coverage": round(memory.coverage(), 3), "objects_seen": len(memory.sightings),
            "rooms_seen": len(memory.rooms_seen())}


def objectnav(layout: Layout, policy: Policy, start_xy, goal: str, budget_m: float = 60.0) -> dict:
    """Find `goal` in a house the body has never seen."""
    mem = Memory(layout)
    body = Body(np.asarray(start_xy, float))
    look_around(mem, body)
    r = find(mem, body, policy, goal, budget_m)
    opt = path_len_full(layout, start_xy, goal)
    r.update(goal=goal, optimal_m=round(opt, 2),
             spl=round(r["success"] * opt / max(r["walked_m"], opt, 1e-6), 3) if math.isfinite(opt) else 0.0)
    return r


def explore_then_find(layout: Layout, policy: Policy, start_xy, goal: str, explore_m: float = 40.0,
                      find_m: float = 40.0, rng=None) -> dict:
    """Explore with no goal, get asked for `goal` at a random known spot, then find it.

    Returns the find result plus whether the goal was already in memory when asked.
    """
    mem = Memory(layout)
    body = Body(np.asarray(start_xy, float))
    look_around(mem, body)
    ex = explore(mem, body, policy, explore_m)
    already = bool(mem.seen_of(goal))
    query_xy = body.xy.copy()
    r = find(mem, body, policy, goal, find_m)
    opt = path_len_full(layout, query_xy, goal)
    r.update(goal=goal, optimal_m=round(opt, 2), in_memory=already, explore=ex,
             spl=round(r["success"] * opt / max(r["walked_m"], opt, 1e-6), 3) if math.isfinite(opt) else 0.0)
    return r
