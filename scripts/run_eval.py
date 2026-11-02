"""Compare exploration policies on houses the body has never seen.

    python scripts/run_eval.py --houses 40 --policies nearest,size_over_dist,random,teacher --out runs/eval1

Episodes are fixed by seed, so every policy gets the same houses, starts and goals.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scout.agent import Body, look_around, path_len_full  # noqa: E402
from scout.eval.episodes import explore_then_find, objectnav  # noqa: E402
from scout.memory.explored import Memory  # noqa: E402
from scout.policies import LargestNearFrontier, NearestFrontier, OracleFrontier, RandomFrontier  # noqa: E402
from scout.world.layout import GOAL_TYPES, Layout  # noqa: E402


def make_policy(name: str, seed: int, log_path: str | None):
    if name == "nearest":
        return NearestFrontier()
    if name == "size_over_dist":
        return LargestNearFrontier()
    if name == "oracle_frontier":
        return OracleFrontier()
    if name == "random":
        return RandomFrontier(seed)
    if name.startswith("teacher"):
        from scout.teacher.llm import BedrockChat, OpenAICompatChat
        from scout.teacher.policy import TeacherPolicy
        backend = name.split(":", 1)[1] if ":" in name else "bedrock"
        llm = OpenAICompatChat() if backend == "openai" else BedrockChat(model_id=backend if backend != "bedrock" else
                                                                        "us.meta.llama4-maverick-17b-instruct-v1:0")
        return TeacherPolicy(llm, log_path)
    raise ValueError(name)


def sample_episodes(paths: list[str], n_houses: int, per_house: int, seed: int, min_rooms: int = 3):
    """(house path, start xy, goal type) with the goal out of sight at the start and at least 4 m away."""
    rng = np.random.default_rng(seed)
    eps = []
    for p in paths:
        if len({e[0] for e in eps}) >= n_houses:
            break
        lay = Layout.load(p)
        if len(lay.rooms) < min_rooms or lay.grid.free.sum() < 500:
            continue
        cells = np.argwhere(lay.grid.free)
        types = [t for t in lay.goal_types() if t in GOAL_TYPES]
        got = 0
        for _ in range(40):
            if got >= per_house or not types:
                break
            start = lay.grid.pos(cells[rng.integers(len(cells))])
            goal = types[rng.integers(len(types))]
            m = Memory(lay)
            look_around(m, Body(start.copy()))
            if m.seen_of(goal):
                continue
            if not (4.0 <= path_len_full(lay, start, goal) < 40.0):
                continue
            eps.append((p, start.tolist(), goal))
            got += 1
    return eps


BUDGETS = {"find_m": 40.0, "explore_m": 20.0, "after_m": 30.0}


def run_one(task, policy_name, mode, seed, log_dir, budgets=None):
    b = budgets or BUDGETS
    p, start, goal = task
    lay = Layout.load(p)
    pol = make_policy(policy_name, seed, os.path.join(log_dir, f"teacher_{mode}_{os.getpid()}.jsonl") if log_dir else None)
    t = time.time()
    if mode == "find":
        r = objectnav(lay, pol, start, goal, budget_m=b["find_m"])
    else:
        r = explore_then_find(lay, pol, start, goal, explore_m=b["explore_m"], find_m=b["after_m"])
    r.update(house=lay.name, policy=policy_name, mode=mode, start=start, seconds=round(time.time() - t, 2))
    if hasattr(pol, "calls"):
        r.update(llm_calls=pol.calls, llm_fallbacks=pol.fallbacks,
                 llm_latency_s=round(float(np.median(pol.latency)), 2) if pol.latency else None)
    return r


def summarise(rows):
    out = {}
    for (pol, mode) in sorted({(r["policy"], r["mode"]) for r in rows}):
        rs = [r for r in rows if r["policy"] == pol and r["mode"] == mode]
        succ = np.array([r["success"] for r in rs], float)
        k = len(rs)
        se = float(np.sqrt(succ.mean() * (1 - succ.mean()) / max(k, 1)))
        out[f"{mode}/{pol}"] = dict(n=k, success=round(float(succ.mean()), 3), success_se=round(se, 3),
                                    spl=round(float(np.mean([r["spl"] for r in rs])), 3),
                                    walked_m=round(float(np.mean([r["walked_m"] for r in rs])), 1),
                                    decisions=round(float(np.mean([r["decisions"] for r in rs])), 1))
        if mode == "explore_find":
            out[f"{mode}/{pol}"]["in_memory"] = round(float(np.mean([r["in_memory"] for r in rs])), 3)
            out[f"{mode}/{pol}"]["explore_coverage"] = round(float(np.mean([r["explore"]["coverage"] for r in rs])), 3)
        if "llm_calls" in rs[0]:
            out[f"{mode}/{pol}"]["llm_fallback_rate"] = round(sum(r["llm_fallbacks"] for r in rs) / max(sum(r["llm_calls"] for r in rs), 1), 3)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", default="data/layouts")
    ap.add_argument("--houses", type=int, default=40)
    ap.add_argument("--per-house", type=int, default=2)
    ap.add_argument("--offset", type=int, default=0, help="skip this many layout files (keeps splits apart)")
    ap.add_argument("--policies", default="nearest,size_over_dist,random")
    ap.add_argument("--modes", default="find,explore_find")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-rooms", type=int, default=3)
    ap.add_argument("--find-m", type=float, default=BUDGETS["find_m"])
    ap.add_argument("--explore-m", type=float, default=BUDGETS["explore_m"])
    ap.add_argument("--after-m", type=float, default=BUDGETS["after_m"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="runs/eval")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    paths = sorted(glob.glob(f"{a.layouts}/*.npz"))[a.offset:]
    eps = sample_episodes(paths, a.houses, a.per_house, a.seed, a.min_rooms)
    budgets = {"find_m": a.find_m, "explore_m": a.explore_m, "after_m": a.after_m}
    json.dump(vars(a), open(f"{a.out}/args.json", "w"), indent=1)
    print(f"{len(eps)} episodes in {len({e[0] for e in eps})} houses", flush=True)
    jobs = [(e, pol, mode) for mode in a.modes.split(",") for pol in a.policies.split(",") for e in eps]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex, open(f"{a.out}/episodes.jsonl", "w") as f:
        futs = [ex.submit(run_one, e, pol, mode, a.seed + i, a.out, budgets) for i, (e, pol, mode) in enumerate(jobs)]
        for k, fu in enumerate(futs):
            r = fu.result()
            rows.append(r)
            f.write(json.dumps(r) + "\n")
            if (k + 1) % 20 == 0:
                print(f"{k + 1}/{len(jobs)}", flush=True)
    s = summarise(rows)
    json.dump(s, open(f"{a.out}/summary.json", "w"), indent=1)
    for k, v in s.items():
        print(k, v)


if __name__ == "__main__":
    main()
