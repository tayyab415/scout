"""Run one find episode and save the route plus memory snapshots for the 3D replay.

    python scripts/export_route.py --house procthor-10k-val_val_105 --start 9.85,11.56 --goal Sofa \
        --policy teacher --out runs/routes/val105_sofa_teacher
writes <out>.json (route and metadata) and <out>_snaps.npz (what the body had seen at each look).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from visualize import record  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scout.world.layout import GOAL_TYPES, Layout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--policy", default="teacher")
    ap.add_argument("--budget", type=float, default=40.0)
    ap.add_argument("--layouts", default="data/layouts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    lay = Layout.load(f"{a.layouts}/{a.house}.npz")
    start = [float(v) for v in a.start.split(",")]
    frames, res = record(lay, a.policy, start, a.goal, a.budget)
    route = [frames[0]["xy"].tolist()] + [f["xy"].tolist() for f in frames if len(f["trail"])]
    clean = [route[0]]
    for p in route[1:]:
        if np.linalg.norm(np.subtract(p, clean[-1])) > 1e-3:
            clean.append(p)
    first_yaw = float(frames[min(4, len(frames) - 1)]["yaw"]) if len(clean) < 2 else float(
        np.arctan2(clean[1][1] - clean[0][1], clean[1][0] - clean[0][0]))
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    meta = dict(house=a.house, goal=a.goal, goal_text=GOAL_TYPES.get(a.goal, a.goal), policy=a.policy,
                start=start, start_yaw=first_yaw, route=clean, result={k: (float(v) if isinstance(v, np.floating) else v)
                                                                         for k, v in res.items()},
                reasons=[f["why"] for f in frames if f["why"]][:1] + list(dict.fromkeys(f["why"] for f in frames if f["why"]))[1:])
    json.dump(meta, open(a.out + ".json", "w"), indent=1)
    np.savez_compressed(a.out + "_snaps.npz", seen=np.array([f["seen"] for f in frames]),
                        xy=np.array([f["xy"] for f in frames]), yaw=np.array([f["yaw"] for f in frames]),
                        walked=np.array([f["walked"] for f in frames]))
    with open(a.out + "_frames.json", "w") as fh:
        json.dump([dict(sightings=[(t, xy.tolist()) for t, xy in f["sightings"]],
                        frontiers=[x.tolist() for x in f["frontiers"]], choice=f["choice"], why=f["why"])
                   for f in frames], fh)
    print(a.out, res, len(clean), "route points", len(frames), "snapshots")


if __name__ == "__main__":
    main()
