"""Run Scout episodes on any GPU machine (no Modal): one episode, or every missing episode of a frozen plan.

    MUJOCO_GL=egl python scripts/run_local.py --house procthor-10k-val_val_935 --brain rules --tag dev
    MUJOCO_GL=egl python scripts/run_local.py --plan configs/frozen_final3.json --tag final3 --parallel 2

Data layout under --data-root (default /data): house/{scenes,objects,layouts}, scout/z_motion.pt, outputs in
scout/env/<tag>/<episode>/. Each episode runs in its own process (its own GL context); a crash is logged and kept.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))


def _one(args):
    house, brain, seed, tag, explore_s, data_root = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from scout.env.episode import run_episode
    t0 = time.time()
    try:
        r = run_episode(house, brain, seed, tag, explore_s, data_root=data_root, sam3_dir=os.environ.get("SAM3_DIR", "/opt/sam3"))
        return dict(brain=brain, house=house, ok=True, minutes=round((time.time() - t0) / 60, 1),
                    target=r["manifest"].get("target"), answers=r["answers"])
    except Exception as e:  # a crash is a result too
        return dict(brain=brain, house=house, ok=False, error=f"{type(e).__name__}: {str(e)[:400]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house")
    ap.add_argument("--brain", default="rules")
    ap.add_argument("--plan")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--explore-s", type=float, default=240.0)
    ap.add_argument("--data-root", default="/data")
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--only", default="", help="comma list of brain:house to run from the plan (e.g. the missing ones)")
    a = ap.parse_args()
    os.environ["SCOUT_COMMIT"] = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    os.environ["SCOUT_HOST"] = os.environ.get("SCOUT_HOST", "aws")
    if a.plan:
        plan = json.load(open(a.plan))
        seed, explore = plan.get("seed", 0), plan.get("explore_s", a.explore_s)
        only = {tuple(x.split(":")) for x in a.only.split(",") if x}
        jobs = [(h, b, seed, a.tag, explore, a.data_root) for b in plan["brains"] for h in plan["houses"]
                if (not only or (b, h) in only)
                and not os.path.exists(f"{a.data_root}/scout/env/{a.tag}/{b}_{h}_s{seed}/manifest.json")]
    else:
        jobs = [(a.house, a.brain, a.seed, a.tag, a.explore_s, a.data_root)]
    print(f"{len(jobs)} episodes to run", flush=True)
    log = f"{a.data_root}/scout/env/{a.tag}/run_local.jsonl"
    os.makedirs(os.path.dirname(log), exist_ok=True)
    import multiprocessing as mp
    # one fresh process per episode: a worker that keeps the previous episode's GPU memory (SAM 3.1, renderers)
    # runs the next one out of memory
    with ProcessPoolExecutor(max_workers=a.parallel, mp_context=mp.get_context("spawn"), max_tasks_per_child=1) as ex:
        for r in ex.map(_one, jobs):
            print("EP", json.dumps(r, default=str), flush=True)
            open(log, "a").write(json.dumps(r, default=str) + "\n")


if __name__ == "__main__":
    main()
