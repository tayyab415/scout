"""Top-down replay of an episode, one or more policies side by side.

    python scripts/visualize.py --house procthor-10k-val_val_105 --start 9.85,11.56 --goal Sofa \
        --policies teacher,nearest --out runs/viz/val105_sofa.mp4
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
from run_eval import make_policy  # noqa: E402
from scout.agent import Body, look_around  # noqa: E402
from scout.eval.episodes import find  # noqa: E402
from scout.memory.explored import Memory  # noqa: E402
from scout.teacher.prompt import ROOM_NAMES  # noqa: E402
from scout.world.layout import GOAL_TYPES, Layout  # noqa: E402

LABEL = {"teacher": "LLM teacher (Llama 4 Maverick)", "nearest": "Nearest opening (classic rule)",
         "size_over_dist": "Biggest-nearby opening (rule)", "random": "Random opening",
         "oracle_frontier": "Oracle (knows the map)"}


def record(lay: Layout, policy_name: str, start, goal: str, budget: float):
    """Run one find episode and keep a snapshot after every look and every decision."""
    mem, body = Memory(lay), Body(np.asarray(start, float))
    pol = make_policy(policy_name, 0, None)
    frames, state = [], {"choice": None, "frontiers": [], "why": ""}
    real_look = mem.look

    def look(xy, yaw, t=0.0, fov_deg=75.0):
        new = real_look(xy, yaw, t, fov_deg)
        frames.append(dict(seen=mem.seen.copy(), xy=np.asarray(xy, float).copy(), yaw=yaw, walked=body.walked,
                           sightings=[(s.type, s.xy.copy()) for s in mem.sightings.values()],
                           frontiers=[f.xy.copy() for f, _ in state["frontiers"]], choice=state["choice"],
                           why=state["why"], trail=[p.copy() for p in body.route]))
        return new
    mem.look = look

    real_choose = pol.choose

    def choose(memory, b, frontiers, g, left):
        k = real_choose(memory, b, frontiers, g, left)
        state.update(choice=k, frontiers=frontiers)
        if hasattr(pol, "log_path"):
            state["why"] = getattr(pol, "last_why", "")
        return k
    pol.choose = choose
    if policy_name.startswith("teacher"):
        from scout.teacher import policy as tp
        orig = tp.parse

        def parse_keep(text, n):
            k, why = orig(text, n)
            pol.last_why = why
            return k, why
        tp.parse = parse_keep
    look_around(mem, body)
    res = find(mem, body, pol, goal, budget)
    if policy_name.startswith("teacher"):
        tp.parse = orig
    return frames, res


def draw(ax, lay: Layout, fr: dict, goal: str, title: str, done: str | None):
    g = lay.grid
    img = np.ones(g.shape + (3,)) * 0.08
    img[g.free] = [0.30, 0.30, 0.33]
    img[fr["seen"] & g.free] = [0.82, 0.86, 0.80]
    img[lay.walls] = [0.0, 0.0, 0.0]
    ext = [g.origin[0], g.origin[0] + g.shape[0] * g.res, g.origin[1], g.origin[1] + g.shape[1] * g.res]
    ax.imshow(img.transpose(1, 0, 2), origin="lower", extent=ext, interpolation="nearest")
    for o in lay.objects_of(goal):  # true goal locations, for the viewer only
        ax.plot(*o.xy, marker="*", ms=16, mfc="none", mec="#ffcc00", mew=1.5)
    for t, xy in fr["sightings"]:
        c = "#ffcc00" if t == goal else "#3a7bd5"
        ax.plot(*xy, "o", ms=5 if t != goal else 9, color=c)
        ax.text(xy[0] + 0.15, xy[1] + 0.15, GOAL_TYPES.get(t, t), fontsize=6.5, color=c)
    for k, f in enumerate(fr["frontiers"]):
        sel = k == fr["choice"]
        ax.plot(*f, "s", ms=9 if sel else 6, color="#e0218a" if sel else "#b06ab3", mec="white" if sel else "none")
        ax.text(f[0] + 0.12, f[1] - 0.35, str(k), fontsize=6, color="#e0218a")
    if len(fr["trail"]) > 1:
        tr = np.array(fr["trail"])
        ax.plot(tr[:, 0], tr[:, 1], "-", color="#ff6b35", lw=1.6)
    x, y = fr["xy"]
    ax.arrow(x, y, 0.6 * np.cos(fr["yaw"]), 0.6 * np.sin(fr["yaw"]), width=0.12, color="#ff6b35", length_includes_head=True)
    for r in lay.rooms:
        if fr["seen"][tuple(np.clip(g.cell(r.polygon.mean(0)), 0, np.array(g.shape) - 1))]:
            c = r.polygon.mean(0)
            ax.text(c[0], c[1], ROOM_NAMES.get(r.type, r.type), fontsize=8, color="#1b5e20", ha="center", alpha=0.8)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"{title}\nwalked {fr['walked']:.1f} m" + (f"  ·  {done}" if done else ""), fontsize=10)
    if fr["why"]:
        ax.text(0.01, -0.02, "\n".join(textwrap.wrap("Teacher: " + fr["why"], 70)), transform=ax.transAxes,
                fontsize=7.5, va="top", color="#333")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--policies", default="teacher,nearest")
    ap.add_argument("--budget", type=float, default=40.0)
    ap.add_argument("--layouts", default="data/layouts")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=int, default=8)
    a = ap.parse_args()
    lay = Layout.load(f"{a.layouts}/{a.house}.npz")
    start = [float(v) for v in a.start.split(",")]
    runs = []
    for p in a.policies.split(","):
        frames, res = record(lay, p, start, a.goal, a.budget)
        done = f"FOUND the {GOAL_TYPES.get(a.goal, a.goal)}" if res["success"] else f"not found ({res['status']})"
        runs.append((p, frames, res, done))
        print(p, res, len(frames), "frames")
    n = max(len(f) for _, f, _, _ in runs) + a.fps * 2
    tmp = tempfile.mkdtemp()
    w = 6.0 * len(runs)
    for i in range(n):
        fig, axes = plt.subplots(1, len(runs), figsize=(w, 6.6))
        axes = np.atleast_1d(axes)
        for ax, (p, frames, res, done) in zip(axes, runs):
            k = min(i, len(frames) - 1)
            draw(ax, lay, frames[k], a.goal, LABEL.get(p, p), done if i >= len(frames) - 1 else None)
        fig.suptitle(f"Task: find the {GOAL_TYPES.get(a.goal, a.goal)} in a house it has never seen  "
                     f"(light = seen, dark = unknown, squares = openings, star = where it really is)", fontsize=10)
        fig.tight_layout(rect=[0, 0.02, 1, 0.95])
        fig.savefig(f"{tmp}/f{i:04d}.png", dpi=90)
        plt.close(fig)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.fps), "-i", f"{tmp}/f%04d.png",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", a.out], check=True)
    last = f"{tmp}/f{n - 1:04d}.png"
    subprocess.run(["cp", last, a.out.rsplit(".", 1)[0] + "_final.png"], check=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
