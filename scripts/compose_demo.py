"""Put the 3D body video, the head camera and the brain's map side by side, with the task and reasoning.

    python scripts/compose_demo.py --route runs/routes/val105_sofa_teacher --body runs/body/val105_sofa_teacher \
        --out runs/demo/val105_sofa.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import textwrap

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from visualize import draw  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scout.body.follow import densify  # noqa: E402
from scout.world.layout import Layout  # noqa: E402

BRAIN = {"teacher": "Llama 4 Maverick teacher (stand-in for Muse Spark)", "nearest": "nearest-opening rule",
         "oracle_frontier": "oracle"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--route", required=True, help="prefix written by export_route.py")
    ap.add_argument("--body", required=True, help="prefix of <name>_body.mp4/json from modal_body.py")
    ap.add_argument("--layouts", default="data/layouts")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    meta = json.load(open(a.route + ".json"))
    snaps = np.load(a.route + "_snaps.npz")
    info = json.load(open(a.route + "_frames.json"))
    body = json.load(open(a.body + "_body.json"))
    lay = Layout.load(f"{a.layouts}/{meta['house']}.npz")
    pts = densify(meta["route"], 0.1)
    s_route = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    walked = snaps["walked"]
    reader = imageio.get_reader(a.body + "_body.mp4")
    track = body["track"]
    tmp = tempfile.mkdtemp()
    cache = {}
    n_out = 0
    for i, frame in enumerate(reader):
        sim_i = min(i * 2, len(track) - 1)
        s = s_route[min(track[sim_i]["progress"], len(s_route) - 1)]
        k = int(np.searchsorted(walked, s + 1e-6, side="right") - 1)
        k = int(np.clip(k, 0, len(walked) - 1))
        if k not in cache:
            fr = dict(seen=snaps["seen"][k], xy=snaps["xy"][k], yaw=float(snaps["yaw"][k]), walked=float(walked[k]),
                      sightings=[(t, np.array(xy)) for t, xy in info[k]["sightings"]],
                      frontiers=[np.array(f) for f in info[k]["frontiers"]], choice=info[k]["choice"], why="",
                      trail=[snaps["xy"][j] for j in range(1, k + 1)])
            fig, ax = plt.subplots(figsize=(4.8, 4.8), dpi=100)
            draw(ax, lay, fr, meta["goal"], "What the brain knows", None)
            fig.tight_layout()
            fig.canvas.draw()
            cache[k] = (np.asarray(fig.canvas.buffer_rgba())[..., :3].copy(), info[k]["why"])
            plt.close(fig)
        mini, why = cache[k]
        why = why or next((info[j]["why"] for j in range(k, -1, -1) if info[j]["why"]), "")
        h = frame.shape[0]
        if mini.shape[0] != h:
            import PIL.Image
            mini = np.asarray(PIL.Image.fromarray(mini).resize((int(mini.shape[1] * h / mini.shape[0]), h)))
        top = np.concatenate([frame, mini], 1)
        fig = plt.figure(figsize=(top.shape[1] / 100, (h + 90) / 100), dpi=100)
        ax = fig.add_axes([0, 90 / (h + 90), 1, h / (h + 90)])
        ax.imshow(top)
        ax.axis("off")
        fig.text(0.01, 0.62 * 90 / (h + 90), f"Task: find the {meta['goal_text']} in a house it has never seen   "
                 f"|   body: Meta Motivo (physics)   |   eyes: head camera   |   brain: {BRAIN.get(meta['policy'], meta['policy'])}",
                 fontsize=10, weight="bold")
        if why:
            fig.text(0.01, 0.12 * 90 / (h + 90), "\n".join(textwrap.wrap("Brain: " + why, 190)[:2]), fontsize=9, color="#333")
        fig.savefig(f"{tmp}/f{n_out:05d}.png")
        plt.close(fig)
        n_out += 1
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "15", "-i", f"{tmp}/f%05d.png",
                    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-pix_fmt", "yuv420p", a.out], check=True)
    print("wrote", a.out, n_out, "frames")


if __name__ == "__main__":
    main()
