"""Render a planning run: follow camera | head camera | goal photo, and the imagined next view for each action.

Imagined views are the closest real views from the body's earlier walks in the same house to what the world
model predicts (a lookup in memory, not a generated image).
"""
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

NAMES = ["stand", "forward", "side-left", "side-right", "back", "walk-turn-R"]


def render(p, out):
    z = np.load(p)
    bank = {int(i): im for i, im in zip(z["bank_ids"], z["bank"])}
    tmp = tempfile.mkdtemp()
    n = 0
    T = len(z["head"])
    for k in range(T):
        W, H = 1280, 640
        c = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(c)
        c.paste(Image.fromarray(z["follow"][k]).resize((480, 360)), (0, 40))
        c.paste(Image.fromarray(z["head"][k]).resize((360, 360)), (490, 40))
        c.paste(Image.fromarray(z["goal_img"]).resize((360, 360)), (860, 40))
        d.text((6, 8), f"{os.path.basename(p)[:-10]}   step {k + 1}/{T}   chose: {NAMES[int(z['actions'][k])]}   "
                       f"distance to goal spot: {z['pos_err'][k]:.2f} m", fill="black")
        d.text((6, 24), "follow camera", fill="gray")
        d.text((496, 24), "head camera now", fill="gray")
        d.text((866, 24), "GOAL photo", fill="#c0392b")
        d.text((6, 410), "Imagined next view for each action (nearest real view in memory to the world model's prediction):",
               fill="black")
        for j, idx in enumerate(z["imagined"][k]):
            x = 6 + j * 212
            c.paste(Image.fromarray(bank[int(idx)]).resize((200, 200)), (x, 430))
            col = "#c0392b" if j == int(z["actions"][k]) else "black"
            d.text((x, 632 - 0), "", fill=col)
            d.rectangle([x - 2, 428, x + 202, 632], outline=col, width=3 if col != "black" else 1)
            d.text((x + 4, 434), NAMES[j], fill="white")
        for _ in range(3):
            c.save(f"{tmp}/f{n:05d}.png")
            n += 1
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "8", "-i", f"{tmp}/f%05d.png",
                    "-pix_fmt", "yuv420p", out], check=True)
    return out


if __name__ == "__main__":
    for p in sorted(glob.glob(f"{sys.argv[1]}/*_video.npz")):
        print(render(p, p.replace("_video.npz", ".mp4")))
