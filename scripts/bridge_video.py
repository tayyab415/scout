"""Side-by-side clips from the bridge test: original behaviour | copy from video alone | average behaviour.

    python scripts/bridge_video.py --npz runs/bridge/showcase.npz --out runs/bridge/bridge.mp4
"""
from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw


def label(img, text, sub=""):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 34], fill=(0, 0, 0))
    d.text((6, 3), text, fill=(255, 255, 255))
    if sub:
        d.text((6, 18), sub, fill=(200, 200, 200))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    s = np.load(a.npz)
    ids = sorted({k.split("_")[0] for k in s.files}, key=int)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    w = imageio.get_writer(a.out, fps=7.5, codec="libx264", quality=8, macro_block_size=1)
    for i in ids:
        err_copy, err_avg = s[f"{i}_err"]
        for t in range(len(s[f"{i}_source"])):
            row = np.concatenate([
                label(s[f"{i}_source"][t], "ORIGINAL (what the camera saw)", f"clip {i}"),
                label(s[f"{i}_copy"][t], "COPY: body driven by V-JEPA 2 -> z", f"body error {err_copy:.2f} m"),
                label(s[f"{i}_average"][t], "BASELINE: average behaviour", f"body error {err_avg:.2f} m"),
            ], 1)
            w.append_data(row)
        for _ in range(4):
            w.append_data(np.zeros_like(row))
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
