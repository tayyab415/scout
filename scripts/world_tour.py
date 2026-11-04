"""Montage of recorded random-action episodes: follow camera | head camera, with the action and bump flag."""
import glob
import os
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image, ImageDraw

NAMES = ["stand", "forward", "side-left", "side-right", "back", "walk-turn-right"]


def main(src, out):
    tmp = tempfile.mkdtemp()
    n = 0
    for p in sorted(glob.glob(f"{src}/*.npz")):
        z = np.load(p)
        house = os.path.basename(p)[:-4]
        for k in range(len(z["frames"])):
            fo = Image.fromarray(z["follow"][k]).resize((480, 360))
            hd = Image.fromarray(z["frames"][k]).resize((360, 360))
            canvas = Image.new("RGB", (840, 410), "white")
            canvas.paste(fo, (0, 50))
            canvas.paste(hd, (480, 50))
            d = ImageDraw.Draw(canvas)
            act = NAMES[int(z["actions"][k - 1])] if k > 0 else "start"
            bump = bool(z["contact"][k])
            d.text((8, 6), f"{house}   step {k}   action: {act}", fill="black")
            d.text((8, 24), "follow camera", fill="gray")
            d.text((488, 24), "head camera (what V-JEPA 2 sees)", fill="gray")
            if bump:
                d.rectangle([700, 4, 832, 22], fill="#d9534f")
                d.text((706, 6), "touching house", fill="white")
            for _ in range(2):  # ~3.75 control steps per second -> show each step for 2 video frames at 8 fps
                canvas.save(f"{tmp}/f{n:05d}.png")
                n += 1
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", "8", "-i", f"{tmp}/f%05d.png",
                    "-pix_fmt", "yuv420p", out], check=True)
    print("wrote", out, n, "frames")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
