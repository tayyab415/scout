"""Assemble the demo video from one recorded episode and a leaderboard. Every frame comes from a real recording.

    python scripts/demo_cut.py --episode <dir with unchanged.mp4, moved.mp4, manifest.json> --leaderboard leaderboard.md --out demo.mp4

Sections (about 3 minutes; the narrator records the voice over it, see docs/demo_script.md):
  0:00  the moved arm's evidence card (what it remembered | what it found when it went back), as the hook
  0:12  title card
  0:18  exploration, the three-panel view (unchanged arm, sped up 3x)
  ~1:05 the evaluator's hidden move and the question, then the walk back and the look (moved arm, real time)
  ~1:50 both evidence cards: memory-persistence answer vs checked answer, moved and unchanged
  ~2:25 leaderboard card
  ~2:45 disclosure card
"""
from __future__ import annotations

import argparse
import json
import os
import textwrap

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

W, H, FPS = 1280, 720, 15
FONT = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf")
             if os.path.exists(p)), None)


def font(size):
    from PIL import ImageFont
    return ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()


def card(lines, sizes=None, colors=None):
    im = Image.new("RGB", (W, H), (14, 14, 18))
    dr = ImageDraw.Draw(im)
    y = 120
    for k, ln in enumerate(lines):
        sz = (sizes or [34] * len(lines))[k]
        col = (colors or [(235, 235, 235)] * len(lines))[k]
        for wl in textwrap.wrap(ln, int(1900 / sz)) or [""]:
            dr.text((80, y), wl, fill=col, font=font(sz))
            y += int(sz * 1.35)
        y += 10
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", required=True)
    ap.add_argument("--leaderboard", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--branch", default="", help="branch shot from scripts/branch_video.py (both worlds side by side)")
    a = ap.parse_args()
    man = json.load(open(f"{a.episode}/manifest.json"))
    card_len = FPS * 6  # env_video appends each evidence card for 6 s at the end of an arm video
    rm, ru = imageio.get_reader(f"{a.episode}/moved.mp4"), imageio.get_reader(f"{a.episode}/unchanged.mp4")
    nm, nu = rm.count_frames(), ru.count_frames()
    mv_card, un_card = rm.get_data(nm - card_len // 2), ru.get_data(nu - card_len // 2)
    n_prefix = int(man["explore_s"] * FPS)  # observer frames: 15 per simulated second
    w = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=8, macro_block_size=1)

    def put(img, secs=None, n=None):
        for _ in range(n if n is not None else int(secs * FPS)):
            w.append_data(img)

    branch = []
    if a.branch:
        rb = imageio.get_reader(a.branch)
        nb = rb.count_frames()
        branch_card = rb.get_data(nb - 3)
        split_frames = nb - FPS * 8  # branch_video appends an 8 s verdict card
    if a.branch:
        for i, f in enumerate(rb):   # hook: the last 10 s of the split, both worlds answering
            if split_frames - FPS * 8 <= i < split_frames:
                w.append_data(f)
        put(branch_card, 3)
    else:
        put(mv_card, 10)
    tgt = man.get("target") or {}
    put(card(["Scout", "A physics-simulated home that tests whether a robot's memory is still true.",
              "Body: Meta Motivo, in physics. Eyes: SAM 3.1. Brain: plug in yours.",
              f"This episode: house {man['house']}, never seen before  |  brain: {man['brain']}"], [64, 32, 26, 20],
             [(255, 210, 90), (235, 235, 235), (200, 200, 200), (150, 150, 150)]), 5)
    speed = max(1, int(np.ceil(n_prefix / (FPS * 20))))  # about 20 s of exploration on screen
    for i, f in enumerate(ru):      # exploration (shared by both arms), sped up and labelled
        if i >= n_prefix:
            break
        if i % speed == 0:
            im = Image.fromarray(f)
            d = ImageDraw.Draw(im)
            d.rectangle([W - 230, 44, W - 10, 84], fill=(0, 0, 0))
            d.text((W - 220, 50), f"{speed}x playback", fill=(255, 210, 90), font=font(26))
            w.append_data(np.asarray(im))
    if a.branch:                    # the split, both worlds in real time, then the graded verdict
        for i, f in enumerate(imageio.get_reader(a.branch)):
            if i >= split_frames:
                break
            w.append_data(f)
        put(branch_card, 6)
    else:
        for i, f in enumerate(rm):  # the hidden move, the question, the walk back, the look: real time
            if i >= nm - card_len:
                break
            if i >= n_prefix:
                w.append_data(f)
    put(mv_card, 7)
    put(un_card, 7)
    rows = [ln for ln in open(a.leaderboard).read().splitlines() if ln.startswith("| ") and "---" not in ln][:5]
    cols = [c.strip() for c in rows[0].strip("|").split("|")]
    keep = [i for i, c in enumerate(cols) if c.startswith(("brain", "episodes planned", "verify unchanged", "verify moved",
                                                             "memory-persistence", "quiz accuracy", "affirmative", "falls"))]
    im = Image.new("RGB", (W, H), (14, 14, 18))
    dr = ImageDraw.Draw(im)
    dr.text((40, 30), "Leaderboard: unseen homes, same body, same eyes, same budgets", fill=(255, 210, 90), font=font(30))
    xs = np.linspace(40, W - 140, len(keep)).astype(int)
    for r, row in enumerate(rows):
        cells = [c.strip() for c in row.strip("|").split("|")]
        for x, i in zip(xs, keep):
            txt = cells[i] if i < len(cells) else ""
            for k, ln in enumerate(textwrap.wrap(txt, 15 if r == 0 else 22)[:6]):
                dr.text((x, 100 + (r * 90 + (40 if r else 0)) + k * 19), ln, fill=(255, 255, 255) if r else (170, 170, 170),
                        font=font(18 if r else 14))
    dr.text((40, 640), "verify columns: correct / abstain / wrong / other.  Full outcome breakdown and every failure: docs/results/",
            fill=(150, 150, 150), font=font(16))
    put(np.asarray(im), 14)
    put(card(["Plug in your own brain",
              "from scout.brains import register_brain",
              "class MyBrain:  decide(memory, state)   resolve(memory, question)   phrase(...)   answer(memory, question)",
              "register_brain('mybrain', MyBrain)",
              "python scripts/run_local.py --brain-module mybrain.py --brain mybrain --house procthor-10k-val_val_752 --tag mine",
              "...or any Meta Model API chat model:  --brain llm:<model id>",
              "Score a recorded episode on a laptop:  python scripts/score_episode.py examples/sample_episode",
              "github.com/tayyab415/scout"], [34, 19, 19, 19, 17, 19, 19, 30],
             [(255, 210, 90)] + [(200, 230, 255)] * 6 + [(255, 210, 90)]), 9)
    put(card(["Built during the hackathon (Oct 27 - Nov 4)",
              "The environment: head RGB-D camera, the robot's own depth map, SAM 3.1 placed in 3D, episodic memory",
              "The body controller on Meta Motivo: blended skills, turning to look, getting back up after falls",
              "The evaluator: object identities, per-frame visibility, hidden moves, branching one episode into two worlds",
              "Scoring, the plug-in brain interface, the synchronised videos, this demo",
              "Used as is: Meta Motivo M-1, SAM 3.1, ProcTHOR houses (MolmoSpaces), Muse Spark 1.3 and Llama 4 on Meta Model API"],
             [34, 22, 22, 22, 22, 20], [(255, 210, 90)] + [(235, 235, 235)] * 4 + [(170, 170, 170)]), 8)
    put(card(["What is real and what is given",
              "Exploration uses a fixed nearest-frontier rule for every brain. Brains decide re-checks and answer.",
              "Depth and the body's pose come from the simulator (perfect odometry). Object recognition is SAM 3.1 on the head camera.",
              "'Not found' means the depth camera now sees past the remembered spot. It never claims to have seen the move.",
              "The brain in these runs is Muse Spark 1.3 on Meta Model API. Any model plugs in.",
              "The memory-persistence answer is a rule: whatever memory says, without looking.",
              "Targets are objects with no look-alike within 2.5 m, so the question names one identifiable object."],
             [34, 22, 22, 22, 22, 22, 22]), 9)
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
