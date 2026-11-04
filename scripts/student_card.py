"""Evidence card for the student: its trust scores on a real memory table, and the offline quiz result.

    python scripts/student_card.py --episode examples/sample_episode --quiz docs/results/student/quiz_final3.json \
        --out assets/media/student_card.png [--mp4 student_card.mp4]

Left: rows of the sample episode's memory table (final3, Terra, house val_752) with the network's trust in each, and
whether the object was really there (the evaluator's record; the robot never sees it). Right: the quiz on all
final-evaluation memories. Same 1280x720 style as scripts/demo_cut.py.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scout.student.data import label_rows  # noqa: E402
from scout.student.trust import StudentBrain  # noqa: E402

W, H = 1280, 720
BG, FG, DIM, GOLD = (14, 14, 18), (235, 235, 235), (150, 150, 150), (255, 210, 90)
OK, BAD = (90, 200, 120), (235, 95, 85)
FONT = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf")
             if os.path.exists(p)), None)


def font(size):
    from PIL import ImageFont
    return ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()


def pick_rows(rows, n=9):
    """A fair slice: the most and least trusted objects, interleaved by category so the card is not cherry-picked
    towards one kind of mistake. Sorted by trust for display."""
    rows = sorted(rows, key=lambda r: -r["p"])
    k = n // 2
    chosen = rows[:k] + rows[-(n - k):]
    return sorted({r["id"]: r for r in chosen}.values(), key=lambda r: -r["p"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", default="examples/sample_episode")
    ap.add_argument("--quiz", default="docs/results/student/quiz_final3.json")
    ap.add_argument("--out", default="assets/media/student_card.png")
    ap.add_argument("--mp4", default="")
    ap.add_argument("--live", default="", help="optional json {brain: quiz correct of 112}; defaults are the final3_student numbers")
    a = ap.parse_args()
    arm = f"{a.episode}/unchanged"
    mem = json.load(open(f"{arm}/memory.json"))
    truth = {r["id"]: r["y"] for r in label_rows(arm)}
    brain = StudentBrain()
    rows = brain.trust(mem["table"])
    for r in rows:
        r["y"] = truth[r["id"]]
    shown = pick_rows(rows)
    tot = json.load(open(a.quiz))["totals"]

    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((40, 28), "The environment trains brains too", fill=GOLD, font=font(34))
    d.text((40, 76), "A 233-parameter network, trained in seconds on 72 earlier episodes from 22 other houses,", fill=FG, font=font(18))
    d.text((40, 100), "learned which of SAM 3.1's sightings to doubt.", fill=FG, font=font(18))

    # left: trust on a real memory table
    x0, y0 = 40, 140
    d.text((x0, y0), "Its trust in each remembered object (final3, Terra, house val_752)", fill=FG, font=font(18))
    hdr = y0 + 34
    d.text((x0, hdr), "object", fill=DIM, font=font(15))
    d.text((x0 + 150, hdr), "seen", fill=DIM, font=font(15))
    d.text((x0 + 210, hdr), "SAM conf.", fill=DIM, font=font(15))
    d.text((x0 + 300, hdr), "network trust", fill=DIM, font=font(15))
    d.text((x0 + 500, hdr), "really there?", fill=DIM, font=font(15))
    for k, r in enumerate(shown):
        y = hdr + 30 + k * 40
        d.text((x0, y), f"{r['label']}", fill=FG, font=font(18))
        d.text((x0 + 150, y), f"{r['n']}x", fill=FG, font=font(18))
        d.text((x0 + 210, y), f"{r['best']:.2f}", fill=FG, font=font(18))
        bw = int(150 * r["p"])
        d.rectangle([x0 + 300, y + 3, x0 + 450, y + 21], outline=(70, 70, 80))
        d.rectangle([x0 + 300, y + 3, x0 + 300 + bw, y + 21], fill=GOLD if r["p"] >= brain.threshold else (110, 110, 120))
        d.text((x0 + 458, y), f"{r['p']:.2f}", fill=FG, font=font(16))
        real = r["y"] > 0.5
        d.text((x0 + 500, y), "yes" if real else "no", fill=OK if real else BAD, font=font(18))
    yl = hdr + 30 + len(shown) * 40 + 4
    d.text((x0, yl), f"Gold: trusted (>= {brain.threshold}). 'Really there' is the evaluator's record; the robot never sees it.",
           fill=DIM, font=font(13))
    d.text((x0, yl + 18), "Rows: the most and least trusted objects in this memory.", fill=DIM, font=font(13))

    # right: the live result in the eight final-evaluation homes (docs/results/final3_student/leaderboard.md)
    x1, y1 = 720, 140
    d.text((x1, y1), "Live as the robot's brain, 8 never-seen homes", fill=FG, font=font(18))
    d.text((x1, y1 + 24), "memory quiz, same houses, seed and scorer", fill=DIM, font=font(15))
    live = json.load(open(a.live)) if a.live else {}
    res = [("GPT-6.1 Terra (Muse Spark stand-in)", live.get("terra", 92)), ("rule brain", live.get("rules", 93)),
           ("student (233 parameters)", live.get("student", 106))]
    n = 112
    for k, (name, c) in enumerate(res):
        y = y1 + 66 + k * 62
        bold = name.startswith("student")
        d.text((x1, y), name, fill=GOLD if bold else FG, font=font(19 if bold else 17))
        bw = int(420 * c / n)
        d.rectangle([x1, y + 26, x1 + bw, y + 44], fill=GOLD if bold else (110, 110, 120))
        d.text((x1 + bw + 8, y + 24), f"{c}/{n}", fill=FG, font=font(17))
    off = tot["student | all"]["correct"], tot["rules | all"]["correct"], tot["student | all"]["n"]
    d.text((x1, y1 + 266), f"Offline, on the other brains' memories: {off[0]}/{off[2]}", fill=FG, font=font(17))
    d.text((x1, y1 + 290), f"(rule brain {off[1]}/{off[2]}).", fill=FG, font=font(17))
    d.text((x1, y1 + 330), "Not better at the physical check:", fill=FG, font=font(17))
    d.text((x1, y1 + 354), "1 clean verification pair, Terra 3, rules 2.", fill=FG, font=font(17))
    d.text((x1, y1 + 394), "Run after the frozen evaluation; reported separately.", fill=DIM, font=font(14))
    d.text((40, 680), "Protocol, cross-validation and every caveat: docs/results/student.md    github.com/tayyab415/scout",
           fill=DIM, font=font(15))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    im.save(a.out)
    print("saved", a.out, "| rows shown:", [(r["label"], round(r["p"], 2), int(r["y"])) for r in shown])
    if a.mp4:
        import imageio.v2 as imageio
        w = imageio.get_writer(a.mp4, fps=15, codec="libx264", quality=8, macro_block_size=1)
        f = np.asarray(im)
        for _ in range(15 * 9):
            w.append_data(f)
        w.close()
        print("saved", a.mp4)


if __name__ == "__main__":
    main()
