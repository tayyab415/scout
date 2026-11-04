"""The branch shot: both worlds side by side from the moment the episode splits, time-synchronised.

Left: the unchanged world. Right: the world where the evaluator moved the object. Same robot, same memory, same
question. Each side shows the observer view with the head camera inset and the events as they happen, then both
answers, graded by the scorer.

    python scripts/branch_video.py --run <episode dir> --out branch.mp4 [--max-seconds 70]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from env_video import State, font  # noqa: E402

W, H, FPS = 1280, 720, 15
GREEN, RED, AMBER, GOLD, GREY = (120, 230, 140), (255, 110, 110), (255, 200, 90), (255, 210, 90), (190, 190, 190)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-seconds", type=float, default=75.0)
    a = ap.parse_args()
    from scout.env.scoring import score_verify
    man = json.load(open(f"{a.run}/manifest.json"))
    tgt = man["target"]
    sides = {}
    for arm in ("unchanged", "moved"):
        st = State(f"{a.run}/{arm}")
        ev = json.load(open(f"{a.run}/{arm}/evaluator.json"))
        rd = imageio.get_reader(f"{a.run}/observer_{arm}.mp4")
        ans = [q for q in st.log["qa"] if q["kind"] == "answer"]
        rec = (st.log.get("recall") or [None])[-1]
        grade = lambda q: score_verify(arm, dict(qa=[q]), st.mem, tgt, ev.get("moves"), ev)["outcome"] if q else None  # noqa: E731
        sides[arm] = dict(st=st, rd=rd, times=st.log["observer_times"], ans=ans[-1] if ans else None, rec=rec,
                          g_ans=grade(ans[-1]) if ans else None, g_rec=grade(rec))
    n = min(len(sides["unchanged"]["times"]), len(sides["moved"]["times"]), int(a.max_seconds * FPS))
    t_move = (json.load(open(f"{a.run}/moved/evaluator.json")).get("moves") or [{}])[0].get("t", man["explore_s"])
    w = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
    big, mid, small = font(30), font(20), font(16)
    frames = {arm: iter(sides[arm]["rd"]) for arm in sides}
    for i in range(n):
        canvas = Image.new("RGB", (W, H), (12, 12, 16))
        dr = ImageDraw.Draw(canvas)
        dr.text((24, 10), "Same robot. Same memory. Same question.", fill=GOLD, font=big)
        for k, arm in enumerate(("unchanged", "moved")):
            s = sides[arm]
            t = s["times"][i]
            x0 = k * 640
            obs = Image.fromarray(next(frames[arm])).resize((632, 474))
            canvas.paste(obs, (x0 + 4, 56))
            fi = s["st"].head_frame(t)
            if fi is not None:
                head = Image.open(f"{a.run}/{arm}/frames/{fi:04d}.jpg").convert("RGB").resize((180, 180))
                canvas.paste(head, (x0 + 448, 344))
                dr.rectangle([x0 + 448, 344, x0 + 628, 524], outline=(255, 255, 255), width=2)
            title = "World A: nothing moved" if arm == "unchanged" else f"World B: we moved the {tgt['category']}"
            dr.rectangle([x0 + 4, 56, x0 + 4 + 16 + int(dr.textlength(title, font=mid)), 86], fill=(0, 0, 0))
            dr.text((x0 + 12, 60), title, fill=GREEN if arm == "unchanged" else RED, font=mid)
            if arm == "moved" and t >= t_move:
                dr.text((x0 + 12, 92), f"(the robot was not told; moved at t={t_move:.0f} s)", fill=GREY, font=small)
            evs = [e for e in s["st"].events_until(t) if e["kind"] in ("question", "check_start", "check_result", "answer")]
            y = 540
            for e in evs[-3:]:
                txt = {"question": lambda e: f'Q: "{e["text"]}"',
                       "check_start": lambda e: f"walking back to look at the {e['label']}",
                       "check_result": lambda e: f"looked: {e['result']}",
                       "answer": lambda e: f'A: "{e["say"]}"'}[e["kind"]](e)
                col = GOLD if e["kind"] == "question" else (GREEN if (e["kind"] == "answer" and s["g_ans"] == "correct") else
                                                            (RED if e["kind"] == "answer" else (230, 230, 230)))
                for ln in textwrap.wrap(txt, 58)[:2]:
                    dr.text((x0 + 12, y), ln, fill=col, font=small)
                    y += 20
            dr.text((x0 + 12, 690), f"t = {t:5.1f} s", fill=(200, 200, 120), font=mid)
        dr.line([(640, 50), (640, 710)], fill=(80, 80, 90), width=2)
        w.append_data(np.asarray(canvas))
    # verdict card: both worlds, both ways of answering, graded by the scorer
    card = Image.new("RGB", (W, H), (12, 12, 16))
    dr = ImageDraw.Draw(card)
    q = next((e["text"] for e in sides["moved"]["st"].log["events"] if e["kind"] == "question"), "")
    dr.text((40, 30), f'"{q}"', fill=GOLD, font=big)
    col = lambda g: GREEN if g == "correct" else (AMBER if g == "abstain" else RED)  # noqa: E731
    for k, arm in enumerate(("unchanged", "moved")):
        s = sides[arm]
        x0 = 40 + k * 620
        dr.text((x0, 110), "World A: nothing moved" if arm == "unchanged" else "World B: it was moved", fill=GREY, font=mid)
        dr.text((x0, 160), "Answering from memory:", fill=GREY, font=small)
        for j, ln in enumerate(textwrap.wrap(f'"{s["rec"]["say"]}"' if s["rec"] else "-", 48)[:3]):
            dr.text((x0, 185 + 24 * j), ln, fill=col(s["g_rec"]), font=mid)
        dr.text((x0, 270), f"[{s['g_rec']}]", fill=col(s["g_rec"]), font=mid)
        dr.text((x0, 330), "Walking back to look:", fill=GREY, font=small)
        for j, ln in enumerate(textwrap.wrap(f'"{s["ans"]["say"]}"' if s["ans"] else "-", 48)[:3]):
            dr.text((x0, 355 + 24 * j), ln, fill=col(s["g_ans"]), font=mid)
        dr.text((x0, 440), f"[{s['g_ans']}]", fill=col(s["g_ans"]), font=mid)
    dr.text((40, 560), "Memory can't tell the two worlds apart. Looking can.", fill=(235, 235, 235), font=big)
    dr.text((40, 620), f"brain: {man['brain']}   |   body: Meta Motivo   |   eyes: SAM 3.1   |   house {man['house']} (never seen before)",
            fill=GREY, font=small)
    for _ in range(FPS * 8):
        w.append_data(np.asarray(card))
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
