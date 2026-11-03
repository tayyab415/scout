"""Compose a synchronised video of one environment episode arm: stable observer view (walls cut away) | head camera
with SAM 3.1 detections on their own frames | the agent's self-built map with memory as known at that moment.
Every panel is chosen by simulation time t, so nothing shows information the agent did not have yet.

    python scripts/env_video.py --run <episode dir> --arm moved --out out.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, "/pkg")
W, H = 1280, 720
FONT = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf")
             if os.path.exists(p)), None)


def font(size):
    from PIL import ImageFont
    return ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()


def _confident(ss) -> bool:
    """Same rule as memory (Track.confident), using only the sightings available at that time."""
    if len(ss) < 3:
        return False
    views = np.array([s["body_xy"] for s in ss], float)
    pos = np.array([s["xyz"][:2] for s in ss], float)
    spread = float(np.linalg.norm(pos - np.median(pos, 0), axis=1).mean())
    return float(np.linalg.norm(views.max(0) - views.min(0))) >= 1.0 and spread < 0.5


class State:
    """What the agent knew at time t, rebuilt from its own logs."""

    def __init__(self, arm_dir: str):
        self.d = arm_dir
        self.log = json.load(open(f"{arm_dir}/log.json"))
        self.mem = json.load(open(f"{arm_dir}/memory.json"))
        self.looks = self.log["looks"]
        self.look_t = np.array([lk["t"] for lk in self.looks])
        self.S = {s["id"]: s for s in self.mem["sightings"]}
        self.status_t = {}  # track -> (t, status) when a check changed its status
        for e in self.log["events"]:
            if e["kind"] == "check_result" and e.get("status") in ("retracted", "gone"):
                self.status_t[e["track"]] = (e["t"], e["status"])

    def head_frame(self, t: float):
        """The latest head frame captured at or before t, or None before the first capture."""
        i = int(np.searchsorted(self.look_t, t + 1e-9, side="right") - 1)
        return self.looks[i]["frame"] if i >= 0 else None

    def boxes_for_frame(self, frame: int) -> list:
        lk = self.looks[frame]
        return list(lk.get("dets") or [])

    def pins(self, t: float, confident_only: bool = False):
        out = []
        for tr in self.mem["tracks"]:
            ss = [self.S[i] for i in tr["sightings"] if self.S[i]["t"] <= t + 1e-9]
            if not ss:
                continue
            st = self.status_t.get(tr["id"])
            if st and st[0] <= t:
                continue  # retracted or found gone by then
            if confident_only and not _confident(ss):
                continue
            out.append((tr["label"], list(ss[-1]["xyz"][:2])))
        return out

    def active_check(self, t: float):
        """Label of the object being checked at time t (from the agent's own check events), else None."""
        cur = None
        for e in self.log["events"]:
            if e["t"] > t + 1e-9:
                break
            if e["kind"] == "check_start":
                cur = e["label"]
            elif e["kind"] in ("check_result", "check_unreachable"):
                cur = None
        return cur

    def events_until(self, t: float, kinds=None):
        return [e for e in self.log["events"] if e["t"] <= t + 1e-9 and (kinds is None or e["kind"] in kinds)]


CAPTION = {
    "new_object": lambda e: f"I see a {e['label']}",
    "stuck": lambda e: "stuck - trying another way",
    "new_waypoint": lambda e: "heading to an unexplored opening",
    "expected_not_seen": lambda e: f"the {e['label']} is not where I saw it?",
    "fell": lambda e: "I fell",
    "got_up": lambda e: f"got back up ({e['seconds']:.1f} s)",
    "explored_all": lambda e: "no unexplored openings left",
    "check_start": lambda e: (f"asked - going back to look at the {e['label']}" if e.get("question") else f"going back to check the {e['label']}"),
    "check_result": lambda e: f"checked the {e['label']}: {e['result']}",
    "question": lambda e: f'QUESTION: "{e["text"]}"',
    "answer": lambda e: f'ANSWER: "{e["say"]}"',
}


def map_panel(m, idx, st: State, t, size, meta):
    from PIL import Image, ImageDraw
    shape, origin, res = tuple(m["shape"]), m["origin"], float(m["res"])
    seen = np.unpackbits(m[f"seen_{idx}"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    walk = np.unpackbits(m[f"walk_{idx}"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    img = np.full(shape + (3,), 20, np.uint8)
    img[seen] = (70, 72, 82)
    img[walk] = (196, 206, 196)
    allxy = np.array(m["xy"])
    c0 = np.floor(((allxy.min(0) + allxy.max(0)) / 2 - origin) / res).astype(int)
    n = int(max(float(np.ptp(allxy, 0).max()) / 2 + 4.0, 6.0) / res)
    sub = img[max(c0[0] - n, 0):c0[0] + n, max(c0[1] - n, 0):c0[1] + n]
    im = Image.fromarray(np.ascontiguousarray(np.transpose(sub, (1, 0, 2))[::-1])).resize((size, size), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    sc = size / (2 * n)

    def px(p):
        ij = (np.asarray(p) - origin) / res - (c0 - n)
        return float(ij[0] * sc), float(size - ij[1] * sc)
    trail = [px(p) for p in allxy[: idx + 1]]
    if len(trail) > 1:
        dr.line(trail, fill=(255, 130, 40), width=3)
    for lab, p in st.pins(t):
        x, y = px(p)
        dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(40, 110, 230))
    mm = meta[idx]
    tgt = st.active_check(t)
    if mm.get("check"):
        x, y = px(mm["check"])
        dr.ellipse([x - 11, y - 11, x + 11, y + 11], outline=(230, 0, 150), width=4)
        if tgt:
            f = font(18)
            w_ = dr.textlength(tgt, font=f)
            dr.rectangle([x + 12, y - 12, x + 20 + w_, y + 12], fill=(255, 255, 255))
            dr.text((x + 16, y - 11), tgt, fill=(160, 0, 110), font=f)
    x, y = px(allxy[idx])
    yaw = float(m["yaw"][idx])
    dr.ellipse([x - 7, y - 7, x + 7, y + 7], fill=(255, 60, 0))
    dr.line([x, y, x + 16 * np.cos(yaw), y - 16 * np.sin(yaw)], fill=(255, 60, 0), width=3)
    return im


def compose(run_dir: str, arm: str, out: str, title: str = ""):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw
    adir = f"{run_dir}/{arm}"
    st = State(adir)
    m = np.load(f"{adir}/maps.npz")
    meta = json.load(open(f"{adir}/map_meta.json"))
    man = json.load(open(f"{run_dir}/manifest.json"))
    times = st.log["observer_times_prefix"] + st.log["observer_times"]
    readers = [imageio.get_reader(f"{run_dir}/observer_prefix.mp4")]
    if os.path.exists(f"{run_dir}/observer_{arm}.mp4"):
        readers.append(imageio.get_reader(f"{run_dir}/observer_{arm}.mp4"))
    frames = (fr for r in readers for fr in r)
    w = imageio.get_writer(out, fps=15, codec="libx264", quality=8, macro_block_size=1)
    big, mid, small = font(26), font(20), font(16)
    mt = np.array(m["t"])
    brain = man["brain"]
    moved = (man.get("arms", {}).get(arm) or {}).get("moved")
    for t, fr in zip(times, frames):
        canvas = Image.new("RGB", (W, H), (14, 14, 18))
        dr = ImageDraw.Draw(canvas)
        obs = Image.fromarray(fr).resize((640, 480))
        canvas.paste(obs, (0, 40))
        fi = st.head_frame(t)
        if fi is None:
            head = Image.new("RGB", (512, 512), (30, 30, 34))
            ImageDraw.Draw(head).text((150, 240), "no head image yet", fill=(160, 160, 160), font=mid)
        else:
            head = Image.open(f"{adir}/frames/{fi:04d}.jpg").convert("RGB")
        hd = ImageDraw.Draw(head)
        for b in (st.boxes_for_frame(fi) if fi is not None else []):
            x0, y0, x1, y1 = b["box"]
            hd.rectangle([x0, y0, x1, y1], outline=(255, 220, 0), width=4)
            hd.text((x0 + 4, max(y0 - 22, 0)), f"{b['label']} {b['score']:.2f}", fill=(255, 220, 0), font=font(20))
        canvas.paste(head.resize((320, 320)), (640, 40))
        mi = max(int(np.searchsorted(mt, t + 1e-9, side="right") - 1), 0)
        canvas.paste(map_panel(m, mi, st, t, 320, meta), (960, 40))
        dr.text((8, 8), title or f"Scout environment  |  {man['house']}  |  brain: {brain}", fill=(235, 235, 235), font=mid)
        dr.text((648, 362), "head camera + SAM 3.1", fill=(170, 170, 170), font=small)
        dr.text((968, 362), "its own map + memory", fill=(170, 170, 170), font=small)
        import textwrap
        evs = [e for e in st.events_until(t) if e["kind"] in CAPTION]
        lines = []
        for e in evs[-5:]:
            txt = f"t={e['t']:5.1f}s  " + CAPTION[e["kind"]](e)
            col = (120, 230, 140) if e["kind"] == "answer" else ((255, 210, 90) if e["kind"] == "question" else (225, 225, 225))
            for k, ln in enumerate(textwrap.wrap(txt, 56)):
                lines.append((("      " if k else "") + ln, col))
        y = 396
        for ln, col in lines[-8:]:
            dr.text((652, y), ln, fill=col, font=small)
            y += 22
        if moved and t >= moved["t"]:
            dr.text((8, 530), f"[evaluator, hidden from the agent] moved the {moved['category']} at t={moved['t']:.0f}s",
                    fill=(255, 120, 120), font=mid)
        dr.text((8, 560), "exploration: nearest-frontier rule  |  re-checks + answers: " + brain, fill=(170, 170, 170), font=small)
        dr.text((8, 585), "body: Meta Motivo (physics)  |  eyes: SAM 3.1  |  sensing: RGB-D head camera, perfect odometry (disclosed)",
                fill=(170, 170, 170), font=small)
        dr.text((8, 640), f"simulation time {t:6.1f} s", fill=(200, 200, 120), font=big)
        w.append_data(np.asarray(canvas))
    # evidence cards for each answer: question, answer, the old sighting and the new look, full width
    for q in [q for q in st.log["qa"] if q["kind"] == "answer"]:
        card = Image.new("RGB", (W, H), (14, 14, 18))
        dr = ImageDraw.Draw(card)
        ques = next((x["text"] for x in st.log["qa"] if x["kind"] == "question"), "")
        import json
        import os
        import textwrap

        from scout.env.scoring import score_verify  # the card uses the scorer's own grading
        ep_dir = os.path.dirname(adir.rstrip("/"))
        man_ = json.load(open(f"{ep_dir}/manifest.json")) if os.path.exists(f"{ep_dir}/manifest.json") else {}
        ev_ = json.load(open(f"{adir}/evaluator.json")) if os.path.exists(f"{adir}/evaluator.json") else {}
        colour = {"correct": (120, 230, 140), "wrong": (255, 110, 110), "wrong target": (255, 110, 110),
                  "abstain": (255, 200, 90)}

        def graded(answer):
            o = score_verify(arm, dict(qa=[answer]), st.mem, man_.get("target"), ev_.get("moves"), ev_)["outcome"]
            return o, colour.get(o, (200, 200, 200))
        dr.text((30, 16), f'Q: "{ques}"', fill=(255, 210, 90), font=mid)
        rec = (st.log.get("recall") or [None])[-1]
        y = 48
        if rec:
            o, col = graded(rec)
            for ln in textwrap.wrap(f'Without looking (memory-persistence rule) [{o}]: "{rec["say"]}"', 100)[:3]:
                dr.text((30, y), ln, fill=col, font=small)
                y += 20
        o, col = graded(q)
        for ln in textwrap.wrap(f'After walking back to look [{o}]: "{q["say"]}"', 100)[:3]:
            dr.text((30, y), ln, fill=col, font=small)
            y += 20
        dr.text((30, y + 2), f"truth: {'the evaluator moved it' if arm == 'moved' else 'nothing was moved'}   |   "
                f"verdict scored: {q.get('result')}   |   brain: {brain}", fill=(200, 200, 200), font=small)
        label_y = y + 30                       # image captions sit below the measured text
        img_y = label_y + 20
        side = min(520, H - img_y - 40)
        bf = q.get("best_frame", q.get("before_frame"))
        if bf is not None:
            b = Image.open(f"{adir}/frames/{bf:04d}.jpg").convert("RGB")
            bd = ImageDraw.Draw(b)
            for det in st.boxes_for_frame(bf):
                if det.get("track") == q.get("track"):
                    bd.rectangle(det["box"], outline=(255, 220, 0), width=5)
            card.paste(b.resize((side, side)), (40, img_y))
            dr.text((40, label_y), f"what it remembered: its best sighting (t={st.looks[bf]['t']:.0f}s)", fill=(200, 200, 200), font=small)
        if q.get("now_frames"):
            nf = q["now_frames"][-1]
            lk = st.looks[nf]
            b = Image.open(f"{adir}/frames/{nf:04d}.jpg").convert("RGB")
            bd = ImageDraw.Draw(b)
            if lk.get("check_uv"):
                u, v = lk["check_uv"]
                bd.ellipse([u - 18, v - 18, u + 18, v + 18], outline=(230, 0, 150), width=5)
            if lk.get("zoom"):
                bd.rectangle(lk["zoom"]["box"], outline=(120, 200, 255), width=3)
            for det in st.boxes_for_frame(nf):
                if det.get("track") == q.get("track"):
                    bd.rectangle(det["box"], outline=(255, 220, 0), width=5)
            card.paste(b.resize((side, side)), (680, img_y))
            dr.text((680, label_y), f"when it went back (t={lk['t']:.0f}s); magenta = remembered spot", fill=(200, 200, 200), font=small)
        ev = q.get("evidence")
        if ev:
            dr.text((30, img_y + side + 8), f"evidence over {ev['views']} looks at the spot: recognised {ev['hits']}, depth saw past it "
                    f"{ev['through']}, something there {ev['occupied']}", fill=(170, 170, 170), font=small)
        for _ in range(15 * 6):
            w.append_data(np.asarray(card))
    w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", default="moved")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    compose(a.run, a.arm, a.out)


if __name__ == "__main__":
    main()
