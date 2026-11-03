"""Compose a synchronised video of one environment episode arm: stable observer view (walls cut away) | head camera
with SAM 3.1 detections on their own frames | the agent's self-built map with memory as known at that moment.
Every panel is chosen by simulation time t, so nothing shows information the agent did not have yet.

    python scripts/env_video.py --run <episode dir> --arm moved --out out.mp4
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

W, H = 1280, 720
FONT = next((p for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf")
             if os.path.exists(p)), None)


def font(size):
    from PIL import ImageFont
    return ImageFont.truetype(FONT, size) if FONT else ImageFont.load_default()


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

    def head_frame(self, t: float) -> int:
        i = int(np.searchsorted(self.look_t, t + 1e-9, side="right") - 1)
        return self.looks[max(i, 0)]["frame"]

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
            if confident_only and len(ss) < 3:
                continue
            out.append((tr["label"], list(ss[-1]["xyz"][:2])))
        return out

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
    f = font(14)
    labelled = set()
    for lab, p in st.pins(t):
        x, y = px(p)
        dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(40, 120, 255))
    for lab, p in st.pins(t, confident_only=True):  # one label per category, confident objects only
        if lab in labelled:
            continue
        labelled.add(lab)
        x, y = px(p)
        dr.text((x + 6, y - 8), lab, fill=(10, 30, 110), font=f)
    mm = meta[idx]
    if mm.get("check"):
        x, y = px(mm["check"])
        dr.ellipse([x - 10, y - 10, x + 10, y + 10], outline=(230, 0, 150), width=3)
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
    readers = [imageio.get_reader(f"{run_dir}/observer_prefix.mp4"), imageio.get_reader(f"{run_dir}/observer_{arm}.mp4")]
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
        head = Image.open(f"{adir}/frames/{fi:04d}.jpg").convert("RGB")
        hd = ImageDraw.Draw(head)
        for b in st.boxes_for_frame(fi):
            x0, y0, x1, y1 = b["box"]
            hd.rectangle([x0, y0, x1, y1], outline=(255, 220, 0), width=4)
            hd.text((x0 + 4, max(y0 - 22, 0)), f"{b['label']} {b['score']:.2f}", fill=(255, 220, 0), font=font(20))
        canvas.paste(head.resize((320, 320)), (640, 40))
        mi = max(int(np.searchsorted(mt, t + 1e-9, side="right") - 1), 0)
        canvas.paste(map_panel(m, mi, st, t, 320, meta), (960, 40))
        dr.text((8, 8), title or f"Scout environment  |  {man['house']}  |  brain: {brain}", fill=(235, 235, 235), font=mid)
        dr.text((648, 362), "head camera + SAM 3.1 (boxes only on their own frame)", fill=(170, 170, 170), font=small)
        dr.text((968, 362), "its own map from depth; dots = memory", fill=(170, 170, 170), font=small)
        evs = [e for e in st.events_until(t) if e["kind"] in CAPTION]
        y = 400
        for e in evs[-5:]:
            txt = f"t={e['t']:5.1f}s  " + CAPTION[e["kind"]](e)
            col = (120, 230, 140) if e["kind"] == "answer" else ((255, 210, 90) if e["kind"] == "question" else (225, 225, 225))
            dr.text((656, y), txt[:70], fill=col, font=small if len(txt) > 50 else mid)
            y += 28
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
        dr.text((30, 24), f'Q: "{ques}"', fill=(255, 210, 90), font=big)
        dr.text((30, 66), f'A: "{q["say"]}"', fill=(120, 230, 140), font=mid)
        dr.text((30, 100), f"verdict: {q.get('result')}   (brain: {brain})", fill=(200, 200, 200), font=small)
        if q.get("before_frame") is not None:
            b = Image.open(f"{adir}/frames/{q['before_frame']:04d}.jpg").convert("RGB").resize((560, 560))
            card.paste(b, (40, 140))
            tb = st.looks[q["before_frame"]]["t"]
            dr.text((40, 120), f"what it remembered (t={tb:.0f}s)", fill=(200, 200, 200), font=small)
        if q.get("now_frames"):
            nf = q["now_frames"][-1]
            b = Image.open(f"{adir}/frames/{nf:04d}.jpg").convert("RGB").resize((560, 560))
            card.paste(b, (680, 140))
            dr.text((680, 120), f"what it saw when it went back (t={st.looks[nf]['t']:.0f}s)", fill=(200, 200, 200), font=small)
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
