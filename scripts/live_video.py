"""Three-panel video of a live run: the body in the house | its head camera with what SAM 3.1 recognised | the map
it built from its own depth camera, with remembered objects pinned where it placed them.

    python scripts/live_video.py --run runs/live/gate60 --out runs/live/gate60/live.mp4
"""
from __future__ import annotations

import argparse
import json

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw

H = 360
PANEL = 360


def unpack(m, key, shape):
    return np.unpackbits(m[key])[: shape[0] * shape[1]].reshape(shape).astype(bool)


def map_panel(seen, walk, origin, res, trail, pins, goal, xy, yaw, crop_c, crop_half):
    img = np.zeros(seen.shape + (3,), np.uint8)
    img[seen] = (70, 70, 80)
    img[walk] = (190, 200, 190)
    c0 = np.floor((crop_c - origin) / res).astype(int)
    n = int(crop_half / res)
    sub = img[max(c0[0] - n, 0):c0[0] + n, max(c0[1] - n, 0):c0[1] + n]
    im = Image.fromarray(np.ascontiguousarray(np.transpose(sub, (1, 0, 2))[::-1])).resize((PANEL, PANEL), Image.NEAREST)
    dr = ImageDraw.Draw(im)
    sc = PANEL / (2 * n)

    def px(p):
        ij = (np.asarray(p) - origin) / res - (c0 - n)
        return float(ij[0] * sc), float(PANEL - ij[1] * sc)
    if len(trail) > 1:
        dr.line([px(p) for p in trail], fill=(255, 120, 40), width=3)
    for lab, p in pins:
        x, y = px(p)
        dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(40, 120, 255))
        dr.text((x + 6, y - 6), lab, fill=(20, 40, 120))
    if goal is not None:
        x, y = px(goal)
        dr.rectangle([x - 5, y - 5, x + 5, y + 5], outline=(220, 0, 160), width=2)
    x, y = px(xy)
    dr.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 60, 0))
    dr.line([x, y, x + 14 * np.cos(yaw), y - 14 * np.sin(yaw)], fill=(255, 60, 0), width=3)
    dr.rectangle([0, 0, PANEL, 18], fill=(0, 0, 0))
    dr.text((6, 3), "its own map (from head depth) + what it remembers", fill=(255, 255, 255))
    return np.asarray(im)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    log = json.load(open(f"{a.run}/log.json"))
    mem = json.load(open(f"{a.run}/memory.json"))
    m = np.load(f"{a.run}/maps.npz")
    shape, origin, res = tuple(m["shape"]), m["origin"], float(m["res"])
    map_t = m["t"]
    looks = log["looks"]
    S = {s["id"]: s for s in mem["sightings"]}
    first_seen = {tr["id"]: (S[tr["sightings"][0]]["t"], tr["label"], S[tr["sightings"][-1]]["xyz"][:2]) for tr in mem["tracks"]}
    all_xy = np.array([lk["xy"] for lk in looks])
    crop_c = (all_xy.min(0) + all_xy.max(0)) / 2
    crop_half = max(float(np.ptp(all_xy, 0).max()) / 2 + 4.0, 6.0)
    waypoints = [(e["t"], np.array(e["xy"])) for e in log["events"] if e["kind"] == "new_waypoint"]
    captions = [(e["t"], {"new_object": lambda e: f"t={e['t']:.0f}s  I see a {e['label']} (remembered)",
                          "stuck": lambda e: f"t={e['t']:.0f}s  I'm stuck, trying another way",
                          "new_waypoint": lambda e: f"t={e['t']:.0f}s  heading to an unexplored opening",
                          "expected_not_seen": lambda e: f"t={e['t']:.0f}s  the {e['label']} is not where I saw it",
                          "fell": lambda e: f"t={e['t']:.0f}s  I fell"}[e["kind"]](e)) for e in log["events"]]
    reader = imageio.get_reader(f"{a.run}/follow.mp4")
    w = imageio.get_writer(a.out, fps=15, codec="libx264", quality=8, macro_block_size=1)
    cache_map = {}
    for i, fr in enumerate(reader):
        t = i * 2 / 30
        li = min(int(t / 0.5), len(looks) - 1)
        lk = looks[li]
        head = Image.open(f"{a.run}/frame_{lk['frame']:04d}.jpg").convert("RGB")
        dets_look = next((looks[j] for j in range(li, -1, -1) if looks[j]["dets"]), None)
        dr = ImageDraw.Draw(head)
        if dets_look is not None and lk["t"] - dets_look["t"] < 0.6:
            for d in dets_look["dets"]:
                x0, y0, x1, y1 = d["box"]
                dr.rectangle([x0, y0, x1, y1], outline=(255, 220, 0), width=3)
                dr.text((x0 + 3, max(y0 - 12, 0)), f"{d['label']} {d['score']:.2f}", fill=(255, 220, 0))
        head = head.resize((H, H))
        hd = ImageDraw.Draw(head)
        hd.rectangle([0, 0, H, 18], fill=(0, 0, 0))
        hd.text((6, 3), "head camera  |  SAM 3.1 sees", fill=(255, 255, 255))
        mi = min(int(np.searchsorted(map_t, t, side="right") - 1), len(map_t) - 1)
        key = (mi, li)
        if key not in cache_map:
            seen = unpack(m, f"seen_{max(mi, 0)}", shape)
            walk = unpack(m, f"walk_{max(mi, 0)}", shape)
            trail = [looks[j]["xy"] for j in range(li + 1)]
            pins = [(lab, xy) for (t0, lab, xy) in first_seen.values() if t0 <= t]
            goal = next((g for tg, g in reversed(waypoints) if tg <= t), None)
            cache_map[key] = map_panel(seen, walk, origin, res, trail, pins, goal, np.array(lk["xy"]), lk["yaw"], crop_c, crop_half)
        body = Image.fromarray(fr).resize((int(fr.shape[1] * H / fr.shape[0]), H))
        bd = ImageDraw.Draw(body)
        bd.rectangle([0, 0, body.width, 18], fill=(0, 0, 0))
        bd.text((6, 3), "Meta Motivo body, physics (MuJoCo)", fill=(255, 255, 255))
        top = np.concatenate([np.asarray(body), np.asarray(head), cache_map[key]], 1)
        bar = Image.new("RGB", (top.shape[1], 56), (15, 15, 20))
        bd = ImageDraw.Draw(bar)
        recent = [c for tc, c in captions if tc <= t][-2:]
        for k, c in enumerate(recent):
            bd.text((10, 8 + 22 * k), c, fill=(230, 230, 230) if k == len(recent) - 1 else (150, 150, 150))
        bd.text((top.shape[1] - 330, 8), f"unseen house  |  no instructions  |  t = {t:4.1f} s", fill=(200, 200, 120))
        bd.text((top.shape[1] - 330, 30), f"objects remembered: {sum(1 for v in first_seen.values() if v[0] <= t)}",
                fill=(200, 200, 120))
        w.append_data(np.concatenate([top, np.asarray(bar)], 0))
    w.close()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
