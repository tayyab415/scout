"""Training examples for the student, from recorded arms: one row per remembered object, labelled by the evaluator.

The label is the memory quiz's own criterion: the object's category was visible (>= 0.5% of a captured head frame)
and the remembered position is within 1.5 m of a visible instance of that category. The features come from the
memory table only; the label comes from `evaluator.json`, which the agent never sees.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from ..env.scoring import NEAR_M, captured_visibility, truth_at
from .trust import features, parse_table


def label_rows(arm_dir: str) -> list[dict]:
    log = json.load(open(f"{arm_dir}/log.json"))
    mem = json.load(open(f"{arm_dir}/memory.json"))
    ev = json.load(open(f"{arm_dir}/evaluator.json"))
    quiz = log.get("quiz") or []
    t_q = quiz[0]["t"] if quiz and quiz[0].get("t") is not None else log["t_end"]
    cats = {int(i): d["category"] for i, d in ev["instances"].items()}
    vis_rows = [v for v in captured_visibility(log, ev) if v["t"] <= t_q]
    seen = {}
    for v in vis_rows:
        for i in v["visible"]:
            seen.setdefault(cats[int(i)], set()).add(int(i))
    pos = truth_at(ev, t_q)
    S = {s["id"]: s for s in mem["sightings"]}
    tracks = {tr["id"]: tr for tr in mem["tracks"]}
    out = []
    for r in parse_table(mem["table"]):
        tr = tracks[r["id"]]
        xy = np.array(S[tr["sightings"][-1]]["xyz"][:2])
        inst = seen.get(r["label"], set())
        real = bool(inst) and min(float(np.linalg.norm(pos[i][1] - xy)) for i in inst) < NEAR_M
        r["y"] = float(real)
        out.append(r)
    return out


def episode_arm(ep_dir: str) -> str | None:
    """The one arm per episode whose memory is used: unchanged (quiz-time memory), else explore-only."""
    for arm in ("unchanged", "explore_only"):
        if os.path.exists(f"{ep_dir}/{arm}/evaluator.json") and os.path.exists(f"{ep_dir}/{arm}/memory.json"):
            return f"{ep_dir}/{arm}"
    return None


def dataset(roots: list[str], exclude_houses: set[str] = frozenset()) -> dict:
    """Features, labels and group ids (house) for every episode under the given roots."""
    X, y, house, ep_names = [], [], [], []
    for root in roots:
        for ep in sorted(glob.glob(f"{root}/*/manifest.json")):
            man = json.load(open(ep))
            if man["house"] in exclude_houses:
                continue
            arm = episode_arm(os.path.dirname(ep))
            if arm is None:
                continue
            rows = label_rows(arm)
            if not rows:
                continue
            X.append(features(rows))
            y += [r["y"] for r in rows]
            house += [man["house"]] * len(rows)
            ep_names.append(os.path.basename(os.path.dirname(ep)))
    return dict(X=np.concatenate(X) if X else np.zeros((0, 0)), y=np.array(y), house=np.array(house), episodes=ep_names)
