"""Scorers for Scout environment episodes. Pure python: they read the saved agent logs and the evaluator files.

Tasks:
  stay_up   falls, recoveries, early end
  explore   share of the house's walkable floor the agent's own map saw, rooms visited, metres walked, stuck, idle time
  verify    "Is the <x> still where you last saw it?" after the evaluator moved it (correct: not found) or not
            (correct: still there); "unclear" is an abstention, reported separately
  remember  the memory quiz, graded against visibility history: a category counts as seen only if one of its instances
            covered >= 0.5% of a head frame the agent captured; a "yes" must also place it within 1.5 m of one
  cite      every cited sighting must exist, come from a frame captured before the question, and that frame must show
            an instance of the category (evaluator segmentation)
  checks    the brain's own re-checks, scored against the truth at that time; abstentions reported
"""
from __future__ import annotations

import json
import os

import numpy as np

NEAR_M = 1.5


def _load(arm_dir):
    j = lambda f: json.load(open(f"{arm_dir}/{f}"))  # noqa: E731
    return j("log.json"), j("memory.json"), j("evaluator.json")


def truth_at(ev, t):
    """Object positions at time t: the start positions with every evaluator move up to t applied."""
    pos = {o["id"]: (o["category"], np.array(o["xy"], float)) for o in ev["objects_start"]}
    for mv in ev["moves"]:
        if mv["t"] <= t + 1e-9:
            c, _ = pos[mv["instance"]]
            pos[mv["instance"]] = (c, np.array(mv["to_xy"], float))
    return pos


def score_verify(arm, log):
    ans = [q for q in log["qa"] if q["kind"] == "answer"]
    if not ans:
        return dict(arm=arm, verdict=None, outcome="no answer")
    v = ans[-1].get("result")
    want = "still there" if arm == "unchanged" else "not found"
    outcome = "correct" if v == want else ("abstain" if v in ("unclear", "no check") else "wrong")
    return dict(arm=arm, verdict=v, outcome=outcome, say=ans[-1].get("say"))


def score_quiz(log, mem, ev):
    quiz = log.get("quiz") or []
    if not quiz:
        return None
    t_q = log["t_end"]
    seen_inst = {}
    for v in ev["visibility"]:
        if v["t"] <= t_q:
            for i in v["visible"]:
                seen_inst.setdefault(int(i), v["t"])
    cats = {int(i): d["category"] for i, d in ev["instances"].items()}
    pos = truth_at(ev, t_q)
    rows = []
    for q in quiz:
        c = q["category"]
        vis = [i for i in seen_inst if cats[i] == c]
        observable = bool(vis)
        loc_ok = None
        if q["observed"] and observable and q.get("xy") is not None:
            loc_ok = min(np.linalg.norm(pos[i][1] - np.array(q["xy"])) for i in vis) < NEAR_M
        correct = (q["observed"] == observable) and (loc_ok is not False)
        rows.append(dict(category=c, observable=observable, said_seen=q["observed"], located=loc_ok, correct=correct))
    yes = [r for r in rows if r["said_seen"]]
    return dict(n=len(rows), accuracy=round(np.mean([r["correct"] for r in rows]), 3),
                said_seen=len(yes), said_seen_correct=sum(r["observable"] and r["located"] is not False for r in yes),
                observable=sum(r["observable"] for r in rows),
                recall=round(sum(r["said_seen"] and r["correct"] for r in rows) / max(1, sum(r["observable"] for r in rows)), 3),
                rows=rows)


def score_citations(log, mem, ev):
    S = {s["id"]: s for s in mem["sightings"]}
    vis_by_frame = {v["frame"]: v["visible"] for v in ev["visibility"]}
    cats = {int(i): d["category"] for i, d in ev["instances"].items()}
    out = dict(cited=0, valid=0, invalid_id=0, future=0, not_visible=0)
    t_q = log["t_end"]
    for q in log.get("quiz") or []:
        for sid in q.get("cited", []):
            out["cited"] += 1
            s = S.get(sid)
            if s is None:
                out["invalid_id"] += 1
                continue
            if s["t"] > t_q:
                out["future"] += 1
                continue
            vis = vis_by_frame.get(s["frame"], {})
            if any(cats[int(i)] == s["label"] for i in vis):
                out["valid"] += 1
            else:
                out["not_visible"] += 1
    return out


def score_checks(log, mem, ev):
    S = {s["id"]: s for s in mem["sightings"]}
    tracks = {tr["id"]: tr for tr in mem["tracks"]}
    res = dict(n=0, correct=0, wrong=0, abstain=0)
    for e in log["events"]:
        if e["kind"] != "check_result" or e.get("question"):
            continue
        res["n"] += 1
        if e["result"] == "unclear":
            res["abstain"] += 1
            continue
        tr = tracks[e["track"]]
        ss = [S[i] for i in tr["sightings"] if S[i]["t"] <= e["t"]] or [S[tr["sightings"][0]]]
        p = np.array(ss[-1]["xyz"][:2])
        pos = truth_at(ev, e["t"])
        real = any(c == tr["label"] and np.linalg.norm(xy - p) < NEAR_M for c, xy in pos.values())
        ok = (e["result"] == "still there") == real
        res["correct" if ok else "wrong"] += 1
    return res


def score_explore(log, ev, layout, maps, t_cut):
    """Floor seen by the agent's own map by time t_cut, against the house's true walkable floor."""
    shape, origin, res = tuple(maps["shape"]), maps["origin"], float(maps["res"])
    mt = np.array(maps["t"])
    i = max(int(np.searchsorted(mt, t_cut + 1e-9, side="right") - 1), 0)
    seen = np.unpackbits(maps[f"seen_{i}"])[: shape[0] * shape[1]].reshape(shape).astype(bool)
    g = layout.grid
    P = np.array([g.pos(c) for c in np.argwhere(g.free)])
    ij = np.floor((P - origin) / res).astype(int)
    ok = (ij >= 0).all(1) & (ij < shape[0]).all(1)
    cov = float(seen[ij[ok, 0], ij[ok, 1]].mean()) if ok.any() else 0.0
    looks = [lk for lk in log["looks"] if lk["t"] <= t_cut]
    xy = np.array([lk["xy"] for lk in looks])
    walked = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else 0.0
    rooms = {getattr(layout.room_at(p), "id", None) for p in xy} - {None}
    sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) / 0.5 if len(xy) > 1 else np.zeros(0)
    return dict(floor_seen=round(cov, 3), rooms=len(rooms), rooms_total=len(layout.rooms), walked_m=round(walked, 1),
                stuck=sum(e["kind"] == "stuck" and e["t"] <= t_cut for e in log["events"]),
                idle_frac=round(float((sp < 0.1).mean()) if len(sp) else 1.0, 3))


def score_episode(ep_dir: str, layout=None) -> dict:
    man = json.load(open(f"{ep_dir}/manifest.json"))
    out = dict(episode=man["episode"], house=man["house"], brain=man["brain"], target=man.get("target"),
               no_target_reason=man.get("no_target_reason"))
    arms = [a for a in ("unchanged", "moved") if os.path.exists(f"{ep_dir}/{a}/log.json")]
    if not arms and os.path.exists(f"{ep_dir}/explore_only/log.json"):
        arms = ["explore_only"]
    for a in arms:
        log, mem, ev = _load(f"{ep_dir}/{a}")
        r = dict(falls=log["falls"], got_up=log["got_up"], ended=log["ended"], checks=score_checks(log, mem, ev))
        if a in ("unchanged", "moved"):
            r["verify"] = score_verify(a, log)
        if log.get("quiz"):
            r["quiz"] = score_quiz(log, mem, ev)
            r["cite"] = score_citations(log, mem, ev)
        if layout is not None and os.path.exists(f"{ep_dir}/{a}/maps.npz"):
            r["explore"] = score_explore(log, ev, layout, np.load(f"{ep_dir}/{a}/maps.npz"), man["explore_s"])
        out[a] = r
    return out
