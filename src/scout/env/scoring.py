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


def captured_visibility(log, ev):
    """Evaluator visibility rows that belong to head frames the agent actually captured (same frame id and time)."""
    look_t = {lk["frame"]: lk["t"] for lk in log.get("looks", [])}
    return [v for v in ev.get("visibility", []) if v["frame"] in look_t and abs(look_t[v["frame"]] - v["t"]) < 1e-6]


def truth_at(ev, t):
    """Object positions at time t: the start positions with every evaluator move up to t applied."""
    pos = {o["id"]: (o["category"], np.array(o["xy"], float)) for o in ev["objects_start"]}
    for mv in ev["moves"]:
        if mv["t"] <= t + 1e-9:
            c, _ = pos[mv["instance"]]
            pos[mv["instance"]] = (c, np.array(mv["to_xy"], float))
    return pos


def _bound_to_target(anchor, target, ev):
    """The checked spot must belong to the frozen instance: within 1.5 m of it and nearer to it than to any other
    instance of the same category (a neighbouring chair is a different chair)."""
    if anchor is None:
        return False
    a = np.array(anchor, float)
    d_t = float(np.linalg.norm(a - np.array(target["from_xy"])))
    if d_t >= NEAR_M:
        return False
    if ev is None or "instance" not in target:
        return True
    for o in ev.get("objects_start", []):
        if o["category"] == target["category"] and o["id"] != target["instance"]:
            if float(np.linalg.norm(a - np.array(o["xy"]))) < d_t:
                return False
    return True


def score_verify(arm, log, mem=None, target=None, moves=None, ev=None):
    """The body's verdict on the frozen question, bound to the frozen target. Ineligible when the moved arm did not
    actually move anything or the check inspected a different object."""
    ans = [q for q in log["qa"] if q["kind"] == "answer"]
    if arm == "moved" and moves is not None and not moves:
        return dict(arm=arm, verdict=None, outcome="ineligible", reason="no intervention happened")
    if not ans:
        return dict(arm=arm, verdict=None, outcome="no answer")
    a = ans[-1]
    v = a.get("result")
    if target is not None and mem is not None and v not in ("no check", None):
        tid = a.get("track")
        tr = next((t for t in mem["tracks"] if t["id"] == tid), None)
        S = {s["id"]: s for s in mem["sightings"]}
        anchor = a.get("anchor_xy")
        if anchor is None and tr is not None:
            pre = [S[i] for i in tr["sightings"] if S[i]["t"] <= target.get("t_freeze", 1e9)] or [S[tr["sightings"][0]]]
            anchor = pre[-1]["xyz"][:2]
        if tr is None or tr["label"] != target["category"] or not _bound_to_target(anchor, target, ev):
            return dict(arm=arm, verdict=v, outcome="wrong target", say=a.get("say"))
    want = "still there" if arm == "unchanged" else "not found"
    outcome = "correct" if v == want else ("abstain" if v in ("unclear", "no check") else "wrong")
    return dict(arm=arm, verdict=v, outcome=outcome, say=a.get("say"))


def score_quiz(log, mem, ev):
    """Graded against what the agent's own captured frames showed. A "yes" needs a location within 1.5 m of a seen
    instance; a failed or malformed brain reply is an error, never a correct "no"."""
    quiz = log.get("quiz") or []
    if not quiz:
        return None
    seen_inst = {}
    cats = {int(i): d["category"] for i, d in ev["instances"].items()}
    vis_rows = captured_visibility(log, ev)
    rows = []
    for q in quiz:
        t_q = q.get("t") if q.get("t") is not None else log["t_end"]
        c = q["category"]
        vis = sorted({int(i) for v in vis_rows if v["t"] <= t_q for i in v["visible"] if cats[int(i)] == c})
        observable = bool(vis)
        pos = truth_at(ev, t_q)
        # the agent marks every reply valid/invalid when it records it; records without that flag need a boolean
        # "observed" (and, if they carry an answer, a non-empty string)
        if "valid" in q:
            well_formed = q["valid"] is True and not q.get("error")
        else:
            well_formed = isinstance(q.get("observed"), bool) and (
                "answer" not in q or (isinstance(q["answer"], str) and bool(q["answer"].strip())))
        if not well_formed:
            rows.append(dict(category=c, observable=observable, said_seen=None, located=None, correct=False, error=True))
            continue
        said = bool(q["observed"])
        located = None
        if said:
            located = bool(q.get("xy") is not None and observable and
                           min(float(np.linalg.norm(pos[i][1] - np.array(q["xy"]))) for i in vis) < NEAR_M)
            correct = located
        else:
            correct = not observable
        rows.append(dict(category=c, observable=observable, said_seen=said, located=located, correct=bool(correct), error=False))
    yes = [r for r in rows if r["said_seen"]]
    return dict(n=len(rows), accuracy=round(float(np.mean([r["correct"] for r in rows])), 3),
                errors=sum(r["error"] for r in rows), said_seen=len(yes), said_seen_correct=sum(bool(r["located"]) for r in yes),
                observable=sum(r["observable"] for r in rows),
                recall=round(sum(bool(r["said_seen"]) and r["correct"] for r in rows) / max(1, sum(r["observable"] for r in rows)), 3),
                rows=rows)


def score_citations(log, mem, ev):
    """A citation is valid only if the sighting exists, comes from a frame captured before that question, is of the
    question's category, and the evaluator saw an instance of that category in that exact frame."""
    S = {s["id"]: s for s in mem["sightings"]}
    vis_by_frame = {v["frame"]: v["visible"] for v in captured_visibility(log, ev)}
    cats = {int(i): d["category"] for i, d in ev["instances"].items()}
    out = dict(cited=0, valid=0, invalid_id=0, future=0, wrong_category=0, not_visible=0,
               affirmative=0, affirmative_supported=0)
    look_t = {lk["frame"]: lk["t"] for lk in log.get("looks", [])}
    for q in log.get("quiz") or []:
        t_q = q.get("t") if q.get("t") is not None else log["t_end"]
        cat = q["category"]
        supported = False
        bad = q.get("malformed_citations", [])
        out["cited"] += len(bad)
        out["invalid_id"] += len(bad)
        for sid in q.get("cited", []):
            out["cited"] += 1
            s = S.get(sid) if isinstance(sid, int) else None
            if s is None:
                out["invalid_id"] += 1
            elif s["t"] > t_q or look_t.get(s["frame"], float("inf")) > t_q:
                out["future"] += 1  # the sighting, or the frame it claims to come from, is later than the question
            elif s["label"] != cat:
                out["wrong_category"] += 1
            elif not any(cats[int(i)] == cat for i in vis_by_frame.get(s["frame"], {})):
                out["not_visible"] += 1
            else:
                out["valid"] += 1
                supported = True
        if q.get("observed") is True and not q.get("error"):
            out["affirmative"] += 1
            out["affirmative_supported"] += int(supported)
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
        if e.get("anchor_xy") is not None:
            p = np.array(e["anchor_xy"])
        else:
            t0 = e.get("t_start", e["t"])
            ss = [S[i] for i in tr["sightings"] if S[i]["t"] <= t0] or [S[tr["sightings"][0]]]
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
    cov = float(seen[ij[ok, 0], ij[ok, 1]].sum()) / max(len(P), 1)  # true floor outside the map counts as unseen
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
            r["verify"] = score_verify(a, log, mem, man.get("target"), ev.get("moves"), ev)
            if log.get("recall"):
                r["verify_recall"] = score_verify(a, dict(qa=log["recall"]), mem, man.get("target"), ev.get("moves"), ev)
        if log.get("quiz"):
            r["quiz"] = score_quiz(log, mem, ev)
            r["cite"] = score_citations(log, mem, ev)
        if layout is not None and os.path.exists(f"{ep_dir}/{a}/maps.npz"):
            r["explore"] = score_explore(log, ev, layout, np.load(f"{ep_dir}/{a}/maps.npz"), man["explore_s"])
        out[a] = r
    return out
