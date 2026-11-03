"""The agent that plays in ScoutEnv: its own depth map, its episodic memory, a plug-in brain, and a body controller.

Everything here uses only `ScoutEnv.observe()`; nothing reads the evaluator's truth. One clock: `env.t`.

Modes: explore (nearest reachable frontier, a fixed rule), check (walk back to a remembered object and look),
recover (after a fall, hold Motivo's standing skill), done.
A check ends in one of three verdicts:
  still there  the object was detected again at its remembered place (same memory track or within 1.5 m)
  not found    no detection, AND on >= 2 looks the depth camera saw past the remembered spot (it is now empty)
  unclear      anything else: out of view, occluded, or something is there but SAM did not recognise it
"""
from __future__ import annotations

import io
import math
import time

import numpy as np

from ..body.follow import lookahead, wrap_deg
from ..memory.episodic import Episodic
from ..perception.selfmap import SelfMap, depth_to_points
from ..world.grid import descend, geodesic
from .core import FOVY, SIZE

MIX_LEFT = ([-18, 12, 71, 119], [0.0, 0.25, 0.5, 0.75])   # measured in docs/results/motion_tests.md
MIX_RIGHT = ([18, 33, 85, 122], [0.0, 0.25, 0.5, 0.75])
LOOK_EVERY, DETECT_EVERY = 15, 30                         # steps: map every 0.5 s, SAM 3.1 every 1 s


class Agent:
    def __init__(self, brain, mind: bool = True, mind_every: float = 20.0):
        self.brain, self.mind, self.mind_every = brain, mind, mind_every
        self.mem = Episodic()
        self.smap = None
        self.events, self.looks, self.decisions, self.qa = [], [], [], []
        self.mode, self.goal, self.path, self.prog = "explore", None, np.zeros((0, 2)), 0
        self.best_gd, self.last_progress_t, self.banned, self.all_done = 1e9, 0.0, [], False
        self.next_mind_t, self.mind_due = mind_every, False
        self.miss_run, self.checks_done, self.last_check = {}, {}, None
        self.check = None          # dict describing an ongoing check
        self.pending_q = None      # a question waiting to be handled
        self.recover = None        # dict while recovering from a fall
        self.falls = self.got_up = 0
        self.ended = None
        self.z = None
        self.last_rel = None

    # --- helpers --------------------------------------------------------------------------------------------------
    def log(self, t, kind, **kw):
        self.events.append(dict(t=round(t, 3), wall=round(time.time(), 2), kind=kind, **kw))

    def _target_z(self, Z, norm, rel):
        if rel >= -18:
            w, side = float(np.interp(rel, *MIX_LEFT)), Z["move-ego-90-1"]
        else:
            w, side = float(np.interp(-rel, *MIX_RIGHT)), Z["move-ego--90-1"]
        z = (1 - w) * Z["move-ego-0-1"] + w * side
        return z / z.norm() * norm

    def _set_goal(self, p, t):
        self.goal, self.path, self.prog = (p[-1] if len(p) else None), p, 0
        self.best_gd, self.last_progress_t = 1e9, t

    def _frontiers(self, xy):
        wk = self.smap.walkable()
        c = self.smap.cell(xy)
        sl = (slice(max(c[0] - 3, 0), c[0] + 4), slice(max(c[1] - 3, 0), c[1] + 4))
        wk[sl] |= ~self.smap.inflated()[sl]
        dist = geodesic(wk, [c], self.smap.res)
        fr = [(f, n) for f, n in self.smap.frontiers() if np.isfinite(dist[self.smap.cell(f)])
              and np.linalg.norm(f - xy) > 0.8 and all(np.linalg.norm(f - b) > 1.0 for b in self.banned)]
        return wk, c, dist, fr

    def _viewpoint_path(self, xy, obj):
        away = xy - obj
        if np.linalg.norm(away) < 4.0:
            p = self.smap.path(xy, obj + 2.0 * away / (np.linalg.norm(away) + 1e-6), goal_radius=0.6)
            if len(p) >= 2:
                return p
        base = math.atan2(away[1], away[0])
        for dang in (0, 0.6, -0.6, 1.2, -1.2, 2.0, -2.0):
            for rr in (2.5, 2.0, 3.0):
                v = obj + rr * np.array([math.cos(base + dang), math.sin(base + dang)])
                p = self.smap.path(xy, v, goal_radius=0.4)
                if len(p) >= 2:
                    return p
        return np.zeros((0, 2))

    def _start_check(self, tid, t, xy, why, question=None):
        obj = self.mem.tracks[tid].last_xy(self.mem).copy()
        p = self._viewpoint_path(xy, obj)
        if len(p) < 2 and np.linalg.norm(obj - xy) > 2.5:
            self.log(t, "check_unreachable", track=tid, label=self.mem.tracks[tid].label)
            return False
        last = self.mem.sightings[self.mem.tracks[tid].sightings[-1]]
        best = max((self.mem.sightings[i] for i in self.mem.tracks[tid].sightings), key=lambda s: s.score)
        self.check = dict(tid=tid, t0=t, xy=obj, xyz=last.xyz.copy(), arrived=None, turn=0, hits=0, through=0,
                          occupied=0, views=0, frames=[], question=question, n_before=len(self.mem.tracks[tid].sightings),
                          before_frame=last.frame, before_t=last.t, best_frame=best.frame)
        self.mode = "check"
        self._set_goal(p, t)
        self.log(t, "check_start", track=tid, label=self.mem.tracks[tid].label, why=why, question=question)
        return True

    def ask(self, question: str):
        self.pending_q = question

    # --- perception -------------------------------------------------------------------------------------------------
    def _perceive(self, env, obs, detect):
        t, xy = obs["t"], obs["xy"]
        if self.smap is None:
            self.smap = SelfMap(origin=xy, size_m=40)
        pts = depth_to_points(obs["depth"], obs["cam_pos"], obs["cam_mat"], FOVY, stride=4)
        pts = pts[np.linalg.norm(pts[:, :2] - xy, axis=1) > 0.6]  # its own arms and legs
        self.smap.integrate(pts, cam_xy=obs["cam_pos"][:2])
        self.smap.stand(xy)
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(obs["rgb"]).save(buf, "JPEG", quality=88)
        look = dict(t=round(t, 3), frame=len(self.looks), xy=[float(v) for v in xy], yaw=float(obs["yaw"]),
                    head_z=round(float(obs["head_z"]), 3), mode=self.mode, cmd_rel=self.last_rel, jpg=buf.getvalue(),
                    dets=None, cam_pos=[round(float(v), 4) for v in obs["cam_pos"]],
                    cam_mat=[round(float(v), 5) for v in np.asarray(obs["cam_mat"]).ravel()])
        self.looks.append(look)
        if not detect:
            return look
        depth, cp, cm = obs["depth"], obs["cam_pos"], obs["cam_mat"]
        h = depth.shape[0]
        f = (h / 2) / math.tan(math.radians(FOVY) / 2)
        look["dets"] = []
        from scipy.ndimage import binary_erosion
        for label, insts in (obs["dets"] or {}).items():
            for score, m in insts:
                if score < 0.5 or m.sum() < 30:
                    continue
                core = binary_erosion(m, iterations=4)
                core = core if core.sum() >= 20 else m
                vs, us = np.nonzero(core)
                dv = depth[vs, us]
                ok = (dv > 0.1) & (dv < 8)
                if ok.sum() < 10:
                    continue
                front = np.percentile(dv[ok], 30)
                ok &= np.abs(dv - front) < 0.4
                if ok.sum() < 10:
                    continue
                x = (us[ok] + 0.5 - h / 2) / f * dv[ok]
                y = -(vs[ok] + 0.5 - h / 2) / f * dv[ok]
                xyz = np.median(np.stack([x, y, -dv[ok]], 1) @ cm.T + cp, 0)
                if not np.isfinite(xyz).all():
                    continue
                s, tr, new = self.mem.add(label, score, xyz, t, look["frame"], xy)
                vs_all, us_all = np.nonzero(m)
                look["dets"].append(dict(label=label, score=round(score, 3), sighting=s.id, track=tr.id,
                                         box=[int(us_all.min()), int(vs_all.min()), int(us_all.max()), int(vs_all.max())]))
                if new:
                    self.log(t, "new_object", label=label, track=tr.id, sighting=s.id, frame=look["frame"])
        seen_tracks = {d["track"] for d in look["dets"]}
        # spontaneous misses, per object: expected in view (1-3.5 m, central, not occluded) but not detected
        for tr, (u, v) in self.mem.expected_in_view(cp, cm, FOVY, depth, max_range=3.5):
            eligible = abs(u - SIZE / 2) < SIZE / 3 and abs(v - SIZE / 2) < SIZE / 3 and \
                np.linalg.norm(tr.last_xy(self.mem) - xy) > 1.0 and t - self.mem.sightings[tr.sightings[-1]].t > 3.0
            if not eligible or tr.id in seen_tracks:
                self.miss_run.pop(tr.id, None)
                continue
            self.miss_run[tr.id] = self.miss_run.get(tr.id, 0) + 1
            if self.miss_run[tr.id] == 3:
                tr.misses.append((t, look["frame"]))
                self.log(t, "expected_not_seen", label=tr.label, track=tr.id, frame=look["frame"])
                self.mind_due = self.mind_due or self.mind
        expected_ids = {tr.id for tr, _ in self.mem.expected_in_view(cp, cm, FOVY, depth, max_range=3.5)}
        for tid in list(self.miss_run):
            if tid not in expected_ids:
                self.miss_run.pop(tid)  # left the view: the run of consecutive misses is broken
        # evidence for an ongoing check, once it has arrived and faces the spot
        c = self.check
        if c is not None and c["arrived"] is not None:
            trc = self.mem.tracks[c["tid"]]
            hit = any(d["label"] == trc.label and (d["track"] == trc.id or
                      np.linalg.norm(self.mem.sightings[d["sighting"]].xyz[:2] - c["xy"]) < 1.5) for d in look["dets"])
            pc = cm.T @ (c["xyz"] - cp)
            if pc[2] < -0.3:
                dd = -pc[2]
                u = int(pc[0] / dd * f + SIZE / 2)
                v = int(-pc[1] / dd * f + SIZE / 2)
                if 8 <= u < SIZE - 8 and 8 <= v < SIZE - 8:
                    c["views"] += 1
                    look["check_uv"] = [u, v]
                    patch = depth[max(v - 6, 0):v + 7, max(u - 6, 0):u + 7]
                    med = float(np.median(patch))
                    if med > dd + 0.35:
                        c["through"] += 1   # the camera sees past where the object was
                    elif abs(med - dd) <= 0.35:
                        c["occupied"] += 1  # something is at that spot
                        if not hit and env.eyes is not None:
                            # look closer: SAM 3.1 on a zoomed crop around the spot, asked only for this category
                            half = 128
                            x0, y0 = min(max(u - half, 0), SIZE - 2 * half), min(max(v - half, 0), SIZE - 2 * half)
                            from PIL import Image
                            crop = np.asarray(Image.fromarray(obs["rgb"][y0:y0 + 2 * half, x0:x0 + 2 * half]).resize((SIZE, SIZE)))
                            zd = env.eyes.detect(crop, [trc.label]).get(trc.label, [])
                            cu, cv = (u - x0) * SIZE // (2 * half), (v - y0) * SIZE // (2 * half)
                            zhit = any(sc >= 0.5 and m[max(cv - 20, 0):cv + 21, max(cu - 20, 0):cu + 21].any() for sc, m in zd)
                            look["zoom"] = dict(box=[x0, y0, x0 + 2 * half, y0 + 2 * half], hit=bool(zhit),
                                                score=round(max([sc for sc, _ in zd], default=0.0), 3))
                            hit = hit or zhit
            c["hits"] += int(hit)
            c["frames"].append(look["frame"])
        return look

    # --- decisions --------------------------------------------------------------------------------------------------
    def _mind_state(self, t, xy, n_frontiers):
        stale = sorted([tr for tr in self.mem.tracks if tr.status == "ok" and tr.confident(self.mem)
                        and self.checks_done.get(tr.id, 0) < 2], key=lambda tr: self.mem.sightings[tr.sightings[-1]].t)[:5]
        return (f"time {t:.0f} s; position x={xy[0]:.1f} y={xy[1]:.1f}; unexplored openings reachable: {n_frontiers}; "
                f"last object checked: {self.last_check}; objects already checked twice (do not choose): "
                f"{[k for k, v in self.checks_done.items() if v >= 2]}; confident objects not seen for the longest: " +
                ", ".join(f"{tr.id} {tr.label} (last seen {t - self.mem.sightings[tr.sightings[-1]].t:.0f} s ago)" for tr in stale) +
                "; recent events: " + "; ".join(f"{e['kind']} {e.get('label', '')} at {e['t']:.0f}s" for e in self.events[-8:]
                                                 if e["kind"] in ("expected_not_seen", "check_result", "stuck")))

    def _decide(self, t, xy):
        # a person's question comes first
        if self.pending_q and self.mode != "check":
            q, self.pending_q = self.pending_q, None
            r = self.brain.resolve(self.mem.table(), q) or {}
            tid = r.get("object_id")
            self.qa.append(dict(t=round(t, 3), kind="question", text=q, say=r.get("say", ""), object_id=tid))
            self.log(t, "question", text=q, say=r.get("say", ""), object_id=tid)
            if not (isinstance(tid, int) and 0 <= tid < len(self.mem.tracks) and self._start_check(tid, t, xy, "asked", q)):
                self.qa.append(dict(t=round(t, 3), kind="answer", result="no check", say=r.get("say", "") or "I can't check that."))
            return
        if self.mind and self.mode == "explore" and (t >= self.next_mind_t or self.mind_due) and self.mem.tracks:
            _, _, _, fr = self._frontiers(xy)
            state = self._mind_state(t, xy, len(fr))
            t0 = time.time()
            dec = self.brain.decide(self.mem.table(), state) or {"do": "explore"}
            self.decisions.append(dict(t=round(t, 3), decision=dec, latency_s=round(time.time() - t0, 2)))
            self.log(t, "mind", decision=dec)
            self.next_mind_t, self.mind_due = t + self.mind_every, False
            tid = dec.get("object_id")
            if dec.get("do") == "check" and isinstance(tid, int) and 0 <= tid < len(self.mem.tracks) \
                    and tid != self.last_check and self.checks_done.get(tid, 0) < 2 and self.mem.tracks[tid].status == "ok":
                self.checks_done[tid] = self.checks_done.get(tid, 0) + 1
                self._start_check(tid, t, xy, dec.get("why", ""))

    def _check_step(self, t, xy, yaw, stuck):
        c = self.check
        tr = self.mem.tracks[c["tid"]]
        near = len(self.path) >= 2 and np.linalg.norm(xy - self.path[-1]) < 0.5
        if c["arrived"] is None and (near or stuck or len(self.path) < 2):
            to = c["xy"] - xy
            rel = wrap_deg(math.degrees(math.atan2(to[1], to[0]) - yaw))
            if abs(rel) < 25 or t - c["t0"] > 30:
                c["arrived"], c["turn"] = t, 0
                self.path = np.zeros((0, 2))
            elif np.linalg.norm(to) < 1.6:
                self.path, c["turn"] = np.zeros((0, 2)), (1 if rel > 0 else -1)
            else:
                step_to = xy + to / np.linalg.norm(to) * min(1.0, np.linalg.norm(to) - 1.5)
                self._set_goal(np.array([xy, step_to]), t)
        done = c["arrived"] is not None and (t - c["arrived"] >= 4.5 or (c["hits"] and t - c["arrived"] >= 1.5))
        if done or t - c["t0"] > 40:
            if c["hits"]:
                res = "still there"
            elif c["through"] >= 2:
                res = "not found"
                tr.status = "retracted" if len(tr.sightings) < 2 else "gone"
            else:
                res = "unclear"
            self.log(t, "check_result", track=c["tid"], label=tr.label, result=res, status=tr.status, hits=c["hits"],
                     through=c["through"], occupied=c["occupied"], views=c["views"], question=c["question"])
            if c["question"]:
                remembered = f"a {tr.label} seen {c['n_before']} times, last at t={c['before_t']:.0f}s"
                finding = {"still there": "still there", "not found": "not found", "unclear": "unclear"}[res]
                if res == "unclear" and c["occupied"] >= 2:
                    finding = "unclear: something is still at that exact spot, but I could not recognise it as the " + tr.label
                say = self.brain.phrase(c["question"], remembered, finding)
                self.qa.append(dict(t=round(t, 3), kind="answer", result=res, say=say, track=c["tid"],
                                    before_frame=c["before_frame"], best_frame=c["best_frame"], now_frames=c["frames"][-3:],
                                    evidence=dict(hits=c["hits"], through=c["through"], occupied=c["occupied"], views=c["views"])))
                self.log(t, "answer", say=say, result=res)
            self.last_check, self.mode, self.check, self.goal = c["tid"], "explore", None, None
            self.mind_due = self.mind_due or (self.mind and (res == "not found" or self.all_done))

    def _explore_step(self, t, xy, stuck):
        if self.goal is not None and np.linalg.norm(xy - self.goal) >= 0.6 and not stuck:
            return
        if stuck and self.goal is not None:
            self.banned.append(self.goal.copy())
            self.log(t, "stuck", goal=[round(float(v), 2) for v in self.goal])
        wk, c, dist, fr = self._frontiers(xy)
        if fr:
            f, n = min(fr, key=lambda q: dist[self.smap.cell(q[0])])
            cells = descend(geodesic(wk, [self.smap.cell(f)], self.smap.res), c)
            p = np.array([self.smap.pos(q) for q in cells]) if cells else np.zeros((0, 2))
            if self.goal is None or np.linalg.norm(f - self.goal) > 0.5:
                self.log(t, "new_waypoint", xy=[round(float(v), 2) for v in f], frontier_cells=int(n))
            self._set_goal(p, t)
            self.all_done = False
        else:
            if not self.all_done:
                self.log(t, "explored_all")
                self.all_done, self.mind_due = True, self.mind_due or self.mind
            self.goal, self.path = None, np.zeros((0, 2))

    # --- one control step -----------------------------------------------------------------------------------------
    def tick(self, env):
        """Decide and act for one 1/30 s step. Returns False when the episode has ended for the agent."""
        Z = env.Z
        if self.z is None:
            self.norm = float(np.mean([float(Z[k].norm()) for k in ("move-ego-0-1", "move-ego-90-1", "move-ego--90-1", "move-ego-0-0")]))
            self.z = Z["move-ego-0-0"].clone()
        t = env.t
        body = env.body
        xy, yaw = body.xy, body.yaw
        if self.recover is not None:  # fallen: hold the standing skill, on the same clock as everything else
            r = self.recover
            if body.head_z > 1.35 and t - r["t0"] > 1.0:
                self.got_up += 1
                self.log(t, "got_up", seconds=round(t - r["t0"], 2))
                self.recover = None
                self.mode = r["mode_before"] if r["mode_before"] != "check" else "explore"
                self.check, self.goal, self.path = None, None, np.zeros((0, 2))
                self.last_progress_t = t
            elif t - r["t0"] > 8.0:
                self.log(t, "stayed_down")
                self.ended = "stayed_down"
                return False
            ok = env.step(Z["move-ego-0-0"])
            if not ok:
                self.ended = "physics_invalid"
            return ok
        if body.head_z < 0.8:
            self.falls += 1
            self.log(t, "fell", xy=[round(float(v), 2) for v in xy])
            self.recover = dict(t0=t, mode_before=self.mode)
            return env.step(Z["move-ego-0-0"])
        if env.k % LOOK_EVERY == 0:
            obs = env.observe(detect=env.k % DETECT_EVERY == 0)
            self._perceive(env, obs, detect=obs["dets"] is not None)
            gd = np.linalg.norm(xy - self.goal) if self.goal is not None else 0.0
            if self.goal is None or gd < self.best_gd - 0.3:
                self.best_gd, self.last_progress_t = gd, t
            stuck = self.goal is not None and t - self.last_progress_t > 6.0 and \
                (self.check is None or self.check["arrived"] is None)
            self._decide(t, xy)
            if self.mode == "check":
                self._check_step(t, xy, yaw, stuck)
            elif self.mode == "explore":
                self._explore_step(t, xy, stuck)
        # steer
        if self.mode == "check" and self.check is not None and self.check["turn"] and self.check["arrived"] is None:
            zt = Z["yaw-L-2-free"] if self.check["turn"] > 0 else Z["yaw-R-2-free"]
        elif len(self.path) >= 2:
            tgt, self.prog = lookahead(self.path, xy, self.prog, 0.8)
            d = tgt - xy
            rel = wrap_deg(math.degrees(math.atan2(d[1], d[0]) - yaw)) if np.linalg.norm(d) > 0.05 else 0.0
            self.last_rel = round(float(rel), 1)
            zt = self._target_z(Z, self.norm, rel)
        elif self.mode in ("check", "done"):
            zt = Z["move-ego-0-0"]
        else:
            zt = Z["yaw-L-2-free"]  # nothing left to explore: look around
        self.z = self.z + (zt - self.z) / 15.0 * 2.0
        self.z = self.z / self.z.norm() * self.norm
        ok = env.step(self.z)
        if not ok:
            self.ended = "physics_invalid"
        return ok

    def quiz(self, prompts) -> list[dict]:
        out = []
        for cat in prompts:
            q = f"Did you see a {cat}? If so, where was it?"
            a = self.brain.answer(self.mem.table(), q) or {}
            oid = a.get("object_id")
            xy = self.mem.tracks[oid].last_xy(self.mem).tolist() if isinstance(oid, int) and 0 <= oid < len(self.mem.tracks) else None
            out.append(dict(category=cat, question=q, answer=a.get("answer", ""), observed=bool(a.get("observed")),
                            object_id=oid, xy=xy, cited=[c for c in a.get("cited_sightings", []) if isinstance(c, int)]))
        return out

    def memory_dump(self) -> dict:
        m = self.mem
        return dict(table=m.table(),
                    sightings=[dict(id=s.id, label=s.label, score=s.score, xyz=[round(float(v), 3) for v in s.xyz], t=round(s.t, 3),
                                    frame=s.frame, body_xy=[round(float(v), 3) for v in s.body_xy]) for s in m.sightings],
                    tracks=[dict(id=tr.id, label=tr.label, sightings=tr.sightings, misses=tr.misses, status=tr.status,
                                 confident=tr.confident(m)) for tr in m.tracks])
