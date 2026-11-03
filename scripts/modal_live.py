"""Closed loop in an unseen house: physics body, head RGB-D, SAM 3.1 eyes, self-built map, episodic memory.

    modal run scripts/modal_live.py --house procthor-10k-val_val_935 --seconds 60

Every 30 Hz step the Motivo body acts on a blended skill vector. Every 0.5 s the head camera's depth updates the
body's own floor map and the next waypoint is chosen (nearest reachable frontier, a fixed rule). Every 1 s SAM 3.1
looks at the head RGB frame; each detection is placed in 3D with head depth and stored with its frame. Tracks
expected in view but not detected are logged as misses. Simulator truth is used only to pick the start spot and is
saved separately for the evaluator. Output: /data/scout/live/<run>/ (videos, memory, log, map snapshots).
"""
import os

import modal

app = modal.App("scout-live")
vol = modal.Volume.from_name("tribe-lab-claude")
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
CFG = os.path.join(os.path.dirname(__file__), "..", "configs")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "ffmpeg", "libgl1", "libegl1", "libglib2.0-0")
         .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
         .run_commands("git clone https://github.com/facebookresearch/sam3.git /sam3 && pip install -e /sam3")
         .pip_install("imageio[ffmpeg]", "pillow", "einops", "decord", "pycocotools", "psutil", "scipy", "pandas",
                      "opencv-python-headless", "mujoco==3.17.0", "h5py", "boto3")
         .pip_install("metamotivo[huggingface,humenv] @ git+https://github.com/facebookresearch/metamotivo.git")
         .run_commands("python -c 'import torch; print(torch.__version__)'")
         .env({"HF_HOME": "/data/hf", "MUJOCO_GL": "egl"})
         .add_local_dir(SRC, "/pkg")
         .add_local_dir(CFG, "/cfg"))
_tok = os.path.expanduser("~/.cache/huggingface/token")
secrets = [modal.Secret.from_dict({"HF_TOKEN": open(_tok).read().strip() if modal.is_local() and os.path.exists(_tok) else ""})]


def _aws_env() -> dict:
    """Bedrock credentials for the mind, passed as an ephemeral Modal secret (never written to disk)."""
    if not modal.is_local():
        return {"AWS_DEFAULT_REGION": "us-east-1"}
    import configparser
    cp = configparser.ConfigParser()
    cp.read(os.path.expanduser("~/.aws/credentials"))
    if "default" not in cp:
        return {"AWS_DEFAULT_REGION": "us-east-1"}
    d = cp["default"]
    env = {"AWS_ACCESS_KEY_ID": d.get("aws_access_key_id", ""), "AWS_SECRET_ACCESS_KEY": d.get("aws_secret_access_key", ""),
           "AWS_DEFAULT_REGION": "us-east-1"}
    if d.get("aws_session_token"):
        env["AWS_SESSION_TOKEN"] = d["aws_session_token"]
    return env


secrets.append(modal.Secret.from_dict(_aws_env()))
# objects the evaluator may move during a run (disclosed test of stale memory): prompt -> THOR body-name prefixes
MOVABLE = {"floor lamp": ["floorlamp"], "armchair": ["armchair"], "television": ["television"], "laptop": ["laptop"],
           "house plant": ["houseplant"], "sofa": ["sofa"], "dresser": ["dresser", "chestofdrawers"]}

PROMPTS = ["bed", "sofa", "television", "dining table", "fridge", "kitchen counter", "chair", "armchair",
           "floor lamp", "painting", "laptop", "toilet", "sink", "dresser", "bathtub", "house plant"]
FOVY = 75
EYE_PITCH = 20  # head camera tilted down: sees the floor from ~1 m ahead instead of ~2 m
SIZE = 512
# walking direction (deg, + = left) reached by mixing forward with a sidestep, measured in docs/results/motion_tests.md
MIX_LEFT = ([-18, 12, 71, 119], [0.0, 0.25, 0.5, 0.75])
MIX_RIGHT = ([18, 33, 85, 122], [0.0, 0.25, 0.5, 0.75])


def move_object(mujoco, body, mem, lay, rng, t, in_view=()):
    """Evaluator: pick a remembered movable object that is >= 5 m from the body, move it to another room.

    Returns a record of the move (kept out of the body's log), or None if no candidate yet."""
    import numpy as np
    M, D = body.M, body.D
    names = [mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(M.nbody)]
    for tr in sorted(mem.tracks, key=lambda tr: -len(tr.sightings)):
        if tr.label not in MOVABLE or len(tr.sightings) < 2:
            continue
        xy = tr.last_xy(mem)
        if np.linalg.norm(xy - body.xy) < 6.0 or tr in in_view:
            continue
        pre = tuple(p + "_" for p in MOVABLE[tr.label])
        cand = [b for b in range(M.nbody) if names[b].startswith("h_") and names[b][2:].lower().startswith(pre)]
        if not cand:
            continue
        b0 = min(cand, key=lambda b: np.linalg.norm(D.xpos[b][:2] - xy))
        if np.linalg.norm(D.xpos[b0][:2] - xy) > 1.5:
            continue
        stem = "_".join(names[b0].split("_")[:3])
        group = [b for b in cand if names[b].startswith(stem) and np.linalg.norm(D.xpos[b][:2] - D.xpos[b0][:2]) < 1.5]
        group = [b for b in group if M.body_parentid[b] not in group]
        here = lay.room_at(D.xpos[b0][:2])
        cells = np.argwhere(lay.grid.free)
        rng.shuffle(cells)
        for c in cells[:4000]:
            p = lay.grid.pos(c)
            r = lay.room_at(p)
            if r is None or (here is not None and r.id == here.id):
                continue
            if np.linalg.norm(p - D.xpos[b0][:2]) < 4.0 or np.linalg.norm(p - body.xy) < 6.0:
                continue
            delta = p - D.xpos[b0][:2]
            src = D.xpos[b0][:2].copy()
            for b in group:
                M.body_pos[b][:2] += delta
            mujoco.mj_forward(M, D)
            return dict(t=round(t, 1), label=tr.label, track_believed=tr.id, bodies=[names[b] for b in group],
                        from_xy=src.round(2).tolist(), to_xy=p.round(2).tolist(),
                        from_room=here.type if here else None, to_room=r.type)
    return None


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=65536, timeout=5400, secrets=secrets)
def live_run(house: str, seconds: float = 60.0, seed: int = 0, run_name: str = "", mind: bool = False,
             mind_every: float = 20.0, move_at: float = -1.0):
    import io
    import json
    import math
    import sys
    import time

    import imageio.v2 as imageio
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0, "/pkg")
    from scout.body.follow import lookahead, wrap_deg
    from scout.body.motivo import MotivoBody
    from scout.memory.episodic import Episodic
    from scout.mind import ask_mind
    from scout.perception.sam import Sam31
    from scout.perception.selfmap import SelfMap, depth_to_points
    from scout.world.grid import descend, geodesic
    from scout.world.layout import Layout
    t0 = time.time()
    log = lambda *a: print(f"{time.time() - t0:7.1f}s", *a, flush=True)  # noqa: E731
    run = run_name or f"{house}_s{seed}"
    out = f"/data/scout/live/{run}"
    os.makedirs(out, exist_ok=True)

    eyes = Sam31()
    log("sam loaded")
    body = MotivoBody("/data/house", house, "/data/scout/z_motion.pt", device="cuda", eye_pitch_deg=EYE_PITCH)
    body.M.vis.global_.offwidth, body.M.vis.global_.offheight = 640, 640
    lay = Layout.load(f"/data/house/layouts/{house}.npz")  # evaluator side: start spot only
    rng = np.random.default_rng(seed)
    cells = np.argwhere(lay.grid.free)
    start = lay.grid.pos(cells[rng.integers(len(cells))])
    body.place(start, float(rng.uniform(-math.pi, math.pi)))
    for _ in range(20):
        body.step("move-ego-0-0")
    log("body placed", start.round(2).tolist())

    Z = body.Z
    keys = ["move-ego-0-1", "move-ego-90-1", "move-ego--90-1", "move-ego-0-0"]
    norm = float(np.mean([float(Z[k].norm()) for k in keys]))

    def target_z(rel_deg):
        if rel_deg >= -18:
            w, side = float(np.interp(rel_deg, *MIX_LEFT)), Z["move-ego-90-1"]
        else:
            w, side = float(np.interp(-rel_deg, *MIX_RIGHT)), Z["move-ego--90-1"]
        z = (1 - w) * Z["move-ego-0-1"] + w * side
        return z / z.norm() * norm

    smap = SelfMap(origin=body.xy, size_m=40)
    miss_run = {}
    mode, check_tid, check_t0, arrived_t, last_check_tid = "explore", None, 0.0, None, None
    next_mind_t, mind_due, decisions, evaluator, all_done = (mind_every if mind else 1e9), False, [], [], False
    import mujoco
    mem = Episodic()
    events, looks, maps = [], [], []
    follow_w = imageio.get_writer(f"{out}/follow.mp4", fps=15, codec="libx264", quality=7, macro_block_size=1)
    head_w = imageio.get_writer(f"{out}/head.mp4", fps=2, codec="libx264", quality=8, macro_block_size=1)
    z = Z["move-ego-0-0"].clone()
    path, prog, goal = np.zeros((0, 2)), 0, None
    last_xy, last_move_t, fell, falls, best_gd, banned, checks_done = body.xy, 0.0, False, 0, 1e9, [], {}
    n_steps = int(seconds * 30)
    for k in range(n_steps):
        t = k / 30.0
        if k % 15 == 0:
            rgb, depth, cp, cm = body.head_rgbd(SIZE)
            pts = depth_to_points(depth, cp, cm, FOVY, stride=4)
            pts = pts[np.linalg.norm(pts[:, :2] - body.xy, axis=1) > 0.6]  # drop its own arms and legs
            smap.integrate(pts, cam_xy=cp[:2])
            smap.stand(body.xy)
            frame_id = len(looks)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, "JPEG", quality=88)
            look = dict(t=t, frame=frame_id, xy=body.xy.tolist(), yaw=body.yaw, jpg=buf.getvalue(), dets=[])
            if k % 30 == 0:
                det = eyes.detect(rgb, PROMPTS)
                seen_labels = set()
                for label, insts in det.items():
                    for score, m in insts:
                        if score < 0.5 or m.sum() < 30:
                            continue
                        vs, us = np.nonzero(m)
                        dvals = depth[vs, us]
                        ok = (dvals > 0.1) & (dvals < 8)
                        if ok.sum() < 10:
                            continue
                        h = depth.shape[0]
                        f = (h / 2) / math.tan(math.radians(FOVY) / 2)
                        x = (us[ok] + 0.5 - h / 2) / f * dvals[ok]
                        y = -(vs[ok] + 0.5 - h / 2) / f * dvals[ok]
                        pc = np.stack([x, y, -dvals[ok]], 1) @ cm.T + cp
                        xyz = np.median(pc, 0)
                        s, tr, new = mem.add(label, score, xyz, t, frame_id, body.xy)
                        seen_labels.add(label)
                        bb = [int(us.min()), int(vs.min()), int(us.max()), int(vs.max())]
                        look["dets"].append(dict(label=label, score=round(score, 3), box=bb, sighting=s.id, track=tr.id))
                        if new:
                            events.append(dict(t=t, kind="new_object", label=label, track=tr.id, sighting=s.id,
                                               xyz=xyz.round(2).tolist(), frame=frame_id))
                # a remembered object that should be in view but is not: count it only after three looks in a row,
                # 1-3.5 m away and near the centre of view, so one missed detection does not become a false "it moved"
                expected = {tr.id: tr for tr, (u, v) in mem.expected_in_view(cp, cm, FOVY, depth, max_range=3.5)
                            if abs(u - SIZE / 2) < SIZE / 3 and abs(v - SIZE / 2) < SIZE / 3
                            and np.linalg.norm(tr.last_xy(mem) - body.xy) > 1.0}
                for tr in mem.tracks:
                    if tr.label in seen_labels and tr.id not in expected:
                        miss_run.pop(tr.id, None)
                for tid, tr in expected.items():
                    if tr.label in seen_labels or t - mem.sightings[tr.sightings[-1]].t <= 3.0:
                        miss_run.pop(tid, None)
                        continue
                    miss_run[tid] = miss_run.get(tid, 0) + 1
                    if miss_run[tid] == 3:
                        tr.misses.append((t, frame_id))
                        events.append(dict(t=t, kind="expected_not_seen", label=tr.label, track=tr.id, frame=frame_id))
                        mind_due = mind_due or mind
                if mode == "check" and arrived_t is not None:
                    trc = mem.tracks[check_tid]
                    uv = next((uv for tr, uv in mem.expected_in_view(cp, cm, FOVY, depth, max_range=5.0) if tr is trc), None)
                    hit = uv is not None and any(
                        d["label"] == trc.label and d["box"][0] - 40 <= uv[0] <= d["box"][2] + 40 and d["box"][1] - 40 <= uv[1] <= d["box"][3] + 40
                        for d in look["dets"])
                    check_views += int(uv is not None)
                    check_hits += int(hit)
                head_w.append_data(rgb)
            looks.append(look)
            if k % 30 == 0:
                maps.append(dict(t=t, seen=np.packbits(smap.seen), walk=np.packbits(smap.walkable()), xy=body.xy.tolist()))

            # evaluator only: move one remembered object to another room while the body is far away (disclosed)
            if move_at >= 0 and t >= move_at and not evaluator:
                vis = [tr for tr, _ in mem.expected_in_view(cp, cm, FOVY, depth, max_range=12.0)]
                evaluator.append(move_object(mujoco, body, mem, lay, rng, t, vis))
                if evaluator[-1] is None:
                    evaluator.clear()
                    move_at = t + 10.0

            # stuck = no progress toward the current goal for 6 s (oscillating in place counts as stuck)
            gd = np.linalg.norm(body.xy - goal) if goal is not None else 0.0
            if goal is None or gd < best_gd - 0.3:
                best_gd, last_move_t = gd, t
            stuck = goal is not None and t - last_move_t > 6.0 and arrived_t is None
            frs = None

            def reachable_frontiers():
                wk = smap.walkable()
                c = smap.cell(body.xy)
                wk[max(c[0] - 3, 0):c[0] + 4, max(c[1] - 3, 0):c[1] + 4] |= ~smap.blocked[max(c[0] - 3, 0):c[0] + 4, max(c[1] - 3, 0):c[1] + 4]
                dist = geodesic(wk, [c], smap.res)
                return wk, c, dist, [(f, n) for f, n in smap.frontiers() if np.isfinite(dist[smap.cell(f)]) and np.linalg.norm(f - body.xy) > 0.8]

            # the mind (language model) decides: keep exploring, or go back and check a remembered object
            if mind and mode == "explore" and (t >= next_mind_t or mind_due) and mem.tracks:
                wk, c, dist, frs = reachable_frontiers()
                stale = sorted([tr for tr in mem.tracks if tr.status == "ok" and len(tr.sightings) >= 3
                                and checks_done.get(tr.id, 0) < 2],
                               key=lambda tr: mem.sightings[tr.sightings[-1]].t)[:5]
                state = (f"time {t:.0f} s; position x={body.xy[0]:.1f} y={body.xy[1]:.1f}; unexplored openings reachable: {len(frs)}; "
                         f"last object checked: {last_check_tid}; objects already checked twice (do not choose): "
                         f"{[k for k, v in checks_done.items() if v >= 2]}; "
                         "confident objects not seen for the longest: " +
                         ", ".join(f"{tr.id} {tr.label} (last seen {t - mem.sightings[tr.sightings[-1]].t:.0f} s ago)" for tr in stale) +
                         "; recent events: " +
                         "; ".join(f"{e['kind']} {e.get('label', '')} at {e['t']:.0f}s" for e in events[-6:]
                                   if e["kind"] in ("expected_not_seen", "check_result", "stuck")))
                dec, lat = ask_mind(mem.table(), state)
                decisions.append(dict(t=t, decision=dec, latency_s=round(lat, 1), state=state))
                events.append(dict(t=t, kind="mind", decision=dec))
                next_mind_t, mind_due = t + mind_every, False
                tid = dec.get("object_id")
                if dec.get("do") == "check" and isinstance(tid, int) and 0 <= tid < len(mem.tracks) and tid != last_check_tid \
                        and checks_done.get(tid, 0) < 2:
                    checks_done[tid] = checks_done.get(tid, 0) + 1
                    obj = mem.tracks[tid].last_xy(mem)
                    p = np.zeros((0, 2))
                    away = body.xy - obj
                    base = math.atan2(away[1], away[0])
                    for dang in (0, 0.6, -0.6, 1.2, -1.2, 2.0, -2.0):  # stand 2.5 m off, facing it; nearest side first
                        for rr in (2.5, 2.0, 3.0):
                            v = obj + rr * np.array([math.cos(base + dang), math.sin(base + dang)])
                            p = smap.path(body.xy, v, goal_radius=0.4)
                            if len(p) >= 2:
                                break
                        if len(p) >= 2:
                            break
                    if len(p) >= 2:
                        mode, check_tid, check_t0, arrived_t = "check", tid, t, None
                        check_xy, check_views, check_hits = obj.copy(), 0, 0
                        goal, path, prog = p[-1], p, 0
                        n_before = len(mem.tracks[tid].sightings)
                        events.append(dict(t=t, kind="check_start", track=tid, label=mem.tracks[tid].label, why=dec.get("why", "")))
                    else:
                        events.append(dict(t=t, kind="check_unreachable", track=tid))

            if mode == "check":
                tr = mem.tracks[check_tid]
                near = len(path) and np.linalg.norm(body.xy - path[-1]) < 0.5
                if arrived_t is None and (near or stuck):
                    arrived_t, path = t, np.zeros((0, 2))
                need = 2 if len(tr.sightings) < 2 else 3  # a confident object needs more evidence before it is "gone"
                if arrived_t is not None and (t - arrived_t >= 4.5 or (check_hits and t - arrived_t >= 1.5)) or t - check_t0 > 40:
                    if check_hits:
                        res = "still there"
                    elif check_views >= need:
                        res = "not there"
                        tr.status = "retracted" if len(tr.sightings) < 2 else "gone"
                    else:
                        res = "could not tell"
                    events.append(dict(t=t, kind="check_result", track=check_tid, label=tr.label, result=res,
                                       status=tr.status, looks_in_view=check_views, hits=check_hits))
                    last_check_tid, mode, check_tid, arrived_t, goal = check_tid, "explore", None, None, None
                    mind_due = mind_due or (mind and (res == "not there" or all_done))
            elif goal is None or (np.linalg.norm(body.xy - goal) < 0.6) or stuck:
                # explore: the nearest reachable frontier (fixed rule, no language model)
                if stuck and goal is not None:
                    banned.append(goal.copy())  # never pick a frontier here again
                    events.append(dict(t=t, kind="stuck", goal=goal.round(2).tolist()))
                    last_move_t = t
                wk, c, dist, fr = reachable_frontiers()
                fr = [(f, n) for f, n in fr if all(np.linalg.norm(f - b) > 1.0 for b in banned)]
                if fr:
                    f, n = min(fr, key=lambda q: dist[smap.cell(q[0])])
                    cells_path = descend(geodesic(wk, [smap.cell(f)], smap.res), c)
                    new_path = np.array([smap.pos(q) for q in cells_path]) if cells_path else np.zeros((0, 2))
                    if goal is None or np.linalg.norm(f - goal) > 0.5:
                        events.append(dict(t=t, kind="new_waypoint", xy=f.round(2).tolist(), frontier_cells=int(n)))
                    goal, path, prog, all_done = f, new_path, 0, False
                    best_gd, last_move_t = 1e9, t
                else:
                    if not all_done:
                        events.append(dict(t=t, kind="explored_all"))
                        mind_due, all_done = mind_due or mind, True
                    goal, path = None, np.zeros((0, 2))

        # steer along the path with a blended skill vector
        if len(path) >= 2:
            tgt, prog = lookahead(path, body.xy, prog, 0.8)
            d = tgt - body.xy
            rel = wrap_deg(math.degrees(math.atan2(d[1], d[0]) - body.yaw)) if np.linalg.norm(d) > 0.05 else 0.0
            zt = target_z(rel)
        elif mode == "check":
            zt = Z["move-ego-0-0"]  # arrived: stand and look
        else:
            zt = Z["yaw-L-2-free"] if "yaw-L-2-free" in Z else Z["move-ego-0-0"]  # nothing left to explore: look around
        z = z + (zt - z) / 15.0 * 2.0
        z = z / z.norm() * norm
        try:
            body.step_z(z)
        except ValueError:
            fell = True
        if k % 2 == 0:
            follow_w.append_data(body.follow_view(math.degrees(body.yaw) + 180, 360, 480))
        if not fell and body.head_z < 0.8:
            # fallen: hold the standing skill for up to 8 s and see whether Motivo gets the body back up
            events.append(dict(t=t, kind="fell", xy=body.xy.round(2).tolist()))
            log("fell at", round(t, 1))
            up = False
            for j in range(240):
                try:
                    body.step_z(Z["move-ego-0-0"])
                except ValueError:
                    break
                if j % 2 == 0:
                    follow_w.append_data(body.follow_view(math.degrees(body.yaw) + 180, 360, 480))
                if body.head_z > 1.35 and j > 30:
                    up = True
                    break
            k_extra = j + 1
            events.append(dict(t=t + k_extra / 30.0, kind="got_up" if up else "stayed_down", seconds=round(k_extra / 30.0, 1)))
            log("got up" if up else "stayed down", round(k_extra / 30.0, 1), "s")
            falls += 1
            if not up:
                fell = True
                break
            z, path, goal, last_move_t = Z["move-ego-0-0"].clone(), np.zeros((0, 2)), None, t
        if fell:
            events.append(dict(t=t, kind="fell", xy=body.xy.round(2).tolist()))
            break
        if k % 300 == 0:
            log(f"t={t:.0f}s xy={body.xy.round(1).tolist()} objects={len(mem.tracks)} sightings={len(mem.sightings)}")
    follow_w.close()
    head_w.close()

    # save the body's memory and log (agent side) and the truth (evaluator side) separately
    np.savez_compressed(f"{out}/maps.npz", shape=np.array(smap.floor.shape), origin=smap.origin, res=smap.res,
                        **{f"seen_{i}": m["seen"] for i, m in enumerate(maps)},
                        **{f"walk_{i}": m["walk"] for i, m in enumerate(maps)},
                        t=np.array([m["t"] for m in maps]), xy=np.array([m["xy"] for m in maps]))
    for lk in looks:
        open(f"{out}/frame_{lk['frame']:04d}.jpg", "wb").write(lk.pop("jpg"))
    memory = dict(table=mem.table(),
                  sightings=[dict(id=s.id, label=s.label, score=s.score, xyz=s.xyz.round(2).tolist(), t=s.t,
                                  frame=s.frame, body_xy=s.body_xy.round(2).tolist()) for s in mem.sightings],
                  tracks=[dict(id=tr.id, label=tr.label, sightings=tr.sightings, misses=tr.misses, status=tr.status) for tr in mem.tracks])
    json.dump(memory, open(f"{out}/memory.json", "w"), indent=1)
    json.dump(dict(house=house, seed=seed, seconds_requested=seconds, seconds_run=round(t, 2), fell=fell,
                   start=start.tolist(), events=events, looks=looks, prompts=PROMPTS,
                   perception="RGB-D head camera, SAM 3.1 open weights; perfect odometry from the simulator",
                   explorer="nearest reachable frontier on the self-built map (rule, no language model)",
                   mind=("GPT-6.1 Terra (Bedrock) as stand-in for Muse Spark" if mind else "off"), decisions=decisions),
              open(f"{out}/log.json", "w"))
    truth = dict(objects=[dict(type=o.type, xy=np.asarray(o.xy).tolist(), room=o.room) for o in lay.objects],
                 rooms=[dict(id=r.id, type=r.type) for r in lay.rooms], evaluator_events=evaluator)
    json.dump(truth, open(f"{out}/truth_evaluator_only.json", "w"))
    vol.commit()
    summary = dict(run=run, seconds_run=round(t, 2), fell=fell, falls=falls,
                   got_up=sum(e["kind"] == "got_up" for e in events), objects=len(mem.tracks), sightings=len(mem.sightings),
                   misses=sum(len(tr.misses) for tr in mem.tracks), explored_cells=int(smap.seen.sum()),
                   walked_waypoints=sum(e["kind"] == "new_waypoint" for e in events),
                   stuck=sum(e["kind"] == "stuck" for e in events), minutes=round((time.time() - t0) / 60, 1),
                   mind_calls=len(decisions), checks=[{k: e[k] for k in ("t", "label", "result", "status")} for e in events if e["kind"] == "check_result"],
                   evaluator=evaluator)
    log(summary)
    return summary


@app.local_entrypoint()
def main(house: str = "procthor-10k-val_val_935", seconds: float = 60.0, seed: int = 0, run_name: str = "",
         mind: bool = False, mind_every: float = 20.0, move_at: float = -1.0):
    print(live_run.remote(house, seconds, seed, run_name, mind, mind_every, move_at))


@app.local_entrypoint()
def batch(seconds: float = 600.0, tag: str = "r10", mind: bool = False, move_at: float = -1.0):
    """The preselected robustness houses (configs/robustness_houses.json), in parallel. Every run is kept."""
    import json
    houses = json.load(open(os.path.join(CFG, "robustness_houses.json")))
    calls = [live_run.spawn(h, seconds, 0, f"{tag}_{h}", mind, 20.0, move_at) for h in houses]
    for h, c in zip(houses, calls):
        try:
            print(h, c.get())
        except Exception as e:  # keep going; a crash is a result too
            print(h, "CRASHED", type(e).__name__, str(e)[:300])
