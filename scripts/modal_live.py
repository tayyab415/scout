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
                      "opencv-python-headless", "mujoco==3.17.0", "h5py")
         .pip_install("metamotivo[huggingface,humenv] @ git+https://github.com/facebookresearch/metamotivo.git")
         .run_commands("python -c 'import torch; print(torch.__version__)'")
         .env({"HF_HOME": "/data/hf", "MUJOCO_GL": "egl"})
         .add_local_dir(SRC, "/pkg")
         .add_local_dir(CFG, "/cfg"))
_tok = os.path.expanduser("~/.cache/huggingface/token")
secrets = [modal.Secret.from_dict({"HF_TOKEN": open(_tok).read().strip() if modal.is_local() and os.path.exists(_tok) else ""})]

PROMPTS = ["bed", "sofa", "television", "dining table", "fridge", "kitchen counter", "chair", "armchair",
           "floor lamp", "painting", "laptop", "toilet", "sink", "dresser", "bathtub", "house plant"]
FOVY = 75
SIZE = 512
# walking direction (deg, + = left) reached by mixing forward with a sidestep, measured in docs/results/motion_tests.md
MIX_LEFT = ([-18, 12, 71, 119], [0.0, 0.25, 0.5, 0.75])
MIX_RIGHT = ([18, 33, 85, 122], [0.0, 0.25, 0.5, 0.75])


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=65536, timeout=5400, secrets=secrets)
def live_run(house: str, seconds: float = 60.0, seed: int = 0, run_name: str = ""):
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
    body = MotivoBody("/data/house", house, "/data/scout/z_motion.pt", device="cuda")
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
    mem = Episodic()
    events, looks, maps = [], [], []
    follow_w = imageio.get_writer(f"{out}/follow.mp4", fps=15, codec="libx264", quality=7, macro_block_size=1)
    head_w = imageio.get_writer(f"{out}/head.mp4", fps=2, codec="libx264", quality=8, macro_block_size=1)
    z = Z["move-ego-0-0"].clone()
    path, prog, goal = np.zeros((0, 2)), 0, None
    last_xy, last_move_t, fell = body.xy, 0.0, False
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
                # a remembered object that should be in view but is not: count it only after two looks in a row,
                # from within 3 m, so one missed detection does not become a false "it moved"
                expected = {tr.id: tr for tr, _ in mem.expected_in_view(cp, cm, FOVY, depth, max_range=3.0)}
                for tr in mem.tracks:
                    if tr.label in seen_labels and tr.id not in expected:
                        miss_run.pop(tr.id, None)
                for tid, tr in expected.items():
                    if tr.label in seen_labels or t - mem.sightings[tr.sightings[-1]].t <= 3.0:
                        miss_run.pop(tid, None)
                        continue
                    miss_run[tid] = miss_run.get(tid, 0) + 1
                    if miss_run[tid] == 2:
                        tr.misses.append((t, frame_id))
                        events.append(dict(t=t, kind="expected_not_seen", label=tr.label, track=tr.id, frame=frame_id))
                head_w.append_data(rgb)
            looks.append(look)
            if k % 30 == 0:
                maps.append(dict(t=t, seen=np.packbits(smap.seen), walk=np.packbits(smap.walkable()), xy=body.xy.tolist()))

            # choose a waypoint: the nearest reachable frontier (fixed rule, no language model)
            moved = np.linalg.norm(body.xy - last_xy)
            if moved > 0.3:
                last_xy, last_move_t = body.xy, t
            stuck = t - last_move_t > 3.0
            reached = goal is not None and np.linalg.norm(body.xy - goal) < 0.6
            if goal is None or reached or stuck or k % 60 == 0:
                if stuck and goal is not None:
                    gi = smap.cell(goal)
                    smap.obst[max(gi[0] - 2, 0):gi[0] + 3, max(gi[1] - 2, 0):gi[1] + 3] += 5
                    events.append(dict(t=t, kind="stuck", goal=goal.round(2).tolist()))
                    last_move_t = t
                wk = smap.walkable()
                c = smap.cell(body.xy)
                wk[max(c[0] - 3, 0):c[0] + 4, max(c[1] - 3, 0):c[1] + 4] |= ~smap.blocked[max(c[0] - 3, 0):c[0] + 4, max(c[1] - 3, 0):c[1] + 4]
                dist = geodesic(wk, [c], smap.res)
                fr = [(f, n) for f, n in smap.frontiers() if np.isfinite(dist[smap.cell(f)]) and np.linalg.norm(f - body.xy) > 0.8]
                if fr:
                    f, n = min(fr, key=lambda q: dist[smap.cell(q[0])])
                    cells_path = descend(geodesic(wk, [smap.cell(f)], smap.res), c)
                    new_path = np.array([smap.pos(q) for q in cells_path]) if cells_path else np.zeros((0, 2))
                    if goal is None or np.linalg.norm(f - goal) > 0.5:
                        events.append(dict(t=t, kind="new_waypoint", xy=f.round(2).tolist(), frontier_cells=int(n)))
                    goal, path, prog = f, new_path, 0
                else:
                    goal, path = None, np.zeros((0, 2))

        # steer along the path with a blended skill vector
        if len(path) >= 2:
            tgt, prog = lookahead(path, body.xy, prog, 0.8)
            d = tgt - body.xy
            rel = wrap_deg(math.degrees(math.atan2(d[1], d[0]) - body.yaw)) if np.linalg.norm(d) > 0.05 else 0.0
            zt = target_z(rel)
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
        if fell or body.head_z < 0.8:
            events.append(dict(t=t, kind="fell", xy=body.xy.round(2).tolist()))
            log("fell at", round(t, 1))
            fell = True
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
                  tracks=[dict(id=tr.id, label=tr.label, sightings=tr.sightings, misses=tr.misses) for tr in mem.tracks])
    json.dump(memory, open(f"{out}/memory.json", "w"), indent=1)
    json.dump(dict(house=house, seed=seed, seconds_requested=seconds, seconds_run=round(t, 2), fell=fell,
                   start=start.tolist(), events=events, looks=looks, prompts=PROMPTS,
                   perception="RGB-D head camera, SAM 3.1 open weights; perfect odometry from the simulator",
                   explorer="nearest reachable frontier on the self-built map (rule, no language model)"),
              open(f"{out}/log.json", "w"))
    truth = dict(objects=[dict(type=o.type, xy=np.asarray(o.xy).tolist(), room=o.room) for o in lay.objects],
                 rooms=[dict(id=r.id, type=r.type) for r in lay.rooms])
    json.dump(truth, open(f"{out}/truth_evaluator_only.json", "w"))
    vol.commit()
    summary = dict(run=run, seconds_run=round(t, 2), fell=fell, objects=len(mem.tracks), sightings=len(mem.sightings),
                   misses=sum(len(tr.misses) for tr in mem.tracks), explored_cells=int(smap.seen.sum()),
                   walked_waypoints=sum(e["kind"] == "new_waypoint" for e in events),
                   stuck=sum(e["kind"] == "stuck" for e in events), minutes=round((time.time() - t0) / 60, 1))
    log(summary)
    return summary


@app.local_entrypoint()
def main(house: str = "procthor-10k-val_val_935", seconds: float = 60.0, seed: int = 0, run_name: str = ""):
    print(live_run.remote(house, seconds, seed, run_name))


@app.local_entrypoint()
def batch(seconds: float = 600.0, tag: str = "r10"):
    """The preselected robustness houses (configs/robustness_houses.json), in parallel. Every run is kept."""
    import json
    houses = json.load(open(os.path.join(CFG, "robustness_houses.json")))
    calls = [live_run.spawn(h, seconds, 0, f"{tag}_{h}") for h in houses]
    for h, c in zip(houses, calls):
        try:
            print(h, c.get())
        except Exception as e:  # keep going; a crash is a result too
            print(h, "CRASHED", type(e).__name__, str(e)[:300])
