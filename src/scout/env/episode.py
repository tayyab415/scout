"""Run one Scout episode: explore, then a frozen moved / unchanged verification pair branched from one state.

Plain Python (GPU machine with MuJoCo EGL, Motivo, sam3). `scripts/modal_env.py` wraps it for Modal and
`scripts/run_local.py` runs it on any GPU box (AWS). Protocol: docs/environment.md.
"""
from __future__ import annotations

import os


def run_episode(house: str, brain: str = "terra", seed: int = 0, tag: str = "dev", explore_s: float = 240.0,
                ask_after_s: float = 20.0, answer_timeout_s: float = 90.0, quiz: bool = True, mind: bool = True,
                data_root: str = "/data", sam3_dir: str = "/sam3", commit=None):
    """One Scout episode (see the module docstring). `commit` is called after the outputs are written."""
    import copy
    import json
    import subprocess
    import sys
    import time

    import imageio.v2 as imageio
    import numpy as np
    from ..brains import make_brain
    from .agent import LOOK_EVERY, Agent
    from .core import MOVABLE, PROMPTS, ScoutEnv
    from ..perception.sam import Sam31
    t_wall = time.time()
    ep = f"{brain}_{house}_s{seed}"
    out = f"{data_root}/scout/env/{tag}/{ep}"
    os.makedirs(out, exist_ok=True)
    log = lambda *a: print(f"[{ep} {time.time() - t_wall:6.0f}s]", *a, flush=True)  # noqa: E731
    git = subprocess.run(["git", "-C", sam3_dir, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    env = ScoutEnv(house, seed, root=f"{data_root}/house", skills=f"{data_root}/scout/z_motion.pt", eyes=Sam31())
    agent = Agent(make_brain(brain), mind=mind)
    env.reset()
    log("reset at", env.start)
    vis = []            # evaluator: visible object instances per look
    maps = []           # agent map snapshots once per second, for the video

    def run_until(t_end, writer, times, stop=None):
        while env.t < t_end - 1e-9:
            # evaluator visibility is rendered at the same instant as a possible head capture, but kept only if the
            # agent really captured a frame on this tick (it does not while recovering from a fall)
            pending = dict(t=round(env.t, 3), visible=env.visible_instances()) if env.k % LOOK_EVERY == 0 else None
            n_looks = len(agent.looks)
            if env.k % 2 == 0:
                writer.append_data(env.observer_view())
                times.append(round(env.t, 3))
            if env.k % 30 == 0 and agent.smap is not None:
                maps.append(dict(t=round(env.t, 3), seen=np.packbits(agent.smap.seen), walk=np.packbits(agent.smap.walkable()),
                                 xy=[float(v) for v in env.body.xy], yaw=float(env.body.yaw),
                                 goal=None if agent.goal is None else [float(v) for v in agent.goal],
                                 check=None if agent.check is None else [float(v) for v in agent.check["xy"]]))
            alive = agent.tick(env)
            if pending is not None and len(agent.looks) > n_looks:
                lk = agent.looks[-1]
                vis.append(dict(t=lk["t"], frame=lk["frame"], visible=pending["visible"]))
            if not alive:
                return "ended"
            if stop is not None and stop():
                return "stopped"
        return "time"

    # 1. explore
    w = imageio.get_writer(f"{out}/observer_prefix.mp4", fps=15, codec="libx264", quality=7, macro_block_size=1)
    times = []
    end = run_until(explore_s, w, times)
    w.close()
    log("explored:", end, "objects", len(agent.mem.tracks), "falls", agent.falls)

    # 2. freeze the target (evaluator side)
    truth0 = env.truth_objects()
    visible_now = env.visible_instances()
    cand = []
    for tr in agent.mem.tracks:
        if tr.label not in MOVABLE or tr.status != "ok" or len(tr.sightings) < 3:  # the evaluator checks it is real
            continue
        p = tr.last_xy(agent.mem)
        inst = [o for o in truth0 if o["category"] == tr.label and np.linalg.norm(np.array(o["xy"]) - p) < 1.5]
        if not inst:
            continue
        o = min(inst, key=lambda o: np.linalg.norm(np.array(o["xy"]) - p))
        far = np.linalg.norm(np.array(o["xy"]) - env.body.xy) >= 4.0 and o["id"] not in visible_now
        # the question must name an identifiable object: no other instance of its category within 2.5 m
        unique = all(np.linalg.norm(np.array(q["xy"]) - np.array(o["xy"])) > 2.5 for q in truth0
                     if q["category"] == o["category"] and q["id"] != o["id"])
        far = far and unique
        cand.append((len(tr.sightings), far, tr, o))
    cand = [c for c in cand if c[1]]
    manifest = dict(episode=ep, house=house, brain=agent.brain.name, brain_arg=brain,
                    brain_model=getattr(agent.brain, "model", None), seed=seed, explore_s=explore_s,
                    scout_commit=os.environ.get("SCOUT_COMMIT", ""), host=os.environ.get("SCOUT_HOST", "modal"),
                    ask_after_s=ask_after_s, answer_timeout_s=answer_timeout_s, start=env.start, explore_end=end,
                    sam3_commit=git, observation_contract=(
                        "head RGB-D 512x512, 75 deg FOV, 20 deg down; ideal depth; exact head-camera pose and body xy/yaw "
                        "from the simulator (perfect odometry); SAM 3.1 detections every 1 s; map from depth every 0.5 s"))
    arms_out = {}
    if end != "time" or not cand:
        manifest["target"] = None
        manifest["no_target_reason"] = "episode ended early" if end != "time" else "no eligible movable object (seen 3+ times, matched to a real instance, no same-category instance within 2.5 m, >= 4 m away, not in view)"
        log("no target:", manifest["no_target_reason"])
    else:
        n, _, tr, o = max(cand, key=lambda c: c[0])
        rng = np.random.default_rng(seed + 17)
        here = env.layout.room_at(np.array(o["xy"]))
        cells = np.argwhere(env.layout.grid.free)
        rng.shuffle(cells)
        dest = None
        for c in cells[:5000]:
            p = env.layout.grid.pos(c)
            r = env.layout.room_at(p)
            if r is None or (here is not None and r.id == here.id):
                continue
            if np.linalg.norm(p - np.array(o["xy"])) >= 4.0 and np.linalg.norm(p - env.body.xy) >= 6.0:
                dest = p
                break
        ref_t = agent.mem.sightings[tr.sightings[-1]].t  # agent-visible reference: its own sighting at this time
        question = f"Is the {tr.label} you saw at {ref_t:.0f} s still where it was?"
        manifest["target"] = dict(instance=o["id"], category=o["category"], from_xy=o["xy"], track_at_freeze=tr.id,
                                  t_freeze=round(env.t, 3), reference_sighting_t=round(ref_t, 3),
                                  sightings_at_freeze=n, dest_xy=None if dest is None else dest.round(3).tolist(),
                                  dest_room=None if dest is None else env.layout.room_at(dest).type, question=question)
        snap_env, snap_agent = env.snapshot(), copy.deepcopy(agent)
        snap_vis, snap_maps = copy.deepcopy(vis), copy.deepcopy(maps)
        arms = ("unchanged", "moved") if dest is not None else ("unchanged",)
        if dest is None:
            manifest["moved_arm_skipped"] = "no destination in another room >= 4 m from the object and >= 6 m from the body"
        for arm in arms:
            env.restore(snap_env)
            agent = copy.deepcopy(snap_agent)
            vis[:], maps[:] = copy.deepcopy(snap_vis), copy.deepcopy(snap_maps)
            moved = env.move_object(o["id"], dest) if (arm == "moved" and dest is not None) else None
            w = imageio.get_writer(f"{out}/observer_{arm}.mp4", fps=15, codec="libx264", quality=7, macro_block_size=1)
            atimes = []
            t_ask = env.t + ask_after_s
            e1 = run_until(t_ask, w, atimes)
            if e1 == "time":
                # paired baseline: the same brain, same memory, same moment, answering from memory without checking
                rec = copy.deepcopy(agent)
                rec.brain.recall = True
                rec.mode, rec.check = "explore", None  # it answers at once from memory; it is never stepped
                rec.ask(question)
                rec._decide(env.t, env.body.xy)
                recall = [q for q in rec.qa if q["kind"] == "answer"][-1:]
                agent.ask(question)
                n_qa = len(agent.qa)
                e2 = run_until(env.t + answer_timeout_s, w, atimes,
                               stop=lambda: any(q["kind"] == "answer" for q in agent.qa[n_qa:]) and agent.mode != "check")
                run_until(env.t + 3.0, w, atimes)
            else:
                e2, recall = e1, []
            w.close()
            qz = agent.quiz(PROMPTS, t=round(env.t, 3)) if (quiz and arm == "unchanged") else []
            arms_out[arm] = dict(moved=moved, end=e2, qa=agent.qa, quiz=qz)
            adir = f"{out}/{arm}"
            os.makedirs(f"{adir}/frames", exist_ok=True)
            for lk in agent.looks:
                p = f"{adir}/frames/{lk['frame']:04d}.jpg"
                open(p, "wb").write(lk["jpg"])
            looks = [{k: v for k, v in lk.items() if k != "jpg"} for lk in agent.looks]
            json.dump(dict(events=agent.events, looks=looks, decisions=agent.decisions, qa=agent.qa, quiz=qz, recall=recall,
                           falls=agent.falls, got_up=agent.got_up, ended=agent.ended, t_end=round(env.t, 3),
                           observer_times_prefix=times, observer_times=atimes), open(f"{adir}/log.json", "w"))
            json.dump(agent.memory_dump(), open(f"{adir}/memory.json", "w"))
            json.dump(dict(objects_start=truth0, objects_end=env.truth_objects(), moves=env.moves, visibility=vis,
                           instances={i: dict(category=v["category"], key=v["key"]) for i, v in env.instances.items()}),
                      open(f"{adir}/evaluator.json", "w"))
            m0 = agent.smap
            np.savez_compressed(f"{adir}/maps.npz", shape=np.array(m0.floor.shape), origin=m0.origin, res=m0.res,
                                t=np.array([m["t"] for m in maps]), xy=np.array([m["xy"] for m in maps]),
                                yaw=np.array([m["yaw"] for m in maps]),
                                **{f"seen_{i}": m["seen"] for i, m in enumerate(maps)},
                                **{f"walk_{i}": m["walk"] for i, m in enumerate(maps)})
            json.dump([dict(t=m["t"], goal=m["goal"], check=m["check"]) for m in maps], open(f"{adir}/map_meta.json", "w"))
            log(arm, "answer:", [q.get("say") for q in agent.qa if q["kind"] == "answer"])
    if not arms_out:  # still save the whole exploration, with everything the video and scorer need
        adir = f"{out}/explore_only"
        os.makedirs(f"{adir}/frames", exist_ok=True)
        for lk in agent.looks:
            open(f"{adir}/frames/{lk['frame']:04d}.jpg", "wb").write(lk["jpg"])
        json.dump(dict(events=agent.events, looks=[{k: v for k, v in lk.items() if k != "jpg"} for lk in agent.looks],
                       decisions=agent.decisions, qa=agent.qa, quiz=[], falls=agent.falls, got_up=agent.got_up, ended=agent.ended,
                       t_end=round(env.t, 3), observer_times_prefix=times, observer_times=[]), open(f"{adir}/log.json", "w"))
        if agent.smap is not None:
            m0 = agent.smap
            np.savez_compressed(f"{adir}/maps.npz", shape=np.array(m0.floor.shape), origin=m0.origin, res=m0.res,
                                t=np.array([m["t"] for m in maps]), xy=np.array([m["xy"] for m in maps]),
                                yaw=np.array([m["yaw"] for m in maps]),
                                **{f"seen_{i}": m["seen"] for i, m in enumerate(maps)},
                                **{f"walk_{i}": m["walk"] for i, m in enumerate(maps)})
            json.dump([dict(t=m["t"], goal=m["goal"], check=m["check"]) for m in maps], open(f"{adir}/map_meta.json", "w"))
        json.dump(agent.memory_dump(), open(f"{adir}/memory.json", "w"))
        json.dump(dict(objects_start=truth0, objects_end=env.truth_objects(), moves=env.moves, visibility=vis,
                       instances={i: dict(category=v["category"], key=v["key"]) for i, v in env.instances.items()}),
                  open(f"{adir}/evaluator.json", "w"))
    manifest["arms"] = {a: dict(end=v["end"], moved=v["moved"]) for a, v in arms_out.items()}
    manifest["wall_minutes"] = round((time.time() - t_wall) / 60, 1)
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
    if commit is not None:
        commit()
    return json.loads(json.dumps(dict(manifest=manifest, answers={a: [q for q in v["qa"] if q["kind"] == "answer"]
                                                                  for a, v in arms_out.items()}), default=str))
