"""Run Scout environment episodes: explore, then a frozen moved / unchanged verification pair branched from one state.

    modal run scripts/modal_env.py::episode --house procthor-10k-val_val_935 --brain terra
    modal run scripts/modal_env.py::compare --houses-file leaderboard_houses.json --brains terra,llama4,rules,random

One episode:
  1. reset in a held-out house; the agent explores on its own for `explore_s` seconds (brain decides re-checks).
  2. the evaluator freezes a target: the movable object instance the agent has seen most (matched to a confident
     memory track), plus a destination in another room and the question "Is the <category> still where you last saw it?".
  3. snapshot env + agent. Arm "moved": the evaluator moves the target. Arm "unchanged": nothing changes. Both arms then
     run `ask_after_s` seconds on their own, get the same question, and run until the answer (or `answer_timeout_s`).
  4. memory quiz (unchanged arm): one question per category, graded later against visibility history.
Outputs on the volume under /data/scout/env/<tag>/<episode>/ (see README in that folder of the repo: docs/environment.md).
"""
import os

import modal

app = modal.App("scout-env")
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
         .env({"HF_HOME": "/data/hf", "MUJOCO_GL": "egl"})
         .add_local_dir(SRC, "/pkg")
         .add_local_dir(CFG, "/cfg"))
_tok = os.path.expanduser("~/.cache/huggingface/token")


def _aws_env() -> dict:
    if not modal.is_local():
        return {"AWS_DEFAULT_REGION": "us-east-1"}
    import configparser
    cp = configparser.ConfigParser()
    cp.read(os.path.expanduser("~/.aws/credentials"))
    d = cp["default"] if "default" in cp else {}
    env = {"AWS_ACCESS_KEY_ID": d.get("aws_access_key_id", ""), "AWS_SECRET_ACCESS_KEY": d.get("aws_secret_access_key", ""),
           "AWS_DEFAULT_REGION": "us-east-1"}
    if d.get("aws_session_token"):
        env["AWS_SESSION_TOKEN"] = d["aws_session_token"]
    return env


secrets = [modal.Secret.from_dict({"HF_TOKEN": open(_tok).read().strip() if modal.is_local() and os.path.exists(_tok) else ""}),
           modal.Secret.from_dict(_aws_env())]


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=65536, timeout=7200, secrets=secrets)
def run_episode(house: str, brain: str = "terra", seed: int = 0, tag: str = "dev", explore_s: float = 240.0,
                ask_after_s: float = 20.0, answer_timeout_s: float = 90.0, quiz: bool = True, mind: bool = True):
    import copy
    import json
    import subprocess
    import sys
    import time

    import imageio.v2 as imageio
    import numpy as np
    sys.path.insert(0, "/pkg")
    from scout.brains import make_brain
    from scout.env.agent import LOOK_EVERY, Agent
    from scout.env.core import MOVABLE, PROMPTS, ScoutEnv
    from scout.perception.sam import Sam31
    t_wall = time.time()
    ep = f"{brain}_{house}_s{seed}"
    out = f"/data/scout/env/{tag}/{ep}"
    os.makedirs(out, exist_ok=True)
    log = lambda *a: print(f"[{ep} {time.time() - t_wall:6.0f}s]", *a, flush=True)  # noqa: E731
    git = subprocess.run(["git", "-C", "/sam3", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()

    env = ScoutEnv(house, seed, eyes=Sam31())
    agent = Agent(make_brain(brain), mind=mind)
    env.reset()
    log("reset at", env.start)
    vis = []            # evaluator: visible object instances per look
    maps = []           # agent map snapshots once per second, for the video

    def run_until(t_end, writer, times, stop=None):
        while env.t < t_end - 1e-9:
            if env.k % LOOK_EVERY == 0:
                vis.append(dict(t=round(env.t, 3), frame=len(agent.looks), visible=env.visible_instances()))
            if env.k % 2 == 0:
                writer.append_data(env.observer_view())
                times.append(round(env.t, 3))
            if env.k % 30 == 0 and agent.smap is not None:
                maps.append(dict(t=round(env.t, 3), seen=np.packbits(agent.smap.seen), walk=np.packbits(agent.smap.walkable()),
                                 xy=[float(v) for v in env.body.xy], yaw=float(env.body.yaw),
                                 goal=None if agent.goal is None else [float(v) for v in agent.goal],
                                 check=None if agent.check is None else [float(v) for v in agent.check["xy"]]))
            if not agent.tick(env):
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
        if tr.label not in MOVABLE or tr.status != "ok" or not tr.confident(agent.mem):
            continue
        p = tr.last_xy(agent.mem)
        inst = [o for o in truth0 if o["category"] == tr.label and np.linalg.norm(np.array(o["xy"]) - p) < 1.5]
        if not inst:
            continue
        o = min(inst, key=lambda o: np.linalg.norm(np.array(o["xy"]) - p))
        far = np.linalg.norm(np.array(o["xy"]) - env.body.xy) >= 6.0 and o["id"] not in visible_now
        cand.append((len(tr.sightings), far, tr, o))
    cand = [c for c in cand if c[1]]
    manifest = dict(episode=ep, house=house, brain=agent.brain.name, seed=seed, explore_s=explore_s,
                    ask_after_s=ask_after_s, answer_timeout_s=answer_timeout_s, start=env.start, explore_end=end,
                    sam3_commit=git, observation_contract=(
                        "head RGB-D 512x512, 75 deg FOV, 20 deg down; ideal depth; exact head-camera pose and body xy/yaw "
                        "from the simulator (perfect odometry); SAM 3.1 detections every 1 s; map from depth every 0.5 s"))
    arms_out = {}
    if end != "time" or not cand:
        manifest["target"] = None
        manifest["no_target_reason"] = "episode ended early" if end != "time" else "no eligible movable object (confident, matched, >= 6 m away, not in view)"
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
        question = f"Is the {tr.label} still where you last saw it?"
        manifest["target"] = dict(instance=o["id"], category=o["category"], from_xy=o["xy"], track_at_freeze=tr.id,
                                  sightings_at_freeze=n, dest_xy=None if dest is None else dest.round(3).tolist(),
                                  dest_room=None if dest is None else env.layout.room_at(dest).type, question=question)
        snap_env, snap_agent = env.snapshot(), copy.deepcopy(agent)
        snap_vis, snap_maps = copy.deepcopy(vis), copy.deepcopy(maps)
        for arm in ("unchanged", "moved"):
            env.restore(snap_env)
            agent = copy.deepcopy(snap_agent)
            vis[:], maps[:] = copy.deepcopy(snap_vis), copy.deepcopy(snap_maps)
            moved = env.move_object(o["id"], dest) if (arm == "moved" and dest is not None) else None
            w = imageio.get_writer(f"{out}/observer_{arm}.mp4", fps=15, codec="libx264", quality=7, macro_block_size=1)
            atimes = []
            t_ask = env.t + ask_after_s
            e1 = run_until(t_ask, w, atimes)
            if e1 == "time":
                agent.ask(question)
                n_qa = len(agent.qa)
                e2 = run_until(env.t + answer_timeout_s, w, atimes,
                               stop=lambda: any(q["kind"] == "answer" for q in agent.qa[n_qa:]) and agent.mode != "check")
                run_until(env.t + 3.0, w, atimes)
            else:
                e2 = e1
            w.close()
            qz = agent.quiz(PROMPTS) if (quiz and arm == "unchanged") else []
            arms_out[arm] = dict(moved=moved, end=e2, qa=agent.qa, quiz=qz)
            adir = f"{out}/{arm}"
            os.makedirs(f"{adir}/frames", exist_ok=True)
            for lk in agent.looks:
                p = f"{adir}/frames/{lk['frame']:04d}.jpg"
                open(p, "wb").write(lk["jpg"])
            looks = [{k: v for k, v in lk.items() if k != "jpg"} for lk in agent.looks]
            json.dump(dict(events=agent.events, looks=looks, decisions=agent.decisions, qa=agent.qa, quiz=qz,
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
    if not arms_out:  # still save the exploration
        adir = f"{out}/explore_only"
        os.makedirs(f"{adir}/frames", exist_ok=True)
        for lk in agent.looks:
            open(f"{adir}/frames/{lk['frame']:04d}.jpg", "wb").write(lk["jpg"])
        json.dump(dict(events=agent.events, looks=[{k: v for k, v in lk.items() if k != "jpg"} for lk in agent.looks],
                       decisions=agent.decisions, falls=agent.falls, got_up=agent.got_up, ended=agent.ended,
                       t_end=round(env.t, 3), observer_times_prefix=times), open(f"{adir}/log.json", "w"))
        json.dump(agent.memory_dump(), open(f"{adir}/memory.json", "w"))
        json.dump(dict(objects_start=truth0, objects_end=env.truth_objects(), moves=env.moves, visibility=vis,
                       instances={i: dict(category=v["category"], key=v["key"]) for i, v in env.instances.items()}),
                  open(f"{adir}/evaluator.json", "w"))
    manifest["arms"] = {a: dict(end=v["end"], moved=v["moved"]) for a, v in arms_out.items()}
    manifest["wall_minutes"] = round((time.time() - t_wall) / 60, 1)
    json.dump(manifest, open(f"{out}/manifest.json", "w"), indent=1)
    vol.commit()
    return json.loads(json.dumps(dict(manifest=manifest, answers={a: [q for q in v["qa"] if q["kind"] == "answer"]
                                                                  for a, v in arms_out.items()}), default=str))


@app.local_entrypoint()
def episode(house: str = "procthor-10k-val_val_935", brain: str = "terra", seed: int = 0, tag: str = "dev",
            explore_s: float = 240.0):
    import json
    print(json.dumps(run_episode.remote(house, brain, seed, tag, explore_s), indent=1))


@app.local_entrypoint()
def compare(houses_file: str = "leaderboard_houses.json", brains: str = "terra,llama4,rules,random", tag: str = "cmp",
            explore_s: float = 240.0):
    """Frozen comparison: every brain, same houses, same seeds, same budgets. Every episode is kept, including errors."""
    import json
    houses = json.load(open(os.path.join(CFG, houses_file)))
    plan = [(b, h) for b in brains.split(",") for h in houses]
    json.dump(dict(houses=houses, brains=brains.split(","), explore_s=explore_s, seed=0, tag=tag), open(
        os.path.join(CFG, f"frozen_{tag}.json"), "w"), indent=1)
    calls = [(b, h, run_episode.spawn(h, b, 0, tag, explore_s)) for b, h in plan]
    for b, h, c in calls:
        try:
            r = c.get()
            print("EP", b, h, json.dumps({"target": r["manifest"].get("target"), "answers": r["answers"]}))
        except Exception as e:
            print("EP", b, h, "CRASHED", type(e).__name__, str(e)[:400])
