"""Run Scout environment episodes: explore, then a frozen moved / unchanged verification pair branched from one state.

    modal run scripts/modal_env.py::episode --house procthor-10k-val_val_935 --brain spark
    modal run scripts/modal_env.py::compare --houses-file leaderboard_houses.json --brains spark,llama4,rules,random

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
VOLUME = os.environ.get("SCOUT_VOLUME", "tribe-lab-claude")
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
CFG = os.path.join(os.path.dirname(__file__), "..", "configs")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "ffmpeg", "libgl1", "libegl1", "libglib2.0-0")
         .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
         .run_commands("git clone https://github.com/facebookresearch/sam3.git /sam3 && pip install -e /sam3")
         .pip_install("imageio[ffmpeg]", "pillow", "einops", "decord", "pycocotools", "psutil", "scipy", "pandas",
                      "opencv-python-headless", "mujoco==3.17.0", "h5py", "requests")
         .pip_install("metamotivo[huggingface,humenv] @ git+https://github.com/facebookresearch/metamotivo.git")
         .env({"HF_HOME": "/data/hf", "MUJOCO_GL": "egl"})
         .add_local_dir(SRC, "/pkg")
         .add_local_dir(CFG, "/cfg"))
_tok = os.path.expanduser("~/.cache/huggingface/token")


def _model_env() -> dict:
    """Meta Model API key for the brains, passed as an ephemeral Modal secret (never written to disk)."""
    if not modal.is_local():
        return {}
    key = os.environ.get("MODEL_API_KEY", "")
    if not key:
        p = os.path.expanduser(os.environ.get("MODEL_API_KEY_FILE", "~/.config/meta/model_api_key"))
        if os.path.exists(p):
            key = open(p).read().strip()
    return {"MODEL_API_KEY": key, "LLAMA_API_KEY": os.environ.get("LLAMA_API_KEY", "")}


secrets = [modal.Secret.from_dict({"HF_TOKEN": open(_tok).read().strip() if modal.is_local() and os.path.exists(_tok) else ""}),
           modal.Secret.from_dict(_model_env())]


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=65536, timeout=7200, secrets=secrets)
def run_episode(house: str, brain: str = "spark", seed: int = 0, tag: str = "dev", explore_s: float = 240.0,
                ask_after_s: float = 20.0, answer_timeout_s: float = 90.0, quiz: bool = True, mind: bool = True):
    import sys
    sys.path.insert(0, "/pkg")
    from scout.env.episode import run_episode as _run
    return _run(house, brain, seed, tag, explore_s, ask_after_s, answer_timeout_s, quiz, mind, commit=vol.commit)


@app.local_entrypoint()
def episode(house: str = "procthor-10k-val_val_935", brain: str = "spark", seed: int = 0, tag: str = "dev",
            explore_s: float = 240.0):
    import json
    print(json.dumps(run_episode.remote(house, brain, seed, tag, explore_s), indent=1))


@app.local_entrypoint()
def compare(houses_file: str = "leaderboard_houses.json", brains: str = "spark,llama4,rules,random", tag: str = "cmp",
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


render_image = (modal.Image.debian_slim(python_version="3.12").apt_install("ffmpeg", "fonts-dejavu-core")
                .pip_install("imageio[ffmpeg]", "pillow", "numpy")
                .add_local_file(os.path.join(os.path.dirname(__file__), "env_video.py"), "/scripts/env_video.py")
                .add_local_file(os.path.join(os.path.dirname(__file__), "demo_cut.py"), "/scripts/demo_cut.py")
                .add_local_dir(SRC, "/pkg"))


@app.function(image=render_image, volumes={"/data": vol}, cpu=4, memory=16384, timeout=3600)
def render(tag: str, episode: str, arm: str) -> str:
    import sys
    sys.path.insert(0, "/scripts")
    import env_video
    d = f"/data/scout/env/{tag}/{episode}"
    if not os.path.exists(f"{d}/{arm}/log.json"):
        return f"skip {episode}/{arm} (not in this episode)"
    out = f"{d}/{arm}.mp4"
    env_video.compose(d, arm, out)
    vol.commit()
    return out


@app.local_entrypoint()
def videos(tag: str = "cmp", episodes: str = ""):
    """Render the moved and unchanged videos for every episode of a tag (or a comma list of episodes)."""
    import subprocess
    if episodes:
        eps = episodes.split(",")
    else:
        import json
        r = subprocess.run(["modal", "volume", "ls", "--json", VOLUME, f"/scout/env/{tag}"], capture_output=True, text=True)
        if r.returncode != 0:
            raise SystemExit(r.stderr[:300])
        eps = [os.path.basename(x["filename"].rstrip("/")) for x in json.loads(r.stdout) if x.get("type", "dir") == "dir"]
    calls = [(e, a, render.spawn(tag, e, a)) for e in eps for a in ("unchanged", "moved", "explore_only")]
    for e, a, c in calls:
        try:
            print("VIDEO", c.get())
        except Exception as ex:
            print("VIDEO", e, a, "FAILED", type(ex).__name__, str(ex)[:200])


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=65536, timeout=1800, secrets=secrets)
def replay_check(house: str = "procthor-10k-val_val_935", steps: int = 120) -> dict:
    """Branch replay proof: from one snapshot, run the same deterministic action sequence twice and compare the
    physics state and head images step by step. No SAM, no language model."""
    import sys

    import numpy as np
    sys.path.insert(0, "/pkg")
    from scout.env.core import ScoutEnv
    env = ScoutEnv(house, 0, eyes=None)
    env.reset()
    Z = env.Z
    for _ in range(60):
        env.step(Z["move-ego-0-1"])
    snap = env.snapshot()
    seq = [Z["move-ego-0-1"] if (i // 30) % 2 == 0 else Z["move-ego-90-1"] for i in range(steps)]
    traces = []
    for rep in range(2):
        env.restore(snap)
        q, imgs = [], []
        for i, z in enumerate(seq):
            env.step(z)
            q.append(env.D.qpos.copy())
            if i % 15 == 0:
                imgs.append(env.observe()["rgb"].astype(np.int16))
        traces.append((np.array(q), imgs, env.t))
    dq = float(np.abs(traces[0][0] - traces[1][0]).max())
    pix = [np.abs(a - b) for a, b in zip(traces[0][1], traces[1][1])]
    # same state, rendered twice: is the renderer itself deterministic?
    r1, r2 = env.observe()["rgb"].astype(np.int16), env.observe()["rgb"].astype(np.int16)
    return dict(house=house, steps=steps, max_abs_qpos_diff=dq, n_images=len(pix),
                images_identical=sum(int(p.max() == 0) for p in pix),
                max_pixel_diff=[int(p.max()) for p in pix], frac_pixels_differing=[round(float((p > 0).mean()), 5) for p in pix],
                same_state_rerender_max_diff=int(np.abs(r1 - r2).max()), t_end=[traces[0][2], traces[1][2]])


@app.local_entrypoint()
def replay(house: str = "procthor-10k-val_val_935"):
    import json
    print(json.dumps(replay_check.remote(house), indent=1))


score_image = (modal.Image.debian_slim(python_version="3.12").pip_install("numpy")
               .add_local_dir(SRC, "/pkg").add_local_dir(CFG, "/cfg")
               .add_local_file(os.path.join(os.path.dirname(__file__), "score_env.py"), "/scripts/score_env.py"))


@app.function(image=score_image, volumes={"/data": vol}, cpu=4, memory=16384, timeout=3600)
def score_remote(tag: str) -> str:
    """Run scripts/score_env.py next to the data (no downloads), writing scores.json and leaderboard.md to the volume."""
    import shutil
    import sys
    import tempfile
    w = tempfile.mkdtemp()
    os.makedirs(f"{w}/runs/env", exist_ok=True)
    os.symlink(f"/data/scout/env/{tag}", f"{w}/runs/env/{tag}")
    os.makedirs(f"{w}/configs", exist_ok=True)
    if os.path.exists(f"/cfg/frozen_{tag}.json"):
        shutil.copy(f"/cfg/frozen_{tag}.json", f"{w}/configs/frozen_{tag}.json")
    os.chdir(w)
    sys.path.insert(0, "/scripts")
    sys.path.insert(0, "/pkg")
    sys.argv = ["score_env.py", "--tag", tag, "--no-fetch", "--layouts", "/data/house/layouts"]
    import score_env
    score_env.main()
    vol.commit()
    return open(f"/data/scout/env/{tag}/leaderboard.md").read()


@app.local_entrypoint()
def score(tag: str = "final"):
    print(score_remote.remote(tag))


@app.function(image=score_image, volumes={"/data": vol}, cpu=2, timeout=900)
def check_evidence(tag: str) -> list:
    """Evidence counts of every question-triggered check in a tag (for diagnosing abstentions)."""
    import glob
    import json
    out = []
    for f in sorted(glob.glob(f"/data/scout/env/{tag}/*/*/log.json")):
        arm = f.split("/")[-2]
        ep = f.split("/")[-3]
        log = json.load(open(f))
        for e in log["events"]:
            if e["kind"] == "check_result" and e.get("question"):
                out.append(dict(ep=ep, arm=arm, result=e["result"], hits=e["hits"], through=e["through"],
                                occupied=e["occupied"], views=e["views"], dur=round(e["t"] - e.get("t_start", e["t"]), 1)))
            if e["kind"] in ("check_unreachable",) and any(q["kind"] == "question" and abs(q["t"] - e["t"]) < 0.01 for q in log["qa"]):
                out.append(dict(ep=ep, arm=arm, result="unreachable"))
        if not any(e["kind"] == "check_result" and e.get("question") for e in log["events"]) and any(q["kind"] == "question" for q in log["qa"]):
            out.append(dict(ep=ep, arm=arm, result="no check result", qa=[q.get("result") for q in log["qa"] if q["kind"] == "answer"]))
    return out


@app.local_entrypoint()
def evidence(tag: str = "final"):
    for r in check_evidence.remote(tag):
        print(r)


@app.function(image=render_image, volumes={"/data": vol}, cpu=4, memory=16384, timeout=3600)
def demo_remote(tag: str, episode: str) -> str:
    import sys
    sys.path.insert(0, "/scripts")
    import demo_cut
    d = f"/data/scout/env/{tag}/{episode}"
    out = f"/data/scout/env/{tag}/demo_{episode}.mp4"
    sys.argv = ["demo_cut.py", "--episode", d, "--leaderboard", f"/data/scout/env/{tag}/leaderboard.md", "--out", out]
    demo_cut.main()
    vol.commit()
    return out


@app.local_entrypoint()
def demo(tag: str = "final2", episode: str = ""):
    print(demo_remote.remote(tag, episode))


@app.local_entrypoint()
def resume(tag: str = "final3", explore_s: float = 240.0):
    """Re-run only the planned episodes of a frozen plan that have no manifest yet (crash or client loss), with the
    same spec and seed. Run with `modal run --detach` so losing the local client does not stop the app."""
    import json
    import subprocess
    plan = json.load(open(os.path.join(CFG, f"frozen_{tag}.json")))
    r = subprocess.run(["modal", "volume", "ls", "--json", VOLUME, f"/scout/env/{tag}"], capture_output=True, text=True)
    have = set()
    for x in json.loads(r.stdout or "[]"):
        ep = os.path.basename(x["filename"].rstrip("/"))
        r2 = subprocess.run(["modal", "volume", "ls", "--json", VOLUME, f"/scout/env/{tag}/{ep}"], capture_output=True, text=True)
        if any(os.path.basename(y["filename"]) == "manifest.json" for y in json.loads(r2.stdout or "[]")):
            have.add(ep)
    todo = [(b, h) for b in plan["brains"] for h in plan["houses"] if f"{b}_{h}_s{plan.get('seed', 0)}" not in have]
    print("complete:", len(have), "to run:", todo, flush=True)
    calls = [(b, h, run_episode.spawn(h, b, plan.get("seed", 0), tag, plan.get("explore_s", explore_s))) for b, h in todo]
    for b, h, c in calls:
        try:
            r = c.get()
            print("EP", b, h, json.dumps({"target": r["manifest"].get("target"), "answers": r["answers"]}))
        except Exception as e:
            print("EP", b, h, "CRASHED", type(e).__name__, str(e)[:400])
