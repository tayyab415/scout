"""Learn extra Motivo skills by reward inference and test whether the body can do them without falling.

    modal run scripts/modal_skills.py

New skills: slow turning in place, and walking while turning. Results go to /data/scout/skills.json and the
merged skill library to /data/scout/z.pt (the original /data/stack/z.pt is left untouched).
"""
import modal

app = modal.App("scout-skills")
vol = modal.Volume.from_name("tribe-lab-claude")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git", "libgl1", "libegl1", "libglib2.0-0")
         .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("metamotivo[huggingface,humenv] @ git+https://github.com/facebookresearch/metamotivo.git", "h5py")
         .env({"HF_HOME": "/data/hf"}))

# name -> (base locomotion reward or None, target yaw rate rad/s)
NEW = {
    "turn-left-0.5": (None, 0.5), "turn-right-0.5": (None, -0.5),
    "turn-left-1": (None, 1.0), "turn-right-1": (None, -1.0),
    "walkturn-left": ("move-ego-0-0.7", 0.6), "walkturn-right": ("move-ego-0-0.7", -0.6),
}
_G = {}
YAW_AXIS = 1  # pelvis-frame axis that points up when standing (SMPL: x left, y up, z forward)


def _reward(args):
    """Rewards for one chunk of buffer transitions, computed in a forked worker."""
    name, qpos, qvel, ctrl = args
    import mujoco
    import numpy as np
    from dm_control.utils import rewards
    from humenv import make_humenv
    from humenv.env import make_from_name
    if "m" not in _G:
        e, _ = make_humenv(num_envs=1, state_init="Default")
        _G["m"] = e.unwrapped.model
        _G["d"] = mujoco.MjData(_G["m"])
        _G["gyro"] = _G["m"].sensor("Pelvis_gyro").adr[0]
    m, d = _G["m"], _G["d"]
    base, w = NEW[name]
    stand = make_from_name("move-ego-0-0") if base is None else make_from_name(base)
    out = []
    for i in range(len(qpos)):
        d.qpos[:] = qpos[i]
        d.qvel[:] = qvel[i]
        d.ctrl[:] = ctrl[i]
        mujoco.mj_forward(m, d)
        r_base = stand.compute(m, d)
        yaw_rate = d.sensordata[_G["gyro"] + YAW_AXIS]  # pelvis gyro; SMPL body frames have +y up
        r_turn = rewards.tolerance(np.sign(w) * yaw_rate, bounds=(abs(w), abs(w) + 0.4), margin=abs(w) / 2,
                                   value_at_margin=0, sigmoid="linear")
        out.append(float(r_base * r_turn))
    return out


@app.function(image=image, volumes={"/data": vol}, cpu=16, memory=32768, timeout=1800, retries=0, max_containers=1)
def learn(seeds: int = 5, seconds: float = 6.0):
    import json
    import math
    import multiprocessing as mp
    import os
    import time

    import h5py
    import numpy as np
    t0 = time.time()
    log = lambda *a: print(round(time.time() - t0, 1), *a, flush=True)  # noqa: E731
    with h5py.File("/data/stack/m1/data/buffer_inference_500000.hdf5", "r") as h:
        idx = np.sort(np.random.default_rng(0).choice(h["qpos"].shape[0], 100000, replace=False))
        B = {k: h[k][idx] for k in ("next_qpos", "next_qvel", "action", "next_observation")}
    pool = mp.get_context("fork").Pool(16)
    parts = np.array_split(np.arange(len(idx)), 64)
    import torch
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    torch.set_num_threads(16)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1")
    nxt = torch.tensor(B["next_observation"], dtype=torch.float32)
    Z = torch.load("/data/stack/z.pt")
    Z = {k: v.cpu() for k, v in Z.items()}
    stats = {}
    for name in NEW:
        r = np.concatenate(pool.map(_reward, [(name, B["next_qpos"][p], B["next_qvel"][p], B["action"][p]) for p in parts]))
        Z[name] = model.reward_wr_inference(next_obs=nxt, reward=torch.tensor(r, dtype=torch.float32).reshape(-1, 1)).cpu()
        stats[name] = {"buffer_reward_mean": round(float(r.mean()), 4), "buffer_reward_frac_pos": round(float((r > 0.05).mean()), 4)}
        log("inferred", name, stats[name])
    pool.terminate()

    # Test: can the body do each skill (new and a few old ones) without falling?
    env, _ = make_humenv(num_envs=1, state_init="Default")
    u = env.unwrapped
    pid = u.model.body("Pelvis").id
    hid = u.model.body("Head").id
    gyro = u.model.sensor("Pelvis_gyro").adr[0]
    tests = list(NEW) + ["rotate-z-5-0.8", "move-ego-0-1", "move-ego-90-1", "move-ego-0-0"]
    for name in tests:
        if name not in Z:
            continue
        runs = []
        for s in range(seeds):
            env.reset(seed=s)
            rng = np.random.default_rng(s)
            o = torch.tensor(u.get_obs()["proprio"].reshape(1, -1), dtype=torch.float32)
            p0 = u.data.xpos[pid][:2].copy()
            R0 = u.data.xmat[pid].reshape(3, 3)
            yaw_rates, min_head = [], 9.0
            heading = []
            for i in range(int(seconds * 30)):
                a = model.act(o, Z[name], mean=True).numpy().ravel()
                env.step(np.clip(a + 0.02 * rng.standard_normal(a.shape), -1, 1))
                o = torch.tensor(u.get_obs()["proprio"].reshape(1, -1), dtype=torch.float32)
                yaw_rates.append(u.data.sensordata[gyro:gyro + 3].copy())
                min_head = min(min_head, float(u.data.xpos[hid][2]))
                R = u.data.xmat[pid].reshape(3, 3)
                heading.append(math.atan2(R[1, 2], R[0, 2]))  # any horizontal body axis works for total turn
            turn = float(np.sum(np.diff(np.unwrap(heading))))
            runs.append(dict(fell=min_head < 0.8, min_head=round(min_head, 2), turned_deg=round(math.degrees(turn), 1),
                             mean_gyro_xyz=[round(float(v), 2) for v in np.mean(yaw_rates[30:], axis=0)],
                             moved_m=round(float(np.linalg.norm(u.data.xpos[pid][:2] - p0)), 2)))
        stats.setdefault(name, {})["test"] = dict(
            falls=sum(r["fell"] for r in runs), seeds=seeds,
            turned_deg_mean=round(float(np.mean([r["turned_deg"] for r in runs])), 1),
            moved_m_mean=round(float(np.mean([r["moved_m"] for r in runs])), 2),
            gyro_xyz_mean=[round(float(v), 2) for v in np.mean([r["mean_gyro_xyz"] for r in runs], axis=0)], runs=runs)
        log("test", name, {k: v for k, v in stats[name]["test"].items() if k != "runs"})
    os.makedirs("/data/scout", exist_ok=True)
    torch.save(Z, "/data/scout/z.pt")
    json.dump(stats, open("/data/scout/skills.json", "w"), indent=1)
    vol.commit()
    return {k: {kk: vv for kk, vv in v.get("test", {}).items() if kk != "runs"} for k, v in stats.items()}


@app.local_entrypoint()
def main():
    import json
    print(json.dumps(learn.remote(), indent=1))
