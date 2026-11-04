"""Why does the body move badly, and can a small adapter fix it? Three quick tests on frozen Motivo, empty world.

1. Turning: reward-inferred yaw skills on the correct (pelvis y) axis; does the body turn, does it fall?
2. Blending: switch skills instantly vs slide the skill vector over 0.5 s; falls and wobble.
3. Mixing: z = mix of forward and sidestep; does walking direction vary smoothly with the mix?

    modal run scripts/modal_motion_tests.py
Writes /data/scout/motion/results.json and short clips (npz frames) for the videos.
"""
import modal

app = modal.App("scout-motion")
vol = modal.Volume.from_name("tribe-lab-claude")
image = (modal.Image.debian_slim(python_version="3.11").apt_install("git", "libgl1", "libosmesa6", "libglib2.0-0")
         .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cpu")
         .pip_install("metamotivo[huggingface,humenv] @ git+https://github.com/facebookresearch/metamotivo.git", "h5py")
         .env({"HF_HOME": "/data/hf", "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}))

YAW = 1  # pelvis-frame axis that is vertical when standing (SMPL: x left, y up, z forward)
VARIANTS = {  # name: (target yaw rate rad/s, cap linear speed?)
    "yaw-L-0.5": (0.5, True), "yaw-R-0.5": (-0.5, True), "yaw-L-1": (1.0, True), "yaw-R-1": (-1.0, True),
    "yaw-L-1-free": (1.0, False), "yaw-R-1-free": (-1.0, False), "yaw-L-2-free": (2.0, False), "yaw-R-2-free": (-2.0, False),
}
_G = {}


def _reward(args):
    name, qpos, qvel, ctrl = args
    import mujoco
    import numpy as np
    from dm_control.utils import rewards
    from humenv import make_humenv
    if "m" not in _G:
        e, _ = make_humenv(num_envs=1, state_init="Default")
        m = e.unwrapped.model
        _G.update(m=m, d=mujoco.MjData(m), gyro=m.sensor("Pelvis_gyro").adr[0], head=m.body("Head").id)
    m, d = _G["m"], _G["d"]
    w, cap = VARIANTS[name]
    out = []
    for i in range(len(qpos)):
        d.qpos[:], d.qvel[:], d.ctrl[:] = qpos[i], qvel[i], ctrl[i]
        mujoco.mj_forward(m, d)
        up = rewards.tolerance(d.xpos[_G["head"]][2], bounds=(1.4, 3.0), margin=0.6, sigmoid="linear")
        yaw = d.sensordata[_G["gyro"] + YAW]
        turn = rewards.tolerance(np.sign(w) * yaw, bounds=(abs(w), abs(w) + 0.6), margin=abs(w), sigmoid="linear")
        r = up * turn
        if cap:
            r *= rewards.tolerance(float(np.linalg.norm(d.qvel[0:2])), bounds=(0, 0.3), margin=0.5, sigmoid="linear")
        out.append(float(r))
    return out


@app.function(image=image, volumes={"/data": vol}, cpu=16, memory=32768, timeout=3000, retries=0, max_containers=1)
def run(seeds: int = 6):
    import json
    import math
    import multiprocessing as mp
    import os
    import time

    import h5py
    import mujoco
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
    Z = {k: v.cpu() for k, v in torch.load("/data/scout/z.pt").items()}
    nxt = torch.tensor(B["next_observation"], dtype=torch.float32)
    res = {"buffer": {}}
    for name in VARIANTS:
        r = np.concatenate(pool.map(_reward, [(name, B["next_qpos"][p], B["next_qvel"][p], B["action"][p]) for p in parts]))
        Z[name] = model.reward_wr_inference(next_obs=nxt, reward=torch.tensor(r, dtype=torch.float32).reshape(-1, 1)).cpu()
        res["buffer"][name] = dict(mean=round(float(r.mean()), 4), frac_gt_0_5=round(float((r > 0.5).mean()), 4))
        log("inferred", name, res["buffer"][name])
    pool.terminate()

    env, _ = make_humenv(num_envs=1, state_init="Default")
    u = env.unwrapped
    M, D = u.model, u.data
    pid, hid = M.body("Pelvis").id, M.body("Head").id
    gyro = M.sensor("Pelvis_gyro").adr[0]
    obs = lambda: torch.tensor(u.get_obs()["proprio"].reshape(1, -1), dtype=torch.float32)  # noqa: E731
    rd = mujoco.Renderer(M, 240, 320)

    def frame(follow_yaw=None):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = D.xpos[pid] + np.array([0, 0, 0.0])
        cam.distance, cam.elevation = 3.2, -15
        cam.azimuth = 135 if follow_yaw is None else follow_yaw
        rd.update_scene(D, camera=cam)
        return rd.render().copy()

    # forward axis of the pelvis (as in scout.body.motivo)
    env.reset(seed=0)
    p0, R0 = D.xpos[pid][:2].copy(), D.xmat[pid].reshape(3, 3).copy()
    o = obs()
    for _ in range(60):
        env.step(model.act(o, Z["move-ego-0-1"], mean=True).numpy().ravel())
        o = obs()
    dv = D.xpos[pid][:2] - p0
    dv /= np.linalg.norm(dv)
    ax = max(((s, c) for c in range(3) for s in (1, -1)), key=lambda k: k[0] * R0[:2, k[1]] @ dv)
    yaw_of = lambda: math.atan2(*(ax[0] * D.xmat[pid].reshape(3, 3)[:2, ax[1]])[::-1])  # noqa: E731

    def head_tilt_deg():
        up_local = D.xmat[hid].reshape(3, 3)[:, 1]  # head's y axis = up in SMPL frames
        return math.degrees(math.acos(float(np.clip(up_local[2], -1, 1))))

    # ---- 1. turning ------------------------------------------------------------------------------
    res["turning"] = {}
    clips = {}
    for name in list(VARIANTS) + ["move-ego-90-1", "move-ego-0-0"]:
        runs = []
        for s in range(seeds):
            env.reset(seed=s)
            rng = np.random.default_rng(s)
            o = obs()
            y0, p0 = yaw_of(), D.xpos[pid][:2].copy()
            yaws, minh, fr = [y0], 9.0, []
            for i in range(180):  # 6 s
                env.step(np.clip(model.act(o, Z[name], mean=True).numpy().ravel() + 0.02 * rng.standard_normal(69), -1, 1))
                o = obs()
                yaws.append(yaw_of())
                minh = min(minh, float(D.xpos[hid][2]))
                if s == 0 and i % 2 == 0:
                    fr.append(frame())
            turned = math.degrees(float(np.sum(np.diff(np.unwrap(yaws)))))
            runs.append(dict(turned_deg=round(turned, 1), fell=minh < 0.8, moved_m=round(float(np.linalg.norm(D.xpos[pid][:2] - p0)), 2)))
            if s == 0:
                clips[f"turn_{name}"] = np.stack(fr)
        res["turning"][name] = dict(turned_deg_mean=round(float(np.mean([r["turned_deg"] for r in runs])), 1),
                                    turned_deg_min=round(float(np.min(np.abs([r["turned_deg"] for r in runs]))), 1),
                                    falls=sum(r["fell"] for r in runs),
                                    moved_m_mean=round(float(np.mean([r["moved_m"] for r in runs])), 2))
        log("turn", name, res["turning"][name])

    # ---- 2. switching vs blending ------------------------------------------------------------------
    skills = ["move-ego-0-0", "move-ego-0-1", "move-ego-90-1", "move-ego--90-1", "move-ego-180-1", "walkturn-right"]
    norm = float(np.mean([float(Z[k].norm()) for k in skills]))
    res["switching"] = {}
    for mode in ("instant", "blend_0.5s"):
        runs = []
        for s in range(20):
            env.reset(seed=100 + s)
            rng = np.random.default_rng(100 + s)
            seq = rng.integers(len(skills), size=40)
            o = obs()
            z = Z[skills[1]].clone()
            minh, tilts, ang, fr = 9.0, [], [], []
            prev_R = D.xmat[hid].reshape(3, 3).copy()
            for k in range(30 * 15):  # 15 s, new skill every 0.5 s
                target = Z[skills[int(seq[k // 15])]]
                if mode == "instant":
                    z = target
                else:
                    z = z + (target - z) / 15.0 * 2.0  # reaches the target in ~0.5 s
                    z = z / z.norm() * norm
                env.step(np.clip(model.act(o, z, mean=True).numpy().ravel(), -1, 1))
                o = obs()
                minh = min(minh, float(D.xpos[hid][2]))
                tilts.append(head_tilt_deg())
                R = D.xmat[hid].reshape(3, 3)
                ang.append(math.degrees(math.acos(float(np.clip((np.trace(prev_R.T @ R) - 1) / 2, -1, 1)))) * 30)
                prev_R = R.copy()
                if s == 0 and k % 2 == 0:
                    fr.append(frame())
            runs.append(dict(fell=minh < 0.8, tilt_mean=float(np.mean(tilts)), tilt_p95=float(np.percentile(tilts, 95)),
                             head_ang_speed=float(np.mean(ang))))
            if s == 0:
                clips[f"switch_{mode}"] = np.stack(fr)
        res["switching"][mode] = dict(falls=sum(r["fell"] for r in runs), n=len(runs),
                                      head_tilt_mean_deg=round(float(np.mean([r["tilt_mean"] for r in runs])), 1),
                                      head_tilt_p95_deg=round(float(np.mean([r["tilt_p95"] for r in runs])), 1),
                                      head_turn_speed_deg_s=round(float(np.mean([r["head_ang_speed"] for r in runs])), 1))
        log("switching", mode, res["switching"][mode])

    # ---- 3. mixing forward with sidesteps ----------------------------------------------------------
    res["mixing"] = {}
    for side in ("move-ego-90-1", "move-ego--90-1"):
        for w in (0.0, 0.25, 0.5, 0.75, 1.0):
            z = (1 - w) * Z["move-ego-0-1"] + w * Z[side]
            z = z / z.norm() * norm
            dirs, falls = [], 0
            for s in range(4):
                env.reset(seed=200 + s)
                o = obs()
                for _ in range(20):
                    env.step(model.act(o, Z["move-ego-0-0"], mean=True).numpy().ravel())
                    o = obs()
                y0, p0, minh = yaw_of(), D.xpos[pid][:2].copy(), 9.0
                for _ in range(90):
                    env.step(model.act(o, z, mean=True).numpy().ravel())
                    o = obs()
                    minh = min(minh, float(D.xpos[hid][2]))
                d = D.xpos[pid][:2] - p0
                dirs.append(math.degrees((math.atan2(d[1], d[0]) - y0 + math.pi) % (2 * math.pi) - math.pi))
                falls += minh < 0.8
            res["mixing"][f"{side}@{w}"] = dict(direction_deg_mean=round(float(np.mean(dirs)), 1),
                                               direction_deg_sd=round(float(np.std(dirs)), 1), falls=falls)
            log("mix", side, w, res["mixing"][f"{side}@{w}"])
    os.makedirs("/data/scout/motion", exist_ok=True)
    json.dump(res, open("/data/scout/motion/results.json", "w"), indent=1)
    np.savez_compressed("/data/scout/motion/clips.npz", **clips)
    torch.save(Z, "/data/scout/z_motion.pt")
    vol.commit()
    return res


@app.local_entrypoint()
def main():
    import json
    r = run.remote()
    print(json.dumps({k: v for k, v in r.items() if k != "buffer"}, indent=1))
