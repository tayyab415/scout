"""Bridge test: can a frozen V-JEPA 2 video embedding tell Motivo what to do?

    modal run scripts/modal_bridge.py

1. Make ~3,300 different Motivo behaviours (skill vectors z): random z, z from 8-step windows of the motion
   buffer (as Motivo's tracking does), and mixes of named skills. Run each for 128 physics steps (4.3 s) from the
   same standing pose and film it from a random camera.
2. Encode each clip with frozen V-JEPA 2 (ViT-L), twice: the normal render and a white-on-black silhouette (the
   silhouette is what SAM 3.1 would give us from a real video).
3. Fit a linear (ridge) map from video features to z on 3,000 clips.
4. On 300 held-out clips: predict z from the video alone, run the body with it, and compare the body motion with
   the original. Baselines: average z, a random training z, raw-pixel ridge, nearest training clip. Ceiling: the
   true z rerun with small action noise.

Motion error is the mean distance of the 24 body parts between source and copy over the 4.3 s, in metres,
including where the body went. Results: /data/scout/bridge/results.json, showcase frames in showcase.npz.
"""
import modal

app = modal.App("scout-bridge")
vol = modal.Volume.from_name("tribe-lab-claude")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git", "ffmpeg", "libgl1", "libegl1", "libglib2.0-0")
         .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("metamotivo[huggingface,humenv] @ git+https://github.com/facebookresearch/metamotivo.git",
                      "mujoco==3.17.0", "transformers==4.57.1", "scikit-learn", "h5py")
         .env({"HF_HOME": "/data/hf", "MUJOCO_GL": "egl"}))

STEPS = 128          # 4.27 s at 30 Hz
EVERY = 4            # film every 4th step -> 32 frames at 7.5 fps
RES = 256
OUT = "/data/scout/bridge"
_G = {}


def _setup():
    if "env" in _G:
        return
    import mujoco
    import torch
    from humenv import make_humenv
    from metamotivo.fb_cpr.huggingface import FBcprModel
    torch.set_num_threads(1)
    env, _ = make_humenv(num_envs=1, state_init="Default")
    u = env.unwrapped
    M = u.model
    _G.update(env=env, u=u, M=M, model=FBcprModel.from_pretrained("facebook/metamotivo-M-1").eval(),
              pid=M.body("Pelvis").id, hid=M.body("Head").id, rd=mujoco.Renderer(M, RES, RES),
              robot_geom=(M.geom_bodyid >= 1) & (M.geom_bodyid <= 24))


def _rollout(job):
    """Forked worker. job = (z (256,), seed, camera or None, noise). Returns body-part tracks and optional frames."""
    import mujoco
    import numpy as np
    import torch
    _setup()
    z, seed, cam_p, noise = job
    env, u, M, model, rd = _G["env"], _G["u"], _G["M"], _G["model"], _G["rd"]
    D = u.data
    env.reset(seed=0)
    rng = np.random.default_rng(seed)
    zt = torch.tensor(z, dtype=torch.float32).reshape(1, -1)
    o = torch.tensor(u.get_obs()["proprio"].reshape(1, -1), dtype=torch.float32)
    p0 = D.xpos[_G["pid"]][:2].copy()
    track, rgb, sil, minh = [], [], [], 9.0
    cam = None
    if cam_p is not None:
        cam = mujoco.MjvCamera()
        cam.azimuth, cam.elevation, cam.distance = cam_p
    for t in range(STEPS):
        a = model.act(o, zt, mean=True).numpy().ravel()
        if noise:
            a = np.clip(a + noise * rng.standard_normal(a.shape), -1, 1)
        env.step(a)
        o = torch.tensor(u.get_obs()["proprio"].reshape(1, -1), dtype=torch.float32)
        minh = min(minh, float(D.xpos[_G["hid"]][2]))
        if t % EVERY == EVERY - 1:
            pos = D.xpos[1:25].copy()
            pos[:, :2] -= p0
            track.append(pos)
            if cam is not None:
                cam.lookat[:] = [D.xpos[_G["pid"]][0], D.xpos[_G["pid"]][1], 0.9]
                rd.update_scene(D, camera=cam)
                rgb.append(rd.render().copy())
                rd.enable_segmentation_rendering()
                rd.update_scene(D, camera=cam)
                seg = rd.render()
                rd.disable_segmentation_rendering()
                ids, types = seg[..., 0], seg[..., 1]
                m = (types == int(mujoco.mjtObj.mjOBJ_GEOM)) & (ids >= 0)
                m[m] = _G["robot_geom"][ids[m]]
                sil.append(m)
    out = dict(track=np.stack(track).astype(np.float32), fell=minh < 0.8)
    if cam is not None:
        out["rgb"] = np.stack(rgb)
        out["sil"] = np.packbits(np.stack(sil), axis=-1)
    return out


def _vjepa():
    import torch
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", torch_dtype=torch.float16).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1)

    @torch.no_grad()
    def encode(clips):  # (N, T, H, W, 3) uint8 -> (N, 5 * D): whole-clip mean + 4 time-bin means of patch tokens
        x = torch.from_numpy(clips).cuda().permute(0, 1, 4, 2, 3).float().div(255)
        x = ((x - mean) / std).half()
        h = m.get_vision_features(pixel_values_videos=x).float()  # (N, T/2 * 256, D)
        n, L, d = h.shape
        h = h.view(n, L // 256, 256, d).mean(2)  # (N, T/2, D)
        bins = h.view(n, 4, -1, d).mean(2)
        return torch.cat([h.mean(1), bins.reshape(n, -1)], 1).cpu().numpy()
    return encode


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=16, memory=98304, timeout=7200, retries=0,
              max_containers=1)
def run(n_train: int = 3000, n_test: int = 300, workers: int = 15):
    import json
    import multiprocessing as mp
    import os
    import time

    import h5py
    import numpy as np
    import torch
    from metamotivo.fb_cpr.huggingface import FBcprModel
    from sklearn.linear_model import RidgeCV
    t0 = time.time()
    log = lambda *a: print(f"{time.time() - t0:7.1f}s", *a, flush=True)  # noqa: E731
    rng = np.random.default_rng(2026)
    model = FBcprModel.from_pretrained("facebook/metamotivo-M-1").eval()
    N = n_train + n_test

    # ---- 1. behaviours ---------------------------------------------------------------------------
    with h5py.File("/data/stack/m1/data/buffer_inference_500000.hdf5", "r") as h:
        n_buf = h["qpos"].shape[0]
        starts = np.sort(rng.choice(n_buf - 16, N, replace=False))
        qp = h["qpos"][: 20001]
        nqp = h["next_qpos"][: 20000]
        contiguous = float(np.mean(np.abs(nqp - qp[1:]).max(1) < 1e-5))
        wins = np.stack([h["next_observation"][s:s + 8] for s in starts])
    seq_ok = contiguous > 0.9
    log("buffer rows contiguous", round(contiguous, 3))
    named = {k: v.cpu().numpy().ravel() for k, v in torch.load("/data/scout/z_motion.pt").items()}
    names = list(named)
    Z, src = [], []
    for i in range(N):
        r = rng.random()
        if r < 0.4:
            Z.append(model.sample_z(1).numpy().ravel()); src.append("random")
        elif r < 0.7:
            w = torch.tensor(wins[i] if seq_ok else wins[i][:1], dtype=torch.float32)
            Z.append(model.project_z(model.backward_map(w).mean(0, keepdim=True)).numpy().ravel())
            src.append("buffer_window" if seq_ok else "buffer_goal")
        else:
            a, b = rng.choice(len(names), 2, replace=False)
            w = rng.uniform(0.2, 0.8)
            Z.append(model.project_z(torch.tensor(w * named[names[a]] + (1 - w) * named[names[b]])[None]).numpy().ravel())
            src.append("named_mix")
    Z = np.stack(Z).astype(np.float32)
    cams = [(float(rng.uniform(0, 360)), float(rng.uniform(-25, -5)), float(rng.uniform(3.0, 4.5))) for _ in range(N)]

    # ---- 2. film, then encode with V-JEPA 2 ------------------------------------------------------------
    pool = mp.get_context("fork").Pool(workers)
    enc = _vjepa()
    feats = {"rgb": [], "sil": [], "pix": []}
    tracks, fell = [], []
    keep_rgb = {}
    B = 8
    for i0 in range(0, N, 120):
        res = pool.map(_rollout, [(Z[i], i, cams[i], 0.0) for i in range(i0, min(N, i0 + 120))])
        for j in range(0, len(res), B):
            chunk = res[j:j + B]
            rgb = np.stack([r["rgb"] for r in chunk])
            sil = np.stack([np.unpackbits(r["sil"], axis=-1)[..., :RES] for r in chunk]).astype(np.uint8) * 255
            feats["rgb"].append(enc(rgb))
            feats["sil"].append(enc(np.repeat(sil[..., None], 3, -1)))
            small = sil.reshape(len(chunk), -1, 16, RES // 16, 16, RES // 16).mean((3, 5))  # 16x16 per frame
            feats["pix"].append(small.reshape(len(chunk), -1).astype(np.float32) / 255)
        for k, r in enumerate(res):
            tracks.append(r["track"]); fell.append(r["fell"])
            if i0 + k >= n_train and len(keep_rgb) < 12:
                keep_rgb[i0 + k] = r["rgb"]
        log("filmed+encoded", min(N, i0 + 120), "/", N)
    F = {k: np.concatenate(v) for k, v in feats.items()}
    tracks, fell = np.stack(tracks), np.array(fell)
    tr, te = np.arange(n_train), np.arange(n_train, N)

    # ---- 3. linear maps video -> z ----------------------------------------------------------------------
    def fit(X):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        reg = RidgeCV(alphas=np.logspace(-1, 7, 17)).fit((X[tr] - mu) / sd, Z[tr])
        P = reg.predict((X[te] - mu) / sd)
        return P / np.linalg.norm(P, axis=1, keepdims=True) * 16.0, float(reg.alpha_)

    def nearest(X):
        Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
        return Z[tr][np.argmax(Xn[te] @ Xn[tr].T, 1)]

    preds, alphas = {}, {}
    for k in ("sil", "rgb", "pix"):
        preds[f"ridge_{k}"], alphas[k] = fit(F[k])
    preds["nearest_sil"] = nearest(F["sil"])
    zbar = Z[tr].mean(0)
    preds["average_z"] = np.repeat((zbar / np.linalg.norm(zbar) * 16)[None], len(te), 0)
    preds["random_train_z"] = Z[tr][rng.integers(n_train, size=len(te))]
    cos = {k: float(np.mean(np.sum(v * Z[te], 1) / 256.0)) for k, v in preds.items()}
    log("alphas", alphas, "cosine to true z", cos)

    # ---- 4. run the body with each predicted z and compare motion -------------------------------------------
    res = {}
    copy_tracks = {}
    for k, P in list(preds.items()) + [("true_z_noisy", Z[te])]:
        noise = 0.05 if k == "true_z_noisy" else 0.0
        out = pool.map(_rollout, [(P[j], 10_000 + j, None, noise) for j in range(len(te))])
        ct = np.stack([o["track"] for o in out])
        copy_tracks[k] = ct
        err = np.linalg.norm(ct - tracks[te], axis=-1).mean((1, 2))  # per clip, metres
        cf = np.array([o["fell"] for o in out])
        # identification: is the copy closer to its own source than to every other held-out source?
        d = np.linalg.norm(ct[:, None] - tracks[te][None], axis=-1).mean((2, 3))
        top1 = float(np.mean(np.argmin(d, 1) == np.arange(len(te))))
        rank = np.mean([int(np.sum(d[i] < d[i, i])) for i in range(len(te))])
        res[k] = dict(motion_err_m_mean=round(float(err.mean()), 3), motion_err_m_median=round(float(np.median(err)), 3),
                      ident_top1=round(top1, 3), ident_mean_rank=round(float(rank), 1),
                      fall_agreement=round(float(np.mean(cf == fell[te])), 3))
        log("copy", k, res[k])
    by_src = {}
    srcs = np.array(src)[te]
    for s in sorted(set(srcs)):
        m = srcs == s
        by_src[s] = {k: round(float(np.linalg.norm(copy_tracks[k][m] - tracks[te][m], axis=-1).mean()), 3)
                     for k in copy_tracks} | {"n": int(m.sum())}
    log("by source", by_src)

    # ---- showcase: re-film a few copies from the source camera -------------------------------------------------
    show = {}
    for i in list(keep_rgb)[:8]:
        j = i - n_train
        a = _rollout((preds["ridge_sil"][j], 0, cams[i], 0.0))
        b = _rollout((preds["average_z"][j], 0, cams[i], 0.0))
        show[f"{i}_source"] = keep_rgb[i]
        show[f"{i}_copy"] = a["rgb"]
        show[f"{i}_average"] = b["rgb"]
        show[f"{i}_err"] = np.array([np.linalg.norm(copy_tracks["ridge_sil"][j] - tracks[i], axis=-1).mean(),
                                     np.linalg.norm(copy_tracks["average_z"][j] - tracks[i], axis=-1).mean()])
    pool.terminate()
    os.makedirs(OUT, exist_ok=True)
    summary = dict(n_train=n_train, n_test=n_test, buffer_contiguous=contiguous, fell_rate=float(fell.mean()),
                   source_counts={s: int(np.sum(np.array(src) == s)) for s in set(src)}, ridge_alpha=alphas,
                   cosine_to_true_z=cos, copy=res, by_source=by_src, minutes=round((time.time() - t0) / 60, 1))
    json.dump(summary, open(f"{OUT}/results.json", "w"), indent=1)
    np.savez_compressed(f"{OUT}/showcase.npz", **show)
    np.savez_compressed(f"{OUT}/data.npz", Z=Z, tracks=tracks, fell=fell, src=np.array(src), cams=np.array(cams),
                        F_sil=F["sil"].astype(np.float16), F_rgb=F["rgb"].astype(np.float16))
    vol.commit()
    return json.loads(json.dumps(summary, default=float))


@app.local_entrypoint()
def main(n_train: int = 3000, n_test: int = 300):
    import json
    print(run.remote(n_train, n_test)["copy"])
