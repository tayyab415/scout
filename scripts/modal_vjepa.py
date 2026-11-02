"""V-JEPA 2 world model for the Motivo body, following the V-JEPA 2-AC recipe (frozen encoder, new predictor).

    modal run scripts/modal_vjepa.py::collect      # head-camera rollouts with random skills, encoded by V-JEPA 2
    modal run scripts/modal_vjepa.py::train        # action-conditioned predictor on frozen features
    modal run scripts/modal_vjepa.py::evaluate     # prediction, action identification, surprise vs events

Data and results live on the Modal volume under /data/scout/vj/.
"""
import json
import os

import modal

app = modal.App("scout-vjepa")
vol = modal.Volume.from_name("tribe-lab-claude")
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
CFG = os.path.join(os.path.dirname(__file__), "..", "configs")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git", "ffmpeg", "libgl1", "libegl1", "libglib2.0-0")
         .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("metamotivo[huggingface,humenv] @ git+https://github.com/facebookresearch/metamotivo.git",
                      "mujoco==3.17.0", "imageio[ffmpeg]", "transformers==4.57.1", "scikit-learn")
         .env({"HF_HOME": "/data/hf", "MUJOCO_GL": "egl"})
         .add_local_dir(SRC, "/pkg")
         .add_local_dir(CFG, "/cfg"))

ACTIONS = ["move-ego-0-0", "move-ego-0-1", "move-ego-90-1", "move-ego--90-1", "move-ego-180-1", "walkturn-right"]
ACTION_NAMES = ["stand", "forward", "side-left", "side-right", "back", "walk-turn-right"]
SIM_PER_CTRL = 8          # 8 physics steps of 1/30 s = 0.267 s per control step (~3.75 fps, the paper uses 4 fps)
OUT = "/data/scout/vj"


def _rollouts(job):
    """Forked worker: one house, several random-action episodes. Writes raw frames to /tmp, returns paths."""
    house, n_eps, seed, steps, video_eps = job
    import sys
    import time
    sys.path.insert(0, "/pkg")
    import numpy as np
    import torch
    torch.set_num_threads(1)
    from scout.body.motivo import MotivoBody
    from scout.world.layout import Layout
    t0 = time.time()
    try:
        body = MotivoBody("/data/house", house, "/data/scout/z.pt", device="cpu")
    except Exception as e:  # keep the run going; report it
        return house, [], f"build failed: {type(e).__name__}: {str(e)[:200]}"
    lay = Layout.load(f"/data/house/layouts/{house}.npz")
    M = body.M
    robot = {g for g in range(M.ngeom) if 1 <= M.geom_bodyid[g] <= 24}
    import mujoco
    house_geoms = set()
    for g in range(M.ngeom):
        b = M.geom_bodyid[g]
        name = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if b > 24 and "room_" not in name:
            house_geoms.add(g)
    rng = np.random.default_rng(seed)
    cells = np.argwhere(lay.grid.free)
    paths = []
    for ep in range(n_eps):
        c = cells[rng.integers(len(cells))]
        xy0, yaw0 = lay.grid.pos(c), float(rng.uniform(-np.pi, np.pi))
        body.place(xy0, yaw0)
        for _ in range(15):
            body.step("move-ego-0-0")
        frames, follow = [body.head_view(256)], []
        acts, xy, yaw, headz, contact, room = [], [body.xy], [body.yaw], [body.head_z], [False], []
        r0 = lay.room_at(body.xy)
        room.append(r0.id if r0 else "")
        keep_video = ep < video_eps
        if keep_video:
            follow.append(body.follow_view(np.degrees(body.yaw) + 180, 360, 480))
        a, hold, fell = 1, 0, False
        for k in range(steps):
            if hold <= 0:
                a = int(rng.integers(len(ACTIONS)))
                hold = int(rng.integers(2, 5))
            hold -= 1
            touched = False
            try:
                for _ in range(SIM_PER_CTRL):
                    body.step(ACTIONS[a])
                    d = body.D
                    for i in range(d.ncon):
                        c = d.contact[i]
                        g1, g2 = c.geom1, c.geom2
                        # above the floor only: feet on rugs and door thresholds are not bumps
                        if c.pos[2] > 0.2 and ((g1 in robot and g2 in house_geoms) or (g2 in robot and g1 in house_geoms)):
                            touched = True
                            break
            except ValueError:
                fell = True
                break
            acts.append(a)
            frames.append(body.head_view(256))
            xy.append(body.xy)
            yaw.append(body.yaw)
            headz.append(body.head_z)
            contact.append(touched)
            r = lay.room_at(body.xy)
            room.append(r.id if r else "")
            if keep_video:
                follow.append(body.follow_view(np.degrees(body.yaw) + 180, 360, 480))
            if body.head_z < 0.8:
                fell = True
                break
        p = f"/tmp/vjraw/{house}_{ep:03d}.npz"
        os.makedirs("/tmp/vjraw", exist_ok=True)
        np.savez(p, frames=np.stack(frames), actions=np.array(acts, np.int8), xy=np.array(xy, np.float32),
                 yaw=np.array(yaw, np.float32), headz=np.array(headz, np.float32), contact=np.array(contact),
                 room=np.array(room), fell=fell, follow=np.stack(follow) if follow else np.zeros((0,), np.uint8))
        paths.append(p)
    return house, paths, f"ok {time.time() - t0:.0f}s"


def _encoder():
    import torch
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", torch_dtype=torch.float16).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1)

    @torch.no_grad()
    def encode(frames):  # (N, 256, 256, 3) uint8 -> (N, 16, D) float16, 4x4 average-pooled patch tokens
        out = []
        for i in range(0, len(frames), 64):
            x = torch.from_numpy(frames[i:i + 64]).cuda().permute(0, 3, 1, 2).float().div(255)
            x = x[:, None].expand(-1, 2, -1, -1, -1)  # each image as a 2-frame clip, as in V-JEPA 2-AC
            x = ((x - mean) / std).half()
            h = m.get_vision_features(pixel_values_videos=x)  # (n, 256, D) for 16x16 patches
            n, t, d = h.shape
            g = int(round(t ** 0.5))
            h = h.view(n, g, g, d).permute(0, 3, 1, 2)
            h = torch.nn.functional.avg_pool2d(h.float(), g // 4).permute(0, 2, 3, 1).reshape(n, 16, d)
            out.append(h.half().cpu())
        return torch.cat(out).numpy()
    return encode


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=16, memory=98304, timeout=5400, retries=0,
              max_containers=1, scaledown_window=2)
def collect_run(train_houses: list, test_houses: list, eps_train: int = 30, eps_test: int = 10, steps: int = 72,
                workers: int = 14):
    import multiprocessing as mp
    import time

    import numpy as np
    t0 = time.time()
    log = lambda *a: print(round(time.time() - t0, 1), *a, flush=True)  # noqa: E731
    jobs = [(h, eps_test, 1000 + i, steps, 2 if i < 2 else 0) for i, h in enumerate(test_houses)] + \
           [(h, eps_train, i, steps, 1 if i < 2 else 0) for i, h in enumerate(train_houses)]
    split = {h: "test" for h in test_houses} | {h: "train" for h in train_houses}
    pool = mp.get_context("fork").Pool(workers)
    it = pool.imap_unordered(_rollouts, jobs)
    encode = _encoder()
    n_frames = 0
    summary = []
    for house, paths, status in it:
        log(house, split[house], status, len(paths), "episodes")
        for p in paths:
            z = np.load(p)
            feats = encode(z["frames"])
            name = os.path.basename(p)[:-4]
            d = f"{OUT}/{split[house]}"
            os.makedirs(d, exist_ok=True)
            np.savez(f"{d}/{name}.npz", feats=feats, actions=z["actions"], xy=z["xy"], yaw=z["yaw"], headz=z["headz"],
                     contact=z["contact"], room=z["room"], fell=z["fell"])
            if len(z["follow"]):
                os.makedirs(f"{OUT}/video", exist_ok=True)
                np.savez_compressed(f"{OUT}/video/{name}.npz", frames=z["frames"], follow=z["follow"],
                                    actions=z["actions"], contact=z["contact"], room=z["room"])
            n_frames += len(feats)
            summary.append(dict(ep=name, split=split[house], frames=len(feats), fell=bool(z["fell"]),
                                contacts=int(z["contact"].sum()), rooms=len(set(z["room"].tolist()) - {""})))
            os.remove(p)
        log("frames so far", n_frames)
    pool.close()
    json.dump(dict(summary=summary, actions=ACTION_NAMES, sim_per_ctrl=SIM_PER_CTRL), open(f"{OUT}/collect.json", "w"))
    vol.commit()
    return dict(frames=n_frames, episodes=len(summary), falls=sum(s["fell"] for s in summary),
                minutes=round((time.time() - t0) / 60, 1))


@app.local_entrypoint()
def collect(n_train: int = 48, n_test: int = 12, eps_train: int = 30, eps_test: int = 10, steps: int = 72,
            smoke: bool = False):
    import random
    splits = json.load(open(os.path.join(CFG, "splits.json")))
    tr, te = list(splits["train"]), list(splits["test"])
    random.Random(7).shuffle(tr)
    random.Random(8).shuffle(te)
    if smoke:
        n_train, n_test, eps_train, eps_test, steps = 2, 1, 2, 2, 24
    print(collect_run.remote(tr[:n_train], te[:n_test], eps_train, eps_test, steps))


# --------------------------------------------------------------------------------------------------
# Predictor: block-causal transformer over (frame tokens, action token) steps, as in V-JEPA 2-AC.
CTX = 8      # frames of context
DIM = 384


def _model(n_actions, feat_dim=1024, layers=6, heads=6):
    import torch
    import torch.nn as nn

    class Predictor(nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = nn.Linear(feat_dim, DIM)
            self.act = nn.Embedding(n_actions + 1, DIM)  # last index = "no action given"
            self.tok = nn.Parameter(torch.zeros(17, DIM))
            self.time = nn.Parameter(torch.zeros(CTX + 4, DIM))
            layer = nn.TransformerEncoderLayer(DIM, heads, 4 * DIM, dropout=0.0, batch_first=True, norm_first=True)
            self.tf = nn.TransformerEncoder(layer, layers)
            self.out = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, feat_dim))
            nn.init.normal_(self.tok, std=0.02)
            nn.init.normal_(self.time, std=0.02)

        def forward(self, z, a):
            """z: (B, T, 16, F) normalised features; a: (B, T) actions taken after each frame.
            Returns predictions for frames 1..T: (B, T, 16, F)."""
            B, T, N, _ = z.shape
            x = torch.cat([self.inp(z), self.act(a)[:, :, None]], 2)  # (B, T, 17, D)
            x = x + self.tok[None, None] + self.time[None, :T, None]
            x = x.reshape(B, T * (N + 1), DIM)
            step = torch.arange(T, device=z.device).repeat_interleave(N + 1)
            mask = step[None, :] > step[:, None]  # True = blocked: no attending to later steps
            h = self.tf(x, mask=mask).reshape(B, T, N + 1, DIM)[:, :, :N]
            return self.out(h)
    return Predictor()


def _load_split(split):
    import glob

    import numpy as np
    eps = []
    for p in sorted(glob.glob(f"{OUT}/{split}/*.npz")):
        z = np.load(p)
        if len(z["actions"]) < 4:
            continue
        eps.append(dict(name=os.path.basename(p)[:-4], feats=z["feats"], actions=z["actions"].astype(np.int64),
                        contact=z["contact"], room=z["room"], headz=z["headz"], xy=z["xy"]))
    return eps


def _norm(x):
    import torch
    return torch.nn.functional.layer_norm(x.float(), x.shape[-1:])


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=65536, timeout=3600, retries=0,
              max_containers=2, scaledown_window=2)
def train_run(blind: bool = False, steps: int = 12000, batch: int = 256, seed: int = 0):
    import time

    import numpy as np
    import torch
    torch.manual_seed(seed)
    t0 = time.time()
    log = lambda *a: print(round(time.time() - t0, 1), *a, flush=True)  # noqa: E731
    eps = _load_split("train")
    F = torch.cat([torch.from_numpy(e["feats"]) for e in eps]).cuda()  # (Nf, 16, D) fp16
    A = torch.cat([torch.from_numpy(np.append(e["actions"], 0)) for e in eps]).cuda()
    starts, off = [], 0
    for e in eps:
        n = len(e["feats"])
        starts += [off + s for s in range(0, n - (CTX + 1) + 1)]
        off += n
    starts = torch.tensor(starts, device="cuda")
    n_act = len(ACTIONS)
    log("train episodes", len(eps), "frames", len(F), "windows", len(starts))
    net = _model(n_act).cuda()
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    ar = torch.arange(CTX + 1, device="cuda")
    for it in range(steps):
        idx = starts[torch.randint(len(starts), (batch,), device="cuda")][:, None] + ar  # (B, CTX+1)
        z = _norm(F[idx])
        a = A[idx][:, :CTX]
        if blind:
            a = torch.full_like(a, n_act)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = net(z[:, :CTX], a)                                 # predicts frames 1..CTX
            tf = (p.float() - z[:, 1:]).abs().mean()
            z2 = torch.cat([z[:, :1], p[:, :-1].float()], 1)       # one-step rollout: feed own predictions back
            p2 = net(z2, a)
            ro = (p2[:, 1:].float() - z[:, 2:]).abs().mean()
            loss = tf + ro
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 1000 == 0 or it == steps - 1:
            log(it, "teacher-forcing", round(tf.item(), 4), "rollout", round(ro.item(), 4))
    name = "blind" if blind else "ac"
    os.makedirs(f"{OUT}/models", exist_ok=True)
    torch.save(net.state_dict(), f"{OUT}/models/{name}.pt")
    vol.commit()
    return dict(model=name, params=sum(p.numel() for p in net.parameters()), final_tf=round(tf.item(), 4),
                final_rollout=round(ro.item(), 4), minutes=round((time.time() - t0) / 60, 1))


@app.local_entrypoint()
def train(steps: int = 12000):
    for r in train_run.starmap([(False, steps), (True, steps)]):
        print(r)


@app.function(image=image, volumes={"/data": vol}, gpu="T4", cpu=4, memory=16384, timeout=900, retries=0)
def debug_contacts(house: str, n: int = 4):
    import collections
    import sys
    sys.path.insert(0, "/pkg")
    import mujoco
    import numpy as np
    import torch
    torch.set_num_threads(4)
    from scout.body.motivo import MotivoBody
    from scout.world.layout import Layout
    body = MotivoBody("/data/house", house, "/data/scout/z.pt", device="cpu")
    lay = Layout.load(f"/data/house/layouts/{house}.npz")
    M = body.M
    rng = np.random.default_rng(0)
    cells = np.argwhere(lay.grid.free)
    seen = collections.Counter()
    for _ in range(n):
        body.place(lay.grid.pos(cells[rng.integers(len(cells))]), float(rng.uniform(-3, 3)))
        for k in range(60):
            body.step("move-ego-0-0" if k < 15 else "move-ego-0-1")
            d = body.D
            for i in range(d.ncon):
                c = d.contact[i]
                for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                    if 1 <= M.geom_bodyid[g] <= 24 and M.geom_bodyid[o] > 24 and c.pos[2] > 0.2:
                        bn = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, M.geom_bodyid[o])
                        gn = mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_GEOM, o)
                        seen[(bn, gn, int(M.geom_type[o]), round(float(c.pos[2]), 1),
                              mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_GEOM, g))] += 1
    return [list(k) + [v] for k, v in seen.most_common(12)]


@app.local_entrypoint()
def dbg(house: str = "procthor-10k-val_val_695"):
    for r in debug_contacts.remote(house):
        print(r)
