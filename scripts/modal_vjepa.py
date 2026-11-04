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


def _auc(pos, neg):
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return None
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1.0
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=65536, timeout=3600, retries=0,
              max_containers=1, scaledown_window=2)
def evaluate_run():
    import time

    import numpy as np
    import torch
    t0 = time.time()
    log = lambda *a: print(round(time.time() - t0, 1), *a, flush=True)  # noqa: E731
    n_act = len(ACTIONS)
    nets = {}
    for name in ("ac", "blind"):
        net = _model(n_act).cuda().eval()
        net.load_state_dict(torch.load(f"{OUT}/models/{name}.pt"))
        nets[name] = net
    eps = _load_split("test")
    log("test episodes", len(eps), "frames", sum(len(e["feats"]) for e in eps))

    @torch.no_grad()
    def predict(net, z, a):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return net(z, a).float()

    res = {"actions": ACTION_NAMES}
    # 1) prediction error at horizons 1..4 (autoregressive, true actions), vs copy-last
    H = 4
    errs = {k: {h: [] for h in range(1, H + 1)} for k in ("ac", "blind", "copy_last")}
    # 2) action identification at horizon 1 and 2
    conf = np.zeros((n_act, n_act), int)
    conf2 = np.zeros((n_act, n_act), int)
    spread, err_true = [], []
    # 3) per-step surprise for event analysis
    rows = []
    for e in eps:
        z = _norm(torch.from_numpy(e["feats"]).cuda())  # (T+1, 16, D)
        acts = torch.from_numpy(e["actions"]).cuda()
        T = len(acts)
        for k in range(1, T):  # predict frame k+1 from frames up to k
            lo = max(0, k - CTX + 1)
            ctx, act = z[lo:k + 1][None], acts[lo:k + 1][None]
            p_ac = predict(nets["ac"], ctx, act)[0, -1]
            p_bl = predict(nets["blind"], ctx, torch.full_like(act, n_act))[0, -1]
            tgt = z[k + 1]
            e_ac = (p_ac - tgt).abs().mean().item()
            e_bl = (p_bl - tgt).abs().mean().item()
            e_cp = (z[k] - tgt).abs().mean().item()
            # action identification: which action's prediction is closest to what happened?
            ctx_r = ctx.expand(n_act, -1, -1, -1)
            act_r = act.repeat(n_act, 1)
            act_r[:, -1] = torch.arange(n_act, device="cuda")
            P = predict(nets["ac"], ctx_r, act_r)[:, -1]
            d = (P - tgt[None]).abs().mean((1, 2))
            conf[acts[k].item(), int(d.argmin())] += 1
            spread.append(float(d.std() / d.mean()))
            err_true.append(e_ac)
            if k + 2 <= T and acts[k + 1].item() == acts[k].item():  # same action held for two steps
                ctx2 = torch.cat([ctx_r, P[:, None]], 1)[:, -CTX:]
                act2 = torch.cat([act_r, act_r[:, -1:]], 1)[:, -CTX:]
                P2 = predict(nets["ac"], ctx2, act2)[:, -1]
                d2 = (P2 - z[k + 2][None]).abs().mean((1, 2))
                conf2[acts[k].item(), int(d2.argmin())] += 1
            prev_contact, contact = bool(e["contact"][k]), bool(e["contact"][k + 1])
            room_change = e["room"][k] != e["room"][k + 1] and e["room"][k] != "" and e["room"][k + 1] != ""
            drop = float(e["headz"][k] - e["headz"][k + 1])
            moved = float(np.linalg.norm(e["xy"][k + 1] - e["xy"][k]))
            rows.append(dict(ep=e["name"], k=k, action=int(acts[k]), e_ac=e_ac, e_blind=e_bl, e_copy=e_cp,
                             bump_onset=contact and not prev_contact, touching=contact, room_change=bool(room_change),
                             head_drop=drop, moved=moved))
        # horizons
        for k in range(1, T - H + 1):
            lo = max(0, k - CTX + 1)
            for name in ("ac", "blind"):
                ctx = z[lo:k + 1][None]
                act = acts[lo:k + H][None] if name == "ac" else torch.full((1, k + H - lo), n_act, device="cuda")
                cur = ctx
                for h in range(1, H + 1):
                    p = predict(nets[name], cur[:, -CTX:], act[:, :cur.shape[1]][:, -CTX:])[:, -1:]
                    errs[name][h].append((p[0, 0] - z[k + h]).abs().mean().item())
                    cur = torch.cat([cur, p], 1)
            for h in range(1, H + 1):
                errs["copy_last"][h].append((z[k] - z[k + h]).abs().mean().item())
    res["prediction_l1"] = {name: {h: round(float(np.mean(v)), 4) for h, v in d.items()} for name, d in errs.items()}
    res["action_id_acc_h1"] = round(float(np.trace(conf) / conf.sum()), 3)
    res["action_id_acc_h2"] = round(float(np.trace(conf2) / max(conf2.sum(), 1)), 3)
    res["action_id_confusion_h1"] = conf.tolist()
    res["action_id_confusion_h2"] = conf2.tolist()
    res["chosen_action_hist_h1"] = conf.sum(0).tolist()
    res["prediction_spread_over_actions"] = round(float(np.median(spread)), 4)
    log("prediction", res["prediction_l1"])
    log("action id", res["action_id_acc_h1"], res["action_id_acc_h2"], "chosen", res["chosen_action_hist_h1"])

    # surprise vs events, with the plain visual-change baseline (copy-last error) as control
    def auc_for(key, flag, cond=None):
        sel = [r for r in rows if cond is None or cond(r)]
        return _auc([r[key] for r in sel if flag(r)], [r[key] for r in sel if not flag(r)])
    ev = {
        "bump_onset": lambda r: r["bump_onset"],
        "room_change": lambda r: r["room_change"],
        "head_drop_gt_5cm": lambda r: r["head_drop"] > 0.05,
    }
    moving = lambda r: r["action"] != 0  # noqa: E731  commanded to move
    out = {}
    for evn, f in ev.items():
        out[evn] = dict(n=sum(f(r) for r in rows),
                        auc_ac_surprise=auc_for("e_ac", f), auc_blind_surprise=auc_for("e_blind", f),
                        auc_visual_change=auc_for("e_copy", f),
                        when_commanded_to_move=dict(auc_ac_surprise=auc_for("e_ac", f, moving),
                                                    auc_visual_change=auc_for("e_copy", f, moving)))
    res["events"] = out
    log("events", out)
    os.makedirs(f"{OUT}/eval", exist_ok=True)
    json.dump(res, open(f"{OUT}/eval/results.json", "w"), indent=1)
    np.save(f"{OUT}/eval/rows.npy", np.array([[r["e_ac"], r["e_blind"], r["e_copy"], r["action"], r["bump_onset"],
                                               r["touching"], r["room_change"], r["head_drop"], r["moved"], r["k"]]
                                              for r in rows], np.float32))
    json.dump([r["ep"] for r in rows], open(f"{OUT}/eval/rows_ep.json", "w"))
    vol.commit()
    return {k: v for k, v in res.items() if "confusion" not in k}


@app.local_entrypoint()
def evaluate():
    print(json.dumps(evaluate_run.remote(), indent=1))


@app.function(image=image, volumes={"/data": vol}, cpu=4, memory=16384, timeout=900, retries=0)
def data_stats_run():
    import collections
    import glob

    import numpy as np
    out = {}
    for split in ("train", "test"):
        n = outside = touching = ends_outside = 0
        acts = collections.Counter()
        exits = []
        for p in glob.glob(f"{OUT}/{split}/*.npz"):
            z = np.load(p)
            room = z["room"]
            n += len(room)
            outside += int((room == "").sum())
            touching += int(z["contact"].sum())
            ends_outside += int(room[-1] == "")
            acts.update(z["actions"].tolist())
            if (room == "").any():
                k = int(np.argmax(room == ""))
                exits.append(z["xy"][k].tolist())
        out[split] = dict(frames=n, frac_outside_rooms=round(outside / n, 4), frac_touching=round(touching / n, 4),
                          episodes_ending_outside=ends_outside, episodes=len(glob.glob(f"{OUT}/{split}/*.npz")),
                          action_counts={ACTION_NAMES[k]: v for k, v in sorted(acts.items())})
    return out


@app.local_entrypoint()
def data_stats():
    print(json.dumps(data_stats_run.remote(), indent=1))


# --------------------------------------------------------------------------------------------------
# Imagination demo and image-goal planning in held-out houses.
def _plan_scores(net, ctx, acts_ctx, goal, n_act, horizon):
    """L1 distance to the goal after every action sequence of length `horizon` (all combinations).

    ctx: (1, T, 16, F) observed frames; acts_ctx: (1, T) actions after each frame, the last one to be chosen."""
    import itertools

    import torch
    seqs = torch.tensor(list(itertools.product(range(n_act), repeat=horizon)), device="cuda")  # (S, H)
    z = ctx.expand(len(seqs), -1, -1, -1)
    a = acts_ctx.expand(len(seqs), -1).clone()
    for h in range(horizon):
        a[:, -1] = seqs[:, h]
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = net(z[:, -CTX:], a[:, -CTX:]).float()[:, -1:]
        z = torch.cat([z, p], 1)
        a = torch.cat([a, torch.zeros_like(a[:, :1])], 1)
    return seqs, (z[:, -1] - goal[None]).abs().mean((1, 2))


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=65536, timeout=5400, retries=0,
              max_containers=3, scaledown_window=2)
def plan_run(house: str, n_goals: int = 12, bank_eps: int = 24, steps_max: int = 40, horizon: int = 2,
             seed: int = 0, video_goals: int = 2):
    """In one held-out house: record a bank of views (memory), then try to walk to goal photos.

    Policies per goal (same start pose): world-model planning, random actions, and an oracle-free 'greedy'
    that picks the action whose single predicted next view is closest to the goal (horizon 1).
    """
    import math
    import sys
    import time

    import numpy as np
    import torch
    sys.path.insert(0, "/pkg")
    from scout.body.motivo import MotivoBody
    from scout.world.layout import Layout
    t0 = time.time()
    log = lambda *a: print(round(time.time() - t0, 1), *a, flush=True)  # noqa: E731
    n_act = len(ACTIONS)
    net = _model(n_act).cuda().eval()
    net.load_state_dict(torch.load(f"{OUT}/models/ac.pt"))
    encode = _encoder()
    body = MotivoBody("/data/house", house, "/data/scout/z.pt", device="cuda")
    lay = Layout.load(f"/data/house/layouts/{house}.npz")
    rng = np.random.default_rng(seed)
    cells = np.argwhere(lay.grid.free)

    def enc(img):
        return _norm(torch.from_numpy(encode(img[None]))[0].cuda())

    # 1) memory bank: random walks in this house (raw head views + features + poses)
    bank_img, bank_xy, bank_yaw = [], [], []
    for ep in range(bank_eps):
        body.place(lay.grid.pos(cells[rng.integers(len(cells))]), float(rng.uniform(-math.pi, math.pi)))
        for _ in range(15):
            body.step("move-ego-0-0")
        a, hold = 1, 0
        for k in range(40):
            if hold <= 0:
                a, hold = int(rng.integers(n_act)), int(rng.integers(2, 5))
            hold -= 1
            for _ in range(SIM_PER_CTRL):
                body.step(ACTIONS[a])
            bank_img.append(body.head_view(256))
            bank_xy.append(body.xy)
            bank_yaw.append(body.yaw)
            if body.head_z < 0.8:
                break
    bank_img = np.stack(bank_img)
    bank_f = _norm(torch.from_numpy(encode(bank_img)).cuda())
    bank_xy, bank_yaw = np.array(bank_xy), np.array(bank_yaw)
    log("bank", len(bank_img))

    def nearest(f):
        return int((bank_f - f[None]).abs().mean((1, 2)).argmin())

    # 2) goals: a bank view 2-5 m (straight line) from a free start; start pose random
    results, frames_out = [], []
    for gi in range(n_goals):
        for _ in range(200):
            g = int(rng.integers(len(bank_img)))
            c = cells[rng.integers(len(cells))]
            s_xy = lay.grid.pos(c)
            dist = float(np.linalg.norm(bank_xy[g] - s_xy))
            if 2.0 <= dist <= 5.0:
                break
        s_yaw = float(rng.uniform(-math.pi, math.pi))
        goal_f = bank_f[g]
        for policy in ("plan", "greedy1", "random"):
            prng = np.random.default_rng(seed * 1000 + gi)
            body.place(s_xy, s_yaw)
            for _ in range(15):
                body.step("move-ego-0-0")
            img = body.head_view(256)
            ctx = enc(img)[None, None]                                  # (1, T, 16, F)
            acts = torch.zeros(1, 1, dtype=torch.long, device="cuda")  # (1, T), last entry = action to choose
            track, best_d, success, vid = [], 9.0, False, []
            record = gi < video_goals and policy == "plan"
            for k in range(steps_max):
                if policy == "random":
                    a = int(prng.integers(n_act))
                    imagined = None
                else:
                    h = horizon if policy == "plan" else 1
                    seqs, d = _plan_scores(net, ctx[:, -CTX:], acts[:, -CTX:], goal_f, n_act, h)
                    a = int(seqs[int(d.argmin()), 0])
                    imagined = None
                    if record:  # what each single action is imagined to show
                        imagined = []
                        for j in range(n_act):
                            aa = acts[:, -CTX:].clone()
                            aa[:, -1] = j
                            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                                pj = net(ctx[:, -CTX:], aa).float()[0, -1]
                            imagined.append(nearest(pj))
                acts[:, -1] = a
                fell = False
                for _ in range(SIM_PER_CTRL):
                    body.step(ACTIONS[a])
                if body.head_z < 0.8:
                    fell = True
                img = body.head_view(256)
                f = enc(img)
                ctx = torch.cat([ctx, f[None, None]], 1)
                acts = torch.cat([acts, torch.zeros(1, 1, dtype=torch.long, device="cuda")], 1)
                pos_err = float(np.linalg.norm(body.xy - bank_xy[g]))
                yaw_err = abs(math.degrees((body.yaw - bank_yaw[g] + math.pi) % (2 * math.pi) - math.pi))
                feat_d = float((f - goal_f).abs().mean())
                track.append(dict(k=k, a=a, pos_err=round(pos_err, 2), yaw_err=round(yaw_err, 1), feat_d=round(feat_d, 4)))
                best_d = min(best_d, pos_err)
                if record:
                    vid.append(dict(head=img, follow=body.follow_view(math.degrees(body.yaw) + 180, 360, 480),
                                    imagined=imagined, action=a, pos_err=pos_err, yaw_err=yaw_err))
                if pos_err < 0.75 and yaw_err < 45:
                    success = True
                    break
                if fell:
                    break
            results.append(dict(goal=gi, policy=policy, success=success, steps=len(track), start_dist=round(dist, 2),
                                final_pos_err=track[-1]["pos_err"], final_yaw_err=track[-1]["yaw_err"],
                                best_pos_err=round(best_d, 2), track=track))
            if record:
                frames_out.append(dict(goal=gi, goal_img=bank_img[g], start_dist=dist, steps=vid,
                                       bank_imgs={i: bank_img[i] for st in vid for i in (st["imagined"] or [])}))
        log("goal", gi, {r["policy"]: (r["success"], r["final_pos_err"]) for r in results if r["goal"] == gi})
    os.makedirs(f"{OUT}/plan", exist_ok=True)
    json.dump(results, open(f"{OUT}/plan/{house}.json", "w"))
    for fo in frames_out:
        np.savez_compressed(f"{OUT}/plan/{house}_goal{fo['goal']}_video.npz", goal_img=fo["goal_img"],
                            head=np.stack([s["head"] for s in fo["steps"]]),
                            follow=np.stack([s["follow"] for s in fo["steps"]]),
                            imagined=np.array([s["imagined"] for s in fo["steps"]]),
                            actions=np.array([s["action"] for s in fo["steps"]]),
                            pos_err=np.array([s["pos_err"] for s in fo["steps"]]),
                            bank_ids=np.array(sorted(fo["bank_imgs"])),
                            bank=np.stack([fo["bank_imgs"][i] for i in sorted(fo["bank_imgs"])]))
    vol.commit()
    summ = {}
    for p in ("plan", "greedy1", "random"):
        rs = [r for r in results if r["policy"] == p]
        summ[p] = dict(n=len(rs), success=sum(r["success"] for r in rs), mean_final_pos_err=round(float(np.mean([r["final_pos_err"] for r in rs])), 2),
                       mean_best_pos_err=round(float(np.mean([r["best_pos_err"] for r in rs])), 2))
    return dict(house=house, summary=summ, minutes=round((time.time() - t0) / 60, 1))


@app.local_entrypoint()
def plan(n_houses: int = 3, n_goals: int = 12):
    import random
    splits = json.load(open(os.path.join(CFG, "splits.json")))
    te = list(splits["test"])
    random.Random(8).shuffle(te)
    houses = te[:n_houses]  # the same held-out houses used for test episodes
    for r in plan_run.starmap([(h, n_goals) for h in houses]):
        print(json.dumps(r))
