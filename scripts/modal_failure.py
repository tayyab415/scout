"""Does the head camera, through frozen V-JEPA 2, tell the body it is about to fail (get stuck or fall)?

    modal run scripts/modal_failure.py --tag u20

Data: the body's own runs (configs/untouched20.json). Every 0.5 s look has a head frame, the movement command and its
position. Label: a "stuck" or "fell" event within the next 4 s. Features:
  - vjepa: V-JEPA 2 ViT-L on the last two head frames (0.5 s apart) as a 2-frame clip, mean + 2x2 pooled tokens
  - motion: speed over the last 1 s and 2 s, change in heading, commanded direction (no camera)
  - both
Split by house (15 train, 5 test). Logistic regression; report test AUC and average precision. Results:
/data/scout/failure/results.json and per-look predictions for the video.
"""
import os

import modal

app = modal.App("scout-failure")
vol = modal.Volume.from_name("tribe-lab-claude")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("transformers==4.57.1", "scikit-learn", "pillow", "numpy")
         .env({"HF_HOME": "/data/hf"}))
CFG = os.path.join(os.path.dirname(__file__), "..", "configs")
HORIZON = 4.0


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", timeout=5400, max_containers=1)
def run(tag: str, houses: list):
    import io
    import json
    import math

    import numpy as np
    import torch
    from PIL import Image
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.preprocessing import StandardScaler
    from transformers import AutoModel
    m = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", torch_dtype=torch.float16).cuda().eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 1, 3, 1, 1)

    @torch.no_grad()
    def enc(pairs):  # (N, 2, 256, 256, 3) uint8
        x = torch.from_numpy(pairs).cuda().permute(0, 1, 4, 2, 3).float().div(255)
        x = ((x - mean) / std).half()
        h = m.get_vision_features(pixel_values_videos=x).float()  # (N, 256, D)
        n, L, d = h.shape
        g = int(round(L ** 0.5))
        h2 = torch.nn.functional.adaptive_avg_pool2d(h.view(n, g, g, d).permute(0, 3, 1, 2), 2).reshape(n, -1)
        return torch.cat([h.mean(1), h2], 1).cpu().numpy()

    rows = []
    for house in houses:
        d = f"/data/scout/live/{tag}_{house}"
        if not os.path.exists(f"{d}/log.json"):
            continue
        log = json.load(open(f"{d}/log.json"))
        looks = log["looks"]
        bad_t = [e["t"] for e in log["events"] if e["kind"] in ("stuck", "fell")]
        frames = [np.asarray(Image.open(f"{d}/frame_{lk['frame']:04d}.jpg").convert("RGB").resize((256, 256))) for lk in looks]
        xy = np.array([lk["xy"] for lk in looks])
        yaw = np.array([lk["yaw"] for lk in looks])
        feats = []
        for i in range(0, len(looks), 64):
            idx = list(range(i, min(i + 64, len(looks))))
            pairs = np.stack([np.stack([frames[max(j - 1, 0)], frames[j]]) for j in idx])
            feats.append(enc(pairs))
        F = np.concatenate(feats)
        for j, lk in enumerate(looks):
            t = lk["t"]
            y = any(t < b <= t + HORIZON for b in bad_t)
            sp1 = np.linalg.norm(xy[j] - xy[max(j - 2, 0)]) / 1.0
            sp2 = np.linalg.norm(xy[j] - xy[max(j - 4, 0)]) / 2.0
            dyaw = abs(math.remainder(yaw[j] - yaw[max(j - 2, 0)], 2 * math.pi))
            rel = lk.get("cmd_rel")
            rel = 0.0 if rel is None else rel
            motion = [sp1, sp2, dyaw, math.cos(math.radians(rel)), math.sin(math.radians(rel)), lk.get("head_z", 1.5)]
            rows.append(dict(house=house, t=t, y=int(y), v=F[j], motion=np.array(motion, float), frame=lk["frame"]))
        print(house, len(looks), "looks", sum(r["y"] for r in rows if r["house"] == house), "positive", flush=True)

    hs = sorted({r["house"] for r in rows})
    test_h = set(hs[-5:])
    tr = [r for r in rows if r["house"] not in test_h]
    te = [r for r in rows if r["house"] in test_h]
    Y_tr, Y_te = np.array([r["y"] for r in tr]), np.array([r["y"] for r in te])
    res = dict(n_train=len(tr), n_test=len(te), pos_train=int(Y_tr.sum()), pos_test=int(Y_te.sum()),
               test_houses=sorted(test_h), horizon_s=HORIZON)
    preds = {}
    for name, f in {"motion": lambda r: r["motion"], "vjepa": lambda r: r["v"],
                    "both": lambda r: np.concatenate([r["v"], r["motion"]])}.items():
        X_tr, X_te = np.stack([f(r) for r in tr]), np.stack([f(r) for r in te])
        sc = StandardScaler().fit(X_tr)
        best = None
        for C in (0.001, 0.01, 0.1, 1.0):
            # choose C on the last training house (no peeking at test houses)
            val_h = sorted({r["house"] for r in tr})[-3:]
            vm = np.array([r["house"] in val_h for r in tr])
            clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X_tr[~vm]), Y_tr[~vm])
            ap = average_precision_score(Y_tr[vm], clf.predict_proba(sc.transform(X_tr[vm]))[:, 1]) if Y_tr[vm].sum() else 0
            if best is None or ap > best[0]:
                best = (ap, C)
        clf = LogisticRegression(C=best[1], max_iter=3000, class_weight="balanced").fit(sc.transform(X_tr), Y_tr)
        p = clf.predict_proba(sc.transform(X_te))[:, 1]
        preds[name] = p
        res[name] = dict(C=best[1], auc=round(float(roc_auc_score(Y_te, p)), 3) if Y_te.sum() else None,
                         ap=round(float(average_precision_score(Y_te, p)), 3) if Y_te.sum() else None,
                         base_rate=round(float(Y_te.mean()), 3))
        print(name, res[name], flush=True)
    os.makedirs("/data/scout/failure", exist_ok=True)
    json.dump(res, open("/data/scout/failure/results.json", "w"), indent=1)
    np.savez_compressed("/data/scout/failure/test_preds.npz", house=np.array([r["house"] for r in te]),
                        t=np.array([r["t"] for r in te]), y=Y_te, frame=np.array([r["frame"] for r in te]),
                        **{f"p_{k}": v for k, v in preds.items()})
    vol.commit()
    return res


@app.local_entrypoint()
def main(tag: str = "u20"):
    import json
    houses = json.load(open(os.path.join(CFG, "untouched20.json")))
    print(json.dumps(run.remote(tag, houses), indent=1))
