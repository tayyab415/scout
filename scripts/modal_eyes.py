"""Can SAM 3.1 see house furniture from the head camera? Measured against the simulator's segmentation.

    modal run scripts/modal_eyes.py

For a few held-out houses: random standing spots, render the eye-height view (RGB + segmentation), ask SAM for each
object category, and compare with which categories really cover >= 0.5% of the image. Compares SAM 3.1 (multiplex
video model on a 1-frame clip) with the SAM 3 image model. Writes /data/scout/eyes/results.json and example frames.
"""
import os

import modal

app = modal.App("scout-eyes")
vol = modal.Volume.from_name("tribe-lab-claude")
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
CFG = os.path.join(os.path.dirname(__file__), "..", "configs")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "ffmpeg", "libgl1", "libegl1", "libglib2.0-0")
         .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
         .run_commands("git clone https://github.com/facebookresearch/sam3.git /sam3 && pip install -e /sam3")
         .pip_install("imageio[ffmpeg]", "pillow", "einops", "decord", "pycocotools", "psutil", "scipy", "pandas",
                      "opencv-python-headless", "mujoco==3.17.0")
         .env({"HF_HOME": "/data/hf", "MUJOCO_GL": "egl"})
         .add_local_dir(SRC, "/pkg")
         .add_local_dir(CFG, "/cfg"))
_tok = os.path.expanduser("~/.cache/huggingface/token")
secrets = [modal.Secret.from_dict({"HF_TOKEN": open(_tok).read().strip() if modal.is_local() and os.path.exists(_tok) else ""})]
RES = 512
MIN_FRAC = 0.005
CATS = {  # prompt -> body-name prefixes used by the THOR assets
    "bed": ["bed"], "toilet": ["crapper", "toilet"], "sofa": ["sofa", "couch"], "television": ["television", "tv"],
    "dining table": ["diningtable"], "fridge": ["refrigerator", "fridge"], "desk": ["desk"],
    "bathtub": ["bathtub", "tub"], "sink": ["sink"], "dresser": ["chestofdrawers", "dresser"],
    "armchair": ["armchair"], "kitchen counter": ["countertop"], "microwave": ["microwave"], "toaster": ["toaster"],
    "coffee machine": ["coffeemaker"], "laptop": ["laptop"], "house plant": ["houseplant"],
    "floor lamp": ["floorlamp"], "painting": ["painting"], "chair": ["chair"], "pillow": ["pillow"],
    "cup": ["cup", "mug"], "bowl": ["bowl"], "book": ["book"], "apple": ["apple"],
}
PREFIX = {pre: c for c, pres in CATS.items() for pre in pres}


def sam31_predictor():
    import inspect

    from sam3.model_builder import build_sam3_predictor
    pred = build_sam3_predictor(version="sam3.1", use_fa3=False)
    init, ok = pred.model.init_state, set(inspect.signature(pred.model.init_state).parameters)
    pred.model.init_state = lambda **kw: init(**{k: v for k, v in kw.items() if k in ok})  # upstream kwarg bug
    return pred


def sam31_detect(pred, img, prompts):
    """img (H, W, 3) uint8 -> {prompt: (best score, union mask)} using SAM 3.1 on a one-frame clip."""
    import tempfile

    import numpy as np
    from PIL import Image
    d = tempfile.mkdtemp()
    Image.fromarray(img).save(f"{d}/00000.jpg", quality=95)
    out = {}
    sid = pred.handle_request({"type": "start_session", "resource_path": d})["session_id"]
    for p in prompts:
        pred.handle_request({"type": "reset_session", "session_id": sid})
        r = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": p})["outputs"]
        probs = np.asarray(r["out_probs"], float).ravel()
        masks = np.asarray(r["out_binary_masks"]).reshape(-1, *img.shape[:2])
        out[p] = (float(probs.max()) if len(probs) else 0.0, masks.any(0) if len(masks) else np.zeros(img.shape[:2], bool))
    pred.handle_request({"type": "close_session", "session_id": sid})
    return out


def sam3_image():
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model
    return Sam3Processor(build_sam3_image_model())


def sam3_detect(proc, img, prompts):
    import numpy as np
    from PIL import Image
    st = proc.set_image(Image.fromarray(img))
    out = {}
    for p in prompts:
        o = proc.set_text_prompt(state=st, prompt=p)
        s = o["scores"].float().cpu().numpy().ravel() if len(o["scores"]) else np.zeros(0)
        m = o["masks"].cpu().numpy().reshape(-1, *img.shape[:2]) if len(o["masks"]) else np.zeros((0, *img.shape[:2]), bool)
        out[p] = (float(s.max()) if len(s) else 0.0, m.any(0) if len(m) else np.zeros(img.shape[:2], bool))
    return out


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=8, memory=65536, timeout=3600, secrets=secrets)
def run(n_houses: int = 4, views: int = 8):
    import json
    import sys
    import time

    import mujoco
    import numpy as np
    sys.path.insert(0, "/pkg")
    from scout.world.house import EYE_Z, build_house
    from scout.world.layout import GOAL_TYPES, Layout
    t0 = time.time()
    test = json.load(open("/cfg/splits.json"))["test"]
    rng = np.random.default_rng(7)
    prompts = sorted(CATS)
    models = {"sam3.1": (sam31_predictor(), sam31_detect), "sam3_image": (sam3_image(), sam3_detect)}
    print("models loaded", round(time.time() - t0), flush=True)
    rows, examples, names_seen = [], {}, set()
    for house in rng.choice(test, n_houses, replace=False):
        print("building", house, flush=True)
        M = build_house("/data/house", house, ceiling=True)
        M.vis.global_.fovy = 75
        M.vis.global_.offwidth, M.vis.global_.offheight = RES, RES
        D = mujoco.MjData(M)
        mujoco.mj_forward(M, D)
        lay = Layout.load(f"/data/house/layouts/{house}.npz")
        btype = {}
        for b in range(M.nbody):
            n = (mujoco.mj_id2name(M, mujoco.mjtObj.mjOBJ_BODY, b) or "")
            n2 = n[2:] if n.startswith("h_") else n
            names_seen.add(n2.split("_")[0].lower())
            c = PREFIX.get(n2.split("_")[0].lower())
            if c:
                btype[b] = c
        rd = mujoco.Renderer(M, RES, RES)
        cells = np.argwhere(lay.grid.free)
        for v in range(views):
            xy = lay.grid.pos(cells[rng.integers(len(cells))])
            yaw = float(rng.uniform(0, 360))
            cam = mujoco.MjvCamera()
            pitch = -15.0
            fwd = np.array([np.cos(np.radians(yaw)) * np.cos(np.radians(pitch)), np.sin(np.radians(yaw)) * np.cos(np.radians(pitch)), np.sin(np.radians(pitch))])
            cam.lookat[:] = np.array([xy[0], xy[1], EYE_Z]) + fwd
            cam.distance, cam.azimuth, cam.elevation = 1.0, yaw, pitch
            rd.update_scene(D, camera=cam)
            img = rd.render().copy()
            rd.enable_segmentation_rendering()
            rd.update_scene(D, camera=cam)
            seg = rd.render().copy()
            rd.disable_segmentation_rendering()
            gid = seg[..., 0]
            isg = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (gid >= 0)
            body = np.full(gid.shape, -1)
            body[isg] = M.geom_bodyid[gid[isg]]
            gt = {}
            for p in prompts:
                bs = [b for b, t in btype.items() if t == p]
                m = np.isin(body, bs) if bs else np.zeros(gid.shape, bool)
                gt[p] = m
            for name, (mdl, fn) in models.items():
                ts = time.time()
                det = fn(mdl, img, prompts)
                dt = time.time() - ts
                for p in prompts:
                    score, m = det[p]
                    g = gt[p]
                    inter, union = float((m & g).sum()), float((m | g).sum())
                    rows.append(dict(model=name, house=house, view=v, prompt=p, visible=bool(g.mean() >= MIN_FRAC),
                                     gt_frac=float(g.mean()), score=score, iou=inter / union if union else None,
                                     sec_per_image=dt))
            print("view", house, v, {n: round(float(np.mean([r["sec_per_image"] for r in rows if r["model"] == n][-1:])), 1) for n in models}, flush=True)
            key = f"{house}_{v}"
            if len(examples) < 12:
                examples[key] = img
        rd.close()
        print("house done", house, round(time.time() - t0), flush=True)

    # summary per model at a few score thresholds
    summ = {}
    for name in models:
        R = [r for r in rows if r["model"] == name]
        s = {"sec_per_image": round(float(np.mean([r["sec_per_image"] for r in R])) / 1.0, 2)}
        for th in (0.3, 0.5, 0.7):
            tp = sum(r["visible"] and r["score"] >= th for r in R)
            fp = sum((not r["visible"]) and r["score"] >= th for r in R)
            fn_ = sum(r["visible"] and r["score"] < th for r in R)
            s[f"th{th}"] = dict(precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn_, 1), 3), tp=tp, fp=fp, fn=fn_)
        ious = [r["iou"] for r in R if r["visible"] and r["score"] >= 0.5 and r["iou"] is not None]
        s["mean_iou_when_found"] = round(float(np.mean(ious)), 3) if ious else None
        per = {}
        for p in prompts:
            Rp = [r for r in R if r["prompt"] == p]
            vis = [r for r in Rp if r["visible"]]
            per[p] = dict(visible=len(vis), found=sum(r["score"] >= 0.5 for r in vis),
                          false_alarms=sum(r["score"] >= 0.5 and not r["visible"] for r in Rp))
        s["per_category"] = per
        summ[name] = s
    os.makedirs("/data/scout/eyes", exist_ok=True)
    out = dict(summary=summ, n_images=n_houses * views, min_frac=MIN_FRAC, body_name_samples=sorted(names_seen),
               minutes=round((time.time() - t0) / 60, 1))
    json.dump(out, open("/data/scout/eyes/results.json", "w"), indent=1)
    json.dump(rows, open("/data/scout/eyes/rows.json", "w"))
    np.savez_compressed("/data/scout/eyes/examples.npz", **examples)
    vol.commit()
    return json.loads(json.dumps(out, default=float))


@app.local_entrypoint()
def main(n_houses: int = 4, views: int = 8):
    import json
    r = run.remote(n_houses, views)
    print(json.dumps({k: v for k, v in r.items() if k != "body_name_samples"}, indent=1))
    print(r["body_name_samples"])
