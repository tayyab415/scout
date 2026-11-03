"""SAM 3.1 (open weights, facebook/sam3.1) turns a video of a person into a silhouette clip for the bridge.

    modal run scripts/modal_sam.py::check            # SAM 3.1 vs the simulator's exact silhouette on bridge clips
    modal run scripts/modal_sam.py::video --path clip.mp4 --out runs/sam/clip_sil.npz

The bridge was trained on white-on-black silhouettes filmed by a camera that follows the pelvis, with the body
filling about 57% of the frame height. `to_bridge_clip` crops real video the same way.
"""
import os

import modal

app = modal.App("scout-sam")
vol = modal.Volume.from_name("tribe-lab-claude")
image = (modal.Image.debian_slim(python_version="3.12")
         .apt_install("git", "ffmpeg", "libgl1", "libglib2.0-0")
         .pip_install("torch==2.10.0", "torchvision", index_url="https://download.pytorch.org/whl/cu128")
         .run_commands("git clone https://github.com/facebookresearch/sam3.git /sam3 && pip install -e /sam3")
         .pip_install("imageio[ffmpeg]", "pillow", "einops", "decord", "pycocotools", "psutil", "scipy", "pandas", "opencv-python-headless")
         .env({"HF_HOME": "/data/hf"}))
_tok = os.path.expanduser("~/.cache/huggingface/token")
secrets = [modal.Secret.from_dict({"HF_TOKEN": open(_tok).read().strip() if modal.is_local() and os.path.exists(_tok) else ""})]
RES = 256


def _predictor():
    import inspect

    from sam3.model_builder import build_sam3_predictor
    pred = build_sam3_predictor(version="sam3.1", use_fa3=False)
    # upstream bug (main, Nov 2026): start_session passes kwargs the multiplex model's init_state does not take
    init, ok = pred.model.init_state, set(inspect.signature(pred.model.init_state).parameters)
    pred.model.init_state = lambda **kw: init(**{k: v for k, v in kw.items() if k in ok})
    return pred


def _segment(pred, frames, prompt="person"):
    """frames (T, H, W, 3) uint8 -> (T, H, W) bool mask of the largest-tracked person."""
    import tempfile

    import numpy as np
    from PIL import Image
    d = tempfile.mkdtemp()
    for i, f in enumerate(frames):
        Image.fromarray(f).save(f"{d}/{i:05d}.jpg", quality=95)
    sid = pred.handle_request({"type": "start_session", "resource_path": d})["session_id"]
    pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": prompt})
    masks = np.zeros(frames.shape[:3], bool)
    for out in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
        fi = out["frame_index"]
        m = out["outputs"]["out_binary_masks"] if "outputs" in out else out.get("out_binary_masks")
        m = np.asarray(m.cpu() if hasattr(m, "cpu") else m)
        if m.size:
            masks[fi] = m.reshape(-1, *frames.shape[1:3]).any(0)
    pred.handle_request({"type": "close_session", "session_id": sid})
    return masks


def to_bridge_clip(masks, n=32, fill=0.57):
    """Crop and scale a person mask video like the bridge camera: body centred, ~57% of frame height, 32 frames."""
    import numpy as np
    from PIL import Image
    idx = np.linspace(0, len(masks) - 1, n).round().astype(int)
    boxes = []
    for m in masks[idx]:
        ys, xs = np.nonzero(m)
        boxes.append((xs.mean(), ys.mean(), ys.max() - ys.min()) if len(ys) else (np.nan, np.nan, np.nan))
    b = np.array(boxes, float)
    h = np.nanmedian(b[:, 2])
    side = h / fill
    out = np.zeros((n, RES, RES), np.uint8)
    for k, i in enumerate(idx):
        cx, cy = b[k, 0], b[k, 1]
        if np.isnan(cx):
            continue
        img = Image.fromarray(masks[i].astype(np.uint8) * 255)
        x0, y0 = cx - side / 2, cy - side / 2
        out[k] = np.asarray(img.crop((int(x0), int(y0), int(x0 + side), int(y0 + side))).resize((RES, RES)))
    return out


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", timeout=3600, secrets=secrets)
def check_run(n_clips: int = 8):
    """SAM 3.1 on the bridge showcase clips (rendered RGB) vs the simulator's exact body mask."""
    import json

    import numpy as np
    s = np.load("/data/scout/bridge/showcase.npz")
    pred = _predictor()
    ids = sorted({k.split("_")[0] for k in s.files}, key=int)[:n_clips]
    res = {}
    for i in ids:
        rgb = s[f"{i}_source"]
        m = _segment(pred, rgb)
        res[i] = dict(frames_with_mask=int(m.any((1, 2)).sum()), n=len(m), mean_area=float(m.mean()))
    json.dump(res, open("/data/scout/bridge/sam_check.json", "w"), indent=1)
    vol.commit()
    return res


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", timeout=3600, secrets=secrets)
def video_run(data: bytes, prompt: str = "person"):
    import tempfile

    import imageio.v2 as imageio
    import numpy as np
    p = tempfile.mktemp(suffix=".mp4")
    open(p, "wb").write(data)
    frames = np.stack([f[..., :3] for f in imageio.get_reader(p)])
    pred = _predictor()
    m = _segment(pred, frames, prompt)
    return dict(sil=to_bridge_clip(m), raw_mask_frac=m.any((1, 2)).mean(), fps_frames=len(frames))


@app.local_entrypoint()
def check():
    print(check_run.remote())


@app.local_entrypoint()
def video(path: str, out: str, prompt: str = "person"):
    import numpy as np
    r = video_run.remote(open(path, "rb").read(), prompt)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, sil=r["sil"])
    print("frames", r["fps_frames"], "mask found in", r["raw_mask_frac"])


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", timeout=1800, secrets=secrets)
def debug_run(clip: str = "3006"):
    import tempfile

    import numpy as np
    from PIL import Image
    s = np.load("/data/scout/bridge/showcase.npz")
    frames = s[f"{clip}_source"]
    pred = _predictor()
    d = tempfile.mkdtemp()
    for i, f in enumerate(frames):
        Image.fromarray(f).save(f"{d}/{i:05d}.jpg", quality=95)
    rep = {}
    for prompt in ("person", "mannequin", "human figure", "doll"):
        sid = pred.handle_request({"type": "start_session", "resource_path": d})["session_id"]
        r = pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": prompt})
        desc = lambda o: {k: (getattr(v, "shape", None) or (type(v).__name__ + (f" len {len(v)}" if hasattr(v, "__len__") else ""))) for k, v in o.items()} if isinstance(o, dict) else type(o).__name__  # noqa: E731
        first = None
        n = 0
        for out in pred.handle_stream_request({"type": "propagate_in_video", "session_id": sid}):
            n += 1
            if first is None:
                first = {k: desc(v) if isinstance(v, dict) else str(getattr(v, "shape", type(v).__name__)) for k, v in out.items()}
        rep[prompt] = dict(add_prompt=desc(r) if isinstance(r, dict) else str(r)[:200],
                           add_outputs=desc(r.get("outputs", {})) if isinstance(r, dict) else None, n_stream=n, first=first)
        pred.handle_request({"type": "close_session", "session_id": sid})
    return rep


@app.local_entrypoint()
def debug(clip: str = "3006"):
    import json
    print(json.dumps(debug_run.remote(clip), indent=1, default=str))
