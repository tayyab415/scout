"""Speed-up test: SAM 3.1 weights in SAM 3's image mode (encode the frame once, then one cheap pass per prompt)
against the pipeline the environment uses (SAM 3.1 video predictor, one session per category).

    python scripts/sam_speed_test.py --frames <dir of head-camera jpgs> [--n 20]

Reports seconds per look for both, and agreement: for each frame and category, do both detect it, and how much do
the best masks overlap (IoU)? The new path is only acceptable if detections agree.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scout.env.core import PROMPTS  # noqa: E402
from scout.perception.sam import Sam31  # noqa: E402


class Sam31Image:
    """SAM 3.1 checkpoint loaded into the SAM 3 image model: one image encoding per frame, then 16 text prompts."""

    def __init__(self):
        import torch
        from huggingface_hub import hf_hub_download
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        ckpt = hf_hub_download("facebook/sam3.1", "sam3.1_multiplex.pt")
        model = build_sam3_image_model(checkpoint_path=None, load_from_HF=False)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        # keep the detector weights; the multiplex checkpoint prefixes them with "detector."
        det = {k[len("detector."):]: v for k, v in sd.items() if k.startswith("detector.")}
        missing, unexpected = model.load_state_dict(det, strict=False)
        self.load_report = dict(loaded=len(det), missing=len(missing), unexpected=len(unexpected),
                                missing_examples=list(missing)[:8])
        model = model.cuda().eval()
        self.proc = Sam3Processor(model)

    def detect(self, img, prompts):
        st = self.proc.set_image(Image.fromarray(img))
        out = {}
        for p in prompts:
            o = self.proc.set_text_prompt(state=st, prompt=p)
            if len(o["scores"]) == 0:
                out[p] = []
                continue
            s = o["scores"].float().cpu().numpy().ravel()
            m = o["masks"].cpu().numpy().reshape(-1, *img.shape[:2]).astype(bool)
            out[p] = [(float(a), b) for a, b in zip(s, m)]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", default="sam_speed_test.json")
    a = ap.parse_args()
    paths = sorted(glob.glob(f"{a.frames}/*.jpg"))
    paths = [paths[i] for i in np.linspace(0, len(paths) - 1, a.n).astype(int)]
    imgs = [np.asarray(Image.open(p).convert("RGB")) for p in paths]
    ref, new = Sam31(), Sam31Image()
    res = dict(load_report=new.load_report, frames=len(imgs))
    times = {"sam31_video_per_category": [], "sam31_image_mode": []}
    agree = []
    for img in imgs:
        t0 = time.time()
        a_ = ref.detect(img, PROMPTS)
        times["sam31_video_per_category"].append(time.time() - t0)
        t0 = time.time()
        b_ = new.detect(img, PROMPTS)
        times["sam31_image_mode"].append(time.time() - t0)
        for p in PROMPTS:
            da = [m for s, m in a_[p] if s >= 0.5]
            db = [m for s, m in b_[p] if s >= 0.5]
            iou = None
            if da and db:
                ua, ub = np.any(da, 0), np.any(db, 0)
                iou = float((ua & ub).sum() / max((ua | ub).sum(), 1))
            agree.append(dict(prompt=p, ref=bool(da), new=bool(db), iou=iou))
    both = [x for x in agree if x["ref"] and x["new"]]
    res.update(sec_per_look={k: round(float(np.mean(v[1:] or v)), 3) for k, v in times.items()},
               detections_ref=sum(x["ref"] for x in agree), detections_new=sum(x["new"] for x in agree),
               agree_presence=round(float(np.mean([x["ref"] == x["new"] for x in agree])), 3),
               mean_iou_when_both=round(float(np.mean([x["iou"] for x in both])), 3) if both else None)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
