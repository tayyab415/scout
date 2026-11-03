"""Open-vocabulary eyes: SAM 3.1 (or the SAM 3 image model) on one head-camera frame. Needs the sam3 package and a GPU."""
from __future__ import annotations

import inspect
import tempfile

import numpy as np


class Sam31:
    """SAM 3.1 multiplex video model, run on a one-frame clip per look."""
    name = "sam3.1"

    def __init__(self):
        from sam3.model_builder import build_sam3_predictor
        self.pred = build_sam3_predictor(version="sam3.1", use_fa3=False)
        init, ok = self.pred.model.init_state, set(inspect.signature(self.pred.model.init_state).parameters)
        self.pred.model.init_state = lambda **kw: init(**{k: v for k, v in kw.items() if k in ok})  # upstream kwarg bug

    def detect(self, img: np.ndarray, prompts) -> dict:
        """-> {prompt: [(score, mask), ...]} for every instance found."""
        from PIL import Image
        d = tempfile.mkdtemp()
        Image.fromarray(img).save(f"{d}/00000.jpg", quality=95)
        sid = self.pred.handle_request({"type": "start_session", "resource_path": d})["session_id"]
        out = {}
        for p in prompts:
            self.pred.handle_request({"type": "reset_session", "session_id": sid})
            r = self.pred.handle_request({"type": "add_prompt", "session_id": sid, "frame_index": 0, "text": p})["outputs"]
            probs = np.asarray(r["out_probs"], float).ravel()
            masks = np.asarray(r["out_binary_masks"]).reshape(-1, *img.shape[:2])
            out[p] = [(float(s), m) for s, m in zip(probs, masks)]
        self.pred.handle_request({"type": "close_session", "session_id": sid})
        return out


class Sam3Image:
    """SAM 3 image model: one backbone pass per frame, then one cheap decoder pass per prompt."""
    name = "sam3_image"

    def __init__(self):
        from sam3.model.sam3_image_processor import Sam3Processor
        from sam3.model_builder import build_sam3_image_model
        self.proc = Sam3Processor(build_sam3_image_model())

    def detect(self, img: np.ndarray, prompts) -> dict:
        from PIL import Image
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
