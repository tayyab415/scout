"""Branch replay check on any GPU box: one snapshot, the same 120 actions twice, compare state and head images.

    MUJOCO_GL=egl python scripts/replay_local.py --house procthor-10k-val_val_935 --data-root /data
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--house", default="procthor-10k-val_val_935")
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--data-root", default="/data")
    a = ap.parse_args()
    from scout.env.core import ScoutEnv
    print(f"house {a.house}: reset, walk 2 s, snapshot", flush=True)
    env = ScoutEnv(a.house, 0, root=f"{a.data_root}/house", skills=f"{a.data_root}/scout/z_motion.pt", eyes=None)
    env.reset()
    Z = env.Z
    for _ in range(60):
        env.step(Z["move-ego-0-1"])
    snap = env.snapshot()
    seq = [Z["move-ego-0-1"] if (i // 30) % 2 == 0 else Z["move-ego-90-1"] for i in range(a.steps)]
    traces = []
    for rep in range(2):
        env.restore(snap)
        q, imgs = [], []
        for i, z in enumerate(seq):
            env.step(z)
            q.append(env.D.qpos.copy())
            if i % 15 == 0:
                imgs.append(env.observe()["rgb"].astype(np.int16))
        traces.append((np.array(q), imgs, env.t))
        print(f"run {rep + 1}: {a.steps} steps from the snapshot, ended at t={env.t:.2f} s", flush=True)
    pix = [np.abs(x - y) for x, y in zip(traces[0][1], traces[1][1])]
    res = dict(house=a.house, steps=a.steps, max_abs_qpos_diff=float(np.abs(traces[0][0] - traces[1][0]).max()),
               n_images=len(pix), images_identical=sum(int(p.max() == 0) for p in pix),
               max_pixel_diff=[int(p.max()) for p in pix], frac_pixels_differing=[round(float((p > 0).mean()), 6) for p in pix],
               gpu=os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip())
    print(json.dumps(res, indent=1), flush=True)
    json.dump(res, open("replay_result.json", "w"), indent=1)


if __name__ == "__main__":
    main()
