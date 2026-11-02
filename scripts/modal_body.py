"""Walk the Motivo body along exported routes in the 3D house and film it.

    modal run scripts/modal_body.py --routes runs/routes/val105_sofa_teacher.json
Writes <route>_body.mp4 (third person | head camera) and <route>_body.json (body track) next to the route.
"""
import json
import os

import modal

app = modal.App("scout-body")
vol = modal.Volume.from_name("tribe-lab-claude")
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("git", "ffmpeg", "libgl1", "libegl1", "libglib2.0-0")
         .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cu124")
         .pip_install("metamotivo[huggingface,humenv] @ git+https://github.com/facebookresearch/metamotivo.git",
                      "mujoco==3.17.0", "imageio[ffmpeg]")
         .env({"HF_HOME": "/data/hf", "MUJOCO_GL": "egl"})
         .add_local_dir(SRC, "/pkg"))


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=4, memory=32768, timeout=1800, retries=0,
              max_containers=4, scaledown_window=2)
def walk(route: dict, max_seconds: float = 90.0) -> dict:
    import math
    import sys
    import time

    import imageio
    import numpy as np
    sys.path.insert(0, "/pkg")
    from scout.body.follow import DIRECTIONS, STAND, densify, steer
    from scout.body.motivo import MotivoBody

    t0 = time.time()
    body = MotivoBody("/data/house", route["house"], "/data/stack/z.pt")
    available = {a: s for a, s in DIRECTIONS.items() if s in body.Z}
    print("walking directions available:", sorted(available), flush=True)
    pts = densify(route["route"], 0.1)
    body.place(pts[0], route["start_yaw"])
    for _ in range(20):  # settle
        body.step(STAND)
    frames, track, k, cam_az = [], [], 0, math.degrees(body.yaw) + 180
    done, fell, err, steps = False, False, None, int(max_seconds * 30)
    min_head = 9.0
    skill = STAND
    for i in range(steps):
        if i % 6 == 0:  # decide five times a second
            skill, k, done = steer(body.xy, body.yaw, pts, k, available=available)
        try:
            body.step(skill)
        except ValueError as e:
            err = str(e)[:120]
            break
        min_head = min(min_head, body.head_z)
        if body.head_z < 0.8:
            fell = True
        track.append(dict(t=i / 30, x=float(body.xy[0]), y=float(body.xy[1]), yaw=body.yaw, skill=skill, progress=k))
        if i % 2 == 0:
            move = math.degrees(math.atan2(*(pts[min(k + 10, len(pts) - 1)] - body.xy)[::-1]))
            cam_az += 0.08 * (((move + 180 - cam_az) + 180) % 360 - 180)
            third = body.follow_view(cam_az)
            head = body.head_view(480)
            frames.append(np.concatenate([third, head], 1))
        if done:
            for _ in range(30):
                body.step(STAND)
            break
    name = os.path.basename(route["_path"]).rsplit(".", 1)[0]
    out = f"/data/house/body/{name}"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    imageio.mimsave(out + "_body.mp4", frames, fps=15, quality=7, macro_block_size=8)
    end_d = float(np.linalg.norm(pts[-1] - body.xy))
    res = dict(route=name, reached_end=bool(done), end_dist_m=round(end_d, 2), fell=fell, error=err,
               sim_seconds=round(len(track) / 30, 1), min_head_z=round(min_head, 2), wall_seconds=round(time.time() - t0, 1),
               route_m=round(float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum()), 1))
    json.dump(dict(result=res, track=track), open(out + "_body.json", "w"))
    vol.commit()
    return res


@app.local_entrypoint()
def main(routes: str):
    items = []
    for p in routes.split(","):
        r = json.load(open(p))
        r["_path"] = p
        items.append(r)
    for res in walk.map(items, order_outputs=False):
        print(json.dumps(res))
