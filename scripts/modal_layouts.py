"""Build a Layout (walkable grid, walls, rooms, objects) for every house on the Modal volume.

    modal run scripts/modal_layouts.py            # all houses
    modal run scripts/modal_layouts.py --limit 5  # quick check
"""
import os

import modal

app = modal.App("scout-layouts")
vol = modal.Volume.from_name("tribe-lab-claude")
ROOT = "/data/house"
SRC = os.path.join(os.path.dirname(__file__), "..", "src")
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("mujoco==3.17.0", "numpy")
         .add_local_dir(SRC, "/pkg"))


@app.function(image=image, volumes={"/data": vol}, cpu=2, memory=4096, timeout=900, retries=0, max_containers=30)
def build(name: str) -> tuple[str, str]:
    import sys
    import time
    sys.path.insert(0, "/pkg")
    from scout.world.house import layout_from_house
    out = f"{ROOT}/layouts/{name}.npz"
    if os.path.exists(out):
        return name, "cached"
    t = time.time()
    try:
        lay = layout_from_house(ROOT, name)
    except Exception as e:  # keep going; report at the end
        return name, f"error: {type(e).__name__}: {str(e)[:200]}"
    os.makedirs(f"{ROOT}/layouts", exist_ok=True)
    lay.save(out)  # the volume is committed when the container exits
    return name, f"ok {time.time() - t:.0f}s free={int(lay.grid.free.sum())} walls={int(lay.walls.sum())} objs={len(lay.objects)}"


@app.function(image=image, volumes={"/data": vol})
def house_names() -> list[str]:
    import json
    return sorted(json.load(open(f"{ROOT}/rooms.json")))


@app.local_entrypoint()
def main(limit: int = 0):
    names = house_names.remote()
    if limit:
        names = names[:limit]
    ok = err = 0
    for name, status in build.map(names, order_outputs=False):
        if status.startswith("error"):
            err += 1
            print(name, status)
        else:
            ok += 1
    print(f"done: {ok} ok, {err} errors")
