"""Download the MolmoSpaces ProcTHOR-10K validation houses (CC BY 4.0) and THOR object assets onto the Modal volume.

    SCOUT_VOLUME=scout-data modal run scripts/modal_fetch_houses.py

Source: https://huggingface.co/datasets/allenai/molmospaces (mujoco/scenes/procthor-10k-val, mujoco/objects/thor).
Writes <volume>/house/{scenes,objects/thor,rooms.json}. Then run scripts/modal_layouts.py for the evaluator layouts.
Adapted from the house smoke test written earlier in this hackathon.
"""
import os

import modal

VOLUME = os.environ.get("SCOUT_VOLUME", "tribe-lab-claude")
app = modal.App("scout-fetch-houses")
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.11").apt_install("zstd").pip_install("requests")
R = "/data/house"
REPO = "https://huggingface.co/datasets/allenai/molmospaces/resolve/main/mujoco"


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=16384, timeout=3600)
def fetch():
    import glob
    import json
    import subprocess

    import requests
    os.makedirs(f"{R}/raw", exist_ok=True)

    def get(url, out):
        if os.path.exists(out):
            return
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(out + ".part", "wb") as f:
                for c in r.iter_content(1 << 22):
                    f.write(c)
        os.rename(out + ".part", out)
    get(f"{REPO}/scenes/procthor-10k-val/20251217/shards/00000.tar", f"{R}/raw/scenes.tar")
    get(f"{REPO}/objects/thor/20251117/shards/00000.tar", f"{R}/raw/thor.tar")
    for name, dst in (("scenes", "scenes_pk"), ("thor", "thor_pk")):  # each shard is a tar of .tar.zst packages
        os.makedirs(f"/tmp/{dst}", exist_ok=True)
        subprocess.run(["tar", "-xf", f"{R}/raw/{name}.tar", "-C", f"/tmp/{dst}"], check=True)
    os.makedirs("/tmp/objx", exist_ok=True)
    for p in sorted(glob.glob("/tmp/thor_pk/**/*.tar.zst", recursive=True)):
        subprocess.run(f'zstd -dc "{p}" | tar -x -C /tmp/objx', shell=True, check=True)
    tex = [d for d in glob.glob("/tmp/objx/**/Textures", recursive=True) if os.path.isdir(d)]
    root = os.path.dirname(sorted(tex, key=len)[0])
    os.makedirs(f"{R}/objects", exist_ok=True)
    subprocess.run(["rm", "-rf", f"{R}/objects/thor"])
    subprocess.run(["cp", "-r", root, f"{R}/objects/thor"], check=True)
    rooms = {}
    os.makedirs(f"{R}/scenes", exist_ok=True)
    for p in sorted(glob.glob("/tmp/scenes_pk/**/*.tar.zst", recursive=True)):
        n = os.path.basename(p)[:-8]
        if "_val_" not in n:
            continue
        d = f"{R}/scenes/{n}"
        os.makedirs(d, exist_ok=True)
        subprocess.run(f'zstd -dc "{p}" | tar -x -C "{d}"', shell=True, check=True)
        js = [j for j in glob.glob(f"{d}/*.json") if "metadata" not in j]
        if js:
            rooms[n] = len(json.load(open(js[0]))["rooms"])
    json.dump(rooms, open(f"{R}/rooms.json", "w"))
    vol.commit()
    return len(rooms)


@app.local_entrypoint()
def main():
    print("houses:", fetch.remote())
