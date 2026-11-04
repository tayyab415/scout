"""Copy the Scout data and every result off the Modal volume to S3 (one-off, before shutting Modal down).

    modal run --detach scripts/modal_to_s3.py
"""
import os

import modal

VOLUME = os.environ.get("SCOUT_VOLUME", "tribe-lab-claude")
BUCKET = os.environ.get("SCOUT_S3", "s3://meta-hackathon-570044538587/scout")
app = modal.App("scout-to-s3")
vol = modal.Volume.from_name(VOLUME)
image = modal.Image.debian_slim(python_version="3.12").pip_install("awscli")


def _aws_env():
    if not modal.is_local():
        return {"AWS_DEFAULT_REGION": "us-east-1"}
    import configparser
    cp = configparser.ConfigParser()
    cp.read(os.path.expanduser("~/.aws/credentials"))
    d = cp["default"]
    env = {"AWS_ACCESS_KEY_ID": d["aws_access_key_id"], "AWS_SECRET_ACCESS_KEY": d["aws_secret_access_key"], "AWS_DEFAULT_REGION": "us-east-1"}
    if d.get("aws_session_token"):
        env["AWS_SESSION_TOKEN"] = d["aws_session_token"]
    return env


@app.function(image=image, volumes={"/data": vol}, cpu=8, memory=16384, timeout=7200,
              secrets=[modal.Secret.from_dict(_aws_env())])
def copy(parts: list) -> dict:
    import subprocess
    out = {}
    for src, dst in parts:
        du = subprocess.run(["du", "-sh", src], capture_output=True, text=True).stdout.split("\t")[0]
        r = subprocess.run(["aws", "s3", "sync", "--only-show-errors", src, f"{BUCKET}/{dst}"], capture_output=True, text=True)
        out[src] = dict(size=du, ok=r.returncode == 0, err=r.stderr[-300:])
        print(src, out[src], flush=True)
    return out


@app.local_entrypoint()
def one(src: str = "/data/scout/env/final3", dst: str = "env/final3"):
    print(copy.remote([(src, dst)]))


@app.local_entrypoint()
def main():
    parts = [("/data/scout/env", "env"), ("/data/scout/live", "live"), ("/data/house/layouts", "house/layouts"),
             ("/data/house/scenes", "house/scenes"), ("/data/house/objects", "house/objects")]
    calls = [copy.spawn([p]) for p in parts]  # in parallel
    for c in calls:
        print(c.get())
