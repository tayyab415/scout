"""Ask the body about what it saw. The language model answers only from its memory and cites sightings; the cited
head-camera frames are saved with the detection box drawn.

    python scripts/ask.py --run runs/live/gate60 "Where did you see a sofa?" "Did you see a bathtub?"

Model: GPT-6.1 Terra on Bedrock as a stand-in for Muse Spark (set SCOUT_LLM_* to use the Meta Model API instead).
"""
from __future__ import annotations

import argparse
import json
import os
import re

import boto3
from PIL import Image, ImageDraw

MODEL = os.environ.get("SCOUT_QA_MODEL", "us.openai.gpt-6.1-terra")
SYSTEM = """You are the memory of a humanoid body that has been exploring a house it had never seen before.
You know ONLY what is in the memory table below: things its head camera recognised, with time (seconds since it
woke up), confidence, and position on its own map (x, y in metres). Lines saying "NOT seen at its last known place"
mean it looked at where something had been and did not find it.

Answer the question in one or two short sentences, in the first person ("I saw ...").
- Cite the sighting ids that support the answer.
- If the memory does not contain the answer, say plainly that you did not see it. Never guess.
- Describe places by what was nearby (other objects you saw close to it), and give the time.

Reply with JSON only: {"answer": "...", "cited_sightings": [ids], "observed": true|false}"""


def track_summary(mem: dict) -> str:
    S = {s["id"]: s for s in mem["sightings"]}
    lines = ["object_id | object | first_seen_s | last_seen_s | times_seen | best_confidence | x_m | y_m | sighting_ids (first, best, last)"]
    for tr in mem["tracks"]:
        ss = [S[i] for i in tr["sightings"]]
        best = max(ss, key=lambda s: s["score"])
        ids = sorted({ss[0]["id"], best["id"], ss[-1]["id"]})
        x, y = best["xyz"][0], best["xyz"][1]
        lines.append(f"{tr['id']} | {tr['label']} | {ss[0]['t']:.1f} | {ss[-1]['t']:.1f} | {len(ss)} | {best['score']:.2f} | "
                     f"{x:.1f} | {y:.1f} | {ids}")
        for t, fr in tr.get("misses", []):
            lines.append(f"{tr['id']} | {tr['label']} NOT seen at its last known place at t={t:.1f}s (frame {fr})")
        if tr.get("status") == "retracted":
            lines.append(f"{tr['id']} | {tr['label']}: a second look found nothing there, probably a misrecognition")
        elif tr.get("status") == "gone":
            lines.append(f"{tr['id']} | {tr['label']}: was seen here repeatedly, but a deliberate check found it gone")
    return "\n".join(lines)


def ask(mem: dict, question: str) -> dict:
    c = boto3.client("bedrock-runtime", region_name="us-east-1")
    msg = f"Memory table:\n{track_summary(mem)}\n\nQuestion: {question}"
    r = c.converse(modelId=MODEL, system=[{"text": SYSTEM}], messages=[{"role": "user", "content": [{"text": msg}]}],
                   inferenceConfig={"maxTokens": 4000})
    text = [b["text"] for b in r["output"]["message"]["content"] if "text" in b][-1]
    m = re.search(r"\{.*\}", text, re.S)
    return json.loads(m.group(0)) if m else {"answer": text, "cited_sightings": [], "observed": False}


def draw_cited(run: str, mem: dict, log: dict, ids: list[int], out_prefix: str) -> list[str]:
    S = {s["id"]: s for s in mem["sightings"]}
    dets = {d["sighting"]: d for lk in log["looks"] for d in lk["dets"]}
    paths = []
    for i in ids:
        if i not in S:
            continue
        s = S[i]
        im = Image.open(f"{run}/frame_{s['frame']:04d}.jpg").convert("RGB")
        dr = ImageDraw.Draw(im)
        if i in dets:
            x0, y0, x1, y1 = dets[i]["box"]
            dr.rectangle([x0, y0, x1, y1], outline=(255, 220, 0), width=4)
            dr.text((x0 + 4, max(y0 - 14, 0)), f"{s['label']} {s['score']:.2f}", fill=(255, 220, 0))
        dr.text((6, 6), f"t = {s['t']:.0f} s   sighting {i}", fill=(255, 255, 255))
        p = f"{out_prefix}_s{i}.jpg"
        im.save(p, quality=90)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("questions", nargs="+")
    a = ap.parse_args()
    mem = json.load(open(f"{a.run}/memory.json"))
    log = json.load(open(f"{a.run}/log.json"))
    os.makedirs(f"{a.run}/qa", exist_ok=True)
    results = []
    for k, q in enumerate(a.questions):
        r = ask(mem, q)
        r["question"] = q
        r["frames"] = draw_cited(a.run, mem, log, [int(i) for i in r.get("cited_sightings", [])], f"{a.run}/qa/q{k}")
        results.append(r)
        print(json.dumps(r, indent=1))
    json.dump(results, open(f"{a.run}/qa/answers.json", "w"), indent=1)


if __name__ == "__main__":
    main()
