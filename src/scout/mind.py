"""The mind: a language model reads the body's memory and decides whether to keep exploring or go back and check
something. Muse Spark is the intended model; Muse Spark 1.3 on Meta Model API stands in until the Meta Model API key arrives.

It never sees simulator truth: only the memory table (what SAM 3.1 recognised, where and when), misses, and how much
of the house is still unexplored.
"""
from __future__ import annotations

import json
import os
import re
import time

MODEL = os.environ.get("SCOUT_MIND_MODEL", "muse-spark-1.3")
SYSTEM = """You are the mind of a humanoid body exploring a house it has never seen. Nobody gives you tasks.
Your own standing goal: learn the house and keep an accurate memory of what is where.

Each time you are asked, choose ONE action:
- {"do": "explore"}: keep walking to unexplored openings.
- {"do": "check", "object_id": N, "why": "..."}: walk back to remembered object N and look at it again.

When to check:
- An object was seen only once or with low confidence (below 0.8). It may be a misrecognition, so a second look
  confirms it.
- An object was NOT seen where you remembered it. Go back and look properly before believing it moved.
- Exploration is finished (no unexplored openings left). Keep your memory fresh: check the object you have not
  seen for the longest time, or one you are unsure about.
An object seen only once is tentative. Lines saying "probably a misrecognition" or "found it gone" are settled;
do not check those again.

Do not check the same object twice in a row. Prefer exploring while much of the house is unexplored, unless
something was not where you remembered it.
Reply with JSON only: {"do": ..., "object_id": ..., "why": "one short sentence in the first person"}"""


def ask_mind(table: str, state: str, region: str = "us-east-1", retries: int = 2) -> tuple[dict, float]:
    import boto3
    c = boto3.client("modelapi", region_name=region)
    msg = f"Memory:\n{table}\n\nNow:\n{state}"
    t0 = time.time()
    for k in range(retries + 1):
        try:
            r = c.converse(modelId=MODEL, system=[{"text": SYSTEM}], messages=[{"role": "user", "content": [{"text": msg}]}],
                           inferenceConfig={"maxTokens": 3000})
            text = [b["text"] for b in r["output"]["message"]["content"] if "text" in b][-1]
            m = re.search(r"\{.*\}", text, re.S)
            d = json.loads(m.group(0)) if m else {"do": "explore", "why": "(unparsed reply)"}
            if d.get("do") not in ("explore", "check"):
                d = {"do": "explore", "why": "(invalid action)"}
            return d, time.time() - t0
        except Exception as e:  # network or throttling: keep the body moving
            if k == retries:
                return {"do": "explore", "why": f"(mind unavailable: {type(e).__name__})"}, time.time() - t0
            time.sleep(2)
    return {"do": "explore"}, time.time() - t0
