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
- Exploration is finished (no unexplored openings left). Keep your memory fresh: check the first object in the
  list "confident objects not seen for the longest", unless something was just found missing.
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


QUESTION_SYSTEM = """You are the memory of a humanoid body that has explored a house on its own. Someone now asks you
a question about something you saw. Use ONLY the memory table (objects your head camera recognised, with times in
seconds since you woke up, positions on your own map, and whether each is confident or tentative).

Pick the remembered object the question is about. If several match, prefer a confident one, then the most recent.
Then say what you knew before checking, in one short first-person sentence with the time you last saw it, and that
you will go and look. If nothing in memory matches, say you never saw one.
Reply with JSON only: {"object_id": N or null, "say": "..."}"""

ANSWER_SYSTEM = """You are a humanoid body that just walked back to check something you remembered, because someone
asked. You are given the question, what you remembered, and what the check found. Answer in one or two short
first-person sentences. Separate what you knew before from what you see now. Be precise about evidence:
- "still there": say you just saw it there again.
- "not found": say you could not find it where you last saw it (do NOT claim it was moved or taken; you did not see that).
- "unclear": say you could not get a clear look.
- "found elsewhere": say where you saw one now; say you cannot be sure it is the same one unless the memory shows
  the old place empty AND only one such object was ever seen.
Reply with JSON only: {"say": "..."}"""


def _converse(system: str, msg: str, region: str = "us-east-1") -> dict:
    import boto3
    c = boto3.client("modelapi", region_name=region)
    for k in range(3):
        try:
            r = c.converse(modelId=MODEL, system=[{"text": system}], messages=[{"role": "user", "content": [{"text": msg}]}],
                           inferenceConfig={"maxTokens": 3000})
            text = [b["text"] for b in r["output"]["message"]["content"] if "text" in b][-1]
            m = re.search(r"\{.*\}", text, re.S)
            return json.loads(m.group(0)) if m else {"say": text}
        except Exception:
            time.sleep(2)
    return {}


def resolve_question(table: str, question: str) -> dict:
    return _converse(QUESTION_SYSTEM, f"Memory:\n{table}\n\nQuestion: {question}")


def phrase_answer(question: str, remembered: str, finding: str) -> str:
    d = _converse(ANSWER_SYSTEM, f"Question: {question}\nWhat I remembered: {remembered}\nWhat the check found: {finding}")
    return d.get("say", finding)
