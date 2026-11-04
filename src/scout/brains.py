"""Plug-in brains for the environment. Every brain gets the same body, eyes and memory table; it only decides.

A brain answers three kinds of call:
  decide(table, state)        -> {"do": "explore"} or {"do": "check", "object_id": N, "why": ...}
  resolve(table, question)    -> {"object_id": N or None, "say": ...}   which remembered object a question is about
  phrase(question, remembered, finding) -> str                          the spoken answer after a check
  answer(table, question)     -> {"answer": ..., "cited_sightings": [...], "observed": bool}   memory QA

Brains:
  llm:<Meta Model API model id>   a language model (Muse Spark 1.3 stands in for Muse Spark; Llama 4 Maverick)
  rules                    fixed rules, no language model
  random                   random choices (a floor for the leaderboard)
  student                  a 233-parameter trust network trained on Scout's own recorded episodes (src/scout/student/)
"""
from __future__ import annotations

import json
import random
import re
import time

from . import mind as _mind

SPARK = "muse-spark-1.3"
LLAMA4 = "Llama-4-Maverick-17B-128E-Instruct-FP8"

QA_SYSTEM = """You are the memory of a humanoid body that explored a house it had never seen. You know ONLY the memory
table (objects its head camera recognised, with times in seconds since it woke up, positions on its own map, and
whether each is confident or tentative). Answer in one or two short first-person sentences, cite the supporting
sighting ids, and if the memory does not contain the answer say plainly you did not see it. Never guess.
Reply with JSON only: {"answer": "...", "cited_sightings": [ids], "observed": true|false, "object_id": N or null}"""


def _rows(table: str):
    """Parse the memory table back into (object_id, label, times_seen, last_seen, confident) rows."""
    out = []
    for line in table.splitlines()[1:]:
        p = [x.strip() for x in line.split("|")]
        if len(p) >= 10 and p[0].isdigit():
            out.append(dict(id=int(p[0]), label=p[1], last=float(p[3]), n=int(p[4]), conf=p[9] == "confident",
                            ids=[int(x) for x in re.findall(r"\d+", p[8])]))
    return out


class LLMBrain:
    def __init__(self, model: str):
        self.model, self.name = model, {SPARK: "spark (Muse Spark stand-in)", LLAMA4: "llama4-maverick"}.get(model, model)

    def _call(self, system, msg):
        import boto3
        c = boto3.client("modelapi", region_name="us-east-1")
        for k in range(3):
            try:
                r = c.converse(modelId=self.model, system=[{"text": system}], messages=[{"role": "user", "content": [{"text": msg}]}],
                               inferenceConfig={"maxTokens": 3000})
                text = [b["text"] for b in r["output"]["message"]["content"] if "text" in b][-1]
                m = re.search(r"\{.*\}", text, re.S)
                if not m:
                    return {"_error": "no JSON in reply", "_raw": text[:500]}
                return json.loads(m.group(0))
            except Exception as e:  # provider or parse failure: report it, never turn it into an answer
                err = f"{type(e).__name__}: {str(e)[:200]}"
                time.sleep(2)
        return {"_error": err}

    def decide(self, table, state):
        d = self._call(_mind.SYSTEM, f"Memory:\n{table}\n\nNow:\n{state}")
        return d if d.get("do") in ("explore", "check") else {"do": "explore", "why": "(no valid reply)"}

    def resolve(self, table, question):
        return self._call(_mind.QUESTION_SYSTEM, f"Memory:\n{table}\n\nQuestion: {question}")

    def phrase(self, question, remembered, finding):
        return self._call(_mind.ANSWER_SYSTEM, f"Question: {question}\nWhat I remembered: {remembered}\nWhat the check found: {finding}").get("say", finding)

    def answer(self, table, question):
        return self._call(QA_SYSTEM, f"Memory:\n{table}\n\nQuestion: {question}")


class RuleBrain:
    """No language model: explore until done, then re-check confident objects oldest-first; questions are matched
    to memory by the object name appearing in the question."""
    name = "rules"

    def decide(self, table, state):
        m = re.search(r"unexplored openings reachable: (\d+)", state)
        if m and int(m.group(1)) > 0:
            return {"do": "explore"}
        stale = re.search(r"confident objects not seen for the longest: (\d+)", state)
        return {"do": "check", "object_id": int(stale.group(1)), "why": "oldest confident object"} if stale else {"do": "explore"}

    def _match(self, table, question):
        rows = [r for r in _rows(table) if r["label"] in question.lower()]
        rows.sort(key=lambda r: (not r["conf"], -r["last"]))
        return rows[0] if rows else None

    def resolve(self, table, question):
        r = self._match(table, question)
        return {"object_id": r["id"], "say": f"I last saw the {r['label']} at {r['last']:.0f} s. I'll go and look."} if r \
            else {"object_id": None, "say": "I never saw one."}

    def phrase(self, question, remembered, finding):
        if finding.startswith("answered from memory"):
            return "Yes, it is where I saw it." if finding.endswith("still there") else "No, it is gone."
        if finding.startswith("unclear: something"):
            return "Something is still at that spot, but I could not recognise it."
        return {"still there": "I just saw it there again.", "not found": "I could not find it where I last saw it.",
                "unclear": "I could not get a clear look."}.get(finding, finding)

    def answer(self, table, question):
        r = self._match(table, question)
        if not r:
            return {"answer": "I did not see one.", "cited_sightings": [], "observed": False, "object_id": None}
        return {"answer": f"I saw a {r['label']}, last at {r['last']:.0f} s.", "cited_sightings": r["ids"][-1:],
                "observed": True, "object_id": r["id"]}


class RandomBrain:
    name = "random"

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def decide(self, table, state):
        rows = _rows(table)
        if rows and self.rng.random() < 0.3:
            return {"do": "check", "object_id": self.rng.choice(rows)["id"], "why": "random"}
        return {"do": "explore"}

    def resolve(self, table, question):
        rows = _rows(table)
        return {"object_id": self.rng.choice(rows)["id"] if rows else None, "say": "I'll go and look."}

    def phrase(self, question, remembered, finding):
        return self.rng.choice(["It is still there.", "I could not find it.", "I could not get a clear look."])

    def answer(self, table, question):
        rows = _rows(table)
        if not rows or self.rng.random() < 0.5:
            return {"answer": "I did not see one.", "cited_sightings": [], "observed": False, "object_id": None}
        r = self.rng.choice(rows)
        return {"answer": f"I saw a {r['label']}.", "cited_sightings": r["ids"][-1:], "observed": True, "object_id": r["id"]}


_REGISTRY = {}


def register_brain(name: str, factory) -> None:
    """Plug in your own brain: register_brain("mybrain", MyBrain) makes `--brains mybrain` work. A brain implements
    decide, resolve, phrase and answer (see the module docstring); any Meta Model API chat model also works as "llm:<model id>"."""
    _REGISTRY[name] = factory


def make_brain(name: str):
    """'<brain>-recall' is the same brain answering verification questions from memory only, without walking back."""
    if name.endswith("-recall"):
        b = make_brain(name[: -len("-recall")])
        b.recall = True
        b.name = f"{b.name} + memory only (no physical check)"
        return b
    if name in _REGISTRY:
        return _REGISTRY[name]()
    if name in ("spark", "spark"):
        return LLMBrain(SPARK)
    if name in ("llama4", "llama"):
        return LLMBrain(LLAMA4)
    if name.startswith("llm:"):
        return LLMBrain(name[4:])
    if name == "rules":
        return RuleBrain()
    if name == "random":
        return RandomBrain()
    if name == "student":
        from .student.trust import StudentBrain
        return StudentBrain()
    raise ValueError(f"unknown brain {name}")
