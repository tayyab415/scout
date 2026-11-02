"""The teacher as an exploration policy: an LLM picks the next opening from the body's memory."""
from __future__ import annotations

import json

import numpy as np

from ..policies import Policy
from .llm import ChatModel
from .prompt import SYSTEM, describe, parse


class TeacherPolicy(Policy):
    def __init__(self, llm: ChatModel, log_path: str | None = None):
        self.llm = llm
        self.name = f"teacher:{llm.name}"
        self.log_path = log_path
        self.calls = 0
        self.fallbacks = 0
        self.latency = []

    def choose(self, memory, body, frontiers, goal, budget_left):
        user = describe(memory, body, frontiers, goal, budget_left)
        try:
            text, info = self.llm.complete(SYSTEM, user, max_tokens=300)
        except Exception as e:  # network or quota trouble: fall back, but count it
            text, info = f"ERROR {type(e).__name__}: {e}", {}
        self.calls += 1
        if info.get("latency_s"):
            self.latency.append(info["latency_s"])
        k, why = parse(text, len(frontiers))
        if k is None:
            self.fallbacks += 1
            k = int(np.argmin([d for _, d in frontiers]))
        if self.log_path:
            with open(self.log_path, "a") as f:
                f.write(json.dumps({"house": memory.layout.name, "goal": goal, "prompt": user, "reply": text,
                                    "choice": k, "why": why, "n": len(frontiers), **info}) + "\n")
        return k
