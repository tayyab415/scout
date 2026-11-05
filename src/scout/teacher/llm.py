"""Chat model backends for the teacher. Meta Model API by default, any OpenAI-compatible endpoint on request."""
from __future__ import annotations

import os

from ..modelapi import BASE_URL, DEFAULT_MODEL, chat


class ChatModel:
    name = "chat"

    def complete(self, system: str, user: str, max_tokens: int = 200) -> tuple[str, dict]:
        """Return (text, info) where info has latency and token counts when available."""
        raise NotImplementedError


class MetaModelChat(ChatModel):
    """Meta Model API (https://api.meta.ai/v1). Default: Muse Spark 1.3."""

    def __init__(self, model_id: str = DEFAULT_MODEL, temperature: float = 0.0):
        self.model_id, self.temperature = model_id, temperature
        self.name = model_id

    def complete(self, system, user, max_tokens=200):
        return chat(system, user, model=self.model_id, max_tokens=max_tokens, temperature=self.temperature)


class OpenAICompatChat(ChatModel):
    """Any OpenAI-compatible chat endpoint, e.g. Muse Spark Open behind SGLang or a hosted provider.

    Configure with SCOUT_LLM_BASE_URL, SCOUT_LLM_MODEL and SCOUT_LLM_API_KEY (or SCOUT_LLM_API_KEY_FILE).
    """

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0):
        self.base_url = (base_url or os.environ.get("SCOUT_LLM_BASE_URL", BASE_URL)).rstrip("/")
        self.model = model or os.environ.get("SCOUT_LLM_MODEL", DEFAULT_MODEL)
        key = api_key or os.environ.get("SCOUT_LLM_API_KEY")
        if not key and os.environ.get("SCOUT_LLM_API_KEY_FILE"):
            key = open(os.environ["SCOUT_LLM_API_KEY_FILE"]).read().strip()
        self.key, self.temperature, self.name = key, temperature, self.model

    def complete(self, system, user, max_tokens=200):
        import requests
        import time
        t = time.time()
        r = requests.post(f"{self.base_url}/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "temperature": self.temperature, "max_tokens": max_tokens,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        j = r.json()
        u = j.get("usage", {})
        return j["choices"][0]["message"]["content"], {"latency_s": round(time.time() - t, 2),
                                                       "in_tokens": u.get("prompt_tokens"),
                                                       "out_tokens": u.get("completion_tokens")}
