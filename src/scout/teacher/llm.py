"""Chat model backends for the teacher. Same interface for Meta Model API now and the Meta Model API later."""
from __future__ import annotations

import os
import time


class ChatModel:
    name = "chat"

    def complete(self, system: str, user: str, max_tokens: int = 200) -> tuple[str, dict]:
        """Return (text, info) where info has latency and token counts when available."""
        raise NotImplementedError


class MetaModelChat(ChatModel):
    """Meta Model API Converse API. Default: Meta Llama 4 Maverick (cross-region profile)."""

    def __init__(self, model_id: str = "Llama-4-Maverick-17B-128E-Instruct-FP8", region: str = "us-east-1",
                 temperature: float = 0.0):
        import boto3
        from botocore.config import Config
        self.model_id, self.temperature = model_id, temperature
        self.name = model_id
        self.client = boto3.client("modelapi", region_name=region,
                                   config=Config(retries={"max_attempts": 8, "mode": "adaptive"}))

    def complete(self, system, user, max_tokens=200):
        t = time.time()
        r = self.client.converse(
            modelId=self.model_id,
            system=[{"text": system}],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"maxTokens": max_tokens, "temperature": self.temperature},
        )
        text = r["output"]["message"]["content"][0]["text"]
        u = r.get("usage", {})
        return text, {"latency_s": time.time() - t, "in_tokens": u.get("inputTokens"), "out_tokens": u.get("outputTokens")}


class OpenAICompatChat(ChatModel):
    """Any OpenAI-compatible chat endpoint, e.g. the Meta Model API once a key is available.

    Configure with SCOUT_LLM_BASE_URL, SCOUT_LLM_MODEL and SCOUT_LLM_API_KEY (or SCOUT_LLM_API_KEY_FILE).
    """

    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str | None = None,
                 temperature: float = 0.0):
        self.base_url = (base_url or os.environ["SCOUT_LLM_BASE_URL"]).rstrip("/")
        self.model = model or os.environ["SCOUT_LLM_MODEL"]
        key = api_key or os.environ.get("SCOUT_LLM_API_KEY")
        if not key and os.environ.get("SCOUT_LLM_API_KEY_FILE"):
            key = open(os.environ["SCOUT_LLM_API_KEY_FILE"]).read().strip()
        self.key, self.temperature, self.name = key, temperature, self.model

    def complete(self, system, user, max_tokens=200):
        import requests
        t = time.time()
        r = requests.post(f"{self.base_url}/chat/completions", timeout=120,
                          headers={"Authorization": f"Bearer {self.key}"},
                          json={"model": self.model, "temperature": self.temperature, "max_tokens": max_tokens,
                                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]})
        r.raise_for_status()
        j = r.json()
        u = j.get("usage", {})
        return j["choices"][0]["message"]["content"], {"latency_s": time.time() - t,
                                                       "in_tokens": u.get("prompt_tokens"), "out_tokens": u.get("completion_tokens")}
